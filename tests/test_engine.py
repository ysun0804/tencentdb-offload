#!/usr/bin/env python3
"""Test tencentdb-offload plugin: import, health check, compact, fallback, ingest."""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
sys.path.insert(0, _PARENT)
sys.path.insert(0, _HERE)

import engine as engine_mod
from engine import TencentDBOffloadEngine, _estimate_tokens


def test_engine():
    """End-to-end exercise of TencentDBOffloadEngine including ingest-before-compact.

    Sections [1]–[8] cover the pre-existing surface; [9]–[14] cover the new
    ingest-before-compact flow required by the reconstruction task.
    """
    print("=" * 60)
    print("tencentdb-offload plugin test")
    print("=" * 60)

    # 1. Instantiation
    engine = TencentDBOffloadEngine()
    assert engine.name == "tencentdb-offload", f"name mismatch: {engine.name}"
    print(f"[1] name: {engine.name} ✅")

    # 2. Health check
    available = engine._check_available()
    assert available, "Gateway should be reachable at :8420"
    print(f"[2] gateway health: {'✅ reachable' if available else '❌ unreachable'}")

    # 3. Token estimation
    msgs = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "hello " * 100},
        {"role": "assistant", "content": "response " * 50},
    ]
    est = _estimate_tokens(msgs)
    assert est > 0, "token estimate should be positive"
    print(f"[3] token estimate: {est} for {len(msgs)} messages ✅")

    # 4. Compact via API (also exercises ingest-before-compact on real gateway)
    engine._session_id = "test-hermes-001"
    engine.context_length = 200000
    engine.last_prompt_tokens = 200
    result = engine.compress(msgs)
    assert len(result) > 0, "compact should return non-empty list"
    print(f"[4] compact: {len(msgs)} -> {len(result)} messages ✅")

    # 5. Fallback compress
    big_msgs = [{"role": "system", "content": "system"}]
    big_msgs += [{"role": "user", "content": f"message {i} " * 200} for i in range(50)]
    big_msgs += [{"role": "assistant", "content": f"reply {i} " * 100} for i in range(50)]
    big_msgs += [{"role": "user", "content": "final question"}]

    engine._available = False  # force fallback
    engine.context_length = 5000  # force aggressive cut
    fallback = engine._fallback_compress(big_msgs)
    assert len(fallback) < len(big_msgs), "fallback should reduce message count"
    assert fallback[0] == big_msgs[0], "should keep system message"
    assert fallback[-1] == big_msgs[-1], "should keep last message"
    print(f"[5] fallback: {len(big_msgs)} -> {len(fallback)} messages ✅")

    # 6. should_compress
    engine._available = None  # reset
    engine.context_length = 100000
    engine.threshold_tokens = int(100000 * 0.75)
    assert not engine.should_compress(50000), "50k < 75k threshold"
    assert engine.should_compress(80000), "80k > 75k threshold"
    print(f"[6] should_compress: threshold={engine.threshold_tokens} ✅")

    # 7. update_from_response
    engine.update_from_response({
        "prompt_tokens": 1500,
        "completion_tokens": 500,
        "total_tokens": 2000,
    })
    assert engine.last_prompt_tokens == 1500
    assert engine.last_total_tokens == 2000
    print(f"[7] update_from_response: {engine.last_total_tokens} tokens ✅")

    # 8. update_model (flexible signature)
    engine.update_model("test-model", 200000)
    assert engine.context_length == 200000
    print(f"[8] update_model: ctx={engine.context_length} ✅")

    # 9. _extract_tool_pairs — OpenAI format
    openai_msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "list files"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {"name": "terminal", "arguments": '{"command": "ls"}'},
            }],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "file_a\nfile_b"},
        {"role": "assistant", "content": "done"},
    ]
    pairs = engine._extract_tool_pairs(openai_msgs)
    assert len(pairs) == 1, f"expected 1 pair, got {len(pairs)}"
    assert pairs[0]["tool_name"] == "terminal"
    assert pairs[0]["tool_call_id"] == "call_1"
    assert pairs[0]["params"] == {"command": "ls"}
    assert pairs[0]["result"] == "file_a\nfile_b"
    print(f"[9] extract_tool_pairs (OpenAI): {len(pairs)} pair ✅")

    # 10. _extract_tool_pairs — Anthropic format
    anthropic_msgs = [
        {"role": "user", "content": "list files"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "running ls"},
                {"type": "tool_use", "id": "toolu_1", "name": "terminal", "input": {"command": "ls"}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1", "content": "file_a\nfile_b"},
            ],
        },
    ]
    pairs = engine._extract_tool_pairs(anthropic_msgs)
    assert len(pairs) == 1, f"expected 1 pair, got {len(pairs)}"
    assert pairs[0]["tool_name"] == "terminal"
    assert pairs[0]["tool_call_id"] == "toolu_1"
    assert pairs[0]["params"] == {"command": "ls"}
    assert pairs[0]["result"] == "file_a\nfile_b"
    print(f"[10] extract_tool_pairs (Anthropic): {len(pairs)} pair ✅")

    # 11. _build_recent_messages — skip tool msgs, truncate, max
    recent_src = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "short"},
        {"role": "user", "content": "hello world"},
        {"role": "assistant", "content": "thanks for asking"},
        {"role": "tool", "tool_call_id": "x", "content": "raw tool output"},
        {"role": "user", "content": "HEARTBEAT ping"},
        {"role": "assistant", "content": "x" * 500},  # long, should be truncated
    ] + [{"role": "user", "content": f"msg {i}"} for i in range(15)]
    recent = engine._build_recent_messages(recent_src, max_msgs=5)
    assert len(recent) <= 5, f"expected ≤5, got {len(recent)}"
    roles = [r["role"] for r in recent]
    assert "tool" not in roles, "tool messages must be skipped"
    for r in recent:
        assert len(r["content"]) <= 400, f"content >400 chars: {len(r['content'])}"
    assert all("HEARTBEAT" not in r["content"] for r in recent), "heartbeat must be skipped"
    print(f"[11] build_recent_messages: {len(recent)} msgs (max 5), tool/heartbeat filtered ✅")

    # 12. _ingest_before_compact does not raise when gateway is unavailable
    engine._available = False  # simulate unreachable gateway
    try:
        engine._ingest_before_compact(openai_msgs, "test-session-unreachable")
        print("[12] ingest_before_compact: skipped cleanly when unavailable ✅")
    except Exception as exc:
        raise AssertionError(f"ingest must not raise when unavailable: {exc}")
    finally:
        engine._available = None  # reset cache

    # 13. compress() calls ingest BEFORE compact (mock _post_json to capture order)
    _captured: list = []
    _orig_post = engine_mod._post_json

    def _mock_post(url, body, headers, timeout_ms):
        _captured.append(url)
        if "/ingest" in url:
            return {"code": 0, "data": {}}
        if "/compact" in url:
            # Echo back first message as "compacted"
            return {"code": 0, "data": {"messages": body["messages"][:1], "report": {}}}
        return {"code": 0, "data": {}}

    engine_mod._post_json = _mock_post
    try:
        engine._available = True  # skip real health check
        engine._session_id = "test-order-001"
        result = engine.compress(openai_msgs)
        ingest_calls = [i for i, u in enumerate(_captured) if "/ingest" in u]
        compact_calls = [i for i, u in enumerate(_captured) if "/compact" in u]
        assert ingest_calls, "ingest should have been called"
        assert compact_calls, "compact should have been called"
        assert ingest_calls[0] < compact_calls[0], (
            f"ingest must precede compact; got order={_captured}"
        )
        assert len(result) > 0
        print(
            f"[13] compress calls ingest→compact in order: "
            f"{[u.split('/')[-1] for u in _captured]} ✅"
        )
    finally:
        engine_mod._post_json = _orig_post
        engine._available = None  # reset for any later use

    # 14. compress() tolerates ingest failure and still calls compact (fire-and-forget)
    _captured.clear()

    def _mock_post_fail_ingest(url, body, headers, timeout_ms):
        _captured.append(url)
        if "/ingest" in url:
            return None  # simulate ingest HTTP failure
        if "/compact" in url:
            return {"code": 0, "data": {"messages": body["messages"][:1], "report": {}}}
        return {"code": 0, "data": {}}

    engine_mod._post_json = _mock_post_fail_ingest
    try:
        result = engine.compress(openai_msgs)
        assert any("/ingest" in u for u in _captured), "ingest should have been attempted"
        assert any("/compact" in u for u in _captured), "compact should still run after ingest failure"
        assert len(result) > 0, "compress should still return a result"
        print("[14] compress survives ingest failure, compact still runs ✅")
    finally:
        engine_mod._post_json = _orig_post
        engine._available = None

    print()
    print("=" * 60)
    print("ALL TESTS PASSED ✅")
    print("=" * 60)


if __name__ == "__main__":
    test_engine()
