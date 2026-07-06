#!/usr/bin/env python3
"""Test tencentdb-offload plugin: import, health check, compact, fallback, ingest."""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
sys.path.insert(0, _PARENT)
sys.path.insert(0, _HERE)

import engine as engine_mod
from engine import TencentDBOffloadEngine, _estimate_tokens, SessionRegistry, SessionState, _hash_content


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


# ==========================================================================
# Tests for new features (15–24)
# ==========================================================================

def test_new_features():
    """Tests for the 5 new features:
      15. SessionRegistry + SessionState
      16. _filter_heartbeat_messages
      17. _build_active_mmd_text
      18. _build_history_mmd_text
      19. _inject_active_mmd (insertion point)
      20. _inject_history_mmds (insertion point)
      21. _remove_existing_mmd_injections
      22. pre_llm_call with mocked Gateway
      23. reclaim
      24. _hash_content
    """
    print("=" * 60)
    print("tencentdb-offload new features test")
    print("=" * 60)

    # 15. SessionRegistry + SessionState
    sr = SessionRegistry(max_cached=3)
    s1 = sr.resolve("session-1")
    assert sr.size == 1
    assert s1.session_id == "session-1"
    assert not s1.is_processed("call_1")
    s1.mark_processed("call_1")
    assert s1.is_processed("call_1")
    s1.mark_offloaded("call_1")
    assert s1.is_offloaded("call_1")
    s1.mark_deleted("call_2")
    assert s1.is_deleted("call_2")
    s1.set_mmd_version("001-task.mmd", "abc123")
    assert s1.get_mmd_version("001-task.mmd") == "abc123"

    # LRU eviction
    s2 = sr.resolve("session-2")
    s3 = sr.resolve("session-3")
    assert sr.size == 3
    s4 = sr.resolve("session-4")  # should evict oldest
    assert sr.size == 3

    # get() doesn't create
    assert sr.get("nonexistent") is None
    assert sr.size == 3  # no new entry created

    # prune_expired
    import time
    sr2 = SessionRegistry(max_cached=10)
    sa = sr2.resolve("old-session")
    sa.last_access_ms = 0  # very old
    sb = sr2.resolve("new-session")
    sb.last_access_ms = time.time() * 1000
    pruned = sr2.prune_expired(cutoff_epoch_s=time.time())
    assert pruned == 1
    assert sr2.size == 1
    assert sr2.get("old-session") is None
    print("[15] SessionRegistry + SessionState: LRU eviction, prune ✅")

    # 16. _filter_heartbeat_messages
    engine = TencentDBOffloadEngine()
    hb_msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hello"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "checking"},
                {"type": "tool_use", "id": "hb_1", "name": "terminal",
                 "input": {"command": "cat HEARTBEAT.md"}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "hb_1",
                 "content": "heartbeat OK"},
            ],
        },
        {"role": "assistant", "content": "done"},
    ]
    removed = engine._filter_heartbeat_messages(hb_msgs)
    assert removed > 0, f"expected heartbeat removal, got {removed}"
    # heartbeat tool_use and tool_result should be gone
    for msg in hb_msgs:
        content = msg.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    assert "HEARTBEAT" not in str(block.get("input", "")), \
                        f"heartbeat tool_use not filtered: {block}"
    print(f"[16] filter_heartbeat_messages: removed {removed} ✅")

    # 17. _build_active_mmd_text
    active_text = engine._build_active_mmd_text("001-task.mmd", "graph TD\nA-->B")
    assert "<current_task_context>" in active_text
    assert "```mermaid" in active_text
    assert "001-task.mmd" in active_text
    print("[17] build_active_mmd_text: correct format ✅")

    # 18. _build_history_mmd_text
    hist_text = engine._build_history_mmd_text("002-old.mmd", "graph LR\nX-->Y")
    assert "<history_task_context" in hist_text
    assert "```mermaid" in hist_text
    assert "002-old.mmd" in hist_text
    print("[18] build_history_mmd_text: correct format ✅")

    # 19. _inject_active_mmd insertion point
    msgs_19 = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "first q"},
        {"role": "assistant", "content": "first a"},
        {"role": "user", "content": "second q"},
        {"role": "assistant", "content": "second a"},
        {"role": "user", "content": "final q"},
    ]
    engine_19 = TencentDBOffloadEngine()
    engine_19._inject_active_mmd(msgs_19, "001-task.mmd", "graph TD\nA-->B", "v1")
    # Should find an MMD message with marker
    mmd_found = any(
        m.get(engine_19._MMD_CONTEXT_MARKER) == "active" for m in msgs_19
    )
    assert mmd_found, "active MMD should be injected"
    # System message should still be first
    assert msgs_19[0].get("role") == "system"
    print("[19] inject_active_mmd: insertion point correct ✅")

    # 20. _inject_history_mmds insertion
    msgs_20 = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a"},
        {"role": "user", "content": "final"},
    ]
    hist_list = [
        {"filename": "002-old.mmd", "content": "graph LR\nX-->Y", "version": "v2"},
    ]
    count = engine_19._inject_history_mmds(msgs_20, hist_list, token_budget=10000)
    assert count == 1, f"expected 1 history MMD injected, got {count}"
    hist_found = any(
        m.get(engine_19._MMD_INJECTION_MARKER) for m in msgs_20
    )
    assert hist_found, "history MMD should be injected"
    print("[20] inject_history_mmds: insertion correct ✅")

    # 21. _remove_existing_mmd_injections
    msgs_21 = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "normal", "_mmdInjection": True},
        {"role": "user", "content": "also normal", "_mmdContextMessage": "active"},
        {"role": "user", "content": "keep me"},
    ]
    removed = engine_19._remove_existing_mmd_injections(msgs_21)
    assert removed == 2, f"expected 2 MMD removals, got {removed}"
    assert len(msgs_21) == 2
    assert msgs_21[-1].get("content") == "keep me"
    print("[21] remove_existing_mmd_injections: 2 removed ✅")

    # 22. pre_llm_call with mocked Gateway
    _captured_22: list = []
    _orig_post_22 = engine_mod._post_json

    def _mock_post_22(url, body, headers, timeout_ms):
        _captured_22.append(url)
        if "/query-mmd" in url:
            return {"code": 0, "data": {
                "mmds": [
                    {"filename": "001-task.mmd", "content": "graph TD\nA-->B",
                     "version": "v1"},
                ],
                "currentMmd": "001-task.mmd",
            }}
        if "/compact" in url:
            return {"code": 0, "data": {"messages": body["messages"][:2], "report": {}}}
        return {"code": 0, "data": {}}

    engine_22 = TencentDBOffloadEngine()
    engine_22._available = True
    engine_22._session_id = "test-pre-llm-22"
    engine_22.context_length = 200000
    engine_22.threshold_tokens = int(200000 * 0.75)
    engine_mod._post_json = _mock_post_22

    try:
        msgs_22 = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello " * 100},
            {"role": "assistant", "content": "response " * 50},
            {"role": "user", "content": "final"},
        ]
        result = engine_22.pre_llm_call(msgs_22)
        # pre_llm_call returns None (no context injection, modifies in-place)
        assert result is None
        # Should have called query-mmd at minimum
        mmd_called = any("/query-mmd" in u for u in _captured_22)
        assert mmd_called, f"expected query-mmd call, got {_captured_22}"
        print("[22] pre_llm_call: query-mmd called, no crash ✅")
    finally:
        engine_mod._post_json = _orig_post_22
        engine_22._available = None

    # 23. reclaim
    engine_23 = TencentDBOffloadEngine()
    # reclaim with retention < 3 should be no-op
    stats = engine_23.reclaim(retention_days=2)
    assert stats["local_sessions_pruned"] == 0

    # reclaim with registry
    sr_23 = SessionRegistry(max_cached=10)
    sr_23.resolve("old-sess")
    engine_23._session_registry = sr_23
    stats = engine_23.reclaim(retention_days=7)
    # "old-sess" was just created so not expired
    assert stats["local_sessions_pruned"] == 0
    print("[23] reclaim: no-op for fresh sessions, respects retention < 3 ✅")

    # 24. _hash_content
    h1 = _hash_content("hello world")
    h2 = _hash_content("hello world")
    h3 = _hash_content("different")
    assert h1 == h2, "same content → same hash"
    assert h1 != h3, "different content → different hash"
    assert len(h1) == 12, f"hash should be 12 chars, got {len(h1)}"
    print(f"[24] _hash_content: deterministic, 12-char hex ✅")

    print()
    print("=" * 60)
    print("ALL NEW FEATURE TESTS PASSED ✅")
    print("=" * 60)


if __name__ == "__main__":
    test_engine()
    test_new_features()  # separate function to keep original tests untouched


if __name__ == "__main__":
    test_engine()
