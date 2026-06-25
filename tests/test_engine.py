#!/usr/bin/env python3
"""Test tencentdb-offload plugin: import, health check, compact, fallback."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine import TencentDBOffloadEngine

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
est = 0
from engine import _estimate_tokens
est = _estimate_tokens(msgs)
assert est > 0, "token estimate should be positive"
print(f"[3] token estimate: {est} for {len(msgs)} messages ✅")

# 4. Compact via API
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

print()
print("=" * 60)
print("ALL TESTS PASSED ✅")
print("=" * 60)
