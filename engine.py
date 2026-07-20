"""tencentdb-offload — Hermes Context Engine plugin backed by TencentDB offload V2 API.

Wraps the TencentDB Agent Memory Gateway's offload endpoints as a Hermes
ContextEngine.  Compression is delegated to the Gateway server which performs
multi-layer offload (L1 tool-pair summarisation, L2 Mermaid canvas, L3
compaction).

The plugin talks exclusively to the stable HTTP V2 API:
    POST /v2/offload/compact   — synchronous compaction
    POST /v2/offload/ingest    — fire-and-forget tool-pair ingestion

It does NOT depend on TencentDB internals, so any server upgrade that keeps
the V2 API contract is automatically compatible.

Configuration via environment variables (or ``plugin.yaml``):
    TENCENTDB_OFFLOAD_GATEWAY_URL   — Gateway base URL (default: http://127.0.0.1:8420)
    TENCENTDB_OFFLOAD_API_KEY       — API key for auth (default: local)
    TENCENTDB_OFFLOAD_INSTANCE_ID   — x-tdai-service-id (default: default)
    TENCENTDB_OFFLOAD_COMPACT_RATIO — target context ratio after compaction (default: 0.5)
    TENCENTDB_OFFLOAD_TIMEOUT_MS    — HTTP timeout for compact calls (default: 30000)
    TENCENTDB_OFFLOAD_INGEST_TIMEOUT_MS — timeout for ingest calls (default: 5000)
    TENCENTDB_OFFLOAD_ENABLED       — master switch (default: false; set to switch from LCM)
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# HTTP client — minimal, no external deps
# ---------------------------------------------------------------------------

import json as _json
import urllib.error
import urllib.request


def _post_json(
    url: str,
    body: dict,
    headers: dict,
    timeout_ms: int,
) -> Optional[dict]:
    """POST JSON and return parsed response, or None on failure."""
    data = _json.dumps(body).encode("utf-8")
    h = {**headers, "Content-Type": "application/json"}
    req = urllib.request.Request(url, data=data, headers=h, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout_ms / 1000) as resp:
            return _json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, _json.JSONDecodeError) as exc:
        logger.warning("[tencentdb-offload] HTTP %s failed: %s", url, exc)
        return None
    except Exception as exc:
        logger.warning("[tencentdb-offload] HTTP %s unexpected error: %s", url, exc)
        return None


# ---------------------------------------------------------------------------
# Token estimation — rough but sufficient for should_compress decisions
# ---------------------------------------------------------------------------

_ROUGH_CHARS_PER_TOKEN = 3.5  # conservative for mixed CJK/English


def _is_tool_result(msg: Dict[str, Any]) -> bool:
    """Detect tool-result messages in OpenAI or Anthropic format.

    Mirrors OpenClaw ``context-engine.ts`` ``isToolResult()``:
      - OpenAI: role in {tool, function, toolResult, tool_result}
      - Anthropic: user message whose content list contains ``tool_result`` blocks
    """
    role = msg.get("role", "")
    if role in ("tool", "function", "toolResult", "tool_result"):
        return True
    content = msg.get("content")
    if role == "user" and isinstance(content, list):
        return any(
            isinstance(b, dict) and b.get("type") == "tool_result"
            for b in content
        )
    return False


def _truncate_tool_result(msg: Dict[str, Any], max_chars: int = 500) -> Dict[str, Any]:
    """Truncate tool_result content in a message to avoid huge HTTP bodies.

    Handles both OpenAI format (role="tool"/"function" with string content) and
    Anthropic format (user messages with ``tool_result`` content blocks).
    Mirrors OpenClaw ``context-engine.ts`` ``truncateToolResult()``.
    """
    _MARKER = "\n...[truncated for compact]"
    content = msg.get("content", "")

    # OpenAI format: tool/function role with string content
    if isinstance(content, str) and len(content) > max_chars:
        if _is_tool_result(msg):
            msg = dict(msg)
            msg["content"] = content[:max_chars] + _MARKER
    # List content blocks (OpenAI multi-block, or Anthropic tool_result blocks)
    elif isinstance(content, list):
        changed = False
        new_content = []
        for block in content:
            if isinstance(block, dict):
                text = block.get("text", block.get("content", ""))
                if isinstance(text, str) and len(text) > max_chars:
                    block = dict(block)
                    if "text" in block:
                        block["text"] = text[:max_chars] + _MARKER
                    elif "content" in block:
                        block["content"] = text[:max_chars] + _MARKER
                    changed = True
            new_content.append(block)
        if changed:
            msg = dict(msg)
            msg["content"] = new_content
    return msg


def _estimate_tokens(messages: List[Dict[str, Any]]) -> int:
    """Rough token estimate for a message list."""
    total_chars = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total_chars += len(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    total_chars += len(str(block.get("text", block.get("content", ""))))
                else:
                    total_chars += len(str(block))
    return int(total_chars / _ROUGH_CHARS_PER_TOKEN)


def _extract_text_content(content: Any) -> str:
    """Flatten message content (string or list of blocks) into a single text string.

    Only ``text`` blocks contribute; ``tool_use`` / ``tool_result`` blocks are
    skipped so we don't pull raw tool I/O into recent-message snapshots.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text" and isinstance(block.get("text"), str):
                    parts.append(block["text"])
                elif "text" in block and isinstance(block["text"], str):
                    parts.append(block["text"])
        return " ".join(parts)
    return ""


def _extract_last_user_prompt(messages: List[Dict[str, Any]]) -> str:
    """Return text of the most recent user message that is NOT a tool_result.

    Used as the L1.5 ``prompt`` field for ingest.  Returns "" if no suitable
    user message exists.
    """
    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        if _is_tool_result(msg):
            continue
        return _extract_text_content(msg.get("content", ""))
    return ""


# ---------------------------------------------------------------------------
# Context Engine
# ---------------------------------------------------------------------------

from agent.context_engine import ContextEngine


class TencentDBOffloadEngine(ContextEngine):
    """Hermes ContextEngine backed by TencentDB offload V2 API.

    Lifecycle:
        1. ``update_from_response`` — track real token usage after each LLM call
        2. ``should_compress`` — trigger when usage exceeds threshold
        3. ``compress`` — POST messages to Gateway, get back compacted list
        4. (background) ``ingest_tool_pairs`` — fire-and-forget offload for L1/L2

    If the Gateway is unreachable, compression falls back to tail-truncation
    (keep system + last N messages), so the session never deadlocks.
    """

    # -- Identity --------------------------------------------------------

    @property
    def name(self) -> str:
        return "tencentdb-offload"

    # -- State (read by run_agent.py) ------------------------------------

    # inherited defaults; overridden as real data arrives
    # last_prompt_tokens, last_completion_tokens, last_total_tokens,
    # threshold_tokens, context_length, compression_count

    # -- Config ----------------------------------------------------------

    def __init__(self) -> None:
        self._gateway_url = os.environ.get(
            "TENCENTDB_OFFLOAD_GATEWAY_URL", "http://127.0.0.1:8420"
        ).rstrip("/")
        self._api_key = os.environ.get("TENCENTDB_OFFLOAD_API_KEY", "local")
        self._instance_id = os.environ.get(
            "TENCENTDB_OFFLOAD_INSTANCE_ID", "default"
        )
        self._compact_ratio = float(
            os.environ.get("TENCENTDB_OFFLOAD_COMPACT_RATIO", "0.5")
        )
        self._compact_timeout_ms = int(
            os.environ.get("TENCENTDB_OFFLOAD_TIMEOUT_MS", "90000")
        )
        self._ingest_timeout_ms = int(
            os.environ.get("TENCENTDB_OFFLOAD_INGEST_TIMEOUT_MS", "5000")
        )

        # Config validation
        if not self._gateway_url.startswith(("http://", "https://")):
            logger.error(
                "[tencentdb-offload] invalid gateway URL %r — must start with http:// or https://",
                self._gateway_url,
            )

        # Compaction defaults — visible to run_agent.py
        self.threshold_percent = float(
            os.environ.get("TENCENTDB_OFFLOAD_THRESHOLD", "0.75")
        )
        self.protect_first_n = 3
        self.protect_last_n = 6

        # Session tracking
        self._session_id: str = ""
        self._lock = threading.Lock()
        self._available: Optional[bool] = None  # lazy health check
        self._session_registry: Optional[SessionRegistry] = None  # Feature 5

        # L1.5 state (v0.5.0)
        self._cached_prompt: str = ""
        self._cached_recent_messages: List[Dict[str, str]] = []
        self._last_l15_hash: str = ""

        logger.info(
            "[tencentdb-offload] init: gateway=%s, instance=%s, ratio=%.2f",
            self._gateway_url,
            self._instance_id,
            self._compact_ratio,
        )

    # -- Deepcopy support (for v0.18.0 subagent fork) -------------------

    def __deepcopy__(self, memo: dict) -> "TencentDBOffloadEngine":
        """Create a copy suitable for child-agent context engine inheritance.

        v0.18.0 ``agent_init.py`` deep-copies the shared plugin singleton so a
        child agent's ``update_model()`` can't mutate the parent's compressor
        budget (#42449).  The default ``copy.deepcopy`` chokes on
        ``threading.Lock`` and would silently fall back to the built-in
        compressor.  We side-step that by constructing a fresh instance with
        config re-read from env vars — only the mutable budget state carried by
        the base ``ContextEngine`` (``last_prompt_tokens``,
        ``compression_count``, etc.) is copied across.
        """
        import copy as _copy

        cls = self.__class__
        new = cls.__new__(cls)
        memo[id(self)] = new

        # Re-read config from env (same as __init__) — ensures the child
        # picks up the same gateway / instance / ratio.
        new._gateway_url = self._gateway_url
        new._api_key = self._api_key
        new._instance_id = self._instance_id
        new._compact_ratio = self._compact_ratio
        new._compact_timeout_ms = self._compact_timeout_ms
        new._ingest_timeout_ms = self._ingest_timeout_ms

        # Mutable budget state — copy from parent so the child starts with the
        # same thresholds, not zeroes.
        new.threshold_percent = self.threshold_percent
        new.protect_first_n = self.protect_first_n
        new.protect_last_n = self.protect_last_n
        new.context_length = getattr(self, "context_length", 0)
        new.last_prompt_tokens = getattr(self, "last_prompt_tokens", 0)
        new.last_completion_tokens = getattr(self, "last_completion_tokens", 0)
        new.last_total_tokens = getattr(self, "last_total_tokens", 0)
        new.threshold_tokens = getattr(self, "threshold_tokens", 0)
        new.compression_count = self.compression_count

        # Fresh state — child agent gets its own lock + session tracking.
        new._session_id = ""  # child will bind its own session
        new._lock = threading.Lock()
        new._available = self._available  # reuse cached health-check result
        new._session_registry = None  # child starts with empty registry

        # L1.5 state — child starts fresh (no cached prompt/hash)
        new._cached_prompt = ""
        new._cached_recent_messages = []
        new._last_l15_hash = ""

        return new

    # -- HTTP helpers ----------------------------------------------------

    @property
    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "x-tdai-service-id": self._instance_id,
        }

    def _check_available(self) -> bool:
        """Lazy one-shot health check.  Cached after first result."""
        if self._available is not None:
            return self._available
        self._available = False  # default until proven otherwise
        try:
            req = urllib.request.Request(
                f"{self._gateway_url}/health", headers=self._headers
            )
            with urllib.request.urlopen(req, timeout=3) as resp:
                result = _json.loads(resp.read().decode("utf-8"))
                self._available = result.get("status") == "ok"
        except Exception as exc:
            logger.warning("[tencentdb-offload] health check failed: %s", exc)
            self._available = False

        if self._available:
            logger.info("[tencentdb-offload] Gateway reachable — offload active")
        else:
            logger.warning(
                "[tencentdb-offload] Gateway unreachable at %s — "
                "compression will fall back to tail-truncation",
                self._gateway_url,
            )
        return self._available

    # -- ContextEngine abstract methods ----------------------------------

    def update_from_response(self, usage: Dict[str, Any]) -> None:
        """Track real token usage from LLM API response."""
        with self._lock:
            self.last_prompt_tokens = usage.get("prompt_tokens", usage.get("input_tokens", 0))
            self.last_completion_tokens = usage.get(
                "completion_tokens", usage.get("output_tokens", 0)
            )
            self.last_total_tokens = usage.get("total_tokens", 0)
            if self.last_total_tokens == 0:
                self.last_total_tokens = self.last_prompt_tokens + self.last_completion_tokens

            # Update threshold based on context length
            if self.context_length > 0:
                self.threshold_tokens = int(self.context_length * self.threshold_percent)

    def should_compress(self, prompt_tokens: int = None) -> bool:
        """Trigger compaction when token usage exceeds threshold."""
        tokens = prompt_tokens if prompt_tokens is not None else self.last_prompt_tokens
        if self.threshold_tokens <= 0 or tokens <= 0:
            return False
        return tokens >= self.threshold_tokens

    def compress(
        self,
        messages: List[Dict[str, Any]],
        current_tokens: int = None,
        focus_topic: str = None,
    ) -> List[Dict[str, Any]]:
        """Compact messages via TencentDB offload server.

        Falls back to tail-truncation if server is unavailable.
        """
        # Snapshot mutable state under lock; the HTTP call below runs lock-free
        # so concurrent update_from_response() calls don't block for ~30s.
        # The base ContextEngine has no locking of its own, so this lock guards
        # every mutation of last_prompt_tokens / compression_count / _session_id.
        with self._lock:
            self.compression_count += 1
            count = self.compression_count
            last_prompt = self.last_prompt_tokens
            session_id = self._session_id or "hermes-default"

        tokens = current_tokens if current_tokens is not None else last_prompt or _estimate_tokens(messages)

        if not self._check_available():
            logger.info(
                "[tencentdb-offload] compress #%d: server unavailable, using fallback",
                count,
            )
            return self._fallback_compress(messages)

        logger.info(
            "[tencentdb-offload] compress #%d: %d messages, ~%d tokens, session=%s",
            count,
            len(messages),
            tokens,
            session_id,
        )

        # NEW: ingest full messages BEFORE _prepare_for_compact truncates them,
        # so tool pairs and recent context are preserved in TencentDB even if
        # the compaction step later drops them.  Fire-and-forget — failures
        # are logged but never block the compact flow.
        self._ingest_before_compact(messages, session_id)

        # Limit messages sent to Gateway to avoid HTTP body too large (Broken pipe).
        # Keep system + first N + last M, truncate middle tool results.
        send_msgs = self._prepare_for_compact(messages)

        result = _post_json(
            f"{self._gateway_url}/v2/offload/compact",
            {
                "session_id": session_id,
                "messages": send_msgs,
                "ratio": self._compact_ratio,
                # Compress hard: pick a context_window that lands the resolved level
                # at "emergency" (>= 0.95) once our threshold triggers. Emergency runs
                # the full cascade — fastpath + mild(entries-based) + aggressive +
                # emergency deletion — so compression works whether or not L1 has
                # finished populating entries.jsonl for this session's recent pairs.
                #
                # Math:
                #   - aggressive target  = floor(W * (0.85 - 0.05)) = 0.80 * W
                #   - emergency target   = floor(W * (0.85 - 0.10)) = 0.75 * W
                #   - we want final size = compactRatio * context_length
                #   - so W = compactRatio * context_length / 0.80  (rounds down through
                #     aggressive then emergency, leaving ~compactRatio * context_length)
                "context_window": max(
                    int(self.context_length * self._compact_ratio / 0.80), 100000
                ) if self.context_length else max(int(tokens / 0.7), 100000),
                "total_tokens": tokens,
                "instance": self._instance_id,
            },
            self._headers,
            self._compact_timeout_ms,
        )

        if result is None or result.get("code") != 0 or not result.get("data"):
            err_msg = result.get("message", "unknown") if result else "HTTP request failed"
            err_req = result.get("request_id", "?") if result else "?"
            logger.warning(
                "[tencentdb-offload] compact failed (code=%s, req=%s): %s — using fallback",
                result.get("code", "?") if result else "?",
                err_req,
                err_msg,
            )
            return self._fallback_compress(messages)

        compacted = result["data"].get("messages", [])
        report = result["data"].get("report", {})

        logger.info(
            "[tencentdb-offload] compact #%d done: %d→%d messages, level=%s, "
            "fastPath=%d/%d, mild=%d, aggressive=%d",
            count,
            len(messages),
            len(compacted),
            report.get("resolvedLevel", "?"),
            report.get("fastPathReplaced", 0),
            report.get("fastPathDeleted", 0),
            report.get("mildReplacements", 0),
            report.get("aggressiveDeleted", 0),
        )

        # Return compacted messages, or original if server returned empty
        if len(compacted) > 0:
            return compacted
        logger.warning("[tencentdb-offload] server returned 0 messages, keeping original")
        return messages

    # -- Message preparation for compact API ---------------------------------

    def _prepare_for_compact(
        self, messages: List[Dict[str, Any]], max_body_mb: float = 5.0
    ) -> List[Dict[str, Any]]:
        """Reduce message payload before sending to compact API.

        Strategy: send the FULL messages when possible so Gateway sees the real
        token ratio and picks the right compaction level. Only truncate when the
        serialized body exceeds ``max_body_mb`` (risk of Broken pipe / HTTP 413),
        and even then only truncate the largest tool_results to 2000 chars
        (matching Gateway's TOOL_RESULT_TRUNCATE_CHARS constant) — never destroy
        the whole conversation shape.
        """
        import json as _json

        def _body_mb(msgs: List[Dict[str, Any]]) -> float:
            return len(_json.dumps(msgs, ensure_ascii=False).encode("utf-8")) / (1024 * 1024)

        n = len(messages)
        before = _body_mb(messages)

        if before <= max_body_mb:
            logger.info(
                "[tencentdb-offload] prepare: %d msgs, %.2fMB ≤ %.2fMB cap — sending full",
                n, before, max_body_mb,
            )
            return messages

        # Body too large — truncate largest tool_results progressively until under cap.
        # Work on a deep copy so caller's messages aren't mutated.
        import copy
        result = copy.deepcopy(messages)
        TRUNC_CHARS = 2000  # matches Gateway's TOOL_RESULT_TRUNCATE_CHARS

        def _tool_result_size(msg: Dict[str, Any]) -> int:
            content = msg.get("content", "")
            if isinstance(content, str):
                return len(content)
            if isinstance(content, list):
                return sum(
                    len(b.get("text", b.get("content", ""))) if isinstance(b, dict) else 0
                    for b in content
                )
            return 0

        # Iteratively truncate the single largest tool_result until we fit
        while _body_mb(result) > max_body_mb:
            # Find the largest tool_result
            biggest_idx = -1
            biggest_size = 0
            for i, msg in enumerate(result):
                if not _is_tool_result(msg):
                    continue
                size = _tool_result_size(msg)
                if size > biggest_size and size > TRUNC_CHARS:
                    biggest_size = size
                    biggest_idx = i
            if biggest_idx < 0:
                break  # nothing left to truncate
            result[biggest_idx] = _truncate_tool_result(result[biggest_idx], max_chars=TRUNC_CHARS)

        after = _body_mb(result)
        logger.info(
            "[tencentdb-offload] prepare: %d msgs, %.2fMB → %.2fMB (largest tool_results→%d chars)",
            n, before, after, TRUNC_CHARS,
        )
        return result

    # -- Fallback compaction ---------------------------------------------

    def _fallback_compress(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Tail-truncation fallback when offload server is unavailable.

        Keeps system + first N + last (total - head) messages.
        """
        target_tokens = int((self.context_length or 200000) * self._compact_ratio)
        head_keep = self.protect_first_n
        tail_keep = self.protect_last_n

        if len(messages) <= head_keep + tail_keep:
            return messages

        # Scan from tail, accumulate tokens until target reached
        cum = 0
        cut = len(messages)
        for i in range(len(messages) - 1, head_keep - 1, -1):
            cum += _estimate_tokens([messages[i]])
            if cum > target_tokens:
                cut = i + 1
                break
            cut = i

        # Respect tool-call/tool-result pairs (don't split them)
        # If message at cut is a tool result, advance past the pair
        while cut < len(messages) and messages[cut].get("role") == "tool":
            cut += 1
        # If message before cut is assistant with tool_calls, pull back
        while cut > head_keep and cut < len(messages):
            prev = messages[cut - 1]
            if prev.get("role") == "assistant" and prev.get("tool_calls"):
                cut -= 1
            else:
                break

        if cut <= head_keep:
            return messages  # don't delete everything

        retained = messages[:head_keep] + messages[cut:]
        deleted = len(messages) - len(retained)
        logger.info(
            "[tencentdb-offload] fallback: deleted %d/%d messages, kept %d",
            deleted,
            len(messages),
            len(retained),
        )
        return retained

    # -- Optional preflight (inherited defaults are fine) -----------------

    # should_compress_preflight returns False by default — sufficient.
    # Override only if we want pre-LLM-call compaction.

    # -- Session binding (called by plugin lifecycle) ---------------------

    def bind_session(self, session_id: str) -> None:
        """Bind the current Hermes session ID for offload tracking.

        Resets health check cache so a new session re-probes the gateway.
        """
        with self._lock:
            self._session_id = session_id
            # _available is process-global: the gateway's reachability doesn't
            # depend on which session we're in, but resetting here lets a new
            # session recover automatically if the gateway was down at startup
            # and has since come back.
            self._available = None

    def on_session_start(self, session_id: str, **kwargs) -> None:
        """Base-class hook — bind the session ID directly.

        This is the canonical session-binding path. The plugin's
        ``register_session_hook`` in ``__init__.py`` is a fallback for
        older Hermes hosts that don't call this method on the engine.
        """
        self.bind_session(session_id)

    # -- Observability ----------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """Return engine status snapshot for diagnostics."""
        registry = self._session_registry
        return {
            "engine": "tencentdb-offload",
            "gateway_url": self._gateway_url,
            "instance_id": self._instance_id,
            "available": self._available,
            "session_id": self._session_id,
            "compression_count": self.compression_count,
            "last_prompt_tokens": self.last_prompt_tokens,
            "last_total_tokens": self.last_total_tokens,
            "threshold_tokens": self.threshold_tokens,
            "context_length": self.context_length,
            "compact_ratio": self._compact_ratio,
            # Feature 5: session registry
            "cached_sessions": registry.size if registry else 0,
            "features": ["pre_llm_call", "mmd_injection", "reclaimer", "session_registry"],
        }

    # -- Background ingestion (called by hook, not part of abstract) ------

    def ingest_tool_pairs(
        self,
        tool_pairs: List[Dict[str, Any]],
        prompt: str = None,
        recent_messages: Optional[List[Dict[str, str]]] = None,
    ) -> None:
        """Fire-and-forget: send tool-call/result pairs to offload server for L1 processing.

        Called after each tool execution.  Does not block the conversation.
        When prompt and recent_messages are provided, Gateway uses them for
        richer L1 extraction (ingestWithContext pattern).
        """
        if not self._check_available() or not tool_pairs:
            return

        session_id = self._session_id or "hermes-default"
        body: Dict[str, Any] = {
            "session_id": session_id,
            "tool_pairs": tool_pairs,
        }
        if prompt:
            body["prompt"] = prompt[:500]
        if recent_messages:
            body["recent_messages"] = [
                {"role": m["role"], "content": m["content"][:300]}
                for m in recent_messages[-5:]
            ]

        _post_json(
            f"{self._gateway_url}/v2/offload/ingest",
            body,
            self._headers,
            self._ingest_timeout_ms,
        )
        # Intentionally ignore result — fire and forget

    # -- Pre-compact ingestion (NEW — preserves info before truncation) ---

    def _extract_tool_pairs(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Scan ``messages`` and match assistant(tool_use) → tool(tool_result) pairs.

        Handles both OpenAI format (``tool_calls`` + role="tool" results) and
        Anthropic format (``tool_use`` / ``tool_result`` content blocks).
        Returns ``[{tool_name, tool_call_id, params, result, timestamp, duration_ms}]``.
        """
        try:
            import json as _json
        except ImportError:  # pragma: no cover — json is stdlib
            _json = None

        pending: Dict[str, Dict[str, Any]] = {}  # tool_call_id → call metadata
        pairs: List[Dict[str, Any]] = []

        for msg in messages:
            role = msg.get("role", "")

            # ---- assistant: collect tool_use calls (OpenAI + Anthropic) ----
            if role == "assistant":
                # OpenAI: tool_calls=[{id, function:{name, arguments}}]
                for tc in msg.get("tool_calls") or []:
                    if not isinstance(tc, dict):
                        continue
                    tc_id = tc.get("id", "")
                    if not tc_id:
                        continue
                    fn = tc.get("function") or {}
                    raw_args = fn.get("arguments", "{}")
                    if isinstance(raw_args, str):
                        try:
                            params = _json.loads(raw_args) if _json else raw_args
                        except (ValueError, TypeError):
                            params = raw_args
                    elif isinstance(raw_args, dict):
                        params = raw_args
                    else:
                        params = {}
                    pending[tc_id] = {
                        "tool_name": fn.get("name", ""),
                        "params": params,
                        "timestamp": msg.get("timestamp", ""),
                    }
                # Anthropic: content=[{type:"tool_use", id, name, input}]
                content = msg.get("content")
                if isinstance(content, list):
                    for block in content:
                        if not isinstance(block, dict):
                            continue
                        if block.get("type") != "tool_use":
                            continue
                        tc_id = block.get("id", "")
                        if not tc_id:
                            continue
                        pending[tc_id] = {
                            "tool_name": block.get("name", ""),
                            "params": block.get("input", {}) or {},
                            "timestamp": msg.get("timestamp", ""),
                        }

            # ---- OpenAI tool result: role in {tool, function, ...} ----
            elif role in ("tool", "function", "toolResult", "tool_result"):
                tc_id = msg.get("tool_call_id", "")
                if tc_id and tc_id in pending:
                    call = pending.pop(tc_id)
                    pairs.append(self._build_tool_pair(
                        call["tool_name"], tc_id, call["params"],
                        msg.get("content", ""), call["timestamp"],
                        msg.get("duration_ms"),
                    ))

            # ---- Anthropic tool_result: user msg with tool_result blocks ----
            elif role == "user" and isinstance(msg.get("content"), list):
                for block in msg["content"]:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") != "tool_result":
                        continue
                    tc_id = block.get("tool_use_id", "")
                    if not tc_id or tc_id not in pending:
                        continue
                    raw = block.get("content", "")
                    if isinstance(raw, list):
                        result_text = " ".join(
                            b.get("text", "") for b in raw
                            if isinstance(b, dict) and isinstance(b.get("text"), str)
                        )
                    else:
                        result_text = str(raw)
                    call = pending.pop(tc_id)
                    pairs.append(self._build_tool_pair(
                        call["tool_name"], tc_id, call["params"],
                        result_text, call["timestamp"],
                        msg.get("duration_ms"),
                    ))

        return pairs

    @staticmethod
    def _build_tool_pair(
        tool_name: str,
        tool_call_id: str,
        params: Any,
        result: str,
        timestamp: Any,
        duration_ms: Any,
    ) -> Dict[str, Any]:
        """Build a tool-pair payload entry, truncating huge results."""
        pair: Dict[str, Any] = {
            "tool_name": tool_name,
            "tool_call_id": tool_call_id,
            "params": params,
            "result": result[:2000] if isinstance(result, str) else str(result)[:2000],
            "timestamp": timestamp or "",
        }
        if duration_ms is not None:
            pair["duration_ms"] = duration_ms
        return pair

    def _build_recent_messages(
        self, messages: List[Dict[str, Any]], max_msgs: int = 10
    ) -> List[Dict[str, Any]]:
        """Return recent user/assistant text messages, each truncated to 400 chars.

        Mirrors OpenClaw ``buildRecentMessages``: skips tool messages, tool_result
        user messages, assistant tool_use-only messages, heartbeats, and very
        short messages.  Keeps the last ``max_msgs`` entries.
        """
        out: List[Dict[str, Any]] = []
        for msg in messages:
            role = msg.get("role", "")

            # Skip OpenAI-format tool results
            if role in ("tool", "function", "toolResult", "tool_result"):
                continue
            # Skip Anthropic-format tool_result (user msg with tool_result blocks)
            if _is_tool_result(msg):
                continue

            if role == "user":
                text = _extract_text_content(msg.get("content", ""))
                if len(text) <= 5:
                    continue
            elif role == "assistant":
                text = _extract_text_content(msg.get("content", ""))
                if len(text) <= 10:
                    continue
            else:
                # Skip system / unknown roles
                continue

            if "HEARTBEAT" in text or "heartbeat" in text:
                continue

            out.append({"role": role, "content": text[:400]})

        return out[-max_msgs:]

    # -- L1.5 + ingestWithContext + localCompact (v0.5.0) ----------------

    def _is_internal_prompt(self, prompt: str) -> bool:
        """Detect internal prompts that should NOT trigger L1.5."""
        if prompt.startswith("Pre-compaction"):
            return True
        if prompt.startswith("[Inter-session"):
            return True
        if "HEARTBEAT" in prompt or "heartbeat" in prompt:
            return True
        return False

    def _format_context_for_l1(self, prompt: str, recent_messages: List[Dict[str, str]]) -> str:
        """Format prompt + recent_messages into a context string for L1 extraction."""
        parts = [f"User: {prompt[:500]}"]
        if recent_messages:
            parts.append("")
            parts.append("Recent:")
            for m in recent_messages[-5:]:
                role = m.get("role", "?")
                content = m.get("content", "")[:200]
                parts.append(f"  {role}: {content}")
        return "\n".join(parts)

    def _trigger_l15_if_needed(
        self, messages: List[Dict[str, Any]], session_id: str
    ) -> None:
        """Fire-and-forget L1.5: send prompt + recent_messages to Gateway.

        Mirrors OpenClaw ``triggerL15IfNeeded``.  Runs on every new user
        prompt so Gateway has real-time context for L1 extraction.
        """
        prompt = _extract_last_user_prompt(messages)
        if not prompt:
            logger.info("[tencentdb-offload] L1.5 skip: no prompt extracted")
            return
        if self._is_internal_prompt(prompt):
            logger.info("[tencentdb-offload] L1.5 skip: internal prompt")
            return

        # Update cache for ingestWithContext
        recent = self._build_recent_messages(messages, max_msgs=5)
        self._cached_prompt = prompt[:500]
        self._cached_recent_messages = recent

        # Dedup: skip if same prompt hash
        import hashlib
        h = hashlib.md5(prompt.encode()).hexdigest()[:16]
        if h == self._last_l15_hash:
            logger.info("[tencentdb-offload] L1.5 skip: dup hash=%s", h)
            return
        self._last_l15_hash = h

        if not self._check_available():
            logger.info("[tencentdb-offload] L1.5 skip: gateway not available")
            return

        logger.info(
            "[tencentdb-offload] L1.5 firing: hash=%s, session=%s, prompt=%s, recent=%d",
            h, session_id, prompt[:80], len(recent),
        )

        # Fire-and-forget in daemon thread
        def _fire():
            try:
                resp = _post_json(
                    f"{self._gateway_url}/v2/offload/ingest",
                    {
                        "session_id": session_id,
                        "tool_pairs": [],
                        "prompt": prompt[:500],
                        "recent_messages": [
                            {"role": m["role"], "content": m["content"][:300]}
                            for m in recent[-5:]
                        ] if recent else [],
                    },
                    self._headers,
                    self._ingest_timeout_ms,
                )
                logger.info(
                    "[tencentdb-offload] L1.5 sent OK: hash=%s, resp=%s",
                    h, str(resp)[:200] if resp else "(empty)",
                )
            except Exception as exc:
                logger.warning("[tencentdb-offload] L1.5 failed: %s", exc)

        threading.Thread(target=_fire, daemon=True).start()

    def _trigger_l15_from_cache(self, session_id: str) -> None:
        """L1.5 fallback: trigger from post_tool_call.

        Hermes v0.18.0 never fires pre_llm_call, so we use this method
        in post_tool_call to send L1.5 to Gateway.  Reads the latest
        user prompt from Hermes state.db.
        """
        # Read latest user message from Hermes state.db
        prompt = self._read_latest_user_prompt()
        if not prompt or self._is_internal_prompt(prompt):
            return

        # Also cache for ingestWithContext
        self._cached_prompt = prompt[:500]

        import hashlib
        h = hashlib.md5(prompt.encode()).hexdigest()[:16]
        if h == self._last_l15_hash:
            return
        self._last_l15_hash = h

        if not self._check_available():
            return

        recent = self._cached_recent_messages or []
        logger.info(
            "[tencentdb-offload] L1.5 (post_tool_call) firing: hash=%s, session=%s, prompt=%s",
            h, session_id, prompt[:80],
        )

        def _fire():
            try:
                resp = _post_json(
                    f"{self._gateway_url}/v2/offload/ingest",
                    {
                        "session_id": session_id,
                        "tool_pairs": [],
                        "prompt": prompt[:500],
                        "recent_messages": [
                            {"role": m["role"], "content": m["content"][:300]}
                            for m in recent[-5:]
                        ] if recent else [],
                    },
                    self._headers,
                    self._ingest_timeout_ms,
                )
                logger.info(
                    "[tencentdb-offload] L1.5 (post_tool_call) sent OK: hash=%s, resp=%s",
                    h, str(resp)[:200] if resp else "(empty)",
                )
            except Exception as exc:
                logger.warning("[tencentdb-offload] L1.5 (post_tool_call) failed: %s", exc)

        threading.Thread(target=_fire, daemon=True).start()

    def _read_latest_user_prompt(self) -> Optional[str]:
        """Read latest user message from Hermes state.db."""
        try:
            import sqlite3
            import os
            db_path = os.path.expanduser("~/.hermes/state.db")
            if not os.path.exists(db_path):
                return None
            conn = sqlite3.connect(db_path, timeout=2)
            row = conn.execute(
                "SELECT content FROM messages WHERE role='user' "
                "AND content IS NOT NULL AND content != '' "
                "ORDER BY timestamp DESC LIMIT 1"
            ).fetchone()
            conn.close()
            if row and row[0]:
                return str(row[0])[:500]
        except Exception as exc:
            logger.debug("[tencentdb-offload] _read_latest_user_prompt error: %s", exc)
        return None

    def _local_compact(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Tool-pair-aware local compaction (fallback when server unavailable).

        Mirrors OpenClaw ``localCompact()``: scans from tail to find cut index,
        then expands cut boundary to preserve tool_use/tool_result pairs.
        """
        target_tokens = int((self.context_length or 200000) * self._compact_ratio)
        head_keep = self.protect_first_n
        n = len(messages)

        if n <= head_keep + 2:
            return messages

        # Step 1: scan from tail, accumulate tokens
        cum = 0
        cut = n
        for i in range(n - 1, head_keep - 1, -1):
            cum += _estimate_tokens([messages[i]])
            if cum > target_tokens:
                cut = i + 1
                break
            cut = i

        # Step 2: expand cut to respect tool pairs
        # 2a: if msg at cut is a tool_result, advance past the pair
        while cut < n and _is_tool_result(messages[cut]):
            cut += 1

        # 2b: if msg at cut-1 (last deleted) is assistant with tool_use, pull back
        while cut > head_keep and cut < n:
            prev = messages[cut - 1]
            if prev.get("role") == "assistant":
                content = prev.get("content")
                if isinstance(content, list) and any(
                    isinstance(b, dict) and b.get("type") == "tool_use"
                    for b in content
                ):
                    cut -= 1
                    continue
            break

        if cut <= head_keep:
            return messages  # don't delete everything

        retained = messages[:head_keep] + messages[cut:]
        deleted = n - len(retained)
        logger.info(
            "[tencentdb-offload] local_compact: deleted %d/%d msgs, kept %d, target=%d tokens",
            deleted, n, len(retained), target_tokens,
        )
        return retained

    def _ingest_before_compact(
        self, messages: List[Dict[str, Any]], session_id: str
    ) -> None:
        """Defensive backup ingest — primary ingest path is the post_tool_call hook.

        The hook fires per-tool-call and gives L1 time to populate entries.jsonl
        before compression triggers. This method catches two edge cases the hook
        misses:
          1. Tool pairs from before the plugin loaded (e.g. session restore).
          2. Hermes hosts where post_tool_call isn't emitted.

        Fire-and-forget — failures are logged and swallowed so ``compress()``
        continues to the compact step uninterrupted.
        """
        if not self._check_available():
            return

        try:
            tool_pairs = self._extract_tool_pairs(messages)
            recent_messages = self._build_recent_messages(messages)
            prompt = _extract_last_user_prompt(messages)

            if not tool_pairs and not recent_messages and not prompt:
                logger.debug("[tencentdb-offload] ingest: nothing to send")
                return

            body: Dict[str, Any] = {
                "session_id": session_id,
                "tool_pairs": tool_pairs,
                "recent_messages": recent_messages,
            }
            if prompt:
                body["prompt"] = prompt[:500]

            logger.info(
                "[tencentdb-offload] ingest: %d tool_pairs, %d recent_messages",
                len(tool_pairs),
                len(recent_messages),
            )

            result = _post_json(
                f"{self._gateway_url}/v2/offload/ingest",
                body,
                self._headers,
                self._ingest_timeout_ms,
            )
            if result is None:
                logger.warning(
                    "[tencentdb-offload] ingest failed: HTTP request failed "
                    "(continuing with compact)"
                )
            elif result.get("code", 0) != 0:
                logger.warning(
                    "[tencentdb-offload] ingest failed: code=%s msg=%s "
                    "(continuing with compact)",
                    result.get("code", "?"),
                    result.get("message", "unknown"),
                )
        except Exception as exc:
            logger.warning(
                "[tencentdb-offload] ingest failed: %s (continuing with compact)", exc
            )

    # -- New session carry-over ------------------------------------------

    def carry_over_new_session_context(
        self, old_session_id: str, new_session_id: str
    ) -> int:
        """Carry context reference when Hermes creates a new session."""
        self._session_id = new_session_id
        return 0

    # -- Model update -----------------------------------------------------

    def update_model(self, *args, **kwargs) -> None:
        """Called when model changes — update context length and threshold.

        Accepts variable positional/keyword args because different Hermes
        versions call this with different parameter counts.
        """
        model = args[0] if args else kwargs.get("model", "unknown")
        context_length = args[1] if len(args) > 1 else kwargs.get("context_length", 0)
        self.context_length = context_length
        self.threshold_tokens = int(context_length * self.threshold_percent)
        logger.info(
            "[tencentdb-offload] model=%s, context_length=%d, threshold=%d",
            model,
            context_length,
            self.threshold_tokens,
        )

    # ======================================================================
    # Feature 1: pre_llm_call — L3 incremental compression + fastpath
    # ======================================================================

    def pre_llm_call(self, messages: List[Dict[str, Any]], **kwargs) -> Optional[Dict[str, Any]]:
        """Called before each LLM call (via pre_llm_call hook).

        Mirrors OpenClaw's ``before_prompt_build`` / ``llm_input_l3``:
          1. L1.5 trigger (fire-and-forget prompt + recent_messages to Gateway)
          2. Filter heartbeat messages
          3. Request MMD injection data from Gateway
          4. If approaching threshold, trigger incremental compression
        """
        if not messages:
            return None

        session_id = self._session_id or kwargs.get("session_id", "hermes-default")
        if session_id and session_id != self._session_id:
            self.bind_session(session_id)

        # Step 0: L1.5 — send prompt + recent_messages to Gateway (fire-and-forget)
        try:
            self._trigger_l15_if_needed(messages, session_id)
        except Exception as exc:
            logger.debug("[tencentdb-offload] L1.5 error: %s", exc)

        # Step 1: Filter heartbeat messages (best-effort, in-place)
        self._filter_heartbeat_messages(messages)

        # Step 2: Request MMD injection from Gateway
        mmd_injected = self._inject_mmd_from_gateway(messages, session_id)

        # Step 3: Incremental compression check
        # If we're approaching the threshold (>60%), request a light compact
        # to do fastpath replay without waiting for the full threshold trigger.
        estimated = _estimate_tokens(messages)
        soft_threshold = int((self.threshold_tokens or self.context_length or 200000) * 0.60)
        if soft_threshold > 0 and estimated >= soft_threshold and self._check_available():
            logger.info(
                "[tencentdb-offload] pre_llm_call: %d tokens >= %d soft threshold — requesting incremental compact",
                estimated, soft_threshold,
            )
            # Use compact with ratio to do fastpath replay + mild compression
            # This is lighter than the full threshold-triggered compact()
            try:
                self._incremental_compact(messages, session_id, estimated)
            except Exception as exc:
                logger.debug("[tencentdb-offload] pre_llm_call incremental compact error: %s", exc)

        return None  # no context injection needed — modifications are in-place

    def _filter_heartbeat_messages(self, messages: List[Dict[str, Any]]) -> int:
        """Remove heartbeat tool_use/tool_result pairs from messages.

        Mirrors OpenClaw's ``filterHeartbeatMessages``.  Heartbeat tool calls
        are internal probes that add noise to the context.  Removing them
        saves tokens and avoids confusing the LLM.

        Returns count of removed messages.
        """
        # Collect heartbeat tool_use IDs
        heartbeat_ids: set = set()
        for msg in messages:
            role = msg.get("role", "")
            if role != "assistant":
                continue
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") not in ("tool_use", "toolCall"):
                    continue
                raw_input = block.get("input", block.get("arguments", ""))
                raw_str = str(raw_input) if not isinstance(raw_input, str) else raw_input
                if "HEARTBEAT" in raw_str:
                    heartbeat_ids.add(block.get("id", ""))

        if not heartbeat_ids:
            return 0

        removed = 0
        for i in range(len(messages) - 1, -1, -1):
            msg = messages[i]
            role = msg.get("role", "")

            # OpenAI tool result with heartbeat tool_call_id
            if role in ("tool", "function", "toolResult", "tool_result"):
                tc_id = msg.get("tool_call_id", "")
                if tc_id and tc_id in heartbeat_ids:
                    messages.pop(i)
                    removed += 1
                    continue

            # Anthropic user msg with tool_result blocks
            if role == "user" and isinstance(msg.get("content"), list):
                new_blocks = []
                msg_removed = False
                for block in msg["content"]:
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        if block.get("tool_use_id", "") in heartbeat_ids:
                            msg_removed = True
                            continue
                    new_blocks.append(block)
                if msg_removed:
                    if new_blocks:
                        messages[i] = dict(msg)
                        messages[i]["content"] = new_blocks
                    else:
                        messages.pop(i)
                    removed += 1
                    continue

            # Assistant with heartbeat tool_use blocks
            if role == "assistant" and isinstance(msg.get("content"), list):
                new_blocks = []
                msg_changed = False
                for block in msg["content"]:
                    if isinstance(block, dict) and block.get("type") in ("tool_use", "toolCall"):
                        if block.get("id", "") in heartbeat_ids:
                            msg_changed = True
                            continue
                    new_blocks.append(block)
                if msg_changed:
                    if new_blocks:
                        messages[i] = dict(msg)
                        messages[i]["content"] = new_blocks
                    else:
                        messages.pop(i)
                    removed += 1

        if removed > 0:
            logger.info("[tencentdb-offload] filtered %d heartbeat messages", removed)
        return removed

    def _incremental_compact(
        self, messages: List[Dict[str, Any]], session_id: str, estimated_tokens: int
    ) -> None:
        """Request lightweight incremental compression from Gateway.

        Unlike the full compress(), this targets a higher ratio (keeping more)
        and focuses on fastpath replay + mild replacement.  Used in pre_llm_call
        when approaching but not yet at the threshold.
        """
        send_msgs = self._prepare_for_compact(messages, max_body_mb=0.5)

        result = _post_json(
            f"{self._gateway_url}/v2/offload/compact",
            {
                "session_id": session_id,
                "messages": send_msgs,
                "ratio": 0.65,  # lighter than the full compact ratio
                "context_window": max(
                    int((self.context_length or 200000) * 0.65 / 0.80), 100000
                ),
                "total_tokens": estimated_tokens,
                "instance": self._instance_id,
            },
            self._headers,
            30000,  # shorter timeout for incremental
        )

        if result is None or result.get("code") != 0 or not result.get("data"):
            return

        compacted = result["data"].get("messages", [])
        if not compacted:
            return

        report = result["data"].get("report", {})
        # Only apply if the compacted list is meaningfully shorter
        if len(compacted) < len(messages):
            messages.clear()
            messages.extend(compacted)
            logger.info(
                "[tencentdb-offload] incremental compact: %d→%d messages (level=%s)",
                len(send_msgs),
                len(compacted),
                report.get("resolvedLevel", "?"),
            )

    # ======================================================================
    # Feature 2: L2 Mermaid canvas injection
    # ======================================================================

    _MMD_CONTEXT_MARKER = "_mmdContextMessage"
    _MMD_INJECTION_MARKER = "_mmdInjection"

    def _inject_mmd_from_gateway(
        self, messages: List[Dict[str, Any]], session_id: str
    ) -> bool:
        """Fetch MMD files from Gateway and inject context messages.

        Calls POST /v2/offload/query-mmd to get active + history MMD files,
        then injects them as user messages with <current_task_context> and
        <history_task_context> tags.

        Mirrors OpenClaw's ``injectActiveMmd`` + ``injectHistoryMmds`` from
        ``mmd-injector.ts`` and ``buildHistoryMmdInjection`` from
        ``llm-input-l3.ts``.

        Returns True if any MMD was injected.
        """
        if not self._check_available():
            return False

        result = _post_json(
            f"{self._gateway_url}/v2/offload/query-mmd",
            {"session_id": session_id},
            self._headers,
            5000,
        )

        if result is None or result.get("code") != 0 or not result.get("data"):
            return False

        mmd_data = result["data"]
        mmds = mmd_data.get("mmds", [])
        current_mmd = mmd_data.get("currentMmd")

        if not mmds:
            return False

        # Remove existing MMD injections (version dedup)
        self._remove_existing_mmd_injections(messages)

        injected = False
        token_budget = int((self.context_length or 200000) * 0.05)  # 5% of context for MMD

        # Separate active from history MMDs
        active_mmd_content = None
        active_mmd_filename = None
        active_version = None
        history_mmds: List[Dict[str, str]] = []

        for mmd in mmds:
            filename = mmd.get("filename", "")
            content = mmd.get("content", "")
            version = mmd.get("version", "")
            if not content or not content.strip():
                continue

            if filename == current_mmd:
                active_mmd_content = content
                active_mmd_filename = filename
                active_version = version
            else:
                history_mmds.append({"filename": filename, "content": content, "version": version})

        # Inject history MMDs first (oldest → newest)
        hist_injected = self._inject_history_mmds(
            messages, history_mmds, token_budget
        )

        # Inject active MMD
        active_injected = False
        if active_mmd_content:
            active_injected = self._inject_active_mmd(
                messages, active_mmd_filename, active_mmd_content, active_version
            )

        injected = hist_injected or active_injected
        if injected:
            logger.info(
                "[tencentdb-offload] MMD injection: active=%s, history=%d",
                "yes" if active_injected else "no",
                hist_injected,
            )
        return injected

    def _inject_active_mmd(
        self,
        messages: List[Dict[str, Any]],
        filename: Optional[str],
        content: str,
        version: Optional[str],
    ) -> bool:
        """Inject the active MMD as a <current_task_context> message.

        Mirrors OpenClaw's ``injectActiveMmd`` + ``buildActiveMmdText``.
        """
        text = self._build_active_mmd_text(filename, content)

        mmd_msg: Dict[str, Any] = {
            "role": "user",
            "content": text,
            self._MMD_CONTEXT_MARKER: "active",
            "_mmdVersion": version or _hash_content(content),
            "_mmdFilename": filename,
        }

        insert_idx = self._find_active_mmd_insertion_point(messages)
        messages.insert(insert_idx, mmd_msg)
        return True

    def _inject_history_mmds(
        self,
        messages: List[Dict[str, Any]],
        history_mmds: List[Dict[str, str]],
        token_budget: int,
    ) -> int:
        """Inject history MMD files for deleted/aggressive-compressed entries.

        Mirrors OpenClaw's ``injectHistoryMmds`` + ``buildHistoryMmdText``.
        Injects newest first, respects token budget.
        """
        if not history_mmds:
            return 0

        # Sort newest first (by version), then inject oldest-first in messages
        history_mmds.sort(key=lambda m: m.get("version", ""), reverse=True)

        injected: List[Dict[str, Any]] = []
        used_tokens = 0

        for mmd in history_mmds:
            filename = mmd.get("filename", "")
            content = mmd.get("content", "")
            text = self._build_history_mmd_text(filename, content)
            tokens = len(text) // 3  # rough estimate

            if used_tokens + tokens > token_budget:
                continue

            injected.append({
                "role": "user",
                "content": text,
                self._MMD_INJECTION_MARKER: True,
                "_mmdFilename": filename,
            })
            used_tokens += tokens

        if not injected:
            return 0

        # Reverse to chronological order (oldest first)
        injected.reverse()

        insert_idx = self._find_history_mmd_insertion_point(messages)
        for i, msg in enumerate(injected):
            messages.insert(insert_idx + i, msg)

        return len(injected)

    def _remove_existing_mmd_injections(self, messages: List[Dict[str, Any]]) -> int:
        """Remove existing MMD injection messages for version dedup.

        Mirrors OpenClaw's ``removeExistingMmdInjections`` + ``removeMmdMessages``.
        """
        removed = 0
        for i in range(len(messages) - 1, -1, -1):
            msg = messages[i]
            if msg.get(self._MMD_INJECTION_MARKER) or msg.get(self._MMD_CONTEXT_MARKER):
                messages.pop(i)
                removed += 1
        return removed

    def _find_active_mmd_insertion_point(self, messages: List[Dict[str, Any]]) -> int:
        """Find the insertion point for the active MMD message.

        Strategy: insert after the latest user message in the second half,
        or before the trailing tool loop.  Never before system message.
        Mirrors OpenClaw's ``findActiveMmdInsertionPoint``.
        """
        if len(messages) <= 2:
            return min(1, len(messages))

        # Find latest non-MMD user message
        latest_user_idx = -1
        for i in range(len(messages) - 1, -1, -1):
            msg = messages[i]
            if msg.get(self._MMD_CONTEXT_MARKER) or msg.get(self._MMD_INJECTION_MARKER):
                continue
            if msg.get("role") == "user":
                latest_user_idx = i
                break

        if latest_user_idx >= 0:
            if latest_user_idx == len(messages) - 1:
                insert_idx = latest_user_idx  # before last (user prompt stays last)
            else:
                insert_idx = latest_user_idx + 1
        else:
            # Fallback: in the second half
            insert_idx = max(1, len(messages) - 30)

        # Guard: don't insert between assistant(tool_use) and its tool_result
        insert_idx = self._adjust_for_tool_call_pair(messages, insert_idx)

        # Never insert before system message
        if insert_idx == 0 and messages and messages[0].get("role") == "system":
            insert_idx = 1

        return insert_idx

    def _find_history_mmd_insertion_point(self, messages: List[Dict[str, Any]]) -> int:
        """Find insertion point for history MMD (before active MMD)."""
        for i, msg in enumerate(messages):
            if msg.get(self._MMD_CONTEXT_MARKER) == "active":
                return i
        return self._find_active_mmd_insertion_point(messages)

    def _adjust_for_tool_call_pair(
        self, messages: List[Dict[str, Any]], insert_idx: int
    ) -> int:
        """Adjust insertion index to avoid splitting tool_call/tool_result pairs."""
        if insert_idx <= 0 or insert_idx >= len(messages):
            return insert_idx

        msg = messages[insert_idx]
        role = msg.get("role", "")

        # Don't insert at a tool_result position
        if _is_tool_result(msg):
            # Walk back past the assistant tool_use
            i = insert_idx - 1
            while i >= 0:
                prev_role = messages[i].get("role", "")
                if prev_role == "assistant":
                    return i
                if not _is_tool_result(messages[i]):
                    break
                i -= 1

        # Don't insert right after assistant with tool_calls (before its tool_result)
        if insert_idx > 0:
            prev = messages[insert_idx - 1]
            if prev.get("role") == "assistant" and prev.get("tool_calls"):
                return insert_idx - 1

        return insert_idx

    @staticmethod
    def _build_active_mmd_text(filename: Optional[str], content: str) -> str:
        """Build active MMD injection text with <current_task_context> tag.

        Mirrors OpenClaw's ``buildActiveMmdText``.
        """
        task_goal = ""
        import re
        meta_match = re.match(r"^%%\{\s*(.*?)\s*\}%%", content)
        if meta_match:
            try:
                import json as _json
                meta = _json.loads("{" + meta_match.group(1) + "}")
                task_goal = meta.get("taskGoal", "")
            except Exception:
                pass

        lines = [
            "<current_task_context>",
            "【当前活跃任务的mermaid流程图】这是你最近正在执行的任务的阶段性记录。",
        ]
        if task_goal:
            lines.append(f"**任务目标:** {task_goal}")
        if filename:
            lines.append(f"**任务文件:** {filename}")
        lines.extend([
            "```mermaid",
            content,
            "```",
            "标记为 \"doing\" 的节点是近期焦点，\"done\" 的已完成。请参考此保持方向感，避免重复已完成的工作。",
            "</current_task_context>",
        ])
        return "\n".join(lines)

    @staticmethod
    def _build_history_mmd_text(filename: str, content: str) -> str:
        """Build history MMD injection text with <history_task_context> tag.

        Mirrors OpenClaw's ``buildHistoryMmdText``.
        """
        task_goal = ""
        import re
        meta_match = re.match(r"^%%\{\s*(.*?)\s*\}%%", content)
        if meta_match:
            try:
                import json as _json
                meta = _json.loads("{" + meta_match.group(1) + "}")
                task_goal = meta.get("taskGoal", "")
            except Exception:
                pass

        lines = [
            f'<history_task_context file="{filename}">',
            "【历史任务记录】以下是此前完成的任务的概要。",
        ]
        if task_goal:
            lines.append(f"**任务目标:** {task_goal}")
        lines.extend([
            "",
            "```mermaid",
            content,
            "```",
            "</history_task_context>",
        ])
        return "\n".join(lines)

    # ======================================================================
    # Feature 3: Reclaimer — cleanup of stale session data
    # ======================================================================

    def reclaim(self, retention_days: int = 7) -> Dict[str, int]:
        """Reclaim stale session data by calling Gateway cleanup.

        In the official OpenClaw plugin, the reclaimer runs a 5-step cleanup
        over local disk (expired JSONL, orphan refs, expired MMDs, log rotation,
        registry pruning).  Since our plugin delegates storage to the Gateway,
        we trigger cleanup via a Gateway request.

        Also clears local session registry entries for sessions older than
        ``retention_days``.
        """
        stats = {"local_sessions_pruned": 0}

        if retention_days < 3:
            return stats

        cutoff = time.time() - retention_days * 86400

        # Prune local session registry
        if self._session_registry:
            pruned = self._session_registry.prune_expired(cutoff)
            stats["local_sessions_pruned"] = pruned
            logger.info("[tencentdb-offload] reclaim: pruned %d expired local sessions", pruned)

        return stats

    # ======================================================================
    # Feature 4: pre_llm_call hook (registered in __init__.py)
    # ======================================================================
    # See pre_llm_call method above — registered via pre_llm_call hook
    # in __init__.py.

    # ======================================================================
    # Feature 5: SessionRegistry — multi-session state management
    # ======================================================================

    def get_session_registry(self) -> "SessionRegistry":
        """Return the SessionRegistry, creating one if needed."""
        if self._session_registry is None:
            self._session_registry = SessionRegistry(max_cached=20)
        return self._session_registry


# ---------------------------------------------------------------------------
# Helper: content hashing (for MMD version dedup)
# ---------------------------------------------------------------------------

def _hash_content(s: str) -> str:
    """Hash content for MMD version dedup.  Mirrors OpenClaw hashContent."""
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:12]


# ---------------------------------------------------------------------------
# SessionRegistry — per-session state with LRU eviction
# ---------------------------------------------------------------------------

class SessionRegistry:
    """Per-session state tracking with LRU eviction.

    Mirrors OpenClaw's ``SessionRegistry`` (session-registry.ts).  Each
    session gets an isolated ``SessionState`` holding processed tool call IDs,
    offloaded IDs, and MMD injection versions.  LRU eviction keeps memory
    bounded.

    In our Gateway-backed architecture, the heavy state lives server-side.
    This registry tracks lightweight client-side state for fastpath replay
    and MMD version dedup.
    """

    def __init__(self, max_cached: int = 20) -> None:
        self._sessions: Dict[str, SessionState] = {}
        self._max_cached = max_cached
        self._lock = threading.Lock()

    def resolve(self, session_id: str) -> SessionState:
        """Get or create a per-session state.  Thread-safe with LRU eviction."""
        with self._lock:
            entry = self._sessions.get(session_id)
            if entry is not None:
                entry.last_access_ms = time.time() * 1000
                return entry

            entry = SessionState(session_id=session_id)
            self._sessions[session_id] = entry

            # LRU eviction
            if len(self._sessions) > self._max_cached:
                self._evict_oldest()

            return entry

    def get(self, session_id: str) -> Optional[SessionState]:
        """Look up an existing session (does not create)."""
        with self._lock:
            entry = self._sessions.get(session_id)
            if entry is not None:
                entry.last_access_ms = time.time() * 1000
            return entry

    def prune_expired(self, cutoff_epoch_s: float) -> int:
        """Remove sessions whose last access predates cutoff.  Thread-safe."""
        with self._lock:
            to_remove = [
                sid for sid, state in self._sessions.items()
                if state.last_access_ms / 1000 < cutoff_epoch_s
            ]
            for sid in to_remove:
                del self._sessions[sid]
            return len(to_remove)

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._sessions)

    def keys(self):
        with self._lock:
            return list(self._sessions.keys())

    def values(self):
        with self._lock:
            return list(self._sessions.values())

    def _evict_oldest(self) -> None:
        """Evict the least-recently-accessed session.  Caller holds lock."""
        oldest_key: Optional[str] = None
        oldest_ms = float("inf")
        for sid, state in self._sessions.items():
            if state.last_access_ms < oldest_ms:
                oldest_ms = state.last_access_ms
                oldest_key = sid
        if oldest_key is not None:
            del self._sessions[oldest_key]


class SessionState:
    """Per-session state for offload tracking.

    Mirrors a subset of OpenClaw's ``OffloadStateManager`` that's relevant
    for client-side operations (fastpath replay, MMD dedup, processed ID
    tracking).  Server-side state (entries.jsonl, offload entries) lives
    on the Gateway.
    """

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.last_access_ms: float = time.time() * 1000

        # Tool call IDs already processed (dedup prevention)
        self.processed_tool_call_ids: set = set()
        # Tool call IDs confirmed offloaded (fastpath replay candidates)
        self.confirmed_offload_ids: set = set()
        # Tool call IDs that were aggressively deleted
        self.deleted_offload_ids: set = set()

        # MMD injection version tracking (filename → version hash)
        self.injected_mmd_versions: Dict[str, str] = {}

        # Cached MMD data from last query-mmd call
        self.cached_mmds: Optional[List[Dict[str, str]]] = None
        self.cached_current_mmd: Optional[str] = None
        self.cached_mmd_time: float = 0.0

        # Compression tracking
        self.compression_count: int = 0
        self.last_compact_time: float = 0.0

    def is_processed(self, tool_call_id: str) -> bool:
        return tool_call_id in self.processed_tool_call_ids

    def mark_processed(self, tool_call_id: str) -> None:
        self.processed_tool_call_ids.add(tool_call_id)

    def is_offloaded(self, tool_call_id: str) -> bool:
        return tool_call_id in self.confirmed_offload_ids

    def mark_offloaded(self, tool_call_id: str) -> None:
        self.confirmed_offload_ids.add(tool_call_id)

    def is_deleted(self, tool_call_id: str) -> bool:
        return tool_call_id in self.deleted_offload_ids

    def mark_deleted(self, tool_call_id: str) -> None:
        self.deleted_offload_ids.add(tool_call_id)

    def get_mmd_version(self, filename: str) -> Optional[str]:
        return self.injected_mmd_versions.get(filename)

    def set_mmd_version(self, filename: str, version: str) -> None:
        self.injected_mmd_versions[filename] = version

    def clear_mmd_versions(self) -> None:
        self.injected_mmd_versions.clear()
