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

import logging
import os
import threading
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

        logger.info(
            "[tencentdb-offload] init: gateway=%s, instance=%s, ratio=%.2f",
            self._gateway_url,
            self._instance_id,
            self._compact_ratio,
        )

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
                # Pass a context_window that makes Gateway's ratio land in mild/aggressive range.
                # Gateway uses: ratio = totalTokens / contextWindow, then resolveLevel(ratio)
                #   mild >= 0.5, aggressive >= 0.85, emergency >= 0.95
                # We want ratio ≈ 0.6-0.8 when our threshold triggers, so:
                # contextWindow = totalTokens / 0.7
                "context_window": max(int(tokens / 0.7), 100000),
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
        self, messages: List[Dict[str, Any]], max_body_mb: float = 4.0
    ) -> List[Dict[str, Any]]:
        """Reduce message payload before sending to compact API.

        Always truncate tool results to 500 chars (unconditionally).
        This is critical because:
          1. Gateway's L1 extraction is async and entries.jsonl may be empty
          2. Short tool results keep ratio low → Gateway stays in fastpath/mild
          3. Original tool results are useless after compression anyway

        Then apply head+tail truncation if body still exceeds max_body_mb.
        """
        import json as _json

        def _body_mb(msgs: List[Dict[str, Any]]) -> float:
            return len(_json.dumps(msgs, ensure_ascii=False).encode("utf-8")) / (1024 * 1024)

        n = len(messages)
        before = _body_mb(messages)

        # Step 1: Always truncate tool results to 500 chars (unconditional)
        result = [_truncate_tool_result(msg, max_chars=500) for msg in messages]
        after = _body_mb(result)
        logger.info(
            "[tencentdb-offload] prepare: %d msgs, %.2fMB → %.2fMB (tool→500, unconditional)",
            n, before, after,
        )

        # Step 2: head+tail truncation if body still exceeds limit
        if after > max_body_mb:
            keep = min(max(8, int(n * max_body_mb / after)), len(result))
            head = result[:4]
            tail = result[-(keep - 4):] if keep > 4 else []
            result = head + tail
            after = _body_mb(result)
            logger.info(
                "[tencentdb-offload] prepare step2 (head+tail): %d→%d msgs, %.2fMB → %.2fMB",
                n, len(result), before, after,
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
        }

    # -- Background ingestion (called by hook, not part of abstract) ------

    def ingest_tool_pairs(
        self,
        tool_pairs: List[Dict[str, Any]],
        prompt: str = None,
    ) -> None:
        """Fire-and-forget: send tool-call/result pairs to offload server for L1 processing.

        Called after each tool execution.  Does not block the conversation.
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

    def _ingest_before_compact(
        self, messages: List[Dict[str, Any]], session_id: str
    ) -> None:
        """Ingest full ``messages`` to ``/v2/offload/ingest`` before compaction truncates them.

        Extracts tool pairs + recent text messages + last user prompt and fires
        them at the Gateway.  Fire-and-forget: any failure (network, server,
        unexpected exception) is logged and swallowed so ``compress()`` continues
        to the compact step uninterrupted.
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
