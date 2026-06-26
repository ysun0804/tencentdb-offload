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


def _truncate_tool_result(msg: Dict[str, Any], max_chars: int = 500) -> Dict[str, Any]:
    """Truncate tool_result content in a message to avoid huge HTTP bodies."""
    content = msg.get("content", "")
    if isinstance(content, str) and len(content) > max_chars:
        role = msg.get("role", "")
        if role in ("tool", "function"):
            msg = dict(msg)
            msg["content"] = content[:max_chars] + "\n...[truncated for compact]"
    elif isinstance(content, list):
        changed = False
        new_content = []
        for block in content:
            if isinstance(block, dict):
                text = block.get("text", block.get("content", ""))
                if isinstance(text, str) and len(text) > max_chars:
                    block = dict(block)
                    if "text" in block:
                        block["text"] = text[:max_chars] + "\n...[truncated for compact]"
                    elif "content" in block:
                        block["content"] = text[:max_chars] + "\n...[truncated for compact]"
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
            os.environ.get("TENCENTDB_OFFLOAD_TIMEOUT_MS", "30000")
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

        # Limit messages sent to Gateway to avoid HTTP body too large (Broken pipe).
        # Keep system + first N + last M, truncate middle tool results.
        send_msgs = self._prepare_for_compact(messages)

        result = _post_json(
            f"{self._gateway_url}/v2/offload/compact",
            {
                "session_id": session_id,
                "messages": send_msgs,
                "ratio": self._compact_ratio,
                "context_window": self.context_length or 200000,
                "total_tokens": tokens,
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
        self, messages: List[Dict[str, Any]], max_chars_per_msg: int = 2000
    ) -> List[Dict[str, Any]]:
        """Reduce message payload size before sending to compact API.

        The TencentDB Gateway has practical HTTP body limits. When the
        conversation has hundreds of messages with large tool outputs,
        sending them all in one POST causes Broken pipe.

        Strategy: keep ALL messages (preserve conversation structure),
        but truncate tool_result content to max_chars_per_msg.
        This lets the compact API see the full conversation and make
        intelligent compression decisions, while keeping body size manageable.
        """
        result = []
        for msg in messages:
            result.append(_truncate_tool_result(msg, max_chars=max_chars_per_msg))

        # If still too many messages, drop middle with a higher cap
        # (only as last resort — the compact API should handle this)
        if len(result) > 500:
            head = result[:4]
            tail = result[-(500 - 4):]
            result = head + tail
            logger.info(
                "[tencentdb-offload] extreme case: %d→%d messages (dropped middle)",
                len(messages), len(result),
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
