"""tencentdb-offload — Hermes plugin registration.

Plugin entry point.  When ``context.engine: tencentdb-offload`` is set in
config.yaml, Hermes calls ``register(ctx)`` which creates the engine and
hooks it into the lifecycle.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def register(ctx):
    """Plugin entry point — register TencentDB offload as context engine."""
    logger.warning('[tencentdb-offload] register() CALLED')
    from .engine import TencentDBOffloadEngine

    engine = TencentDBOffloadEngine()

    # Register as the context engine
    if not hasattr(ctx, "register_context_engine") or not callable(ctx.register_context_engine):
        logger.error(
            "[tencentdb-offload] this Hermes host does not support "
            "register_context_engine — plugin disabled"
        )
        return

    ctx.register_context_engine(engine)

    # Register post_tool_call hook for background ingest.
    # IMPORTANT: the valid Hermes hook name is "post_tool_call" (see
    # hermes_cli.plugins.VALID_HOOKS); "after_tool_call" silently fails.
    # This is the primary ingest path — every tool call triggers an async
    # /ingest so the Gateway's L1 has time to populate entries.jsonl
    # before should_compress() fires many turns later.
    register_hook = getattr(ctx, "register_hook", None)
    if callable(register_hook):
        try:
            register_hook(
                hook_name="post_tool_call",
                callback=_make_post_tool_call_handler(engine),
            )
            logger.info("[tencentdb-offload] registered post_tool_call hook")
        except Exception as exc:
            logger.warning(
                "[tencentdb-offload] post_tool_call hook registration failed: %s "
                "(L1 entries will not populate — compression will degrade)",
                exc,
            )

        # Register pre_llm_call hook for incremental L3 compression +
        # MMD injection + heartbeat filtering.
        # Equivalent to OpenClaw's before_prompt_build / llm_input_l3.
        try:
            register_hook(
                hook_name="pre_llm_call",
                callback=_make_pre_llm_call_handler(engine),
            )
            logger.info("[tencentdb-offload] registered pre_llm_call hook")
        except Exception as exc:
            logger.warning(
                "[tencentdb-offload] pre_llm_call hook registration failed: %s "
                "(MMD injection and incremental compression will be unavailable)",
                exc,
            )

    # Optional: bind session ID when session starts
    register_session_hook = getattr(ctx, "on_session_start", None) or getattr(
        ctx, "register_session_hook", None
    )
    if callable(register_session_hook):
        try:
            register_session_hook(
                callback=_make_session_start_handler(engine),
            )
            logger.info("[tencentdb-offload] registered session_start hook")
        except Exception as exc:
            logger.debug(
                "[tencentdb-offload] session hook registration skipped: %s", exc
            )

    # Optional: register /tencentdb-offload slash command
    register_command = getattr(ctx, "register_command", None)
    if callable(register_command):
        try:
            register_command(
                name="tencentdb-offload",
                callback=_make_status_command_handler(engine),
                description="Show TencentDB offload context engine status",
            )
            logger.info("[tencentdb-offload] registered /tencentdb-offload command")
        except Exception as exc:
            logger.debug(
                "[tencentdb-offload] command registration skipped: %s", exc
            )

    logger.info(
        "[tencentdb-offload] plugin registered (gateway=%s)",
        engine._gateway_url,
    )


def _make_post_tool_call_handler(engine):
    """Create a post_tool_call hook handler that ingests tool pairs for L1 offload.

    Hermes emits ``post_tool_call`` with kwargs (see ``model_tools._emit_post_tool_call_hook``):
      tool_name, args, result, tool_call_id, session_id, task_id, turn_id,
      duration_ms, status, error_type, error_message
    """

    def _handler(*args, **kwargs):
        logger.info('[tencentdb-offload] post_tool_call FIRED: tool=%s', kwargs.get("tool_name", "?"))
        try:
            tool_name = kwargs.get("tool_name") or ""
            tool_call_id = (
                kwargs.get("tool_call_id")
                or kwargs.get("call_id")
                or kwargs.get("task_id")
                or ""
            )
            params = kwargs.get("args")
            if params is None:
                params = kwargs.get("params") or kwargs.get("arguments") or {}
            result = kwargs.get("result")
            error = kwargs.get("error_message") or kwargs.get("error")
            duration_ms = kwargs.get("duration_ms")
            session_id = kwargs.get("session_id")

            if not tool_call_id:
                logger.debug(
                    "[tencentdb-offload] post_tool_call: skip (no tool_call_id, tool=%s)",
                    tool_name,
                )
                return

            # Bind session id if engine doesn't have one yet
            if session_id and not engine._session_id:
                engine.bind_session(str(session_id))

            # Coerce result to a string — Hermes may pass dict / list / object.
            # Match OpenClaw's ToolPair format: toolName/toolCallId/params/result/error/timestamp/durationMs.
            result_str = result if isinstance(result, str) else _truncate(
                str(result), 8000,
            )

            from datetime import datetime, timezone

            pair = {
                "tool_name": str(tool_name)[:200],
                "tool_call_id": str(tool_call_id)[:200],
                "params": _coerce_params(params),
                "result": _truncate(result_str, 8000),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            if error:
                pair["error"] = _truncate(str(error), 2000)
            if duration_ms is not None:
                try:
                    pair["duration_ms"] = int(duration_ms)
                except (TypeError, ValueError):
                    pass

            engine.ingest_tool_pairs([pair])
        except Exception as exc:
            logger.debug("[tencentdb-offload] post_tool_call hook error: %s", exc)

    return _handler


def _coerce_params(params: Any) -> Any:
    """Keep params as-is if dict/list; otherwise truncate the string form."""
    if isinstance(params, (dict, list)):
        return params
    return _truncate(str(params), 4000)


def _make_pre_llm_call_handler(engine):
    """Create a pre_llm_call hook handler for incremental L3 + MMD injection.

    Hermes emits ``pre_llm_call`` before each LLM API call.  The handler
    receives kwargs including ``messages`` (the message list to be sent).
    We modify messages in-place for:
      - Heartbeat filtering (remove internal probe tool_use/result pairs)
      - MMD canvas injection (active + history task context)
      - Incremental L3 compression (when approaching threshold)

    Equivalent to OpenClaw's ``before_prompt_build`` + ``llm_input_l3``.
    """

    def _handler(*args, **kwargs):
        messages = kwargs.get("messages") or (args[0] if args else None)
        if not isinstance(messages, list) or not messages:
            return

        try:
            engine.pre_llm_call(messages, **kwargs)
        except Exception as exc:
            logger.debug(
                "[tencentdb-offload] pre_llm_call hook error: %s", exc
            )

    return _handler


def _make_session_start_handler(engine):
    """Create a handler that binds the session ID on new sessions."""

    def _handler(session_id=None, *args, **kwargs):
        sid = session_id or kwargs.get("session_id", "")
        if sid:
            engine.bind_session(str(sid))

    return _handler


def _make_status_command_handler(engine):
    """Create a slash command handler that prints engine status."""

    def _handler(*args, **kwargs):
        import json
        status = engine.get_status()
        lines = [f"  {k}: {v}" for k, v in status.items()]
        return "TencentDB Offload Status\n" + "\n".join(lines)

    return _handler


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[truncated]"
