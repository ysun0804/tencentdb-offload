# tencentdb-offload

Hermes Agent **ContextEngine** plugin backed by TencentDB Agent Memory Gateway's offload V2 API.

Replaces [hermes-lcm](https://github.com/stephenschoettler/hermes-lcm) with TencentDB's multi-layer offload system for context compression.

## Features

- **L1 Offload**: Tool-call/result pair summarisation (replaces verbose tool output with concise summaries)
- **L2 Mermaid Canvas**: Visual symbol map of conversation state
- **L3 Compaction**: Synchronous compression (fast-path → mild → aggressive → emergency)
- **Graceful Fallback**: Tail-truncation with tool-pair preservation when Gateway is down
- **Thread-Safe**: Lock-free HTTP calls, snapshot-based state mutation
- **Zero External Dependencies**: Python stdlib only (urllib, json, threading, logging)
- **Slash Command**: `/tencentdb-offload` for runtime status inspection

## Architecture

```
Hermes Agent
  │
  ├── ContextEngine interface (4 abstract methods)
  │     └── TencentDBOffloadEngine (this plugin)
  │           │
  │           ├── compress()     ──→  POST :8420/v2/offload/compact
  │           ├── ingest()       ──→  POST :8420/v2/offload/ingest (fire-and-forget)
  │           ├── should_compress()  →  threshold check (0.4 × context_window)
  │           └── _fallback_compress()  →  local tail-truncation (if Gateway down)
  │
  └── TencentDB Gateway :8420 (Node.js, separate process)
        ├── /v2/offload/compact   — synchronous multi-layer compaction
        ├── /v2/offload/ingest    — async L1/L2 processing
        └── /health               — health check
```

## Configuration

Environment variables (set in `~/.hermes/.env`):

| Variable | Default | Description |
|----------|---------|-------------|
| `TENCENTDB_OFFLOAD_GATEWAY_URL` | `http://127.0.0.1:8420` | Gateway base URL |
| `TENCENTDB_OFFLOAD_API_KEY` | `local` | Bearer token for Gateway auth |
| `TENCENTDB_OFFLOAD_INSTANCE_ID` | `default` | x-tdai-service-id header |
| `TENCENTDB_OFFLOAD_COMPACT_RATIO` | `0.5` | Target context ratio after compaction |
| `TENCENTDB_OFFLOAD_THRESHOLD` | `0.75` | Compress trigger (fraction of context window) |
| `TENCENTDB_OFFLOAD_TIMEOUT_MS` | `30000` | Compact HTTP timeout (ms) |
| `TENCENTDB_OFFLOAD_INGEST_TIMEOUT_MS` | `5000` | Ingest HTTP timeout (ms) |

## Installation

### Option A: Direct install

```bash
# Clone into Hermes plugins directory
git clone https://gitea.ysun0804.cn/Hermes/tencentdb-offload.git ~/.hermes/plugins/tencentdb-offload
```

### Option B: From scratch (for development)

```bash
mkdir -p ~/.hermes/plugins/tencentdb-offload
# Copy engine.py, __init__.py, plugin.yaml from this repo
```

## How to switch from LCM

1. Enable the plugin in `~/.hermes/config.yaml`:
   ```yaml
   plugins:
     enabled:
       - tencentdb-offload
       # - hermes-lcm  ← remove or comment out

   context:
     engine: tencentdb-offload
   ```

2. Set threshold in `~/.hermes/.env`:
   ```
   TENCENTDB_OFFLOAD_THRESHOLD=0.4
   ```

3. Restart gateway: `hermes gateway restart`

4. Verify in gateway logs:
   ```
   [tencentdb-offload] model=glm-5.2, context_length=1000000, threshold=400000
   ```

## Fallback Behavior

If the TencentDB Gateway is unreachable, the engine automatically falls back to **tail-truncation**:

- Keeps system prompt + first 3 messages + tail messages within token budget
- Respects tool-call/tool-result pairs (never splits them)
- Truncates oversized tool results (>2000 chars)

This ensures sessions never deadlock, even if the Gateway is down.

## File Structure

```
tencentdb-offload/
├── __init__.py      — Plugin registration + hooks + slash command
├── engine.py        — TencentDBOffloadEngine (ContextEngine implementation)
├── plugin.yaml      — Plugin manifest
├── README.md        — This file
├── .gitignore
└── tests/
    └── test_engine.py — 8 unit tests
```

## Compatibility

- **TencentDB Agent Memory v1.0.0+** (requires offload V2 API)
- **Hermes Agent v0.16.0+** (requires `ContextEngine` abstract class + `register_context_engine`)
- **Python 3.10+** (uses `from __future__ import annotations`)
- No external Python dependencies — stdlib only
- Plugin talks HTTP only; no Node.js or TypeScript runtime required

## Development

Developed by [ClawCompany](https://gitea.ysun0804.cn/Hermes) as a replacement for hermes-lcm.

Key design decisions:
1. **HTTP-only API contract** — No dependency on TencentDB internals, survives server upgrades
2. **Process-global health cache** — One health check per gateway lifetime, reset on session bind
3. **Lock-free HTTP in compress()** — Snapshot mutable state under lock, release before HTTP call

## License

MIT
