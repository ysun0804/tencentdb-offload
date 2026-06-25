# tencentdb-offload

Hermes Agent ContextEngine plugin backed by TencentDB Agent Memory Gateway's offload V2 API.

## What it does

Replaces the built-in context compressor (or LCM) with TencentDB's multi-layer offload system:

- **L1**: Tool-call/result pair summarisation (replaces verbose tool output with concise summaries)
- **L2**: Mermaid canvas generation (visual symbol map of conversation state)
- **L3**: Synchronous compaction (fast-path truncation → mild compression → aggressive deletion)

## Architecture

```
Hermes Agent
  │
  ├── ContextEngine interface
  │     └── TencentDBOffloadEngine (this plugin)
  │           │
  │           ├── compress()  ──→  POST :8420/v2/offload/compact
  │           ├── ingest()    ──→  POST :8420/v2/offload/ingest (fire-and-forget)
  │           └── fallback    ──→  local tail-truncation (if Gateway down)
  │
  └── TencentDB Gateway :8420 (Node.js, separate process)
        ├── /v2/offload/compact  — synchronous compaction
        ├── /v2/offload/ingest   — async L1/L2 processing
        └── /health              — health check
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
| `TENCENTDB_OFFLOAD_TIMEOUT_MS` | `30000` | Compact HTTP timeout |
| `TENCENTDB_OFFLOAD_INGEST_TIMEOUT_MS` | `5000` | Ingest HTTP timeout |

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

2. Restart gateway: `hermes gateway restart`

3. Verify: check gateway logs for `[tencentdb-offload] init: gateway=...`

## Fallback behavior

If the TencentDB Gateway is unreachable, the engine automatically falls back to **tail-truncation**:
- Keeps system prompt + first 3 messages + last 6 messages
- Respects tool-call/tool-result pairs (never splits them)
- Truncates oversized tool results

This ensures sessions never deadlock, even if the Gateway is down.

## Compatibility

- Requires TencentDB Agent Memory **v1.0.0+** (offload V2 API)
- No external Python dependencies — stdlib only
- Plugin talks HTTP only; no Node.js or TypeScript dependency
- TencentDB server upgrades that preserve the V2 API contract are automatically compatible
