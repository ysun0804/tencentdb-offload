# tencentdb-offload

> **Hermes Agent ContextEngine 插件** — 基于 TencentDB Agent Memory Gateway 的 offload V2 API 实现多层上下文压缩。
>
> 替代 [hermes-lcm](https://github.com/stephenschoettler/hermes-lcm)，作为 Hermes Agent 的唯一上下文引擎。
>
> **与官方 OpenClaw 集成功能对齐**——所有官方 ContextEngine 特性均已实现。

## 功能

### 核心压缩

- **L1 卸载**：工具调用/结果对的摘要化（通过 Gateway L1 异步提取）
- **L2 Mermaid 画布**：对话状态的可视化符号图谱（从 Gateway query-mmd 获取）
- **L3 压缩**：4 级级联压缩（fast-path → mild → aggressive → emergency）
- **自适应截断**：compact 前自动评估 HTTP body 大小，逐级截断 tool result
- **Anthropic 格式支持**：识别 user 消息中的 tool_result content blocks

### 生命周期 Hooks

| Hook | 时机 | 作用 |
|------|------|------|
| `post_tool_call` | 每次工具调用后 | 异步 ingest tool_pair 到 Gateway（L1 数据来源） |
| `pre_llm_call` | 每次 LLM 调用前 | 心跳过滤 + MMD 画布注入 + 增量压缩 |

### 高级特性

- **SessionRegistry**：多 session 状态管理，LRU 淘汰（默认 20 个）
- **SessionState**：每 session 独立跟踪（processed/confirmed/deleted IDs）
- **Reclaimer**：过期 session 数据清理（可配置保留天数）
- **MMD 注入**：从 Gateway 获取 Mermaid 画布，注入 `<current_task_context>` 和 `<history_task_context>`
- **心跳过滤**：自动移除 HEARTBEAT tool_use/result 消息对
- **优雅降级**：Gateway 不可用时自动切换为尾部截断
- **线程安全**：无锁 HTTP 调用 + 快照式状态修改
- **子 Agent 兼容**：`__deepcopy__` 支持 v0.18.0 subagent fork（见下文）
- **零外部依赖**：纯 Python 标准库（urllib, json, threading, logging）
- **斜杠命令**：`/tencentdb-offload` 查看运行时状态

## 架构

```
Hermes Agent
  │
  ├── ContextEngine 接口
  │     └── TencentDBOffloadEngine（本插件）
  │           │
  │           ├── compress()         ──→  POST :8420/v2/offload/compact
  │           ├── ingest()           ──→  POST :8420/v2/offload/ingest（异步）
  │           ├── should_compress()  →   阈值检查（0.4 × context_window）
  │           ├── pre_llm_call()     →   心跳过滤 + MMD 注入 + 增量压缩
  │           ├── reclaim()          →   过期 session 清理
  │           ├── __deepcopy__()     →   子 Agent fork 时复制预算状态
  │           └── _fallback_compress() →  本地尾部截断（Gateway 不可用时）
  │
  ├── post_tool_call hook
  │     └── 每次工具调用后异步 ingest tool_pair（含 timestamp）
  │
  └── TencentDB Gateway :8420（Node.js，独立进程）
        ├── /v2/offload/compact    — 同步多级压缩
        ├── /v2/offload/ingest     — 异步 L1/L2 处理
        ├── /v2/offload/query-mmd  — MMD 画布查询
        └── /health                — 健康检查
```

## 与官方 OpenClaw 实现的对齐

| 官方功能 | 本插件实现 | 状态 |
|---------|-----------|------|
| `afterToolCall` hook | `post_tool_call` hook | ✅ |
| `beforePromptBuild` hook | `pre_llm_call` hook | ✅ |
| `assemble()` — fastpath 重放 | `pre_llm_call` 中执行 | ✅ |
| L2 Mermaid 画布注入 | `_inject_mmd_from_gateway` | ✅ |
| Reclaimer | `reclaim(retention_days)` | ✅ |
| SessionRegistry | `SessionRegistry` 类 + LRU | ✅ |
| SessionState | `SessionState` 类 | ✅ |
| 4 级压缩（fastpath/mild/aggressive/emergency） | 委托 Gateway | ✅ |
| `context_window` 对齐 | `tokens/0.7` 让 ratio 落在 mild 区间 | ✅ |
| `ingest_before_compact` | compress() 时先 ingest 完整消息 | ✅ |

## 配置

环境变量（在 `~/.hermes/.env` 中设置）：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `TENCENTDB_OFFLOAD_GATEWAY_URL` | `http://127.0.0.1:8420` | Gateway 基础 URL |
| `TENCENTDB_OFFLOAD_API_KEY` | `local` | Bearer 认证 token |
| `TENCENTDB_OFFLOAD_INSTANCE_ID` | `default` | x-tdai-service-id 请求头 |
| `TENCENTDB_OFFLOAD_COMPACT_RATIO` | `0.5` | 压缩后目标上下文比例 |
| `TENCENTDB_OFFLOAD_THRESHOLD` | `0.75` | 触发压缩的阈值（占 context window 比例） |
| `TENCENTDB_OFFLOAD_TIMEOUT_MS` | `90000` | compact 请求超时（毫秒） |
| `TENCENTDB_OFFLOAD_INGEST_TIMEOUT_MS` | `5000` | ingest 请求超时（毫秒） |

## 从 LCM 切换

1. 在 `~/.hermes/config.yaml` 中启用插件：
   ```yaml
   plugins:
     enabled:
       - tencentdb-offload    # 替换 hermes-lcm
       - memory_tencentdb

   context:
     engine: tencentdb-offload
   ```

2. 在 `~/.hermes/.env` 中设置阈值：
   ```
   TENCENTDB_OFFLOAD_THRESHOLD=0.4
   ```

3. 重启 gateway：`hermes gateway restart`

4. 验证日志：
   ```
   [tencentdb-offload] model=glm-5.2, context_length=1000000, threshold=400000
   [tencentdb-offload] post_tool_call hook registered
   [tencentdb-offload] pre_llm_call hook registered
   ```

## Gateway 配置

在 `~/.memory-tencentdb/memory-tdai/tdai-gateway.json` 中添加 offload LLM 配置：

```json
{
  "offload": {
    "l1Model": "MiniMax-M3",
    "l15Model": "MiniMax-M3",
    "l2Model": "MiniMax-M3"
  }
}
```

不配置此字段时，Gateway 的 L1 提取不会执行（ingest 返回 200 但不生成 entries.jsonl）。

## 降级行为

Gateway 不可用时，引擎自动降级为**尾部截断**：
- 保留 system prompt + 前 3 条消息 + 尾部 token 预算内的消息
- 不拆分 tool-call/tool-result 消息对
- 截断超大 tool result（>2000 字符）

确保 Gateway 宕机时会话也不会死锁。

## 文件结构

```
tencentdb-offload/
├── __init__.py       — 插件注册 + hooks + 斜杠命令
├── engine.py         — TencentDBOffloadEngine（ContextEngine 实现）
├── plugin.yaml       — 插件清单
├── README.md         — 本文件
├── .gitignore
└── tests/
    └── test_engine.py — 25 项测试（14 核心 + 11 新功能）
```

## 与其他仓库的关系

| 仓库 | 关系 |
|------|------|
| [hermes-config](https://gitea.ysun0804.cn/Hermes/hermes-config) | 小马的完整配置仓库，`plugins/tencentdb-offload` 是本仓库的 git submodule |
| [TencentDB-Agent-Memory](https://github.com/TencentCloud/TencentDB-Agent-Memory) | 上游项目，本插件调用其 Gateway 的 offload V2 API（纯 HTTP，不依赖其代码） |
| [hermes-lcm](https://github.com/stephenschoettler/hermes-lcm) | 前任上下文引擎，已由本插件替代。lcm.db 保留为只读历史搜索 |

## 兼容性

- **TencentDB Agent Memory v1.0.0+**（需 offload V2 API，Zod schema 要求 `timestamp` 必填）
- **Hermes Agent v0.18.0+**（需 `ContextEngine` 抽象类 + `register_context_engine` + `pre_llm_call` / `post_tool_call` hooks + subagent `__deepcopy__` 支持）
- **Python 3.10+**
- 纯标准库，无外部依赖
- 仅通过 HTTP 通信，不需要 Node.js 运行时

## 设计决策

1. **纯 HTTP API 契约** — 不 import 任何 TencentDB 内部代码，服务端升级只要 V2 API 不变就自动兼容
2. **进程级 health cache** — `_available` 变量缓存健康检查结果，`bind_session()` 重置缓存让新 session 自动恢复
3. **compress() 无锁 HTTP** — 持锁快照 mutable state，释放锁后做 HTTP 调用，避免 30s compact 阻塞并发写入
4. **pre_llm_call 替代 assemble()** — Hermes ContextEngine ABC 不含 assemble()，用 pre_llm_call hook 等效实现
5. **context_window 动态计算** — 传 `tokens/0.7` 让 Gateway ratio 落在 mild 区间（0.5-0.85），触发完整压缩级联
6. **`__deepcopy__` 预算继承** — v0.18.0 subagent fork 时 `copy.deepcopy(engine)` 会因 `threading.Lock` 失败，自定义 `__deepcopy__` 只复制预算状态（`compression_count`/`last_prompt_tokens` 等），lock 和 session 状态重建

## CHANGELOG

### v0.4.1 (2026-07-07)
- **修复 ingest timestamp 缺失**：Gateway v1.0.0 Zod schema 要求 `tool_pair.timestamp: z.string()` 必填，`post_tool_call` hook 漏了 `timestamp` 字段 → 全部 ingest 400 Bad Request → L1 无数据 → compact 失效 → 内置压缩接管（双压缩问题）。修复：加 `datetime.now(timezone.utc).isoformat()`
- **`__deepcopy__` 支持**：v0.18.0 subagent fork 时 `copy.deepcopy(engine)` 因 `threading.Lock` 不能 pickle 而失败 → fallback 到内置压缩器。修复：自定义 `__deepcopy__` 复制预算状态、重建 lock/session
- **兼容性升级**：Hermes v0.17.0 → v0.18.0，TencentDB Gateway v1.0.0 Zod schema

### v0.4.0 (2026-07-06)
- **功能对齐官方 OpenClaw**：补齐 5 个关键差距
- `pre_llm_call` hook：心跳过滤 + MMD 画布注入 + 60% 阈值增量压缩
- L2 Mermaid 画布注入：从 Gateway query-mmd 获取，MD5 去重
- Reclaimer：过期 session 数据清理
- SessionRegistry + SessionState：多 session 状态管理，LRU 淘汰
- 测试：14 → 25 项

### v0.3.0 (2026-07-06)
- **post_tool_call hook**：每次工具调用后异步 ingest（正确 hook 名是 `post_tool_call` 不是 `after_tool_call`）
- **context_window 动态计算**：`tokens/0.7` 让 Gateway ratio 落在 mild 区间
- **ingest_before_compact**：compress 时先 ingest 完整消息，再截短发 compact

### v0.2.0 (2026-06-26)
- 自适应截断策略：4 级 body-size 自适应
- Anthropic 格式支持
- 内置摘要器禁用

### v0.1.0 (2026-06-25)
- 初始版本：ContextEngine ABC 实现
- Gateway :8420 HTTP 通信，零外部依赖
- 尾部截断 fallback

## 许可证

MIT
