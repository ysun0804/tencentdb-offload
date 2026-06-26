# tencentdb-offload

> **Hermes Agent ContextEngine 插件** — 基于 TencentDB Agent Memory Gateway 的 offload V2 API 实现多层上下文压缩。
>
> 替代 [hermes-lcm](https://github.com/stephenschoettler/hermes-lcm)，作为 Hermes Agent 的唯一上下文引擎。

## 功能

- **L1 卸载**：工具调用/结果对的摘要化（将冗长工具输出替换为精简摘要）
- **L2 Mermaid 画布**：对话状态的可视化符号图谱
- **L3 压缩**：同步多级压缩（fast-path → mild → aggressive → emergency）
- **自适应截断**：compact 前自动评估 HTTP body 大小，逐级截断 tool result（2000→500 字符），仅超限才丢消息
- **Anthropic 格式支持**：识别 user 消息中的 tool_result content blocks
- **优雅降级**：Gateway 不可用时自动切换为尾部截断（保持 tool-call/result 完整性）
- **线程安全**：无锁 HTTP 调用 + 快照式状态修改
- **零外部依赖**：纯 Python 标准库（urllib, json, threading, logging）
- **斜杠命令**：`/tencentdb-offload` 查看运行时状态

## 架构

```
Hermes Agent
  │
  ├── ContextEngine 接口（4 个抽象方法）
  │     └── TencentDBOffloadEngine（本插件）
  │           │
  │           ├── compress()        ──→  POST :8420/v2/offload/compact
  │           ├── ingest()          ──→  POST :8420/v2/offload/ingest（异步）
  │           ├── should_compress()  →   阈值检查（0.4 × context_window）
  │           └── _fallback_compress() →  本地尾部截断（Gateway 不可用时）
  │
  └── TencentDB Gateway :8420（Node.js，独立进程）
        ├── /v2/offload/compact   — 同步多级压缩
        ├── /v2/offload/ingest    — 异步 L1/L2 处理
        └── /health               — 健康检查
```

## 配置

环境变量（在 `~/.hermes/.env` 中设置）：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `TENCENTDB_OFFLOAD_GATEWAY_URL` | `http://127.0.0.1:8420` | Gateway 基础 URL |
| `TENCENTDB_OFFLOAD_API_KEY` | `local` | Bearer 认证 token |
| `TENCENTDB_OFFLOAD_INSTANCE_ID` | `default` | x-tdai-service-id 请求头 |
| `TENCENTDB_OFFLOAD_COMPACT_RATIO` | `0.5` | 压缩后目标上下文比例 |
| `TENCENTDB_OFFLOAD_THRESHOLD` | `0.75` | 触发压缩的阈值（占 context window 比例） |
| `TENCENTDB_OFFLOAD_TIMEOUT_MS` | `30000` | compact 请求超时（毫秒） |
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
   ```

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
    └── test_engine.py — 8 项单元测试
```

## 与其他仓库的关系

| 仓库 | 关系 |
|------|------|
| [hermes-config](https://gitea.ysun0804.cn/Hermes/hermes-config) | 小马的完整配置仓库，`plugins/tencentdb-offload` 是本仓库的 git submodule |
| [TencentDB-Agent-Memory](https://github.com/TencentCloud/TencentDB-Agent-Memory) | 上游项目，本插件调用其 Gateway 的 offload V2 API（纯 HTTP，不依赖其代码） |
| [hermes-lcm](https://github.com/stephenschoettler/hermes-lcm) | 前任上下文引擎，已由本插件替代。lcm.db 保留为只读历史搜索 |

## 兼容性

- **TencentDB Agent Memory v1.0.0+**（需 offload V2 API）
- **Hermes Agent v0.16.0+**（需 `ContextEngine` 抽象类 + `register_context_engine`）
- **Python 3.10+**
- 纯标准库，无外部依赖
- 仅通过 HTTP 通信，不需要 Node.js 运行时

## 设计决策

1. **纯 HTTP API 契约** — 不 import 任何 TencentDB 内部代码，服务端升级只要 V2 API 不变就自动兼容
2. **进程级 health cache** — `_available` 变量缓存健康检查结果，`bind_session()` 重置缓存让新 session 自动恢复
3. **compress() 无锁 HTTP** — 持锁快照 mutable state，释放锁后做 HTTP 调用，避免 30s compact 阻塞并发写入

## CHANGELOG

### v0.2.0 (2026-06-26)
- **自适应截断策略**：对齐 OpenClaw `context-engine.ts`，4 级 body-size 自适应（≤4MB 不截断 → 2000 字符 → 500 字符 → head+tail 兜底）
- **Anthropic 格式支持**：`_is_tool_result()` 检测 user 消息中的 tool_result content blocks
- **内置摘要器禁用**：配合 `auxiliary.compression.model: ''`，压缩只走 tencentdb-offload
- **compact body 日志**：每级截断记录 before→after MB 和消息数

### v0.1.0 (2026-06-25)
- 初始版本：ContextEngine ABC 实现，compress/ingest/health 接口
- Gateway :8420 HTTP 通信，零外部依赖
- 尾部截断 fallback（保持 tool pair 完整性）

## 许可证

MIT
