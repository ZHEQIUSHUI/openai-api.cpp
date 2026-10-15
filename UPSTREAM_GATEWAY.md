# 统一网关 (Upstream 透传)

## 概述

把多个独立运行的 OpenAI 兼容服务（例如每个模型一个 `axllm serve` 进程，各占一个端口）聚合到一个端口上：
网关按请求里的 `model` 字段，把请求**原样**转发给对应的上游，响应（包括 SSE 流式响应）**原样逐块**转回。

和[集群模式](CLUSTER_MODE.md)的区别：

| | 集群模式 (Master-Worker) | 统一网关 (Upstream) |
|---|---|---|
| 谁来注册 | Worker 启动后主动连接 Master 上报 | 由网关（或管理平台）主动注册上游地址 |
| 后端要求 | 后端必须基于本库并以 Worker 模式运行 | 任何 OpenAI 兼容的 HTTP 服务，无需改造 |
| 网关与后端之间 | 自定义内部协议 | 普通 HTTP，可以直接用 curl 排查 |
| 请求和响应 | 解析后重新编码 | 不解析、不改写，`tool_calls`、`reasoning_content`、`usage` 等字段完整保留 |

## 架构

```
管理平台 ──── POST/DELETE /admin/upstreams ────┐
                                              ▼
客户端 ──── /v1/chat/completions ───► 网关 (openai_api_gateway :8000)
            /v1/embeddings ...                 │ 按 model 字段转发
                                ┌──────────────┼──────────────┐
                                ▼              ▼              ▼
                          axllm serve    axllm serve    其他 OpenAI 兼容服务
                          (:18001)       (:18002)       (:180xx)
```

## 快速开始

```bash
# 启动网关，启动时注册一个上游
./openai_api_gateway --port 8000 --state upstreams.json \
    --upstream qwen3-4b=http://127.0.0.1:18001

# 运行时注册上游（管理接口默认只允许本机访问）
curl -X POST http://127.0.0.1:8000/admin/upstreams -d '{
  "model": "qwen3-vl-2b",
  "base_url": "http://127.0.0.1:18002",
  "upstream_model": "AXERA-TECH/Qwen3-VL-2B-Instruct-GPTQ-Int4"
}'

# 调用：只需要网关地址和模型名
curl http://127.0.0.1:8000/v1/chat/completions -d '{
  "model": "qwen3-vl-2b",
  "messages": [{"role": "user", "content": "你好"}]
}'

# 查看上游状态和最近一次调用统计
curl http://127.0.0.1:8000/admin/upstreams

# 注销上游（模型名中的 / 需要写成 %2F）
curl -X DELETE http://127.0.0.1:8000/admin/upstreams/qwen3-vl-2b
```

### 命令行参数

| 参数 | 说明 |
|---|---|
| `--host <addr>` | 监听地址，默认 `0.0.0.0` |
| `--port <port>` | 监听端口，默认 `8000` |
| `--api-key <key>` | 客户端调用需要的 API key，默认不校验 |
| `--upstream <model>=<url>` | 启动时注册上游，可重复 |
| `--upstream-timeout <ms>` | `--upstream` 注册的上游的读超时，默认 600000 |
| `--state <file>` | 启动时从文件恢复上游，注册表每次变化后写回文件；网关重启后不需要重新注册 |
| `--no-admin` | 关闭 `/admin/upstreams` 管理接口 |
| `--admin-key <key>` | 管理接口需要的 key（`Authorization: Bearer <key>`） |
| `--admin-remote` | 允许非本机访问管理接口（建议同时设置 `--admin-key`） |
| `--health-interval <ms>` | 上游健康检查间隔，默认 5000，0 表示关闭 |
| `--threads <n>` | HTTP 工作线程数，默认 32；每个进行中的流式请求占用一个线程 |

## 管理接口

管理接口与对外 API 使用同一个端口，但默认只接受来自 `127.0.0.1` / `::1` 的请求；对外的 `--api-key` 不作用于管理接口。

### `POST /admin/upstreams` 注册或替换上游

| 字段 | 必填 | 说明 |
|---|---|---|
| `model` | 是 | 对外的模型名，即客户端请求里的 `model` |
| `base_url` | 是 | 上游地址，如 `http://127.0.0.1:18001`，可带路径前缀 |
| `upstream_model` | 否 | 上游实际的模型名。设置后，转发时把请求里的 `model` 改成这个值（JSON 和 multipart 请求都支持） |
| `api_key` | 否 | 转发给上游时带的 key |
| `health_path` | 否 | 健康检查路径，默认 `/v1/models`，空字符串表示不检查 |
| `connect_timeout_ms` | 否 | 连接超时，默认 3000 |
| `read_timeout_ms` | 否 | 读超时，默认 600000 |
| `extra` | 否 | 附加到 `/v1/models` 中该模型条目的字段，例如 `{"max_context": 4096}` |

同名上游会被替换（统计数据清零）。`base_url` 不合法时返回 400。

### `GET /admin/upstreams` 查看上游

返回每个上游的配置（不回显 `api_key`，只给出 `has_api_key`）、健康状态和调用统计：

```json
{
  "model": "qwen3-0.6b",
  "base_url": "http://127.0.0.1:18011",
  "health": "healthy",
  "model_info": {"max_token_len": 2559, "prefill_max_token_num": 2176, "capabilities": {"vision": false}},
  "requests": 12, "errors": 0, "active": 1,
  "last_error": "",
  "last_call": {
    "endpoint": "/v1/chat/completions", "status": 200, "stream": true,
    "started_at": 1792034543, "duration_ms": 2936.5, "ttft_ms": 558.4,
    "tokens_per_second": 16.8,
    "usage": {"prompt_tokens": 18, "completion_tokens": 41, "total_tokens": 59}
  }
}
```

- `last_call.usage`：上游响应中的 `usage` 原样记录；流式响应取最后一个带 `usage` 的数据块。上游不返回时为 `null`，网关不自行估算。
- `last_call.ttft_ms`：流式请求是网关实测的首个 token 延迟（含网络）；非流式请求取上游 `usage.ttft_ms`（如果有）。
- `last_call.tokens_per_second`：优先取上游 `usage.decode_tps`，否则用 `completion_tokens / 生成耗时` 估算。
- 统计只保存在内存里，网关重启后清零。

### `DELETE /admin/upstreams/{model}` 注销上游

不存在时返回 404。

## 行为说明

- **路由**：对外端点（`/v1/chat/completions`、`/v1/embeddings`、`/v1/audio/*`、`/v1/images/*`、`/v1/completions` 等）收到请求后，先看 `model` 是否是已注册的上游；是则透传，否则按原有逻辑交给本地注册的模型。同名时上游优先。
- **路径**：按客户端请求的原始路径（含查询参数）转发，前面拼上 `base_url` 里的路径前缀。
- **`/v1/models`**：本地模型和健康的上游模型一起列出。上游条目包含从上游 `/v1/models` 读到的字段（如 `max_token_len`、`capabilities`），再叠加注册时的 `extra`。
- **健康检查**：定期请求 `health_path`。连续 2 次失败标记为 `unhealthy`：此时调用该模型直接返回 `503 model_not_ready`，且不出现在 `/v1/models`。恢复后自动变回 `healthy`。转发时连不上上游也计入失败次数。
- **错误**：上游返回的错误（状态码和内容）原样转回；连不上上游返回 `502 upstream_unavailable`。
- **客户端断开**：流式请求中途客户端断开时，网关立即断开与上游的连接，上游可以据此停止生成。
- **并发**：透传请求不占用本库的 `max_concurrency` 名额，由上游自己排队；网关的并发连接数受 `--threads` 限制。

## 在代码中使用

```cpp
#include <openai_api/server.hpp>

openai_api::Server server;

openai_api::UpstreamOptions up;
up.base_url = "http://127.0.0.1:18001";
up.upstream_model = "AXERA-TECH/Qwen3-0.6B";
server.registerUpstream("qwen3-0.6b", up);   // 也可以在 run() 之后随时调用

server.setUpstreamChangedCallback([&] {      // 注册表变化时回调，可用于持久化
    for (const auto& info : server.listUpstreams()) { /* ... */ }
});

openai_api::ServerOptions options;
options.port = 8000;
options.admin_api = true;                    // 启用 /admin/upstreams
options.upstream_health_interval = std::chrono::milliseconds(5000);
options.thread_pool_size = 32;
server.run(options);
```

上游注册与本地模型可以同时存在：同一个进程既可以直接跑本地模型，也可以把部分模型名转发给其他进程。
