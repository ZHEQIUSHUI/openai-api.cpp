# 集群模式 (Cluster Mode) 功能文档

## 概述

OpenAI API Server Library 支持主从分布式架构，允许在多台机器上部署 Worker 节点，由一个 Master 节点统一接收请求并转发给相应的 Worker 处理。

## 架构

```
┌─────────────┐      HTTP API      ┌─────────────┐
│   Client    │ ◄────────────────► │    Master   │◄────┐
└─────────────┘                    │   (Port N)  │     │
                                   └──────┬──────┘     │
                                          │            │
                    ┌─────────────────────┼────────┐   │
                    │                     │        │   │
                    ▼                     ▼        ▼   │
              ┌─────────┐          ┌─────────┐ ┌──────┴───┐
              │ Worker1 │          │ Worker2 │ │ Worker3  │
              │(Model A)│          │(Model B)│ │(Model C) │
              └─────────┘          └─────────┘ └──────────┘
              (Port N+1000 内部通信)
```

## 核心组件

### 1. ClusterServer
主入口类，支持三种运行模式：
- **STANDALONE**: 独立模式（单进程）
- **MASTER**: 主节点模式
- **WORKER**: 工作节点模式

```cpp
openai_api::ClusterServer server;

// Master 模式
server.runAsMaster(8080);

// Worker 模式
server.runAsWorker("192.168.1.100", 9080);  // 连接到 Master 内部端口

// 自动检测模式
server.run(8080);  // 自动选择 Master 或 Worker 模式
```

### 2. WorkerManager (Master 端)
- 接受 Worker 连接
- 管理模型注册表
- 转发请求到对应的 Worker
- 维护心跳检测

### 3. WorkerClient (Worker 端)
- 连接到 Master
- 注册本地模型
- 接收并处理转发的请求
- 发送心跳保活

### 4. RemoteWorkerProvider
Master 端包装远程 Worker 响应的 Provider，实现透明的请求转发。

## 内部协议

Worker 和 Master 之间通过 HTTP + 二进制协议通信：

| 消息类型 | 说明 |
|---------|------|
| HANDSHAKE | 握手建立连接 |
| REGISTER_MODEL | 注册模型 |
| HEARTBEAT | 心跳保活 |
| FORWARD_REQUEST | 转发请求 |
| FORWARD_RESPONSE | 转发响应 |

内部端口 = 外部端口 + 1000

## API 参考

### ClusterServer

```cpp
// 构造
ClusterServer();
ClusterServer(int port);
ClusterServer(const ClusterServerOptions& options);

// 配置
void setMaxConcurrency(int max);
void setTimeout(std::chrono::milliseconds timeout);
void setApiKey(const std::string& api_key);
void setWorkerListenAddress(const std::string& host, int port = 0);

// 模型注册
void registerChat(const std::string& model_name, ChatCallback callback);
void registerEmbedding(const std::string& model_name, EmbeddingCallback callback);
void registerASR(const std::string& model_name, ASRCallback callback);
void registerTTS(const std::string& model_name, TTSCallback callback);
void registerImageGeneration(const std::string& model_name, ImageGenCallback callback);

// 模型管理
std::vector<std::string> listModels() const;
bool hasModel(const std::string& model_name) const;
void unregisterModel(const std::string& model_name);

// 运行控制
ClusterMode run(int port = 8080);                    // 自动检测模式
bool runAsMaster(int port = 8080);                   // 强制 Master 模式
bool runAsWorker(const std::string& host, int master_port);  // 强制 Worker 模式
void stop();
bool isRunning() const;
ClusterMode getMode() const;

// 获取内部组件
Server* getServer();
WorkerClient* getWorkerClient();
```

### 配置选项

```cpp
struct ServerOptions {
    std::string host = "0.0.0.0";
    int port = 8080;
    int max_concurrency = 10;
    std::chrono::milliseconds default_timeout{60000};
    std::string api_key;
    std::string worker_id;  // Worker ID（集群模式下使用，留空自动生成）
};

struct ClusterServerOptions {
    ServerOptions server;
    bool enable_cluster = true;
    std::chrono::milliseconds worker_timeout{30000};
    std::chrono::milliseconds heartbeat_interval{5000};
};
```

## 使用示例

### Master 节点

```cpp
#include <openai_api/cluster_server.hpp>

int main() {
    openai_api::ClusterServer master;
    
    // Master 可以有自己的本地模型
    master.registerChat("master-model", [](const auto& req, auto provider) {
        // 处理请求
        provider->push(OutputChunk::TextDelta("Hello"));
        provider->end();
    });
    
    // 启动 Master，监听 8080 端口
    // 内部 Worker 管理端口为 8080 + 1000 = 9080
    master.runAsMaster(8080);
    
    return 0;
}
```

### Worker 节点

```cpp
#include <openai_api/cluster_server.hpp>

int main() {
    openai_api::ClusterServer worker;
    
    // 设置 Worker 监听地址（用于跨机器部署）
    worker.setWorkerListenAddress("0.0.0.0", 0);  // 0 表示自动分配端口
    
    // 注册 Worker 的模型
    worker.registerChat("worker-model", [](const auto& req, auto provider) {
        // 处理请求
        provider->push(OutputChunk::TextDelta("Hello from Worker"));
        provider->end();
    });
    
    // 连接到 Master 的内部端口
    worker.runAsWorker("192.168.1.100", 9080);
    
    return 0;
}
```

### 自动模式检测

```cpp
openai_api::ClusterServer server;

// 注册模型
server.registerChat("my-model", callback);

// 自动检测模式：
// - 如果端口空闲，作为 Master 启动
// - 如果端口被占用且是集群服务，作为 Worker 连接
// - 如果端口被占用但不是集群服务，报错
auto mode = server.run(8080);

if (mode == openai_api::ClusterMode::MASTER) {
    std::cout << "Running as Master" << std::endl;
} else if (mode == openai_api::ClusterMode::WORKER) {
    std::cout << "Running as Worker" << std::endl;
}
```

## 特性

### 已实现

- ✅ **端口检测**：自动检测端口是否可用
- ✅ **服务识别**：通过魔数识别是否为本项目服务
- ✅ **远程注册**：Worker 可以注册模型到 Master
- ✅ **模型冲突检查**：同名模型注册会失败
- ✅ **请求转发**：Master 自动转发请求到对应 Worker
- ✅ **心跳保活**：自动检测 Worker 断开
- ✅ **跨机器部署**：Worker 可以运行在不同机器上
- ✅ **全模型支持**：Chat/Embedding/ASR/TTS/Image Generation
- ✅ **流式响应**：支持 SSE 流式转发
- ✅ **并发控制**：可配置的并发限制
- ✅ **API Key 认证**：支持 Master 端认证

### 测试覆盖

- 基础连接测试
- Worker 注册测试
- 请求转发测试
- 多 Worker 测试
- 模型冲突测试
- 自动模式检测测试
- 所有模型类型转发测试
- 配置传递测试
- 组件获取测试

## 运行测试

```bash
# 基础集群测试
./openai_api_test_cluster

# 完整集群测试
./openai_api_test_cluster_full

# 运行所有测试
ctest --output-on-failure
```

## 示例程序

```bash
# 终端 1：启动 Master
./openai_api_example_cluster_master 8080

# 终端 2：启动 Worker 1
./openai_api_example_cluster_worker -n worker-1 192.168.1.100 8080

# 终端 3：启动 Worker 2
./openai_api_example_cluster_worker -n worker-2 192.168.1.100 8080

# 测试请求
curl http://192.168.1.100:8080/v1/models
curl http://192.168.1.100:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "worker-1-model", "messages": [...]}'
```

## 注意事项

1. **端口规划**：Master 占用两个端口，外部端口和内部端口（+1000）
2. **防火墙**：确保内部端口允许 Worker 机器访问
3. **网络延迟**：跨机器部署时考虑网络延迟对推理的影响
4. **模型名称**：Worker 注册的模型名不能与 Master 或其他 Worker 冲突
5. **心跳超时**：Worker 断开后，Master 需要 30 秒（默认）才能清理模型
