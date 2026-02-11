# OpenAI API Server Library

一个 C++ 库，用于构建兼容 OpenAI API 的推理服务。支持多模型路由，同一端点可以注册多个算法实现。

## 特性

- **库形式**：作为 CMake 库供其他项目引入
- **多模型路由**：同一端点支持多个算法实现（如 ASR 同时支持 Whisper 和 SenseVoice）
- **集群模式**：支持 Master-Worker 架构，多个 Worker 进程可以注册到同一个 Master
- **模型验证**：自动验证请求的模型是否已注册
- **完整 API 支持**：Chat、Embedding、ASR、TTS、Image Generation
- **流式输出**：支持 SSE 流式响应
- **并发控制**：可配置的并发和超时管理
- **Python SDK 兼容**：与 OpenAI Python SDK 完全兼容

## 快速开始

### 方式一：FetchContent

```cmake
include(FetchContent)
FetchContent_Declare(
    openai_api
    GIT_REPOSITORY https://github.com/yourusername/openai-api.git
    GIT_TAG main
)
FetchContent_MakeAvailable(openai_api)

target_link_libraries(your_app PRIVATE openai_api::server)
```

### 方式二：子目录

```cmake
add_subdirectory(third_party/openai-api)
target_link_libraries(your_app PRIVATE openai_api::server)
```

## 使用示例

### Qwen 模型服务

```cpp
#include <openai_api/server.hpp>
#include <iostream>

using namespace openai_api;

int main() {
    // 创建服务器
    Server server(8080);
    
    // 注册 Qwen-0.6B 模型
    server.registerChat("qwen-0.6b", [](const ChatRequest& req, auto provider) {
        // 调用你的 Qwen 推理代码
        std::string response = your_qwen_inference(req.messages);
        
        if (req.stream) {
            // 流式输出
            for (const auto& token : tokenize(response)) {
                provider->push(OutputChunk::TextDelta(token, req.model));
            }
        } else {
            provider->push(OutputChunk::FinalText(response, req.model));
        }
        provider->end();
    });
    
    // 注册 Qwen-7B 模型
    server.registerChat("qwen-7b", [](const ChatRequest& req, auto provider) {
        std::string response = your_qwen7b_inference(req.messages);
        provider->push(OutputChunk::FinalText(response, req.model));
        provider->end();
    });
    
    // 运行服务器
    server.run();
}
```

### 多 ASR 模型服务（Whisper + SenseVoice）

```cpp
#include <openai_api/server.hpp>

using namespace openai_api;

int main() {
    Server server(8080);
    
    // 注册 Whisper
    server.registerASR("whisper-1", [](const ASRRequest& req, auto provider) {
        std::string text = whisper_transcribe(req.audio_data, req.language);
        provider->push(OutputChunk::FinalText(text, req.model));
        provider->end();
    });
    
    // 注册 SenseVoice（更好的中文支持）
    server.registerASR("sensevoice", [](const ASRRequest& req, auto provider) {
        std::string text = sensevoice_transcribe(req.audio_data);
        provider->push(OutputChunk::FinalText(text, req.model));
        provider->end();
    });
    
    server.run();
}
```

客户端请求时通过 `model` 字段自动路由：
```bash
# 使用 Whisper
curl -X POST http://localhost:8080/v1/audio/transcriptions \
  -F "model=whisper-1" \
  -F "file=@audio.mp3"

# 使用 SenseVoice
curl -X POST http://localhost:8080/v1/audio/transcriptions \
  -F "model=sensevoice" \
  -F "file=@audio.mp3"
```

## API 参考

### Server 类

```cpp
class Server {
public:
    explicit Server(int port = 8080);
    explicit Server(const ServerOptions& options);
    
    // 注册模型回调
    void registerChat(const std::string& model_name, ChatCallback callback);
    void registerEmbedding(const std::string& model_name, EmbeddingCallback callback);
    void registerASR(const std::string& model_name, ASRCallback callback);
    void registerTTS(const std::string& model_name, TTSCallback callback);
    void registerImageGeneration(const std::string& model_name, ImageGenCallback callback);
    
    // 配置
    void setMaxConcurrency(int max);
    void setTimeout(std::chrono::milliseconds timeout);
    void setApiKey(const std::string& api_key);
    
    // 运行
    void run();           // 阻塞
    std::thread runAsync();  // 非阻塞
    void stop();
    
    // 查询
    std::vector<std::string> listModels() const;
    bool hasModel(const std::string& model_name) const;
};
```

### 回调函数签名

```cpp
// Chat
using ChatCallback = std::function<void(
    const ChatRequest& request,
    std::shared_ptr<BaseDataProvider> provider
)>;

// ASR
using ASRCallback = std::function<void(
    const ASRRequest& request,
    std::shared_ptr<BaseDataProvider> provider
)>;

// 其他类型类似...
```

### 请求结构体

```cpp
struct ChatRequest {
    std::string model;              // 模型名称（路由用）
    std::vector<Message> messages;  // 对话消息
    bool stream = false;            // 是否流式
    float temperature = 1.0f;
    int max_tokens = 2048;
    nlohmann::json raw;             // 原始 JSON
};

struct ASRRequest {
    std::string model;
    std::vector<uint8_t> audio_data;
    std::string language;
    std::string response_format;
};
```

## 构建

```bash
mkdir build && cd build
cmake .. -DOPENAI_API_BUILD_EXAMPLES=ON -DOPENAI_API_BUILD_TESTS=ON
make -j4

# 运行测试
./openai_api_test_provider
./openai_api_test_encoder
./openai_api_test_integration

# 运行示例
./examples/qwen_server/openai_api_example_qwen 8080
./examples/whisper_sensevoice/openai_api_example_whisper 8080
```

## 安装

```bash
cd build
sudo make install
```

安装后使用：
```cmake
find_package(openai_api REQUIRED)
target_link_libraries(your_app PRIVATE openai_api::server)
```

## 模型路由机制

```
HTTP Request
    ↓
Endpoint (/v1/chat/completions)
    ↓
Model Router (根据 model 字段)
    ↓
qwen-0.6b? ──→ Qwen 0.6B 回调
qwen-7b?   ──→ Qwen 7B 回调
gpt-4?     ──→ GPT-4 回调
whisper-1? ──→ Whisper 回调
sensevoice? ──→ SenseVoice 回调
```

如果请求的模型未注册，返回：
```json
{
  "error": {
    "message": "Model 'xxx' is not available. Available models: qwen-0.6b, qwen-7b",
    "type": "invalid_request_error"
  }
}
```

## 集群模式 (Cluster Mode)

支持 Master-Worker 分布式架构，多个 Worker 进程可以连接到同一个 Master，实现模型服务的水平扩展。

### 架构

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

### 使用示例

**Master 进程（第一个启动）：**

```cpp
#include <openai_api/cluster_server.hpp>

int main() {
    openai_api::ClusterServer master;
    
    // Master 也可以有自己的本地模型
    master.registerChat("master-model", [](const auto& req, auto provider) {
        // 处理请求
    });
    
    // 启动 Master，监听 8080 端口
    // 内部 Worker 管理端口为 8080 + 1000 = 9080
    master.runAsMaster(8080);
}
```

**Worker 进程（后续启动）：**

```cpp
#include <openai_api/cluster_server.hpp>

int main() {
    openai_api::ClusterServer worker;
    
    // Worker 设置自己的监听地址（用于跨机器部署）
    worker.setWorkerListenAddress("0.0.0.0", 0);  // 0 表示自动分配端口
    
    // 注册 Worker 的模型
    worker.registerChat("worker-model", [](const auto& req, auto provider) {
        // 处理请求
    });
    
    // 连接到 Master 的内部端口 (8080 + 1000)
    worker.runAsWorker("192.168.1.100", 9080);
}
```

### 特性

- **自动端口检测**：Worker 自动计算 Master 的内部端口（外部端口 + 1000）
- **模型冲突检测**：同名模型注册会失败，避免冲突
- **请求转发**：Master 自动将请求转发到对应的 Worker
- **心跳保活**：Worker 定期发送心跳，Master 自动检测断线
- **跨机器部署**：Worker 可以运行在不同的机器上，只需网络可达

### 运行示例

```bash
# 终端 1：启动 Master
./openai_api_example_cluster_master 8080

# 终端 2：启动 Worker 1（连接 Master）
./openai_api_example_cluster_worker -n worker-1 192.168.1.100 8080

# 终端 3：启动 Worker 2（连接 Master）
./openai_api_example_cluster_worker -n worker-2 192.168.1.100 8080

# 测试 - 请求会转发到对应的 Worker
curl http://192.168.1.100:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "worker-1-model", "messages": [...]}'
```

## 目录结构

```
openai-api/
├── CMakeLists.txt
├── README.md
├── include/openai_api/     # 公开头文件
│   ├── server.hpp          # 主入口
│   ├── types.hpp           # 请求/响应类型
│   ├── router.hpp          # 模型路由
│   ├── core/               # 核心组件
│   │   ├── output_chunk.hpp
│   │   └── data_provider.hpp
│   └── encoder/            # 编码器
│       └── encoder.hpp
├── src/                    # 实现
│   ├── server.cpp
│   └── router.cpp
├── examples/               # 使用示例
│   ├── qwen_server/        # Qwen 服务示例
│   └── whisper_sensevoice/ # 多 ASR 示例
└── tests/                  # 测试
```

## 许可证

MIT License
