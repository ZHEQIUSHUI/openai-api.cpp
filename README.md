# OpenAI API Server Library

一个 C++ 库，用于构建兼容 OpenAI API 的推理服务。支持多模型路由，同一端点可以注册多个算法实现。

## 特性

- **库形式**：作为 CMake 库供其他项目引入
- **多模型路由**：同一端点支持多个算法实现（如 ASR 同时支持 Whisper 和 SenseVoice）
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
cmake .. -DBUILD_EXAMPLES=ON -DBUILD_TESTS=ON
make -j4

# 运行测试
./test_provider
./test_encoder
./test_integration

# 运行示例
./examples/qwen_server/qwen_server 8080
./examples/whisper_sensevoice/whisper_sensevoice 8080
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
