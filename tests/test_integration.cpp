#include "openai_api/server.hpp"
#include "openai_api/core/data_provider.hpp"

#include <iostream>
#include <thread>
#include <cassert>
#include <chrono>

using namespace openai_api;

// 测试 ServerOptions
void test_server_options() {
    std::cout << "Test: server_options... " << std::flush;
    
    ServerOptions options;
    assert(options.host == "0.0.0.0");
    assert(options.port == 8080);
    assert(options.max_concurrency == 10);
    
    std::cout << "PASSED" << std::endl;
}

// 测试模型注册
void test_model_registration() {
    std::cout << "Test: model_registration... " << std::flush;
    
    // 创建服务器（不传配置，run() 时再传入）
    Server server;
    
    // 注册模型
    server.registerChat("gpt-4", [](const ChatRequest& req, auto provider) {
        provider->push(OutputChunk::FinalText("Hello", req.model));
        provider->end();
    });
    
    server.registerASR("whisper-1", [](const ASRRequest& req, auto provider) {
        provider->push(OutputChunk::FinalText("Transcription", req.model));
        provider->end();
    });
    
    // 验证
    auto models = server.listModels();
    assert(models.size() == 2);
    assert(server.hasModel("gpt-4"));
    assert(server.hasModel("whisper-1"));
    assert(!server.hasModel("nonexistent"));
    
    std::cout << "PASSED" << std::endl;
}

// 测试完整的端到端流程
void test_end_to_end() {
    std::cout << "Test: end_to_end... " << std::flush;
    
    // 创建 Provider
    auto provider = std::make_shared<QueueProvider>(std::chrono::milliseconds(5000));
    
    // 模拟模型推理（生产者线程）
    std::thread producer([provider]() {
        provider->push(OutputChunk::TextDelta("Hello", "gpt-4"));
        provider->push(OutputChunk::TextDelta(" ", "gpt-4"));
        provider->push(OutputChunk::TextDelta("World", "gpt-4"));
        provider->push(OutputChunk::FinalText("Hello World", "gpt-4"));
        provider->end();
    });
    
    // 模拟消费者
    std::vector<std::string> deltas;
    ChatCompletionsSSEEncoder encoder;
    
    while (true) {
        auto chunk = provider->wait_pop_for(std::chrono::milliseconds(1000));
        if (!chunk.has_value() || chunk->is_end()) {
            break;
        }
        deltas.push_back(encoder.encode(chunk.value()));
    }
    
    producer.join();
    assert(deltas.size() >= 3);
    
    std::cout << "PASSED" << std::endl;
}

// 测试错误处理
void test_error_handling() {
    std::cout << "Test: error_handling... " << std::flush;
    
    auto provider = std::make_shared<QueueProvider>();
    provider->push(OutputChunk::Error("test_error", "Test error message"));
    provider->end();
    
    auto chunk = provider->pop();
    assert(chunk.has_value());
    assert(chunk->is_error());
    assert(chunk->error_code == "test_error");
    assert(chunk->error_message == "Test error message");
    
    std::cout << "PASSED" << std::endl;
}

// 测试路由功能
void test_model_routing() {
    std::cout << "Test: model_routing... " << std::flush;
    
    ModelRouter router;
    
    // 注册多个模型
    bool whisper_called = false;
    bool sensevoice_called = false;
    
    router.registerASR("whisper-1", [&whisper_called](const ASRRequest& req, auto provider) {
        whisper_called = true;
        provider->push(OutputChunk::FinalText("Whisper result", req.model));
        provider->end();
    });
    
    router.registerASR("sensevoice", [&sensevoice_called](const ASRRequest& req, auto provider) {
        sensevoice_called = true;
        provider->push(OutputChunk::FinalText("SenseVoice result", req.model));
        provider->end();
    });
    
    // 路由到 whisper
    {
        ASRRequest req;
        req.model = "whisper-1";
        auto provider = std::make_shared<QueueProvider>();
        assert(router.routeASR(req, provider));
        auto chunk = provider->wait_pop_for(std::chrono::seconds(1));
        assert(chunk.has_value());
        assert(whisper_called);
    }
    
    // 路由到 sensevoice
    {
        ASRRequest req;
        req.model = "sensevoice";
        auto provider = std::make_shared<QueueProvider>();
        assert(router.routeASR(req, provider));
        auto chunk = provider->wait_pop_for(std::chrono::seconds(1));
        assert(chunk.has_value());
        assert(sensevoice_called);
    }
    
    // 不存在的模型
    {
        ASRRequest req;
        req.model = "nonexistent";
        auto provider = std::make_shared<QueueProvider>();
        assert(!router.routeASR(req, provider));
    }
    
    std::cout << "PASSED" << std::endl;
}

int main() {
    std::cout << "=== Integration Tests ===" << std::endl;
    
    test_server_options();
    test_model_registration();
    test_end_to_end();
    test_error_handling();
    test_model_routing();
    
    std::cout << "\nAll tests PASSED!" << std::endl;
    return 0;
}
