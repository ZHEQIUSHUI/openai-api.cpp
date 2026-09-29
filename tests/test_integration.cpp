#include "openai_api/server.hpp"
#include "openai_api/core/data_provider.hpp"

#include <iostream>
#include <thread>
#include <cassert>
#include <chrono>
#include <atomic>

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

void test_chat_multimodal_parsing() {
    std::cout << "Test: chat_multimodal_parsing... " << std::flush;

    nlohmann::json req_json = {
        {"model", "vision-model"},
        {"messages", nlohmann::json::array({
            {
                {"role", "user"},
                {"content", nlohmann::json::array({
                    {
                        {"type", "text"},
                        {"text", "Describe this image"}
                    },
                    {
                        {"type", "image_url"},
                        {"image_url", {
                            {"url", "https://example.com/cat.png"},
                            {"detail", "high"}
                        }}
                    }
                })}
            }
        })}
    };

    auto req = ChatRequest::from_json(req_json);
    assert(req.has_image_inputs());
    assert(req.image_input_count() == 1);
    assert(req.image_urls().size() == 1);
    assert(req.image_urls()[0] == "https://example.com/cat.png");
    assert(req.flattened_text() == "Describe this image");

    std::cout << "PASSED" << std::endl;
}

void test_chat_multimodal_capabilities() {
    std::cout << "Test: chat_multimodal_capabilities... " << std::flush;

    Server server;
    std::atomic<bool> vision_callback_called{false};
    ChatModelOptions text_only_options;
    text_only_options.supports_vision = false;
    text_only_options.extra_fields["context_window"] = 8192;
    text_only_options.extra_fields["max_output_tokens"] = 2048;

    ChatModelOptions vision_options;
    vision_options.supports_vision = true;
    vision_options.extra_fields["context_window"] = 32768;
    vision_options.extra_fields["max_output_tokens"] = 8192;

    server.registerChat("text-only", [](const ChatRequest& req, auto provider) {
        provider->push(OutputChunk::FinalText("text-only", req.model));
        provider->end();
    }, text_only_options);

    server.registerChat("vision-model", [&vision_callback_called](const ChatRequest& req, auto provider) {
        vision_callback_called = req.has_image_inputs();
        provider->push(OutputChunk::FinalText("vision-ok", req.model));
        provider->end();
    }, vision_options);

    server.registerEmbedding("embed-model", [](const EmbeddingRequest& req, auto provider) {
        provider->end();
    });
    server.setModelExtraFields("embed-model", {{"prefill_max_token_num", 1408}, {"max_token_len", 1407}});
    // generic fields merge over the chat options' extra_fields
    server.setModelExtraFields("text-only", {{"max_output_tokens", 1024}});

    ServerOptions options;
    options.host = "127.0.0.1";
    options.port = 18126;
    options.default_timeout = std::chrono::milliseconds(3000);

    std::thread server_thread([&server, options]() {
        server.run(options);
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(300));

    httplib::Client client("127.0.0.1", 18126);
    client.set_connection_timeout(3);
    client.set_read_timeout(3);

    auto models_res = client.Get("/v1/models");
    assert(models_res && models_res->status == 200);
    auto models_json = nlohmann::json::parse(models_res->body);

    bool saw_text_only = false;
    bool saw_vision = false;
    bool saw_embed = false;
    for (const auto& model : models_json["data"]) {
        if (model["id"] == "text-only") {
            saw_text_only = true;
            assert(model["capabilities"]["vision"] == false);
            assert(model["context_window"] == 8192);
            assert(model["max_output_tokens"] == 1024);
            assert(model["input_modalities"].size() == 1);
        } else if (model["id"] == "vision-model") {
            saw_vision = true;
            assert(model["capabilities"]["vision"] == true);
            assert(model["context_window"] == 32768);
            assert(model["max_output_tokens"] == 8192);
            assert(model["input_modalities"].size() == 2);
        } else if (model["id"] == "embed-model") {
            saw_embed = true;
            assert(model["prefill_max_token_num"] == 1408);
            assert(model["max_token_len"] == 1407);
            assert(!model.contains("capabilities"));
        }
    }
    assert(saw_text_only);
    assert(saw_vision);
    assert(saw_embed);

    nlohmann::json image_chat = {
        {"model", "text-only"},
        {"messages", nlohmann::json::array({
            {
                {"role", "user"},
                {"content", nlohmann::json::array({
                    {{"type", "text"}, {"text", "what is in the image?"}},
                    {{"type", "image_url"}, {"image_url", {{"url", "https://example.com/a.png"}}}}
                })}
            }
        })}
    };
    auto reject_res = client.Post("/v1/chat/completions", image_chat.dump(), "application/json");
    assert(reject_res && reject_res->status == 400);

    image_chat["model"] = "vision-model";
    auto accept_res = client.Post("/v1/chat/completions", image_chat.dump(), "application/json");
    assert(accept_res && accept_res->status == 200);
    assert(vision_callback_called.load());

    server.stop();
    server_thread.join();

    std::cout << "PASSED" << std::endl;
}

// 测试完整的端到端流程
void test_embedding_errors() {
    std::cout << "Test: embedding_errors... " << std::flush;

    Server server;
    server.registerEmbedding("embed-error", [](const EmbeddingRequest& req, auto provider) {
        provider->push(OutputChunk::Error("model_error", "input too long"));
        provider->end();
    });
    server.registerEmbedding("embed-silent", [](const EmbeddingRequest& req, auto provider) {
        provider->end();  // no output at all
    });

    ServerOptions options;
    options.host = "127.0.0.1";
    options.port = 18127;
    options.default_timeout = std::chrono::milliseconds(5000);
    std::thread server_thread([&server, options]() { server.run(options); });
    std::this_thread::sleep_for(std::chrono::milliseconds(300));

    httplib::Client client("127.0.0.1", 18127);
    client.set_read_timeout(10);

    auto res = client.Post("/v1/embeddings", R"({"model":"embed-error","input":"hi"})", "application/json");
    assert(res && res->status == 400);
    auto j = nlohmann::json::parse(res->body);
    assert(j["error"]["message"] == "input too long");
    assert(j["error"]["code"] == "model_error");

    auto t0 = std::chrono::steady_clock::now();
    res = client.Post("/v1/embeddings", R"({"model":"embed-silent","input":"hi"})", "application/json");
    auto elapsed = std::chrono::steady_clock::now() - t0;
    assert(res && res->status == 500);  // not a 504 "timeout" after waiting default_timeout
    assert(nlohmann::json::parse(res->body)["error"]["message"] == "Model returned no output");
    assert(elapsed < std::chrono::milliseconds(3000));

    server.stop();
    server_thread.join();
    std::cout << "PASSED" << std::endl;
}

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
    test_chat_multimodal_parsing();
    test_chat_multimodal_capabilities();
    test_embedding_errors();
    test_end_to_end();
    test_error_handling();
    test_model_routing();
    
    std::cout << "\nAll tests PASSED!" << std::endl;
    return 0;
}
