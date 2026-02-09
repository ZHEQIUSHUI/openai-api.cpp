#include "openai_api/encoder/encoder.hpp"

#include <iostream>
#include <cassert>
#include "utils/json.hpp"

using namespace openai_api;

// 测试 ChatCompletionsSSEEncoder
void test_chat_sse_encoder() {
    std::cout << "Test: chat_sse_encoder... " << std::flush;
    
    ChatCompletionsSSEEncoder encoder;
    
    // 测试 TextDelta
    auto chunk = OutputChunk::TextDelta("Hello", "gpt-4");
    chunk.id = "test-id";
    
    std::string encoded = encoder.encode(chunk);
    assert(encoded.find("data: ") == 0);
    assert(encoded.find("chat.completion.chunk") != std::string::npos);
    assert(encoded.find("Hello") != std::string::npos);
    
    // 测试 End
    auto end_chunk = OutputChunk::EndMarker();
    std::string end_encoded = encoder.encode(end_chunk);
    assert(end_encoded.find("[DONE]") != std::string::npos);
    
    // 测试 Error
    auto error_chunk = OutputChunk::Error("test_error", "Error message");
    std::string error_encoded = encoder.encode(error_chunk);
    assert(error_encoded.find("error") != std::string::npos);
    
    std::cout << "PASSED" << std::endl;
}

// 测试 ChatCompletionsJSONEncoder
void test_chat_json_encoder() {
    std::cout << "Test: chat_json_encoder... " << std::flush;
    
    ChatCompletionsJSONEncoder encoder;
    
    auto chunk = OutputChunk::FinalText("Hello, World!", "gpt-4");
    chunk.id = "test-id";
    chunk.created = 1234567890;
    
    std::string encoded = encoder.encode(chunk);
    
    // 解析 JSON 验证
    nlohmann::json j = nlohmann::json::parse(encoded);
    assert(j["object"] == "chat.completion");
    assert(j["model"] == "gpt-4");
    assert(j["choices"][0]["message"]["content"] == "Hello, World!");
    assert(j["choices"][0]["message"]["role"] == "assistant");
    assert(j["choices"][0]["finish_reason"] == "stop");
    
    std::cout << "PASSED" << std::endl;
}

// 测试 EmbeddingsJSONEncoder
void test_embeddings_encoder() {
    std::cout << "Test: embeddings_encoder... " << std::flush;
    
    EmbeddingsJSONEncoder encoder;
    
    // 测试批量 Embedding
    std::vector<std::vector<float>> embeds = {
        {0.1f, 0.2f, 0.3f},
        {0.4f, 0.5f, 0.6f}
    };
    
    auto chunk = OutputChunk::BatchEmbeddings(embeds, "text-embedding-ada-002");
    std::string encoded = encoder.encode(chunk);
    
    nlohmann::json j = nlohmann::json::parse(encoded);
    assert(j["object"] == "list");
    assert(j["data"].size() == 2);
    assert(j["data"][0]["object"] == "embedding");
    assert(j["data"][0]["index"] == 0);
    assert(j["data"][0]["embedding"].size() == 3);
    assert(j["model"] == "text-embedding-ada-002");
    
    std::cout << "PASSED" << std::endl;
}

// 测试 ASR encoders
void test_asr_encoders() {
    std::cout << "Test: asr_encoders... " << std::flush;
    
    // JSON Encoder
    {
        ASRJSONEncoder encoder;
        auto chunk = OutputChunk::FinalText("Hello world", "whisper-1");
        std::string encoded = encoder.encode(chunk);
        
        nlohmann::json j = nlohmann::json::parse(encoded);
        assert(j["text"] == "Hello world");
    }
    
    // Text Encoder
    {
        ASRTextEncoder encoder;
        auto chunk = OutputChunk::FinalText("Hello world", "whisper-1");
        std::string encoded = encoder.encode(chunk);
        assert(encoded == "Hello world");
    }
    
    std::cout << "PASSED" << std::endl;
}

// 测试 ErrorEncoder
void test_error_encoder() {
    std::cout << "Test: error_encoder... " << std::flush;
    
    std::string encoded = ErrorEncoder::invalid_request("Invalid parameter");
    nlohmann::json j = nlohmann::json::parse(encoded);
    
    assert(j["error"]["type"] == "invalid_request_error");
    assert(j["error"]["message"] == "Invalid parameter");
    
    // 测试 rate_limit
    std::string rate_limit = ErrorEncoder::rate_limit();
    nlohmann::json j2 = nlohmann::json::parse(rate_limit);
    assert(j2["error"]["type"] == "rate_limit_exceeded");
    
    std::cout << "PASSED" << std::endl;
}

// 测试 Encoder 的 done_marker
void test_done_marker() {
    std::cout << "Test: done_marker... " << std::flush;
    
    ChatCompletionsSSEEncoder encoder;
    std::string done = encoder.done_marker();
    assert(done == "data: [DONE]\n\n");
    
    std::cout << "PASSED" << std::endl;
}

// 测试 is_done
void test_is_done() {
    std::cout << "Test: is_done... " << std::flush;
    
    ChatCompletionsSSEEncoder encoder;
    
    auto end_chunk = OutputChunk::EndMarker();
    assert(encoder.is_done(end_chunk));
    
    auto text_chunk = OutputChunk::TextDelta("test", "gpt-4");
    assert(!encoder.is_done(text_chunk));
    
    std::cout << "PASSED" << std::endl;
}

int main() {
    std::cout << "=== Encoder Tests ===" << std::endl;
    
    test_chat_sse_encoder();
    test_chat_json_encoder();
    test_embeddings_encoder();
    test_asr_encoders();
    test_error_encoder();
    test_done_marker();
    test_is_done();
    
    std::cout << "\nAll tests PASSED!" << std::endl;
    return 0;
}
