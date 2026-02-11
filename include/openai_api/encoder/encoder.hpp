#pragma once

#include "openai_api/core/api_export.hpp"
#include "openai_api/core/output_chunk.hpp"
#include "openai_api/core/data_provider.hpp"

#include <string>
#include <sstream>
#include <iomanip>
#include <random>

namespace openai_api {

/**
 * Encoder 基类
 * 负责将 OutputChunk 编码为具体的协议格式
 */
class OPENAI_API_API Encoder {
public:
    virtual ~Encoder() = default;
    
    // 编码单个 chunk 为字符串
    virtual std::string encode(const OutputChunk& chunk) = 0;
    
    // 检查是否是结束标记
    virtual bool is_done(const OutputChunk& chunk) const {
        return chunk.is_end();
    }
    
    // 生成完成标记（如 SSE 的 [DONE]）
    virtual std::string done_marker() const {
        return "";
    }
    
protected:
    // 生成唯一的 ID
    std::string generate_id(const std::string& prefix = "chatcmpl") const {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        static std::uniform_int_distribution<> dis(0, 15);
        
        std::stringstream ss;
        ss << prefix << "-";
        for (int i = 0; i < 24; ++i) {
            ss << std::hex << dis(gen);
        }
        return ss.str();
    }
};

/**
 * Chat Completions SSE Encoder
 * 将 OutputChunk 编码为 OpenAI SSE 流格式
 */
class ChatCompletionsSSEEncoder : public Encoder {
public:
    std::string encode(const OutputChunk& chunk) override {
        nlohmann::json data;
        
        switch (chunk.type) {
            case OutputChunkType::TextDelta:
                data = create_sse_delta(chunk);
                break;
            case OutputChunkType::FinalText:
                data = create_sse_finish(chunk);
                break;
            case OutputChunkType::Error:
                data = create_sse_error(chunk);
                break;
            case OutputChunkType::End:
                return "data: [DONE]\n\n";
            default:
                return "";
        }
        
        return "data: " + data.dump() + "\n\n";
    }
    
    std::string done_marker() const override {
        return "data: [DONE]\n\n";
    }

private:
    nlohmann::json create_sse_delta(const OutputChunk& chunk) {
        nlohmann::json j;
        j["id"] = chunk.id.empty() ? generate_id() : chunk.id;
        j["object"] = "chat.completion.chunk";
        j["created"] = chunk.created ? chunk.created : std::time(nullptr);
        j["model"] = chunk.model.empty() ? "gpt-4" : chunk.model;
        
        nlohmann::json choice;
        choice["index"] = chunk.index;
        choice["delta"]["content"] = chunk.text;
        choice["delta"]["role"] = "assistant";
        choice["finish_reason"] = nullptr;
        
        j["choices"] = nlohmann::json::array({choice});
        return j;
    }
    
    nlohmann::json create_sse_finish(const OutputChunk& chunk) {
        nlohmann::json j;
        j["id"] = chunk.id.empty() ? generate_id() : chunk.id;
        j["object"] = "chat.completion.chunk";
        j["created"] = chunk.created ? chunk.created : std::time(nullptr);
        j["model"] = chunk.model.empty() ? "gpt-4" : chunk.model;
        
        nlohmann::json choice;
        choice["index"] = chunk.index;
        choice["delta"] = nlohmann::json::object();
        choice["finish_reason"] = "stop";
        
        j["choices"] = nlohmann::json::array({choice});
        return j;
    }
    
    nlohmann::json create_sse_error(const OutputChunk& chunk) {
        nlohmann::json j;
        j["error"]["message"] = chunk.error_message;
        j["error"]["type"] = chunk.error_code;
        return j;
    }
};

/**
 * Chat Completions JSON Encoder
 * 将 OutputChunk 编码为 OpenAI 非流式 JSON 格式
 */
class ChatCompletionsJSONEncoder : public Encoder {
public:
    std::string encode(const OutputChunk& chunk) override {
        nlohmann::json j;
        j["id"] = chunk.id.empty() ? generate_id() : chunk.id;
        j["object"] = "chat.completion";
        j["created"] = chunk.created ? chunk.created : std::time(nullptr);
        j["model"] = chunk.model.empty() ? "gpt-4" : chunk.model;
        
        nlohmann::json choice;
        choice["index"] = chunk.index;
        choice["message"]["role"] = "assistant";
        choice["message"]["content"] = chunk.text;
        choice["finish_reason"] = "stop";
        
        j["choices"] = nlohmann::json::array({choice});
        
        // 默认 usage 信息
        j["usage"]["prompt_tokens"] = 0;
        j["usage"]["completion_tokens"] = 0;
        j["usage"]["total_tokens"] = 0;
        
        return j.dump(2);
    }
};

/**
 * Embeddings JSON Encoder
 */
class EmbeddingsJSONEncoder : public Encoder {
public:
    std::string encode(const OutputChunk& chunk) override {
        nlohmann::json j;
        j["object"] = "list";
        
        if (chunk.type == OutputChunkType::Embedding) {
            // 单个 embedding
            j["data"] = nlohmann::json::array({embedding_to_json(chunk, chunk.index)});
        } else if (chunk.type == OutputChunkType::Embeddings) {
            // 批量 embeddings
            j["data"] = nlohmann::json::array();
            for (size_t i = 0; i < chunk.embeds.size(); ++i) {
                OutputChunk emb_chunk;
                emb_chunk.embedding = chunk.embeds[i];
                j["data"].push_back(embedding_to_json(emb_chunk, static_cast<int>(i)));
            }
        }
        
        j["model"] = chunk.model.empty() ? "text-embedding-ada-002" : chunk.model;
        j["usage"]["prompt_tokens"] = 0;
        j["usage"]["total_tokens"] = 0;
        
        return j.dump(2);
    }

private:
    nlohmann::json embedding_to_json(const OutputChunk& chunk, int index) {
        nlohmann::json j;
        j["object"] = "embedding";
        j["index"] = index;
        j["embedding"] = chunk.embedding;
        return j;
    }
};

/**
 * ASR JSON Encoder (Whisper API 格式)
 */
class ASRJSONEncoder : public Encoder {
public:
    std::string encode(const OutputChunk& chunk) override {
        nlohmann::json j;
        j["text"] = chunk.text;
        return j.dump(2);
    }
};

/**
 * ASR Text Encoder (纯文本格式)
 */
class ASRTextEncoder : public Encoder {
public:
    std::string encode(const OutputChunk& chunk) override {
        return chunk.text;
    }
};

/**
 * ASR Verbose JSON Encoder (带详细信息的 Whisper 格式)
 */
class ASRVerboseJSONEncoder : public Encoder {
public:
    std::string encode(const OutputChunk& chunk) override {
        nlohmann::json j;
        j["task"] = "transcribe";
        j["language"] = "zh";
        j["duration"] = 0.0;
        j["text"] = chunk.text;
        j["segments"] = nlohmann::json::array();
        
        if (chunk.obj.contains("segments")) {
            j["segments"] = chunk.obj["segments"];
        }
        
        return j.dump(2);
    }
};

/**
 * TTS Binary Encoder
 * 直接返回音频字节（不需要字符串编码）
 */
class TTSBinaryEncoder : public Encoder {
public:
    std::string encode(const OutputChunk& chunk) override {
        // 对于二进制数据，直接返回空字符串
        // 实际数据通过 bytes 字段传递
        return "";
    }
    
    // 获取 MIME 类型
    std::string get_mime_type(const OutputChunk& chunk) const {
        return chunk.mime_type.empty() ? "audio/mp3" : chunk.mime_type;
    }
};

/**
 * Images JSON Encoder (DALL-E 格式)
 */
class ImagesJSONEncoder : public Encoder {
public:
    std::string encode(const OutputChunk& chunk) override {
        nlohmann::json j;
        j["created"] = chunk.created ? chunk.created : std::time(nullptr);
        
        nlohmann::json item;
        if (chunk.type == OutputChunkType::ImageBytes) {
            // Base64 编码图片
            item["b64_json"] = base64_encode(chunk.bytes);
        } else if (chunk.type == OutputChunkType::JsonObject) {
            // URL 格式
            return chunk.obj.dump(2);
        }
        
        item["revised_prompt"] = "";
        j["data"] = nlohmann::json::array({item});
        
        return j.dump(2);
    }

private:
    static std::string base64_encode(const std::vector<uint8_t>& data) {
        static const char base64_chars[] = 
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        
        std::string encoded;
        int i = 0;
        uint8_t array_3[3];
        uint8_t array_4[4];
        
        for (uint8_t c : data) {
            array_3[i++] = c;
            if (i == 3) {
                array_4[0] = (array_3[0] & 0xfc) >> 2;
                array_4[1] = ((array_3[0] & 0x03) << 4) + ((array_3[1] & 0xf0) >> 4);
                array_4[2] = ((array_3[1] & 0x0f) << 2) + ((array_3[2] & 0xc0) >> 6);
                array_4[3] = array_3[2] & 0x3f;
                
                for (int j = 0; j < 4; ++j)
                    encoded += base64_chars[array_4[j]];
                i = 0;
            }
        }
        
        if (i) {
            for (int j = i; j < 3; ++j)
                array_3[j] = '\0';
            
            array_4[0] = (array_3[0] & 0xfc) >> 2;
            array_4[1] = ((array_3[0] & 0x03) << 4) + ((array_3[1] & 0xf0) >> 4);
            array_4[2] = ((array_3[1] & 0x0f) << 2) + ((array_3[2] & 0xc0) >> 6);
            
            for (int j = 0; j < (i + 1); ++j)
                encoded += base64_chars[array_4[j]];
            
            while ((i++ < 3))
                encoded += '=';
        }
        
        return encoded;
    }
};

/**
 * 错误响应 Encoder（OpenAI 错误格式）
 */
class ErrorEncoder {
public:
    static std::string encode(const std::string& code, const std::string& message, int http_status = 400) {
        nlohmann::json j;
        j["error"]["message"] = message;
        j["error"]["type"] = code;
        j["error"]["code"] = code;
        return j.dump(2);
    }
    
    static std::string invalid_request(const std::string& message) {
        return encode("invalid_request_error", message, 400);
    }
    
    static std::string rate_limit() {
        return encode("rate_limit_exceeded", "Rate limit exceeded", 429);
    }
    
    static std::string server_error(const std::string& message) {
        return encode("server_error", message, 500);
    }
    
    static std::string not_found() {
        return encode("not_found", "The requested resource was not found", 404);
    }
};

} // namespace openai_api
