#pragma once

#include "utils/json.hpp"
#include <string>
#include <vector>
#include <cstdint>

namespace openai_api {

// ============ 请求类型 ============

struct ChatRequest {
    std::string model;
    nlohmann::json messages;
    bool stream = false;
    float temperature = 1.0f;
    float top_p = 1.0f;
    int max_tokens = 2048;
    int n = 1;
    std::vector<std::string> stop;
    float presence_penalty = 0.0f;
    float frequency_penalty = 0.0f;
    
    // 原始 JSON（供扩展用）
    nlohmann::json raw;
    
    static ChatRequest from_json(const nlohmann::json& j) {
        ChatRequest req;
        req.raw = j;
        
        if (j.contains("model")) req.model = j["model"].get<std::string>();
        if (j.contains("stream")) req.stream = j["stream"].get<bool>();
        if (j.contains("temperature")) req.temperature = j["temperature"].get<float>();
        if (j.contains("top_p")) req.top_p = j["top_p"].get<float>();
        if (j.contains("max_tokens")) req.max_tokens = j["max_tokens"].get<int>();
        if (j.contains("n")) req.n = j["n"].get<int>();
        if (j.contains("presence_penalty")) req.presence_penalty = j["presence_penalty"].get<float>();
        if (j.contains("frequency_penalty")) req.frequency_penalty = j["frequency_penalty"].get<float>();
        
        if (j.contains("messages") && j["messages"].is_array()) {
            req.messages = j["messages"];
        }
        
        if (j.contains("stop")) {
            if (j["stop"].is_string()) {
                req.stop.push_back(j["stop"].get<std::string>());
            } else if (j["stop"].is_array()) {
                for (const auto& s : j["stop"]) {
                    req.stop.push_back(s.get<std::string>());
                }
            }
        }
        
        return req;
    }
};

struct EmbeddingRequest {
    std::string model;
    std::vector<std::string> inputs;  // 支持批量输入
    std::string encoding_format = "float";
    int dimensions = -1;
    
    nlohmann::json raw;
    
    static EmbeddingRequest from_json(const nlohmann::json& j) {
        EmbeddingRequest req;
        req.raw = j;
        
        if (j.contains("model")) req.model = j["model"].get<std::string>();
        if (j.contains("encoding_format")) req.encoding_format = j["encoding_format"].get<std::string>();
        if (j.contains("dimensions")) req.dimensions = j["dimensions"].get<int>();
        
        if (j.contains("input")) {
            if (j["input"].is_string()) {
                req.inputs.push_back(j["input"].get<std::string>());
            } else if (j["input"].is_array()) {
                for (const auto& item : j["input"]) {
                    if (item.is_string()) {
                        req.inputs.push_back(item.get<std::string>());
                    }
                }
            }
        }
        
        return req;
    }
};

struct ASRRequest {
    std::string model;
    std::vector<uint8_t> audio_data;
    std::string filename;  // 原始文件名
    std::string language;
    std::string prompt;
    std::string response_format = "json";  // json, text, srt, verbose_json, vtt
    float temperature = 0.0f;
    
    // multipart form data 原始内容
    std::string raw_body;
    
    static ASRRequest from_multipart(const std::string& body, const std::string& content_type);
};

struct TTSRequest {
    std::string model;
    std::string input;
    std::string voice = "alloy";  // alloy, echo, fable, onyx, nova, shimmer
    std::string response_format = "mp3";  // mp3, opus, aac, flac, wav, pcm
    float speed = 1.0f;
    
    nlohmann::json raw;
    
    static TTSRequest from_json(const nlohmann::json& j) {
        TTSRequest req;
        req.raw = j;
        
        if (j.contains("model")) req.model = j["model"].get<std::string>();
        if (j.contains("input")) req.input = j["input"].get<std::string>();
        if (j.contains("voice")) req.voice = j["voice"].get<std::string>();
        if (j.contains("response_format")) req.response_format = j["response_format"].get<std::string>();
        if (j.contains("speed")) req.speed = j["speed"].get<float>();
        
        return req;
    }
};

struct ImageGenRequest {
    std::string prompt;
    std::string model = "dall-e-2";
    int n = 1;
    std::string quality = "standard";
    std::string response_format = "url";  // url, b64_json
    std::string size = "1024x1024";
    std::string style = "vivid";
    
    nlohmann::json raw;
    
    static ImageGenRequest from_json(const nlohmann::json& j) {
        ImageGenRequest req;
        req.raw = j;
        
        if (j.contains("prompt")) req.prompt = j["prompt"].get<std::string>();
        if (j.contains("model")) req.model = j["model"].get<std::string>();
        if (j.contains("n")) req.n = j["n"].get<int>();
        if (j.contains("quality")) req.quality = j["quality"].get<std::string>();
        if (j.contains("response_format")) req.response_format = j["response_format"].get<std::string>();
        if (j.contains("size")) req.size = j["size"].get<std::string>();
        if (j.contains("style")) req.style = j["style"].get<std::string>();
        
        return req;
    }
};

} // namespace openai_api
