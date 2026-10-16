#pragma once

#include "openai_api/core/api_export.hpp"
#include "utils/json.hpp"
#include <string>
#include <vector>
#include <cstdint>
#include <optional>
#include <algorithm>

namespace openai_api {

// ============ 请求类型 ============

struct ChatContentPart {
    std::string type;
    std::string text;
    std::string image_url;
    std::string image_detail;
    nlohmann::json raw;

    bool is_text() const { return type == "text"; }
    bool is_image() const { return type == "image_url" || type == "image"; }

    // Input modality a content part carries: "text", "image", "video", "audio", or "" when unknown.
    // Accepts the OpenAI part types plus the vLLM/SGLang extensions (video_url, audio_url).
    std::string modality() const {
        if (type == "text") return "text";
        if (type == "image_url" || type == "image") return "image";
        if (type == "video_url" || type == "video") return "video";
        if (type == "input_audio" || type == "audio_url" || type == "audio") return "audio";
        return "";
    }
};

struct ParsedChatMessage {
    std::string role;
    std::vector<ChatContentPart> content_parts;
    nlohmann::json raw;

    bool has_images() const {
        for (const auto& part : content_parts) {
            if (part.is_image()) return true;
        }
        return false;
    }
};

struct ChatModelOptions {
    std::optional<bool> supports_vision;
    // Input modalities the model accepts, e.g. {"text", "image", "video"}. When set, requests that
    // carry any other modality are rejected with 400 and /v1/models lists exactly these. When unset,
    // supports_vision decides ({"text", "image"} or {"text"}); when both are unset nothing is checked.
    std::optional<std::vector<std::string>> input_modalities;
    nlohmann::json extra_fields = nlohmann::json::object();

    std::optional<std::vector<std::string>> resolved_input_modalities() const {
        if (input_modalities.has_value()) return input_modalities;
        if (supports_vision.has_value()) {
            if (supports_vision.value()) return std::vector<std::string>{"text", "image"};
            return std::vector<std::string>{"text"};
        }
        return std::nullopt;
    }
};

struct ChatRequest {
    std::string model;
    nlohmann::json messages;
    std::vector<ParsedChatMessage> parsed_messages;
    bool stream = false;
    bool has_temperature = false;
    float temperature = 1.0f;
    bool has_top_p = false;
    float top_p = 1.0f;
    bool has_enable_thinking = false;
    bool enable_thinking = true;
    bool has_thinking_mode = false;
    std::string thinking_mode; // raw string ("think"/"no_think"/"default"); interpreted by the app
    bool has_max_tokens = false; // max_tokens or max_completion_tokens was given
    int max_tokens = 2048;
    int n = 1;
    std::vector<std::string> stop;
    bool has_presence_penalty = false;
    float presence_penalty = 0.0f;
    bool has_frequency_penalty = false;
    float frequency_penalty = 0.0f;
    // vLLM / SGLang sampling extensions, plus OpenAI seed
    bool has_top_k = false;
    int top_k = -1;
    bool has_min_p = false;
    float min_p = 0.0f;
    bool has_repetition_penalty = false;
    float repetition_penalty = 1.0f;
    bool has_seed = false;
    int64_t seed = 0;
    // stream_options.include_usage: send usage in a final chunk with empty choices
    bool include_usage = false;
    // Legacy /v1/completions request routed to the chat callback: `prompt` is raw text that the
    // model continues without a chat template; `echo` prepends it to the returned text.
    bool is_completion = false;
    std::string prompt;
    bool echo = false;
    
    // 原始 JSON（供扩展用）
    nlohmann::json raw;

    bool has_image_inputs() const {
        for (const auto& message : parsed_messages) {
            if (message.has_images()) return true;
        }
        return false;
    }

    size_t image_input_count() const {
        size_t count = 0;
        for (const auto& message : parsed_messages) {
            for (const auto& part : message.content_parts) {
                if (part.is_image()) ++count;
            }
        }
        return count;
    }

    // Distinct non-text modalities used by the request, in first-seen order.
    std::vector<std::string> media_modalities() const {
        std::vector<std::string> out;
        for (const auto& message : parsed_messages) {
            for (const auto& part : message.content_parts) {
                const std::string m = part.modality();
                if (m.empty() || m == "text") continue;
                if (std::find(out.begin(), out.end(), m) == out.end()) out.push_back(m);
            }
        }
        return out;
    }

    std::vector<std::string> image_urls() const {
        std::vector<std::string> urls;
        for (const auto& message : parsed_messages) {
            for (const auto& part : message.content_parts) {
                if (part.is_image() && !part.image_url.empty()) {
                    urls.push_back(part.image_url);
                }
            }
        }
        return urls;
    }

    std::string flattened_text() const {
        std::string text;
        for (const auto& message : parsed_messages) {
            for (const auto& part : message.content_parts) {
                if (part.is_text() && !part.text.empty()) {
                    if (!text.empty()) text += "\n";
                    text += part.text;
                }
            }
        }
        return text;
    }
    
    static ChatRequest from_json(const nlohmann::json& j) {
        ChatRequest req;
        req.raw = j;
        
        // Fields are read only when they have the expected JSON type; clients routinely send
        // null for "not set" (e.g. "temperature": null), which must not fail the request.
        if (j.contains("model") && j["model"].is_string()) req.model = j["model"].get<std::string>();
        if (j.contains("stream") && j["stream"].is_boolean()) req.stream = j["stream"].get<bool>();
        if (j.contains("temperature") && j["temperature"].is_number()) { req.has_temperature = true; req.temperature = j["temperature"].get<float>(); }
        if (j.contains("top_p") && j["top_p"].is_number()) { req.has_top_p = true; req.top_p = j["top_p"].get<float>(); }
        if (j.contains("enable_thinking") && j["enable_thinking"].is_boolean()) {
            req.has_enable_thinking = true;
            req.enable_thinking = j["enable_thinking"].get<bool>();
        } else if (j.contains("chat_template_kwargs") &&
                   j["chat_template_kwargs"].is_object() &&
                   j["chat_template_kwargs"].contains("enable_thinking") &&
                   j["chat_template_kwargs"]["enable_thinking"].is_boolean()) {
            req.has_enable_thinking = true;
            req.enable_thinking = j["chat_template_kwargs"]["enable_thinking"].get<bool>();
        }
        if (j.contains("thinking_mode") && j["thinking_mode"].is_string()) {
            req.has_thinking_mode = true;
            req.thinking_mode = j["thinking_mode"].get<std::string>();
        } else if (j.contains("chat_template_kwargs") &&
                   j["chat_template_kwargs"].is_object() &&
                   j["chat_template_kwargs"].contains("thinking_mode") &&
                   j["chat_template_kwargs"]["thinking_mode"].is_string()) {
            req.has_thinking_mode = true;
            req.thinking_mode = j["chat_template_kwargs"]["thinking_mode"].get<std::string>();
        }
        // max_completion_tokens is the current OpenAI name; max_tokens the legacy one (vLLM takes both).
        if (j.contains("max_completion_tokens") && j["max_completion_tokens"].is_number_integer()) {
            req.has_max_tokens = true;
            req.max_tokens = j["max_completion_tokens"].get<int>();
        } else if (j.contains("max_tokens") && j["max_tokens"].is_number_integer()) {
            req.has_max_tokens = true;
            req.max_tokens = j["max_tokens"].get<int>();
        }
        if (j.contains("n") && j["n"].is_number_integer()) req.n = j["n"].get<int>();
        if (j.contains("presence_penalty") && j["presence_penalty"].is_number()) { req.has_presence_penalty = true; req.presence_penalty = j["presence_penalty"].get<float>(); }
        if (j.contains("frequency_penalty") && j["frequency_penalty"].is_number()) { req.has_frequency_penalty = true; req.frequency_penalty = j["frequency_penalty"].get<float>(); }
        if (j.contains("top_k") && j["top_k"].is_number()) { req.has_top_k = true; req.top_k = j["top_k"].get<int>(); }
        if (j.contains("min_p") && j["min_p"].is_number()) { req.has_min_p = true; req.min_p = j["min_p"].get<float>(); }
        if (j.contains("repetition_penalty") && j["repetition_penalty"].is_number()) {
            req.has_repetition_penalty = true;
            req.repetition_penalty = j["repetition_penalty"].get<float>();
        }
        if (j.contains("seed") && j["seed"].is_number_integer()) { req.has_seed = true; req.seed = j["seed"].get<int64_t>(); }
        if (j.contains("stream_options") && j["stream_options"].is_object()) {
            req.include_usage = j["stream_options"].value("include_usage", false);
        }
        
        if (j.contains("messages") && j["messages"].is_array()) {
            req.messages = j["messages"];
            for (const auto& message : j["messages"]) {
                if (!message.is_object()) continue;

                ParsedChatMessage parsed;
                parsed.raw = message;
                parsed.role = message.value("role", "");

                if (!message.contains("content")) {
                    req.parsed_messages.push_back(std::move(parsed));
                    continue;
                }

                const auto& content = message["content"];
                if (content.is_string()) {
                    ChatContentPart part;
                    part.type = "text";
                    part.text = content.get<std::string>();
                    part.raw = content;
                    parsed.content_parts.push_back(std::move(part));
                } else if (content.is_array()) {
                    for (const auto& item : content) {
                        if (!item.is_object()) continue;

                        ChatContentPart part;
                        part.raw = item;
                        part.type = item.value("type", "");

                        if (part.type == "text") {
                            part.text = item.value("text", "");
                        } else if (part.type == "image_url" && item.contains("image_url")) {
                            const auto& image_url = item["image_url"];
                            if (image_url.is_object()) {
                                part.image_url = image_url.value("url", "");
                                part.image_detail = image_url.value("detail", "");
                            } else if (image_url.is_string()) {
                                part.image_url = image_url.get<std::string>();
                            }
                        }

                        parsed.content_parts.push_back(std::move(part));
                    }
                }

                req.parsed_messages.push_back(std::move(parsed));
            }
        }
        
        if (j.contains("stop")) {
            if (j["stop"].is_string()) {
                req.stop.push_back(j["stop"].get<std::string>());
            } else if (j["stop"].is_array()) {
                for (const auto& s : j["stop"]) {
                    if (s.is_string()) req.stop.push_back(s.get<std::string>());
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

        // vLLM-style extension: allow `messages` (chat-style content, including multimodal parts) for embeddings.
        // The actual handling is delegated to the embedding callback via `req.raw`.
        // To keep the server-side validation logic happy, ensure `inputs` is non-empty when `messages` exists.
        if (req.inputs.empty() && j.contains("messages") && j["messages"].is_array()) {
            req.inputs.push_back("");
        }
        
        return req;
    }
};

struct ASRRequest {
    std::string model;
    std::vector<uint8_t> audio_data;
    std::string filename;  // 原始文件名
    std::string task = "transcription"; // transcription | translation
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
    std::string model;
    int n = 1;
    std::string quality = "standard";
    std::string response_format = "url";  // url, b64_json
    std::string size = "1024x1024";
    std::string style = "vivid";
    int seed = -1;
    int num_inference_steps = 4;
    float guidance_scale = 0.0f;
    std::string init_image;
    std::string mask;
    std::string request_base_url;
    bool is_edit = false;
    std::vector<uint8_t> image_data;
    std::string image_filename;
    std::vector<uint8_t> mask_data;
    std::string mask_filename;
    
    nlohmann::json raw;
    
    static ImageGenRequest from_json(const nlohmann::json& j) {
        ImageGenRequest req;
        req.raw = j;
        
        if (j.contains("prompt")) req.prompt = j["prompt"].get<std::string>();
        if (j.contains("model")) req.model = j["model"].get<std::string>();
        if (j.contains("n") && j["n"].is_number_integer()) req.n = j["n"].get<int>();
        if (j.contains("quality")) req.quality = j["quality"].get<std::string>();
        if (j.contains("response_format")) req.response_format = j["response_format"].get<std::string>();
        if (j.contains("size")) req.size = j["size"].get<std::string>();
        if (j.contains("style")) req.style = j["style"].get<std::string>();
        if (j.contains("seed")) req.seed = j["seed"].get<int>();
        if (j.contains("num_inference_steps")) req.num_inference_steps = j["num_inference_steps"].get<int>();
        if (j.contains("guidance_scale")) req.guidance_scale = j["guidance_scale"].get<float>();
        if (j.contains("init_image")) req.init_image = j["init_image"].get<std::string>();
        else if (j.contains("image") && j["image"].is_string()) req.init_image = j["image"].get<std::string>();
        else if (j.contains("input_image")) req.init_image = j["input_image"].get<std::string>();
        if (j.contains("mask") && j["mask"].is_string()) req.mask = j["mask"].get<std::string>();
        
        return req;
    }

    bool has_image_input() const {
        return !image_data.empty();
    }
};

} // namespace openai_api
