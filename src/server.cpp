#include "openai_api/server.hpp"
#include "openai_api/encoder/encoder.hpp"

#include <iostream>
#include <sstream>
#include <chrono>

namespace openai_api {

// ============ 构造函数/析构函数 ============

Server::Server() {
    setupRoutes();
}

Server::~Server() {
    if (running_.load()) {
        stop();
    }
}

// ============ 配置 ============

void Server::setMaxConcurrency(int max) {
    options_.max_concurrency = max;
}

void Server::setTimeout(std::chrono::milliseconds timeout) {
    options_.default_timeout = timeout;
}

void Server::setApiKey(const std::string& api_key) {
    options_.api_key = api_key;
}

void Server::setOwner(const std::string& owner) {
    options_.owner = owner;
}

// ============ 模型注册 ============

void Server::registerChat(const std::string& model_name, ChatCallback callback) {
    router_.registerChat(model_name, std::move(callback));
}

void Server::registerEmbedding(const std::string& model_name, EmbeddingCallback callback) {
    router_.registerEmbedding(model_name, std::move(callback));
}

void Server::registerASR(const std::string& model_name, ASRCallback callback) {
    router_.registerASR(model_name, std::move(callback));
}

void Server::registerTTS(const std::string& model_name, TTSCallback callback) {
    router_.registerTTS(model_name, std::move(callback));
}

void Server::registerImageGeneration(const std::string& model_name, ImageGenCallback callback) {
    router_.registerImageGeneration(model_name, std::move(callback));
}

// ============ 模型管理 ============

std::vector<std::string> Server::listModels() const {
    return router_.listAllModels();
}

bool Server::hasModel(const std::string& model_name) const {
    return router_.hasChatModel(model_name) ||
           router_.hasEmbeddingModel(model_name) ||
           router_.hasASRModel(model_name) ||
           router_.hasTTSModel(model_name) ||
           router_.hasImageGenModel(model_name);
}

void Server::unregisterModel(const std::string& model_name) {
    router_.unregisterChat(model_name);
    router_.unregisterEmbedding(model_name);
    router_.unregisterASR(model_name);
    router_.unregisterTTS(model_name);
    router_.unregisterImageGeneration(model_name);
}

// ============ 运行控制 ============

void Server::run(int port) {
    options_.port = port;
    run(options_);
}

void Server::run(const ServerOptions& options) {
    options_ = options;
    running_ = true;
    std::cout << "OpenAI API Server starting on http://" << options_.host << ":" << options_.port << std::endl;
    std::cout << "Max concurrency: " << options_.max_concurrency << std::endl;
    std::cout << "Models: " << [this]() {
        auto models = listModels();
        if (models.empty()) return std::string("none");
        std::string s;
        for (const auto& m : models) {
            if (!s.empty()) s += ", ";
            s += m;
        }
        return s;
    }() << std::endl;
    
    http_server_.listen(options_.host.c_str(), options_.port);
}

std::thread Server::runAsync(int port) {
    return std::thread([this, port]() { run(port); });
}

std::thread Server::runAsync(const ServerOptions& options) {
    return std::thread([this, options]() { run(options); });
}

void Server::stop() {
    running_ = false;
    http_server_.stop();
}

bool Server::isRunning() const {
    return running_.load();
}

// ============ 并发控制 ============

bool Server::acquireSlot() {
    std::unique_lock<std::mutex> lock(slot_mutex_);
    bool acquired = slot_cv_.wait_for(lock, options_.wait_timeout, [this] {
        return current_concurrency_.load() < options_.max_concurrency;
    });
    if (acquired) {
        current_concurrency_++;
    }
    return acquired;
}

void Server::releaseSlot() {
    {
        std::lock_guard<std::mutex> lock(slot_mutex_);
        current_concurrency_--;
    }
    slot_cv_.notify_one();
}

// ============ API Key 验证 ============

bool Server::verifyApiKey(const httplib::Request& req) const {
    if (options_.api_key.empty()) {
        return true;
    }
    
    auto auth_header = req.get_header_value("Authorization");
    if (auth_header.empty()) {
        return false;
    }
    
    const std::string bearer_prefix = "Bearer ";
    if (auth_header.substr(0, bearer_prefix.length()) == bearer_prefix) {
        std::string provided_key = auth_header.substr(bearer_prefix.length());
        return provided_key == options_.api_key;
    }
    
    return auth_header == options_.api_key;
}

// ============ 路由设置 ============

void Server::setupRoutes() {
    // Health
    http_server_.Get("/health", [this](const httplib::Request& req, httplib::Response& res) {
        handleHealth(req, res);
    });
    
    // Models (支持 /v1/models 和 /models)
    http_server_.Get("/v1/models", [this](const httplib::Request& req, httplib::Response& res) {
        handleModels(req, res);
    });
    http_server_.Get("/models", [this](const httplib::Request& req, httplib::Response& res) {
        handleModels(req, res);
    });
    
    // Chat Completions
    http_server_.Post("/v1/chat/completions", [this](const httplib::Request& req, httplib::Response& res) {
        handleChatCompletions(req, res);
    });
    http_server_.Post("/chat/completions", [this](const httplib::Request& req, httplib::Response& res) {
        handleChatCompletions(req, res);
    });
    
    // Embeddings
    http_server_.Post("/v1/embeddings", [this](const httplib::Request& req, httplib::Response& res) {
        handleEmbeddings(req, res);
    });
    http_server_.Post("/embeddings", [this](const httplib::Request& req, httplib::Response& res) {
        handleEmbeddings(req, res);
    });
    
    // ASR
    http_server_.Post("/v1/audio/transcriptions", [this](const httplib::Request& req, httplib::Response& res) {
        handleTranscriptions(req, res);
    });
    http_server_.Post("/audio/transcriptions", [this](const httplib::Request& req, httplib::Response& res) {
        handleTranscriptions(req, res);
    });
    
    http_server_.Post("/v1/audio/translations", [this](const httplib::Request& req, httplib::Response& res) {
        handleTranslations(req, res);
    });
    http_server_.Post("/audio/translations", [this](const httplib::Request& req, httplib::Response& res) {
        handleTranslations(req, res);
    });
    
    // TTS
    http_server_.Post("/v1/audio/speech", [this](const httplib::Request& req, httplib::Response& res) {
        handleSpeech(req, res);
    });
    http_server_.Post("/audio/speech", [this](const httplib::Request& req, httplib::Response& res) {
        handleSpeech(req, res);
    });
    
    // Image Generation
    http_server_.Post("/v1/images/generations", [this](const httplib::Request& req, httplib::Response& res) {
        handleImageGenerations(req, res);
    });
    http_server_.Post("/images/generations", [this](const httplib::Request& req, httplib::Response& res) {
        handleImageGenerations(req, res);
    });
    
    // CORS
    http_server_.Options("/.*", [](const httplib::Request& req, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
        res.set_header("Access-Control-Allow-Headers", "Content-Type, Authorization");
        res.status = 200;
    });
}

// ============ 端点处理 ============

void Server::handleHealth(const httplib::Request& req, httplib::Response& res) {
    nlohmann::json j;
    j["status"] = "healthy";
    j["concurrency"] = current_concurrency_.load();
    j["max_concurrency"] = options_.max_concurrency;
    res.set_content(j.dump(2), "application/json");
}

void Server::handleModels(const httplib::Request& req, httplib::Response& res) {
    nlohmann::json j;
    j["object"] = "list";
    j["data"] = nlohmann::json::array();
    
    auto models = listModels();
    auto now = std::time(nullptr);
    for (const auto& model_name : models) {
        nlohmann::json model_j;
        model_j["id"] = model_name;
        model_j["object"] = "model";
        model_j["created"] = now;
        model_j["owned_by"] = options_.owner;
        j["data"].push_back(model_j);
    }
    
    res.set_content(j.dump(2), "application/json");
}

void Server::handleChatCompletions(const httplib::Request& req, httplib::Response& res) {
    // API Key 验证
    if (!verifyApiKey(req)) {
        res.status = 401;
        res.set_content(ErrorEncoder::encode("unauthorized", "Invalid API key"), "application/json");
        return;
    }
    
    // 并发控制
    if (!acquireSlot()) {
        res.status = 503;
        res.set_content(ErrorEncoder::rate_limit(), "application/json");
        return;
    }
    struct SlotGuard { Server* s; ~SlotGuard() { s->releaseSlot(); } } guard{this};
    
    // 解析请求
    nlohmann::json req_json;
    try {
        req_json = nlohmann::json::parse(req.body);
    } catch (const std::exception& e) {
        res.status = 400;
        res.set_content(ErrorEncoder::invalid_request("Invalid JSON: " + std::string(e.what())), "application/json");
        return;
    }
    
    auto request = ChatRequest::from_json(req_json);
    if (request.model.empty()) {
        res.status = 400;
        res.set_content(ErrorEncoder::invalid_request("Missing 'model' field"), "application/json");
        return;
    }
    
    // 检查模型是否存在
    if (!router_.hasChatModel(request.model)) {
        res.status = 400;
        auto available = router_.listChatModels();
        std::string msg = "Model '" + request.model + "' is not available";
        if (!available.empty()) {
            msg += ". Available models: ";
            for (size_t i = 0; i < available.size(); ++i) {
                if (i > 0) msg += ", ";
                msg += available[i];
            }
        }
        res.set_content(ErrorEncoder::invalid_request(msg), "application/json");
        return;
    }
    
    // 创建 Provider 并路由
    auto provider = std::make_shared<QueueProvider>(options_.default_timeout);
    if (!router_.routeChat(request, provider)) {
        res.status = 500;
        res.set_content(ErrorEncoder::server_error("Failed to route request"), "application/json");
        return;
    }
    
    // 处理响应
    if (request.stream) {
        // 流式响应 - 使用 chunked 传输实现真正的流式输出
        res.set_header("Content-Type", "text/event-stream");
        res.set_header("Cache-Control", "no-cache");
        res.set_header("Connection", "keep-alive");
        
        // 使用 shared_ptr 管理状态，确保生命周期
        struct StreamState {
            std::shared_ptr<BaseDataProvider> provider;
            ChatCompletionsSSEEncoder encoder;
            std::chrono::steady_clock::time_point start;
            std::chrono::milliseconds timeout;
            std::atomic<bool> done{false};
        };
        auto state = std::make_shared<StreamState>();
        state->provider = provider;
        state->start = std::chrono::steady_clock::now();
        state->timeout = options_.default_timeout;
        
        // 使用 chunked content provider 支持流式传输
        res.set_chunked_content_provider("text/event-stream",
            [state](size_t offset, httplib::DataSink& sink) -> bool {
                if (state->done.load()) {
                    sink.done();  // 关闭连接
                    return false;
                }
                
                auto elapsed = std::chrono::steady_clock::now() - state->start;
                auto remaining = state->timeout - std::chrono::duration_cast<std::chrono::milliseconds>(elapsed);
                if (remaining <= std::chrono::milliseconds(0)) {
                    // 超时，发送 [DONE] 并关闭
                    const std::string done_marker = "data: [DONE]\n\n";
                    sink.write(done_marker.data(), done_marker.size());
                    sink.done();  // 关键：关闭连接
                    state->done = true;
                    return false;
                }
                
                // 检查 provider 是否已结束（减少等待时间）
                if (state->provider->is_ended()) {
                    const std::string done_marker = "data: [DONE]\n\n";
                    sink.write(done_marker.data(), done_marker.size());
                    sink.done();  // 关闭连接
                    state->done = true;
                    return false;
                }
                
                // 等待数据（短超时，快速响应结束）
                auto chunk = state->provider->wait_pop_for(std::chrono::milliseconds(10));
                if (!chunk.has_value()) {
                    return true;  // 没有数据，继续等待
                }
                
                if (chunk->is_end()) {
                    const std::string done_marker = "data: [DONE]\n\n";
                    sink.write(done_marker.data(), done_marker.size());
                    sink.done();  // 关键：关闭连接让 Python SDK 知道结束
                    state->done = true;
                    return false;
                }
                
                // 编码并写入数据
                std::string encoded = state->encoder.encode(chunk.value());
                if (!encoded.empty()) {
                    sink.write(encoded.data(), encoded.size());
                }
                
                return true;  // 继续等待
            }
        );
        
        return;
    } else {
        // 非流式响应
        auto chunk = provider->wait_pop_for(options_.default_timeout);
        if (!chunk.has_value()) {
            res.status = 504;
            res.set_content(ErrorEncoder::server_error("Request timeout"), "application/json");
            return;
        }
        if (chunk->is_error()) {
            res.status = 400;
            ChatCompletionsJSONEncoder encoder;
            res.set_content(encoder.encode(chunk.value()), "application/json");
            return;
        }
        
        ChatCompletionsJSONEncoder encoder;
        res.set_content(encoder.encode(chunk.value()), "application/json");
    }
}

void Server::handleEmbeddings(const httplib::Request& req, httplib::Response& res) {
    if (!verifyApiKey(req)) {
        res.status = 401;
        res.set_content(ErrorEncoder::encode("unauthorized", "Invalid API key"), "application/json");
        return;
    }
    
    if (!acquireSlot()) {
        res.status = 503;
        res.set_content(ErrorEncoder::rate_limit(), "application/json");
        return;
    }
    struct SlotGuard { Server* s; ~SlotGuard() { s->releaseSlot(); } } guard{this};
    
    nlohmann::json req_json;
    try {
        req_json = nlohmann::json::parse(req.body);
    } catch (const std::exception& e) {
        res.status = 400;
        res.set_content(ErrorEncoder::invalid_request("Invalid JSON"), "application/json");
        return;
    }
    
    auto request = EmbeddingRequest::from_json(req_json);
    if (request.model.empty()) {
        res.status = 400;
        res.set_content(ErrorEncoder::invalid_request("Missing 'model' field"), "application/json");
        return;
    }
    if (request.inputs.empty()) {
        res.status = 400;
        res.set_content(ErrorEncoder::invalid_request("Missing 'input' field"), "application/json");
        return;
    }
    
    if (!router_.hasEmbeddingModel(request.model)) {
        res.status = 400;
        auto available = router_.listEmbeddingModels();
        std::string msg = "Model '" + request.model + "' is not available";
        if (!available.empty()) {
            msg += ". Available models: ";
            for (size_t i = 0; i < available.size(); ++i) {
                if (i > 0) msg += ", ";
                msg += available[i];
            }
        }
        res.set_content(ErrorEncoder::invalid_request(msg), "application/json");
        return;
    }
    
    auto provider = std::make_shared<QueueProvider>(options_.default_timeout);
    if (!router_.routeEmbedding(request, provider)) {
        res.status = 500;
        res.set_content(ErrorEncoder::server_error("Failed to route request"), "application/json");
        return;
    }
    
    auto chunk = provider->wait_pop_for(options_.default_timeout);
    if (!chunk.has_value()) {
        res.status = 504;
        res.set_content(ErrorEncoder::server_error("Request timeout"), "application/json");
        return;
    }
    if (chunk->is_error()) {
        res.status = 400;
        EmbeddingsJSONEncoder encoder;
        res.set_content(encoder.encode(chunk.value()), "application/json");
        return;
    }
    
    EmbeddingsJSONEncoder encoder;
    res.set_content(encoder.encode(chunk.value()), "application/json");
}

void Server::handleTranscriptions(const httplib::Request& req, httplib::Response& res) {
    if (!verifyApiKey(req)) {
        res.status = 401;
        res.set_content(ErrorEncoder::encode("unauthorized", "Invalid API key"), "application/json");
        return;
    }
    
    if (!acquireSlot()) {
        res.status = 503;
        res.set_content(ErrorEncoder::rate_limit(), "application/json");
        return;
    }
    struct SlotGuard { Server* s; ~SlotGuard() { s->releaseSlot(); } } guard{this};
    
    // TODO: 解析 multipart form data 提取 audio 和 model
    // 简化实现：从 body 中提取 model 参数
    ASRRequest request;
    request.raw_body = req.body;
    
    // 尝试从 body 中提取 model
    size_t model_pos = req.body.find("name=\"model\"");
    if (model_pos != std::string::npos) {
        size_t val_start = req.body.find("\r\n\r\n", model_pos);
        if (val_start != std::string::npos) {
            val_start += 4;
            size_t val_end = req.body.find("\r\n", val_start);
            if (val_end != std::string::npos) {
                request.model = req.body.substr(val_start, val_end - val_start);
            }
        }
    }
    
    if (request.model.empty()) {
        res.status = 400;
        res.set_content(ErrorEncoder::invalid_request("Missing 'model' field"), "application/json");
        return;
    }
    
    if (!router_.hasASRModel(request.model)) {
        res.status = 400;
        auto available = router_.listASRModels();
        std::string msg = "Model '" + request.model + "' is not available";
        if (!available.empty()) {
            msg += ". Available models: ";
            for (size_t i = 0; i < available.size(); ++i) {
                if (i > 0) msg += ", ";
                msg += available[i];
            }
        }
        res.set_content(ErrorEncoder::invalid_request(msg), "application/json");
        return;
    }
    
    auto provider = std::make_shared<QueueProvider>(options_.default_timeout);
    if (!router_.routeASR(request, provider)) {
        res.status = 500;
        res.set_content(ErrorEncoder::server_error("Failed to route request"), "application/json");
        return;
    }
    
    auto chunk = provider->wait_pop_for(options_.default_timeout);
    if (!chunk.has_value()) {
        res.status = 504;
        res.set_content(ErrorEncoder::server_error("Request timeout"), "application/json");
        return;
    }
    if (chunk->is_error()) {
        res.status = 400;
        res.set_content(ErrorEncoder::encode(chunk->error_code, chunk->error_message), "application/json");
        return;
    }
    
    ASRJSONEncoder encoder;
    res.set_content(encoder.encode(chunk.value()), "application/json");
}

void Server::handleTranslations(const httplib::Request& req, httplib::Response& res) {
    // 翻译端点与转录类似
    handleTranscriptions(req, res);
}

void Server::handleSpeech(const httplib::Request& req, httplib::Response& res) {
    if (!verifyApiKey(req)) {
        res.status = 401;
        res.set_content(ErrorEncoder::encode("unauthorized", "Invalid API key"), "application/json");
        return;
    }
    
    if (!acquireSlot()) {
        res.status = 503;
        res.set_content(ErrorEncoder::rate_limit(), "application/json");
        return;
    }
    struct SlotGuard { Server* s; ~SlotGuard() { s->releaseSlot(); } } guard{this};
    
    nlohmann::json req_json;
    try {
        req_json = nlohmann::json::parse(req.body);
    } catch (const std::exception& e) {
        res.status = 400;
        res.set_content(ErrorEncoder::invalid_request("Invalid JSON"), "application/json");
        return;
    }
    
    auto request = TTSRequest::from_json(req_json);
    if (request.model.empty()) {
        res.status = 400;
        res.set_content(ErrorEncoder::invalid_request("Missing 'model' field"), "application/json");
        return;
    }
    if (request.input.empty()) {
        res.status = 400;
        res.set_content(ErrorEncoder::invalid_request("Missing 'input' field"), "application/json");
        return;
    }
    
    if (!router_.hasTTSModel(request.model)) {
        res.status = 400;
        auto available = router_.listTTSModels();
        std::string msg = "Model '" + request.model + "' is not available";
        if (!available.empty()) {
            msg += ". Available models: ";
            for (size_t i = 0; i < available.size(); ++i) {
                if (i > 0) msg += ", ";
                msg += available[i];
            }
        }
        res.set_content(ErrorEncoder::invalid_request(msg), "application/json");
        return;
    }
    
    auto provider = std::make_shared<QueueProvider>(options_.default_timeout);
    if (!router_.routeTTS(request, provider)) {
        res.status = 500;
        res.set_content(ErrorEncoder::server_error("Failed to route request"), "application/json");
        return;
    }
    
    auto chunk = provider->wait_pop_for(options_.default_timeout);
    if (!chunk.has_value()) {
        res.status = 504;
        res.set_content(ErrorEncoder::server_error("Request timeout"), "application/json");
        return;
    }
    if (chunk->is_error()) {
        res.status = 400;
        res.set_content(ErrorEncoder::encode(chunk->error_code, chunk->error_message), "application/json");
        return;
    }
    
    std::string mime = chunk->mime_type.empty() ? "audio/mpeg" : chunk->mime_type;
    std::string audio_data(chunk->bytes.begin(), chunk->bytes.end());
    res.set_content(audio_data, mime);
}

void Server::handleImageGenerations(const httplib::Request& req, httplib::Response& res) {
    if (!verifyApiKey(req)) {
        res.status = 401;
        res.set_content(ErrorEncoder::encode("unauthorized", "Invalid API key"), "application/json");
        return;
    }
    
    if (!acquireSlot()) {
        res.status = 503;
        res.set_content(ErrorEncoder::rate_limit(), "application/json");
        return;
    }
    struct SlotGuard { Server* s; ~SlotGuard() { s->releaseSlot(); } } guard{this};
    
    nlohmann::json req_json;
    try {
        req_json = nlohmann::json::parse(req.body);
    } catch (const std::exception& e) {
        res.status = 400;
        res.set_content(ErrorEncoder::invalid_request("Invalid JSON"), "application/json");
        return;
    }
    
    auto request = ImageGenRequest::from_json(req_json);
    if (request.prompt.empty()) {
        res.status = 400;
        res.set_content(ErrorEncoder::invalid_request("Missing 'prompt' field"), "application/json");
        return;
    }
    if (request.model.empty()) {
        request.model = "dall-e-2";
    }
    
    if (!router_.hasImageGenModel(request.model)) {
        res.status = 400;
        auto available = router_.listImageGenModels();
        std::string msg = "Model '" + request.model + "' is not available";
        if (!available.empty()) {
            msg += ". Available models: ";
            for (size_t i = 0; i < available.size(); ++i) {
                if (i > 0) msg += ", ";
                msg += available[i];
            }
        }
        res.set_content(ErrorEncoder::invalid_request(msg), "application/json");
        return;
    }
    
    auto provider = std::make_shared<QueueProvider>(options_.default_timeout);
    if (!router_.routeImageGeneration(request, provider)) {
        res.status = 500;
        res.set_content(ErrorEncoder::server_error("Failed to route request"), "application/json");
        return;
    }
    
    auto chunk = provider->wait_pop_for(options_.default_timeout);
    if (!chunk.has_value()) {
        res.status = 504;
        res.set_content(ErrorEncoder::server_error("Request timeout"), "application/json");
        return;
    }
    if (chunk->is_error()) {
        res.status = 400;
        res.set_content(ErrorEncoder::encode(chunk->error_code, chunk->error_message), "application/json");
        return;
    }
    
    ImagesJSONEncoder encoder;
    res.set_content(encoder.encode(chunk.value()), "application/json");
}

} // namespace openai_api
