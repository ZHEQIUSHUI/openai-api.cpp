#include "openai_api/cluster/worker_manager.hpp"
#include "openai_api/types.hpp"
#include "openai_api/encoder/encoder.hpp"

#include <sstream>
#include <iomanip>

namespace openai_api {
namespace cluster {

WorkerManager::WorkerManager() = default;

WorkerManager::~WorkerManager() {
    stop();
}

bool WorkerManager::start(int port) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (running_) return true;
    
    http_server_ = std::make_unique<httplib::Server>();
    
    // 设置路由
    http_server_->Post("/internal/handshake", [this](const httplib::Request& req, httplib::Response& res) {
        handle_handshake(req, res);
    });
    
    http_server_->Post("/internal/register", [this](const httplib::Request& req, httplib::Response& res) {
        handle_register(req, res);
    });
    
    http_server_->Post("/internal/heartbeat", [this](const httplib::Request& req, httplib::Response& res) {
        handle_heartbeat(req, res);
    });
    
    http_server_->Post("/internal/forward", [this](const httplib::Request& req, httplib::Response& res) {
        handle_forward(req, res);
    });
    
    http_server_->Post("/internal/response", [this](const httplib::Request& req, httplib::Response& res) {
        handle_response(req, res);
    });
    
    // 启动服务器
    bool started = false;
    if (port == 0) {
        // 使用随机端口
        for (int p = 18080; p < 18180; ++p) {
            if (http_server_->bind_to_port("0.0.0.0", p)) {
                port_ = p;
                started = true;
                break;
            }
        }
    } else {
        if (http_server_->bind_to_port("0.0.0.0", port)) {
            port_ = port;
            started = true;
        }
    }
    
    if (!started) {
        http_server_.reset();
        return false;
    }
    
    running_ = true;
    server_thread_ = std::thread([this]() {
        http_server_->listen_after_bind();
    });
    
    // 启动心跳线程
    heartbeat_thread_ = std::thread([this]() {
        heartbeat_loop();
    });
    
    return true;
}

void WorkerManager::stop() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!running_) return;
        running_ = false;
    }
    
    if (http_server_) {
        http_server_->stop();
    }
    
    if (server_thread_.joinable()) {
        server_thread_.join();
    }
    
    if (heartbeat_thread_.joinable()) {
        heartbeat_thread_.join();
    }
}

bool WorkerManager::register_worker(const std::string& worker_id, 
                                     const std::string& worker_host,
                                     int worker_port,
                                     std::shared_ptr<httplib::Client> client) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto conn = std::make_shared<WorkerConnection>(worker_id, worker_host, worker_port, client);
    workers_[worker_id] = conn;
    
    return true;
}

void WorkerManager::unregister_worker(const std::string& worker_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = workers_.find(worker_id);
    if (it == workers_.end()) return;
    
    // 清理该 Worker 注册的所有模型
    for (const auto& model : it->second->registered_models) {
        model_to_worker_.erase(model);
        if (on_model_unregistered_) {
            on_model_unregistered_(model);
        }
    }
    
    workers_.erase(it);
}

bool WorkerManager::register_model(const std::string& worker_id, 
                                    ModelType type, 
                                    const std::string& model_name) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // 检查模型是否已存在
    if (model_to_worker_.count(model_name)) {
        return false;  // 模型名冲突
    }
    
    auto it = workers_.find(worker_id);
    if (it == workers_.end()) {
        return false;
    }
    
    // 记录模型归属
    model_to_worker_[model_name] = worker_id;
    it->second->registered_models.insert(model_name);
    
    // 通知回调
    if (on_model_registered_) {
        on_model_registered_(model_name, type);
    }
    
    return true;
}

bool WorkerManager::has_model(const std::string& model_name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return model_to_worker_.count(model_name) > 0;
}

std::string WorkerManager::get_worker_for_model(const std::string& model_name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = model_to_worker_.find(model_name);
    if (it == model_to_worker_.end()) return "";
    return it->second;
}

bool WorkerManager::forward_request(const std::string& model_name,
                                     ModelType type,
                                     const nlohmann::json& request_data,
                                     std::shared_ptr<BaseDataProvider> provider) {
    std::string worker_id;
    std::string worker_host;
    int worker_port;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = model_to_worker_.find(model_name);
        if (it == model_to_worker_.end()) return false;
        worker_id = it->second;
        
        auto wit = workers_.find(worker_id);
        if (wit == workers_.end()) {
            provider->push(OutputChunk::Error("worker_not_found", "Worker not found"));
            return false;
        }
        worker_host = wit->second->worker_host;
        worker_port = wit->second->worker_port;
    }
    
    auto request_id = generate_request_id();
    
    // 创建请求上下文
    auto context = std::make_shared<RemoteRequestContext>();
    context->request_id = request_id;
    context->provider = provider;
    context->start_time = std::chrono::steady_clock::now();
    
    {
        std::lock_guard<std::mutex> lock(mutex_);
        pending_requests_[request_id] = context;
    }
    
    // 发送转发请求给 Worker
    auto payload = make_forward_request(request_id, type, request_data);
    auto msg = build_message(MessageType::FORWARD_REQUEST, payload);
    
    // 异步发送（使用 Worker 的 host:port 创建新连接）
    std::thread([this, worker_host, worker_port, msg, request_id]() {
        httplib::Client client(worker_host, worker_port);
        client.set_connection_timeout(5);
        client.set_read_timeout(300);  // 较长的读取超时，因为推理可能需要时间
        
        httplib::Headers headers;
        headers.emplace("Content-Type", "application/octet-stream");
        
        std::string body(reinterpret_cast<const char*>(msg.data()), msg.size());
        auto res = client.Post("/internal/forward", headers, body, "application/octet-stream");
        
        if (!res || res->status != 200) {
            handle_worker_response(request_id, make_error("forward_failed", 
                "Failed to forward request to " + worker_host + ":" + std::to_string(worker_port)), true);
        }
    }).detach();
    
    return true;
}

void WorkerManager::handle_worker_response(const std::string& request_id,
                                            const nlohmann::json& response,
                                            bool is_error) {
    std::shared_ptr<RemoteRequestContext> context;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = pending_requests_.find(request_id);
        if (it == pending_requests_.end()) return;
        context = it->second;
        pending_requests_.erase(it);
    }
    
    if (!context->provider) return;
    
    if (is_error) {
        std::string code = response.value("error_code", "worker_error");
        std::string msg = response.value("error_message", "Unknown error");
        context->provider->push(OutputChunk::Error(code, msg));
        context->provider->end();
    } else {
        // 处理响应数据
        if (response.contains("chunks")) {
            // 流式响应
            for (const auto& chunk_data : response["chunks"]) {
                OutputChunk chunk;
                chunk.type = chunk_data.value("is_delta", true) ? 
                             OutputChunkType::TextDelta : OutputChunkType::FinalText;
                chunk.text = chunk_data.value("text", "");
                if (chunk_data.contains("finish_reason")) {
                    chunk.obj["finish_reason"] = chunk_data.value("finish_reason", "");
                }
                context->provider->push(chunk);
            }
        } else {
            // 非流式响应
            OutputChunk chunk;
            chunk.type = OutputChunkType::FinalText;
            chunk.text = response.value("text", "");
            context->provider->push(chunk);
        }
        context->provider->end();
    }
    
    context->completed = true;
}

std::vector<std::string> WorkerManager::list_models() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::string> models;
    for (const auto& [name, _] : model_to_worker_) {
        models.push_back(name);
    }
    return models;
}

void WorkerManager::heartbeat_loop() {
    while (running_) {
        std::this_thread::sleep_for(std::chrono::seconds(5));
        
        if (!running_) break;
        
        cleanup_dead_workers();
    }
}

void WorkerManager::cleanup_dead_workers() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto now = std::chrono::steady_clock::now();
    std::vector<std::string> dead_workers;
    
    for (const auto& [id, conn] : workers_) {
        auto elapsed = now - conn->last_heartbeat;
        if (elapsed > std::chrono::seconds(30)) {
            dead_workers.push_back(id);
        }
    }
    
    for (const auto& id : dead_workers) {
        auto it = workers_.find(id);
        if (it == workers_.end()) continue;
        
        // 清理模型
        for (const auto& model : it->second->registered_models) {
            model_to_worker_.erase(model);
        }
        
        workers_.erase(it);
    }
}

std::string WorkerManager::generate_request_id() {
    std::stringstream ss;
    ss << "req_" << std::hex << std::setfill('0') << std::setw(16) 
       << (std::chrono::steady_clock::now().time_since_epoch().count());
    return ss.str();
}

// HTTP 处理器
void WorkerManager::handle_handshake(const httplib::Request& req, httplib::Response& res) {
    // 检查魔数
    if (req.body.size() < sizeof(MessageHeader)) {
        res.status = 400;
        return;
    }
    
    auto header = deserialize_header(reinterpret_cast<const uint8_t*>(req.body.data()));
    if (!header.is_valid()) {
        res.status = 400;
        return;
    }
    
    // 解析 payload
    nlohmann::json payload;
    if (header.payload_length > 0) {
        try {
            payload = nlohmann::json::parse(req.body.substr(sizeof(MessageHeader)));
        } catch (...) {
            res.status = 400;
            return;
        }
    }
    
    std::string worker_id = payload.value("worker_id", "");
    std::string worker_host = payload.value("worker_host", "");
    int worker_port = payload.value("worker_port", 0);
    
    if (worker_id.empty()) {
        res.status = 400;
        return;
    }
    
    // 如果 Worker 没有报告地址，使用连接的远程地址
    if (worker_host.empty()) {
        worker_host = req.remote_addr;
    }
    
    // 注册 Worker（暂不创建 client，转发时动态创建）
    register_worker(worker_id, worker_host, worker_port, nullptr);
    
    // 发送确认，携带 Master 的信息
    auto ack = make_handshake_ack(true, "Welcome");
    ack["master_host"] = req.local_addr;
    ack["master_port"] = port_;
    auto msg = build_message(MessageType::HANDSHAKE_ACK, ack);
    
    res.set_content(std::string(reinterpret_cast<const char*>(msg.data()), msg.size()), 
                    "application/octet-stream");
}

void WorkerManager::handle_register(const httplib::Request& req, httplib::Response& res) {
    if (req.body.size() < sizeof(MessageHeader)) {
        res.status = 400;
        return;
    }
    
    auto header = deserialize_header(reinterpret_cast<const uint8_t*>(req.body.data()));
    if (!header.is_valid() || static_cast<MessageType>(header.type) != MessageType::REGISTER_MODEL) {
        res.status = 400;
        return;
    }
    
    nlohmann::json payload;
    try {
        payload = nlohmann::json::parse(req.body.substr(sizeof(MessageHeader)));
    } catch (...) {
        res.status = 400;
        return;
    }
    
    std::string worker_id = payload.value("worker_id", "");
    std::string model_name = payload.value("model_name", "");
    std::string worker_host = payload.value("worker_host", "");
    int worker_port = payload.value("worker_port", 0);
    ModelType type = static_cast<ModelType>(payload.value("model_type", 1));
    
    if (worker_id.empty() || model_name.empty()) {
        res.status = 400;
        return;
    }
    
    // 更新 Worker 的地址信息（如果 Worker 报告了新的端口）
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = workers_.find(worker_id);
        if (it != workers_.end()) {
            if (!worker_host.empty()) {
                it->second->worker_host = worker_host;
            }
            if (worker_port > 0) {
                it->second->worker_port = worker_port;
            }
        }
    }
    
    // 检查模型名冲突
    if (has_model(model_name)) {
        auto ack = make_register_ack(false, "Model name already exists: " + model_name);
        auto msg = build_message(MessageType::REGISTER_ACK, ack);
        res.set_content(std::string(reinterpret_cast<const char*>(msg.data()), msg.size()), 
                        "application/octet-stream");
        return;
    }
    
    // 注册模型
    bool success = register_model(worker_id, type, model_name);
    
    auto ack = make_register_ack(success, success ? "" : "Registration failed");
    auto msg = build_message(MessageType::REGISTER_ACK, ack);
    res.set_content(std::string(reinterpret_cast<const char*>(msg.data()), msg.size()), 
                    "application/octet-stream");
}

void WorkerManager::handle_heartbeat(const httplib::Request& req, httplib::Response& res) {
    if (req.body.size() < sizeof(MessageHeader)) {
        res.status = 400;
        return;
    }
    
    auto header = deserialize_header(reinterpret_cast<const uint8_t*>(req.body.data()));
    if (!header.is_valid()) {
        res.status = 400;
        return;
    }
    
    nlohmann::json payload;
    try {
        payload = nlohmann::json::parse(req.body.substr(sizeof(MessageHeader)));
    } catch (...) {
        res.status = 400;
        return;
    }
    
    std::string worker_id = payload.value("worker_id", "");
    std::string worker_host = payload.value("worker_host", "");
    int worker_port = payload.value("worker_port", 0);
    
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = workers_.find(worker_id);
    if (it != workers_.end()) {
        it->second->last_heartbeat = std::chrono::steady_clock::now();
        // 更新 Worker 地址信息
        if (!worker_host.empty()) {
            it->second->worker_host = worker_host;
        }
        if (worker_port > 0) {
            it->second->worker_port = worker_port;
        }
    }
    
    auto ack = make_heartbeat_ack();
    auto msg = build_message(MessageType::HEARTBEAT_ACK, ack);
    res.set_content(std::string(reinterpret_cast<const char*>(msg.data()), msg.size()), 
                    "application/octet-stream");
}

void WorkerManager::handle_forward(const httplib::Request& req, httplib::Response& res) {
    // Worker 收到转发的请求并处理，然后返回响应
    // 这里只是接收确认，实际处理在 Worker 端
    res.status = 200;
    res.set_content("OK", "text/plain");
}

void WorkerManager::handle_response(const httplib::Request& req, httplib::Response& res) {
    if (req.body.size() < sizeof(MessageHeader)) {
        res.status = 400;
        return;
    }
    
    auto header = deserialize_header(reinterpret_cast<const uint8_t*>(req.body.data()));
    if (!header.is_valid()) {
        res.status = 400;
        return;
    }
    
    nlohmann::json payload;
    try {
        payload = nlohmann::json::parse(req.body.substr(sizeof(MessageHeader)));
    } catch (...) {
        res.status = 400;
        return;
    }
    
    std::string request_id = payload.value("request_id", "");
    bool is_error = payload.value("is_error", false);
    nlohmann::json response = payload.value("response", nlohmann::json::object());
    
    handle_worker_response(request_id, response, is_error);
    
    res.status = 200;
    res.set_content("OK", "text/plain");
}

} // namespace cluster
} // namespace openai_api
