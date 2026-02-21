#include "openai_api/cluster/worker_client.hpp"
#include "openai_api/cluster/internal_protocol.hpp"
#include <sstream>
#include <iomanip>
#include <random>
#include <cctype>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#include <iphlpapi.h>
#pragma comment(lib, "ws2_32.lib")
#pragma comment(lib, "iphlpapi.lib")
#else
#include <ifaddrs.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#endif

namespace openai_api {
namespace cluster {

namespace {

std::string base64_encode(const std::vector<uint8_t>& data) {
    static const char* kChars =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string encoded;
    encoded.reserve(((data.size() + 2) / 3) * 4);

    size_t i = 0;
    while (i + 2 < data.size()) {
        const uint32_t n = (static_cast<uint32_t>(data[i]) << 16) |
                           (static_cast<uint32_t>(data[i + 1]) << 8) |
                           static_cast<uint32_t>(data[i + 2]);
        encoded.push_back(kChars[(n >> 18) & 0x3F]);
        encoded.push_back(kChars[(n >> 12) & 0x3F]);
        encoded.push_back(kChars[(n >> 6) & 0x3F]);
        encoded.push_back(kChars[n & 0x3F]);
        i += 3;
    }

    const size_t rem = data.size() - i;
    if (rem == 1) {
        const uint32_t n = static_cast<uint32_t>(data[i]) << 16;
        encoded.push_back(kChars[(n >> 18) & 0x3F]);
        encoded.push_back(kChars[(n >> 12) & 0x3F]);
        encoded.push_back('=');
        encoded.push_back('=');
    } else if (rem == 2) {
        const uint32_t n = (static_cast<uint32_t>(data[i]) << 16) |
                           (static_cast<uint32_t>(data[i + 1]) << 8);
        encoded.push_back(kChars[(n >> 18) & 0x3F]);
        encoded.push_back(kChars[(n >> 12) & 0x3F]);
        encoded.push_back(kChars[(n >> 6) & 0x3F]);
        encoded.push_back('=');
    }
    return encoded;
}

}  // namespace

// 获取本机非回环 IP 地址
static std::string get_local_ip() {
    std::string result = "127.0.0.1";  // 默认回退
    
#ifdef _WIN32
    // Windows 实现
    ULONG flags = GAA_FLAG_INCLUDE_PREFIX;
    ULONG family = AF_INET;  // IPv4
    ULONG bufferSize = 15000;
    
    std::vector<BYTE> buffer(bufferSize);
    PIP_ADAPTER_ADDRESSES adapterAddresses = reinterpret_cast<PIP_ADAPTER_ADDRESSES>(buffer.data());
    
    DWORD ret = GetAdaptersAddresses(family, flags, nullptr, adapterAddresses, &bufferSize);
    if (ret != ERROR_SUCCESS) {
        return result;
    }
    
    for (PIP_ADAPTER_ADDRESSES adapter = adapterAddresses; adapter != nullptr; adapter = adapter->Next) {
        // 跳过回环和未启用的适配器
        if (adapter->IfType == IF_TYPE_SOFTWARE_LOOPBACK || adapter->OperStatus != IfOperStatusUp) {
            continue;
        }
        
        for (PIP_ADAPTER_UNICAST_ADDRESS unicast = adapter->FirstUnicastAddress; 
             unicast != nullptr; unicast = unicast->Next) {
            sockaddr* addr = unicast->Address.lpSockaddr;
            if (addr->sa_family == AF_INET) {
                char ipStr[INET_ADDRSTRLEN];
                sockaddr_in* sin = reinterpret_cast<sockaddr_in*>(addr);
                inet_ntop(AF_INET, &(sin->sin_addr), ipStr, INET_ADDRSTRLEN);
                
                std::string ip(ipStr);
                if (ip != "127.0.0.1") {
                    return ip;
                }
            }
        }
    }
#else
    // Linux/macOS 实现
    struct ifaddrs *ifaddr, *ifa;
    char addrstr[INET_ADDRSTRLEN];
    
    if (getifaddrs(&ifaddr) == -1) {
        return result;
    }
    
    for (ifa = ifaddr; ifa != nullptr; ifa = ifa->ifa_next) {
        if (ifa->ifa_addr == nullptr) continue;
        if (ifa->ifa_addr->sa_family != AF_INET) continue;  // 只处理 IPv4
        
        void* addr = &((struct sockaddr_in*)ifa->ifa_addr)->sin_addr;
        inet_ntop(AF_INET, addr, addrstr, INET_ADDRSTRLEN);
        
        std::string ip(addrstr);
        // 跳过回环地址
        if (ip != "127.0.0.1") {
            result = ip;
            break;  // 返回第一个非回环地址
        }
    }
    
    freeifaddrs(ifaddr);
#endif
    
    return result;
}

// 检查是否为集群服务
bool check_is_cluster_server(const std::string& host, int port) {
    httplib::Client client(host, port);
    client.set_connection_timeout(2);
    
    // 构建握手消息
    auto payload = make_handshake("probe");
    auto msg = build_message(MessageType::HANDSHAKE, payload);
    
    httplib::Headers headers;
    headers.emplace("Content-Type", "application/octet-stream");
    
    std::string body(reinterpret_cast<const char*>(msg.data()), msg.size());
    auto res = client.Post("/internal/handshake", headers, body, "application/octet-stream");
    
    if (!res || res->status != 200) {
        return false;
    }
    
    // 检查响应是否为有效的握手确认
    if (res->body.size() < sizeof(MessageHeader)) {
        return false;
    }
    
    auto header = deserialize_header(reinterpret_cast<const uint8_t*>(res->body.data()));
    return header.is_valid() && static_cast<MessageType>(header.type) == MessageType::HANDSHAKE_ACK;
}

WorkerClient::WorkerClient() {
    worker_id_ = generate_worker_id();
}

WorkerClient::~WorkerClient() {
    disconnect();
}

void WorkerClient::set_listen_address(const std::string& host, int port) {
    listen_host_ = host;
    listen_port_ = port;
}

std::string WorkerClient::get_listen_address() const {
    if (listen_host_ == "0.0.0.0") {
        // 如果监听所有接口，返回实际的本机 IP
        return get_local_ip();
    }
    return listen_host_;
}

std::string WorkerClient::generate_worker_id() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 15);
    
    std::stringstream ss;
    ss << "worker_";
    for (int i = 0; i < 8; ++i) {
        ss << std::hex << dis(gen);
    }
    return ss.str();
}

bool WorkerClient::connect(const std::string& host, int port) {
    if (connected_.load()) {
        return true;
    }
    
    master_host_ = host;
    master_port_ = port;
    
    client_ = std::make_shared<httplib::Client>(host, port);
    client_->set_connection_timeout(5);
    client_->set_read_timeout(60);
    
    // 发送握手，携带 Worker 的监听地址信息
    auto payload = make_handshake(worker_id_);
    payload["worker_host"] = get_listen_address();
    payload["worker_port"] = listen_port_;  // 0 表示自动分配
    auto msg = build_message(MessageType::HANDSHAKE, payload);
    
    httplib::Headers headers;
    headers.emplace("Content-Type", "application/octet-stream");
    
    std::string body(reinterpret_cast<const char*>(msg.data()), msg.size());
    auto res = client_->Post("/internal/handshake", headers, body, "application/octet-stream");
    
    if (!res || res->status != 200) {
        client_.reset();
        return false;
    }
    
    // 验证响应
    if (res->body.size() < sizeof(MessageHeader)) {
        client_.reset();
        return false;
    }
    
    auto header = deserialize_header(reinterpret_cast<const uint8_t*>(res->body.data()));
    if (!header.is_valid() || static_cast<MessageType>(header.type) != MessageType::HANDSHAKE_ACK) {
        client_.reset();
        return false;
    }
    
    connected_ = true;
    should_stop_ = false;
    actual_listen_port_ = 0;
    
    // 启动处理线程（启动本地 HTTP 服务接收 Master 转发）
    process_thread_ = std::thread([this]() {
        process_loop();
    });
    
    // 等待本地服务启动
    while (actual_listen_port_.load() == 0 && !should_stop_) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    if (actual_listen_port_.load() <= 0 || should_stop_) {
        should_stop_ = true;
        connected_ = false;
        if (process_thread_.joinable()) {
            process_thread_.join();
        }
        client_.reset();
        return false;
    }
    
    // 启动心跳线程
    heartbeat_thread_ = std::thread([this]() {
        heartbeat_loop();
    });
    
    return true;
}

void WorkerClient::disconnect() {
    should_stop_ = true;
    connected_ = false;
    
    // 发送断开消息
    if (client_) {
        auto msg = build_message(MessageType::DISCONNECT, {});
        httplib::Headers headers;
        headers.emplace("Content-Type", "application/octet-stream");
        std::string body(reinterpret_cast<const char*>(msg.data()), msg.size());
        client_->Post("/internal/disconnect", headers, body, "application/octet-stream");
    }
    
    if (process_thread_.joinable()) {
        process_thread_.join();
    }
    
    if (heartbeat_thread_.joinable()) {
        heartbeat_thread_.join();
    }

    actual_listen_port_ = 0;
    client_.reset();
}

bool WorkerClient::register_model(ModelType type, const std::string& model_name) {
    if (!connected_.load()) {
        return false;
    }
    
    {
        std::lock_guard<std::mutex> lock(models_mutex_);
        if (registered_models_.count(model_name)) {
            return true;  // 已注册
        }
    }
    
    auto payload = make_register_model(type, model_name);
    payload["worker_id"] = worker_id_;
    payload["worker_host"] = get_listen_address();
    payload["worker_port"] = actual_listen_port_.load();
    auto msg = build_message(MessageType::REGISTER_MODEL, payload);
    
    httplib::Headers headers;
    headers.emplace("Content-Type", "application/octet-stream");
    
    std::string body(reinterpret_cast<const char*>(msg.data()), msg.size());
    auto res = client_->Post("/internal/register", headers, body, "application/octet-stream");
    
    if (!res || res->status != 200) {
        return false;
    }
    
    // 解析响应
    if (res->body.size() < sizeof(MessageHeader)) {
        return false;
    }
    
    auto header = deserialize_header(reinterpret_cast<const uint8_t*>(res->body.data()));
    if (!header.is_valid() || static_cast<MessageType>(header.type) != MessageType::REGISTER_ACK) {
        return false;
    }
    
    try {
        nlohmann::json ack = nlohmann::json::parse(res->body.substr(sizeof(MessageHeader)));
        if (!ack.value("success", false)) {
            std::string error_msg = ack.value("message", "Registration failed");
            // 可以记录错误
            return false;
        }
    } catch (...) {
        return false;
    }
    
    {
        std::lock_guard<std::mutex> lock(models_mutex_);
        registered_models_.insert(model_name);
    }
    
    return true;
}

bool WorkerClient::send_response(const std::string& request_id, 
                                  const nlohmann::json& response,
                                  bool is_error) {
    if (!connected_.load()) {
        return false;
    }
    
    auto payload = make_forward_response(request_id, response, is_error);
    auto msg = build_message(MessageType::FORWARD_RESPONSE, payload);
    
    httplib::Headers headers;
    headers.emplace("Content-Type", "application/octet-stream");
    
    std::string body(reinterpret_cast<const char*>(msg.data()), msg.size());
    auto res = client_->Post("/internal/response", headers, body, "application/octet-stream");
    
    return res && res->status == 200;
}

void WorkerClient::process_loop() {
    // 启动一个本地 HTTP 服务器接收 Master 转发的请求
    httplib::Server server;
    
    server.Post("/internal/forward", [this](const httplib::Request& req, httplib::Response& res) {
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
        
        handle_forward_request(payload);
        res.status = 200;
        res.set_content("OK", "text/plain");
    });
    
    // 使用配置的端口或随机端口启动
    int port = listen_port_;
    if (port > 0) {
        if (!server.bind_to_port(listen_host_, port)) {
            actual_listen_port_ = -1;
            should_stop_ = true;
            return;  // 端口绑定失败
        }
        actual_listen_port_ = port;
    } else {
        // 尝试随机端口
        for (int p = 28080; p < 28180; ++p) {
            if (server.bind_to_port(listen_host_, p)) {
                port = p;
                actual_listen_port_ = port;
                break;
            }
        }
    }
    
    if (port == 0) {
        actual_listen_port_ = -1;
        should_stop_ = true;
        return;  // 绑定失败
    }
    
    // 在后台线程中运行服务器
    std::thread server_thread([&server]() {
        server.listen_after_bind();
    });
    
    // 等待停止信号
    while (!should_stop_) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    
    server.stop();
    if (server_thread.joinable()) {
        server_thread.join();
    }
}

void WorkerClient::handle_forward_request(const nlohmann::json& data) {
    std::string request_id = data.value("request_id", "");
    ModelType type = static_cast<ModelType>(data.value("model_type", 1));
    nlohmann::json request = data.value("request", nlohmann::json::object());
    
    // 创建 provider 接收处理结果
    auto provider = std::make_shared<QueueProvider>();
    
    {
        std::lock_guard<std::mutex> lock(requests_mutex_);
        active_requests_[request_id] = provider;
    }
    
    // 使用 Router 处理请求
    if (router_) {
        bool routed = false;
        switch (type) {
            case ModelType::CHAT: {
                ChatRequest chat_req = ChatRequest::from_json(request);
                routed = router_->routeChat(chat_req, provider);
                break;
            }
            case ModelType::EMBEDDING: {
                EmbeddingRequest emb_req = EmbeddingRequest::from_json(request);
                routed = router_->routeEmbedding(emb_req, provider);
                break;
            }
            case ModelType::TTS: {
                TTSRequest tts_req = TTSRequest::from_json(request);
                routed = router_->routeTTS(tts_req, provider);
                break;
            }
            case ModelType::ASR: {
                ASRRequest asr_req;
                asr_req.model = request.value("model", "");
                asr_req.language = request.value("language", "");
                asr_req.prompt = request.value("prompt", "");
                asr_req.response_format = request.value("response_format", "json");
                asr_req.temperature = request.value("temperature", 0.0f);
                routed = router_->routeASR(asr_req, provider);
                break;
            }
            case ModelType::IMAGE_GEN: {
                ImageGenRequest img_req = ImageGenRequest::from_json(request);
                routed = router_->routeImageGeneration(img_req, provider);
                break;
            }
            default:
                break;
        }
        if (!routed) {
            provider->push(OutputChunk::Error("model_not_found", "Model is not registered on worker"));
            provider->end();
        }
    } else if (request_handler_) {
        request_handler_(type, request, provider);
    } else {
        provider->push(OutputChunk::Error("worker_handler_missing", "No worker request handler configured"));
        provider->end();
    }
    
    // 收集响应并发送回 Master
    std::thread([this, request_id, provider]() {
        nlohmann::json response;
        std::vector<nlohmann::json> chunks;
        
        while (true) {
            auto chunk = provider->wait_pop_for(std::chrono::milliseconds(100));
            if (!chunk.has_value()) {
                if (provider->is_ended()) break;
                continue;
            }
            
            if (chunk->is_end()) break;
            if (chunk->is_error()) {
                send_response(request_id, make_error(chunk->error_code, chunk->error_message), true);
                return;
            }
            
            nlohmann::json chunk_json;
            if (!chunk->text.empty()) {
                chunk_json["text"] = chunk->text;
                // is_delta 通过 type 判断
                chunk_json["is_delta"] = (chunk->type == OutputChunkType::TextDelta);
                if (chunk->obj.contains("finish_reason")) {
                    chunk_json["finish_reason"] = chunk->obj["finish_reason"];
                }
            } else if (!chunk->embeds.empty()) {
                chunk_json["embeddings"] = chunk->embeds;
            } else if (!chunk->bytes.empty()) {
                chunk_json["bytes_b64"] = base64_encode(chunk->bytes);
                chunk_json["mime_type"] = chunk->mime_type;
            }
            chunks.push_back(chunk_json);
            
            if (chunk->obj.contains("finish_reason") && 
                chunk->obj["finish_reason"] == "stop") {
                break;
            }
        }
        
        if (chunks.size() == 1) {
            response = chunks[0];
        } else {
            response["chunks"] = chunks;
        }
        
        send_response(request_id, response, false);
        
        {
            std::lock_guard<std::mutex> lock(requests_mutex_);
            active_requests_.erase(request_id);
        }
    }).detach();
}

void WorkerClient::heartbeat_loop() {
    while (!should_stop_) {
        std::this_thread::sleep_for(std::chrono::seconds(5));
        
        if (!connected_.load() || should_stop_) break;
        
        auto payload = make_heartbeat();
        payload["worker_id"] = worker_id_;
        payload["worker_host"] = get_listen_address();
        payload["worker_port"] = actual_listen_port_.load();
        auto msg = build_message(MessageType::HEARTBEAT, payload);
        
        httplib::Headers headers;
        headers.emplace("Content-Type", "application/octet-stream");
        
        std::string body(reinterpret_cast<const char*>(msg.data()), msg.size());
        auto res = client_->Post("/internal/heartbeat", headers, body, "application/octet-stream");
        
        if (!res || res->status != 200) {
            // 心跳失败，可能已断开
            connected_ = false;
            break;
        }
    }
}

} // namespace cluster
} // namespace openai_api
