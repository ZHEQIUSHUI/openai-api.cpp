#pragma once

#include "openai_api/core/api_export.hpp"
#include "internal_protocol.hpp"
#include "../core/data_provider.hpp"
#include "../types.hpp"

#include "utils/httplib.h"
#include <thread>
#include <mutex>
#include <map>
#include <set>
#include <atomic>
#include <condition_variable>
#include <queue>
#include <functional>
#include <random>

namespace openai_api {
namespace cluster {

// 前置声明
class ClusterServer;

/**
 * Worker 连接信息
 */
struct WorkerConnection {
    std::string worker_id;
    std::string worker_host;  // Worker 监听地址
    int worker_port = 0;      // Worker 监听端口
    std::shared_ptr<httplib::Client> client;
    std::chrono::steady_clock::time_point last_heartbeat;
    std::atomic<bool> alive{true};
    std::set<std::string> registered_models;  // 该 Worker 注册的模型
    
    WorkerConnection(const std::string& id, const std::string& host, int port,
                     std::shared_ptr<httplib::Client> c)
        : worker_id(id), worker_host(host), worker_port(port), client(c), 
          last_heartbeat(std::chrono::steady_clock::now()) {}
          
    // 获取或创建指向 Worker 的 HTTP 客户端
    std::shared_ptr<httplib::Client> get_client() {
        if (!client) {
            client = std::make_shared<httplib::Client>(worker_host, worker_port);
            client->set_connection_timeout(5);
            client->set_read_timeout(60);
        }
        return client;
    }
};

/**
 * 远程请求上下文
 */
struct RemoteRequestContext {
    std::string request_id;
    std::shared_ptr<BaseDataProvider> provider;
    std::chrono::steady_clock::time_point start_time;
    std::atomic<bool> completed{false};
};

/**
 * WorkerManager - Master 端管理 Worker 连接
 * 
 * 职责：
 * 1. 接受 Worker 连接
 * 2. 管理模型注册
 * 3. 转发请求到 Worker
 * 4. 维护心跳
 */
class OPENAI_API_API WorkerManager {
public:
    // 模型注册回调
    using ModelRegisteredCallback = std::function<void(const std::string& model_name, ModelType type)>;
    using ModelUnregisteredCallback = std::function<void(const std::string& model_name)>;
    using ForwardHandler = std::function<void(const std::string& worker_id, 
                                               const nlohmann::json& request,
                                               std::shared_ptr<BaseDataProvider> provider)>;
    
    WorkerManager();
    ~WorkerManager();
    
    // 启动 Worker 监听服务
    bool start(int port = 0);  // port=0 表示使用随机端口
    void stop();
    
    // 获取监听端口
    int get_port() const { return port_; }
    
    // 注册 Worker（由 Server 调用处理握手后）
    bool register_worker(const std::string& worker_id, 
                         const std::string& worker_host,
                         int worker_port,
                         std::shared_ptr<httplib::Client> client);
    
    // 注销 Worker
    void unregister_worker(const std::string& worker_id);
    
    // 注册模型到 Worker
    bool register_model(const std::string& worker_id, 
                        ModelType type, 
                        const std::string& model_name);
    
    // 检查模型是否存在
    bool has_model(const std::string& model_name) const;
    
    // 获取模型所在的 Worker
    std::string get_worker_for_model(const std::string& model_name) const;
    
    // 转发请求到 Worker
    bool forward_request(const std::string& model_name,
                         ModelType type,
                         const nlohmann::json& request_data,
                         std::shared_ptr<BaseDataProvider> provider);
    
    // 处理 Worker 响应
    void handle_worker_response(const std::string& request_id,
                                const nlohmann::json& response,
                                bool is_error);
    
    // 获取所有注册的模型
    std::vector<std::string> list_models() const;
    
    // HTTP 处理器
    void handle_handshake(const httplib::Request& req, httplib::Response& res);
    void handle_register(const httplib::Request& req, httplib::Response& res);
    void handle_heartbeat(const httplib::Request& req, httplib::Response& res);
    void handle_forward(const httplib::Request& req, httplib::Response& res);
    void handle_response(const httplib::Request& req, httplib::Response& res);
    
    // 设置回调
    void set_model_registered_callback(ModelRegisteredCallback cb) { on_model_registered_ = cb; }
    void set_model_unregistered_callback(ModelUnregisteredCallback cb) { on_model_unregistered_ = cb; }
    
private:
    // 心跳检查线程
    void heartbeat_loop();
    
    // 清理超时 Worker
    void cleanup_dead_workers();
    
    // 生成请求 ID
    std::string generate_request_id();
    
    mutable std::mutex mutex_;
    std::map<std::string, std::shared_ptr<WorkerConnection>> workers_;  // worker_id -> connection
    std::map<std::string, std::string> model_to_worker_;  // model_name -> worker_id
    std::map<std::string, std::shared_ptr<RemoteRequestContext>> pending_requests_;  // request_id -> context
    
    std::unique_ptr<httplib::Server> http_server_;
    std::thread server_thread_;
    std::thread heartbeat_thread_;
    std::atomic<bool> running_{false};
    int port_ = 0;
    
    std::random_device rd_;
    mutable std::mt19937 rng_{rd_()};
    
    ModelRegisteredCallback on_model_registered_;
    ModelUnregisteredCallback on_model_unregistered_;
};

} // namespace cluster
} // namespace openai_api
