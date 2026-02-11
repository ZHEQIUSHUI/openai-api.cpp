#pragma once

#include "internal_protocol.hpp"
#include "../core/data_provider.hpp"
#include "../router.hpp"

#include "utils/httplib.h"
#include <thread>
#include <atomic>
#include <functional>
#include <map>
#include <queue>
#include <mutex>
#include <condition_variable>

namespace openai_api {
namespace cluster {

/**
 * WorkerClient - Worker 进程连接 Master
 * 
 * 职责：
 * 1. 连接 Master
 * 2. 注册本地模型
 * 3. 接收并处理转发请求
 * 4. 发送心跳
 */
class WorkerClient {
public:
    using RequestHandler = std::function<void(ModelType type,
                                               const nlohmann::json& request,
                                               std::shared_ptr<BaseDataProvider> provider)>;
    
    WorkerClient();
    ~WorkerClient();
    
    // 设置 Worker 监听地址（用于接收 Master 转发的请求）
    // 如果不设置，会自动检测本地 IP
    void set_listen_address(const std::string& host, int port = 0);
    
    // 连接到 Master
    bool connect(const std::string& host, int port);
    void disconnect();
    
    // 检查是否已连接
    bool is_connected() const { return connected_.load(); }
    
    // 注册模型（向 Master 注册）
    bool register_model(ModelType type, const std::string& model_name);
    
    // 设置请求处理器（由应用层设置，用于处理 Master 转发的请求）
    void set_request_handler(RequestHandler handler) { request_handler_ = handler; }
    
    // 设置 Router（用于本地模型路由）
    void set_router(ModelRouter* router) { router_ = router; }
    
    // 发送响应给 Master
    bool send_response(const std::string& request_id, 
                       const nlohmann::json& response,
                       bool is_error = false);
    
    // 获取 Worker ID
    std::string get_worker_id() const { return worker_id_; }
    
    // 获取 Worker 监听地址（用于 Master 连接）
    std::string get_listen_address() const;
    int get_listen_port() const { return actual_listen_port_.load(); }
    
private:
    // 内部处理循环
    void process_loop();
    
    // 心跳线程
    void heartbeat_loop();
    
    // 处理转发的请求
    void handle_forward_request(const nlohmann::json& data);
    
    // 生成 Worker ID
    std::string generate_worker_id();
    
    std::string worker_id_;
    std::string master_host_;
    int master_port_ = 0;
    
    // Worker 监听地址（用于接收 Master 转发的请求）
    std::string listen_host_ = "0.0.0.0";  // 默认监听所有接口
    int listen_port_ = 0;  // 0 表示自动分配
    std::atomic<int> actual_listen_port_{0};  // 实际监听的端口
    
    std::shared_ptr<httplib::Client> client_;
    std::atomic<bool> connected_{false};
    std::atomic<bool> should_stop_{false};
    
    std::thread process_thread_;
    std::thread heartbeat_thread_;
    
    RequestHandler request_handler_;
    ModelRouter* router_ = nullptr;
    
    // 本地模型集合（已注册到 Master 的）
    std::set<std::string> registered_models_;
    std::mutex models_mutex_;
    
    // 活跃请求映射
    std::map<std::string, std::shared_ptr<BaseDataProvider>> active_requests_;
    std::mutex requests_mutex_;
};

/**
 * 检查端口是否被占用，如果是本项目服务则返回 true
 */
bool check_is_cluster_server(const std::string& host, int port);

} // namespace cluster
} // namespace openai_api
