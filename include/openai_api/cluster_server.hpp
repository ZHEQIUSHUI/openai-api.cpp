#pragma once

/**
 * OpenAI API Cluster Server
 * 
 * 支持主从分布式架构的服务器类。
 * 
 * 特性：
 * 1. 端口检测：启动时检测端口是否被占用
 * 2. 服务识别：判断占用端口的是否为本项目服务
 * 3. 远程注册：Worker 进程可以连接到 Master 并注册模型
 * 4. 请求转发：Master 将请求转发给对应的 Worker 处理
 * 
 * 使用示例：
 * @code
 * // Master 进程（第一个启动）
 * openai_api::ClusterServer master(8080);
 * master.run();
 * 
 * // Worker 进程（后续启动，相同端口）
 * openai_api::ClusterServer worker(8080);
 * worker.registerChat("qwen-7b", [](const auto& req, auto provider) {
 *     // 模型推理逻辑
 * });
 * worker.run();  // 会自动连接到 Master 并注册模型
 * @endcode
 */

#include "server.hpp"
#include "cluster/worker_manager.hpp"
#include "cluster/worker_client.hpp"
#include "cluster/remote_worker_provider.hpp"

namespace openai_api {

/**
 * 集群服务器运行模式
 */
enum class ClusterMode {
    STANDALONE,     // 独立模式（普通单进程）
    MASTER,         // 主节点模式
    WORKER          // 工作节点模式
};

/**
 * 集群服务器配置
 */
struct ClusterServerOptions {
    ServerOptions server;
    
    // 集群相关配置
    bool enable_cluster = true;              // 启用集群功能
    std::string worker_id;                    // Worker ID（留空自动生成）
    std::chrono::milliseconds worker_timeout{30000};  // Worker 超时时间
    std::chrono::milliseconds heartbeat_interval{5000}; // 心跳间隔
};

/**
 * ClusterServer - 支持主从分布式架构的服务器
 * 
 * 自动检测端口状态：
 * - 端口空闲：以 Master 模式启动
 * - 端口被占用且为本项目服务：以 Worker 模式连接
 * - 端口被占用但不是本项目服务：报错
 */
class ClusterServer {
public:
    ClusterServer();
    explicit ClusterServer(int port);
    explicit ClusterServer(const ClusterServerOptions& options);
    
    ~ClusterServer();
    
    // 禁止拷贝
    ClusterServer(const ClusterServer&) = delete;
    ClusterServer& operator=(const ClusterServer&) = delete;
    
    // ============ 配置 ============
    
    void setMaxConcurrency(int max);
    void setTimeout(std::chrono::milliseconds timeout);
    void setApiKey(const std::string& api_key);
    
    // 设置 Worker 监听地址（用于跨机器部署，Worker 模式下有效）
    // 默认自动检测本机 IP，监听所有接口
    void setWorkerListenAddress(const std::string& host, int port = 0);
    
    // ============ 模型注册（本地模型） ============
    
    void registerChat(const std::string& model_name, ChatCallback callback);
    void registerEmbedding(const std::string& model_name, EmbeddingCallback callback);
    void registerASR(const std::string& model_name, ASRCallback callback);
    void registerTTS(const std::string& model_name, TTSCallback callback);
    void registerImageGeneration(const std::string& model_name, ImageGenCallback callback);
    
    // ============ 模型管理 ============
    
    std::vector<std::string> listModels() const;
    bool hasModel(const std::string& model_name) const;
    void unregisterModel(const std::string& model_name);
    
    // ============ 运行控制 ============
    
    /**
     * 启动服务器（自动检测模式）
     * @param port 端口号
     * @return 实际运行的模式
     */
    ClusterMode run(int port = 8080);
    
    /**
     * 使用配置启动服务器
     */
    ClusterMode run(const ClusterServerOptions& options);
    
    /**
     * 强制以 Master 模式启动（即使端口被占用也会尝试）
     */
    bool runAsMaster(int port = 8080);
    
    /**
     * 强制以 Worker 模式启动（连接到指定 Master）
     */
    bool runAsWorker(const std::string& master_host, int master_port);
    
    /**
     * 停止服务器
     */
    void stop();
    
    /**
     * 检查是否正在运行
     */
    bool isRunning() const;
    
    /**
     * 获取当前运行模式
     */
    ClusterMode getMode() const { return mode_; }
    
    /**
     * 获取底层 Server 实例（Master 模式下有效）
     */
    Server* getServer() { return server_.get(); }
    
    /**
     * 获取 WorkerClient 实例（Worker 模式下有效）
     */
    cluster::WorkerClient* getWorkerClient() { return worker_client_.get(); }

private:
    // 尝试作为 Master 启动
    bool tryStartMaster(int port);
    
    // 尝试作为 Worker 连接
    bool tryStartWorker(const std::string& host, int port);
    
    // 检测端口是否为集群服务
    bool detectClusterService(const std::string& host, int port);
    
    // 注册本地模型到 Master（Worker 模式）
    void registerLocalModelsToMaster();
    
    // 设置 Worker 请求处理器
    void setupWorkerHandler();

private:
    ClusterServerOptions options_;
    ClusterMode mode_ = ClusterMode::STANDALONE;
    
    // Worker 监听地址设置
    std::string worker_listen_host_ = "0.0.0.0";
    int worker_listen_port_ = 0;
    
    // Master 模式组件
    std::unique_ptr<Server> server_;
    std::unique_ptr<cluster::WorkerManager> worker_manager_;
    
    // Worker 模式组件
    std::unique_ptr<cluster::WorkerClient> worker_client_;
    
    // 本地模型注册（Worker 模式下暂存，连接后发送给 Master）
    struct LocalModel {
        cluster::ModelType type;
        std::string name;
        std::function<void(const nlohmann::json&, std::shared_ptr<BaseDataProvider>)> callback;
    };
    std::vector<LocalModel> local_models_;
    mutable std::mutex models_mutex_;
    
    std::atomic<bool> running_{false};
};

} // namespace openai_api
