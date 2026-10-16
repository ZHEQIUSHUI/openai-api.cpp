#pragma once

#include "openai_api/core/api_export.hpp"

/**
 * OpenAI API Compatible Server Library
 * 
 * 一个 C++ 库，用于构建兼容 OpenAI API 的推理服务。
 * 支持多模型路由：同一端点可以注册多个模型实现。
 * 
 * 示例用法：
 * @code
 * #include <openai_api/server.hpp>
 * 
 * int main() {
 *     // 创建服务器
 *     openai_api::Server server(8080);
 *     
 *     // 注册多个 ASR 实现
 *     server.registerASR("whisper-1", [](const auto& req, auto provider) {
 *         // Whisper 实现
 *     });
 *     server.registerASR("sensevoice", [](const auto& req, auto provider) {
 *         // SenseVoice 实现
 *     });
 *     
 *     // 注册多个 LLM 实现
 *     server.registerChat("qwen-0.6b", [](const auto& req, auto provider) {
 *         // Qwen 实现
 *     });
 *     server.registerChat("llama-7b", [](const auto& req, auto provider) {
 *         // Llama 实现
 *     });
 *     
 *     // 运行服务器
 *     server.run();
 * }
 * @endcode
 */

#include "types.hpp"
#include "router.hpp"
#include "upstream.hpp"
#include "openai_api/core/data_provider.hpp"
#include "encoder/encoder.hpp"
#include "utils/httplib.h"

#include <string>
#include <vector>
#include <memory>
#include <thread>
#include <atomic>
#include <chrono>

namespace openai_api {

/**
 * 服务器配置
 */
struct ServerOptions {
    std::string host = "0.0.0.0";
    int port = 8080;
    int max_concurrency = 10;
    std::chrono::milliseconds default_timeout{60000};
    std::chrono::milliseconds wait_timeout{5000};
    std::string api_key;  // 空表示不启用认证
    std::string worker_id;  // Worker ID（集群模式下使用，留空自动生成）
    std::string owner = "openai-api";  // 模型列表中显示的拥有者名称

    // 上游透传 / 统一网关
    bool admin_api = false;            // 是否启用 /admin/upstreams 管理接口
    std::string admin_key;             // 管理接口 key（Authorization: Bearer <key>），空表示不校验
    bool admin_allow_remote = false;   // 是否允许非本机访问管理接口（开启时建议同时设置 admin_key）
    std::chrono::milliseconds upstream_health_interval{5000};  // 上游健康检查间隔，0 表示关闭
    int thread_pool_size = 0;          // HTTP 工作线程数，0 使用 httplib 默认值；每个流式连接占用一个线程
};

/**
 * OpenAI API 兼容服务器
 */
class OPENAI_API_API Server {
public:
    Server();
    
    ~Server();
    
    // 禁止拷贝
    Server(const Server&) = delete;
    Server& operator=(const Server&) = delete;
    
    // ============ 配置 ============
    
    /**
     * 设置最大并发数
     */
    void setMaxConcurrency(int max);
    
    /**
     * 设置默认超时时间
     */
    void setTimeout(std::chrono::milliseconds timeout);

    /**
     * 设置并发槽排队等待超时（请求等待空闲推理槽的最长时间，超时返回 503）。
     * 默认 5000ms；串行推理场景建议设为与请求超时一致，让请求排队而非被拒。
     */
    void setWaitTimeout(std::chrono::milliseconds timeout);

    /**
     * 设置 API key（启用认证）
     */
    void setApiKey(const std::string& api_key);
    
    /**
     * 设置模型拥有者名称（显示在 /v1/models 中）
     */
    void setOwner(const std::string& owner);
    void setImageGenerationOutputDir(const std::string& dir);
    
    // ============ 模型注册 ============
    
    /**
     * 注册 LLM/VLM 模型
     * @param model_name 模型名称（如 "qwen-0.6b", "llama-7b"）
     * @param callback 模型推理回调
     */
    void registerChat(const std::string& model_name, ChatCallback callback,
                      ChatModelOptions options = {});
    
    /**
     * 注册 Embedding 模型
     */
    void registerEmbedding(const std::string& model_name, EmbeddingCallback callback);

    /**
     * 为任意已注册(或即将注册)的模型设置 /v1/models 中附带的额外字段,
     * 例如 {"prefill_max_token_num": 1408, "max_token_len": 1407}
     */
    void setModelExtraFields(const std::string& model_name, nlohmann::json fields);
    
    /**
     * 注册 ASR 模型
     * @param model_name 模型名称（如 "whisper-1", "sensevoice"）
     * @param callback 模型推理回调
     */
    void registerASR(const std::string& model_name, ASRCallback callback);
    
    /**
     * 注册 TTS 模型
     */
    void registerTTS(const std::string& model_name, TTSCallback callback);
    
    /**
     * 注册图像生成模型
     */
    void registerImageGeneration(const std::string& model_name, ImageGenCallback callback);
    
    // ============ 模型管理 ============
    
    /**
     * 获取已注册的模型列表
     */
    std::vector<std::string> listModels() const;
    
    /**
     * 检查模型是否已注册
     */
    bool hasModel(const std::string& model_name) const;
    
    /**
     * 卸载模型（同时移除同名上游）
     */
    void unregisterModel(const std::string& model_name);

    // ============ 上游透传 ============

    /**
     * 注册（或替换）上游：该模型名的请求原样转发到 options.base_url，
     * 响应（含 SSE 流式）原样转回。上游优先于同名的本地模型。
     * base_url 不合法时抛出 std::invalid_argument。
     */
    void registerUpstream(const std::string& model_name, UpstreamOptions options);

    /**
     * 注销上游，不存在时返回 false
     */
    bool unregisterUpstream(const std::string& model_name);

    /**
     * 上游列表（含健康状态和最近一次调用统计）
     */
    std::vector<UpstreamInfo> listUpstreams() const;

    /**
     * 上游注册表变化时回调（可用于持久化）
     */
    void setUpstreamChangedCallback(std::function<void()> callback);

    /**
     * 启用 /admin/upstreams 管理接口
     * @param admin_key 管理接口 key，空表示不校验
     * @param allow_remote 是否允许非本机访问
     */
    void setAdminApi(bool enabled, const std::string& admin_key = "", bool allow_remote = false);
    
    // ============ 运行控制 ============
    
    /**
     * 启动服务器（阻塞调用）
     * @param port 端口号
     */
    void run(int port = 8080);
    
    /**
     * 使用配置启动服务器（阻塞调用）
     * @param options 服务器配置
     */
    void run(const ServerOptions& options);
    
    /**
     * 启动服务器（非阻塞）
     * @param port 端口号
     * @return 后台线程
     */
    std::thread runAsync(int port = 8080);
    
    /**
     * 使用配置启动服务器（非阻塞）
     * @param options 服务器配置
     * @return 后台线程
     */
    std::thread runAsync(const ServerOptions& options);
    
    /**
     * 停止服务器
     */
    void stop();

    /**
     * Extra JSON endpoint (e.g. /tokenize, /version). The handler gets the parsed request body
     * ({} for GET) and fills the response JSON; its return value is the HTTP status.
     * API-key checks apply as for the built-in routes. Register before run().
     */
    using JsonHandler = std::function<int(const nlohmann::json& body, nlohmann::json& out)>;
    void addJsonRoute(const std::string& method, const std::string& path, JsonHandler handler);
    
    /**
     * 检查服务器是否正在运行
     */
    bool isRunning() const;

private:
    void setupRoutes();
    bool verifyApiKey(const httplib::Request& req) const;
    bool verifyAdmin(const httplib::Request& req, httplib::Response& res) const;
    
    // 端点处理函数
    void handleHealth(const httplib::Request& req, httplib::Response& res);
    void handleModels(const httplib::Request& req, httplib::Response& res);
    void handleChatCompletions(const httplib::Request& req, httplib::Response& res);
    void handleCompletions(const httplib::Request& req, httplib::Response& res);
    // Shared by /v1/chat/completions and the legacy /v1/completions (completion = true).
    void handleGeneration(const httplib::Request& req, httplib::Response& res, bool completion);
    void handleMetrics(const httplib::Request& req, httplib::Response& res);
    void countUsage(const OutputChunk& chunk);
    void handleEmbeddings(const httplib::Request& req, httplib::Response& res);
    // llama.cpp compatibility endpoints:
    // - POST /embedding
    // - POST /embeddings (non-OAI, response is a JSON array)
    void handleLlamaEmbedding(const httplib::Request& req, httplib::Response& res);
    void handleLlamaEmbeddings(const httplib::Request& req, httplib::Response& res);
    void handleTranscriptions(const httplib::Request& req, httplib::Response& res);
    void handleTranslations(const httplib::Request& req, httplib::Response& res);
    void handleSpeech(const httplib::Request& req, httplib::Response& res);
    void handleImageGenerations(const httplib::Request& req, httplib::Response& res);
    void handleImageEdits(const httplib::Request& req, httplib::Response& res);
    // 只支持上游透传的端点（如 /v1/completions）
    void handleUpstreamOnly(const httplib::Request& req, httplib::Response& res);
    // 管理接口
    void handleAdminListUpstreams(const httplib::Request& req, httplib::Response& res);
    void handleAdminAddUpstream(const httplib::Request& req, httplib::Response& res);
    void handleAdminRemoveUpstream(const httplib::Request& req, httplib::Response& res);
    
    // 并发控制
    bool acquireSlot();
    void releaseSlot();
    
private:
    ServerOptions options_;
    ModelRouter router_;
    UpstreamManager upstreams_;
    httplib::Server http_server_;
    std::string image_generation_output_dir_;
    
    std::atomic<bool> running_{false};
    std::atomic<int> current_concurrency_{0};
    // /metrics counters (vLLM metric names)
    std::atomic<int> waiting_requests_{0};
    std::atomic<uint64_t> prompt_tokens_total_{0};
    std::atomic<uint64_t> generation_tokens_total_{0};
    std::atomic<uint64_t> requests_ok_stop_{0};
    std::atomic<uint64_t> requests_ok_length_{0};
    std::atomic<uint64_t> requests_ok_tool_calls_{0};
    std::atomic<uint64_t> requests_failed_{0};
    std::mutex slot_mutex_;
    std::condition_variable slot_cv_;
};

} // namespace openai_api
