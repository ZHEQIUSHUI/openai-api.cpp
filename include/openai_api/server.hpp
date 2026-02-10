#pragma once

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
};

/**
 * OpenAI API 兼容服务器
 */
class Server {
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
     * 设置 API key（启用认证）
     */
    void setApiKey(const std::string& api_key);
    
    // ============ 模型注册 ============
    
    /**
     * 注册 LLM/VLM 模型
     * @param model_name 模型名称（如 "qwen-0.6b", "llama-7b"）
     * @param callback 模型推理回调
     */
    void registerChat(const std::string& model_name, ChatCallback callback);
    
    /**
     * 注册 Embedding 模型
     */
    void registerEmbedding(const std::string& model_name, EmbeddingCallback callback);
    
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
     * 卸载模型
     */
    void unregisterModel(const std::string& model_name);
    
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
     * 检查服务器是否正在运行
     */
    bool isRunning() const;

private:
    void setupRoutes();
    bool verifyApiKey(const httplib::Request& req) const;
    
    // 端点处理函数
    void handleHealth(const httplib::Request& req, httplib::Response& res);
    void handleModels(const httplib::Request& req, httplib::Response& res);
    void handleChatCompletions(const httplib::Request& req, httplib::Response& res);
    void handleEmbeddings(const httplib::Request& req, httplib::Response& res);
    void handleTranscriptions(const httplib::Request& req, httplib::Response& res);
    void handleTranslations(const httplib::Request& req, httplib::Response& res);
    void handleSpeech(const httplib::Request& req, httplib::Response& res);
    void handleImageGenerations(const httplib::Request& req, httplib::Response& res);
    
    // 并发控制
    bool acquireSlot();
    void releaseSlot();
    
private:
    ServerOptions options_;
    ModelRouter router_;
    httplib::Server http_server_;
    
    std::atomic<bool> running_{false};
    std::atomic<int> current_concurrency_{0};
    std::mutex slot_mutex_;
    std::condition_variable slot_cv_;
};

} // namespace openai_api
