#pragma once

/**
 * 上游透传（Upstream）
 *
 * 把某个模型名的请求原样转发给另一个 OpenAI 兼容服务（例如本机另一个端口上的 axllm serve），
 * 响应（包括 SSE 流式响应）原样逐块转回，不解析、不改写内容。
 *
 * 用于搭建统一网关：网关对外只开一个端口，管理平台启动模型进程后，
 * 通过 registerUpstream() 或管理接口 /admin/upstreams 把 "模型名 -> 地址" 注册进来。
 */

#include "openai_api/core/api_export.hpp"
#include "utils/json.hpp"
#include "utils/httplib.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <string>
#include <thread>
#include <vector>

namespace openai_api {

/**
 * 上游配置
 */
struct OPENAI_API_API UpstreamOptions {
    std::string base_url;                       // 上游地址，如 http://127.0.0.1:18001（可带路径前缀）
    std::string upstream_model;                 // 上游实际使用的模型名，空表示与注册名相同
    std::string api_key;                        // 转发给上游时使用的 key，空表示不带
    std::string health_path = "/v1/models";     // 健康检查路径，空表示不做健康检查
    std::chrono::milliseconds connect_timeout{3000};
    std::chrono::milliseconds read_timeout{600000};
    nlohmann::json extra;                       // 附加到 /v1/models 中该模型条目的字段

    nlohmann::json to_json() const;
    static UpstreamOptions from_json(const nlohmann::json& j);
};

enum class UpstreamHealth { Unknown, Healthy, Unhealthy };

OPENAI_API_API const char* to_string(UpstreamHealth health);

/**
 * 最近一次调用的统计
 */
struct OPENAI_API_API UpstreamCallStats {
    std::string endpoint;                       // 请求路径，如 /v1/chat/completions
    int status = 0;                             // 上游返回的 HTTP 状态码，0 表示没有连上上游
    bool stream = false;
    int64_t started_at = 0;                     // unix 时间戳（秒）
    double duration_ms = 0;                     // 总耗时
    std::optional<double> ttft_ms;              // 首字延迟：流式为网关实测，非流式取上游 usage.ttft_ms
    nlohmann::json usage;                       // 上游返回的 usage，没有则为 null
    std::optional<double> tokens_per_second;    // 优先取上游 usage.decode_tps，否则 completion_tokens / 生成耗时

    nlohmann::json to_json() const;
};

/**
 * 上游状态（供管理接口和调用方查看）
 */
struct OPENAI_API_API UpstreamInfo {
    std::string model;
    UpstreamOptions options;
    UpstreamHealth health = UpstreamHealth::Unknown;
    nlohmann::json model_info;                  // 从上游 /v1/models 读到的该模型字段
    uint64_t requests = 0;
    uint64_t errors = 0;
    int active = 0;
    std::string last_error;
    std::optional<UpstreamCallStats> last_call;

    nlohmann::json to_json() const;
};

/**
 * 上游管理器：注册表、请求透传、健康检查
 */
class OPENAI_API_API UpstreamManager {
public:
    UpstreamManager();
    ~UpstreamManager();

    UpstreamManager(const UpstreamManager&) = delete;
    UpstreamManager& operator=(const UpstreamManager&) = delete;

    /**
     * 注册（或替换）上游。base_url 不合法时抛出 std::invalid_argument。
     */
    void add(const std::string& model_name, UpstreamOptions options);
    bool remove(const std::string& model_name);
    bool has(const std::string& model_name) const;
    bool empty() const;
    std::vector<std::string> names() const;
    std::vector<UpstreamInfo> list() const;

    /**
     * /v1/models 中该模型的条目字段（上游字段 + extra），不健康时返回 std::nullopt
     */
    std::optional<nlohmann::json> listedModelFields(const std::string& model_name) const;

    /**
     * 如果请求的 model 是已注册的上游，则转发并填好 res，返回 true；否则返回 false。
     */
    bool tryProxy(const httplib::Request& req, httplib::Response& res);

    /**
     * 注册表变化时回调（用于持久化）
     */
    void setOnChange(std::function<void()> callback);

    void startHealthCheck(std::chrono::milliseconds interval);
    void stopHealthCheck();

private:
    struct Entry;

    std::shared_ptr<Entry> find(const std::string& model_name) const;
    void notifyChange();
    void healthLoop();
    void checkHealth(const std::shared_ptr<Entry>& entry);

    mutable std::shared_mutex mutex_;
    std::map<std::string, std::shared_ptr<Entry>> entries_;

    std::mutex change_mutex_;
    std::function<void()> on_change_;

    std::mutex health_mutex_;
    std::condition_variable health_cv_;
    std::thread health_thread_;
    bool health_stop_ = false;
    bool health_wakeup_ = false;
    std::chrono::milliseconds health_interval_{5000};
};

} // namespace openai_api
