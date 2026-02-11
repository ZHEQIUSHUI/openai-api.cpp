#pragma once

#include "openai_api/core/api_export.hpp"
#include "../core/data_provider.hpp"
#include "internal_protocol.hpp"

#include <atomic>
#include <mutex>
#include <condition_variable>
#include <queue>

namespace openai_api {
namespace cluster {

class WorkerManager;

/**
 * RemoteWorkerProvider - Master 端包装远程 Worker 的 Provider
 * 
 * 当 Master 收到请求需要转发给 Worker 时，使用此 Provider
 * 它会将数据转发给 Worker，并将 Worker 的响应返回给 Server
 */
class OPENAI_API_API RemoteWorkerProvider : public BaseDataProvider {
public:
    RemoteWorkerProvider(const std::string& request_id,
                         WorkerManager* manager,
                         std::chrono::milliseconds timeout = std::chrono::milliseconds(60000));
    
    // 由 WorkerManager 调用，当收到 Worker 响应时
    void on_response(const nlohmann::json& data, bool is_error);
    void on_end();
    void on_error(const std::string& code, const std::string& message);
    
    // BaseDataProvider 接口
    bool push(const OutputChunk& chunk) override;
    bool push(OutputChunk&& chunk) override;
    void end() override;
    bool is_ended() override;
    bool is_writable() const override;
    bool is_alive() const override;
    void reset_timeout() override;
    std::optional<OutputChunk> pop() override;
    std::optional<OutputChunk> wait_pop() override;
    std::optional<OutputChunk> wait_pop_for(std::chrono::milliseconds timeout) override;
    size_t size() const override;
    bool empty() const override;

private:
    std::string request_id_;
    WorkerManager* manager_;
    
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::queue<OutputChunk> queue_;
    
    std::atomic<bool> ended_{false};
    std::atomic<bool> writable_{true};
    std::chrono::milliseconds timeout_;
    std::chrono::steady_clock::time_point last_activity_;
};

} // namespace cluster
} // namespace openai_api
