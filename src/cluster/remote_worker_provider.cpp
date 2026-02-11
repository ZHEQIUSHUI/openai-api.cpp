#include "openai_api/cluster/remote_worker_provider.hpp"
#include "openai_api/cluster/worker_manager.hpp"

namespace openai_api {
namespace cluster {

RemoteWorkerProvider::RemoteWorkerProvider(const std::string& request_id,
                                            WorkerManager* manager,
                                            std::chrono::milliseconds timeout)
    : request_id_(request_id)
    , manager_(manager)
    , timeout_(timeout)
    , last_activity_(std::chrono::steady_clock::now())
{}

void RemoteWorkerProvider::on_response(const nlohmann::json& data, bool is_error) {
    if (is_error) {
        std::string code = data.value("error_code", "worker_error");
        std::string msg = data.value("error_message", "Unknown worker error");
        push(OutputChunk::Error(code, msg));
    } else {
        // 根据响应类型构造 OutputChunk
        if (data.contains("text")) {
            // Chat completion
            OutputChunk chunk;
            chunk.type = OutputChunkType::TextDelta;
            chunk.text = data.value("text", "");
            // is_delta 和 finish_reason 通过 obj 传递
            if (data.contains("finish_reason")) {
                chunk.obj["finish_reason"] = data.value("finish_reason", "");
            }
            push(chunk);
        } else if (data.contains("embeddings") || data.contains("embedding")) {
            // Embedding
            OutputChunk chunk;
            chunk.type = OutputChunkType::Embedding;
            if (data.contains("embeddings")) {
                for (const auto& emb : data["embeddings"]) {
                    std::vector<float> vec;
                    for (const auto& v : emb) {
                        vec.push_back(v.get<float>());
                    }
                    chunk.embeds.push_back(vec);
                }
            }
            push(chunk);
        } else if (data.contains("bytes")) {
            // Binary data (ASR/TTS/Image)
            OutputChunk chunk;
            chunk.type = OutputChunkType::AudioBytes;
            std::string bytes_str = data.value("bytes", "");
            chunk.bytes.assign(bytes_str.begin(), bytes_str.end());
            chunk.mime_type = data.value("mime_type", "application/octet-stream");
            push(chunk);
        }
    }
}

void RemoteWorkerProvider::on_end() {
    end();
}

void RemoteWorkerProvider::on_error(const std::string& code, const std::string& message) {
    push(OutputChunk::Error(code, message));
    end();
}

bool RemoteWorkerProvider::push(const OutputChunk& chunk) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (ended_ || !writable_) return false;
        queue_.push(chunk);
        last_activity_ = std::chrono::steady_clock::now();
    }
    cv_.notify_one();
    return true;
}

bool RemoteWorkerProvider::push(OutputChunk&& chunk) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (ended_ || !writable_) return false;
        queue_.push(std::move(chunk));
        last_activity_ = std::chrono::steady_clock::now();
    }
    cv_.notify_one();
    return true;
}

void RemoteWorkerProvider::end() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        ended_ = true;
    }
    cv_.notify_all();
}

bool RemoteWorkerProvider::is_ended() {
    std::lock_guard<std::mutex> lock(mutex_);
    return ended_ && queue_.empty();
}

bool RemoteWorkerProvider::is_writable() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return writable_ && !ended_;
}

bool RemoteWorkerProvider::is_alive() const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto elapsed = std::chrono::steady_clock::now() - last_activity_;
    return elapsed <= timeout_;
}

void RemoteWorkerProvider::reset_timeout() {
    std::lock_guard<std::mutex> lock(mutex_);
    last_activity_ = std::chrono::steady_clock::now();
}

std::optional<OutputChunk> RemoteWorkerProvider::pop() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (queue_.empty()) return std::nullopt;
    OutputChunk chunk = std::move(queue_.front());
    queue_.pop();
    return chunk;
}

std::optional<OutputChunk> RemoteWorkerProvider::wait_pop() {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this] { return !queue_.empty() || ended_; });
    if (queue_.empty()) return std::nullopt;
    OutputChunk chunk = std::move(queue_.front());
    queue_.pop();
    return chunk;
}

std::optional<OutputChunk> RemoteWorkerProvider::wait_pop_for(std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lock(mutex_);
    bool has_data = cv_.wait_for(lock, timeout, [this] { return !queue_.empty() || ended_; });
    if (!has_data || queue_.empty()) return std::nullopt;
    OutputChunk chunk = std::move(queue_.front());
    queue_.pop();
    return chunk;
}

size_t RemoteWorkerProvider::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.size();
}

bool RemoteWorkerProvider::empty() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.empty();
}

} // namespace cluster
} // namespace openai_api
