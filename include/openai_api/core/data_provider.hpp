#pragma once

#include "output_chunk.hpp"

#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <chrono>
#include <thread>
#include <functional>
#include <optional>

namespace openai_api {

/**
 * BaseDataProvider - 通用数据提供者基类
 * 
 * 职责：
 * 1. 线程安全队列管理
 * 2. 生命周期管理（end）
 * 3. 超时管理
 * 4. 阻塞/非阻塞读取
 * 5. 可写入状态判断
 */
class BaseDataProvider {
public:
    virtual ~BaseDataProvider() = default;
    
    /**
     * 推入数据
     * @return true 成功，false 失败（已结束或已断开）
     */
    virtual bool push(const OutputChunk& chunk) = 0;
    virtual bool push(OutputChunk&& chunk) = 0;
    
    /**
     * 标记结束
     */
    virtual void end() = 0;
    
    /**
     * 检查是否已结束
     */
    virtual bool is_ended() = 0;
    
    /**
     * 检查是否可写入
     * @return true 可以 push，false 已结束或已断开
     */
    virtual bool is_writable() const = 0;
    
    /**
     * 检查连接是否存活
     * @return true 连接正常，false 已超时或断开
     */
    virtual bool is_alive() const = 0;
    
    /**
     * 重置超时计时器
     */
    virtual void reset_timeout() = 0;
    
    // 非阻塞弹出
    virtual std::optional<OutputChunk> pop() = 0;
    
    // 阻塞弹出，等待直到有数据或结束
    virtual std::optional<OutputChunk> wait_pop() = 0;
    
    // 带超时的阻塞弹出
    virtual std::optional<OutputChunk> wait_pop_for(std::chrono::milliseconds timeout) = 0;
    
    // 获取队列大小
    virtual size_t size() const = 0;
    
    // 检查是否为空
    virtual bool empty() const = 0;
};

/**
 * QueueProvider - 基于队列的默认数据提供者实现
 * 
 * 特性：
 * 1. 线程安全（使用 mutex + condition_variable）
 * 2. 支持超时自动 end
 * 3. push 自动刷新超时
 * 4. 支持阻塞/非阻塞读取
 * 5. 可写入状态判断
 */
class QueueProvider : public BaseDataProvider {
public:
    explicit QueueProvider(std::chrono::milliseconds timeout = std::chrono::milliseconds(60000))
        : timeout_(timeout)
        , ended_(false)
        , disconnected_(false)
        , last_activity_(std::chrono::steady_clock::now())
    {}
    
    ~QueueProvider() override = default;
    
    /**
     * 推入数据 - 刷新超时计时器
     * @return true 成功，false 已结束/断开或超时
     */
    bool push(const OutputChunk& chunk) override {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            
            // 检查是否可以写入
            if (ended_ || disconnected_) {
                return false;
            }
            
            // 检查超时
            if (check_timeout_locked()) {
                return false;
            }
            
            queue_.push(chunk);
            last_activity_ = std::chrono::steady_clock::now();
        }
        cv_.notify_one();
        return true;
    }
    
    bool push(OutputChunk&& chunk) override {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            
            // 检查是否可以写入
            if (ended_ || disconnected_) {
                return false;
            }
            
            // 检查超时
            if (check_timeout_locked()) {
                return false;
            }
            
            queue_.push(std::move(chunk));
            last_activity_ = std::chrono::steady_clock::now();
        }
        cv_.notify_one();
        return true;
    }
    
    /**
     * 标记结束
     */
    void end() override {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            ended_ = true;
        }
        cv_.notify_all();
    }
    
    /**
     * 标记断开连接（客户端断开）
     */
    void disconnect() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            disconnected_ = true;
            ended_ = true;
        }
        cv_.notify_all();
    }
    
    /**
     * 检查是否已结束
     */
    bool is_ended() override {
        std::lock_guard<std::mutex> lock(mutex_);
        check_timeout_locked();
        return ended_ && queue_.empty();
    }
    
    /**
     * 检查是否可写入
     */
    bool is_writable() const override {
        std::lock_guard<std::mutex> lock(mutex_);
        if (ended_ || disconnected_) {
            return false;
        }
        // 检查是否超时
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - last_activity_);
        return elapsed <= timeout_;
    }
    
    /**
     * 检查连接是否存活
     */
    bool is_alive() const override {
        std::lock_guard<std::mutex> lock(mutex_);
        if (disconnected_ || ended_) {
            return false;
        }
        // 检查是否超时
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - last_activity_);
        return elapsed <= timeout_;
    }
    
    /**
     * 重置超时计时器
     */
    void reset_timeout() override {
        std::lock_guard<std::mutex> lock(mutex_);
        last_activity_ = std::chrono::steady_clock::now();
    }
    
    // 非阻塞弹出
    std::optional<OutputChunk> pop() override {
        std::lock_guard<std::mutex> lock(mutex_);
        
        // 检查超时
        if (check_timeout_locked()) {
            return std::nullopt;
        }
        
        if (queue_.empty()) {
            return std::nullopt;
        }
        
        OutputChunk chunk = std::move(queue_.front());
        queue_.pop();
        return chunk;
    }
    
    // 阻塞弹出
    std::optional<OutputChunk> wait_pop() override {
        std::unique_lock<std::mutex> lock(mutex_);
        
        cv_.wait(lock, [this] {
            return !queue_.empty() || ended_ || disconnected_ || check_timeout_locked();
        });
        
        if (check_timeout_locked()) {
            return std::nullopt;
        }
        
        if (queue_.empty()) {
            return std::nullopt;
        }
        
        OutputChunk chunk = std::move(queue_.front());
        queue_.pop();
        return chunk;
    }
    
    // 带超时的阻塞弹出
    std::optional<OutputChunk> wait_pop_for(std::chrono::milliseconds wait_timeout) override {
        std::unique_lock<std::mutex> lock(mutex_);
        
        bool has_data = cv_.wait_for(lock, wait_timeout, [this] {
            return !queue_.empty() || ended_ || disconnected_ || check_timeout_locked();
        });
        
        if (!has_data) {
            return std::nullopt;  // 等待超时
        }
        
        if (check_timeout_locked()) {
            return std::nullopt;
        }
        
        if (queue_.empty()) {
            return std::nullopt;
        }
        
        OutputChunk chunk = std::move(queue_.front());
        queue_.pop();
        return chunk;
    }
    
    // 获取队列大小
    size_t size() const override {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }
    
    // 检查是否为空
    bool empty() const override {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.empty();
    }
    
    // 设置超时
    void set_timeout(std::chrono::milliseconds timeout) {
        std::lock_guard<std::mutex> lock(mutex_);
        timeout_ = timeout;
    }

private:
    // 检查是否超时（必须在持有锁的情况下调用）
    bool check_timeout_locked() const {
        if (ended_) return false;
        
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - last_activity_);
        
        if (elapsed > timeout_) {
            // 超时，自动标记结束
            const_cast<QueueProvider*>(this)->ended_ = true;
            return true;
        }
        return false;
    }
    
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::queue<OutputChunk> queue_;
    
    std::chrono::milliseconds timeout_;
    std::atomic<bool> ended_;
    std::atomic<bool> disconnected_;  // 客户端断开标记
    std::chrono::steady_clock::time_point last_activity_;
};

} // namespace openai_api
