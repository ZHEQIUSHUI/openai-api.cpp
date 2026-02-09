#include "openai_api/core/data_provider.hpp"
#include "openai_api/core/output_chunk.hpp"

#include <iostream>
#include <thread>
#include <cassert>
#include <chrono>

using namespace openai_api;

// 测试基本 push/pop 功能
void test_basic_push_pop() {
    std::cout << "Test: basic_push_pop... " << std::flush;
    
    QueueProvider provider;
    
    // 推入数据
    provider.push(OutputChunk::TextDelta("Hello", "gpt-4"));
    provider.push(OutputChunk::TextDelta(" World", "gpt-4"));
    provider.end();
    
    // 弹出数据
    auto chunk1 = provider.pop();
    assert(chunk1.has_value());
    assert(chunk1->type == OutputChunkType::TextDelta);
    assert(chunk1->text == "Hello");
    
    auto chunk2 = provider.pop();
    assert(chunk2.has_value());
    assert(chunk2->text == " World");
    
    std::cout << "PASSED" << std::endl;
}

// 测试 wait_pop 阻塞功能
void test_wait_pop() {
    std::cout << "Test: wait_pop... " << std::flush;
    
    QueueProvider provider;
    
    std::thread producer([&provider]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        provider.push(OutputChunk::TextDelta("delayed", "gpt-4"));
        provider.end();
    });
    
    auto chunk = provider.wait_pop();
    assert(chunk.has_value());
    assert(chunk->text == "delayed");
    
    producer.join();
    std::cout << "PASSED" << std::endl;
}

// 测试超时
void test_timeout() {
    std::cout << "Test: timeout... " << std::flush;
    
    // 设置 200ms 超时
    QueueProvider provider(std::chrono::milliseconds(200));
    
    // 推入数据
    provider.push(OutputChunk::TextDelta("data", "gpt-4"));
    
    // 等待 50ms（不超过超时）
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    
    // 应该还能取出数据
    auto chunk = provider.pop();
    assert(chunk.has_value());
    assert(chunk->text == "data");
    
    // 现在队列为空，等待 300ms 让超时发生
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    
    // 此时应该已经超时
    assert(provider.is_ended());
    
    std::cout << "PASSED" << std::endl;
}

// 测试 push 刷新超时
void test_push_refresh_timeout() {
    std::cout << "Test: push_refresh_timeout... " << std::flush;
    
    // 设置 200ms 超时
    QueueProvider provider(std::chrono::milliseconds(200));
    
    // 推入第一个数据
    provider.push(OutputChunk::TextDelta("1", "gpt-4"));
    
    // 等待 150ms（不超过超时）
    std::this_thread::sleep_for(std::chrono::milliseconds(150));
    
    // 推入第二个数据，应该刷新超时计时器
    provider.push(OutputChunk::TextDelta("2", "gpt-4"));
    
    // 再等待 150ms
    std::this_thread::sleep_for(std::chrono::milliseconds(150));
    
    // 此时不应该超时
    assert(!provider.is_ended());
    
    // 应该还能弹出数据
    auto chunk = provider.pop();
    assert(chunk.has_value());
    assert(chunk->text == "1");
    
    chunk = provider.pop();
    assert(chunk.has_value());
    assert(chunk->text == "2");
    
    std::cout << "PASSED" << std::endl;
}

// 测试线程安全
void test_thread_safety() {
    std::cout << "Test: thread_safety... " << std::flush;
    
    QueueProvider provider;
    const int num_items = 1000;
    
    std::thread producer([&provider]() {
        for (int i = 0; i < num_items; ++i) {
            provider.push(OutputChunk::TextDelta(std::to_string(i), "gpt-4"));
        }
        provider.end();
    });
    
    int count = 0;
    while (true) {
        auto chunk = provider.wait_pop();
        if (!chunk.has_value() || chunk->is_end()) {
            break;
        }
        count++;
    }
    
    producer.join();
    assert(count == num_items);
    
    std::cout << "PASSED" << std::endl;
}

// 测试 wait_pop_for 超时
void test_wait_pop_for_timeout() {
    std::cout << "Test: wait_pop_for_timeout... " << std::flush;
    
    QueueProvider provider;
    
    auto start = std::chrono::steady_clock::now();
    auto chunk = provider.wait_pop_for(std::chrono::milliseconds(100));
    auto elapsed = std::chrono::steady_clock::now() - start;
    
    assert(!chunk.has_value());
    assert(elapsed >= std::chrono::milliseconds(90));  // 允许 10ms 误差
    
    std::cout << "PASSED" << std::endl;
}

// 测试 empty 和 size
void test_empty_and_size() {
    std::cout << "Test: empty_and_size... " << std::flush;
    
    QueueProvider provider;
    
    assert(provider.empty());
    assert(provider.size() == 0);
    
    provider.push(OutputChunk::TextDelta("test", "gpt-4"));
    
    assert(!provider.empty());
    assert(provider.size() == 1);
    
    provider.pop();
    
    assert(provider.empty());
    assert(provider.size() == 0);
    
    std::cout << "PASSED" << std::endl;
}

// 测试 Error chunk
void test_error_chunk() {
    std::cout << "Test: error_chunk... " << std::flush;
    
    QueueProvider provider;
    
    provider.push(OutputChunk::Error("test_error", "Something went wrong"));
    provider.end();
    
    auto chunk = provider.pop();
    assert(chunk.has_value());
    assert(chunk->is_error());
    assert(chunk->error_code == "test_error");
    assert(chunk->error_message == "Something went wrong");
    
    std::cout << "PASSED" << std::endl;
}

int main() {
    std::cout << "=== DataProvider Tests ===" << std::endl;
    
    test_basic_push_pop();
    test_wait_pop();
    test_timeout();
    test_push_refresh_timeout();
    test_thread_safety();
    test_wait_pop_for_timeout();
    test_empty_and_size();
    test_error_chunk();
    
    std::cout << "\nAll tests PASSED!" << std::endl;
    return 0;
}
