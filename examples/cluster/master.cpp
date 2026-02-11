/**
 * Cluster Master Server Example
 * 
 * 第一个启动的进程，监听指定端口
 * 后续 Worker 进程可以连接到这个 Master 并注册模型
 */

#include <openai_api/cluster_server.hpp>
#include <iostream>
#include <thread>
#include <chrono>

using namespace openai_api;

int main(int argc, char* argv[]) {
    int port = 8080;
    if (argc > 1) {
        port = std::atoi(argv[1]);
    }
    
    std::cout << "=== OpenAI API Cluster Master ===" << std::endl;
    std::cout << "Port: " << port << std::endl;
    std::cout << std::endl;
    
    ClusterServer server;
    
    // Master 也可以有自己的本地模型
    server.registerChat("master-model", [](const ChatRequest& req, std::shared_ptr<BaseDataProvider> provider) {
        std::cout << "[Master] Processing request for model: master-model" << std::endl;
        
        // 模拟流式响应
        std::vector<std::string> tokens = {"Hello", " from", " Master", "!"};
        for (const auto& token : tokens) {
            provider->push(OutputChunk::TextDelta(token));
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        
        provider->push(OutputChunk::FinalText("Hello from Master!"));
        provider->end();
    });
    
    std::cout << "Starting Master server..." << std::endl;
    std::cout << "Workers can connect to register their models." << std::endl;
    std::cout << "Press Ctrl+C to stop." << std::endl;
    std::cout << std::endl;
    
    // 启动 Master 模式
    server.runAsMaster(port);
    
    return 0;
}
