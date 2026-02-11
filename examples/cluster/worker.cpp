/**
 * Cluster Worker Node Example
 * 
 * 连接到已存在的 Master 服务器，注册自己的模型
 * Master 会将匹配的请求转发给 Worker 处理
 * 
 * 支持跨机器部署：
 * - 默认自动检测本机 IP
 * - 可以通过参数指定监听地址（用于多网卡或 NAT 场景）
 */

#include <openai_api/cluster_server.hpp>
#include <iostream>
#include <thread>
#include <chrono>
#include <cstdlib>

using namespace openai_api;

void printUsage(const char* prog) {
    std::cout << "Usage: " << prog << " [options] [master_host] [master_port]" << std::endl;
    std::cout << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  -n, --name NAME          Worker name (default: random)" << std::endl;
    std::cout << "  -l, --listen ADDR:PORT   Worker listen address (default: auto)" << std::endl;
    std::cout << "  -h, --help               Show this help" << std::endl;
    std::cout << std::endl;
    std::cout << "Examples:" << std::endl;
    std::cout << "  # Local deployment" << std::endl;
    std::cout << "  " << prog << " 127.0.0.1 8080" << std::endl;
    std::cout << std::endl;
    std::cout << "  # Cross-machine deployment" << std::endl;
    std::cout << "  " << prog << " -l 192.168.1.100:28080 192.168.1.50 8080" << std::endl;
}

int main(int argc, char* argv[]) {
    std::string master_host = "127.0.0.1";
    int master_port = 8080;
    std::string worker_name = "worker-" + std::to_string(std::rand() % 1000);
    std::string listen_host = "0.0.0.0";  // 默认监听所有接口
    int listen_port = 0;  // 0 表示自动分配
    
    // 解析参数
    int arg_idx = 1;
    while (arg_idx < argc) {
        std::string arg = argv[arg_idx];
        if (arg == "-h" || arg == "--help") {
            printUsage(argv[0]);
            return 0;
        } else if ((arg == "-n" || arg == "--name") && arg_idx + 1 < argc) {
            worker_name = argv[++arg_idx];
        } else if ((arg == "-l" || arg == "--listen") && arg_idx + 1 < argc) {
            std::string listen_addr = argv[++arg_idx];
            size_t colon_pos = listen_addr.find(':');
            if (colon_pos != std::string::npos) {
                listen_host = listen_addr.substr(0, colon_pos);
                listen_port = std::stoi(listen_addr.substr(colon_pos + 1));
            } else {
                listen_port = std::stoi(listen_addr);
            }
        } else if (arg[0] != '-') {
            break;  // 非选项参数，后面处理
        }
        arg_idx++;
    }
    
    // 解析 Master 地址
    if (arg_idx < argc) master_host = argv[arg_idx++];
    if (arg_idx < argc) master_port = std::atoi(argv[arg_idx++]);
    
    // Worker 内部通信端口 = Master 端口 + 1000
    int master_internal_port = master_port + 1000;
    
    std::cout << "=== OpenAI API Cluster Worker ===" << std::endl;
    std::cout << "Worker Name: " << worker_name << std::endl;
    std::cout << "Master API: " << master_host << ":" << master_port << std::endl;
    std::cout << "Master Internal: " << master_host << ":" << master_internal_port << std::endl;
    if (listen_port > 0) {
        std::cout << "Listen: " << listen_host << ":" << listen_port << std::endl;
    } else {
        std::cout << "Listen: auto (all interfaces)" << std::endl;
    }
    std::cout << std::endl;
    
    ClusterServer server;
    
    // 设置 Worker 监听地址（用于跨机器部署）
    server.setWorkerListenAddress(listen_host, listen_port);
    
    // 注册 Worker 自己的模型
    std::string model_name = worker_name + "-model";
    server.registerChat(model_name, [worker_name](const ChatRequest& req, std::shared_ptr<BaseDataProvider> provider) {
        std::cout << "[" << worker_name << "] Processing request for model: " 
                  << req.model << std::endl;
        
        // 模拟流式响应
        std::vector<std::string> tokens = {"Hello", " from", " " + worker_name, "!"};
        for (const auto& token : tokens) {
            provider->push(OutputChunk::TextDelta(token));
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        
        provider->push(OutputChunk::FinalText("Hello from " + worker_name + "!"));
        provider->end();
    });
    
    std::cout << "Connecting to Master..." << std::endl;
    std::cout << "Registering model: " << model_name << std::endl;
    std::cout << std::endl;
    
    // 启动 Worker 模式，连接到 Master 的内部端口
    if (!server.runAsWorker(master_host, master_internal_port)) {
        std::cerr << "Failed to connect to Master!" << std::endl;
        return 1;
    }
    
    std::cout << "Disconnected from Master." << std::endl;
    return 0;
}
