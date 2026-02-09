/**
 * OpenAI API Server - Standalone executable
 * 
 * 这是一个使用 openai_api 库构建的独立服务器示例
 */

#include <openai_api/server.hpp>
#include <iostream>
#include <csignal>
#include <memory>
#include <random>
#include <sstream>

using namespace openai_api;

std::unique_ptr<Server> g_server;

void signal_handler(int sig) {
    std::cout << "\nReceived signal " << sig << ", shutting down..." << std::endl;
    if (g_server) {
        g_server->stop();
    }
}

void print_usage(const char* program) {
    std::cout << "Usage: " << program << " [options] [port] [max_concurrency]" << std::endl;
    std::cout << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --models <names>  Comma-separated list of supported models" << std::endl;
    std::cout << "  --api-key <key>   API key for authentication" << std::endl;
    std::cout << "  -h, --help        Show this help message" << std::endl;
    std::cout << std::endl;
    std::cout << "Examples:" << std::endl;
    std::cout << "  " << program << "                           # Run on port 8080" << std::endl;
    std::cout << "  " << program << " 3000                     # Run on port 3000" << std::endl;
    std::cout << "  " << program << " --api-key my-key 8080    # With API key auth" << std::endl;
}

int main(int argc, char* argv[]) {
    // 解析命令行参数
    std::vector<std::string> models_to_register;
    std::string api_key;
    int port = 8080;
    int max_concurrency = 10;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        }
        else if (arg == "--models" && i + 1 < argc) {
            std::string models_str = argv[++i];
            std::stringstream ss(models_str);
            std::string model;
            while (std::getline(ss, model, ',')) {
                model.erase(0, model.find_first_not_of(" \t"));
                model.erase(model.find_last_not_of(" \t") + 1);
                if (!model.empty()) {
                    models_to_register.push_back(model);
                }
            }
        }
        else if (arg == "--api-key" && i + 1 < argc) {
            api_key = argv[++i];
        }
        else {
            // 尝试解析为数字
            try {
                int val = std::stoi(arg);
                if (port == 8080) {
                    port = val;
                } else {
                    max_concurrency = val;
                }
            } catch (...) {
                std::cerr << "Error: Unknown argument: " << arg << std::endl;
                return 1;
            }
        }
    }
    
    // 设置信号处理
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);
    
    // 创建服务器
    ServerOptions options;
    options.port = port;
    options.max_concurrency = max_concurrency;
    options.api_key = api_key;
    
    g_server = std::make_unique<Server>(options);
    
    // 注册模型
    if (models_to_register.empty()) {
        // 默认注册一些模型
        models_to_register = {"gpt-4", "gpt-4o", "whisper-1", "text-embedding-ada-002"};
    }
    
    for (const auto& model : models_to_register) {
        // 根据模型名称前缀判断类型
        if (model.find("gpt-") == 0 || model.find("qwen") == 0 || model.find("llama") == 0) {
            // LLM 模型
            g_server->registerChat(model, [model](const ChatRequest& req, auto provider) {
                std::string response = "[Mock " + model + "] This is a response from " + model;
                
                if (req.stream) {
                    std::istringstream iss(response);
                    std::string word;
                    while (iss >> word) {
                        provider->push(OutputChunk::TextDelta(word + " ", req.model));
                        std::this_thread::sleep_for(std::chrono::milliseconds(50));
                    }
                    provider->push(OutputChunk::FinalText("", req.model));
                } else {
                    provider->push(OutputChunk::FinalText(response, req.model));
                }
                provider->end();
            });
        }
        else if (model.find("whisper") == 0 || model.find("sensevoice") == 0) {
            // ASR 模型
            g_server->registerASR(model, [model](const ASRRequest& req, auto provider) {
                std::string transcript = "[Mock " + model + "] Transcription result";
                provider->push(OutputChunk::FinalText(transcript, req.model));
                provider->end();
            });
        }
        else if (model.find("embedding") != std::string::npos) {
            // Embedding 模型
            g_server->registerEmbedding(model, [model](const EmbeddingRequest& req, auto provider) {
                std::vector<std::vector<float>> embeddings;
                for (size_t i = 0; i < req.inputs.size(); ++i) {
                    std::vector<float> emb(1536);
                    for (float& v : emb) v = static_cast<float>(rand()) / RAND_MAX;
                    embeddings.push_back(emb);
                }
                provider->push(OutputChunk::BatchEmbeddings(embeddings, req.model));
                provider->end();
            });
        }
    }
    
    std::cout << "OpenAI API Server" << std::endl;
    std::cout << "=================" << std::endl;
    std::cout << "Port: " << port << std::endl;
    std::cout << "Max Concurrency: " << max_concurrency << std::endl;
    std::cout << "API Key: " << (api_key.empty() ? "disabled" : "enabled") << std::endl;
    std::cout << "Models:" << std::endl;
    for (const auto& model : g_server->listModels()) {
        std::cout << "  - " << model << std::endl;
    }
    std::cout << std::endl;
    std::cout << "Press Ctrl+C to stop" << std::endl;
    std::cout << std::endl;
    
    g_server->run();
    return 0;
}
