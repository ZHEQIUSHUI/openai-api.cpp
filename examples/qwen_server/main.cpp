/**
 * Qwen Server Example
 * 
 * 展示如何使用 openai_api 库构建一个支持 Qwen 模型的推理服务
 */

#include <openai_api/server.hpp>
#include <iostream>
#include <thread>
#include <chrono>

using namespace openai_api;

int main(int argc, char* argv[]) {
    int port = 8080;
    if (argc > 1) {
        port = std::stoi(argv[1]);
    }
    
    // 创建服务器
    Server server(port);
    server.setMaxConcurrency(10);
    
    // 注册 Qwen-0.6B 模型
    server.registerChat("qwen-0.6b", [](const ChatRequest& req, std::shared_ptr<BaseDataProvider> provider) {
        std::cout << "[Qwen-0.6B] Processing request with " << req.messages.size() << " messages" << std::endl;
        
        // 获取用户输入
        std::string user_input;
        for (const auto& msg : req.messages) {
            if (msg.role == "user") {
                user_input = msg.content;
                break;
            }
        }
        
        // 模拟 Qwen 推理
        std::string response = "[Qwen-0.6B] You said: \"" + user_input + "\"\n";
        response += "This is a response from Qwen 0.6B model.";
        
        if (req.stream) {
            // 流式输出
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
    
    // 注册 Qwen-7B 模型
    server.registerChat("qwen-7b", [](const ChatRequest& req, std::shared_ptr<BaseDataProvider> provider) {
        std::cout << "[Qwen-7B] Processing request" << std::endl;
        
        std::string user_input;
        for (const auto& msg : req.messages) {
            if (msg.role == "user") {
                user_input = msg.content;
                break;
            }
        }
        
        // 模拟更大的模型输出
        std::string response = "[Qwen-7B] You asked: \"" + user_input + "\"\n\n";
        response += "As Qwen-7B, I can provide a more detailed response. ";
        response += "This model has better understanding and reasoning capabilities compared to 0.6B.";
        
        if (req.stream) {
            std::istringstream iss(response);
            std::string word;
            while (iss >> word) {
                provider->push(OutputChunk::TextDelta(word + " ", req.model));
                std::this_thread::sleep_for(std::chrono::milliseconds(30));
            }
            provider->push(OutputChunk::FinalText("", req.model));
        } else {
            provider->push(OutputChunk::FinalText(response, req.model));
        }
        
        provider->end();
    });
    
    // 注册 Embedding 模型
    server.registerEmbedding("text-embedding-qwen", [](const EmbeddingRequest& req, std::shared_ptr<BaseDataProvider> provider) {
        std::cout << "[Qwen-Embedding] Generating embeddings for " << req.inputs.size() << " inputs" << std::endl;
        
        // 模拟生成 embedding
        std::vector<std::vector<float>> embeddings;
        for (size_t i = 0; i < req.inputs.size(); ++i) {
            std::vector<float> emb(1536);
            for (float& v : emb) {
                v = static_cast<float>(rand()) / RAND_MAX;
            }
            embeddings.push_back(emb);
        }
        
        provider->push(OutputChunk::BatchEmbeddings(embeddings, req.model));
        provider->end();
    });
    
    std::cout << "========================================" << std::endl;
    std::cout << "Qwen Server Example" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Registered models:" << std::endl;
    for (const auto& model : server.listModels()) {
        std::cout << "  - " << model << std::endl;
    }
    std::cout << std::endl;
    std::cout << "Test with:" << std::endl;
    std::cout << "  curl http://localhost:" << port << "/v1/models" << std::endl;
    std::cout << "  curl -X POST http://localhost:" << port << "/v1/chat/completions \\" << std::endl;
    std::cout << "    -H \"Content-Type: application/json\" \\" << std::endl;
    std::cout << "    -d '{\"model\":\"qwen-0.6b\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}]}'" << std::endl;
    std::cout << std::endl;
    
    server.run();
    return 0;
}
