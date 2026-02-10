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

// 从 messages JSON 中提取用户输入
std::string extract_user_input(const nlohmann::json& messages) {
    if (!messages.is_array()) return "";
    
    for (const auto& msg : messages) {
        if (msg.contains("role") && msg["role"] == "user" && msg.contains("content")) {
            if (msg["content"].is_string()) {
                return msg["content"].get<std::string>();
            }
            // TODO: 处理多模态内容（图像+文本）
        }
    }
    return "";
}

// 构建 prompt（简化版）
std::string build_prompt(const nlohmann::json& messages) {
    std::string prompt;
    if (messages.is_array()) {
        for (const auto& msg : messages) {
            if (msg.contains("role") && msg.contains("content")) {
                std::string role = msg["role"].get<std::string>();
                std::string content;
                if (msg["content"].is_string()) {
                    content = msg["content"].get<std::string>();
                } else {
                    content = "[多模态内容]";  // 简化处理
                }
                prompt += role + ": " + content + "\n";
            }
        }
    }
    return prompt;
}

int main(int argc, char* argv[]) {
    int port = 8080;
    if (argc > 1) {
        port = std::stoi(argv[1]);
    }
    
    // 创建服务器（全局变量可以在 run() 时再配置）
    static Server server;
    
    // 注册 Qwen-0.6B 模型
    server.registerChat("qwen-0.6b", [](const ChatRequest& req, std::shared_ptr<BaseDataProvider> provider) {
        // 检查是否可以写入
        if (!provider->is_writable()) {
            return;
        }
        
        std::cout << "[Qwen-0.6B] Processing request" << std::endl;
        std::cout << "  Model: " << req.model << std::endl;
        std::cout << "  Stream: " << (req.stream ? "yes" : "no") << std::endl;
        std::cout << "  Temperature: " << req.temperature << std::endl;
        
        // 从 messages JSON 中提取用户输入
        std::string user_input = extract_user_input(req.messages);
        std::cout << "  User input: " << user_input << std::endl;
        
        // 模拟 Qwen 推理
        std::string response = "[Qwen-0.6B] You said: \"" + user_input + "\"\n";
        response += "This is a response from Qwen 0.6B model.";
        
        if (req.stream) {
            // 流式输出
            std::istringstream iss(response);
            std::string word;
            while (iss >> word) {
                if (!provider->is_writable()) {
                    std::cout << "  Connection lost, stopping generation" << std::endl;
                    break;
                }
                if (!provider->push(OutputChunk::TextDelta(word + " ", req.model))) {
                    std::cerr << "  Failed to push chunk" << std::endl;
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
            if (provider->is_writable()) {
                provider->push(OutputChunk::FinalText("", req.model));
            }
        } else {
            // 非流式输出
            if (provider->is_writable()) {
                provider->push(OutputChunk::FinalText(response, req.model));
            }
        }
        
        if (provider->is_writable()) {
            provider->end();
        }
        std::cout << "  Done" << std::endl;
    });
    
    // 注册 Qwen-7B 模型
    server.registerChat("qwen-7b", [](const ChatRequest& req, std::shared_ptr<BaseDataProvider> provider) {
        if (!provider->is_writable()) {
            return;
        }
        
        std::cout << "[Qwen-7B] Processing request" << std::endl;
        
        std::string user_input = extract_user_input(req.messages);
        
        // 模拟更大的模型输出
        std::string response = "[Qwen-7B] You asked: \"" + user_input + "\"\n\n";
        response += "As Qwen-7B, I can provide a more detailed response. ";
        response += "This model has better understanding and reasoning capabilities compared to 0.6B.";
        
        if (req.stream) {
            std::istringstream iss(response);
            std::string word;
            while (iss >> word) {
                if (!provider->is_writable()) break;
                if (!provider->push(OutputChunk::TextDelta(word + " ", req.model))) break;
                std::this_thread::sleep_for(std::chrono::milliseconds(30));
            }
            if (provider->is_writable()) {
                provider->push(OutputChunk::FinalText("", req.model));
            }
        } else {
            if (provider->is_writable()) {
                provider->push(OutputChunk::FinalText(response, req.model));
            }
        }
        
        if (provider->is_writable()) {
            provider->end();
        }
    });
    
    // 注册 Qwen-VL（多模态版本）- 展示如何处理多模态 messages
    server.registerChat("qwen-vl", [](const ChatRequest& req, std::shared_ptr<BaseDataProvider> provider) {
        if (!provider->is_writable()) {
            return;
        }
        
        std::cout << "[Qwen-VL] Processing multimodal request" << std::endl;
        
        // 检查 messages 中是否包含图像
        bool has_image = false;
        if (req.messages.is_array()) {
            for (const auto& msg : req.messages) {
                if (msg.contains("content")) {
                    // 检查 content 是否为数组（多模态）
                    if (msg["content"].is_array()) {
                        has_image = true;
                        std::cout << "  Found multimodal content" << std::endl;
                        // TODO: 处理图像数据
                    }
                }
            }
        }
        
        std::string response = "[Qwen-VL] I received your ";
        response += (has_image ? "image and text input." : "text input.");
        response += " This is a multimodal response.";
        
        if (provider->is_writable()) {
            provider->push(OutputChunk::FinalText(response, req.model));
            provider->end();
        }
    });
    
    // 注册 Embedding 模型
    server.registerEmbedding("text-embedding-qwen", [](const EmbeddingRequest& req, std::shared_ptr<BaseDataProvider> provider) {
        if (!provider->is_writable()) {
            return;
        }
        
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
        
        if (provider->is_writable()) {
            provider->push(OutputChunk::BatchEmbeddings(embeddings, req.model));
            provider->end();
        }
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
    std::cout << "Test multimodal (text only for demo):" << std::endl;
    std::cout << "  curl -X POST http://localhost:" << port << "/v1/chat/completions \\" << std::endl;
    std::cout << "    -H \"Content-Type: application/json\" \\" << std::endl;
    std::cout << "    -d '{\"model\":\"qwen-vl\",\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"Describe this\"},{\"type\":\"image_url\",\"image_url\":{\"url\":\"http://example.com/image.jpg\"}}]}]}'" << std::endl;
    std::cout << std::endl;
    
    // 配置并启动服务器
    ServerOptions options;
    options.port = port;
    options.max_concurrency = 10;
    
    server.run(options);
    return 0;
}
