#include "openai_api/cluster_server.hpp"
#include <iostream>

#ifdef _WIN32
#include <winsock2.h>
#pragma comment(lib, "ws2_32.lib")
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#endif

namespace openai_api {

ClusterServer::ClusterServer() = default;

ClusterServer::ClusterServer(int port) {
    options_.server.port = port;
}

ClusterServer::ClusterServer(const ClusterServerOptions& options) 
    : options_(options) 
{}

ClusterServer::~ClusterServer() {
    stop();
}

void ClusterServer::setMaxConcurrency(int max) {
    options_.server.max_concurrency = max;
    if (server_) {
        server_->setMaxConcurrency(max);
    }
}

void ClusterServer::setTimeout(std::chrono::milliseconds timeout) {
    options_.server.default_timeout = timeout;
    if (server_) {
        server_->setTimeout(timeout);
    }
}

void ClusterServer::setApiKey(const std::string& api_key) {
    options_.server.api_key = api_key;
    if (server_) {
        server_->setApiKey(api_key);
    }
}

void ClusterServer::setWorkerListenAddress(const std::string& host, int port) {
    worker_listen_host_ = host;
    worker_listen_port_ = port;
    if (worker_client_) {
        worker_client_->set_listen_address(host, port);
    }
}

void ClusterServer::registerChat(const std::string& model_name, ChatCallback callback) {
    std::lock_guard<std::mutex> lock(models_mutex_);
    
    if (mode_ == ClusterMode::WORKER && worker_client_) {
        // Worker 模式：先暂存，连接后注册
        LocalModel model;
        model.type = cluster::ModelType::CHAT;
        model.name = model_name;
        model.callback = [callback](const nlohmann::json& req, std::shared_ptr<BaseDataProvider> provider) {
            ChatRequest chat_req = ChatRequest::from_json(req);
            callback(chat_req, provider);
        };
        local_models_.push_back(model);
    } else if ((mode_ == ClusterMode::MASTER || mode_ == ClusterMode::STANDALONE) && server_) {
        // Master/Standalone 运行中：直接注册到本地 Server
        server_->registerChat(model_name, callback);
    } else {
        // STANDALONE 模式或尚未确定模式：暂存
        LocalModel model;
        model.type = cluster::ModelType::CHAT;
        model.name = model_name;
        model.callback = [callback](const nlohmann::json& req, std::shared_ptr<BaseDataProvider> provider) {
            ChatRequest chat_req = ChatRequest::from_json(req);
            callback(chat_req, provider);
        };
        local_models_.push_back(model);
    }
}

void ClusterServer::registerEmbedding(const std::string& model_name, EmbeddingCallback callback) {
    std::lock_guard<std::mutex> lock(models_mutex_);
    
    if (mode_ == ClusterMode::WORKER && worker_client_) {
        LocalModel model;
        model.type = cluster::ModelType::EMBEDDING;
        model.name = model_name;
        model.callback = [callback](const nlohmann::json& req, std::shared_ptr<BaseDataProvider> provider) {
            EmbeddingRequest emb_req = EmbeddingRequest::from_json(req);
            callback(emb_req, provider);
        };
        local_models_.push_back(model);
    } else if ((mode_ == ClusterMode::MASTER || mode_ == ClusterMode::STANDALONE) && server_) {
        server_->registerEmbedding(model_name, callback);
    } else {
        LocalModel model;
        model.type = cluster::ModelType::EMBEDDING;
        model.name = model_name;
        model.callback = [callback](const nlohmann::json& req, std::shared_ptr<BaseDataProvider> provider) {
            EmbeddingRequest emb_req = EmbeddingRequest::from_json(req);
            callback(emb_req, provider);
        };
        local_models_.push_back(model);
    }
}

void ClusterServer::registerASR(const std::string& model_name, ASRCallback callback) {
    std::lock_guard<std::mutex> lock(models_mutex_);
    
    if (mode_ == ClusterMode::WORKER && worker_client_) {
        LocalModel model;
        model.type = cluster::ModelType::ASR;
        model.name = model_name;
        model.callback = [callback](const nlohmann::json& req, std::shared_ptr<BaseDataProvider> provider) {
            // ASRRequest 没有 from_json，直接使用原始请求
            ASRRequest asr_req;
            asr_req.model = req.value("model", "");
            asr_req.language = req.value("language", "");
            asr_req.prompt = req.value("prompt", "");
            asr_req.response_format = req.value("response_format", "json");
            asr_req.temperature = req.value("temperature", 0.0f);
            callback(asr_req, provider);
        };
        local_models_.push_back(model);
    } else if ((mode_ == ClusterMode::MASTER || mode_ == ClusterMode::STANDALONE) && server_) {
        server_->registerASR(model_name, callback);
    } else {
        LocalModel model;
        model.type = cluster::ModelType::ASR;
        model.name = model_name;
        model.callback = [callback](const nlohmann::json& req, std::shared_ptr<BaseDataProvider> provider) {
            ASRRequest asr_req;
            asr_req.model = req.value("model", "");
            asr_req.language = req.value("language", "");
            asr_req.prompt = req.value("prompt", "");
            asr_req.response_format = req.value("response_format", "json");
            asr_req.temperature = req.value("temperature", 0.0f);
            callback(asr_req, provider);
        };
        local_models_.push_back(model);
    }
}

void ClusterServer::registerTTS(const std::string& model_name, TTSCallback callback) {
    std::lock_guard<std::mutex> lock(models_mutex_);
    
    if (mode_ == ClusterMode::WORKER && worker_client_) {
        LocalModel model;
        model.type = cluster::ModelType::TTS;
        model.name = model_name;
        model.callback = [callback](const nlohmann::json& req, std::shared_ptr<BaseDataProvider> provider) {
            TTSRequest tts_req = TTSRequest::from_json(req);
            callback(tts_req, provider);
        };
        local_models_.push_back(model);
    } else if ((mode_ == ClusterMode::MASTER || mode_ == ClusterMode::STANDALONE) && server_) {
        server_->registerTTS(model_name, callback);
    } else {
        LocalModel model;
        model.type = cluster::ModelType::TTS;
        model.name = model_name;
        model.callback = [callback](const nlohmann::json& req, std::shared_ptr<BaseDataProvider> provider) {
            TTSRequest tts_req = TTSRequest::from_json(req);
            callback(tts_req, provider);
        };
        local_models_.push_back(model);
    }
}

void ClusterServer::registerImageGeneration(const std::string& model_name, ImageGenCallback callback) {
    std::lock_guard<std::mutex> lock(models_mutex_);
    
    if (mode_ == ClusterMode::WORKER && worker_client_) {
        LocalModel model;
        model.type = cluster::ModelType::IMAGE_GEN;
        model.name = model_name;
        model.callback = [callback](const nlohmann::json& req, std::shared_ptr<BaseDataProvider> provider) {
            ImageGenRequest img_req = ImageGenRequest::from_json(req);
            callback(img_req, provider);
        };
        local_models_.push_back(model);
    } else if ((mode_ == ClusterMode::MASTER || mode_ == ClusterMode::STANDALONE) && server_) {
        server_->registerImageGeneration(model_name, callback);
    } else {
        LocalModel model;
        model.type = cluster::ModelType::IMAGE_GEN;
        model.name = model_name;
        model.callback = [callback](const nlohmann::json& req, std::shared_ptr<BaseDataProvider> provider) {
            ImageGenRequest img_req = ImageGenRequest::from_json(req);
            callback(img_req, provider);
        };
        local_models_.push_back(model);
    }
}

std::vector<std::string> ClusterServer::listModels() const {
    if ((mode_ == ClusterMode::MASTER || mode_ == ClusterMode::STANDALONE) && server_) {
        return server_->listModels();
    }
    return {};
}

bool ClusterServer::hasModel(const std::string& model_name) const {
    if ((mode_ == ClusterMode::MASTER || mode_ == ClusterMode::STANDALONE) && server_) {
        return server_->hasModel(model_name);
    }
    return false;
}

void ClusterServer::unregisterModel(const std::string& model_name) {
    if ((mode_ == ClusterMode::MASTER || mode_ == ClusterMode::STANDALONE) && server_) {
        server_->unregisterModel(model_name);
    }
}

ClusterMode ClusterServer::run(int port) {
    if (!options_.enable_cluster) {
        runAsStandalone(port);
        return mode_;
    }

    // 尝试自动检测模式
    // 1. 先尝试作为 Master 启动
    // 2. 如果端口被占用，检测是否为集群服务
    // 3. 如果是集群服务，作为 Worker 连接
    // 4. 如果不是集群服务，报错
    
    if (tryStartMaster(port)) {
        return ClusterMode::MASTER;
    }
    
    // 端口被占用，检测是否为集群服务
    if (detectClusterService("127.0.0.1", port + 1000)) {
        // 是集群服务，作为 Worker 连接
        if (tryStartWorker("127.0.0.1", port)) {
            return ClusterMode::WORKER;
        }
    }
    
    // 无法启动
    std::cerr << "Failed to start cluster mode on port " << port << std::endl;
    std::cerr << "Port is occupied and not a cluster service" << std::endl;
    running_ = false;
    mode_ = ClusterMode::STANDALONE;
    return mode_;
}

ClusterMode ClusterServer::run(const ClusterServerOptions& options) {
    options_ = options;
    return run(options.server.port);
}

bool ClusterServer::runAsMaster(int port) {
    mode_ = ClusterMode::MASTER;
    
    // 创建 Server
    server_ = std::make_unique<Server>();
    server_->setMaxConcurrency(options_.server.max_concurrency);
    server_->setTimeout(options_.server.default_timeout);
    if (!options_.server.api_key.empty()) {
        server_->setApiKey(options_.server.api_key);
    }
    
    registerLocalModelsToServer();
    
    // 启动 WorkerManager（内部端口 = 外部端口 + 1000）
    if (options_.enable_cluster) {
        worker_manager_ = std::make_unique<cluster::WorkerManager>();
        
        // 先设置回调，再启动 WorkerManager，避免错过早期注册
        worker_manager_->set_model_registered_callback(
            [this](const std::string& model_name, cluster::ModelType type) {
                // 将 Worker 模型注册到 Server 的 Router
                // 根据类型注册到 Server
                switch (type) {
                    case cluster::ModelType::CHAT: {
                        auto forward_callback = [this, model_name](const ChatRequest& req, std::shared_ptr<BaseDataProvider> provider) {
                            worker_manager_->forward_request(model_name, cluster::ModelType::CHAT, req.raw, provider);
                        };
                        server_->registerChat(model_name, forward_callback);
                        break;
                    }
                    case cluster::ModelType::EMBEDDING: {
                        auto forward_callback = [this, model_name](const EmbeddingRequest& req, std::shared_ptr<BaseDataProvider> provider) {
                            worker_manager_->forward_request(model_name, cluster::ModelType::EMBEDDING, req.raw, provider);
                        };
                        server_->registerEmbedding(model_name, forward_callback);
                        break;
                    }
                    case cluster::ModelType::ASR: {
                        auto forward_callback = [this, model_name](const ASRRequest& req, std::shared_ptr<BaseDataProvider> provider) {
                            nlohmann::json req_json;
                            req_json["model"] = req.model;
                            req_json["language"] = req.language;
                            req_json["prompt"] = req.prompt;
                            req_json["response_format"] = req.response_format;
                            req_json["temperature"] = req.temperature;
                            worker_manager_->forward_request(model_name, cluster::ModelType::ASR, req_json, provider);
                        };
                        server_->registerASR(model_name, forward_callback);
                        break;
                    }
                    case cluster::ModelType::TTS: {
                        auto forward_callback = [this, model_name](const TTSRequest& req, std::shared_ptr<BaseDataProvider> provider) {
                            worker_manager_->forward_request(model_name, cluster::ModelType::TTS, req.raw, provider);
                        };
                        server_->registerTTS(model_name, forward_callback);
                        break;
                    }
                    case cluster::ModelType::IMAGE_GEN: {
                        auto forward_callback = [this, model_name](const ImageGenRequest& req, std::shared_ptr<BaseDataProvider> provider) {
                            worker_manager_->forward_request(model_name, cluster::ModelType::IMAGE_GEN, req.raw, provider);
                        };
                        server_->registerImageGeneration(model_name, forward_callback);
                        break;
                    }
                }
                std::cout << "Worker model registered: " << model_name << std::endl;
            }
        );
        
        worker_manager_->set_model_unregistered_callback(
            [this](const std::string& model_name) {
                server_->unregisterModel(model_name);
                std::cout << "Worker model unregistered: " << model_name << std::endl;
            }
        );
        
        // 启动 WorkerManager
        int internal_port = port + 1000;
        if (!worker_manager_->start(internal_port)) {
            std::cerr << "Failed to start WorkerManager on port " << internal_port << std::endl;
            return false;
        }
        std::cout << "WorkerManager listening on port " << internal_port << std::endl;
    }
    
    running_ = true;
    
    // 启动 Server（阻塞）
    server_->run(port);
    
    return true;
}

bool ClusterServer::runAsStandalone(int port) {
    mode_ = ClusterMode::STANDALONE;

    server_ = std::make_unique<Server>();
    server_->setMaxConcurrency(options_.server.max_concurrency);
    server_->setTimeout(options_.server.default_timeout);
    if (!options_.server.api_key.empty()) {
        server_->setApiKey(options_.server.api_key);
    }

    registerLocalModelsToServer();

    running_ = true;
    server_->run(port);
    return true;
}

bool ClusterServer::runAsWorker(const std::string& master_host, int master_port) {
    mode_ = ClusterMode::WORKER;
    
    // 创建 WorkerClient
    worker_client_ = std::make_unique<cluster::WorkerClient>();
    
    // 设置 Worker 监听地址
    worker_client_->set_listen_address(worker_listen_host_, worker_listen_port_);
    
    // 设置 Router 用于处理转发的请求
    auto router = std::make_unique<ModelRouter>();
    
    // 注册本地模型到 Router（复制一份，因为后面还要注册到 Master）
    std::vector<LocalModel> models_copy;
    {
        std::lock_guard<std::mutex> lock(models_mutex_);
        models_copy = local_models_;
    }
    
    for (const auto& model : models_copy) {
        switch (model.type) {
            case cluster::ModelType::CHAT:
                router->registerChat(model.name, [model](const ChatRequest& req, std::shared_ptr<BaseDataProvider> provider) {
                    model.callback(req.raw, provider);
                });
                break;
            case cluster::ModelType::EMBEDDING:
                router->registerEmbedding(model.name, [model](const EmbeddingRequest& req, std::shared_ptr<BaseDataProvider> provider) {
                    model.callback(req.raw, provider);
                });
                break;
            case cluster::ModelType::ASR:
                router->registerASR(model.name, [model](const ASRRequest& req, std::shared_ptr<BaseDataProvider> provider) {
                    // ASRRequest 没有 raw 字段，构造一个
                    nlohmann::json req_json;
                    req_json["model"] = req.model;
                    req_json["language"] = req.language;
                    req_json["prompt"] = req.prompt;
                    req_json["response_format"] = req.response_format;
                    req_json["temperature"] = req.temperature;
                    model.callback(req_json, provider);
                });
                break;
            case cluster::ModelType::TTS:
                router->registerTTS(model.name, [model](const TTSRequest& req, std::shared_ptr<BaseDataProvider> provider) {
                    model.callback(req.raw, provider);
                });
                break;
            case cluster::ModelType::IMAGE_GEN:
                router->registerImageGeneration(model.name, [model](const ImageGenRequest& req, std::shared_ptr<BaseDataProvider> provider) {
                    model.callback(req.raw, provider);
                });
                break;
        }
    }
    
    worker_client_->set_router(router.get());
    
    // 连接到 Master
    if (!worker_client_->connect(master_host, master_port)) {
        std::cerr << "Failed to connect to Master at " << master_host << ":" << master_port << std::endl;
        return false;
    }
    
    // 注册所有模型到 Master
    {
        std::lock_guard<std::mutex> lock(models_mutex_);
        for (const auto& model : models_copy) {
            if (!worker_client_->register_model(model.type, model.name)) {
                std::cerr << "Failed to register model: " << model.name << std::endl;
            }
        }
        local_models_.clear();  // 清空原始列表
    }
    
    running_ = true;
    
    // 保持运行直到断开
    while (worker_client_->is_connected() && running_) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    
    return true;
}

void ClusterServer::stop() {
    running_ = false;
    
    if (server_) {
        server_->stop();
    }
    
    if (worker_manager_) {
        worker_manager_->stop();
    }
    
    if (worker_client_) {
        worker_client_->disconnect();
    }
}

bool ClusterServer::isRunning() const {
    return running_.load();
}

bool ClusterServer::tryStartMaster(int port) {
    // 检查端口是否可用
#ifdef _WIN32
    SOCKET sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock == INVALID_SOCKET) {
        return false;
    }
    
    sockaddr_in addr;
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(port);
    
    bool available = (bind(sock, (sockaddr*)&addr, sizeof(addr)) == 0);
    closesocket(sock);
#else
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) return false;
    
    struct sockaddr_in addr;
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(port);
    
    bool available = (bind(sock, (struct sockaddr*)&addr, sizeof(addr)) == 0);
    close(sock);
#endif
    
    if (!available) {
        return false;  // 端口被占用
    }
    
    // 端口可用，启动 Master（异步）
    std::thread([this, port]() {
        runAsMaster(port);
    }).detach();
    
    // 等待启动完成
    for (int i = 0; i < 50; ++i) {
        if (running_ && mode_ == ClusterMode::MASTER) {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    
    return false;
}

bool ClusterServer::tryStartWorker(const std::string& host, int port) {
    // 计算 Master 内部端口
    int master_internal_port = port + 1000;
    
    // 尝试连接
    if (runAsWorker(host, master_internal_port)) {
        return true;
    }
    
    return false;
}

bool ClusterServer::detectClusterService(const std::string& host, int port) {
    // 使用 WorkerClient 的检测函数
    return cluster::check_is_cluster_server(host, port);
}

void ClusterServer::setupWorkerHandler() {
    // 设置 Worker 请求处理器
    // 这个方法在 Worker 模式下用于处理 Master 转发的请求
    // 实际处理逻辑已经在 runAsWorker 中通过 Router 设置
}

void ClusterServer::registerLocalModelsToMaster() {
    if (!worker_client_ || !worker_client_->is_connected()) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(models_mutex_);
    for (const auto& model : local_models_) {
        worker_client_->register_model(model.type, model.name);
    }
    local_models_.clear();
}

void ClusterServer::registerLocalModelsToServer() {
    if (!server_) return;

    std::lock_guard<std::mutex> lock(models_mutex_);
    for (const auto& model : local_models_) {
        switch (model.type) {
            case cluster::ModelType::CHAT: {
                auto callback = [model](const ChatRequest& req, std::shared_ptr<BaseDataProvider> provider) {
                    model.callback(req.raw, provider);
                };
                server_->registerChat(model.name, callback);
                break;
            }
            case cluster::ModelType::EMBEDDING: {
                auto callback = [model](const EmbeddingRequest& req, std::shared_ptr<BaseDataProvider> provider) {
                    model.callback(req.raw, provider);
                };
                server_->registerEmbedding(model.name, callback);
                break;
            }
            case cluster::ModelType::ASR: {
                auto callback = [model](const ASRRequest& req, std::shared_ptr<BaseDataProvider> provider) {
                    nlohmann::json req_json;
                    req_json["model"] = req.model;
                    req_json["language"] = req.language;
                    req_json["prompt"] = req.prompt;
                    req_json["response_format"] = req.response_format;
                    req_json["temperature"] = req.temperature;
                    model.callback(req_json, provider);
                };
                server_->registerASR(model.name, callback);
                break;
            }
            case cluster::ModelType::TTS: {
                auto callback = [model](const TTSRequest& req, std::shared_ptr<BaseDataProvider> provider) {
                    model.callback(req.raw, provider);
                };
                server_->registerTTS(model.name, callback);
                break;
            }
            case cluster::ModelType::IMAGE_GEN: {
                auto callback = [model](const ImageGenRequest& req, std::shared_ptr<BaseDataProvider> provider) {
                    model.callback(req.raw, provider);
                };
                server_->registerImageGeneration(model.name, callback);
                break;
            }
        }
    }
    local_models_.clear();
}

} // namespace openai_api
