#pragma once

#include "types.hpp"
#include "openai_api/core/data_provider.hpp"
#include <functional>
#include <unordered_map>
#include <mutex>
#include <shared_mutex>

namespace openai_api {

// 回调函数类型定义
using ChatCallback = std::function<void(const ChatRequest&, std::shared_ptr<BaseDataProvider>)>;
using EmbeddingCallback = std::function<void(const EmbeddingRequest&, std::shared_ptr<BaseDataProvider>)>;
using ASRCallback = std::function<void(const ASRRequest&, std::shared_ptr<BaseDataProvider>)>;
using TTSCallback = std::function<void(const TTSRequest&, std::shared_ptr<BaseDataProvider>)>;
using ImageGenCallback = std::function<void(const ImageGenRequest&, std::shared_ptr<BaseDataProvider>)>;

/**
 * ModelRouter - 模型路由管理器
 * 
 * 管理不同类型（Chat/ASR/TTS等）的多个模型实现
 * 根据请求中的 model 字段路由到对应的回调函数
 */
class ModelRouter {
public:
    ModelRouter() = default;
    ~ModelRouter() = default;
    
    // 注册模型回调
    void registerChat(const std::string& model_name, ChatCallback callback);
    void registerEmbedding(const std::string& model_name, EmbeddingCallback callback);
    void registerASR(const std::string& model_name, ASRCallback callback);
    void registerTTS(const std::string& model_name, TTSCallback callback);
    void registerImageGeneration(const std::string& model_name, ImageGenCallback callback);
    
    // 路由调用
    bool routeChat(const ChatRequest& req, std::shared_ptr<BaseDataProvider> provider);
    bool routeEmbedding(const EmbeddingRequest& req, std::shared_ptr<BaseDataProvider> provider);
    bool routeASR(const ASRRequest& req, std::shared_ptr<BaseDataProvider> provider);
    bool routeTTS(const TTSRequest& req, std::shared_ptr<BaseDataProvider> provider);
    bool routeImageGeneration(const ImageGenRequest& req, std::shared_ptr<BaseDataProvider> provider);
    
    // 检查模型是否存在
    bool hasChatModel(const std::string& model_name) const;
    bool hasEmbeddingModel(const std::string& model_name) const;
    bool hasASRModel(const std::string& model_name) const;
    bool hasTTSModel(const std::string& model_name) const;
    bool hasImageGenModel(const std::string& model_name) const;
    
    // 获取模型列表
    std::vector<std::string> listChatModels() const;
    std::vector<std::string> listEmbeddingModels() const;
    std::vector<std::string> listASRModels() const;
    std::vector<std::string> listTTSModels() const;
    std::vector<std::string> listImageGenModels() const;
    
    // 获取所有模型（用于 /v1/models 接口）
    std::vector<std::string> listAllModels() const;
    
    // 卸载模型
    void unregisterChat(const std::string& model_name);
    void unregisterEmbedding(const std::string& model_name);
    void unregisterASR(const std::string& model_name);
    void unregisterTTS(const std::string& model_name);
    void unregisterImageGeneration(const std::string& model_name);

private:
    mutable std::shared_mutex mutex_;
    
    std::unordered_map<std::string, ChatCallback> chat_models_;
    std::unordered_map<std::string, EmbeddingCallback> embedding_models_;
    std::unordered_map<std::string, ASRCallback> asr_models_;
    std::unordered_map<std::string, TTSCallback> tts_models_;
    std::unordered_map<std::string, ImageGenCallback> image_gen_models_;
};

} // namespace openai_api
