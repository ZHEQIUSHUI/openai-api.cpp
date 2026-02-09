#include "openai_api/router.hpp"
#include <algorithm>

namespace openai_api {

// ============ 注册模型 ============

void ModelRouter::registerChat(const std::string& model_name, ChatCallback callback) {
    std::unique_lock lock(mutex_);
    chat_models_[model_name] = std::move(callback);
}

void ModelRouter::registerEmbedding(const std::string& model_name, EmbeddingCallback callback) {
    std::unique_lock lock(mutex_);
    embedding_models_[model_name] = std::move(callback);
}

void ModelRouter::registerASR(const std::string& model_name, ASRCallback callback) {
    std::unique_lock lock(mutex_);
    asr_models_[model_name] = std::move(callback);
}

void ModelRouter::registerTTS(const std::string& model_name, TTSCallback callback) {
    std::unique_lock lock(mutex_);
    tts_models_[model_name] = std::move(callback);
}

void ModelRouter::registerImageGeneration(const std::string& model_name, ImageGenCallback callback) {
    std::unique_lock lock(mutex_);
    image_gen_models_[model_name] = std::move(callback);
}

// ============ 路由调用 ============

bool ModelRouter::routeChat(const ChatRequest& req, std::shared_ptr<BaseDataProvider> provider) {
    std::shared_lock lock(mutex_);
    auto it = chat_models_.find(req.model);
    if (it == chat_models_.end()) {
        return false;
    }
    
    // 在线程中执行回调，避免阻塞 HTTP 线程
    std::thread([callback = it->second, req, provider]() {
        try {
            callback(req, provider);
        } catch (const std::exception& e) {
            provider->push(OutputChunk::Error("model_error", e.what()));
            provider->end();
        }
    }).detach();
    
    return true;
}

bool ModelRouter::routeEmbedding(const EmbeddingRequest& req, std::shared_ptr<BaseDataProvider> provider) {
    std::shared_lock lock(mutex_);
    auto it = embedding_models_.find(req.model);
    if (it == embedding_models_.end()) {
        return false;
    }
    
    std::thread([callback = it->second, req, provider]() {
        try {
            callback(req, provider);
        } catch (const std::exception& e) {
            provider->push(OutputChunk::Error("model_error", e.what()));
            provider->end();
        }
    }).detach();
    
    return true;
}

bool ModelRouter::routeASR(const ASRRequest& req, std::shared_ptr<BaseDataProvider> provider) {
    std::shared_lock lock(mutex_);
    auto it = asr_models_.find(req.model);
    if (it == asr_models_.end()) {
        return false;
    }
    
    std::thread([callback = it->second, req, provider]() {
        try {
            callback(req, provider);
        } catch (const std::exception& e) {
            provider->push(OutputChunk::Error("model_error", e.what()));
            provider->end();
        }
    }).detach();
    
    return true;
}

bool ModelRouter::routeTTS(const TTSRequest& req, std::shared_ptr<BaseDataProvider> provider) {
    std::shared_lock lock(mutex_);
    auto it = tts_models_.find(req.model);
    if (it == tts_models_.end()) {
        return false;
    }
    
    std::thread([callback = it->second, req, provider]() {
        try {
            callback(req, provider);
        } catch (const std::exception& e) {
            provider->push(OutputChunk::Error("model_error", e.what()));
            provider->end();
        }
    }).detach();
    
    return true;
}

bool ModelRouter::routeImageGeneration(const ImageGenRequest& req, std::shared_ptr<BaseDataProvider> provider) {
    std::shared_lock lock(mutex_);
    auto it = image_gen_models_.find(req.model);
    if (it == image_gen_models_.end()) {
        return false;
    }
    
    std::thread([callback = it->second, req, provider]() {
        try {
            callback(req, provider);
        } catch (const std::exception& e) {
            provider->push(OutputChunk::Error("model_error", e.what()));
            provider->end();
        }
    }).detach();
    
    return true;
}

// ============ 检查模型是否存在 ============

bool ModelRouter::hasChatModel(const std::string& model_name) const {
    std::shared_lock lock(mutex_);
    return chat_models_.find(model_name) != chat_models_.end();
}

bool ModelRouter::hasEmbeddingModel(const std::string& model_name) const {
    std::shared_lock lock(mutex_);
    return embedding_models_.find(model_name) != embedding_models_.end();
}

bool ModelRouter::hasASRModel(const std::string& model_name) const {
    std::shared_lock lock(mutex_);
    return asr_models_.find(model_name) != asr_models_.end();
}

bool ModelRouter::hasTTSModel(const std::string& model_name) const {
    std::shared_lock lock(mutex_);
    return tts_models_.find(model_name) != tts_models_.end();
}

bool ModelRouter::hasImageGenModel(const std::string& model_name) const {
    std::shared_lock lock(mutex_);
    return image_gen_models_.find(model_name) != image_gen_models_.end();
}

// ============ 获取模型列表 ============

std::vector<std::string> ModelRouter::listChatModels() const {
    std::shared_lock lock(mutex_);
    std::vector<std::string> result;
    for (const auto& [name, _] : chat_models_) {
        result.push_back(name);
    }
    return result;
}

std::vector<std::string> ModelRouter::listEmbeddingModels() const {
    std::shared_lock lock(mutex_);
    std::vector<std::string> result;
    for (const auto& [name, _] : embedding_models_) {
        result.push_back(name);
    }
    return result;
}

std::vector<std::string> ModelRouter::listASRModels() const {
    std::shared_lock lock(mutex_);
    std::vector<std::string> result;
    for (const auto& [name, _] : asr_models_) {
        result.push_back(name);
    }
    return result;
}

std::vector<std::string> ModelRouter::listTTSModels() const {
    std::shared_lock lock(mutex_);
    std::vector<std::string> result;
    for (const auto& [name, _] : tts_models_) {
        result.push_back(name);
    }
    return result;
}

std::vector<std::string> ModelRouter::listImageGenModels() const {
    std::shared_lock lock(mutex_);
    std::vector<std::string> result;
    for (const auto& [name, _] : image_gen_models_) {
        result.push_back(name);
    }
    return result;
}

std::vector<std::string> ModelRouter::listAllModels() const {
    std::shared_lock lock(mutex_);
    std::vector<std::string> result;
    
    for (const auto& [name, _] : chat_models_) result.push_back(name);
    for (const auto& [name, _] : embedding_models_) result.push_back(name);
    for (const auto& [name, _] : asr_models_) result.push_back(name);
    for (const auto& [name, _] : tts_models_) result.push_back(name);
    for (const auto& [name, _] : image_gen_models_) result.push_back(name);
    
    // 去重
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    
    return result;
}

// ============ 卸载模型 ============

void ModelRouter::unregisterChat(const std::string& model_name) {
    std::unique_lock lock(mutex_);
    chat_models_.erase(model_name);
}

void ModelRouter::unregisterEmbedding(const std::string& model_name) {
    std::unique_lock lock(mutex_);
    embedding_models_.erase(model_name);
}

void ModelRouter::unregisterASR(const std::string& model_name) {
    std::unique_lock lock(mutex_);
    asr_models_.erase(model_name);
}

void ModelRouter::unregisterTTS(const std::string& model_name) {
    std::unique_lock lock(mutex_);
    tts_models_.erase(model_name);
}

void ModelRouter::unregisterImageGeneration(const std::string& model_name) {
    std::unique_lock lock(mutex_);
    image_gen_models_.erase(model_name);
}

} // namespace openai_api
