#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include "utils/json.hpp"

namespace openai_api {

/**
 * OutputChunk 类型枚举
 * 定义所有支持的语义输出事件类型
 */
enum class OutputChunkType {
    TextDelta,      // LLM/VLM 流式文本片段
    FinalText,      // LLM/VLM 最终完整文本
    Embedding,      // 单个向量
    Embeddings,     // 批量向量
    JsonObject,     // 通用 JSON 对象
    AudioBytes,     // TTS 音频字节
    ImageBytes,     // 生图图片字节
    Error,          // 错误信息
    End             // 流结束标记
};

/**
 * OutputChunk - 统一语义输出事件
 * 
 * 核心设计原则：
 * 1. 模型回调层只输出语义事件，不关心 HTTP/JSON/SSE 协议
 * 2. Encoder 层负责将所有事件编码为具体的协议格式
 */
struct OutputChunk {
    OutputChunkType type;
    
    // 文本相关字段
    std::string text;
    
    // Embedding 相关字段
    std::vector<float> embedding;
    std::vector<std::vector<float>> embeds;
    
    // 通用 JSON 对象
    nlohmann::json obj;
    
    // 二进制数据相关字段（音频/图片）
    std::vector<uint8_t> bytes;
    std::string mime_type;  // 如 "audio/mp3", "image/png"
    
    // 错误信息
    std::string error_message;
    std::string error_code;
    
    // 元数据（用于 Encoder 生成 OpenAI 格式）
    std::string model;
    std::string id;
    int64_t created;
    int index;  // 用于批量结果中的序号
    
    // 构造函数
    OutputChunk() : type(OutputChunkType::End), created(0), index(0) {}
    
    // 静态工厂方法：创建文本 delta
    static OutputChunk TextDelta(const std::string& delta, const std::string& model_id = "") {
        OutputChunk chunk;
        chunk.type = OutputChunkType::TextDelta;
        chunk.text = delta;
        chunk.model = model_id;
        chunk.created = std::time(nullptr);
        return chunk;
    }
    
    // 静态工厂方法：创建最终文本
    static OutputChunk FinalText(const std::string& content, const std::string& model_id = "") {
        OutputChunk chunk;
        chunk.type = OutputChunkType::FinalText;
        chunk.text = content;
        chunk.model = model_id;
        chunk.created = std::time(nullptr);
        return chunk;
    }
    
    // 静态工厂方法：创建单个 Embedding
    static OutputChunk SingleEmbedding(const std::vector<float>& emb, 
                                        const std::string& model_id = "",
                                        int idx = 0) {
        OutputChunk chunk;
        chunk.type = OutputChunkType::Embedding;
        chunk.embedding = emb;
        chunk.model = model_id;
        chunk.index = idx;
        chunk.created = std::time(nullptr);
        return chunk;
    }
    
    // 静态工厂方法：创建批量 Embeddings
    static OutputChunk BatchEmbeddings(const std::vector<std::vector<float>>& embs,
                                        const std::string& model_id = "") {
        OutputChunk chunk;
        chunk.type = OutputChunkType::Embeddings;
        chunk.embeds = embs;
        chunk.model = model_id;
        chunk.created = std::time(nullptr);
        return chunk;
    }
    
    // 静态工厂方法：创建音频字节
    static OutputChunk AudioData(const std::vector<uint8_t>& data,
                                  const std::string& mime,
                                  const std::string& model_id = "") {
        OutputChunk chunk;
        chunk.type = OutputChunkType::AudioBytes;
        chunk.bytes = data;
        chunk.mime_type = mime;
        chunk.model = model_id;
        chunk.created = std::time(nullptr);
        return chunk;
    }
    
    // 静态工厂方法：创建图片字节
    static OutputChunk ImageData(const std::vector<uint8_t>& data,
                                  const std::string& mime,
                                  const std::string& model_id = "") {
        OutputChunk chunk;
        chunk.type = OutputChunkType::ImageBytes;
        chunk.bytes = data;
        chunk.mime_type = mime;
        chunk.model = model_id;
        chunk.created = std::time(nullptr);
        return chunk;
    }
    
    // 静态工厂方法：创建 JSON 对象
    static OutputChunk Json(const nlohmann::json& json_obj, const std::string& model_id = "") {
        OutputChunk chunk;
        chunk.type = OutputChunkType::JsonObject;
        chunk.obj = json_obj;
        chunk.model = model_id;
        chunk.created = std::time(nullptr);
        return chunk;
    }
    
    // 静态工厂方法：创建错误
    static OutputChunk Error(const std::string& code, const std::string& message) {
        OutputChunk chunk;
        chunk.type = OutputChunkType::Error;
        chunk.error_code = code;
        chunk.error_message = message;
        chunk.created = std::time(nullptr);
        return chunk;
    }
    
    // 静态工厂方法：创建结束标记
    static OutputChunk EndMarker() {
        OutputChunk chunk;
        chunk.type = OutputChunkType::End;
        return chunk;
    }
    
    // 检查是否已结束
    bool is_end() const {
        return type == OutputChunkType::End;
    }
    
    // 检查是否是错误
    bool is_error() const {
        return type == OutputChunkType::Error;
    }
};

} // namespace openai_api
