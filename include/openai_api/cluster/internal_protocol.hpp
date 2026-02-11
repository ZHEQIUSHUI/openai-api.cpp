#pragma once

#include "utils/json.hpp"
#include <cstdint>
#include <string>
#include <vector>
#include <cstring>
#include <chrono>

namespace openai_api {
namespace cluster {

/**
 * 内部集群通信协议
 * 用于 Master 和 Worker 进程之间的通信
 */

// 魔数，用于识别本项目服务
constexpr uint32_t CLUSTER_MAGIC = 0x4F414943; // "OAIC" (OpenAI Cluster)
constexpr uint32_t PROTOCOL_VERSION = 1;

// 消息类型
enum class MessageType : uint32_t {
    // 握手
    HANDSHAKE = 1,
    HANDSHAKE_ACK,
    
    // 模型注册
    REGISTER_MODEL,
    REGISTER_ACK,
    
    // 心跳
    HEARTBEAT,
    HEARTBEAT_ACK,
    
    // 请求转发
    FORWARD_REQUEST,
    FORWARD_RESPONSE,
    
    // 错误
    ERROR,
    
    // 断开连接
    DISCONNECT
};

// 模型类型
enum class ModelType : uint32_t {
    CHAT = 1,
    EMBEDDING,
    ASR,
    TTS,
    IMAGE_GEN
};

// 消息头
struct MessageHeader {
    uint32_t magic;           // 魔数
    uint32_t version;         // 协议版本
    uint32_t type;            // 消息类型
    uint32_t payload_length;  // 负载长度
    
    bool is_valid() const {
        return magic == CLUSTER_MAGIC && version == PROTOCOL_VERSION;
    }
};

// 序列化消息头
inline std::vector<uint8_t> serialize_header(const MessageHeader& header) {
    std::vector<uint8_t> data(sizeof(MessageHeader));
    memcpy(data.data(), &header, sizeof(MessageHeader));
    return data;
}

// 反序列化消息头
inline MessageHeader deserialize_header(const uint8_t* data) {
    MessageHeader header;
    memcpy(&header, data, sizeof(MessageHeader));
    return header;
}

// 构建完整消息
inline std::vector<uint8_t> build_message(MessageType type, const nlohmann::json& payload = {}) {
    std::string payload_str = payload.empty() ? "{}" : payload.dump();
    
    MessageHeader header;
    header.magic = CLUSTER_MAGIC;
    header.version = PROTOCOL_VERSION;
    header.type = static_cast<uint32_t>(type);
    header.payload_length = static_cast<uint32_t>(payload_str.size());
    
    std::vector<uint8_t> message;
    message.reserve(sizeof(MessageHeader) + payload_str.size());
    
    auto header_bytes = serialize_header(header);
    message.insert(message.end(), header_bytes.begin(), header_bytes.end());
    message.insert(message.end(), payload_str.begin(), payload_str.end());
    
    return message;
}

// 握手消息
inline nlohmann::json make_handshake(const std::string& worker_id) {
    return {
        {"worker_id", worker_id},
        {"timestamp", std::chrono::system_clock::now().time_since_epoch().count()}
    };
}

// 握手确认
inline nlohmann::json make_handshake_ack(bool accepted, const std::string& message = "") {
    return {
        {"accepted", accepted},
        {"message", message}
    };
}

// 注册模型消息
inline nlohmann::json make_register_model(ModelType type, const std::string& model_name) {
    return {
        {"model_type", static_cast<uint32_t>(type)},
        {"model_name", model_name}
    };
}

// 注册确认
inline nlohmann::json make_register_ack(bool success, const std::string& message = "") {
    return {
        {"success", success},
        {"message", message}
    };
}

// 转发请求消息
inline nlohmann::json make_forward_request(const std::string& request_id, 
                                            ModelType type,
                                            const nlohmann::json& request_data) {
    return {
        {"request_id", request_id},
        {"model_type", static_cast<uint32_t>(type)},
        {"request", request_data}
    };
}

// 转发响应消息
inline nlohmann::json make_forward_response(const std::string& request_id,
                                             const nlohmann::json& response_data,
                                             bool is_error = false) {
    return {
        {"request_id", request_id},
        {"response", response_data},
        {"is_error", is_error}
    };
}

// 错误消息
inline nlohmann::json make_error(const std::string& code, const std::string& message) {
    return {
        {"error_code", code},
        {"error_message", message}
    };
}

// 心跳
inline nlohmann::json make_heartbeat() {
    return {{"ping", true}};
}

inline nlohmann::json make_heartbeat_ack() {
    return {{"pong", true}};
}

} // namespace cluster
} // namespace openai_api
