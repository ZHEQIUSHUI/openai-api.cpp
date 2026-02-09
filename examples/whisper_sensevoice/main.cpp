/**
 * Whisper + SenseVoice Server Example
 * 
 * 展示如何使用 openai_api 库构建一个支持多个 ASR 算法的服务
 * 支持 Whisper 和 SenseVoice 两种 ASR 模型
 */

#include <openai_api/server.hpp>
#include <iostream>

using namespace openai_api;

int main(int argc, char* argv[]) {
    int port = 8080;
    if (argc > 1) {
        port = std::stoi(argv[1]);
    }
    
    // 创建服务器
    Server server(port);
    
    // 注册 Whisper-1 模型
    server.registerASR("whisper-1", [](const ASRRequest& req, std::shared_ptr<BaseDataProvider> provider) {
        std::cout << "[Whisper-1] Transcribing audio..." << std::endl;
        std::cout << "  Language: " << (req.language.empty() ? "auto" : req.language) << std::endl;
        std::cout << "  Audio size: " << req.raw_body.size() << " bytes" << std::endl;
        
        // 模拟 Whisper 转录
        std::string transcript;
        if (req.language == "zh" || req.language == "zh-CN") {
            transcript = "这是一段使用 Whisper 模型转录的中文语音。（模拟结果）";
        } else if (req.language == "en" || req.language.empty()) {
            transcript = "This is an English transcription using Whisper model. (Mock result)";
        } else {
            transcript = "Transcription in " + req.language + " using Whisper. (Mock result)";
        }
        
        provider->push(OutputChunk::FinalText(transcript, req.model));
        provider->end();
    });
    
    // 注册 Whisper-Large-V3 模型（更强大的版本）
    server.registerASR("whisper-large-v3", [](const ASRRequest& req, std::shared_ptr<BaseDataProvider> provider) {
        std::cout << "[Whisper-Large-V3] Transcribing with large model..." << std::endl;
        
        // 模拟更好的转录结果
        std::string transcript = "[Whisper-Large-V3] This is a high-quality transcription "
                                "with better accuracy and punctuation support. (Mock result)";
        
        provider->push(OutputChunk::FinalText(transcript, req.model));
        provider->end();
    });
    
    // 注册 SenseVoice 模型（阿里开源的中文 ASR）
    server.registerASR("sensevoice", [](const ASRRequest& req, std::shared_ptr<BaseDataProvider> provider) {
        std::cout << "[SenseVoice] Processing audio with Alibaba SenseVoice..." << std::endl;
        
        // 模拟 SenseVoice 转录（更好的中文支持）
        std::string transcript = "这是一段使用阿里 SenseVoice 模型转录的中文语音。";
        transcript += "SenseVoice 对中文语音有更好的识别效果。（模拟结果）";
        
        provider->push(OutputChunk::FinalText(transcript, req.model));
        provider->end();
    });
    
    // 注册 SenseVoice-Small（轻量版）
    server.registerASR("sensevoice-small", [](const ASRRequest& req, std::shared_ptr<BaseDataProvider> provider) {
        std::cout << "[SenseVoice-Small] Fast transcription..." << std::endl;
        
        std::string transcript = "[SenseVoice-Small] 快速中文语音识别结果。（模拟结果）";
        
        provider->push(OutputChunk::FinalText(transcript, req.model));
        provider->end();
    });
    
    // 同时注册一些 LLM 模型（作为额外功能）
    server.registerChat("gpt-4", [](const ChatRequest& req, std::shared_ptr<BaseDataProvider> provider) {
        std::string response = "This is GPT-4 responding to your ASR-related queries.";
        provider->push(OutputChunk::FinalText(response, req.model));
        provider->end();
    });
    
    std::cout << "========================================" << std::endl;
    std::cout << "Whisper + SenseVoice Server" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "ASR Models:" << std::endl;
    std::cout << "  - whisper-1           : OpenAI Whisper" << std::endl;
    std::cout << "  - whisper-large-v3    : OpenAI Whisper Large V3" << std::endl;
    std::cout << "  - sensevoice          : Alibaba SenseVoice" << std::endl;
    std::cout << "  - sensevoice-small    : Alibaba SenseVoice (lightweight)" << std::endl;
    std::cout << std::endl;
    std::cout << "Test commands:" << std::endl;
    std::cout << "  # List models" << std::endl;
    std::cout << "  curl http://localhost:" << port << "/v1/models" << std::endl;
    std::cout << std::endl;
    std::cout << "  # Transcribe with Whisper" << std::endl;
    std::cout << "  curl -X POST http://localhost:" << port << "/v1/audio/transcriptions \\" << std::endl;
    std::cout << "    -F \"model=whisper-1\" \\" << std::endl;
    std::cout << "    -F \"file=@audio.mp3\"" << std::endl;
    std::cout << std::endl;
    std::cout << "  # Transcribe with SenseVoice (better for Chinese)" << std::endl;
    std::cout << "  curl -X POST http://localhost:" << port << "/v1/audio/transcriptions \\" << std::endl;
    std::cout << "    -F \"model=sensevoice\" \\" << std::endl;
    std::cout << "    -F \"file=@audio.mp3\" \\" << std::endl;
    std::cout << "    -F \"language=zh\"" << std::endl;
    std::cout << std::endl;
    
    server.run();
    return 0;
}
