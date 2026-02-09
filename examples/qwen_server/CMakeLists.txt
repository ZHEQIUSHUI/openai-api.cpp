add_executable(qwen_server main.cpp)
target_link_libraries(qwen_server PRIVATE openai_api::server)
