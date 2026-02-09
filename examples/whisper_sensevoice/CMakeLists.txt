add_executable(whisper_sensevoice main.cpp)
target_link_libraries(whisper_sensevoice PRIVATE openai_api::server)
