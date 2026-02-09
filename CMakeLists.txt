cmake_minimum_required(VERSION 3.14)
project(OpenAIAPIServer VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 选项
option(BUILD_EXAMPLES "Build examples" ON)
option(BUILD_TESTS "Build tests" ON)

# 包含目录
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# 查找线程库
find_package(Threads REQUIRED)

# ============ 核心库 ============
add_library(openai_api STATIC
    src/server.cpp
    src/router.cpp
)

target_include_directories(openai_api PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(openai_api PUBLIC
    Threads::Threads
)

# 别名目标（方便 FetchContent 使用）
add_library(openai_api::server ALIAS openai_api)

# ============ 可执行文件 ============
add_executable(openai_server src/main.cpp)
target_link_libraries(openai_server PRIVATE openai_api)

# ============ 测试 ============
if(BUILD_TESTS)
    enable_testing()
    
    add_executable(test_provider tests/test_provider.cpp)
    target_link_libraries(test_provider PRIVATE openai_api)
    add_test(NAME provider_test COMMAND test_provider)
    
    add_executable(test_encoder tests/test_encoder.cpp)
    target_link_libraries(test_encoder PRIVATE openai_api)
    add_test(NAME encoder_test COMMAND test_encoder)
    
    add_executable(test_integration tests/test_integration.cpp)
    target_link_libraries(test_integration PRIVATE openai_api)
    add_test(NAME integration_test COMMAND test_integration)
endif()

# ============ 示例 ============
if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# ============ 安装配置 ============
include(GNUInstallDirs)

# 安装库
install(TARGETS openai_api openai_server
    EXPORT openai_apiTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# 安装头文件
install(DIRECTORY include/openai_api
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# 导出配置
install(EXPORT openai_apiTargets
    FILE openai_apiTargets.cmake
    NAMESPACE openai_api::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/openai_api
)

# 生成配置文件
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/openai_apiConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

configure_file(cmake/openai_apiConfig.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/openai_apiConfig.cmake"
    @ONLY
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/openai_apiConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/openai_apiConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/openai_api
)

# ============ 打印配置 ============
message(STATUS "")
message(STATUS "OpenAI API Server Library Configuration:")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  Build examples: ${BUILD_EXAMPLES}")
message(STATUS "  Build tests: ${BUILD_TESTS}")
message(STATUS "  Install prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "")
