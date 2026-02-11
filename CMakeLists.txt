cmake_minimum_required(VERSION 3.14)
project(OpenAIAPIServer VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 选项（使用项目前缀避免与其他项目冲突）
option(OPENAI_API_BUILD_EXAMPLES "Build openai-api.cpp examples" ON)
option(OPENAI_API_BUILD_TESTS "Build openai-api.cpp tests" ON)
option(OPENAI_API_BUILD_SHARED "Build openai-api.cpp as shared library" OFF)

# 包含目录
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# 查找线程库
find_package(Threads REQUIRED)

# ============ 核心库 ============
if(OPENAI_API_BUILD_SHARED)
    add_library(openai_api SHARED
        src/server.cpp
        src/router.cpp
        src/cluster/remote_worker_provider.cpp
        src/cluster/worker_manager.cpp
        src/cluster/worker_client.cpp
        src/cluster_server.cpp
    )
    # 定义导出宏（构建 DLL 时使用 dllexport）
    target_compile_definitions(openai_api PRIVATE OPENAI_API_EXPORTS)
    
    # Windows 上需要导出所有符号
    if(WIN32)
        set_target_properties(openai_api PROPERTIES
            WINDOWS_EXPORT_ALL_SYMBOLS ON
        )
    endif()
else()
    add_library(openai_api STATIC
        src/server.cpp
        src/router.cpp
        src/cluster/remote_worker_provider.cpp
        src/cluster/worker_manager.cpp
        src/cluster/worker_client.cpp
        src/cluster_server.cpp
    )
    # 静态库需要定义 OPENAI_API_STATIC
    target_compile_definitions(openai_api PUBLIC OPENAI_API_STATIC)
endif()

target_include_directories(openai_api PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(openai_api PUBLIC
    Threads::Threads
)

# Windows 需要链接 ws2_32 (Winsock2) 和 iphlpapi (IP Helper API)
if(WIN32)
    target_link_libraries(openai_api PUBLIC ws2_32 iphlpapi)
endif()

# 别名目标（方便 FetchContent 使用）
add_library(openai_api::server ALIAS openai_api)

# ============ 可执行文件 ============
add_executable(openai_api_server src/main.cpp)
target_link_libraries(openai_api_server PRIVATE openai_api)

# ============ 测试 ============
if(OPENAI_API_BUILD_TESTS)
    enable_testing()
    
    add_executable(openai_api_test_provider tests/test_provider.cpp)
    target_link_libraries(openai_api_test_provider PRIVATE openai_api)
    add_test(NAME openai_api_provider_test COMMAND openai_api_test_provider)
    
    add_executable(openai_api_test_encoder tests/test_encoder.cpp)
    target_link_libraries(openai_api_test_encoder PRIVATE openai_api)
    add_test(NAME openai_api_encoder_test COMMAND openai_api_test_encoder)
    
    add_executable(openai_api_test_integration tests/test_integration.cpp)
    target_link_libraries(openai_api_test_integration PRIVATE openai_api)
    add_test(NAME openai_api_integration_test COMMAND openai_api_test_integration)
    
    add_executable(openai_api_test_cluster tests/test_cluster.cpp)
    target_link_libraries(openai_api_test_cluster PRIVATE openai_api)
    add_test(NAME openai_api_cluster_test COMMAND openai_api_test_cluster)
    
    add_executable(openai_api_test_cluster_full tests/test_cluster_full.cpp)
    target_link_libraries(openai_api_test_cluster_full PRIVATE openai_api)
    add_test(NAME openai_api_cluster_full_test COMMAND openai_api_test_cluster_full)
endif()

# ============ 示例 ============
if(OPENAI_API_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# ============ 安装配置 ============
include(GNUInstallDirs)

# 安装库
install(TARGETS openai_api openai_api_server
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
message(STATUS "  Build examples: ${OPENAI_API_BUILD_EXAMPLES}")
message(STATUS "  Build tests: ${OPENAI_API_BUILD_TESTS}")
message(STATUS "  Install prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "")
