# API 导出宏使用说明

## 概述

本项目使用 `OPENAI_API_API` 宏来支持跨平台的动态库导出/导入。

## 宏定义

```cpp
#include "openai_api/core/api_export.hpp"

// 类导出
class OPENAI_API_API MyClass { ... };

// 函数导出
OPENAI_API_API void my_function();
```

## 平台支持

### Windows (MSVC)

- 编译库时定义 `OPENAI_API_EXPORTS`，使用 `__declspec(dllexport)`
- 使用库时（未定义 `OPENAI_API_EXPORTS`），使用 `__declspec(dllimport)`

### Linux/macOS (GCC/Clang)

- 使用 `__attribute__((visibility("default")))`
- 默认隐藏符号，显式导出需要的符号

## CMake 配置

### 静态库（默认）
```cmake
# 默认构建静态库，不需要特殊处理
add_subdirectory(openai-api)
target_link_libraries(my_app PRIVATE openai_api::server)
```

### 动态库（DLL/SO/Dylib）
```cmake
# 构建动态库
set(OPENAI_API_BUILD_SHARED ON CACHE BOOL "" FORCE)
add_subdirectory(openai-api)
target_link_libraries(my_app PRIVATE openai_api::server)

# Windows 上需要复制 DLL 到输出目录
if(WIN32)
    add_custom_command(TARGET my_app POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:openai_api> $<TARGET_FILE_DIR:my_app>
    )
endif()
```

## 导出的类

以下类已添加导出宏：

| 类名 | 头文件 |
|------|--------|
| `Server` | `openai_api/server.hpp` |
| `ClusterServer` | `openai_api/cluster_server.hpp` |
| `ModelRouter` | `openai_api/router.hpp` |
| `BaseDataProvider` | `openai_api/core/data_provider.hpp` |
| `QueueProvider` | `openai_api/core/data_provider.hpp` |
| `WorkerManager` | `openai_api/cluster/worker_manager.hpp` |
| `WorkerClient` | `openai_api/cluster/worker_client.hpp` |
| `RemoteWorkerProvider` | `openai_api/cluster/remote_worker_provider.hpp` |
| `Encoder` | `openai_api/encoder/encoder.hpp` |

## 注意事项

1. **模板类不需要导出**：模板类（如各种 Encoder 实现）是 header-only，不需要导出
2. **内联函数不需要导出**：内联函数定义在头文件中，不需要导出
3. **静态成员变量**：如果类有静态成员变量，需要在 .cpp 文件中定义，确保正确导出

## 示例

### 导出类
```cpp
// my_class.hpp
#pragma once
#include "openai_api/core/api_export.hpp"

class OPENAI_API_API MyClass {
public:
    void public_method();
    
private:
    void private_method();  // 不需要导出，但类的所有方法都可以访问
};
```

### 导出函数
```cpp
// my_api.hpp
#pragma once
#include "openai_api/core/api_export.hpp"
#include <string>

OPENAI_API_API std::string get_version();
OPENAI_API_API int initialize();
OPENAI_API_API void cleanup();
```

### 导出全局变量（不推荐）
```cpp
// 在 .cpp 中定义
OPENAI_API_API extern int g_global_counter;

// 在 .cpp 中初始化
int g_global_counter = 0;
```
