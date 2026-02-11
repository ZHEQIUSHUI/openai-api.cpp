#pragma once

/**
 * API Export Macros
 * 
 * 跨平台动态库导出支持
 * 
 * 使用示例:
 * class OPENAI_API_API Server { ... };
 * void OPENAI_API_API some_function();
 */

// 定义导出宏
#if defined(_WIN32) || defined(__CYGWIN__)
    #ifdef OPENAI_API_EXPORTS
        // 编译库时使用 dllexport
        #define OPENAI_API_API __declspec(dllexport)
    #else
        // 使用库时使用 dllimport
        #define OPENAI_API_API __declspec(dllimport)
    #endif
    #define OPENAI_API_LOCAL
#else
    // GCC/Clang 使用可见性属性
    #if __GNUC__ >= 4
        #define OPENAI_API_API __attribute__((visibility("default")))
        #define OPENAI_API_LOCAL __attribute__((visibility("hidden")))
    #else
        #define OPENAI_API_API
        #define OPENAI_API_LOCAL
    #endif
#endif

// 废弃标记
#if defined(__GNUC__) || defined(__clang__)
    #define OPENAI_API_DEPRECATED(msg) __attribute__((deprecated(msg)))
#elif defined(_MSC_VER)
    #define OPENAI_API_DEPRECATED(msg) __declspec(deprecated(msg))
#else
    #define OPENAI_API_DEPRECATED(msg)
#endif

// 未使用参数标记
#if defined(__GNUC__) || defined(__clang__)
    #define OPENAI_API_UNUSED __attribute__((unused))
#elif defined(_MSC_VER)
    #define OPENAI_API_UNUSED
#else
    #define OPENAI_API_UNUSED
#endif

// 内联提示
#if defined(__GNUC__) || defined(__clang__)
    #define OPENAI_API_INLINE inline __attribute__((always_inline))
#elif defined(_MSC_VER)
    #define OPENAI_API_INLINE __forceinline
#else
    #define OPENAI_API_INLINE inline
#endif
