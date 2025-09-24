/*
 * Copyright (C) 2025 Aless Microsystems
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published
 * by the Free Software Foundation, version 3 of the License, or under
 * alternative licensing terms as granted by Aless Microsystems.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 */

#pragma once
#ifdef __AVX2__
#define SIMDE_ENABLE_NATIVE_ALIASES
#endif
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstdlib>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/pair.h>

#if defined(__AVX2__) || defined(__ARM_NEON)
//#include <immintrin.h>
#include <simde/x86/sse2.h>
#include <simde/x86/sse3.h>
#include <simde/x86/sse4.1.h>
#include <simde/x86/sse4.2.h>
#include <simde/x86/avx2.h>
#include <simde/arm/neon.h>
// Disable AVX2 SIMD for JSON canonicalization
//#define DISABLE_AVX2_JSON_SIMD
// Separate controls for base64 encoder/decoder
#define DISABLE_SSE_BASE64_ENCODER //pass
#define DISABLE_SSE_BASE64_ENCODER_MULA //pass
//#define DISABLE_AVX2_BASE64_DECODER
#define DISABLE_SSE_BASE64_ENCODER_ALIGNED //pass
#define DISABLE_SSE_BASE64_ENCODER_LEMIRE //pass
#define DISABLE_SSE_BASE64_ENCODER_AVX //pass
//#define DISABLE_NEON_BASE64_ENCODER //pass

#endif


// Compiler-specific optimization attributes
#ifdef __clang__
#define ALWAYS_INLINE [[clang::always_inline]] inline
#define HOT_FUNCTION [[clang::hot]]
#define FLATTEN_FUNCTION [[clang::flatten]]
#elif defined(__GNUC__)
#define ALWAYS_INLINE [[gnu::always_inline]] inline
#define HOT_FUNCTION [[gnu::hot]]
#define FLATTEN_FUNCTION [[gnu::flatten]]
#else
#define ALWAYS_INLINE inline
#define HOT_FUNCTION
#define FLATTEN_FUNCTION
#endif


// Debug logging infrastructure - check environment variable at runtime
inline bool is_debug_enabled() {
    static bool cached_result = []() {
        const char* env1 = std::getenv("SYNAPSE_CPP_CRYPTO_DEBUG");
        const char* env2 = std::getenv("SYNAPSE_RUST_CRYPTO_DEBUG");
        return (env1 && std::string(env1) == "1") || (env2 && std::string(env2) == "1");
    }();
    return cached_result;
}

#define DEBUG_LOG(msg) do { \
    std::ofstream logfile("/tmp/cpp_crypto_debug.log", std::ios::app); \
    if (logfile.is_open()) { \
        logfile << "DEBUG C++ crypto: " << msg << " (debug_enabled=" << (is_debug_enabled() ? "true" : "false") << ")" << std::endl; \
        logfile.close(); \
    } \
    if (is_debug_enabled()) { \
        std::cout << "DEBUG C++ crypto: " << msg << std::endl; \
    } \
} while(0)

// Keep the old variable for compatibility
static bool debug_enabled = is_debug_enabled();

// Hex lookup table for debug output
static constexpr char hex_lut[] = "0123456789abcdef";

// Maximum event size to prevent DoS attacks
static constexpr size_t MAX_EVENT_SIZE = 256 * 1024; // 256 KB

// Shared exception classes
class SignatureVerifyException : public std::runtime_error {
public:
    explicit SignatureVerifyException(const std::string& msg) : std::runtime_error(msg) {}
};

// Shared enums and structs for JSON processing
enum class TaskType { VALUE, ARRAY_START, ARRAY_ITEM, DICT_START, DICT_KEY, DICT_VALUE };

struct Task {
    TaskType type;
    nanobind::object obj;
    nanobind::handle seq;
    size_t index = 0;
    size_t seq_size = 0;
    std::vector<std::pair<std::string, nanobind::object>> dict_pairs;
};

// Namespace alias
namespace nb = nanobind;

// Debug initialization function
inline void debug_init() {
    std::ofstream logfile("/tmp/cpp_crypto_debug.log", std::ios::app);
    if (logfile.is_open()) {
        logfile << "DEBUG INIT: C++ crypto module loaded, debug_enabled=" << (is_debug_enabled() ? "true" : "false") << std::endl;
        logfile.close();
    }
}

// Each module can define its own thread-local buffers locally
// No need for shared buffers across modules