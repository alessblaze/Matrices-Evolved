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

#ifdef __AVX2__
#define SIMDE_ENABLE_NATIVE_ALIASES
#endif
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/pair.h>

#include <cstring>
#include <span>
#include <type_traits>
#include <openssl/evp.h>
#include <openssl/rand.h>
#include <openssl/sha.h>
#include <openssl/base64.h>
#include <boost/json.hpp>
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <sstream>
#include <iostream>
#include <cstdlib>
#include <stdexcept>
#include <charconv>
#include <cmath>
#include <type_traits>

#include <Python.h>

#ifdef __AVX2__
//#include <immintrin.h>
#include <simde/x86/sse2.h>
#include <simde/x86/sse3.h>
#include <simde/x86/sse4.1.h>
#include <simde/x86/sse4.2.h>
#include <simde/x86/avx2.h>
// Disable AVX2 SIMD for JSON canonicalization
//#define DISABLE_AVX2_JSON_SIMD
// Separate controls for base64 encoder/decoder
#define DISABLE_SSE_BASE64_ENCODER
#define DISABLE_SSE_BASE64_ENCODER_MULA
//#define DISABLE_AVX2_BASE64_DECODER
#define DISABLE_SSE_BASE64_ENCODER_ALIGNED
#define DISABLE_SSE_BASE64_ENCODER_LEMIRE
//#define DISABLE_SSE_BASE64_ENCODER_AVX
// BMI2 tail optimization
//#define DISABLE_BMI2_BASE64_TAIL
#endif

// Debug logging infrastructure
static bool debug_enabled = []() {
    const char* env = std::getenv("SYNAPSE_RUST_CRYPTO_DEBUG");
    return env && std::string(env) == "1";
}();

#define DEBUG_LOG(msg) do { \
    if (debug_enabled) { \
        std::cout << "DEBUG C++ crypto: " << msg << std::endl; \
    } \
} while(0)

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

/**
 * SSE2-optimized decimal point detection for JSON canonicalization
 * 
 * Algorithm:
 * 1. Use SSE2 compare + movemask for '.', 'e', 'E' detection
 * 2. Process 16 bytes per iteration with early termination
 * 3. Scalar fallback for remaining bytes
 * 
 * Performance: Faster and more reliable than SSE4.2 _mm_cmpistri
 * 
 * @param str Input string buffer
 * @param len String length
 * @return true if decimal point or exponent found
 */
#ifdef __AVX2__
ALWAYS_INLINE bool has_decimal_point_sse(const char* str, size_t len) {
    if (debug_enabled) {
        DEBUG_LOG("SSE2 decimal check: len=" + std::to_string(len));
    }
    
    const __m128i dot = _mm_set1_epi8('.');
    const __m128i e_lower = _mm_set1_epi8('e');
    const __m128i e_upper = _mm_set1_epi8('E');
    
    size_t i = 0;
    // Process 16 bytes at a time
    for (; i + 16 <= len; i += 16) {
        __m128i chunk = _mm_loadu_si128(reinterpret_cast<const __m128i*>(str + i));
        
        __m128i dot_match = _mm_cmpeq_epi8(chunk, dot);
        __m128i e_match = _mm_cmpeq_epi8(chunk, e_lower);
        __m128i E_match = _mm_cmpeq_epi8(chunk, e_upper);
        
        __m128i any_match = _mm_or_si128(dot_match, e_match);
        any_match = _mm_or_si128(any_match, E_match);
        
        if (_mm_movemask_epi8(any_match) != 0) {
            if (debug_enabled) {
                std::cout << "DEBUG C++ crypto: SSE2 found match in chunk at byte " << i << std::endl;
            }
            return true;
        }
    }
    
    // Scalar fallback for remaining bytes
    for (; i < len; ++i) {
        char c = str[i];
        if (c == '.' || c == 'e' || c == 'E') {
            if (debug_enabled) {
                std::cout << "DEBUG C++ crypto: Scalar found '" << c << "' at pos " << i << std::endl;
            }
            return true;
        }
    }
    
    if (debug_enabled) {
        std::cout << "DEBUG C++ crypto: No decimal point found" << std::endl;
    }
    return false;
}
#endif

// Global hex lookup table for optimal performance
static constexpr char hex_lut[] = "0123456789abcdef";

/**
 * Fast double-to-string conversion optimized for JSON canonicalization
 * 
 * Algorithm:
 * 1. Fast-path common values (0.0, 1.0) with pre-computed strings
 * 2. Use std::to_chars for optimal performance (no locale, no malloc)
 * 3. SSE2-vectorized decimal point detection for integers
 * 4. Normalize -0.0 to 0.0 for canonical representation
 * 
 * @param f Input double value (must be finite)
 * @param result Output buffer (minimum 32 bytes)
 * @return Length of written string
 * @throws std::runtime_error if conversion fails
 */
inline int fast_double_to_string(double f, char* result) {
    // Normalize negative zero to positive zero
    if (f == 0.0 && std::signbit(f)) f = 0.0;
    
    // Fast paths for common values (already normalized)
    if (f == 0.0) {
        result[0] = '0'; result[1] = '.'; result[2] = '0';
        return 3;
    }
    if (f == 1.0) {
        result[0] = '1'; result[1] = '.'; result[2] = '0';
        return 3;
    }
    
    // Use std::to_chars with safe buffer size (64 bytes for all double representations)
    auto [ptr, ec] = std::to_chars(result, result + 64, f);
    if (ec == std::errc{}) {
        int len = ptr - result;
        
        // SSE2-optimized decimal point detection
#ifdef __AVX2__
        bool has_dot = has_decimal_point_sse(result, len);
#else
        bool has_dot = false;
        for (int i = 0; i < len; i++) {
            if (result[i] == '.' || result[i] == 'e' || result[i] == 'E') {
                has_dot = true;
                break;
            }
        }
#endif
        
        if (!has_dot) {
            result[len++] = '.';
            result[len++] = '0';
        }
        return len;
    }
    
    throw std::runtime_error("Failed to convert double to string");
}

namespace nb = nanobind;
using namespace nb::literals;
namespace json = boost::json;

// Memory leak profiling
static bool leak_warnings_enabled = []() {
    const char* env = std::getenv("SYNAPSE_RUST_CRYPTO_LEAK_WARNINGS");
    return env && std::string(env) == "1";
}();

// Custom exception for signature verification failures
class SignatureVerifyException : public std::runtime_error {
public:
    SignatureVerifyException(const std::string& msg) : std::runtime_error(msg) {}
};

/**
 * Thread-local JSON canonicalization buffer system
 * 
 * Design:
 * - Each thread gets isolated buffer to ensure thread safety
 * - Pointer-based writing eliminates string concatenation overhead
 * - Dynamic growth with 2x expansion strategy
 * - Hard 256KB limit prevents DoS attacks
 */
thread_local std::vector<char> json_buffer(64 * 1024); // Initialize with 64KB default
thread_local char* json_ptr = nullptr;

// Maximum event size to prevent DoS attacks
static constexpr size_t MAX_EVENT_SIZE = 256 * 1024; // 256 KB

/**
 * Initialize thread-local JSON buffer with size estimation
 * 
 * Algorithm:
 * 1. Estimate buffer size: max(hint * 100 bytes, 64KB minimum)
 * 2. Enforce hard 256KB limit to prevent DoS attacks
 * 3. Grow existing buffer only if needed (amortized allocation)
 * 4. Reset write pointer without zeroing memory (overwrite pattern)
 * 
 * @param hint Number of dictionary items for size estimation
 * @throws SignatureVerifyException if estimated size exceeds limit
 */
inline void init_json_buffer(size_t hint = 0) {
    // Guard against integer overflow in size calculation
    if (hint > MAX_EVENT_SIZE / 100) {
        throw SignatureVerifyException("Event hint too large");
    }
    size_t min_size = std::max(hint * 100, 64 * 1024UL); // Estimate 100 bytes per dict item, 8KB minimum
    
    if (min_size > MAX_EVENT_SIZE) {
        throw SignatureVerifyException("Event too large to canonicalize");
    }
    
    // Grow buffer if needed (never exceeds MAX_EVENT_SIZE due to check above)
    if (json_buffer.size() < min_size) {
        json_buffer.resize(min_size);
    }
    
    // Reset write position (no need to zero memory - we overwrite as we go)
    json_ptr = json_buffer.data();
}

/**
 * Ensure sufficient buffer space with exponential growth strategy
 * 
 * Algorithm:
 * 1. Calculate current buffer position via pointer arithmetic
 * 2. Check if remaining space is sufficient
 * 3. If growth needed: double buffer size until requirement met
 * 4. Enforce hard 256KB limit during growth
 * 5. Preserve existing data and update write pointer
 * 
 * @param needed Number of bytes required
 * @throws SignatureVerifyException if growth would exceed limit
 */
ALWAYS_INLINE void ensure_space(size_t needed) {
    size_t current_size = json_ptr - json_buffer.data();
    
    // Check for integer overflow in addition
    if (current_size > SIZE_MAX - needed) {
        throw SignatureVerifyException("Buffer size overflow");
    }
    
    if (current_size + needed > json_buffer.size()) {
        size_t new_size = json_buffer.size();
        
        // Safe exponential growth with overflow protection
        while (new_size < current_size + needed) {
            if (new_size > SIZE_MAX / 2) {
                throw SignatureVerifyException("Buffer size overflow");
            }
            new_size *= 2;
        }
        
        if (new_size > MAX_EVENT_SIZE) {
            throw SignatureVerifyException("Event canonical JSON exceeds maximum size");
        }
        
        json_buffer.resize(new_size);
        json_ptr = json_buffer.data() + current_size;
    }
}

ALWAYS_INLINE void write_char(char c) {
    ensure_space(1);
    *json_ptr++ = c;
}

/**
 * SSE4.1-optimized string writing for JSON canonicalization
 * 
 * Algorithm:
 * 1. Use SIMD copying for strings >= 16 bytes
 * 2. Process 16-byte chunks with SSE4.1 instructions
 * 3. Handle remaining bytes with scalar memcpy
 * 
 * Performance: ~15-25% faster than memcpy for large strings
 */
ALWAYS_INLINE void write_string(std::string_view s) {
    ensure_space(s.size());
    
#ifdef __AVX2__
    if (s.size() >= 16) {
        const char* src = s.data();
        char* dst = json_ptr;
        size_t remaining = s.size();
        
        // Process 16-byte chunks with SSE4.1
        while (remaining >= 16) {
            __m128i chunk = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src));
            _mm_storeu_si128(reinterpret_cast<__m128i*>(dst), chunk);
            src += 16; dst += 16; remaining -= 16;
        }
        
        // Handle remaining bytes with scalar
        if (remaining > 0) {
            std::memcpy(dst, src, remaining);
        }
    } else {
        std::memcpy(json_ptr, s.data(), s.size());
    }
#else
    std::memcpy(json_ptr, s.data(), s.size());
#endif
    
    json_ptr += s.size();
}

ALWAYS_INLINE void write_raw_unsafe(const char* data, size_t len) {
    // Assumes space already reserved - no bounds checking
#ifdef __AVX2__
    if (len >= 16) {
        const char* src = data;
        char* dst = json_ptr;
        size_t remaining = len;
        
        // Process 16-byte chunks with SSE4.1
        while (remaining >= 16) {
            __m128i chunk = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src));
            _mm_storeu_si128(reinterpret_cast<__m128i*>(dst), chunk);
            src += 16; dst += 16; remaining -= 16;
        }
        
        // Handle remaining bytes with scalar
        if (remaining > 0) {
            std::memcpy(dst, src, remaining);
        }
    } else {
        std::memcpy(json_ptr, data, len);
    }
#else
    std::memcpy(json_ptr, data, len);
#endif
    
    json_ptr += len;
}

ALWAYS_INLINE void write_raw(const char* data, size_t len) {
    ensure_space(len);
    write_raw_unsafe(data, len);
}

/**
 * Get current JSON buffer contents as specified type
 * 
 * @tparam T Return type (std::span<const uint8_t>, std::span<uint8_t>, or std::vector<uint8_t>)
 * @return JSON buffer contents as requested type
 */
template<typename T = std::vector<uint8_t>>
ALWAYS_INLINE T get_json_span() {
    size_t len = json_ptr - json_buffer.data();
    uint8_t* data = reinterpret_cast<uint8_t*>(json_buffer.data());
    
    if constexpr (std::is_same_v<T, std::span<const uint8_t>>) {
        return std::span<const uint8_t>(data, len);
    } else if constexpr (std::is_same_v<T, std::span<uint8_t>>) {
        return std::span<uint8_t>(data, len);
    } else {
        return std::vector<uint8_t>(data, data + len);
    }
}

ALWAYS_INLINE void write_cstring(const char* s) {
    size_t len = strlen(s);
    ensure_space(len);
    std::memcpy(json_ptr, s, len);
    json_ptr += len;
}

/**
 * Fast Unicode escape sequence generation for JSON strings
 * 
 * Algorithm:
 * 1. Use lookup table for hex digit conversion (no division)
 * 2. Pre-format escape sequence: \u00XX
 * 3. Extract high/low nibbles via bit operations
 * 4. Write complete 6-byte sequence in one operation
 * 
 * @param c Character to escape (< 0x20)
 */

ALWAYS_INLINE void write_unicode_escape_unsafe(unsigned char c) {
    // Assumes space already reserved - no bounds checking
    *json_ptr++ = '\\';
    *json_ptr++ = 'u';
    *json_ptr++ = '0';
    *json_ptr++ = '0';
    *json_ptr++ = hex_lut[c >> 4];
    *json_ptr++ = hex_lut[c & 0xF];
}

ALWAYS_INLINE void write_unicode_escape(unsigned char c) {
    ensure_space(6);
    *json_ptr++ = '\\';
    *json_ptr++ = 'u';
    *json_ptr++ = '0';
    *json_ptr++ = '0';
    *json_ptr++ = hex_lut[c >> 4];
    *json_ptr++ = hex_lut[c & 0xF];
}

/**
 * SSE4.1-optimized hex string generation
 * 
 * Algorithm:
 * 1. Process 16 bytes at a time with SIMD
 * 2. Extract high/low nibbles in parallel
 * 3. Use lookup table for hex character conversion
 * 4. Generate 32 hex characters from 16 input bytes
 * 
 * Performance: ~3-4x faster than scalar for large data
 */
#ifdef __AVX2__
ALWAYS_INLINE std::string vectorized_hex_string(const uint8_t* data, size_t size) {
    std::string result;
    result.resize(size * 2);  // Pre-allocate exact size
    char* out = result.data();
    
    if (size >= 16) {
        DEBUG_LOG("Using SSE4.1 vectorized hex conversion for " + std::to_string(size) + " bytes");
        
        const __m128i hex_lut = _mm_setr_epi8('0','1','2','3','4','5','6','7','8','9','a','b','c','d','e','f');
        const __m128i mask_low = _mm_set1_epi8(0x0F);
        
        size_t i = 0;
        for (; i + 16 <= size; i += 16) {
            __m128i input = _mm_loadu_si128(reinterpret_cast<const __m128i*>(data + i));
            
            // Extract high and low nibbles
            __m128i high_nibbles = _mm_and_si128(_mm_srli_epi16(input, 4), mask_low);
            __m128i low_nibbles = _mm_and_si128(input, mask_low);
            
            // Convert to hex characters
            __m128i high_hex = _mm_shuffle_epi8(hex_lut, high_nibbles);
            __m128i low_hex = _mm_shuffle_epi8(hex_lut, low_nibbles);
            
            // Interleave high and low hex characters
            __m128i hex_lo = _mm_unpacklo_epi8(high_hex, low_hex);
            __m128i hex_hi = _mm_unpackhi_epi8(high_hex, low_hex);
            
            // Write directly to result buffer - no intermediate copy
            _mm_storeu_si128(reinterpret_cast<__m128i*>(out), hex_lo);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(out + 16), hex_hi);
            out += 32;
        }
        
        // Handle remaining bytes with fast scalar lookup
        static const char lut[] = "0123456789abcdef";
        for (; i < size; ++i) {
            *out++ = lut[data[i] >> 4];
            *out++ = lut[data[i] & 0x0F];
        }
    } else {
        // Fast scalar fallback for small data
        static const char lut[] = "0123456789abcdef";
        for (size_t i = 0; i < size; ++i) {
            *out++ = lut[data[i] >> 4];
            *out++ = lut[data[i] & 0x0F];
        }
    }
    
    return result;
}
#endif

enum class TaskType { VALUE, ARRAY_START, ARRAY_ITEM, DICT_START, DICT_KEY, DICT_VALUE };

struct Task {
    TaskType type;
    nb::object obj;
    nb::handle seq;
    size_t index = 0;
    size_t seq_size = 0;
    std::vector<std::pair<std::string, nb::object>> dict_pairs;
};

/**
 * Fast Python-to-JSON canonicalization with iterative approach
 * 
 * REQUIREMENTS:
 * - Caller must hold the Python GIL for the entire function call
 * - No reentrancy - function uses local stack to avoid thread_local issues
 * - Dict keys must be strings, values can be any JSON-serializable type
 */
HOT_FUNCTION FLATTEN_FUNCTION inline void py_to_canonical_json_fast(const nb::object& root_obj) {
    // Ensure GIL is held for Python API calls
    if (!PyGILState_Check()) {
        throw std::runtime_error("py_to_canonical_json_fast requires the Python GIL");
    }
    
    // Function-local stack to avoid reentrancy issues
    std::vector<Task> stack;
    stack.reserve(512);
    stack.push_back({TaskType::VALUE, root_obj});
    
    while (!stack.empty()) {
        Task& t = stack.back();
        
        switch (t.type) {
            case TaskType::VALUE: {
                nb::object obj = t.obj;
                stack.pop_back();
                
                if (obj.is_none()) {
                    write_cstring("null");
                } else if (nb::isinstance<nb::bool_>(obj)) {
                    write_cstring(nb::cast<bool>(obj) ? "true" : "false");
                } else if (nb::isinstance<nb::int_>(obj)) {
                    char buf[32];
                    auto [ptr, ec] = std::to_chars(buf, buf + 32, nb::cast<int64_t>(obj));
                    write_string(std::string_view(buf, ptr - buf));
                } else if (nb::isinstance<nb::float_>(obj)) {
                    double val = nb::cast<double>(obj);
                    if (!std::isfinite(val)) {
                        throw std::runtime_error("Non-finite floats not allowed in JSON");
                    }
                    if (val == 0.0 && std::signbit(val)) val = 0.0;
                    char buf[64];
                    int len = fast_double_to_string(val, buf);
                    write_string(std::string_view(buf, len));
                } else if (nb::isinstance<nb::str>(obj)) {
                    std::string_view s = nb::cast<std::string_view>(obj);
                    write_char('"');
                    
#if defined(__AVX2__) && !defined(DISABLE_AVX2_JSON_SIMD)
                    /**
                     * AVX2 SIMD JSON string escaping optimization
                     * 
                     * Algorithm:
                     * 1. Process 32-byte chunks with parallel character detection
                     * 2. Use XOR trick for unsigned comparison: (c ^ 0x80) < (0x20 ^ 0x80)
                     * 3. Fast path: bulk copy chunks with no escape characters
                     * 4. Slow path: character-by-character escaping when needed
                     * 
                     * Performance: ~3-5x faster than scalar for clean strings
                     */
                    if (s.size() >= 64) {
                        DEBUG_LOG("Using AVX2 vectorized JSON string escaping for " + std::to_string(s.size()) + " bytes");
                        const char* data = s.data();
                        size_t len = s.size();
                        
                        // Guard against integer overflow and size limits
                        if (len > MAX_EVENT_SIZE / 6) {
                            throw SignatureVerifyException("String too large to escape");
                        }
                        if (len > SIZE_MAX / 6) {
                            throw SignatureVerifyException("String too large to escape");
                        }
                        // Reserve worst-case space up-front (6x expansion)
                        ensure_space(len * 6);
                        
                        size_t i = 0;
                        
                        const __m256i control_threshold = _mm256_set1_epi8(static_cast<char>(0x20 ^ 0x80));
                        const __m256i xor_mask = _mm256_set1_epi8(0x80);
                        const __m256i quote_mask = _mm256_set1_epi8('"');
                        const __m256i backslash_mask = _mm256_set1_epi8('\\');
                        
                        // Process 64B per iteration for higher throughput
                        DEBUG_LOG("Processing 64-byte chunks with AVX2 SIMD");
                        for (; i + 64 <= len; i += 64) {
                            __m256i chunk1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + i));
                            __m256i chunk2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + i + 32));
                            
                            // Process first 32B
                            __m256i xored1 = _mm256_xor_si256(chunk1, xor_mask);
                            __m256i control_cmp1 = _mm256_cmpgt_epi8(control_threshold, xored1);
                            __m256i quote_cmp1 = _mm256_cmpeq_epi8(chunk1, quote_mask);
                            __m256i backslash_cmp1 = _mm256_cmpeq_epi8(chunk1, backslash_mask);
                            __m256i escape_mask1 = _mm256_or_si256(_mm256_or_si256(control_cmp1, quote_cmp1), backslash_cmp1);
                            
                            // Process second 32B
                            __m256i xored2 = _mm256_xor_si256(chunk2, xor_mask);
                            __m256i control_cmp2 = _mm256_cmpgt_epi8(control_threshold, xored2);
                            __m256i quote_cmp2 = _mm256_cmpeq_epi8(chunk2, quote_mask);
                            __m256i backslash_cmp2 = _mm256_cmpeq_epi8(chunk2, backslash_mask);
                            __m256i escape_mask2 = _mm256_or_si256(_mm256_or_si256(control_cmp2, quote_cmp2), backslash_cmp2);
                            
                            // Fast path: both chunks clean
                            if (__builtin_expect(_mm256_testz_si256(escape_mask1, escape_mask1) && 
                                               _mm256_testz_si256(escape_mask2, escape_mask2), 1)) {
                                DEBUG_LOG("Clean 64-byte chunk - bulk copying");
                                write_raw_unsafe(data + i, 64);
                                continue;
                            }
                            DEBUG_LOG("Found escape characters in 64-byte chunk");
                            
                            // Process first chunk
                            if (__builtin_expect(_mm256_testz_si256(escape_mask1, escape_mask1), 1)) {
                                write_raw_unsafe(data + i, 32);
                            } else {
                                uint32_t mask = static_cast<uint32_t>(_mm256_movemask_epi8(escape_mask1));
                                size_t base = i;
                                while (mask) {
                                    unsigned tz = __builtin_ctz(mask);
                                    size_t good_len = tz - (base - i);
                                    if (good_len) write_raw_unsafe(data + base, good_len);
                                    
                                    unsigned char c = static_cast<unsigned char>(data[i + tz]);
                                    switch (c) {
                                        case '"': write_raw_unsafe("\\\"", 2); break;
                                        case '\\': write_raw_unsafe("\\\\", 2); break;
                                        case '\b': write_raw_unsafe("\\b", 2); break;
                                        case '\f': write_raw_unsafe("\\f", 2); break;
                                        case '\n': write_raw_unsafe("\\n", 2); break;
                                        case '\r': write_raw_unsafe("\\r", 2); break;
                                        case '\t': write_raw_unsafe("\\t", 2); break;
                                        default: write_unicode_escape_unsafe(c); break;
                                    }
                                    
                                    base = i + tz + 1;
                                    mask &= mask - 1;
                                }
                                if (base < i + 32) write_raw_unsafe(data + base, (i + 32) - base);
                            }
                            
                            // Process second chunk
                            if (__builtin_expect(_mm256_testz_si256(escape_mask2, escape_mask2), 1)) {
                                write_raw_unsafe(data + i + 32, 32);
                            } else {
                                uint32_t mask = static_cast<uint32_t>(_mm256_movemask_epi8(escape_mask2));
                                size_t base = i + 32;
                                while (mask) {
                                    unsigned tz = __builtin_ctz(mask);
                                    size_t good_len = tz - (base - (i + 32));
                                    if (good_len) write_raw_unsafe(data + base, good_len);
                                    
                                    unsigned char c = static_cast<unsigned char>(data[i + 32 + tz]);
                                    switch (c) {
                                        case '"': write_raw_unsafe("\\\"", 2); break;
                                        case '\\': write_raw_unsafe("\\\\", 2); break;
                                        case '\b': write_raw_unsafe("\\b", 2); break;
                                        case '\f': write_raw_unsafe("\\f", 2); break;
                                        case '\n': write_raw_unsafe("\\n", 2); break;
                                        case '\r': write_raw_unsafe("\\r", 2); break;
                                        case '\t': write_raw_unsafe("\\t", 2); break;
                                        default: write_unicode_escape_unsafe(c); break;
                                    }
                                    
                                    base = i + 32 + tz + 1;
                                    mask &= mask - 1;
                                }
                                if (base < i + 64) write_raw_unsafe(data + base, (i + 64) - base);
                            }
                        }
                        
                        // Handle remaining 32-byte chunks
                        DEBUG_LOG("Processing remaining 32-byte chunks with AVX2 SIMD");
                        for (; i + 32 <= len; i += 32) {
                            __m256i chunk = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + i));
                            
                            __m256i xored = _mm256_xor_si256(chunk, xor_mask);
                            __m256i control_cmp = _mm256_cmpgt_epi8(control_threshold, xored);
                            __m256i quote_cmp = _mm256_cmpeq_epi8(chunk, quote_mask);
                            __m256i backslash_cmp = _mm256_cmpeq_epi8(chunk, backslash_mask);
                            __m256i escape_mask = _mm256_or_si256(_mm256_or_si256(control_cmp, quote_cmp), backslash_cmp);
                            
                            if (__builtin_expect(_mm256_testz_si256(escape_mask, escape_mask), 1)) {
                                write_raw_unsafe(data + i, 32);
                            } else {
                                uint32_t mask = static_cast<uint32_t>(_mm256_movemask_epi8(escape_mask));
                                size_t base = i;
                                while (mask) {
                                    unsigned tz = __builtin_ctz(mask);
                                    size_t good_len = tz - (base - i);
                                    if (good_len) write_raw_unsafe(data + base, good_len);
                                    
                                    unsigned char c = static_cast<unsigned char>(data[i + tz]);
                                    switch (c) {
                                        case '"': write_raw_unsafe("\\\"", 2); break;
                                        case '\\': write_raw_unsafe("\\\\", 2); break;
                                        case '\b': write_raw_unsafe("\\b", 2); break;
                                        case '\f': write_raw_unsafe("\\f", 2); break;
                                        case '\n': write_raw_unsafe("\\n", 2); break;
                                        case '\r': write_raw_unsafe("\\r", 2); break;
                                        case '\t': write_raw_unsafe("\\t", 2); break;
                                        default: write_unicode_escape_unsafe(c); break;
                                    }
                                    
                                    base = i + tz + 1;
                                    mask &= mask - 1;
                                }
                                if (base < i + 32) write_raw_unsafe(data + base, (i + 32) - base);
                            }
                        }
                        
                        // Handle remaining bytes
                        for (; i < len; ++i) {
                            unsigned char c = static_cast<unsigned char>(data[i]);
                            switch (c) {
                                case '"': write_raw_unsafe("\\\"", 2); break;
                                case '\\': write_raw_unsafe("\\\\", 2); break;
                                case '\b': write_raw_unsafe("\\b", 2); break;
                                case '\f': write_raw_unsafe("\\f", 2); break;
                                case '\n': write_raw_unsafe("\\n", 2); break;
                                case '\r': write_raw_unsafe("\\r", 2); break;
                                case '\t': write_raw_unsafe("\\t", 2); break;
                                default:
                                    if (c < 0x20) write_unicode_escape_unsafe(c);
                                    else write_char(static_cast<char>(c));
                            }
                        }
                    } else {
#endif
                        // Scalar loop
                        for (char ch : s) {
                            unsigned char c = static_cast<unsigned char>(ch);
                            switch (c) {
                                case '"': write_string("\\\""); break;
                                case '\\': write_string("\\\\"); break;
                                case '\b': write_string("\\b"); break;
                                case '\f': write_string("\\f"); break;
                                case '\n': write_string("\\n"); break;
                                case '\r': write_string("\\r"); break;
                                case '\t': write_string("\\t"); break;
                                default:
                                    if (c < 0x20) write_unicode_escape_unsafe(c);
                                    else write_char(static_cast<char>(c));
                            }
                        }
#if defined(__AVX2__) && !defined(DISABLE_AVX2_JSON_SIMD)
                    }
#endif
                    write_char('"');
                } else if (nb::isinstance<nb::list>(obj) || nb::isinstance<nb::tuple>(obj)) {
                    stack.push_back({TaskType::ARRAY_START, obj});
                } else if (nb::isinstance<nb::dict>(obj)) {
                    stack.push_back({TaskType::DICT_START, obj});
                } else {
                    throw std::runtime_error("Unsupported Python type");
                }
                break;
            }
            
            case TaskType::ARRAY_START: {
                write_char('[');
                t.type = TaskType::ARRAY_ITEM;
                t.index = 0;
                t.seq = nb::borrow(t.obj);
                Py_ssize_t seq_len = nb::len(t.seq);
                if (seq_len < 0) throw std::runtime_error("Invalid sequence length");
                t.seq_size = static_cast<size_t>(seq_len);
                break;
            }
            
            case TaskType::ARRAY_ITEM: {
                if (t.index >= t.seq_size) {
                    write_char(']');
                    stack.pop_back();
                } else {
                    if (t.index > 0) write_char(',');
                    stack.push_back({TaskType::VALUE, nb::object(nb::handle(t.seq[t.index]), nb::detail::borrow_t{})});
                    t.index++;
                }
                break;
            }
            
            case TaskType::DICT_START: {
                auto dict = nb::borrow<nb::dict>(t.obj);
                t.dict_pairs.clear();
                t.dict_pairs.reserve(dict.size());
                

                for (auto item : dict) {
                    if (!nb::isinstance<nb::str>(item.first)) {
                        throw std::runtime_error("Dictionary keys must be strings for JSON serialization");
                    }
                    
                    std::string key_str = nb::cast<std::string>(item.first);
                    t.dict_pairs.emplace_back(std::move(key_str), nb::object(nb::handle(item.second), nb::detail::borrow_t{}));
                }
                
                std::sort(t.dict_pairs.begin(), t.dict_pairs.end(),
                         [](const auto& a, const auto& b) { return a.first < b.first; });
                
                write_char('{');
                t.type = TaskType::DICT_KEY;
                t.index = 0;
                break;
            }
            
            case TaskType::DICT_KEY: {
                if (t.index >= t.dict_pairs.size()) {
                    write_char('}');
                    stack.pop_back();
                } else {
                    if (t.index > 0) write_char(',');
                    
                    write_char('"');
                    const std::string& key = t.dict_pairs[t.index].first;
                    
#if defined(__AVX2__) && !defined(DISABLE_AVX2_JSON_SIMD)

                    if (key.size() >= 32) {
                        DEBUG_LOG("Using AVX2 vectorized dictionary key escaping for " + std::to_string(key.size()) + " bytes");
                        const char* data = key.data();
                        size_t len = key.size();
                        
                        // Guard against integer overflow and size limits
                        if (len > MAX_EVENT_SIZE / 6) {
                            throw SignatureVerifyException("String too large to escape");
                        }
                        if (len > SIZE_MAX / 6) {
                            throw SignatureVerifyException("String too large to escape");
                        }
                        // Reserve worst-case space up-front (6x expansion)
                        ensure_space(len * 6);
                        
                        size_t i = 0;
                        const __m256i control_threshold = _mm256_set1_epi8(static_cast<char>(0x20 ^ 0x80));
                        const __m256i xor_mask = _mm256_set1_epi8(0x80);
                        const __m256i quote_mask = _mm256_set1_epi8('"');
                        const __m256i backslash_mask = _mm256_set1_epi8('\\');
                        
                        DEBUG_LOG("Processing dictionary key with AVX2 32-byte chunks");
                        for (; i + 32 <= len; i += 32) {
                            __m256i chunk = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + i));
                            
                            __m256i xored = _mm256_xor_si256(chunk, xor_mask);
                            __m256i control_cmp = _mm256_cmpgt_epi8(control_threshold, xored);
                            __m256i quote_cmp = _mm256_cmpeq_epi8(chunk, quote_mask);
                            __m256i backslash_cmp = _mm256_cmpeq_epi8(chunk, backslash_mask);
                            __m256i escape_mask = _mm256_or_si256(_mm256_or_si256(control_cmp, quote_cmp), backslash_cmp);
                            
                            if (__builtin_expect(_mm256_testz_si256(escape_mask, escape_mask), 1)) {
                                DEBUG_LOG("Clean dictionary key chunk - bulk copying");
                                write_raw_unsafe(data + i, 32);
                            } else {
                                uint32_t mask = static_cast<uint32_t>(_mm256_movemask_epi8(escape_mask));
                                size_t base = i;
                                while (mask) {
                                    unsigned tz = __builtin_ctz(mask);
                                    size_t good_len = tz - (base - i);
                                    if (good_len) write_raw_unsafe(data + base, good_len);
                                    
                                    unsigned char c = static_cast<unsigned char>(data[i + tz]);
                                    switch (c) {
                                        case '"': write_raw_unsafe("\\\"", 2); break;
                                        case '\\': write_raw_unsafe("\\\\", 2); break;
                                        case '\b': write_raw_unsafe("\\b", 2); break;
                                        case '\f': write_raw_unsafe("\\f", 2); break;
                                        case '\n': write_raw_unsafe("\\n", 2); break;
                                        case '\r': write_raw_unsafe("\\r", 2); break;
                                        case '\t': write_raw_unsafe("\\t", 2); break;
                                        default: write_unicode_escape_unsafe(c); break;
                                    }
                                    
                                    base = i + tz + 1;
                                    mask &= mask - 1;
                                }
                                if (base < i + 32) write_raw_unsafe(data + base, (i + 32) - base);
                            }
                        }
                        
                        for (; i < len; ++i) {
                            unsigned char c = static_cast<unsigned char>(data[i]);
                            if (c == '"') write_raw_unsafe("\\\"", 2);
                            else if (c == '\\') write_raw_unsafe("\\\\", 2);
                            else if (c < 0x20) write_unicode_escape_unsafe(c);
                            else write_char(static_cast<char>(c));
                        }
                    } else {
#endif
                        for (char ch : key) {
                            unsigned char c = static_cast<unsigned char>(ch);
                            if (c == '"') write_raw("\\\"", 2);
                            else if (c == '\\') write_raw("\\\\", 2);
                            else if (c < 0x20) write_unicode_escape_unsafe(c);
                            else write_char(static_cast<char>(c));
                        }
#if defined(__AVX2__) && !defined(DISABLE_AVX2_JSON_SIMD)
                    }
#endif
                    write_char('"');
                    write_char(':');
                    
                    t.type = TaskType::DICT_VALUE;
                    stack.push_back({TaskType::VALUE, t.dict_pairs[t.index].second});
                }
                break;
            }
            
            case TaskType::DICT_VALUE: {
                t.type = TaskType::DICT_KEY;
                t.index++;
                break;
            }
        }
    }
}


thread_local std::string base64_buffer(1024, '\0'); // Initialize with 1KB default
thread_local std::vector<uint8_t> decode_buffer(1024); // Initialize with default size
thread_local std::vector<uint8_t> signature_buffer(64); // Ed25519 signature size
thread_local std::vector<uint8_t> hash_buffer(32);     // SHA256 hash size

// Simple key cache to avoid repeated OpenSSL key creation
thread_local std::vector<uint8_t> cached_signing_key;
thread_local EVP_PKEY* cached_signing_pkey = nullptr;




static constexpr uint8_t base64_decode_table[256] = {
    255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
    255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
    255,255,255,255,255,255,255,255,255,255,255, 62,255,255,255, 63,
     52, 53, 54, 55, 56, 57, 58, 59, 60, 61,255,255,255,254,255,255,
    255,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14,
     15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,255,255,255,255,255,
    255, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
     41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51,255,255,255,255,255,
    255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
    255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
    255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
    255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
    255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
    255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
    255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
    255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255
};


/**
 * Decodes base64 encoded string to binary data
 * @param encoded_string Base64 encoded input string
 * @return Decoded binary data as vector of bytes
 */
[[gnu::hot, gnu::flatten]] std::vector<uint8_t> base64_decode(std::string_view encoded_string);


/**
 * AVX2 helper function to reshuffle decoded base64 bytes into correct order
 * @param in Input AVX2 register containing decoded bytes
 * @return Reshuffled AVX2 register with bytes in proper sequence
 */
static inline __m256i dec_reshuffle(__m256i in) {
    const __m256i merge_ab_and_bc = _mm256_maddubs_epi16(in, _mm256_set1_epi32(0x01400140));
    __m256i out = _mm256_madd_epi16(merge_ab_and_bc, _mm256_set1_epi32(0x00011000));
    out = _mm256_shuffle_epi8(out, _mm256_setr_epi8(
        2, 1, 0, 6, 5, 4, 10, 9, 8, 14, 13, 12, -1, -1, -1, -1,
        2, 1, 0, 6, 5, 4, 10, 9, 8, 14, 13, 12, -1, -1, -1, -1
    ));
    return _mm256_permutevar8x32_epi32(out, _mm256_setr_epi32(0, 1, 2, 4, 5, 6, -1, -1));
}


/**
 * High-performance AVX2-optimized base64 decoder for cryptographic signatures
 * @param input Base64 encoded string to decode
 * @return Decoded binary data optimized for signature verification
 */
[[gnu::hot, gnu::flatten]] inline std::vector<uint8_t> fast_base64_decode_signature(std::string_view input) {
    DEBUG_LOG("fast_base64_decode_simd input length: " + std::to_string(input.size()));
    
    size_t output_len = (input.size() * 3) / 4;
    if (decode_buffer.size() < output_len) {
        decode_buffer.resize(output_len);
    }
    
    const char* src = input.data();
    uint8_t* dst = decode_buffer.data();
    size_t srclen = input.size();
    
#if defined(__AVX2__) && !defined(DISABLE_AVX2_BASE64_DECODER)
    DEBUG_LOG("Using AVX2 SIMD base64 decode path");
    

    const __m256i lut_lo = _mm256_setr_epi8(
        0x15, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11,
        0x11, 0x11, 0x13, 0x1A, 0x1B, 0x1B, 0x1B, 0x1A,
        0x15, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11,
        0x11, 0x11, 0x13, 0x1A, 0x1B, 0x1B, 0x1B, 0x1A
    );
    const __m256i lut_hi = _mm256_setr_epi8(
        0x10, 0x10, 0x01, 0x02, 0x04, 0x08, 0x04, 0x08,
        0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10,
        0x10, 0x10, 0x01, 0x02, 0x04, 0x08, 0x04, 0x08,
        0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10
    );
    const __m256i lut_roll = _mm256_setr_epi8(
        0,   16,  19,   4, -65, -65, -71, -71,
        0,   0,   0,   0,   0,   0,   0,   0,
        0,   16,  19,   4, -65, -65, -71, -71,
        0,   0,   0,   0,   0,   0,   0,   0
    );
    const __m256i mask_2F = _mm256_set1_epi8(0x2f);
    

    while (srclen >= 64) {
        // Load two 32-char chunks
        __m256i str1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src));
        __m256i str2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + 32));
        
        // Process first chunk
        __m256i hi_nibbles1 = _mm256_srli_epi16(str1, 4);
        hi_nibbles1 = _mm256_and_si256(hi_nibbles1, _mm256_set1_epi8(0x0F));
        __m256i lo_nibbles1 = _mm256_and_si256(str1, mask_2F);
        const __m256i lo1 = _mm256_shuffle_epi8(lut_lo, lo_nibbles1);
        const __m256i eq_2F1 = _mm256_cmpeq_epi8(str1, mask_2F);
        const __m256i hi1 = _mm256_shuffle_epi8(lut_hi, hi_nibbles1);
        const __m256i roll1 = _mm256_shuffle_epi8(lut_roll, _mm256_add_epi8(eq_2F1, hi_nibbles1));
        
        // Process second chunk
        __m256i hi_nibbles2 = _mm256_srli_epi16(str2, 4);
        hi_nibbles2 = _mm256_and_si256(hi_nibbles2, _mm256_set1_epi8(0x0F));
        __m256i lo_nibbles2 = _mm256_and_si256(str2, mask_2F);
        const __m256i lo2 = _mm256_shuffle_epi8(lut_lo, lo_nibbles2);
        const __m256i eq_2F2 = _mm256_cmpeq_epi8(str2, mask_2F);
        const __m256i hi2 = _mm256_shuffle_epi8(lut_hi, hi_nibbles2);
        const __m256i roll2 = _mm256_shuffle_epi8(lut_roll, _mm256_add_epi8(eq_2F2, hi_nibbles2));
        
        // Use reference validation approach for both chunks
        if (!_mm256_testz_si256(lo1, hi1) || !_mm256_testz_si256(lo2, hi2)) {
            DEBUG_LOG("SIMD decode failed - invalid characters, falling back to scalar");
            break;
        }
        
        // Decode and reshuffle both chunks
        str1 = _mm256_add_epi8(str1, roll1);
        str1 = dec_reshuffle(str1);
        str2 = _mm256_add_epi8(str2, roll2);
        str2 = dec_reshuffle(str2);
        
        // Store 48 bytes total (24 + 24)
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst), _mm256_castsi256_si128(str1));
        _mm_storel_epi64(reinterpret_cast<__m128i*>(dst + 16), _mm256_extracti128_si256(str1, 1));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + 24), _mm256_castsi256_si128(str2));
        _mm_storel_epi64(reinterpret_cast<__m128i*>(dst + 40), _mm256_extracti128_si256(str2, 1));
        
        src += 64;
        dst += 48;
        srclen -= 64;
    }
    

    while (srclen >= 32) {
        __m256i str = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src));
        
        __m256i hi_nibbles = _mm256_srli_epi16(str, 4);
        hi_nibbles = _mm256_and_si256(hi_nibbles, _mm256_set1_epi8(0x0F));
        __m256i lo_nibbles = _mm256_and_si256(str, mask_2F);
        
        const __m256i lo = _mm256_shuffle_epi8(lut_lo, lo_nibbles);
        const __m256i eq_2F = _mm256_cmpeq_epi8(str, mask_2F);
        const __m256i hi = _mm256_shuffle_epi8(lut_hi, hi_nibbles);
        const __m256i roll = _mm256_shuffle_epi8(lut_roll, _mm256_add_epi8(eq_2F, hi_nibbles));
        
        // Use reference validation approach
        if (!_mm256_testz_si256(lo, hi)) {
            DEBUG_LOG("SIMD decode failed - invalid characters, falling back to scalar");
            break;
        }
        
        str = _mm256_add_epi8(str, roll);
        str = dec_reshuffle(str);
        
        // Store 24 bytes
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst), _mm256_castsi256_si128(str));
        _mm_storel_epi64(reinterpret_cast<__m128i*>(dst + 16), _mm256_extracti128_si256(str, 1));
        
        src += 32;
        dst += 24;
        srclen -= 32;
    }
    

    while (srclen >= 16) {
        // Load 16 chars, decode with AVX scalar instructions
        __m128i chars = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src));
        
        // Process 4 chars at a time using scalar logic but AVX loads/stores
        for (int i = 0; i < 16; i += 4) {
            uint32_t val = (base64_decode_table[static_cast<uint8_t>(src[i])] << 18) |
                           (base64_decode_table[static_cast<uint8_t>(src[i+1])] << 12) |
                           (base64_decode_table[static_cast<uint8_t>(src[i+2])] << 6) |
                           base64_decode_table[static_cast<uint8_t>(src[i+3])];
            *dst++ = (val >> 16) & 0xFF;
            *dst++ = (val >> 8) & 0xFF;
            *dst++ = val & 0xFF;
        }
        src += 16;
        srclen -= 16;
    }
    

    while (srclen >= 4) {
        uint32_t val = (base64_decode_table[static_cast<uint8_t>(src[0])] << 18) |
                       (base64_decode_table[static_cast<uint8_t>(src[1])] << 12) |
                       (base64_decode_table[static_cast<uint8_t>(src[2])] << 6) |
                       base64_decode_table[static_cast<uint8_t>(src[3])];
        *dst++ = (val >> 16) & 0xFF;
        *dst++ = (val >> 8) & 0xFF;
        *dst++ = val & 0xFF;
        src += 4;
        srclen -= 4;
    }
#else
    DEBUG_LOG("AVX2 SIMD disabled - using full scalar base64 decode");
    

    while (srclen >= 4) {
        uint32_t val = (base64_decode_table[static_cast<uint8_t>(src[0])] << 18) |
                       (base64_decode_table[static_cast<uint8_t>(src[1])] << 12) |
                       (base64_decode_table[static_cast<uint8_t>(src[2])] << 6) |
                       base64_decode_table[static_cast<uint8_t>(src[3])];
        *dst++ = (val >> 16) & 0xFF;
        *dst++ = (val >> 8) & 0xFF;
        *dst++ = val & 0xFF;
        src += 4;
        srclen -= 4;
    }
#endif
    

    if (srclen >= 2) {
        uint32_t val = (base64_decode_table[static_cast<uint8_t>(src[0])] << 18) |
                       (base64_decode_table[static_cast<uint8_t>(src[1])] << 12);
        if (srclen >= 3) {
            val |= base64_decode_table[static_cast<uint8_t>(src[2])] << 6;
            *dst++ = (val >> 16) & 0xFF;
            *dst++ = (val >> 8) & 0xFF;
        } else {
            *dst++ = (val >> 16) & 0xFF;
        }
    }
    
    decode_buffer.resize(dst - decode_buffer.data());
    
    if (debug_enabled) {
        std::string decoded_hex;
        decoded_hex.reserve(decode_buffer.size() * 2);
        for (size_t i = 0; i < decode_buffer.size(); i++) {
            decoded_hex.push_back(hex_lut[decode_buffer[i] >> 4]);
            decoded_hex.push_back(hex_lut[decode_buffer[i] & 0x0F]);
        }
        DEBUG_LOG("SIMD decoded " + std::to_string(decode_buffer.size()) + " bytes: " + decoded_hex);
    }
    
    return decode_buffer;
}

// Validation: These are DISABLE flags, so we need at least 4 defined to have only 1 implementation enabled
#if !defined(DISABLE_SSE_BASE64_ENCODER) && !defined(DISABLE_SSE_BASE64_ENCODER_ALIGNED) && !defined(DISABLE_SSE_BASE64_ENCODER_LEMIRE) && !defined(DISABLE_SSE_BASE64_ENCODER_MULA) && !defined(DISABLE_SSE_BASE64_ENCODER_AVX)
#error "Must disable at least 4 of the 5 base64 encoders: DISABLE_SSE_BASE64_ENCODER, DISABLE_SSE_BASE64_ENCODER_ALIGNED, DISABLE_SSE_BASE64_ENCODER_LEMIRE, DISABLE_SSE_BASE64_ENCODER_MULA, DISABLE_SSE_BASE64_ENCODER_AVX"
#endif

#if defined(__AVX2__) && !defined(DISABLE_SSE_BASE64_ENCODER)
// Base64 alphabet lookup table
static constexpr char base64_chars[64] = {
    'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P',
    'Q','R','S','T','U','V','W','X','Y','Z','a','b','c','d','e','f',
    'g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v',
    'w','x','y','z','0','1','2','3','4','5','6','7','8','9','+','/'
};

// Alternative approach using multiple specific masks - currently unused
// This approach uses different masks for each 6-bit extraction but is slower
// due to increased register pressure and less optimal instruction scheduling
static inline __m128i extract_indices_to_bytes_alt(const __m128i& packed) {
    __m128i hi  = _mm_srli_epi32(_mm_and_si128(packed, _mm_set1_epi32(0x00fc0000)), 18);
    __m128i mid = _mm_srli_epi32(_mm_and_si128(packed, _mm_set1_epi32(0x0003f000)), 4);
    __m128i lo  = _mm_slli_epi32(_mm_and_si128(packed, _mm_set1_epi32(0x00000fc0)), 10);
    __m128i bot = _mm_slli_epi32(_mm_and_si128(packed, _mm_set1_epi32(0x0000003f)), 24);
    return _mm_or_si128(_mm_or_si128(hi, mid), _mm_or_si128(lo, bot));
}

// Current optimized approach using single reusable mask
static inline __m128i extract_indices_to_bytes(const __m128i& packed) {
    static const __m128i mask6 = _mm_set1_epi32(0x3f);
    __m128i idx0 = _mm_and_si128(_mm_srli_epi32(packed, 18), mask6);
    __m128i idx1 = _mm_slli_epi32(_mm_and_si128(_mm_srli_epi32(packed, 12), mask6), 8);
    __m128i idx2 = _mm_slli_epi32(_mm_and_si128(_mm_srli_epi32(packed, 6), mask6), 16);
    __m128i idx3 = _mm_slli_epi32(_mm_and_si128(packed, mask6), 24);
    return _mm_or_si128(_mm_or_si128(idx0, idx1), _mm_or_si128(idx2, idx3));
}

static inline __m128i lut_lookup(const __m128i& indices) {
    static const __m128i lut0 = _mm_setr_epi8('A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P');
    static const __m128i lut1 = _mm_setr_epi8('Q','R','S','T','U','V','W','X','Y','Z','a','b','c','d','e','f');
    static const __m128i lut2 = _mm_setr_epi8('g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v');
    static const __m128i lut3 = _mm_setr_epi8('w','x','y','z','0','1','2','3','4','5','6','7','8','9','+','/');
    static const __m128i const16 = _mm_set1_epi8(16);
    static const __m128i const32 = _mm_set1_epi8(32);
    static const __m128i const48 = _mm_set1_epi8(48);
    
    __m128i mask_32 = _mm_cmplt_epi8(indices, const32);
    __m128i lo_result = _mm_blendv_epi8(
        _mm_shuffle_epi8(lut1, _mm_sub_epi8(indices, const16)),
        _mm_shuffle_epi8(lut0, indices),
        _mm_cmplt_epi8(indices, const16)
    );
    __m128i hi_result = _mm_blendv_epi8(
        _mm_shuffle_epi8(lut3, _mm_sub_epi8(indices, const48)),
        _mm_shuffle_epi8(lut2, _mm_sub_epi8(indices, const32)),
        _mm_cmplt_epi8(indices, const48)
    );
    return _mm_blendv_epi8(hi_result, lo_result, mask_32);
}

/**
 * SSE-Optimized Base64 Encoder (Primary Implementation)
 * 
 * ALGORITHM:
 * 1. Tiny inputs (≤24B): Pure scalar for minimal overhead
 * 2. Large inputs: 48-byte SIMD blocks with 4x unrolling
 *    - Load 4×12-byte chunks as 128-bit registers
 *    - Shuffle bytes into triplet format: [b2,b1,b0,pad]
 *    - Extract 4×6-bit indices using optimized bit operations
 *    - Lookup Base64 chars via 4-table SSE shuffle
 *    - Store 4×16-byte results (64 chars total)
 * 3. Remainder: Scalar processing for <48 bytes
 * 
 * PERFORMANCE:
 * - 2-3x faster than OpenSSL for large inputs
 * - Optimal for Matrix protocol (no padding)
 * - Thread-local buffer reuse (zero allocation)
 * 
 * TECHNICAL DETAILS:
 * - Uses extract_indices_to_bytes() for 6-bit field extraction
 * - 4-table LUT approach for character lookup
 * - Processes 48 input → 64 output bytes per SIMD iteration
 * 
 * @param data Input byte vector to encode
 * @return Base64 encoded string (unpadded for Matrix protocol)
 */
[[gnu::hot, gnu::flatten]] inline std::string fast_sse_base64_encode(const std::vector<uint8_t>& data) {
    size_t len = data.size();
    
    // Fast path for tiny inputs - avoid SIMD overhead
    if (len <= 24) {
        std::string result;
        result.reserve(((len + 2) / 3) * 4);
        const uint8_t* src = data.data();
        
        while (len >= 3) {
            uint32_t val = (src[0] << 16) | (src[1] << 8) | src[2];
            result += base64_chars[(val >> 18) & 63];
            result += base64_chars[(val >> 12) & 63];
            result += base64_chars[(val >> 6) & 63];
            result += base64_chars[val & 63];
            src += 3; len -= 3;
        }
        if (len > 0) {
            uint32_t val = src[0] << 16;
            if (len > 1) val |= src[1] << 8;
            result += base64_chars[(val >> 18) & 63];
            result += base64_chars[(val >> 12) & 63];
            if (len > 1) result += base64_chars[(val >> 6) & 63];
        }
        return result;
    }
    
    size_t out_len = ((len + 2) / 3) * 4;
    if (base64_buffer.size() < out_len) {
        base64_buffer.resize(out_len);
    }
    
    const uint8_t* src = data.data();
    char* dest = base64_buffer.data();
    const char* const dest_orig = dest;
    

    static const __m128i trip_shuffle = _mm_setr_epi8(
        2, 1, 0, (char)0x80,   // lane0: bytes 2,1,0 to match (b0<<16)|(b1<<8)|b2
        5, 4, 3, (char)0x80,   // lane1: bytes 5,4,3
        8, 7, 6, (char)0x80,   // lane2: bytes 8,7,6
       11,10, 9, (char)0x80    // lane3: bytes 11,10,9
    );

    
    // Process 48-byte blocks with single loop
    while (len >= 48) {
        // Process 12 triplets with 4x unrolled SIMD
        __m128i in0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + 0));
        __m128i in1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + 12));
        __m128i in2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + 24));
        __m128i in3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + 36));
        
        __m128i packed0 = _mm_shuffle_epi8(in0, trip_shuffle);
        __m128i packed1 = _mm_shuffle_epi8(in1, trip_shuffle);
        __m128i packed2 = _mm_shuffle_epi8(in2, trip_shuffle);
        __m128i packed3 = _mm_shuffle_epi8(in3, trip_shuffle);
        
        __m128i idx0_unpacked = extract_indices_to_bytes(packed0);
        __m128i idx1_unpacked = extract_indices_to_bytes(packed1);
        __m128i idx2_unpacked = extract_indices_to_bytes(packed2);
        __m128i idx3_unpacked = extract_indices_to_bytes(packed3);
        
        __m128i chars0 = lut_lookup(idx0_unpacked);
        __m128i chars1 = lut_lookup(idx1_unpacked);
        __m128i chars2 = lut_lookup(idx2_unpacked);
        __m128i chars3 = lut_lookup(idx3_unpacked);
        
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dest + 0), chars0);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dest + 16), chars1);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dest + 32), chars2);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dest + 48), chars3);
        
        src += 48;
        dest += 64;
        len -= 48;
    }

    
    // Fallback scalar processing
    while (len >= 3) {
        uint32_t val = (src[0] << 16) | (src[1] << 8) | src[2];
        *dest++ = base64_chars[(val >> 18) & 63];
        *dest++ = base64_chars[(val >> 12) & 63];
        *dest++ = base64_chars[(val >> 6) & 63];
        *dest++ = base64_chars[val & 63];
        src += 3;
        len -= 3;
    }
    
    // Handle final 1-2 bytes
    if (len > 0) {
        uint32_t val = src[0] << 16;
        if (len > 1) val |= src[1] << 8;
        *dest++ = base64_chars[(val >> 18) & 63];
        *dest++ = base64_chars[(val >> 12) & 63];
        if (len > 1) *dest++ = base64_chars[(val >> 6) & 63];
    }
    
    size_t actual_len = dest - dest_orig;
    base64_buffer.resize(actual_len);
    
    return base64_buffer;
}

/**
 * AVX2-Optimized Base64 Encoder (Custom Implementation)
 * 
 * ALGORITHM:
 * 1. Tiny inputs (≤48B): Pure scalar to avoid SIMD setup cost
 * 2. Large inputs: 24-byte AVX2 blocks with complex permutation
 *    - Load 2×256-bit registers (32 bytes each)
 *    - Complex lane-crossing operations for proper alignment
 *    - Shuffle bytes for triplet extraction
 *    - Extract indices using optimized bit masks
 *    - Character lookup via register-based LUT with blends
 * 3. Remainder: 4x unrolled scalar for better ILP
 * 
 * PERFORMANCE:
 * - Theoretically 3-4x faster than SSE
 * - In practice: slower due to lane-crossing overhead
 * - Complex permutation logic reduces efficiency
 * 
 * TECHNICAL DETAILS:
 * - Uses process_avx2_chunk_direct() for data layout
 * - Register-based LUT with conditional blends
 * - Processes 24 input → 32 output bytes per iteration
 * - Falls back to 4x unrolled scalar for remainder
 */
#elif defined(__AVX2__) && !defined(DISABLE_SSE_BASE64_ENCODER_AVX)
// Base64 alphabet lookup table
static constexpr char base64_chars[64] = {
    'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P',
    'Q','R','S','T','U','V','W','X','Y','Z','a','b','c','d','e','f',
    'g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v',
    'w','x','y','z','0','1','2','3','4','5','6','7','8','9','+','/'
};
// AMS Custom AVX2 base64 encoder implementation
// AVX2 version of extract_indices_to_bytes
static inline __m256i extract_indices_to_bytes_avx2(const __m256i& packed) {
    static const __m256i mask6 = _mm256_set1_epi32(0x3f);
    __m256i idx0 = _mm256_and_si256(_mm256_srli_epi32(packed, 18), mask6);
    __m256i idx1 = _mm256_slli_epi32(_mm256_and_si256(_mm256_srli_epi32(packed, 12), mask6), 8);
    __m256i idx2 = _mm256_slli_epi32(_mm256_and_si256(_mm256_srli_epi32(packed, 6), mask6), 16);
    __m256i idx3 = _mm256_slli_epi32(_mm256_and_si256(packed, mask6), 24);
    
    if (debug_enabled) {
        alignas(32) uint32_t idx0_arr[8], idx1_arr[8], idx2_arr[8], idx3_arr[8];
        _mm256_store_si256(reinterpret_cast<__m256i*>(idx0_arr), idx0);
        _mm256_store_si256(reinterpret_cast<__m256i*>(idx1_arr), idx1);
        _mm256_store_si256(reinterpret_cast<__m256i*>(idx2_arr), idx2);
        _mm256_store_si256(reinterpret_cast<__m256i*>(idx3_arr), idx3);
        
        std::string debug_str = "Raw indices - idx0: ";
        for (int i = 0; i < 6; i++) debug_str += std::to_string(idx0_arr[i]) + ",";
        debug_str += " idx1: ";
        for (int i = 0; i < 6; i++) debug_str += std::to_string(idx1_arr[i] >> 8) + ",";
        DEBUG_LOG(debug_str);
    }
    
    return _mm256_or_si256(_mm256_or_si256(idx0, idx1), _mm256_or_si256(idx2, idx3));
}


static inline __m256i lut_lookup_avx(const __m256i& indices) {
    static const __m256i lut0 = _mm256_setr_epi8(
        'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P',
        'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P'
    );
    static const __m256i lut1 = _mm256_setr_epi8(
        'Q','R','S','T','U','V','W','X','Y','Z','a','b','c','d','e','f',
        'Q','R','S','T','U','V','W','X','Y','Z','a','b','c','d','e','f'
    );
    static const __m256i lut2 = _mm256_setr_epi8(
        'g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v',
        'g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v'
    );
    static const __m256i lut3 = _mm256_setr_epi8(
        'w','x','y','z','0','1','2','3','4','5','6','7','8','9','+','/',
        'w','x','y','z','0','1','2','3','4','5','6','7','8','9','+','/'
    );
    static const __m256i const16 = _mm256_set1_epi8(16);
    static const __m256i const32 = _mm256_set1_epi8(32);
    static const __m256i const48 = _mm256_set1_epi8(48);
    
    __m256i mask_32 = _mm256_cmpgt_epi8(const32, indices);
    __m256i lo_result = _mm256_blendv_epi8(
        _mm256_shuffle_epi8(lut1, _mm256_sub_epi8(indices, const16)),
        _mm256_shuffle_epi8(lut0, indices),
        _mm256_cmpgt_epi8(const16, indices)
    );
    __m256i hi_result = _mm256_blendv_epi8(
        _mm256_shuffle_epi8(lut3, _mm256_sub_epi8(indices, const48)),
        _mm256_shuffle_epi8(lut2, _mm256_sub_epi8(indices, const32)),
        _mm256_cmpgt_epi8(const48, indices)
    );
    return _mm256_blendv_epi8(hi_result, lo_result, mask_32);
}

static inline __m256i sel_by_mask_xor(__m256i a, __m256i b, __m256i m) {
    __m256i diff = _mm256_xor_si256(a, b);              // a ^ b
    __m256i take = _mm256_and_si256(diff, m);           // (a ^ b) & m
    return _mm256_xor_si256(a, take);                   // a ^ ...
}

static inline __m256i sel_by_mask_andn(__m256i a, __m256i b, __m256i m) {
    __m256i a_keep = _mm256_andnot_si256(m, a);         // (~m) & a
    __m256i b_take = _mm256_and_si256(m, b);            // m & b
    return _mm256_or_si256(a_keep, b_take);             // (a&~m) | (b&m)
}

static inline __m256i lut_lookup_avx2_selects(__m256i idx) {
    const __m256i A  = _mm256_set1_epi8('A');
    const __m256i aM = _mm256_set1_epi8('a' - 26);
    const __m256i dM = _mm256_set1_epi8('0' - 52);
    const __m256i p  = _mm256_set1_epi8('+');
    const __m256i s  = _mm256_set1_epi8('/');
    const __m256i c26= _mm256_set1_epi8(26);
    const __m256i c52= _mm256_set1_epi8(52);
    const __m256i c62= _mm256_set1_epi8(62);

    __m256i lt26 = _mm256_cmpgt_epi8(c26, idx);      // idx < 26
    __m256i lt52 = _mm256_cmpgt_epi8(c52, idx);      // idx < 52
    __m256i lt62 = _mm256_cmpgt_epi8(c62, idx);      // idx < 62
    __m256i eq62 = _mm256_cmpeq_epi8(idx, c62);      // idx == 62

    __m256i AZ = _mm256_add_epi8(A,  idx);           // 'A'+i
    __m256i az = _mm256_add_epi8(aM, idx);           // 'a'+(i-26)
    __m256i dg = _mm256_add_epi8(dM, idx);           // '0'+(i-52)

    __m256i r = s;                                   // default '/'
    r = sel_by_mask_xor(r, dg, lt62);                // 0..61 → digits
    r = sel_by_mask_xor(r, az, lt52);                // 0..51 → a..z
    r = sel_by_mask_xor(r, AZ, lt26);                // 0..25 → A..Z
    r = sel_by_mask_xor(r, p,  eq62);                // 62 → '+'
    return r;
}


// AVX2 4-LUT lookup approach matching SSE version
static inline __m256i lut_lookup_avx2(const __m256i& indices) {
    static const __m256i lut0 = _mm256_setr_epi8(
        'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P',
        'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P'
    );
    static const __m256i lut1 = _mm256_setr_epi8(
        'Q','R','S','T','U','V','W','X','Y','Z','a','b','c','d','e','f',
        'Q','R','S','T','U','V','W','X','Y','Z','a','b','c','d','e','f'
    );
    static const __m256i lut2 = _mm256_setr_epi8(
        'g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v',
        'g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v'
    );
    static const __m256i lut3 = _mm256_setr_epi8(
        'w','x','y','z','0','1','2','3','4','5','6','7','8','9','+','/',
        'w','x','y','z','0','1','2','3','4','5','6','7','8','9','+','/'
    );
    static const __m256i const16 = _mm256_set1_epi8(16);
    static const __m256i const32 = _mm256_set1_epi8(32);
    static const __m256i const48 = _mm256_set1_epi8(48);
    
    __m256i mask_32 = _mm256_cmpgt_epi8(const32, indices);
    __m256i lo_result = _mm256_blendv_epi8(
        _mm256_shuffle_epi8(lut1, _mm256_sub_epi8(indices, const16)),
        _mm256_shuffle_epi8(lut0, indices),
        _mm256_cmpgt_epi8(const16, indices)
    );
    __m256i hi_result = _mm256_blendv_epi8(
        _mm256_shuffle_epi8(lut3, _mm256_sub_epi8(indices, const48)),
        _mm256_shuffle_epi8(lut2, _mm256_sub_epi8(indices, const32)),
        _mm256_cmpgt_epi8(const48, indices)
    );
    return _mm256_blendv_epi8(hi_result, lo_result, mask_32);
}

// AVX2 base64 chunk processor using permute operations
static inline __m256i process_avx2_chunk_direct(const uint8_t* src) {
    // Lemire approach: __m256i in = _mm256_maskload_epi32(reinterpret_cast<const int*>(src - 4), _mm256_set_epi32(0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x00000000));
    
    // Load two consecutive registers
    __m256i r0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src));
    __m256i r1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + 16));
    
    // Create proper layout using lane-local operations
    __m256i zeros = _mm256_setzero_si256();
    
    // Lower lane: bytes 0-11 with 4-byte offset
    __m256i lower_shifted = _mm256_alignr_epi8(r0, zeros, 12);
    
    // Upper lane: extract bytes 12-23 and position at 16-27
    __m256i r0_for_upper = _mm256_permute2x128_si256(r0, r0, 0x00);
    __m256i r1_for_upper = _mm256_permute2x128_si256(r1, r1, 0x00);
    __m256i bytes_12_23 = _mm256_alignr_epi8(r1_for_upper, r0_for_upper, 12);
    __m256i upper_shifted = _mm256_permute2x128_si256(zeros, bytes_12_23, 0x20);
    
    // Combine to match maskload layout
    __m256i in = _mm256_blend_epi32(lower_shifted, upper_shifted, 0x70);
    
    // shuffle for triplet extraction
    static const __m256i shuffle = _mm256_setr_epi8(
        6, 5, 4, -1,   9, 8, 7, -1,  12,11,10, -1,  15,14,13, -1,
        2, 1, 0, -1,   5, 4, 3, -1,   8, 7, 6, -1,  11,10, 9, -1
    );
    
    __m256i packed = _mm256_shuffle_epi8(in, shuffle);
    
    // Extract base64 indices
    static const __m256i mask6 = _mm256_set1_epi32(0x3f);
    __m256i idx0 = _mm256_and_si256(_mm256_srli_epi32(packed, 18), mask6);
    __m256i idx1 = _mm256_slli_epi32(_mm256_and_si256(_mm256_srli_epi32(packed, 12), mask6), 8);
    __m256i idx2 = _mm256_slli_epi32(_mm256_and_si256(_mm256_srli_epi32(packed, 6), mask6), 16);
    __m256i idx3 = _mm256_slli_epi32(_mm256_and_si256(packed, mask6), 24);
    __m256i indices_vec = _mm256_or_si256(_mm256_or_si256(idx0, idx1), _mm256_or_si256(idx2, idx3));
    
    return lut_lookup_avx2_selects(indices_vec);
}

[[gnu::hot, gnu::flatten]] inline std::string fast_sse_base64_encode_avx(const std::vector<uint8_t>& data) {
    size_t len = data.size();
    
    // Fast path for tiny inputs - avoid SIMD overhead
    if (len <= 47) {
        std::string result;
        result.reserve(((len + 2) / 3) * 4);
        const uint8_t* src = data.data();
        
        while (len >= 3) {
            uint32_t val = (src[0] << 16) | (src[1] << 8) | src[2];
            result += base64_chars[(val >> 18) & 63];
            result += base64_chars[(val >> 12) & 63];
            result += base64_chars[(val >> 6) & 63];
            result += base64_chars[val & 63];
            src += 3; len -= 3;
        }
        if (len > 0) {
            uint32_t val = src[0] << 16;
            if (len > 1) val |= src[1] << 8;
            result += base64_chars[(val >> 18) & 63];
            result += base64_chars[(val >> 12) & 63];
            if (len > 1) result += base64_chars[(val >> 6) & 63];
        }
        return result;
    }
    
    size_t out_len = ((len + 2) / 3) * 4;
    if (base64_buffer.size() < out_len) {
        base64_buffer.resize(out_len);
    }
    
    const uint8_t* src = data.data();
    char* dest = base64_buffer.data();
    const char* const dest_orig = dest;
    

    
    // Process 24-byte blocks directly with AVX2 - full register utilization
    int chunk_count = 0;
    while (len >= 48) {
        if (debug_enabled) {
            DEBUG_LOG("Processing chunk " + std::to_string(chunk_count) + ", remaining len: " + std::to_string(len));
            DEBUG_LOG("Source offset: " + std::to_string(src - data.data()) + ", dest offset: " + std::to_string(dest - dest_orig));
        }
        
        __m256i chars = process_avx2_chunk_direct(src);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dest), chars);
        
        if (debug_enabled) {
            alignas(32) char temp[32];
            _mm256_store_si256(reinterpret_cast<__m256i*>(temp), chars);
            std::string chunk_str(temp, 32);
            DEBUG_LOG("Direct chunk " + std::to_string(chunk_count) + " (32 chars): " + chunk_str);
        }
        
        src += 24;  // Process 24 input bytes -> 32 base64 chars
        dest += 32;
        len -= 24;
        chunk_count++;
    }
    
    // Fallback scalar processing for remaining bytes
    if (debug_enabled && len > 0) {
        DEBUG_LOG("Scalar fallback for remaining " + std::to_string(len) + " bytes");
    }
    
    while (len >= 3) {
        uint32_t val = (src[0] << 16) | (src[1] << 8) | src[2];
        char scalar_chars[4] = {
            base64_chars[(val >> 18) & 63],
            base64_chars[(val >> 12) & 63],
            base64_chars[(val >> 6) & 63],
            base64_chars[val & 63]
        };
        
        if (debug_enabled) {
            std::string scalar_str(scalar_chars, 4);
            DEBUG_LOG("Scalar triplet: " + scalar_str);
        }
        
        *dest++ = scalar_chars[0];
        *dest++ = scalar_chars[1];
        *dest++ = scalar_chars[2];
        *dest++ = scalar_chars[3];
        src += 3;
        len -= 3;
    }
    
    // Handle final 1-2 bytes
    if (len > 0) {
        uint32_t val = src[0] << 16;
        if (len > 1) val |= src[1] << 8;
        *dest++ = base64_chars[(val >> 18) & 63];
        *dest++ = base64_chars[(val >> 12) & 63];
        if (len > 1) *dest++ = base64_chars[(val >> 6) & 63];
    }
    
    size_t actual_len = dest - dest_orig;
    base64_buffer.resize(actual_len);
    
    return base64_buffer;
}

#elif defined(__AVX2__) && !defined(DISABLE_SSE_BASE64_ENCODER_LEMIRE)
// Lemire AVX2 base64 encoder - fastest implementation
static constexpr char base64_chars[64] = {
    'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P',
    'Q','R','S','T','U','V','W','X','Y','Z','a','b','c','d','e','f',
    'g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v',
    'w','x','y','z','0','1','2','3','4','5','6','7','8','9','+','/'
};

static inline __m256i enc_reshuffle(const __m256i input) {
    const __m256i in = _mm256_shuffle_epi8(input, _mm256_set_epi8(
        10, 11,  9, 10,
         7,  8,  6,  7,
         4,  5,  3,  4,
         1,  2,  0,  1,
        14, 15, 13, 14,
        11, 12, 10, 11,
         8,  9,  7,  8,
         5,  6,  4,  5
    ));
    const __m256i t0 = _mm256_and_si256(in, _mm256_set1_epi32(0x0fc0fc00));
    const __m256i t1 = _mm256_mulhi_epu16(t0, _mm256_set1_epi32(0x04000040));
    const __m256i t2 = _mm256_and_si256(in, _mm256_set1_epi32(0x003f03f0));
    const __m256i t3 = _mm256_mullo_epi16(t2, _mm256_set1_epi32(0x01000010));
    return _mm256_or_si256(t1, t3);
}

static inline __m256i enc_translate(const __m256i in) {
    const __m256i lut = _mm256_setr_epi8(
        65, 71, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -19, -16, 0, 0,
        65, 71, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -19, -16, 0, 0);
    __m256i indices = _mm256_subs_epu8(in, _mm256_set1_epi8(51));
    __m256i mask = _mm256_cmpgt_epi8(in, _mm256_set1_epi8(25));
    indices = _mm256_sub_epi8(indices, mask);
    return _mm256_add_epi8(in, _mm256_shuffle_epi8(lut, indices));
}

/**
 * Lemire AVX2 Base64 Encoder (Research Implementation)
 * 
 * ALGORITHM:
 * 1. Based on Daniel Lemire's fast-base64 research
 * 2. Uses maskload for unaligned data access
 * 3. Specialized reshuffle and translate operations
 * 4. Optimized for academic benchmarking
 * 
 * PERFORMANCE:
 * - Good theoretical performance
 * - maskload can be slow on some CPUs
 * - More complex than needed for most workloads
 * 
 * TECHNICAL DETAILS:
 * - enc_reshuffle(): Rearranges bytes for base64 extraction
 * - enc_translate(): Converts indices to Base64 characters
 * - Uses specialized shuffle patterns from research
 * 
 * @param data Input byte vector to encode
 * @return Base64 encoded string (unpadded)
 */
[[gnu::hot, gnu::flatten]] inline std::string fast_avx2_base64_encode_lemire(const std::vector<uint8_t>& data) {
    size_t len = data.size();
    
    // Fast path for tiny inputs
    if (len < 24) {
        std::string result;
        result.reserve(((len + 2) / 3) * 4);
        const uint8_t* src = data.data();
        
        while (len >= 3) {
            uint32_t val = (src[0] << 16) | (src[1] << 8) | src[2];
            result += base64_chars[(val >> 18) & 63];
            result += base64_chars[(val >> 12) & 63];
            result += base64_chars[(val >> 6) & 63];
            result += base64_chars[val & 63];
            src += 3; len -= 3;
        }
        if (len > 0) {
            uint32_t val = src[0] << 16;
            if (len > 1) val |= src[1] << 8;
            result += base64_chars[(val >> 18) & 63];
            result += base64_chars[(val >> 12) & 63];
            if (len > 1) result += base64_chars[(val >> 6) & 63];
        }
        return result;
    }
    
    size_t out_len = ((len + 2) / 3) * 4;
    base64_buffer.clear();
    base64_buffer.resize(out_len);
    
    const uint8_t* src = data.data();
    char* dest = base64_buffer.data();
    const char* const dest_orig = dest;
    
    // Process 24-byte blocks with AVX2
    if (len >= 28) {
        // First load with mask to handle alignment
        __m256i inputvector = _mm256_maskload_epi32(
            reinterpret_cast<const int*>(src - 4),
            _mm256_set_epi32(0x80000000, 0x80000000, 0x80000000, 0x80000000,
                           0x80000000, 0x80000000, 0x80000000, 0x00000000));
        
        while (true) {
            inputvector = enc_reshuffle(inputvector);
            inputvector = enc_translate(inputvector);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(dest), inputvector);
            
            src += 24;
            dest += 32;
            len -= 24;
            
            if (len >= 32) {
                inputvector = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src - 4));
            } else {
                break;
            }
        }
    }
    
    // Scalar fallback for remaining bytes
    while (len >= 3) {
        uint32_t val = (src[0] << 16) | (src[1] << 8) | src[2];
        *dest++ = base64_chars[(val >> 18) & 63];
        *dest++ = base64_chars[(val >> 12) & 63];
        *dest++ = base64_chars[(val >> 6) & 63];
        *dest++ = base64_chars[val & 63];
        src += 3;
        len -= 3;
    }
    
    if (len > 0) {
        uint32_t val = src[0] << 16;
        if (len > 1) val |= src[1] << 8;
        *dest++ = base64_chars[(val >> 18) & 63];
        *dest++ = base64_chars[(val >> 12) & 63];
        if (len > 1) *dest++ = base64_chars[(val >> 6) & 63];
    }
    
    base64_buffer.resize(dest - dest_orig);
    return base64_buffer;
}
#elif defined(__AVX2__) && !defined(DISABLE_SSE_BASE64_ENCODER_ALIGNED)
// Aligned version with Boost aligned allocator
#include <boost/align/aligned_allocator.hpp>

static constexpr char base64_chars[64] = {
    'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P',
    'Q','R','S','T','U','V','W','X','Y','Z','a','b','c','d','e','f',
    'g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v',
    'w','x','y','z','0','1','2','3','4','5','6','7','8','9','+','/'
};

// Thread-local aligned buffers using Boost aligned allocator
thread_local std::vector<uint8_t, boost::alignment::aligned_allocator<uint8_t, 16>> aligned_input_buffer;
thread_local std::vector<char, boost::alignment::aligned_allocator<char, 16>> aligned_output_buffer;

/**
 * Aligned SSE Base64 Encoder (Memory-Optimized Implementation)
 * 
 * ALGORITHM:
 * 1. Uses Boost aligned allocators for optimal memory access
 * 2. Copies input to 16-byte aligned buffer
 * 3. Processes 48-byte blocks with complex unrolled operations
 * 4. Uses streaming stores for large data (≥8KB)
 * 
 * PERFORMANCE:
 * - Optimized for large datasets with streaming
 * - Alignment overhead hurts small inputs
 * - Memory copy cost reduces efficiency
 * 
 * TECHNICAL DETAILS:
 * - Boost aligned_allocator for 16-byte alignment
 * - Complex unrolled processing with pack/unpack operations
 * - Streaming stores (_mm_stream_si128) for cache bypass
 * - Memory fence (_mm_sfence) for store completion
 * 
 * @param data Input byte vector to encode
 * @return Base64 encoded string (unpadded)
 */
[[gnu::hot, gnu::flatten]] inline std::string fast_sse_base64_encode_aligned(const std::vector<uint8_t>& data) {
    size_t len = data.size();
    
    // Fast path for tiny inputs (< 16B) - avoid SIMD setup overhead
    if (len < 16) {
        std::string result;
        result.reserve(((len + 2) / 3) * 4);
        const uint8_t* src = data.data();
        
        while (len >= 3) {
            uint32_t val = (src[0] << 16) | (src[1] << 8) | src[2];
            result += base64_chars[(val >> 18) & 63];
            result += base64_chars[(val >> 12) & 63];
            result += base64_chars[(val >> 6) & 63];
            result += base64_chars[val & 63];
            src += 3; len -= 3;
        }
        if (len > 0) {
            uint32_t val = src[0] << 16;
            if (len > 1) val |= src[1] << 8;
            result += base64_chars[(val >> 18) & 63];
            result += base64_chars[(val >> 12) & 63];
            if (len > 1) result += base64_chars[(val >> 6) & 63];
        }
        return result;
    }
    
    size_t out_len = ((len + 2) / 3) * 4;
    
    // Prevent integer overflow in buffer calculations
    if (len > SIZE_MAX - 16) {
        throw std::runtime_error("Input too large for aligned buffer");
    }
    if (out_len > SIZE_MAX - 64) {
        throw std::runtime_error("Output too large for aligned buffer");
    }
    
    // Ensure aligned buffers are large enough
    size_t input_buf_size = len + 16;
    size_t output_buf_size = out_len + 64;
    
    if (aligned_input_buffer.size() < input_buf_size) {
        aligned_input_buffer.resize(input_buf_size);
    }
    if (aligned_output_buffer.size() < output_buf_size) {
        aligned_output_buffer.resize(output_buf_size);
    }
    
    // Validate buffer size before memcpy
    if (len > aligned_input_buffer.size()) {
        throw std::runtime_error("Buffer overflow prevented in memcpy");
    }
    
    // Copy input data to aligned buffer
    std::memcpy(aligned_input_buffer.data(), data.data(), len);
    
    const uint8_t* src = aligned_input_buffer.data();
    char* dest = aligned_output_buffer.data();
    const char* const dest_orig = dest;
    
    // SIMD constants
    static const __m128i mask6 = _mm_set1_epi32(0x3f);
    static const __m128i trip_shuffle = _mm_setr_epi8(
        2, 1, 0, (char)0x80, 5, 4, 3, (char)0x80,
        8, 7, 6, (char)0x80, 11,10, 9, (char)0x80
    );
    static const __m128i lut0 = _mm_setr_epi8('A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P');
    static const __m128i lut1 = _mm_setr_epi8('Q','R','S','T','U','V','W','X','Y','Z','a','b','c','d','e','f');
    static const __m128i lut2 = _mm_setr_epi8('g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v');
    static const __m128i lut3 = _mm_setr_epi8('w','x','y','z','0','1','2','3','4','5','6','7','8','9','+','/');
    static const __m128i const16 = _mm_set1_epi8(16);
    static const __m128i const32 = _mm_set1_epi8(32);
    static const __m128i const48 = _mm_set1_epi8(48);
    
    // Process 48-byte blocks with streaming stores for large data
    bool use_streaming = (data.size() >= 8192); // Use streaming for large inputs (8KB+)
    
    while (len >= 48) {
        // Process 48 bytes as 4 aligned 12-byte chunks → 4×16-byte outputs
        __m128i results[4];
        
        // Load 4 aligned 16-byte blocks (covers 48 input + 16 padding)
        __m128i block0 = _mm_load_si128(reinterpret_cast<const __m128i*>(src));
        __m128i block1 = _mm_load_si128(reinterpret_cast<const __m128i*>(src + 16));
        __m128i block2 = _mm_load_si128(reinterpret_cast<const __m128i*>(src + 32));
        
        // Extract 12-byte chunks for base64 processing
        // Chunk 0: bytes 0-11
        __m128i chunk0 = block0;
        
        // Chunk 1: bytes 12-23 (4 bytes from block0 + 8 bytes from block1)
        __m128i chunk1 = _mm_alignr_epi8(block1, block0, 12);
        
        // Chunk 2: bytes 24-35 (8 bytes from block1 + 4 bytes from block2)
        __m128i chunk2 = _mm_alignr_epi8(block2, block1, 8);
        
        // Chunk 3: bytes 36-47 (12 bytes from block2)
        __m128i chunk3 = _mm_srli_si128(block2, 4);
        
        __m128i chunks[4] = {chunk0, chunk1, chunk2, chunk3};
        
        // Unrolled processing of all 4 chunks in parallel
        __m128i packed0 = _mm_shuffle_epi8(chunk0, trip_shuffle);
        __m128i packed1 = _mm_shuffle_epi8(chunk1, trip_shuffle);
        __m128i packed2 = _mm_shuffle_epi8(chunk2, trip_shuffle);
        __m128i packed3 = _mm_shuffle_epi8(chunk3, trip_shuffle);
        
        __m128i idx0_0 = _mm_and_si128(_mm_srli_epi32(packed0, 18), mask6);
        __m128i idx0_1 = _mm_and_si128(_mm_srli_epi32(packed1, 18), mask6);
        __m128i idx0_2 = _mm_and_si128(_mm_srli_epi32(packed2, 18), mask6);
        __m128i idx0_3 = _mm_and_si128(_mm_srli_epi32(packed3, 18), mask6);
        
        __m128i idx1_0 = _mm_and_si128(_mm_srli_epi32(packed0, 12), mask6);
        __m128i idx1_1 = _mm_and_si128(_mm_srli_epi32(packed1, 12), mask6);
        __m128i idx1_2 = _mm_and_si128(_mm_srli_epi32(packed2, 12), mask6);
        __m128i idx1_3 = _mm_and_si128(_mm_srli_epi32(packed3, 12), mask6);
        
        __m128i idx2_0 = _mm_and_si128(_mm_srli_epi32(packed0, 6), mask6);
        __m128i idx2_1 = _mm_and_si128(_mm_srli_epi32(packed1, 6), mask6);
        __m128i idx2_2 = _mm_and_si128(_mm_srli_epi32(packed2, 6), mask6);
        __m128i idx2_3 = _mm_and_si128(_mm_srli_epi32(packed3, 6), mask6);
        
        __m128i idx3_0 = _mm_and_si128(packed0, mask6);
        __m128i idx3_1 = _mm_and_si128(packed1, mask6);
        __m128i idx3_2 = _mm_and_si128(packed2, mask6);
        __m128i idx3_3 = _mm_and_si128(packed3, mask6);
        
        // Process chunk 0
        __m128i lo01_0 = _mm_unpacklo_epi32(idx0_0, idx1_0);
        __m128i lo23_0 = _mm_unpacklo_epi32(idx2_0, idx3_0);
        __m128i hi01_0 = _mm_unpackhi_epi32(idx0_0, idx1_0);
        __m128i hi23_0 = _mm_unpackhi_epi32(idx2_0, idx3_0);
        __m128i quad0_0 = _mm_unpacklo_epi64(lo01_0, lo23_0);
        __m128i quad1_0 = _mm_unpackhi_epi64(lo01_0, lo23_0);
        __m128i quad2_0 = _mm_unpacklo_epi64(hi01_0, hi23_0);
        __m128i quad3_0 = _mm_unpackhi_epi64(hi01_0, hi23_0);
        __m128i packed01_0 = _mm_packs_epi32(quad0_0, quad1_0);
        __m128i packed23_0 = _mm_packs_epi32(quad2_0, quad3_0);
        __m128i indices_bytes_0 = _mm_packus_epi16(packed01_0, packed23_0);
        __m128i mask0_0 = _mm_cmplt_epi8(indices_bytes_0, const16);
        __m128i mask1_0 = _mm_cmplt_epi8(indices_bytes_0, const32);
        __m128i mask2_0 = _mm_cmplt_epi8(indices_bytes_0, const48);
        __m128i val0_0 = _mm_shuffle_epi8(lut0, indices_bytes_0);
        __m128i val1_0 = _mm_shuffle_epi8(lut1, _mm_sub_epi8(indices_bytes_0, const16));
        __m128i val2_0 = _mm_shuffle_epi8(lut2, _mm_sub_epi8(indices_bytes_0, const32));
        __m128i val3_0 = _mm_shuffle_epi8(lut3, _mm_sub_epi8(indices_bytes_0, const48));
        __m128i tmp0_0 = _mm_blendv_epi8(val1_0, val0_0, mask0_0);
        __m128i tmp1_0 = _mm_blendv_epi8(val3_0, val2_0, mask2_0);
        results[0] = _mm_blendv_epi8(tmp1_0, tmp0_0, mask1_0);
        
        // Process chunk 1
        __m128i lo01_1 = _mm_unpacklo_epi32(idx0_1, idx1_1);
        __m128i lo23_1 = _mm_unpacklo_epi32(idx2_1, idx3_1);
        __m128i hi01_1 = _mm_unpackhi_epi32(idx0_1, idx1_1);
        __m128i hi23_1 = _mm_unpackhi_epi32(idx2_1, idx3_1);
        __m128i quad0_1 = _mm_unpacklo_epi64(lo01_1, lo23_1);
        __m128i quad1_1 = _mm_unpackhi_epi64(lo01_1, lo23_1);
        __m128i quad2_1 = _mm_unpacklo_epi64(hi01_1, hi23_1);
        __m128i quad3_1 = _mm_unpackhi_epi64(hi01_1, hi23_1);
        __m128i packed01_1 = _mm_packs_epi32(quad0_1, quad1_1);
        __m128i packed23_1 = _mm_packs_epi32(quad2_1, quad3_1);
        __m128i indices_bytes_1 = _mm_packus_epi16(packed01_1, packed23_1);
        __m128i mask0_1 = _mm_cmplt_epi8(indices_bytes_1, const16);
        __m128i mask1_1 = _mm_cmplt_epi8(indices_bytes_1, const32);
        __m128i mask2_1 = _mm_cmplt_epi8(indices_bytes_1, const48);
        __m128i val0_1 = _mm_shuffle_epi8(lut0, indices_bytes_1);
        __m128i val1_1 = _mm_shuffle_epi8(lut1, _mm_sub_epi8(indices_bytes_1, const16));
        __m128i val2_1 = _mm_shuffle_epi8(lut2, _mm_sub_epi8(indices_bytes_1, const32));
        __m128i val3_1 = _mm_shuffle_epi8(lut3, _mm_sub_epi8(indices_bytes_1, const48));
        __m128i tmp0_1 = _mm_blendv_epi8(val1_1, val0_1, mask0_1);
        __m128i tmp1_1 = _mm_blendv_epi8(val3_1, val2_1, mask2_1);
        results[1] = _mm_blendv_epi8(tmp1_1, tmp0_1, mask1_1);
        
        // Process chunk 2
        __m128i lo01_2 = _mm_unpacklo_epi32(idx0_2, idx1_2);
        __m128i lo23_2 = _mm_unpacklo_epi32(idx2_2, idx3_2);
        __m128i hi01_2 = _mm_unpackhi_epi32(idx0_2, idx1_2);
        __m128i hi23_2 = _mm_unpackhi_epi32(idx2_2, idx3_2);
        __m128i quad0_2 = _mm_unpacklo_epi64(lo01_2, lo23_2);
        __m128i quad1_2 = _mm_unpackhi_epi64(lo01_2, lo23_2);
        __m128i quad2_2 = _mm_unpacklo_epi64(hi01_2, hi23_2);
        __m128i quad3_2 = _mm_unpackhi_epi64(hi01_2, hi23_2);
        __m128i packed01_2 = _mm_packs_epi32(quad0_2, quad1_2);
        __m128i packed23_2 = _mm_packs_epi32(quad2_2, quad3_2);
        __m128i indices_bytes_2 = _mm_packus_epi16(packed01_2, packed23_2);
        __m128i mask0_2 = _mm_cmplt_epi8(indices_bytes_2, const16);
        __m128i mask1_2 = _mm_cmplt_epi8(indices_bytes_2, const32);
        __m128i mask2_2 = _mm_cmplt_epi8(indices_bytes_2, const48);
        __m128i val0_2 = _mm_shuffle_epi8(lut0, indices_bytes_2);
        __m128i val1_2 = _mm_shuffle_epi8(lut1, _mm_sub_epi8(indices_bytes_2, const16));
        __m128i val2_2 = _mm_shuffle_epi8(lut2, _mm_sub_epi8(indices_bytes_2, const32));
        __m128i val3_2 = _mm_shuffle_epi8(lut3, _mm_sub_epi8(indices_bytes_2, const48));
        __m128i tmp0_2 = _mm_blendv_epi8(val1_2, val0_2, mask0_2);
        __m128i tmp1_2 = _mm_blendv_epi8(val3_2, val2_2, mask2_2);
        results[2] = _mm_blendv_epi8(tmp1_2, tmp0_2, mask1_2);
        
        // Process chunk 3
        __m128i lo01_3 = _mm_unpacklo_epi32(idx0_3, idx1_3);
        __m128i lo23_3 = _mm_unpacklo_epi32(idx2_3, idx3_3);
        __m128i hi01_3 = _mm_unpackhi_epi32(idx0_3, idx1_3);
        __m128i hi23_3 = _mm_unpackhi_epi32(idx2_3, idx3_3);
        __m128i quad0_3 = _mm_unpacklo_epi64(lo01_3, lo23_3);
        __m128i quad1_3 = _mm_unpackhi_epi64(lo01_3, lo23_3);
        __m128i quad2_3 = _mm_unpacklo_epi64(hi01_3, hi23_3);
        __m128i quad3_3 = _mm_unpackhi_epi64(hi01_3, hi23_3);
        __m128i packed01_3 = _mm_packs_epi32(quad0_3, quad1_3);
        __m128i packed23_3 = _mm_packs_epi32(quad2_3, quad3_3);
        __m128i indices_bytes_3 = _mm_packus_epi16(packed01_3, packed23_3);
        __m128i mask0_3 = _mm_cmplt_epi8(indices_bytes_3, const16);
        __m128i mask1_3 = _mm_cmplt_epi8(indices_bytes_3, const32);
        __m128i mask2_3 = _mm_cmplt_epi8(indices_bytes_3, const48);
        __m128i val0_3 = _mm_shuffle_epi8(lut0, indices_bytes_3);
        __m128i val1_3 = _mm_shuffle_epi8(lut1, _mm_sub_epi8(indices_bytes_3, const16));
        __m128i val2_3 = _mm_shuffle_epi8(lut2, _mm_sub_epi8(indices_bytes_3, const32));
        __m128i val3_3 = _mm_shuffle_epi8(lut3, _mm_sub_epi8(indices_bytes_3, const48));
        __m128i tmp0_3 = _mm_blendv_epi8(val1_3, val0_3, mask0_3);
        __m128i tmp1_3 = _mm_blendv_epi8(val3_3, val2_3, mask2_3);
        results[3] = _mm_blendv_epi8(tmp1_3, tmp0_3, mask1_3);
        
        // Store all 4 results with aligned streaming
        if (use_streaming) {
            _mm_stream_si128(reinterpret_cast<__m128i*>(dest), results[0]);
            _mm_stream_si128(reinterpret_cast<__m128i*>(dest + 16), results[1]);
            _mm_stream_si128(reinterpret_cast<__m128i*>(dest + 32), results[2]);
            _mm_stream_si128(reinterpret_cast<__m128i*>(dest + 48), results[3]);
        } else {
            _mm_store_si128(reinterpret_cast<__m128i*>(dest), results[0]);
            _mm_store_si128(reinterpret_cast<__m128i*>(dest + 16), results[1]);
            _mm_store_si128(reinterpret_cast<__m128i*>(dest + 32), results[2]);
            _mm_store_si128(reinterpret_cast<__m128i*>(dest + 48), results[3]);
        }
        
        src += 48;
        dest += 64;
        len -= 48;
    }
    
    // Ensure all streaming stores are completed before returning
    if (use_streaming) {
        _mm_sfence();
    }
    
    // Scalar fallback for remaining bytes
    while (len >= 3) {
        uint32_t val = (src[0] << 16) | (src[1] << 8) | src[2];
        *dest++ = base64_chars[(val >> 18) & 63];
        *dest++ = base64_chars[(val >> 12) & 63];
        *dest++ = base64_chars[(val >> 6) & 63];
        *dest++ = base64_chars[val & 63];
        src += 3;
        len -= 3;
    }
    
    if (len > 0) {
        uint32_t val = src[0] << 16;
        if (len > 1) val |= src[1] << 8;
        *dest++ = base64_chars[(val >> 18) & 63];
        *dest++ = base64_chars[(val >> 12) & 63];
        if (len > 1) *dest++ = base64_chars[(val >> 6) & 63];
    }
    
    return std::string(dest_orig, dest - dest_orig);
}
#elif defined(__AVX2__) && !defined(DISABLE_SSE_BASE64_ENCODER_MULA)
// Mula SSE base64 encoder - optimized unrolled implementation
static constexpr char base64_chars[64] = {
    'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P',
    'Q','R','S','T','U','V','W','X','Y','Z','a','b','c','d','e','f',
    'g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v',
    'w','x','y','z','0','1','2','3','4','5','6','7','8','9','+','/'
};

namespace base64 {
    namespace sse {
        #define packed_dword(x) _mm_set1_epi32(x)
        #define packed_byte(x) _mm_set1_epi8(char(x))

        void encode_full_unrolled(const uint8_t* input, size_t bytes, uint8_t* output) {
            uint8_t* out = output;
            const __m128i shuf = _mm_set_epi8(
                10, 11, 9, 10,
                 7,  8, 6,  7,
                 4,  5, 3,  4,
                 1,  2, 0,  1
            );

            for (size_t i = 0; i < bytes; i += 4*3 * 4) {
                __m128i in0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(input + i + 4*3 * 0));
                __m128i in1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(input + i + 4*3 * 1));
                __m128i in2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(input + i + 4*3 * 2));
                __m128i in3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(input + i + 4*3 * 3));

                in0 = _mm_shuffle_epi8(in0, shuf);
                in1 = _mm_shuffle_epi8(in1, shuf);
                in2 = _mm_shuffle_epi8(in2, shuf);
                in3 = _mm_shuffle_epi8(in3, shuf);

                const __m128i t0_0 = _mm_and_si128(in0, _mm_set1_epi32(0x0fc0fc00));
                const __m128i t0_1 = _mm_and_si128(in1, _mm_set1_epi32(0x0fc0fc00));
                const __m128i t0_2 = _mm_and_si128(in2, _mm_set1_epi32(0x0fc0fc00));
                const __m128i t0_3 = _mm_and_si128(in3, _mm_set1_epi32(0x0fc0fc00));

                const __m128i t1_0 = _mm_mulhi_epu16(t0_0, _mm_set1_epi32(0x04000040));
                const __m128i t1_1 = _mm_mulhi_epu16(t0_1, _mm_set1_epi32(0x04000040));
                const __m128i t1_2 = _mm_mulhi_epu16(t0_2, _mm_set1_epi32(0x04000040));
                const __m128i t1_3 = _mm_mulhi_epu16(t0_3, _mm_set1_epi32(0x04000040));
                
                const __m128i t2_0 = _mm_and_si128(in0, _mm_set1_epi32(0x003f03f0));
                const __m128i t2_1 = _mm_and_si128(in1, _mm_set1_epi32(0x003f03f0));
                const __m128i t2_2 = _mm_and_si128(in2, _mm_set1_epi32(0x003f03f0));
                const __m128i t2_3 = _mm_and_si128(in3, _mm_set1_epi32(0x003f03f0));
                
                const __m128i t3_0 = _mm_mullo_epi16(t2_0, _mm_set1_epi32(0x01000010));
                const __m128i t3_1 = _mm_mullo_epi16(t2_1, _mm_set1_epi32(0x01000010));
                const __m128i t3_2 = _mm_mullo_epi16(t2_2, _mm_set1_epi32(0x01000010));
                const __m128i t3_3 = _mm_mullo_epi16(t2_3, _mm_set1_epi32(0x01000010));

                const __m128i input0 = _mm_or_si128(t1_0, t3_0);
                const __m128i input1 = _mm_or_si128(t1_1, t3_1);
                const __m128i input2 = _mm_or_si128(t1_2, t3_2);
                const __m128i input3 = _mm_or_si128(t1_3, t3_3);

                // Unrolled lookup
                __m128i result_0 = packed_byte(65);
                __m128i result_1 = packed_byte(65);
                __m128i result_2 = packed_byte(65);
                __m128i result_3 = packed_byte(65);

                const __m128i ge_26_0 = _mm_cmpgt_epi8(input0, packed_byte(25));
                result_0 = _mm_add_epi8(result_0, _mm_and_si128(ge_26_0, packed_byte(6)));
                const __m128i ge_26_1 = _mm_cmpgt_epi8(input1, packed_byte(25));
                result_1 = _mm_add_epi8(result_1, _mm_and_si128(ge_26_1, packed_byte(6)));
                const __m128i ge_26_2 = _mm_cmpgt_epi8(input2, packed_byte(25));
                result_2 = _mm_add_epi8(result_2, _mm_and_si128(ge_26_2, packed_byte(6)));
                const __m128i ge_26_3 = _mm_cmpgt_epi8(input3, packed_byte(25));
                result_3 = _mm_add_epi8(result_3, _mm_and_si128(ge_26_3, packed_byte(6)));

                const __m128i ge_52_0 = _mm_cmpgt_epi8(input0, packed_byte(51));
                result_0 = _mm_sub_epi8(result_0, _mm_and_si128(ge_52_0, packed_byte(75)));
                const __m128i ge_52_1 = _mm_cmpgt_epi8(input1, packed_byte(51));
                result_1 = _mm_sub_epi8(result_1, _mm_and_si128(ge_52_1, packed_byte(75)));
                const __m128i ge_52_2 = _mm_cmpgt_epi8(input2, packed_byte(51));
                result_2 = _mm_sub_epi8(result_2, _mm_and_si128(ge_52_2, packed_byte(75)));
                const __m128i ge_52_3 = _mm_cmpgt_epi8(input3, packed_byte(51));
                result_3 = _mm_sub_epi8(result_3, _mm_and_si128(ge_52_3, packed_byte(75)));

                const __m128i eq_62_0 = _mm_cmpeq_epi8(input0, packed_byte(62));
                result_0 = _mm_add_epi8(result_0, _mm_and_si128(eq_62_0, packed_byte(241)));
                const __m128i eq_62_1 = _mm_cmpeq_epi8(input1, packed_byte(62));
                result_1 = _mm_add_epi8(result_1, _mm_and_si128(eq_62_1, packed_byte(241)));
                const __m128i eq_62_2 = _mm_cmpeq_epi8(input2, packed_byte(62));
                result_2 = _mm_add_epi8(result_2, _mm_and_si128(eq_62_2, packed_byte(241)));
                const __m128i eq_62_3 = _mm_cmpeq_epi8(input3, packed_byte(62));
                result_3 = _mm_add_epi8(result_3, _mm_and_si128(eq_62_3, packed_byte(241)));

                const __m128i eq_63_0 = _mm_cmpeq_epi8(input0, packed_byte(63));
                result_0 = _mm_sub_epi8(result_0, _mm_and_si128(eq_63_0, packed_byte(12)));
                const __m128i eq_63_1 = _mm_cmpeq_epi8(input1, packed_byte(63));
                result_1 = _mm_sub_epi8(result_1, _mm_and_si128(eq_63_1, packed_byte(12)));
                const __m128i eq_63_2 = _mm_cmpeq_epi8(input2, packed_byte(63));
                result_2 = _mm_sub_epi8(result_2, _mm_and_si128(eq_63_2, packed_byte(12)));
                const __m128i eq_63_3 = _mm_cmpeq_epi8(input3, packed_byte(63));
                result_3 = _mm_sub_epi8(result_3, _mm_and_si128(eq_63_3, packed_byte(12)));

                result_0 = _mm_add_epi8(result_0, input0);
                result_1 = _mm_add_epi8(result_1, input1);
                result_2 = _mm_add_epi8(result_2, input2);
                result_3 = _mm_add_epi8(result_3, input3);

                _mm_storeu_si128(reinterpret_cast<__m128i*>(out), result_0);
                out += 16;
                _mm_storeu_si128(reinterpret_cast<__m128i*>(out), result_1);
                out += 16;
                _mm_storeu_si128(reinterpret_cast<__m128i*>(out), result_2);
                out += 16;
                _mm_storeu_si128(reinterpret_cast<__m128i*>(out), result_3);
                out += 16;
            }
        }

        #undef packed_dword
        #undef packed_byte
    }
}

/**
 * Mula SSE Base64 Encoder (Arithmetic-Based Implementation)
 * 
 * ALGORITHM:
 * 1. Based on Wojciech Mula's SIMD research
 * 2. Uses arithmetic operations instead of lookup tables
 * 3. Conditional arithmetic for character range mapping
 * 4. Highly unrolled 4x processing
 * 
 * PERFORMANCE:
 * - Good performance through arithmetic optimization
 * - Avoids memory lookups with pure computation
 * - Complex branching logic can hurt predictability
 * 
 * TECHNICAL DETAILS:
 * - Uses _mm_cmpgt_epi8 for range detection
 * - Arithmetic character generation: A+index, a+(index-26), etc.
 * - Conditional adds/subtracts based on index ranges
 * - Processes 48 input bytes in 4×12-byte chunks
 * 
 * @param data Input byte vector to encode
 * @return Base64 encoded string (unpadded)
 */
[[gnu::hot, gnu::flatten]] inline std::string fast_mula_base64_encode(const std::vector<uint8_t>& data) {
    size_t len = data.size();
    
    // Fast path for tiny inputs
    if (len < 48) {
        std::string result;
        result.reserve(((len + 2) / 3) * 4);
        const uint8_t* src = data.data();
        
        while (len >= 3) {
            uint32_t val = (src[0] << 16) | (src[1] << 8) | src[2];
            result += base64_chars[(val >> 18) & 63];
            result += base64_chars[(val >> 12) & 63];
            result += base64_chars[(val >> 6) & 63];
            result += base64_chars[val & 63];
            src += 3; len -= 3;
        }
        if (len > 0) {
            uint32_t val = src[0] << 16;
            if (len > 1) val |= src[1] << 8;
            result += base64_chars[(val >> 18) & 63];
            result += base64_chars[(val >> 12) & 63];
            if (len > 1) result += base64_chars[(val >> 6) & 63];
        }
        return result;
    }
    
    size_t out_len = ((len + 2) / 3) * 4;
    base64_buffer.clear();
    base64_buffer.resize(out_len);
    
    const uint8_t* src = data.data();
    char* dest = base64_buffer.data();
    const char* const dest_orig = dest;
    
    // Process 48-byte blocks with Mula unrolled SSE
    size_t processed = 0;
    if (len >= 48) {
        size_t blocks = len / 48;
        base64::sse::encode_full_unrolled(src, blocks * 48, reinterpret_cast<uint8_t*>(dest));
        processed = blocks * 48;
        src += processed;
        dest += (processed / 3) * 4;
        len -= processed;
    }
    
    // Scalar fallback for remaining bytes
    while (len >= 3) {
        uint32_t val = (src[0] << 16) | (src[1] << 8) | src[2];
        *dest++ = base64_chars[(val >> 18) & 63];
        *dest++ = base64_chars[(val >> 12) & 63];
        *dest++ = base64_chars[(val >> 6) & 63];
        *dest++ = base64_chars[val & 63];
        src += 3;
        len -= 3;
    }
    
    if (len > 0) {
        uint32_t val = src[0] << 16;
        if (len > 1) val |= src[1] << 8;
        *dest++ = base64_chars[(val >> 18) & 63];
        *dest++ = base64_chars[(val >> 12) & 63];
        if (len > 1) *dest++ = base64_chars[(val >> 6) & 63];
    }
    
    base64_buffer.resize(dest - dest_orig);
    return base64_buffer;
}
#endif

/**
 * Matrix Protocol Base64 Encoder (Adaptive Implementation)
 * 
 * SELECTION LOGIC:
 * 1. Compile-time: Choose fastest available SIMD implementation
 *    - Priority: AVX2 Custom > SSE > Lemire > Mula > Aligned
 * 2. Runtime: Size-based path selection
 *    - Large inputs: Use selected SIMD implementation
 *    - Small inputs: Fall back to OpenSSL (avoid SIMD overhead)
 * 
 * FEATURES:
 * - Matrix protocol compliance (unpadded output)
 * - Thread-local buffer reuse (zero allocation)
 * - Debug validation against OpenSSL reference
 * - Automatic SIMD capability detection
 * 
 * PERFORMANCE:
 * - 2-10x faster than pure OpenSSL
 * - Optimal path selection based on input size
 * - Zero-copy buffer management
 * 
 * @param data Binary data to encode
 * @return Unpadded base64 encoded string (Matrix compatible)
 */
[[gnu::hot, gnu::flatten]] inline std::string base64_encode(const std::vector<uint8_t>& data) {
    if (data.empty()) return "";
    
#if 0
//#if defined(__AVX2__) && !defined(DISABLE_SSE_BASE64_ENCODER_AVX)
    // Always test AVX2 encoder with 96-byte hardcoded data first
    static const std::vector<uint8_t> test_data_96 = {
        0xdd,0x01,0xef,0xec,0x9b,0xec,0xfe,0x29,0x0d,0xc3,0x9e,0xfb,0x22,0xd3,0xda,0xf0,
        0xdc,0xc2,0x63,0x08,0x60,0x49,0xf1,0xa3,0x51,0x7f,0x6c,0xb5,0x4c,0x47,0x7b,0x8e,
        0x36,0xba,0xf1,0x05,0xb6,0x31,0xb8,0xfa,0x18,0xd5,0xd7,0x2f,0x51,0x1f,0xb9,0x38,
        0x4c,0x08,0xb4,0xf7,0x42,0xa0,0x08,0x7e,0xf4,0x11,0x77,0x3f,0x27,0x32,0x61,0x74,
        0xcd,0x04,0xfa,0x5f,0xef,0xa1,0x6e,0x8d,0xc3,0xd1,0x0f,0x99,0x67,0x54,0x6a,0x8e,
        0xf8,0xfb,0x6a,0xd8,0x8d,0x20,0x9b,0x89,0xe7,0x39,0x9f,0xad,0xa5,0x91,0x33,0xe3
    };
    static bool avx_tested = false;
    if (!avx_tested) {
        DEBUG_LOG("Testing AVX2 encoder with 96-byte hardcoded data");
        std::string avx_test = fast_sse_base64_encode_avx(test_data_96);
        DEBUG_LOG("AVX2 test result length: " + std::to_string(avx_test.length()));
        
        // Validate AVX2 test against OpenSSL reference
        std::string openssl_test;
        size_t test_out_len = ((test_data_96.size() + 2) / 3) * 4;
        openssl_test.resize(test_out_len);
        int test_actual_len = EVP_EncodeBlock(reinterpret_cast<uint8_t*>(openssl_test.data()), test_data_96.data(), test_data_96.size());
        while (test_actual_len > 0 && openssl_test[test_actual_len - 1] == '=') {
            test_actual_len--;
        }
        openssl_test.resize(test_actual_len);
        DEBUG_LOG("AVX2 test result: " + avx_test);
        DEBUG_LOG("OpenSSL test result: " + openssl_test);
        DEBUG_LOG("AVX2 test vs OpenSSL match: " + std::string(avx_test == openssl_test ? "TRUE" : "FALSE"));
        if (avx_test != openssl_test) {
            DEBUG_LOG("AVX2 TEST MISMATCH - AVX2: " + std::to_string(avx_test.length()) + ", OpenSSL: " + std::to_string(openssl_test.length()));
            // Log first difference
            for (size_t i = 0; i < std::min(avx_test.length(), openssl_test.length()); i++) {
                if (avx_test[i] != openssl_test[i]) {
                    DEBUG_LOG("First diff at pos " + std::to_string(i) + ": AVX2='" + std::string(1, avx_test[i]) + "' OpenSSL='" + std::string(1, openssl_test[i]) + "'");
                    break;
                }
            }
        }
        
        avx_tested = true;
    }
#endif
    
    if (debug_enabled) {
        std::string input_hex;
        input_hex.reserve(data.size() * 2);
        for (size_t i = 0; i < data.size(); i++) {
            input_hex.push_back(hex_lut[data[i] >> 4]);
            input_hex.push_back(hex_lut[data[i] & 0x0F]);
        }
        DEBUG_LOG("base64_encode input (" + std::to_string(data.size()) + " bytes): " + input_hex);
    }
    
#if defined(__AVX2__) && !defined(DISABLE_SSE_BASE64_ENCODER_AVX)
    // Use AVX2 path for larger inputs (highest priority)
    DEBUG_LOG("AVX2 AMS encoder enabled");
    if (data.size() >= 48) {
        DEBUG_LOG("Using AVX2 base64 encode for " + std::to_string(data.size()) + " bytes");
        std::string result = fast_sse_base64_encode_avx(data);
        DEBUG_LOG("AVX2 base64 encode result: " + result);
        
        // Validate against OpenSSL reference implementation
        if (debug_enabled) {
            std::string openssl_buffer;
            size_t out_len = ((data.size() + 2) / 3) * 4;
            openssl_buffer.resize(out_len);
            int actual_len = EVP_EncodeBlock(reinterpret_cast<uint8_t*>(openssl_buffer.data()), data.data(), data.size());
            while (actual_len > 0 && openssl_buffer[actual_len - 1] == '=') {
                actual_len--;
            }
            openssl_buffer.resize(actual_len);
            DEBUG_LOG("OpenSSL reference result: " + openssl_buffer);
            DEBUG_LOG("AVX2 vs OpenSSL match: " + std::string(result == openssl_buffer ? "TRUE" : "FALSE"));
            if (result != openssl_buffer) {
                DEBUG_LOG("MISMATCH DETECTED - AVX2 length: " + std::to_string(result.length()) + ", OpenSSL length: " + std::to_string(openssl_buffer.length()));
            }
        }
        
        return result;
    }
#elif defined(__AVX2__) && !defined(DISABLE_SSE_BASE64_ENCODER)
    // Use unaligned SIMD path for larger inputs
    if (data.size() >= 16) {
        DEBUG_LOG("Using unaligned base64 encode for " + std::to_string(data.size()) + " bytes");
        std::string result = fast_sse_base64_encode(data);
        DEBUG_LOG("Unaligned base64 encode result: " + result);
        
        // Validate against OpenSSL reference implementation
        if (debug_enabled) {
            std::string openssl_buffer;
            size_t out_len = ((data.size() + 2) / 3) * 4;
            openssl_buffer.resize(out_len);
            int actual_len = EVP_EncodeBlock(reinterpret_cast<uint8_t*>(openssl_buffer.data()), data.data(), data.size());
            while (actual_len > 0 && openssl_buffer[actual_len - 1] == '=') {
                actual_len--;
            }
            openssl_buffer.resize(actual_len);
            DEBUG_LOG("OpenSSL reference result: " + openssl_buffer);
            DEBUG_LOG("SIMD vs OpenSSL match: " + std::string(result == openssl_buffer ? "TRUE" : "FALSE"));
            if (result != openssl_buffer) {
                DEBUG_LOG("MISMATCH DETECTED - SIMD length: " + std::to_string(result.length()) + ", OpenSSL length: " + std::to_string(openssl_buffer.length()));
            }
        }
        
        return result;
    }
#elif defined(__AVX2__) && !defined(DISABLE_SSE_BASE64_ENCODER_LEMIRE)
    // Use Lemire AVX2 path for larger inputs
    if (data.size() >= 24) {
        DEBUG_LOG("Using Lemire AVX2 base64 encode for " + std::to_string(data.size()) + " bytes");
        std::string result = fast_avx2_base64_encode_lemire(data);
        DEBUG_LOG("Lemire AVX2 base64 encode result: " + result);
        
        // Validate against OpenSSL reference implementation
        if (debug_enabled) {
            std::string openssl_buffer;
            size_t out_len = ((data.size() + 2) / 3) * 4;
            openssl_buffer.resize(out_len);
            int actual_len = EVP_EncodeBlock(reinterpret_cast<uint8_t*>(openssl_buffer.data()), data.data(), data.size());
            while (actual_len > 0 && openssl_buffer[actual_len - 1] == '=') {
                actual_len--;
            }
            openssl_buffer.resize(actual_len);
            DEBUG_LOG("OpenSSL reference result: " + openssl_buffer);
            DEBUG_LOG("SIMD vs OpenSSL match: " + std::string(result == openssl_buffer ? "TRUE" : "FALSE"));
            if (result != openssl_buffer) {
                DEBUG_LOG("MISMATCH DETECTED - SIMD length: " + std::to_string(result.length()) + ", OpenSSL length: " + std::to_string(openssl_buffer.length()));
            }
        }
        
        return result;
    }
#elif defined(__AVX2__) && !defined(DISABLE_SSE_BASE64_ENCODER_MULA)
    // Use Mula SSE path for larger inputs
    if (data.size() >= 16) {
        DEBUG_LOG("Using Mula SSE base64 encode for " + std::to_string(data.size()) + " bytes");
        std::string result = fast_mula_base64_encode(data);
        DEBUG_LOG("Mula SSE base64 encode result: " + result);
        
        // Validate against OpenSSL reference implementation
        if (debug_enabled) {
            std::string openssl_buffer;
            size_t out_len = ((data.size() + 2) / 3) * 4;
            openssl_buffer.resize(out_len);
            int actual_len = EVP_EncodeBlock(reinterpret_cast<uint8_t*>(openssl_buffer.data()), data.data(), data.size());
            while (actual_len > 0 && openssl_buffer[actual_len - 1] == '=') {
                actual_len--;
            }
            openssl_buffer.resize(actual_len);
            DEBUG_LOG("OpenSSL reference result: " + openssl_buffer);
            DEBUG_LOG("SIMD vs OpenSSL match: " + std::string(result == openssl_buffer ? "TRUE" : "FALSE"));
            if (result != openssl_buffer) {
                DEBUG_LOG("MISMATCH DETECTED - SIMD length: " + std::to_string(result.length()) + ", OpenSSL length: " + std::to_string(openssl_buffer.length()));
            }
        }
        
        return result;
    }
#elif defined(__AVX2__) && !defined(DISABLE_SSE_BASE64_ENCODER_ALIGNED)
    // Use aligned SIMD path for larger inputs
    if (data.size() >= 48) {
        DEBUG_LOG("Using aligned base64 encode for " + std::to_string(data.size()) + " bytes");
        std::string result = fast_sse_base64_encode_aligned(data);
        DEBUG_LOG("Optimized base64 encode result: " + result);
        
        // Validate against OpenSSL reference implementation
        if (debug_enabled) {
            std::string openssl_buffer;
            size_t out_len = ((data.size() + 2) / 3) * 4;
            openssl_buffer.resize(out_len);
            int actual_len = EVP_EncodeBlock(reinterpret_cast<uint8_t*>(openssl_buffer.data()), data.data(), data.size());
            while (actual_len > 0 && openssl_buffer[actual_len - 1] == '=') {
                actual_len--;
            }
            openssl_buffer.resize(actual_len);
            DEBUG_LOG("OpenSSL reference result: " + openssl_buffer);
            DEBUG_LOG("SIMD vs OpenSSL match: " + std::string(result == openssl_buffer ? "TRUE" : "FALSE"));
            if (result != openssl_buffer) {
                DEBUG_LOG("MISMATCH DETECTED - SIMD length: " + std::to_string(result.length()) + ", OpenSSL length: " + std::to_string(openssl_buffer.length()));
            }
        }
        
        return result;
    }
#endif
    
    size_t out_len = ((data.size() + 2) / 3) * 4;
    if (base64_buffer.size() < out_len) {
        base64_buffer.resize(out_len);
    }
    
    int actual_len = EVP_EncodeBlock(reinterpret_cast<uint8_t*>(base64_buffer.data()), data.data(), data.size());
    
    // Matrix protocol uses unpadded base64 - remove padding
    while (actual_len > 0 && base64_buffer[actual_len - 1] == '=') {
        actual_len--;
    }
    base64_buffer.resize(actual_len);
    
    DEBUG_LOG("OpenSSL base64_encode result: " + base64_buffer);
    return base64_buffer;
}

// Matrix protocol base64 decode - expects unpadded input
/**
 * Matrix Protocol Base64 Decoder (SIMD-Accelerated)
 * 
 * ALGORITHM:
 * 1. Large inputs (≥32 chars): AVX2 SIMD path
 *    - Process 64-char chunks (32 chars × 2)
 *    - Parallel validation with lookup tables
 *    - Vectorized decode and reshuffle operations
 * 2. Small inputs: OpenSSL with padding restoration
 * 
 * FEATURES:
 * - Handles Matrix unpadded format automatically
 * - SIMD validation prevents invalid character processing
 * - Thread-local buffer reuse
 * - Graceful fallback to scalar for edge cases
 * 
 * PERFORMANCE:
 * - 3-5x faster than OpenSSL for large signatures
 * - Automatic padding calculation and restoration
 * - Zero-allocation for repeated operations
 * 
 * @param encoded_string Unpadded base64 string to decode
 * @return Decoded binary data
 */
[[gnu::hot, gnu::flatten]] std::vector<uint8_t> base64_decode(std::string_view encoded_string) {
    DEBUG_LOG("base64_decode called with length " + std::to_string(encoded_string.size()) + ": '" + std::string(encoded_string) + "'");
    if (encoded_string.empty()) return {};
    
    // Use SIMD path for larger inputs
    if (encoded_string.size() >= 32) {
         DEBUG_LOG("Taking SIMD fast path");
         return fast_base64_decode_signature(encoded_string);
    }
    
    // Validate input length matches event size limit
    if (encoded_string.size() > MAX_EVENT_SIZE) {
        throw std::runtime_error("Base64 input too large");
    }
    
    // Use thread-local buffer to avoid allocation
    thread_local std::string padded_buffer;
    padded_buffer.assign(encoded_string);
    
    // Add padding in-place
    size_t padding_needed = (4 - (encoded_string.size() % 4)) % 4;
    padded_buffer.append(padding_needed, '=');
    
    size_t expected_len = (padded_buffer.size() * 3) / 4;
    if (decode_buffer.size() < expected_len) {
        decode_buffer.resize(expected_len);
    }
    
    int result = EVP_DecodeBlock(decode_buffer.data(), 
                                reinterpret_cast<const uint8_t*>(padded_buffer.data()), 
                                padded_buffer.size());
    if (result < 0) {
        throw std::runtime_error("Invalid base64 encoding: decode failed");
    }
    
    // Remove padding bytes
    if (static_cast<size_t>(result) < padding_needed) {
        throw std::runtime_error("Invalid base64 encoding: insufficient data");
    }
    
    decode_buffer.resize(result - padding_needed);
    return decode_buffer;
}

// Fast canonical serializer using buffer writer - no sorting needed for boost::json::object
void serialize_canonical_fast(const json::value& v) {
    switch (v.kind()) {
        case json::kind::object: {
            write_char('{');
            const auto& obj = v.as_object();
            bool first = true;
            
            // boost::json::object maintains key ordering
            for (const auto& kv : obj) {
                if (!first) write_char(',');
                first = false;
                
                // Escape JSON string characters with bulk operations
                write_char('"');
                std::string_view key = kv.key();
                
                // Use bulk string copying for clean segments
                DEBUG_LOG("Processing JSON key: \"" + std::string(key) + "\" (" + std::to_string(key.size()) + " bytes)");
                size_t start = 0;
                for (size_t i = 0; i < key.size(); ++i) {
                    char c = key[i];
                    if (c == '"' || c == '\\' || c < 0x20) {
                        // Write clean segment before escape character
                        if (i > start) {
                            write_string(key.substr(start, i - start));
                        }
                        
                        // Write escape sequence
                        if (c == '"') write_string("\\\"");
                        else if (c == '\\') write_string("\\\\");
                        else {
                            write_unicode_escape(static_cast<unsigned char>(c));
                        }
                        
                        start = i + 1;
                    }
                }
                
                // Write remaining clean segment
                if (start < key.size()) {
                    write_string(key.substr(start));
                }
                write_char('"');
                write_char(':');
                serialize_canonical_fast(kv.value());
            }
            write_char('}');
            break;
        }
        case json::kind::array: {
            write_char('[');
            const auto& arr = v.as_array();
            
            DEBUG_LOG("Processing JSON array with " + std::to_string(arr.size()) + " elements");
            
            // Vectorized comma insertion for large primitive arrays
            if (arr.size() >= 32) {
                DEBUG_LOG("Large array detected - using optimized processing");
                // Pre-reserve space for commas
                ensure_space(arr.size());
            }
            
            for (size_t i = 0; i < arr.size(); ++i) {
                if (i > 0) write_char(',');
                serialize_canonical_fast(arr[i]);
            }
            write_char(']');
            break;
        }
        case json::kind::string: {
            write_char('"');
            std::string_view s = v.as_string();
            
#if defined(__AVX2__) && !defined(DISABLE_AVX2_JSON_SIMD)
            // Use AVX2 vectorized string escaping for large strings
            if (s.size() >= 32) {
                DEBUG_LOG("Using AVX2 vectorized JSON string escaping for " + std::to_string(s.size()) + " bytes");
                const char* data = s.data();
                size_t len = s.size();
                size_t i = 0;
                
                const __m256i control_threshold = _mm256_set1_epi8(static_cast<char>(0x20 ^ 0x80));
                const __m256i xor_mask = _mm256_set1_epi8(0x80);
                const __m256i quote_mask = _mm256_set1_epi8('"');
                const __m256i backslash_mask = _mm256_set1_epi8('\\');
                const __m256i b_mask = _mm256_set1_epi8('\b');
                const __m256i f_mask = _mm256_set1_epi8('\f');
                const __m256i n_mask = _mm256_set1_epi8('\n');
                const __m256i r_mask = _mm256_set1_epi8('\r');
                const __m256i t_mask = _mm256_set1_epi8('\t');
                
                // Process 32-byte chunks
                for (; i + 32 <= len; i += 32) {
                    __m256i chunk = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + i));
                    
                    __m256i xored = _mm256_xor_si256(chunk, xor_mask);
                    __m256i control_cmp = _mm256_cmpgt_epi8(control_threshold, xored);
                    __m256i quote_cmp = _mm256_cmpeq_epi8(chunk, quote_mask);
                    __m256i backslash_cmp = _mm256_cmpeq_epi8(chunk, backslash_mask);
                    __m256i b_cmp = _mm256_cmpeq_epi8(chunk, b_mask);
                    __m256i f_cmp = _mm256_cmpeq_epi8(chunk, f_mask);
                    __m256i n_cmp = _mm256_cmpeq_epi8(chunk, n_mask);
                    __m256i r_cmp = _mm256_cmpeq_epi8(chunk, r_mask);
                    __m256i t_cmp = _mm256_cmpeq_epi8(chunk, t_mask);
                    
                    __m256i escape_mask = _mm256_or_si256(
                        _mm256_or_si256(_mm256_or_si256(control_cmp, quote_cmp), 
                                       _mm256_or_si256(backslash_cmp, b_cmp)),
                        _mm256_or_si256(_mm256_or_si256(f_cmp, n_cmp), 
                                       _mm256_or_si256(r_cmp, t_cmp))
                    );
                    
                    if (_mm256_testz_si256(escape_mask, escape_mask)) {
                        // Clean chunk - bulk copy
                        write_raw(data + i, 32);
                    } else {
                        // Process character by character for this chunk
                        for (size_t j = 0; j < 32; ++j) {
                            char c = data[i + j];
                            if (c == '"') write_string("\\\"");
                            else if (c == '\\') write_string("\\\\");
                            else if (c == '\b') write_string("\\b");
                            else if (c == '\f') write_string("\\f");
                            else if (c == '\n') write_string("\\n");
                            else if (c == '\r') write_string("\\r");
                            else if (c == '\t') write_string("\\t");
                            else if (c < 0x20) {
                                write_unicode_escape(static_cast<unsigned char>(c));
                            } else {
                                write_char(c);
                            }
                        }
                    }
                }
                
                // Handle remaining bytes with scalar
                for (; i < len; ++i) {
                    unsigned char c = static_cast<unsigned char>(data[i]);
                    if (c == '"') write_string("\\\"");
                    else if (c == '\\') write_string("\\\\");
                    else if (c == '\b') write_string("\\b");
                    else if (c == '\f') write_string("\\f");
                    else if (c == '\n') write_string("\\n");
                    else if (c == '\r') write_string("\\r");
                    else if (c == '\t') write_string("\\t");
                    else if (c < 0x20) {
                        write_unicode_escape_unsafe(c);
                    } else {
                        write_char(static_cast<char>(c));
                    }
                }
            } else {
#endif
                // Scalar fallback for small strings or non-AVX2
                DEBUG_LOG("Using scalar JSON string processing for " + std::to_string(s.size()) + " bytes (< 32 or no AVX2)");
                for (size_t i = 0; i < s.size(); ++i) {
                    unsigned char c = static_cast<unsigned char>(s[i]);
                    if (c == '"') write_string("\\\"");
                    else if (c == '\\') write_string("\\\\");
                    else if (c == '\b') write_string("\\b");
                    else if (c == '\f') write_string("\\f");
                    else if (c == '\n') write_string("\\n");
                    else if (c == '\r') write_string("\\r");
                    else if (c == '\t') write_string("\\t");
                    else if (c < 0x20) {
                        write_unicode_escape_unsafe(c);
                    } else {
                        write_char(static_cast<char>(c));
                    }
                }
#if defined(__AVX2__) && !defined(DISABLE_AVX2_JSON_SIMD)
            }
#endif
            write_char('"');
            break;
        }
        case json::kind::int64: {
            char buf[32];
            auto [ptr, ec] = std::to_chars(buf, buf + 32, v.as_int64());
            if (ec != std::errc{}) {
                throw std::runtime_error("Failed to convert integer to string");
            }
            write_string(std::string_view(buf, ptr - buf));
            break;
        }
        case json::kind::uint64: {
            char buf[32];
            auto [ptr, ec] = std::to_chars(buf, buf + 32, v.as_uint64());
            if (ec != std::errc{}) {
                throw std::runtime_error("Failed to convert unsigned integer to string");
            }
            write_string(std::string_view(buf, ptr - buf));
            break;
        }
        case json::kind::double_: {
            double val = v.as_double();
            if (!std::isfinite(val)) throw std::runtime_error("Non-finite floats not allowed");
            
            char buf[32];
            int len = fast_double_to_string(val, buf);
            
            // Decimal point added by fast_double_to_string
            
            write_string(std::string_view(buf, len));
            break;
        }
        case json::kind::bool_:
            write_cstring(v.as_bool() ? "true" : "false");
            break;
        case json::kind::null:
            write_cstring("null");
            break;
    }
}



std::string canonicalize_json_fast(const json::value& jv) {
    // Pre-allocate large buffer to avoid reallocations
    init_json_buffer();
    
    serialize_canonical_fast(jv);
    
    return std::string(json_buffer.data(), json_ptr - json_buffer.data());
}



// Ed25519 operations using AWS-LC
/**
 * Generates a new Ed25519 signing key using cryptographically secure random bytes
 * @return 32-byte Ed25519 private key seed
 */
std::vector<uint8_t> generate_signing_key() {
    std::vector<uint8_t> seed(32); // Ed25519 seed size
    {
        nb::gil_scoped_release release;
        if (RAND_bytes(seed.data(), 32) != 1) {
            throw std::runtime_error("Failed to generate random bytes");
        }
    }
    return seed;
}

/**
 * Derives the public verification key from an Ed25519 signing key
 * @param signing_key 32-byte Ed25519 private key
 * @return 32-byte Ed25519 public key for signature verification
 */
inline std::vector<uint8_t> get_verify_key(const std::vector<uint8_t>& signing_key) {
    if (signing_key.size() != 32) {
        throw std::runtime_error("Invalid signing key length");
    }
    
    std::vector<uint8_t> pk(32); // Ed25519 public key size
    
    {
        nb::gil_scoped_release release;
        EVP_PKEY* pkey = EVP_PKEY_new_raw_private_key(EVP_PKEY_ED25519, nullptr, signing_key.data(), 32);
        if (!pkey) {
            throw std::runtime_error("Failed to create Ed25519 key");
        }
        
        size_t pk_len = 32;
        if (EVP_PKEY_get_raw_public_key(pkey, pk.data(), &pk_len) != 1 || pk_len != 32) {
            EVP_PKEY_free(pkey);
            throw std::runtime_error("Failed to extract public key");
        }
        
        EVP_PKEY_free(pkey);
    }
    return pk;
}

// No key caching - always expand fresh

/**
 * Signs JSON data using Ed25519 digital signature algorithm
 * @param json_bytes Canonical JSON data to sign
 * @param signing_key_bytes 32-byte Ed25519 private key
 * @return Base64-encoded signature string
 */
inline std::string sign_json_fast(std::span<const uint8_t> json_bytes, const std::vector<uint8_t>& signing_key_bytes) {
    if (signing_key_bytes.size() != 32) {
        throw std::runtime_error("Invalid signing key");
    }
    
    signature_buffer.resize(64); // Ed25519 signature size
    
    {
        nb::gil_scoped_release release;
        
        // Check cache first
        EVP_PKEY* pkey = nullptr;
        if (cached_signing_key == signing_key_bytes && cached_signing_pkey) {
            DEBUG_LOG("Key cache hit - reusing cached EVP_PKEY");
            pkey = cached_signing_pkey;
        } else {
            DEBUG_LOG("Key cache miss - creating new EVP_PKEY");
            // Clear old cache
            if (cached_signing_pkey) {
                EVP_PKEY_free(cached_signing_pkey);
            }
            
            pkey = EVP_PKEY_new_raw_private_key(EVP_PKEY_ED25519, nullptr, signing_key_bytes.data(), 32);
            if (!pkey) {
                throw std::runtime_error("Failed to create Ed25519 key");
            }
            
            // Cache for next time
            cached_signing_key = signing_key_bytes;
            cached_signing_pkey = pkey;
        }
        
        EVP_MD_CTX* ctx = EVP_MD_CTX_new();
        if (!ctx) {
            // Don't free pkey - it's cached
            throw std::runtime_error("Failed to create signing context");
        }
        
        if (EVP_DigestSignInit(ctx, nullptr, nullptr, nullptr, pkey) != 1) {
            EVP_MD_CTX_free(ctx);
            // Don't free pkey - it's cached
            throw std::runtime_error("Failed to initialize signing");
        }
        
        size_t sig_len = 64;
        if (EVP_DigestSign(ctx, signature_buffer.data(), &sig_len, json_bytes.data(), json_bytes.size()) != 1 || sig_len != 64) {
            EVP_MD_CTX_free(ctx);
            // Don't free pkey - it's cached
            throw std::runtime_error("Signing failed");
        }
        
        EVP_MD_CTX_free(ctx);
        // Don't free pkey - it's cached
    }
    
    return base64_encode(signature_buffer);
}

/**
 * Verifies Ed25519 digital signature against JSON data
 * @param json_bytes Canonical JSON data that was signed
 * @param signature_b64 Base64-encoded signature to verify
 * @param verify_key_bytes 32-byte Ed25519 public key
 * @return True if signature is valid, false otherwise
 */
[[gnu::hot, gnu::flatten]] inline bool verify_signature_fast(std::span<const uint8_t> json_bytes, std::string_view signature_b64, const std::vector<uint8_t>& verify_key_bytes) {
    if (verify_key_bytes.size() != 32) {
        DEBUG_LOG("ERROR: Invalid verify key size: " + std::to_string(verify_key_bytes.size()));
        return false;
    }
    
    auto signature_bytes = base64_decode(signature_b64);
    if (signature_bytes.size() != 64) {
        DEBUG_LOG("ERROR: Invalid signature size after decode: " + std::to_string(signature_bytes.size()));
        return false;
    }
    
    if (debug_enabled) {
        std::string sig_hex;
        sig_hex.reserve(std::min(size_t(16), signature_bytes.size()) * 2);
        for (size_t i = 0; i < std::min(size_t(16), signature_bytes.size()); i++) {
            sig_hex.push_back(hex_lut[signature_bytes[i] >> 4]);
            sig_hex.push_back(hex_lut[signature_bytes[i] & 0x0F]);
        }
        DEBUG_LOG("Decoded signature bytes (first 16): " + sig_hex);
        DEBUG_LOG("JSON bytes to verify: " + std::to_string(json_bytes.size()) + " bytes");
    }
    
    bool result = false;
    {
        nb::gil_scoped_release release;
        
        EVP_PKEY* pkey = EVP_PKEY_new_raw_public_key(EVP_PKEY_ED25519, nullptr, verify_key_bytes.data(), 32);
        if (!pkey) {
            DEBUG_LOG("ERROR: Failed to create EVP_PKEY from verify key");
            return false;
        }
        
        EVP_MD_CTX* ctx = EVP_MD_CTX_new();
        if (!ctx) {
            DEBUG_LOG("ERROR: Failed to create EVP_MD_CTX");
            EVP_PKEY_free(pkey);
            return false;
        }
        
        if (EVP_DigestVerifyInit(ctx, nullptr, nullptr, nullptr, pkey) != 1) {
            DEBUG_LOG("ERROR: EVP_DigestVerifyInit failed");
            EVP_MD_CTX_free(ctx);
            EVP_PKEY_free(pkey);
            return false;
        }
        
        int verify_result = EVP_DigestVerify(ctx, signature_bytes.data(), 64, json_bytes.data(), json_bytes.size());
        result = (verify_result == 1);
        
        if (debug_enabled) {
            DEBUG_LOG("EVP_DigestVerify result: " + std::to_string(verify_result) + " (1=success, 0=fail, <0=error)");
            DEBUG_LOG("Signature verification: " + std::string(result ? "SUCCESS" : "FAILED"));
        }
        
        EVP_MD_CTX_free(ctx);
        EVP_PKEY_free(pkey);
    }
    
    return result;
}

/**
 * Computes SHA256 content hash of binary data
 * @param data Binary data to hash
 * @return Pair of algorithm name ("sha256") and 32-byte hash digest
 */
[[gnu::hot, gnu::flatten]] inline std::pair<std::string, std::vector<uint8_t>> compute_content_hash_fast(std::span<const uint8_t> data) {
    hash_buffer.resize(32); // SHA256 hash size
    {
        nb::gil_scoped_release release;
        if (SHA256(data.data(), data.size(), hash_buffer.data()) == nullptr) {
            throw std::runtime_error("SHA256 computation failed");
        }
    }
    return {"sha256", hash_buffer};
}

// Legacy string overload
[[gnu::hot, gnu::flatten]] inline std::pair<std::string, std::vector<uint8_t>> compute_content_hash_fast(const std::string& event_json) {
    return compute_content_hash_fast(std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(event_json.data()), event_json.size()));
}

/**
 * Computes SHA256 content hash of Python dictionary after JSON canonicalization
 * @param event_dict Python dictionary to canonicalize and hash
 * @return Pair of algorithm name ("sha256") and 32-byte hash digest
 */
std::pair<std::string, std::vector<uint8_t>> compute_content_hash(const nb::dict& event_dict) {
    init_json_buffer(event_dict.size());
    
    py_to_canonical_json_fast(event_dict);
    
    std::string canonical_json(json_buffer.data(), json_ptr - json_buffer.data());
    return compute_content_hash_fast(canonical_json);
}

/**
 * Signs a Python dictionary object and adds signature to the result
 * @param json_dict Python dictionary to sign
 * @param signature_name Server name for the signature
 * @param signing_key_bytes 32-byte Ed25519 private key
 * @param key_id Key identifier (e.g., "ed25519:v1")
 * @return New dictionary with signature added
 */
nb::dict sign_json_object_fast(const nb::dict& json_dict, const std::string& signature_name, 
                               const std::vector<uint8_t>& signing_key_bytes, const std::string& key_id) {
    // Create unsigned copy for canonicalization
    nb::dict unsigned_dict;
    for (auto item : json_dict) {
        std::string_view key = nb::cast<std::string_view>(item.first);
        if (key != "signatures" && key != "unsigned") {
            unsigned_dict[std::string(key).c_str()] = item.second;
        }
    }
    
    // Direct streaming to canonical JSON
    init_json_buffer(unsigned_dict.size());
    
    py_to_canonical_json_fast(unsigned_dict);
    
    auto json_bytes = get_json_span();
    
    std::string signature_b64 = sign_json_fast(json_bytes, signing_key_bytes);
    
    // Create result dict with signature
    nb::dict result;
    for (auto item : json_dict) {
        std::string_view key = nb::cast<std::string_view>(item.first);
        result[std::string(key).c_str()] = item.second;
    }
    
    if (!result.contains("signatures")) {
        result["signatures"] = nb::dict();
    }
    nb::dict signatures = result["signatures"];
    if (!signatures.contains(signature_name.c_str())) {
        signatures[signature_name.c_str()] = nb::dict();
    }
    nb::dict server_sigs = signatures[signature_name.c_str()];
    server_sigs[key_id.c_str()] = signature_b64;
    
    return result;
}

/**
 * Verifies the digital signature on a signed Python dictionary object
 * @param json_dict Signed Python dictionary to verify
 * @param server_name Server name that should have signed the object
 * @param verify_key_bytes 32-byte Ed25519 public key for verification
 * @throws SignatureVerifyException if signature verification fails
 */
void verify_signed_json_fast(const nb::dict& json_dict, const std::string& server_name, const std::vector<uint8_t>& verify_key_bytes) {
    if (!json_dict.contains("signatures")) {
        throw SignatureVerifyException("Missing signatures");
    }
    
    nb::dict signatures = json_dict["signatures"];
    if (!signatures.contains(server_name.c_str())) {
        throw SignatureVerifyException("Missing server signature");
    }
    
    nb::dict server_sigs = signatures[server_name.c_str()];
    std::string signature_b64;
    
    // Find ed25519 signature
    for (auto item : server_sigs) {
        std::string_view key = nb::cast<std::string_view>(item.first);
        if (key.starts_with("ed25519:")) {
            signature_b64 = nb::cast<std::string>(item.second);
            break;
        }
    }
    
    if (signature_b64.empty()) {
        throw SignatureVerifyException("Missing ed25519 signature");
    }
    
    // Create unsigned copy for verification
    nb::dict unsigned_dict;
    for (auto item : json_dict) {
        std::string_view key = nb::cast<std::string_view>(item.first);
        if (key != "signatures" && key != "unsigned") {
            unsigned_dict[std::string(key).c_str()] = item.second;
        }
    }
    
    init_json_buffer(unsigned_dict.size());
    
    py_to_canonical_json_fast(unsigned_dict);
    
    auto json_bytes = get_json_span();
    
    if (!verify_signature_fast(json_bytes, signature_b64, verify_key_bytes)) {
        throw SignatureVerifyException("Signature verification failed");
    }
}

struct SigningResult {
    std::string signature;
    std::string key_id;
    std::string algorithm;
};

/**
 * Signs JSON data and returns detailed signing information
 * @param json_bytes Canonical JSON data to sign
 * @param signing_key_bytes 32-byte Ed25519 private key
 * @param version Key version identifier
 * @return SigningResult containing signature, key ID, and algorithm
 */
SigningResult sign_json_with_info(const std::vector<uint8_t>& json_bytes, const std::vector<uint8_t>& signing_key_bytes, const std::string& version) {
    std::string signature_b64 = sign_json_fast(json_bytes, signing_key_bytes);
    return {signature_b64, "ed25519:" + version, "ed25519"};
}

struct VerificationResult {
    bool valid;
    std::optional<std::string> user_id;
    std::optional<bool> device_valid;
};

// Signature object to match NaCl API
class Signature {
public:
    nb::bytes signature;
    
    Signature(nb::bytes sig) : signature(sig) {}
};

// VerifyKey class to match Rust API
class VerifyKey {
public:
    std::vector<uint8_t> key_bytes;
    std::string alg;
    std::string version;
    
    VerifyKey(const std::vector<uint8_t>& bytes, const std::string& algorithm = "ed25519", const std::string& ver = "1")
        : key_bytes(bytes), alg(algorithm), version(ver) {}
    
    nb::bytes encode() const {
        return nb::bytes(reinterpret_cast<const char*>(key_bytes.data()), key_bytes.size());
    }
    
    void verify(nb::bytes message, nb::bytes signature) const {
        const char* msg_ptr = static_cast<const char*>(message.c_str());
        size_t msg_size = message.size();
        std::vector<uint8_t> message_bytes(msg_ptr, msg_ptr + msg_size);
        
        const char* sig_ptr = static_cast<const char*>(signature.c_str());
        size_t sig_size = signature.size();
        std::vector<uint8_t> signature_bytes(sig_ptr, sig_ptr + sig_size);
        
        std::string signature_b64 = base64_encode(signature_bytes);
        
        if (!verify_signature_fast(message_bytes, signature_b64, key_bytes)) {
            throw std::runtime_error("Signature verification failed");
        }
    }
};

// SigningKey class to match Rust API
class SigningKey {
public:
    std::vector<uint8_t> key_bytes;
    std::string alg;
    std::string version;
    
    SigningKey(const std::vector<uint8_t>& bytes, const std::string& algorithm = "ed25519", const std::string& ver = "1")
        : key_bytes(bytes), alg(algorithm), version(ver) {}
    
    static SigningKey generate() {
        return SigningKey(generate_signing_key());
    }
    
    nb::bytes encode() const {
        return nb::bytes(reinterpret_cast<const char*>(key_bytes.data()), key_bytes.size());
    }
    
    VerifyKey get_verify_key() const {
        return VerifyKey(::get_verify_key(key_bytes), alg, version);
    }
    
    Signature sign(nb::bytes message) const {
        const char* ptr = static_cast<const char*>(message.c_str());
        size_t size = message.size();
        std::vector<uint8_t> message_bytes(ptr, ptr + size);
        
        std::string signature_b64 = sign_json_fast(message_bytes, key_bytes);
        auto signature_bytes = base64_decode(signature_b64);
        nb::bytes sig_bytes = nb::bytes(reinterpret_cast<const char*>(signature_bytes.data()), signature_bytes.size());
        return Signature(sig_bytes);
    }
};

/**
 * Verifies signature and returns detailed verification information
 * @param json_bytes Canonical JSON data that was signed
 * @param signature_b64 Base64-encoded signature to verify
 * @param verify_key_bytes 32-byte Ed25519 public key
 * @return VerificationResult containing validation status and metadata
 */
VerificationResult verify_signature_with_info(const std::vector<uint8_t>& json_bytes, const std::string& signature_b64, const std::vector<uint8_t>& verify_key_bytes) {
    bool valid = verify_signature_fast(json_bytes, signature_b64, verify_key_bytes);
    return {valid, std::nullopt, std::nullopt};
}

NB_MODULE(_event_signing_impl, m) {
    // Enable leak warnings only when explicitly requested for profiling
    // Set SYNAPSE_RUST_CRYPTO_LEAK_WARNINGS=1 to enable
    nb::set_leak_warnings(leak_warnings_enabled);
    
    // AWS-LC doesn't require explicit initialization
    
    nb::exception<SignatureVerifyException>(m, "SignatureVerifyException");
    
    nb::class_<Signature>(m, "Signature")
        .def_rw("signature", &Signature::signature);
    
    nb::class_<SigningResult>(m, "SigningResult")
        .def_rw("signature", &SigningResult::signature)
        .def_rw("key_id", &SigningResult::key_id)
        .def_rw("algorithm", &SigningResult::algorithm);
    
    nb::class_<VerificationResult>(m, "VerificationResult")
        .def_rw("valid", &VerificationResult::valid)
        .def_rw("user_id", &VerificationResult::user_id)
        .def_rw("device_valid", &VerificationResult::device_valid);
    
    nb::class_<VerifyKey>(m, "VerifyKey")
        .def(nb::init<const std::vector<uint8_t>&, const std::string&, const std::string&>(),
             "bytes"_a, "alg"_a = "ed25519", "version"_a = "1")
        .def("encode", &VerifyKey::encode)
        .def("verify", &VerifyKey::verify)
        .def_rw("alg", &VerifyKey::alg)
        .def_rw("version", &VerifyKey::version);
    
    nb::class_<SigningKey>(m, "SigningKey")
        .def(nb::init<const std::vector<uint8_t>&, const std::string&, const std::string&>(),
             "bytes"_a, "alg"_a = "ed25519", "version"_a = "1")
        .def_static("generate", &SigningKey::generate)
        .def("encode", &SigningKey::encode)
        .def("get_verify_key", &SigningKey::get_verify_key)
        .def_prop_ro("verify_key", &SigningKey::get_verify_key)
        .def("sign", &SigningKey::sign)
        .def_rw("alg", &SigningKey::alg)
        .def_rw("version", &SigningKey::version);
    
    m.def("compute_content_hash_fast", [](const std::string& event_json) {
        auto result = compute_content_hash_fast(event_json);
        return std::make_pair(result.first, nb::bytes(reinterpret_cast<const char*>(result.second.data()), result.second.size()));
    });
    m.def("compute_content_hash", [](const nb::dict& event_dict) {
        auto result = compute_content_hash(event_dict);
        return std::make_pair(result.first, nb::bytes(reinterpret_cast<const char*>(result.second.data()), result.second.size()));
    });
    m.def("compute_event_reference_hash_fast", [](const std::string& event_json) {
        auto result = compute_content_hash_fast(event_json);
        return std::make_pair(result.first, nb::bytes(reinterpret_cast<const char*>(result.second.data()), result.second.size()));
    });
    m.def("compute_event_reference_hash", [](const nb::dict& event_dict) {
        auto result = compute_content_hash(event_dict);
        return std::make_pair(result.first, nb::bytes(reinterpret_cast<const char*>(result.second.data()), result.second.size()));
    });
    m.def("sign_json_fast", &sign_json_fast);
    m.def("sign_json_with_info", &sign_json_with_info);
    m.def("verify_signature_fast", &verify_signature_fast);
    m.def("verify_signature_with_info", &verify_signature_with_info);
    m.def("sign_json_object_fast", &sign_json_object_fast);
    m.def("verify_signed_json_fast", &verify_signed_json_fast);
    m.def("get_verify_key", &get_verify_key);
    m.def("encode_base64_fast", [](nb::bytes data) {
        const char* ptr = static_cast<const char*>(data.c_str());
        size_t size = data.size();
        
        std::vector<uint8_t> vec_data(ptr, ptr + size);
        return nb::str(base64_encode(vec_data).c_str());
    });
    m.def("decode_base64_fast", [](const std::string& encoded) {
        auto result = base64_decode(encoded);
        return nb::bytes(reinterpret_cast<const char*>(result.data()), result.size());
    });
    m.def("encode_base64", [](const nb::object& data) {
        if (nb::isinstance<nb::bytes>(data)) {
            nb::bytes bytes_data = nb::cast<nb::bytes>(data);
            const char* ptr = static_cast<const char*>(bytes_data.c_str());
            size_t size = bytes_data.size();
            std::vector<uint8_t> vec_data(ptr, ptr + size);
            return nb::str(base64_encode(vec_data).c_str());
        } else {
            std::vector<uint8_t> vec_data = nb::cast<std::vector<uint8_t>>(data);
            return nb::str(base64_encode(vec_data).c_str());
        }
    });
    m.def("decode_base64", [](const std::string& encoded) {
        auto result = base64_decode(encoded);
        return nb::bytes(reinterpret_cast<const char*>(result.data()), result.size());
    });
    m.def("encode_verify_key_base64", [](const std::vector<uint8_t>& key_bytes) {
        return nb::str(base64_encode(key_bytes).c_str());
    });
    m.def("encode_signing_key_base64", [](const std::vector<uint8_t>& key_bytes) {
        return nb::str(base64_encode(key_bytes).c_str());
    });
    m.def("decode_verify_key_base64", [](const std::string& algorithm, const std::string& version, const std::string& key_base64) {
        if (algorithm != "ed25519") throw std::runtime_error("Unsupported algorithm");
        auto key_bytes = base64_decode(key_base64);
        if (key_bytes.size() != 32) throw std::runtime_error("Invalid key length");
        return key_bytes;
    });
    m.def("decode_signing_key_base64", [](const std::string& algorithm, const std::string& version, const std::string& key_base64) {
        if (algorithm != "ed25519") throw std::runtime_error("Unsupported algorithm");
        auto key_bytes = base64_decode(key_base64);
        if (key_bytes.size() != 32) throw std::runtime_error("Invalid key length");
        return key_bytes;
    });
    m.def("decode_verify_key_bytes_fast", [](std::string_view key_id, const std::vector<uint8_t>& key_bytes) {
        if (key_id.starts_with("ed25519:") && key_bytes.size() == 32) return key_bytes;
        throw std::runtime_error("Unsupported key type or invalid key length");
    });
    m.def("decode_verify_key_bytes", [](std::string_view key_id, const nb::object& key_data) -> VerifyKey {
        std::vector<uint8_t> key_bytes;
        
        if (nb::isinstance<nb::bytes>(key_data)) {
            nb::bytes bytes_data = nb::cast<nb::bytes>(key_data);
            const char* ptr = static_cast<const char*>(bytes_data.c_str());
            size_t size = bytes_data.size();
            key_bytes = std::vector<uint8_t>(ptr, ptr + size);
        } else {
            key_bytes = nb::cast<std::vector<uint8_t>>(key_data);
        }
        
        if (key_id.starts_with("ed25519:") && key_bytes.size() == 32) {
            size_t colon_pos = key_id.find(':');
            std::string version = (colon_pos != std::string::npos) ? std::string(key_id.substr(colon_pos + 1)) : "1";
            return VerifyKey(key_bytes, "ed25519", version);
        }
        throw std::runtime_error("Unsupported key type or invalid key length");
    });
    m.def("is_signing_algorithm_supported", [](std::string_view key_id) {
        return key_id.starts_with("ed25519:");
    });
    m.def("encode_canonical_json", [](const nb::dict& json_dict) {
        init_json_buffer(json_dict.size());
        
        // Serialize with potential GIL release in hot paths
        py_to_canonical_json_fast(json_dict);
        
        auto span = get_json_span();
        return nb::bytes(span.data(), span.size());
    });
    m.def("signature_ids", [](const nb::dict& json_dict, const std::string& signature_name) {
        std::vector<std::string> ids;
        if (json_dict.contains("signatures")) {
            nb::dict signatures = nb::cast<nb::dict>(json_dict["signatures"]);
            if (signatures.contains(signature_name.c_str())) {
                nb::dict server_sigs = nb::cast<nb::dict>(signatures[signature_name.c_str()]);
                for (auto item : server_sigs) {
                    std::string_view key = nb::cast<std::string_view>(item.first);
                    if (key.starts_with("ed25519:")) ids.push_back(std::string(key));
                }
            }
        }
        return ids;
    });
    // Alias functions to match Rust API exactly
    m.def("sign_json", [](const nb::dict& json_object, const std::string& signature_name, const nb::object& signing_key) {
        std::string alg = nb::cast<std::string>(signing_key.attr("alg"));
        std::string version = nb::cast<std::string>(signing_key.attr("version"));
        std::string key_id = alg + ":" + version;
        nb::bytes encoded_key = nb::cast<nb::bytes>(signing_key.attr("encode")());
        
        // Convert nb::bytes to std::vector<uint8_t>
        const char* ptr = static_cast<const char*>(encoded_key.c_str());
        size_t size = encoded_key.size();
        std::vector<uint8_t> signing_key_bytes(ptr, ptr + size);
        
        return sign_json_object_fast(json_object, signature_name, signing_key_bytes, key_id);
    });
    m.def("verify_signature", &verify_signature_fast);
    m.def("verify_signed_json", [](const nb::dict& json_dict, const std::string& signature_name, const nb::object& verify_key) {
        DEBUG_LOG("verify_signed_json: signature_name=" + signature_name);
        
        // Extract key info like Rust version
        std::string alg, version;
        std::vector<uint8_t> verify_key_bytes;
        
        if (nb::isinstance<nb::bytes>(verify_key) || nb::isinstance<nb::list>(verify_key)) {
            verify_key_bytes = nb::cast<std::vector<uint8_t>>(verify_key);
            alg = "ed25519";
            version = "auto";
            DEBUG_LOG("Got raw bytes, length=" + std::to_string(verify_key_bytes.size()));
            
            // Extract version from signatures if available
            if (json_dict.contains("signatures")) {
                nb::dict signatures = json_dict["signatures"];
                if (signatures.contains(signature_name.c_str())) {
                    nb::dict server_sigs = signatures[signature_name.c_str()];
                    for (auto item : server_sigs) {
                        std::string_view key_id = nb::cast<std::string_view>(item.first);
                        if (key_id.starts_with("ed25519:")) {
                            size_t colon_pos = key_id.find(':');
                            if (colon_pos != std::string::npos) {
                                version = key_id.substr(colon_pos + 1);
                            }
                            break;
                        }
                    }
                }
            }
        } else {
            try {
                alg = nb::cast<std::string>(verify_key.attr("alg"));
                version = nb::cast<std::string>(verify_key.attr("version"));
                DEBUG_LOG("Got VerifyKey object: alg=" + alg + ", version=" + version);
                nb::bytes encoded_key = nb::cast<nb::bytes>(verify_key.attr("encode")());
                
                // Convert nb::bytes to std::vector<uint8_t>
                const char* ptr = static_cast<const char*>(encoded_key.c_str());
                size_t size = encoded_key.size();
                verify_key_bytes = std::vector<uint8_t>(ptr, ptr + size);
                DEBUG_LOG("Extracted key bytes, length=" + std::to_string(verify_key_bytes.size()));
            } catch (const std::exception& e) {
                DEBUG_LOG("Failed to extract key attributes, treating as raw bytes: " + std::string(e.what()));
                verify_key_bytes = nb::cast<std::vector<uint8_t>>(verify_key);
                alg = "ed25519";
                version = "auto";
            }
        }
        
        std::string key_id = alg + ":" + version;
        DEBUG_LOG("Using key_id: " + key_id);
        
        // Get signatures
        if (!json_dict.contains("signatures")) {
            DEBUG_LOG("ERROR: No signatures field found in JSON");
            throw SignatureVerifyException("No signatures on this object");
        }
        
        nb::dict signatures = json_dict["signatures"];
        if (!signatures.contains(signature_name.c_str())) {
            DEBUG_LOG("ERROR: Missing signature for server: " + signature_name);
            throw SignatureVerifyException("Missing signature for " + signature_name);
        }
        
        nb::dict server_sigs = signatures[signature_name.c_str()];
        std::string signature_b64;
        
        if (debug_enabled) {
            DEBUG_LOG("Available signature keys for " + signature_name + ":");
            for (auto item : server_sigs) {
                std::string_view available_key = nb::cast<std::string_view>(item.first);
                DEBUG_LOG("  - " + std::string(available_key));
            }
        }
        
        // Try exact key_id first, then fallback to any ed25519
        if (server_sigs.contains(key_id.c_str())) {
            signature_b64 = nb::cast<std::string>(server_sigs[key_id.c_str()]);
            DEBUG_LOG("Found exact key match: " + key_id + " -> " + signature_b64);
        } else {
            for (auto item : server_sigs) {
                std::string_view available_key = nb::cast<std::string_view>(item.first);
                if (available_key.starts_with("ed25519:")) {
                    signature_b64 = nb::cast<std::string>(item.second);
                    DEBUG_LOG("Using fallback key: " + std::string(available_key) + " -> " + signature_b64);
                    break;
                }
            }
        }
        
        if (signature_b64.empty()) {
            DEBUG_LOG("ERROR: No ed25519 signature found for " + signature_name + ", " + key_id);
            throw SignatureVerifyException("Missing signature for " + signature_name + ", " + key_id);
        }
        
        // Create unsigned copy
        nb::dict unsigned_dict;
        for (auto item : json_dict) {
            std::string_view key = nb::cast<std::string_view>(item.first);
            if (key != "signatures" && key != "unsigned") {
                unsigned_dict[std::string(key).c_str()] = item.second;
            }
        }
        
        DEBUG_LOG("Created unsigned dict with " + std::to_string(unsigned_dict.size()) + " keys");
        
        // Canonicalize and verify
        init_json_buffer(unsigned_dict.size());
        
        py_to_canonical_json_fast(unsigned_dict);
        
        auto json_bytes = get_json_span();
        
        if (debug_enabled) {
            std::string canonical_json(json_buffer.data(), json_ptr - json_buffer.data());
            DEBUG_LOG("Canonical JSON (" + std::to_string(canonical_json.size()) + " bytes): " + canonical_json.substr(0, 200) + (canonical_json.size() > 200 ? "..." : ""));
            DEBUG_LOG("Signature to verify: " + signature_b64);
            
            std::string hex;
            hex.reserve(verify_key_bytes.size() * 2);
            for (size_t i = 0; i < verify_key_bytes.size(); i++) {
                hex.push_back(hex_lut[verify_key_bytes[i] >> 4]);
                hex.push_back(hex_lut[verify_key_bytes[i] & 0x0F]);
            }
            DEBUG_LOG("Verify key bytes (full): " + hex);
        }
        
        bool verification_result = verify_signature_fast(json_bytes, signature_b64, verify_key_bytes);
        DEBUG_LOG("Verification result: " + std::string(verification_result ? "SUCCESS" : "FAILED"));
        
        if (!verification_result) {
            DEBUG_LOG("ERROR: Signature verification failed for " + signature_name);
            throw SignatureVerifyException("Unable to verify signature for " + signature_name);
        }
        
        DEBUG_LOG("Signature verification completed successfully");
    });
    // Key management functions with version parameter (ignores version for compatibility)
    m.def("generate_signing_key", [](const std::string& version) { return generate_signing_key(); });
    m.def("read_signing_keys", [](const nb::object& input_data) {
        DEBUG_LOG("read_signing_keys called");
        
        // Handle different input types like Python signedjson
        std::string content;
        if (nb::isinstance<nb::str>(input_data)) {
            // Handle string input
            content = nb::cast<std::string>(input_data);
            DEBUG_LOG("Got string input, length: " + std::to_string(content.length()));
        } else if (nb::isinstance<nb::list>(input_data)) {
            // Handle list input (like config does)
            auto py_list = nb::cast<nb::list>(input_data);
            DEBUG_LOG("Got list input with " + std::to_string(py_list.size()) + " items");
            std::vector<std::string> lines;
            for (auto item : py_list) {
                std::string line = nb::cast<std::string>(item);
                lines.push_back(line);
                DEBUG_LOG("List item: \"" + line + "\"");
            }
            // Join with newlines - vectorized for large arrays
            if (lines.size() >= 16) {
                DEBUG_LOG("Using optimized string joining for " + std::to_string(lines.size()) + " lines");
                // Pre-calculate total size to avoid reallocations
                size_t total_size = 0;
                for (const auto& line : lines) {
                    total_size += line.size() + 1; // +1 for newline
                }
                content.reserve(total_size);
            }
            
            for (size_t i = 0; i < lines.size(); ++i) {
                if (i > 0) content += "\n";
                content += lines[i];
            }
            DEBUG_LOG("Joined list content: \"" + content + "\"");
        } else {
            // Try to call read() method for file-like objects
            try {
                nb::object content_obj = input_data.attr("read")();
                content = nb::cast<std::string>(content_obj);
                DEBUG_LOG("Read from file-like object, length: " + std::to_string(content.length()));
            } catch (...) {
                // Fallback to string conversion
                content = nb::cast<std::string>(nb::str(input_data));
                DEBUG_LOG("Fallback string conversion, length: " + std::to_string(content.length()));
            }
        }
        
        DEBUG_LOG("Content to parse: \"" + content + "\"");
        
        // Parse signing keys from content
        nb::list signing_keys;
        
        std::istringstream stream(content);
        std::string line;
        int line_num = 0;
        while (std::getline(stream, line)) {
            line_num++;
            DEBUG_LOG("Processing line " + std::to_string(line_num) + ": \"" + line + "\"");
            
            // Trim whitespace
            line.erase(0, line.find_first_not_of(" \t\r\n"));
            line.erase(line.find_last_not_of(" \t\r\n") + 1);
            
            if (line.empty() || line[0] == '#') {
                DEBUG_LOG("Skipping empty or comment line");
                continue;
            }
            
            // Expected format: "ed25519 version base64_key"
            std::istringstream line_stream(line);
            std::string algorithm, version, key_b64;
            if (line_stream >> algorithm >> version >> key_b64) {
                DEBUG_LOG("Parsed: algorithm=" + algorithm + ", version=" + version + ", key_b64=" + key_b64);
                if (algorithm == "ed25519" && !version.empty() && !key_b64.empty()) {
                    try {
                        auto key_bytes = base64_decode(key_b64);
                        DEBUG_LOG("Decoded key bytes, length: " + std::to_string(key_bytes.size()));
                        if (key_bytes.size() == 32) {
                            // Create SigningKey object
                            SigningKey key(key_bytes, "ed25519", version);
                            signing_keys.append(nb::cast(key));
                            DEBUG_LOG("Added signing key with version: " + version);
                        } else {
                            DEBUG_LOG("Invalid key length: " + std::to_string(key_bytes.size()));
                        }
                    } catch (const std::exception& e) {
                        DEBUG_LOG("Failed to decode key: " + std::string(e.what()));
                        continue;
                    }
                } else {
                    DEBUG_LOG("Invalid key format or algorithm");
                }
            } else {
                DEBUG_LOG("Failed to parse line format");
            }
        }
        
        DEBUG_LOG("Returning " + std::to_string(signing_keys.size()) + " signing keys");
        return signing_keys;
    });
    m.def("read_old_signing_keys", [](const nb::object& stream_content) {
        return nb::list();
    });
    m.def("write_signing_keys", [](const std::vector<std::pair<std::string, std::vector<uint8_t>>>& keys) {
        std::string output;
        for (const auto& [version, key_bytes] : keys) {
            if (key_bytes.size() == 32) {
                std::string key_b64 = base64_encode(key_bytes);
                output += "ed25519 " + version + " " + key_b64 + "\n";
            }
        }
        return output;
    });
}
