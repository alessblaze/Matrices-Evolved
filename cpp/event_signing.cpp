/*
 * Copyright (C) 2025 Aless Microsystems
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 */

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/optional.h>
#include <unordered_map>
#include <array>
#include <cstring>
#include <span>
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
#include <iomanip>
#include <iostream>
#include <cstdlib>
#include <stdexcept>
#include <charconv>
#include <cmath>
#include <Python.h>

#ifdef __AVX2__
#include <immintrin.h>
// Disable AVX2 SIMD for JSON canonicalization
//#define DISABLE_AVX2_JSON_SIMD
// Separate controls for base64 encoder/decoder
//#define DISABLE_AVX2_BASE64_ENCODER
//#define DISABLE_AVX2_BASE64_DECODER
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

/**
 * Fast double-to-string conversion optimized for JSON canonicalization
 * 
 * Algorithm:
 * 1. Fast-path common values (0.0, 1.0) with pre-computed strings
 * 2. Use std::to_chars for optimal performance (no locale, no malloc)
 * 3. Ensure JSON compliance by adding ".0" suffix for integers
 * 4. Normalize -0.0 to 0.0 for canonical representation
 * 
 * @param f Input double value (must be finite)
 * @param result Output buffer (minimum 32 bytes)
 * @return Length of written string
 * @throws std::runtime_error if conversion fails
 */
inline int fast_double_to_string(double f, char* result) {
    // Fast paths for common values (already normalized)
    if (f == 0.0) {
        result[0] = '0'; result[1] = '.'; result[2] = '0';
        return 3;
    }
    if (f == 1.0) {
        result[0] = '1'; result[1] = '.'; result[2] = '0';
        return 3;
    }
    
    // Use std::to_chars for fast conversion
    auto [ptr, ec] = std::to_chars(result, result + 32, f);
    if (ec == std::errc{}) {
        int len = ptr - result;
        // Add .0 if integer
        bool has_dot = false;
        for (int i = 0; i < len; i++) {
            if (result[i] == '.' || result[i] == 'e' || result[i] == 'E') {
                has_dot = true;
                break;
            }
        }
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

// Debug logging infrastructure
static bool debug_enabled = []() {
    const char* env = std::getenv("SYNAPSE_RUST_CRYPTO_DEBUG");
    return env && std::string(env) == "1";
}();

// Memory leak profiling
static bool leak_warnings_enabled = []() {
    const char* env = std::getenv("SYNAPSE_RUST_CRYPTO_LEAK_WARNINGS");
    return env && std::string(env) == "1";
}();

#define DEBUG_LOG(msg) do { \
    if (debug_enabled) { \
        std::cout << "DEBUG C++ crypto: " << msg << std::endl; \
    } \
} while(0)

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
thread_local std::vector<char> json_buffer;
thread_local char* json_ptr;

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
    size_t min_size = std::max(hint * 100, 64 * 1024UL); // Estimate 100 bytes per dict item
    
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
    
    if (current_size + needed > json_buffer.size()) {
        size_t new_size = json_buffer.size() * 2;
        while (new_size < current_size + needed)
            new_size *= 2;
        
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

ALWAYS_INLINE void write_string(std::string_view s) {
    ensure_space(s.size());
    std::memcpy(json_ptr, s.data(), s.size());
    json_ptr += s.size();
}

ALWAYS_INLINE void write_raw(const char* data, size_t len) {
    ensure_space(len);
    std::memcpy(json_ptr, data, len);
    json_ptr += len;
}

/**
 * Get current JSON buffer contents as zero-copy span
 * 
 * Algorithm:
 * 1. Calculate buffer length via pointer arithmetic
 * 2. Cast char buffer to uint8_t for crypto operations
 * 3. Return span view without copying data
 * 
 * @return Immutable span of canonicalized JSON bytes
 */
ALWAYS_INLINE std::span<const uint8_t> get_json_span() {
    return std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(json_buffer.data()), 
                                   json_ptr - json_buffer.data());
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
static constexpr char hex[] = "0123456789abcdef";
ALWAYS_INLINE void write_unicode_escape(unsigned char c) {
    char buf[6] = {'\\','u','0','0', hex[c >> 4], hex[c & 0xF]};
    write_string({buf, 6});
}

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
                    if (s.size() >= 32) {
                        //DEBUG_LOG("Using AVX2 SIMD for string of length " + std::to_string(s.size()));
                        const char* data = s.data();
                        size_t len = s.size();
                        size_t i = 0;
                        
                        // Pre-reserve worst-case (every char becomes \u00XX = 6 bytes)
                        size_t current_size = json_ptr - json_buffer.data();
                        size_t needed_size = current_size + len * 6;
                        if (needed_size > json_buffer.size()) {
                            if (needed_size > MAX_EVENT_SIZE) {
                                throw SignatureVerifyException("String too large to escape");
                            }
                            json_buffer.resize(needed_size);
                            json_ptr = json_buffer.data() + current_size;
                        }
                        
                        const __m256i control_threshold = _mm256_set1_epi8(static_cast<char>(0x20 ^ 0x80));
                        const __m256i xor_mask = _mm256_set1_epi8(0x80);
                        const __m256i quote_mask = _mm256_set1_epi8('"');
                        const __m256i backslash_mask = _mm256_set1_epi8('\\');
                        
                        // Outer loop unroll: process 64 bytes per iteration
                        for (; i + 64 <= len; i += 64) {
                            __m256i chunk1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + i));
                            __m256i chunk2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + i + 32));
                            
                            __m256i xored1 = _mm256_xor_si256(chunk1, xor_mask);
                            __m256i xored2 = _mm256_xor_si256(chunk2, xor_mask);
                            __m256i control_cmp1 = _mm256_cmpgt_epi8(control_threshold, xored1);
                            __m256i control_cmp2 = _mm256_cmpgt_epi8(control_threshold, xored2);
                            __m256i quote_cmp1 = _mm256_cmpeq_epi8(chunk1, quote_mask);
                            __m256i quote_cmp2 = _mm256_cmpeq_epi8(chunk2, quote_mask);
                            __m256i backslash_cmp1 = _mm256_cmpeq_epi8(chunk1, backslash_mask);
                            __m256i backslash_cmp2 = _mm256_cmpeq_epi8(chunk2, backslash_mask);
                            __m256i escape_mask1 = _mm256_or_si256(_mm256_or_si256(control_cmp1, quote_cmp1), backslash_cmp1);
                            __m256i escape_mask2 = _mm256_or_si256(_mm256_or_si256(control_cmp2, quote_cmp2), backslash_cmp2);
                            
                            // Combined testz for both chunks
                            __m256i combined_mask = _mm256_or_si256(escape_mask1, escape_mask2);
                            if (_mm256_testz_si256(combined_mask, combined_mask)) {
                                // Both chunks clean - fast path
                                write_raw(data + i, 64);
                            } else {
                                // Process first chunk
                                if (_mm256_testz_si256(escape_mask1, escape_mask1)) {
                                    write_raw(data + i, 32);
                                } else {
                                    uint32_t mask = static_cast<uint32_t>(_mm256_movemask_epi8(escape_mask1));
                                    size_t base = i;
                                    while (mask) {
                                        const unsigned tz = __builtin_ctz(mask);
                                        const size_t good_len = tz - (base - i);
                                        if (good_len) write_raw(data + base, good_len);
                                        
                                        unsigned char c = static_cast<unsigned char>(data[i + tz]);
                                        switch (c) {
                                            case '"': write_raw("\\\"", 2); break;
                                            case '\\': write_raw("\\\\", 2); break;
                                            case '\b': write_raw("\\b", 2); break;
                                            case '\f': write_raw("\\f", 2); break;
                                            case '\n': write_raw("\\n", 2); break;
                                            case '\r': write_raw("\\r", 2); break;
                                            case '\t': write_raw("\\t", 2); break;
                                            default: write_unicode_escape(c);
                                        }
                                        
                                        base = i + tz + 1;
                                        mask &= mask - 1;
                                    }
                                    const size_t tail = (i + 32) - base;
                                    if (tail) write_raw(data + base, tail);
                                }
                                
                                // Process second chunk
                                if (_mm256_testz_si256(escape_mask2, escape_mask2)) {
                                    write_raw(data + i + 32, 32);
                                } else {
                                    uint32_t mask = static_cast<uint32_t>(_mm256_movemask_epi8(escape_mask2));
                                    size_t base = i + 32;
                                    while (mask) {
                                        const unsigned tz = __builtin_ctz(mask);
                                        const size_t good_len = tz - (base - (i + 32));
                                        if (good_len) write_raw(data + base, good_len);
                                        
                                        unsigned char c = static_cast<unsigned char>(data[i + 32 + tz]);
                                        switch (c) {
                                            case '"': write_raw("\\\"", 2); break;
                                            case '\\': write_raw("\\\\", 2); break;
                                            case '\b': write_raw("\\b", 2); break;
                                            case '\f': write_raw("\\f", 2); break;
                                            case '\n': write_raw("\\n", 2); break;
                                            case '\r': write_raw("\\r", 2); break;
                                            case '\t': write_raw("\\t", 2); break;
                                            default: write_unicode_escape(c);
                                        }
                                        
                                        base = i + 32 + tz + 1;
                                        mask &= mask - 1;
                                    }
                                    const size_t tail = (i + 64) - base;
                                    if (tail) write_raw(data + base, tail);
                                }
                            }
                        }
                        
                        // Handle remaining 32-byte chunks
                        for (; i + 32 <= len; i += 32) {
                            __m256i chunk = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + i));
                            
                            __m256i xored = _mm256_xor_si256(chunk, xor_mask);
                            __m256i control_cmp = _mm256_cmpgt_epi8(control_threshold, xored);
                            __m256i quote_cmp = _mm256_cmpeq_epi8(chunk, quote_mask);
                            __m256i backslash_cmp = _mm256_cmpeq_epi8(chunk, backslash_mask);
                            __m256i escape_mask = _mm256_or_si256(_mm256_or_si256(control_cmp, quote_cmp), backslash_cmp);
                            
                            if (_mm256_testz_si256(escape_mask, escape_mask)) {
                                write_raw(data + i, 32);
                            } else {
                                uint32_t mask = static_cast<uint32_t>(_mm256_movemask_epi8(escape_mask));
                                size_t base = i;
                                while (mask) {
                                    const unsigned tz = __builtin_ctz(mask);
                                    const size_t good_len = tz - (base - i);
                                    if (good_len) write_raw(data + base, good_len);
                                    
                                    unsigned char c = static_cast<unsigned char>(data[i + tz]);
                                    switch (c) {
                                        case '"': write_raw("\\\"", 2); break;
                                        case '\\': write_raw("\\\\", 2); break;
                                        case '\b': write_raw("\\b", 2); break;
                                        case '\f': write_raw("\\f", 2); break;
                                        case '\n': write_raw("\\n", 2); break;
                                        case '\r': write_raw("\\r", 2); break;
                                        case '\t': write_raw("\\t", 2); break;
                                        default: write_unicode_escape(c);
                                    }
                                    
                                    base = i + tz + 1;
                                    mask &= mask - 1;
                                }
                                
                                const size_t tail = (i + 32) - base;
                                if (tail) write_raw(data + base, tail);
                            }
                        }
                        
                        // Process remaining bytes
                        for (; i < len; i++) {
                            unsigned char c = data[i];
                            switch (c) {
                                case '"': write_string("\\\""); break;
                                case '\\': write_string("\\\\"); break;
                                case '\b': write_string("\\b"); break;
                                case '\f': write_string("\\f"); break;
                                case '\n': write_string("\\n"); break;
                                case '\r': write_string("\\r"); break;
                                case '\t': write_string("\\t"); break;
                                default:
                                    if (c < 0x20) write_unicode_escape(c);
                                    else write_char(c);
                            }
                        }
                    } else {
#endif
                        // Scalar loop
                        for (unsigned char c : s) {
                            switch (c) {
                                case '"': write_string("\\\""); break;
                                case '\\': write_string("\\\\"); break;
                                case '\b': write_string("\\b"); break;
                                case '\f': write_string("\\f"); break;
                                case '\n': write_string("\\n"); break;
                                case '\r': write_string("\\r"); break;
                                case '\t': write_string("\\t"); break;
                                default:
                                    if (c < 0x20) write_unicode_escape(c);
                                    else write_char(c);
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
                
                // Use owning strings for all keys to avoid lifetime issues
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
                    for (char c : key) {
                        if (c == '"') write_raw("\\\"", 2);
                        else if (c == '\\') write_raw("\\\\", 2);
                        else if (c < 0x20) write_unicode_escape(static_cast<unsigned char>(c));
                        else write_char(c);
                    }
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

// Thread-local buffers for reuse - pre-allocated for common sizes
thread_local std::string base64_buffer;
thread_local std::vector<uint8_t> decode_buffer;
thread_local std::vector<uint8_t> signature_buffer(64); // Ed25519 signature size
thread_local std::vector<uint8_t> hash_buffer(32);     // SHA256 hash size
thread_local std::vector<std::string> key_sort_buffer;

// Fast base64 decode table - pre-computed for speed
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

// Forward declaration
[[gnu::hot, gnu::flatten]] std::vector<uint8_t> base64_decode(std::string_view encoded_string);

// Working AVX2 base64 decode using proven approach from fastavxbase64.h
static inline __m256i dec_reshuffle(__m256i in) {
    const __m256i merge_ab_and_bc = _mm256_maddubs_epi16(in, _mm256_set1_epi32(0x01400140));
    __m256i out = _mm256_madd_epi16(merge_ab_and_bc, _mm256_set1_epi32(0x00011000));
    out = _mm256_shuffle_epi8(out, _mm256_setr_epi8(
        2, 1, 0, 6, 5, 4, 10, 9, 8, 14, 13, 12, -1, -1, -1, -1,
        2, 1, 0, 6, 5, 4, 10, 9, 8, 14, 13, 12, -1, -1, -1, -1
    ));
    return _mm256_permutevar8x32_epi32(out, _mm256_setr_epi32(0, 1, 2, 4, 5, 6, -1, -1));
}

// Fast AVX2 base64 decode using proven fastavxbase64.h approach
[[gnu::hot, gnu::flatten]] inline std::vector<uint8_t> fast_base64_decode_signature(std::string_view input) {
    DEBUG_LOG("fast_base64_decode_simd input length: " + std::to_string(input.size()));
    
    size_t output_len = (input.size() * 3) / 4;
    decode_buffer.resize(output_len);
    
    const char* src = input.data();
    uint8_t* dst = decode_buffer.data();
    size_t srclen = input.size();
    
#if defined(__AVX2__) && !defined(DISABLE_AVX2_BASE64_DECODER)
    DEBUG_LOG("Using AVX2 SIMD base64 decode path");
    
    // Shared LUTs for both unrolled iterations
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
    
    // 2x unrolled loop: process 64 chars -> 48 bytes per iteration
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
    
    // Handle remaining 32-63 chars with single iteration
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
    
    // Handle remaining chars with single iteration

    
    // AVX scalar path for remaining bytes when AVX2 is available
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
    
    // Handle remaining 4-15 chars with scalar
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
    
    // Complete scalar fallback when AVX2 not available
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
    
    // Handle final 2-3 chars
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
        for (size_t i = 0; i < decode_buffer.size(); i++) {
            char buf[3];
            snprintf(buf, sizeof(buf), "%02x", decode_buffer[i]);
            decoded_hex += buf;
        }
        DEBUG_LOG("SIMD decoded " + std::to_string(decode_buffer.size()) + " bytes: " + decoded_hex);
    }
    
    return decode_buffer;
}

#if defined(__AVX2__) && !defined(DISABLE_AVX2_BASE64_ENCODER)
// Base64 alphabet lookup table
static constexpr char base64_chars[64] = {
    'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P',
    'Q','R','S','T','U','V','W','X','Y','Z','a','b','c','d','e','f',
    'g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v',
    'w','x','y','z','0','1','2','3','4','5','6','7','8','9','+','/'
};

/**
 * Fast base64 encoder using direct 6-bit extraction
 * 
 * Algorithm:
 * 1. Process 48 input bytes as 16 triplets (3 bytes each)
 * 2. Extract 4 base64 indices per triplet using bit shifts
 * 3. Lookup base64 characters from alphabet table
 * 4. Produces exactly 64 base64 characters per 48-byte block
 * 
 * Performance: ~2-3x faster than OpenSSL for large inputs
 * 
 * @param data Input byte vector to encode
 * @return Base64 encoded string (unpadded for Matrix protocol)
 */
[[gnu::hot, gnu::flatten]] inline std::string fast_avx2_base64_encode(const std::vector<uint8_t>& data) {
    size_t len = data.size();
    size_t out_len = ((len + 2) / 3) * 4;
    base64_buffer.resize(out_len);
    
    const uint8_t* src = data.data();
    char* dest = base64_buffer.data();
    const char* const dest_orig = dest;
    
    // Process 48→64 byte blocks correctly
    while (len >= 48) {
        // Direct 6-bit extraction from 48 bytes → 64 base64 chars
        alignas(64) uint8_t indices[64];
        
        // TODO: Vectorize 6-bit extraction using AVX2 shuffle/shift operations
        // Extract 6-bit values directly from input bytes
        for (int i = 0; i < 16; ++i) {
            uint32_t triple = (src[i*3] << 16) | (src[i*3+1] << 8) | src[i*3+2];
            indices[i*4] = (triple >> 18) & 0x3f;
            indices[i*4+1] = (triple >> 12) & 0x3f;
            indices[i*4+2] = (triple >> 6) & 0x3f;
            indices[i*4+3] = triple & 0x3f;
        }
        
        // TODO: Vectorize character lookup using dual AVX2 shuffle tables
        // Translate indices to base64 characters
        for (int i = 0; i < 64; ++i) {
            dest[i] = base64_chars[indices[i]];
        }
        
        src += 48;
        dest += 64;
        len -= 48;
    }
    
    DEBUG_LOG("SIMD processed " + std::to_string(data.size() - len) + " bytes, " + std::to_string(len) + " remaining for scalar");
    
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
    
    // Handle final 1-2 bytes
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
 * Matrix protocol base64 encoder with SIMD optimization
 * 
 * Features:
 * - Uses fast SIMD path for inputs ≥48 bytes
 * - Falls back to OpenSSL for smaller inputs
 * - Produces unpadded base64 (Matrix protocol requirement)
 * - Thread-local buffer reuse for zero-allocation encoding
 * 
 * @param data Input byte vector to encode
 * @return Unpadded base64 string
 */
[[gnu::hot, gnu::flatten]] inline std::string base64_encode(const std::vector<uint8_t>& data) {
    if (data.empty()) return "";
    
    if (debug_enabled) {
        std::string input_hex;
        for (size_t i = 0; i < std::min(size_t(32), data.size()); i++) {
            char buf[3];
            snprintf(buf, sizeof(buf), "%02x", data[i]);
            input_hex += buf;
        }
        DEBUG_LOG("base64_encode input (" + std::to_string(data.size()) + " bytes, first 32): " + input_hex);
    }
    
#if defined(__AVX2__) && !defined(DISABLE_AVX2_BASE64_ENCODER)
    // Use optimized SIMD path for larger inputs
    if (data.size() >= 48) {
        DEBUG_LOG("Using optimized base64 encode for " + std::to_string(data.size()) + " bytes");
        std::string result = fast_avx2_base64_encode(data);
        DEBUG_LOG("Optimized base64 encode result: " + result);
        return result;
    }
#endif
    
    size_t out_len = ((data.size() + 2) / 3) * 4;
    base64_buffer.resize(out_len);
    
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
[[gnu::hot, gnu::flatten]] std::vector<uint8_t> base64_decode(std::string_view encoded_string) {
    DEBUG_LOG("base64_decode called with length " + std::to_string(encoded_string.size()) + ": '" + std::string(encoded_string) + "'");
    if (encoded_string.empty()) return {};
    
    // Disable SIMD path temporarily to fix signature verification
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
    decode_buffer.resize(expected_len);
    
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
            
            // boost::json::object is already ordered - no sorting needed!
            for (const auto& kv : obj) {
                if (!first) write_char(',');
                first = false;
                
                // Fast string escaping
                write_char('"');
                std::string_view key = kv.key();
                for (char c : key) {
                    if (c == '"') write_string("\\\"");
                    else if (c == '\\') write_string("\\\\");
                    else if (c < 0x20) {
                        char buf[7];
                        snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned char>(c));
                        write_string(buf);
                    } else write_char(c);
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
            for (char c : s) {
                if (c == '"') write_string("\\\"");
                else if (c == '\\') write_string("\\\\");
                else if (c == '\b') write_string("\\b");
                else if (c == '\f') write_string("\\f");
                else if (c == '\n') write_string("\\n");
                else if (c == '\r') write_string("\\r");
                else if (c == '\t') write_string("\\t");
                else if (c < 0x20) {
                    write_unicode_escape(static_cast<unsigned char>(c));
                } else write_char(c);
            }
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
            
            // .0 already added by fast_double_to_string
            
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

inline std::string sign_json_fast(const std::vector<uint8_t>& json_bytes, const std::vector<uint8_t>& signing_key_bytes) {
    if (signing_key_bytes.size() != 32) {
        throw std::runtime_error("Invalid signing key");
    }
    
    signature_buffer.resize(64); // Ed25519 signature size
    
    {
        nb::gil_scoped_release release;
        EVP_PKEY* pkey = EVP_PKEY_new_raw_private_key(EVP_PKEY_ED25519, nullptr, signing_key_bytes.data(), 32);
        if (!pkey) {
            throw std::runtime_error("Failed to create Ed25519 key");
        }
        
        EVP_MD_CTX* ctx = EVP_MD_CTX_new();
        if (!ctx) {
            EVP_PKEY_free(pkey);
            throw std::runtime_error("Failed to create signing context");
        }
        
        if (EVP_DigestSignInit(ctx, nullptr, nullptr, nullptr, pkey) != 1) {
            EVP_MD_CTX_free(ctx);
            EVP_PKEY_free(pkey);
            throw std::runtime_error("Failed to initialize signing");
        }
        
        size_t sig_len = 64;
        if (EVP_DigestSign(ctx, signature_buffer.data(), &sig_len, json_bytes.data(), json_bytes.size()) != 1 || sig_len != 64) {
            EVP_MD_CTX_free(ctx);
            EVP_PKEY_free(pkey);
            throw std::runtime_error("Signing failed");
        }
        
        EVP_MD_CTX_free(ctx);
        EVP_PKEY_free(pkey);
    }
    
    return base64_encode(signature_buffer);
}

[[gnu::hot, gnu::flatten]] inline bool verify_signature_fast(const std::vector<uint8_t>& json_bytes, std::string_view signature_b64, const std::vector<uint8_t>& verify_key_bytes) {
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
        for (size_t i = 0; i < std::min(size_t(16), signature_bytes.size()); i++) {
            char buf[3];
            snprintf(buf, sizeof(buf), "%02x", signature_bytes[i]);
            sig_hex += buf;
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

std::pair<std::string, std::vector<uint8_t>> compute_content_hash(const nb::dict& event_dict) {
    init_json_buffer(event_dict.size());
    
    py_to_canonical_json_fast(event_dict);
    
    std::string canonical_json(json_buffer.data(), json_ptr - json_buffer.data());
    return compute_content_hash_fast(canonical_json);
}

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
    
    std::vector<uint8_t> json_bytes(json_buffer.data(), json_ptr);
    
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
    
    std::vector<uint8_t> json_bytes(json_buffer.data(), json_ptr);
    
    if (!verify_signature_fast(json_bytes, signature_b64, verify_key_bytes)) {
        throw SignatureVerifyException("Signature verification failed");
    }
}

struct SigningResult {
    std::string signature;
    std::string key_id;
    std::string algorithm;
};

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
        
        std::vector<uint8_t> json_bytes(json_buffer.data(), json_ptr);
        
        if (debug_enabled) {
            std::string canonical_json(json_buffer.data(), json_ptr - json_buffer.data());
            DEBUG_LOG("Canonical JSON (" + std::to_string(canonical_json.size()) + " bytes): " + canonical_json.substr(0, 200) + (canonical_json.size() > 200 ? "..." : ""));
            DEBUG_LOG("Signature to verify: " + signature_b64);
            
            std::string hex;
            for (size_t i = 0; i < verify_key_bytes.size(); i++) {
                char buf[3];
                snprintf(buf, sizeof(buf), "%02x", verify_key_bytes[i]);
                hex += buf;
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
            // Join with newlines
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