/*

Matrices-Evolved - High Performance ops offloaded to Rust and C++
Copyright (c) 2025 Albert Blasczykowski (Aless Microsystems)

This program is licensed under the Aless Microsystems Source-Available License (Non-Commercial, No Military) v1.0 Available in the Root
Directory of the project as LICENSE in Text Format.
You may use, copy, modify, and distribute this program for Non-Commercial purposes only, subject to the terms of that license.
Use by or for military, intelligence, or defense entities or purposes is strictly prohibited.

If you distribute this program in object form or make it available to others over a network, you must provide the complete
corresponding source code for the provided functionality under this same license.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the License for details.

You should have received a copy of the License along with this program; if not, see the LICENSE file included with this source.

*/

#include "canonicalization.h"
#include <boost/json/src.hpp>

namespace nb = nanobind;
namespace json = boost::json;

// Compiler attributes
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

// Shared definitions are now in debug.h



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
#if defined (__AVX2__) || defined (__ARM_NEON)
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

// hex_lut is now provided by debug.h

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
int fast_double_to_string(double f, char* result) {
    // Reject non-finite values (NaN, infinity)
    if (!std::isfinite(f)) {
        throw std::runtime_error("Non-finite floats not allowed in JSON");
    }
    
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
    
    // Use std::to_chars with safe buffer size (32 bytes minimum required)
    auto [ptr, ec] = std::to_chars(result, result + 32, f);
    if (ec == std::errc{}) {
        int len = ptr - result;
        
        // SSE2-optimized decimal point detection
#if defined (__AVX2__) || defined (__ARM_NEON)
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

// MAX_EVENT_SIZE is now provided by debug.h

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
void init_json_buffer(size_t hint) {
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
        // Calculate required size directly to avoid loop overhead
        size_t required_size = current_size + needed;
        size_t new_size = json_buffer.size();
        
        // Find next power of 2 >= required_size
        while (new_size < required_size) {
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

// Non-inline versions for external linkage
void write_char_external(char c) {
    write_char(c);
}

void write_string_external(std::string_view s) {
    write_string(s);
}

void write_cstring_external(const char* s) {
    write_cstring(s);
}

void write_unicode_escape_external(unsigned char c) {
    write_unicode_escape(c);
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
    
#if defined (__AVX2__) || defined (__ARM_NEON)
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
#if defined (__AVX2__) || defined (__ARM_NEON)
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
template<typename T>
T get_json_span() {
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
    // hex_lut is guaranteed to have 16 elements for nibble values 0-15
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
#if defined (__AVX2__) || defined (__ARM_NEON)
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

HOT_FUNCTION FLATTEN_FUNCTION void py_to_canonical_json_fast(const nb::object& root_obj) {
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
                    
#if defined(__AVX2__) && !defined(DISABLE_AVX2_JSON_SIMD) || defined(__ARM_NEON) && !defined(DISABLE_AVX2_JSON_SIMD)
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
#if defined(__AVX2__) && !defined(DISABLE_AVX2_JSON_SIMD) || defined(__ARM_NEON) && !defined(DISABLE_AVX2_JSON_SIMD)
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
                    
#if defined(__AVX2__) && !defined(DISABLE_AVX2_JSON_SIMD) || defined(__ARM_NEON) && !defined(DISABLE_AVX2_JSON_SIMD)

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
#if defined(__AVX2__) && !defined(DISABLE_AVX2_JSON_SIMD) || defined(__ARM_NEON) && !defined(DISABLE_AVX2_JSON_SIMD)
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

// Local buffers for JSON canonicalization module
thread_local std::vector<uint8_t> json_signature_buffer(64); // Ed25519 signature size
thread_local std::vector<uint8_t> json_hash_buffer(32);     // SHA256 hash size

// Simple key cache to avoid repeated OpenSSL key creation
thread_local std::vector<uint8_t> cached_signing_key;
thread_local EVP_PKEY* cached_signing_pkey = nullptr;




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
            
#if defined(__AVX2__) && !defined(DISABLE_AVX2_JSON_SIMD) || defined(__ARM_NEON) && !defined(DISABLE_AVX2_JSON_SIMD)
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
#if defined(__AVX2__) && !defined(DISABLE_AVX2_JSON_SIMD) || defined(__ARM_NEON) && !defined(DISABLE_AVX2_JSON_SIMD)
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
 * Signs JSON data using Ed25519 digital signature algorithm
 * @param json_bytes Canonical JSON data to sign
 * @param signing_key_bytes 32-byte Ed25519 private key
 * @return Base64-encoded signature string
 */
std::string sign_json_fast(std::span<const uint8_t> json_bytes, const std::vector<uint8_t>& signing_key_bytes) {
    if (signing_key_bytes.size() != 32) {
        throw std::runtime_error("Invalid signing key");
    }
    
    json_signature_buffer.resize(64); // Ed25519 signature size
    
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
        if (EVP_DigestSign(ctx, json_signature_buffer.data(), &sig_len, json_bytes.data(), json_bytes.size()) != 1 || sig_len != 64) {
            EVP_MD_CTX_free(ctx);
            // Don't free pkey - it's cached
            throw std::runtime_error("Signing failed");
        }
        
        EVP_MD_CTX_free(ctx);
        // Don't free pkey - it's cached
    }
    
    return base64_encode(json_signature_buffer);
}

/**
 * Reset JSON pointer to buffer start
 */
void reset_json_pointer() {
    json_ptr = json_buffer.data();
}

// Explicit template instantiations for get_json_span
template std::vector<uint8_t> get_json_span<std::vector<uint8_t>>();
template std::span<const uint8_t> get_json_span<std::span<const uint8_t>>();
template std::span<uint8_t> get_json_span<std::span<uint8_t>>();
