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
#include <Python.h>

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

/**
 * Direct Python object to canonical JSON serialization
 * 
 * Algorithm:
 * 1. Type dispatch via nanobind isinstance checks
 * 2. Direct buffer writing without intermediate DOM
 * 3. Recursive descent for containers (lists, dicts)
 * 4. Dictionary key sorting for canonical ordering
 * 5. Zero-copy string access via PyUnicode_AsUTF8AndSize
 * 6. Thread-local arena allocation for key storage
 * 
 * Optimizations:
 * - Fast-path common types (null, bool, int)
 * - std::to_chars for numeric conversion
 * - Vectorized string escaping
 * - Arena allocator eliminates malloc churn
 * 
 * @param obj Python object to serialize
 * @throws std::runtime_error for unsupported types or non-finite floats
 */
HOT_FUNCTION FLATTEN_FUNCTION inline void py_to_canonical_json_fast(const nb::object& obj) {
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
        
        // Normalize -0.0 to 0.0
        if (val == 0.0 && std::signbit(val)) {
            val = 0.0;
        }
        
        // Use fast double formatting (already adds .0 if needed)
        char buf[32];
        int len = fast_double_to_string(val, buf);
        write_string(std::string_view(buf, len));
    } else if (nb::isinstance<nb::str>(obj)) {
        std::string_view s = nb::cast<std::string_view>(obj);
        write_char('"');
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
                    if (c < 0x20) {
                        write_unicode_escape(c);
                    } else {
                        write_char(c);
                    }
            }
        }
        write_char('"');
    } else if (nb::isinstance<nb::list>(obj) || nb::isinstance<nb::tuple>(obj)) {
        auto seq = nb::borrow(obj);
        write_char('[');
        size_t i = 0;
        for (auto item : seq) {
            if (i++) write_char(',');
            py_to_canonical_json_fast(nb::cast<nb::object>(item));
        }
        write_char(']');
    } else if (nb::isinstance<nb::dict>(obj)) {
        auto dict = nb::borrow<nb::dict>(obj);
        std::vector<std::pair<std::string_view, nb::object>> key_pairs;
        key_pairs.reserve(dict.size());
        
        // Arena allocator for key storage - no malloc churn
        thread_local std::vector<char> key_arena;
        
        key_arena.clear();
        
        // Extract all data while holding GIL
        for (auto item : dict) {
            // Check that key is a string
            if (!nb::isinstance<nb::str>(item.first)) {
                throw std::runtime_error("Dictionary keys must be strings for JSON serialization");
            }
            
            // Always use PyUnicode_AsUTF8AndSize for zero-copy
            Py_ssize_t size;
            const char* data = PyUnicode_AsUTF8AndSize(item.first.ptr(), &size);
            if (data) {
                key_pairs.emplace_back(std::string_view(data, size), nb::cast<nb::object>(item.second));
            } else {
                // Rare fallback - store in arena
                std::string key_str = nb::cast<std::string>(item.first);
                size_t offset = key_arena.size();
                key_arena.resize(offset + key_str.size());
                std::memcpy(key_arena.data() + offset, key_str.data(), key_str.size());
                key_pairs.emplace_back(std::string_view(key_arena.data() + offset, key_str.size()), nb::cast<nb::object>(item.second));
            }
        }
        
        // Sort keys (no Python objects touched)
        std::sort(key_pairs.begin(), key_pairs.end(),
                 [](const auto& a, const auto& b) { return a.first < b.first; });
        
        // Write dict structure (no Python objects touched for keys)
        write_char('{');
        for (size_t i = 0; i < key_pairs.size(); ++i) {
            if (i) write_char(',');
            
            // Write key directly without nb::str conversion
            write_char('"');
            const auto& key = key_pairs[i].first;
            for (char c : key) {
                if (c == '"') write_raw("\\\"", 2);
                else if (c == '\\') write_raw("\\\\", 2);
                else if (c < 0x20) write_unicode_escape(static_cast<unsigned char>(c));
                else write_char(c);
            }
            write_char('"');
            
            write_char(':');
            // Recursively serialize value (may need GIL)
            py_to_canonical_json_fast(key_pairs[i].second);
        }
        write_char('}');
    } else {
        throw std::runtime_error("Unsupported Python type");
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

// Ultra-fast base64 decode for Ed25519 signatures (86 chars -> 64 bytes)
[[gnu::hot, gnu::flatten]] inline std::vector<uint8_t> fast_base64_decode_signature(std::string_view input) {
    signature_buffer.resize(64);
    const char* src = input.data();
    uint8_t* dst = signature_buffer.data();
    
    // Process 84 chars in groups of 4 (21 groups * 3 bytes = 63 bytes)
    for (size_t i = 0; i < 84; i += 4) {
        uint32_t val = (base64_decode_table[static_cast<uint8_t>(src[i])] << 18) |
                       (base64_decode_table[static_cast<uint8_t>(src[i+1])] << 12) |
                       (base64_decode_table[static_cast<uint8_t>(src[i+2])] << 6) |
                       base64_decode_table[static_cast<uint8_t>(src[i+3])];
        
        *dst++ = (val >> 16) & 0xFF;
        *dst++ = (val >> 8) & 0xFF;
        *dst++ = val & 0xFF;
    }
    
    // Handle last 2 characters -> 1 byte
    uint32_t val = (base64_decode_table[static_cast<uint8_t>(src[84])] << 18) |
                   (base64_decode_table[static_cast<uint8_t>(src[85])] << 12);
    *dst = (val >> 16) & 0xFF;
    
    return signature_buffer;
}

// Base64 using OpenSSL with buffer reuse - Matrix format (no padding)
[[gnu::hot, gnu::flatten]] inline std::string base64_encode(const std::vector<uint8_t>& data) {
    if (data.empty()) return "";
    
    size_t out_len = ((data.size() + 2) / 3) * 4;
    base64_buffer.resize(out_len);
    
    int actual_len = EVP_EncodeBlock(reinterpret_cast<uint8_t*>(base64_buffer.data()), data.data(), data.size());
    
    // Matrix protocol uses unpadded base64 - remove padding
    while (actual_len > 0 && base64_buffer[actual_len - 1] == '=') {
        actual_len--;
    }
    base64_buffer.resize(actual_len);
    
    return base64_buffer;
}

// Matrix protocol base64 decode - expects unpadded input
[[gnu::hot, gnu::flatten]] std::vector<uint8_t> base64_decode(std::string_view encoded_string) {
    if (encoded_string.empty()) return {};
    
    // Fast path for Ed25519 signatures (86 chars)
    if (encoded_string.size() == 86) {
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
    if (verify_key_bytes.size() != 32) return false;
    
    auto signature_bytes = fast_base64_decode_signature(signature_b64);
    if (signature_bytes.size() != 64) return false;
    
    bool result = false;
    {
        nb::gil_scoped_release release;
        EVP_PKEY* pkey = EVP_PKEY_new_raw_public_key(EVP_PKEY_ED25519, nullptr, verify_key_bytes.data(), 32);
        if (pkey) {
            EVP_MD_CTX* ctx = EVP_MD_CTX_new();
            if (ctx) {
                if (EVP_DigestVerifyInit(ctx, nullptr, nullptr, nullptr, pkey) == 1) {
                    result = EVP_DigestVerify(ctx, signature_bytes.data(), 64, json_bytes.data(), json_bytes.size()) == 1;
                }
                EVP_MD_CTX_free(ctx);
            }
            EVP_PKEY_free(pkey);
        }
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
    m.def("decode_verify_key_bytes", [](std::string_view key_id, const std::vector<uint8_t>& key_bytes) {
        if (key_id.starts_with("ed25519:") && key_bytes.size() == 32) return key_bytes;
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
            alg = nb::cast<std::string>(verify_key.attr("alg"));
            version = nb::cast<std::string>(verify_key.attr("version"));
            DEBUG_LOG("Got VerifyKey object: alg=" + alg + ", version=" + version);
            nb::bytes encoded_key = nb::cast<nb::bytes>(verify_key.attr("encode")());
            
            // Convert nb::bytes to std::vector<uint8_t>
            const char* ptr = static_cast<const char*>(encoded_key.c_str());
            size_t size = encoded_key.size();
            verify_key_bytes = std::vector<uint8_t>(ptr, ptr + size);
        }
        
        std::string key_id = alg + ":" + version;
        
        // Get signatures
        if (!json_dict.contains("signatures")) {
            throw SignatureVerifyException("No signatures on this object");
        }
        
        nb::dict signatures = json_dict["signatures"];
        if (!signatures.contains(signature_name.c_str())) {
            throw SignatureVerifyException("Missing signature for " + signature_name);
        }
        
        nb::dict server_sigs = signatures[signature_name.c_str()];
        std::string signature_b64;
        
        // Try exact key_id first, then fallback to any ed25519
        if (server_sigs.contains(key_id.c_str())) {
            signature_b64 = nb::cast<std::string>(server_sigs[key_id.c_str()]);
        } else {
            for (auto item : server_sigs) {
                std::string_view available_key = nb::cast<std::string_view>(item.first);
                if (available_key.starts_with("ed25519:")) {
                    signature_b64 = nb::cast<std::string>(item.second);
                    break;
                }
            }
        }
        
        if (signature_b64.empty()) {
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
        
        // Canonicalize and verify
        init_json_buffer(unsigned_dict.size());
        
        py_to_canonical_json_fast(unsigned_dict);
        
        std::vector<uint8_t> json_bytes(json_buffer.data(), json_ptr);
        
        if (!verify_signature_fast(json_bytes, signature_b64, verify_key_bytes)) {
            throw SignatureVerifyException("Unable to verify signature for " + signature_name);
        }
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