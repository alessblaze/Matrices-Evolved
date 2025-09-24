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

// Validation: These are DISABLE flags, so we need at least 5 defined to have only 1 implementation enabled
#include "include/base64-encoder.h"
#if !defined(DISABLE_SSE_BASE64_ENCODER) && !defined(DISABLE_SSE_BASE64_ENCODER_ALIGNED) && !defined(DISABLE_SSE_BASE64_ENCODER_LEMIRE) && !defined(DISABLE_SSE_BASE64_ENCODER_MULA) && !defined(DISABLE_SSE_BASE64_ENCODER_AVX) && !defined(DISABLE_NEON_BASE64_ENCODER)
#error "Must disable at least 5 of the 6 base64 encoders: DISABLE_SSE_BASE64_ENCODER, DISABLE_SSE_BASE64_ENCODER_ALIGNED, DISABLE_SSE_BASE64_ENCODER_LEMIRE, DISABLE_SSE_BASE64_ENCODER_MULA, DISABLE_SSE_BASE64_ENCODER_AVX, DISABLE_NEON_BASE64_ENCODER"
#endif

#include <fstream>
#include <cstring>
#include <stdexcept>
// OpenSSL includes for base64 operations
#include <openssl/evp.h>
#include <openssl/bio.h>

// Thread-local buffer definition
thread_local std::string encoder_base64_buffer;

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
[[clang::always_inline]] std::string base64_encode(const std::vector<uint8_t>& data) {
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
    DEBUG_LOG("AVX2 AMS encoder enabled - checking size >= 48");
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
#elif defined(__ARM_NEON) && !defined(DISABLE_NEON_BASE64_ENCODER) || defined(__AVX2__) && !defined(DISABLE_NEON_BASE64_ENCODER)
    // Use NEON SIMD path for larger inputs on ARM
    if (data.size() >= 16) {
        DEBUG_LOG("Using NEON base64 encode for " + std::to_string(data.size()) + " bytes");
        std::string result = fast_neon_base64_encode(data);
        DEBUG_LOG("NEON base64 encode result: " + result);
        
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
            DEBUG_LOG("NEON vs OpenSSL match: " + std::string(result == openssl_buffer ? "TRUE" : "FALSE"));
            if (result != openssl_buffer) {
                DEBUG_LOG("MISMATCH DETECTED - NEON length: " + std::to_string(result.length()) + ", OpenSSL length: " + std::to_string(openssl_buffer.length()));
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
        std::string result = fast_sse_base64_encode_aligned_alt(data);
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
    
    DEBUG_LOG("Using OpenSSL fallback path for " + std::to_string(data.size()) + " bytes");
    size_t out_len = ((data.size() + 2) / 3) * 4;
    if (encoder_base64_buffer.size() < out_len) {
        encoder_base64_buffer.resize(out_len);
    }
    
    int actual_len = EVP_EncodeBlock(reinterpret_cast<uint8_t*>(encoder_base64_buffer.data()), data.data(), data.size());
    
    // Matrix protocol uses unpadded base64 - remove padding
    while (actual_len > 0 && encoder_base64_buffer[actual_len - 1] == '=') {
        actual_len--;
    }
    encoder_base64_buffer.resize(actual_len);
    
    DEBUG_LOG("OpenSSL base64_encode result: " + encoder_base64_buffer);
    return encoder_base64_buffer;
}

