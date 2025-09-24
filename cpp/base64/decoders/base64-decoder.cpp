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

#include "include/base64-decoder.h"
//#include "include/lemire-avx.h"
//#include "include/ams-avx.h"
//#include "include/ams-sse.h"
#include "include/ams-neon.h"
// Thread-local decode buffer
thread_local std::vector<uint8_t> decode_buffer(1024);



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
std::vector<uint8_t> base64_decode(std::string_view encoded_string) {
    DEBUG_LOG("base64_decode called with length " + std::to_string(encoded_string.size()) + ": '" + std::string(encoded_string) + "'");
    if (encoded_string.empty()) return {};
    
    // Use SIMD path for larger inputs
    if (encoded_string.size() >= 32) {
         DEBUG_LOG("Taking SIMD fast path");
         //auto simd_result = fast_base64_decode_signature(encoded_string);
         //auto simd_result = fast_base64_decode_avx2_rangecmp(encoded_string);
         auto simd_result = fast_base64_decode_neon_rangecmp(encoded_string);
         if (debug_enabled) {
             // OpenSSL verification for debugging
             std::string padded_input(encoded_string);
             size_t padding_needed = (4 - (encoded_string.size() % 4)) % 4;
             padded_input.append(padding_needed, '=');
             
             std::vector<uint8_t> openssl_buffer(((padded_input.size() * 3) / 4));
             int openssl_result = EVP_DecodeBlock(openssl_buffer.data(),
                                                reinterpret_cast<const uint8_t*>(padded_input.data()),
                                                padded_input.size());
             if (openssl_result >= 0) {
                 openssl_buffer.resize(openssl_result - padding_needed);
                 if (simd_result != openssl_buffer) {
                     DEBUG_LOG("SIMD/OpenSSL mismatch - SIMD: " + std::to_string(simd_result.size()) + " bytes, OpenSSL: " + std::to_string(openssl_buffer.size()) + " bytes");
                 } else {
                     DEBUG_LOG("SIMD/OpenSSL match - " + std::to_string(simd_result.size()) + " bytes");
                 }
             }
         }
         return simd_result;
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

// String overload that delegates to string_view version
std::vector<uint8_t> base64_decode(const std::string& encoded) {
    return base64_decode(std::string_view(encoded));
}
