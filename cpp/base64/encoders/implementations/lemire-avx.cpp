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

#include "../include/lemire-avx.h"

#if defined(__AVX2__) && !defined(DISABLE_SSE_BASE64_ENCODER_LEMIRE)
thread_local std::string base64_buffer;
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
    const __m256i t1 = _mm256_mulhi_epu16(t0, _mm256_set1_epi32(0x04010040));
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
[[gnu::hot, gnu::flatten, clang::always_inline]] std::string fast_avx2_base64_encode_lemire(const std::vector<uint8_t>& data) {
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
    // Clear upper 128 bits of YMM registers before transitioning to scalar code
    _mm256_zeroupper();
    
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