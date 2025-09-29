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
#include "../../../global.h"

#ifdef __AVX2__
// Thread-local decode buffer
thread_local std::vector<uint8_t> lemire_decode_buffer(1024);

// Local hex lookup table for this file (using lowercase)
static constexpr char lemire_hex_lut[16] = {'0','1','2','3','4','5','6','7','8','9','a','b','c','d','e','f'};

// Debug logging macro

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
[[gnu::hot, gnu::flatten]] std::vector<uint8_t> fast_base64_decode_signature(std::string_view input) {
    DEBUG_LOG("fast_base64_decode_simd input length: " + std::to_string(input.size()));
    
    size_t output_len = (input.size() * 3) / 4;
    if (lemire_decode_buffer.size() < output_len) {
        lemire_decode_buffer.resize(output_len);
    }
    
    const char* src = input.data();
    uint8_t* dst = lemire_decode_buffer.data();
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
    // Clear upper 128 bits of YMM registers before transitioning to scalar code
    //This is a tweak to apply only on x86_64 targets because simde is used thus the same code gets compiled for ARM
    //simd manages the intrinstic mappings but it does not support zeroupper on non native targets
    //simde defines __AVX2__ even when not compiling for x86. So we need to limit this to x86_64 only as zeroupper is not needed on ARM.
    #ifdef __x86_64__ 
    _mm256_zeroupper();
    #endif

    while (srclen >= 16) {
        // Load 16 chars, decode with AVX scalar instructions
        //__m128i chars = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src));
        
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
    
    lemire_decode_buffer.resize(dst - lemire_decode_buffer.data());
    
    if (debug_enabled) {
        std::string decoded_hex;
        decoded_hex.reserve(lemire_decode_buffer.size() * 2);
        for (size_t i = 0; i < lemire_decode_buffer.size(); i++) {
            decoded_hex.push_back(lemire_hex_lut[lemire_decode_buffer[i] >> 4]);
            decoded_hex.push_back(lemire_hex_lut[lemire_decode_buffer[i] & 0x0F]);
        }
        DEBUG_LOG("SIMD decoded " + std::to_string(lemire_decode_buffer.size()) + " bytes: " + decoded_hex);
    }
    
    return lemire_decode_buffer;
}
#endif
