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

#include "../include/mulla-sse.h"

#if defined(__AVX2__) && !defined(DISABLE_SSE_BASE64_ENCODER_MULA)
thread_local std::string base64_buffer;
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
[[gnu::hot, gnu::flatten]] std::string fast_mula_base64_encode(const std::vector<uint8_t>& data) {
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