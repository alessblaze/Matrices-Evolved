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

#include "../include/ams-sse-aligned.h"

#if defined(__AVX2__) && !defined(DISABLE_SSE_BASE64_ENCODER_ALIGNED)
// Aligned version with Boost aligned allocator
#include <boost/align/aligned_allocator.hpp>
thread_local std::string base64_buffer;
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
[[gnu::hot, gnu::flatten]] std::string fast_sse_base64_encode_aligned(const std::vector<uint8_t>& data) {
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
#endif