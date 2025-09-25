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

#include "../include/ams-avx.h"

#if defined(__AVX2__) && !defined(DISABLE_SSE_BASE64_ENCODER_AVX)
thread_local std::string base64_buffer;
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

static inline __m256i map_base64_1pshufb(__m256i idx) {
    // Classify indices into 4 classes: 0:[0..25], 1:[26..51], 2:[52..61], 3:[62..63]
    __m256i gt25 = _mm256_cmpgt_epi8(idx, _mm256_set1_epi8(25));
    __m256i gt51 = _mm256_cmpgt_epi8(idx, _mm256_set1_epi8(51));
    __m256i gt61 = _mm256_cmpgt_epi8(idx, _mm256_set1_epi8(61));
    __m256i one  = _mm256_set1_epi8(1);
    __m256i cls  = _mm256_add_epi8(_mm256_and_si256(gt25, one),
                    _mm256_add_epi8(_mm256_and_si256(gt51, one),
                                    _mm256_and_si256(gt61, one)));

    // Build 16-byte LUT and broadcast to 256 bits (both lanes get the same 16B)
    const __m128i lut128 = _mm_setr_epi8(
        65, 71, (char)-4, 0, 65, 71, (char)-4, 0,
        65, 71, (char)-4, 0, 65, 71, (char)-4, 0
    );
    const __m256i lut = _mm256_broadcastsi128_si256(lut128);

    // One VPSHUFB: fetch ASCII deltas by class; MSB of cls is 0 so no zeroing
    __m256i delta = _mm256_shuffle_epi8(lut, cls);

    // Apply delta to indices (maps 0..25→'A'+i, 26..51→'a'+i-26, 52..61→'0'+i-52)
    __m256i out = _mm256_add_epi8(idx, delta);

    // Patch specials: 62 -> '+', 63 -> '/'
    __m256i eq62  = _mm256_cmpeq_epi8(idx, _mm256_set1_epi8(62));
    __m256i eq63  = _mm256_cmpeq_epi8(idx, _mm256_set1_epi8(63));
    __m256i plus  = _mm256_set1_epi8('+');
    __m256i slash = _mm256_set1_epi8('/');

    out = sel_by_mask_xor(out, plus,  eq62);
    out = sel_by_mask_xor(out, slash, eq63);
    return out;
}

static inline __m256i map_base64_1pshufb_32(__m256i idx) {
    // Classify indices into 4 classes: 0:[0..25], 1:[26..51], 2:[52..61], 3:[62..63]
    __m256i gt25 = _mm256_cmpgt_epi8(idx, _mm256_set1_epi8(25));
    __m256i gt51 = _mm256_cmpgt_epi8(idx, _mm256_set1_epi8(51));
    __m256i gt61 = _mm256_cmpgt_epi8(idx, _mm256_set1_epi8(61));
    __m256i one  = _mm256_set1_epi8(1);
    __m256i cls  = _mm256_add_epi8(_mm256_and_si256(gt25, one),
                    _mm256_add_epi8(_mm256_and_si256(gt51, one),
                                    _mm256_and_si256(gt61, one)));

    // 32-arg LUT: duplicate the 16-byte pattern in both lanes (VPSHUFB is lane-local)
    const __m256i lut = _mm256_setr_epi8(
        65, 71, (char)-4, 0, 65, 71, (char)-4, 0,
        65, 71, (char)-4, 0, 65, 71, (char)-4, 0,
        65, 71, (char)-4, 0, 65, 71, (char)-4, 0,
        65, 71, (char)-4, 0, 65, 71, (char)-4, 0
    );

    // One VPSHUFB: fetch ASCII deltas by class (class codes 0..3 keep MSB clear -> no zeroing)
    __m256i delta = _mm256_shuffle_epi8(lut, cls);

    // Apply delta to indices
    __m256i out = _mm256_add_epi8(idx, delta);

    // Patch specials: 62 -> '+', 63 -> '/'
    __m256i eq62  = _mm256_cmpeq_epi8(idx, _mm256_set1_epi8(62));
    __m256i eq63  = _mm256_cmpeq_epi8(idx, _mm256_set1_epi8(63));
    __m256i plus  = _mm256_set1_epi8('+');
    __m256i slash = _mm256_set1_epi8('/');

    out = sel_by_mask_xor(out, plus,  eq62);
    out = sel_by_mask_xor(out, slash, eq63);
    return out;
}

[[clang::always_inline]] static inline __m256i map_base64_1pshufb_merged(__m256i idx) {
    __m256i gt25 = _mm256_cmpgt_epi8(idx, _mm256_set1_epi8(25));
    __m256i gt51 = _mm256_cmpgt_epi8(idx, _mm256_set1_epi8(51));
    __m256i gt61 = _mm256_cmpgt_epi8(idx, _mm256_set1_epi8(61));
    __m256i one  = _mm256_set1_epi8(1);
    __m256i cls  = _mm256_add_epi8(_mm256_and_si256(gt25, one),
                    _mm256_add_epi8(_mm256_and_si256(gt51, one),
                                    _mm256_and_si256(gt61, one)));

    // LUT deltas by class: 0:+65 ('A'), 1:+71 ('a'-26), 2:-4 ('0'-52), 3:-19 (base for '+'/'/')
    const __m256i lut = _mm256_setr_epi8(
        65, 71, (char)-4, (char)-19, 65, 71, (char)-4, (char)-19,
        65, 71, (char)-4, (char)-19, 65, 71, (char)-4, (char)-19,
        65, 71, (char)-4, (char)-19, 65, 71, (char)-4, (char)-19,
        65, 71, (char)-4, (char)-19, 65, 71, (char)-4, (char)-19
    );
    __m256i delta = _mm256_shuffle_epi8(lut, cls);

    // Compute s3 = 3*(idx&1) only for idx>=62: adds 0 for 62, 3 for 63
    __m256i lsb  = _mm256_and_si256(idx, one);
    __m256i s3   = _mm256_add_epi8(lsb, _mm256_add_epi8(lsb, lsb));   // 3*lsb
    s3 = _mm256_and_si256(s3, gt61);

    // Final ASCII
    return _mm256_add_epi8(_mm256_add_epi8(idx, delta), s3);
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
[[clang::always_inline]] static inline __m256i process_avx2_chunk_direct(const uint8_t* src) {
    // Lemire approach: __m256i in = _mm256_maskload_epi32(reinterpret_cast<const int*>(src - 4), _mm256_set_epi32(0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x00000000));
    
     __m256i r0 = _mm256_loadu_si256((const __m256i*)(src +  0));   // 0..31
    __m256i r1 = _mm256_loadu_si256((const __m256i*)(src + 16));   // 16..47
    __m256i z  = _mm256_setzero_si256();

    __m256i low12 = _mm256_alignr_epi8(r0, z, 12);

    __m256i r0low_dup = _mm256_permute2x128_si256(r0, r0, 0x00);
    __m256i r1low_dup = _mm256_permute2x128_si256(r1, r1, 0x00);
    __m256i hi12_low  = _mm256_alignr_epi8(r1low_dup, r0low_dup, 12);

    __m256i in = _mm256_permute2x128_si256(low12, hi12_low, 0x20); 

    // lane-local triplet gather
    const __m256i gather = _mm256_setr_epi8(
        6,5,4,-1,  9,8,7,-1,  12,11,10,-1, 15,14,13,-1,
        2,1,0,-1,  5,4,3,-1,   8,7,6,-1,  11,10, 9,-1
    );
    __m256i packed = _mm256_shuffle_epi8(in, gather);               

    // extract 6-bit indices into byte 0..3 of each dword
    const __m256i m6 = _mm256_set1_epi32(0x3f);
    __m256i i0 = _mm256_and_si256(_mm256_srli_epi32(packed, 18), m6);
    __m256i i1 = _mm256_slli_epi32(_mm256_and_si256(_mm256_srli_epi32(packed, 12), m6), 8);
    __m256i i2 = _mm256_slli_epi32(_mm256_and_si256(_mm256_srli_epi32(packed,  6), m6), 16);
    __m256i i3 = _mm256_slli_epi32(_mm256_and_si256(packed, m6), 24);
    __m256i indices = _mm256_or_si256(_mm256_or_si256(i0, i1), _mm256_or_si256(i2, i3));
    // index -> ASCII via one-pshufb delta LUT + tiny patch (reuse your map_base64_1pshufb_merged)
    return map_base64_1pshufb_merged(indices);  
}

[[gnu::hot, gnu::flatten, clang::always_inline]] std::string fast_sse_base64_encode_avx(const std::vector<uint8_t>& data) {
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
    // Clear upper 128 bits of YMM registers before transitioning to scalar code
    _mm256_zeroupper();
    
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
#endif