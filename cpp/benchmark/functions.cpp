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
#define SIMDE_ENABLE_NATIVE_ALIASES

#include <simde/x86/avx2.h>
#include <simde/x86/sse4.1.h>
#include <simde/x86/sse4.2.h>
#include <simde/x86/sse3.h>
#include <simde/x86/sse2.h>
#include <simde/arm/neon.h>
#include <vector>
#include <string>
#include <cstring>
#include <stdexcept>
#include <cstdint>
#include <cstddef>
#include <openssl/bio.h>
#include <openssl/evp.h>
#include <openssl/buffer.h>
// Thread-local buffers for base64 encoding (separate per function)
thread_local std::string sse_buffer;
thread_local std::string avx_buffer;
thread_local std::string lemire_buffer;
thread_local std::string mula_buffer;
thread_local std::string aligned_buffer;
thread_local std::string neon_buffer;

// Debug flag
static bool debug_enabled = false;

// Debug logging macro
#define DEBUG_LOG(msg) do { if (debug_enabled) { /* log message */ } } while(0)

// Base64 alphabet lookup table
static constexpr char base64_chars[64] = {
    'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P',
    'Q','R','S','T','U','V','W','X','Y','Z','a','b','c','d','e','f',
    'g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v',
    'w','x','y','z','0','1','2','3','4','5','6','7','8','9','+','/'
};

// Alternative approach using multiple specific masks - currently unused
// This approach uses different masks for each 6-bit extraction but is slower
// due to increased register pressure and less optimal instruction scheduling
static inline __m128i extract_indices_to_bytes_alt(const __m128i& packed) {
    __m128i hi  = _mm_srli_epi32(_mm_and_si128(packed, _mm_set1_epi32(0x00fc0000)), 18);
    __m128i mid = _mm_srli_epi32(_mm_and_si128(packed, _mm_set1_epi32(0x0003f000)), 4);
    __m128i lo  = _mm_slli_epi32(_mm_and_si128(packed, _mm_set1_epi32(0x00000fc0)), 10);
    __m128i bot = _mm_slli_epi32(_mm_and_si128(packed, _mm_set1_epi32(0x0000003f)), 24);
    return _mm_or_si128(_mm_or_si128(hi, mid), _mm_or_si128(lo, bot));
}

// Current optimized approach using single reusable mask
static inline __m128i extract_indices_to_bytes(const __m128i& packed) {
    static const __m128i mask6 = _mm_set1_epi32(0x3f);
    __m128i idx0 = _mm_and_si128(_mm_srli_epi32(packed, 18), mask6);
    __m128i idx1 = _mm_slli_epi32(_mm_and_si128(_mm_srli_epi32(packed, 12), mask6), 8);
    __m128i idx2 = _mm_slli_epi32(_mm_and_si128(_mm_srli_epi32(packed, 6), mask6), 16);
    __m128i idx3 = _mm_slli_epi32(_mm_and_si128(packed, mask6), 24);
    return _mm_or_si128(_mm_or_si128(idx0, idx1), _mm_or_si128(idx2, idx3));
}

static inline __m128i lut_lookup(const __m128i& indices) {
    static const __m128i lut0 = _mm_setr_epi8('A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P');
    static const __m128i lut1 = _mm_setr_epi8('Q','R','S','T','U','V','W','X','Y','Z','a','b','c','d','e','f');
    static const __m128i lut2 = _mm_setr_epi8('g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v');
    static const __m128i lut3 = _mm_setr_epi8('w','x','y','z','0','1','2','3','4','5','6','7','8','9','+','/');
    static const __m128i const16 = _mm_set1_epi8(16);
    static const __m128i const32 = _mm_set1_epi8(32);
    static const __m128i const48 = _mm_set1_epi8(48);
    
    __m128i mask_32 = _mm_cmplt_epi8(indices, const32);
    __m128i lo_result = _mm_blendv_epi8(
        _mm_shuffle_epi8(lut1, _mm_sub_epi8(indices, const16)),
        _mm_shuffle_epi8(lut0, indices),
        _mm_cmplt_epi8(indices, const16)
    );
    __m128i hi_result = _mm_blendv_epi8(
        _mm_shuffle_epi8(lut3, _mm_sub_epi8(indices, const48)),
        _mm_shuffle_epi8(lut2, _mm_sub_epi8(indices, const32)),
        _mm_cmplt_epi8(indices, const48)
    );
    return _mm_blendv_epi8(hi_result, lo_result, mask_32);
}

/**
 * Fast base64 encoder using SSE 128-bit SIMD instructions
 * 
 * Algorithm:
 * 1. Process 48 input bytes as 16 triplets (3 bytes each)
 * 2. Extract 4 base64 indices per triplet using bit shifts
 * 3. Lookup base64 characters from alphabet table using SSE shuffle
 * 4. Produces exactly 64 base64 characters per 48-byte block
 * 
 * Performance: ~2-3x faster than OpenSSL for large inputs
 * Complete redesign.
 * Nearly same with Mullas method better in some cases.
 * @param data Input byte vector to encode
 * @return Base64 encoded string (unpadded for Matrix protocol)
 */
[[gnu::hot, gnu::flatten]] inline std::string fast_sse_base64_encode(const std::vector<uint8_t>& data) {
    size_t len = data.size();
    
    // Fast path for tiny inputs - avoid SIMD overhead
    if (len <= 24) {
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
    if (sse_buffer.size() < out_len) {
        sse_buffer.resize(out_len);
    }
    
    const uint8_t* src = data.data();
    char* dest = sse_buffer.data();
    const char* const dest_orig = dest;
    

    static const __m128i trip_shuffle = _mm_setr_epi8(
        2, 1, 0, (char)0x80,   // lane0: bytes 2,1,0 to match (b0<<16)|(b1<<8)|b2
        5, 4, 3, (char)0x80,   // lane1: bytes 5,4,3
        8, 7, 6, (char)0x80,   // lane2: bytes 8,7,6
       11,10, 9, (char)0x80    // lane3: bytes 11,10,9
    );

    
    // Process 48-byte blocks with single loop
    while (len >= 48) {
        // Process 12 triplets with 4x unrolled SIMD
        __m128i in0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + 0));
        __m128i in1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + 12));
        __m128i in2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + 24));
        __m128i in3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + 36));
        
        __m128i packed0 = _mm_shuffle_epi8(in0, trip_shuffle);
        __m128i packed1 = _mm_shuffle_epi8(in1, trip_shuffle);
        __m128i packed2 = _mm_shuffle_epi8(in2, trip_shuffle);
        __m128i packed3 = _mm_shuffle_epi8(in3, trip_shuffle);
        
        __m128i idx0_unpacked = extract_indices_to_bytes(packed0);
        __m128i idx1_unpacked = extract_indices_to_bytes(packed1);
        __m128i idx2_unpacked = extract_indices_to_bytes(packed2);
        __m128i idx3_unpacked = extract_indices_to_bytes(packed3);
        
        __m128i chars0 = lut_lookup(idx0_unpacked);
        __m128i chars1 = lut_lookup(idx1_unpacked);
        __m128i chars2 = lut_lookup(idx2_unpacked);
        __m128i chars3 = lut_lookup(idx3_unpacked);
        
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dest + 0), chars0);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dest + 16), chars1);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dest + 32), chars2);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dest + 48), chars3);
        
        src += 48;
        dest += 64;
        len -= 48;
    }

    
    // Fallback scalar processing
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
    
    size_t actual_len = dest - dest_orig;
    sse_buffer.resize(actual_len);
    
    return sse_buffer;
}

/**
 * Fast AVX2 base64 encoder using 256-bit SIMD instructions
 * 
 * Algorithm:
 * 1. Process 96 input bytes as 32 triplets (3 bytes each) 
 * 2. Extract 4 base64 indices per triplet using bit shifts
 * 3. Lookup base64 characters from alphabet table using AVX2 shuffle
 * 4. Produces exactly 128 base64 characters per 96-byte block
 * 
 * Performance: ~3-4x faster than SSE for large inputs
 */

 // Base64 alphabet lookup table
// AVX2 version of extract_indices_to_bytes
static inline __m256i extract_indices_to_bytes_avx2(const __m256i& packed) {
    static const __m256i mask6 = _mm256_set1_epi32(0x3f);
    __m256i idx0 = _mm256_and_si256(_mm256_srli_epi32(packed, 18), mask6);
    __m256i idx1 = _mm256_slli_epi32(_mm256_and_si256(_mm256_srli_epi32(packed, 12), mask6), 8);
    __m256i idx2 = _mm256_slli_epi32(_mm256_and_si256(_mm256_srli_epi32(packed, 6), mask6), 16);
    __m256i idx3 = _mm256_slli_epi32(_mm256_and_si256(packed, mask6), 24);
    
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

// Register-based LUT approach - load all characters into registers
static inline __m256i lut_lookup_avx2(const __m256i& indices) {
    // Hoist constants to avoid repeated broadcasts
    static const __m256i A_base = _mm256_set1_epi8('A');
    static const __m256i a_base = _mm256_set1_epi8('a' - 26);
    static const __m256i digit_base = _mm256_set1_epi8('0' - 52);
    static const __m256i plus = _mm256_set1_epi8('+');
    static const __m256i slash = _mm256_set1_epi8('/');
    static const __m256i const26 = _mm256_set1_epi8(26);
    static const __m256i const52 = _mm256_set1_epi8(52);
    static const __m256i const62 = _mm256_set1_epi8(62);
    
    // Create masks for different ranges
    __m256i lt26 = _mm256_cmpgt_epi8(const26, indices);  // 0-25: A-Z
    __m256i lt52 = _mm256_cmpgt_epi8(const52, indices);  // 26-51: a-z  
    __m256i lt62 = _mm256_cmpgt_epi8(const62, indices);  // 52-61: 0-9
    __m256i eq62 = _mm256_cmpeq_epi8(indices, const62);  // 62: +
    // index 63 is '/' (default case)
    
    // Calculate characters for each range
    __m256i AZ_chars = _mm256_add_epi8(A_base, indices);               // A + (0-25)
    __m256i az_chars = _mm256_add_epi8(a_base, indices);               // a + (26-51) - 26
    __m256i digit_chars = _mm256_add_epi8(digit_base, indices);        // 0 + (52-61) - 52
    
    // Select appropriate character based on index range (default is '/')
    __m256i result = slash;                                            // Default: /
    result = _mm256_blendv_epi8(result, digit_chars, lt62);            // 0-9 if < 62
    result = _mm256_blendv_epi8(result, az_chars, lt52);               // a-z if < 52
    result = _mm256_blendv_epi8(result, AZ_chars, lt26);               // A-Z if < 26
    result = _mm256_blendv_epi8(result, plus, eq62);                   // + if == 62
    
    return result;
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

static inline __m256i map_base64_1pshufb_merged(__m256i idx) {
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




// AVX2 base64 chunk processor using permute operations
static inline __m256i process_avx2_chunk_direct(const uint8_t* src) {
    // Lemire approach: __m256i in = _mm256_maskload_epi32(reinterpret_cast<const int*>(src - 4), _mm256_set_epi32(0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x00000000));
    
    // Load two consecutive registers
    __m256i r0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src));
    __m256i r1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + 16));
    
    // Create proper layout using lane-local operations
    __m256i zeros = _mm256_setzero_si256();
    
    // Lower lane: bytes 0-11 with 4-byte offset
    __m256i lower_shifted = _mm256_alignr_epi8(r0, zeros, 12);
    
    // Upper lane: extract bytes 12-23 and position at 16-27
    __m256i r0_for_upper = _mm256_permute2x128_si256(r0, r0, 0x00);
    __m256i r1_for_upper = _mm256_permute2x128_si256(r1, r1, 0x00);
    __m256i bytes_12_23 = _mm256_alignr_epi8(r1_for_upper, r0_for_upper, 12);
    __m256i upper_shifted = _mm256_permute2x128_si256(zeros, bytes_12_23, 0x20);
    
    // Combine to match maskload layout
    __m256i in = _mm256_blend_epi32(lower_shifted, upper_shifted, 0x70);
    
    // shuffle for triplet extraction
    static const __m256i shuffle = _mm256_setr_epi8(
        6, 5, 4, -1,   9, 8, 7, -1,  12,11,10, -1,  15,14,13, -1,
        2, 1, 0, -1,   5, 4, 3, -1,   8, 7, 6, -1,  11,10, 9, -1
    );
    
    __m256i packed = _mm256_shuffle_epi8(in, shuffle);
    
    // Extract base64 indices
    static const __m256i mask6 = _mm256_set1_epi32(0x3f);
    __m256i idx0 = _mm256_and_si256(_mm256_srli_epi32(packed, 18), mask6);
    __m256i idx1 = _mm256_slli_epi32(_mm256_and_si256(_mm256_srli_epi32(packed, 12), mask6), 8);
    __m256i idx2 = _mm256_slli_epi32(_mm256_and_si256(_mm256_srli_epi32(packed, 6), mask6), 16);
    __m256i idx3 = _mm256_slli_epi32(_mm256_and_si256(packed, mask6), 24);
    __m256i indices_vec = _mm256_or_si256(_mm256_or_si256(idx0, idx1), _mm256_or_si256(idx2, idx3));
    
    //return lut_lookup_avx2_selects(indices_vec);
    return map_base64_1pshufb_32(indices_vec);
}

[[gnu::hot, gnu::flatten]] inline std::string fast_sse_base64_encode_avx(const std::vector<uint8_t>& data) {
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
    if (avx_buffer.size() < out_len) {
        avx_buffer.resize(out_len);
    }
    
    const uint8_t* src = data.data();
    char* dest = avx_buffer.data();
    const char* const dest_orig = dest;
    

    
    // Process 24-byte blocks directly with AVX2 - full register utilization
    while (len >= 48) {
        __m256i chars = process_avx2_chunk_direct(src);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dest), chars);
        
        src += 24;  // Process 24 input bytes -> 32 base64 chars
        dest += 32;
        len -= 24;
    }
    
    _mm256_zeroupper();
    
    // Fallback scalar processing for remaining bytes
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
    
    size_t actual_len = dest - dest_orig;
    avx_buffer.resize(actual_len);
    
    return avx_buffer;
}


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
    const __m256i t1 = _mm256_mulhi_epu16(t0, _mm256_set1_epi32(0x04000040));
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

[[gnu::hot, gnu::flatten]] inline std::string fast_avx2_base64_encode_lemire(const std::vector<uint8_t>& data) {
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
    lemire_buffer.clear();
    lemire_buffer.resize(out_len);
    
    const uint8_t* src = data.data();
    char* dest = lemire_buffer.data();
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
    
    lemire_buffer.resize(dest - dest_orig);
    return lemire_buffer;
}
// Custom aligned allocator implementation
template<typename T, std::size_t Alignment = 16>
class aligned_allocator {
public:
    static_assert((Alignment & (Alignment - 1)) == 0, "Alignment must be a power of two");
    static_assert(Alignment >= alignof(T), "Alignment must be >= alignof(T)");

    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using size_type = std::size_t;

    template<typename U>
    struct rebind { using other = aligned_allocator<U, Alignment>; };

    using is_always_equal = std::true_type;

    aligned_allocator() = default;
    template<typename U>
    aligned_allocator(const aligned_allocator<U, Alignment>&) {}

    pointer allocate(size_type n) {
        if (n > std::numeric_limits<size_type>::max() / sizeof(T))
            throw std::bad_array_new_length();
        const std::size_t bytes = n * sizeof(T);
        void* p = ::operator new[](bytes, std::align_val_t(Alignment));
        return static_cast<pointer>(p);
    }

    void deallocate(pointer p, size_type) noexcept {
        ::operator delete[](p, std::align_val_t(Alignment));
    }

    template<typename U>
    bool operator==(const aligned_allocator<U, Alignment>&) const noexcept { return true; }
    template<typename U>
    bool operator!=(const aligned_allocator<U, Alignment>&) const noexcept { return false; }
};

// Thread-local aligned buffers using custom aligned allocator
thread_local std::vector<uint8_t, aligned_allocator<uint8_t, 16>> aligned_input_buffer;
thread_local std::vector<char, aligned_allocator<char, 16>> aligned_output_buffer;

[[gnu::hot, gnu::flatten]] inline std::string fast_sse_base64_encode_aligned(const std::vector<uint8_t>& data) {
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

static inline void sse2_copy_unaligned_to_aligned(void* dstv, const void* srcv, size_t n) {
    unsigned char*       dst = static_cast<unsigned char*>(dstv);
    const unsigned char* src = static_cast<const unsigned char*>(srcv);

    // Optional: assert 16B alignment for dst if desired
    // assert((reinterpret_cast<uintptr_t>(dst) & 15) == 0);

    // 64B unrolled body
    while (n >= 64) {
        __m128i a0 = _mm_loadu_si128((const __m128i*)(src +  0));
        __m128i a1 = _mm_loadu_si128((const __m128i*)(src + 16));
        __m128i a2 = _mm_loadu_si128((const __m128i*)(src + 32));
        __m128i a3 = _mm_loadu_si128((const __m128i*)(src + 48));
        _mm_store_si128((__m128i*)(dst +  0), a0);  // dst must be 16B-aligned
        _mm_store_si128((__m128i*)(dst + 16), a1);
        _mm_store_si128((__m128i*)(dst + 32), a2);
        _mm_store_si128((__m128i*)(dst + 48), a3);
        src += 64; dst += 64; n -= 64;
    }

    // 16B body
    while (n >= 16) {
        __m128i a = _mm_loadu_si128((const __m128i*)src);
        _mm_store_si128((__m128i*)dst, a);
        src += 16; dst += 16; n -= 16;
    }

    // Tail
    for (size_t i = 0; i < n; ++i) dst[i] = src[i];
}

[[gnu::hot, gnu::flatten, clang::always_inline]] std::string fast_sse_base64_encode_aligned_alt(const std::vector<uint8_t>& data) {
    if (debug_enabled) {
        DEBUG_LOG("fast_sse_base64_encode_aligned_alt called with " + std::to_string(data.size()) + " bytes");
    }
    size_t len = data.size();
    
    // Fast path for tiny inputs - avoid SIMD overhead
    if (len <= 24) {
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
    
    // Ensure aligned buffers are large enough (need extra padding for aligned loads)
    size_t input_buf_size = len + 32;  // Extra padding for aligned loads past end
    size_t output_buf_size = out_len + 64;
    
    if (aligned_input_buffer.size() < input_buf_size) {
        aligned_input_buffer.resize(input_buf_size);
    }
    if (aligned_output_buffer.size() < output_buf_size) {
        aligned_output_buffer.resize(output_buf_size);
    }
    
    // Check if input data is already aligned
    bool input_is_aligned = (reinterpret_cast<uintptr_t>(data.data()) % 16 == 0);
    if (debug_enabled) {
        DEBUG_LOG("Alt: Input data alignment check: " + std::to_string(input_is_aligned) + " (addr: " + std::to_string(reinterpret_cast<uintptr_t>(data.data())) + ")");
    }
    
    const uint8_t* src;
    bool use_aligned_loads;
    if (input_is_aligned) {
        // Use original data directly if aligned
        src = data.data();
        use_aligned_loads = true;
        if (debug_enabled) {
            DEBUG_LOG("Alt: Using original aligned data directly");
        }
    } else {
        // Copy input data to aligned buffer using optimized SSE2 copy
        if (debug_enabled) {
            DEBUG_LOG("Alt: Copying " + std::to_string(len) + " bytes to aligned buffer with SSE2");
        }
        sse2_copy_unaligned_to_aligned(aligned_input_buffer.data(), data.data(), len);
        src = aligned_input_buffer.data();
        use_aligned_loads = true;
    }
    char* dest = aligned_output_buffer.data();
    const char* const dest_orig = dest;
    
    if (debug_enabled) {
        DEBUG_LOG("Alt: Buffer setup complete - src aligned: " + std::to_string(reinterpret_cast<uintptr_t>(src) % 16 == 0));
    }

    static const __m128i trip_shuffle = _mm_setr_epi8(
        2, 1, 0, (char)0x80,   // lane0: bytes 2,1,0 to match (b0<<16)|(b1<<8)|b2
        5, 4, 3, (char)0x80,   // lane1: bytes 5,4,3
        8, 7, 6, (char)0x80,   // lane2: bytes 8,7,6
       11,10, 9, (char)0x80    // lane3: bytes 11,10,9
    );

    // Use streaming stores for large data (≥8KB)
    bool use_streaming = (data.size() >= 8192);
    
    // Process 48-byte blocks with single loop
    if (debug_enabled && len >= 48) {
        DEBUG_LOG("SSE Aligned Alt encoder: processing " + std::to_string(len) + " bytes with SIMD");
    }
    while (len >= 48) {
        if (debug_enabled) {
            DEBUG_LOG("Alt: Processing 48-byte block, remaining: " + std::to_string(len));
        }
        // Load 3 16-byte blocks (covers 48 input bytes)
        __m128i block0, block1, block2;
        if (use_aligned_loads) {
            block0 = _mm_load_si128(reinterpret_cast<const __m128i*>(src));
            block1 = _mm_load_si128(reinterpret_cast<const __m128i*>(src + 16));
            block2 = _mm_load_si128(reinterpret_cast<const __m128i*>(src + 32));
        } else {
            block0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src));
            block1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + 16));
            block2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + 32));
        }
        
        if (debug_enabled) {
            DEBUG_LOG("Alt: Loaded 3 " + std::string(use_aligned_loads ? "aligned" : "unaligned") + " blocks");
        }
        
        // Extract 12-byte chunks using alignr
        __m128i chunk0 = block0;                                    // bytes 0-11
        __m128i chunk1 = _mm_alignr_epi8(block1, block0, 12);      // bytes 12-23
        __m128i chunk2 = _mm_alignr_epi8(block2, block1, 8);       // bytes 24-35
        __m128i chunk3 = _mm_srli_si128(block2, 4);                // bytes 36-47
        
        __m128i packed0 = _mm_shuffle_epi8(chunk0, trip_shuffle);
        __m128i packed1 = _mm_shuffle_epi8(chunk1, trip_shuffle);
        __m128i packed2 = _mm_shuffle_epi8(chunk2, trip_shuffle);
        __m128i packed3 = _mm_shuffle_epi8(chunk3, trip_shuffle);
        
        __m128i idx0_unpacked = extract_indices_to_bytes(packed0);
        __m128i idx1_unpacked = extract_indices_to_bytes(packed1);
        __m128i idx2_unpacked = extract_indices_to_bytes(packed2);
        __m128i idx3_unpacked = extract_indices_to_bytes(packed3);
        
        __m128i chars0 = lut_lookup(idx0_unpacked);
        __m128i chars1 = lut_lookup(idx1_unpacked);
        __m128i chars2 = lut_lookup(idx2_unpacked);
        __m128i chars3 = lut_lookup(idx3_unpacked);
        
        if (use_streaming) {
            _mm_stream_si128(reinterpret_cast<__m128i*>(dest + 0), chars0);
            _mm_stream_si128(reinterpret_cast<__m128i*>(dest + 16), chars1);
            _mm_stream_si128(reinterpret_cast<__m128i*>(dest + 32), chars2);
            _mm_stream_si128(reinterpret_cast<__m128i*>(dest + 48), chars3);
        } else {
            _mm_store_si128(reinterpret_cast<__m128i*>(dest + 0), chars0);
            _mm_store_si128(reinterpret_cast<__m128i*>(dest + 16), chars1);
            _mm_store_si128(reinterpret_cast<__m128i*>(dest + 32), chars2);
            _mm_store_si128(reinterpret_cast<__m128i*>(dest + 48), chars3);
        }
        
        src += 48;
        dest += 64;
        len -= 48;
        if (debug_enabled) {
            DEBUG_LOG("Alt: Block complete, remaining: " + std::to_string(len));
        }
    }
    
    // Ensure all streaming stores are completed
    if (use_streaming) {
        if (debug_enabled) {
            DEBUG_LOG("Alt: Memory fence for streaming stores");
        }
        _mm_sfence();
    }
    
    _mm256_zeroupper();
    
    // Fallback scalar processing
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
    
    size_t actual_len = dest - dest_orig;
    
    if (debug_enabled) {
        DEBUG_LOG("Alt: Complete - output length: " + std::to_string(actual_len));
    }
    
    return std::string(dest_orig, actual_len);
}

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

[[gnu::hot, gnu::flatten]] inline std::string fast_mula_base64_encode(const std::vector<uint8_t>& data) {
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
    mula_buffer.clear();
    mula_buffer.resize(out_len);
    
    const uint8_t* src = data.data();
    char* dest = mula_buffer.data();
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
    
    _mm256_zeroupper();
    
    mula_buffer.resize(dest - dest_orig);
    return mula_buffer;
}



// NEON helper functions for base64 encoding
static inline uint8x16_t extract_indices_to_bytes_neon(const uint32x4_t& packed) {
    const uint32x4_t mask6 = vdupq_n_u32(0x3f);
    uint32x4_t idx0 = vandq_u32(vshrq_n_u32(packed, 18), mask6);
    uint32x4_t idx1 = vshlq_n_u32(vandq_u32(vshrq_n_u32(packed, 12), mask6), 8);
    uint32x4_t idx2 = vshlq_n_u32(vandq_u32(vshrq_n_u32(packed, 6), mask6), 16);
    uint32x4_t idx3 = vshlq_n_u32(vandq_u32(packed, mask6), 24);
    uint32x4_t combined = vorrq_u32(vorrq_u32(idx0, idx1), vorrq_u32(idx2, idx3));
    return vreinterpretq_u8_u32(combined);
}

alignas(16) static const uint8_t b64_tbl_bytes[64] = {
    'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P',
    'Q','R','S','T','U','V','W','X','Y','Z','a','b','c','d','e','f',
    'g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v',
    'w','x','y','z','0','1','2','3','4','5','6','7','8','9','+','/'
};

// indices: 16 bytes, each in 0..63
static inline uint8x16_t lut_lookup_neon64(uint8x16_t indices) {
    // Build a 64-byte table in four q-registers once; compilers typically hoist this.
    uint8x16x4_t tbl;
    tbl.val[0] = vld1q_u8(b64_tbl_bytes +  0);
    tbl.val[1] = vld1q_u8(b64_tbl_bytes + 16);
    tbl.val[2] = vld1q_u8(b64_tbl_bytes + 32);
    tbl.val[3] = vld1q_u8(b64_tbl_bytes + 48);

    // Single 64-byte table lookup: indices 0..63 select across {val[0],val[1],val[2],val[3]}
    return vqtbl4q_u8(tbl, indices);
}

static inline uint8x16_t lut_lookup_neon(const uint8x16_t& indices) {
    // NEON lookup tables (16 bytes each)
    const uint8x16_t lut0 = {65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80}; // A-P
    const uint8x16_t lut1 = {81,82,83,84,85,86,87,88,89,90,97,98,99,100,101,102}; // Q-Z,a-f
    const uint8x16_t lut2 = {103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118}; // g-v
    const uint8x16_t lut3 = {119,120,121,122,48,49,50,51,52,53,54,55,56,57,43,47}; // w-z,0-9,+,/
    
    const uint8x16_t const16 = vdupq_n_u8(16);
    const uint8x16_t const32 = vdupq_n_u8(32);
    const uint8x16_t const48 = vdupq_n_u8(48);
    
    uint8x16_t mask_32 = vcltq_u8(indices, const32);
    uint8x16_t lo_result = vbslq_u8(
        vcltq_u8(indices, const16),
        vqtbl1q_u8(lut0, indices),
        vqtbl1q_u8(lut1, vsubq_u8(indices, const16))
    );
    uint8x16_t hi_result = vbslq_u8(
        vcltq_u8(indices, const48),
        vqtbl1q_u8(lut2, vsubq_u8(indices, const32)),
        vqtbl1q_u8(lut3, vsubq_u8(indices, const48))
    );
    return vbslq_u8(mask_32, lo_result, hi_result);
}

/**
 * Fast NEON base64 encoder using 128-bit SIMD instructions
 * 
 * Algorithm:
 * 1. Process 48 input bytes as 16 triplets (3 bytes each)
 * 2. Extract 4 base64 indices per triplet using bit shifts
 * 3. Lookup base64 characters from alphabet table using NEON table lookup
 * 4. Produces exactly 64 base64 characters per 48-byte block
 * 
 * Performance: ~2-3x faster than scalar for large inputs
 * ARM NEON equivalent of SSE implementation
 * 
 * @param data Input byte vector to encode
 * @return Base64 encoded string (unpadded for Matrix protocol)
 */
[[gnu::hot, gnu::flatten]] inline std::string fast_neon_base64_encode(const std::vector<uint8_t>& data) {
    size_t len = data.size();
    
    // Fast path for tiny inputs - avoid SIMD overhead
    if (len <= 24) {
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
    if (neon_buffer.size() < out_len) {
        neon_buffer.resize(out_len);
    }
    
    const uint8_t* src = data.data();
    char* dest = neon_buffer.data();
    const char* const dest_orig = dest;
    
    // NEON shuffle pattern for triplet extraction
    // Note: NEON uses different byte ordering than x86
    const uint8x16_t trip_shuffle = {2,1,0,255, 5,4,3,255, 8,7,6,255, 11,10,9,255};
    
    // Process 48-byte blocks with NEON
    while (len >= 48) {
        // Process 12 triplets with 4x unrolled NEON
        uint8x16_t in0 = vld1q_u8(src + 0);
        uint8x16_t in1 = vld1q_u8(src + 12);
        uint8x16_t in2 = vld1q_u8(src + 24);
        uint8x16_t in3 = vld1q_u8(src + 36);
        
        uint8x16_t packed0 = vqtbl1q_u8(in0, trip_shuffle);
        uint8x16_t packed1 = vqtbl1q_u8(in1, trip_shuffle);
        uint8x16_t packed2 = vqtbl1q_u8(in2, trip_shuffle);
        uint8x16_t packed3 = vqtbl1q_u8(in3, trip_shuffle);
        
        uint8x16_t idx0_unpacked = extract_indices_to_bytes_neon(vreinterpretq_u32_u8(packed0));
        uint8x16_t idx1_unpacked = extract_indices_to_bytes_neon(vreinterpretq_u32_u8(packed1));
        uint8x16_t idx2_unpacked = extract_indices_to_bytes_neon(vreinterpretq_u32_u8(packed2));
        uint8x16_t idx3_unpacked = extract_indices_to_bytes_neon(vreinterpretq_u32_u8(packed3));
        
        uint8x16_t chars0 = lut_lookup_neon64(idx0_unpacked);
        uint8x16_t chars1 = lut_lookup_neon64(idx1_unpacked);
        uint8x16_t chars2 = lut_lookup_neon64(idx2_unpacked);
        uint8x16_t chars3 = lut_lookup_neon64(idx3_unpacked);
        
        vst1q_u8(reinterpret_cast<uint8_t*>(dest + 0), chars0);
        vst1q_u8(reinterpret_cast<uint8_t*>(dest + 16), chars1);
        vst1q_u8(reinterpret_cast<uint8_t*>(dest + 32), chars2);
        vst1q_u8(reinterpret_cast<uint8_t*>(dest + 48), chars3);
        
        src += 48;
        dest += 64;
        len -= 48;
    }
    
    _mm256_zeroupper();
    
    // Fallback scalar processing
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
    
    size_t actual_len = dest - dest_orig;
    neon_buffer.resize(actual_len);
    
    return neon_buffer;
}

std::string openssl_base64_encode(const std::vector<uint8_t>& data) {
    BIO *bio, *b64;
    BUF_MEM *bufferPtr;
    
    b64 = BIO_new(BIO_f_base64());
    bio = BIO_new(BIO_s_mem());
    bio = BIO_push(b64, bio);
    
    BIO_set_flags(bio, BIO_FLAGS_BASE64_NO_NL);
    BIO_write(bio, data.data(), data.size());
    BIO_flush(bio);
    BIO_get_mem_ptr(bio, &bufferPtr);
    
    std::string result(bufferPtr->data, bufferPtr->length);
    BIO_free_all(bio);
    return result;
}

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
    

    while (srclen >= 16) {
        // Process 4 chars at a time using scalar logic
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
    
    _mm256_zeroupper();

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

/**
 * Pack 4x6-bit sextets into 3x8-bit bytes using AVX2 bit manipulation
 * 
 * Input:  __m256i containing 32 bytes, each representing a 6-bit sextet (0-63)
 * Output: __m256i with packed 24 bytes of decoded binary data
 * 
 * Algorithm:
 * 1. Use multiply-add to combine adjacent sextets: (a<<6)|b and (c<<6)|d
 * 2. Use multiply-add again to create 24-bit triplets: ((a<<6)|b)<<12 | ((c<<6)|d)
 * 3. Shuffle bytes to extract the 3 meaningful bytes from each 32-bit word
 * 4. Permute to compact the result into the lower 192 bits (24 bytes)
 * 
 * @param sextets AVX2 register containing 32 base64 indices (0-63)
 * @return AVX2 register with 24 bytes of packed binary data
 */
// Hoisted constants for pack_4x6_to_3x8 to avoid re-materialization
static const __m256i pack_mul1 = _mm256_set1_epi32(0x01400140);
static const __m256i pack_mul2 = _mm256_set1_epi32(0x00011000);
static const __m256i pack_shuf = _mm256_setr_epi8(
    2,1,0, 6,5,4, 10,9,8, 14,13,12, -1,-1,-1,-1,
    2,1,0, 6,5,4, 10,9,8, 14,13,12, -1,-1,-1,-1
);
static const __m256i pack_perm = _mm256_setr_epi32(0,1,2,4,5,6,-1,-1);

static inline __m256i pack_4x6_to_3x8(__m256i sextets) {
    // Step 1: Combine adjacent sextets using multiply-add
    __m256i ab_bc = _mm256_maddubs_epi16(sextets, pack_mul1);
    
    // Step 2: Combine pairs into 24-bit values
    __m256i packed = _mm256_madd_epi16(ab_bc, pack_mul2);
    
    // Step 3: Extract meaningful bytes from each 32-bit word
    packed = _mm256_shuffle_epi8(packed, pack_shuf);
    
    // Step 4: Compact result by removing unused high bytes
    return _mm256_permutevar8x32_epi32(packed, pack_perm);
}

/**
 * Convert 32 ASCII base64 characters to sextets using AVX2 range comparisons
 * 
 * Uses parallel range checking to validate and convert base64 characters:
 * - 'A'-'Z' (65-90)  -> 0-25
 * - 'a'-'z' (97-122) -> 26-51  
 * - '0'-'9' (48-57)  -> 52-61
 * - '+'     (43)     -> 62
 * - '/'     (47)     -> 63
 * 
 * Algorithm:
 * 1. Load 32 characters into AVX2 register
 * 2. Create range masks for each character class using comparisons
 * 3. Compute sextet values for each class in parallel
 * 4. Blend results based on character class masks
 * 5. Validate all characters are valid base64 (reject padding '=')
 * 
 * @param src Pointer to 32 base64 characters
 * @param out_idx Output AVX2 register containing 32 sextets (0-63)
 * @return true if all characters valid, false if any invalid characters found
 */

static const __m256i kA      = _mm256_set1_epi8('A');
static const __m256i kZ      = _mm256_set1_epi8('Z');
static const __m256i ka      = _mm256_set1_epi8('a');
static const __m256i kz      = _mm256_set1_epi8('z');
static const __m256i k0      = _mm256_set1_epi8('0');
static const __m256i k9      = _mm256_set1_epi8('9');
static const __m256i kPlus   = _mm256_set1_epi8('+');
static const __m256i kSlash  = _mm256_set1_epi8('/');
static const __m256i kAll1   = _mm256_set1_epi8((char)0xFF);

// class offsets relative to base idx = c - 'A'
static const __m256i off_az  = _mm256_set1_epi8((char)-6);  // +26 for 'a'..'z'
static const __m256i off_09  = _mm256_set1_epi8((char)69);  // +52 for '0'..'9'
static const __m256i off_p   = _mm256_set1_epi8((char)84);  //  62 for '+'
static const __m256i off_s   = _mm256_set1_epi8((char)81);  //  63 for '/'

// Saturated-subtract range test helper: x in [lo, hi]  <=>  subs_epu8(x,lo) == subs_epu8(x,hi+1)
static inline __m256i in_range_u8(__m256i x, __m256i lo, __m256i hip1) {
    return _mm256_cmpeq_epi8(_mm256_subs_epu8(x, lo), _mm256_subs_epu8(x, hip1));
}

static inline bool map32_ascii_to_sextets_avx2_alt2(const char* src, __m256i& out_idx) {
    __m256i c = _mm256_loadu_si256((const __m256i*)src);

    // signed compares: x >= k  => cmpgt(x, k-1);  x <= k => cmpgt(k+1, x)
    auto ge = [](__m256i x, __m256i k){ return _mm256_cmpgt_epi8(x, _mm256_sub_epi8(k, _mm256_set1_epi8(1))); };
    auto le = [](__m256i x, __m256i k){ return _mm256_cmpgt_epi8(_mm256_add_epi8(k, _mm256_set1_epi8(1)), x); };

    __m256i is_AZ = _mm256_and_si256(ge(c,kA), le(c,kZ));
    __m256i is_az = _mm256_and_si256(ge(c,ka), le(c,kz));
    __m256i is_09 = _mm256_and_si256(ge(c,k0), le(c,k9));
    __m256i is_p  = _mm256_cmpeq_epi8(c, kPlus);
    __m256i is_s  = _mm256_cmpeq_epi8(c, kSlash);

    __m256i cls = _mm256_or_si256(_mm256_or_si256(is_AZ, is_az),
                   _mm256_or_si256(_mm256_or_si256(is_09, is_p), is_s));
    if (!_mm256_testc_si256(cls, kAll1)) return false;

    __m256i idx = _mm256_sub_epi8(c, kA);
    idx = _mm256_add_epi8(idx, _mm256_and_si256(is_az, off_az));
    idx = _mm256_add_epi8(idx, _mm256_and_si256(is_09, off_09));
    idx = _mm256_add_epi8(idx, _mm256_and_si256(is_p,  off_p));
    idx = _mm256_add_epi8(idx, _mm256_and_si256(is_s,  off_s));

    out_idx = idx;
    return true;
}

// Hoisted constants for map32_ascii_to_sextets_avx2_alt
static const __m256i char_A = _mm256_set1_epi8('A');
static const __m256i char_Z = _mm256_set1_epi8('Z');
static const __m256i char_a = _mm256_set1_epi8('a');
static const __m256i char_z = _mm256_set1_epi8('z');
static const __m256i char_0 = _mm256_set1_epi8('0');
static const __m256i char_9 = _mm256_set1_epi8('9');
static const __m256i char_plus = _mm256_set1_epi8('+');
static const __m256i char_slash = _mm256_set1_epi8('/');
static const __m256i offset_neg6 = _mm256_set1_epi8((char)-6);
static const __m256i offset_69 = _mm256_set1_epi8((char)69);
static const __m256i offset_84 = _mm256_set1_epi8((char)84);
static const __m256i offset_81 = _mm256_set1_epi8((char)81);
static const __m256i all_ones = _mm256_set1_epi8((char)0xFF);


static inline bool map32_ascii_to_sextets_avx2_alt(const char* src, __m256i& out_idx) {
    __m256i c = _mm256_loadu_si256((const __m256i*)src);

    // Range masks: x >= k and x <= k
    auto ge = [](__m256i x, __m256i k){ return _mm256_cmpgt_epi8(x, _mm256_sub_epi8(k, _mm256_set1_epi8(1))); };
    auto le = [](__m256i x, __m256i k){ return _mm256_cmpgt_epi8(_mm256_add_epi8(k, _mm256_set1_epi8(1)), x); };

    __m256i is_AZ = _mm256_and_si256(ge(c, char_A), le(c, char_Z));
    __m256i is_az = _mm256_and_si256(ge(c, char_a), le(c, char_z));
    __m256i is_09 = _mm256_and_si256(ge(c, char_0), le(c, char_9));
    __m256i is_p  = _mm256_cmpeq_epi8(c, char_plus);
    __m256i is_s  = _mm256_cmpeq_epi8(c, char_slash);

    // Validate: every byte must be in exactly one class
    __m256i cls = _mm256_or_si256(_mm256_or_si256(is_AZ, is_az),
                   _mm256_or_si256(_mm256_or_si256(is_09, is_p), is_s));
    if (!_mm256_testc_si256(cls, all_ones)) return false;

    // Base index for 'A'..'Z'
    __m256i idx = _mm256_sub_epi8(c, char_A);

    // Masked adds to map other classes to their 0..63 indices
    idx = _mm256_add_epi8(idx, _mm256_and_si256(is_az, offset_neg6));
    idx = _mm256_add_epi8(idx, _mm256_and_si256(is_09, offset_69));
    idx = _mm256_add_epi8(idx, _mm256_and_si256(is_p, offset_84));
    idx = _mm256_add_epi8(idx, _mm256_and_si256(is_s, offset_81));

    out_idx = idx;
    return true;
}


static inline bool map32_ascii_to_sextets_avx2(const char* src, __m256i& out_idx) {
    // Load 32 ASCII characters
    __m256i c = _mm256_loadu_si256((const __m256i*)src);
    
    // Helper lambdas for range comparisons (x >= k and x <= k)
    auto ge = [](__m256i x, int k){ return _mm256_cmpgt_epi8(x, _mm256_set1_epi8(k-1)); };  // x >= k
    auto le = [](__m256i x, int k){ return _mm256_cmpgt_epi8(_mm256_set1_epi8(k+1), x); };  // x <= k

    // Create masks for each base64 character class
    __m256i is_AZ = _mm256_and_si256(ge(c,'A'), le(c,'Z'));  // 'A'-'Z': uppercase letters
    __m256i is_az = _mm256_and_si256(ge(c,'a'), le(c,'z'));  // 'a'-'z': lowercase letters  
    __m256i is_09 = _mm256_and_si256(ge(c,'0'), le(c,'9'));  // '0'-'9': digits
    __m256i is_p  = _mm256_cmpeq_epi8(c, _mm256_set1_epi8('+'));  // '+': plus sign
    __m256i is_s  = _mm256_cmpeq_epi8(c, _mm256_set1_epi8('/'));  // '/': slash

    // Compute sextet values for each character class in parallel
    __m256i v_AZ = _mm256_sub_epi8(c, _mm256_set1_epi8('A'));                        // 'A'-'Z' -> 0-25
    __m256i v_az = _mm256_add_epi8(_mm256_sub_epi8(c, _mm256_set1_epi8('a')), _mm256_set1_epi8(26)); // 'a'-'z' -> 26-51
    __m256i v_09 = _mm256_add_epi8(_mm256_sub_epi8(c, _mm256_set1_epi8('0')), _mm256_set1_epi8(52)); // '0'-'9' -> 52-61
    __m256i v_p  = _mm256_set1_epi8(62);  // '+' -> 62
    __m256i v_s  = _mm256_set1_epi8(63);  // '/' -> 63

    // Blend results based on character class masks
    // Start with zeros, then conditionally replace with computed values
    __m256i idx = _mm256_setzero_si256();
    idx = _mm256_blendv_epi8(idx, v_AZ, is_AZ);  // Apply uppercase letter values
    idx = _mm256_blendv_epi8(idx, v_az, is_az);  // Apply lowercase letter values
    idx = _mm256_blendv_epi8(idx, v_09, is_09);  // Apply digit values
    idx = _mm256_blendv_epi8(idx, v_p,  is_p);   // Apply plus sign value
    idx = _mm256_blendv_epi8(idx, v_s,  is_s);   // Apply slash value

    // Validation: ensure all characters belong to valid base64 alphabet
    // Combine all class masks - every character must match exactly one class
    __m256i cls = _mm256_or_si256(_mm256_or_si256(is_AZ, is_az),
                   _mm256_or_si256(_mm256_or_si256(is_09, is_p), is_s));
    const __m256i all1 = _mm256_set1_epi8((char)0xFF);
    if (!_mm256_testc_si256(cls, all1)) return false;  // Reject any non-alphabet chars (including '=')

    out_idx = idx;
    return true;
}

/**
 * High-performance AVX2 base64 decoder using range comparison validation
 * 
 * Optimized for Matrix protocol unpadded base64 data (cryptographic signatures,
 * event IDs, etc.). Uses AVX2 SIMD instructions for parallel character validation
 * and conversion, with scalar fallback for edge cases.
 * 
 * Features:
 * - Processes 32 characters per AVX2 iteration (4x faster than scalar)
 * - Range-based validation rejects invalid characters early
 * - Handles unpadded base64 (no '=' padding required)
 * - Zero-copy output with minimal memory allocation
 * - Graceful fallback to scalar processing for remainder
 * 
 * Algorithm Flow:
 * 1. Validate input length (reject length % 4 == 1)
 * 2. AVX2 loop: process 32-char chunks -> 24 decoded bytes
 * 3. Scalar loop: process remaining 4-char groups -> 3 decoded bytes  
 * 4. Handle unpadded tail (2-3 remaining characters)
 * 
 * @param input Base64 encoded string view (unpadded)
 * @return Decoded binary data as byte vector, empty if invalid
 */
std::vector<uint8_t> fast_base64_decode_avx2_rangecmp(std::string_view input) {
    const char* src = input.data();
    size_t      n   = input.size();

    // Input validation: reject invalid lengths for unpadded base64
    // Length % 4 == 1 is impossible for valid base64 (would require fractional input bytes)
    if ((n & 3U) == 1U) return {};  // Invalid length for unpadded base64
    
    // Pre-allocate output buffer with upper bound estimate
    // Each 4 base64 chars -> 3 bytes, +2 extra for unpadded tail handling
    std::vector<uint8_t> out((n/4)*3 + 2);
    uint8_t* dst = out.data();

#if defined(__AVX2__)
    // AVX2 SIMD processing loop: handle 32 characters at a time
    // 32 base64 chars -> 24 decoded bytes per iteration
    while (n >= 32) {
        __m256i idx;
        // Convert ASCII characters to base64 indices (0-63)
        // Returns false if any invalid characters found
        if (!map32_ascii_to_sextets_avx2_alt2(src, idx)) break;
        
        // Pack 32 sextets (6-bit values) into 24 bytes (8-bit values)
        __m256i packed = pack_4x6_to_3x8(idx);
        
        // Store 24 decoded bytes: 16 bytes + 8 bytes
        // Use unaligned stores since output buffer may not be aligned
        _mm_storeu_si128((__m128i*)dst, _mm256_castsi256_si128(packed));           // Store lower 16 bytes
        _mm_storel_epi64((__m128i*)(dst + 16), _mm256_extracti128_si256(packed, 1)); // Store upper 8 bytes
        
        src += 32; dst += 24; n -= 32;  // Advance pointers and decrement remaining count
    }
#endif

    // Scalar decode lookup table for remaining characters
    // Maps ASCII values to base64 indices (0-63), invalid chars -> 255
    // Note: '=' (padding) maps to 255 since we handle unpadded base64
    static constexpr uint8_t D[256] = {
        /* 0-31: Control chars */
        255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
        255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
        /* 32-47: Space, punctuation */
        255,255,255,255,255,255,255,255,255,255,255, 62,255,255,255, 63,  // '+' -> 62, '/' -> 63
        /* 48-63: Digits, punctuation */
         52, 53, 54, 55, 56, 57, 58, 59, 60, 61,255,255,255,255,255,255,  // '0'-'9' -> 52-61, '=' -> 255
        /* 64-79: @, A-O */
        255,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14,  // 'A'-'O' -> 0-14
        /* 80-95: P-Z, punctuation */
         15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,255,255,255,255,255,  // 'P'-'Z' -> 15-25
        /* 96-111: `, a-o */
        255, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,  // 'a'-'o' -> 26-40
        /* 112-127: p-z, punctuation */
         41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51,255,255,255,255,255,  // 'p'-'z' -> 41-51
        /* 128-255: Extended ASCII (all invalid) */
        255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
        255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
        255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
        255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
        255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
        255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
        255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
        255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255
    };

    // Scalar processing loop: handle remaining complete 4-character groups
    // Each group of 4 base64 chars decodes to exactly 3 bytes
    while (n >= 4) {
        // Look up base64 indices for all 4 characters
        uint8_t a = D[(unsigned char)src[0]];  // First sextet (bits 23-18)
        uint8_t b = D[(unsigned char)src[1]];  // Second sextet (bits 17-12)
        uint8_t c = D[(unsigned char)src[2]];  // Third sextet (bits 11-6)
        uint8_t d4= D[(unsigned char)src[3]];  // Fourth sextet (bits 5-0)
        
        // Check for invalid characters (any lookup returned 255)
        if ((a|b|c|d4) == 255) break; // Invalid character found, stop processing
        
        // Combine 4 sextets into 24-bit value: aaaaaabbbbbbccccccdddddd
        uint32_t v = (a<<18)|(b<<12)|(c<<6)|d4;
        
        // Extract 3 bytes from the 24-bit value
        dst[0] = (v >> 16) & 0xFF;  // Bits 23-16: first output byte
        dst[1] = (v >>  8) & 0xFF;  // Bits 15-8:  second output byte  
        dst[2] =  v        & 0xFF;  // Bits 7-0:   third output byte
        
        dst += 3; src += 4; n -= 4;  // Advance pointers
    }

    // Handle unpadded tail cases (remaining 0, 2, or 3 characters)
    // Note: n==1 is invalid for base64 and was rejected earlier
    if (n == 2) {
        // 2 chars -> 1 byte: decode first 12 bits only
        uint8_t a = D[(unsigned char)src[0]];
        uint8_t b = D[(unsigned char)src[1]];
        if ((a|b) == 255) return {};  // Invalid characters
        uint32_t v = (a<<18)|(b<<12);  // Only first 2 sextets matter
        dst[0] = (v >> 16) & 0xFF;     // Extract first byte (bits 23-16)
        dst += 1;
    } else if (n == 3) {
        // 3 chars -> 2 bytes: decode first 18 bits
        uint8_t a = D[(unsigned char)src[0]];
        uint8_t b = D[(unsigned char)src[1]];
        uint8_t c = D[(unsigned char)src[2]];
        if ((a|b|c) == 255) return {};  // Invalid characters
        uint32_t v = (a<<18)|(b<<12)|(c<<6);  // First 3 sextets
        dst[0] = (v >> 16) & 0xFF;     // Extract first byte (bits 23-16)
        dst[1] = (v >>  8) & 0xFF;     // Extract second byte (bits 15-8)
        dst += 2;
    } else if (n == 1) {
        return {}; // Invalid: single character cannot be decoded
    }

    // Resize output vector to actual decoded length and return
    out.resize(dst - out.data());
    return out;
}

// Hoisted constants for mapping and validation
static const __m128i kA_sse     = _mm_set1_epi8('A');
static const __m128i kZ_sse     = _mm_set1_epi8('Z');
static const __m128i ka_sse     = _mm_set1_epi8('a');
static const __m128i kz_sse     = _mm_set1_epi8('z');
static const __m128i k0_sse     = _mm_set1_epi8('0');
static const __m128i k9_sse     = _mm_set1_epi8('9');
static const __m128i kPlus_sse  = _mm_set1_epi8('+');
static const __m128i kSlash_sse = _mm_set1_epi8('/');
static const __m128i kAll1_sse  = _mm_set1_epi8((char)0xFF);

// Offsets for masked-add mapper (relative to base idx = c - 'A')
static const __m128i off_az_sse = _mm_set1_epi8((char)-6);  // 'a'..'z' -> +26
static const __m128i off_09_sse = _mm_set1_epi8((char) 69); // '0'..'9' -> +52
static const __m128i off_p_sse  = _mm_set1_epi8((char) 84); // '+' -> 62
static const __m128i off_s_sse  = _mm_set1_epi8((char) 81); // '/' -> 63

// Signed-compare helpers: x >= k => cmpgt(x, k-1); x <= k => cmpgt(k+1, x)
static inline __m128i ge_epi8(__m128i x, __m128i k) {
    return _mm_cmpgt_epi8(x, _mm_sub_epi8(k, _mm_set1_epi8(1)));
}
static inline __m128i le_epi8(__m128i x, __m128i k) {
    return _mm_cmpgt_epi8(_mm_add_epi8(k, _mm_set1_epi8(1)), x);
}

// Map 16 ASCII Base64 chars -> 16 sextets (0..63); return false if any invalid
static inline bool map16_ascii_to_sextets_sse2(const char* src, __m128i& out_idx) {
    __m128i c = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src));

    // Class masks
    __m128i is_AZ = _mm_and_si128(ge_epi8(c, kA_sse), le_epi8(c, kZ_sse)); // 'A'..'Z'
    __m128i is_az = _mm_and_si128(ge_epi8(c, ka_sse), le_epi8(c, kz_sse)); // 'a'..'z'
    __m128i is_09 = _mm_and_si128(ge_epi8(c, k0_sse), le_epi8(c, k9_sse)); // '0'..'9'
    __m128i is_p  = _mm_cmpeq_epi8(c, kPlus_sse);                      // '+'
    __m128i is_s  = _mm_cmpeq_epi8(c, kSlash_sse);                     // '/'

    // Validate: every byte must be in some class
    __m128i cls = _mm_or_si128(_mm_or_si128(is_AZ, is_az), _mm_or_si128(_mm_or_si128(is_09, is_p), is_s));
    unsigned mask = static_cast<unsigned>(_mm_movemask_epi8(cls));
    if (mask != 0xFFFFu) return false;  // at least one invalid byte

    // Base index 'A'..'Z'
    __m128i idx = _mm_sub_epi8(c, kA_sse);

    // Masked adds to map other classes
    idx = _mm_add_epi8(idx, _mm_and_si128(is_az, off_az_sse)); // 'a'..'z' -> +26
    idx = _mm_add_epi8(idx, _mm_and_si128(is_09, off_09_sse)); // '0'..'9' -> +52
    idx = _mm_add_epi8(idx, _mm_and_si128(is_p,  off_p_sse));  // '+' -> 62
    idx = _mm_add_epi8(idx, _mm_and_si128(is_s,  off_s_sse));  // '/' -> 63

    out_idx = idx;
    return true;
}

// Scalar table for tail/cleanup (strict: '=' is invalid -> 255)
static constexpr uint8_t Dtbl[256] = {
    // 0..31
    255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
    255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
    // 32..47   '+'/0x2B -> 62, '/'/0x2F -> 63
    255,255,255,255,255,255,255,255,255,255,255, 62,255,255,255, 63,
    // 48..63   '0'..'9' -> 52..61; '=' -> 255
     52, 53, 54, 55, 56, 57, 58, 59, 60, 61,255,255,255,255,255,255,
    // 64..79   'A'..'O' -> 0..14
    255,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14,
    // 80..95   'P'..'Z' -> 15..25
     15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,255,255,255,255,255,
    // 96..111  'a'..'o' -> 26..40
    255, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
    // 112..127 'p'..'z' -> 41..51
     41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51,255,255,255,255,255,
    // 128..255 invalid
    255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
    255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
    255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
    255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
    255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
    255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
    255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
    255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255
};

// Unpadded SSE2 Base64 decoder (range-compare mapper)
std::vector<uint8_t> fast_base64_decode_sse2_rangecmp(std::string_view input) {
    const char* src = input.data();
    size_t      n   = input.size();

    // Unpadded rule: reject length % 4 == 1
    if ((n & 3U) == 1U) return {};  // invalid unpadded Base64

    // Upper-bound allocate; +2 covers unpadded tails (mod4==2/3)
    std::vector<uint8_t> out((n/4)*3 + 2);
    uint8_t* dst = out.data();

    // SIMD loop: 16 chars -> 12 bytes per iter
    while (n >= 16) {
        __m128i idx16;
        if (!map16_ascii_to_sextets_sse2(src, idx16)) break;

        // Store sextets and pack quartets with scalar ops for SSE2 portability
        alignas(16) uint8_t sext[16];
        _mm_storeu_si128(reinterpret_cast<__m128i*>(sext), idx16);

        // Pack 4 sextets -> 3 bytes, four times
        uint32_t v;

        v = (sext[0]<<18) | (sext[1]<<12) | (sext[2]<<6) | sext[3];
        dst[0] = (uint8_t)(v >> 16);
        dst[1] = (uint8_t)(v >>  8);
        dst[2] = (uint8_t)(v      );

        v = (sext[4]<<18) | (sext[5]<<12) | (sext[6]<<6) | sext[7];
        dst[3] = (uint8_t)(v >> 16);
        dst[4] = (uint8_t)(v >>  8);
        dst[5] = (uint8_t)(v      );

        v = (sext[8]<<18) | (sext[9]<<12) | (sext[10]<<6) | sext[11];
        dst[6] = (uint8_t)(v >> 16);
        dst[7] = (uint8_t)(v >>  8);
        dst[8] = (uint8_t)(v      );

        v = (sext[12]<<18) | (sext[13]<<12) | (sext[14]<<6) | sext[15];
        dst[9]  = (uint8_t)(v >> 16);
        dst[10] = (uint8_t)(v >>  8);
        dst[11] = (uint8_t)(v      );

        src += 16;
        dst += 12;
        n   -= 16;
    }

    // Scalar full quartets
    while (n >= 4) {
        uint8_t a = Dtbl[(unsigned char)src[0]];
        uint8_t b = Dtbl[(unsigned char)src[1]];
        uint8_t c = Dtbl[(unsigned char)src[2]];
        uint8_t d = Dtbl[(unsigned char)src[3]];
        if ((a|b|c|d) == 255) break;

        uint32_t v = (a<<18) | (b<<12) | (c<<6) | d;
        dst[0] = (uint8_t)(v >> 16);
        dst[1] = (uint8_t)(v >>  8);
        dst[2] = (uint8_t)(v      );
        dst += 3; src += 4; n -= 4;
    }

    // Unpadded tail: n in {0,2,3}; n==1 invalid
    if (n == 2) {
        uint8_t a = Dtbl[(unsigned char)src[0]];
        uint8_t b = Dtbl[(unsigned char)src[1]];
        if ((a|b) == 255) return {};
        uint32_t v = (a<<18) | (b<<12);
        dst[0] = (uint8_t)(v >> 16);
        dst += 1;
    } else if (n == 3) {
        uint8_t a = Dtbl[(unsigned char)src[0]];
        uint8_t b = Dtbl[(unsigned char)src[1]];
        uint8_t c = Dtbl[(unsigned char)src[2]];
        if ((a|b|c) == 255) return {};
        uint32_t v = (a<<18) | (b<<12) | (c<<6);
        dst[0] = (uint8_t)(v >> 16);
        dst[1] = (uint8_t)(v >>  8);
        dst += 2;
    } else if (n == 1) {
        return {};
    }

    out.resize(static_cast<size_t>(dst - out.data()));
    return out;
}

//this is experimental and untested
static const uint8x16_t kA_neon      = vdupq_n_u8('A');
static const uint8x16_t kZ_neon      = vdupq_n_u8('Z');
static const uint8x16_t ka_neon      = vdupq_n_u8('a');
static const uint8x16_t kz_neon      = vdupq_n_u8('z');
static const uint8x16_t k0_neon      = vdupq_n_u8('0');
static const uint8x16_t k9_neon      = vdupq_n_u8('9');
static const uint8x16_t kPlus_neon   = vdupq_n_u8('+');
static const uint8x16_t kSlash_neon  = vdupq_n_u8('/');

// Offsets for masked-add mapper relative to base idx = (c - 'A')
// -6 mod 256 = 0xFA maps 'a'..'z' to +26; others are direct adds
static const uint8x16_t off_az_neon  = vdupq_n_u8((uint8_t)0xFA); // -6
static const uint8x16_t off_09_neon  = vdupq_n_u8((uint8_t)69);   // +69
static const uint8x16_t off_p_neon   = vdupq_n_u8((uint8_t)84);   // +84
static const uint8x16_t off_s_neon   = vdupq_n_u8((uint8_t)81);   // +81

static inline bool map16_ascii_to_sextets_neon(const char* src, uint8x16_t& out_idx) {
    uint8x16_t c = vld1q_u8(reinterpret_cast<const uint8_t*>(src));

    // Class masks with unsigned compares
    uint8x16_t is_AZ = vandq_u8(vcgeq_u8(c, kA_neon), vcleq_u8(c, kZ_neon)); // 'A'..'Z'
    uint8x16_t is_az = vandq_u8(vcgeq_u8(c, ka_neon), vcleq_u8(c, kz_neon)); // 'a'..'z'
    uint8x16_t is_09 = vandq_u8(vcgeq_u8(c, k0_neon), vcleq_u8(c, k9_neon)); // '0'..'9'
    uint8x16_t is_p  = vceqq_u8(c, kPlus_neon);                         // '+'
    uint8x16_t is_s  = vceqq_u8(c, kSlash_neon);                        // '/'

    // Validate: every lane is in some class (strict: '=' not allowed)
    uint8x16_t cls = vorrq_u8(vorrq_u8(is_AZ, is_az), vorrq_u8(vorrq_u8(is_09, is_p), is_s));
    // All-true lanes are 0xFF; require min element == 0xFF
    if (vminvq_u8(cls) != 0xFFu) return false;

    // Base index: 'A'..'Z' -> 0..25
    uint8x16_t idx = vsubq_u8(c, kA_neon);

    // Masked adds to map other classes into 0..63
    idx = vaddq_u8(idx, vandq_u8(is_az, off_az_neon));  // 'a'..'z' -> +26
    idx = vaddq_u8(idx, vandq_u8(is_09, off_09_neon));  // '0'..'9' -> +52
    idx = vaddq_u8(idx, vandq_u8(is_p,  off_p_neon));   // '+'      -> 62
    idx = vaddq_u8(idx, vandq_u8(is_s,  off_s_neon));   // '/'      -> 63

    out_idx = idx;
    return true;
}

// Input:  x = 16 sextets (0..63) in uint8x16_t
// Output: y[0..11] = 12 decoded bytes; y[12..15] unspecified (ignored)
static inline uint8x16_t neon_pack_4x6_to_3x8_tbl(uint8x16_t x) {
    // Neighbor vectors: a,b,c,d are the 0..15, 1..16, 2..17, 3..18 windows
    uint8x16_t a = x;
    uint8x16_t b = vextq_u8(x, x, 1);
    uint8x16_t c = vextq_u8(x, x, 2);
    uint8x16_t d = vextq_u8(x, x, 3);

    // Compute byte streams:
    // b0 = (a<<2) | (b>>4)
    // b1 = ((b & 0x0F)<<4) | (c>>2)
    // b2 = ((c & 0x03)<<6) | d
    const uint8x16_t m4 = vdupq_n_u8(0x0F);
    const uint8x16_t m2 = vdupq_n_u8(0x03);

    uint8x16_t b0 = vorrq_u8(vshlq_n_u8(a, 2), vshrq_n_u8(b, 4));
    uint8x16_t b1 = vorrq_u8(vshlq_n_u8(vandq_u8(b, m4), 4), vshrq_n_u8(c, 2));
    uint8x16_t b2 = vorrq_u8(vshlq_n_u8(vandq_u8(c, m2), 6), d);

    // Gather pairs P = [b0[0],b1[0], b0[4],b1[4], b0[8],b1[8], b0[12],b1[12]]
    uint8x16x2_t tbl01 = { b0, b1 };
    const uint8x16_t idxP = { 0,16, 4,20, 8,24, 12,28, 0xFF,0xFF,0xFF,0xFF, 0xFF,0xFF,0xFF,0xFF };
    uint8x16_t P = vqtbl2q_u8(tbl01, idxP);

    // Gather Q = [b2[0], b2[4], b2[8], b2[12]]
    const uint8x16_t idxQ = { 0,4,8,12, 0xFF,0xFF,0xFF,0xFF, 0xFF,0xFF,0xFF,0xFF, 0xFF,0xFF,0xFF,0xFF };
    uint8x16_t Q = vqtbl1q_u8(b2, idxQ);

    // Interleave P and Q into 12 bytes: [P0,P1,Q0, P2,P3,Q1, P4,P5,Q2, P6,P7,Q3]
    uint8x16x2_t tblPQ = { P, Q };
    const uint8x16_t idxOut = { 0,1,16, 2,3,17, 4,5,18, 6,7,19, 0xFF,0xFF,0xFF,0xFF };
    return vqtbl2q_u8(tblPQ, idxOut);
}

// Store 12 bytes from y (8 + 4)
static inline void store12_neon(uint8_t* dst, uint8x16_t y12) {
    // 8 bytes
    vst1_u64(reinterpret_cast<uint64_t*>(dst), vget_low_u64(vreinterpretq_u64_u8(y12)));
    // next 4 bytes at offset 8 (lane index 2 of uint32x4_t)
    uint32x4_t y32 = vreinterpretq_u32_u8(y12);
    vst1q_lane_u32(reinterpret_cast<uint32_t*>(dst + 8), y32, 2);
}

static constexpr uint8_t Dtbl_neon[256] = {
    // 0..31
    255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
    255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
    // 32..47 ('+'=62, '/'=63)
    255,255,255,255,255,255,255,255,255,255,255, 62,255,255,255, 63,
    // 48..63 ('0'..'9'=52..61; '=' invalid)
     52, 53, 54, 55, 56, 57, 58, 59, 60, 61,255,255,255,255,255,255,
    // 64..79 ('A'..'O'=0..14)
    255,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14,
    // 80..95 ('P'..'Z'=15..25)
     15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,255,255,255,255,255,
    // 96..111 ('a'..'o'=26..40)
    255, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
    // 112..127 ('p'..'z'=41..51)
     41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51,255,255,255,255,255,
    // 128..255 invalid
    255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
    255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
    255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
    255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
    255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
    255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
    255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
    255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255
};

std::vector<uint8_t> fast_base64_decode_neon_rangecmp(std::string_view input) {
    const char* src = input.data();
    size_t      n   = input.size();

    // Unpadded rule: length % 4 must be 0,2,3 (reject 1)
    if ((n & 3U) == 1U) return {};

    std::vector<uint8_t> out((n/4)*3 + 2);
    uint8_t* dst = out.data();

//#if defined(__aarch64__) || defined(__ARM_NEON)
    while (n >= 16) {
        uint8x16_t idx16;
        if (!map16_ascii_to_sextets_neon(src, idx16)) break;

        // Fully vectorized pack and 12-byte store
        uint8x16_t packed = neon_pack_4x6_to_3x8_tbl(idx16);
        store12_neon(dst, packed);

        src += 16;
        dst += 12;
        n   -= 16;
    }
//#endif

    // Scalar full quartets
    while (n >= 4) {
        uint8_t a = Dtbl_neon[(unsigned char)src[0]];
        uint8_t b = Dtbl_neon[(unsigned char)src[1]];
        uint8_t c = Dtbl_neon[(unsigned char)src[2]];
        uint8_t d = Dtbl_neon[(unsigned char)src[3]];
        if ((a|b|c|d) == 255) break;

        uint32_t v = (a<<18) | (b<<12) | (c<<6) | d;
        dst[0] = (uint8_t)(v >> 16);
        dst[1] = (uint8_t)(v >>  8);
        dst[2] = (uint8_t)(v      );
        dst += 3; src += 4; n -= 4;
    }

    // Unpadded tails
    if (n == 2) {
        uint8_t a = Dtbl_neon[(unsigned char)src[0]];
        uint8_t b = Dtbl_neon[(unsigned char)src[1]];
        if ((a|b) == 255) return {};
        uint32_t v = (a<<18) | (b<<12);
        dst[0] = (uint8_t)(v >> 16);
        dst += 1;
    } else if (n == 3) {
        uint8_t a = Dtbl_neon[(unsigned char)src[0]];
        uint8_t b = Dtbl_neon[(unsigned char)src[1]];
        uint8_t c = Dtbl_neon[(unsigned char)src[2]];
        if ((a|b|c) == 255) return {};
        uint32_t v = (a<<18) | (b<<12) | (c<<6);
        dst[0] = (uint8_t)(v >> 16);
        dst[1] = (uint8_t)(v >>  8);
        dst += 2;
    } else if (n == 1) {
        return {};
    }

    out.resize(static_cast<size_t>(dst - out.data()));
    return out;
}



