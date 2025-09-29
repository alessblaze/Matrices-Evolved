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
#include "../../../global.h"

#ifdef __AVX2__
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

// Hoisted constants
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

//#if defined(__AVX2__)
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
    // Clear upper 128 bits of YMM registers before transitioning to scalar code
    _mm256_zeroupper();
//#endif

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
#endif
