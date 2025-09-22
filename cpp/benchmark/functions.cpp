#include <immintrin.h>
#include <emmintrin.h>
#include <smmintrin.h>
#include <tmmintrin.h>
#include <vector>
#include <string>
#include <cstring>
#include <stdexcept>
#include <cstdint>
#include <cstddef>

// Thread-local buffers for base64 encoding (separate per function)
thread_local std::string sse_buffer;
thread_local std::string avx_buffer;
thread_local std::string lemire_buffer;
thread_local std::string mula_buffer;
thread_local std::string aligned_buffer;

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
    
    return lut_lookup_avx2_selects(indices_vec);
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
// Aligned version with Boost aligned allocator
#include <boost/align/aligned_allocator.hpp>


// Thread-local aligned buffers using Boost aligned allocator
thread_local std::vector<uint8_t, boost::alignment::aligned_allocator<uint8_t, 16>> aligned_input_buffer;
thread_local std::vector<char, boost::alignment::aligned_allocator<char, 16>> aligned_output_buffer;

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
    
    mula_buffer.resize(dest - dest_orig);
    return mula_buffer;
}

#include <openssl/bio.h>
#include <openssl/evp.h>
#include <openssl/buffer.h>

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