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
#include "../include/ams-neon.h"

// Hoisted constants for mapping and validation
static const __m128i kA     = _mm_set1_epi8('A');
static const __m128i kZ     = _mm_set1_epi8('Z');
static const __m128i ka     = _mm_set1_epi8('a');
static const __m128i kz     = _mm_set1_epi8('z');
static const __m128i k0     = _mm_set1_epi8('0');
static const __m128i k9     = _mm_set1_epi8('9');
static const __m128i kPlus  = _mm_set1_epi8('+');
static const __m128i kSlash = _mm_set1_epi8('/');

// Offsets for masked-add mapper (relative to base idx = c - 'A')
static const __m128i off_az = _mm_set1_epi8((char)-6);   // +26 for 'a'..'z'
static const __m128i off_09 = _mm_set1_epi8((char) 69);  // +52 for '0'..'9'
static const __m128i off_p  = _mm_set1_epi8((char) 84);  //  62 for '+'
static const __m128i off_s  = _mm_set1_epi8((char) 81);  //  63 for '/'

// Signed-compare helpers: x >= k => cmpgt(x,k-1); x <= k => cmpgt(k+1,x)
static inline __m128i ge_epi8(__m128i x, __m128i k) { return _mm_cmpgt_epi8(x, _mm_sub_epi8(k, _mm_set1_epi8(1))); }  // [web:116]
static inline __m128i le_epi8(__m128i x, __m128i k) { return _mm_cmpgt_epi8(_mm_add_epi8(k, _mm_set1_epi8(1)), x); }  // [web:116]

// Map 16 ASCII Base64 chars -> 16 sextets (0..63); return false if any invalid
static inline bool map16_ascii_to_sextets_sse2(const char* src, __m128i& out_idx) {
    __m128i c = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src));  // [web:116]

    // Class masks
    __m128i is_AZ = _mm_and_si128(ge_epi8(c, kA), le_epi8(c, kZ));       // [web:116]
    __m128i is_az = _mm_and_si128(ge_epi8(c, ka), le_epi8(c, kz));       // [web:116]
    __m128i is_09 = _mm_and_si128(ge_epi8(c, k0), le_epi8(c, k9));       // [web:116]
    __m128i is_p  = _mm_cmpeq_epi8(c, kPlus);                            // [web:116]
    __m128i is_s  = _mm_cmpeq_epi8(c, kSlash);                           // [web:116]

    // Validate: every byte must be in some class
    __m128i cls = _mm_or_si128(_mm_or_si128(is_AZ, is_az), _mm_or_si128(_mm_or_si128(is_09, is_p), is_s));  // [web:116]
    unsigned mask = static_cast<unsigned>(_mm_movemask_epi8(cls));       // [web:116]
    if (mask != 0xFFFFu) return false;                                   // [web:116]

    // Base index 'A'..'Z' and masked adds for other classes
    __m128i idx = _mm_sub_epi8(c, kA);                                   // [web:116]
    idx = _mm_add_epi8(idx, _mm_and_si128(is_az, off_az));               // [web:116]
    idx = _mm_add_epi8(idx, _mm_and_si128(is_09, off_09));               // [web:116]
    idx = _mm_add_epi8(idx, _mm_and_si128(is_p,  off_p));                // [web:116]
    idx = _mm_add_epi8(idx, _mm_and_si128(is_s,  off_s));                // [web:116]

    out_idx = idx;                                                        // [web:116]
    return true;                                                          // [web:116]
}

// SSSE3 pack: 4x6-bit -> 3x8-bit for 16 sextets (output low 12 bytes)
static inline __m128i pack_4x6_to_3x8_sse(__m128i sextets) {
    // pmaddubsw with pattern {1,64,1,64,...} to form (a | b<<6) pairs
    const __m128i mul1 = _mm_set1_epi32(0x01400140);                     // [web:234]
    __m128i ab_bc = _mm_maddubs_epi16(sextets, mul1);                    // [web:234]

    // pmaddwd with pattern {1,4096} to combine pairs into 24-bit dwords
    const __m128i mul2 = _mm_set1_epi32(0x00011000);                     // [web:234]
    __m128i packed = _mm_madd_epi16(ab_bc, mul2);                        // [web:234]

    // pshufb to extract bytes 2,1,0 of each dword; 0x80 lanes zeroed
    const __m128i shuf = _mm_setr_epi8( 2,1,0, 6,5,4, 10,9,8, 14,13,12, -128,-128,-128,-128 );  // [web:116]
    return _mm_shuffle_epi8(packed, shuf);                               // [web:116]
}

// Store 12 bytes from x to dst (8 + 4)
static inline void store12(uint8_t* dst, __m128i x) {
    _mm_storel_epi64(reinterpret_cast<__m128i*>(dst), x);                // 8B [web:116]
    __m128i hi = _mm_srli_si128(x, 8);                                   // [web:116]
    *(uint32_t*)(dst + 8) = (uint32_t)_mm_cvtsi128_si32(hi);             // 4B [web:116]
}

// Strict scalar table for cleanup and tails (255 = invalid, '=' invalid)
static constexpr uint8_t Dtbl[256] = {
    255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
    255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
    255,255,255,255,255,255,255,255,255,255,255, 62,255,255,255, 63,
     52, 53, 54, 55, 56, 57, 58, 59, 60, 61,255,255,255,255,255,255,
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
};  // [web:234]

// Fully vectorized SSSE3/SSE4.x unpadded Base64 decoder
std::vector<uint8_t> fast_base64_decode_sse_rangecmp(std::string_view input) {
    const char* src = input.data();                                        // [web:116]
    size_t      n   = input.size();                                        // [web:116]

    // Unpadded rule: reject length % 4 == 1
    if ((n & 3U) == 1U) return {};                                         // [web:21]

    std::vector<uint8_t> out((n/4)*3 + 2);
    uint8_t* dst = out.data();                                             // [web:21]

    while (n >= 16) {
        __m128i sextets;
        if (!map16_ascii_to_sextets_sse2(src, sextets)) break;             // [web:116]

        __m128i packed = pack_4x6_to_3x8_sse(sextets);                     // [web:234]
        store12(dst, packed);                                              // [web:116]

        src += 16; dst += 12; n -= 16;                                     // [web:116]
    }

    // Scalar full quartets
    while (n >= 4) {
        uint8_t a = Dtbl[(unsigned char)src[0]];
        uint8_t b = Dtbl[(unsigned char)src[1]];
        uint8_t c = Dtbl[(unsigned char)src[2]];
        uint8_t d = Dtbl[(unsigned char)src[3]];
        if ((a|b|c|d) == 255) break;                                       // [web:234]

        uint32_t v = (a<<18) | (b<<12) | (c<<6) | d;
        dst[0] = (uint8_t)(v >> 16);
        dst[1] = (uint8_t)(v >>  8);
        dst[2] = (uint8_t)(v      );
        dst += 3; src += 4; n -= 4;                                        // [web:234]
    }

    // Unpadded tails: n in {0,2,3}; n==1 invalid
    if (n == 2) {
        uint8_t a = Dtbl[(unsigned char)src[0]];
        uint8_t b = Dtbl[(unsigned char)src[1]];
        if ((a|b) == 255) return {};
        uint32_t v = (a<<18) | (b<<12);
        dst[0] = (uint8_t)(v >> 16);
        dst += 1;                                                          // [web:21]
    } else if (n == 3) {
        uint8_t a = Dtbl[(unsigned char)src[0]];
        uint8_t b = Dtbl[(unsigned char)src[1]];
        uint8_t c = Dtbl[(unsigned char)src[2]];
        if ((a|b|c) == 255) return {};
        uint32_t v = (a<<18) | (b<<12) | (c<<6);
        dst[0] = (uint8_t)(v >> 16);
        dst[1] = (uint8_t)(v >>  8);
        dst += 2;                                                          // [web:21]
    } else if (n == 1) {
        return {};                                                         // [web:21]
    }

    out.resize(static_cast<size_t>(dst - out.data()));                     // [web:21]
    return out;                                                            // [web:21]
}
