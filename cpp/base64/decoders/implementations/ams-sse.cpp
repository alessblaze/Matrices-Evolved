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
#include "../include/ams-sse.h"

// Hoisted constants for mapping and validation
static const __m128i kA     = _mm_set1_epi8('A');
static const __m128i kZ     = _mm_set1_epi8('Z');
static const __m128i ka     = _mm_set1_epi8('a');
static const __m128i kz     = _mm_set1_epi8('z');
static const __m128i k0     = _mm_set1_epi8('0');
static const __m128i k9     = _mm_set1_epi8('9');
static const __m128i kPlus  = _mm_set1_epi8('+');
static const __m128i kSlash = _mm_set1_epi8('/');
static const __m128i kAll1  = _mm_set1_epi8((char)0xFF);

// Offsets for masked-add mapper (relative to base idx = c - 'A')
static const __m128i off_az = _mm_set1_epi8((char)-6);  // 'a'..'z' -> +26
static const __m128i off_09 = _mm_set1_epi8((char) 69); // '0'..'9' -> +52
static const __m128i off_p  = _mm_set1_epi8((char) 84); // '+' -> 62
static const __m128i off_s  = _mm_set1_epi8((char) 81); // '/' -> 63

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
    __m128i is_AZ = _mm_and_si128(ge_epi8(c, kA), le_epi8(c, kZ)); // 'A'..'Z'
    __m128i is_az = _mm_and_si128(ge_epi8(c, ka), le_epi8(c, kz)); // 'a'..'z'
    __m128i is_09 = _mm_and_si128(ge_epi8(c, k0), le_epi8(c, k9)); // '0'..'9'
    __m128i is_p  = _mm_cmpeq_epi8(c, kPlus);                      // '+'
    __m128i is_s  = _mm_cmpeq_epi8(c, kSlash);                     // '/'

    // Validate: every byte must be in some class
    __m128i cls = _mm_or_si128(_mm_or_si128(is_AZ, is_az), _mm_or_si128(_mm_or_si128(is_09, is_p), is_s));
    unsigned mask = static_cast<unsigned>(_mm_movemask_epi8(cls));
    if (mask != 0xFFFFu) return false;  // at least one invalid byte

    // Base index 'A'..'Z'
    __m128i idx = _mm_sub_epi8(c, kA);

    // Masked adds to map other classes
    idx = _mm_add_epi8(idx, _mm_and_si128(is_az, off_az)); // 'a'..'z' -> +26
    idx = _mm_add_epi8(idx, _mm_and_si128(is_09, off_09)); // '0'..'9' -> +52
    idx = _mm_add_epi8(idx, _mm_and_si128(is_p,  off_p));  // '+' -> 62
    idx = _mm_add_epi8(idx, _mm_and_si128(is_s,  off_s));  // '/' -> 63

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
