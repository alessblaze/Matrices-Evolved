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
#ifdef __ARM_NEON
//this is experimental 
static const uint8x16_t kA      = vdupq_n_u8('A');
static const uint8x16_t kZ      = vdupq_n_u8('Z');
static const uint8x16_t ka      = vdupq_n_u8('a');
static const uint8x16_t kz      = vdupq_n_u8('z');
static const uint8x16_t k0      = vdupq_n_u8('0');
static const uint8x16_t k9      = vdupq_n_u8('9');
static const uint8x16_t kPlus   = vdupq_n_u8('+');
static const uint8x16_t kSlash  = vdupq_n_u8('/');

// Offsets for masked-add mapper relative to base idx = (c - 'A')
// -6 mod 256 = 0xFA maps 'a'..'z' to +26; others are direct adds
static const uint8x16_t off_az  = vdupq_n_u8((uint8_t)0xFA); // -6
static const uint8x16_t off_09  = vdupq_n_u8((uint8_t)69);   // +69
static const uint8x16_t off_p   = vdupq_n_u8((uint8_t)84);   // +84
static const uint8x16_t off_s   = vdupq_n_u8((uint8_t)81);   // +81

static inline bool map16_ascii_to_sextets_neon(const char* src, uint8x16_t& out_idx) {
    uint8x16_t c = vld1q_u8(reinterpret_cast<const uint8_t*>(src));

    // Class masks with unsigned compares
    uint8x16_t is_AZ = vandq_u8(vcgeq_u8(c, kA), vcleq_u8(c, kZ)); // 'A'..'Z'
    uint8x16_t is_az = vandq_u8(vcgeq_u8(c, ka), vcleq_u8(c, kz)); // 'a'..'z'
    uint8x16_t is_09 = vandq_u8(vcgeq_u8(c, k0), vcleq_u8(c, k9)); // '0'..'9'
    uint8x16_t is_p  = vceqq_u8(c, kPlus);                         // '+'
    uint8x16_t is_s  = vceqq_u8(c, kSlash);                        // '/'

    // Validate: every lane is in some class (strict: '=' not allowed)
    uint8x16_t cls = vorrq_u8(vorrq_u8(is_AZ, is_az), vorrq_u8(vorrq_u8(is_09, is_p), is_s));
    // All-true lanes are 0xFF; require min element == 0xFF
    if (vminvq_u8(cls) != 0xFFu) return false;

    // Base index: 'A'..'Z' -> 0..25
    uint8x16_t idx = vsubq_u8(c, kA);

    // Masked adds to map other classes into 0..63
    idx = vaddq_u8(idx, vandq_u8(is_az, off_az));  // 'a'..'z' -> +26
    idx = vaddq_u8(idx, vandq_u8(is_09, off_09));  // '0'..'9' -> +52
    idx = vaddq_u8(idx, vandq_u8(is_p,  off_p));   // '+'      -> 62
    idx = vaddq_u8(idx, vandq_u8(is_s,  off_s));   // '/'      -> 63

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

static constexpr uint8_t Dtbl[256] = {
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

    // Unpadded tails
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
#endif