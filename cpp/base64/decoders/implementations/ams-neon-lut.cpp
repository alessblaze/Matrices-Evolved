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
// 256-entry ASCII -> sextet table (255 = invalid)
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
}; // [web:234]

// Table bundle for one 64-byte slice (for vqtbl4q_u8)
struct Tbl64x4 { uint8x16_t v0, v1, v2, v3; };

// Load four 16B vectors from a 64-byte table slice
static inline Tbl64x4 load_tbl64x4(const uint8_t* base) {
  return Tbl64x4{ vld1q_u8(base +  0), vld1q_u8(base + 16),
                  vld1q_u8(base + 32), vld1q_u8(base + 48) }; // [web:398]
}

// 64-byte table lookup with vqtbl4q_u8
static inline uint8x16_t tbl64_lookup(const Tbl64x4& t, uint8x16_t idx) {
  uint8x16x4_t pack{ t.v0, t.v1, t.v2, t.v3 };
  return vqtbl4q_u8(pack, idx); // indices 0..63 per lane [web:516]
}

// 2×128 hierarchical LUT mapper: 16 ASCII -> 16 sextets (0..63)
static inline bool map16_ascii_to_sextets_neon_lut_2x128(const char* src, uint8x16_t& out_idx) {
  static const Tbl64x4 T0 = load_tbl64x4(Dtbl +   0);  // 0..63   [web:398]
  static const Tbl64x4 T1 = load_tbl64x4(Dtbl +  64);  // 64..127 [web:398]
  static const Tbl64x4 T2 = load_tbl64x4(Dtbl + 128);  // 128..191 [web:398]
  static const Tbl64x4 T3 = load_tbl64x4(Dtbl + 192);  // 192..255 [web:398]

  uint8x16_t c     = vld1q_u8(reinterpret_cast<const uint8_t*>(src));      // [web:398]
  uint8x16_t idx64 = vandq_u8(c, vdupq_n_u8(63));                           // low 6 bits [web:398]
  uint8x16_t blk   = vshrq_n_u8(c, 6);                                      // block id [web:398]

  // Lookups per 64-byte block
  uint8x16_t r0 = tbl64_lookup(T0, idx64);
  uint8x16_t r1 = tbl64_lookup(T1, idx64);
  uint8x16_t r2 = tbl64_lookup(T2, idx64);
  uint8x16_t r3 = tbl64_lookup(T3, idx64);                                   // [web:516]

  // Masks for low/high bit of blk
  uint8x16_t mBit0 = vceqq_u8(vandq_u8(blk, vdupq_n_u8(1)), vdupq_n_u8(1)); // [web:398]
  uint8x16_t mBit1 = vceqq_u8(vandq_u8(blk, vdupq_n_u8(2)), vdupq_n_u8(2)); // [web:398]

  // Stage 1: select inside low half (blocks 0 vs 1) and high half (2 vs 3)
  uint8x16_t r01 = vbslq_u8(mBit0, r1, r0);
  uint8x16_t r23 = vbslq_u8(mBit0, r3, r2);                                   // [web:506]

  // Stage 2: select between halves using high bit
  uint8x16_t res = vbslq_u8(mBit1, r23, r01);                                  // [web:506]

  // Validate: 255 indicates invalid
  uint8x16_t inv = vceqq_u8(res, vdupq_n_u8(255));                             // [web:398]
  if (vmaxvq_u8(inv)) return false;                                            // [web:398]

  out_idx = res;
  return true;
}

// Fully vectorized NEON pack: 16 sextets -> 12 bytes (in low lanes)
static inline uint8x16_t neon_pack_4x6_to_3x8_tbl(uint8x16_t x) {
  uint8x16_t a = x;
  uint8x16_t b = vextq_u8(x, x, 1);
  uint8x16_t c = vextq_u8(x, x, 2);
  uint8x16_t d = vextq_u8(x, x, 3);                                            // [web:360]

  const uint8x16_t m4 = vdupq_n_u8(0x0F);
  const uint8x16_t m2 = vdupq_n_u8(0x03);

  uint8x16_t b0 = vorrq_u8(vshlq_n_u8(a, 2), vshrq_n_u8(b, 4));
  uint8x16_t b1 = vorrq_u8(vshlq_n_u8(vandq_u8(b, m4), 4), vshrq_n_u8(c, 2));
  uint8x16_t b2 = vorrq_u8(vshlq_n_u8(vandq_u8(c, m2), 6), d);                 // [web:360]

  // Gather P = [b0[0],b1[0], b0[4],b1[4], b0[8],b1[8], b0[12],b1[12]]
  uint8x16x2_t tbl01 = { b0, b1 };
  const uint8x16_t idxP = { 0,16, 4,20, 8,24, 12,28, 0xFF,0xFF,0xFF,0xFF, 0xFF,0xFF,0xFF,0xFF };
  uint8x16_t P = vqtbl2q_u8(tbl01, idxP);                                      // [web:516]

  // Gather Q = [b2[0], b2[4], b2[8], b2[12]]
  const uint8x16_t idxQ = { 0,4,8,12, 0xFF,0xFF,0xFF,0xFF, 0xFF,0xFF,0xFF,0xFF, 0xFF,0xFF,0xFF,0xFF };
  uint8x16_t Q = vqtbl1q_u8(b2, idxQ);                                         // [web:516]

  // Interleave into 12 bytes: [P0,P1,Q0, P2,P3,Q1, P4,P5,Q2, P6,P7,Q3]
  uint8x16x2_t tblPQ = { P, Q };
  const uint8x16_t idxOut = { 0,1,16, 2,3,17, 4,5,18, 6,7,19, 0xFF,0xFF,0xFF,0xFF };
  return vqtbl2q_u8(tblPQ, idxOut);                                            // [web:516]
}

// Store 12 bytes from y (8 + 4)
static inline void store12_neon(uint8_t* dst, uint8x16_t y12) {
  vst1_u64(reinterpret_cast<uint64_t*>(dst), vget_low_u64(vreinterpretq_u64_u8(y12)));
  uint32x4_t y32 = vreinterpretq_u32_u8(y12);
  vst1q_lane_u32(reinterpret_cast<uint32_t*>(dst + 8), y32, 2);               // [web:398]
}

// Unpadded NEON decoder using 2×128 LUT mapper
std::vector<uint8_t> fast_base64_decode_neon_lut2x128(std::string_view input) {
  const char* src = input.data();
  size_t      n   = input.size();

  // Unpadded rule: reject length % 4 == 1 (no '=' allowed anywhere)
  if ((n & 3U) == 1U) return {};                                              // [web:21]

  std::vector<uint8_t> out((n/4)*3 + 2);
  uint8_t* dst = out.data();

//#if defined(__aarch64__) || defined(__ARM_NEON)
  while (n >= 16) {
    uint8x16_t idx16;
    if (!map16_ascii_to_sextets_neon_lut_2x128(src, idx16)) break;            // [web:398]

    uint8x16_t packed = neon_pack_4x6_to_3x8_tbl(idx16);
    store12_neon(dst, packed);

    src += 16; dst += 12; n -= 16;                                            // [web:360]
  }
//#endif

  // Scalar cleanup for full quartets
  while (n >= 4) {
    uint8_t a = Dtbl[(unsigned char)src[0]];
    uint8_t b = Dtbl[(unsigned char)src[1]];
    uint8_t c = Dtbl[(unsigned char)src[2]];
    uint8_t d = Dtbl[(unsigned char)src[3]];
    if ((a|b|c|d) == 255) break;                                              // [web:234]

    uint32_t v = (a<<18) | (b<<12) | (c<<6) | d;
    dst[0] = (uint8_t)(v >> 16);
    dst[1] = (uint8_t)(v >>  8);
    dst[2] = (uint8_t)(v      );
    dst += 3; src += 4; n -= 4;                                               // [web:234]
  }

  // Unpadded tails: n in {0,2,3}; n==1 invalid
  if (n == 2) {
    uint8_t a = Dtbl[(unsigned char)src[0]];
    uint8_t b = Dtbl[(unsigned char)src[1]];
    if ((a|b) == 255) return {};
    uint32_t v = (a<<18) | (b<<12);
    dst[0] = (uint8_t)(v >> 16);
    dst += 1;                                                                 // [web:21]
  } else if (n == 3) {
    uint8_t a = Dtbl[(unsigned char)src[0]];
    uint8_t b = Dtbl[(unsigned char)src[1]];
    uint8_t c = Dtbl[(unsigned char)src[2]];
    if ((a|b|c) == 255) return {};
    uint32_t v = (a<<18) | (b<<12) | (c<<6);
    dst[0] = (uint8_t)(v >> 16);
    dst[1] = (uint8_t)(v >>  8);
    dst += 2;                                                                 // [web:21]
  } else if (n == 1) {
    return {};                                                                // [web:21]
  }

  out.resize(static_cast<size_t>(dst - out.data()));
  return out;
}
#endif
