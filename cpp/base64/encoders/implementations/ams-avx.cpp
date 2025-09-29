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

#if defined(__AVX2__) && !defined(DISABLE_SSE_BASE64_ENCODER_AVX) || defined(__ARM_NEON) && !defined(DISABLE_SSE_BASE64_ENCODER_AVX)
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

 /*
// During our extended research we discovered a family of mulhi/mullo
// multiplier/mask parameterizations for SIMD sextet extraction.
//
// The following methods and constant families were derived by Aless
// Microsystems from first principles (bit-layout equations + multiply
// identities). They are provided under this project's license; see the
// LICENSE header for details.
//
// NOTE: The high-level multiply-and-mask trick (use of vpmulhi/vpmullo
// with masks and a pre-shuffle is a known SIMD technique used in prior
// work on Base64.  What this project documents and supplies are the
// explicit derivations, alternative byte-order variants, and tested
// constant families for different pre-shuffle layouts.  Those
// derivations are independently produced and documented here.
// These constants and the derivation process were produced independently by Aless Microsystems; 
// they are not copied from any externally published constant tables.
### Setup

- Define per 24‑bit group bytes $a,b,c$ and halfwords per lane as $h = (a \ll 8) + b$ and $\ell = (b \ll 8) + c$, with target sextets $i_0=a\gg 2$, $i_1=((a\&3)\ll 4)\,|\, (b\gg 4)$, $i_2=((b\&15)\ll 2)\,|\, (c\gg 6)$, $i_3=c\&63$ packed into bytes via a lane‑local vpshufb pre‑arrangement as in the Muła–Lemire method [^1][^4].
- The algorithm computes two masked paths, $t_1 = \mathrm{mulhi}(x_R, C_{hi})$ and $t_3 = \mathrm{mullo}(x_L, C_{lo})$, where $x_R = \mathrm{in} \,\&\, 0x0fc0fc00$ and $x_L = \mathrm{in} \,\&\, 0x003f03f0$, then ORs them to produce the four sextets per dword without variable shifts.[^2][^1]


### Mulhi implements right shifts

- For any 16‑bit lane value $x$ and power‑of‑two multiplier $2^k$, vpmulhi_epu16 computes the high half of the product: $\mathrm{mulhi}(x,2^k) = \left\lfloor\frac{x\cdot 2^k}{2^{16}}\right\rfloor = x \gg (16-k)$, so choosing $k=6$ yields a right shift by 10, and $k=10$ yields a right shift by 6, exactly the two right shifts needed from $h$ and $\ell$ respectively.[^3][^1]
- The canonical constant 0x04000040 encodes 16‑bit multipliers {0x0400, 0x0040} per halfword in each 32‑bit dword, giving $h \gg 6$ (for 0x0400 = $2^{10}$) and $\ell \gg 10$ (for 0x0040 = $2^6$) or vice‑versa depending on byte order; swapping these two halfword multipliers yields equivalent results under LE vs BE pre‑shuffles (e.g., 0x00400400/0x04000040 families).[^1][^2]
- Adding 1 to a mulhi power‑of‑two does not change the result: $\mathrm{mulhi}(x,2^k+1) = \left\lfloor\frac{x\cdot 2^k}{2^{16}}\right\rfloor + \left\lfloor\frac{x}{2^{16}}\right\rfloor = \mathrm{mulhi}(x,2^k)$ because $x<2^{16}$ implies $x\gg 16 = 0$, which explains variants like 0x04010040 or 0x04000041 producing identical high‑half bytes under the same masks.[^3][^1]


### Mullo implements left shifts modulo $2^{16}$

- For any 16‑bit lane value $x$ and power‑of‑two multiplier $2^m$, vpmullo_epi16 computes $\mathrm{mullo}(x,2^m) = (x \ll m) \bmod 2^{16}$, so choosing $m=8$ and $m=4$ yields the left‑shift parts needed to place the low‑order contributions of $h$ and $\ell$ for $i_1$ and $i_2$ respectively.[^1][^3]
- The canonical constant 0x01000010 encodes 16‑bit multipliers {0x0100, 0x0010} per halfword, i.e., left shifts by 8 and 4 within each dword, and swapping these per halfword (e.g., 0x00100100) matches the LE vs BE arrangement when the pre‑shuffle flips which halfword carries which group’s contribution.[^2][^1]
- Adding 1 to a mullo power‑of‑two is harmless when the masked input has its low $m$ bits zero: $\mathrm{mullo}(x,2^m+1) \equiv (x \ll m) + x \pmod{2^{16}} = (x \ll m)$ if $x \equiv 0 \pmod{2^m}$, which is ensured by using maskL so that the low 8 bits of the “$m=8$” lane and the low 4 bits of the “$m=4$” lane are zero before the multiply.[^3][^1]


### Why multiple constant families appear

- Different “bcab vs bacb” labels reflect whether the pre‑shuffle presents the two 16‑bit halfwords as $[h,\ell]$ or $[\ell,h]$ inside each 32‑bit dword, which simply swaps which halfword multiplier must be 0x0400 vs 0x0040 for mulhi and 0x0100 vs 0x0010 for mullo, yielding the two canonical families 0x04000040/0x01000010 (BE) and 0x00400400/0x00100100 (LE) under equivalent masks.[^4][^1]
- The “+1” variants in the mulhi constants are mathematically equivalent for the high‑half as shown above, and “+1” variants for mullo are equivalent when maskL zeroes the low $m$ bits of each lane’s operand, which is precisely how the Muła–Lemire masks are designed prior to the multiplies.[^2][^1]


### Copyable math identities

- $\mathrm{mulhi}(x,2^k) = x \gg (16-k)$ and $\mathrm{mulhi}(x,2^k+1) = \mathrm{mulhi}(x,2^k)$ for $x<2^{16}$.[^3]
- $\mathrm{mullo}(x,2^m) \equiv (x \ll m) \pmod{2^{16}}$ and $\mathrm{mullo}(x,2^m+1) \equiv (x \ll m)$ if $x \equiv 0 \pmod{2^m}$ due to prior masking that zeroes the low $m$ bits.[^1][^3]
- With $h=(a\ll 8)+b$ and $\ell=(b\ll 8)+c$, choose per‑halfword multipliers $\{2^{10},2^{6}\}$ for mulhi and $\{2^{8},2^{4}\}$ for mullo (order depends on LE/BE pre‑shuffle) to realize $i_0,i_1,i_2,i_3$ in one mulhi, one mullo, and one OR per lane under masks $0x0fc0fc00$ and $0x003f03f0$.[^2][^1]

## I believe it’s not as simple as it sounds and there are more variants to be discovered.
## Our methods can be slower, but in practical use this won’t matter.
## I would like to see more methods coming from the younger generation of programmers,
## which do not rely on these old methods.
## These tables are available under the project license. They are not guaranteed to work.
## If you use these elsewhere, you must abide by the license and cite this project and author.
## Always test if it works before using it in anything meaningful.

bacb (little‑endian dword view), idx_order=(0,1,2,3)

Per 128‑bit lane input (12 bytes = 4 groups):
a0 b0 c0 a1 b1 c1 a2 b2 c2 a3 b3 c3

Lane‑local preshuffle (halves swapped relative to bcab):
D0: [ b0 ][ c0 ] | [ a0 ][ b0 ]
D1: [ b1 ][ c1 ] | [ a1 ][ b1 ]
D2: [ b2 ][ c2 ] | [ a2 ][ b2 ]
D3: [ b3 ][ c3 ] | [ a3 ][ b3 ]

Masks (same bitfields, expressed in LE dword view notation):
maskR = 0x0fc0fc0f (or 0xfc0f0fc0 under the alternate naming)
maskL = 0xf03f03f0 (or 0x03f0f03f under the alternate naming)

Multipliers (per 16‑bit lane, swapped halves vs bcab):
mulhi = 0x00400400 // {0x0040, 0x0400} → right shifts {10, 6}
mullo = 0x00100100 // {0x0010, 0x0100} → left shifts {4, 8}

How the multipliers implement shifts (same identities as bcab):
mulhi(x, 2^k) = x >> (16 − k); mullo(x, 2^m) = (x << m) mod 2^16
(LE vs BE just swaps which halfword gets which 2^k / 2^m power.)

Post (t1 | t3) — per dword bytes:
[ i0 i1 i2 i3 ] from the same abc groups as above.

Equivalences:
mulhi +1 variants are identical for 16‑bit lanes.
mullo +1 variants are identical under maskL because low m bits are zero before multiply.
Legend: [x] is one byte; ‘|’ separates the two 16‑bit halves inside the 32‑bit dword.

SOLVED abbc  dword=BE  idx_order=(2,3,0,1)  maskR=0xfc0f0fc0 maskL=0x03f0f03f  mulhi=0x00400400 mullo=0x00100100
SOLVED abbc  dword=BE  idx_order=(2,3,0,1)  maskR=0xfc0f0fc0 maskL=0x03f0f03f  mulhi=0x00410400 mullo=0x00100100
SOLVED abbc  dword=BE  idx_order=(2,3,0,1)  maskR=0xfc0f0fc0 maskL=0x03f0f03f  mulhi=0x00400401 mullo=0x00100100
SOLVED abbc  dword=BE  idx_order=(2,3,0,1)  maskR=0xfc0f0fc0 maskL=0x03f0f03f  mulhi=0x00410401 mullo=0x00100100
SOLVED abbc  dword=BE  idx_order=(2,3,0,1)  maskR=0xfc0f0fc0 maskL=0x03f0f03f  mulhi=0x00400402 mullo=0x00100100
SOLVED abbc  dword=BE  idx_order=(2,3,0,1)  maskR=0xfc0f0fc0 maskL=0x03f0f03f  mulhi=0x00410402 mullo=0x00100100
SOLVED abbc  dword=BE  idx_order=(2,3,0,1)  maskR=0xfc0f0fc0 maskL=0x03f0f03f  mulhi=0x00400403 mullo=0x00100100
SOLVED abbc  dword=BE  idx_order=(2,3,0,1)  maskR=0xfc0f0fc0 maskL=0x03f0f03f  mulhi=0x00410403 mullo=0x00100100
SOLVED abbc  dword=BE  idx_order=(2,3,0,1)  maskR=0xfc0f0fc0 maskL=0x03f0f03f  mulhi=0x00400404 mullo=0x00100100
SOLVED abbc  dword=BE  idx_order=(2,3,0,1)  maskR=0xfc0f0fc0 maskL=0x03f0f03f  mulhi=0x00410404 mullo=0x00100100
SOLVED abbc  dword=BE  idx_order=(2,3,0,1)  maskR=0xfc0f0fc0 maskL=0x03f0f03f  mulhi=0x00400405 mullo=0x00100100
SOLVED abbc  dword=BE  idx_order=(2,3,0,1)  maskR=0xfc0f0fc0 maskL=0x03f0f03f  mulhi=0x00410405 mullo=0x00100100
SOLVED abbc  dword=BE  idx_order=(2,3,0,1)  maskR=0xfc0f0fc0 maskL=0x03f0f03f  mulhi=0x00400406 mullo=0x00100100
SOLVED abbc  dword=BE  idx_order=(2,3,0,1)  maskR=0xfc0f0fc0 maskL=0x03f0f03f  mulhi=0x00410406 mullo=0x00100100
SOLVED abbc  dword=BE  idx_order=(2,3,0,1)  maskR=0xfc0f0fc0 maskL=0x03f0f03f  mulhi=0x00400407 mullo=0x00100100
SOLVED abbc  dword=BE  idx_order=(2,3,0,1)  maskR=0xfc0f0fc0 maskL=0x03f0f03f  mulhi=0x00410407 mullo=0x00100100
SOLVED abbc  dword=BE  idx_order=(2,3,0,1)  maskR=0xfc0f0fc0 maskL=0x03f0f03f  mulhi=0x00400408 mullo=0x00100100
SOLVED abbc  dword=BE  idx_order=(2,3,0,1)  maskR=0xfc0f0fc0 maskL=0x03f0f03f  mulhi=0x00410408 mullo=0x00100100
SOLVED bacb  dword=LE  idx_order=(0,1,2,3)  maskR=0x0fc0fc0f maskL=0xf03f03f0  mulhi=0x04000040 mullo=0x01000010
SOLVED bacb  dword=LE  idx_order=(0,1,2,3)  maskR=0x0fc0fc0f maskL=0xf03f03f0  mulhi=0x04010040 mullo=0x01000010
SOLVED bacb  dword=LE  idx_order=(0,1,2,3)  maskR=0x0fc0fc0f maskL=0xf03f03f0  mulhi=0x04020040 mullo=0x01000010
SOLVED bacb  dword=LE  idx_order=(0,1,2,3)  maskR=0x0fc0fc0f maskL=0xf03f03f0  mulhi=0x04030040 mullo=0x01000010
SOLVED bacb  dword=LE  idx_order=(0,1,2,3)  maskR=0x0fc0fc0f maskL=0xf03f03f0  mulhi=0x04040040 mullo=0x01000010
SOLVED bacb  dword=LE  idx_order=(0,1,2,3)  maskR=0x0fc0fc0f maskL=0xf03f03f0  mulhi=0x04050040 mullo=0x01000010
SOLVED bacb  dword=LE  idx_order=(0,1,2,3)  maskR=0x0fc0fc0f maskL=0xf03f03f0  mulhi=0x04060040 mullo=0x01000010
SOLVED bacb  dword=LE  idx_order=(0,1,2,3)  maskR=0x0fc0fc0f maskL=0xf03f03f0  mulhi=0x04070040 mullo=0x01000010
SOLVED bacb  dword=LE  idx_order=(0,1,2,3)  maskR=0x0fc0fc0f maskL=0xf03f03f0  mulhi=0x04080040 mullo=0x01000010
SOLVED bacb  dword=LE  idx_order=(0,1,2,3)  maskR=0x0fc0fc0f maskL=0xf03f03f0  mulhi=0x04000041 mullo=0x01000010
SOLVED bacb  dword=LE  idx_order=(0,1,2,3)  maskR=0x0fc0fc0f maskL=0xf03f03f0  mulhi=0x04010041 mullo=0x01000010
SOLVED bacb  dword=LE  idx_order=(0,1,2,3)  maskR=0x0fc0fc0f maskL=0xf03f03f0  mulhi=0x04020041 mullo=0x01000010
SOLVED bacb  dword=LE  idx_order=(0,1,2,3)  maskR=0x0fc0fc0f maskL=0xf03f03f0  mulhi=0x04030041 mullo=0x01000010
SOLVED bacb  dword=LE  idx_order=(0,1,2,3)  maskR=0x0fc0fc0f maskL=0xf03f03f0  mulhi=0x04040041 mullo=0x01000010
SOLVED bacb  dword=LE  idx_order=(0,1,2,3)  maskR=0x0fc0fc0f maskL=0xf03f03f0  mulhi=0x04050041 mullo=0x01000010
SOLVED bacb  dword=LE  idx_order=(0,1,2,3)  maskR=0x0fc0fc0f maskL=0xf03f03f0  mulhi=0x04060041 mullo=0x01000010
SOLVED bacb  dword=LE  idx_order=(0,1,2,3)  maskR=0x0fc0fc0f maskL=0xf03f03f0  mulhi=0x04070041 mullo=0x01000010
SOLVED bacb  dword=LE  idx_order=(0,1,2,3)  maskR=0x0fc0fc0f maskL=0xf03f03f0  mulhi=0x04080041 mullo=0x01000010
SOLVED bcab  dword=BE  idx_order=(0,1,2,3)  maskR=0x0fc0fc0f maskL=0xf03f03f0  mulhi=0x04010040 mullo=0x01000010
SOLVED bcab  dword=BE  idx_order=(0,1,2,3)  maskR=0x0fc0fc0f maskL=0xf03f03f0  mulhi=0x04020040 mullo=0x01000010
SOLVED bcab  dword=BE  idx_order=(0,1,2,3)  maskR=0x0fc0fc0f maskL=0xf03f03f0  mulhi=0x04030040 mullo=0x01000010
SOLVED bcab  dword=BE  idx_order=(0,1,2,3)  maskR=0x0fc0fc0f maskL=0xf03f03f0  mulhi=0x04040040 mullo=0x01000010
SOLVED bcab  dword=BE  idx_order=(0,1,2,3)  maskR=0x0fc0fc0f maskL=0xf03f03f0  mulhi=0x04050040 mullo=0x01000010
SOLVED bcab  dword=BE  idx_order=(0,1,2,3)  maskR=0x0fc0fc0f maskL=0xf03f03f0  mulhi=0x04060040 mullo=0x01000010
SOLVED bcab  dword=BE  idx_order=(0,1,2,3)  maskR=0x0fc0fc0f maskL=0xf03f03f0  mulhi=0x04070040 mullo=0x01000010
SOLVED bcab  dword=BE  idx_order=(0,1,2,3)  maskR=0x0fc0fc0f maskL=0xf03f03f0  mulhi=0x04080040 mullo=0x01000010
SOLVED bcab  dword=BE  idx_order=(0,1,2,3)  maskR=0x0fc0fc0f maskL=0xf03f03f0  mulhi=0x04000041 mullo=0x01000010
SOLVED bcab  dword=BE  idx_order=(0,1,2,3)  maskR=0x0fc0fc0f maskL=0xf03f03f0  mulhi=0x04010041 mullo=0x01000010
SOLVED bcab  dword=BE  idx_order=(0,1,2,3)  maskR=0x0fc0fc0f maskL=0xf03f03f0  mulhi=0x04020041 mullo=0x01000010
SOLVED bcab  dword=BE  idx_order=(0,1,2,3)  maskR=0x0fc0fc0f maskL=0xf03f03f0  mulhi=0x04030041 mullo=0x01000010
SOLVED bcab  dword=BE  idx_order=(0,1,2,3)  maskR=0x0fc0fc0f maskL=0xf03f03f0  mulhi=0x04040041 mullo=0x01000010
SOLVED bcab  dword=BE  idx_order=(0,1,2,3)  maskR=0x0fc0fc0f maskL=0xf03f03f0  mulhi=0x04050041 mullo=0x01000010
SOLVED bcab  dword=BE  idx_order=(0,1,2,3)  maskR=0x0fc0fc0f maskL=0xf03f03f0  mulhi=0x04060041 mullo=0x01000010
SOLVED bcab  dword=BE  idx_order=(0,1,2,3)  maskR=0x0fc0fc0f maskL=0xf03f03f0  mulhi=0x04070041 mullo=0x01000010
SOLVED bcab  dword=BE  idx_order=(0,1,2,3)  maskR=0x0fc0fc0f maskL=0xf03f03f0  mulhi=0x04080041 mullo=0x01000010
SOLVED cbba  dword=LE  idx_order=(2,3,0,1)  maskR=0xfc0f0fc0 maskL=0x03f0f03f  mulhi=0x00400400 mullo=0x00100100
SOLVED cbba  dword=LE  idx_order=(2,3,0,1)  maskR=0xfc0f0fc0 maskL=0x03f0f03f  mulhi=0x00410400 mullo=0x00100100
SOLVED cbba  dword=LE  idx_order=(2,3,0,1)  maskR=0xfc0f0fc0 maskL=0x03f0f03f  mulhi=0x00400401 mullo=0x00100100
SOLVED cbba  dword=LE  idx_order=(2,3,0,1)  maskR=0xfc0f0fc0 maskL=0x03f0f03f  mulhi=0x00410401 mullo=0x00100100
SOLVED cbba  dword=LE  idx_order=(2,3,0,1)  maskR=0xfc0f0fc0 maskL=0x03f0f03f  mulhi=0x00400402 mullo=0x00100100
SOLVED cbba  dword=LE  idx_order=(2,3,0,1)  maskR=0xfc0f0fc0 maskL=0x03f0f03f  mulhi=0x00410402 mullo=0x00100100
SOLVED cbba  dword=LE  idx_order=(2,3,0,1)  maskR=0xfc0f0fc0 maskL=0x03f0f03f  mulhi=0x00400403 mullo=0x00100100
SOLVED cbba  dword=LE  idx_order=(2,3,0,1)  maskR=0xfc0f0fc0 maskL=0x03f0f03f  mulhi=0x00410403 mullo=0x00100100
SOLVED cbba  dword=LE  idx_order=(2,3,0,1)  maskR=0xfc0f0fc0 maskL=0x03f0f03f  mulhi=0x00400404 mullo=0x00100100
SOLVED cbba  dword=LE  idx_order=(2,3,0,1)  maskR=0xfc0f0fc0 maskL=0x03f0f03f  mulhi=0x00410404 mullo=0x00100100
SOLVED cbba  dword=LE  idx_order=(2,3,0,1)  maskR=0xfc0f0fc0 maskL=0x03f0f03f  mulhi=0x00400405 mullo=0x00100100
SOLVED cbba  dword=LE  idx_order=(2,3,0,1)  maskR=0xfc0f0fc0 maskL=0x03f0f03f  mulhi=0x00410405 mullo=0x00100100
SOLVED cbba  dword=LE  idx_order=(2,3,0,1)  maskR=0xfc0f0fc0 maskL=0x03f0f03f  mulhi=0x00400406 mullo=0x00100100
SOLVED cbba  dword=LE  idx_order=(2,3,0,1)  maskR=0xfc0f0fc0 maskL=0x03f0f03f  mulhi=0x00410406 mullo=0x00100100
SOLVED cbba  dword=LE  idx_order=(2,3,0,1)  maskR=0xfc0f0fc0 maskL=0x03f0f03f  mulhi=0x00400407 mullo=0x00100100
SOLVED cbba  dword=LE  idx_order=(2,3,0,1)  maskR=0xfc0f0fc0 maskL=0x03f0f03f  mulhi=0x00410407 mullo=0x00100100
SOLVED cbba  dword=LE  idx_order=(2,3,0,1)  maskR=0xfc0f0fc0 maskL=0x03f0f03f  mulhi=0x00400408 mullo=0x00100100
SOLVED cbba  dword=LE  idx_order=(2,3,0,1)  maskR=0xfc0f0fc0 maskL=0x03f0f03f  mulhi=0x00410408 mullo=0x00100100
SOLVED abbc  dword=BE  idx_order=(2,3,0,1)  maskR=0xfc0f0fc0 maskL=0x03f0f03f  mulhi=0x00400400 mullo=0x00100100
SOLVED bacb  dword=LE  idx_order=(0,1,2,3)  maskR=0x0fc0fc0f maskL=0xf03f03f0  mulhi=0x04000040 mullo=0x01000010
SOLVED cbba  dword=LE  idx_order=(2,3,0,1)  maskR=0xfc0f0fc0 maskL=0x03f0f03f  mulhi=0x00400400 mullo=0x00100100
SOLVED abbc  dword=BE  idx_order=(2,3,0,1)  maskR=0xfc0f0fc0 maskL=0x03f0f03f  mulhi=0x00400400 mullo=0x00100100
SOLVED bacb  dword=LE  idx_order=(0,1,2,3)  maskR=0x0fc0fc0f maskL=0xf03f03f0  mulhi=0x04000040 mullo=0x01000010
SOLVED cbba  dword=LE  idx_order=(2,3,0,1)  maskR=0xfc0f0fc0 maskL=0x03f0f03f  mulhi=0x00400400 mullo=0x00100100

Final Notes: the 0x0fc0fc00 and 0x003f03f0 masks would also work on LE and vice versa, it depends on the input arrangement
and the mask.
on some instances and input alignment would be needed like this below code.
        __m128i lo = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src));      // bytes 0-15
        __m128i hi = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + 12)); // bytes 12-27
        __m256i inputvector = _mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1);
also in some combinations setr instead of set might be needed, it depends on data arrangement.        
on some combaniations like cbba there might be needed one final shuffle to arrange the bytes before the LUT or other methods to 
put the bytes in the correct order.


These are some more methods we have discovered, but not implemented here.
we are keeping these for ourselves for now.
using pure maddubs for all indices extraction is possible, but it is slower than the mulhi/mullo methods.
all of them may not work as expected thats why they are in comments but either functions work fine.
would need to change just a bit to work.

They are very much experimental and not tested much.
we arent and havent used them properly either but we kept them for future reference.

static inline __m256i extract_indices_pure_madd(__m256i v) {
    // Index 0: a>>2 using MADDUBS (byte 2 in [c][b][a][0] format)
    const __m256i mask_a = _mm256_set1_epi32(0x00FC0000);
    const __m256i weights_a = _mm256_setr_epi8(
        0, 0, 1, 0,  0, 0, 1, 0,  0, 0, 1, 0,  0, 0, 1, 0,
        0, 0, 1, 0,  0, 0, 1, 0,  0, 0, 1, 0,  0, 0, 1, 0
    );
    __m256i va = _mm256_and_si256(v, mask_a);
    __m256i ta = _mm256_maddubs_epi16(va, weights_a);
    __m256i i0 = _mm256_and_si256(_mm256_srli_epi32(ta, 18), _mm256_set1_epi32(0x3F));
    
    // Index 1: ((a&3)<<4)|(b>>4) using MADDUBS with correct byte positions
    // a is at byte 2, b is at byte 1 in [c][b][a][0] format
    __m256i a_part = _mm256_and_si256(v, _mm256_set1_epi32(0x00030000));  // a&3 at byte 2
    __m256i b_part = _mm256_and_si256(v, _mm256_set1_epi32(0x0000F000));  // b>>4 at byte 1
    __m256i a_shifted = _mm256_srli_epi32(a_part, 16);  // Move a&3 to byte 0
    __m256i b_shifted = _mm256_srli_epi32(b_part, 4);   // Move b>>4 to byte 1
    __m256i combined_ab = _mm256_or_si256(a_shifted, b_shifted);
    __m256i weights_ab = _mm256_setr_epi8(
        16, 1, 0, 0,  16, 1, 0, 0,  16, 1, 0, 0,  16, 1, 0, 0,
        16, 1, 0, 0,  16, 1, 0, 0,  16, 1, 0, 0,  16, 1, 0, 0
    );
    __m256i tab_i1 = _mm256_maddubs_epi16(combined_ab, weights_ab);
    __m256i i1 = _mm256_and_si256(tab_i1, _mm256_set1_epi32(0x3F));
    
    // Index 2: Bits 11-6 using MADDUBS
    __m256i shifted_i2 = _mm256_srli_epi32(v, 6);  // Shift right by 6 to get bits 11-6 in position 5-0
    __m256i mask_i2 = _mm256_set1_epi32(0x0000003F);  // Mask bottom 6 bits
    __m256i masked_i2 = _mm256_and_si256(shifted_i2, mask_i2);
    __m256i weights_i2 = _mm256_setr_epi8(
        1, 0, 0, 0,  1, 0, 0, 0,  1, 0, 0, 0,  1, 0, 0, 0,
        1, 0, 0, 0,  1, 0, 0, 0,  1, 0, 0, 0,  1, 0, 0, 0
    );
    __m256i tab_i2 = _mm256_maddubs_epi16(masked_i2, weights_i2);
    __m256i i2 = _mm256_and_si256(tab_i2, _mm256_set1_epi32(0x3F));
    
    // Index 3: Bottom 6 bits using MADDUBS
    __m256i mask_i3 = _mm256_set1_epi32(0x0000003F);  // Bottom 6 bits
    __m256i masked_i3 = _mm256_and_si256(v, mask_i3);
    __m256i weights_i3 = _mm256_setr_epi8(
        1, 0, 0, 0,  1, 0, 0, 0,  1, 0, 0, 0,  1, 0, 0, 0,
        1, 0, 0, 0,  1, 0, 0, 0,  1, 0, 0, 0,  1, 0, 0, 0
    );
    __m256i tab_i3 = _mm256_maddubs_epi16(masked_i3, weights_i3);
    __m256i i3 = _mm256_and_si256(tab_i3, _mm256_set1_epi32(0x3F));
    
    // Pack results
    return _mm256_or_si256(
        _mm256_or_si256(i0, _mm256_slli_epi32(i1, 8)),
        _mm256_or_si256(_mm256_slli_epi32(i2, 16), _mm256_slli_epi32(i3, 24))
    );
}




// cba0 per 32-bit dword: byte0=c, byte1=b, byte2=a, byte3=0  [arXiv:1704.00605]
alignas(32) static const __m256i w_even = _mm256_setr_epi8(
    1,0, 1,0,  1,0, 1,0,   1,0, 1,0,  1,0, 1,0,
    1,0, 1,0,  1,0, 1,0,   1,0, 1,0,  1,0, 1,0
); // select even byte of each pair (0,1) and (2,3) per dword [VPMADDUBSW semantics]

// 16-bit lane masks: keep low byte of even or odd 16-bit words
alignas(32) static const __m256i keep_even_words = _mm256_setr_epi16(
    0x00FF,0x0000, 0x00FF,0x0000, 0x00FF,0x0000, 0x00FF,0x0000,
    0x00FF,0x0000, 0x00FF,0x0000, 0x00FF,0x0000, 0x00FF,0x0000
); // words 0,2,4,6,... (c-words) [even 16b lanes]

alignas(32) static const __m256i keep_odd_words = _mm256_setr_epi16(
    0x0000,0x00FF, 0x0000,0x00FF, 0x0000,0x00FF, 0x0000,0x00FF,
    0x0000,0x00FF, 0x0000,0x00FF, 0x0000,0x00FF, 0x0000,0x00FF
); // words 1,3,5,7,... (a-words) [odd 16b lanes]

alignas(32) static const __m256i m6_32 = _mm256_set1_epi32(0x3F); // sextet mask

static inline __m256i extract_indices_madd_avx2_min(__m256i v) {
    // 1) One maddubs selects both c (pair 0,1) and a (pair 2,3) per dword into alternating 16-bit words [even,odd]
    __m256i t16 = _mm256_maddubs_epi16(v, w_even);

    // 2) Split: i0 from odd words (a), i3 from even words (c)
    __m256i a16 = _mm256_and_si256(t16, keep_odd_words);
    __m256i c16 = _mm256_and_si256(t16, keep_even_words);

    // 3) Finish i0 and i3 in 16-bit lanes
    __m256i i0w = _mm256_srli_epi16(a16, 2);
    __m256i i3w = _mm256_and_si256(c16, _mm256_set1_epi16(0x003F));

    // 4) Widen to 32-bit - extract from correct positions
    __m256i i0_32 = _mm256_and_si256(_mm256_srli_epi32(i0w, 16), m6_32);
    __m256i i3_32 = _mm256_and_si256(i3w, m6_32);

    // 5) Canonical i1, i2
    __m256i i1 = _mm256_and_si256(_mm256_srli_epi32(v, 12), m6_32);
    __m256i i2 = _mm256_and_si256(_mm256_srli_epi32(v,  6), m6_32);

    // 6) Pack to bytes [i0,i1,i2,i3] per dword
    return _mm256_or_si256(_mm256_or_si256(i0_32, _mm256_slli_epi32(i1, 8)),
                          _mm256_or_si256(_mm256_slli_epi32(i2, 16), _mm256_slli_epi32(i3_32, 24)));
}


// cba0 per 32-bit dword: byte0=c, byte1=b, byte2=a, byte3=0 [1]
alignas(32) static const __m256i w_even = _mm256_setr_epi8(
    1,0, 1,0,  1,0, 1,0,   1,0, 1,0,  1,0, 1,0,
    1,0, 1,0,  1,0, 1,0,   1,0, 1,0,  1,0, 1,0
); // select even byte of both pairs per dword (c from (0,1), a from (2,3)) [2]

alignas(32) static const __m256i keep_even_words = _mm256_setr_epi16(
    0x00FF,0x0000, 0x00FF,0x0000, 0x00FF,0x0000, 0x00FF,0x0000,
    0x00FF,0x0000, 0x00FF,0x0000, 0x00FF,0x0000, 0x00FF,0x0000
); // c-words (word 0 per dword) [3]

alignas(32) static const __m256i keep_odd_words = _mm256_setr_epi16(
    0x0000,0x00FF, 0x0000,0x00FF, 0x0000,0x00FF, 0x0000,0x00FF,
    0x0000,0x00FF, 0x0000,0x00FF, 0x0000,0x00FF, 0x0000,0x00FF
); // a-words (word 1 per dword) [3]

alignas(32) static const __m256i m6_32     = _mm256_set1_epi32(0x3F);      // sextet mask [1]
alignas(32) static const __m256i m_i2_b_lo = _mm256_set1_epi32(0x00000F00); // b low nibble (byte1) [1]
alignas(32) static const __m256i m_i2_c_hi = _mm256_set1_epi32(0x000000C0); // c high 2 bits (byte0) [1]

// weights for i2 on pair (0,1) = (c,b): even=1 for c, odd=4 for b; zero for pair (2,3) [2]
alignas(32) static const __m256i w_i2 = _mm256_setr_epi8(
    1,4, 0,0,  1,4, 0,0,  1,4, 0,0,  1,4, 0,0,
    1,4, 0,0,  1,4, 0,0,  1,4, 0,0,  1,4, 0,0
);

static inline __m256i extract_indices_madd_avx2_min(__m256i v) {
    // 1) One maddubs: select c (pair 0,1 even) and a (pair 2,3 even) into alternating 16-bit words [2]
    __m256i t16 = _mm256_maddubs_epi16(v, w_even);

    // 2) Split: i0 from odd words (a), i3 from even words (c) [3]
    __m256i a16 = _mm256_and_si256(t16, keep_odd_words);
    __m256i c16 = _mm256_and_si256(t16, keep_even_words);

    // 3) Finish i0 (a>>2) and i3 (c&63) in 16-bit lanes, then extract to 32-bit bitfields [1][3]
    __m256i i0w   = _mm256_srli_epi16(a16, 2);                  // a>>2 in word1 [1]
    __m256i i3w   = _mm256_and_si256(c16, _mm256_set1_epi16(0x003F)); // c&63 in word0 [1]
    __m256i i0_32 = _mm256_and_si256(_mm256_srli_epi32(i0w, 16), m6_32); // pick word1 -> dword [3]
    __m256i i3_32 = _mm256_and_si256(i3w, m6_32);                       // word0 already in low 16 [3]

    // 4) i1 stays canonical: (v>>12) & 63 (mixes a and b across pairs; dot-prod would need an extra shuffle) [1][4]
    __m256i i1 = _mm256_and_si256(_mm256_srli_epi32(v, 12), m6_32);     // sextet in bits 5..0 [1]

    // 5) i2 via one maddubs: pre-shift c>>6, scale b_low4<<2, sum in pair (0,1) -> word0 [1][2]
    __m256i vb    = _mm256_and_si256(v, m_i2_b_lo);              // b & 0x0F at byte1 [1]
    __m256i vc    = _mm256_and_si256(v, m_i2_c_hi);              // c & 0xC0 at byte0 [1]
    __m256i vc6   = _mm256_srli_epi16(vc, 6);                    // c>>6 into bits 0..1 [1]
    __m256i i2src = _mm256_or_si256(vb, vc6);                    // pair (0,1) holds both terms [1]
    __m256i i2w   = _mm256_maddubs_epi16(i2src, w_i2);           // (b&0xF)*4 + (c>>6) in word0 [2]
    __m256i i2    = _mm256_and_si256(i2w, m6_32);                // sextet in low 6 bits [1]

    // 6) Pack to bytes [i0,i1,i2,i3] per dword [1]
    __m256i r0 = i0_32;
    __m256i r1 = _mm256_slli_epi32(i1, 8);
    __m256i r2 = _mm256_slli_epi32(i2, 16);
    __m256i r3 = _mm256_slli_epi32(i3_32, 24);
    return _mm256_or_si256(_mm256_or_si256(r0, r1), _mm256_or_si256(r2, r3));
}


// Maddubs #1: select even bytes of both pairs -> c (pair 0,1) and a (pair 2,3). [2]
alignas(32) static const __m256i w_even = _mm256_setr_epi8(
    1,0, 1,0,  1,0, 1,0,   1,0, 1,0,  1,0, 1,0,
    1,0, 1,0,  1,0, 1,0,   1,0, 1,0,  1,0, 1,0
);

// Maddubs #2: weights per dword bytes [b0,b1,b2,b3] = [1,4,16,1] -> word0=i2, word1=i1. [2]
alignas(32) static const __m256i w_i12 = _mm256_setr_epi8(
    1,4,16,1,  1,4,16,1,  1,4,16,1,  1,4,16,1,
    1,4,16,1,  1,4,16,1,  1,4,16,1,  1,4,16,1
);

// 16-bit lane masks: keep even or odd words (word0->even, word1->odd) per dword. [3]
alignas(32) static const __m256i keep_even_words = _mm256_setr_epi16(
    0x00FF,0x0000, 0x00FF,0x0000, 0x00FF,0x0000, 0x00FF,0x0000,
    0x00FF,0x0000, 0x00FF,0x0000, 0x00FF,0x0000, 0x00FF,0x0000
);
alignas(32) static const __m256i keep_odd_words = _mm256_setr_epi16(
    0x0000,0x00FF, 0x0000,0x00FF, 0x0000,0x00FF, 0x0000,0x00FF,
    0x0000,0x00FF, 0x0000,0x00FF, 0x0000,0x00FF, 0x0000,0x00FF
);

// Common masks. [1]
alignas(32) static const __m256i m6_32     = _mm256_set1_epi32(0x3F);
alignas(32) static const __m256i m_i2_c_hi = _mm256_set1_epi32(0x000000C0); // c>>6 source [1]
alignas(32) static const __m256i m_i2_b_lo = _mm256_set1_epi32(0x00000F00); // b&0x0F source [1]
alignas(32) static const __m256i m_i1_a_lo = _mm256_set1_epi32(0x00030000); // a&3 source [1]
alignas(32) static const __m256i m_i1_b_hi = _mm256_set1_epi32(0x0000F000); // b>>4 source [1]

// Extract i1 and i2 with one VPMADDUBSW. [2][1]
static inline void compute_i1_i2_with_maddubs(__m256i v, __m256i& i1_32, __m256i& i2_32) {
    __m256i vc    = _mm256_and_si256(v, m_i2_c_hi);                 // c&0xC0 at byte0 [1]
    __m256i vc6   = _mm256_srli_epi16(vc, 6);                       // c>>6 -> bits 0..1 of byte0 [1]
    __m256i vb_lo = _mm256_and_si256(v, m_i2_b_lo);                 // b&0x0F at byte1 [1]
    __m256i va2   = _mm256_and_si256(v, m_i1_a_lo);                 // a&3 at byte2 [1]
    __m256i vb_hi = _mm256_slli_epi32(_mm256_and_si256(v, m_i1_b_hi), 12); // b>>4 into byte3 low nibble [1]

    __m256i T     = _mm256_or_si256(_mm256_or_si256(vc6, vb_lo), _mm256_or_si256(va2, vb_hi)); // [b0,b1,b2,b3] [1]
    __m256i i12w  = _mm256_maddubs_epi16(T, w_i12);                 // word0=i2, word1=i1 per dword [2]

    __m256i i2w   = _mm256_and_si256(i12w, keep_even_words);        // word0 [3]
    __m256i i1w   = _mm256_and_si256(i12w, keep_odd_words);         // word1 [3]
    i2_32         = _mm256_and_si256(i2w, m6_32);                   // sextet in low 6 bits [1]
    i1_32         = _mm256_and_si256(_mm256_srli_epi32(i1w, 16), m6_32); // bring word1 down [3]
}

static inline __m256i extract_indices_madd_avx2_full(__m256i v) {
    // Maddubs #1: c and a into alternating words. [2]
    __m256i t16 = _mm256_maddubs_epi16(v, w_even);

    // Split and finish i0 (a>>2) and i3 (c&63). [1][3]
    __m256i a16 = _mm256_and_si256(t16, keep_odd_words);
    __m256i c16 = _mm256_and_si256(t16, keep_even_words);
    __m256i i0w = _mm256_srli_epi16(a16, 2);                        // a>>2 [1]
    __m256i i3w = _mm256_and_si256(c16, _mm256_set1_epi16(0x003F)); // c&63 [1]
    __m256i i0_32 = _mm256_and_si256(_mm256_srli_epi32(i0w, 16), m6_32); // pick odd word [3]
    __m256i i3_32 = _mm256_and_si256(i3w, m6_32);                        // pick even word [3]

    // Maddubs #2: compute i1 and i2 in one pass. [2][1]
    __m256i i1_32, i2_32;
    compute_i1_i2_with_maddubs(v, i1_32, i2_32);

    // Pack [i0,i1,i2,i3] per dword. [1]
    __m256i r0 = i0_32;
    __m256i r1 = _mm256_slli_epi32(i1_32, 8);
    __m256i r2 = _mm256_slli_epi32(i2_32, 16);
    __m256i r3 = _mm256_slli_epi32(i3_32, 24);
    return _mm256_or_si256(_mm256_or_si256(r0, r1), _mm256_or_si256(r2, r3));
}


Now it is possible  simply to integrate directly character mappings like below.

// idx: 32 sextets (u8), each in [0..63] with 0x3F mask already applied
// Build a “reduced nibble” key per Muła–Lemire, then VPSHUFB offsets and add.
// Lane-local 16B lookup replicated in both 128-bit lanes.
const __m256i tbl = _mm256_setr_epi8(
    71,-4,-4,-4,-4,-4,-4,-4, -4,-4,-4,-19,-16, 65, 0, 0,
    71,-4,-4,-4,-4,-4,-4,-4, -4,-4,-4,-19,-16, 65, 0, 0);

// ...compute 'key' (reduced nibble per byte) with compares/arith per paper...
__m256i offs = _mm256_shuffle_epi8(tbl, key);   // VPSHUFB per 128-bit lane
__m256i ascii = _mm256_add_epi8(idx, offs);     // add offset -> ASCII bytes

### How it works
After extracting the four sextets per 3-byte group with VPMADDUBSW and simple shifts/masks, treat each sextet as an 8-bit element, 
compute a small “range key” per element, use a lane-local VPSHUFB to fetch the ASCII offset for that range, 
and add it to the sextet to get the ASCII code directly in bytes. 
This avoids a 64-entry table and keeps everything lane-local, which fits AVX2’s 128-bit-per-lane VPSHUFB semantics for high throughput.
The five offsets from sextet value $x\in$ to ASCII are $+65$ for $x\in$, $+71$ for $x\in$, $-4$ for $x\in$, $-19$ for $x=62$, 
and $-16$ for $x=63$, so the final character is $x + \text{offset}(x)$ per byte lane. Muła–Lemire derive a cheap “reduced nibble” 
per $x$ (e.g., 13 for A–Z, 0 for a–z, $x+1$ for digits, 11 for ‘+’, 12 for ‘/’) and then use VPSHUFB with a 16-byte table of 
offsets like [71, −4, …, −19, −16, 65, 0, 0] to materialize the correct per-lane offset to add to $x$.
Base64’s alphabet is five contiguous ranges, so each sextet 
x∈ can be turned into its ASCII code by adding a small per-range offset, implemented with a 16-byte l
ane-local VPSHUFB of offsets followed by a byte add, producing final characters directly from the 
sextets in YMM registers. This eliminates the need to feed a vector of indices into a separate 
“index-to-ASCII” packing routine, because the mapping stage itself yields the final ASCII bytes per lane.
There is still a lightweight byte-ordering step to arrange the four character streams per 3-byte group 
into a contiguous 32-byte output block, but this is a lane-local shuffle/transpose and not the heavier “pack indices” 
function that operates on 32-bit slots of bitfields. On AVX2, keep in mind that VPSHUFB is lane-local, so any final 
interleave must either stay within 128-bit halves or use one cross-lane permute to stitch halves, which is standard in published Base64 encoders

*/

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
    #ifdef __x86_64__ 
    _mm256_zeroupper();
    #endif
    
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