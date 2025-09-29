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

#pragma once
#include "../../../global.h"

#if defined(__AVX2__) && !defined(DISABLE_SSE_BASE64_ENCODER) || defined(__ARM_NEON) && !defined(DISABLE_NEON_BASE64_ENCODER) || defined(__AVX2__) && !defined(DISABLE_NEON_BASE64_ENCODER)
#include <vector>
#include <string>
#include <cstdint>
#endif
#if defined(__AVX2__) && !defined(DISABLE_SSE_BASE64_ENCODER) || defined(__ARM_NEON) && !defined(DISABLE_SSE_BASE64_ENCODER)
extern thread_local std::string base64_buffer;
[[gnu::hot, gnu::flatten, clang::always_inline]] std::string fast_sse_base64_encode(const std::vector<uint8_t>& data);
#endif
#if defined(__ARM_NEON) && !defined(DISABLE_NEON_BASE64_ENCODER) || defined(__AVX2__) && !defined(DISABLE_NEON_BASE64_ENCODER)
extern thread_local std::string base64_buffer;
[[gnu::hot, gnu::flatten, clang::always_inline]] std::string fast_neon_base64_encode(const std::vector<uint8_t>& data);
#endif