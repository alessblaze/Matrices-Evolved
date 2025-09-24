/*
 * Copyright (C) 2025 Aless Microsystems
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published
 * by the Free Software Foundation, version 3 of the License, or under
 * alternative licensing terms as granted by Aless Microsystems.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 */

#pragma once
#if defined(__AVX2__) && !defined(DISABLE_SSE_BASE64_ENCODER) || defined(__ARM_NEON) && !defined(DISABLE_NEON_BASE64_ENCODER) || defined(__AVX2__) && !defined(DISABLE_NEON_BASE64_ENCODER)
#include <vector>
#include <string>
#include <cstdint>
#include "../../../global.h"
#endif
#if defined(__AVX2__) && !defined(DISABLE_SSE_BASE64_ENCODER)
extern thread_local std::string base64_buffer;
std::string fast_sse_base64_encode(const std::vector<uint8_t>& data);
#endif
#if defined(__ARM_NEON) && !defined(DISABLE_NEON_BASE64_ENCODER) || defined(__AVX2__) && !defined(DISABLE_NEON_BASE64_ENCODER)
extern thread_local std::string base64_buffer;
std::string fast_neon_base64_encode(const std::vector<uint8_t>& data);
#endif