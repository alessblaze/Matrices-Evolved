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

#if defined(__AVX2__) && !defined(DISABLE_SSE_BASE64_ENCODER_AVX)
#pragma once
#include <vector>
#include <string>
#include <cstdint>
#include "../../../global.h"
extern thread_local std::string base64_buffer;
std::string fast_sse_base64_encode_avx(const std::vector<uint8_t>& data);
#endif