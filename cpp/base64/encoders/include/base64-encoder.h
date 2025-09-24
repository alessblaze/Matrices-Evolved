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
#include <vector>
#include <string>
#include <cstdint>
#include "../../../global.h"


#if !defined(DISABLE_SSE_BASE64_ENCODER_AVX)
#include "ams-avx.h"
#endif
#if !defined(DISABLE_SSE_BASE64_ENCODER) || !defined(DISABLE_SSE_BASE64_ENCODER_SSE)
#include "ams-sse.h"
#endif
#if !defined(DISABLE_SSE_BASE64_ENCODER_ALIGNED)
#include "ams-sse-aligned.h"
#endif
#if !defined(DISABLE_SSE_BASE64_ENCODER_LEMIRE)
#include "lemire-avx.h"
#endif
#if !defined(DISABLE_SSE_BASE64_ENCODER_MULA)
#include "mulla-sse.h"
#endif


// Thread-local variable declarations
extern thread_local std::string encoder_base64_buffer;

// Function declarations
std::string base64_encode(const std::vector<uint8_t>& data);