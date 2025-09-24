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
#include <string_view>
#include <cstdint>

/**
 * High-performance AVX2-optimized base64 decoder for cryptographic signatures
 * @param input Base64 encoded string to decode
 * @return Decoded binary data optimized for signature verification
 */
extern thread_local std::vector<uint8_t> decode_buffer;

[[gnu::hot, gnu::flatten]] std::vector<uint8_t> fast_base64_decode_signature(std::string_view input);