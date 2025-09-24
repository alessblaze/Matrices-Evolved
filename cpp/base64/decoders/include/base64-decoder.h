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
#include <string_view>
#include <cstdint>
#include <iostream>
#include <cstdlib>
#include <openssl/evp.h>
#include <openssl/bio.h>
#include "../../../global.h"

// Constants

// Thread-local decode buffer
extern thread_local std::vector<uint8_t> decode_buffer;

// Function declarations
std::vector<uint8_t> base64_decode(const std::string& encoded);
std::vector<uint8_t> base64_decode(std::string_view encoded_string);
std::vector<uint8_t> fast_base64_decode_signature(std::string_view input);