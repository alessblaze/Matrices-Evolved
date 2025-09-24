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
#include <vector>
#include <string>
#include <string_view>
#include <span>
#include <cstdint>
#include <iostream>
#include <cstdlib>
#include <stdexcept>
#include <charconv>
#include <cmath>
#include <cstring>
#include <type_traits>
#include <boost/json.hpp>
#include <openssl/evp.h>
#include <openssl/rand.h>
#include <Python.h>
#include "../global.h"
#include "../base64/encoders/include/base64-encoder.h"
#include "../base64/decoders/include/base64-decoder.h"

namespace nb = nanobind;
namespace json = boost::json;

// Thread-local variable declarations
extern thread_local std::vector<char> json_buffer;
extern thread_local char* json_ptr;
extern thread_local std::vector<uint8_t> json_signature_buffer;
extern thread_local std::vector<uint8_t> json_hash_buffer;
extern thread_local std::vector<uint8_t> cached_signing_key;
extern thread_local EVP_PKEY* cached_signing_pkey;

// Function declarations
template<typename T = std::vector<uint8_t>>
T get_json_span();

// Explicit template instantiation declarations
extern template std::vector<uint8_t> get_json_span<std::vector<uint8_t>>();
extern template std::span<const uint8_t> get_json_span<std::span<const uint8_t>>();
extern template std::span<uint8_t> get_json_span<std::span<uint8_t>>();

[[gnu::hot, gnu::flatten]] void py_to_canonical_json_fast(const nb::object& root_obj);
std::string sign_json_fast(std::span<const uint8_t> json_bytes, const std::vector<uint8_t>& signing_key_bytes);
void init_json_buffer(size_t hint = 0);
std::vector<uint8_t> generate_signing_key();
std::string canonicalize_json_fast(const json::value& jv);
void serialize_canonical_fast(const json::value& v);
std::string base64_encode(const std::vector<uint8_t>& data);

// Internal helper functions
void ensure_space(size_t needed);
void write_char(char c);
void write_string(std::string_view s);
void write_raw(const char* data, size_t len);
void write_raw_unsafe(const char* data, size_t len);
void write_cstring(const char* s);
void write_unicode_escape(unsigned char c);
void write_unicode_escape_unsafe(unsigned char c);
int fast_double_to_string(double f, char* result);

#ifdef __AVX2__
bool has_decimal_point_sse(const char* str, size_t len);
std::string vectorized_hex_string(const uint8_t* data, size_t size);
#endif