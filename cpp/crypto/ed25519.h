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
#include <stdexcept>
#include <optional>
#include <utility>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <openssl/evp.h>
#include <openssl/sha.h>
#include <openssl/rand.h>
#include <openssl/err.h>
#include "../global.h"
#include "../json/canonicalization.h"
#include "../base64/encoders/include/base64-encoder.h"
#include "../base64/decoders/include/base64-decoder.h"

namespace nb = nanobind;

// Thread-local variable declarations
extern thread_local std::vector<uint8_t> ed25519_hash_buffer;

// Forward declarations
class VerifyKey;

// Class declarations
class Signature {
public:
    nb::bytes signature;
    Signature(nb::bytes sig) : signature(sig) {}
};

class VerifyKey {
public:
    std::vector<uint8_t> key_bytes;
    std::string alg;
    std::string version;
    
    VerifyKey(const std::vector<uint8_t>& bytes, const std::string& algorithm = "ed25519", const std::string& ver = "1");
    nb::bytes encode() const;
    void verify(nb::bytes message, nb::bytes signature) const;
};

class VerifyKeyWithExpiry : public VerifyKey {
public:
    int expired = 0;
    
    VerifyKeyWithExpiry(const std::vector<uint8_t>& bytes, const std::string& algorithm = "ed25519", const std::string& ver = "1")
        : VerifyKey(bytes, algorithm, ver) {}
};

class SigningKey {
public:
    std::vector<uint8_t> key_bytes;
    std::string alg;
    std::string version;
    
    SigningKey(const std::vector<uint8_t>& bytes, const std::string& algorithm = "ed25519", const std::string& ver = "1");
    static SigningKey generate();
    nb::bytes encode() const;
    VerifyKey get_verify_key() const;
    Signature sign(nb::bytes message) const;
};

struct SigningResult {
    std::string signature;
    std::string key_id;
    std::string algorithm;
};

struct VerificationResult {
    bool valid;
    std::optional<std::string> user_id;
    std::optional<bool> device_valid;
};

// Function declarations
std::vector<uint8_t> get_verify_key(const std::vector<uint8_t>& signing_key);
[[gnu::hot, gnu::flatten]] bool verify_signature_fast(std::span<const uint8_t> json_bytes, std::string_view signature_b64, const std::vector<uint8_t>& verify_key_bytes);
[[gnu::hot, gnu::flatten]] std::pair<std::string, std::vector<uint8_t>> compute_content_hash_fast(std::span<const uint8_t> data);
[[gnu::hot, gnu::flatten]] std::pair<std::string, std::vector<uint8_t>> compute_content_hash_fast(const std::string& event_json);
std::pair<std::string, std::vector<uint8_t>> compute_content_hash(const nb::dict& event_dict);
nb::dict sign_json_object_fast(const nb::dict& json_dict, const std::string& signature_name, 
                               const std::vector<uint8_t>& signing_key_bytes, const std::string& key_id);
void verify_signed_json_fast(const nb::dict& json_dict, const std::string& server_name, const std::vector<uint8_t>& verify_key_bytes);
SigningResult sign_json_with_info(const std::vector<uint8_t>& json_bytes, const std::vector<uint8_t>& signing_key_bytes, const std::string& version);
VerificationResult verify_signature_with_info(const std::vector<uint8_t>& json_bytes, const std::string& signature_b64, const std::vector<uint8_t>& verify_key_bytes);