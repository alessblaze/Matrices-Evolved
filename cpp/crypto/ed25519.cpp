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

#include "ed25519.h"

// Local thread-local buffers for ed25519 module
thread_local std::vector<uint8_t> ed25519_hash_buffer(32);


/**
 * Derives the public verification key from an Ed25519 signing key
 * @param signing_key 32-byte Ed25519 private key
 * @return 32-byte Ed25519 public key for signature verification
 */
std::vector<uint8_t> get_verify_key(const std::vector<uint8_t>& signing_key) {
    if (signing_key.size() != 32) {
        throw std::runtime_error("Invalid signing key length");
    }
    
    std::vector<uint8_t> pk(32); // Ed25519 public key size
    
    try {
        nb::gil_scoped_release release;
        EVP_PKEY* pkey = EVP_PKEY_new_raw_private_key(EVP_PKEY_ED25519, nullptr, signing_key.data(), 32);
        if (!pkey) {
            throw std::runtime_error("Failed to create Ed25519 key");
        }
        
        size_t pk_len = 32;
        if (EVP_PKEY_get_raw_public_key(pkey, pk.data(), &pk_len) != 1 || pk_len != 32) {
            EVP_PKEY_free(pkey);
            throw std::runtime_error("Failed to extract public key");
        }
        
        EVP_PKEY_free(pkey);
    } catch (const std::exception& e) {
        throw std::runtime_error("Key derivation failed: " + std::string(e.what()));
    }
    return pk;
}

// No key caching - always expand fresh


/**
 * Verifies Ed25519 digital signature against JSON data
 * @param json_bytes Canonical JSON data that was signed
 * @param signature_b64 Base64-encoded signature to verify
 * @param verify_key_bytes 32-byte Ed25519 public key
 * @return True if signature is valid, false otherwise
 */
[[gnu::hot, gnu::flatten]] bool verify_signature_fast(std::span<const uint8_t> json_bytes, std::string_view signature_b64, const std::vector<uint8_t>& verify_key_bytes) {
    try {
        if (verify_key_bytes.size() != 32) {
            DEBUG_LOG("ERROR: Invalid verify key size: " + std::to_string(verify_key_bytes.size()));
            return false;
        }
        
        auto signature_bytes = base64_decode(signature_b64);
        if (signature_bytes.size() != 64) {
            DEBUG_LOG("ERROR: Invalid signature size after decode: " + std::to_string(signature_bytes.size()));
            return false;
        }
    
    if (debug_enabled) {
        std::string sig_hex;
        sig_hex.reserve(std::min(size_t(16), signature_bytes.size()) * 2);
        for (size_t i = 0; i < std::min(size_t(16), signature_bytes.size()); i++) {
            sig_hex.push_back(hex_lut[signature_bytes[i] >> 4]);
            sig_hex.push_back(hex_lut[signature_bytes[i] & 0x0F]);
        }
        DEBUG_LOG("Decoded signature bytes (first 16): " + sig_hex);
        DEBUG_LOG("JSON bytes to verify: " + std::to_string(json_bytes.size()) + " bytes");
    }
    
    bool result = false;
    {
        nb::gil_scoped_release release;
        
        EVP_PKEY* pkey = EVP_PKEY_new_raw_public_key(EVP_PKEY_ED25519, nullptr, verify_key_bytes.data(), 32);
        if (!pkey) {
            DEBUG_LOG("ERROR: Failed to create EVP_PKEY from verify key");
            return false;
        }
        
        EVP_MD_CTX* ctx = EVP_MD_CTX_new();
        if (!ctx) {
            DEBUG_LOG("ERROR: Failed to create EVP_MD_CTX");
            EVP_PKEY_free(pkey);
            return false;
        }
        
        if (EVP_DigestVerifyInit(ctx, nullptr, nullptr, nullptr, pkey) != 1) {
            DEBUG_LOG("ERROR: EVP_DigestVerifyInit failed");
            EVP_MD_CTX_free(ctx);
            EVP_PKEY_free(pkey);
            return false;
        }
        
        int verify_result = EVP_DigestVerify(ctx, signature_bytes.data(), 64, json_bytes.data(), json_bytes.size());
        result = (verify_result == 1);
        
        if (debug_enabled) {
            DEBUG_LOG("EVP_DigestVerify result: " + std::to_string(verify_result) + " (1=success, 0=fail, <0=error)");
            DEBUG_LOG("Signature verification: " + std::string(result ? "SUCCESS" : "FAILED"));
        }
        
        EVP_MD_CTX_free(ctx);
        EVP_PKEY_free(pkey);
    }
    
    return result;
    } catch (const std::exception& e) {
        DEBUG_LOG("ERROR: Signature verification failed with exception: " + std::string(e.what()));
        return false;
    }
}

/**
 * Computes SHA256 content hash of binary data
 * @param data Binary data to hash
 * @return Pair of algorithm name ("sha256") and 32-byte hash digest
 */
[[gnu::hot, gnu::flatten]] std::pair<std::string, std::vector<uint8_t>> compute_content_hash_fast(std::span<const uint8_t> data) {
    ed25519_hash_buffer.resize(32); // SHA256 hash size
    {
        nb::gil_scoped_release release;
        if (SHA256(data.data(), data.size(), ed25519_hash_buffer.data()) == nullptr) {
            throw std::runtime_error("SHA256 computation failed");
        }
    }
    return {"sha256", ed25519_hash_buffer};
}

// Legacy string overload
[[gnu::hot, gnu::flatten]] std::pair<std::string, std::vector<uint8_t>> compute_content_hash_fast(const std::string& event_json) {
    return compute_content_hash_fast(std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(event_json.data()), event_json.size()));
}

/**
 * Computes SHA256 content hash of Python dictionary after JSON canonicalization
 * @param event_dict Python dictionary to canonicalize and hash
 * @return Pair of algorithm name ("sha256") and 32-byte hash digest
 */
std::pair<std::string, std::vector<uint8_t>> compute_content_hash(const nb::dict& event_dict) {
    init_json_buffer(event_dict.size());
    
    py_to_canonical_json_fast(event_dict);
    
    std::string canonical_json(json_buffer.data(), json_ptr - json_buffer.data());
    return compute_content_hash_fast(canonical_json);
}

/**
 * Signs a Python dictionary object and adds signature to the result
 * @param json_dict Python dictionary to sign
 * @param signature_name Server name for the signature
 * @param signing_key_bytes 32-byte Ed25519 private key
 * @param key_id Key identifier (e.g., "ed25519:v1")
 * @return New dictionary with signature added
 */
nb::dict sign_json_object_fast(const nb::dict& json_dict, const std::string& signature_name, 
                               const std::vector<uint8_t>& signing_key_bytes, const std::string& key_id) {
    // Create unsigned copy for canonicalization
    nb::dict unsigned_dict;
    for (auto item : json_dict) {
        std::string_view key = nb::cast<std::string_view>(item.first);
        if (key != "signatures" && key != "unsigned") {
            unsigned_dict[std::string(key).c_str()] = item.second;
        }
    }
    
    // Direct streaming to canonical JSON
    init_json_buffer(unsigned_dict.size());
    
    py_to_canonical_json_fast(unsigned_dict);
    
    auto json_bytes = get_json_span();
    
    std::string signature_b64 = sign_json_fast(json_bytes, signing_key_bytes);
    
    // Create result dict with signature
    nb::dict result;
    for (auto item : json_dict) {
        std::string_view key = nb::cast<std::string_view>(item.first);
        result[std::string(key).c_str()] = item.second;
    }
    
    if (!result.contains("signatures")) {
        result["signatures"] = nb::dict();
    }
    nb::dict signatures = result["signatures"];
    if (!signatures.contains(signature_name.c_str())) {
        signatures[signature_name.c_str()] = nb::dict();
    }
    nb::dict server_sigs = signatures[signature_name.c_str()];
    server_sigs[key_id.c_str()] = signature_b64;
    
    return result;
}

/**
 * Verifies the digital signature on a signed Python dictionary object
 * @param json_dict Signed Python dictionary to verify
 * @param server_name Server name that should have signed the object
 * @param verify_key_bytes 32-byte Ed25519 public key for verification
 * @throws SignatureVerifyException if signature verification fails
 */
void verify_signed_json_fast(const nb::dict& json_dict, const std::string& server_name, const std::vector<uint8_t>& verify_key_bytes) {
    if (!json_dict.contains("signatures")) {
        throw SignatureVerifyException("Missing signatures");
    }
    
    nb::dict signatures = json_dict["signatures"];
    if (!signatures.contains(server_name.c_str())) {
        throw SignatureVerifyException("Missing server signature");
    }
    
    nb::dict server_sigs = signatures[server_name.c_str()];
    std::string signature_b64;
    
    // Find ed25519 signature
    for (auto item : server_sigs) {
        std::string_view key = nb::cast<std::string_view>(item.first);
        if (key.starts_with("ed25519:")) {
            signature_b64 = nb::cast<std::string>(item.second);
            break;
        }
    }
    
    if (signature_b64.empty()) {
        throw SignatureVerifyException("Missing ed25519 signature");
    }
    
    // Create unsigned copy for verification
    nb::dict unsigned_dict;
    for (auto item : json_dict) {
        std::string_view key = nb::cast<std::string_view>(item.first);
        if (key != "signatures" && key != "unsigned") {
            unsigned_dict[std::string(key).c_str()] = item.second;
        }
    }
    
    init_json_buffer(unsigned_dict.size());
    
    py_to_canonical_json_fast(unsigned_dict);
    
    auto json_bytes = get_json_span();
    
    if (!verify_signature_fast(json_bytes, signature_b64, verify_key_bytes)) {
        throw SignatureVerifyException("Signature verification failed");
    }
}

// Class method implementations
VerifyKey::VerifyKey(const std::vector<uint8_t>& bytes, const std::string& algorithm, const std::string& ver)
    : key_bytes(bytes), alg(algorithm), version(ver) {}

nb::bytes VerifyKey::encode() const {
    return nb::bytes(reinterpret_cast<const char*>(key_bytes.data()), key_bytes.size());
}

void VerifyKey::verify(nb::bytes message, nb::bytes signature) const {
    const char* msg_ptr = static_cast<const char*>(message.c_str());
    size_t msg_size = message.size();
    std::vector<uint8_t> message_bytes(msg_ptr, msg_ptr + msg_size);
    
    const char* sig_ptr = static_cast<const char*>(signature.c_str());
    size_t sig_size = signature.size();
    std::vector<uint8_t> signature_bytes(sig_ptr, sig_ptr + sig_size);
    
    std::string signature_b64 = base64_encode(signature_bytes);
    
    if (!verify_signature_fast(message_bytes, signature_b64, key_bytes)) {
        throw std::runtime_error("Signature verification failed");
    }
}

SigningKey::SigningKey(const std::vector<uint8_t>& bytes, const std::string& algorithm, const std::string& ver)
    : key_bytes(bytes), alg(algorithm), version(ver) {}

SigningKey SigningKey::generate() {
    return SigningKey(generate_signing_key());
}

nb::bytes SigningKey::encode() const {
    return nb::bytes(reinterpret_cast<const char*>(key_bytes.data()), key_bytes.size());
}

VerifyKey SigningKey::get_verify_key() const {
    return VerifyKey(::get_verify_key(key_bytes), alg, version);
}

Signature SigningKey::sign(nb::bytes message) const {
    const char* ptr = static_cast<const char*>(message.c_str());
    size_t size = message.size();
    std::vector<uint8_t> message_bytes(ptr, ptr + size);
    
    std::string signature_b64 = sign_json_fast(message_bytes, key_bytes);
    auto signature_bytes = base64_decode(signature_b64);
    nb::bytes sig_bytes = nb::bytes(reinterpret_cast<const char*>(signature_bytes.data()), signature_bytes.size());
    return Signature(sig_bytes);
}

SigningResult sign_json_with_info(const std::vector<uint8_t>& json_bytes, const std::vector<uint8_t>& signing_key_bytes, const std::string& version) {
    try {
        std::string signature_b64 = sign_json_fast(json_bytes, signing_key_bytes);
        return {signature_b64, "ed25519:" + version, "ed25519"};
    } catch (const std::exception& e) {
        return {"", "ed25519:" + version, "ed25519"};
    }
}

VerificationResult verify_signature_with_info(const std::vector<uint8_t>& json_bytes, const std::string& signature_b64, const std::vector<uint8_t>& verify_key_bytes) {
    try {
        bool valid = verify_signature_fast(json_bytes, signature_b64, verify_key_bytes);
        return {valid, std::nullopt, std::nullopt};
    } catch (const std::exception& e) {
        return {false, std::nullopt, std::nullopt};
    }
}
