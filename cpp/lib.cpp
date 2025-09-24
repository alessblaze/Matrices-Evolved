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


#include "global.h"

#include <cstring>
#include <span>
#include <type_traits>
#include <openssl/evp.h>
#include <openssl/rand.h>
#include <openssl/sha.h>
#include <openssl/base64.h>
#include <boost/json.hpp>
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <sstream>
#include <iostream>
#include <cstdlib>
#include <stdexcept>
#include <charconv>
#include <cmath>
#include <type_traits>

#include <Python.h>

// Module includes - restructured cpp directory (header-only)
#include "base64/encoders/include/base64-encoder.h"
#include "base64/decoders/include/base64-decoder.h"
#include "json/canonicalization.h"
#include "crypto/ed25519.h"

// Debug logging infrastructure

// Force recompilation - timestamp change 2




namespace nb = nanobind;
using namespace nb::literals;
namespace json = boost::json;

// Memory leak profiling
static bool leak_warnings_enabled = []() {
    const char* env = std::getenv("SYNAPSE_RUST_CRYPTO_LEAK_WARNINGS");
    return env && std::string(env) == "1";
}();

// SignatureVerifyException is now defined in debug.h



NB_MODULE(_event_signing_impl, m) {
    // Initialize debug system
    debug_init();
    
    // Enable leak warnings only when explicitly requested for profiling
    // Set SYNAPSE_RUST_CRYPTO_LEAK_WARNINGS=1 to enable
    nb::set_leak_warnings(leak_warnings_enabled);
    
    // AWS-LC doesn't require explicit initialization
    
    nb::exception<SignatureVerifyException>(m, "SignatureVerifyException");
    
    nb::class_<Signature>(m, "Signature")
        .def_rw("signature", &Signature::signature);
    
    nb::class_<SigningResult>(m, "SigningResult")
        .def_rw("signature", &SigningResult::signature)
        .def_rw("key_id", &SigningResult::key_id)
        .def_rw("algorithm", &SigningResult::algorithm);
    
    nb::class_<VerificationResult>(m, "VerificationResult")
        .def_rw("valid", &VerificationResult::valid)
        .def_rw("user_id", &VerificationResult::user_id)
        .def_rw("device_valid", &VerificationResult::device_valid);
    
    nb::class_<VerifyKey>(m, "VerifyKey")
        .def(nb::init<const std::vector<uint8_t>&, const std::string&, const std::string&>(),
             "bytes"_a, "alg"_a = "ed25519", "version"_a = "1")
        .def("encode", &VerifyKey::encode)
        .def("verify", &VerifyKey::verify)
        .def_rw("alg", &VerifyKey::alg)
        .def_rw("version", &VerifyKey::version);
    
    nb::class_<SigningKey>(m, "SigningKey")
        .def(nb::init<const std::vector<uint8_t>&, const std::string&, const std::string&>(),
             "bytes"_a, "alg"_a = "ed25519", "version"_a = "1")
        .def_static("generate", &SigningKey::generate)
        .def("encode", &SigningKey::encode)
        .def("get_verify_key", &SigningKey::get_verify_key)
        .def_prop_ro("verify_key", &SigningKey::get_verify_key)
        .def("sign", &SigningKey::sign)
        .def_rw("alg", &SigningKey::alg)
        .def_rw("version", &SigningKey::version);
    
    m.def("compute_content_hash_fast", [](const std::vector<uint8_t>& json_bytes) {
        auto result = compute_content_hash_fast(std::span<const uint8_t>(json_bytes));
        return std::make_pair(result.first, nb::bytes(reinterpret_cast<const char*>(result.second.data()), result.second.size()));
    });
    m.def("compute_content_hash", [](const nb::dict& event_dict) {
        auto result = compute_content_hash(event_dict);
        return std::make_pair(result.first, nb::bytes(reinterpret_cast<const char*>(result.second.data()), result.second.size()));
    });
    m.def("compute_event_reference_hash_fast", [](const std::string& event_json) {
        auto result = compute_content_hash_fast(event_json);
        return std::make_pair(result.first, nb::bytes(reinterpret_cast<const char*>(result.second.data()), result.second.size()));
    });
    m.def("compute_event_reference_hash", [](const nb::dict& event_dict) {
        auto result = compute_content_hash(event_dict);
        return std::make_pair(result.first, nb::bytes(reinterpret_cast<const char*>(result.second.data()), result.second.size()));
    });
    m.def("sign_json_fast", [](const std::vector<uint8_t>& json_bytes, const std::vector<uint8_t>& signing_key_bytes) {
        return sign_json_fast(std::span<const uint8_t>(json_bytes), signing_key_bytes);
    });
    m.def("sign_json_with_info", &sign_json_with_info);
    m.def("verify_signature_fast", [](const std::vector<uint8_t>& json_bytes, const std::string& signature_b64, const std::vector<uint8_t>& verify_key_bytes) {
        return verify_signature_fast(std::span<const uint8_t>(json_bytes), signature_b64, verify_key_bytes);
    });
    m.def("verify_signature_with_info", &verify_signature_with_info);
    m.def("sign_json_object_fast", &sign_json_object_fast);
    m.def("verify_signed_json_fast", &verify_signed_json_fast);
    m.def("get_verify_key", &get_verify_key);
    m.def("encode_base64_fast", [](nb::bytes data) {
        const char* ptr = static_cast<const char*>(data.c_str());
        size_t size = data.size();
        
        std::vector<uint8_t> vec_data(ptr, ptr + size);
        return nb::str(base64_encode(vec_data).c_str());
    });
    m.def("decode_base64_fast", [](const std::string& encoded) {
        auto result = base64_decode(encoded);
        return nb::bytes(reinterpret_cast<const char*>(result.data()), result.size());
    });
    m.def("encode_base64", [](const nb::object& data) {
        if (nb::isinstance<nb::bytes>(data)) {
            nb::bytes bytes_data = nb::cast<nb::bytes>(data);
            const char* ptr = static_cast<const char*>(bytes_data.c_str());
            size_t size = bytes_data.size();
            std::vector<uint8_t> vec_data(ptr, ptr + size);
            return nb::str(base64_encode(vec_data).c_str());
        } else {
            std::vector<uint8_t> vec_data = nb::cast<std::vector<uint8_t>>(data);
            return nb::str(base64_encode(vec_data).c_str());
        }
    });
    m.def("decode_base64", [](const std::string& encoded) {
        auto result = base64_decode(encoded);
        return nb::bytes(reinterpret_cast<const char*>(result.data()), result.size());
    });
    m.def("encode_verify_key_base64", [](const std::vector<uint8_t>& key_bytes) {
        return nb::str(base64_encode(key_bytes).c_str());
    });
    m.def("encode_signing_key_base64", [](const std::vector<uint8_t>& key_bytes) {
        return nb::str(base64_encode(key_bytes).c_str());
    });
    m.def("decode_verify_key_base64", [](const std::string& algorithm, const std::string& version, const std::string& key_base64) {
        if (algorithm != "ed25519") throw std::runtime_error("Unsupported algorithm");
        auto key_bytes = base64_decode(key_base64);
        if (key_bytes.size() != 32) throw std::runtime_error("Invalid key length");
        return key_bytes;
    });
    m.def("decode_signing_key_base64", [](const std::string& algorithm, const std::string& version, const std::string& key_base64) {
        if (algorithm != "ed25519") throw std::runtime_error("Unsupported algorithm");
        auto key_bytes = base64_decode(key_base64);
        if (key_bytes.size() != 32) throw std::runtime_error("Invalid key length");
        return key_bytes;
    });
    m.def("decode_verify_key_bytes_fast", [](std::string_view key_id, const std::vector<uint8_t>& key_bytes) {
        if (key_id.starts_with("ed25519:") && key_bytes.size() == 32) return key_bytes;
        throw std::runtime_error("Unsupported key type or invalid key length");
    });
    m.def("decode_verify_key_bytes", [](std::string_view key_id, const nb::object& key_data) -> VerifyKey {
        std::vector<uint8_t> key_bytes;
        
        if (nb::isinstance<nb::bytes>(key_data)) {
            nb::bytes bytes_data = nb::cast<nb::bytes>(key_data);
            const char* ptr = static_cast<const char*>(bytes_data.c_str());
            size_t size = bytes_data.size();
            key_bytes = std::vector<uint8_t>(ptr, ptr + size);
        } else {
            key_bytes = nb::cast<std::vector<uint8_t>>(key_data);
        }
        
        if (key_id.starts_with("ed25519:") && key_bytes.size() == 32) {
            size_t colon_pos = key_id.find(':');
            std::string version = (colon_pos != std::string::npos) ? std::string(key_id.substr(colon_pos + 1)) : "1";
            return VerifyKey(key_bytes, "ed25519", version);
        }
        throw std::runtime_error("Unsupported key type or invalid key length");
    });
    m.def("is_signing_algorithm_supported", [](std::string_view key_id) {
        return key_id.starts_with("ed25519:");
    });
    m.def("encode_canonical_json", [](const nb::dict& json_dict) {
        init_json_buffer(json_dict.size());
        
        // Serialize with potential GIL release in hot paths
        py_to_canonical_json_fast(json_dict);
        
        auto span = get_json_span();
        return nb::bytes(span.data(), span.size());
    });
    m.def("signature_ids", [](const nb::dict& json_dict, const std::string& signature_name) {
        std::vector<std::string> ids;
        if (json_dict.contains("signatures")) {
            nb::dict signatures = nb::cast<nb::dict>(json_dict["signatures"]);
            if (signatures.contains(signature_name.c_str())) {
                nb::dict server_sigs = nb::cast<nb::dict>(signatures[signature_name.c_str()]);
                for (auto item : server_sigs) {
                    std::string_view key = nb::cast<std::string_view>(item.first);
                    if (key.starts_with("ed25519:")) ids.push_back(std::string(key));
                }
            }
        }
        return ids;
    });
    // Alias functions to match Rust API exactly
    m.def("sign_json", [](const nb::dict& json_object, const std::string& signature_name, const nb::object& signing_key) {
        std::string alg = nb::cast<std::string>(signing_key.attr("alg"));
        std::string version = nb::cast<std::string>(signing_key.attr("version"));
        std::string key_id = alg + ":" + version;
        nb::bytes encoded_key = nb::cast<nb::bytes>(signing_key.attr("encode")());
        
        // Convert nb::bytes to std::vector<uint8_t>
        const char* ptr = static_cast<const char*>(encoded_key.c_str());
        size_t size = encoded_key.size();
        std::vector<uint8_t> signing_key_bytes(ptr, ptr + size);
        
        return sign_json_object_fast(json_object, signature_name, signing_key_bytes, key_id);
    });
    m.def("verify_signature", [](const std::vector<uint8_t>& json_bytes, const std::string& signature_b64, const std::vector<uint8_t>& verify_key_bytes) {
        return verify_signature_fast(std::span<const uint8_t>(json_bytes), signature_b64, verify_key_bytes);
    });
    m.def("verify_signed_json", [](const nb::dict& json_dict, const std::string& signature_name, const nb::object& verify_key) {
        DEBUG_LOG("verify_signed_json: signature_name=" + signature_name);
        
        // Extract key info like Rust version
        std::string alg, version;
        std::vector<uint8_t> verify_key_bytes;
        
        if (nb::isinstance<nb::bytes>(verify_key) || nb::isinstance<nb::list>(verify_key)) {
            verify_key_bytes = nb::cast<std::vector<uint8_t>>(verify_key);
            alg = "ed25519";
            version = "auto";
            DEBUG_LOG("Got raw bytes, length=" + std::to_string(verify_key_bytes.size()));
            
            // Extract version from signatures if available
            if (json_dict.contains("signatures")) {
                nb::dict signatures = json_dict["signatures"];
                if (signatures.contains(signature_name.c_str())) {
                    nb::dict server_sigs = signatures[signature_name.c_str()];
                    for (auto item : server_sigs) {
                        std::string_view key_id = nb::cast<std::string_view>(item.first);
                        if (key_id.starts_with("ed25519:")) {
                            size_t colon_pos = key_id.find(':');
                            if (colon_pos != std::string::npos) {
                                version = key_id.substr(colon_pos + 1);
                            }
                            break;
                        }
                    }
                }
            }
        } else {
            try {
                alg = nb::cast<std::string>(verify_key.attr("alg"));
                version = nb::cast<std::string>(verify_key.attr("version"));
                DEBUG_LOG("Got VerifyKey object: alg=" + alg + ", version=" + version);
                nb::bytes encoded_key = nb::cast<nb::bytes>(verify_key.attr("encode")());
                
                // Convert nb::bytes to std::vector<uint8_t>
                const char* ptr = static_cast<const char*>(encoded_key.c_str());
                size_t size = encoded_key.size();
                verify_key_bytes = std::vector<uint8_t>(ptr, ptr + size);
                DEBUG_LOG("Extracted key bytes, length=" + std::to_string(verify_key_bytes.size()));
            } catch (const std::exception& e) {
                DEBUG_LOG("Failed to extract key attributes, treating as raw bytes: " + std::string(e.what()));
                verify_key_bytes = nb::cast<std::vector<uint8_t>>(verify_key);
                alg = "ed25519";
                version = "auto";
            }
        }
        
        std::string key_id = alg + ":" + version;
        DEBUG_LOG("Using key_id: " + key_id);
        
        // Get signatures
        if (!json_dict.contains("signatures")) {
            DEBUG_LOG("ERROR: No signatures field found in JSON");
            throw SignatureVerifyException("No signatures on this object");
        }
        
        nb::dict signatures = json_dict["signatures"];
        if (!signatures.contains(signature_name.c_str())) {
            DEBUG_LOG("ERROR: Missing signature for server: " + signature_name);
            throw SignatureVerifyException("Missing signature for " + signature_name);
        }
        
        nb::dict server_sigs = signatures[signature_name.c_str()];
        std::string signature_b64;
        
        if (debug_enabled) {
            DEBUG_LOG("Available signature keys for " + signature_name + ":");
            for (auto item : server_sigs) {
                std::string_view available_key = nb::cast<std::string_view>(item.first);
                DEBUG_LOG("  - " + std::string(available_key));
            }
        }
        
        // Try exact key_id first, then fallback to any ed25519
        if (server_sigs.contains(key_id.c_str())) {
            signature_b64 = nb::cast<std::string>(server_sigs[key_id.c_str()]);
            DEBUG_LOG("Found exact key match: " + key_id + " -> " + signature_b64);
        } else {
            for (auto item : server_sigs) {
                std::string_view available_key = nb::cast<std::string_view>(item.first);
                if (available_key.starts_with("ed25519:")) {
                    signature_b64 = nb::cast<std::string>(item.second);
                    DEBUG_LOG("Using fallback key: " + std::string(available_key) + " -> " + signature_b64);
                    break;
                }
            }
        }
        
        if (signature_b64.empty()) {
            DEBUG_LOG("ERROR: No ed25519 signature found for " + signature_name + ", " + key_id);
            throw SignatureVerifyException("Missing signature for " + signature_name + ", " + key_id);
        }
        
        // Create unsigned copy
        nb::dict unsigned_dict;
        for (auto item : json_dict) {
            std::string_view key = nb::cast<std::string_view>(item.first);
            if (key != "signatures" && key != "unsigned") {
                unsigned_dict[std::string(key).c_str()] = item.second;
            }
        }
        
        DEBUG_LOG("Created unsigned dict with " + std::to_string(unsigned_dict.size()) + " keys");
        
        // Canonicalize and verify
        init_json_buffer(unsigned_dict.size());
        
        py_to_canonical_json_fast(unsigned_dict);
        
        auto json_bytes = get_json_span();
        
        if (debug_enabled) {
            std::string canonical_json(json_buffer.data(), json_ptr - json_buffer.data());
            DEBUG_LOG("Canonical JSON (" + std::to_string(canonical_json.size()) + " bytes): " + canonical_json.substr(0, 200) + (canonical_json.size() > 200 ? "..." : ""));
            DEBUG_LOG("Signature to verify: " + signature_b64);
            
            std::string hex;
            hex.reserve(verify_key_bytes.size() * 2);
            for (size_t i = 0; i < verify_key_bytes.size(); i++) {
                hex.push_back(hex_lut[verify_key_bytes[i] >> 4]);
                hex.push_back(hex_lut[verify_key_bytes[i] & 0x0F]);
            }
            DEBUG_LOG("Verify key bytes (full): " + hex);
        }
        
        bool verification_result = verify_signature_fast(json_bytes, signature_b64, verify_key_bytes);
        DEBUG_LOG("Verification result: " + std::string(verification_result ? "SUCCESS" : "FAILED"));
        
        if (!verification_result) {
            DEBUG_LOG("ERROR: Signature verification failed for " + signature_name);
            throw SignatureVerifyException("Unable to verify signature for " + signature_name);
        }
        
        DEBUG_LOG("Signature verification completed successfully");
    });
    // Key management functions with version parameter (ignores version for compatibility)
    m.def("generate_signing_key", [](const std::string& version) { return generate_signing_key(); });
    m.def("read_signing_keys", [](const nb::object& input_data) {
        DEBUG_LOG("read_signing_keys called");
        
        // Handle different input types like Python signedjson
        std::string content;
        if (nb::isinstance<nb::str>(input_data)) {
            // Handle string input
            content = nb::cast<std::string>(input_data);
            DEBUG_LOG("Got string input, length: " + std::to_string(content.length()));
        } else if (nb::isinstance<nb::list>(input_data)) {
            // Handle list input (like config does)
            auto py_list = nb::cast<nb::list>(input_data);
            DEBUG_LOG("Got list input with " + std::to_string(py_list.size()) + " items");
            std::vector<std::string> lines;
            for (auto item : py_list) {
                std::string line = nb::cast<std::string>(item);
                lines.push_back(line);
                DEBUG_LOG("List item: \"" + line + "\"");
            }
            // Join with newlines - vectorized for large arrays
            if (lines.size() >= 16) {
                DEBUG_LOG("Using optimized string joining for " + std::to_string(lines.size()) + " lines");
                // Pre-calculate total size to avoid reallocations
                size_t total_size = 0;
                for (const auto& line : lines) {
                    total_size += line.size() + 1; // +1 for newline
                }
                content.reserve(total_size);
            }
            
            for (size_t i = 0; i < lines.size(); ++i) {
                if (i > 0) content += "\n";
                content += lines[i];
            }
            DEBUG_LOG("Joined list content: \"" + content + "\"");
        } else {
            // Try to call read() method for file-like objects
            try {
                nb::object content_obj = input_data.attr("read")();
                content = nb::cast<std::string>(content_obj);
                DEBUG_LOG("Read from file-like object, length: " + std::to_string(content.length()));
            } catch (...) {
                // Fallback to string conversion
                content = nb::cast<std::string>(nb::str(input_data));
                DEBUG_LOG("Fallback string conversion, length: " + std::to_string(content.length()));
            }
        }
        
        DEBUG_LOG("Content to parse: \"" + content + "\"");
        
        // Parse signing keys from content
        nb::list signing_keys;
        
        std::istringstream stream(content);
        std::string line;
        int line_num = 0;
        while (std::getline(stream, line)) {
            line_num++;
            DEBUG_LOG("Processing line " + std::to_string(line_num) + ": \"" + line + "\"");
            
            // Trim whitespace
            line.erase(0, line.find_first_not_of(" \t\r\n"));
            line.erase(line.find_last_not_of(" \t\r\n") + 1);
            
            if (line.empty() || line[0] == '#') {
                DEBUG_LOG("Skipping empty or comment line");
                continue;
            }
            
            // Expected format: "ed25519 version base64_key"
            std::istringstream line_stream(line);
            std::string algorithm, version, key_b64;
            if (line_stream >> algorithm >> version >> key_b64) {
                DEBUG_LOG("Parsed: algorithm=" + algorithm + ", version=" + version + ", key_b64=" + key_b64);
                if (algorithm == "ed25519" && !version.empty() && !key_b64.empty()) {
                    try {
                        auto key_bytes = base64_decode(key_b64);
                        DEBUG_LOG("Decoded key bytes, length: " + std::to_string(key_bytes.size()));
                        if (key_bytes.size() == 32) {
                            // Create SigningKey object
                            SigningKey key(key_bytes, "ed25519", version);
                            signing_keys.append(nb::cast(key));
                            DEBUG_LOG("Added signing key with version: " + version);
                        } else {
                            DEBUG_LOG("Invalid key length: " + std::to_string(key_bytes.size()));
                        }
                    } catch (const std::exception& e) {
                        DEBUG_LOG("Failed to decode key: " + std::string(e.what()));
                        continue;
                    }
                } else {
                    DEBUG_LOG("Invalid key format or algorithm");
                }
            } else {
                DEBUG_LOG("Failed to parse line format");
            }
        }
        
        DEBUG_LOG("Returning " + std::to_string(signing_keys.size()) + " signing keys");
        return signing_keys;
    });
    m.def("read_old_signing_keys", [](const nb::object& stream_content) {
        return nb::list();
    });
    m.def("write_signing_keys", [](const std::vector<std::pair<std::string, std::vector<uint8_t>>>& keys) {
        std::string output;
        for (const auto& [version, key_bytes] : keys) {
            if (key_bytes.size() == 32) {
                std::string key_b64 = base64_encode(key_bytes);
                output += "ed25519 " + version + " " + key_b64 + "\n";
            }
        }
        return output;
    });
}