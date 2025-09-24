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
use pyo3::prelude::*;
use pyo3::types::PyDict;
use aws_lc_rs::{digest, signature};
use base64::{Engine as _, engine::general_purpose::STANDARD_NO_PAD};
use serde_json;
use std::sync::LazyLock;
use std::cell::RefCell;
use pythonize::{pythonize, PythonizeError};
use serde::{Deserialize, Serialize};

// Cached debug flag for optimal performance
static CRYPTO_DEBUG_ENABLED: LazyLock<bool> = LazyLock::new(|| std::env::var("SYNAPSE_RUST_CRYPTO_DEBUG").unwrap_or_default() == "1");

// Thread-local key cache to avoid repeated key pair creation
thread_local! {
    static CACHED_SIGNING_KEY: RefCell<Option<Vec<u8>>> = RefCell::new(None);
    static CACHED_KEY_PAIR: RefCell<Option<signature::Ed25519KeyPair>> = RefCell::new(None);
}

/// Verification result with automatic Python conversion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationResult {
    pub valid: bool,
    pub user_id: Option<String>,
    pub device_valid: Option<bool>,
}

impl<'py> IntoPyObject<'py> for VerificationResult {
    type Target = PyAny;
    type Output = Bound<'py, Self::Target>;
    type Error = PythonizeError;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        pythonize(py, &self)
    }
}

/// Signing result with automatic Python conversion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SigningResult {
    pub signature: String,
    pub key_id: String,
    pub algorithm: String,
}

impl<'py> IntoPyObject<'py> for SigningResult {
    type Target = PyAny;
    type Output = Bound<'py, Self::Target>;
    type Error = PythonizeError;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        pythonize(py, &self)
    }
}


/// Escape string for JSON while preserving UTF-8 (like Python canonicaljson)
/// Only escapes control characters and quotes, not Unicode
fn escape_json_string(s: &str) -> String {
    let mut result = String::with_capacity(s.len() + 2);
    escape_json_string_into(&mut result, s);
    result
}

/// Escape string directly into existing buffer for better performance
fn escape_json_string_into(result: &mut String, s: &str) {
    result.push('"');
    
    for ch in s.chars() {
        match ch {
            '"' => result.push_str("\\\""),
            '\\' => result.push_str("\\\\"),
            '\u{08}' => result.push_str("\\b"),
            '\u{0C}' => result.push_str("\\f"),
            '\n' => result.push_str("\\n"),
            '\r' => result.push_str("\\r"),
            '\t' => result.push_str("\\t"),
            ch if ch.is_control() => {
                use std::fmt::Write;
                let _ = write!(result, "\\u{:04x}", ch as u32);
            }
            ch => result.push(ch),
        }
    }
    
    result.push('"');
}

/// Estimate JSON size for pre-allocation
fn estimate_json_size(value: &serde_json::Value) -> usize {
    match value {
        serde_json::Value::Null => 4,
        serde_json::Value::Bool(_) => 5,
        serde_json::Value::Number(_) => 20,
        serde_json::Value::String(s) => s.len() + 10,
        serde_json::Value::Array(arr) => {
            2 + arr.iter().map(estimate_json_size).sum::<usize>() + arr.len().saturating_sub(1)
        }
        serde_json::Value::Object(obj) => {
            2 + obj.iter().map(|(k, v)| k.len() + 10 + estimate_json_size(v)).sum::<usize>() + obj.len().saturating_sub(1)
        }
    }
}

/// Canonicalize JSON to match Python's canonicaljson library exactly
/// - Keys are sorted alphabetically
/// - No whitespace
/// - Deterministic float representation using ryu
/// - UTF-8 preservation in strings
/// - Rejects non-finite floats
#[inline(always)]
fn canonicalize_json(value: &serde_json::Value) -> Result<String, String> {
    let mut result = String::with_capacity(estimate_json_size(value));
    canonicalize_json_into(&mut result, value)?;
    Ok(result)
}

#[inline(always)]
fn canonicalize_json_into(result: &mut String, value: &serde_json::Value) -> Result<(), String> {
    match value {
        serde_json::Value::Null => result.push_str("null"),
        serde_json::Value::Bool(b) => result.push_str(&b.to_string()),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                result.push_str(&i.to_string());
            } else if let Some(u) = n.as_u64() {
                result.push_str(&u.to_string());
            } else if let Some(f) = n.as_f64() {
                if !f.is_finite() {
                    return Err("Non-finite floats (NaN, Infinity) are not allowed in canonical JSON".to_string());
                }
                let mut buf = ryu::Buffer::new();
                let formatted = buf.format(f);
                if !formatted.contains('.') && !formatted.contains('e') {
                    result.push_str(formatted);
                    result.push_str(".0");
                } else {
                    result.push_str(formatted);
                }
            } else {
                return Err("Invalid number format".to_string());
            }
        }
        serde_json::Value::String(s) => {
            escape_json_string_into(result, s);
        }
        serde_json::Value::Array(arr) => {
            result.push('[');
            for (i, item) in arr.iter().enumerate() {
                if i > 0 {
                    result.push(',');
                }
                canonicalize_json_into(result, item)?;
            }
            result.push(']');
        }
        serde_json::Value::Object(obj) => {
            let mut sorted_keys: Vec<_> = obj.keys().collect();
            sorted_keys.sort_unstable();
            
            result.push('{');
            for (i, key) in sorted_keys.iter().enumerate() {
                if i > 0 {
                    result.push(',');
                }
                escape_json_string_into(result, key);
                result.push(':');
                if let Some(value) = obj.get(*key) {
                    canonicalize_json_into(result, value)?;
                } else {
                    result.push_str("null");
                }
            }
            result.push('}');
        }
    }
    Ok(())
}

#[pyfunction]
pub fn compute_content_hash_fast(py: Python, event_json: String) -> PyResult<(String, Py<pyo3::types::PyBytes>)> {
    let hash_bytes = digest::digest(&digest::SHA256, event_json.as_bytes());
    Ok(("sha256".to_string(), pyo3::types::PyBytes::new(py, hash_bytes.as_ref()).into()))
}

#[pyfunction]
pub fn compute_event_reference_hash_fast(py: Python, event_json: String) -> PyResult<(String, Py<pyo3::types::PyBytes>)> {
    let hash_bytes = digest::digest(&digest::SHA256, event_json.as_bytes());
    Ok(("sha256".to_string(), pyo3::types::PyBytes::new(py, hash_bytes.as_ref()).into()))
}

#[pyfunction]
pub fn sign_json_fast(
    json_bytes: Vec<u8>,
    signing_key_bytes: Vec<u8>
) -> PyResult<String> {
    // Validate key length first
    if signing_key_bytes.len() != 32 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("Invalid signing key length: expected 32 bytes, got {}", signing_key_bytes.len())
        ));
    }
    
    // Perform signing inside thread_local closure to avoid borrow checker issues
    CACHED_SIGNING_KEY.with(|cached_key| {
        CACHED_KEY_PAIR.with(|cached_pair| {
            let mut cached_key = cached_key.borrow_mut();
            let mut cached_pair = cached_pair.borrow_mut();
            
            // Check if we have a cached key pair for this key
            if cached_key.as_ref() != Some(&signing_key_bytes) || cached_pair.is_none() {
                if *CRYPTO_DEBUG_ENABLED {
                    println!("DEBUG Rust crypto: Key cache miss - creating new Ed25519KeyPair");
                }
                // Create new key pair and cache it
                let new_key_pair = signature::Ed25519KeyPair::from_seed_unchecked(&signing_key_bytes)
                    .map_err(|_| PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid signing key"))?;
                
                *cached_key = Some(signing_key_bytes);
                *cached_pair = Some(new_key_pair);
            } else {
                if *CRYPTO_DEBUG_ENABLED {
                    println!("DEBUG Rust crypto: Key cache hit - reusing cached Ed25519KeyPair");
                }
            }
            
            // Sign with cached key pair
            let signature = cached_pair.as_ref().unwrap().sign(&json_bytes);
            let signature_b64 = STANDARD_NO_PAD.encode(signature.as_ref());
            
            Ok(signature_b64)
        })
    })
}

#[pyfunction]
pub fn sign_json_with_info(
    py: Python,
    json_bytes: Vec<u8>,
    signing_key_bytes: Vec<u8>,
    version: String
) -> PyResult<PyObject> {
    // Validate key length first
    if signing_key_bytes.len() != 32 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("Invalid signing key length: expected 32 bytes, got {}", signing_key_bytes.len())
        ));
    }
    
    // Perform signing inside thread_local closure
    let signature_b64: String = CACHED_SIGNING_KEY.with(|cached_key| {
        CACHED_KEY_PAIR.with(|cached_pair| -> PyResult<String> {
            let mut cached_key = cached_key.borrow_mut();
            let mut cached_pair = cached_pair.borrow_mut();
            
            if cached_key.as_ref() != Some(&signing_key_bytes) || cached_pair.is_none() {
                if *CRYPTO_DEBUG_ENABLED {
                    println!("DEBUG Rust crypto: Key cache miss in sign_json_with_info - creating new Ed25519KeyPair");
                }
                let new_key_pair = signature::Ed25519KeyPair::from_seed_unchecked(&signing_key_bytes)
                    .map_err(|_| PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid signing key"))?;
                
                *cached_key = Some(signing_key_bytes);
                *cached_pair = Some(new_key_pair);
            } else {
                if *CRYPTO_DEBUG_ENABLED {
                    println!("DEBUG Rust crypto: Key cache hit in sign_json_with_info - reusing cached Ed25519KeyPair");
                }
            }
            
            let signature = cached_pair.as_ref().unwrap().sign(&json_bytes);
            Ok(STANDARD_NO_PAD.encode(signature.as_ref()))
        })
    })?;
    
    let result = SigningResult {
        signature: signature_b64,
        key_id: format!("ed25519:{}", version),
        algorithm: "ed25519".to_string(),
    };
    
    // Automatic conversion to Python dict via serde+pythonize
    result.into_pyobject(py).map(|bound| bound.unbind())
        .map_err(|_| PyErr::new::<pyo3::exceptions::PyValueError, _>("Failed to convert result"))
}

#[pyfunction]
pub fn sign_json_object_fast(
    py: Python,
    json_dict: &Bound<'_, PyDict>,
    signature_name: String,
    signing_key_bytes: Vec<u8>,
    key_id: String
) -> PyResult<PyObject> {
    // Validate key length first
    if signing_key_bytes.len() != 32 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("Invalid signing key length: expected 32 bytes, got {}", signing_key_bytes.len())
        ));
    }
    
    // Convert PyDict to serde_json::Value
    let mut json_value: serde_json::Value = pythonize::depythonize(json_dict)
        .map_err(|_| PyErr::new::<pyo3::exceptions::PyValueError, _>("Failed to convert Python dict"))?;
    
    // Create canonical JSON without signatures (avoid clone)
    let canonical_json = {
        let mut temp_value = json_value.clone();
        if let Some(obj) = temp_value.as_object_mut() {
            obj.remove("signatures");
        }
        canonicalize_json(&temp_value)
    }
        .map_err(|_| PyErr::new::<pyo3::exceptions::PyValueError, _>("JSON canonicalization failed"))?;
    
    // Sign using cached key pair
    let signature_b64: String = CACHED_SIGNING_KEY.with(|cached_key| {
        CACHED_KEY_PAIR.with(|cached_pair| -> PyResult<String> {
            let mut cached_key = cached_key.borrow_mut();
            let mut cached_pair = cached_pair.borrow_mut();
            
            if cached_key.as_ref() != Some(&signing_key_bytes) || cached_pair.is_none() {
                if *CRYPTO_DEBUG_ENABLED {
                    println!("DEBUG Rust crypto: Key cache miss in sign_json_object_fast - creating new Ed25519KeyPair");
                }
                let new_key_pair = signature::Ed25519KeyPair::from_seed_unchecked(&signing_key_bytes)
                    .map_err(|_| PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid signing key"))?;
                
                *cached_key = Some(signing_key_bytes);
                *cached_pair = Some(new_key_pair);
            } else {
                if *CRYPTO_DEBUG_ENABLED {
                    println!("DEBUG Rust crypto: Key cache hit in sign_json_object_fast - reusing cached Ed25519KeyPair");
                }
            }
            
            let signature = cached_pair.as_ref().unwrap().sign(canonical_json.as_bytes());
            Ok(STANDARD_NO_PAD.encode(signature.as_ref()))
        })
    })?;
    
    // Add signature to JSON object
    if let Some(obj) = json_value.as_object_mut() {
        let signatures = obj.entry("signatures").or_insert(serde_json::Value::Object(serde_json::Map::new()));
        if let Some(sig_obj) = signatures.as_object_mut() {
            let server_sigs = sig_obj.entry(&signature_name).or_insert(serde_json::Value::Object(serde_json::Map::new()));
            if let Some(server_sig_obj) = server_sigs.as_object_mut() {
                server_sig_obj.insert(key_id, serde_json::Value::String(signature_b64));
            }
        }
    }
    
    // Convert back to Python dict
    pythonize::pythonize(py, &json_value)
        .map_err(|_| PyErr::new::<pyo3::exceptions::PyValueError, _>("Failed to convert to Python dict"))
        .map(|bound| bound.unbind())
}

#[pyfunction]
pub fn verify_signature_fast(
    json_bytes: Vec<u8>,
    signature_b64: String,
    verify_key_bytes: Vec<u8>
) -> PyResult<bool> {
    let public_key = signature::UnparsedPublicKey::new(&signature::ED25519, &verify_key_bytes);
    
    // Try both standard and URL-safe base64 decoding
    let signature_bytes = STANDARD_NO_PAD.decode(&signature_b64)
        .or_else(|_| base64::engine::general_purpose::URL_SAFE_NO_PAD.decode(&signature_b64))
        .map_err(|_| PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid signature encoding"))?;
    
    Ok(public_key.verify(&json_bytes, &signature_bytes).is_ok())
}

#[pyfunction]
pub fn verify_signature_with_info(
    py: Python,
    json_bytes: Vec<u8>,
    signature_b64: String,
    verify_key_bytes: Vec<u8>
) -> PyResult<PyObject> {
    let public_key = signature::UnparsedPublicKey::new(&signature::ED25519, &verify_key_bytes);
    
    let signature_bytes = STANDARD_NO_PAD.decode(&signature_b64)
        .or_else(|_| base64::engine::general_purpose::URL_SAFE_NO_PAD.decode(&signature_b64))
        .map_err(|_| PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid signature encoding"))?;
    
    let valid = public_key.verify(&json_bytes, &signature_bytes).is_ok();
    
    let result = VerificationResult {
        valid,
        user_id: None,
        device_valid: None,
    };
    
    // Automatic conversion to Python dict via serde+pythonize
    result.into_pyobject(py).map(|bound| bound.unbind())
        .map_err(|_| PyErr::new::<pyo3::exceptions::PyValueError, _>("Failed to convert result"))
}

#[pyfunction]
pub fn encode_base64_fast(py: Python, data: &[u8]) -> PyResult<Py<pyo3::types::PyString>> {
    let encoded = STANDARD_NO_PAD.encode(data);
    Ok(pyo3::types::PyString::new(py, &encoded).into())
}

#[pyfunction]
pub fn decode_base64_fast(py: Python, data: &str) -> PyResult<Py<pyo3::types::PyBytes>> {
    // Try both standard and URL-safe base64 decoding
    let decoded = STANDARD_NO_PAD.decode(data)
        .or_else(|_| base64::engine::general_purpose::URL_SAFE_NO_PAD.decode(data))
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Base64 decode error: {}", e)))?;
    Ok(pyo3::types::PyBytes::new(py, &decoded).into())
}

#[pyfunction]
pub fn verify_signed_json_fast(
    json_dict: &Bound<'_, PyDict>,
    server_name: String,
    verify_key_bytes: Vec<u8>
) -> PyResult<()> {
    let public_key = signature::UnparsedPublicKey::new(&signature::ED25519, &verify_key_bytes);
    
    // Convert PyDict to serde_json::Value
    let json_value: serde_json::Value = pythonize::depythonize(json_dict)
        .map_err(|_| PyErr::new::<pyo3::exceptions::PyValueError, _>("Failed to convert Python dict"))?;
    
    let signatures = json_value.get("signatures")
        .and_then(|s| s.as_object())
        .ok_or_else(|| super::SignatureVerifyException::new_err("Missing signatures"))?;
    
    let server_sigs = signatures.get(&server_name)
        .and_then(|s| s.as_object())
        .ok_or_else(|| super::SignatureVerifyException::new_err("Missing server signature"))?;
    
    // Get first ed25519 signature
    let signature_b64 = server_sigs.iter()
        .find(|(key, _)| key.starts_with("ed25519:"))
        .and_then(|(_, sig)| sig.as_str())
        .ok_or_else(|| super::SignatureVerifyException::new_err("Missing ed25519 signature"))?;
    
    // Remove signatures from JSON for verification
    let mut unsigned_json = json_value.clone();
    if let Some(obj) = unsigned_json.as_object_mut() {
        obj.remove("signatures");
        obj.remove("unsigned");
    }
    
    // Use canonical JSON encoding
    let canonical_json = canonicalize_json(&unsigned_json)
        .map_err(|_| PyErr::new::<pyo3::exceptions::PyValueError, _>("JSON canonicalization failed"))?;
    
    let signature_bytes = base64::engine::general_purpose::STANDARD_NO_PAD.decode(signature_b64)
        .map_err(|_| super::SignatureVerifyException::new_err("Invalid signature encoding"))?;
    
    public_key.verify(canonical_json.as_bytes(), &signature_bytes)
        .map_err(|_| super::SignatureVerifyException::new_err("Signature verification failed"))?;
    
    Ok(())
}

#[pyfunction]
pub fn decode_verify_key_bytes_fast(
    key_id: String,
    key_bytes: Vec<u8>
) -> PyResult<Vec<u8>> {
    // For Ed25519 keys, just return the raw key bytes
    // The key_id format is "ed25519:version" 
    if key_id.starts_with("ed25519:") && key_bytes.len() == 32 {
        Ok(key_bytes)
    } else {
        Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Unsupported key type or invalid key length"))
    }
}

// Standard name functions - optimized versions
#[pyfunction]
pub fn compute_content_hash(py: Python, event_dict: &Bound<'_, PyDict>) -> PyResult<(String, Py<pyo3::types::PyBytes>)> {
    // Convert PyDict to serde_json::Value
    let json_value: serde_json::Value = pythonize::depythonize(event_dict)
        .map_err(|_| PyErr::new::<pyo3::exceptions::PyValueError, _>("Failed to convert Python dict"))?;
    
    // Create canonical JSON
    let canonical_json = canonicalize_json(&json_value)
        .map_err(|_| PyErr::new::<pyo3::exceptions::PyValueError, _>("JSON canonicalization failed"))?;
    
    let hash_bytes = digest::digest(&digest::SHA256, canonical_json.as_bytes());
    Ok(("sha256".to_string(), pyo3::types::PyBytes::new(py, hash_bytes.as_ref()).into()))
}

#[pyfunction]
pub fn compute_event_reference_hash(py: Python, event_dict: &Bound<'_, PyDict>) -> PyResult<(String, Py<pyo3::types::PyBytes>)> {
    // Convert PyDict to serde_json::Value
    let json_value: serde_json::Value = pythonize::depythonize(event_dict)
        .map_err(|_| PyErr::new::<pyo3::exceptions::PyValueError, _>("Failed to convert Python dict"))?;
    
    // Create canonical JSON
    let canonical_json = canonicalize_json(&json_value)
        .map_err(|_| PyErr::new::<pyo3::exceptions::PyValueError, _>("JSON canonicalization failed"))?;
    
    let hash_bytes = digest::digest(&digest::SHA256, canonical_json.as_bytes());
    Ok(("sha256".to_string(), pyo3::types::PyBytes::new(py, hash_bytes.as_ref()).into()))
}

#[pyfunction]
pub fn sign_json(py: Python, json_object: &Bound<'_, PyDict>, signature_name: String, signing_key: PyObject) -> PyResult<PyObject> {
    // Extract key components from signing_key object
    let alg = signing_key.getattr(py, "alg")?.extract::<String>(py)?;
    let version = signing_key.getattr(py, "version")?.extract::<String>(py)?;
    let key_id = format!("{}:{}", alg, version);
    
    // Get signing key bytes
    let key_encode_method = signing_key.getattr(py, "encode")?;
    let encoded_key = key_encode_method.call0(py)?;
    let signing_key_bytes = encoded_key.extract::<Vec<u8>>(py)?;
    
    if *CRYPTO_DEBUG_ENABLED {
        println!("DEBUG Rust crypto: sign_json called with key_id: {}", key_id);
    }
    
    // Do everything in Rust - avoid Python update() call
    sign_json_object_fast(py, json_object, signature_name, signing_key_bytes, key_id)
}

#[pyfunction]
pub fn verify_signature(json_bytes: Vec<u8>, signature_b64: String, verify_key_bytes: Vec<u8>) -> PyResult<bool> {
    verify_signature_fast(json_bytes, signature_b64, verify_key_bytes)
}

#[pyfunction]
pub fn encode_base64(py: Python, data: Vec<u8>) -> PyResult<Py<pyo3::types::PyString>> {
    encode_base64_fast(py, &data)
}

#[pyfunction]
pub fn decode_base64(py: Python, data: String) -> PyResult<Py<pyo3::types::PyBytes>> {
    decode_base64_fast(py, &data)
}

#[pyfunction]
pub fn verify_signed_json(py: Python, json_dict: &Bound<'_, PyDict>, signature_name: String, verify_key: PyObject) -> PyResult<()> {
    let debug_enabled = *CRYPTO_DEBUG_ENABLED;
    
    // Convert once and reuse
    let json_value: serde_json::Value = pythonize::depythonize(json_dict)
        .map_err(|_| PyErr::new::<pyo3::exceptions::PyValueError, _>("Failed to convert Python dict"))?;
    
    if debug_enabled {
        println!("DEBUG verify_signed_json: signature_name={}", signature_name);
        println!("DEBUG verify_key type: {:?}", verify_key.bind(py).get_type().name());
        let json_module = py.import("json")?;
        let json_str_obj = json_module.call_method1("dumps", (json_dict,))?;
        let json_str = json_str_obj.extract::<String>()?;
        println!("DEBUG JSON object: {}", json_str);
        if let Some(sigs) = json_dict.get_item("signatures")? {
            if let Ok(sigs_str) = json_module.call_method1("dumps", (sigs,)) {
                println!("DEBUG signatures: {}", sigs_str.extract::<String>()?);
            }
        }
    }
    let (alg, version, verify_key_bytes) = if let Ok(bytes_data) = verify_key.extract::<Vec<u8>>(py) {
        if debug_enabled {
            println!("DEBUG: Got raw bytes, length={}", bytes_data.len());
        }
        let mut found_version = "auto".to_string();
        if let Some(signatures) = json_value.get("signatures") {
            if let Some(server_sigs) = signatures.get(&signature_name) {
                if let Some(sig_obj) = server_sigs.as_object() {
                    for key_id in sig_obj.keys() {
                        if key_id.starts_with("ed25519:") {
                            found_version = key_id.split(':').nth(1).unwrap_or("auto").to_string();
                            if debug_enabled {
                                println!("DEBUG: Using version from signatures: {}", found_version);
                            }
                            break;
                        }
                    }
                }
            }
        }
        ("ed25519".to_string(), found_version, bytes_data)
    } else {
        if debug_enabled {
            println!("DEBUG: Got VerifyKey object");
        }
        let alg = verify_key.getattr(py, "alg")?.extract::<String>(py)?;
        let version = verify_key.getattr(py, "version")?.extract::<String>(py)?;
        if debug_enabled {
            println!("DEBUG: alg={}, version={}", alg, version);
        }
        let encode_method = verify_key.getattr(py, "encode")?;
        let encoded_key = encode_method.call0(py)?;
        let key_bytes = encoded_key.extract::<Vec<u8>>(py)?;
        (alg, version, key_bytes)
    };
    let key_id = format!("{}:{}", alg, version);
    let signatures = json_value.get("signatures")
        .and_then(|s| s.as_object())
        .ok_or_else(|| super::SignatureVerifyException::new_err("No signatures on this object"))?;
    let server_sigs = signatures.get(&signature_name)
        .and_then(|s| s.as_object())
        .ok_or_else(|| super::SignatureVerifyException::new_err(
            format!("Missing signature for {}", signature_name)
        ))?;
    let keys_vec: Vec<String> = server_sigs.keys().cloned().collect();
    if debug_enabled {
        println!("DEBUG: Available signature keys: {:?}", keys_vec);
    }
    let signature_b64 = if let Some(sig) = server_sigs.get(&key_id).and_then(|v| v.as_str()) {
        sig
    } else {
        let mut fallback_sig = None;
        for available_key in &keys_vec {
            if available_key.starts_with("ed25519:") {
                if let Some(sig) = server_sigs.get(available_key).and_then(|v| v.as_str()) {
                    if debug_enabled {
                        println!("DEBUG: Using fallback signature key: {}", available_key);
                    }
                    fallback_sig = Some(sig);
                    break;
                }
            }
        }
        fallback_sig.ok_or_else(|| super::SignatureVerifyException::new_err(
            format!("Missing signature for {}, {}", signature_name, key_id)
        ))?
    };
    let signature_bytes = STANDARD_NO_PAD.decode(&signature_b64)
        .map_err(|_| super::SignatureVerifyException::new_err(
            format!("Invalid signature base64 for {}, {}", signature_name, key_id)
        ))?;
    let mut unsigned_json = json_value.clone();
    if let Some(obj) = unsigned_json.as_object_mut() {
        obj.remove("signatures");
        obj.remove("unsigned");
    }
    let canonical_json = canonicalize_json(&unsigned_json)
        .map_err(|_| PyErr::new::<pyo3::exceptions::PyValueError, _>("JSON canonicalization failed"))?;
    let message = canonical_json.as_bytes();
    if debug_enabled {
        println!("DEBUG Python canonical JSON: {}", String::from_utf8_lossy(&message));
        println!("DEBUG signature_b64: {}", signature_b64);
        println!("DEBUG signature_bytes length: {}", signature_bytes.len());
        println!("DEBUG verify_key_bytes length: {}", verify_key_bytes.len());
    }
    let public_key = signature::UnparsedPublicKey::new(&signature::ED25519, &verify_key_bytes);
    if debug_enabled {
        println!("DEBUG: About to verify signature with message length: {}", message.len());
    }
    match public_key.verify(&message, &signature_bytes) {
        Ok(_) => {
            if debug_enabled {
                println!("DEBUG: Signature verification succeeded!");
            }
        }
        Err(e) => {
            if debug_enabled {
                println!("DEBUG: Signature verification failed: {:?}", e);
            }
            return Err(super::SignatureVerifyException::new_err(
                format!("Unable to verify signature for {}: {:?}", signature_name, e)
            ));
        }
    }
    Ok(())
}

#[pyfunction]
pub fn decode_verify_key_bytes(key_id: String, key_bytes: Vec<u8>) -> PyResult<Vec<u8>> {
    decode_verify_key_bytes_fast(key_id, key_bytes)
}

// Additional signedjson compatibility functions
#[pyfunction]
pub fn encode_canonical_json(py: Python, json_dict: &Bound<'_, PyDict>) -> PyResult<Vec<u8>> {
    // Direct conversion from PyDict to serde_json::Value
    let json_value: serde_json::Value = pythonize::depythonize(json_dict)
        .map_err(|_| PyErr::new::<pyo3::exceptions::PyValueError, _>("Failed to convert Python dict"))?;
    
    let canonical = canonicalize_json(&json_value)
        .map_err(|_| PyErr::new::<pyo3::exceptions::PyValueError, _>("JSON canonicalization failed"))?;
    
    Ok(canonical.into_bytes())
}

#[pyfunction]
pub fn signature_ids(
    _py: Python,
    json_dict: &Bound<'_, PyDict>,
    signature_name: String
) -> PyResult<Vec<String>> {
    let json_value: serde_json::Value = pythonize::depythonize(json_dict)
        .map_err(|_| PyErr::new::<pyo3::exceptions::PyValueError, _>("Failed to convert Python dict"))?;
    let mut ids = Vec::new();
    if let Some(signatures) = json_value.get("signatures") {
        if let Some(server_sigs) = signatures.get(&signature_name) {
            if let Some(sig_obj) = server_sigs.as_object() {
                for key in sig_obj.keys() {
                    if key.starts_with("ed25519:") {
                        ids.push(key.clone());
                    }
                }
            }
        }
    }
    Ok(ids)
}

// signedjson.key module functions
#[pyfunction]
pub fn generate_signing_key(_version: String) -> PyResult<Vec<u8>> {
    use aws_lc_rs::rand::{SystemRandom, SecureRandom};
    let rng = SystemRandom::new();
    let mut seed = [0u8; 32];
    rng.fill(&mut seed)
        .map_err(|_| PyErr::new::<pyo3::exceptions::PyValueError, _>("Random generation failed"))?;
    Ok(seed.to_vec())
}

#[pyfunction]
pub fn get_verify_key(signing_key_bytes: Vec<u8>) -> PyResult<Vec<u8>> {
    use aws_lc_rs::signature::KeyPair;
    
    if signing_key_bytes.len() != 32 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid signing key length"));
    }
    
    let key_pair = signature::Ed25519KeyPair::from_seed_unchecked(&signing_key_bytes)
        .map_err(|_| PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid signing key"))?;
    
    Ok(key_pair.public_key().as_ref().to_vec())
}

#[pyfunction]
pub fn encode_verify_key_base64(verify_key_bytes: Vec<u8>) -> PyResult<String> {
    Ok(STANDARD_NO_PAD.encode(&verify_key_bytes))
}

#[pyfunction]
pub fn encode_signing_key_base64(signing_key_bytes: Vec<u8>) -> PyResult<String> {
    Ok(STANDARD_NO_PAD.encode(&signing_key_bytes))
}

#[pyfunction]
pub fn decode_verify_key_base64(algorithm: String, _version: String, key_base64: String) -> PyResult<Vec<u8>> {
    if algorithm != "ed25519" {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Unsupported algorithm"));
    }
    
    let key_bytes = STANDARD_NO_PAD.decode(key_base64.as_bytes())
        .map_err(|_| PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid base64"))?;
    
    if key_bytes.len() != 32 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid key length"));
    }
    
    Ok(key_bytes)
}

#[pyfunction]
pub fn decode_signing_key_base64(algorithm: String, _version: String, key_base64: String) -> PyResult<Vec<u8>> {
    if algorithm != "ed25519" {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Unsupported algorithm"));
    }
    
    let key_bytes = STANDARD_NO_PAD.decode(key_base64.as_bytes())
        .map_err(|_| PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid base64"))?;
    
    if key_bytes.len() != 32 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid key length"));
    }
    
    Ok(key_bytes)
}

#[pyfunction]
pub fn is_signing_algorithm_supported(key_id: String) -> PyResult<bool> {
    Ok(key_id.starts_with("ed25519:"))
}

#[pyfunction]
pub fn read_signing_keys(py: Python, input_data: PyObject) -> PyResult<PyObject> {
    use pyo3::types::PyList;
    
    // Handle different input types like Python signedjson
    let content = if let Ok(py_list) = input_data.extract::<Vec<String>>(py) {
        // Handle list input (like config does)
        py_list.join("\n")
    } else if let Ok(py_str) = input_data.extract::<String>(py) {
        // Handle string input
        py_str
    } else {
        // Try to call read() method for file-like objects
        match input_data.call_method0(py, "read") {
            Ok(content_obj) => content_obj.extract::<String>(py)?,
            Err(_) => {
                let py_str = input_data.bind(py).str()?;
                py_str.to_string()
            },
        }
    };
    
    // Parse signing keys from content
    let mut signing_keys = Vec::new();
    
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        
        // Expected format: "ed25519 version base64_key"
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() >= 3 {
            let algorithm = parts.get(0).unwrap_or(&"");
            let version = parts.get(1).unwrap_or(&"");
            let key_b64 = parts.get(2).unwrap_or(&"");
            
            if algorithm == &"ed25519" && !version.is_empty() && !key_b64.is_empty() {
                match STANDARD_NO_PAD.decode(key_b64.as_bytes()) {
                Ok(key_bytes) if key_bytes.len() == 32 => {
                    // Create NaCl SigningKey object
                    let nacl_module = py.import("nacl.signing")?;
                    let signing_key_class = nacl_module.getattr("SigningKey")?;
                    let key_obj = signing_key_class.call1((key_bytes,))?;
                    
                    // Set algorithm and version attributes
                    key_obj.setattr("alg", "ed25519")?;
                    key_obj.setattr("version", version)?;
                    
                    signing_keys.push(key_obj);
                }
                _ => continue,
            }
            }
        }
    }
    
    Ok(PyList::new(py, signing_keys)?.into())
}

#[pyfunction]
pub fn read_old_signing_keys(py: Python, stream_content: PyObject) -> PyResult<PyObject> {
    // Same implementation as read_signing_keys for compatibility
    read_signing_keys(py, stream_content)
}

#[pyfunction]
pub fn write_signing_keys(keys: Vec<(String, Vec<u8>)>) -> PyResult<String> {
    let mut output = String::new();
    
    for (version, key_bytes) in keys {
        if key_bytes.len() == 32 {
            let key_b64 = STANDARD_NO_PAD.encode(&key_bytes);
            output.push_str(&format!("ed25519 {} {}\n", version, key_b64));
        }
    }
    
    Ok(output)
}
