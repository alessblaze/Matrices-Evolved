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
use base64::{Engine as _, engine::general_purpose::STANDARD_NO_PAD, engine::{GeneralPurpose, GeneralPurposeConfig}};
use serde_json;
use std::sync::LazyLock;
use std::cell::RefCell;
use pythonize::{pythonize, PythonizeError};
use serde::{Deserialize, Serialize};

// Cached debug flag for optimal performance
static CRYPTO_DEBUG_ENABLED: LazyLock<bool> = LazyLock::new(|| std::env::var("SYNAPSE_RUST_CRYPTO_DEBUG").unwrap_or_default() == "1");

// Lenient base64 decoder for compatibility with Python's base64 module
static LENIENT_BASE64: LazyLock<GeneralPurpose> = LazyLock::new(|| {
    let config = GeneralPurposeConfig::new().with_decode_allow_trailing_bits(true);
    GeneralPurpose::new(&base64::alphabet::STANDARD, config)
});

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

/// VerifyKey class to match C++ implementation
#[pyclass(subclass)]
#[derive(Debug, Clone)]
pub struct VerifyKey {
    #[pyo3(get, set)]
    pub alg: String,
    #[pyo3(get, set)]
    pub version: String,
    pub key_bytes: Vec<u8>,
}

#[pymethods]
impl VerifyKey {
    #[new]
    fn new(bytes: Vec<u8>, alg: Option<String>, version: Option<String>) -> Self {
        Self {
            key_bytes: bytes,
            alg: alg.unwrap_or_else(|| "ed25519".to_string()),
            version: version.unwrap_or_else(|| "1".to_string()),
        }
    }
    
    fn encode(&self, py: Python) -> PyResult<Py<pyo3::types::PyBytes>> {
        Ok(pyo3::types::PyBytes::new(py, &self.key_bytes).into())
    }
    
    fn verify(&self, message: Vec<u8>, signature: String) -> PyResult<bool> {
        let public_key = signature::UnparsedPublicKey::new(&signature::ED25519, &self.key_bytes);
        let signature_bytes = STANDARD_NO_PAD.decode(&signature)
            .map_err(|_| PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid signature encoding"))?;
        Ok(public_key.verify(&message, &signature_bytes).is_ok())
    }
    
    fn __eq__(&self, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        // Handle comparison with VerifyKey objects
        if let Ok(verify_key) = other.extract::<PyRef<VerifyKey>>() {
            return Ok(self.key_bytes == verify_key.key_bytes && self.alg == verify_key.alg && self.version == verify_key.version);
        }
        
        // Handle comparison with external VerifyKey objects (from signedjson)
        if let (Ok(alg), Ok(version)) = (other.getattr("alg").and_then(|v| v.extract::<String>()), other.getattr("version").and_then(|v| v.extract::<String>())) {
            if let Ok(encode_method) = other.getattr("encode") {
                if let Ok(encoded) = encode_method.call0() {
                    if let Ok(key_bytes) = encoded.extract::<Vec<u8>>() {
                        return Ok(self.key_bytes == key_bytes && self.alg == alg && self.version == version);
                    }
                }
            }
        }
        
        Ok(false)
    }
    

}

/// VerifyKeyWithExpiry class to match C++ implementation
#[pyclass(extends=VerifyKey)]
#[derive(Debug, Clone)]
pub struct VerifyKeyWithExpiry {
    #[pyo3(get, set)]
    pub expired: i32,
}

#[pymethods]
impl VerifyKeyWithExpiry {
    #[new]
    fn new(bytes: Vec<u8>, alg: Option<String>, version: Option<String>) -> PyClassInitializer<Self> {
        let base = VerifyKey::new(bytes, alg, version);
        let expiry = Self {
            expired: 0,
        };
        PyClassInitializer::from(base).add_subclass(expiry)
    }
}

/// SigningKey class to match C++ implementation
#[pyclass]
#[derive(Debug, Clone)]
pub struct SigningKey {
    #[pyo3(get, set)]
    pub alg: String,
    #[pyo3(get, set)]
    pub version: String,
    pub key_bytes: Vec<u8>,
}

#[pymethods]
impl SigningKey {
    #[new]
    fn new(bytes: Vec<u8>, alg: Option<String>, version: Option<String>) -> Self {
        Self {
            key_bytes: bytes,
            alg: alg.unwrap_or_else(|| "ed25519".to_string()),
            version: version.unwrap_or_else(|| "1".to_string()),
        }
    }
    
    #[staticmethod]
    fn generate() -> PyResult<Self> {
        use aws_lc_rs::rand::{SystemRandom, SecureRandom};
        let rng = SystemRandom::new();
        let mut seed = [0u8; 32];
        rng.fill(&mut seed)
            .map_err(|_| PyErr::new::<pyo3::exceptions::PyValueError, _>("Random generation failed"))?;
        Ok(Self::new(seed.to_vec(), None, None))
    }
    
    fn encode(&self, py: Python) -> PyResult<Py<pyo3::types::PyBytes>> {
        Ok(pyo3::types::PyBytes::new(py, &self.key_bytes).into())
    }
    
    fn get_verify_key(&self) -> PyResult<VerifyKey> {
        use aws_lc_rs::signature::KeyPair;
        let key_pair = signature::Ed25519KeyPair::from_seed_unchecked(&self.key_bytes)
            .map_err(|_| PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid signing key"))?;
        let public_key_bytes = key_pair.public_key().as_ref().to_vec();
        Ok(VerifyKey::new(public_key_bytes, Some(self.alg.clone()), Some(self.version.clone())))
    }
    
    fn sign(&self, message: Vec<u8>) -> PyResult<String> {
        let key_pair = signature::Ed25519KeyPair::from_seed_unchecked(&self.key_bytes)
            .map_err(|_| PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid signing key"))?;
        let signature = key_pair.sign(&message);
        Ok(STANDARD_NO_PAD.encode(signature.as_ref()))
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
    signing_key: PyObject,
    key_id: String
) -> PyResult<PyObject> {
    // Extract signing key bytes from either bytes or SigningKey object
    let signing_key_bytes = if let Ok(bytes) = signing_key.extract::<Vec<u8>>(py) {
        bytes
    } else {
        // Try to get bytes from SigningKey object
        if let Ok(encode_method) = signing_key.getattr(py, "encode") {
            encode_method.call0(py)?.extract::<Vec<u8>>(py)?
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>("Expected bytes or SigningKey object"));
        }
    };
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
    
    // Convert back to Python dict and update original in-place
    let signed_dict = pythonize::pythonize(py, &json_value)
        .map_err(|_| PyErr::new::<pyo3::exceptions::PyValueError, _>("Failed to convert to Python dict"))?;
    
    if let Ok(signed_dict_bound) = signed_dict.downcast::<PyDict>() {
        json_dict.clear();
        json_dict.update(signed_dict_bound.as_mapping())?;
        Ok(json_dict.clone().unbind().into())
    } else {
        Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Failed to create signed dict"))
    }
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
#[pyo3(signature = (input_bytes, urlsafe = false))]
pub fn encode_base64(py: Python, input_bytes: Vec<u8>, urlsafe: bool) -> PyResult<Py<pyo3::types::PyString>> {
    let encoded = if urlsafe {
        base64::engine::general_purpose::URL_SAFE.encode(&input_bytes)
    } else {
        base64::engine::general_purpose::STANDARD.encode(&input_bytes)
    };
    // Remove padding like Python implementation
    let unpadded = encoded.trim_end_matches('=');
    Ok(pyo3::types::PyString::new(py, unpadded).into())
}

#[pyfunction]
pub fn decode_base64(py: Python, input_string: String) -> PyResult<Py<pyo3::types::PyBytes>> {
    let input_len = input_string.len();
    let padding_count = 3 - ((input_len + 3) % 4);
    let mut padded_input = input_string;
    for _ in 0..padding_count {
        padded_input.push('=');
    }
    
    // Try standard base64 with altchars for URL-safe characters
    let decoded = base64::engine::general_purpose::STANDARD.decode(padded_input.replace('-', "+").replace('_', "/"))
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Base64 decode error: {}", e)))?;
    Ok(pyo3::types::PyBytes::new(py, &decoded).into())
}

// Keep fast versions for internal use
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
    
    // Convert PyDict to serde_json::Value
    let mut json_value: serde_json::Value = pythonize::depythonize(json_object)
        .map_err(|_| PyErr::new::<pyo3::exceptions::PyValueError, _>("Failed to convert Python dict"))?;
    
    if *CRYPTO_DEBUG_ENABLED {
        println!("DEBUG: JSON conversion completed");
    }
    
    // Create canonical JSON without signatures
    let mut unsigned_json = json_value.clone();
    if let Some(obj) = unsigned_json.as_object_mut() {
        obj.remove("signatures");
        obj.remove("unsigned");
    }
    
    if *CRYPTO_DEBUG_ENABLED {
        println!("DEBUG: About to canonicalize JSON");
    }
    
    let canonical_json = canonicalize_json(&unsigned_json)
        .map_err(|_| PyErr::new::<pyo3::exceptions::PyValueError, _>("JSON canonicalization failed"))?;
    
    if *CRYPTO_DEBUG_ENABLED {
        println!("DEBUG: JSON canonicalization completed");
    }
    
    if *CRYPTO_DEBUG_ENABLED {
        println!("DEBUG: About to sign with cached key pair");
    }
    
    // Sign using cached key pair
    let signature_b64: String = CACHED_SIGNING_KEY.with(|cached_key| {
        CACHED_KEY_PAIR.with(|cached_pair| -> PyResult<String> {
            if *CRYPTO_DEBUG_ENABLED {
                println!("DEBUG: Inside cached key pair closure");
            }
            
            let mut cached_key = cached_key.borrow_mut();
            let mut cached_pair = cached_pair.borrow_mut();
            
            if cached_key.as_ref() != Some(&signing_key_bytes) || cached_pair.is_none() {
                if *CRYPTO_DEBUG_ENABLED {
                    println!("DEBUG: Creating new key pair from seed");
                }
                
                let new_key_pair = signature::Ed25519KeyPair::from_seed_unchecked(&signing_key_bytes)
                    .map_err(|_| PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid signing key"))?;
                
                *cached_key = Some(signing_key_bytes);
                *cached_pair = Some(new_key_pair);
                
                if *CRYPTO_DEBUG_ENABLED {
                    println!("DEBUG: New key pair created and cached");
                }
            }
            
            if *CRYPTO_DEBUG_ENABLED {
                println!("DEBUG: About to sign canonical JSON");
            }
            
            let signature = cached_pair.as_ref().unwrap().sign(canonical_json.as_bytes());
            
            if *CRYPTO_DEBUG_ENABLED {
                println!("DEBUG: Signature created, encoding to base64");
            }
            
            Ok(STANDARD_NO_PAD.encode(signature.as_ref()))
        })
    })?;
    
    if *CRYPTO_DEBUG_ENABLED {
        println!("DEBUG: Signature base64 encoding completed");
    }
    
    if *CRYPTO_DEBUG_ENABLED {
        println!("DEBUG: About to add signature to JSON object");
    }
    
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
    
    if *CRYPTO_DEBUG_ENABLED {
        println!("DEBUG: Signature added to JSON, about to convert back to Python dict");
    }
    
    // Convert back to Python dict and update original in-place like C++ version
    let signed_dict = pythonize::pythonize(py, &json_value)
        .map_err(|_| PyErr::new::<pyo3::exceptions::PyValueError, _>("Failed to convert to Python dict"))?;
    
    if *CRYPTO_DEBUG_ENABLED {
        println!("DEBUG: About to clear and update original dictionary in-place");
    }
    
    // Clear the original dictionary and copy all items from signed dict (like C++ version)
    json_object.clear();
    
    if let Ok(signed_dict_bound) = signed_dict.downcast::<PyDict>() {
        for (key, value) in signed_dict_bound.iter() {
            json_object.set_item(key, value)?;
        }
    }
    
    if *CRYPTO_DEBUG_ENABLED {
        println!("DEBUG: Dictionary updated in-place, returning signed dictionary");
    }
    
    // Return the signed dictionary for compatibility (like C++ version)
    Ok(signed_dict.into())
}

#[pyfunction]
pub fn verify_signature(json_bytes: Vec<u8>, signature_b64: String, verify_key_bytes: Vec<u8>) -> PyResult<bool> {
    verify_signature_fast(json_bytes, signature_b64, verify_key_bytes)
}



#[pyfunction]
pub fn verify_signed_json(py: Python, json_dict: &Bound<'_, PyDict>, signature_name: String, verify_key: PyObject) -> PyResult<()> {
    // Convert once and reuse
    let json_value: serde_json::Value = pythonize::depythonize(json_dict)
        .map_err(|_| PyErr::new::<pyo3::exceptions::PyValueError, _>("Failed to convert Python dict"))?;
    
    // Extract verify key information - handle both VerifyKey objects and raw bytes
    let (alg, version, verify_key_bytes) = if let Ok(bytes_data) = verify_key.extract::<Vec<u8>>(py) {
        // Raw bytes - need to find version from signatures
        let mut found_version = "auto".to_string();
        if let Some(signatures) = json_value.get("signatures") {
            if let Some(server_sigs) = signatures.get(&signature_name) {
                if let Some(sig_obj) = server_sigs.as_object() {
                    // Find the first ed25519 key
                    for key_id in sig_obj.keys() {
                        if key_id.starts_with("ed25519:") {
                            if let Some(version_part) = key_id.split(':').nth(1) {
                                found_version = version_part.to_string();
                            }
                            break;
                        }
                    }
                }
            }
        }
        ("ed25519".to_string(), found_version, bytes_data)
    } else {
        // VerifyKey object - extract attributes safely
        let alg = verify_key.getattr(py, "alg")
            .and_then(|v| v.extract::<String>(py))
            .unwrap_or_else(|_| "ed25519".to_string());
        let version = verify_key.getattr(py, "version")
            .and_then(|v| v.extract::<String>(py))
            .unwrap_or_else(|_| "1".to_string());
        
        // Get key bytes - try encode() method first, then direct extraction
        let key_bytes = if let Ok(encode_method) = verify_key.getattr(py, "encode") {
            encode_method.call0(py)?.extract::<Vec<u8>>(py)?
        } else {
            // Fallback: try to get key_bytes attribute directly
            verify_key.getattr(py, "key_bytes")?.extract::<Vec<u8>>(py)?
        };
        
        (alg, version, key_bytes)
    };
    
    let key_id = format!("{}:{}", alg, version);
    
    // Get signatures with better error handling
    let signatures = json_value.get("signatures")
        .and_then(|s| s.as_object())
        .ok_or_else(|| super::SignatureVerifyException::new_err("No signatures on this object"))?;
    
    let server_sigs = signatures.get(&signature_name)
        .and_then(|s| s.as_object())
        .ok_or_else(|| super::SignatureVerifyException::new_err(
            format!("Missing signature for {}", signature_name)
        ))?;
    
    // Find the signature - try exact match first, then any ed25519 key
    let signature_b64 = if let Some(sig) = server_sigs.get(&key_id).and_then(|v| v.as_str()) {
        sig
    } else {
        // Fallback: find any ed25519 signature
        let mut fallback_sig = None;
        for (available_key, sig_value) in server_sigs.iter() {
            if available_key.starts_with("ed25519:") {
                if let Some(sig) = sig_value.as_str() {
                    fallback_sig = Some(sig);
                    break;
                }
            }
        }
        fallback_sig.ok_or_else(|| super::SignatureVerifyException::new_err(
            format!("Missing signature for {}, {}", signature_name, key_id)
        ))?
    };
    
    // Decode signature with better error handling
    let signature_bytes = STANDARD_NO_PAD.decode(&signature_b64)
        .map_err(|e| super::SignatureVerifyException::new_err(
            format!("Invalid signature base64 for {}, {}: {:?}", signature_name, key_id, e)
        ))?;
    
    // Validate key and signature lengths
    if verify_key_bytes.len() != 32 {
        return Err(super::SignatureVerifyException::new_err(
            format!("Invalid verify key length: {} (expected 32)", verify_key_bytes.len())
        ));
    }
    
    if signature_bytes.len() != 64 {
        return Err(super::SignatureVerifyException::new_err(
            format!("Invalid signature length: {} (expected 64)", signature_bytes.len())
        ));
    }
    
    // Create unsigned JSON for verification
    let mut unsigned_json = json_value.clone();
    if let Some(obj) = unsigned_json.as_object_mut() {
        obj.remove("signatures");
        obj.remove("unsigned");
    }
    
    // Canonicalize and verify
    let canonical_json = canonicalize_json(&unsigned_json)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("JSON canonicalization failed: {:?}", e)
        ))?;
    let message = canonical_json.as_bytes();
    
    // Perform verification
    let public_key = signature::UnparsedPublicKey::new(&signature::ED25519, &verify_key_bytes);
    
    match public_key.verify(&message, &signature_bytes) {
        Ok(_) => Ok(()),
        Err(e) => Err(super::SignatureVerifyException::new_err(
            format!("Unable to verify signature for {}: {:?}", signature_name, e)
        ))
    }
}

#[pyfunction]
pub fn decode_verify_key_bytes(py: Python, key_id: String, key_bytes: Vec<u8>) -> PyResult<Py<VerifyKeyWithExpiry>> {
    let _decoded_bytes = decode_verify_key_bytes_fast(key_id.clone(), key_bytes.clone())?;
    
    // Extract version from key_id (format: "ed25519:version")
    let version = if key_id.starts_with("ed25519:") {
        key_id.strip_prefix("ed25519:").unwrap_or("1").to_string()
    } else {
        "1".to_string()
    };
    
    let init = VerifyKeyWithExpiry::new(key_bytes, Some("ed25519".to_string()), Some(version));
    let result = Py::new(py, init)?;
    
    // Set expired to max int32 for non-expired keys
    result.borrow_mut(py).expired = 2147483647;
    
    Ok(result)
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
    // Handle None/empty input gracefully
    if json_dict.is_empty() {
        return Ok(Vec::new());
    }
    
    let json_value: serde_json::Value = pythonize::depythonize(json_dict)
        .map_err(|_| PyErr::new::<pyo3::exceptions::PyValueError, _>("Failed to convert Python dict"))?;
    
    let mut ids = Vec::new();
    
    // Defensive programming - ensure we always return a valid Vec, never None
    if let Some(signatures) = json_value.get("signatures") {
        if let Some(server_sigs) = signatures.get(&signature_name) {
            if let Some(sig_obj) = server_sigs.as_object() {
                // Collect signature IDs safely
                for (key, _) in sig_obj.iter() {
                    if key.starts_with("ed25519:") {
                        ids.push(key.clone());
                        // Reasonable limit to prevent issues
                        if ids.len() >= 100 {
                            break;
                        }
                    }
                }
            }
        }
    }
    
    // Always return a valid Vec, even if empty
    Ok(ids)
}

// signedjson.key module functions
#[pyfunction]
pub fn generate_signing_key(version: String) -> PyResult<SigningKey> {
    use aws_lc_rs::rand::{SystemRandom, SecureRandom};
    let rng = SystemRandom::new();
    let mut seed = [0u8; 32];
    rng.fill(&mut seed)
        .map_err(|_| PyErr::new::<pyo3::exceptions::PyValueError, _>("Random generation failed"))?;
    Ok(SigningKey::new(seed.to_vec(), Some("ed25519".to_string()), Some(version)))
}

#[pyfunction]
pub fn get_verify_key(py: Python, signing_key: PyObject) -> PyResult<VerifyKey> {
    use aws_lc_rs::signature::KeyPair;
    
    // Try to extract bytes and version from SigningKey object first
    let (signing_key_bytes, alg, version) = if let Ok(bytes) = signing_key.extract::<Vec<u8>>(py) {
        (bytes, "ed25519".to_string(), "1".to_string())
    } else {
        // Try to get attributes from SigningKey object
        let alg = signing_key.getattr(py, "alg")
            .and_then(|v| v.extract::<String>(py))
            .unwrap_or_else(|_| "ed25519".to_string());
        let version = signing_key.getattr(py, "version")
            .and_then(|v| v.extract::<String>(py))
            .unwrap_or_else(|_| "1".to_string());
        
        let bytes = if let Ok(encode_method) = signing_key.getattr(py, "encode") {
            encode_method.call0(py)?.extract::<Vec<u8>>(py)?
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>("Expected bytes or SigningKey object"));
        };
        
        (bytes, alg, version)
    };
    
    if signing_key_bytes.len() != 32 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid signing key length"));
    }
    
    let key_pair = signature::Ed25519KeyPair::from_seed_unchecked(&signing_key_bytes)
        .map_err(|_| PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid signing key"))?;
    
    Ok(VerifyKey {
        key_bytes: key_pair.public_key().as_ref().to_vec(),
        alg,
        version,
    })
}

#[pyfunction]
pub fn encode_verify_key_base64(py: Python, verify_key: PyObject) -> PyResult<String> {
    // Handle both raw bytes and VerifyKey objects
    let verify_key_bytes = if let Ok(bytes) = verify_key.extract::<Vec<u8>>(py) {
        bytes
    } else {
        // Try to get bytes from VerifyKey object
        if let Ok(encode_method) = verify_key.getattr(py, "encode") {
            encode_method.call0(py)?.extract::<Vec<u8>>(py)?
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>("Expected bytes or VerifyKey object"));
        }
    };
    Ok(STANDARD_NO_PAD.encode(&verify_key_bytes))
}

#[pyfunction]
pub fn encode_signing_key_base64(signing_key_bytes: Vec<u8>) -> PyResult<String> {
    Ok(STANDARD_NO_PAD.encode(&signing_key_bytes))
}

#[pyfunction]
pub fn decode_verify_key_base64(algorithm: String, version: String, key_base64: String) -> PyResult<VerifyKey> {
    if algorithm != "ed25519" {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Unsupported algorithm"));
    }
    
    // Try both padded and unpadded base64 decoding
    let key_bytes = base64::engine::general_purpose::STANDARD.decode(key_base64.as_bytes())
        .or_else(|_| STANDARD_NO_PAD.decode(key_base64.as_bytes()))
        .map_err(|_| PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid base64"))?;
    
    if key_bytes.len() != 32 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid key length"));
    }
    
    Ok(VerifyKey::new(key_bytes, Some(algorithm), Some(version)))
}

#[pyfunction]
pub fn decode_signing_key_base64(algorithm: String, version: String, key_base64: String) -> PyResult<SigningKey> {
    if algorithm != "ed25519" {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Unsupported algorithm"));
    }
    
    // Handle unpadded base64 with + and / characters like pure Python signedjson
    let key_bytes = {
        let input_len = key_base64.len();
        let padding_count = 3 - ((input_len + 3) % 4);
        let mut padded_key = key_base64.clone();
        for _ in 0..padding_count {
            padded_key.push('=');
        }

        // Use cached lenient decoder that allows trailing bits like Python
        LENIENT_BASE64.decode(padded_key.as_bytes())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Invalid base64: {:?}", e)))?
    };
    
    if key_bytes.len() != 32 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid key length"));
    }
    
    Ok(SigningKey::new(key_bytes, Some(algorithm), Some(version)))
}

#[pyfunction]
pub fn is_signing_algorithm_supported(key_id: String) -> PyResult<bool> {
    Ok(key_id.starts_with("ed25519:"))
}

#[pyfunction]
pub fn read_signing_keys(py: Python, input_data: PyObject) -> PyResult<PyObject> {
    use pyo3::types::PyList;
    
    // Always return empty list for None input to prevent NoneType errors
    if input_data.is_none(py) {
        return Ok(PyList::empty(py).into());
    }
    
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
            Ok(content_obj) => {
                if content_obj.is_none(py) {
                    return Ok(PyList::empty(py).into());
                }
                content_obj.extract::<String>(py).unwrap_or_default()
            },
            Err(_) => {
                // Fallback: try to convert to string
                match input_data.bind(py).str() {
                    Ok(py_str) => py_str.to_string(),
                    Err(_) => return Ok(PyList::empty(py).into()),
                }
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
                // Try both padded and unpadded base64 decoding
                let decode_result = base64::engine::general_purpose::STANDARD.decode(key_b64.as_bytes())
                    .or_else(|_| STANDARD_NO_PAD.decode(key_b64.as_bytes()));
                match decode_result {
                Ok(key_bytes) if key_bytes.len() == 32 => {
                    // Create NaCl SigningKey object - handle potential import failures
                    match py.import("nacl.signing") {
                        Ok(nacl_module) => {
                            if let Ok(signing_key_class) = nacl_module.getattr("SigningKey") {
                                if let Ok(key_obj) = signing_key_class.call1((key_bytes,)) {
                                    let _ = key_obj.setattr("alg", "ed25519");
                                    let _ = key_obj.setattr("version", version);
                                    signing_keys.push(key_obj);
                                }
                            }
                        },
                        Err(_) => continue, // Skip if nacl not available
                    }
                }
                _ => continue,
            }
            }
        }
    }
    
    // Always return a valid PyList, never None
    Ok(PyList::new(py, signing_keys).unwrap_or_else(|_| PyList::empty(py)).into())
}

#[pyfunction]
pub fn read_old_signing_keys(py: Python, stream_content: PyObject) -> PyResult<PyObject> {
    use pyo3::types::PyList;
    // Always return an empty list for compatibility - old signing keys are deprecated
    Ok(PyList::empty(py).into())
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