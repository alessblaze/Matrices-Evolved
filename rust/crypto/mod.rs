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
pub mod event_signing;
pub mod cache;
pub mod stream_change_cache;

use pyo3::prelude::*;
use pyo3::exceptions::PyException;

// Create SignatureVerifyException equivalent
pyo3::create_exception!(synapse_rust, SignatureVerifyException, PyException);

pub use event_signing::{
    compute_content_hash_fast,
    compute_event_reference_hash_fast,
    sign_json_fast,
    sign_json_object_fast,
    verify_signature_fast,
    encode_base64_fast,
    decode_base64_fast,
    verify_signed_json_fast,
    decode_verify_key_bytes_fast,
    compute_content_hash,
    compute_event_reference_hash,
    sign_json,
    verify_signature,
    encode_base64,
    decode_base64,
    verify_signed_json,
    decode_verify_key_bytes,
    encode_canonical_json,
    signature_ids,
    generate_signing_key,
    get_verify_key,
    encode_verify_key_base64,
    encode_signing_key_base64,
    decode_verify_key_base64,
    decode_signing_key_base64,
    is_signing_algorithm_supported,
    read_signing_keys,
    read_old_signing_keys,
    write_signing_keys,
    VerifyKey,
    VerifyKeyWithExpiry,
    SigningKey,
};

pub use cache::{
    RustLruCache,
    AsyncRustLruCache,
    RustCacheNode,
    create_rust_lru_cache,
    create_async_rust_lru_cache,
};

pub use stream_change_cache::{
    RustStreamChangeCache,
    AllEntitiesChangedResult,
    CacheMetrics,
};



pub fn register_module(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Add key classes first
    m.add_class::<VerifyKey>()?;
    m.add_class::<VerifyKeyWithExpiry>()?;
    m.add_class::<SigningKey>()?;
    
    // _fast versions for wrapper compatibility
    m.add_function(wrap_pyfunction!(compute_content_hash_fast, m)?)?;
    m.add_function(wrap_pyfunction!(compute_event_reference_hash_fast, m)?)?;
    m.add_function(wrap_pyfunction!(sign_json_fast, m)?)?;
    m.add_function(wrap_pyfunction!(sign_json_object_fast, m)?)?;
    m.add_function(wrap_pyfunction!(verify_signature_fast, m)?)?;
    m.add_function(wrap_pyfunction!(encode_base64_fast, m)?)?;
    m.add_function(wrap_pyfunction!(decode_base64_fast, m)?)?;
    m.add_function(wrap_pyfunction!(verify_signed_json_fast, m)?)?;
    m.add_function(wrap_pyfunction!(decode_verify_key_bytes_fast, m)?)?;
    
    // Standard names for drop-in replacement
    m.add_function(wrap_pyfunction!(compute_content_hash, m)?)?;
    m.add_function(wrap_pyfunction!(compute_event_reference_hash, m)?)?;
    m.add_function(wrap_pyfunction!(sign_json, m)?)?;
    m.add_function(wrap_pyfunction!(verify_signature, m)?)?;
    m.add_function(wrap_pyfunction!(encode_base64, m)?)?;
    m.add_function(wrap_pyfunction!(decode_base64, m)?)?;
    m.add_function(wrap_pyfunction!(verify_signed_json, m)?)?;
    m.add_function(wrap_pyfunction!(decode_verify_key_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(encode_canonical_json, m)?)?;
    m.add_function(wrap_pyfunction!(signature_ids, m)?)?;
    m.add_function(wrap_pyfunction!(generate_signing_key, m)?)?;
    m.add_function(wrap_pyfunction!(get_verify_key, m)?)?;
    m.add_function(wrap_pyfunction!(encode_verify_key_base64, m)?)?;
    m.add_function(wrap_pyfunction!(encode_signing_key_base64, m)?)?;
    m.add_function(wrap_pyfunction!(decode_verify_key_base64, m)?)?;
    m.add_function(wrap_pyfunction!(decode_signing_key_base64, m)?)?;
    m.add_function(wrap_pyfunction!(is_signing_algorithm_supported, m)?)?;
    m.add_function(wrap_pyfunction!(read_signing_keys, m)?)?;
    m.add_function(wrap_pyfunction!(read_old_signing_keys, m)?)?;
    m.add_function(wrap_pyfunction!(write_signing_keys, m)?)?;
    
    // Rust LRU Cache
    m.add_class::<RustLruCache>()?;
    m.add_class::<AsyncRustLruCache>()?;
    m.add_class::<RustCacheNode>()?;
    m.add_function(wrap_pyfunction!(create_rust_lru_cache, m)?)?;
    m.add_function(wrap_pyfunction!(create_async_rust_lru_cache, m)?)?;
    
    // Rust StreamChangeCache
    m.add_class::<stream_change_cache::RustStreamChangeCache>()?;
    m.add_class::<stream_change_cache::AllEntitiesChangedResult>()?;
    m.add_class::<stream_change_cache::CacheMetrics>()?;
    
    // Add exception class
    m.add("SignatureVerifyException", py.get_type::<SignatureVerifyException>())?;
    
    Ok(())
}