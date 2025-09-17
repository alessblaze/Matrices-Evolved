"""
matrices_evolved - Python bindings for Matrix event signing and cryptography

This package provides high-performance C++ and Rust implementations of Matrix protocol
cryptographic operations.

Usage:
    import matrices_evolved.cpp as cpp_impl
    import matrices_evolved.rust as rust_impl
"""

# Try to import available implementations
try:
    from . import _event_signing_impl as cpp
    _has_cpp = True
except ImportError:
    _has_cpp = False

try:
    from . import matrices_evolved_rust as rust
    _has_rust = True
except ImportError:
    _has_rust = False

# Default to C++ implementation if available, otherwise Rust
if _has_cpp:
    from ._event_signing_impl import *
elif _has_rust:
    from .matrices_evolved_rust import *
else:
    raise ImportError("No implementation available. Build with either C++ or Rust support.")

# Expose available implementations
__implementations__ = []
if _has_cpp:
    __implementations__.append('cpp')
if _has_rust:
    __implementations__.append('rust')

__version__ = "1.0.0"
__author__ = "Aless Microsystems"
__license__ = "AGPL-3.0"

__all__ = [
    # Core signing functions
    'generate_signing_key',
    'get_verify_key', 
    'sign_json_fast',
    'sign_json_with_info',
    'sign_json_object_fast',
    'verify_signature_fast',
    'verify_signature_with_info',
    'verify_signed_json_fast',
    
    # Hash functions
    'compute_content_hash',
    'compute_content_hash_fast',
    'compute_event_reference_hash',
    'compute_event_reference_hash_fast',
    
    # Base64 functions
    'encode_base64',
    'decode_base64',
    'encode_base64_fast',
    'decode_base64_fast',
    
    # Key management
    'encode_verify_key_base64',
    'encode_signing_key_base64',
    'decode_verify_key_base64',
    'decode_signing_key_base64',
    'decode_verify_key_bytes',
    'decode_verify_key_bytes_fast',
    'is_signing_algorithm_supported',
    
    # JSON functions
    'encode_canonical_json',
    'signature_ids',
    
    # High-level API
    'sign_json',
    'verify_signature',
    'verify_signed_json',
    
    # Key file operations
    'read_signing_keys',
    'read_old_signing_keys', 
    'write_signing_keys',
    
    # Result classes
    'SigningResult',
    'VerificationResult',
    
    # Key classes (for compatibility)
    'SigningKey',
    'VerifyKey',
]