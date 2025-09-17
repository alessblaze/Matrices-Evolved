# matrices_evolved API Documentation

Complete API reference for the matrices_evolved library.

## Import Methods

### Auto-Selection (Recommended)
```python
import matrices_evolved

# Uses C++ implementation if available, falls back to Rust
# Provides all functions directly in the namespace
key = matrices_evolved.generate_signing_key('ed25519')
```

### Explicit Implementation Selection
```python
# C++ implementation (nanobind)
import matrices_evolved.cpp as crypto

# Rust implementation (PyO3)  
import matrices_evolved.rust as crypto

# Both provide identical APIs
key = crypto.generate_signing_key('ed25519')
```

### Check Available Implementations
```python
import matrices_evolved
print(matrices_evolved.__implementations__)  # ['cpp', 'rust']
```

## Core Functions

> **Note**: Most functions have two variants:
> - **Standard functions**: Accept `list[int]` for byte data (compatible with some Matrix libraries)
> - **`*_fast` functions**: Accept `bytes` or `str` for better performance and convenience
> 
> Use `*_fast` variants for new code unless you need specific compatibility.

### Key Generation and Management

#### `generate_signing_key(algorithm: str) -> list[int]`
Generate a new Ed25519 signing key.

```python
signing_key = matrices_evolved.generate_signing_key('ed25519')
# Returns: List of 32 integers (0-255) representing Ed25519 private key
```

#### `get_verify_key(signing_key: list[int]) -> list[int]`
Derive the public verification key from a signing key.

```python
verify_key = matrices_evolved.get_verify_key(signing_key)
# Returns: List of 32 integers (0-255) representing Ed25519 public key
```

### JSON Signing

#### `sign_json_fast(json_bytes: list[int], signing_key: list[int]) -> str`
Sign raw JSON bytes with Ed25519.

```python
json_data = list(b'{"type":"m.room.message"}')
signature = matrices_evolved.sign_json_fast(json_data, signing_key)
# Returns: Base64-encoded signature
```

#### `sign_json_object_fast(obj: dict, server_name: str, signing_key: list[int], key_id: str) -> dict`
Sign a Python dictionary and add signature to the object.

```python
event = {"type": "m.room.message", "content": {"body": "Hello!"}}
signed_event = matrices_evolved.sign_json_object_fast(
    event, 
    "example.com",      # Server name
    signing_key,        # 32-byte signing key
    "ed25519:v1"       # Key identifier
)
# Returns: Dictionary with added 'signatures' field
```

### Signature Verification

#### `verify_signature_fast(json_bytes: list[int], signature: str, verify_key: list[int]) -> bool`
Verify a signature against raw JSON bytes.

```python
is_valid = matrices_evolved.verify_signature_fast(
    json_data,          # List of integers from JSON bytes
    signature,          # Base64 signature
    verify_key         # List of 32 integers (public key)
)
# Returns: True if signature is valid
```

#### `verify_signed_json_fast(signed_obj: dict, server_name: str, verify_key: list[int]) -> None`
Verify a signed JSON object. Raises exception if invalid.

```python
try:
    matrices_evolved.verify_signed_json_fast(
        signed_event,       # Dictionary with signatures
        "example.com",      # Server name
        verify_key         # 32-byte public key
    )
    print("✅ Signature valid")
except matrices_evolved.SignatureVerifyException:
    print("❌ Invalid signature")
```

### Hash Functions

#### `compute_content_hash(event_dict: dict) -> tuple[str, bytes]`
#### `compute_content_hash_fast(json_string: str) -> tuple[str, bytes]`
Compute SHA256 hash of event content.

```python
# Standard version (dict input)
algorithm, hash_bytes = matrices_evolved.compute_content_hash(event)
# Fast version (JSON string input)
algorithm, hash_bytes = matrices_evolved.compute_content_hash_fast('{"type":"m.test"}')
# Both return: ("sha256", 32-byte hash)
```

#### `compute_event_reference_hash(event_dict: dict) -> tuple[str, bytes]`
#### `compute_event_reference_hash_fast(json_string: str) -> tuple[str, bytes]`
Compute reference hash for event deduplication.

```python
algorithm, hash_bytes = matrices_evolved.compute_event_reference_hash(event)
# Returns: ("sha256", 32-byte hash)
```

### Base64 Operations

#### `encode_base64(data: list[int]) -> str`
#### `encode_base64_fast(data: bytes) -> str`
Encode bytes to base64 (no padding).

```python
# Standard version (list input)
encoded = matrices_evolved.encode_base64(list(b"hello"))
# Fast version (bytes input)
encoded = matrices_evolved.encode_base64_fast(b"hello")
# Both return: "aGVsbG8"
```

#### `decode_base64(data: str) -> bytes`
#### `decode_base64_fast(data: str) -> bytes`
Decode base64 string to bytes.

```python
decoded = matrices_evolved.decode_base64("aGVsbG8")
# Returns: b"hello"
```

### Key Encoding/Decoding

#### `encode_verify_key_base64(verify_key: list[int]) -> str`
Encode verification key to base64.

```python
key_b64 = matrices_evolved.encode_verify_key_base64(verify_key)
```

#### `decode_verify_key_base64(algorithm: str, version: str, key_b64: str) -> list[int]`
Decode base64 verification key.

```python
verify_key = matrices_evolved.decode_verify_key_base64(
    "ed25519", "v1", key_b64
)
# Returns: List of 32 integers
```

### JSON Canonicalization

#### `encode_canonical_json(json_dict: dict) -> bytes`
Convert dictionary to canonical JSON bytes.

```python
canonical = matrices_evolved.encode_canonical_json({"b": 2, "a": 1})
# Returns: b'{"a":1,"b":2}' (sorted keys, no whitespace)
```

### Key File Operations

#### `read_signing_keys(input_data: str | list) -> list`
Parse signing keys from file content or list.

```python
keys = matrices_evolved.read_signing_keys([
    "ed25519 v1 base64_key_here"
])
# Returns: List of signing key objects
```

#### `write_signing_keys(keys: list[tuple[str, bytes]]) -> str`
Format signing keys for file storage.

```python
content = matrices_evolved.write_signing_keys([
    ("v1", signing_key_bytes)
])
# Returns: "ed25519 v1 base64_encoded_key"
```

## Advanced Functions

### High-Level API (Compatible with signedjson)

#### `sign_json(json_object: dict, signature_name: str, signing_key: object) -> dict`
Sign JSON using key object (signedjson compatibility).

```python
# Requires signing key object with .alg, .version, .encode() methods
signed = matrices_evolved.sign_json(event, "example.com", key_obj)
```

#### `verify_signed_json(json_dict: dict, signature_name: str, verify_key: object) -> None`
Verify using key object (signedjson compatibility).

```python
matrices_evolved.verify_signed_json(signed_event, "example.com", key_obj)
```

### Utility Functions

#### `signature_ids(json_dict: dict, signature_name: str) -> list[str]`
Get all signature key IDs for a server.

```python
key_ids = matrices_evolved.signature_ids(signed_event, "example.com")
# Returns: ["ed25519:v1", "ed25519:v2"]
```

#### `is_signing_algorithm_supported(key_id: str) -> bool`
Check if signing algorithm is supported.

```python
supported = matrices_evolved.is_signing_algorithm_supported("ed25519:v1")
# Returns: True
```

## Error Handling

### SignatureVerifyException
Raised when signature verification fails.

```python
try:
    matrices_evolved.verify_signed_json_fast(event, server, key)
except matrices_evolved.SignatureVerifyException as e:
    print(f"Verification failed: {e}")
```

## Performance Notes

### Function Variants

- **`*_fast` functions**: Optimized versions with minimal overhead
- **Standard functions**: Compatible with existing Matrix libraries
- **Both provide identical functionality**, choose based on your needs

### Best Practices

```python
# Prefer fast variants for performance-critical code
signature = matrices_evolved.sign_json_fast(data, key)

# Use standard variants for drop-in compatibility
signature = matrices_evolved.sign_json(data, server, key_obj)

# Cache verification keys to avoid repeated decoding
verify_key = matrices_evolved.get_verify_key(signing_key)  # Cache this
for event in events:
    matrices_evolved.verify_signed_json_fast(event, server, verify_key)
```

## Implementation Differences

Both C++ and Rust implementations provide identical APIs and behavior:

- **Function signatures**: Identical
- **Return values**: Identical  
- **Error handling**: Identical
- **Performance**: Similar (Rust slightly faster imports, C++ slightly faster execution)

Choose based on your build environment and preferences:

```python
# C++ implementation - requires Clang 20
import matrices_evolved.cpp as crypto

# Rust implementation - requires Rust 1.86+
import matrices_evolved.rust as crypto

# Auto-selection - uses C++ if available
import matrices_evolved as crypto
```

## Thread Safety

All functions are thread-safe and can be called concurrently from multiple threads without synchronization.

## Memory Management

- **Zero-copy operations** where possible
- **Automatic memory management** - no manual cleanup required
- **Minimal allocations** for performance-critical paths