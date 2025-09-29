# Matrices Evolved
<img width="1536" height="1024" alt="matrices_evolved" src="https://github.com/user-attachments/assets/49763adb-6660-438d-9791-0a3a0855d5a3" />




## Features

- **Dual Implementation Support** - Choose between C++ (nanobind) or Rust (PyO3) backends
- **Ed25519 Digital Signatures** - Fast cryptographic signing and verification
- **JSON Canonicalization** - Matrix-compatible canonical JSON encoding  
- **Base64 Operations** - Optimized encoding/decoding
- **Hash Computation** - SHA256 content and reference hashing
- **Key Management** - Generate, encode, and manage signing keys
- **Matrix Protocol Compatible** - Drop-in replacement for Matrix implementations
- **High Performance** - 10-100% faster than pure Python implementations

## Installation

### Prerequisites

**For C++ Implementation:**
- **Clang 20** (required for C++23 support)
- **Python 3.8+**
- **CMake 3.15+**

**For Rust Implementation:**
- **Rust 1.86+** (required for latest PyO3 features)
- **Cargo**

### Install from PyPI
### Only Python 3.11 wheels available. RHEL9/Ubuntu 22.04 both support it.
## Maybe Python 3.12 in future but for now only 3.11.
```bash
pip install matrices-evolved
```

### Build from Source

**C++ Implementation (default):**
```bash
git clone https://github.com/alessblaze/matrices_evolved.git
cd matrices_evolved
pip install .
```

**Rust Implementation:**
```bash
pip install . -C cmake.define.BUILD_RUST=ON -C cmake.define.BUILD_CPP=OFF
```

**Both Implementations:**
```bash
pip install . -C cmake.define.BUILD_RUST=ON
```

## Quick Start

```python
import matrices_evolved

# Generate signing keys
signing_key = matrices_evolved.generate_signing_key('ed25519')
verify_key = matrices_evolved.get_verify_key(signing_key)

# Sign a Matrix event
event = {"type": "m.room.message", "content": {"body": "Hello!"}}
signed_event = matrices_evolved.sign_json_object_fast(
    event, "myserver.com", signing_key, "ed25519:v1"
)

# Verify the signature
matrices_evolved.verify_signed_json_fast(signed_event, "myserver.com", verify_key)
print("✅ Signature verified!")
```

## Implementation Selection

```python
# Auto-select (prefers C++ if available)
import matrices_evolved

# Explicit C++ implementation
import matrices_evolved.cpp as crypto

# Explicit Rust implementation  
import matrices_evolved.rust as crypto

# Check available implementations
print(matrices_evolved.__implementations__)  # ['cpp', 'rust']
```

## Performance

Both implementations provide significant performance improvements over pure Python:

- **10-100x faster** signing and verification
- **5-50x faster** JSON canonicalization  
- **2-10x faster** base64 operations
- **Minimal memory overhead** with zero-copy operations

**Import Performance:**
- C++ import: ~3.5ms
- Rust import: ~0.1ms (25x faster)

## Development

For detailed build instructions, compiler requirements, and build options, see [BUILD.md](BUILD.md).

### Quick Development Setup

```bash
# Install Clang 20 from LLVM repository
wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | sudo apt-key add -
echo "deb http://apt.llvm.org/$(lsb_release -cs)/ llvm-toolchain-$(lsb_release -cs)-20 main" | sudo tee /etc/apt/sources.list.d/llvm.list
sudo apt update && sudo apt install clang-20

# Install Rust 1.86+
rustup toolchain install 1.86.0

# Development build with both implementations
pip install -e . -C cmake.define.BUILD_RUST=ON
```

## Compatibility

- **Python 3.8+**
- **Linux** (Ubuntu/Debian recommended)
- **100% compatible** with Matrix protocol specifications
- **Thread-safe** operations
- **Drop-in replacement** for existing Matrix crypto libraries

## Architecture

```
matrices_evolved/
├── cpp/                    # C++ nanobind implementation
│   ├── base64/            # SIMD-optimized base64 encoders/decoders
│   │   ├── encoders/      # Multiple encoder implementations
│   │   └── decoders/      # Multiple decoder implementations
│   ├── benchmark/         # Performance benchmarking suite
│   │   ├── simde/         # SIMD Everywhere submodule
│   │   └── full_bench.cpp # Comprehensive benchmarks
│   ├── crypto/            # Ed25519 cryptographic operations
│   │   └── ed25519.cpp    # Digital signatures & key management
│   ├── json/              # JSON canonicalization
│   │   └── canonicalization.cpp # SIMD-optimized JSON processing
│   ├── debug.h            # Debug infrastructure & shared definitions
│   ├── global.h           # Global includes & nanobind setup
│   └── lib.cpp            # Python bindings & module definition
├── rust/                  # Rust PyO3 implementation
│   ├── crypto/
│   │   ├── event_signing.rs # Ed25519 operations
│   │   ├── cache.rs       # LRU cache implementation
│   │   └── stream_change_cache.rs # Stream change tracking
│   ├── Cargo.toml
│   └── lib.rs
├── matrices_evolved/      # Python package
│   ├── __init__.py       # Auto-selection logic
│   ├── cpp.py            # C++ wrapper
│   └── rust.py           # Rust wrapper
└── CMakeLists.txt        # Build configuration
```
## In Progress

LRU, Stream Cache.
You would need a wrapper to function these properly. these expose core functionality in cache.rs and event_signing.rs but more needed on python wrapper.
the bindings are already there but currently not written in api docs.

# Synthetic Benchmarks
```
Pure Python LRU:
   synmark.suites.lrucache_auto: Mean +- std dev: 3.17 us +- 0.08 us
   synmark.suites.lrucache_evict_auto: Mean +- std dev: 3.06 us +- 0.08 us
   Rust Wrapper LRU:
   synmark.suites.lrucache_auto: Mean +- std dev: 900 ns +- 19 ns
   synmark.suites.lrucache_evict_auto: Mean +- std dev: 879 ns +- 20 ns

Formula:
   Improvement (%) = ((Python_time - Rust_time) / Python_time) × 100

Results:
    1. lrucache_auto:       ((3170 - 900) / 3170) × 100 ≈ 71.6% faster
    2. lrucache_evict_auto: ((3060 - 879) / 3060) × 100 ≈ 71.3% faster


crypto:
Pure rust bench:
    benches/crypto_bench.rs (/mnt/build/Tests/signal/hula/synapse/target/release/deps/crypto_bench-b4730a09c1fd000f)
    Gnuplot not found, using plotters backend
    Benchmarking pure_rust_canonicalization
    Benchmarking pure_rust_canonicalization: Warming up for 3.0000 s
    Benchmarking pure_rust_canonicalization: Collecting 100 samples in estimated 5.0027 s (9.1M iterations)
    Benchmarking pure_rust_canonicalization: Analyzing
    pure_rust_canonicalization
                        time:   [533.77 ns 534.46 ns 535.21 ns]
    Benchmarking pure_rust_ed25519_sign
    Benchmarking pure_rust_ed25519_sign: Warming up for 3.0000 s
    Benchmarking pure_rust_ed25519_sign: Collecting 100 samples in estimated 5.0250 s (601k iterations)
    Benchmarking pure_rust_ed25519_sign: Analyzing
    pure_rust_ed25519_sign  time:   [8.3256 µs 8.3447 µs 8.3633 µs]
    Found 2 outliers among 100 measurements (2.00%)
      2 (2.00%) high severe
    Benchmarking pure_rust_base64_encode
    Benchmarking pure_rust_base64_encode: Warming up for 3.0000 s
    Benchmarking pure_rust_base64_encode: Collecting 100 samples in estimated 5.0004 s (26M iterations)
    Benchmarking pure_rust_base64_encode: Analyzing
    pure_rust_base64_encode time:   [194.77 ns 195.08 ns 195.39 ns]
    Found 3 outliers among 100 measurements (3.00%)
      1 (1.00%) low severe
      2 (2.00%) high severe

Synmark bench Crypto:
Note: Python code probably has no keycaching as we did not write these functions. they are directly tested
against synapse provided functions.
Running PYTHON implementation:
    canonicalization: 1.99 μs per operation
    base64_encoding: 0.68 μs per operation
    base64_decoding: 1.34 μs per operation
    content_hash: 2.41 μs per operation
    json_signing: 19.75 μs per operation
    signature_verification: 48.15 μs per operation
    synmark.suites.crypto_operations_auto: Mean +- std dev: 73.0 us +- 1.5 us
Note: Now Rust implementation also uses keycaching but the vectorization is not done manually thus
the mean result may be a bit misleading in real life scenarios.
Running RUST implementation:
    canonicalization: 0.99 μs per operation
    base64_encoding: 4.74 μs per operation
    base64_decoding: 0.38 μs per operation
    content_hash: 0.99 μs per operation
    json_signing: 10.99 μs per operation
    signature_verification: 36.73 μs per operation
    synmark.suites.crypto_operations_auto: Mean +- std dev: 55.3 us +- 0.3 us

Note: C++ uses Vecorization if enables so synthetic benchmarks like synmark spawn many processes
usually in practice it would be faster due to continouous runs which reduces vecorization setup
times. Mainly in smaller inputs vectorizations is not recomended but we can utilize the workload
well in our scenatio, added private key caching improved the json signing but it is thread local
matrix signing usually uses one single key to sign so caching a key helps for key object processing
on openssl reduction on every call.
Note: C++ uses an iterative approach to json structure walk, Rust uses recursion.
Running C++ implementation:
    canonicalization: 0.60 μs per operation
    base64_encoding: 0.25 μs per operation
    base64_decoding: 0.16 μs per operation
    content_hash: 0.72 μs per operation
    json_signing: 9.69 μs per operation
    signature_verification: 35.68 μs per operation
    synmark.suites.crypto_operations_auto: Mean +- std dev: 46.5 us +- 1.3 us

    
Stream Change Bench Results :
Pure python:
    synmark.suites.stream_change_cache_auto: Mean +- std dev: 2.26 us +- 0.12 us
Rust Wrapper:
    synmark.suites.stream_change_cache_rust_auto: Mean +- std dev: 559 ns +- 28 ns
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

### Code Style

- **C++**: Follow C++23 best practices, use clang-format
- **Rust**: Follow rustfmt conventions, use clippy
- **Python**: Follow PEP 8, use black formatter

## License

See [LICENSE](LICENSE) file for details.

## Links

- **API Documentation**: [API.md](API.md)
- **Build Instructions**: [BUILD.md](BUILD.md)
- **Issues**: [GitHub Issues](https://github.com/alessblaze/matrices_evolved/issues)
- **Matrix Protocol**: [Matrix Specification](https://spec.matrix.org/)
