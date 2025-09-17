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
- **High Performance** - 10-100x faster than pure Python implementations

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

```bash
pip install matrices_evolved
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
│   └── event_signing.cpp
├── rust/                   # Rust PyO3 implementation
│   ├── crypto/
│   │   ├── event_signing.rs
│   │   ├── cache.rs
│   │   └── stream_change_cache.rs
│   ├── Cargo.toml
│   └── lib.rs
├── matrices_evolved/       # Python package
│   ├── __init__.py        # Auto-selection logic
│   ├── cpp.py             # C++ wrapper
│   └── rust.py            # Rust wrapper
└── CMakeLists.txt         # Build configuration
```
## In Progress
LRU, Stream Cache.
You would need a wrapper to function these properly. these expose core functionality in cache.rs and event_signing.rs but more needed on python wrapper.
the bindings are already there but currently not written in api docs.
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

AGPL-3.0 - See [LICENSE](LICENSE) file for details.

## Links

- **API Documentation**: [API.md](API.md)
- **Build Instructions**: [BUILD.md](BUILD.md)
- **Issues**: [GitHub Issues](https://github.com/alessblaze/matrices_evolved/issues)
- **Matrix Protocol**: [Matrix Specification](https://spec.matrix.org/)
