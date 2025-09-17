# Build Instructions

Detailed build instructions for matrices_evolved with both C++ and Rust implementations.

## Prerequisites

### System Requirements
- **Linux** (Ubuntu/Debian recommended)
- **Python 3.8+**
- **CMake 3.15+**

### C++ Implementation Requirements
- **Clang 20** (required for C++23 support)
- **nanobind 2.0+**

### Rust Implementation Requirements  
- **Rust 1.86+** (required for latest PyO3 features)
- **Cargo** (included with Rust)

## Installation

### Install Clang 20

Clang 20 is required for C++23 support. Install from the official LLVM repository:

```bash
# Add LLVM repository
wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | sudo apt-key add -
echo "deb http://apt.llvm.org/$(lsb_release -cs)/ llvm-toolchain-$(lsb_release -cs)-20 main" | sudo tee /etc/apt/sources.list.d/llvm.list

# Update and install Clang 20
sudo apt update
sudo apt install clang-20
```

For detailed instructions and other distributions, see: https://apt.llvm.org/

### Install Rust 1.86+

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup toolchain install 1.86.0
rustup default 1.86.0
```

## Build Options

### CMake Build Flags

The build system supports several CMake options:

```bash
# Build both implementations (default)
pip install . -C cmake.define.BUILD_RUST=ON -C cmake.define.BUILD_CPP=ON

# C++ only
pip install . -C cmake.define.BUILD_RUST=OFF -C cmake.define.BUILD_CPP=ON

# Rust only  
pip install . -C cmake.define.BUILD_RUST=ON -C cmake.define.BUILD_CPP=OFF
```

### Build Types

```bash
# Release build (default)
pip install . -C cmake.define.CMAKE_BUILD_TYPE=Release

# Debug build
pip install . -C cmake.define.CMAKE_BUILD_TYPE=Debug

# Development build (editable)
pip install -e . -C cmake.define.BUILD_RUST=ON
```

### Compiler Selection

By default, the build uses Clang 20. To override:

```bash
# Use specific Clang version
pip install . -C cmake.define.CMAKE_C_COMPILER=clang-20 -C cmake.define.CMAKE_CXX_COMPILER=clang++-20

# Use system default compiler (not recommended)
pip install . -C cmake.define.CMAKE_C_COMPILER=clang -C cmake.define.CMAKE_CXX_COMPILER=clang++
```

## Build Process

### Automatic Dependency Management

The build system automatically downloads and builds:

- **Boost.JSON 1.89.0** - JSON parsing and serialization
- **AWS-LC** - Cryptographic library (FIPS validated)

### Build Steps

1. **Configure**: CMake configures the build with specified options
2. **Download**: External dependencies are downloaded and cached
3. **Compile**: Dependencies are built with optimization flags
4. **Link**: Static libraries are linked into the Python modules
5. **Install**: Modules are installed to the Python environment

### Build Artifacts

**C++ Implementation:**
- `_event_signing_impl.cpython-*.so` - nanobind module

**Rust Implementation:**  
- `matrices_evolved_rust.cpython-*.so` - PyO3 module

## Optimization Flags

### Compiler Optimizations

The build applies aggressive optimization flags:

**Base Optimizations:**
- `-O3` - Maximum optimization level
- `-march=native` - Target current CPU architecture
- `-ffast-math` - Fast floating-point math
- `-flto` - Link-time optimization

**SIMD Instructions:**
- `-mavx2` - AVX2 vector instructions
- `-mfma` - Fused multiply-add
- `-mbmi2` - Bit manipulation instructions

**Clang-Specific:**
- `-flto=thin` - Thin link-time optimization
- `-fvectorize` - Auto-vectorization
- `-mllvm -polly` - Polyhedral loop optimization

### Static Linking

All dependencies are statically linked for portability:
- **AWS-LC** - Static cryptographic library
- **Boost.JSON** - Static JSON library
- **C++ Runtime** - Static linking with `-static` flag

## Troubleshooting

### Common Issues

**Clang 20 not found:**
```bash
# Install Clang 20
sudo apt install clang-20
# Or specify path
pip install . -C cmake.define.CMAKE_CXX_COMPILER=/usr/bin/clang++-20
```

**Rust version too old:**
```bash
rustup update
rustup toolchain install 1.86.0
rustup default 1.86.0
```

**CMake version too old:**
```bash
# Ubuntu
sudo apt install cmake
# Or install from pip
pip install cmake
```

### Build Logs

Enable verbose build output:
```bash
pip install . -v -C cmake.define.CMAKE_VERBOSE_MAKEFILE=ON
```

### Clean Build

Force clean rebuild:
```bash
pip install . --force-reinstall --no-cache-dir
```

## Development Build

### Editable Installation

```bash
pip install -e . -C cmake.define.BUILD_RUST=ON -C cmake.define.BUILD_CPP=ON
```

### Debug Build

```bash
pip install -e . -C cmake.define.CMAKE_BUILD_TYPE=Debug
```

### Testing Build

```bash
# Build and test
pip install -e .
python3 -c "import matrices_evolved; print(matrices_evolved.__implementations__)"
```

## Platform Notes

### Linux
- Requires Clang 20 from package manager
- Static linking works out of the box
- Tested on Ubuntu 20.04+ and Debian 11+

## Performance Tuning

### CPU-Specific Builds

```bash
# Generic build (portable)
pip install . -C cmake.define.CMAKE_CXX_FLAGS="-march=x86-64"

# CPU-specific build (faster)
pip install . -C cmake.define.CMAKE_CXX_FLAGS="-march=native"
```

### Memory Usage

Large builds may require additional memory:
```bash
# Limit parallel jobs
pip install . -C cmake.define.CMAKE_BUILD_PARALLEL_LEVEL=2
```

## Verification

### Test Installation

```bash
python3 -c "
import matrices_evolved
print('Available implementations:', matrices_evolved.__implementations__)

# Test basic functionality
key = matrices_evolved.generate_signing_key('ed25519')
verify_key = matrices_evolved.get_verify_key(key)
print(' Build successful')
"
```

### Performance Test

```bash
python3 -c "
import time
import matrices_evolved

# Test signing performance
key = matrices_evolved.generate_signing_key('ed25519')
event = {'type': 'm.test', 'content': {'body': 'test'}}

start = time.time()
for _ in range(1000):
    matrices_evolved.sign_json_object_fast(event, 'test.com', key, 'ed25519:v1')
elapsed = time.time() - start

print(f'Signed 1000 events in {elapsed:.3f}s ({1000/elapsed:.0f} ops/sec)')
"
```