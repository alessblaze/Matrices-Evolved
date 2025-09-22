# Base64 Encoder Benchmark

## How to Run Benchmarks

### Prerequisites
- **Clang 20+** (required for C++23 support)
- **OpenSSL development libraries**
- **x86_64 CPU with AVX2 support** (for SIMD implementations)

### Build and Run
```bash
# Build the benchmark
clang-20 -std=c++23 -O3 -march=native -msse4.2 -mavx2 -Wall -Wextra \
  -o full_bench full_bench.cpp -lssl -lcrypto -lstdc++

# Run benchmark
./full_bench
```

### Architecture Dependencies
- **Results are highly architecture-dependent**
- **CPU model**, **cache sizes**, and **memory bandwidth** significantly affect performance
- **Thermal throttling** may impact large input benchmarks
- **Different compilers** (GCC vs Clang) may produce different results
- **SIMD support** varies by CPU generation (SSE4.2, AVX2, AVX-512)

## Sample Results (Intel x86_64 with AVX2)

### Test Configuration
- **Compiler**: clang-20 with C++23
- **Optimization**: -O3 -march=native -msse4.2 -mavx2
- **Platform**: x86_64 with AVX2 support

## Performance Summary

### Overall Rankings (by average performance)
1. **AVX2 Custom**: 4,023.85 μs average (fastest)
2. **SSE**: 4,144.29 μs average
3. **AVX2 Lemire**: 4,210.82 μs average
4. **Mulla SSE**: 4,385.09 μs average
5. **SSE Aligned**: 5,254.03 μs average
6. **OpenSSL**: 19,138.02 μs average (baseline)

### Key Findings

- **AVX2 Custom** is the fastest overall implementation, providing **4.8x speedup** over OpenSSL
- **AVX2 Lemire** excels at medium sizes (256B-8KB) with best cycles/byte ratios
- **SSE implementations** provide consistent 4-5x speedup across all sizes
- **OpenSSL** performance degrades significantly on very large inputs (>1MB)

## Implementation Characteristics

### AVX2 Custom
- **Best for**: Large inputs (>2MB)
- **Strengths**: Excellent large-scale performance, optimized data layout
- **Weaknesses**: Higher overhead for small inputs

### AVX2 Lemire
- **Best for**: Medium inputs (128B-1MB)
- **Strengths**: Lowest cycles/byte ratios, research-optimized algorithms
- **Weaknesses**: Complex implementation, maskload dependency

### SSE Implementations
- **Best for**: Consistent performance across all sizes
- **Strengths**: Good balance of speed and simplicity
- **Weaknesses**: Lower peak performance than AVX2

### Performance Notes

- **Synthetic benchmarks**: Results may vary ±5% in real-world usage
- **Thermal effects**: Large input tests may show thermal throttling
- **Cache effects**: Performance varies significantly with data locality
- **SIMD overhead**: Small inputs (<48B) often favor scalar implementations

## Compiler and Build Information

```bash
clang-20 -std=c++23 -O3 -march=native -msse4.2 -mavx2 -Wall -Wextra \
  -o full_bench full_bench.cpp -lssl -lcrypto -lstdc++
```

## Benchmark Interpretation

### Understanding Results
- **μs (microseconds)**: Absolute execution time
- **cyc/B (cycles per byte)**: CPU cycles per input byte (size-independent metric)
- **Speedup**: Performance improvement over OpenSSL baseline
- **±5% variance**: Expected in real-world usage due to system conditions

### Customizing Benchmarks
Edit `full_bench.cpp` to:
- **Add new test sizes**: Modify `test_configs` array
- **Change iteration counts**: Adjust `iterations` per test size
- **Add implementations**: Include new encoder functions
- **Modify test data**: Change data generation patterns

### Platform-Specific Notes
- **ARM processors**: May need different SIMD implementations (NEON)
- **Older x86 CPUs**: Disable AVX2 flags, use SSE-only builds
- **Different OSes**: Linking flags may vary (Windows: `-lssl -lcrypto`)

## Usage Recommendations (Architecture-Dependent)

**For x86_64 with AVX2:**
- **Cryptographic keys/signatures** (≤64B): Use AVX2 Lemire
- **JSON payloads** (256B-8KB): Use AVX2 Lemire  
- **Large files** (>2MB): Use AVX2 Custom
- **General purpose**: Use SSE for consistent performance
- **Fallback**: OpenSSL for maximum compatibility

**For other architectures:** Run benchmarks to determine optimal implementation