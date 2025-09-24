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

#include "../include/ams-sse-aligned.h"

#if defined(__AVX2__) && !defined(DISABLE_SSE_BASE64_ENCODER_ALIGNED)

template<typename T, std::size_t Alignment = 16>
class aligned_allocator {
public:
    static_assert((Alignment & (Alignment - 1)) == 0, "Alignment must be a power of two");
    static_assert(Alignment >= alignof(T), "Alignment must be >= alignof(T)");

    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using size_type = std::size_t;

    template<typename U>
    struct rebind { using other = aligned_allocator<U, Alignment>; };

    using is_always_equal = std::true_type;

    aligned_allocator() = default;
    template<typename U>
    aligned_allocator(const aligned_allocator<U, Alignment>&) {}

    pointer allocate(size_type n) {
        if (n > std::numeric_limits<size_type>::max() / sizeof(T))
            throw std::bad_array_new_length();
        const std::size_t bytes = n * sizeof(T);
        void* p = ::operator new[](bytes, std::align_val_t(Alignment));
        return static_cast<pointer>(p);
    }

    void deallocate(pointer p, size_type) noexcept {
        ::operator delete[](p, std::align_val_t(Alignment));
    }

    template<typename U>
    bool operator==(const aligned_allocator<U, Alignment>&) const noexcept { return true; }
    template<typename U>
    bool operator!=(const aligned_allocator<U, Alignment>&) const noexcept { return false; }
};

thread_local std::string base64_buffer;
static constexpr char base64_chars[64] = {
    'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P',
    'Q','R','S','T','U','V','W','X','Y','Z','a','b','c','d','e','f',
    'g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v',
    'w','x','y','z','0','1','2','3','4','5','6','7','8','9','+','/'
};

static inline void sse2_copy_unaligned_to_aligned(void* dstv, const void* srcv, size_t n) {
    unsigned char*       dst = static_cast<unsigned char*>(dstv);
    const unsigned char* src = static_cast<const unsigned char*>(srcv);

    // Optional: assert 16B alignment for dst if desired
    // assert((reinterpret_cast<uintptr_t>(dst) & 15) == 0);

    // 64B unrolled body
    while (n >= 64) {
        __m128i a0 = _mm_loadu_si128((const __m128i*)(src +  0));
        __m128i a1 = _mm_loadu_si128((const __m128i*)(src + 16));
        __m128i a2 = _mm_loadu_si128((const __m128i*)(src + 32));
        __m128i a3 = _mm_loadu_si128((const __m128i*)(src + 48));
        _mm_store_si128((__m128i*)(dst +  0), a0);  // dst must be 16B-aligned
        _mm_store_si128((__m128i*)(dst + 16), a1);
        _mm_store_si128((__m128i*)(dst + 32), a2);
        _mm_store_si128((__m128i*)(dst + 48), a3);
        src += 64; dst += 64; n -= 64;
    }

    // 16B body
    while (n >= 16) {
        __m128i a = _mm_loadu_si128((const __m128i*)src);
        _mm_store_si128((__m128i*)dst, a);
        src += 16; dst += 16; n -= 16;
    }

    // Tail
    for (size_t i = 0; i < n; ++i) dst[i] = src[i];
}


// Thread-local aligned buffers using custom aligned allocator
thread_local std::vector<uint8_t, aligned_allocator<uint8_t, 16>> aligned_input_buffer;
thread_local std::vector<char, aligned_allocator<char, 16>> aligned_output_buffer;

/**
 * Aligned SSE Base64 Encoder (Memory-Optimized Implementation)
 * 
 * ALGORITHM:
 * 1. Uses Boost aligned allocators for optimal memory access
 * 2. Copies input to 16-byte aligned buffer
 * 3. Processes 48-byte blocks with complex unrolled operations
 * 4. Uses streaming stores for large data (≥8KB)
 * 
 * PERFORMANCE:
 * - Optimized for large datasets with streaming
 * - Alignment overhead hurts small inputs
 * - Memory copy cost reduces efficiency
 * 
 * TECHNICAL DETAILS:
 * - Boost aligned_allocator for 16-byte alignment
 * - Complex unrolled processing with pack/unpack operations
 * - Streaming stores (_mm_stream_si128) for cache bypass
 * - Memory fence (_mm_sfence) for store completion
 * 
 * @param data Input byte vector to encode
 * @return Base64 encoded string (unpadded)
 */
[[gnu::hot, gnu::flatten, clang::always_inline]] std::string fast_sse_base64_encode_aligned(const std::vector<uint8_t>& data) {
    size_t len = data.size();
    
    // Fast path for tiny inputs (< 16B) - avoid SIMD setup overhead
    if (len < 16) {
        std::string result;
        result.reserve(((len + 2) / 3) * 4);
        const uint8_t* src = data.data();
        
        while (len >= 3) {
            uint32_t val = (src[0] << 16) | (src[1] << 8) | src[2];
            result += base64_chars[(val >> 18) & 63];
            result += base64_chars[(val >> 12) & 63];
            result += base64_chars[(val >> 6) & 63];
            result += base64_chars[val & 63];
            src += 3; len -= 3;
        }
        if (len > 0) {
            uint32_t val = src[0] << 16;
            if (len > 1) val |= src[1] << 8;
            result += base64_chars[(val >> 18) & 63];
            result += base64_chars[(val >> 12) & 63];
            if (len > 1) result += base64_chars[(val >> 6) & 63];
        }
        return result;
    }
    
    size_t out_len = ((len + 2) / 3) * 4;
    
    // Prevent integer overflow in buffer calculations
    if (len > SIZE_MAX - 16) {
        throw std::runtime_error("Input too large for aligned buffer");
    }
    if (out_len > SIZE_MAX - 64) {
        throw std::runtime_error("Output too large for aligned buffer");
    }
    
    // Ensure aligned buffers are large enough
    size_t input_buf_size = len + 16;
    size_t output_buf_size = out_len + 64;
    
    if (aligned_input_buffer.size() < input_buf_size) {
        aligned_input_buffer.resize(input_buf_size);
    }
    if (aligned_output_buffer.size() < output_buf_size) {
        aligned_output_buffer.resize(output_buf_size);
    }
    
    // Validate buffer size before memcpy
    if (len > aligned_input_buffer.size()) {
        throw std::runtime_error("Buffer overflow prevented in memcpy");
    }
    
    // Copy input data to aligned buffer
    std::memcpy(aligned_input_buffer.data(), data.data(), len);
    
    const uint8_t* src = aligned_input_buffer.data();
    char* dest = aligned_output_buffer.data();
    const char* const dest_orig = dest;
    
    // SIMD constants
    static const __m128i mask6 = _mm_set1_epi32(0x3f);
    static const __m128i trip_shuffle = _mm_setr_epi8(
        2, 1, 0, (char)0x80, 5, 4, 3, (char)0x80,
        8, 7, 6, (char)0x80, 11,10, 9, (char)0x80
    );
    static const __m128i lut0 = _mm_setr_epi8('A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P');
    static const __m128i lut1 = _mm_setr_epi8('Q','R','S','T','U','V','W','X','Y','Z','a','b','c','d','e','f');
    static const __m128i lut2 = _mm_setr_epi8('g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v');
    static const __m128i lut3 = _mm_setr_epi8('w','x','y','z','0','1','2','3','4','5','6','7','8','9','+','/');
    static const __m128i const16 = _mm_set1_epi8(16);
    static const __m128i const32 = _mm_set1_epi8(32);
    static const __m128i const48 = _mm_set1_epi8(48);
    
    // Process 48-byte blocks with streaming stores for large data
    bool use_streaming = (data.size() >= 8192); // Use streaming for large inputs (8KB+)
    
    while (len >= 48) {
        // Process 48 bytes as 4 aligned 12-byte chunks → 4×16-byte outputs
        __m128i results[4];
        
        // Load 4 aligned 16-byte blocks (covers 48 input + 16 padding)
        __m128i block0 = _mm_load_si128(reinterpret_cast<const __m128i*>(src));
        __m128i block1 = _mm_load_si128(reinterpret_cast<const __m128i*>(src + 16));
        __m128i block2 = _mm_load_si128(reinterpret_cast<const __m128i*>(src + 32));
        
        // Extract 12-byte chunks for base64 processing
        // Chunk 0: bytes 0-11
        __m128i chunk0 = block0;
        
        // Chunk 1: bytes 12-23 (4 bytes from block0 + 8 bytes from block1)
        __m128i chunk1 = _mm_alignr_epi8(block1, block0, 12);
        
        // Chunk 2: bytes 24-35 (8 bytes from block1 + 4 bytes from block2)
        __m128i chunk2 = _mm_alignr_epi8(block2, block1, 8);
        
        // Chunk 3: bytes 36-47 (12 bytes from block2)
        __m128i chunk3 = _mm_srli_si128(block2, 4);
        
        __m128i chunks[4] = {chunk0, chunk1, chunk2, chunk3};
        
        // Unrolled processing of all 4 chunks in parallel
        __m128i packed0 = _mm_shuffle_epi8(chunk0, trip_shuffle);
        __m128i packed1 = _mm_shuffle_epi8(chunk1, trip_shuffle);
        __m128i packed2 = _mm_shuffle_epi8(chunk2, trip_shuffle);
        __m128i packed3 = _mm_shuffle_epi8(chunk3, trip_shuffle);
        
        __m128i idx0_0 = _mm_and_si128(_mm_srli_epi32(packed0, 18), mask6);
        __m128i idx0_1 = _mm_and_si128(_mm_srli_epi32(packed1, 18), mask6);
        __m128i idx0_2 = _mm_and_si128(_mm_srli_epi32(packed2, 18), mask6);
        __m128i idx0_3 = _mm_and_si128(_mm_srli_epi32(packed3, 18), mask6);
        
        __m128i idx1_0 = _mm_and_si128(_mm_srli_epi32(packed0, 12), mask6);
        __m128i idx1_1 = _mm_and_si128(_mm_srli_epi32(packed1, 12), mask6);
        __m128i idx1_2 = _mm_and_si128(_mm_srli_epi32(packed2, 12), mask6);
        __m128i idx1_3 = _mm_and_si128(_mm_srli_epi32(packed3, 12), mask6);
        
        __m128i idx2_0 = _mm_and_si128(_mm_srli_epi32(packed0, 6), mask6);
        __m128i idx2_1 = _mm_and_si128(_mm_srli_epi32(packed1, 6), mask6);
        __m128i idx2_2 = _mm_and_si128(_mm_srli_epi32(packed2, 6), mask6);
        __m128i idx2_3 = _mm_and_si128(_mm_srli_epi32(packed3, 6), mask6);
        
        __m128i idx3_0 = _mm_and_si128(packed0, mask6);
        __m128i idx3_1 = _mm_and_si128(packed1, mask6);
        __m128i idx3_2 = _mm_and_si128(packed2, mask6);
        __m128i idx3_3 = _mm_and_si128(packed3, mask6);
        
        // Process chunk 0
        __m128i lo01_0 = _mm_unpacklo_epi32(idx0_0, idx1_0);
        __m128i lo23_0 = _mm_unpacklo_epi32(idx2_0, idx3_0);
        __m128i hi01_0 = _mm_unpackhi_epi32(idx0_0, idx1_0);
        __m128i hi23_0 = _mm_unpackhi_epi32(idx2_0, idx3_0);
        __m128i quad0_0 = _mm_unpacklo_epi64(lo01_0, lo23_0);
        __m128i quad1_0 = _mm_unpackhi_epi64(lo01_0, lo23_0);
        __m128i quad2_0 = _mm_unpacklo_epi64(hi01_0, hi23_0);
        __m128i quad3_0 = _mm_unpackhi_epi64(hi01_0, hi23_0);
        __m128i packed01_0 = _mm_packs_epi32(quad0_0, quad1_0);
        __m128i packed23_0 = _mm_packs_epi32(quad2_0, quad3_0);
        __m128i indices_bytes_0 = _mm_packus_epi16(packed01_0, packed23_0);
        __m128i mask0_0 = _mm_cmplt_epi8(indices_bytes_0, const16);
        __m128i mask1_0 = _mm_cmplt_epi8(indices_bytes_0, const32);
        __m128i mask2_0 = _mm_cmplt_epi8(indices_bytes_0, const48);
        __m128i val0_0 = _mm_shuffle_epi8(lut0, indices_bytes_0);
        __m128i val1_0 = _mm_shuffle_epi8(lut1, _mm_sub_epi8(indices_bytes_0, const16));
        __m128i val2_0 = _mm_shuffle_epi8(lut2, _mm_sub_epi8(indices_bytes_0, const32));
        __m128i val3_0 = _mm_shuffle_epi8(lut3, _mm_sub_epi8(indices_bytes_0, const48));
        __m128i tmp0_0 = _mm_blendv_epi8(val1_0, val0_0, mask0_0);
        __m128i tmp1_0 = _mm_blendv_epi8(val3_0, val2_0, mask2_0);
        results[0] = _mm_blendv_epi8(tmp1_0, tmp0_0, mask1_0);
        
        // Process chunk 1
        __m128i lo01_1 = _mm_unpacklo_epi32(idx0_1, idx1_1);
        __m128i lo23_1 = _mm_unpacklo_epi32(idx2_1, idx3_1);
        __m128i hi01_1 = _mm_unpackhi_epi32(idx0_1, idx1_1);
        __m128i hi23_1 = _mm_unpackhi_epi32(idx2_1, idx3_1);
        __m128i quad0_1 = _mm_unpacklo_epi64(lo01_1, lo23_1);
        __m128i quad1_1 = _mm_unpackhi_epi64(lo01_1, lo23_1);
        __m128i quad2_1 = _mm_unpacklo_epi64(hi01_1, hi23_1);
        __m128i quad3_1 = _mm_unpackhi_epi64(hi01_1, hi23_1);
        __m128i packed01_1 = _mm_packs_epi32(quad0_1, quad1_1);
        __m128i packed23_1 = _mm_packs_epi32(quad2_1, quad3_1);
        __m128i indices_bytes_1 = _mm_packus_epi16(packed01_1, packed23_1);
        __m128i mask0_1 = _mm_cmplt_epi8(indices_bytes_1, const16);
        __m128i mask1_1 = _mm_cmplt_epi8(indices_bytes_1, const32);
        __m128i mask2_1 = _mm_cmplt_epi8(indices_bytes_1, const48);
        __m128i val0_1 = _mm_shuffle_epi8(lut0, indices_bytes_1);
        __m128i val1_1 = _mm_shuffle_epi8(lut1, _mm_sub_epi8(indices_bytes_1, const16));
        __m128i val2_1 = _mm_shuffle_epi8(lut2, _mm_sub_epi8(indices_bytes_1, const32));
        __m128i val3_1 = _mm_shuffle_epi8(lut3, _mm_sub_epi8(indices_bytes_1, const48));
        __m128i tmp0_1 = _mm_blendv_epi8(val1_1, val0_1, mask0_1);
        __m128i tmp1_1 = _mm_blendv_epi8(val3_1, val2_1, mask2_1);
        results[1] = _mm_blendv_epi8(tmp1_1, tmp0_1, mask1_1);
        
        // Process chunk 2
        __m128i lo01_2 = _mm_unpacklo_epi32(idx0_2, idx1_2);
        __m128i lo23_2 = _mm_unpacklo_epi32(idx2_2, idx3_2);
        __m128i hi01_2 = _mm_unpackhi_epi32(idx0_2, idx1_2);
        __m128i hi23_2 = _mm_unpackhi_epi32(idx2_2, idx3_2);
        __m128i quad0_2 = _mm_unpacklo_epi64(lo01_2, lo23_2);
        __m128i quad1_2 = _mm_unpackhi_epi64(lo01_2, lo23_2);
        __m128i quad2_2 = _mm_unpacklo_epi64(hi01_2, hi23_2);
        __m128i quad3_2 = _mm_unpackhi_epi64(hi01_2, hi23_2);
        __m128i packed01_2 = _mm_packs_epi32(quad0_2, quad1_2);
        __m128i packed23_2 = _mm_packs_epi32(quad2_2, quad3_2);
        __m128i indices_bytes_2 = _mm_packus_epi16(packed01_2, packed23_2);
        __m128i mask0_2 = _mm_cmplt_epi8(indices_bytes_2, const16);
        __m128i mask1_2 = _mm_cmplt_epi8(indices_bytes_2, const32);
        __m128i mask2_2 = _mm_cmplt_epi8(indices_bytes_2, const48);
        __m128i val0_2 = _mm_shuffle_epi8(lut0, indices_bytes_2);
        __m128i val1_2 = _mm_shuffle_epi8(lut1, _mm_sub_epi8(indices_bytes_2, const16));
        __m128i val2_2 = _mm_shuffle_epi8(lut2, _mm_sub_epi8(indices_bytes_2, const32));
        __m128i val3_2 = _mm_shuffle_epi8(lut3, _mm_sub_epi8(indices_bytes_2, const48));
        __m128i tmp0_2 = _mm_blendv_epi8(val1_2, val0_2, mask0_2);
        __m128i tmp1_2 = _mm_blendv_epi8(val3_2, val2_2, mask2_2);
        results[2] = _mm_blendv_epi8(tmp1_2, tmp0_2, mask1_2);
        
        // Process chunk 3
        __m128i lo01_3 = _mm_unpacklo_epi32(idx0_3, idx1_3);
        __m128i lo23_3 = _mm_unpacklo_epi32(idx2_3, idx3_3);
        __m128i hi01_3 = _mm_unpackhi_epi32(idx0_3, idx1_3);
        __m128i hi23_3 = _mm_unpackhi_epi32(idx2_3, idx3_3);
        __m128i quad0_3 = _mm_unpacklo_epi64(lo01_3, lo23_3);
        __m128i quad1_3 = _mm_unpackhi_epi64(lo01_3, lo23_3);
        __m128i quad2_3 = _mm_unpacklo_epi64(hi01_3, hi23_3);
        __m128i quad3_3 = _mm_unpackhi_epi64(hi01_3, hi23_3);
        __m128i packed01_3 = _mm_packs_epi32(quad0_3, quad1_3);
        __m128i packed23_3 = _mm_packs_epi32(quad2_3, quad3_3);
        __m128i indices_bytes_3 = _mm_packus_epi16(packed01_3, packed23_3);
        __m128i mask0_3 = _mm_cmplt_epi8(indices_bytes_3, const16);
        __m128i mask1_3 = _mm_cmplt_epi8(indices_bytes_3, const32);
        __m128i mask2_3 = _mm_cmplt_epi8(indices_bytes_3, const48);
        __m128i val0_3 = _mm_shuffle_epi8(lut0, indices_bytes_3);
        __m128i val1_3 = _mm_shuffle_epi8(lut1, _mm_sub_epi8(indices_bytes_3, const16));
        __m128i val2_3 = _mm_shuffle_epi8(lut2, _mm_sub_epi8(indices_bytes_3, const32));
        __m128i val3_3 = _mm_shuffle_epi8(lut3, _mm_sub_epi8(indices_bytes_3, const48));
        __m128i tmp0_3 = _mm_blendv_epi8(val1_3, val0_3, mask0_3);
        __m128i tmp1_3 = _mm_blendv_epi8(val3_3, val2_3, mask2_3);
        results[3] = _mm_blendv_epi8(tmp1_3, tmp0_3, mask1_3);
        
        // Store all 4 results with aligned streaming
        if (use_streaming) {
            _mm_stream_si128(reinterpret_cast<__m128i*>(dest), results[0]);
            _mm_stream_si128(reinterpret_cast<__m128i*>(dest + 16), results[1]);
            _mm_stream_si128(reinterpret_cast<__m128i*>(dest + 32), results[2]);
            _mm_stream_si128(reinterpret_cast<__m128i*>(dest + 48), results[3]);
        } else {
            _mm_store_si128(reinterpret_cast<__m128i*>(dest), results[0]);
            _mm_store_si128(reinterpret_cast<__m128i*>(dest + 16), results[1]);
            _mm_store_si128(reinterpret_cast<__m128i*>(dest + 32), results[2]);
            _mm_store_si128(reinterpret_cast<__m128i*>(dest + 48), results[3]);
        }
        
        src += 48;
        dest += 64;
        len -= 48;
    }
    
    // Ensure all streaming stores are completed before returning
    if (use_streaming) {
        _mm_sfence();
    }
    
    // Scalar fallback for remaining bytes
    while (len >= 3) {
        uint32_t val = (src[0] << 16) | (src[1] << 8) | src[2];
        *dest++ = base64_chars[(val >> 18) & 63];
        *dest++ = base64_chars[(val >> 12) & 63];
        *dest++ = base64_chars[(val >> 6) & 63];
        *dest++ = base64_chars[val & 63];
        src += 3;
        len -= 3;
    }
    
    if (len > 0) {
        uint32_t val = src[0] << 16;
        if (len > 1) val |= src[1] << 8;
        *dest++ = base64_chars[(val >> 18) & 63];
        *dest++ = base64_chars[(val >> 12) & 63];
        if (len > 1) *dest++ = base64_chars[(val >> 6) & 63];
    }
    
    return std::string(dest_orig, dest - dest_orig);
}

/**
 * Aligned SSE Base64 Encoder (Streaming Implementation)
 * 
 * ALGORITHM:
 * 1. Tiny inputs (≤24B): Pure scalar to avoid SIMD overhead
 * 2. Large inputs: 48-byte SSE blocks with aligned operations
 *    - Uses aligned loads (_mm_load_si128) from thread-local buffer
 *    - Processes 4×12-byte chunks with triplet extraction
 *    - Uses streaming stores (_mm_stream_si128) to bypass cache
 *    - Memory fence (_mm_sfence) ensures store completion
 * 3. Remainder: Scalar processing for <48 bytes
 * 
 * PERFORMANCE:
 * - Optimal for large datasets with streaming stores
 * - Aligned operations maximize memory bandwidth
 * - Cache bypass reduces memory pressure
 * 
 * TECHNICAL DETAILS:
 * - Uses same SSE helper functions as regular SSE encoder
 * - Aligned loads/stores require 16-byte aligned buffers
 * - Streaming stores ideal for write-once large data
 * - Memory fence prevents reordering of streaming operations
 * 
 * @param data Input byte vector to encode
 * @return Base64 encoded string (unpadded)
 */

// Alternative approach using multiple specific masks - currently unused
// This approach uses different masks for each 6-bit extraction but is slower
// due to increased register pressure and less optimal instruction scheduling
static inline __m128i extract_indices_to_bytes_alt(const __m128i& packed) {
    __m128i hi  = _mm_srli_epi32(_mm_and_si128(packed, _mm_set1_epi32(0x00fc0000)), 18);
    __m128i mid = _mm_srli_epi32(_mm_and_si128(packed, _mm_set1_epi32(0x0003f000)), 4);
    __m128i lo  = _mm_slli_epi32(_mm_and_si128(packed, _mm_set1_epi32(0x00000fc0)), 10);
    __m128i bot = _mm_slli_epi32(_mm_and_si128(packed, _mm_set1_epi32(0x0000003f)), 24);
    return _mm_or_si128(_mm_or_si128(hi, mid), _mm_or_si128(lo, bot));
}

// Current optimized approach using single reusable mask
[[clang::always_inline]] static inline __m128i extract_indices_to_bytes(const __m128i& packed) {
    static const __m128i mask6 = _mm_set1_epi32(0x3f);
    __m128i idx0 = _mm_and_si128(_mm_srli_epi32(packed, 18), mask6);
    __m128i idx1 = _mm_slli_epi32(_mm_and_si128(_mm_srli_epi32(packed, 12), mask6), 8);
    __m128i idx2 = _mm_slli_epi32(_mm_and_si128(_mm_srli_epi32(packed, 6), mask6), 16);
    __m128i idx3 = _mm_slli_epi32(_mm_and_si128(packed, mask6), 24);
    return _mm_or_si128(_mm_or_si128(idx0, idx1), _mm_or_si128(idx2, idx3));
}

[[clang::always_inline]] static inline __m128i lut_lookup(const __m128i& indices) {
    static const __m128i lut0 = _mm_setr_epi8('A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P');
    static const __m128i lut1 = _mm_setr_epi8('Q','R','S','T','U','V','W','X','Y','Z','a','b','c','d','e','f');
    static const __m128i lut2 = _mm_setr_epi8('g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v');
    static const __m128i lut3 = _mm_setr_epi8('w','x','y','z','0','1','2','3','4','5','6','7','8','9','+','/');
    static const __m128i const16 = _mm_set1_epi8(16);
    static const __m128i const32 = _mm_set1_epi8(32);
    static const __m128i const48 = _mm_set1_epi8(48);
    
    __m128i mask_32 = _mm_cmplt_epi8(indices, const32);
    __m128i lo_result = _mm_blendv_epi8(
        _mm_shuffle_epi8(lut1, _mm_sub_epi8(indices, const16)),
        _mm_shuffle_epi8(lut0, indices),
        _mm_cmplt_epi8(indices, const16)
    );
    __m128i hi_result = _mm_blendv_epi8(
        _mm_shuffle_epi8(lut3, _mm_sub_epi8(indices, const48)),
        _mm_shuffle_epi8(lut2, _mm_sub_epi8(indices, const32)),
        _mm_cmplt_epi8(indices, const48)
    );
    return _mm_blendv_epi8(hi_result, lo_result, mask_32);
}


[[gnu::hot, gnu::flatten, clang::always_inline]] std::string fast_sse_base64_encode_aligned_alt(const std::vector<uint8_t>& data) {
    if (debug_enabled) {
        DEBUG_LOG("fast_sse_base64_encode_aligned_alt called with " + std::to_string(data.size()) + " bytes");
    }
    size_t len = data.size();
    
    // Fast path for tiny inputs - avoid SIMD overhead
    if (len <= 24) {
        std::string result;
        result.reserve(((len + 2) / 3) * 4);
        const uint8_t* src = data.data();
        
        while (len >= 3) {
            uint32_t val = (src[0] << 16) | (src[1] << 8) | src[2];
            result += base64_chars[(val >> 18) & 63];
            result += base64_chars[(val >> 12) & 63];
            result += base64_chars[(val >> 6) & 63];
            result += base64_chars[val & 63];
            src += 3; len -= 3;
        }
        if (len > 0) {
            uint32_t val = src[0] << 16;
            if (len > 1) val |= src[1] << 8;
            result += base64_chars[(val >> 18) & 63];
            result += base64_chars[(val >> 12) & 63];
            if (len > 1) result += base64_chars[(val >> 6) & 63];
        }
        return result;
    }
    
    size_t out_len = ((len + 2) / 3) * 4;
    
    // Ensure aligned buffers are large enough (need extra padding for aligned loads)
    size_t input_buf_size = len + 32;  // Extra padding for aligned loads past end
    size_t output_buf_size = out_len + 64;
    
    if (aligned_input_buffer.size() < input_buf_size) {
        aligned_input_buffer.resize(input_buf_size);
    }
    if (aligned_output_buffer.size() < output_buf_size) {
        aligned_output_buffer.resize(output_buf_size);
    }
    
    // Check if input data is already aligned
    bool input_is_aligned = (reinterpret_cast<uintptr_t>(data.data()) % 16 == 0);
    if (debug_enabled) {
        DEBUG_LOG("Alt: Input data alignment check: " + std::to_string(input_is_aligned) + " (addr: " + std::to_string(reinterpret_cast<uintptr_t>(data.data())) + ")");
    }
    
    const uint8_t* src;
    bool use_aligned_loads;
    if (input_is_aligned) {
        // Use original data directly if aligned
        src = data.data();
        use_aligned_loads = true;
        if (debug_enabled) {
            DEBUG_LOG("Alt: Using original aligned data directly");
        }
    } else {
        // Copy input data to aligned buffer using optimized SSE2 copy
        if (debug_enabled) {
            DEBUG_LOG("Alt: Copying " + std::to_string(len) + " bytes to aligned buffer with SSE2");
        }
        sse2_copy_unaligned_to_aligned(aligned_input_buffer.data(), data.data(), len);
        src = aligned_input_buffer.data();
        use_aligned_loads = true;
    }
    char* dest = aligned_output_buffer.data();
    const char* const dest_orig = dest;
    
    if (debug_enabled) {
        DEBUG_LOG("Alt: Buffer setup complete - src aligned: " + std::to_string(reinterpret_cast<uintptr_t>(src) % 16 == 0));
    }

    static const __m128i trip_shuffle = _mm_setr_epi8(
        2, 1, 0, (char)0x80,   // lane0: bytes 2,1,0 to match (b0<<16)|(b1<<8)|b2
        5, 4, 3, (char)0x80,   // lane1: bytes 5,4,3
        8, 7, 6, (char)0x80,   // lane2: bytes 8,7,6
       11,10, 9, (char)0x80    // lane3: bytes 11,10,9
    );

    // Use streaming stores for large data (≥8KB)
    bool use_streaming = (data.size() >= 8192);
    
    // Process 48-byte blocks with single loop
    if (debug_enabled && len >= 48) {
        DEBUG_LOG("SSE Aligned Alt encoder: processing " + std::to_string(len) + " bytes with SIMD");
    }
    while (len >= 48) {
        if (debug_enabled) {
            DEBUG_LOG("Alt: Processing 48-byte block, remaining: " + std::to_string(len));
        }
        // Load 3 16-byte blocks (covers 48 input bytes)
        __m128i block0, block1, block2;
        if (use_aligned_loads) {
            block0 = _mm_load_si128(reinterpret_cast<const __m128i*>(src));
            block1 = _mm_load_si128(reinterpret_cast<const __m128i*>(src + 16));
            block2 = _mm_load_si128(reinterpret_cast<const __m128i*>(src + 32));
        } else {
            block0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src));
            block1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + 16));
            block2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + 32));
        }
        
        if (debug_enabled) {
            DEBUG_LOG("Alt: Loaded 3 " + std::string(use_aligned_loads ? "aligned" : "unaligned") + " blocks");
        }
        
        // Extract 12-byte chunks using alignr
        __m128i chunk0 = block0;                                    // bytes 0-11
        __m128i chunk1 = _mm_alignr_epi8(block1, block0, 12);      // bytes 12-23
        __m128i chunk2 = _mm_alignr_epi8(block2, block1, 8);       // bytes 24-35
        __m128i chunk3 = _mm_srli_si128(block2, 4);                // bytes 36-47
        
        __m128i packed0 = _mm_shuffle_epi8(chunk0, trip_shuffle);
        __m128i packed1 = _mm_shuffle_epi8(chunk1, trip_shuffle);
        __m128i packed2 = _mm_shuffle_epi8(chunk2, trip_shuffle);
        __m128i packed3 = _mm_shuffle_epi8(chunk3, trip_shuffle);
        
        __m128i idx0_unpacked = extract_indices_to_bytes(packed0);
        __m128i idx1_unpacked = extract_indices_to_bytes(packed1);
        __m128i idx2_unpacked = extract_indices_to_bytes(packed2);
        __m128i idx3_unpacked = extract_indices_to_bytes(packed3);
        
        __m128i chars0 = lut_lookup(idx0_unpacked);
        __m128i chars1 = lut_lookup(idx1_unpacked);
        __m128i chars2 = lut_lookup(idx2_unpacked);
        __m128i chars3 = lut_lookup(idx3_unpacked);
        
        if (use_streaming) {
            _mm_stream_si128(reinterpret_cast<__m128i*>(dest + 0), chars0);
            _mm_stream_si128(reinterpret_cast<__m128i*>(dest + 16), chars1);
            _mm_stream_si128(reinterpret_cast<__m128i*>(dest + 32), chars2);
            _mm_stream_si128(reinterpret_cast<__m128i*>(dest + 48), chars3);
        } else {
            _mm_store_si128(reinterpret_cast<__m128i*>(dest + 0), chars0);
            _mm_store_si128(reinterpret_cast<__m128i*>(dest + 16), chars1);
            _mm_store_si128(reinterpret_cast<__m128i*>(dest + 32), chars2);
            _mm_store_si128(reinterpret_cast<__m128i*>(dest + 48), chars3);
        }
        
        src += 48;
        dest += 64;
        len -= 48;
        if (debug_enabled) {
            DEBUG_LOG("Alt: Block complete, remaining: " + std::to_string(len));
        }
    }
    
    // Ensure all streaming stores are completed
    if (use_streaming) {
        if (debug_enabled) {
            DEBUG_LOG("Alt: Memory fence for streaming stores");
        }
        _mm_sfence();
    }
    
    // Fallback scalar processing
    while (len >= 3) {
        uint32_t val = (src[0] << 16) | (src[1] << 8) | src[2];
        *dest++ = base64_chars[(val >> 18) & 63];
        *dest++ = base64_chars[(val >> 12) & 63];
        *dest++ = base64_chars[(val >> 6) & 63];
        *dest++ = base64_chars[val & 63];
        src += 3;
        len -= 3;
    }
    
    // Handle final 1-2 bytes
    if (len > 0) {
        uint32_t val = src[0] << 16;
        if (len > 1) val |= src[1] << 8;
        *dest++ = base64_chars[(val >> 18) & 63];
        *dest++ = base64_chars[(val >> 12) & 63];
        if (len > 1) *dest++ = base64_chars[(val >> 6) & 63];
    }
    
    size_t actual_len = dest - dest_orig;
    
    if (debug_enabled) {
        DEBUG_LOG("Alt: Complete - output length: " + std::to_string(actual_len));
    }
    
    return std::string(dest_orig, actual_len);
}
#endif