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

#include "decoder.h"



namespace nb = nanobind;

// Correct UTF-8 validator with scalar fallback
bool validate_utf8_simd(const char* data, size_t len) {
    size_t pos = 0;
    while (pos < len) {
        unsigned char c = static_cast<unsigned char>(data[pos]);
        if (c < 0x80) { pos++; continue; }
        if ((c & 0xE0) == 0xC0) {
            if (pos + 1 >= len || (data[pos + 1] & 0xC0) != 0x80 || c < 0xC2) return false;
            pos += 2;
        } else if ((c & 0xF0) == 0xE0) {
            if (pos + 2 >= len || (data[pos + 1] & 0xC0) != 0x80 || (data[pos + 2] & 0xC0) != 0x80) return false;
            uint32_t cp = ((c & 0x0F) << 12) | ((data[pos + 1] & 0x3F) << 6) | (data[pos + 2] & 0x3F);
            if (cp < 0x800 || (cp >= 0xD800 && cp <= 0xDFFF)) return false;
            pos += 3;
        } else if ((c & 0xF8) == 0xF0) {
            if (pos + 3 >= len || (data[pos + 1] & 0xC0) != 0x80 || (data[pos + 2] & 0xC0) != 0x80 || (data[pos + 3] & 0xC0) != 0x80) return false;
            uint32_t cp = ((c & 0x07) << 18) | ((data[pos + 1] & 0x3F) << 12) | ((data[pos + 2] & 0x3F) << 6) | (data[pos + 3] & 0x3F);
            if (cp < 0x10000 || cp > 0x10FFFF) return false;
            pos += 4;
        } else return false;
    }
    return true;
}

// Structural scanner with proper odd-backslash detection
struct StructuralIndex {
    std::vector<size_t> positions;
    std::vector<char> chars;
    
    void scan(const char* data, size_t len) {
        size_t pos = 0;
        bool in_string = false;
        uint64_t prev_escaped = 0;
        
#ifdef __AVX2__
        while (pos + 32 <= len) {
            __m256i chunk = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + pos));
            __m256i quotes = _mm256_cmpeq_epi8(chunk, _mm256_set1_epi8('"'));
            __m256i backslashes = _mm256_cmpeq_epi8(chunk, _mm256_set1_epi8('\\'));
            
            uint32_t quote_mask = _mm256_movemask_epi8(quotes);
            uint32_t bs_mask = _mm256_movemask_epi8(backslashes);
            
            // Canonical stage-1 odd-backslash detection with correct carry parity
            uint32_t bs_ext = bs_mask;
            if (prev_escaped) bs_ext ^= 0xFFFFFFFFu;
            uint32_t odd_bits = bs_ext;
            odd_bits ^= odd_bits << 1;
            odd_bits ^= odd_bits << 2;
            odd_bits ^= odd_bits << 4;
            odd_bits ^= odd_bits << 8;
            odd_bits ^= odd_bits << 16;
            uint32_t odd_backslash = odd_bits & bs_mask;
            uint32_t escaped_quotes = quote_mask & (odd_backslash >> 1);
            
            // Compute trailing run parity for next block carry
            bool next_carry;
            if (bs_mask == 0xFFFFFFFFu) {
                // Entire block is backslashes: carry toggles
                next_carry = !prev_escaped;
            } else {
                unsigned hi_run = __builtin_clz(~bs_mask);
                next_carry = (hi_run & 1u) != 0;
            }
            prev_escaped = next_carry;
            
            uint32_t unescaped_quotes = quote_mask & ~escaped_quotes;
            
            // Canonical scalar prefix-XOR for in-string mask
            uint32_t x = unescaped_quotes;
            x ^= x << 1;
            x ^= x << 2;
            x ^= x << 4;
            x ^= x << 8;
            x ^= x << 16;
            uint32_t string_mask = in_string ? ~x : x;
            in_string ^= (__builtin_popcount(unescaped_quotes) & 1);
            
            // Find structural characters outside strings
            __m256i structural = _mm256_or_si256(
                _mm256_or_si256(
                    _mm256_cmpeq_epi8(chunk, _mm256_set1_epi8('{')),
                    _mm256_cmpeq_epi8(chunk, _mm256_set1_epi8('}'))),
                _mm256_or_si256(
                    _mm256_or_si256(
                        _mm256_cmpeq_epi8(chunk, _mm256_set1_epi8('[')),
                        _mm256_cmpeq_epi8(chunk, _mm256_set1_epi8(']'))),
                    _mm256_or_si256(
                        _mm256_cmpeq_epi8(chunk, _mm256_set1_epi8(',')),
                        _mm256_cmpeq_epi8(chunk, _mm256_set1_epi8(':')))));
            
            uint32_t struct_mask = _mm256_movemask_epi8(structural) & ~string_mask;
            
            while (struct_mask) {
                int idx = __builtin_ctz(struct_mask);
                positions.push_back(pos + idx);
                chars.push_back(data[pos + idx]);
                struct_mask &= struct_mask - 1;
            }
            pos += 32;
        }
        _mm256_zeroupper();
#endif
        // Scalar tail
        while (pos < len) {
            char c = data[pos];
            if (c == '"') {
                size_t bs_count = 0, check = pos;
                while (check > 0 && data[--check] == '\\') bs_count++;
                if (!(bs_count & 1)) in_string = !in_string;
            }
            if (!in_string && (c == '{' || c == '}' || c == '[' || c == ']' || c == ',' || c == ':')) {
                positions.push_back(pos);
                chars.push_back(c);
            }
            pos++;
        }
    }
};

struct JsonParser {
    const char* data;
    size_t pos;
    size_t len;
    StructuralIndex structural;
    
    JsonParser(const std::string& json) : data(json.data()), pos(0), len(json.size()) {
        structural.scan(data, len);
    }
    
    void skip_whitespace() {
#ifdef __AVX2__
        // Vectorized whitespace skipping with bulk advance
        while (pos + 64 <= len) {
            __m256i chunk1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + pos));
            __m256i chunk2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + pos + 32));
            
            __m256i ws1 = _mm256_or_si256(_mm256_or_si256(
                _mm256_cmpeq_epi8(chunk1, _mm256_set1_epi8(' ')),
                _mm256_cmpeq_epi8(chunk1, _mm256_set1_epi8('\t'))),
                _mm256_or_si256(
                _mm256_cmpeq_epi8(chunk1, _mm256_set1_epi8('\n')),
                _mm256_cmpeq_epi8(chunk1, _mm256_set1_epi8('\r'))));
            
            __m256i ws2 = _mm256_or_si256(_mm256_or_si256(
                _mm256_cmpeq_epi8(chunk2, _mm256_set1_epi8(' ')),
                _mm256_cmpeq_epi8(chunk2, _mm256_set1_epi8('\t'))),
                _mm256_or_si256(
                _mm256_cmpeq_epi8(chunk2, _mm256_set1_epi8('\n')),
                _mm256_cmpeq_epi8(chunk2, _mm256_set1_epi8('\r'))));
            
            uint32_t mask1 = _mm256_movemask_epi8(ws1);
            uint32_t mask2 = _mm256_movemask_epi8(ws2);
            
            if (mask1 != 0xFFFFFFFF) {
                pos += __builtin_ctz(~mask1);
                return;
            }
            if (mask2 != 0xFFFFFFFF) {
                pos += 32 + __builtin_ctz(~mask2);
                return;
            }
            pos += 64;
        }
        
        while (pos + 32 <= len) {
            __m256i chunk = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + pos));
            __m256i whitespace = _mm256_or_si256(_mm256_or_si256(
                _mm256_cmpeq_epi8(chunk, _mm256_set1_epi8(' ')),
                _mm256_cmpeq_epi8(chunk, _mm256_set1_epi8('\t'))),
                _mm256_or_si256(
                _mm256_cmpeq_epi8(chunk, _mm256_set1_epi8('\n')),
                _mm256_cmpeq_epi8(chunk, _mm256_set1_epi8('\r'))));
            
            uint32_t mask = ~_mm256_movemask_epi8(whitespace);
            if (mask) {
                pos += __builtin_ctz(mask);
                return;
            }
            pos += 32;
        }
        _mm256_zeroupper();
#endif
        while (pos < len && (data[pos] == ' ' || data[pos] == '\t' || data[pos] == '\n' || data[pos] == '\r')) pos++;
    }
    
    nb::object parse_value() {
        skip_whitespace();
        if (pos >= len) throw std::invalid_argument("Unexpected end of JSON");
        
        char c = data[pos];
        switch (c) {
            case 'n': return parse_null();
            case 't': case 'f': return parse_bool();
            case '"': return parse_string();
            case '[': return parse_array();
            case '{': return parse_object();
            case '-': case '0': case '1': case '2': case '3': case '4':
            case '5': case '6': case '7': case '8': case '9':
                return parse_number();
            default:
                throw std::invalid_argument("Invalid JSON character");
        }
    }
    
    nb::object parse_null() {
        if (pos + 4 <= len) {
#ifdef __SSE2__
            // Safe SSE comparison - copy to avoid aliasing UB
            uint32_t word;
            std::memcpy(&word, data + pos, 4);
            __m128i input = _mm_cvtsi32_si128(word);
            __m128i null_pattern = _mm_cvtsi32_si128(0x6c6c756e); // "null"
            __m128i cmp = _mm_cmpeq_epi8(input, null_pattern);
            if ((_mm_movemask_epi8(cmp) & 0x0F) == 0x0F) {
#else
            uint32_t word;
            std::memcpy(&word, data + pos, 4);
            if (word == 0x6c6c756e) { // "null" in little-endian
#endif
                pos += 4;
                if (pos < len && (std::isalnum(static_cast<unsigned char>(data[pos])) || data[pos] == '_')) {
                    throw std::invalid_argument("Invalid null token");
                }
                return nb::none();
            }
        }
        throw std::invalid_argument("Invalid null");
    }
    
    nb::object parse_bool() {
        if (pos + 4 <= len) {
#ifdef __SSE2__
            // Safe SSE comparison - copy to avoid aliasing UB
            uint32_t word;
            std::memcpy(&word, data + pos, 4);
            __m128i input = _mm_cvtsi32_si128(word);
            __m128i true_pattern = _mm_cvtsi32_si128(0x65757274); // "true"
            __m128i cmp = _mm_cmpeq_epi8(input, true_pattern);
            if ((_mm_movemask_epi8(cmp) & 0x0F) == 0x0F) {
#else
            uint32_t word;
            std::memcpy(&word, data + pos, 4);
            if (word == 0x65757274) { // "true" in little-endian
#endif
                pos += 4;
                if (pos < len && (std::isalnum(static_cast<unsigned char>(data[pos])) || data[pos] == '_')) {
                    throw std::invalid_argument("Invalid true token");
                }
                return nb::cast(true);
            }
        }
        if (pos + 5 <= len) {
#ifdef __SSE2__
            // Safe bounded SSE comparison for "false" - copy to local buffer
            char local_buf[16] = {0};
            std::memcpy(local_buf, data + pos, 5);
            __m128i input = _mm_loadu_si128(reinterpret_cast<const __m128i*>(local_buf));
            __m128i false_pattern = _mm_setr_epi8('f','a','l','s','e',0,0,0,0,0,0,0,0,0,0,0);
            __m128i cmp = _mm_cmpeq_epi8(input, false_pattern);
            if ((_mm_movemask_epi8(cmp) & 0x1F) == 0x1F) {
#else
            if (std::memcmp(data + pos, "false", 5) == 0) {
#endif
                pos += 5;
                if (pos < len && (std::isalnum(static_cast<unsigned char>(data[pos])) || data[pos] == '_')) {
                    throw std::invalid_argument("Invalid false token");
                }
                return nb::cast(false);
            }
        }
        throw std::invalid_argument("Invalid boolean");
    }
    
    nb::object parse_string() {
        if (data[pos] != '"') throw std::invalid_argument("Expected '\"'");
        pos++;
        size_t start = pos;
        
        // Fast path: scan for end quote using SIMD
        while (pos + 32 <= len) {
            __m256i chunk = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + pos));
            __m256i quotes = _mm256_cmpeq_epi8(chunk, _mm256_set1_epi8('"'));
            __m256i backslashes = _mm256_cmpeq_epi8(chunk, _mm256_set1_epi8('\\'));
            __m256i controls = _mm256_cmpgt_epi8(_mm256_set1_epi8(0x20), chunk);
            __m256i special = _mm256_or_si256(_mm256_or_si256(quotes, backslashes), controls);
            
            uint32_t mask = _mm256_movemask_epi8(special);
            if (mask) {
                pos += __builtin_ctz(mask);
                break;
            }
            pos += 32;
        }
        
        // Handle escapes and find actual end
        std::string result;
        result.reserve(pos - start + 32);
        pos = start;
        
        for (;;) {
#ifdef __AVX2__
            // SIMD span copy - ASCII-only clean spans
            while (pos + 32 <= len) {
                __m256i chunk = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + pos));
                __m256i quotes = _mm256_cmpeq_epi8(chunk, _mm256_set1_epi8('"'));
                __m256i backslashes = _mm256_cmpeq_epi8(chunk, _mm256_set1_epi8('\\'));
                __m256i controls = _mm256_cmpgt_epi8(_mm256_set1_epi8(0x20), chunk);
                __m256i high_bits = _mm256_cmpgt_epi8(chunk, _mm256_set1_epi8(0x7F));
                __m256i special = _mm256_or_si256(_mm256_or_si256(quotes, backslashes), 
                                                _mm256_or_si256(controls, high_bits));
                
                uint32_t mask = _mm256_movemask_epi8(special);
                if (mask == 0) {
                    // Clean ASCII span - no validation needed
                    result.append(data + pos, 32);
                    pos += 32;
                    continue;
                }
                
                int count = __builtin_ctz(mask);
                result.append(data + pos, count);
                pos += count;
                break;
            }
#endif
            
            // Byte-wise handling
            if (pos >= len) break;
            char c = data[pos];
            if (c == '"') {
                pos++;
                return nb::cast(result);
            }
            if (c == '\\') {
                pos++;
                if (pos >= len) throw std::invalid_argument("Unterminated string escape");
                char esc = data[pos++];
                switch (esc) {
                    case '"': result += '"'; break;
                    case '\\': result += '\\'; break;
                    case '/': result += '/'; break;
                    case 'b': result += '\b'; break;
                    case 'f': result += '\f'; break;
                    case 'n': result += '\n'; break;
                    case 'r': result += '\r'; break;
                    case 't': result += '\t'; break;
                    case 'u': {
                        if (pos + 4 > len) throw std::invalid_argument("Invalid unicode escape");
                        
#ifdef __SSE2__
                        // Safe SSE hex validation - copy to avoid potential OOB
                        if (pos + 4 <= len) {
                            uint32_t hex_word;
                            std::memcpy(&hex_word, data + pos, 4);
                            __m128i hex_chars = _mm_cvtsi32_si128(hex_word);
                            
                            // Use gt/lt with adjusted values: x >= '0' becomes x > ('0'-1)
                            __m128i gt_0_minus1 = _mm_cmpgt_epi8(hex_chars, _mm_set1_epi8('0' - 1));
                            __m128i lt_9_plus1 = _mm_cmplt_epi8(hex_chars, _mm_set1_epi8('9' + 1));
                            __m128i gt_A_minus1 = _mm_cmpgt_epi8(hex_chars, _mm_set1_epi8('A' - 1));
                            __m128i lt_F_plus1 = _mm_cmplt_epi8(hex_chars, _mm_set1_epi8('F' + 1));
                            __m128i gt_a_minus1 = _mm_cmpgt_epi8(hex_chars, _mm_set1_epi8('a' - 1));
                            __m128i lt_f_plus1 = _mm_cmplt_epi8(hex_chars, _mm_set1_epi8('f' + 1));
                            
                            __m128i digits = _mm_and_si128(gt_0_minus1, lt_9_plus1);
                            __m128i upper = _mm_and_si128(gt_A_minus1, lt_F_plus1);
                            __m128i lower = _mm_and_si128(gt_a_minus1, lt_f_plus1);
                            __m128i valid = _mm_or_si128(_mm_or_si128(digits, upper), lower);
                            
                            if ((_mm_movemask_epi8(valid) & 0x0F) != 0x0F) {
                                throw std::invalid_argument("Invalid hex digit");
                            }
                        }
#endif
                        
                        // Scalar parsing after validation
                        uint32_t codepoint = 0;
                        for (int i = 0; i < 4; i++) {
                            char hex = data[pos++];
                            codepoint <<= 4;
                            if (hex >= '0' && hex <= '9') codepoint |= hex - '0';
                            else if (hex >= 'a' && hex <= 'f') codepoint |= hex - 'a' + 10;
                            else if (hex >= 'A' && hex <= 'F') codepoint |= hex - 'A' + 10;
                        }
                        
                        if (codepoint >= 0xD800 && codepoint <= 0xDBFF) {
                            if (pos + 6 > len || data[pos] != '\\' || data[pos+1] != 'u') {
                                throw std::invalid_argument("Unpaired high surrogate");
                            }
                            pos += 2;
                            uint32_t low = 0;
                            for (int i = 0; i < 4; i++) {
                                char hex = data[pos++];
                                low <<= 4;
                                if (hex >= '0' && hex <= '9') low |= hex - '0';
                                else if (hex >= 'a' && hex <= 'f') low |= hex - 'a' + 10;
                                else if (hex >= 'A' && hex <= 'F') low |= hex - 'A' + 10;
                                else throw std::invalid_argument("Invalid hex digit");
                            }
                            if (low < 0xDC00 || low > 0xDFFF) {
                                throw std::invalid_argument("Invalid low surrogate");
                            }
                            codepoint = 0x10000 + ((codepoint & 0x3FF) << 10) + (low & 0x3FF);
                        } else if (codepoint >= 0xDC00 && codepoint <= 0xDFFF) {
                            throw std::invalid_argument("Unpaired low surrogate");
                        }
                        
                        // SSE UTF-8 encoding for small codepoints
                        if (codepoint < 0x80) {
                            result += static_cast<char>(codepoint);
                        } else if (codepoint < 0x800) {
                            // SSE 2-byte UTF-8 encoding
                            __m128i cp = _mm_cvtsi32_si128(codepoint);
                            __m128i byte1 = _mm_or_si128(_mm_set1_epi8(0xC0), _mm_srli_epi32(cp, 6));
                            __m128i byte2 = _mm_or_si128(_mm_set1_epi8(0x80), _mm_and_si128(cp, _mm_set1_epi32(0x3F)));
                            result += static_cast<char>(_mm_cvtsi128_si32(byte1));
                            result += static_cast<char>(_mm_cvtsi128_si32(byte2));
                        } else if (codepoint < 0x10000) {
                            result += static_cast<char>(0xE0 | (codepoint >> 12));
                            result += static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F));
                            result += static_cast<char>(0x80 | (codepoint & 0x3F));
                        } else {
                            result += static_cast<char>(0xF0 | (codepoint >> 18));
                            result += static_cast<char>(0x80 | ((codepoint >> 12) & 0x3F));
                            result += static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F));
                            result += static_cast<char>(0x80 | (codepoint & 0x3F));
                        }
                        break;
                    }
                    default: throw std::invalid_argument("Invalid escape sequence");
                }
                // Continue to next character without jumping back to SIMD loop
            } else if (static_cast<unsigned char>(c) < 0x20) {
                throw std::invalid_argument("Unescaped control character");
            } else if (c & 0x80) {
                // Scalar UTF-8 validation consuming complete sequences
                unsigned char b = static_cast<unsigned char>(c);
                size_t seq_len;
                if ((b & 0xE0) == 0xC0) seq_len = 2;
                else if ((b & 0xF0) == 0xE0) seq_len = 3;
                else if ((b & 0xF8) == 0xF0) seq_len = 4;
                else throw std::invalid_argument("Invalid UTF-8 start byte");
                
                if (pos + seq_len > len || !validate_utf8_simd(data + pos, seq_len))
                    throw std::invalid_argument("Invalid UTF-8 sequence");
                result.append(data + pos, seq_len);
                pos += seq_len;
            } else {
                result += c;
                pos++;
            }
        }
        throw std::invalid_argument("Unterminated string");
    }
    
    nb::object parse_number() {
        size_t start = pos;
        if (data[pos] == '-') pos++;
        
        if (pos >= len || !std::isdigit(static_cast<unsigned char>(data[pos]))) {
            throw std::invalid_argument("Invalid number");
        }
        
        // Integer part - enforce no leading zeros
        if (data[pos] == '0') {
            pos++;
            if (pos < len && std::isdigit(static_cast<unsigned char>(data[pos]))) {
                throw std::invalid_argument("Leading zeros not allowed");
            }
        } else {
            // Vectorized digit scanning with parallel accumulation
            uint64_t acc = 0;
            while (pos + 32 <= len) {
                __m256i chunk = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + pos));
                __m256i digit_vals = _mm256_sub_epi8(chunk, _mm256_set1_epi8('0'));
                __m256i is_digit = _mm256_and_si256(
                    _mm256_cmpgt_epi8(digit_vals, _mm256_set1_epi8(-1)),
                    _mm256_cmpgt_epi8(_mm256_set1_epi8(10), digit_vals)
                );
                uint32_t mask = _mm256_movemask_epi8(is_digit);
                if (mask != 0xFFFFFFFF) {
                    pos += __builtin_ctz(~mask);
                    break;
                }
                
                // Skip SIMD accumulation - use std::from_chars for correctness
                
                pos += 32;
            }
            _mm256_zeroupper();
            while (pos < len && std::isdigit(static_cast<unsigned char>(data[pos]))) pos++;
        }
        
        bool is_float = false;
        
        // Fractional part
        if (pos < len && data[pos] == '.') {
            is_float = true;
            pos++;
            if (pos >= len || !std::isdigit(static_cast<unsigned char>(data[pos]))) {
                throw std::invalid_argument("Invalid number");
            }
            while (pos < len && std::isdigit(static_cast<unsigned char>(data[pos]))) pos++;
        }
        
        // Exponent part
        if (pos < len && (data[pos] == 'e' || data[pos] == 'E')) {
            is_float = true;
            pos++;
            if (pos < len && (data[pos] == '+' || data[pos] == '-')) pos++;
            if (pos >= len || !std::isdigit(static_cast<unsigned char>(data[pos]))) {
                throw std::invalid_argument("Invalid number");
            }
            while (pos < len && std::isdigit(static_cast<unsigned char>(data[pos]))) pos++;
        }
        
        if (is_float) {
            double d;
            auto result = std::from_chars(data + start, data + pos, d);
            if (result.ec != std::errc{}) {
                std::string num_str(data + start, pos - start);
                d = std::stod(num_str); // fallback
            }
            if (!std::isfinite(d)) {
                throw std::invalid_argument("Number out of range");
            }
            return nb::cast(d);
        }
        
        // Try parsing as int64 first
        int64_t i64;
        if (std::from_chars(data + start, data + pos, i64).ec == std::errc{}) {
            return nb::cast(i64);
        }
        
        // Try uint64
        uint64_t u64;
        if (std::from_chars(data + start, data + pos, u64).ec == std::errc{}) {
            return nb::cast(u64);
        }
        
        // Large integer - use Python's arbitrary precision
        std::string num_str(data + start, pos - start);
        PyObject* py_int = PyLong_FromString(num_str.c_str(), nullptr, 10);
        if (py_int) {
            return nb::steal<nb::object>(py_int);
        }
        PyErr_Clear();
        throw std::invalid_argument("Invalid number format");
    }
    
    nb::object parse_array() {
        if (data[pos] != '[') throw std::invalid_argument("Expected '['");
        pos++;
        skip_whitespace();
        
        nb::list result;
        
        if (pos < len && data[pos] == ']') {
            pos++;
            return result;
        }
        
        while (true) {
            result.append(parse_value());
            skip_whitespace();
            
            if (pos >= len) throw std::invalid_argument("Unterminated array");
            
            if (data[pos] == ']') {
                pos++;
                break;
            }
            
            if (data[pos] != ',') throw std::invalid_argument("Expected ',' or ']'");
            pos++;
            skip_whitespace();
        }
        
        return result;
    }
    
    nb::object parse_object() {
        if (data[pos] != '{') throw std::invalid_argument("Expected '{'");
        pos++;
        skip_whitespace();
        
        nb::dict result;
        
        if (pos < len && data[pos] == '}') {
            pos++;
            return result;
        }
        
        while (true) {
            nb::object key = parse_string();
            skip_whitespace();
            
            if (pos >= len || data[pos] != ':') {
                throw std::invalid_argument("Expected ':'");
            }
            pos++;
            
            nb::object value = parse_value();
            result[key] = value;
            skip_whitespace();
            
            if (pos >= len) throw std::invalid_argument("Unterminated object");
            
            if (data[pos] == '}') {
                pos++;
                break;
            }
            
            if (data[pos] != ',') throw std::invalid_argument("Expected ',' or '}'");
            pos++;
            skip_whitespace();
        }
        
        return result;
    }
};

nb::object json_decode(const std::string& json_str) {
    try {
        JsonParser parser(json_str);
        nb::object result = parser.parse_value();
        parser.skip_whitespace();
        if (parser.pos < parser.len) {
            throw std::invalid_argument("Extra data after JSON");
        }
        return result;
    } catch (const std::exception& e) {
        throw std::invalid_argument("JSON decode error: " + std::string(e.what()));
    }
}