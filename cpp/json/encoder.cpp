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

#include "encoder.h"


/**
 * JSON encoder - reuses canonicalization infrastructure without sorting
 * Supports immutabledict via _dict attribute access
 */
// Global flag for UTF-8 preservation mode
thread_local bool g_preserve_utf8 = false;

HOT_FUNCTION FLATTEN_FUNCTION void py_to_json_fast(const nb::object& root_obj) {
    if (!PyGILState_Check()) {
        throw std::runtime_error("py_to_json_fast requires the Python GIL");
    }
    
    std::vector<Task> stack;
    stack.reserve(512);
    stack.push_back({TaskType::VALUE, root_obj});
    
    while (!stack.empty()) {
        Task& t = stack.back();
        
        switch (t.type) {
            case TaskType::VALUE: {
                nb::object obj = t.obj;
                stack.pop_back();
                
                // Handle immutabledict by accessing _dict attribute
                if (nb::hasattr(obj, "_dict")) {
                    try {
                        obj = obj.attr("_dict");
                    } catch (...) {
                        // Fallback to dict() conversion if _dict access fails
                        obj = nb::cast<nb::dict>(obj);
                    }
                }
                
                if (obj.is_none()) {
                    write_cstring_external("null");
                } else if (nb::isinstance<nb::bool_>(obj)) {
                    write_cstring_external(nb::cast<bool>(obj) ? "true" : "false");
                } else if (nb::isinstance<nb::int_>(obj)) {
                    // Try 64-bit fast path
                    int overflow = 0;
                    long long v = PyLong_AsLongLongAndOverflow(obj.ptr(), &overflow);
                    if (!overflow && !(v == -1 && PyErr_Occurred())) {
                        char buf[32];
                        auto [ptr, ec] = std::to_chars(buf, buf + 32, (int64_t)v);
                        if (ec == std::errc()) {
                            write_string_external(std::string_view(buf, ptr - buf));
                        } else {
                            throw std::runtime_error("integer to_chars failed");
                        }
                    } else {
                        PyErr_Clear();
                        // Exact decimal for arbitrary precision integers
                        nb::str s = nb::str(obj);
                        std::string_view sv = nb::cast<std::string_view>(s);
                        write_string_external(sv);
                    }
                } else if (nb::isinstance<nb::float_>(obj)) {
                    double val = nb::cast<double>(obj);
                    if (!std::isfinite(val)) {
                        throw std::runtime_error("Out of range float values are not JSON compliant");
                    }
                    if (val == 0.0 && std::signbit(val)) val = 0.0;
                    char buf[64];
                    int len = fast_double_to_string(val, buf);
                    write_string_external(std::string_view(buf, len));
                } else if (nb::isinstance<nb::str>(obj)) {
                    std::string_view s = nb::cast<std::string_view>(obj);
                    write_char_external('"');
                    
                    // Vectorized string escaping with proper Unicode support
                    const char* data = s.data();
                    size_t len = s.size();
                    size_t i = 0;
                    
#ifdef __AVX2__
                    // AVX2 escape detection for clean spans
                    while (i + 32 <= len) {
                        __m256i chunk = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + i));
                        __m256i quote = _mm256_cmpeq_epi8(chunk, _mm256_set1_epi8('"'));
                        __m256i backslash = _mm256_cmpeq_epi8(chunk, _mm256_set1_epi8('\\'));
                        __m256i control = _mm256_cmpgt_epi8(_mm256_set1_epi8(0x20), chunk);
                        __m256i high_bit = _mm256_cmpgt_epi8(chunk, _mm256_set1_epi8(0x7F));
                        
                        __m256i needs_escape = _mm256_or_si256(_mm256_or_si256(quote, backslash), 
                                                             _mm256_or_si256(control, high_bit));
                        uint32_t mask = _mm256_movemask_epi8(needs_escape);
                        
                        if (mask == 0) {
                            // Clean ASCII span - copy directly
                            write_string_external(std::string_view(data + i, 32));
                            i += 32;
                            continue;
                        } else {
                            // Handle up to the first escape
                            int clean_count = __builtin_ctz(mask);
                            if (clean_count > 0) {
                                write_string_external(std::string_view(data + i, clean_count));
                                i += clean_count;
                            }
                            break;
                        }
                    }
                    _mm256_zeroupper();
#endif
                    
                    while (i < len) {
                        unsigned char c = static_cast<unsigned char>(data[i]);
                        
                        // Handle ASCII control and special characters
                        if (c == '"') { write_string_external("\\\""); i++; continue; }
                        if (c == '\\') { write_string_external("\\\\"); i++; continue; }
                        if (c == '\b') { write_string_external("\\b"); i++; continue; }
                        if (c == '\f') { write_string_external("\\f"); i++; continue; }
                        if (c == '\n') { write_string_external("\\n"); i++; continue; }
                        if (c == '\r') { write_string_external("\\r"); i++; continue; }
                        if (c == '\t') { write_string_external("\\t"); i++; continue; }
                        if (c < 0x20) { write_unicode_escape_external(c); i++; continue; }
                        
                        // Handle ASCII printable characters
                        if (c < 0x80) {
                            write_char_external(static_cast<char>(c));
                            i++;
                            continue;
                        }
                        
                        // Handle UTF-8 sequences - decode to Unicode code point
                        uint32_t codepoint = 0;
                        size_t utf8_len = 0;
                        
                        if ((c & 0x80) == 0) {
                            // ASCII (already handled above)
                            codepoint = c;
                            utf8_len = 1;
                        } else if ((c & 0xE0) == 0xC0) {
                            // 2-byte UTF-8 with validation
                            if (i + 1 < len && (data[i+1] & 0xC0) == 0x80) {
                                codepoint = ((c & 0x1F) << 6) | (data[i+1] & 0x3F);
                                // Check for overlong encoding
                                if (codepoint >= 0x80) {
                                    utf8_len = 2;
                                }
                            }
                        } else if ((c & 0xF0) == 0xE0) {
                            // 3-byte UTF-8 with validation
                            if (i + 2 < len && (data[i+1] & 0xC0) == 0x80 && (data[i+2] & 0xC0) == 0x80) {
                                codepoint = ((c & 0x0F) << 12) | ((data[i+1] & 0x3F) << 6) | (data[i+2] & 0x3F);
                                // Check for overlong encoding and surrogates
                                if (codepoint >= 0x800 && !(codepoint >= 0xD800 && codepoint <= 0xDFFF)) {
                                    utf8_len = 3;
                                }
                            }
                        } else if ((c & 0xF8) == 0xF0) {
                            // 4-byte UTF-8 with validation
                            if (i + 3 < len && (data[i+1] & 0xC0) == 0x80 && (data[i+2] & 0xC0) == 0x80 && (data[i+3] & 0xC0) == 0x80) {
                                codepoint = ((c & 0x07) << 18) | ((data[i+1] & 0x3F) << 12) | ((data[i+2] & 0x3F) << 6) | (data[i+3] & 0x3F);
                                // Check for overlong encoding and valid range
                                if (codepoint >= 0x10000 && codepoint <= 0x10FFFF) {
                                    utf8_len = 4;
                                }
                            }
                        }
                        
                        if (utf8_len == 0) {
                            // Invalid UTF-8, escape the byte
                            write_unicode_escape_external(c);
                            i++;
                            continue;
                        }
                        
                        // Option: preserve UTF-8 (RFC 8259 compliant, smaller output)
                        if (g_preserve_utf8) {
                            // Copy valid UTF-8 sequence directly
                            write_string_external(std::string_view(data + i, utf8_len));
                            i += utf8_len;
                            continue;
                        }
                        
                        // Unicode escape generation
                        if (codepoint <= 0xFFFF) {
#ifdef __AVX2__
                            // AVX2 hex digit generation for \uXXXX
                            __m128i cp_vec = _mm_set1_epi32(codepoint);
                            __m128i shifts = _mm_setr_epi32(12, 8, 4, 0);
                            __m128i nibbles = _mm_and_si128(_mm_srlv_epi32(cp_vec, shifts), _mm_set1_epi32(0xF));
                            
                            // Convert nibbles to hex chars
                            __m128i digits = _mm_add_epi32(nibbles, _mm_set1_epi32('0'));
                            __m128i letters = _mm_add_epi32(nibbles, _mm_set1_epi32('a' - 10));
                            __m128i is_letter = _mm_cmpgt_epi32(nibbles, _mm_set1_epi32(9));
                            __m128i hex_chars = _mm_blendv_epi8(digits, letters, is_letter);
                            
                            char escape[7] = "\\u0000";
                            escape[2] = _mm_extract_epi32(hex_chars, 0);
                            escape[3] = _mm_extract_epi32(hex_chars, 1);
                            escape[4] = _mm_extract_epi32(hex_chars, 2);
                            escape[5] = _mm_extract_epi32(hex_chars, 3);
                            write_string_external(std::string_view(escape, 6));
                            _mm256_zeroupper();
#else
                            // Scalar hex digit generation
                            static const char hex_table[] = "0123456789abcdef";
                            char escape[7] = "\\u0000";
                            escape[2] = hex_table[(codepoint >> 12) & 0xF];
                            escape[3] = hex_table[(codepoint >> 8) & 0xF];
                            escape[4] = hex_table[(codepoint >> 4) & 0xF];
                            escape[5] = hex_table[codepoint & 0xF];
                            write_string_external(std::string_view(escape, 6));
#endif
                        } else {
                            // Supplementary planes - UTF-16 surrogate pair
                            codepoint -= 0x10000;
                            uint16_t high = 0xD800 + ((codepoint >> 10) & 0x3FF);
                            uint16_t low = 0xDC00 + (codepoint & 0x3FF);
                            
                            static const char hex_table[] = "0123456789abcdef";
                            char escape_high[7] = "\\u0000";
                            escape_high[2] = hex_table[(high >> 12) & 0xF];
                            escape_high[3] = hex_table[(high >> 8) & 0xF];
                            escape_high[4] = hex_table[(high >> 4) & 0xF];
                            escape_high[5] = hex_table[high & 0xF];
                            
                            char escape_low[7] = "\\u0000";
                            escape_low[2] = hex_table[(low >> 12) & 0xF];
                            escape_low[3] = hex_table[(low >> 8) & 0xF];
                            escape_low[4] = hex_table[(low >> 4) & 0xF];
                            escape_low[5] = hex_table[low & 0xF];
                            
                            write_string_external(std::string_view(escape_high, 6));
                            write_string_external(std::string_view(escape_low, 6));
                        }
                        
                        i += utf8_len;
                    }
                    write_char_external('"');
                } else if (nb::isinstance<nb::list>(obj) || nb::isinstance<nb::tuple>(obj)) {
                    stack.push_back({TaskType::ARRAY_START, obj});
                } else if (nb::isinstance<nb::dict>(obj)) {
                    stack.push_back({TaskType::DICT_START, obj});
                } else {
                    throw std::runtime_error("Unsupported Python type for JSON encoding");
                }
                break;
            }
            
            case TaskType::ARRAY_START: {
                write_char_external('[');
                t.type = TaskType::ARRAY_ITEM;
                t.index = 0;
                t.seq = nb::borrow(t.obj);
                Py_ssize_t seq_len = nb::len(t.seq);
                if (seq_len < 0) throw std::runtime_error("Invalid sequence length");
                t.seq_size = static_cast<size_t>(seq_len);
                break;
            }
            
            case TaskType::ARRAY_ITEM: {
                if (t.index >= t.seq_size) {
                    write_char_external(']');
                    stack.pop_back();
                } else {
                    if (t.index > 0) write_char_external(',');
                    nb::object item = t.seq[t.index]; // new reference via sequence indexing
                    stack.push_back({TaskType::VALUE, std::move(item)}); // move the owning object
                    t.index++;
                }
                break;
            }
            
            case TaskType::DICT_START: {
                auto dict = nb::borrow<nb::dict>(t.obj);
                t.dict_pairs.clear();
                t.dict_pairs.reserve(dict.size());
                
                // No sorting - preserve insertion order
                for (auto item : dict) {
                    std::string key_str;
                    if (nb::isinstance<nb::str>(item.first)) {
                        key_str = nb::cast<std::string>(item.first);
                    } else {
                        // Convert non-string keys to strings (like Synapse)
                        nb::str key_as_str = nb::str(item.first);
                        key_str = nb::cast<std::string>(key_as_str);
                    }
                    nb::object value = nb::object(item.second, nb::detail::borrow_t{}); // new reference
                    t.dict_pairs.emplace_back(std::move(key_str), std::move(value));
                }
                
                write_char_external('{');
                t.type = TaskType::DICT_KEY;
                t.index = 0;
                break;
            }
            
            case TaskType::DICT_KEY: {
                if (t.index >= t.dict_pairs.size()) {
                    write_char_external('}');
                    stack.pop_back();
                } else {
                    if (t.index > 0) write_char_external(',');
                    
                    write_char_external('"');
                    const std::string& key = t.dict_pairs[t.index].first;
                    
                    // String escaping for keys with proper Unicode support
                    const char* key_data = key.data();
                    size_t key_len = key.size();
                    size_t i = 0;
                    
                    while (i < key_len) {
                        unsigned char c = static_cast<unsigned char>(key_data[i]);
                        
                        if (c == '"') { write_string_external("\\\""); i++; continue; }
                        if (c == '\\') { write_string_external("\\\\"); i++; continue; }
                        if (c < 0x20) { write_unicode_escape_external(c); i++; continue; }
                        if (c < 0x80) { write_char_external(static_cast<char>(c)); i++; continue; }
                        
                        // Handle UTF-8 sequences for keys
                        uint32_t codepoint = 0;
                        size_t utf8_len = 0;
                        
                        if ((c & 0xE0) == 0xC0 && i + 1 < key_len && (key_data[i+1] & 0xC0) == 0x80) {
                            codepoint = ((c & 0x1F) << 6) | (key_data[i+1] & 0x3F);
                            if (codepoint >= 0x80) utf8_len = 2;
                        } else if ((c & 0xF0) == 0xE0 && i + 2 < key_len && (key_data[i+1] & 0xC0) == 0x80 && (key_data[i+2] & 0xC0) == 0x80) {
                            codepoint = ((c & 0x0F) << 12) | ((key_data[i+1] & 0x3F) << 6) | (key_data[i+2] & 0x3F);
                            if (codepoint >= 0x800 && !(codepoint >= 0xD800 && codepoint <= 0xDFFF)) utf8_len = 3;
                        } else if ((c & 0xF8) == 0xF0 && i + 3 < key_len && (key_data[i+1] & 0xC0) == 0x80 && (key_data[i+2] & 0xC0) == 0x80 && (key_data[i+3] & 0xC0) == 0x80) {
                            codepoint = ((c & 0x07) << 18) | ((key_data[i+1] & 0x3F) << 12) | ((key_data[i+2] & 0x3F) << 6) | (key_data[i+3] & 0x3F);
                            if (codepoint >= 0x10000 && codepoint <= 0x10FFFF) utf8_len = 4;
                        }
                        
                        if (utf8_len == 0) {
                            write_unicode_escape_external(c);
                            i++;
                            continue;
                        }
                        
                        // Option: preserve UTF-8 for dictionary keys too
                        if (g_preserve_utf8) {
                            // Copy valid UTF-8 sequence directly
                            write_string_external(std::string_view(key_data + i, utf8_len));
                            i += utf8_len;
                            continue;
                        }
                        
                        static const char hex_table[] = "0123456789abcdef";
                        if (codepoint <= 0xFFFF) {
                            char escape[7] = "\\u0000";
                            escape[2] = hex_table[(codepoint >> 12) & 0xF];
                            escape[3] = hex_table[(codepoint >> 8) & 0xF];
                            escape[4] = hex_table[(codepoint >> 4) & 0xF];
                            escape[5] = hex_table[codepoint & 0xF];
                            write_string_external(std::string_view(escape, 6));
                        } else {
                            codepoint -= 0x10000;
                            uint16_t high = 0xD800 + ((codepoint >> 10) & 0x3FF);
                            uint16_t low = 0xDC00 + (codepoint & 0x3FF);
                            
                            char escape_high[7] = "\\u0000";
                            escape_high[2] = hex_table[(high >> 12) & 0xF];
                            escape_high[3] = hex_table[(high >> 8) & 0xF];
                            escape_high[4] = hex_table[(high >> 4) & 0xF];
                            escape_high[5] = hex_table[high & 0xF];
                            
                            char escape_low[7] = "\\u0000";
                            escape_low[2] = hex_table[(low >> 12) & 0xF];
                            escape_low[3] = hex_table[(low >> 8) & 0xF];
                            escape_low[4] = hex_table[(low >> 4) & 0xF];
                            escape_low[5] = hex_table[low & 0xF];
                            
                            write_string_external(std::string_view(escape_high, 6));
                            write_string_external(std::string_view(escape_low, 6));
                        }
                        
                        i += utf8_len;
                    }
                    
                    write_char_external('"');
                    write_char_external(':');
                    
                    t.type = TaskType::DICT_VALUE;
                    stack.push_back({TaskType::VALUE, t.dict_pairs[t.index].second});
                }
                break;
            }
            
            case TaskType::DICT_VALUE: {
                t.type = TaskType::DICT_KEY;
                t.index++;
                break;
            }
            
            default:
                throw std::runtime_error("Invalid task type in JSON encoder");
        }
    }
}

/**
 * Main JSON encoder function - equivalent to json.dumps() with compact separators
 * 
 * @param obj Python object to encode
 * @param preserve_utf8 If true, preserves valid UTF-8 sequences instead of escaping to \uXXXX.
 *                      Only affects non-ASCII characters; mandatory escapes (quotes, backslashes, 
 *                      control characters) are always applied for RFC 8259 compliance.
 *                      Both modes produce valid JSON with predictable output.
 */
std::string json_encode(const nb::object& obj, bool preserve_utf8) {
    g_preserve_utf8 = preserve_utf8;
    init_json_buffer();
    reset_json_pointer();
    py_to_json_fast(obj);
    return std::string(json_buffer.data(), json_ptr - json_buffer.data());
}