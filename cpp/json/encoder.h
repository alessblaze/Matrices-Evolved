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

#pragma once
#include <string>
#include "../global.h"
#include "canonicalization.h"
namespace nb = nanobind;

// JSON encoder functions
void py_to_json_fast(const nb::object& root_obj);
/**
 * JSON encoder function - equivalent to json.dumps() with compact separators
 * 
 * @param obj Python object to encode to JSON
 * @param preserve_utf8 If true, preserves valid UTF-8 sequences instead of escaping to \uXXXX.
 *                      Only affects non-ASCII characters (>= 0x80); mandatory escapes for
 *                      quotes, backslashes, and control characters are always applied.
 *                      Both modes produce RFC 8259 compliant JSON with predictable output.
 */
std::string json_encode(const nb::object& obj, bool preserve_utf8 = false);