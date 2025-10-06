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

#include "global.h"

#include <cstring>
#include <span>
#include <type_traits>
#include <openssl/evp.h>
#include <openssl/rand.h>
#include <openssl/sha.h>
#include <openssl/base64.h>
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <sstream>
#include <iostream>
#include <cstdlib>
#include <stdexcept>
#include <charconv>
#include <cmath>
#include <type_traits>
#include <Python.h>
// Module includes - restructured cpp directory (header-only)
#include "base64/encoders/include/base64-encoder.h"
#include "base64/decoders/include/base64-decoder.h"
#include "json/encoder.h"
#include "json/decoder.h"
#include "json/canonicalization.h"
#include "crypto/ed25519.h"

namespace nb = nanobind;
using namespace nb::literals;
namespace json = boost::json;

// Memory leak profiling
static bool leak_warnings_enabled = []()
{
    const char *env = std::getenv("SYNAPSE_RUST_CRYPTO_LEAK_WARNINGS");
    return env && std::string(env) == "1";
}();

// Helpers from earlier (unchanged)
struct NormLimits
{
    Py_ssize_t max_container_len = -1;
    Py_ssize_t max_string_bytes = -1;
    Py_ssize_t max_nodes = -1;
};

static inline void check_len(Py_ssize_t len, Py_ssize_t limit, const char *what)
{
    if (limit >= 0 && len > limit)
    {
        std::string msg = std::string(what) + " length exceeds limit";
        throw nb::value_error(msg.c_str());
    }
}

static inline void check_nodes(Py_ssize_t &counter, Py_ssize_t limit)
{
    if (limit >= 0)
    {
        if (++counter > limit)
        {
            throw nb::value_error("normalization node budget exceeded");
        }
    }
}

static inline Py_ssize_t utf8_len_bytes(PyObject *py_str)
{
    Py_ssize_t size = 0;
    const char *data = PyUnicode_AsUTF8AndSize(py_str, &size);
    if (!data)
        throw nb::python_error();
    return size;
}

static inline bool is_scalar_fast(nb::handle o, const NormLimits &lim)
{
    PyObject *p = o.ptr();
    if (p == Py_None)
        return true;
    if (PyBool_Check(p))
        return true;
    if (PyLong_Check(p))
        return true;
    if (PyFloat_Check(p))
        return true;
    if (PyUnicode_Check(p))
    {
        if (lim.max_string_bytes >= 0)
        {
            Py_ssize_t blen = utf8_len_bytes(p);
            check_len(blen, lim.max_string_bytes, "string");
        }
        return true;
    }
    return false;
}

static nb::object normalize_for_canonical_json_with_limits(nb::handle root, const NormLimits &lim)
{
    if (is_scalar_fast(root, lim))
        return nb::object(root, nb::detail::borrow_t{});

    struct Frame
    {
        enum Kind
        {
            ListK,
            TupleK,
            SeqK,
            DictK,
            MapK,
            SetK
        } kind;
        nb::object in, out;
        Py_ssize_t i = 0, n = 0;
        PyObject *seq_raw = nullptr;
        PyObject **seq_items = nullptr;
        PyObject *iter = nullptr;
        nb::list map_items;
        nb::object pending_key;
        bool waiting_child = false;
    };

    Py_ssize_t nodes = 0;
    std::vector<Frame> stack;
    stack.reserve(32);

    auto push_list = [&](nb::object in_list)
    {
        Frame f;
        f.kind = Frame::ListK;
        f.in = in_list;
        f.n = PyList_GET_SIZE(in_list.ptr());
        check_len(f.n, lim.max_container_len, "list");
        check_nodes(nodes, lim.max_nodes);
        f.out = nb::list();
        stack.push_back(std::move(f));
    };

    auto push_tuple = [&](nb::object in_tuple)
    {
        Frame f;
        f.kind = Frame::TupleK;
        f.in = in_tuple;
        f.n = PyTuple_GET_SIZE(in_tuple.ptr());
        check_len(f.n, lim.max_container_len, "tuple");
        check_nodes(nodes, lim.max_nodes);
        f.out = nb::list();
        stack.push_back(std::move(f));
    };

    auto push_seq = [&](nb::handle in_seq)
    {
        Frame f;
        f.kind = Frame::SeqK;
        f.in = nb::object(in_seq, nb::detail::borrow_t{});
        f.seq_raw = PySequence_Fast(in_seq.ptr(), "expected a sequence");
        if (!f.seq_raw)
            throw nb::python_error();
        f.n = PySequence_Fast_GET_SIZE(f.seq_raw);
        check_len(f.n, lim.max_container_len, "sequence");
        check_nodes(nodes, lim.max_nodes);
        f.seq_items = PySequence_Fast_ITEMS(f.seq_raw);
        f.out = nb::list();
        stack.push_back(std::move(f));
    };

    auto push_dict = [&](nb::object in_dict)
    {
        Frame f;
        f.kind = Frame::DictK;
        f.in = in_dict;
        nb::object keys_obj = in_dict.attr("keys")();
        f.seq_raw = PySequence_Fast(keys_obj.ptr(), "expected a sequence");
        if (!f.seq_raw)
            throw nb::python_error();
        f.n = PySequence_Fast_GET_SIZE(f.seq_raw);
        check_len(f.n, lim.max_container_len, "dict");
        check_nodes(nodes, lim.max_nodes);
        f.seq_items = PySequence_Fast_ITEMS(f.seq_raw);
        f.out = nb::dict();
        stack.push_back(std::move(f));
    };

    auto push_map = [&](nb::handle in_map)
    {
        Frame f;
        f.kind = Frame::MapK;
        f.in = nb::object(in_map, nb::detail::borrow_t{});
        PyObject *items_raw = PyMapping_Items(in_map.ptr());
        if (!items_raw)
            throw nb::python_error();
        nb::object items_obj(nb::handle(items_raw), nb::detail::steal_t{});
        f.map_items = nb::cast<nb::list>(items_obj);
        f.n = PyList_GET_SIZE(f.map_items.ptr());
        check_len(f.n, lim.max_container_len, "mapping");
        check_nodes(nodes, lim.max_nodes);
        f.out = nb::dict();
        stack.push_back(std::move(f));
    };

    auto push_set = [&](nb::handle in_set)
    {
        Frame f;
        f.kind = Frame::SetK;
        f.in = nb::object(in_set, nb::detail::borrow_t{});
        Py_ssize_t sz = PyObject_Size(in_set.ptr());
        if (sz < 0)
            throw nb::python_error();
        check_len(sz, lim.max_container_len, "set");
        check_nodes(nodes, lim.max_nodes);
        f.iter = PyObject_GetIter(in_set.ptr());
        if (!f.iter)
            throw nb::python_error();
        f.out = nb::list();
        stack.push_back(std::move(f));
    };

    // Initial classification (mirror recursive ordering)
    if (PyList_Check(root.ptr()))
    {
        push_list(nb::cast<nb::list>(root));
    }
    else if (PyTuple_Check(root.ptr()))
    {
        push_tuple(nb::cast<nb::tuple>(root));
    }
    else if (PySequence_Check(root.ptr()) &&
             !PyMapping_Check(root.ptr()) &&
             !PyDict_Check(root.ptr()) &&
             !PyUnicode_Check(root.ptr()) &&
             !PyBytes_Check(root.ptr()) &&
             !PyByteArray_Check(root.ptr()))
    {
        push_seq(root);
    }
    else if (PyDict_Check(root.ptr()))
    {
        push_dict(nb::cast<nb::dict>(root));
    }
    else if (PyMapping_Check(root.ptr()))
    {
        push_map(root);
    }
    else if (PySet_Check(root.ptr()) || PyFrozenSet_Check(root.ptr()))
    {
        push_set(root);
    }
    else if (PyBytes_Check(root.ptr()) || PyByteArray_Check(root.ptr()))
    {
        throw nb::type_error("bytes/bytearray are not JSON-serialisable");
    }
    else
    {
        std::string msg = "object type is not JSON-serialisable: ";
        msg += Py_TYPE(root.ptr())->tp_name;
        throw nb::type_error(msg.c_str());
    }

    auto attach_done_to_parent = [&](const nb::object &done)
    {
        Frame &parent = stack.back();
        switch (parent.kind)
        {
        case Frame::ListK:
        case Frame::TupleK:
        case Frame::SeqK:
        case Frame::SetK:
        {
            nb::list out = nb::cast<nb::list>(parent.out);
            out.append(done);
            break;
        }
        case Frame::DictK:
        case Frame::MapK:
        {
            if (!parent.waiting_child)
            {
                throw nb::type_error("internal error: mapping parent not waiting for child");
            }
            nb::dict out = nb::cast<nb::dict>(parent.out);
            out[parent.pending_key] = done;
            parent.pending_key = nb::object();
            parent.waiting_child = false;
            break;
        }
        }
    };

    while (!stack.empty())
    {
        Frame &f = stack.back();

        if (f.kind == Frame::ListK)
        {
            if (f.i < f.n)
            {
                nb::handle child(PyList_GET_ITEM(f.in.ptr(), f.i));
                f.i++;
                if (is_scalar_fast(child, lim))
                {
                    nb::list out = nb::cast<nb::list>(f.out);
                    out.append(nb::object(child, nb::detail::borrow_t{}));
                    check_nodes(nodes, lim.max_nodes);
                    continue;
                }
                if (PyList_Check(child.ptr()))
                {
                    push_list(nb::cast<nb::list>(child));
                    continue;
                }
                if (PyTuple_Check(child.ptr()))
                {
                    push_tuple(nb::cast<nb::tuple>(child));
                    continue;
                }
                if (PySequence_Check(child.ptr()) &&
                    !PyMapping_Check(child.ptr()) &&
                    !PyDict_Check(child.ptr()) &&
                    !PyUnicode_Check(child.ptr()) &&
                    !PyBytes_Check(child.ptr()) &&
                    !PyByteArray_Check(child.ptr()))
                {
                    push_seq(child);
                    continue;
                }
                if (PyDict_Check(child.ptr()))
                {
                    push_dict(nb::cast<nb::dict>(child));
                    continue;
                }
                if (PyMapping_Check(child.ptr()))
                {
                    push_map(child);
                    continue;
                }
                if (PySet_Check(child.ptr()) || PyFrozenSet_Check(child.ptr()))
                {
                    push_set(child);
                    continue;
                }
                if (PyBytes_Check(child.ptr()) || PyByteArray_Check(child.ptr()))
                    throw nb::type_error("bytes/bytearray are not JSON-serialisable");
                std::string msg = "object type is not JSON-serialisable: ";
                msg += Py_TYPE(child.ptr())->tp_name;
                throw nb::type_error(msg.c_str());
            }
            nb::object done = f.out;
            stack.pop_back();
            if (stack.empty())
                return done;
            attach_done_to_parent(done);
            continue;
        }

        if (f.kind == Frame::TupleK)
        {
            if (f.i < f.n)
            {
                nb::handle child(PyTuple_GET_ITEM(f.in.ptr(), f.i));
                f.i++;
                if (is_scalar_fast(child, lim))
                {
                    nb::list out = nb::cast<nb::list>(f.out);
                    out.append(nb::object(child, nb::detail::borrow_t{}));
                    check_nodes(nodes, lim.max_nodes);
                    continue;
                }
                if (PyList_Check(child.ptr()))
                {
                    push_list(nb::cast<nb::list>(child));
                    continue;
                }
                if (PyTuple_Check(child.ptr()))
                {
                    push_tuple(nb::cast<nb::tuple>(child));
                    continue;
                }
                if (PySequence_Check(child.ptr()) &&
                    !PyMapping_Check(child.ptr()) &&
                    !PyDict_Check(child.ptr()) &&
                    !PyUnicode_Check(child.ptr()) &&
                    !PyBytes_Check(child.ptr()) &&
                    !PyByteArray_Check(child.ptr()))
                {
                    push_seq(child);
                    continue;
                }
                if (PyDict_Check(child.ptr()))
                {
                    push_dict(nb::cast<nb::dict>(child));
                    continue;
                }
                if (PyMapping_Check(child.ptr()))
                {
                    push_map(child);
                    continue;
                }
                if (PySet_Check(child.ptr()) || PyFrozenSet_Check(child.ptr()))
                {
                    push_set(child);
                    continue;
                }
                if (PyBytes_Check(child.ptr()) || PyByteArray_Check(child.ptr()))
                    throw nb::type_error("bytes/bytearray are not JSON-serialisable");
                std::string msg = "object type is not JSON-serialisable: ";
                msg += Py_TYPE(child.ptr())->tp_name;
                throw nb::type_error(msg.c_str());
            }
            nb::object done = f.out;
            stack.pop_back();
            if (stack.empty())
                return done;
            attach_done_to_parent(done);
            continue;
        }

        if (f.kind == Frame::SeqK)
        {
            if (f.i < f.n)
            {
                nb::handle child(f.seq_items[f.i]);
                f.i++;
                if (is_scalar_fast(child, lim))
                {
                    nb::list out = nb::cast<nb::list>(f.out);
                    out.append(nb::object(child, nb::detail::borrow_t{}));
                    check_nodes(nodes, lim.max_nodes);
                    continue;
                }
                if (PyList_Check(child.ptr()))
                {
                    push_list(nb::cast<nb::list>(child));
                    continue;
                }
                if (PyTuple_Check(child.ptr()))
                {
                    push_tuple(nb::cast<nb::tuple>(child));
                    continue;
                }
                if (PySequence_Check(child.ptr()) &&
                    !PyMapping_Check(child.ptr()) &&
                    !PyDict_Check(child.ptr()) &&
                    !PyUnicode_Check(child.ptr()) &&
                    !PyBytes_Check(child.ptr()) &&
                    !PyByteArray_Check(child.ptr()))
                {
                    push_seq(child);
                    continue;
                }
                if (PyDict_Check(child.ptr()))
                {
                    push_dict(nb::cast<nb::dict>(child));
                    continue;
                }
                if (PyMapping_Check(child.ptr()))
                {
                    push_map(child);
                    continue;
                }
                if (PySet_Check(child.ptr()) || PyFrozenSet_Check(child.ptr()))
                {
                    push_set(child);
                    continue;
                }
                if (PyBytes_Check(child.ptr()) || PyByteArray_Check(child.ptr()))
                    throw nb::type_error("bytes/bytearray are not JSON-serialisable");
                std::string msg = "object type is not JSON-serialisable: ";
                msg += Py_TYPE(child.ptr())->tp_name;
                throw nb::type_error(msg.c_str());
            }
            Py_DECREF(f.seq_raw);
            f.seq_raw = nullptr;
            nb::object done = f.out;
            stack.pop_back();
            if (stack.empty())
                return done;
            attach_done_to_parent(done);
            continue;
        }

        if (f.kind == Frame::DictK)
        {
            if (f.i < f.n)
            {
                nb::handle key_h(f.seq_items[f.i]);
                f.i++;
                nb::str k = nb::str(key_h); // enforce string key
                nb::object val = nb::cast<nb::object>(f.in.attr("__getitem__")(key_h));

                if (is_scalar_fast(val, lim))
                {
                    nb::dict out = nb::cast<nb::dict>(f.out);
                    out[k] = nb::object(val, nb::detail::borrow_t{});
                    check_nodes(nodes, lim.max_nodes);
                    continue;
                }

                f.pending_key = k;
                f.waiting_child = true;

                if (PyList_Check(val.ptr()))
                {
                    push_list(nb::cast<nb::list>(val));
                    continue;
                }
                if (PyTuple_Check(val.ptr()))
                {
                    push_tuple(nb::cast<nb::tuple>(val));
                    continue;
                }
                if (PySequence_Check(val.ptr()) &&
                    !PyMapping_Check(val.ptr()) &&
                    !PyDict_Check(val.ptr()) &&
                    !PyUnicode_Check(val.ptr()) &&
                    !PyBytes_Check(val.ptr()) &&
                    !PyByteArray_Check(val.ptr()))
                {
                    push_seq(val);
                    continue;
                }
                if (PyDict_Check(val.ptr()))
                {
                    push_dict(nb::cast<nb::dict>(val));
                    continue;
                }
                if (PyMapping_Check(val.ptr()))
                {
                    push_map(val);
                    continue;
                }
                if (PySet_Check(val.ptr()) || PyFrozenSet_Check(val.ptr()))
                {
                    push_set(val);
                    continue;
                }
                if (PyBytes_Check(val.ptr()) || PyByteArray_Check(val.ptr()))
                    throw nb::type_error("bytes/bytearray are not JSON-serialisable");
                std::string msg = "object type is not JSON-serialisable: ";
                msg += Py_TYPE(val.ptr())->tp_name;
                throw nb::type_error(msg.c_str());
            }
            if (f.seq_raw)
            {
                Py_DECREF(f.seq_raw);
                f.seq_raw = nullptr;
            }
            nb::object done = f.out;
            stack.pop_back();
            if (stack.empty())
                return done;
            attach_done_to_parent(done);
            continue;
        }

        if (f.kind == Frame::MapK)
        {
            if (f.i < f.n)
            {
                nb::handle kvh = f.map_items[f.i];
                f.i++;
                nb::tuple kv = nb::cast<nb::tuple>(kvh);
                nb::str k = nb::str(kv[0]);
                nb::object val = nb::cast<nb::object>(kv[1]);

                if (is_scalar_fast(val, lim))
                {
                    nb::dict out = nb::cast<nb::dict>(f.out);
                    out[k] = nb::object(val, nb::detail::borrow_t{});
                    check_nodes(nodes, lim.max_nodes);
                    continue;
                }

                f.pending_key = k;
                f.waiting_child = true;

                if (PyList_Check(val.ptr()))
                {
                    push_list(nb::cast<nb::list>(val));
                    continue;
                }
                if (PyTuple_Check(val.ptr()))
                {
                    push_tuple(nb::cast<nb::tuple>(val));
                    continue;
                }
                if (PySequence_Check(val.ptr()) &&
                    !PyMapping_Check(val.ptr()) &&
                    !PyDict_Check(val.ptr()) &&
                    !PyUnicode_Check(val.ptr()) &&
                    !PyBytes_Check(val.ptr()) &&
                    !PyByteArray_Check(val.ptr()))
                {
                    push_seq(val);
                    continue;
                }
                if (PyDict_Check(val.ptr()))
                {
                    push_dict(nb::cast<nb::dict>(val));
                    continue;
                }
                if (PyMapping_Check(val.ptr()))
                {
                    push_map(val);
                    continue;
                }
                if (PySet_Check(val.ptr()) || PyFrozenSet_Check(val.ptr()))
                {
                    push_set(val);
                    continue;
                }
                if (PyBytes_Check(val.ptr()) || PyByteArray_Check(val.ptr()))
                    throw nb::type_error("bytes/bytearray are not JSON-serialisable");
                std::string msg = "object type is not JSON-serialisable: ";
                msg += Py_TYPE(val.ptr())->tp_name;
                throw nb::type_error(msg.c_str());
            }
            nb::object done = f.out;
            stack.pop_back();
            if (stack.empty())
                return done;
            attach_done_to_parent(done);
            continue;
        }

        if (f.kind == Frame::SetK)
        {
            PyObject *nxt = PyIter_Next(f.iter);
            if (nxt)
            {
                nb::object elem(nb::handle(nxt), nb::detail::steal_t{});
                if (is_scalar_fast(elem, lim))
                {
                    nb::list out = nb::cast<nb::list>(f.out);
                    out.append(std::move(elem));
                    check_nodes(nodes, lim.max_nodes);
                    continue;
                }
                if (PyList_Check(elem.ptr()))
                {
                    push_list(nb::cast<nb::list>(elem));
                    continue;
                }
                if (PyTuple_Check(elem.ptr()))
                {
                    push_tuple(nb::cast<nb::tuple>(elem));
                    continue;
                }
                if (PySequence_Check(elem.ptr()) &&
                    !PyMapping_Check(elem.ptr()) &&
                    !PyDict_Check(elem.ptr()) &&
                    !PyUnicode_Check(elem.ptr()) &&
                    !PyBytes_Check(elem.ptr()) &&
                    !PyByteArray_Check(elem.ptr()))
                {
                    push_seq(elem);
                    continue;
                }
                if (PyDict_Check(elem.ptr()))
                {
                    push_dict(nb::cast<nb::dict>(elem));
                    continue;
                }
                if (PyMapping_Check(elem.ptr()))
                {
                    push_map(elem);
                    continue;
                }
                if (PySet_Check(elem.ptr()) || PyFrozenSet_Check(elem.ptr()))
                {
                    push_set(elem);
                    continue;
                }
                if (PyBytes_Check(elem.ptr()) || PyByteArray_Check(elem.ptr()))
                    throw nb::type_error("bytes/bytearray are not JSON-serialisable");
                std::string msg = "object type is not JSON-serialisable: ";
                msg += Py_TYPE(elem.ptr())->tp_name;
                throw nb::type_error(msg.c_str());
            }
            if (PyErr_Occurred())
                throw nb::python_error();
            Py_DECREF(f.iter);
            f.iter = nullptr;
            nb::object done = f.out;
            stack.pop_back();
            if (stack.empty())
                return done;
            attach_done_to_parent(done);
            continue;
        }
    }

    std::string msg = "object type is not JSON-serialisable: ";
    msg += Py_TYPE(root.ptr())->tp_name;
    throw nb::type_error(msg.c_str());
}

// Keep your simple wrapper to preserve existing API
static nb::object normalize_for_canonical_json(nb::handle root)
{
    NormLimits lim;
    return normalize_for_canonical_json_with_limits(root, lim);
}

// iterative version for normalization
static nb::object normalize_for_canonical_json_iterative(nb::handle root)
{
    auto is_scalar = [](nb::handle o) -> bool
    {
        return o.is_none() ||
               nb::isinstance<nb::str>(o) ||
               nb::isinstance<nb::bool_>(o) ||
               nb::isinstance<nb::int_>(o) ||
               nb::isinstance<nb::float_>(o);
    };

    if (is_scalar(root))
        return nb::object(root, nb::detail::borrow_t{});

    struct Frame
    {
        enum Kind
        {
            ListK,
            TupleK,
            SeqK,
            DictK,
            MapK,
            SetK
        } kind;
        nb::object in;    // input container
        nb::object out;   // output container (nb::list or nb::dict)
        Py_ssize_t i = 0; // index/position
        Py_ssize_t n = 0;

        // Fast sequence support (PySequence_Fast)
        PyObject *seq_raw = nullptr;    // owned when set (DECREF on finish)
        PyObject **seq_items = nullptr; // pointer from PySequence_Fast_ITEMS

        // Set iterator
        PyObject *iter = nullptr; // owned when set (DECREF on finish)

        // Mapping items (list of (k, v))
        nb::list map_items;

        // For dict/map pending value attachment
        nb::object pending_key; // valid only when waiting_child == true
        bool waiting_child = false;
    };

    std::vector<Frame> stack;
    stack.reserve(32);

    auto push_list = [&](nb::object in_list)
    {
        Frame f;
        f.kind = Frame::ListK;
        f.in = in_list;
        f.n = PyList_GET_SIZE(in_list.ptr());
        f.out = nb::list();
        stack.push_back(std::move(f));
    };

    auto push_tuple = [&](nb::object in_tuple)
    {
        Frame f;
        f.kind = Frame::TupleK;
        f.in = in_tuple;
        f.n = PyTuple_GET_SIZE(in_tuple.ptr());
        f.out = nb::list(); // tuples become lists
        stack.push_back(std::move(f));
    };

    auto push_seq = [&](nb::handle in_seq)
    {
        Frame f;
        f.kind = Frame::SeqK;
        f.in = nb::object(in_seq, nb::detail::borrow_t{});
        f.seq_raw = PySequence_Fast(in_seq.ptr(), "expected a sequence");
        if (!f.seq_raw)
            throw nb::python_error();
        f.n = PySequence_Fast_GET_SIZE(f.seq_raw);
        f.seq_items = PySequence_Fast_ITEMS(f.seq_raw);
        f.out = nb::list();
        stack.push_back(std::move(f));
    };

    auto push_dict = [&](nb::object in_dict)
    {
        Frame f;
        f.kind = Frame::DictK;
        f.in = in_dict;
        f.out = nb::dict();
        stack.push_back(std::move(f));
    };

    auto push_map = [&](nb::handle in_map)
    {
        Frame f;
        f.kind = Frame::MapK;
        f.in = nb::object(in_map, nb::detail::borrow_t{});
        PyObject *items_raw = PyMapping_Items(in_map.ptr()); // new ref
        if (!items_raw)
            throw nb::python_error();
        nb::object items_obj(nb::handle(items_raw), nb::detail::steal_t{});
        f.map_items = nb::cast<nb::list>(items_obj);
        f.n = PyList_GET_SIZE(f.map_items.ptr());
        f.out = nb::dict();
        stack.push_back(std::move(f));
    };

    auto push_set = [&](nb::handle in_set)
    {
        Frame f;
        f.kind = Frame::SetK;
        f.in = nb::object(in_set, nb::detail::borrow_t{});
        f.iter = PyObject_GetIter(in_set.ptr());
        if (!f.iter)
            throw nb::python_error();
        f.out = nb::list();
        stack.push_back(std::move(f));
    };

    // Initialize from root (preserve your original branching rules and exclusions)
    if (nb::isinstance<nb::list>(root))
    {
        push_list(nb::cast<nb::list>(root));
    }
    else if (nb::isinstance<nb::tuple>(root))
    {
        push_tuple(nb::cast<nb::tuple>(root));
    }
    else if (PySequence_Check(root.ptr()) &&
             !PyMapping_Check(root.ptr()) &&
             !PyDict_Check(root.ptr()) &&
             !nb::isinstance<nb::str>(root) &&
             !PyBytes_Check(root.ptr()))
    {
        push_seq(root);
    }
    else if (nb::isinstance<nb::dict>(root))
    {
        push_dict(nb::cast<nb::dict>(root));
    }
    else if (PyMapping_Check(root.ptr()))
    {
        push_map(root);
    }
    else if (PySet_Check(root.ptr()) || PyFrozenSet_Check(root.ptr()))
    {
        push_set(root);
    }
    else if (PyBytes_Check(root.ptr()) || PyByteArray_Check(root.ptr()))
    {
        throw nb::type_error("bytes/bytearray are not JSON-serialisable");
    }
    else
    {
        throw nb::type_error("object type is not JSON-serialisable");
    }

    // Generic child completion handler: attach 'done' to current parent frame
    auto attach_done_to_parent = [&](const nb::object &done)
    {
        Frame &parent = stack.back();
        switch (parent.kind)
        {
        case Frame::ListK:
        case Frame::TupleK:
        case Frame::SeqK:
        case Frame::SetK:
        {
            nb::list out = nb::cast<nb::list>(parent.out);
            out.append(done);
            break;
        }
        case Frame::DictK:
        case Frame::MapK:
        {
            if (!parent.waiting_child)
                throw nb::type_error("internal error: mapping parent not waiting for child");
            nb::dict out = nb::cast<nb::dict>(parent.out);
            out[parent.pending_key] = done;
            parent.pending_key = nb::object();
            parent.waiting_child = false;
            break;
        }
        }
    };

    nb::object child_result; // holds the result of the most recently completed child

    while (!stack.empty())
    {
        Frame &f = stack.back();

        if (f.kind == Frame::ListK)
        {
            if (f.i < f.n)
            {
                PyObject *item = PyList_GET_ITEM(f.in.ptr(), f.i); // borrowed
                f.i++;
                nb::handle child(item);
                if (is_scalar(child))
                {
                    nb::list out = nb::cast<nb::list>(f.out);
                    out.append(nb::object(child, nb::detail::borrow_t{}));
                    continue;
                }
                if (nb::isinstance<nb::list>(child))
                {
                    push_list(nb::cast<nb::list>(child));
                    continue;
                }
                if (nb::isinstance<nb::tuple>(child))
                {
                    push_tuple(nb::cast<nb::tuple>(child));
                    continue;
                }
                if (PySequence_Check(child.ptr()) &&
                    !PyMapping_Check(child.ptr()) &&
                    !PyDict_Check(child.ptr()) &&
                    !nb::isinstance<nb::str>(child) &&
                    !PyBytes_Check(child.ptr()))
                {
                    push_seq(child);
                    continue;
                }
                if (nb::isinstance<nb::dict>(child))
                {
                    push_dict(nb::cast<nb::dict>(child));
                    continue;
                }
                if (PyMapping_Check(child.ptr()))
                {
                    push_map(child);
                    continue;
                }
                if (PySet_Check(child.ptr()) || PyFrozenSet_Check(child.ptr()))
                {
                    push_set(child);
                    continue;
                }
                if (PyBytes_Check(child.ptr()) || PyByteArray_Check(child.ptr()))
                    throw nb::type_error("bytes/bytearray are not JSON-serialisable");
                throw nb::type_error("object type is not JSON-serialisable");
            }
            // done with this frame
            nb::object done = f.out;
            stack.pop_back();
            if (stack.empty())
                return done;
            attach_done_to_parent(done);
            continue;
        }

        if (f.kind == Frame::TupleK)
        {
            if (f.i < f.n)
            {
                PyObject *item = PyTuple_GET_ITEM(f.in.ptr(), f.i); // borrowed
                f.i++;
                nb::handle child(item);
                if (is_scalar(child))
                {
                    nb::list out = nb::cast<nb::list>(f.out);
                    out.append(nb::object(child, nb::detail::borrow_t{}));
                    continue;
                }
                if (nb::isinstance<nb::list>(child))
                {
                    push_list(nb::cast<nb::list>(child));
                    continue;
                }
                if (nb::isinstance<nb::tuple>(child))
                {
                    push_tuple(nb::cast<nb::tuple>(child));
                    continue;
                }
                if (PySequence_Check(child.ptr()) &&
                    !PyMapping_Check(child.ptr()) &&
                    !PyDict_Check(child.ptr()) &&
                    !nb::isinstance<nb::str>(child) &&
                    !PyBytes_Check(child.ptr()))
                {
                    push_seq(child);
                    continue;
                }
                if (nb::isinstance<nb::dict>(child))
                {
                    push_dict(nb::cast<nb::dict>(child));
                    continue;
                }
                if (PyMapping_Check(child.ptr()))
                {
                    push_map(child);
                    continue;
                }
                if (PySet_Check(child.ptr()) || PyFrozenSet_Check(child.ptr()))
                {
                    push_set(child);
                    continue;
                }
                if (PyBytes_Check(child.ptr()) || PyByteArray_Check(child.ptr()))
                    throw nb::type_error("bytes/bytearray are not JSON-serialisable");
                throw nb::type_error("object type is not JSON-serialisable");
            }
            nb::object done = f.out;
            stack.pop_back();
            if (stack.empty())
                return done;
            attach_done_to_parent(done);
            continue;
        }

        if (f.kind == Frame::SeqK)
        {
            if (f.i < f.n)
            {
                nb::handle child(f.seq_items[f.i]);
                f.i++;
                if (is_scalar(child))
                {
                    nb::list out = nb::cast<nb::list>(f.out);
                    out.append(nb::object(child, nb::detail::borrow_t{}));
                    continue;
                }
                if (nb::isinstance<nb::list>(child))
                {
                    push_list(nb::cast<nb::list>(child));
                    continue;
                }
                if (nb::isinstance<nb::tuple>(child))
                {
                    push_tuple(nb::cast<nb::tuple>(child));
                    continue;
                }
                if (PySequence_Check(child.ptr()) &&
                    !PyMapping_Check(child.ptr()) &&
                    !PyDict_Check(child.ptr()) &&
                    !nb::isinstance<nb::str>(child) &&
                    !PyBytes_Check(child.ptr()))
                {
                    push_seq(child);
                    continue;
                }
                if (nb::isinstance<nb::dict>(child))
                {
                    push_dict(nb::cast<nb::dict>(child));
                    continue;
                }
                if (PyMapping_Check(child.ptr()))
                {
                    push_map(child);
                    continue;
                }
                if (PySet_Check(child.ptr()) || PyFrozenSet_Check(child.ptr()))
                {
                    push_set(child);
                    continue;
                }
                if (PyBytes_Check(child.ptr()) || PyByteArray_Check(child.ptr()))
                    throw nb::type_error("bytes/bytearray are not JSON-serialisable");
                throw nb::type_error("object type is not JSON-serialisable");
            }
            Py_DECREF(f.seq_raw);
            f.seq_raw = nullptr;
            nb::object done = f.out;
            stack.pop_back();
            if (stack.empty())
                return done;
            attach_done_to_parent(done);
            continue;
        }

        if (f.kind == Frame::DictK)
        {
            // Initialize keys fast sequence once
            if (f.i == 0 && f.seq_raw == nullptr)
            {
                nb::object keys_obj = f.in.attr("keys")();
                f.seq_raw = PySequence_Fast(keys_obj.ptr(), "expected a sequence");
                if (!f.seq_raw)
                    throw nb::python_error();
                f.n = PySequence_Fast_GET_SIZE(f.seq_raw);
                f.seq_items = PySequence_Fast_ITEMS(f.seq_raw);
            }

            // If resuming from a child, attach was handled in generic pop; nothing to do here.

            if (f.i < f.n)
            {
                nb::handle key_h(f.seq_items[f.i]); // borrowed
                f.i++;
                nb::str k = nb::str(key_h); // enforce string key
                nb::object val = nb::cast<nb::object>(f.in.attr("__getitem__")(key_h));

                if (is_scalar(val))
                {
                    nb::dict out = nb::cast<nb::dict>(f.out);
                    out[k] = nb::object(val, nb::detail::borrow_t{});
                    continue;
                }

                // Descend; mark pending key
                f.pending_key = k;
                f.waiting_child = true;

                if (nb::isinstance<nb::list>(val))
                {
                    push_list(nb::cast<nb::list>(val));
                    continue;
                }
                if (nb::isinstance<nb::tuple>(val))
                {
                    push_tuple(nb::cast<nb::tuple>(val));
                    continue;
                }
                if (PySequence_Check(val.ptr()) &&
                    !PyMapping_Check(val.ptr()) &&
                    !PyDict_Check(val.ptr()) &&
                    !nb::isinstance<nb::str>(val) &&
                    !PyBytes_Check(val.ptr()))
                {
                    push_seq(val);
                    continue;
                }
                if (nb::isinstance<nb::dict>(val))
                {
                    push_dict(nb::cast<nb::dict>(val));
                    continue;
                }
                if (PyMapping_Check(val.ptr()))
                {
                    push_map(val);
                    continue;
                }
                if (PySet_Check(val.ptr()) || PyFrozenSet_Check(val.ptr()))
                {
                    push_set(val);
                    continue;
                }
                if (PyBytes_Check(val.ptr()) || PyByteArray_Check(val.ptr()))
                    throw nb::type_error("bytes/bytearray are not JSON-serialisable");
                throw nb::type_error("object type is not JSON-serialisable");
            }

            // done dict
            if (f.seq_raw)
            {
                Py_DECREF(f.seq_raw);
                f.seq_raw = nullptr;
            }
            nb::object done = f.out;
            stack.pop_back();
            if (stack.empty())
                return done;
            attach_done_to_parent(done);
            continue;
        }

        if (f.kind == Frame::MapK)
        {
            if (f.i < f.n)
            {
                nb::handle kvh = f.map_items[f.i];
                f.i++;
                nb::tuple kv = nb::cast<nb::tuple>(kvh);
                nb::str k = nb::str(kv[0]); // enforce string key
                nb::object val = nb::cast<nb::object>(kv[1]);

                if (is_scalar(val))
                {
                    nb::dict out = nb::cast<nb::dict>(f.out);
                    out[k] = nb::object(val, nb::detail::borrow_t{});
                    continue;
                }

                f.pending_key = k;
                f.waiting_child = true;

                if (nb::isinstance<nb::list>(val))
                {
                    push_list(nb::cast<nb::list>(val));
                    continue;
                }
                if (nb::isinstance<nb::tuple>(val))
                {
                    push_tuple(nb::cast<nb::tuple>(val));
                    continue;
                }
                if (PySequence_Check(val.ptr()) &&
                    !PyMapping_Check(val.ptr()) &&
                    !PyDict_Check(val.ptr()) &&
                    !nb::isinstance<nb::str>(val) &&
                    !PyBytes_Check(val.ptr()))
                {
                    push_seq(val);
                    continue;
                }
                if (nb::isinstance<nb::dict>(val))
                {
                    push_dict(nb::cast<nb::dict>(val));
                    continue;
                }
                if (PyMapping_Check(val.ptr()))
                {
                    push_map(val);
                    continue;
                }
                if (PySet_Check(val.ptr()) || PyFrozenSet_Check(val.ptr()))
                {
                    push_set(val);
                    continue;
                }
                if (PyBytes_Check(val.ptr()) || PyByteArray_Check(val.ptr()))
                    throw nb::type_error("bytes/bytearray are not JSON-serialisable");
                throw nb::type_error("object type is not JSON-serialisable");
            }
            nb::object done = f.out;
            stack.pop_back();
            if (stack.empty())
                return done;
            attach_done_to_parent(done);
            continue;
        }

        if (f.kind == Frame::SetK)
        {
            PyObject *nxt = PyIter_Next(f.iter);
            if (nxt)
            {
                nb::object elem(nb::handle(nxt), nb::detail::steal_t{});
                if (is_scalar(elem))
                {
                    nb::list out = nb::cast<nb::list>(f.out);
                    out.append(std::move(elem));
                    continue;
                }
                if (nb::isinstance<nb::list>(elem))
                {
                    push_list(nb::cast<nb::list>(elem));
                    continue;
                }
                if (nb::isinstance<nb::tuple>(elem))
                {
                    push_tuple(nb::cast<nb::tuple>(elem));
                    continue;
                }
                if (PySequence_Check(elem.ptr()) &&
                    !PyMapping_Check(elem.ptr()) &&
                    !PyDict_Check(elem.ptr()) &&
                    !nb::isinstance<nb::str>(elem) &&
                    !PyBytes_Check(elem.ptr()))
                {
                    push_seq(elem);
                    continue;
                }
                if (nb::isinstance<nb::dict>(elem))
                {
                    push_dict(nb::cast<nb::dict>(elem));
                    continue;
                }
                if (PyMapping_Check(elem.ptr()))
                {
                    push_map(elem);
                    continue;
                }
                if (PySet_Check(elem.ptr()) || PyFrozenSet_Check(elem.ptr()))
                {
                    push_set(elem);
                    continue;
                }
                if (PyBytes_Check(elem.ptr()) || PyByteArray_Check(elem.ptr()))
                    throw nb::type_error("bytes/bytearray are not JSON-serialisable");
                throw nb::type_error("object type is not JSON-serialisable");
            }
            if (PyErr_Occurred())
                throw nb::python_error();
            Py_DECREF(f.iter);
            f.iter = nullptr;
            nb::object done = f.out;
            stack.pop_back();
            if (stack.empty())
                return done;
            attach_done_to_parent(done);
            continue;
        }
    }

    // Unreachable on success
    throw nb::type_error("object type is not JSON-serialisable");
}

NB_MODULE(_event_signing_impl, m)
{
    // Initialize debug system
    // Enable leak warnings only when explicitly requested for profiling
    // Set SYNAPSE_RUST_CRYPTO_LEAK_WARNINGS=1 to enable
    nb::set_leak_warnings(leak_warnings_enabled);

    // AWS-LC doesn't require explicit initialization

    nb::exception<SignatureVerifyException>(m, "SignatureVerifyException");

    nb::class_<Signature>(m, "Signature")
        .def_rw("signature", &Signature::signature);

    nb::class_<SigningResult>(m, "SigningResult")
        .def_rw("signature", &SigningResult::signature)
        .def_rw("key_id", &SigningResult::key_id)
        .def_rw("algorithm", &SigningResult::algorithm);

    nb::class_<VerificationResult>(m, "VerificationResult")
        .def_rw("valid", &VerificationResult::valid)
        .def_rw("user_id", &VerificationResult::user_id)
        .def_rw("device_valid", &VerificationResult::device_valid);

    nb::class_<VerifyKey>(m, "VerifyKey")
        .def(nb::init<const std::vector<uint8_t> &, const std::string &, const std::string &>(),
             "bytes"_a, "alg"_a = "ed25519", "version"_a = "1")
        .def("encode", &VerifyKey::encode)
        .def("verify", &VerifyKey::verify)
        .def("__eq__", [](const VerifyKey &self, const VerifyKey &other)
             { return self.key_bytes == other.key_bytes && self.alg == other.alg && self.version == other.version; })
        .def_rw("alg", &VerifyKey::alg)
        .def_rw("version", &VerifyKey::version);

    nb::class_<VerifyKeyWithExpiry, VerifyKey>(m, "VerifyKeyWithExpiry")
        .def(nb::init<const std::vector<uint8_t> &, const std::string &, const std::string &>(),
             "bytes"_a, "alg"_a = "ed25519", "version"_a = "1")
        .def("__eq__", [](const VerifyKeyWithExpiry &self, const VerifyKey &other)
             { return self.key_bytes == other.key_bytes && self.alg == other.alg && self.version == other.version; })
        .def_rw("expired", &VerifyKeyWithExpiry::expired);

    nb::class_<SigningKey>(m, "SigningKey")
        .def(nb::init<const std::vector<uint8_t> &, const std::string &, const std::string &>(),
             "bytes"_a, "alg"_a = "ed25519", "version"_a = "1")
        .def_static("generate", &SigningKey::generate)
        .def("encode", &SigningKey::encode)
        .def("get_verify_key", &SigningKey::get_verify_key)
        .def_prop_ro("verify_key", &SigningKey::get_verify_key)
        .def("sign", &SigningKey::sign)
        .def_rw("alg", &SigningKey::alg)
        .def_rw("version", &SigningKey::version)
        .def("__bytes__", &SigningKey::encode)
        .def("__len__", [](const SigningKey &self)
             { return self.key_bytes.size(); })
        .def("__iter__", [](const SigningKey &self)
             { return nb::make_iterator(nb::type<SigningKey>(), "iterator", self.key_bytes.begin(), self.key_bytes.end()); });

    m.def("compute_content_hash_fast", [](const std::vector<uint8_t> &json_bytes)
          {
        auto result = compute_content_hash_fast(std::span<const uint8_t>(json_bytes));
        return std::make_pair(result.first, nb::bytes(reinterpret_cast<const char*>(result.second.data()), result.second.size())); });
    m.def("compute_content_hash", [](const nb::dict &event_dict)
          {
        auto result = compute_content_hash(event_dict);
        return std::make_pair(result.first, nb::bytes(reinterpret_cast<const char*>(result.second.data()), result.second.size())); });
    m.def("compute_event_reference_hash_fast", [](const std::string &event_json)
          {
        auto result = compute_content_hash_fast(event_json);
        return std::make_pair(result.first, nb::bytes(reinterpret_cast<const char*>(result.second.data()), result.second.size())); });
    m.def("compute_event_reference_hash", [](const nb::dict &event_dict)
          {
        auto result = compute_content_hash(event_dict);
        return std::make_pair(result.first, nb::bytes(reinterpret_cast<const char*>(result.second.data()), result.second.size())); });
    m.def("sign_json_fast", [](const std::vector<uint8_t> &json_bytes, const std::vector<uint8_t> &signing_key_bytes)
          { return sign_json_fast(std::span<const uint8_t>(json_bytes), signing_key_bytes); });
    m.def("sign_json_with_info", &sign_json_with_info);
    m.def("verify_signature_fast", [](const std::vector<uint8_t> &json_bytes, const std::string &signature_b64, const std::vector<uint8_t> &verify_key_bytes)
          { return verify_signature_fast(std::span<const uint8_t>(json_bytes), signature_b64, verify_key_bytes); });
    m.def("verify_signature_with_info", &verify_signature_with_info);
    m.def("sign_json_object_fast", &sign_json_object_fast);
    m.def("verify_signed_json_fast", &verify_signed_json_fast);
    m.def("get_verify_key", [](const SigningKey &signing_key)
          { return signing_key.get_verify_key(); });
    m.def("get_verify_key", [](const std::vector<uint8_t> &signing_key_bytes)
          {
        auto verify_key_bytes = get_verify_key(signing_key_bytes);
        return VerifyKey(verify_key_bytes, "ed25519", "1"); });
    m.def("get_verify_key", [](const nb::object &signing_key)
          {
        // Handle nacl.signing.SigningKey and other objects with encode() method
        if (nb::hasattr(signing_key, "encode")) {
            nb::bytes encoded = nb::cast<nb::bytes>(signing_key.attr("encode")());
            const char* ptr = static_cast<const char*>(encoded.c_str());
            size_t size = encoded.size();
            std::vector<uint8_t> key_bytes(ptr, ptr + size);
            
            if (key_bytes.size() == 32) {
                auto verify_key_bytes = get_verify_key(key_bytes);
                // Check if the signing key has a version attribute
                std::string version = "auto";
                if (nb::hasattr(signing_key, "version")) {
                    try {
                        version = nb::cast<std::string>(signing_key.attr("version"));
                    } catch (...) {
                        version = "auto";
                    }
                }
                return VerifyKey(verify_key_bytes, "ed25519", version);
            }
        }
        throw std::runtime_error("Invalid signing key object"); });
    m.def("encode_base64_fast", [](nb::bytes data)
          {
        const char* ptr = static_cast<const char*>(data.c_str());
        size_t size = data.size();
        
        std::vector<uint8_t> vec_data(ptr, ptr + size);
        return nb::str(base64_encode(vec_data).c_str()); });
    m.def("decode_base64_fast", [](const std::string &encoded)
          {
        // Remove whitespace like standard library does
        std::string clean_encoded = encoded;
        clean_encoded.erase(std::remove_if(clean_encoded.begin(), clean_encoded.end(), [](char c) {
            return std::isspace(static_cast<unsigned char>(c));
        }), clean_encoded.end());
        
        auto result = base64_decode(clean_encoded);
        return nb::bytes(reinterpret_cast<const char*>(result.data()), result.size()); });
    m.def("encode_base64", [](const nb::object &data)
          {
        if (nb::isinstance<nb::bytes>(data)) {
            nb::bytes bytes_data = nb::cast<nb::bytes>(data);
            const char* ptr = static_cast<const char*>(bytes_data.c_str());
            size_t size = bytes_data.size();
            std::vector<uint8_t> vec_data(ptr, ptr + size);
            return nb::str(base64_encode(vec_data).c_str());
        } else {
            std::vector<uint8_t> vec_data = nb::cast<std::vector<uint8_t>>(data);
            return nb::str(base64_encode(vec_data).c_str());
        } });
    // Synapse-compatible overload with urlsafe parameter
    m.def("encode_base64", [](nb::bytes data, bool urlsafe = false)
          {
        const char* ptr = static_cast<const char*>(data.c_str());
        size_t size = data.size();
        std::vector<uint8_t> vec_data(ptr, ptr + size);
        std::string result = base64_encode(vec_data);
        
        // Convert to URL-safe if requested (replace + with -, / with _)
        if (urlsafe) {
            std::replace(result.begin(), result.end(), '+', '-');
            std::replace(result.begin(), result.end(), '/', '_');
        }
        
        return nb::str(result.c_str()); }, "input_bytes"_a, "urlsafe"_a = false);
    // Additional overload for vector input (Synapse compatibility)
    m.def("encode_base64", [](const std::vector<uint8_t> &data, bool urlsafe = false)
          {
        std::string result = base64_encode(data);
        
        // Convert to URL-safe if requested
        if (urlsafe) {
            std::replace(result.begin(), result.end(), '+', '-');
            std::replace(result.begin(), result.end(), '/', '_');
        }
        
        return nb::str(result.c_str()); }, "input_bytes"_a, "urlsafe"_a = false);
    // Padded base64 encoding function
    m.def("encode_base64_padded", [](const nb::object &data, bool urlsafe = false)
          {
        DEBUG_LOG("encode_base64_padded called, urlsafe=" + std::string(urlsafe ? "true" : "false"));
        
        if (data.ptr() == Py_None) {
            DEBUG_LOG("encode_base64_padded: None input, throwing exception");
            throw std::invalid_argument("argument should be a bytes-like object or ASCII string, not 'NoneType'");
        }
        
        std::vector<uint8_t> vec_data;
        if (nb::isinstance<nb::bytes>(data)) {
            nb::bytes bytes_data = nb::cast<nb::bytes>(data);
            const char* ptr = static_cast<const char*>(bytes_data.c_str());
            size_t size = bytes_data.size();
            vec_data = std::vector<uint8_t>(ptr, ptr + size);
            DEBUG_LOG("encode_base64_padded: bytes input, size=" + std::to_string(size));
        } else {
            vec_data = nb::cast<std::vector<uint8_t>>(data);
            DEBUG_LOG("encode_base64_padded: vector input, size=" + std::to_string(vec_data.size()));
        }
        
        std::string result = base64_encode(vec_data);
        DEBUG_LOG("encode_base64_padded: unpadded result=" + result + ", length=" + std::to_string(result.length()));
        
        // Add padding
        size_t padding_needed = (4 - (result.length() % 4)) % 4;
        result.append(padding_needed, '=');
        DEBUG_LOG("encode_base64_padded: padded result=" + result + ", padding_added=" + std::to_string(padding_needed));
        
        // Convert to URL-safe if requested
        if (urlsafe) {
            std::replace(result.begin(), result.end(), '+', '-');
            std::replace(result.begin(), result.end(), '/', '_');
            DEBUG_LOG("encode_base64_padded: urlsafe result=" + result);
        }
        
        auto output = nb::bytes(result.c_str(), result.size());
        DEBUG_LOG("encode_base64_padded: returning bytes, size=" + std::to_string(result.size()));
        return output; }, nb::arg("data").none(true), "urlsafe"_a = false);
    m.def("decode_base64", [](const std::string &encoded)
          {
        // Remove whitespace like standard library does
        std::string clean_encoded = encoded;
        clean_encoded.erase(std::remove_if(clean_encoded.begin(), clean_encoded.end(), [](char c) {
            return std::isspace(static_cast<unsigned char>(c));
        }), clean_encoded.end());
        
        // Strip padding if present (matrices_evolved base64_decode expects unpadded input)
        while (!clean_encoded.empty() && clean_encoded.back() == '=') {
            clean_encoded.pop_back();
        }
        
        auto result = base64_decode(clean_encoded);
        return nb::bytes(reinterpret_cast<const char*>(result.data()), result.size()); });
    // Padded base64 decoding function
    m.def("decode_base64_padded", [](const nb::object &encoded)
          {
        DEBUG_LOG("decode_base64_padded called");
        
        if (encoded.ptr() == Py_None) {
            DEBUG_LOG("decode_base64_padded: None input, throwing exception");
            throw std::invalid_argument("argument should be a bytes-like object or ASCII string, not 'NoneType'");
        }
        
        std::string input;
        if (nb::isinstance<nb::str>(encoded)) {
            input = nb::cast<std::string>(encoded);
            DEBUG_LOG("decode_base64_padded: string input=" + input + ", length=" + std::to_string(input.length()));
        } else if (nb::isinstance<nb::bytes>(encoded)) {
            nb::bytes bytes_data = nb::cast<nb::bytes>(encoded);
            const char* ptr = static_cast<const char*>(bytes_data.c_str());
            size_t size = bytes_data.size();
            input = std::string(ptr, size);
            DEBUG_LOG("decode_base64_padded: bytes input=" + input + ", length=" + std::to_string(size));
        } else {
            DEBUG_LOG("decode_base64_padded: invalid input type, throwing exception");
            throw std::invalid_argument("argument should be a bytes-like object or ASCII string");
        }
        
        std::string original_input = input;
        
        // Remove whitespace (spaces, newlines, tabs, etc.) like standard library does
        input.erase(std::remove_if(input.begin(), input.end(), [](char c) {
            return std::isspace(static_cast<unsigned char>(c));
        }), input.end());
        DEBUG_LOG("decode_base64_padded: after whitespace removal=" + input + ", length=" + std::to_string(input.length()));
        
        // Convert URL-safe characters back to standard base64
        std::replace(input.begin(), input.end(), '-', '+');
        std::replace(input.begin(), input.end(), '_', '/');
        if (input != original_input) {
            DEBUG_LOG("decode_base64_padded: after URL-safe conversion=" + input);
        }
        
        // Remove padding since internal base64_decode only accepts unpadded input
        size_t padding_removed = 0;
        while (!input.empty() && input.back() == '=') {
            input.pop_back();
            padding_removed++;
        }
        DEBUG_LOG("decode_base64_padded: after padding removal=" + input + ", padding_removed=" + std::to_string(padding_removed));
        
        auto result = base64_decode(input);
        DEBUG_LOG("decode_base64_padded: decoded " + std::to_string(result.size()) + " bytes");
        
        auto output = nb::bytes(reinterpret_cast<const char*>(result.data()), result.size());
        DEBUG_LOG("decode_base64_padded: returning bytes, size=" + std::to_string(result.size()));
        return output; }, nb::arg("encoded").none(true));
    m.def("encode_verify_key_base64", [](const std::vector<uint8_t> &key_bytes)
          { return nb::str(base64_encode(key_bytes).c_str()); });
    m.def("encode_verify_key_base64", [](const VerifyKey &verify_key)
          {
        nb::bytes encoded = verify_key.encode();
        const char* ptr = static_cast<const char*>(encoded.c_str());
        size_t size = encoded.size();
        std::vector<uint8_t> key_bytes(ptr, ptr + size);
        return nb::str(base64_encode(key_bytes).c_str()); });
    m.def("encode_verify_key_base64", [](const nb::object &key)
          {
        // Handle any object with encode() method (nacl.signing.VerifyKey compatibility)
        nb::bytes encoded = nb::cast<nb::bytes>(key.attr("encode")());
        const char* ptr = static_cast<const char*>(encoded.c_str());
        size_t size = encoded.size();
        std::vector<uint8_t> key_bytes(ptr, ptr + size);
        return nb::str(base64_encode(key_bytes).c_str()); });
    m.def("encode_signing_key_base64", [](const std::vector<uint8_t> &key_bytes)
          { return nb::str(base64_encode(key_bytes).c_str()); });
    m.def("encode_signing_key_base64", [](const SigningKey &signing_key)
          {
        nb::bytes encoded = signing_key.encode();
        const char* ptr = static_cast<const char*>(encoded.c_str());
        size_t size = encoded.size();
        std::vector<uint8_t> key_bytes(ptr, ptr + size);
        return nb::str(base64_encode(key_bytes).c_str()); });
    m.def("decode_verify_key_base64", [](const std::string &algorithm, const std::string &version, const std::string &key_base64)
          {
        if (algorithm != "ed25519") throw std::runtime_error("Unsupported algorithm");
        
        // Remove whitespace like standard library does
        std::string clean_key = key_base64;
        clean_key.erase(std::remove_if(clean_key.begin(), clean_key.end(), [](char c) {
            return std::isspace(static_cast<unsigned char>(c));
        }), clean_key.end());
        
        auto key_bytes = base64_decode(clean_key);
        if (key_bytes.size() != 32) throw std::runtime_error("Invalid key length");
        return VerifyKey(key_bytes, algorithm, version); });
    m.def("decode_signing_key_base64", [](const std::string &algorithm, const std::string &version, const std::string &key_base64)
          {
        if (algorithm != "ed25519") throw std::runtime_error("Unsupported algorithm");
        
        // Remove whitespace like standard library does
        std::string clean_key = key_base64;
        clean_key.erase(std::remove_if(clean_key.begin(), clean_key.end(), [](char c) {
            return std::isspace(static_cast<unsigned char>(c));
        }), clean_key.end());
        
        auto key_bytes = base64_decode(clean_key);
        if (key_bytes.size() != 32) throw std::runtime_error("Invalid key length");
        return SigningKey(key_bytes, algorithm, version); });
    m.def("decode_verify_key_bytes_fast", [](std::string_view key_id, const std::vector<uint8_t> &key_bytes)
          {
        if (key_id.starts_with("ed25519:") && key_bytes.size() == 32) return key_bytes;
        throw std::runtime_error("Unsupported key type or invalid key length"); });
    m.def("decode_verify_key_bytes", [](std::string_view key_id, const nb::object &key_data) -> VerifyKeyWithExpiry
          {
        std::vector<uint8_t> key_bytes;
        
        if (nb::isinstance<nb::bytes>(key_data)) {
            nb::bytes bytes_data = nb::cast<nb::bytes>(key_data);
            const char* ptr = static_cast<const char*>(bytes_data.c_str());
            size_t size = bytes_data.size();
            key_bytes = std::vector<uint8_t>(ptr, ptr + size);
        } else {
            key_bytes = nb::cast<std::vector<uint8_t>>(key_data);
        }
        
        // Accept any key_id that starts with "ed25519:" and has 32-byte key data
        // This matches signedjson behavior which is more lenient
        if (key_id.starts_with("ed25519:") && key_bytes.size() == 32) {
            size_t colon_pos = key_id.find(':');
            std::string version = (colon_pos != std::string::npos) ? std::string(key_id.substr(colon_pos + 1)) : "1";
            return VerifyKeyWithExpiry(key_bytes, "ed25519", version);
        }
        throw std::runtime_error("Unsupported key type or invalid key length"); });
    m.def("is_signing_algorithm_supported", [](std::string_view key_id)
          { return key_id.starts_with("ed25519:"); });
    m.def("encode_canonical_json", [](const nb::object &input)
          {
    if (input.ptr() == Py_None ||
        PyBool_Check(input.ptr()) ||
        PyLong_Check(input.ptr()) ||
        PyFloat_Check(input.ptr()) ||
        PyUnicode_Check(input.ptr()) ||
        PyBytes_Check(input.ptr())) {
        init_json_buffer();
        py_to_canonical_json_fast(input);
        auto span = get_json_span();
        return nb::bytes(span.data(), span.size());
    }

    nb::object processed = normalize_for_canonical_json(input);

    if (PyDict_Check(processed.ptr())) {
        nb::dict d = nb::cast<nb::dict>(processed);
        init_json_buffer(d.size());
    } else {
        init_json_buffer();
    }

    py_to_canonical_json_fast(processed);
    auto span = get_json_span();
    return nb::bytes(span.data(), span.size()); }, nb::arg("input").none(true));

    m.def("signature_ids", [](const nb::dict &json_dict, const std::string &signature_name)
          {
        std::vector<std::string> ids;
        if (json_dict.contains("signatures")) {
            nb::dict signatures = nb::cast<nb::dict>(json_dict["signatures"]);
            if (signatures.contains(signature_name.c_str())) {
                nb::dict server_sigs = nb::cast<nb::dict>(signatures[signature_name.c_str()]);
                for (auto item : server_sigs) {
                    std::string_view key = nb::cast<std::string_view>(item.first);
                    if (key.starts_with("ed25519:")) ids.push_back(std::string(key));
                }
            }
        }
        return ids; });
    // Alias functions to match Rust API exactly
    m.def("sign_json", [](nb::dict &json_object, const std::string &signature_name, const nb::object &signing_key) -> nb::dict
          {
        std::string alg, version, key_id;
        
        // Try to get alg and version attributes (our SigningKey objects)
        try {
            alg = nb::cast<std::string>(signing_key.attr("alg"));
            version = nb::cast<std::string>(signing_key.attr("version"));
            key_id = alg + ":" + version;
        } catch (const std::exception&) {
            // Fallback for nacl.signing.SigningKey objects (no alg/version attributes)
            alg = "ed25519";
            version = "1";  // default version
            key_id = "ed25519:1";
        }
        
        nb::bytes encoded_key = nb::cast<nb::bytes>(signing_key.attr("encode")());
        
        // Convert nb::bytes to std::vector<uint8_t>
        const char* ptr = static_cast<const char*>(encoded_key.c_str());
        size_t size = encoded_key.size();
        std::vector<uint8_t> signing_key_bytes(ptr, ptr + size);
        
        nb::dict signed_dict = sign_json_object_fast(json_object, signature_name, signing_key_bytes, key_id);
        
        // Modify original dictionary in-place to match signedjson behavior
        json_object.clear();
        for (auto item : signed_dict) {
            json_object[item.first] = item.second;
        }
        
        // Return the signed dictionary for compatibility with Synapse's compute_event_signature
        return signed_dict; });
    m.def("verify_signature", [](const std::vector<uint8_t> &json_bytes, const std::string &signature_b64, const std::vector<uint8_t> &verify_key_bytes)
          { return verify_signature_fast(std::span<const uint8_t>(json_bytes), signature_b64, verify_key_bytes); });
    m.def("verify_signed_json", [](const nb::dict &json_dict, const std::string &signature_name, const nb::object &verify_key)
          {
        DEBUG_LOG("verify_signed_json: signature_name=" + signature_name);
        
        // Extract key info like Rust version
        std::string alg, version;
        std::vector<uint8_t> verify_key_bytes;
        
        if (nb::isinstance<nb::bytes>(verify_key) || nb::isinstance<nb::list>(verify_key)) {
            verify_key_bytes = nb::cast<std::vector<uint8_t>>(verify_key);
            alg = "ed25519";
            version = "auto";
            DEBUG_LOG("Got raw bytes, length=" + std::to_string(verify_key_bytes.size()));
            
            // Extract version from signatures if available (match Rust logic)
            if (json_dict.contains("signatures")) {
                nb::dict signatures = json_dict["signatures"];
                if (signatures.contains(signature_name.c_str())) {
                    nb::dict server_sigs = signatures[signature_name.c_str()];
                    for (auto item : server_sigs) {
                        std::string_view key_id = nb::cast<std::string_view>(item.first);
                        if (key_id.starts_with("ed25519:")) {
                            size_t colon_pos = key_id.find(':');
                            if (colon_pos != std::string::npos) {
                                version = key_id.substr(colon_pos + 1);
                                DEBUG_LOG("Using version from signatures: " + version);
                            }
                            break;
                        }
                    }
                }
            }
        } else {
            try {
                alg = nb::cast<std::string>(verify_key.attr("alg"));
                version = nb::cast<std::string>(verify_key.attr("version"));
                DEBUG_LOG("Got VerifyKey object: alg=" + alg + ", version=" + version);
                nb::bytes encoded_key = nb::cast<nb::bytes>(verify_key.attr("encode")());
                
                // Convert nb::bytes to std::vector<uint8_t>
                const char* ptr = static_cast<const char*>(encoded_key.c_str());
                size_t size = encoded_key.size();
                verify_key_bytes = std::vector<uint8_t>(ptr, ptr + size);
                DEBUG_LOG("Extracted key bytes, length=" + std::to_string(verify_key_bytes.size()));
            } catch (const std::exception& e) {
                DEBUG_LOG("Failed to extract key attributes, treating as raw bytes: " + std::string(e.what()));
                verify_key_bytes = nb::cast<std::vector<uint8_t>>(verify_key);
                alg = "ed25519";
                version = "auto";
            }
        }
        
        std::string key_id = alg + ":" + version;
        DEBUG_LOG("Using key_id: " + key_id);
        
        // Get signatures
        if (!json_dict.contains("signatures")) {
            DEBUG_LOG("ERROR: No signatures field found in JSON");
            throw SignatureVerifyException("No signatures on this object");
        }
        
        nb::dict signatures = json_dict["signatures"];
        if (!signatures.contains(signature_name.c_str())) {
            DEBUG_LOG("ERROR: Missing signature for server: " + signature_name);
            throw SignatureVerifyException("Missing signature for " + signature_name);
        }
        
        nb::dict server_sigs = signatures[signature_name.c_str()];
        std::string signature_b64;
        
        if (debug_enabled) {
            DEBUG_LOG("Available signature keys for " + signature_name + ":");
            for (auto item : server_sigs) {
                std::string_view available_key = nb::cast<std::string_view>(item.first);
                DEBUG_LOG("  - " + std::string(available_key));
            }
        }
        
        // Try exact key_id first, then fallback to any ed25519
        if (server_sigs.contains(key_id.c_str())) {
            signature_b64 = nb::cast<std::string>(server_sigs[key_id.c_str()]);
            DEBUG_LOG("Found exact key match: " + key_id + " -> " + signature_b64);
        } else {
            for (auto item : server_sigs) {
                std::string_view available_key = nb::cast<std::string_view>(item.first);
                if (available_key.starts_with("ed25519:")) {
                    signature_b64 = nb::cast<std::string>(item.second);
                    DEBUG_LOG("Using fallback key: " + std::string(available_key) + " -> " + signature_b64);
                    break;
                }
            }
        }
        
        if (signature_b64.empty()) {
            DEBUG_LOG("ERROR: No ed25519 signature found for " + signature_name + ", " + key_id);
            throw SignatureVerifyException("Missing signature for " + signature_name + ", " + key_id);
        }
        
        // Create unsigned copy
        nb::dict unsigned_dict;
        for (auto item : json_dict) {
            std::string_view key = nb::cast<std::string_view>(item.first);
            if (key != "signatures" && key != "unsigned") {
                unsigned_dict[std::string(key).c_str()] = item.second;
            }
        }
        
        DEBUG_LOG("Created unsigned dict with " + std::to_string(unsigned_dict.size()) + " keys");
        
        // Canonicalize and verify
        init_json_buffer(unsigned_dict.size());
        
        py_to_canonical_json_fast(unsigned_dict);
        
        auto json_bytes = get_json_span();
        
        if (debug_enabled) {
            std::string canonical_json(json_buffer.data(), json_ptr - json_buffer.data());
            DEBUG_LOG("Canonical JSON (" + std::to_string(canonical_json.size()) + " bytes): " + canonical_json.substr(0, 200) + (canonical_json.size() > 200 ? "..." : ""));
            DEBUG_LOG("Signature to verify: " + signature_b64);
            
            std::string hex;
            hex.reserve(verify_key_bytes.size() * 2);
            for (size_t i = 0; i < verify_key_bytes.size(); i++) {
                hex.push_back(hex_lut[verify_key_bytes[i] >> 4]);
                hex.push_back(hex_lut[verify_key_bytes[i] & 0x0F]);
            }
            DEBUG_LOG("Verify key bytes (full): " + hex);
        }
        
        bool verification_result = verify_signature_fast(json_bytes, signature_b64, verify_key_bytes);
        DEBUG_LOG("Verification result: " + std::string(verification_result ? "SUCCESS" : "FAILED"));
        
        if (!verification_result) {
            DEBUG_LOG("ERROR: Signature verification failed for " + signature_name);
            throw SignatureVerifyException("Unable to verify signature for " + signature_name);
        }
        
        DEBUG_LOG("Signature verification completed successfully"); });
    // Key management functions with version parameter (ignores version for compatibility)
    m.def("generate_signing_key", [](const std::string &key_id)
          { 
        DEBUG_LOG("generate_signing_key called with key_id: " + key_id);
        auto key_bytes = generate_signing_key();
        // Use key_id as version for compatibility with signedjson
        return SigningKey(key_bytes, "ed25519", key_id); });
    m.def("generate_signing_key", []()
          { 
        auto key_bytes = generate_signing_key();
        return SigningKey(key_bytes, "ed25519", "1"); });
    m.def("read_signing_keys", [](const nb::object &input_data)
          {
        DEBUG_LOG("read_signing_keys called");
        
        // Handle different input types like Python signedjson
        std::string content;
        if (nb::isinstance<nb::str>(input_data)) {
            // Handle string input
            content = nb::cast<std::string>(input_data);
            DEBUG_LOG("Got string input, length: " + std::to_string(content.length()));
        } else if (nb::isinstance<nb::list>(input_data)) {
            // Handle list input (like config does)
            auto py_list = nb::cast<nb::list>(input_data);
            DEBUG_LOG("Got list input with " + std::to_string(py_list.size()) + " items");
            std::vector<std::string> lines;
            for (auto item : py_list) {
                std::string line = nb::cast<std::string>(item);
                lines.push_back(line);
                DEBUG_LOG("List item: \"" + line + "\"");
            }
            // Join with newlines - vectorized for large arrays
            if (lines.size() >= 16) {
                DEBUG_LOG("Using optimized string joining for " + std::to_string(lines.size()) + " lines");
                // Pre-calculate total size to avoid reallocations
                size_t total_size = 0;
                for (const auto& line : lines) {
                    total_size += line.size() + 1; // +1 for newline
                }
                content.reserve(total_size);
            }
            
            for (size_t i = 0; i < lines.size(); ++i) {
                if (i > 0) content += "\n";
                content += lines[i];
            }
            DEBUG_LOG("Joined list content: \"" + content + "\"");
        } else {
            // Try to call read() method for file-like objects
            try {
                nb::object content_obj = input_data.attr("read")();
                content = nb::cast<std::string>(content_obj);
                DEBUG_LOG("Read from file-like object, length: " + std::to_string(content.length()));
            } catch (...) {
                // Fallback to string conversion
                content = nb::cast<std::string>(nb::str(input_data));
                DEBUG_LOG("Fallback string conversion, length: " + std::to_string(content.length()));
            }
        }
        
        DEBUG_LOG("Content to parse: \"" + content + "\"");
        
        // Parse signing keys from content
        nb::list signing_keys;
        
        std::istringstream stream(content);
        std::string line;
        int line_num = 0;
        while (std::getline(stream, line)) {
            line_num++;
            DEBUG_LOG("Processing line " + std::to_string(line_num) + ": \"" + line + "\"");
            
            // Trim whitespace
            line.erase(0, line.find_first_not_of(" \t\r\n"));
            line.erase(line.find_last_not_of(" \t\r\n") + 1);
            
            if (line.empty() || line[0] == '#') {
                DEBUG_LOG("Skipping empty or comment line");
                continue;
            }
            
            // Expected format: "ed25519 version base64_key"
            std::istringstream line_stream(line);
            std::string algorithm, version, key_b64;
            if (line_stream >> algorithm >> version >> key_b64) {
                DEBUG_LOG("Parsed: algorithm=" + algorithm + ", version=" + version + ", key_b64=" + key_b64);
                if (algorithm == "ed25519" && !version.empty() && !key_b64.empty()) {
                    try {
                        auto key_bytes = base64_decode(key_b64);
                        DEBUG_LOG("Decoded key bytes, length: " + std::to_string(key_bytes.size()));
                        if (key_bytes.size() == 32) {
                            // Create SigningKey object
                            SigningKey key(key_bytes, "ed25519", version);
                            signing_keys.append(nb::cast(key));
                            DEBUG_LOG("Added signing key with version: " + version);
                        } else {
                            DEBUG_LOG("Invalid key length: " + std::to_string(key_bytes.size()));
                        }
                    } catch (const std::exception& e) {
                        DEBUG_LOG("Failed to decode key: " + std::string(e.what()));
                        continue;
                    }
                } else {
                    DEBUG_LOG("Invalid key format or algorithm");
                }
            } else {
                DEBUG_LOG("Failed to parse line format");
            }
        }
        
        DEBUG_LOG("Returning " + std::to_string(signing_keys.size()) + " signing keys");
        return signing_keys; });
    m.def("read_old_signing_keys", [](const nb::object &stream_content)
          { return nb::list(); });
    // Original signedjson signature: write_signing_keys(stream, keys)
    m.def("write_signing_keys", [](const nb::object &stream, const nb::object &keys)
          {
        std::string output;
        
        // Convert keys to list (handle both tuples and lists)
        nb::list key_list;
        if (nb::isinstance<nb::tuple>(keys)) {
            auto key_tuple = nb::cast<nb::tuple>(keys);
            key_list = nb::list();
            for (size_t i = 0; i < key_tuple.size(); ++i) {
                key_list.append(key_tuple[i]);
            }
        } else if (nb::isinstance<nb::list>(keys)) {
            key_list = nb::cast<nb::list>(keys);
        } else {
            return; // Invalid input type
        }
        
        // Process each signing key
        for (auto key_obj : key_list) {
            if (nb::hasattr(key_obj, "encode") && nb::hasattr(key_obj, "version")) {
                try {
                    std::string version = nb::cast<std::string>(key_obj.attr("version"));
                    nb::bytes encoded = nb::cast<nb::bytes>(key_obj.attr("encode")());
                    
                    const char* ptr = static_cast<const char*>(encoded.c_str());
                    size_t size = encoded.size();
                    std::vector<uint8_t> key_bytes(ptr, ptr + size);
                    
                    if (key_bytes.size() == 32) {
                        std::string key_b64 = base64_encode(key_bytes);
                        output += "ed25519 " + version + " " + key_b64 + "\n";
                    }
                } catch (const std::exception&) {
                    continue; // Skip invalid keys
                }
            }
        }
        
        // Write to stream
        if (nb::hasattr(stream, "write")) {
            stream.attr("write")(output);
        } });
    
    // JSON encoder - equivalent to json.dumps() with compact separators
    m.def("json_encode", &json_encode, "Encode Python object to JSON string", 
          nb::arg("obj"), nb::arg("preserve_utf8") = false);
    
    // JSON decoder - equivalent to json.loads()
    m.def("json_decode", [](const std::string &json_str)
          { return json_decode(json_str); });
}