/*
 * Copyright (C) 2025 Aless Microsystems
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published
 * by the Free Software Foundation, version 3 of the License, or under
 * alternative licensing terms as granted by Aless Microsystems.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 */


use pyo3::prelude::*;

pub mod crypto;


use pyo3::exceptions::PyException;

// Create SignatureVerifyException equivalent
pyo3::create_exception!(matrices_evolved, SignatureVerifyException, PyException);

#[pymodule]
fn matrices_evolved_rust(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Register all functions and classes
    crypto::register_module(py, m)?;
 
    
    Ok(())
}
