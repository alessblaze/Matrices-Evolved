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
