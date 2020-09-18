use pyo3::prelude::*;

/// Load the Python interpreter into the address space.
///
/// This enables the ability for Rust code to call TVM
/// functionality defined in Python.
///
/// For example registered TVM functions can now be
/// obtained via `Function::get`.
pub fn load() -> Result<String, ()> {
    let gil = Python::acquire_gil();
    let py = gil.python();
    load_python_tvm_(py).map_err(|e| {
        // We can't display Python exceptions via std::fmt::Display,
        // so print the error here manually.
        e.print_and_set_sys_last_vars(py);
    })
}

// const TVMC_CODE: &'static str = include_str!("tvmc.py");

fn load_python_tvm_(py: Python) -> PyResult<String> {
    let sys = py.import("tvm")?;
    let version: String = sys.get("__version__")?.extract()?;
    // py.run(TVMC_CODE, None, None)?;
    Ok(version)
}

#[cfg(test)]
mod tests {
    use super::load_python_tvm_;
    use anyhow::Result;
    use pyo3::prelude::*;

    #[ignore]
    #[test]
    fn test_run() -> Result<()> {
        load_python_tvm_(Python::acquire_gil().python()).unwrap();
        Ok(())
    }
}
