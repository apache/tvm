/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

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
    // let main_mod = initialize();
    //let main_mod = main_mod.as_ref(py);
    load_python_tvm_(py).map_err(|e| {
        // We can't display Python exceptions via std::fmt::Display,
        // so print the error here manually.
        e.print_and_set_sys_last_vars(py);
    })
}

pub fn import(mod_to_import: &str) -> PyResult<()> {
    let gil = Python::acquire_gil();
    let py = gil.python();
    import_python(py, mod_to_import)?;
    Ok(())
}

fn import_python<'p, 'b: 'p>(py: Python<'p>, to_import: &'b str) -> PyResult<&'p PyModule> {
    let imported_mod = py.import(to_import)?;
    Ok(imported_mod)
}

fn load_python_tvm_(py: Python) -> PyResult<String> {
    let imported_mod = import_python(py, "tvm")?;
    let version: String = imported_mod.get("__version__")?.extract()?;
    Ok(version)
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;

    #[ignore]
    #[test]
    fn test_run() -> Result<()> {
        load().unwrap();
        Ok(())
    }
}
