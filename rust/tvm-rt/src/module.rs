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

//! Provides the [`Module`] type and methods for working with runtime TVM modules.

use std::{
    ffi::CString,
    os::raw::{c_char, c_int},
    path::Path,
    ptr,
};

use crate::object::Object;
use tvm_macros::Object;
use tvm_sys::ffi;

use crate::errors::Error;
use crate::String as TString;
use crate::{errors, function::Function};

/// Wrapper around TVM module handle which contains an entry function.
/// The entry function can be applied to an imported module through [`entry_func`].
///
/// [`entry_func`]:struct.Module.html#method.entry_func
#[repr(C)]
#[derive(Object, Debug)]
#[ref_name = "Module"]
#[type_key = "runtime.Module"]
pub struct ModuleNode {
    base: Object,
}

crate::external! {
    #[name("runtime.RuntimeEnabled")]
    fn runtime_enabled(target: CString) -> i32;

    #[name("runtime.ModuleLoadFromFile")]
    fn load_from_file(file_name: CString, format: CString) -> Module;

    #[name("runtime.ModuleSaveToFile")]
    fn save_to_file(module: Module, name: TString, fmt: TString);

    // TODO(@jroesch): we need to refactor this
    #[name("tvm.relay.module_export_library")]
    fn export_library(module: Module, file_name: TString);
}

impl Module {
    pub fn default_fn(&mut self) -> Result<Function, Error> {
        self.get_function("default", true)
    }

    /// Gets a function by name from a registered module.
    pub fn get_function(&self, name: &str, query_import: bool) -> Result<Function, Error> {
        let name = CString::new(name)?;
        let mut fhandle = ptr::null_mut() as ffi::TVMFunctionHandle;

        check_call!(ffi::TVMModGetFunction(
            self.handle(),
            name.as_ptr() as *const c_char,
            query_import as c_int,
            &mut fhandle as *mut _
        ));

        if fhandle.is_null() {
            return Err(errors::Error::NullHandle(name.into_string()?.to_string()));
        }

        Ok(Function::from_raw(fhandle))
    }

    /// Imports a dependent module such as `.ptx` for cuda gpu.
    pub fn import_module(&self, dependent_module: Module) {
        check_call!(ffi::TVMModImport(self.handle(), dependent_module.handle()))
    }

    /// Loads a module shared library from path.
    pub fn load<P: AsRef<Path>>(path: &P) -> Result<Module, Error> {
        let ext = CString::new(
            path.as_ref()
                .extension()
                .unwrap_or_else(|| std::ffi::OsStr::new(""))
                .to_str()
                .ok_or_else(|| Error::ModuleLoadPath(path.as_ref().display().to_string()))?,
        )?;

        let cpath = CString::new(
            path.as_ref()
                .to_str()
                .ok_or_else(|| Error::ModuleLoadPath(path.as_ref().display().to_string()))?,
        )?;

        let module = load_from_file(cpath, ext)?;
        Ok(module)
    }

    pub fn save_to_file(&self, name: String, fmt: String) -> Result<(), Error> {
        save_to_file(self.clone(), name.into(), fmt.into())
    }

    pub fn export_library(&self, name: String) -> Result<(), Error> {
        export_library(self.clone(), name.into())
    }

    /// Checks if a target device is enabled for a module.
    pub fn enabled(&self, target: &str) -> bool {
        let target = CString::new(target).unwrap();
        let enabled = runtime_enabled(target).unwrap();
        enabled != 0
    }

    /// Returns the underlying module handle.
    pub unsafe fn handle(&self) -> ffi::TVMModuleHandle {
        self.0.clone().unwrap().into_raw() as *mut _
    }
}
