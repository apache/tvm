//! Provides the [`Module`] type and methods for working with runtime TVM modules.

use std::{
    convert::TryInto,
    ffi::CString,
    mem,
    os::raw::{c_char, c_int},
    path::Path,
    ptr,
};

use ts;

use function::Function;
use internal_api;
use ErrorKind;
use Result;

const ENTRY_FUNC: &'static str = "__tvm_main__";

/// Wrapper around TVM module handle which contains an entry function.
/// The entry function can be applied to an imported module through [`entry_func`].
/// Also [`is_released`] shows whether the module is dropped or not.
///
/// [`entry_func`]:struct.Module.html#method.entry_func
/// [`is_released`]:struct.Module.html#method.is_released
#[derive(Debug, Clone)]
pub struct Module {
    pub(crate) handle: ts::TVMModuleHandle,
    is_released: bool,
    pub(crate) entry: Option<Function>,
}

impl Module {
    pub(crate) fn new(
        handle: ts::TVMModuleHandle,
        is_released: bool,
        entry: Option<Function>,
    ) -> Self {
        Self {
            handle,
            is_released,
            entry,
        }
    }

    /// Sets the entry function of a module.
    pub fn entry_func(&mut self) {
        if self.entry.is_none() {
            self.entry = self.get_function(ENTRY_FUNC, false).ok();
        }
    }

    /// Gets a function by name from a registered module.
    pub fn get_function(&self, name: &str, query_import: bool) -> Result<Function> {
        let name = CString::new(name)?;
        let mut fhandle = ptr::null_mut() as ts::TVMFunctionHandle;
        check_call!(ts::TVMModGetFunction(
            self.handle,
            name.as_ptr() as *const c_char,
            query_import as c_int,
            &mut fhandle as *mut _
        ));
        if fhandle.is_null() {
            bail!(ErrorKind::NullHandle(format!("{}", name.into_string()?)))
        } else {
            mem::forget(name);
            Ok(Function::new(fhandle, false, false))
        }
    }

    /// Imports a dependent module such as `.ptx` for gpu.
    pub fn import_module(&self, dependent_module: Module) {
        check_call!(ts::TVMModImport(self.handle, dependent_module.handle))
    }

    /// Loads a module shared library from path.
    pub fn load(path: &Path) -> Result<Module> {
        let path = path.to_owned();
        let path_str = path.to_str()?.to_owned();
        let ext = path.extension()?.to_str()?.to_owned();
        let func = internal_api::get_api("module._LoadFromFile".to_owned());
        let ret: Module = call_packed!(func, &path_str, &ext)?.try_into()?;
        mem::forget(path);
        Ok(ret)
    }

    /// Checks if a target device is enabled for a module.
    pub fn enabled(&self, target: &str) -> bool {
        let func = internal_api::get_api("module._Enabled".to_owned());
        // `unwrap` is safe here because if there is any error during the
        // function call, it would occur in `call_packed!`.
        let ret: i64 = call_packed!(func, target).unwrap().try_into().unwrap();
        ret != 0
    }

    /// Returns the underlying module handle.
    pub fn handle(&self) -> ts::TVMModuleHandle {
        self.handle
    }

    /// Returns true if the underlying module has been dropped and false otherwise.
    pub fn is_released(&self) -> bool {
        self.is_released
    }
}

impl Drop for Module {
    fn drop(&mut self) {
        if !self.is_released {
            check_call!(ts::TVMModFree(self.handle));
            self.is_released = true;
        }
    }
}
