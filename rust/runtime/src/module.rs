use std::{
    collections::HashMap, convert::AsRef, ffi::CStr, os::raw::c_char, string::String, sync::Mutex,
};

use tvm_common::{
    ffi::BackendPackedCFunc,
    packed_func::{PackedFunc, TVMArgValue, TVMRetValue, TVMValue},
};

pub trait Module {
    fn get_function<S: AsRef<str>>(&self, name: S) -> Option<&(dyn PackedFunc)>;
}

pub struct SystemLibModule;

lazy_static! {
    static ref SYSTEM_LIB_FUNCTIONS: Mutex<HashMap<String, &'static (dyn PackedFunc)>> =
        Mutex::new(HashMap::new());
}

impl Module for SystemLibModule {
    fn get_function<S: AsRef<str>>(&self, name: S) -> Option<&(dyn PackedFunc)> {
        SYSTEM_LIB_FUNCTIONS
            .lock()
            .unwrap()
            .get(name.as_ref())
            .map(|f| *f)
    }
}

impl Default for SystemLibModule {
    fn default() -> Self {
        SystemLibModule {}
    }
}

// @see `WrapPackedFunc` in `llvm_module.cc`.
pub(super) fn wrap_backend_packed_func(
    func_name: String,
    func: BackendPackedCFunc,
) -> Box<dyn PackedFunc> {
    box move |args: &[TVMArgValue]| {
        let exit_code = func(
            args.iter()
                .map(|ref arg| arg.value)
                .collect::<Vec<TVMValue>>()
                .as_ptr(),
            args.iter()
                .map(|ref arg| arg.type_code as i32)
                .collect::<Vec<i32>>()
                .as_ptr() as *const i32,
            args.len() as i32,
        );
        if exit_code == 0 {
            Ok(TVMRetValue::default())
        } else {
            Err(tvm_common::errors::FuncCallError::get_with_context(
                func_name.clone(),
            ))
        }
    }
}

#[no_mangle]
pub extern "C" fn TVMBackendRegisterSystemLibSymbol(
    cname: *const c_char,
    func: BackendPackedCFunc,
) -> i32 {
    let name = unsafe { CStr::from_ptr(cname).to_str().unwrap() };
    SYSTEM_LIB_FUNCTIONS.lock().unwrap().insert(
        name.to_string(),
        &*Box::leak(wrap_backend_packed_func(name.to_string(), func)),
    );
    return 0;
}
