use std::{
    collections::HashMap, convert::AsRef, ffi::CStr, os::raw::c_char, string::String, sync::Mutex,
};
use tvm_common::{
    ffi::BackendPackedCFunc,
    packed_func::{PackedFunc, TVMArgValue, TVMRetValue, TVMValue},
};

pub trait Module {
    fn get_function<S: AsRef<str>>(&self, name: S) -> Option<PackedFunc>;
}

pub struct SystemLibModule;

lazy_static! {
    static ref SYSTEM_LIB_FUNCTIONS: Mutex<HashMap<String, BackendPackedCFunc>> =
        Mutex::new(HashMap::new());
}

impl Module for SystemLibModule {
    fn get_function<S: AsRef<str>>(&self, name: S) -> Option<PackedFunc> {
        SYSTEM_LIB_FUNCTIONS
            .lock()
            .unwrap()
            .get(name.as_ref())
            .map(|func| wrap_backend_packed_func(func.to_owned()))
    }
}

impl Default for SystemLibModule {
    fn default() -> Self {
        SystemLibModule {}
    }
}

// @see `WrapPackedFunc` in `llvm_module.cc`.
pub(super) fn wrap_backend_packed_func(func: BackendPackedCFunc) -> PackedFunc {
    box move |args: &[TVMArgValue]| {
        func(
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
        TVMRetValue::default()
    }
}

#[no_mangle]
pub extern "C" fn TVMBackendRegisterSystemLibSymbol(
    cname: *const c_char,
    func: BackendPackedCFunc,
) -> i32 {
    let name = unsafe { CStr::from_ptr(cname).to_str().unwrap() };
    SYSTEM_LIB_FUNCTIONS
        .lock()
        .unwrap()
        .insert(name.to_string(), func);
    return 0;
}
