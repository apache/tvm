use std::{
    collections::HashMap, convert::AsRef, ffi::CStr, os::raw::c_char, string::String, sync::Mutex,
};

use tvm_common::{ffi::BackendPackedCFunc, packed_func::PackedFunc};

use super::Module;

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

#[no_mangle]
pub extern "C" fn TVMBackendRegisterSystemLibSymbol(
    cname: *const c_char,
    func: BackendPackedCFunc,
) -> i32 {
    let name = unsafe { CStr::from_ptr(cname).to_str().unwrap() };
    SYSTEM_LIB_FUNCTIONS.lock().unwrap().insert(
        name.to_string(),
        &*Box::leak(super::wrap_backend_packed_func(name.to_string(), func)),
    );
    return 0;
}
