#[cfg(not(any(target_arch = "wasm32", target_env = "sgx")))]
mod dylib;
mod syslib;

use tvm_common::{
    ffi::BackendPackedCFunc,
    packed_func::{PackedFunc, TVMArgValue, TVMRetValue, TVMValue},
};

#[cfg(not(any(target_arch = "wasm32", target_env = "sgx")))]
pub use dylib::DylibModule;
pub use syslib::SystemLibModule;

pub trait Module {
    fn get_function<S: AsRef<str>>(&self, name: S) -> Option<&(dyn PackedFunc)>;
}

// @see `WrapPackedFunc` in `llvm_module.cc`.
fn wrap_backend_packed_func(func_name: String, func: BackendPackedCFunc) -> Box<dyn PackedFunc> {
    box move |args: &[TVMArgValue]| {
        let (values, type_codes): (Vec<TVMValue>, Vec<i32>) = args
            .into_iter()
            .map(|arg| {
                let (val, code) = arg.to_tvm_value();
                (val, code as i32)
            })
            .unzip();
        let exit_code = func(values.as_ptr(), type_codes.as_ptr(), values.len() as i32);
        if exit_code == 0 {
            Ok(TVMRetValue::default())
        } else {
            Err(tvm_common::errors::FuncCallError::get_with_context(
                func_name.clone(),
            ))
        }
    }
}
