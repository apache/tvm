use tvm::runtime::function::{register, Function};
use tvm_sys::ffi;
use std::ffi::{c_void};
use std::os::raw::c_int;
use tvm::runtime::ObjectRef;
use tvm::transform::{PassInfo, create_func_pass};
use tvm::ir::relay;

use tvm::runtime::{TVMArgValue, TVMRetValue};

type IRModule = ObjectRef;

fn function_pass(args: &[TVMArgValue]) -> anyhow::Result<TVMRetValue> {
    use std::convert::TryInto;
    let o: ObjectRef = args[0].try_into()?;
    let var = relay::Var::new("Hi from Rust!".into(), ObjectRef::null());
    Ok(var.into())
}

fn the_pass(args: &[TVMArgValue]) -> anyhow::Result<TVMRetValue> {
    let pass_info = PassInfo::new(15, "RustPass".into(), vec![])?;
    let pass_func = Function::get("__rust_pass").unwrap();
    let pass = create_func_pass(pass_func, pass_info)?;
    Ok(pass.into())
}

#[no_mangle]
pub unsafe extern "C" fn initialize(
    args: *mut ffi::TVMValue,
    type_codes: *mut c_int,
    num_args: c_int,
    ret: ffi::TVMRetValueHandle,
) -> c_int {
    register(function_pass, "__rust_pass", true);
    register(the_pass, "out_of_tree.Pass", true);
    return 0
}
