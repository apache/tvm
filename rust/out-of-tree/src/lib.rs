use tvm::runtime::function::register;
use tvm_sys::ffi;
use std::ffi::{c_void};
use std::os::raw::c_int;
use tvm::runtime::ObjectRef;
use tvm::transform::{PassInfo, create_func_pass};

use tvm::runtime::{TVMArgValue, TVMRetValue};

type IRModule = ObjectRef;

fn pass_function(Object
fn the_pass(args: &[TVMArgValue]) -> anyhow::Result<TVMRetValue> {
    let pass_info = PassInfo::new(15, "RustPass".into(), vec![])?;
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
    register(the_pass, "out_of_tree.Pass", true);
    return 0
}
