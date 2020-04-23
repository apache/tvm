use std::ffi::c_void;
use std::os::raw::c_int;
use tvm::ir::relay;
use tvm::runtime::function::{register, Function};
use tvm::runtime::ObjectRef;
use tvm::transform::{create_func_pass, PassInfo};
use tvm_sys::ffi;

use tvm::runtime::{TVMArgValue, TVMRetValue};

type IRModule = ObjectRef;

fn function_pass(args: &[TVMArgValue]) -> anyhow::Result<TVMRetValue> {
    use std::convert::TryInto;
    println!("{:?}", args[0]);
    let arg_0 = args[0].clone();
    println!("{:?}", arg_0);
    // let optr: ObjectRef = arg_0.clone().try_into()?;
    // println!("{:?}", optr.0.as_ref().map(|o| o.ptr));
    // let func: relay::Function = (&args[0]).try_into()?;
    let func: relay::Function = arg_0.try_into()?;
    let var = relay::Var::new("Hi from Rust!".into(), ObjectRef::null());
    Ok(relay::Function::new(
        func.params.clone(),
        var.to_expr(),
        func.ret_type.clone(),
        func.type_params.clone(),
    )
    .into())
}

// fn function_pass(args: &[TVMArgValue]) -> anyhow::Result<TVMRetValue> {
//     let arg_0 = args[0].clone();
//     let func: relay::Function = arg_0.try_into()?;
//     let var = relay::Var::new("Hi from Rust!".into(), ObjectRef::null());
//     Ok(relay::Function::new(
//         func.params.clone(), var.to_expr(), func.ret_type.clone(), func.type_params.clone()).into())
// }

fn the_pass(args: &[TVMArgValue]) -> anyhow::Result<TVMRetValue> {
    println!("fooooooo");
    let pass_info = PassInfo::new(15, "RustPass".into(), vec![])?;
    let pass_func = Function::get("__rust_pass").unwrap();
    let pass = create_func_pass(pass_func, pass_info)?;
    println!("baz");
    // println!("{:?}", pass);
    Ok(pass.into())
}

#[no_mangle]
pub unsafe extern "C" fn initialize(
    args: *mut ffi::TVMValue,
    type_codes: *mut c_int,
    num_args: c_int,
    ret: ffi::TVMRetValueHandle,
) -> c_int {
    register(function_pass, "__rust_pass", true).unwrap();
    register(the_pass, "out_of_tree.Pass", true).unwrap();
    return 0;
}
