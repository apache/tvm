use std::ffi::c_void;
use std::os::raw::c_int;
use tvm::ir::relay::{self, Function};
use tvm::runtime::ObjectRef;
use tvm::transform::{function_pass, PassInfo, Pass, PassContext, IRModule};
use tvm::runtime::function::{register, Result};
use tvm::export_pass;

fn my_pass_fn(func: relay::Function, module: IRModule, ctx: PassContext) -> Function {
    let var = relay::Var::new("Hi from Rust!".into(), ObjectRef::null());
    relay::Function::new(
        func.params.clone(),
        var.to_expr(),
        func.ret_type.clone(),
        func.type_params.clone())
}

fn my_pass_fn2(func: relay::Function) -> Function {
    let var = relay::Var::new("Hi from Rust!".into(), ObjectRef::null());
    relay::Function::new(
        func.params.clone(),
        var.to_expr(),
        func.ret_type.clone(),
        func.type_params.clone())
}


// fn the_pass() -> Result<Pass> {
//     let pass_info = PassInfo::new(15, "RustPass".into(), vec![])?;
//     function_pass(my_pass_fn, pass_info)
// }

export_pass!("out_of_tree.Pass", my_pass_fn2);
