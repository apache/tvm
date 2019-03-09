#![feature(extern_crate_item_prelude, panic_info_message)]
#![allow(unused_imports)]

use std::panic;

#[macro_use]
extern crate tvm_frontend as tvm;

use tvm::*;

fn main() {
    register_global_func! {
        fn error(_args: &[TVMArgValue]) -> Result<TVMRetValue> {
            Err(ErrorKind::TypeMismatch(
                format!("{}", "i64".to_string()),
                format!("{}", "f64".to_string()),
            ).into())
        }
    }

    let mut registered = function::Builder::default();
    registered.get_function("error", true);
    assert!(registered.func.is_some());
    registered.args(&[10, 20]);

    println!("expected error message is:");
    panic::set_hook(Box::new(|panic_info| {
        if let Some(msg) = panic_info.message() {
            println!("{:?}", msg);
        }
        if let Some(location) = panic_info.location() {
            println!(
                "panic occurred in file '{}' at line {}",
                location.file(),
                location.line()
            );
        } else {
            println!("panic occurred but can't get location information");
        }
    }));

    let _result = registered.invoke();
}
