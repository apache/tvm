#![feature(extern_crate_item_prelude, try_from)]
#![allow(unused_imports)]

#[macro_use]
extern crate tvm_frontend as tvm;

use std::convert::TryInto;
use tvm::*;

fn main() {
    register_global_func! {
        fn sum(args: &[TVMArgValue]) -> Result<TVMRetValue> {
            let mut ret = 0.0;
            for arg in args.iter() {
                let val: f64 = arg.try_into()?;
                ret += val;
            }
            Ok(TVMRetValue::from(&ret))
        }
    }

    let mut registered = function::Builder::default();
    registered.get_function("sum", true);
    assert!(registered.func.is_some());
    let ret: f64 = registered
        .args(&[10.0f64, 20.0, 30.0])
        .invoke()
        .unwrap()
        .try_into()
        .unwrap();
    assert_eq!(ret, 60f64);
}
