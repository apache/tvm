#![allow(unused_imports)]

#[macro_use]
extern crate tvm_frontend as tvm;

use std::convert::TryInto;
use tvm::{errors::Error, *};

fn main() {
    register_global_func! {
        fn sum(args: &[TVMArgValue]) -> Result<TVMRetValue, Error> {
            let mut ret = 0.0;
            for arg in args.into_iter() {
                let val: f64 = arg.try_into()?;
                ret += val;
            }
            Ok(TVMRetValue::from(ret))
        }
    }

    let mut registered = function::Builder::default();
    registered.get_function("sum");
    assert!(registered.func.is_some());
    let ret: f64 = registered
        .args(&[10.0f64, 20.0, 30.0])
        .invoke()
        .unwrap()
        .try_into()
        .unwrap();
    assert_eq!(ret, 60f64);
}
