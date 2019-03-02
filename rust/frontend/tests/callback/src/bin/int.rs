#![feature(extern_crate_item_prelude, try_from)]
#![allow(unused_imports)]

extern crate tvm_frontend as tvm;

use std::convert::TryInto;
use tvm::*;

fn main() {
    fn sum(args: &[TVMArgValue]) -> Result<TVMRetValue> {
        let mut ret = 0i64;
        for arg in args.iter() {
            let val: i64 = arg.try_into()?;
            ret += val;
        }
        Ok(TVMRetValue::from(&ret))
    }

    tvm::function::register(sum, "mysum".to_owned(), false).unwrap();

    let mut registered = function::Builder::default();
    registered.get_function("mysum", true);
    assert!(registered.func.is_some());
    let ret: i64 = registered
        .args(&[10, 20, 30])
        .invoke()
        .unwrap()
        .try_into()
        .unwrap();
    assert_eq!(ret, 60);
}
