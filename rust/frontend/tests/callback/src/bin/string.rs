#![feature(extern_crate_item_prelude, try_from)]
#![allow(unused_imports)]

#[macro_use]
extern crate tvm_frontend as tvm;
use std::convert::TryInto;
use tvm::*;

// FIXME
fn main() {
    register_global_func! {
        fn concate_str(args: &[TVMArgValue]) -> Result<TVMRetValue> {
            let mut ret = "".to_string();
            for arg in args.iter() {
                let val: String = arg.try_into()?;
                ret += val.as_str();
            }
            Ok(TVMRetValue::from(ret))
        }
    }
    let mut registered = function::Builder::default();
    registered.get_function("concate_str", true);
    assert!(registered.func.is_some());
    let a = "a".to_string();
    let b = "b".to_string();
    let c = "c".to_string();
    let ret: String = registered
        .args(&[a, b, c])
        .invoke()
        .unwrap()
        .try_into()
        .unwrap();
    assert_eq!(ret, "abc".to_owned());
}
