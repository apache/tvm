#![allow(unused_imports)]

#[macro_use]
extern crate tvm_frontend as tvm;
use std::convert::TryInto;
use tvm::{errors::Error, *};

// FIXME
fn main() {
    register_global_func! {
        fn concate_str(args: &[TVMArgValue]) -> Result<TVMRetValue, Error> {
            let mut ret = "".to_string();
            for arg in args.iter() {
                let val: &str = arg.try_into()?;
                ret += val;
            }
            Ok(TVMRetValue::from(ret))
        }
    }
    let a = std::ffi::CString::new("a").unwrap();
    let b = std::ffi::CString::new("b").unwrap();
    let c = std::ffi::CString::new("c").unwrap();
    let mut registered = function::Builder::default();
    registered.get_function("concate_str");
    assert!(registered.func.is_some());
    let ret: String = registered
        .arg(&a)
        .arg(&b)
        .arg(&c)
        .invoke()
        .unwrap()
        .try_into()
        .unwrap();
    assert_eq!(ret, "abc".to_owned());
}
