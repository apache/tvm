#![feature(extern_crate_item_prelude, try_from)]
#![allow(unused_imports)]

extern crate ndarray as rust_ndarray;
#[macro_use]
extern crate tvm_frontend as tvm;

use rust_ndarray::ArrayD;
use std::convert::{TryFrom, TryInto};

use tvm::*;

fn main() {
    register_global_func! {
        fn sum(args: &[TVMArgValue]) -> Result<TVMRetValue> {
            let mut ret = 0f32;
            let shape = &mut [2];
            for arg in args.iter() {
                let e = NDArray::empty(shape, TVMContext::cpu(0), TVMType::from("float32"));
                let arg: NDArray = arg.try_into()?;
                let arr = arg.copy_to_ndarray(e)?;
                let rnd: ArrayD<f32> = ArrayD::try_from(&arr)?;
                ret += rnd.scalar_sum();
            }
            Ok(TVMRetValue::from(ret))
        }
    }

    let shape = &mut [2];
    let mut data = vec![3f32, 4.0];
    let mut arr = NDArray::empty(shape, TVMContext::cpu(0), TVMType::from("float32"));
    arr.copy_from_buffer(data.as_mut_slice());

    let mut registered = function::Builder::default();
    let ret: f32 = registered
        .get_function("sum", true)
        .arg(&arr)
        .arg(&arr)
        .invoke()
        .unwrap()
        .try_into()
        .unwrap();
    assert_eq!(ret, 14f32);
}
