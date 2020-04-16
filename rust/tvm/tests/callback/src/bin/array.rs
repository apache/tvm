/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#![allow(unused_imports)]

extern crate ndarray as rust_ndarray;
#[macro_use]
extern crate tvm_frontend as tvm;

use rust_ndarray::ArrayD;
use std::{
    convert::{TryFrom, TryInto},
    str::FromStr,
};

use tvm::{errors::Error, *};

fn main() {
    register_global_func! {
        fn sum(args: &[TVMArgValue]) -> Result<TVMRetValue, Error> {
            let mut ret = 0f32;
            let shape = &mut [2];
            for arg in args.iter() {
                let e = NDArray::empty(
                    shape, TVMContext::cpu(0),
                    DLDataType::from_str("float32").unwrap()
                );
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
    let mut arr = NDArray::empty(
        shape,
        TVMContext::cpu(0),
        DLDataType::from_str("float32").unwrap(),
    );
    arr.copy_from_buffer(data.as_mut_slice());

    let mut registered = function::Builder::default();
    let ret: f32 = registered
        .get_function("sum")
        .arg(&arr)
        .arg(&arr)
        .invoke()
        .unwrap()
        .try_into()
        .unwrap();
    assert_eq!(ret, 7f32);
}
