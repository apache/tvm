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

use rust_ndarray::ArrayD;
use std::{
    convert::{TryFrom, TryInto},
    str::FromStr,
};

use tvm::{
    errors::Error,
    function::register_untyped,
    runtime::{ArgValue, RetValue},
    *,
};

fn main() {
    fn sum(args: Vec<ArgValue<'static>>) -> Result<RetValue, Error> {
        let mut ret = 0f32;
        let shape = &mut [2];
        for arg in args.iter() {
            let e = NDArray::empty(shape, Context::cpu(0), DataType::float(32, 1));
            let arg: NDArray = arg.try_into()?;
            let arr = arg.copy_to_ndarray(e)?;
            let rnd: ArrayD<f32> = ArrayD::try_from(&arr)?;
            ret += rnd.scalar_sum();
        }
        Ok(RetValue::from(ret))
    }

    let shape = &mut [2];
    let mut data = vec![3f32, 4.0];
    let mut arr = NDArray::empty(shape, Context::cpu(0), DataType::float(32, 1));
    arr.copy_from_buffer(data.as_mut_slice());

    register_untyped(sum, "sum", true).unwrap();
    let func = Function::get("sum").expect("function registered");

    let ret: f32 = func
        .invoke(vec![(&arr).into(), (&arr).into()])
        .unwrap()
        .try_into()
        .expect("call should succeed");

    assert_eq!(ret, 7f32);
}
