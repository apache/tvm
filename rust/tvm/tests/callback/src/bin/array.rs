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
    fn sum<'a>(args: Vec<ArgValue<'a>>) -> Result<RetValue, Error> {
        let mut ret = 0.0;
        for arg in args {
            let arg: NDArray = arg.try_into()?;
            let rnd: ArrayD<f32> = ArrayD::try_from(&arg)?;
            ret += rnd.scalar_sum();
        }
        Ok(RetValue::from(ret))
    }

    let shape = &[2];
    let data = vec![3.0, 4.0];
    let mut arr = NDArray::empty(shape, Device::cpu(0), DataType::float(32, 1));
    arr.copy_from_buffer(data.as_slice());

    register_untyped(sum, "sum", true).unwrap();
    let func = Function::get("sum").expect("function registered");

    let ret: f32 = func
        .invoke(vec![(&arr).into()])
        .unwrap()
        .try_into()
        .expect("call should succeed");

    assert_eq!(ret, 7.0);
}
