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

use std::convert::TryInto;
use tvm::{
    errors::Error,
    runtime::{ArgValue, RetValue},
    *,
};

fn main() {
    fn sum<'a>(args: Vec<ArgValue<'a>>) -> Result<RetValue, Error> {
        let mut ret = 0.0;
        for arg in args.into_iter() {
            let val: f64 = arg.try_into()?;
            ret += val;
        }
        Ok(RetValue::from(ret))
    }

    function::register_untyped(sum, "sum", true).expect("registration should succeed");

    let func = Function::get("sum").expect("sum was just registered.");

    let ret: f64 = func
        .invoke(vec![10.0f64.into(), 20.0.into(), 30.0.into()])
        .unwrap()
        .try_into()
        .unwrap();

    assert_eq!(ret, 60f64);
}
