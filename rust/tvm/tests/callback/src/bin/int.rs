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

extern crate tvm_frontend as tvm;

use std::convert::TryInto;
use tvm::{errors::Error, *};

fn main() {
    fn sum(args: &[TVMArgValue]) -> Result<TVMRetValue, Error> {
        let mut ret = 0i64;
        for arg in args.iter() {
            let val: i64 = arg.try_into()?;
            ret += val;
        }
        Ok(TVMRetValue::from(ret))
    }

    tvm::function::register(sum, "mysum".to_owned(), false).unwrap();

    let mut registered = function::Builder::default();
    registered.get_function("mysum");
    assert!(registered.func.is_some());
    let ret: i64 = registered
        .args(&[10, 20, 30])
        .invoke()
        .unwrap()
        .try_into()
        .unwrap();
    assert_eq!(ret, 60);
}
