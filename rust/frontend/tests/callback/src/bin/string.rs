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
        .arg(a.as_c_str())
        .arg(b.as_c_str())
        .arg(c.as_c_str())
        .invoke()
        .unwrap()
        .try_into()
        .unwrap();
    assert_eq!(ret, "abc".to_owned());
}
