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

use std::convert::TryInto;
use tvm::{
    errors::Error,
    runtime::{ArgValue, RetValue},
    *,
};

// FIXME
fn main() {
    fn concat_str<'a>(args: Vec<ArgValue<'a>>) -> Result<RetValue, Error> {
        let mut ret = "".to_string();
        for arg in args.iter() {
            let val: &str = arg.try_into()?;
            ret += val;
        }
        Ok(RetValue::from(ret))
    }

    let a = std::ffi::CString::new("a").unwrap();
    let b = std::ffi::CString::new("b").unwrap();
    let c = std::ffi::CString::new("c").unwrap();

    tvm::function::register_untyped(concat_str, "concat_str".to_owned(), false).unwrap();

    let func = Function::get("concat_str").expect("just registered a function");

    let args = vec![
        a.as_c_str().into(),
        b.as_c_str().into(),
        c.as_c_str().into(),
    ];

    let ret: String = func
        .invoke(args)
        .expect("function call should succeed")
        .try_into()
        .unwrap();

    assert_eq!(ret, "abc".to_owned());
}
