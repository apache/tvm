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

fn main() {
    fn sum<'a>(args: Vec<ArgValue<'a>>) -> Result<RetValue, Error> {
        let mut ret = 0i64;
        for arg in args.iter() {
            let val: i64 = arg.try_into()?;
            ret += val;
        }
        Ok(RetValue::from(ret))
    }

    tvm::function::register_untyped(sum, "mysum".to_owned(), false).unwrap();
    let func = Function::get("mysum").unwrap();
    let ret: i64 = func
        .invoke(vec![10.into(), 20.into(), 30.into()])
        .unwrap()
        .try_into()
        .unwrap();
    assert_eq!(ret, 60);
}
