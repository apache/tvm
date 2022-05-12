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

use std::panic;

use tvm::{
    errors::Error,
    runtime::{ArgValue, RetValue},
    *,
};

fn main() {
    fn error<'a>(_args: Vec<ArgValue<'a>>) -> Result<RetValue, Error> {
        Err(errors::NDArrayError::DataTypeMismatch {
            expected: DataType::int(64, 1),
            actual: DataType::float(64, 1),
        }
        .into())
    }

    function::register_untyped(error, "error", true).unwrap();

    let func = Function::get("error");
    assert!(func.is_some());
    match func.unwrap().invoke(vec![10.into(), 20.into()]) {
        Err(_) => {}
        Ok(_) => panic!("expected error"),
    }
}
