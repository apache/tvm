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

use tvm_frontend::{errors::Error, *};

fn main() {
    register_global_func! {
        fn error(_args: &[TVMArgValue]) -> Result<TVMRetValue, Error> {
            Err(errors::TypeMismatchError{
                expected: "i64".to_string(),
                actual: "f64".to_string(),
            }.into())
        }
    }

    let mut registered = function::Builder::default();
    registered.get_function("error");
    assert!(registered.func.is_some());
    registered.args(&[10, 20]);

    println!("expected error message is:");
    panic::set_hook(Box::new(|panic_info| {
        // if let Some(msg) = panic_info.message() {
        //     println!("{:?}", msg);
        // }
        if let Some(location) = panic_info.location() {
            println!(
                "panic occurred in file '{}' at line {}",
                location.file(),
                location.line()
            );
        } else {
            println!("panic occurred but can't get location information");
        }
    }));

    let _result = registered.invoke();
}
