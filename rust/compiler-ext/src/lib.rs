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

use env_logger;
use tvm;
use tvm::runtime::function::register_override;

fn test_fn() -> Result<(), tvm::Error> {
    println!("Hello Greg from Rust!");
    Ok(())
}

fn test_fn2(message: tvm::runtime::string::String) -> Result<(), tvm::Error> {
    println!("The message: {}", message);
    Ok(())
}

tvm::export!(test_fn, test_fn2);

#[no_mangle]
fn compiler_ext_initialize() -> i32 {
    let _ = env_logger::try_init();
    tvm_export("rust_ext").expect("failed to initialize Rust compiler_ext");
    log::debug!("done!");
    return 0;
}
