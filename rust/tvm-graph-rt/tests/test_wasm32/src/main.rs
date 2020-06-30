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

extern "C" {
    static __tvm_module_ctx: i32;
}

#[no_mangle]
unsafe fn __get_tvm_module_ctx() -> i32 {
    // Refer a symbol in the libtest_wasm32.a to make sure that the link of the
    // library is not optimized out.
    __tvm_module_ctx
}

extern crate ndarray;
#[macro_use]
extern crate tvm_runtime;

use ndarray::Array;
use tvm_runtime::{DLTensor, Module as _, SystemLibModule};

fn main() {
    // try static
    let mut a = Array::from_vec(vec![1f32, 2., 3., 4.]);
    let mut b = Array::from_vec(vec![1f32, 0., 1., 0.]);
    let mut c = Array::from_vec(vec![0f32; 4]);
    let e = Array::from_vec(vec![2f32, 2., 4., 4.]);
    let mut a_dl: DLTensor = (&mut a).into();
    let mut b_dl: DLTensor = (&mut b).into();
    let mut c_dl: DLTensor = (&mut c).into();

    let syslib = SystemLibModule::default();
    let add = syslib
        .get_function("default_function")
        .expect("main function not found");
    call_packed!(add, &mut a_dl, &mut b_dl, &mut c_dl).unwrap();
    assert!(c.all_close(&e, 1e-8f32));
}
