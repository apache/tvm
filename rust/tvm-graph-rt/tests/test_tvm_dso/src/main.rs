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

use ndarray::Array;
use tvm_graph_rt::{DLTensor, DsoModule, Module};

fn main() {
    tvm_graph_rt::TVMGetLastError();
    let module = DsoModule::new(concat!(env!("OUT_DIR"), "/test.so")).unwrap();
    let add = module
        .get_function("__tvm_main__")
        .expect("main function not found");
    let mut a = Array::from_vec(vec![1f32, 2., 3., 4.]);
    let mut b = Array::from_vec(vec![1f32, 0., 1., 0.]);
    let mut c = Array::from_vec(vec![0f32; 4]);
    let e = Array::from_vec(vec![2f32, 2., 4., 4.]);
    let mut a_dl: DLTensor = (&mut a).into();
    let mut b_dl: DLTensor = (&mut b).into();
    let mut c_dl: DLTensor = (&mut c).into();
    let args = vec![(&mut a_dl).into(), (&mut b_dl).into(), (&mut c_dl).into()];
    add(&args[..]).unwrap();
    assert!(c.all_close(&e, 1e-8f32));
}
