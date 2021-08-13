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

use std::str::FromStr;

use tvm::*;

fn main() {
    let shape = &mut [2];
    let mut data = vec![3f32, 4.0];

    let (dev, dev_name) = if cfg!(feature = "cpu") {
        (Device::cpu(0), "cpu")
    } else {
        (Device::cuda(0), "cuda")
    };

    let dtype = DataType::from_str("float32").unwrap();
    let mut arr = NDArray::empty(shape, dev, dtype);
    arr.copy_from_buffer(data.as_mut_slice());
    let ret = NDArray::empty(shape, dev, dtype);
    let fadd = Module::load(&concat!(env!("OUT_DIR"), "/test_add.so")).unwrap();
    if !fadd.enabled(dev_name) {
        return;
    }

    if cfg!(feature = "cuda") {
        fadd.import_module(Module::load(&concat!(env!("OUT_DIR"), "/test_add.ptx")).unwrap());
    }

    // todo(@jroesch): fix the entry_name
    fadd.get_function("__tvm_main__", false)
        .expect("module must have entry point")
        .invoke(vec![(&arr).into(), (&arr).into(), (&ret).into()])
        .unwrap();

    assert_eq!(ret.to_vec::<f32>().unwrap(), vec![6f32, 8.0]);
}
