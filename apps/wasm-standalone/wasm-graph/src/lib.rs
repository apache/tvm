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

#[macro_use]
extern crate lazy_static;
#[macro_use]
extern crate serde_derive;

mod types;
mod utils;

use std::{collections::HashMap, convert::TryFrom, env, sync::Mutex};

use tvm_graph_rt::{Graph, GraphExecutor, SystemLibModule, Tensor as TVMTensor};

use types::Tensor;

extern "C" {
    fn __wasm_call_ctors();
}

lazy_static! {
    static ref SYSLIB: SystemLibModule = SystemLibModule::default();
    static ref GRAPH_EXECUTOR: Mutex<GraphExecutor<'static, 'static>> = {
        unsafe {
            // This is necessary to invoke TVMBackendRegisterSystemLibSymbol
            // API calls.
            __wasm_call_ctors();
        }
        let graph = Graph::try_from(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/lib/graph.json"
        )))
        .unwrap();

        let params_bytes =
            include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/lib/graph.params"));
        let params = tvm_graph_rt::load_param_dict(params_bytes)
            .unwrap()
            .into_iter()
            .map(|(k, v)| (k, v.to_owned()))
            .collect::<HashMap<String, TVMTensor<'static>>>();

        let mut exec = GraphExecutor::new(graph, &*SYSLIB).unwrap();

        exec.load_params(params);

        Mutex::new(exec)
    };
}

#[no_mangle]
pub extern "C" fn run(wasm_addr: i32, in_size: i32) -> i32 {
    let in_tensor = unsafe { utils::load_input(wasm_addr, in_size as usize) };
    let input: TVMTensor = in_tensor.as_dltensor().into();

    // since this executor is not multi-threaded, we can acquire lock once
    let mut executor = GRAPH_EXECUTOR.lock().unwrap();

    executor.set_input("data", input);

    executor.run();

    let output = executor.get_output(0).unwrap().as_dltensor(false);

    let out_tensor: Tensor = output.into();
    let out_size = unsafe { utils::store_output(wasm_addr, out_tensor) };
    out_size as i32
}
