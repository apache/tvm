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

extern crate tvm_runtime;

use std::{
    convert::TryFrom as _,
    io::{Read as _, Write as _},
};

fn main() {
    let syslib = tvm_runtime::SystemLibModule::default();

    let graph_json = include_str!(concat!(env!("OUT_DIR"), "/graph.json"));
    let params_bytes = include_bytes!(concat!(env!("OUT_DIR"), "/params.bin"));
    let params = tvm_runtime::load_param_dict(params_bytes).unwrap();

    let graph = tvm_runtime::Graph::try_from(graph_json).unwrap();
    let mut exec = tvm_runtime::GraphExecutor::new(graph, &syslib).unwrap();
    exec.load_params(params);

    let listener = std::net::TcpListener::bind("127.0.0.1:4242").unwrap();
    for stream in listener.incoming() {
        let mut stream = stream.unwrap();
        if let Err(_) =
            stream.read_exact(exec.get_input("data").unwrap().data().view().as_mut_slice())
        {
            continue;
        }
        exec.run();
        if let Err(_) = stream.write_all(exec.get_output(0).unwrap().data().as_slice()) {
            continue;
        }
    }
}
