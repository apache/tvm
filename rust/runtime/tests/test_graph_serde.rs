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

extern crate serde;
extern crate serde_json;

extern crate tvm_runtime;

use std::{convert::TryFrom, fs, io::Read};

use tvm_runtime::Graph;

#[test]
fn test_load_graph() {
    let mut params_bytes = Vec::new();
    fs::File::open(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/graph.params"))
        .expect("Could not find TVM graph. Did you run `tests/build_model.py`?")
        .read_to_end(&mut params_bytes)
        .unwrap();
    let _params = tvm_runtime::load_param_dict(&params_bytes);

    let graph = Graph::try_from(
        &fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/graph.json")).unwrap(),
    )
    .unwrap();

    assert_eq!(graph.nodes[3].op, "tvm_op");
    assert_eq!(
        graph.nodes[3]
            .attrs
            .as_ref()
            .unwrap()
            .get("func_name")
            .unwrap(),
        "fuse_dense"
    );
    assert_eq!(graph.nodes[5].inputs[0].index, 0);
    assert_eq!(graph.nodes[6].inputs[0].index, 1);
    assert_eq!(graph.heads.len(), 2);
}
