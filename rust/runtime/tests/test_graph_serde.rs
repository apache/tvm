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

macro_rules! mf_dir {
    ($p:literal) => {
        concat!(env!("CARGO_MANIFEST_DIR"), $p)
    };
}

static PARAMS_FIXTURE_PATH: &str = mf_dir!("/tests/graph.params");

#[test]
fn test_load_graph() {
    let output = std::process::Command::new(mf_dir!("/tests/build_model.py"))
        .env(
            "PYTHONPATH",
            concat!(
                mf_dir!("/../../python"),
                ":",
                mf_dir!("/../../nnvm/python"),
                ":",
                mf_dir!("/../../topi/python")
            ),
        )
        .output()
        .expect("Failed to build test model");
    assert!(
        std::path::Path::new(PARAMS_FIXTURE_PATH).exists(),
        "Could not build test graph fixture: STDOUT:\n\n{}\nSTDERR: {}\n\n",
        String::from_utf8(output.stdout).unwrap(),
        String::from_utf8(output.stderr).unwrap()
    );
    let mut params_bytes = Vec::new();
    fs::File::open(PARAMS_FIXTURE_PATH)
        .unwrap()
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
        "fused_nn_dense_nn_bias_add"
    );
    assert_eq!(graph.nodes[3].inputs[0].index, 0);
    assert_eq!(graph.nodes[4].inputs[0].index, 0);
    assert_eq!(graph.heads.len(), 3);
}
