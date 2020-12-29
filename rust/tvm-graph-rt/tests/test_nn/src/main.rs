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

use std::{collections::HashMap, convert::TryFrom, fs, io::Read};

use ndarray::{s, Array};
use tvm_graph_rt::{Graph, GraphExecutor, SystemLibModule, Tensor};

const BATCH_SIZE: usize = 4;
const IN_DIM: usize = 8;

macro_rules! check_sum {
    ($e:expr, $a:ident, $b:ident) => {
        let a = Array::try_from($e.get_input(stringify!($a)).unwrap().to_owned()).unwrap();
        check_sum!(a, $b);
    };
    ($e:expr, $a:expr, $b:ident) => {
        let a = Array::try_from($e.get_output($a).unwrap().to_owned()).unwrap();
        check_sum!(a, $b);
    };
    ($a:ident, $b:ident) => {
        let a_sum: f32 = $a.scalar_sum();
        let b_sum: f32 = $b.scalar_sum();
        assert!((a_sum - b_sum).abs() < 1e-2, "{} != {}", a_sum, b_sum);
    };
}

fn main() {
    let syslib = SystemLibModule::default();

    let mut params_bytes = Vec::new();
    fs::File::open(concat!(env!("OUT_DIR"), "/test_nn/graph.params"))
        .unwrap()
        .read_to_end(&mut params_bytes)
        .unwrap();
    let params = tvm_graph_rt::load_param_dict(&params_bytes)
        .unwrap()
        .into_iter()
        .map(|(k, v)| (k, v.to_owned()))
        .collect::<HashMap<String, Tensor<'static>>>();

    let graph = Graph::try_from(
        &fs::read_to_string(concat!(env!("OUT_DIR"), "/test_nn/graph.json")).unwrap(),
    )
    .unwrap();
    let mut exec = GraphExecutor::new(graph, &syslib).unwrap();

    let x = Array::from_shape_vec(
        (BATCH_SIZE, IN_DIM),
        (0..BATCH_SIZE * IN_DIM)
            .map(|x| x as f32)
            .collect::<Vec<f32>>(),
    )
    .unwrap();

    let p0 = params.get("p0").unwrap().to_owned();
    let p1 = params.get("p1").unwrap().to_owned();
    println!("p0: {:?}", p0.shape());
    println!("p1: {:?}", p1.shape());
    let w = Array::try_from(p0)
        .unwrap()
        .into_shape((BATCH_SIZE * 4, IN_DIM))
        .unwrap();
    let b = Array::try_from(p1).unwrap();
    let dense = x.dot(&w.t()) + &b;
    let left = dense.slice(s![.., 0..IN_DIM]);
    let right = dense.slice(s![.., IN_DIM..]);
    let expected_o0 = &left + 1f32;
    let expected_o1 = &right - 1f32;

    exec.load_params(params);
    exec.set_input("data", (&x).into());

    check_sum!(exec, data, x);
    check_sum!(exec, p0, w);
    check_sum!(exec, p1, b);

    exec.run();

    check_sum!(exec, 0, expected_o0);
    check_sum!(exec, 1, expected_o1);
    check_sum!(exec, 2, dense);
}
