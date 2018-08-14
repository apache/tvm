#![feature(fs_read_write, try_from)]

#[macro_use]
extern crate ndarray;
extern crate serde;
extern crate serde_json;

extern crate tvm;
use std::{collections::HashMap, convert::TryFrom, fs, io::Read};

use ndarray::Array;
use tvm::runtime::{Graph, GraphExecutor, SystemLibModule, Tensor};

const BATCH_SIZE: usize = 4;
const IN_DIM: usize = 8;

macro_rules! assert_sum_eq {
  ($a:expr, $b:expr) => {
    let a_sum = $a.scalar_sum();
    let b_sum = $b.scalar_sum();
    assert!((a_sum - b_sum).abs() < 1e-2, "{} != {}", a_sum, b_sum);
  };
}

fn main() {
  let syslib = SystemLibModule::default();

  let mut params_bytes = Vec::new();
  fs::File::open(concat!(env!("OUT_DIR"), "/graph.params"))
    .unwrap()
    .read_to_end(&mut params_bytes)
    .unwrap();
  let params = tvm::runtime::load_param_dict(&params_bytes)
    .unwrap()
    .into_iter()
    .map(|(k, v)| (k, v.to_owned()))
    .collect::<HashMap<String, Tensor<'static>>>();

  let graph =
    Graph::try_from(&fs::read_to_string(concat!(env!("OUT_DIR"), "/graph.json")).unwrap()).unwrap();
  let mut exec = GraphExecutor::new(graph, &syslib).unwrap();

  let x = Array::from_shape_vec(
    (BATCH_SIZE, IN_DIM),
    (0..BATCH_SIZE * IN_DIM)
      .map(|x| x as f32)
      .collect::<Vec<f32>>(),
  ).unwrap();
  let w = Array::try_from(params.get("dense0_weight").unwrap())
    .unwrap()
    .into_shape((IN_DIM * 2, IN_DIM))
    .unwrap();
  let b = Array::try_from(params.get("dense0_bias").unwrap()).unwrap();
  let dense = x.dot(&w.t()) + &b;
  let left = dense.slice(s![.., 0..IN_DIM]);
  let right = dense.slice(s![.., IN_DIM..]);
  let expected_o0 = &left + 1f32;
  let expected_o1 = &right - 1f32;

  exec.load_params(params);
  exec.set_input("data", x.clone().into());

  assert_sum_eq!(Array::try_from(exec.get_input("data").unwrap()).unwrap(), x);
  assert_sum_eq!(
    Array::try_from(exec.get_input("dense0_weight").unwrap()).unwrap(),
    w
  );
  assert_sum_eq!(
    Array::try_from(exec.get_input("dense0_bias").unwrap()).unwrap(),
    b
  );

  exec.run();

  assert_sum_eq!(
    Array::try_from(exec.get_output(0).unwrap()).unwrap(),
    expected_o0
  );
  assert_sum_eq!(
    Array::try_from(exec.get_output(1).unwrap()).unwrap(),
    expected_o1
  );
  assert_sum_eq!(Array::try_from(exec.get_output(2).unwrap()).unwrap(), dense);
}
