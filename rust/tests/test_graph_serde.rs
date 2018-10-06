#![feature(try_from)]

extern crate serde;
extern crate serde_json;

extern crate tvm;

use std::{convert::TryFrom, fs, io::Read};

use tvm::runtime::Graph;

#[test]
fn test_load_graph() {
  let mut params_bytes = Vec::new();
  fs::File::open(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/graph.params"))
    .expect("Could not find TVM graph. Did you run `tests/build_model.py`?")
    .read_to_end(&mut params_bytes)
    .unwrap();
  let _params = tvm::runtime::load_param_dict(&params_bytes);

  let graph = Graph::try_from(
    &fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/graph.json")).unwrap(),
  ).unwrap();

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
