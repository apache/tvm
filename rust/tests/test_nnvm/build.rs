extern crate ar;

use std::{env, path::PathBuf, process::Command};

use ar::Builder;
use std::fs::File;

fn main() {
  let out_dir = env::var("OUT_DIR").unwrap();

  let output = Command::new(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/src/build_test_graph.py"
  )).arg(&out_dir)
    .output()
    .expect("Failed to execute command");
  if output.stderr.len() > 0 {
    panic!(String::from_utf8(output.stderr).unwrap());
  }

  let in_path: PathBuf = [&out_dir, "graph.o"].iter().collect();
  let out_path: PathBuf = [&out_dir, "libgraph.a"].iter().collect();
  let mut builder = Builder::new(File::create(out_path.to_str().unwrap()).unwrap());
  builder.append_path(in_path.to_str().unwrap()).unwrap();

  println!("cargo:rustc-link-lib=static=graph");
  println!("cargo:rustc-link-search=native={}", out_dir);
}
