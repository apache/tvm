extern crate bindgen;

use std::{env, path::PathBuf};

fn parse_clang_ver(raw_v: String) -> Vec<u32> {
  raw_v
    .split_whitespace()
    .nth(2)
    .unwrap()
    .split('.')
    .map(|v| v.parse::<u32>().unwrap())
    .collect()
}

fn main() {
  let clang_ver = parse_clang_ver(bindgen::clang_version().full);
  let bindings = bindgen::Builder::default()
    .header(concat!(
      env!("CARGO_MANIFEST_DIR"),
      "/../include/tvm/runtime/c_runtime_api.h"
    ))
    .header(concat!(
      env!("CARGO_MANIFEST_DIR"),
      "/../include/tvm/runtime/c_backend_api.h"
    ))
    .rust_target(bindgen::RustTarget::Nightly)
    .clang_arg(concat!(
      "-I",
      env!("CARGO_MANIFEST_DIR"),
      "/../dlpack/include"
    ))
    .clang_arg(format!("--target={}", env::var("HOST").unwrap()))
    .clang_arg("-I/usr/include")
    .clang_arg("-I/usr/local/include")
    .clang_arg(format!(
      "-I/usr/local/lib/clang/{}.{}.{}/include",
      clang_ver[0], clang_ver[1], clang_ver[2]
    ))
    .layout_tests(false)
    .generate()
    .expect("Unable to generate bindings.");

  let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
  bindings
    .write_to_file(out_path.join("c_runtime_api.rs"))
    .expect("Unable to write bindings.");
}
