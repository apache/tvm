extern crate ar;

use std::{env, path::Path, process::Command};

use ar::Builder;
use std::fs::File;

fn main() {
    let out_dir = env::var("OUT_DIR").unwrap();

    let output = Command::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/src/build_test_lib.py"
    ))
    .arg(&out_dir)
    .output()
    .expect("Failed to execute command");
    assert!(
        Path::new(&format!("{}/test.o", out_dir)).exists(),
        "Could not build tvm lib: {}",
        String::from_utf8(output.stderr)
            .unwrap()
            .trim()
            .split("\n")
            .last()
            .unwrap_or("")
    );

    let mut builder = Builder::new(File::create(format!("{}/libtest.a", out_dir)).unwrap());
    builder.append_path(format!("{}/test.o", out_dir)).unwrap();

    println!("cargo:rustc-link-lib=static=test");
    println!("cargo:rustc-link-search=native={}", out_dir);
}
