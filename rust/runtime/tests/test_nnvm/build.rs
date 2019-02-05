extern crate ar;

use std::{env, fs::File, path::Path, process::Command};

use ar::Builder;

fn main() {
    let out_dir = env::var("OUT_DIR").unwrap();

    let output = Command::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/src/build_test_graph.py"
    ))
    .arg(&out_dir)
    .output()
    .expect("Failed to execute command");
    assert!(
        Path::new(&format!("{}/graph.o", out_dir)).exists(),
        "Could not build graph lib: {}",
        String::from_utf8(output.stderr)
            .unwrap()
            .trim()
            .split("\n")
            .last()
            .unwrap_or("")
    );

    let mut builder = Builder::new(File::create(format!("{}/libgraph.a", out_dir)).unwrap());
    builder.append_path(format!("{}/graph.o", out_dir)).unwrap();

    println!("cargo:rustc-link-lib=static=graph");
    println!("cargo:rustc-link-search=native={}", out_dir);
}
