extern crate bindgen;

use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-env-changed=TVM_HOME");
    println!("cargo:rustc-link-lib=dylib=tvm_runtime");
    println!("cargo:rustc-link-search={}/build", env!("TVM_HOME"));
    let bindings = bindgen::Builder::default()
        .header(format!(
            "{}/include/tvm/runtime/c_runtime_api.h",
            env!("TVM_HOME")
        ))
        .clang_arg(format!("-I{}/3rdparty/dlpack/include/", env!("TVM_HOME")))
        .blacklist_type("max_align_t") // @see rust-bindgen#550
        .layout_tests(false)
        .derive_partialeq(true)
        .derive_eq(true)
        .generate()
        .expect("unable to generate bindings");

    bindings
        .write_to_file(PathBuf::from("src/bindgen.rs"))
        .expect("can not write the bindings!");
}
