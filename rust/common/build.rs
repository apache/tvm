extern crate bindgen;

use std::path::PathBuf;

fn main() {
    if cfg!(feature = "bindings") {
        println!("cargo:rerun-if-env-changed=TVM_HOME");
        println!("cargo:rustc-link-lib=dylib=tvm_runtime");
        println!("cargo:rustc-link-search={}/build", env!("TVM_HOME"));
    }

    // @see rust-bindgen#550 for `blacklist_type`
    bindgen::Builder::default()
        .header(format!(
            "{}/include/tvm/runtime/c_runtime_api.h",
            env!("TVM_HOME")
        ))
        .header(format!(
            "{}/include/tvm/runtime/c_backend_api.h",
            env!("TVM_HOME")
        ))
        .clang_arg(format!("-I{}/3rdparty/dlpack/include/", env!("TVM_HOME")))
        .blacklist_type("max_align_t")
        .layout_tests(false)
        .derive_partialeq(true)
        .derive_eq(true)
        .generate()
        .expect("unable to generate bindings")
        .write_to_file(PathBuf::from("src/c_runtime_api.rs"))
        .expect("can not write the bindings!");
}
