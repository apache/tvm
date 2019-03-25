fn main() {
    let out_dir = std::env::var("OUT_DIR").unwrap();

    let output =
        std::process::Command::new(concat!(env!("CARGO_MANIFEST_DIR"), "/src/build_model.py"))
            .arg(&out_dir)
            .output()
            .expect("Failed to execute command");
    assert!(
        ["model.o", "graph.json", "params.bin"].iter().all(|f| {
            std::path::Path::new(&format!("{}/{}", out_dir, f)).exists()
        }),
        "Could not build tvm lib: {}",
        String::from_utf8(output.stderr).unwrap().trim()
    );

    std::process::Command::new("objcopy")
        .arg("--globalize-symbol=__tvm_module_startup")
        .arg(&format!("{}/model.o", out_dir))
        .output()
        .expect("Could not gloablize startup function.");

    std::process::Command::new("llvm-ar")
        .arg("rcs")
        .arg(&format!("{}/libmodel.a", out_dir))
        .arg(&format!("{}/model.o", out_dir))
        .output()
        .expect("Failed to package model archive.");

    println!("cargo:rustc-link-lib=static=model");
    println!("cargo:rustc-link-search=native={}", out_dir);
}
