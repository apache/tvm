fn main() {
    let out_dir = std::env::var("OUT_DIR").unwrap();

    let output =
        std::process::Command::new(concat!(env!("CARGO_MANIFEST_DIR"), "/src/build_model.py"))
            .arg(&out_dir)
            .output()
            .expect("Failed to execute command");
    assert!(
        std::path::Path::new(&format!("{}/model.o", out_dir)).exists(),
        "Could not build tvm lib: {}",
        String::from_utf8(output.stderr).unwrap().trim()
    );

    std::process::Command::new("llvm-ar-8")
        .arg("rcs")
        .arg(&format!("{}/libmodel.a", out_dir))
        .arg(&format!("{}/model.o", out_dir))
        .output()
        .expect("Failed to execute command");

    println!("cargo:rustc-link-lib=static=model");
    println!("cargo:rustc-link-search=native={}", out_dir);
}
