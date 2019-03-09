fn main() {
    let out_dir = std::env::var("OUT_DIR").unwrap();

    let output = std::process::Command::new(concat!(env!("CARGO_MANIFEST_DIR"), "/src/tvm_add.py"))
        .args(&[
            if cfg!(feature = "cpu") {
                "llvm"
            } else {
                "cuda"
            },
            &std::env::var("OUT_DIR").unwrap(),
        ])
        .output()
        .expect("Failed to execute command");
    assert!(
        std::path::Path::new(&format!("{}/test_add.so", out_dir)).exists(),
        "Could not build tvm lib: {}",
        String::from_utf8(output.stderr)
            .unwrap()
            .trim()
            .split("\n")
            .last()
            .unwrap_or("")
    );

    println!("cargo:rustc-link-search=native={}", out_dir);
}
