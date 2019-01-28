fn main() {
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
    if output.stderr.len() > 0 {
        panic!(String::from_utf8(output.stderr).unwrap());
    }
    println!(
        "cargo:rustc-link-search=native={}",
        env!("CARGO_MANIFEST_DIR")
    );
}
