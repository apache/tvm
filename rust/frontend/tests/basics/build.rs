use std::process::Command;

fn main() {
    #[cfg(feature = "cpu")]
    let script_path = concat!(env!("CARGO_MANIFEST_DIR"), "/src/tvm_add_cpu.py");
    #[cfg(feature = "gpu")]
    let script_path = concat!(env!("CARGO_MANIFEST_DIR"), "/src/tvm_add_gpu.py");
    let output = Command::new("python")
        .args(&[&script_path, env!("CARGO_MANIFEST_DIR")])
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
