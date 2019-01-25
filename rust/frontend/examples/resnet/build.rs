use std::process::Command;

fn main() {
    let script_path = concat!(env!("CARGO_MANIFEST_DIR"), "/src/build_resnet.py");
    let output = Command::new("python")
        .arg(script_path)
        .output()
        .expect("Failed to execute command");

    println!("stdout: {}", String::from_utf8_lossy(&output.stdout));
    eprintln!("stderr: {}", String::from_utf8_lossy(&output.stderr));
    println!(
        "cargo:rustc-link-search=native={}",
        env!("CARGO_MANIFEST_DIR")
    );
}
