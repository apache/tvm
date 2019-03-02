use std::process::Command;

fn main() {
    let output = Command::new(concat!(env!("CARGO_MANIFEST_DIR"), "/src/build_resnet.py"))
        .output()
        .expect("Failed to execute command");
    assert!(
        std::path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/deploy_lib.o")).exists(),
        "Could not prepare demo: {}",
        String::from_utf8(output.stderr).unwrap().trim()
    );
    println!(
        "cargo:rustc-link-search=native={}",
        env!("CARGO_MANIFEST_DIR")
    );
}
