fn main() {
    let out_dir = std::env::var("OUT_DIR").unwrap();

    let build_output =
        std::process::Command::new(concat!(env!("CARGO_MANIFEST_DIR"), "/src/build_model.py"))
            .arg(&out_dir)
            .output()
            .expect("Failed to build model");
    assert!(
        ["model.o", "graph.json", "params.bin"]
            .iter()
            .all(|f| { std::path::Path::new(&format!("{}/{}", out_dir, f)).exists() }),
        "Could not build tvm lib: {}",
        String::from_utf8(build_output.stderr).unwrap().trim()
    );

    let sysroot_output = std::process::Command::new("rustc")
        .args(&["--print", "sysroot"])
        .output()
        .expect("Failed to get sysroot");
    let sysroot = String::from_utf8(sysroot_output.stdout).unwrap();
    let sysroot = sysroot.trim();
    let mut llvm_tools_path = std::path::PathBuf::from(&sysroot);
    let target = sysroot.splitn(2, "-").nth(1).expect(&sysroot);
    llvm_tools_path.push("lib/rustlib");
    llvm_tools_path.push(target);
    llvm_tools_path.push("bin");

    std::process::Command::new(llvm_tools_path.join("llvm-objcopy"))
        .arg("--globalize-symbol=__tvm_module_startup")
        .arg(&format!("{}/model.o", out_dir))
        .output()
        .expect("Could not gloablize startup function.");

    std::process::Command::new(llvm_tools_path.join("llvm-ar"))
        .arg("rcs")
        .arg(&format!("{}/libmodel.a", out_dir))
        .arg(&format!("{}/model.o", out_dir))
        .output()
        .expect("Failed to package model archive.");

    println!("cargo:rustc-link-lib=static=model");
    println!("cargo:rustc-link-search=native={}", out_dir);
}
