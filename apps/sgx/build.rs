/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

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
    llvm_tools_path.push("lib/rustlib/x86_64-unknown-linux-gnu/bin");

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
