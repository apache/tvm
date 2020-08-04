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

use std::{path::PathBuf, process::Command};

use anyhow::{Context, Result};

fn main() -> Result<()> {
    let mut out_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    out_dir.push("lib");

    if !out_dir.is_dir() {
        std::fs::create_dir(&out_dir).context("failed to create directory for WASM outputs")?;
    }

    let obj_file = out_dir.join("test.o");
    let lib_file = out_dir.join("libtest_wasm32.a");

    let output = Command::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/src/build_test_lib.py"
    ))
    .arg(&out_dir)
    .output()
    .context("failed to execute Python script for generating TVM library")?;

    assert!(
        obj_file.exists(),
        "Could not build tvm lib: {}",
        String::from_utf8(output.stderr)
            .unwrap()
            .trim()
            .split("\n")
            .last()
            .unwrap_or("")
    );

    let ar = option_env!("LLVM_AR").unwrap_or("llvm-ar-8");

    let output = Command::new(ar)
        .arg("rcs")
        .arg(&lib_file)
        .arg(&obj_file)
        .output()
        .context("failed to run LLVM_AR command")?;

    assert!(
        lib_file.exists(),
        "Could not create archive: {}",
        String::from_utf8(output.stderr)
            .unwrap()
            .trim()
            .split("\n")
            .last()
            .unwrap_or("")
    );

    println!("cargo:rustc-link-lib=static=test_wasm32");
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    Ok(())
}
