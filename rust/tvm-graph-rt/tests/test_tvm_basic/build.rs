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

extern crate ar;

use std::{path::PathBuf, process::Command};

use std::fs::File;

use anyhow::Result;
use ar::Builder;

fn main() -> Result<()> {
    let mut out_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    out_dir.push("lib");

    if !out_dir.is_dir() {
        std::fs::create_dir(&out_dir)?;
    }

    let obj_file = out_dir.join("test.o");
    let lib_file = out_dir.join("libtest_basic.a");

    let output = Command::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/src/build_test_lib.py"
    ))
    .arg(&out_dir)
    .output()?;

    assert!(
        obj_file.exists(),
        "Could not build tvm lib: {}",
        String::from_utf8(output.stderr)?
            .trim()
            .split("\n")
            .last()
            .unwrap_or("")
    );

    let mut builder = Builder::new(File::create(&lib_file)?);
    builder.append_path(&obj_file)?;

    drop(builder);

    let status = Command::new("ranlib").arg(&lib_file).status()?;

    assert!(status.success());

    println!("cargo:rustc-link-lib=static=test_basic");
    println!("cargo:rustc-link-search=native={}", out_dir.display());

    Ok(())
}
