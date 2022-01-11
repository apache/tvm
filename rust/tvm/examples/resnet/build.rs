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

use anyhow::{Context, Result};
use std::{io::Write, path::Path, process::Command};

fn main() -> Result<()> {
    let out_dir = std::env::var("CARGO_MANIFEST_DIR")?;
    let python_script = concat!(env!("CARGO_MANIFEST_DIR"), "/src/build_resnet.py");
    let synset_txt = concat!(env!("CARGO_MANIFEST_DIR"), "/synset.txt");

    println!("cargo:rerun-if-changed={}", python_script);
    println!("cargo:rerun-if-changed={}", synset_txt);

    let output = Command::new("python3")
        .arg(python_script)
        .arg(&format!("--build-dir={}", out_dir))
        .output()
        .with_context(|| anyhow::anyhow!("failed to run python3"))?;

    if !output.status.success() {
        std::io::stdout()
            .write_all(&output.stderr)
            .context("Failed to write error")?;
        panic!("Failed to execute build script");
    }

    assert!(
        Path::new(&format!("{}/deploy_lib.o", out_dir)).exists(),
        "Could not prepare demo: {}",
        String::from_utf8(output.stderr)
            .unwrap()
            .trim()
            .split("\n")
            .last()
            .unwrap_or("")
    );
    println!("cargo:rustc-link-search=native={}", out_dir);

    Ok(())
}
