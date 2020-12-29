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

use std::{env, fs::File, path::Path, process::Command};

use anyhow::{Context, Result};
use ar::Builder;

fn main() -> Result<()> {
    let out_dir = env::var("OUT_DIR")?;
    let out_dir = Path::new(&out_dir).join("test_nn");

    std::fs::create_dir_all(&out_dir)?;

    let manifest_dir = env::var("CARGO_MANIFEST_DIR")?;
    let manifest_dir = Path::new(&manifest_dir);

    let generator = manifest_dir.join("src").join("build_test_graph.py");

    let graph_path = out_dir.join("graph.o");

    let output = Command::new(&generator)
        .arg(&out_dir)
        .output()
        .with_context(|| format!("Failed to execute: {:?}", generator))?;

    assert!(
        graph_path.exists(),
        "Could not build graph lib: {}",
        String::from_utf8(output.stderr)
            .unwrap()
            .trim()
            .split("\n")
            .last()
            .unwrap_or("")
    );

    let lib_file = out_dir.join("libtestnn.a");
    let file = File::create(&lib_file).context("failed to create library file")?;
    let mut builder = Builder::new(file);
    builder.append_path(graph_path)?;

    let status = Command::new("ranlib").arg(&lib_file).status()?;

    assert!(status.success());

    println!("cargo:rustc-link-lib=static=testnn");
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rerun-if-changed={}", generator.display());

    Ok(())
}
