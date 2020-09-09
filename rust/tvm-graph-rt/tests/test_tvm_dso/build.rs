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

use std::{env, path::Path, process::Command};

use anyhow::{Context, Result};

fn main() -> Result<()> {
    let out_dir = env::var("OUT_DIR").unwrap();

    let exe = concat!(env!("CARGO_MANIFEST_DIR"), "/src/build_test_lib.py");

    let output = Command::new(exe)
        .arg(&out_dir)
        .output()
        .with_context(|| anyhow::anyhow!("Failed to execute: {} {}", exe, &out_dir))?;

    assert!(
        Path::new(&format!("{}/test.so", out_dir)).exists(),
        "Could not build tvm lib: {}",
        String::from_utf8(output.stderr)
            .unwrap()
            .trim()
            .split("\n")
            .last()
            .unwrap_or("")
    );

    Ok(())
}
