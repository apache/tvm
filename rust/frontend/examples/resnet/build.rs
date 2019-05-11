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

use std::{path::Path, process::Command};

fn main() {
    let output = Command::new("python3")
        .arg(concat!(env!("CARGO_MANIFEST_DIR"), "/src/build_resnet.py"))
        .arg(&format!("--build-dir={}", env!("CARGO_MANIFEST_DIR")))
        .output()
        .expect("Failed to execute command");
    assert!(
        Path::new(&format!("{}/deploy_lib.o", env!("CARGO_MANIFEST_DIR"))).exists(),
        "Could not prepare demo: {}",
        String::from_utf8(output.stderr)
            .unwrap()
            .trim()
            .split("\n")
            .last()
            .unwrap_or("")
    );
    println!(
        "cargo:rustc-link-search=native={}",
        env!("CARGO_MANIFEST_DIR")
    );
}
