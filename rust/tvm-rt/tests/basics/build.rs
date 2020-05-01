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

    let output = std::process::Command::new(concat!(env!("CARGO_MANIFEST_DIR"), "/src/tvm_add.py"))
        .args(&[
            if cfg!(feature = "cpu") {
                "llvm"
            } else {
                "cuda"
            },
            &std::env::var("OUT_DIR").unwrap(),
        ])
        .output()
        .expect("Failed to execute command");
    assert!(
        std::path::Path::new(&format!("{}/test_add.so", out_dir)).exists(),
        "Could not build tvm lib: {}",
        String::from_utf8(output.stderr)
            .unwrap()
            .trim()
            .split("\n")
            .last()
            .unwrap_or("")
    );

    println!("cargo:rustc-link-search=native={}", out_dir);
}
