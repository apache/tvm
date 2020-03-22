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

use ar::Builder;

fn main() {
    let out_dir = env::var("OUT_DIR").unwrap();

    let output = Command::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/src/build_test_graph.py"
    ))
    .arg(&out_dir)
    .output()
    .expect("Failed to execute command");
    assert!(
        Path::new(&format!("{}/graph.o", out_dir)).exists(),
        "Could not build graph lib: {}",
        String::from_utf8(output.stderr)
            .unwrap()
            .trim()
            .split("\n")
            .last()
            .unwrap_or("")
    );

    let lib_file = format!("{}/libtestnn.a", out_dir);
    let file = File::create(&lib_file).unwrap();
    let mut builder = Builder::new(file);
    builder.append_path(format!("{}/graph.o", out_dir)).unwrap();

    let status = Command::new("ranlib")
        .arg(&lib_file)
        .status()
        .expect("fdjlksafjdsa");

    assert!(status.success());


    println!("cargo:rustc-link-lib=static=testnn");
    println!("cargo:rustc-link-search=native={}", out_dir);
}
