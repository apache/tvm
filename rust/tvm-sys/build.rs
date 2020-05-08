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

extern crate bindgen;

use std::path::PathBuf;

use std::env;

fn main() {
    let tvm_home = option_env!("TVM_HOME").map(str::to_string).unwrap_or({
        let crate_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .canonicalize()
            .unwrap();
        crate_dir
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .to_str()
            .unwrap()
            .to_string()
    });

    if cfg!(feature = "bindings") {
        println!("cargo:rerun-if-env-changed=TVM_HOME");
        println!("cargo:rustc-link-lib=dylib=tvm");
        println!("cargo:rustc-link-search={}/build", tvm_home);
    }

    // @see rust-bindgen#550 for `blacklist_type`
    bindgen::Builder::default()
        .header(format!("{}/include/tvm/runtime/c_runtime_api.h", tvm_home))
        .header(format!("{}/include/tvm/runtime/c_backend_api.h", tvm_home))
        .clang_arg(format!("-I{}/3rdparty/dlpack/include/", tvm_home))
        .clang_arg(format!("-I{}/include/", tvm_home))
        .blacklist_type("max_align_t")
        .layout_tests(false)
        .derive_partialeq(true)
        .derive_eq(true)
        .generate()
        .expect("unable to generate bindings")
        .write_to_file(PathBuf::from("src/c_runtime_api.rs"))
        .expect("can not write the bindings!");
}
