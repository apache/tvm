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

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use tvm_build::BuildConfig;

/// The necessary information for detecting a TVM installation.
struct TVMInstall {
    source_path: PathBuf,
    build_path: PathBuf,
}

/// Find the TVM install using the provided path.
fn find_using_tvm_path<P: AsRef<Path>>(tvm_path: P) -> Result<TVMInstall> {
    Ok(TVMInstall {
        source_path: tvm_path.as_ref().into(),
        build_path: tvm_path.as_ref().into(),
    })
}

#[allow(unused)]
fn if_unset<K: AsRef<std::ffi::OsStr>, V: AsRef<std::ffi::OsStr>>(k: K, v: V) -> Result<()> {
    match std::env::var(k.as_ref()) {
        Ok(other) if other != "" => {
            println!(
                "cargo:warning=Using existing environment variable setting {:?}={:?}",
                k.as_ref(),
                v.as_ref()
            );
        }
        _ => std::env::set_var(k, v),
    }

    Ok(())
}

/// Find a TVM installation using TVM build by either first installing or detecting.
fn find_using_tvm_build() -> Result<TVMInstall> {
    let mut build_config = BuildConfig::default();
    build_config.repository = Some("https://github.com/apache/tvm".to_string());
    build_config.branch = Some(option_env!("TVM_BRANCH").unwrap_or("main").into());
    let build_result = tvm_build::build(build_config)?;
    let source_path = build_result.revision.source_path();
    let build_path = build_result.revision.build_path();
    Ok(TVMInstall {
        source_path,
        build_path,
    })
}

fn main() -> Result<()> {
    let TVMInstall {
        source_path,
        build_path,
    } = match option_env!("TVM_HOME") {
        Some(tvm_path) if tvm_path != "" => find_using_tvm_path(tvm_path),
        _ => find_using_tvm_build(),
    }?;

    // If the TVM_HOME environment variable changed, the LLVM_CONFIG_PATH environment variable
    // changed, the build directory or headers have changed we need to rebuild the Rust bindings.
    println!("cargo:rerun-if-env-changed=TVM_HOME");
    println!("cargo:rerun-if-env-changed=LLVM_CONFIG_PATH");
    println!("cargo:rerun-if-changed={}", build_path.display());
    println!("cargo:rerun-if-changed={}/include", source_path.display());

    let library_name = if cfg!(feature = "runtime-only") {
        "tvm_runtime"
    } else {
        "tvm"
    };

    match &std::env::var("CARGO_CFG_TARGET_ARCH")
        .expect("CARGO_CFG_TARGET_ARCH must be set by CARGO")[..]
    {
        "wasm32" => {}
        _ => {
            if cfg!(feature = "static-linking") {
                println!("cargo:rustc-link-lib=static={}", library_name);
                // TODO(@jroesch): move this to tvm-build as library_path?
                println!(
                    "cargo:rustc-link-search=native={}/build",
                    build_path.display()
                );
            }

            if cfg!(feature = "dynamic-linking") {
                println!("cargo:rustc-link-lib=dylib={}", library_name);
                println!(
                    "cargo:rustc-link-search=native={}/build",
                    build_path.display()
                );
            }
        }
    };

    let runtime_api = source_path.join("include/tvm/runtime/c_runtime_api.h");
    let backend_api = source_path.join("include/tvm/runtime/c_backend_api.h");
    let source_path = source_path.display().to_string();
    let dlpack_include = format!("-I{}/3rdparty/dlpack/include/", source_path);
    let tvm_include = format!("-I{}/include/", source_path);

    let out_file = PathBuf::from(std::env::var("OUT_DIR")?).join("c_runtime_api.rs");

    // @see rust-bindgen#550 for `blacklist_type`
    bindgen::Builder::default()
        .header(runtime_api.display().to_string())
        .header(backend_api.display().to_string())
        .clang_arg(dlpack_include)
        .clang_arg(tvm_include)
        .blacklist_type("max_align_t")
        .layout_tests(false)
        .derive_partialeq(true)
        .derive_eq(true)
        .derive_default(true)
        .generate()
        .map_err(|()| {
            anyhow::anyhow!("bindgen failed to generate the Rust bindings for the C API")
        })?
        .write_to_file(out_file)
        .context("failed to write the generated Rust binding to disk")?;

    Ok(())
}
