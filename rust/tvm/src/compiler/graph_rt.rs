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

use std::convert::TryInto;
use std::io::Read;
use std::path::Path;

use once_cell::sync::Lazy;
use thiserror::Error;

use crate::ir::IRModule;
use crate::python;
use crate::runtime::{map::Map, Function, Module as RtModule, NDArray, String};

#[derive(Error, Debug)]
pub enum Error {
    #[error("{0}")]
    IO(#[from] std::io::Error),
    #[error("{0}")]
    TVM(#[from] crate::errors::Error),
}

static TVM_BUILD: Lazy<Function> = Lazy::new(|| {
    python::import("tvm").unwrap();
    python::import("tvm.relay").unwrap();
    Function::get("tvm.relay.build").unwrap()
});

fn _compile_module(
    module: IRModule,
    target: String,
    target_host: String,
    params: Map<String, NDArray>,
    module_name: String,
) -> Result<RtModule, Error> {
    // The RAW API is Fn(IRModule, String, String, Map<String, NDArray>, String);
    let module = TVM_BUILD.invoke(vec![
        (&module).into(),
        (&target).into(),
        (&target_host).into(),
        (&params).into(),
        (&module_name).into(),
    ])?;
    let module: RtModule = module.try_into().unwrap();
    Ok(module)
}

#[derive(Debug)]
pub struct CompilerConfig {
    target: Option<String>,
    target_host: Option<String>,
    params: Map<String, NDArray>,
    module_name: Option<String>,
}

impl Default for CompilerConfig {
    fn default() -> Self {
        CompilerConfig {
            target: None,
            target_host: None,
            params: Map::empty(),
            module_name: None,
        }
    }
}

/// Compile a module from a configuration and IRModule.
///
/// # Arguments
///
/// * `config` - The configuration for the compiler.
/// * `module` - The IRModule to compile.
pub fn compile_module(config: CompilerConfig, module: IRModule) -> Result<RtModule, Error> {
    let target = config.target.unwrap_or("llvm".into());
    _compile_module(
        module,
        target,
        "llvm".into(),
        Map::<String, NDArray>::empty(),
        "default".into(),
    )
}

/// Compile an IRModule on disk and output a runtime module to disk.
///
/// # Arguments
/// * `config` - The configuration for the compiler.
/// * `ir_mod_path` - The path the serialized IRModule.
//
/// * `output_rt_mod_path` - The path to the output runtime module.
pub fn compile_from_disk<P1, P2>(
    config: CompilerConfig,
    ir_mod_path: P1,
    output_rt_mod_path: P2,
) -> Result<(), Error>
where
    P1: AsRef<Path>,
    P2: AsRef<Path>,
{
    let mut input_file = std::fs::File::open(ir_mod_path.as_ref())?;
    let mut input_module_text = std::string::String::new();
    input_file.read_to_string(&mut input_module_text)?;
    let input_module = IRModule::parse("name", input_module_text)?;
    let rt_module = compile_module(config, input_module)?;
    let output_path_str = output_rt_mod_path.as_ref().display().to_string();
    rt_module.export_library(output_path_str)?;
    Ok(())
}
