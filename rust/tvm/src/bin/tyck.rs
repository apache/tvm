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

use std::path::PathBuf;

use anyhow::Result;
use structopt::StructOpt;

use tvm::ir::diagnostics::codespan;
use tvm::ir::{self, IRModule};
use tvm::runtime::Error;

#[derive(Debug, StructOpt)]
#[structopt(name = "tyck", about = "Parse and type check a Relay program.")]
struct Opt {
    /// Input file
    #[structopt(parse(from_os_str))]
    input: PathBuf,
}

fn main() -> Result<()> {
    codespan::init().expect("Failed to initialize Rust based diagnostics.");
    let opt = Opt::from_args();
    let _module = match IRModule::parse_file(opt.input) {
        Err(ir::module::Error::TVM(Error::DiagnosticError(_))) => return Ok(()),
        Err(e) => {
            return Err(e.into());
        }
        Ok(module) => module,
    };

    Ok(())
}
