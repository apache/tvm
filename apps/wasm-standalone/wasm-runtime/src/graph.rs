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

use anyhow::Result;
use wasmtime::*;
use wasmtime_wasi::{Wasi, WasiCtx};

use super::Tensor;

pub struct GraphExecutor {
    pub(crate) wasm_addr: i32,
    pub(crate) input_size: i32,
    pub(crate) output_size: i32,
    pub(crate) instance: Option<Instance>,
}

#[allow(dead_code)]
impl GraphExecutor {
    pub fn new() -> Self {
        Self {
            wasm_addr: 0,
            input_size: 0,
            output_size: 0,
            instance: None,
        }
    }

    pub fn instantiate(&mut self, wasm_graph_file: String) -> Result<()> {
        let engine = Engine::new(Config::new().wasm_simd(true));
        let store = Store::new(&engine);

        // First set up our linker which is going to be linking modules together. We
        // want our linker to have wasi available, so we set that up here as well.
        let mut linker = Linker::new(&store);
        // Create an instance of `Wasi` which contains a `WasiCtx`. Note that
        // `WasiCtx` provides a number of ways to configure what the target program
        // will have access to.
        let wasi = Wasi::new(&store, WasiCtx::new(std::env::args())?);
        wasi.add_to_linker(&mut linker)?;

        let module = Module::from_file(&store, &wasm_graph_file)?;
        self.instance = Some(linker.instantiate(&module)?);

        Ok(())
    }

    pub fn set_input(&mut self, input_data: Tensor) -> Result<()> {
        let memory = self
            .instance
            .as_ref()
            .unwrap()
            .get_memory("memory")
            .ok_or_else(|| anyhow::format_err!("failed to find `memory` export"))?;

        // Specify the wasm address to access the wasm memory.
        let wasm_addr = memory.data_size();
        // Serialize the data into a JSON string.
        let in_data = serde_json::to_vec(&input_data)?;
        let in_size = in_data.len();
        // Grow up memory size according to in_size to avoid memory leak.
        memory.grow((in_size >> 16) as u32 + 1)?;

        // Insert the input data into wasm memory.
        for i in 0..in_size {
            unsafe {
                memory.data_unchecked_mut()[wasm_addr + i] = *in_data.get(i).unwrap();
            }
        }

        self.wasm_addr = wasm_addr as i32;
        self.input_size = in_size as i32;
        Ok(())
    }

    pub fn run(&mut self) -> Result<()> {
        // Invoke `run` export.
        let run = self
            .instance
            .as_ref()
            .unwrap()
            .get_func("run")
            .ok_or_else(|| anyhow::format_err!("failed to find `run` function export!"))?
            .get2::<i32, i32, i32>()?;

        let out_size = run(self.wasm_addr, self.input_size)?;
        if out_size == 0 {
            panic!("graph run failed!");
        }

        self.output_size = out_size;
        Ok(())
    }

    pub fn get_output(&self) -> Result<Tensor> {
        let memory = self
            .instance
            .as_ref()
            .unwrap()
            .get_memory("memory")
            .ok_or_else(|| anyhow::format_err!("failed to find `memory` export"))?;

        let out_data = unsafe {
            &memory.data_unchecked()[self.wasm_addr as usize..][..self.output_size as usize]
        };
        let out_vec: Tensor = serde_json::from_slice(out_data).unwrap();
        Ok(out_vec)
    }
}

impl Default for GraphExecutor {
    fn default() -> Self {
        Self::new()
    }
}
