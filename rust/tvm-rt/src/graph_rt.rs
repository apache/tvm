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

use crate::Function;
use crate::{function::Result, ByteArray, Device, Module, NDArray};

/// An instance of the C++ graph executor.
///
/// An efficient and light weight runtime for static deep learning models.
pub struct GraphRt {
    /// The backing graph executor module which exposes a set of packed functions
    /// which can be invoked by a client.
    ///
    /// In the graph executor module, it exposes create, load_params, set_input, get_output, and run.
    module: Module,
}

impl GraphRt {
    /// Create a graph executor directly from a runtime module.
    pub fn from_module(module: Module, dev: Device) -> Result<GraphRt> {
        let default: Box<dyn Fn(Device) -> Result<Module>> =
            module.get_function("default", false)?.into();

        Ok(Self {
            module: default(dev)?,
        })
    }

    /// Create a graph executor from the deprecated graph, lib, dev triple.
    pub fn create_from_parts(graph: &str, lib: Module, dev: Device) -> Result<Self> {
        let runtime_create_fn = Function::get("tvm.graph_executor.create").unwrap();

        let runtime_create_fn_ret = runtime_create_fn.invoke(vec![
            graph.into(),
            (&lib).into(),
            (&dev.device_type).into(),
            // NOTE you must pass the device id in as i32 because that's what TVM expects
            (dev.device_id as i32).into(),
        ]);

        let graph_executor_module: Module = runtime_create_fn_ret?.try_into()?;
        Ok(Self {
            module: graph_executor_module,
        })
    }

    /// Load the parameters of the model into the runtime.
    pub fn load_params<P>(&mut self, params: P) -> Result<()>
    where
        P: Into<ByteArray>,
    {
        let load_param_fn = self.module.get_function("load_params", false)?;

        let params: ByteArray = params.into();

        load_param_fn.invoke(vec![(&params).into()])?;

        Ok(())
    }

    /// Set the input with name `name` with the value of `input`.
    pub fn set_input(&mut self, name: &str, input: NDArray) -> Result<()> {
        let ref set_input_fn = self.module.get_function("set_input", false)?;

        set_input_fn.invoke(vec![name.into(), (&input).into()])?;
        Ok(())
    }

    /// Run the graph module, once setting parameters and inputs.
    pub fn run(&mut self) -> Result<()> {
        let ref run_fn = self.module.get_function("run", false)?;

        // execute the run function. Note that it has no argument
        run_fn.invoke(vec![])?;
        Ok(())
    }

    /// Extract the ith output from the graph executor and returns it.
    pub fn get_output(&mut self, i: i64) -> Result<NDArray> {
        let get_output_fn = self.module.get_function("get_output", false)?;
        get_output_fn.invoke(vec![i.into()])?.try_into()
    }

    /// Extract the ith output from the graph executor and write the results into output.
    pub fn get_output_into(&mut self, i: i64, output: NDArray) -> Result<()> {
        let get_output_fn = self.module.get_function("get_output", false)?;
        get_output_fn.invoke(vec![i.into(), (&output).into()])?;
        Ok(())
    }
}
