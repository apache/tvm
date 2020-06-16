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

use std::ffi::c_void;
use std::os::raw::c_int;
use tvm::export_pass;
use tvm::ir::relay::{self, Function};
use tvm::runtime::function::{register, Result};
use tvm::runtime::ObjectRef;
use tvm::transform::{function_pass, IRModule, Pass, PassContext, PassInfo};

fn my_pass_fn(func: relay::Function, module: IRModule, ctx: PassContext) -> Function {
    let var = relay::Var::new("Hi from Rust!".into(), ObjectRef::null());
    relay::Function::new(
        func.params.clone(),
        var.to_expr(),
        func.ret_type.clone(),
        func.type_params.clone(),
    )
}

// fn the_pass() -> Result<Pass> {
//     let pass_info = PassInfo::new(15, "RustPass".into(), vec![])?;
//     function_pass(my_pass_fn, pass_info)
// }

export_pass!("out_of_tree.Pass", my_pass_fn);
