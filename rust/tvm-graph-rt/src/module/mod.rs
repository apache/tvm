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

#[cfg(not(any(target_arch = "wasm32", target_env = "sgx")))]
mod dso;
mod syslib;

use tvm_sys::{
    ffi::BackendPackedCFunc,
    packed_func::{ArgValue, PackedFunc, RetValue, TVMValue},
};

#[cfg(not(any(target_arch = "wasm32", target_env = "sgx")))]
pub use dso::DsoModule;
pub use syslib::SystemLibModule;

pub trait Module {
    fn get_function<S: AsRef<str>>(&self, name: S) -> Option<&(dyn PackedFunc)>;
}

// @see `WrapPackedFunc` in `llvm_module.cc`.
fn wrap_backend_packed_func(func_name: String, func: BackendPackedCFunc) -> Box<dyn PackedFunc> {
    Box::new(move |args: &[ArgValue]| {
        let (values, type_codes): (Vec<TVMValue>, Vec<i32>) = args
            .iter()
            .map(|arg| {
                let (val, code) = arg.to_tvm_value();
                (val, code as i32)
            })
            .unzip();
        let ret: RetValue = RetValue::default();
        let (mut ret_val, mut ret_type_code) = ret.to_tvm_value();
        let exit_code = func(
            values.as_ptr(),
            type_codes.as_ptr(),
            values.len() as i32,
            &mut ret_val,
            &mut ret_type_code,
            std::ptr::null_mut(),
        );
        if exit_code == 0 {
            Ok(RetValue::from_tvm_value(ret_val, ret_type_code))
        } else {
            Err(tvm_sys::errors::FuncCallError::get_with_context(
                func_name.clone(),
            ))
        }
    })
}
