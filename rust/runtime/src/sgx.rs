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

use std::{
    ffi::CString,
    os::raw::{c_char, c_int},
};

pub use crate::threading::tvm_run_worker as run_worker;
use crate::{threading::sgx_join_threads, SystemLibModule, TVMArgValue, TVMRetValue};
use errors::SgxError;
use ffi::TVMValue;

#[macro_export]
macro_rules! tvm_ocall {
    ($func: expr) => {
        match $func {
            0 => Ok(()),
            code => Err(SgxError { code }),
        }
    };
}

pub type SgxStatus = u32;

#[cfg(target_env = "sgx")]
extern "C" {
    fn tvm_ocall_packed_func(
        name: *const c_char,
        arg_values: *const TVMValue,
        type_codes: *const c_int,
        num_args: c_int,
        ret_val: *mut TVMValue,
        ret_type_code: *mut c_int,
    ) -> SgxStatus;
}

pub fn ocall_packed_func<S: AsRef<str>>(
    fn_name: S,
    args: &[TVMArgValue],
) -> Result<TVMRetValue, SgxError> {
    let mut ret_val = TVMValue { v_int64: 0 };
    let ret_type_code = 0i64;
    unsafe {
        tvm_ocall!(tvm_ocall_packed_func(
            CString::new(fn_name.as_ref()).unwrap().as_ptr(),
            args.iter()
                .map(|ref arg| arg.value)
                .collect::<Vec<TVMValue>>()
                .as_ptr(),
            args.iter()
                .map(|ref arg| arg.type_code as i32)
                .collect::<Vec<i32>>()
                .as_ptr() as *const i32,
            args.len() as i32,
            &mut ret_val as *mut TVMValue,
            &mut (ret_type_code as i32) as *mut c_int,
        ))?;
    }
    Ok(TVMRetValue::from_tvm_value(ret_val, ret_type_code as i64))
}

#[macro_export]
macro_rules! ocall_packed {
  ($fn_name:expr, $($args:expr),+) => {
    $crate::sgx::ocall_packed_func($fn_name, &[$($args.into(),)+])
      .expect(concat!("Error calling `", $fn_name, "`"))
  };
  ($fn_name:expr) => {
    $crate::sgx::ocall_packed_func($fn_name, &Vec::new())
      .expect(concat!("Error calling `", $fn_name, "`"))
  }
}

pub fn shutdown() {
    if env!("TVM_NUM_THREADS") != "0" {
        sgx_join_threads()
    }
}

impl Drop for SystemLibModule {
    fn drop(&mut self) {
        shutdown()
    }
}
