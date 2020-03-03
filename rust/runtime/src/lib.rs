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

//! This crate is an implementation of the TVM runtime for modules compiled with `--system-lib`.
//! It's mainly useful for compiling to WebAssembly and SGX,
//! but also native if you prefer Rust to C++.
//!
//! For TVM graphs, the entrypoint to this crate is `runtime::GraphExecutor`.
//! Single-function modules are used via the `packed_func!` macro after obtaining
//! the function from `runtime::SystemLibModule`
//!
//! The main entrypoints to this crate are `GraphExecutor`
//! For examples of use, please refer to the multi-file tests in the `tests` directory.

#[macro_use]
extern crate failure;
#[macro_use]
extern crate itertools;
#[macro_use]
extern crate lazy_static;
extern crate ndarray;
#[macro_use]
extern crate nom;
extern crate num_cpus;
extern crate serde;
#[macro_use]
extern crate serde_derive;
extern crate serde_json;
extern crate tvm_common;

mod allocator;
mod array;
pub mod errors;
mod graph;
mod module;
mod threading;
mod workspace;

pub use tvm_common::{
    call_packed,
    errors::*,
    ffi::{self, DLTensor},
    packed_func::{self, *},
    TVMArgValue, TVMRetValue,
};
pub use tvm_macros::import_module;

pub use self::{array::*, errors::*, graph::*, module::*, threading::*, workspace::*};

lazy_static! {
    static ref LAST_ERROR: std::sync::RwLock<Option<&'static std::ffi::CStr>> =
        std::sync::RwLock::new(None);
}

#[no_mangle]
pub unsafe extern "C" fn TVMAPISetLastError(cmsg: *const i8) {
    *LAST_ERROR.write().unwrap() = Some(std::ffi::CStr::from_ptr(cmsg));
}

#[no_mangle]
pub extern "C" fn TVMGetLastError() -> *const std::os::raw::c_char {
    match *LAST_ERROR.read().unwrap() {
        Some(err) => err.as_ptr(),
        None => std::ptr::null(),
    }
}
