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

#![feature(
    allocator_api,
    box_syntax,
    fn_traits,
    unboxed_closures,
    vec_remove_item
)]

#[cfg(target_env = "sgx")]
extern crate alloc;
extern crate bounded_spsc_queue;
#[cfg(target_env = "sgx")]
extern crate core;
#[macro_use]
extern crate failure;
#[macro_use]
extern crate itertools;
#[macro_use]
extern crate lazy_static;
extern crate ndarray;
#[macro_use]
extern crate nom;
#[cfg(not(target_env = "sgx"))]
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
#[cfg(target_env = "sgx")]
#[macro_use]
pub mod sgx;
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
pub extern "C" fn TVMAPISetLastError(cmsg: *const i8) {
    *LAST_ERROR.write().unwrap() = Some(unsafe { std::ffi::CStr::from_ptr(cmsg) });
    #[cfg(target_env = "sgx")]
    ocall_packed!("__sgx_set_last_error__", cmsg);
}

#[no_mangle]
pub extern "C" fn TVMGetLastError() -> *const std::os::raw::c_char {
    match *LAST_ERROR.read().unwrap() {
        Some(err) => err.as_ptr(),
        None => std::ptr::null(),
    }
}
