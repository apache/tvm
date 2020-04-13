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

//! This crate contains the refactored basic components required
//! for `runtime` and `frontend` TVM crates.

#[macro_use]
extern crate failure;

/// Unified ffi module for both runtime and frontend crates.
pub mod ffi {
    #![allow(non_camel_case_types, non_snake_case, non_upper_case_globals, unused)]

    use std::os::raw::{c_char, c_int, c_void};

    include!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/c_runtime_api.rs"));

    pub type BackendPackedCFunc =
        extern "C" fn(args: *const TVMValue, type_codes: *const c_int, num_args: c_int) -> c_int;
}

pub mod array;
pub mod errors;
#[macro_use]
pub mod packed_func;
pub mod value;

pub use errors::*;
pub use ffi::{DLDataType as TVMType, TVMByteArray, TVMContext};
pub use packed_func::{TVMArgValue, TVMRetValue};
