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

//! This crate contains the minimal interface over TVM's
//! C runtime API.
//!
//! These common bindings are useful to both runtimes
//! written in Rust, as well as higher level API bindings.
//!
//! See the `tvm-rt` or `tvm` crates for full bindings to
//! the TVM API.

/// The low-level C runtime FFI API for TVM.
pub mod ffi {
    #![allow(non_camel_case_types, non_snake_case, non_upper_case_globals, unused)]

    use std::os::raw::{c_char, c_int, c_void};

    include!(concat!(env!("OUT_DIR"), "/c_runtime_api.rs"));

    pub type BackendPackedCFunc = extern "C" fn(
        args: *const TVMValue,
        type_codes: *const c_int,
        num_args: c_int,
        out_ret_value: *mut TVMValue,
        out_ret_tcode: *mut u32,
        resource_handle: *mut c_void,
    ) -> c_int;
}

pub mod array;
pub mod byte_array;
pub mod datatype;
pub mod device;
pub mod errors;
#[macro_use]
pub mod packed_func;
pub mod value;

pub use byte_array::ByteArray;
pub use datatype::DataType;
pub use device::{Device, DeviceType};
pub use errors::*;
pub use packed_func::{ArgValue, RetValue};

impl<T, E> std::convert::TryFrom<Result<T, E>> for RetValue
where
    RetValue: std::convert::TryFrom<T>,
    E: From<<RetValue as std::convert::TryFrom<T>>::Error>,
{
    type Error = E;

    fn try_from(val: Result<T, E>) -> Result<RetValue, Self::Error> {
        val.and_then(|t| RetValue::try_from(t).map_err(|e| e.into()))
    }
}
