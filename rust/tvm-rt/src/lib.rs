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

//! [TVM](https://github.com/apache/tvm) is a compiler stack for deep learning systems.
//!
//! This crate provides an idiomatic Rust API for TVM runtime.
//!
//! The TVM runtime API contains the data structures used by higher-level TVM executors.
//! Specifically it exposes the basic types such as NDArray, as well as the more general object system.
//! The TVM object system enables cross-language interoperability including that of closures for all
//! supported languages including C++, and Python.

// Macro to check the return call to TVM runtime shared library.

#[macro_export]
macro_rules! tvm_call {
    ($e:expr) => {{
        if unsafe { $e } != 0 {
            Err($crate::get_last_error().into())
        } else {
            Ok(())
        }
    }};
}

#[macro_export]
macro_rules! check_call {
    ($e:expr) => {{
        if unsafe { $e } != 0 {
            panic!("{}", $crate::get_last_error());
        }
    }};
}

// Define all sumodules.
pub mod array;
pub mod device;
pub mod errors;
pub mod function;
pub mod graph_rt;
pub mod map;
pub mod module;
pub mod ndarray;
pub mod object;
pub mod string;
mod to_function;

pub use object::*;
pub use string::*;

use std::{
    ffi::{CStr, CString},
    str,
};

pub use crate::{
    device::{Device, DeviceType},
    errors::*,
    function::Function,
    module::Module,
    ndarray::NDArray,
};

pub use function::{ArgValue, RetValue};
pub use tvm_sys::byte_array::ByteArray;
pub use tvm_sys::datatype::DataType;
use tvm_sys::ffi;

pub use tvm_macros::external;

/// Gets the last error message.
pub fn get_last_error() -> &'static str {
    unsafe {
        match CStr::from_ptr(ffi::TVMGetLastError()).to_str() {
            Ok(s) => s,
            Err(_) => "Invalid UTF-8 message",
        }
    }
}

pub(crate) fn set_last_error<E: std::error::Error>(err: &E) {
    let c_string = CString::new(err.to_string()).unwrap();
    unsafe {
        ffi::TVMAPISetLastError(c_string.as_ptr());
    }
}

/// Outputs the current TVM version.
pub fn version() -> &'static str {
    match str::from_utf8(ffi::TVM_VERSION) {
        Ok(s) => s,
        Err(_) => "Invalid UTF-8 string",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ByteArray, DataType, Device};
    use std::{convert::TryInto, str::FromStr};

    #[test]
    fn print_version() {
        println!("TVM version: {}", version());
    }

    #[test]
    fn set_error() {
        let err = errors::NDArrayError::EmptyArray;
        set_last_error(&err);
        assert_eq!(
            get_last_error().trim(),
            errors::NDArrayError::EmptyArray.to_string()
        );
    }

    // todo(@jroesch): #8800 Follow up with ByteArray RetValue ownership.
    // #[test]
    // fn bytearray() {
    //     let w = vec![1u8, 2, 3, 4, 5];
    //     let v = ByteArray::from(w.as_slice());
    //     let tvm: ByteArray = RetValue::from(v).try_into().unwrap();
    //     assert_eq!(
    //         tvm.data(),
    //         w.iter().copied().collect::<Vec<u8>>().as_slice()
    //     );
    // }

    #[test]
    fn ty() {
        let t = DataType::from_str("int32").unwrap();
        let tvm: DataType = RetValue::from(t).try_into().unwrap();
        assert_eq!(tvm, t);
    }

    #[test]
    fn device() {
        let c = Device::from_str("cuda").unwrap();
        let tvm: Device = RetValue::from(c).try_into().unwrap();
        assert_eq!(tvm, c);
    }
}
