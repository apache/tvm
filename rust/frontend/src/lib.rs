//! [TVM](https://github.com/dmlc/tvm) is a compiler stack for deep learning systems.
//!
//! This crate provides an idiomatic Rust API for TVM runtime frontend.
//!
//! One particular use case is that given optimized deep learning model artifacts,
//! (compiled with TVM) which include a shared library
//! `lib.so`, `graph.json` and a byte-array `param.params`, one can load them
//! in Rust idomatically to create a TVM Graph Runtime and
//! run the model for some inputs and get the
//! desired predictions *all in Rust*.
//!
//! Checkout the `examples` repository for more details.

#![crate_name = "tvm_frontend"]
#![recursion_limit = "1024"]
#![allow(non_camel_case_types, unused_unsafe)]
#![feature(
    try_from,
    try_trait,
    fn_traits,
    unboxed_closures,
    box_syntax,
    option_replace
)]

#[macro_use]
extern crate error_chain;
extern crate tvm_common as common;
#[macro_use]
extern crate lazy_static;
extern crate ndarray as rust_ndarray;
extern crate num_traits;

use std::{
    ffi::{CStr, CString},
    str,
};

use crate::common::ffi::ts;

// Macro to check the return call to TVM runtime shared library.
macro_rules! check_call {
    ($e:expr) => {{
        if unsafe { $e } != 0 {
            panic!("{}", $crate::get_last_error());
        }
    }};
}

/// Gets the last error message.
pub fn get_last_error() -> &'static str {
    unsafe {
        match CStr::from_ptr(ts::TVMGetLastError()).to_str() {
            Ok(s) => s,
            Err(_) => "Invalid UTF-8 message",
        }
    }
}

pub(crate) fn set_last_error(err: &Error) {
    let c_string = CString::new(err.to_string()).unwrap();
    unsafe {
        ts::TVMAPISetLastError(c_string.as_ptr());
    }
}

#[macro_use]
pub mod function;
pub mod bytearray;
pub mod context;
pub mod errors;
pub mod module;
pub mod ndarray;
pub mod ty;
pub mod value;

pub use crate::{
    bytearray::TVMByteArray,
    common::{
        errors as common_errors,
        ty::TVMTypeCode,
        value::{TVMArgValue, TVMRetValue, TVMValue},
    },
    context::{TVMContext, TVMDeviceType},
    errors::*,
    function::Function,
    module::Module,
    ndarray::NDArray,
    ty::TVMType,
};

/// Outputs the current TVM version.
pub fn version() -> &'static str {
    match str::from_utf8(ts::TVM_VERSION) {
        Ok(s) => s,
        Err(_) => "Invalid UTF-8 string",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn print_version() {
        println!("TVM version: {}", version());
    }

    #[test]
    fn set_error() {
        let err = ErrorKind::EmptyArray;
        set_last_error(&err.into());
        assert_eq!(get_last_error().trim(), ErrorKind::EmptyArray.to_string());
    }
}
