//! This crate contains the refactored basic components required
//! for `runtime` and `frontend` TVM crates.

#![crate_name = "tvm_common"]
#![recursion_limit = "1024"]
#![allow(non_camel_case_types, unused_imports)]
#![feature(box_syntax, try_from)]

#[macro_use]
extern crate error_chain;

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
pub use ffi::{TVMContext, TVMType};
pub use packed_func::{TVMArgValue, TVMRetValue};
