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
    alloc,
    allocator_api,
    box_syntax,
    fn_traits,
    try_from,
    unboxed_closures,
    vec_remove_item
)]

#[cfg(target_env = "sgx")]
extern crate alloc;
extern crate bounded_spsc_queue;
#[cfg(target_env = "sgx")]
extern crate core;
#[macro_use]
extern crate error_chain;
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
extern crate tvm_common as common;

mod allocator;
mod array;
pub mod errors;
mod module;
#[macro_use]
mod packed_func;
mod graph;
#[cfg(target_env = "sgx")]
#[macro_use]
pub mod sgx;
mod threading;
mod workspace;

pub use crate::common::{errors::*, ffi, TVMArgValue, TVMRetValue};

pub use self::{
    array::*, errors::*, graph::*, module::*, packed_func::*, threading::*, workspace::*,
};

#[cfg(target_env = "sgx")]
use self::sgx::ocall_packed_func;

#[no_mangle]
pub extern "C" fn TVMAPISetLastError(cmsg: *const i8) {
    #[cfg(not(target_env = "sgx"))]
    unsafe {
        panic!(std::ffi::CStr::from_ptr(cmsg).to_str().unwrap());
    }
    #[cfg(target_env = "sgx")]
    ocall_packed!("__sgx_set_last_error__", cmsg);
}
