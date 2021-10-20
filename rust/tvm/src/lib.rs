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
//! This crate provides an idiomatic Rust API for TVM runtime frontend.
//!
//! One particular use case is that given optimized deep learning model artifacts,
//! (compiled with TVM) which include a shared library
//! `lib.so`, `graph.json` and a byte-array `param.params`, one can load them
//! in Rust idiomatically to create a TVM Graph Executor and
//! run the model for some inputs and get the
//! desired predictions *all in Rust*.
//!
//! Checkout the `examples` repository for more details.

pub use crate::{errors::*, function::Function, module::Module, ndarray::NDArray};

pub use tvm_rt::{DataType, Device, DeviceType};

pub use tvm_rt::device;
pub use tvm_rt::errors;
pub use tvm_rt::function;
pub use tvm_rt::module;
pub use tvm_rt::ndarray;

#[cfg(feature = "python")]
pub mod compiler;
pub mod ir;
#[cfg(feature = "python")]
pub mod python;
pub mod runtime;
pub mod transform;

pub use runtime::version;

#[macro_export]
macro_rules! export {
    ($($fn_name:expr),*) => {
        pub fn tvm_export(ns: &str) -> Result<(), tvm::Error> {
            $(
                let name = String::from(ns) + ::std::stringify!($fn_name);
                tvm::runtime::function::register_override($fn_name, name, true)?;
            )*
            Ok(())
        }
    }
}

#[macro_export]
macro_rules! export_mod {
    ($ns:expr, $($mod_name:expr),*) => {
        pub fn tvm_mod_export() -> Result<(), tvm::Error> {
            $(
                $mod_name::tvm_export($ns)?;
            )*
            Ok(())
        }
    }
}
