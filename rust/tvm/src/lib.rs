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

//! [TVM](https://github.com/apache/incubator-tvm) is a compiler stack for deep learning systems.
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

pub use crate::{errors::*, function::Function, module::Module, ndarray::NDArray};

pub use tvm_rt::{Context, DataType, DeviceType};

pub use tvm_rt::context;
pub use tvm_rt::errors;
pub use tvm_rt::function;
pub use tvm_rt::module;
pub use tvm_rt::ndarray;
pub use tvm_rt::value;
pub mod ir;
pub mod runtime;
pub mod transform;

pub use runtime::version;
