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

use crate::DataType;
use thiserror::Error;

#[derive(Debug, Error)]
#[error("Function was not set in `function::Builder`")]
pub struct FunctionNotFoundError;

#[derive(Debug, Error)]
#[error("Expected type `{expected}` but found `{actual}`")]
pub struct TypeMismatchError {
    pub expected: String,
    pub actual: String,
}

#[derive(Debug, Error)]
pub enum NDArrayError {
    #[error("Cannot convert from an empty array.")]
    EmptyArray,
    #[error("Invalid datatype when attempting to convert ndarray.")]
    InvalidDatatype(#[from] tvm_sys::datatype::ParseDataTypeError),
    #[error("a shape error occurred in the Rust ndarray library")]
    ShapeError(#[from] ndarray::ShapeError),
    #[error("Expected type `{expected}` but found `{actual}`")]
    DataTypeMismatch {
        expected: DataType,
        actual: DataType,
    },
}

#[derive(Debug, Error)]
pub enum Error {
    #[error("{0}")]
    Downcast(#[from] tvm_sys::errors::ValueDowncastError),
    #[error("raw pointer passed across boundary was null")]
    Null,
    #[error("failed to load module due to invalid path {0}")]
    ModuleLoadPath(String),
    #[error("failed to convert String into CString due to embedded nul character")]
    ToCString(#[from] std::ffi::NulError),
    #[error("failed to convert CString into String")]
    FromCString(#[from] std::ffi::IntoStringError),
    #[error("Handle `{0}` is null.")]
    NullHandle(String),
    #[error("{0}")]
    NDArray(#[from] NDArrayError),
    #[error("{0}")]
    CallFailed(String),
    #[error("this case will never occur")]
    Infallible(#[from] std::convert::Infallible),
    #[error("a panic occurred while executing a Rust packed function")]
    Panic,
    #[error(
        "one or more error diagnostics were emitted, please check diagnostic render for output."
    )]
    DiagnosticError(String),
    #[error("{0}")]
    Raw(String),
}

impl Error {
    pub fn from_raw_tvm(raw: &str) -> Error {
        let err_header = raw.find(":").unwrap_or(0);
        let (err_ty, err_content) = raw.split_at(err_header);
        match err_ty {
            "DiagnosticError" => Error::DiagnosticError((&err_content[1..]).into()),
            _ => Error::Raw(raw.into()),
        }
    }
}

impl Error {
    pub fn downcast(actual_type: String, expected_type: &'static str) -> Error {
        Self::Downcast(tvm_sys::errors::ValueDowncastError {
            actual_type,
            expected_type,
        })
    }
}
