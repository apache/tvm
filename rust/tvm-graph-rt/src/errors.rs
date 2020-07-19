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

use thiserror::Error;
use tvm_sys::DataType;

#[derive(Debug, Error)]
pub enum GraphFormatError {
    #[error("Could not parse graph json")]
    Parse(#[from] serde_json::Error),
    #[error("Could not parse graph params")]
    Params,
    #[error("{0} is missing attr: {1}")]
    MissingAttr(String, String),
    #[error("Graph has invalid attr that can't be parsed: {0}")]
    InvalidAttr(#[from] std::num::ParseIntError),
    #[error("Missing field: {0}")]
    MissingField(&'static str),
    #[error("Invalid DLType: {0}")]
    InvalidDLType(String),
    #[error("Unsupported Op: {0}")]
    UnsupportedOp(String),
}

#[derive(Debug, Error)]
#[error("Function {0} not found")]
pub struct FunctionNotFound(pub String);

#[derive(Debug, Error)]
#[error("Pointer {0:?} invalid when freeing")]
pub struct InvalidPointer(pub *mut u8);

#[derive(Debug, Error)]
pub enum ArrayError {
    #[error("Cannot convert Tensor with dtype {0} to ndarray")]
    IncompatibleDataType(DataType),
    #[error("Shape error when casting ndarray to TVM Array with shape {0:?}")]
    ShapeError(Vec<i64>),
}
