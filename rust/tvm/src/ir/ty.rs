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

use crate::runtime::{Object, ObjectRef};
use tvm_macros::Object;
use tvm_rt::{array::Array, DataType};

use super::PrimExpr;

#[repr(C)]
#[derive(Object)]
#[ref_name = "Type"]
#[type_key = "Type"]
pub struct TypeNode {
    pub base: Object,
    pub span: ObjectRef,
}

#[repr(C)]
#[derive(Object)]
#[ref_name = "BaseTensorType"]
#[type_key = "relay.BaseTensorType"]
pub struct BaseTensorTypeNode {
    pub base: TypeNode,
}

#[repr(C)]
#[derive(Object)]
#[ref_name = "TensorType"]
#[type_key = "relay.TensorType"]
pub struct TensorTypeNode {
    pub base: TypeNode,
    pub shape: Array<PrimExpr>,
    pub dtype: DataType,
}
