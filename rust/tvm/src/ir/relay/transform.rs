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

use crate::runtime::array::Array;
use crate::runtime::function::Result;
use crate::runtime::string::String as TVMString;
use crate::runtime::{external, IsObjectRef, Object, ObjectRef};
use crate::ir::PrimExpr;
use crate::ir::tir::IntImm;
use crate::ir::relay::Expr;
use crate::function::ffi::DLDataType;

external! {
    #[name("relay.op._make.squeeze")]
    pub fn squeeze(data: Expr, axis: Array<PrimExpr>) -> Expr;

    #[name("relay.op._make.take")]
    pub fn take(data: Expr, indices: Expr, batch_dims: IntImm, axis: IntImm, mode: TVMString) -> Expr;

    #[name("relay.op._make.split")]
    pub fn split(data: Expr, indices_or_sections: ObjectRef, axis: i32) -> Expr;

    #[name("relay.op._make.strided_slice")]
    pub fn strided_slice(data: Expr, begin: Array<PrimExpr>, end: Array<PrimExpr>, strides: Array<PrimExpr>, slice_mode: TVMString, axes: Array<PrimExpr>) -> Expr;

    #[name("relay.op._make.expand_dims")]
    pub fn expand_dims(data: Expr, axis: i32, num_newaxis: i32) -> Expr;
}
