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

use crate::function::ffi::DLDataType;
use crate::ir::relay::Expr;
use crate::ir::tir::IntImm;
use crate::ir::PrimExpr;
use crate::runtime::array::Array;
use crate::runtime::function::Result;
use crate::runtime::map::Map;
use crate::runtime::string::String as TVMString;
use crate::runtime::{external, IsObjectRef, Object, ObjectRef};

external! {
    #[name("relay.op._make.argmax")]
    pub fn argmax(data: Expr, axis: Array<IntImm>, keepdims: bool, exclude : bool, select_last_index: bool) -> Expr;

    #[name("relay.op._make.argmin")]
    pub fn argmin(data: Expr, axis: Array<IntImm>, keepdims: bool, exclude : bool, select_last_index: bool) -> Expr;

    #[name("relay.op._make.sum")]
    pub fn sum(data: Expr, axis: Array<IntImm>, keepdims: bool, exclude : bool) -> Expr;

    #[name("relay.op._make.all")]
    pub fn all(data: Expr, axis: Array<IntImm>, keepdims: bool, exclude : bool) -> Expr;

    #[name("relay.op._make.any")]
    pub fn any(data: Expr, axis: Array<IntImm>, keepdims: bool, exclude : bool) -> Expr;

    #[name("relay.op._make.max")]
    pub fn max(data: Expr, axis: Array<IntImm>, keepdims: bool, exclude : bool) -> Expr;

    #[name("relay.op._make.min")]
    pub fn min(data: Expr, axis: Array<IntImm>, keepdims: bool, exclude : bool) -> Expr;

    #[name("relay.op._make.mean")]
    pub fn mean(data: Expr, axis: Array<IntImm>, keepdims: bool, exclude : bool) -> Expr;

    #[name("relay.op._make.variance")]
    pub fn variance(data: Expr, axis: Array<IntImm>, keepdims: bool, exclude : bool, unbiased: bool, with_mean: bool) -> Expr;

    #[name("relay.op._make.std")]
    pub fn std(data: Expr, axis: Array<IntImm>, keepdims: bool, exclude : bool, unbiased: bool) -> Expr;

    #[name("relay.op._make.mean_variance")]
    pub fn mean_variance(data: Expr, axis: Array<IntImm>, keepdims: bool, exclude : bool, unbiased: bool) -> Expr;

    #[name("relay.op._make.mean_std")]
    pub fn mean_std(data: Expr, axis: Array<IntImm>, keepdims: bool, exclude : bool) -> Expr;

    #[name("relay.op._make.prod")]
    pub fn prod(data: Expr, axis: Array<IntImm>, keepdims: bool) -> Expr;

    #[name("relay.op._make.logsumexp")]
    pub fn logsumexp(data: Expr, axis: Array<IntImm>, keepdims: bool) -> Expr;
}
