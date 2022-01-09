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

use crate::ir::attrs::BaseAttrsNode;
use crate::ir::PrimExpr;
use crate::runtime::array::Array;
use tvm_macros::Object;

type IndexExpr = PrimExpr;

#[repr(C)]
#[derive(Object, Debug)]
#[ref_name = "ReduceAttrs"]
#[type_key = "relay.attrs.ReduceAttrs"]
pub struct ReduceAttrsNode {
    pub base: BaseAttrsNode,
    pub axis: Array<IndexExpr>,
    pub keepdims: bool,
    pub exclude: bool,
}

#[repr(C)]
#[derive(Object, Debug)]
#[ref_name = "VarianceAttrs"]
#[type_key = "relay.attrs.ReduceAttrs"]
pub struct VarianceAttrsNode {
    pub base: BaseAttrsNode,
    pub axis: Array<IndexExpr>,
    pub keepdims: bool,
    pub exclude: bool,
    pub unbiased: bool,
}
