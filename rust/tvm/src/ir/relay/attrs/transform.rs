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
use crate::runtime::ObjectRef;
use tvm_macros::Object;

type IndexExpr = PrimExpr;

#[repr(C)]
#[derive(Object, Debug)]
#[ref_name = "ExpandDimsAttrs"]
#[type_key = "relay.attrs.ExpandDimsAttrs"]
pub struct ExpandDimsAttrsNode {
    pub base: BaseAttrsNode,
    pub axis: i32,
    pub num_newaxis: i32,
}

#[repr(C)]
#[derive(Object, Debug)]
#[ref_name = "ConcatenateAttrs"]
#[type_key = "relay.attrs.ConcatenateAttrs"]
pub struct ConcatenateAttrsNode {
    pub base: BaseAttrsNode,
    pub axis: i32,
}

#[repr(C)]
#[derive(Object, Debug)]
#[ref_name = "ReshapeAttrs"]
#[type_key = "relay.attrs.ReshapeAttrs"]
pub struct ReshapeAttrsNode {
    pub base: BaseAttrsNode,
    pub newshape: Array<IndexExpr>,
    pub reverse: bool,
}

#[repr(C)]
#[derive(Object, Debug)]
#[ref_name = "SplitAttrs"]
#[type_key = "relay.attrs.SplitAttrs"]
pub struct SplitAttrsNode {
    pub base: BaseAttrsNode,
    pub indices_or_sections: ObjectRef,
    pub axis: i32,
}

#[repr(C)]
#[derive(Object, Debug)]
#[ref_name = "TransposeAttrs"]
#[type_key = "relay.attrs.TransposeAttrs"]
pub struct TransposeAttrsNode {
    pub base: BaseAttrsNode,
    pub axes: Array<IndexExpr>,
}

#[repr(C)]
#[derive(Object, Debug)]
#[ref_name = "SqueezeAttrs"]
#[type_key = "relay.attrs.SqueezeAttrs"]
pub struct SqueezeAttrsNode {
    pub base: BaseAttrsNode,
    pub axis: Array<IndexExpr>,
}
