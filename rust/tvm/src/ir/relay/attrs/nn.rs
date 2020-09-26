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
use crate::runtime::DataType;
use crate::runtime::String as TString;
use tvm_macros::Object;

type IndexExpr = PrimExpr;

#[repr(C)]
#[derive(Object)]
#[ref_name = "Conv2DAttrs"]
#[type_key = "relay.attrs.Conv2DAttrs"]
pub struct Conv2DAttrsNode {
    pub base: BaseAttrsNode,
    pub strides: Array<IndexExpr>,
    pub padding: Array<IndexExpr>,
    pub dilation: Array<IndexExpr>,
    // TODO(@gussmith23) groups is "int", what should it be here?
    pub groups: i32,
    pub channels: IndexExpr,
    pub kernel_size: Array<IndexExpr>,
    pub data_layout: TString,
    pub kernel_layout: TString,
    pub out_layout: TString,
    pub out_dtype: DataType,
}

#[repr(C)]
#[derive(Object)]
#[ref_name = "BiasAddAttrs"]
#[type_key = "relay.attrs.BiasAddAttrs"]
pub struct BiasAddAttrsNode {
    pub base: BaseAttrsNode,
    pub axis: i32,
}

#[repr(C)]
#[derive(Object)]
#[ref_name = "DenseAttrs"]
#[type_key = "relay.attrs.DenseAttrs"]
pub struct DenseAttrsNode {
    pub base: BaseAttrsNode,
    pub units: IndexExpr,
    pub out_dtype: DataType,
}

#[repr(C)]
#[derive(Object)]
#[ref_name = "GlobalPool2DAttrs"]
#[type_key = "relay.attrs.GlobalPool2DAttrs"]
pub struct GlobalPool2DAttrsNode {
    pub base: BaseAttrsNode,
    pub layout: TString,
}

#[repr(C)]
#[derive(Object)]
#[ref_name = "MaxPool2DAttrs"]
#[type_key = "relay.attrs.MaxPool2DAttrs"]
pub struct MaxPool2DAttrsNode {
    pub base: BaseAttrsNode,
    pub pool_size: Array<IndexExpr>,
    pub strides: Array<IndexExpr>,
    pub padding: Array<IndexExpr>,
    pub layout: TString,
    pub ceil_mode: bool,
}

#[repr(C)]
#[derive(Object)]
#[ref_name = "SoftmaxAttrs"]
#[type_key = "relay.attrs.SoftmaxAttrs"]
pub struct SoftmaxAttrsNode {
    pub base: BaseAttrsNode,
    pub axis: i32,
}
