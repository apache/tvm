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
#[derive(Object, Debug)]
#[ref_name = "PadAttrs"]
#[type_key = "relay.attrs.PadAttrs"]
pub struct PadAttrsNode {
    pub base: BaseAttrsNode,
    pub pad_width: Array<Array<IndexExpr>>,
    pub pad_mode: TString,
}

#[repr(C)]
#[derive(Object, Debug)]
#[ref_name = "Conv1DAttrs"]
#[type_key = "relay.attrs.Conv1DAttrs"]
pub struct Conv1DAttrsNode {
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
#[derive(Object, Debug)]
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
    pub auto_scheduler_rewritten_layout: TString,
    pub out_dtype: DataType,
}

#[repr(C)]
#[derive(Object, Debug)]
#[ref_name = "Conv3DAttrs"]
#[type_key = "relay.attrs.Conv3DAttrs"]
pub struct Conv3DAttrsNode {
    pub base: BaseAttrsNode,
    pub strides: Array<IndexExpr>,
    pub padding: Array<IndexExpr>,
    pub dilation: Array<IndexExpr>,
    pub groups: i32,
    pub channels: IndexExpr,
    pub kernel_size: Array<IndexExpr>,
    pub data_layout: TString,
    pub kernel_layout: TString,
    pub out_layout: TString,
    pub auto_scheduler_rewritten_layout: TString,
    pub out_dtype: DataType,
}

#[repr(C)]
#[derive(Object, Debug)]
#[ref_name = "Conv3DTransposeAttrs"]
#[type_key = "relay.attrs.Conv3DTransposeAttrs"]
pub struct Conv3DTransposeAttrsNode {
    pub base: BaseAttrsNode,
    pub channels: IndexExpr,
    pub kernel_size: Array<IndexExpr>,
    pub strides: Array<IndexExpr>,
    pub padding: Array<IndexExpr>,
    pub output_padding: Array<IndexExpr>,
    pub dilation: Array<IndexExpr>,
    pub groups: i32,
    pub data_layout: TString,
    pub kernel_layout: TString,
    pub out_layout: TString,
    pub out_dtype: DataType,
}

#[repr(C)]
#[derive(Object, Debug)]
#[ref_name = "BiasAddAttrs"]
#[type_key = "relay.attrs.BiasAddAttrs"]
pub struct BiasAddAttrsNode {
    pub base: BaseAttrsNode,
    pub axis: i32,
}

#[repr(C)]
#[derive(Object, Debug)]
#[ref_name = "MatmulAttrs"]
#[type_key = "relay.attrs.MatmulAttrs"]
pub struct MatmulAttrsNode {
    pub base: BaseAttrsNode,
    pub units: IndexExpr,
    pub out_dtype: DataType,
    pub transpose_a: bool,
    pub transpose_b: bool,
}

#[repr(C)]
#[derive(Object, Debug)]
#[ref_name = "DenseAttrs"]
#[type_key = "relay.attrs.DenseAttrs"]
pub struct DenseAttrsNode {
    pub base: BaseAttrsNode,
    pub units: IndexExpr,
    pub auto_scheduler_rewritten_layout: TString,
    pub out_dtype: DataType,
}

#[repr(C)]
#[derive(Object, Debug)]
#[ref_name = "GlobalPool2DAttrs"]
#[type_key = "relay.attrs.GlobalPool2DAttrs"]
pub struct GlobalPool2DAttrsNode {
    pub base: BaseAttrsNode,
    pub layout: TString,
}

#[repr(C)]
#[derive(Object, Debug)]
#[ref_name = "MaxPool2DAttrs"]
#[type_key = "relay.attrs.MaxPool2DAttrs"]
pub struct MaxPool2DAttrsNode {
    pub base: BaseAttrsNode,
    pub pool_size: Array<IndexExpr>,
    pub strides: Array<IndexExpr>,
    pub padding: Array<IndexExpr>,
    pub dilation: Array<IndexExpr>,
    pub layout: TString,
    pub ceil_mode: bool,
}

#[repr(C)]
#[derive(Object, Debug)]
#[ref_name = "SoftmaxAttrs"]
#[type_key = "relay.attrs.SoftmaxAttrs"]
pub struct SoftmaxAttrsNode {
    pub base: BaseAttrsNode,
    pub axis: i32,
}

#[repr(C)]
#[derive(Object, Debug)]
#[ref_name = "BatchNormAttrs"]
#[type_key = "relay.attrs.BatchNormAttrs"]
pub struct BatchNormAttrsNode {
    pub base: BaseAttrsNode,
    pub axis: i32,
    pub epsilon: f64,
    pub center: bool,
    pub scale: bool,
}

#[repr(C)]
#[derive(Object, Debug)]
#[ref_name = "LeakyReluAttrs"]
#[type_key = "relay.attrs.LeakyReluAttrs"]
pub struct LeakyReluAttrsNode {
    pub base: BaseAttrsNode,
    pub alpha: f64,
}

#[repr(C)]
#[derive(Object, Debug)]
#[ref_name = "AvgPool2DAttrs"]
#[type_key = "relay.attrs.AvgPool2DAttrs"]
pub struct AvgPool2DAttrsNode {
    pub base: BaseAttrsNode,
    pub pool_size: Array<IndexExpr>,
    pub strides: Array<IndexExpr>,
    pub padding: Array<IndexExpr>,
    pub dilation: Array<IndexExpr>,
    pub layout: TString,
    pub ceil_mode: bool,
    pub count_include_pad: bool,
}

#[repr(C)]
#[derive(Object, Debug)]
#[ref_name = "UpSamplingAttrs"]
#[type_key = "relay.attrs.UpSamplingAttrs"]
pub struct UpSamplingAttrsNode {
    pub base: BaseAttrsNode,
    pub scale_h: f64,
    pub scale_w: f64,
    pub layout: TString,
    pub method: TString,
    pub align_corners: bool,
}

#[repr(C)]
#[derive(Object, Debug)]
#[ref_name = "DropoutAttrs"]
#[type_key = "relay.attrs.DropoutAttrs"]
pub struct DropoutAttrsNode {
    pub base: BaseAttrsNode,
    pub rate: f64,
}

#[repr(C)]
#[derive(Object, Debug)]
#[ref_name = "BatchMatmulAttrs"]
#[type_key = "relay.attrs.BatchMatmulAttrs"]
pub struct BatchMatmulAttrsNode {
    pub base: BaseAttrsNode,
    pub auto_scheduler_rewritten_layout: TString,
    pub out_dtype: DataType,
}

#[repr(C)]
#[derive(Object, Debug)]
#[ref_name = "LayerNormAttrs"]
#[type_key = "relay.attrs.LayerNormAttrs"]
pub struct LayerNormAttrsNode {
    pub base: BaseAttrsNode,
    pub axis: i32,
    pub epsilon: f64,
    pub center: bool,
    pub scale: bool,
}
