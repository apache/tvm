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
use crate::runtime::map::Map;
use crate::runtime::string::String as TVMString;
use crate::runtime::{external, IsObjectRef, Object, ObjectRef};

external! {
    #[name("relay.op._make.log")]
    pub fn log(data: Expr) -> Expr;

    #[name("relay.op._make.log10")]
    pub fn log10(data: Expr) -> Expr;

    #[name("relay.op._make.tan")]
    pub fn tan(data: Expr) -> Expr;

    #[name("relay.op._make.cos")]
    pub fn cos(data: Expr) -> Expr;

    #[name("relay.op._make.sin")]
    pub fn sin(data: Expr) -> Expr;

    #[name("relay.op._make.cosh")]
    pub fn cosh(data: Expr) -> Expr;

    #[name("relay.op._make.sinh")]
    pub fn sinh(data: Expr) -> Expr;

    #[name("relay.op._make.acos")]
    pub fn acos(data: Expr) -> Expr;

    #[name("relay.op._make.asin")]
    pub fn asin(data: Expr) -> Expr;

    #[name("relay.op._make.acosh")]
    pub fn acosh(data: Expr) -> Expr;

    #[name("relay.op._make.asinh")]
    pub fn asinh(data: Expr) -> Expr;

    #[name("relay.op._make.atan")]
    pub fn atan(data: Expr) -> Expr;

    #[name("relay.op._make.atanh")]
    pub fn atanh(data: Expr) -> Expr;

    #[name("relay.op._make.exp")]
    pub fn exp(data: Expr) -> Expr;

    #[name("relay.op._make.erf")]
    pub fn erf(data: Expr) -> Expr;

    #[name("relay.op._make.sqrt")]
    pub fn sqrt(data: Expr) -> Expr;

    #[name("relay.op._make.rsqrt")]
    pub fn rsqrt(data: Expr) -> Expr;

    #[name("relay.op._make.sigmoid")]
    pub fn sigmoid(data: Expr) -> Expr;

    #[name("relay.op._make.floor")]
    pub fn floor(data: Expr) -> Expr;

    #[name("relay.op._make.ceil")]
    pub fn ceil(data: Expr) -> Expr;

    #[name("relay.op._make.trunc")]
    pub fn trunc(data: Expr) -> Expr;

    #[name("relay.op._make.round")]
    pub fn round(data: Expr) -> Expr;

    #[name("relay.op._make.abs")]
    pub fn abs(data: Expr) -> Expr;

    #[name("relay.op._make.sign")]
    pub fn sign(data: Expr) -> Expr;

    #[name("relay.op._make.tanh")]
    pub fn tanh(data: Expr) -> Expr;

    #[name("relay.op._make.negative")]
    pub fn negative(data: Expr) -> Expr;

    #[name("relay.op._make.logical_not")]
    pub fn logical_not(data: Expr) -> Expr;

    #[name("relay.op._make.bitwise_not")]
    pub fn bitwise_not(lhs: Expr) -> Expr;

    #[name("relay.op._make.add")]
    pub fn add(lhs: Expr, rhs: Expr) -> Expr;

    #[name("relay.op._make.subtract")]
    pub fn subtract(lhs: Expr, rhs: Expr) -> Expr;

    #[name("relay.op._make.multiply")]
    pub fn multiply(lhs: Expr, rhs: Expr) -> Expr;

    #[name("relay.op._make.divide")]
    pub fn divide(lhs: Expr, rhs: Expr) -> Expr;

    #[name("relay.op._make.floor_divide")]
    pub fn floor_divide(lhs: Expr, rhs: Expr) -> Expr;

    #[name("relay.op._make.trunc_divide")]
    pub fn trunc_divide(lhs: Expr, rhs: Expr) -> Expr;

    #[name("relay.op._make.power")]
    pub fn power(lhs: Expr, rhs: Expr) -> Expr;

    #[name("relay.op._make.mod")]
    pub fn mod_(lhs: Expr, rhs: Expr) -> Expr;

    #[name("relay.op._make.floor_mod")]
    pub fn floor_mod(lhs: Expr, rhs: Expr) -> Expr;

    #[name("relay.op._make.trunc_mod")]
    pub fn trunc_mod(lhs: Expr, rhs: Expr) -> Expr;

    #[name("relay.op._make.logical_and")]
    pub fn logical_and(lhs: Expr, rhs: Expr) -> Expr;

    #[name("relay.op._make.logical_or")]
    pub fn logical_or(lhs: Expr, rhs: Expr) -> Expr;

    #[name("relay.op._make.logical_xor")]
    pub fn logical_xor(lhs: Expr, rhs: Expr) -> Expr;

    #[name("relay.op._make.bitwise_and")]
    pub fn bitwise_and(lhs: Expr, rhs: Expr) -> Expr;

    #[name("relay.op._make.bitwise_or")]
    pub fn bitwise_or(lhs: Expr, rhs: Expr) -> Expr;

    #[name("relay.op._make.bitwise_xor")]
    pub fn bitwise_xor(lhs: Expr, rhs: Expr) -> Expr;

    #[name("relay.op._make.equal")]
    pub fn equal(lhs: Expr, rhs: Expr) -> Expr;

    #[name("relay.op._make.not_equal")]
    pub fn not_equal(lhs: Expr, rhs: Expr) -> Expr;

    #[name("relay.op._make.less")]
    pub fn less(lhs: Expr, rhs: Expr) -> Expr;

    #[name("relay.op._make.less_equal")]
    pub fn less_equal(lhs: Expr, rhs: Expr) -> Expr;

    #[name("relay.op._make.greater")]
    pub fn greater(lhs: Expr, rhs: Expr) -> Expr;

    #[name("relay.op._make.greater_equal")]
    pub fn greater_equal(lhs: Expr, rhs: Expr) -> Expr;

    #[name("relay.op._make.maximum")]
    pub fn maximum(lhs: Expr, rhs: Expr) -> Expr;

    #[name("relay.op._make.minimum")]
    pub fn minimum(lhs: Expr, rhs: Expr) -> Expr;

    #[name("relay.op._make.right_shift")]
    pub fn right_shift(lhs: Expr, rhs: Expr) -> Expr;

    #[name("relay.op._make.left_shift")]
    pub fn left_shift(lhs: Expr, rhs: Expr) -> Expr;

    #[name("relay.op._make.zeros")]
    pub fn zeros(shape: Expr, dtype: DLDataType) -> Expr;

    #[name("relay.op._make.zeros_like")]
    pub fn zeros_like(data: Expr) -> Expr;

    #[name("relay.op._make.ones")]
    pub fn ones(shape: Expr, dtype: DLDataType) -> Expr;

    #[name("relay.op._make.ones_like")]
    pub fn ones_like(data: Expr) -> Expr;

   #[name("relay.op._make.clip")]
    pub fn clip(data: Expr, a_min: f64, a_max: f64) -> Expr;

    #[name("relay.op._make.fixed_point_multiply")]
    pub fn fixed_point_multiply(data: Expr, mulitplier: i32, shift: i32) -> Expr;

    #[name("relay.op._make.concatenate")]
    pub fn concatenate(data: Expr, axis: i32) -> Expr;

    #[name("relay.op._make.einsum")]
    pub fn einsum(inputs: Expr, equation: TVMString) -> Expr;

    #[name("relay.op._make.stack")]
    pub fn stack(data: Expr, axis: i32) -> Expr;

    #[name("relay.op._make.copy")]
    pub fn copy(data: Expr) -> Expr;

    //FIXME: VirtualDevice is not implemented in rust bindings
    //#[name("relay.op._make.DeviceCopy")]
    //pub fn device_copy(data: Expr, src_device: VirtualDevice, dst_device: VirtualDevice) -> Expr;

    #[name("relay.op._make.shape_of")]
    pub fn shape_of(data: Expr) -> Expr;

    #[name("relay.op._make.ndarray_size")]
    pub fn ndarray_size(data: Expr) -> Expr;

    #[name("relay.op._make.isnan")]
    pub fn isnan(data: Expr) -> Expr;

    #[name("relay.op._make.isfinite")]
    pub fn isfinite(data: Expr) -> Expr;

    #[name("relay.op._make.reshape")]
    pub fn reshape(data: Expr, newshape: Array<PrimExpr>, allowzero: bool) -> Expr;

    #[name("relay.op._make.transpose")]
    pub fn transpose(data: Expr, axes: Array<PrimExpr>) -> Expr;

    #[name("relay.ir.cast")]
    pub fn cast(data: Expr, dtype: DLDataType) -> Expr;
}
