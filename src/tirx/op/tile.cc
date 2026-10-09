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

/*!
 * \file tirx/op/tile.cc
 * \brief TIRx tile operations.
 */
#include <tvm/tirx/op/tile.h>
#include <tvm/tirx/op_attr_types.h>

namespace tvm {
namespace tirx {
namespace tile {

const Op& zero_op() {
  static const Op op = Op::Get("tirx.tile.zero");
  return op;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.tile.zero")
      .signature(sig::arg("dst", "The destination."), sig::arg("src", "The source."))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.tile.zero"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("tile_primitive"));
}

const Op& sqrt_op() {
  static const Op op = Op::Get("tirx.tile.sqrt");
  return op;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.tile.sqrt")
      .signature(sig::arg("dst", "The destination."), sig::arg("src", "The source."))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.tile.sqrt"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("tile_primitive"));
}

const Op& sqrt_with_scale_bias_op() {
  static const Op op = Op::Get("tirx.tile.sqrt_with_scale_bias");
  return op;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.tile.sqrt_with_scale_bias")
      .signature(sig::arg("dst", "The destination."), sig::arg("src", "The source."),
                 sig::arg("scale", "The scale factor."), sig::arg("bias", "The bias."))
      .set_attr<TScriptPrinterName>("TScriptPrinterName",
                                    ffi::String("tirx.tile.sqrt_with_scale_bias"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("tile_primitive"));
}

const Op& exp_op() {
  static const Op op = Op::Get("tirx.tile.exp");
  return op;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.tile.exp")
      .signature(sig::arg("dst", "The destination."), sig::arg("src", "The source."))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.tile.exp"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("tile_primitive"));
}

const Op& exp_with_scale_bias_op() {
  static const Op op = Op::Get("tirx.tile.exp_with_scale_bias");
  return op;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.tile.exp_with_scale_bias")
      .signature(sig::arg("dst", "The destination."), sig::arg("src", "The source."),
                 sig::arg("scale", "The scale factor."), sig::arg("bias", "The bias."))
      .set_attr<TScriptPrinterName>("TScriptPrinterName",
                                    ffi::String("tirx.tile.exp_with_scale_bias"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("tile_primitive"));
}

const Op& exp2_op() {
  static const Op op = Op::Get("tirx.tile.exp2");
  return op;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.tile.exp2")
      .signature(sig::arg("dst", "The destination."), sig::arg("src", "The source."))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.tile.exp2"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("tile_primitive"));
}

const Op& exp2_with_scale_bias_op() {
  static const Op op = Op::Get("tirx.tile.exp2_with_scale_bias");
  return op;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.tile.exp2_with_scale_bias")
      .signature(sig::arg("dst", "The destination."), sig::arg("src", "The source."),
                 sig::arg("scale", "The scale factor."), sig::arg("bias", "The bias."))
      .set_attr<TScriptPrinterName>("TScriptPrinterName",
                                    ffi::String("tirx.tile.exp2_with_scale_bias"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("tile_primitive"));
}

const Op& log2_op() {
  static const Op op = Op::Get("tirx.tile.log2");
  return op;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.tile.log2")
      .signature(sig::arg("dst", "The destination."), sig::arg("src", "The source."))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.tile.log2"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("tile_primitive"));
}

const Op& log2_with_scale_bias_op() {
  static const Op op = Op::Get("tirx.tile.log2_with_scale_bias");
  return op;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.tile.log2_with_scale_bias")
      .signature(sig::arg("dst", "The destination."), sig::arg("src", "The source."),
                 sig::arg("scale", "The scale factor."), sig::arg("bias", "The bias."))
      .set_attr<TScriptPrinterName>("TScriptPrinterName",
                                    ffi::String("tirx.tile.log2_with_scale_bias"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("tile_primitive"));
}

const Op& add_op() {
  static const Op op = Op::Get("tirx.tile.add");
  return op;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.tile.add")
      .signature(sig::arg("dst", "The destination."), sig::arg("src1", "The first source."),
                 sig::arg("src2", "The second source."))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.tile.add"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("tile_primitive"));
}

const Op& sub_op() {
  static const Op op = Op::Get("tirx.tile.sub");
  return op;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.tile.sub")
      .signature(sig::arg("dst", "The destination."), sig::arg("src1", "The first source."),
                 sig::arg("src2", "The second source."))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.tile.sub"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("tile_primitive"));
}

const Op& mul_op() {
  static const Op op = Op::Get("tirx.tile.mul");
  return op;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.tile.mul")
      .signature(sig::arg("dst", "The destination."), sig::arg("src1", "The first source."),
                 sig::arg("src2", "The second source."))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.tile.mul"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("tile_primitive"));
}

const Op& fdiv_op() {
  static const Op op = Op::Get("tirx.tile.fdiv");
  return op;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.tile.fdiv")
      .signature(sig::arg("dst", "The destination."), sig::arg("src1", "The first source."),
                 sig::arg("src2", "The second source."))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.tile.fdiv"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("tile_primitive"));
}

const Op& minimum_op() {
  static const Op op = Op::Get("tirx.tile.minimum");
  return op;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.tile.minimum")
      .signature(sig::arg("dst", "The destination."), sig::arg("src1", "The first source."),
                 sig::arg("src2", "The second source."))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.tile.minimum"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("tile_primitive"));
}

const Op& maximum_op() {
  static const Op op = Op::Get("tirx.tile.maximum");
  return op;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.tile.maximum")
      .signature(sig::arg("dst", "The destination."), sig::arg("src1", "The first source."),
                 sig::arg("src2", "The second source."))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.tile.maximum"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("tile_primitive"));
}

const Op& copy_op() {
  static const Op op = Op::Get("tirx.tile.copy");
  return op;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.tile.copy")
      .signature(sig::arg("dst", "The destination."), sig::arg("src", "The source."))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.tile.copy"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("tile_primitive"));
}

const Op& fill_op() {
  static const Op op = Op::Get("tirx.tile.fill");
  return op;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.tile.fill")
      .signature(sig::arg("dst", "The destination."), sig::arg("value", "The value to use."))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.tile.fill"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("tile_primitive"));
}

const Op& gemm_op() {
  static const Op op = Op::Get("tirx.tile.gemm");
  return op;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.tile.gemm")
      .signature(
          sig::arg("D", "The D tile."), sig::arg("A", "The A tile."), sig::arg("B", "The B tile."),
          sig::arg("C", "The C tile."), sig::arg("transpose_A", "Whether to transpose A."),
          sig::arg("transpose_B", "Whether to transpose B."),
          sig::arg("alpha", "The alpha scale factor."), sig::arg("beta", "The beta scale factor."))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.tile.gemm"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("tile_primitive"));
}

const Op& reciprocal_op() {
  static const Op op = Op::Get("tirx.tile.reciprocal");
  return op;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.tile.reciprocal")
      .signature(sig::arg("dst", "The destination."), sig::arg("src", "The source."))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.tile.reciprocal"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("tile_primitive"));
}

const Op& sum_op() {
  static const Op op = Op::Get("tirx.tile.sum");
  return op;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.tile.sum")
      .signature(sig::arg("dst", "The destination."), sig::arg("src", "The source."),
                 sig::arg("axes", "The axes."), sig::arg("accum", "The accumulator."))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.tile.sum"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("tile_primitive"));
}

const Op& max_op() {
  static const Op op = Op::Get("tirx.tile.max");
  return op;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.tile.max")
      .signature(sig::arg("dst", "The destination."), sig::arg("src", "The source."),
                 sig::arg("axes", "The axes."), sig::arg("accum", "The accumulator."))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.tile.max"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("tile_primitive"));
}

const Op& min_op() {
  static const Op op = Op::Get("tirx.tile.min");
  return op;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.tile.min")
      .signature(sig::arg("dst", "The destination."), sig::arg("src", "The source."),
                 sig::arg("axes", "The axes."), sig::arg("accum", "The accumulator."))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.tile.min"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("tile_primitive"));
}

const Op& memset_op() {
  static const Op op = Op::Get("tirx.tile.memset");
  return op;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.tile.memset")
      .signature(sig::arg("dst", "The destination."), sig::arg("value", "The value to use."))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.tile.memset"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("tile_primitive"));
}

const Op& reduce_negate_op() {
  static const Op op = Op::Get("tirx.tile.reduce_negate");
  return op;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.tile.reduce_negate")
      .signature(sig::arg("output", "The output."), sig::arg("input", "The input."),
                 sig::arg("reduce_op", "The reduction operation."),
                 sig::arg("reduce_axes", "The reduction axes."),
                 sig::arg("accum", "The accumulator."))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.tile.reduce_negate"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("tile_primitive"));
}

const Op& binary_reduce_op() {
  static const Op op = Op::Get("tirx.tile.binary_reduce");
  return op;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.tile.binary_reduce")
      .signature(sig::arg("binary_output", "The binary operation output."),
                 sig::arg("reduce_output", "The reduction output."),
                 sig::arg("binary_input1", "The first binary input."),
                 sig::arg("binary_input2", "The second binary input."),
                 sig::arg("binary_op", "The binary operation."),
                 sig::arg("reduce_op", "The reduction operation."),
                 sig::arg("reduce_axes", "The reduction axes."))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.tile.binary_reduce"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("tile_primitive"));
}

const Op& unary_reduce_op() {
  static const Op op = Op::Get("tirx.tile.unary_reduce");
  return op;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.tile.unary_reduce")
      .signature(sig::arg("unary_output", "The unary operation output."),
                 sig::arg("reduce_output", "The reduction output."),
                 sig::arg("unary_input", "The unary operation input."),
                 sig::arg("unary_op", "The unary operation."),
                 sig::arg("reduce_op", "The reduction operation."),
                 sig::arg("reduce_axes", "The reduction axes."))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.tile.unary_reduce"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("tile_primitive"));
}

const Op& unary_reduce_with_scale_bias_op() {
  static const Op op = Op::Get("tirx.tile.unary_reduce_with_scale_bias");
  return op;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.tile.unary_reduce_with_scale_bias")
      .signature(sig::arg("unary_output", "The unary operation output."),
                 sig::arg("reduce_output", "The reduction output."),
                 sig::arg("unary_input", "The unary operation input."),
                 sig::arg("unary_op", "The unary operation."),
                 sig::arg("reduce_op", "The reduction operation."),
                 sig::arg("scale", "The scale factor."), sig::arg("bias", "The bias."),
                 sig::arg("reduce_axes", "The reduction axes."))
      .set_attr<TScriptPrinterName>("TScriptPrinterName",
                                    ffi::String("tirx.tile.unary_reduce_with_scale_bias"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("tile_primitive"));
}

const Op& binary_chain_op() {
  static const Op op = Op::Get("tirx.tile.binary_chain");
  return op;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.tile.binary_chain")
      .signature(sig::arg("output", "The output."), sig::arg("data", "The input data."),
                 sig::arg("operand0", "The first operand."),
                 sig::arg("operand1", "The second operand."),
                 sig::arg("op0", "The first operation."), sig::arg("op1", "The second operation."),
                 sig::arg("reverse1", "Whether to reverse the second operation."))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.tile.binary_chain"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("tile_primitive"));
}

const Op& select_op() {
  static const Op op = Op::Get("tirx.tile.select");
  return op;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.tile.select")
      .signature(sig::arg("dst", "The destination."),
                 sig::arg("true_value", "The value when the condition is true."),
                 sig::arg("false_value", "The value when the condition is false."),
                 sig::arg("pred", "The predicate."))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.tile.select"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("tile_primitive"));
}

const Op& cast_op() {
  static const Op op = Op::Get("tirx.tile.cast");
  return op;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.tile.cast")
      .signature(sig::arg("dst", "The destination."), sig::arg("src", "The source."))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.tile.cast"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("tile_primitive"));
}

const Op& fma_op() {
  static const Op op = Op::Get("tirx.tile.fma");
  return op;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.tile.fma")
      .signature(sig::arg("dst", "The destination."), sig::arg("src", "The source."),
                 sig::arg("scale", "The scale factor."), sig::arg("bias", "The bias."))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.tile.fma"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("tile_primitive"));
}

const Op& silu_op() {
  static const Op op = Op::Get("tirx.tile.silu");
  return op;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.tile.silu")
      .signature(sig::arg("dst", "The destination."), sig::arg("src", "The source."))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.tile.silu"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("tile_primitive"));
}

const Op& permute_layout_op() {
  static const Op op = Op::Get("tirx.tile.permute_layout");
  return op;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.tile.permute_layout")
      .signature(sig::arg("dst", "The destination."), sig::arg("src", "The source."))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.tile.permute_layout"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("tile_primitive"));
}

const Op& copy_async_op() {
  static const Op op = Op::Get("tirx.tile.copy_async");
  return op;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.tile.copy_async")
      .signature(sig::arg("dst", "The destination."), sig::arg("src", "The source."))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.tile.copy_async"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("tile_primitive"));
}

const Op& gemm_async_op() {
  static const Op op = Op::Get("tirx.tile.gemm_async");
  return op;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.tile.gemm_async")
      .signature(sig::arg("C", "The C tile."), sig::arg("A", "The A tile."),
                 sig::arg("B", "The B tile."), sig::var_args("args"))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.tile.gemm_async"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("tile_primitive"));
}

}  // namespace tile
}  // namespace tirx
}  // namespace tvm
