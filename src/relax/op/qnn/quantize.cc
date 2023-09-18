/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file src/relax/op/qnn/quantize.cc
 * \brief implements QNN quantize operator.
 */

#include "../../transform/utils.h"
#include "../op_common.h"
#include "qnn.h"

namespace tvm {
namespace relax {

TVM_REGISTER_NODE_TYPE(QuantizeAttrs);

/* relax.qnn.quantize */

Expr quantize(Expr data, Expr scale, Expr zero_point, int axis, DataType out_dtype) {
  ObjectPtr<QuantizeAttrs> attrs = make_object<QuantizeAttrs>();
  attrs->axis = axis;
  attrs->out_dtype = out_dtype;
  static const Op& op = Op::Get("relax.qnn.quantize");
  return Call(op, {std::move(data), std::move(scale), std::move(zero_point)}, Attrs(attrs));
}

TVM_REGISTER_GLOBAL("relax.op.qnn.quantize").set_body_typed(quantize);

StructInfo InferStructInfoQnnQuantize(const Call& call, const BlockBuilder& ctx) {
  const auto* attrs = call->attrs.as<QuantizeAttrs>();
  if (attrs->out_dtype != DataType::Int(8) && attrs->out_dtype != DataType::UInt(8) &&
      attrs->out_dtype != DataType::Int(16) && attrs->out_dtype != DataType::UInt(16)) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "Unsupported output datatype attribute for operation: '"
                     << attrs->out_dtype);
  }

  TensorStructInfo input_sinfo = GetInputTensorStructInfo(call, ctx)[0];
  TensorStructInfo scale_sinfo = GetInputTensorStructInfo(call, ctx)[1];
  TensorStructInfo zp_sinfo = GetInputTensorStructInfo(call, ctx)[2];

  // Check input datatype:
  if (input_sinfo->dtype != DataType::Float(16) && input_sinfo->dtype != DataType::Float(32)) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "Unsupported input datatype for operation: " << input_sinfo->dtype);
  }

  // Check datatype of scale param:
  if (scale_sinfo->dtype != DataType::Float(32) && scale_sinfo->dtype != DataType::Float(16)) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "scale param datatype should be one of [float16, float32], but got "
                     << scale_sinfo->dtype);
  }

  // Check datatype of zero_point param:
  if (zp_sinfo->dtype != DataType::Int(32)) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "zero_point param datatype should be int32, but got" << scale_sinfo->dtype);
  }

  // Check that "axis" attribute is not out of range:
  int axis = (attrs->axis < 0) ? (input_sinfo->ndim + attrs->axis) : attrs->axis;
  if (axis < 0 || axis > input_sinfo->ndim - 1) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "relax.qnn.quantize: axis param is out of range (" << attrs->axis << ")");
  }

  auto output_sinfo = make_object<TensorStructInfoNode>(*input_sinfo.get());
  output_sinfo->dtype = attrs->out_dtype;
  return TensorStructInfo(output_sinfo);
}

/*!
 * Lower relax.qnn.quantize into the sequence of simple operations.
 * Quantization formula is defined as: out = clip(round(input / scale) + zp, min_val, max_val)
 */
Expr LowerQuantize(const Call& call) {
  const QuantizeAttrs* attrs = call->attrs.as<QuantizeAttrs>();
  Expr data = call->args[0];
  Expr scale = call->args[1];
  Expr zero_point = call->args[2];
  const TensorStructInfoNode* tinfo = GetStructInfoAs<TensorStructInfoNode>(data);
  PrimValue min_value(cast(tinfo->dtype, tvm::min_value(attrs->out_dtype)));
  PrimValue max_value(cast(tinfo->dtype, tvm::max_value(attrs->out_dtype)));

  if (!IsScalarTensor(scale)) {
    scale = ExpandToMatchInput(scale, tinfo->ndim, {attrs->axis});
  }

  if (!IsScalarTensor(zero_point)) {
    zero_point = ExpandToMatchInput(zero_point, tinfo->ndim, {attrs->axis});
  }

  Expr scaled_data = round(divide(data, scale));
  Expr add_zp = add(scaled_data, astype(zero_point, tinfo->dtype));
  return astype(clip(add_zp, min_value, max_value), attrs->out_dtype);
}

TVM_REGISTER_OP("relax.qnn.quantize")
    .set_attrs_type<QuantizeAttrs>()
    .set_num_inputs(3)
    .add_argument("data", "Tensor", "The input tensor.")
    .add_argument("scale", "Tensor", "The quantization scale of the output tensor.")
    .add_argument("zero_point", "Tensor", "The quantization zero_point of the output tensor.")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoQnnQuantize)
    .set_attr<FQnnLower>("FQnnLower", LowerQuantize)
    .set_attr<Bool>("FPurity", Bool(true));

/* relax.qnn.dequantize */

Expr dequantize(Expr data, Expr scale, Expr zero_point, int axis, DataType out_dtype) {
  ObjectPtr<QuantizeAttrs> attrs = make_object<QuantizeAttrs>();
  attrs->axis = axis;
  attrs->out_dtype = out_dtype;
  static const Op& op = Op::Get("relax.qnn.dequantize");
  return Call(op, {std::move(data), std::move(scale), std::move(zero_point)}, Attrs(attrs));
}

TVM_REGISTER_GLOBAL("relax.op.qnn.dequantize").set_body_typed(dequantize);

StructInfo InferStructInfoQnnDequantize(const Call& call, const BlockBuilder& ctx) {
  const auto* attrs = call->attrs.as<QuantizeAttrs>();
  if (attrs->out_dtype != DataType::Float(16) && attrs->out_dtype != DataType::Float(32)) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "Unsupported output datatype attribute for operation: "
                     << attrs->out_dtype);
  }

  TensorStructInfo input_sinfo = GetInputTensorStructInfo(call, ctx)[0];
  TensorStructInfo scale_sinfo = GetInputTensorStructInfo(call, ctx)[1];
  TensorStructInfo zp_sinfo = GetInputTensorStructInfo(call, ctx)[2];

  // Check input datatype:
  if (input_sinfo->dtype != DataType::Int(8) && input_sinfo->dtype != DataType::UInt(8) &&
      input_sinfo->dtype != DataType::Int(16) && input_sinfo->dtype != DataType::UInt(16) &&
      input_sinfo->dtype != DataType::Int(32)) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "Unsupported input datatype for operation: " << attrs->out_dtype);
  }

  // Check datatype of scale param:
  if (scale_sinfo->dtype != DataType::Float(32) && scale_sinfo->dtype != DataType::Float(16)) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "scale param datatype should be one of [float16, float32], but got "
                     << scale_sinfo->dtype);
  }

  // Check datatype of zero_point param:
  if (zp_sinfo->dtype != DataType::Int(32)) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "zero_point param datatype should be int32, but got" << scale_sinfo->dtype);
  }

  // Check that "axis" attribute is not out of range:
  int axis = (attrs->axis < 0) ? (input_sinfo->ndim + attrs->axis) : attrs->axis;
  if (axis < 0 || axis > input_sinfo->ndim - 1) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "relax.qnn.dequantize: axis param is out of range (" << attrs->axis << ")");
  }

  auto output_sinfo = make_object<TensorStructInfoNode>(*input_sinfo.get());
  output_sinfo->dtype = attrs->out_dtype;
  return TensorStructInfo(output_sinfo);
}

/*!
 * Lower relax.qnn.dequantize into the sequence of simple operations.
 * Dequantization formula is defined as: out = scale * (input - zp)
 * Compute datatype: float32
 * Example of lowering:
 *   qnn.dequantize(data, scale, zp, "float32") -->
 *       sub = subtract(cast(data, "int32"), zp)
 *       out = multiply(cast(sub, "float32"), scale)
 *
 *   qnn.dequantize(data, scale, zp, "float16") -->
 *       sub = subtract(cast(data, "int32"), zp)
 *       mul = multiply(cast(sub, "float32"), cast(scale, "float32"))
 *       clipped_out = clip(mul, float32(-65504.0), float32(65504.0))
 *       out = cast(clipped_out, "float16")
 */
Expr LowerDequantize(const Call& call) {
  const QuantizeAttrs* attrs = call->attrs.as<QuantizeAttrs>();
  Expr data = call->args[0];
  Expr scale = call->args[1];
  Expr zero_point = call->args[2];
  const TensorStructInfoNode* tinfo = GetStructInfoAs<TensorStructInfoNode>(data);
  const TensorStructInfoNode* scale_tinfo = GetStructInfoAs<TensorStructInfoNode>(scale);

  if (!IsScalarTensor(scale)) {
    scale = ExpandToMatchInput(scale, tinfo->ndim, {attrs->axis});
  }

  if (!IsScalarTensor(zero_point)) {
    zero_point = ExpandToMatchInput(zero_point, tinfo->ndim, {attrs->axis});
  }

  if (tinfo->dtype != DataType::Int(32)) {
    data = astype(data, DataType::Int(32));
  }
  Expr sub_zp = subtract(data, zero_point);
  if (scale_tinfo->dtype != DataType::Float(32)) {
    scale = astype(scale, DataType::Float(32));
  }
  Expr scaled_output = multiply(astype(sub_zp, DataType::Float(32)), scale);
  if (attrs->out_dtype == DataType::Float(32)) {
    return scaled_output;
  }

  PrimValue min_value(cast(DataType::Float(32), tvm::min_value(attrs->out_dtype)));
  PrimValue max_value(cast(DataType::Float(32), tvm::max_value(attrs->out_dtype)));
  return astype(clip(scaled_output, min_value, max_value), attrs->out_dtype);
}

TVM_REGISTER_OP("relax.qnn.dequantize")
    .set_attrs_type<QuantizeAttrs>()
    .set_num_inputs(3)
    .add_argument("data", "Tensor", "The input tensor.")
    .add_argument("scale", "Tensor", "The quantization scale of the output tensor.")
    .add_argument("zero_point", "Tensor", "The quantization zero_point of the output tensor.")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoQnnDequantize)
    .set_attr<FQnnLower>("FQnnLower", LowerDequantize)
    .set_attr<Bool>("FPurity", Bool(true));

}  // namespace relax
}  // namespace tvm
