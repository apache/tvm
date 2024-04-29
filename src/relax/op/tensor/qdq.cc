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
 * \file src/relax/op/tensor/qdq.cc
 * \brief implements quantize/dequantize operators.
 */

#include "qdq.h"

#include <utility>

#include "../../transform/utils.h"
#include "../op_common.h"

namespace tvm {
namespace relax {

TVM_REGISTER_NODE_TYPE(QuantizeAttrs);

/* relax.quantize */

Expr quantize(Expr data, Expr scale, Expr zero_point, int axis, DataType out_dtype) {
  ObjectPtr<QuantizeAttrs> attrs = make_object<QuantizeAttrs>();
  attrs->axis = axis;
  attrs->out_dtype = out_dtype;
  static const Op& op = Op::Get("relax.quantize");
  return Call(op, {std::move(data), std::move(scale), std::move(zero_point)}, Attrs(attrs));
}

TVM_REGISTER_GLOBAL("relax.op.quantize").set_body_typed(quantize);

StructInfo InferStructInfoQuantize(const Call& call, const BlockBuilder& ctx) {
  const auto* attrs = call->attrs.as<QuantizeAttrs>();
  if (attrs->out_dtype != DataType::Int(8) && attrs->out_dtype != DataType::UInt(8) &&
      attrs->out_dtype != DataType::Int(16) && attrs->out_dtype != DataType::UInt(16) &&
      attrs->out_dtype != DataType::NVFloat8E4M3() &&
      attrs->out_dtype != DataType::NVFloat8E5M2()) {
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
  if (zp_sinfo->dtype != DataType::Int(8) && zp_sinfo->dtype != DataType::Float(16)) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "zero_point param datatype should be 'int8' or 'float16', but got "
                     << zp_sinfo->dtype);
  }

  // Check that "axis" attribute is not out of range:
  int axis = (attrs->axis < 0) ? (input_sinfo->ndim + attrs->axis) : attrs->axis;
  if (axis < 0 || axis > input_sinfo->ndim - 1) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "relax.quantize: axis param is out of range (" << attrs->axis << ")");
  }

  auto check_param_size = [&](const TensorStructInfo& param_sinfo,
                              const TensorStructInfo& data_sinfo, String param_name) {
    const PrimExpr& param_dim = param_sinfo->GetShape().value()[0];
    const PrimExpr& input_dim = data_sinfo->GetShape().value()[axis];
    if (!ctx->GetAnalyzer()->CanProveEqual(param_dim, input_dim)) {
      ctx->ReportFatal(Diagnostic::Error(call)
                       << "Size mismatch: " << call->op << ": the input shape at dim "
                       << attrs->axis << " is '" << input_dim << "', but size of " << param_name
                       << " param is '" << param_dim << "'");
    }
  };

  // Check size matching of scale/zp params with input shape at dim = attrs->axis.
  if (!IsScalarTensor(scale_sinfo)) check_param_size(scale_sinfo, input_sinfo, "scale");
  if (!IsScalarTensor(zp_sinfo)) check_param_size(zp_sinfo, input_sinfo, "zero_point");

  auto output_sinfo = make_object<TensorStructInfoNode>(*input_sinfo.get());
  output_sinfo->dtype = attrs->out_dtype;
  return TensorStructInfo(output_sinfo);
}

TVM_REGISTER_OP("relax.quantize")
    .set_attrs_type<QuantizeAttrs>()
    .set_num_inputs(3)
    .add_argument("data", "Tensor", "The input tensor.")
    .add_argument("scale", "Tensor", "The quantization scale of the output tensor.")
    .add_argument("zero_point", "Tensor", "The quantization zero_point of the output tensor.")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoQuantize)
    .set_attr<Bool>("FPurity", Bool(true));

/* relax.dequantize */

Expr dequantize(Expr data, Expr scale, Expr zero_point, int axis, DataType out_dtype) {
  ObjectPtr<QuantizeAttrs> attrs = make_object<QuantizeAttrs>();
  attrs->axis = axis;
  attrs->out_dtype = out_dtype;
  static const Op& op = Op::Get("relax.dequantize");
  return Call(op, {std::move(data), std::move(scale), std::move(zero_point)}, Attrs(attrs));
}

TVM_REGISTER_GLOBAL("relax.op.dequantize").set_body_typed(dequantize);

StructInfo InferStructInfoDequantize(const Call& call, const BlockBuilder& ctx) {
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
      input_sinfo->dtype != DataType::Int(32) && input_sinfo->dtype != DataType::NVFloat8E4M3() &&
      input_sinfo->dtype != DataType::NVFloat8E5M2() && input_sinfo->dtype != DataType::Float(16) &&
      input_sinfo->dtype != DataType::Float(32)) {
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
  if (zp_sinfo->dtype != DataType::Int(8) && zp_sinfo->dtype != DataType::Float(16)) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "zero_point param datatype should be 'int8' or 'float16', but got "
                     << zp_sinfo->dtype);
  }

  // Check that "axis" attribute is not out of range:
  int axis = (attrs->axis < 0) ? (input_sinfo->ndim + attrs->axis) : attrs->axis;
  if (axis < 0 || axis > input_sinfo->ndim - 1) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "relax.dequantize: axis param is out of range (" << attrs->axis << ")");
  }

  auto check_param_size = [&](const TensorStructInfo& param_sinfo,
                              const TensorStructInfo& data_sinfo, String param_name) {
    const PrimExpr& param_dim = param_sinfo->GetShape().value()[0];
    const PrimExpr& input_dim = data_sinfo->GetShape().value()[axis];
    if (!ctx->GetAnalyzer()->CanProveEqual(param_dim, input_dim)) {
      ctx->ReportFatal(Diagnostic::Error(call)
                       << "Size mismatch: " << call->op << ": the input shape at dim "
                       << attrs->axis << " is '" << input_dim << "', but size of " << param_name
                       << " param is '" << param_dim << "'");
    }
  };

  // Check size matching of scale/zp params with input shape at dim = attrs->axis.
  if (!IsScalarTensor(scale_sinfo)) check_param_size(scale_sinfo, input_sinfo, "scale");
  if (!IsScalarTensor(zp_sinfo)) check_param_size(zp_sinfo, input_sinfo, "zero_point");

  auto output_sinfo = make_object<TensorStructInfoNode>(*input_sinfo.get());
  output_sinfo->dtype = attrs->out_dtype;
  return TensorStructInfo(output_sinfo);
}

TVM_REGISTER_OP("relax.dequantize")
    .set_attrs_type<QuantizeAttrs>()
    .set_num_inputs(3)
    .add_argument("data", "Tensor", "The input tensor.")
    .add_argument("scale", "Tensor", "The quantization scale of the input tensor.")
    .add_argument("zero_point", "Tensor", "The quantization zero_point of the input tensor.")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoDequantize)
    .set_attr<Bool>("FPurity", Bool(true));

}  // namespace relax
}  // namespace tvm
