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

#include <tvm/ffi/extra/visit_error_context.h>
#include <tvm/ffi/reflection/registry.h>

#include <utility>

#include "../../transform/utils.h"
#include "../op_common.h"

namespace tvm {
namespace relax {

TVM_FFI_STATIC_INIT_BLOCK() { QuantizeAttrs::RegisterReflection(); }

/* relax.quantize */

Expr quantize(Expr data, Expr scale, Expr zero_point, int axis, DLDataType out_dtype) {
  ffi::ObjectPtr<QuantizeAttrs> attrs = ffi::make_object<QuantizeAttrs>();
  attrs->axis = axis;
  attrs->out_dtype = out_dtype;
  static const Op& op = Op::Get("relax.quantize");
  return Call(op, {std::move(data), std::move(scale), std::move(zero_point)}, Attrs(attrs));
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.quantize", quantize);
}

Type InferTypeQuantize(const Call& call, const BlockBuilder& ctx) {
  const auto* attrs = call->attrs.as<QuantizeAttrs>();
  if (attrs->out_dtype != DLDataType{kDLInt, 8, 1} &&
      attrs->out_dtype != DLDataType{kDLUInt, 8, 1} &&
      attrs->out_dtype != DLDataType{kDLInt, 16, 1} &&
      attrs->out_dtype != DLDataType{kDLUInt, 16, 1} &&
      attrs->out_dtype != DLDataType{static_cast<uint8_t>(kDLFloat8_e4m3fn),
                                     static_cast<uint8_t>(8), static_cast<uint16_t>(1)} &&
      attrs->out_dtype != DLDataType{static_cast<uint8_t>(kDLFloat8_e5m2), static_cast<uint8_t>(8),
                                     static_cast<uint16_t>(1)}) {
    TVM_FFI_VISIT_THROW(TypeError, call)
        << "Unsupported output datatype attribute for operation: '" << attrs->out_dtype;
  }

  TensorType input_ty = GetInputTensorType(call, ctx)[0];
  TensorType scale_ty = GetInputTensorType(call, ctx)[1];
  TensorType zp_ty = GetInputTensorType(call, ctx)[2];
  PrimType input_dtype = input_ty->dtype.value();
  PrimType scale_dtype = scale_ty->dtype.value();
  PrimType zp_dtype = zp_ty->dtype.value();

  // Check input datatype:
  if (input_dtype != PrimType::Float(16) && input_dtype != PrimType::Float(32)) {
    TVM_FFI_VISIT_THROW(TypeError, call)
        << "Unsupported input datatype for operation: " << input_ty->dtype;
  }

  // Check datatype of scale param:
  if (scale_dtype != PrimType::Float(32) && scale_dtype != PrimType::Float(16)) {
    TVM_FFI_VISIT_THROW(TypeError, call)
        << "scale param datatype should be one of [float16, float32], but got " << scale_ty->dtype;
  }

  // Check datatype of zero_point param:
  if (zp_dtype != PrimType::Int(8) && zp_dtype != PrimType::UInt(8) &&
      zp_dtype != PrimType::Int(16) && zp_dtype != PrimType::UInt(16) &&
      zp_dtype != PrimType::Int(32) && zp_dtype != PrimType::UInt(32) &&
      zp_dtype != PrimType::Float(16)) {
    TVM_FFI_VISIT_THROW(TypeError, call)
        << "zero_point param datatype should be one of "
        << "['int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'float16'], "
        << "but got " << zp_ty->dtype;
  }

  // Check that "axis" attribute is not out of range:
  int axis = (attrs->axis < 0) ? (input_ty->ndim + attrs->axis) : attrs->axis;
  if (axis < 0 || axis > input_ty->ndim - 1) {
    TVM_FFI_VISIT_THROW(ValueError, call)
        << "relax.quantize: axis param is out of range (" << attrs->axis << ")";
  }

  auto check_param_size = [&](const TensorType& param_ty, const TensorType& data_ty,
                              ffi::String param_name) {
    const PrimExpr& param_dim = param_ty->GetShape().value()[0];
    const PrimExpr& input_dim = data_ty->GetShape().value()[axis];
    if (!ctx->GetAnalyzer()->CanProveEqual(param_dim, input_dim)) {
      TVM_FFI_VISIT_THROW(ValueError, call)
          << "Size mismatch: " << call->op << ": the input shape at dim " << attrs->axis << " is '"
          << input_dim << "', but size of " << param_name << " param is '" << param_dim << "'";
    }
  };

  auto is_scalar_or_singleton_vector = [&](const TensorType& param_ty) {
    if (IsScalarTensor(param_ty)) return true;
    if (param_ty->shape.defined() && param_ty->shape->IsInstance<ShapeExprNode>()) {
      const auto& values = param_ty->shape.as<ShapeExprNode>()->values;
      if (!values.empty()) {
        return std::all_of(values.begin(), values.end(), [&](const PrimExpr& dim) {
          return ctx->GetAnalyzer()->CanProveEqual(dim, 1);
        });
      }
    }
    return false;
  };

  // Check size matching of scale/zp params with input shape at dim = attrs->axis.
  if (!is_scalar_or_singleton_vector(scale_ty)) check_param_size(scale_ty, input_ty, "scale");
  if (!is_scalar_or_singleton_vector(zp_ty)) check_param_size(zp_ty, input_ty, "zero_point");

  auto output_ty = ffi::make_object<TensorTypeNode>(*input_ty.get());
  output_ty->dtype = PrimType(attrs->out_dtype);
  return TensorType(output_ty);
}

TVM_REGISTER_OP("relax.quantize")
    .set_attrs_type<QuantizeAttrs>()
    .set_num_inputs(3)
    .add_argument("data", "Tensor", "The input tensor.")
    .add_argument("scale", "Tensor", "The quantization scale of the output tensor.")
    .add_argument("zero_point", "Tensor", "The quantization zero_point of the output tensor.")
    .set_attr<FInferType>("FInferType", InferTypeQuantize)
    .set_attr<bool>("FPurity", true);

/* relax.dequantize */

Expr dequantize(Expr data, Expr scale, Expr zero_point, int axis, DLDataType out_dtype) {
  ffi::ObjectPtr<QuantizeAttrs> attrs = ffi::make_object<QuantizeAttrs>();
  attrs->axis = axis;
  attrs->out_dtype = out_dtype;
  static const Op& op = Op::Get("relax.dequantize");
  return Call(op, {std::move(data), std::move(scale), std::move(zero_point)}, Attrs(attrs));
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.dequantize", dequantize);
}

Type InferTypeDequantize(const Call& call, const BlockBuilder& ctx) {
  const auto* attrs = call->attrs.as<QuantizeAttrs>();
  if (attrs->out_dtype != DLDataType{kDLFloat, 16, 1} &&
      attrs->out_dtype != DLDataType{kDLFloat, 32, 1}) {
    TVM_FFI_VISIT_THROW(TypeError, call)
        << "Unsupported output datatype attribute for operation: " << attrs->out_dtype;
  }

  TensorType input_ty = GetInputTensorType(call, ctx)[0];
  TensorType scale_ty = GetInputTensorType(call, ctx)[1];
  TensorType zp_ty = GetInputTensorType(call, ctx)[2];
  PrimType input_dtype = input_ty->dtype.value();
  PrimType scale_dtype = scale_ty->dtype.value();
  PrimType zp_dtype = zp_ty->dtype.value();

  // Check input datatype:
  if (input_dtype != PrimType::Int(8) && input_dtype != PrimType::UInt(8) &&
      input_dtype != PrimType::Int(16) && input_dtype != PrimType::UInt(16) &&
      input_dtype != PrimType::Int(32) &&
      input_dtype != PrimType(DLDataType{static_cast<uint8_t>(kDLFloat8_e4m3fn),
                                         static_cast<uint8_t>(8), static_cast<uint16_t>(1)}) &&
      input_dtype != PrimType(DLDataType{static_cast<uint8_t>(kDLFloat8_e5m2),
                                         static_cast<uint8_t>(8), static_cast<uint16_t>(1)}) &&
      input_dtype != PrimType::Float(16) && input_dtype != PrimType::Float(32)) {
    TVM_FFI_VISIT_THROW(TypeError, call)
        << "Unsupported input datatype for operation: " << attrs->out_dtype;
  }

  // Check datatype of scale param:
  if (scale_dtype != PrimType::Float(32) && scale_dtype != PrimType::Float(16)) {
    TVM_FFI_VISIT_THROW(TypeError, call)
        << "scale param datatype should be one of [float16, float32], but got " << scale_ty->dtype;
  }

  // Check datatype of zero_point param:
  if (zp_dtype != PrimType::Int(8) && zp_dtype != PrimType::UInt(8) &&
      zp_dtype != PrimType::Int(16) && zp_dtype != PrimType::UInt(16) &&
      zp_dtype != PrimType::Int(32) && zp_dtype != PrimType::UInt(32) &&
      zp_dtype != PrimType::Float(16)) {
    TVM_FFI_VISIT_THROW(TypeError, call)
        << "zero_point param datatype should be one of "
        << "['int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'float16'], "
        << "but got " << zp_ty->dtype;
  }

  // Check that "axis" attribute is not out of range:
  int axis = (attrs->axis < 0) ? (input_ty->ndim + attrs->axis) : attrs->axis;
  if (axis < 0 || axis > input_ty->ndim - 1) {
    TVM_FFI_VISIT_THROW(ValueError, call)
        << "relax.dequantize: axis param is out of range (" << attrs->axis << ")";
  }

  auto check_param_size = [&](const TensorType& param_ty, const TensorType& data_ty,
                              ffi::String param_name) {
    const PrimExpr& param_dim = param_ty->GetShape().value()[0];
    const PrimExpr& input_dim = data_ty->GetShape().value()[axis];
    if (!ctx->GetAnalyzer()->CanProveEqual(param_dim, input_dim)) {
      TVM_FFI_VISIT_THROW(ValueError, call)
          << "Size mismatch: " << call->op << ": the input shape at dim " << attrs->axis << " is '"
          << input_dim << "', but size of " << param_name << " param is '" << param_dim << "'";
    }
  };

  auto is_scalar_or_singleton_vector = [&](const TensorType& param_ty) {
    if (IsScalarTensor(param_ty)) return true;
    if (param_ty->shape.defined() && param_ty->shape->IsInstance<ShapeExprNode>()) {
      const auto& values = param_ty->shape.as<ShapeExprNode>()->values;
      if (!values.empty()) {
        return std::all_of(values.begin(), values.end(), [&](const PrimExpr& dim) {
          return ctx->GetAnalyzer()->CanProveEqual(dim, 1);
        });
      }
    }
    return false;
  };

  // Check size matching of scale/zp params with input shape at dim = attrs->axis.
  if (!is_scalar_or_singleton_vector(scale_ty)) check_param_size(scale_ty, input_ty, "scale");
  if (!is_scalar_or_singleton_vector(zp_ty)) check_param_size(zp_ty, input_ty, "zero_point");

  auto output_ty = ffi::make_object<TensorTypeNode>(*input_ty.get());
  output_ty->dtype = PrimType(attrs->out_dtype);
  return TensorType(output_ty);
}

TVM_REGISTER_OP("relax.dequantize")
    .set_attrs_type<QuantizeAttrs>()
    .set_num_inputs(3)
    .add_argument("data", "Tensor", "The input tensor.")
    .add_argument("scale", "Tensor", "The quantization scale of the input tensor.")
    .add_argument("zero_point", "Tensor", "The quantization zero_point of the input tensor.")
    .set_attr<FInferType>("FInferType", InferTypeDequantize)
    .set_attr<bool>("FPurity", true);

}  // namespace relax
}  // namespace tvm
