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
 * \file src/relay/qnn/op/dequantize.cc
 * \brief QNN dequantize operator. Dequantize operator converts from quantized
 * domain to unquantized domain.
 */

#include <tvm/relay/analysis.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/qnn/attrs.h>

#include "../../transforms/pattern_utils.h"
#include "../utils.h"

namespace tvm {
namespace relay {
namespace qnn {

TVM_REGISTER_NODE_TYPE(DequantizeAttrs);

bool DequantizeRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                   const TypeReporter& reporter) {
  ICHECK_EQ(types.size(), 4);
  const auto* data = types[0].as<TensorTypeNode>();

  if (data == nullptr) {
    return false;
  }

  const auto input_dtype = data->dtype;
  ICHECK(input_dtype == DataType::Int(8) || input_dtype == DataType::UInt(8) ||
         input_dtype == DataType::Int(16) || input_dtype == DataType::UInt(16) ||
         input_dtype == DataType::Int(32))
      << "Input type should be one of the quantized types [int8, unit8, int16, uint16, int32] but "
      << "was " << input_dtype;

  const auto* dequantize_attrs = attrs.as<DequantizeAttrs>();
  int axis = dequantize_attrs->axis;
  auto rank = static_cast<int>(data->shape.size());
  axis = (axis < 0) ? ((rank > 0) ? data->shape.size() + axis : 0) : axis;

  // If zero point and scale are scalar or have arbitrary rank with one element,
  // then axis doesn't matter.
  bool scale_is_scalar = (types[1].as<TensorTypeNode>())->shape.size() == 0 ||
                         get_const_int((types[1].as<TensorTypeNode>())->Size()) == 1;
  bool zp_is_scalar = (types[2].as<TensorTypeNode>())->shape.size() == 0 ||
                      get_const_int((types[2].as<TensorTypeNode>())->Size()) == 1;

  if (!scale_is_scalar || !zp_is_scalar) {
    ICHECK_LT(axis, rank > 0 ? rank : 1) << "axis " << dequantize_attrs->axis << " is out of range";
    ICHECK_GE(axis, 0) << "axis " << dequantize_attrs->axis << " is out of range";
  }

  PrimExpr axis_shape;
  if (!scale_is_scalar || !zp_is_scalar) {
    axis_shape = data->shape[axis];
  } else {
    axis_shape = Integer(1);
  }
  // Check and assign types for scale and zero points.
  AssignType(types[1], DataType::Float(32), axis_shape, reporter);  // scale
  AssignType(types[2], DataType::Int(32), axis_shape, reporter);    // zero point

  const Array<tvm::PrimExpr> oshape = data->shape;
  const DataType out_dtype = dequantize_attrs->out_dtype;
  ICHECK(out_dtype == DataType::Float(16) || out_dtype == DataType::Float(32))
      << "Output type should be one of [float16, float32] but was " << out_dtype;
  // assign output type.
  reporter->Assign(types[3], TensorType(oshape, out_dtype));
  return true;
}

Expr MakeDequantize(Expr data, Expr input_scale, Expr input_zero_point, int axis,
                    DataType out_dtype) {
  // real_value = scale * (quantized_value - zero_point)
  // A more detailed explanation can be found here -
  // https://github.com/google/gemmlowp/blob/master/doc/quantization.md
  auto attrs = make_object<DequantizeAttrs>();
  attrs->axis = axis;
  attrs->out_dtype = out_dtype;
  static const Op& op = Op::Get("qnn.dequantize");
  return Call(op, {data, input_scale, input_zero_point}, Attrs(attrs), {});
}

Expr DequantizeLower(const Expr& input_tensor, const Expr& input_scale,
                     const Expr& input_zero_point, const Array<tvm::relay::Type>& types,
                     const DequantizeAttrs* attrs) {
  auto axis = attrs->axis;

  ICHECK_EQ(types.size(), 4);
  auto in_type = types[0];
  auto in_tensor_type = in_type.as<TensorTypeNode>();
  ICHECK(in_tensor_type != nullptr) << "Type information missing"
                                    << " Please run infer_type pass.";
  Array<IndexExpr> input_shape = in_tensor_type->shape;

  size_t n_dim = input_shape.size();

  // Wrap axis from negative to positive if needed.
  if (axis < 0) {
    axis = static_cast<int>(n_dim) + axis;
  }

  // Expand scale and zero point if the input tensor is channel quantized
  auto expanded_input_scale = input_scale;
  if (!IsConstScalar(input_scale) && !IsScalarType(types[1])) {
    expanded_input_scale = ExpandBiasToMatchAxis(input_scale, n_dim, {axis});
  }

  auto expanded_input_zero_point = input_zero_point;
  if (!IsConstScalar(input_zero_point) && !IsScalarType(types[2])) {
    expanded_input_zero_point = ExpandBiasToMatchAxis(input_zero_point, n_dim, {axis});
  }

  auto shift = Subtract(Cast(input_tensor, DataType::Int(32)), expanded_input_zero_point);
  auto scaled_output = Multiply(Cast(shift, DataType::Float(32)), expanded_input_scale);

  const DataType out_dtype = attrs->out_dtype;
  if (out_dtype.is_float() && out_dtype.bits() == 32) return scaled_output;

  double min_val = tvm::min_value(out_dtype).as<FloatImmNode>()->value;
  double max_val = tvm::max_value(out_dtype).as<FloatImmNode>()->value;
  auto clamped_output = Clip(scaled_output, min_val, max_val);
  return Cast(clamped_output, out_dtype);
}

Expr DequantizeQnnCanonicalize(const Attrs& attrs, const Array<Expr>& new_args,
                               const Array<tvm::relay::Type>& types) {
  ICHECK_EQ(new_args.size(), 3);
  auto& data = new_args[0];
  auto& input_scale = new_args[1];
  auto& input_zero_point = new_args[2];
  ICHECK_EQ(types.size(), 4);

  // Get attrs.
  const auto* dequantize_attrs = attrs.as<DequantizeAttrs>();
  ICHECK(dequantize_attrs != nullptr);

  return DequantizeLower(data, input_scale, input_zero_point, types, dequantize_attrs);
}

RELAY_REGISTER_OP("qnn.dequantize")
    .describe(R"code(Dequantizes the input and produces float32 output.
The input is always quantized (int8, uint8) and will be converted to float32 given input scale and zero_point.
- **data**: Quantized tensor of any shape to dequantize. The input data can be of floating point
)code" TVM_ADD_FILELINE)
    .set_attrs_type<DequantizeAttrs>()
    .set_num_inputs(3)
    .add_argument("data", "Tensor", "The tensor to dequantize.")
    .add_argument("input_scale", "Tensor", "The quantization scale of the input tensor.")
    .add_argument("input_zero_point", "Tensor", "The quantization zero_point of the input tensor.")
    .set_support_level(11)
    .add_type_rel("Dequantize", DequantizeRel)
    .set_attr<TNonComputational>("TNonComputational", true)
    .set_attr<FTVMLegalize>("FTVMQnnCanonicalize", DequantizeQnnCanonicalize);

TVM_REGISTER_GLOBAL("relay.qnn.op._make.dequantize").set_body_typed(MakeDequantize);

}  // namespace qnn
}  // namespace relay
}  // namespace tvm
