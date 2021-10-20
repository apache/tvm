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
 * \file src/relay/qnn/op/op_common.h
 * \brief A set of utilities and common functionality for QNN ops.
 */
#ifndef TVM_RELAY_QNN_OP_OP_COMMON_H_
#define TVM_RELAY_QNN_OP_OP_COMMON_H_

#include <tvm/relay/expr.h>
#include <tvm/relay/op.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/qnn/attrs.h>

#include <vector>

#include "../../op/type_relations.h"
#include "../../transforms/infer_layout_utils.h"
#include "../utils.h"

namespace tvm {
namespace relay {
namespace qnn {

/*
 * Number of inputs for the Qnn binary operators.
 * Refer the QNN_REGISTER_BINARY_OP macro to see
 * what the operators are.
 */
static constexpr int kNumQnnBinaryOpInputs = 8;

/*
 * Number of expected arg types.
 */
static constexpr int kNumQnnBinaryOpArgTypes = 9;

/*
 * \brief Simple struct to organize the inputs to the Qnn
 * binary operators. The main reason to have a struct
 * is to be able to perform the common checks needed at a
 * central location.
 */
struct QnnBinaryOpArguments {
  Expr lhs;
  Expr rhs;
  Expr lhs_scale;
  Expr lhs_zero_point;
  Expr rhs_scale;
  Expr rhs_zero_point;
  Expr output_scale;
  Expr output_zero_point;

  explicit QnnBinaryOpArguments(const Array<Expr>& new_args) {
    ICHECK_EQ(new_args.size(), kNumQnnBinaryOpInputs);
    int idx = 0;
    lhs = new_args[idx++];
    rhs = new_args[idx++];
    lhs_scale = new_args[idx++];
    lhs_zero_point = new_args[idx++];
    rhs_scale = new_args[idx++];
    rhs_zero_point = new_args[idx++];
    output_scale = new_args[idx++];
    output_zero_point = new_args[idx++];
    ICHECK_EQ(idx, kNumQnnBinaryOpInputs);
  }
};

/*
 * \brief Simple structure to hold the input tensor's dtype
 * and shape. This structure allows a common point to do
 * all the validation checks for Qnn binary operators.
 */
struct QnnBinaryOpTensorType {
  DataType dtype;
  Array<PrimExpr> shape;

  explicit QnnBinaryOpTensorType(const Array<tvm::relay::Type>& arg_types, const int32_t arg_idx) {
    ICHECK_EQ(arg_types.size(), kNumQnnBinaryOpArgTypes);
    auto tensor_type = arg_types[arg_idx].as<TensorTypeNode>();
    ICHECK(tensor_type != nullptr);
    dtype = tensor_type->dtype;
    shape = tensor_type->shape;
  }
};

/*
 * \brief Converts the expression from expression's dtype
 * to target dtype. This is mainly used for converting
 * computations done in Int32 to lower precision Int8 or
 * UInt8.
 * \param expr The expression to whose dtype needs conversion.
 * \param target_dtype The dtype of the target expression
 * \return New expression with target dtype and possibly lower
 * precision.
 */
inline Expr ConvertDtype(const Expr& expr, const DataType& target_dtype) {
  auto q_min = GetQmin(target_dtype);
  auto q_max = GetQmax(target_dtype);
  auto output = Clip(expr, q_min, q_max);
  return Cast(output, target_dtype);
}

/*
 * \brief Requantizes the given expression if expression's
 * scale and zero point both do not match target scale and
 * zero point. This is mainly needed for requantizing the
 * input tensors with output tensor's scale and zero point
 * to ease the computation of final quantized tensor.
 * \param expr The expression on which the check needs to be performed.
 * \param expr_scale The scale of the expression.
 * \param expr_zero_point The zero point of the expression.
 * \param target_scale The scale of the output tensor.
 * \param target_zero_point The zero point of the output tensor.
 * \param expr_shape The shape of the input expression.
 * \return New expression that is requantized to target scale and zero
 * point if the expression scale and zero points are different otherwise
 * it simply casts the given expression to Int32 as no requantization is
 * needed in this case.
 */
inline Expr RequantizeOrUpcast(const Expr& expr, const Expr& expr_scale,
                               const Expr& expr_zero_point, const Expr& target_scale,
                               const Expr& target_zero_point, const Array<PrimExpr>& expr_shape,
                               const DataType& target_dtype = DataType::Int(32)) {
  auto result = expr;
  if (!IsEqualScalar(expr_scale, target_scale) ||
      !IsEqualScalar(expr_zero_point, target_zero_point)) {
    result = Requantize(expr, expr_shape, expr_scale, expr_zero_point, target_scale,
                        target_zero_point, target_dtype);
  } else {
    result = Cast(result, target_dtype);
  }
  return result;
}

/*! \brief Infer layout for QNN binary broadcast operators */
inline InferCorrectLayoutOutput QnnBinaryBroadcastLayout(
    const Attrs& attrs, const Array<Layout>& new_in_layouts, const Array<Layout>& old_in_layouts,
    const Array<tvm::relay::Type>& old_in_types) {
  // Use Relay Binary Broadcast Infer correct layout.
  auto layouts = BinaryBroadcastLayout(attrs, new_in_layouts, old_in_layouts, old_in_types);

  // Fill the layouts of remaining input tensors - scales and zero points. The layouts of these
  // tensors can be treated as C.
  Layout channel_layout = Layout("C");
  Array<Layout> input_layouts = {layouts->input_layouts[0],
                                 layouts->input_layouts[1],
                                 channel_layout,
                                 channel_layout,
                                 channel_layout,
                                 channel_layout,
                                 channel_layout,
                                 channel_layout};
  Array<Layout> output_layouts = layouts->output_layouts;
  return InferCorrectLayoutOutput(input_layouts, output_layouts, attrs);
}

static inline bool QnnBroadcastRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                                   const TypeReporter& reporter) {
  // Expected Types: lhs, rhs, lhs_scale, lhs_zero_point, rhs_scale, rhs_zero_point, output_scale,
  // output_zero_point, out_type
  ICHECK_EQ(types.size(), kNumQnnBinaryOpArgTypes);

  // Check the lhs and rhs types
  for (size_t i = 0; i < 2; ++i) {
    if (types[i].as<IncompleteTypeNode>()) {
      return false;
    }
  }
  // Check the scale and zero point types
  for (size_t i = 2; i < 8; ++i) {
    if (types[i].as<IncompleteTypeNode>()) {
      return false;
    }
  }
  ICHECK(IsScalarType(types[2], DataType::Float(32)));  // lhs_scale
  ICHECK(IsScalarType(types[3], DataType::Int(32)));    // lhs_zero_point
  ICHECK(IsScalarType(types[4], DataType::Float(32)));  // rhs_scale
  ICHECK(IsScalarType(types[5], DataType::Int(32)));    // rhs_zero_point
  ICHECK(IsScalarType(types[6], DataType::Float(32)));  // output_scale
  ICHECK(IsScalarType(types[7], DataType::Int(32)));    // output_zero_point

  // Collect the input tensor and output tensor devoid of scale and zero points to reuse Relay
  // BroadcastRel infer type function.
  Array<Type> tensor_types = {types[0], types[1], types[8]};
  return BroadcastRel(tensor_types, 3, attrs, reporter);
}

/*! Quick helper macro
 * - Expose a positional make function to construct the node.
 * - Register op to the registry.
 *
 * We make the decision to always only expose positional argument.
 * We will do rewrapping in the frontend to support language
 * sugars such as keyword arguments and default value.
 *
 * \param OpName the name of registry.
 */
#define QNN_REGISTER_BINARY_OP(OpName)                                                             \
  TVM_REGISTER_GLOBAL("relay.qnn.op._make." OpName)                                                \
      .set_body_typed([](Expr lhs, Expr rhs, Expr lhs_scale, Expr lhs_zero_point, Expr rhs_scale,  \
                         Expr rhs_zero_point, Expr output_scale, Expr output_zero_point) {         \
        static const Op& op = Op::Get("qnn." OpName);                                              \
        return Call(op,                                                                            \
                    {lhs, rhs, lhs_scale, lhs_zero_point, rhs_scale, rhs_zero_point, output_scale, \
                     output_zero_point},                                                           \
                    Attrs(), {});                                                                  \
      });                                                                                          \
  RELAY_REGISTER_OP("qnn." OpName)                                                                 \
      .set_num_inputs(kNumQnnBinaryOpInputs)                                                       \
      .add_argument("lhs", "Tensor", "The left hand side quantized tensor.")                       \
      .add_argument("rhs", "Tensor", "The right hand side quantized tensor.")                      \
      .add_argument("lhs_scale", "Tensor", "The scale of the lhs tensor.")                         \
      .add_argument("lhs_zero_point", "Tensor", "The zero_point of the lhs tensor.")               \
      .add_argument("rhs_scale", "Tensor", "The scale of the rhs tensor.")                         \
      .add_argument("rhs_zero_point", "Tensor", "The zero_point of the rhs tensor.")               \
      .add_argument("output_scale", "Tensor", "The scale of the output tensor.")                   \
      .add_argument("output_zero_point", "Tensor", "The zero_point of the output tensor.")         \
      .add_type_rel("QnnBroadcast", QnnBroadcastRel)                                               \
      .set_attr<TNonComputational>("TNonComputational", true)                                      \
      .set_attr<FInferCorrectLayout>("FInferCorrectLayout", QnnBinaryBroadcastLayout)

}  // namespace qnn
}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_QNN_OP_OP_COMMON_H_
