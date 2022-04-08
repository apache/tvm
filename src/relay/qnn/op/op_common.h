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
#include <tvm/relay/qnn/transform.h>

#include <vector>

#include "../../op/type_relations.h"
#include "../../transforms/infer_layout_utils.h"
#include "../utils.h"

namespace tvm {
namespace relay {
namespace qnn {

TVM_REGISTER_NODE_TYPE(BroadcastAttrs);

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
 * Number of inputs for the Qnn unary operators.
 */
static constexpr int kNumQnnUnaryOpInputs = 5;

/*
 * Number of expected arg types.
 */
static constexpr int kNumQnnUnaryOpArgTypes = 6;

/*
 * \brief Simple struct to organize the inputs to the Qnn
 * unary operators. The main reason to have a struct
 * is to be able to perform the common checks needed at a
 * central location.
 */
struct QnnUnaryOpArguments {
  Expr x;
  Expr scale;
  Expr zero_point;
  Expr output_scale;
  Expr output_zero_point;

  explicit QnnUnaryOpArguments(const Array<Expr>& new_args) {
    ICHECK_EQ(new_args.size(), kNumQnnUnaryOpInputs);
    int idx = 0;
    x = new_args[idx++];
    scale = new_args[idx++];
    zero_point = new_args[idx++];
    output_scale = new_args[idx++];
    output_zero_point = new_args[idx++];
    ICHECK_EQ(idx, kNumQnnUnaryOpInputs);
  }
};

/*
 * \brief Simple structure to hold the input tensor's dtype
 * and shape. This structure allows a common point to do
 * all the validation checks for Qnn unary operators.
 */
struct QnnUnaryOpTensorType {
  DataType dtype;
  Array<PrimExpr> shape;

  explicit QnnUnaryOpTensorType(const Array<tvm::relay::Type>& arg_types, const int32_t arg_idx) {
    ICHECK_EQ(arg_types.size(), kNumQnnUnaryOpArgTypes);
    auto tensor_type = arg_types[arg_idx].as<TensorTypeNode>();
    ICHECK(tensor_type != nullptr);
    dtype = tensor_type->dtype;
    shape = tensor_type->shape;
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
                               const int& axis = -1,
                               const DataType& target_dtype = DataType::Int(32)) {
  auto result = expr;
  if (!IsEqualScalar(expr_scale, target_scale) ||
      !IsEqualScalar(expr_zero_point, target_zero_point)) {
    result = Requantize(expr, expr_shape, expr_scale, expr_zero_point, target_scale,
                        target_zero_point, target_dtype, axis);
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

  const auto* lhs_data = types[0].as<TensorTypeNode>();
  const auto* rhs_data = types[1].as<TensorTypeNode>();

  if (lhs_data == nullptr || rhs_data == nullptr) {
    return false;
  }

  const BroadcastAttrs* broadcast_attrs = attrs.as<BroadcastAttrs>();
  ICHECK(broadcast_attrs);

  auto lhs_rank = static_cast<int>(lhs_data->shape.size());
  auto rhs_rank = static_cast<int>(rhs_data->shape.size());

  auto get_channel_axis = [](int rank, int axis_from_attr) {
    if (rank <= 1) return 0;
    if (axis_from_attr < 0) return rank + axis_from_attr;
    return axis_from_attr;
  };

  const int lhs_axis = get_channel_axis(lhs_rank, broadcast_attrs->lhs_axis);
  const int rhs_axis = get_channel_axis(rhs_rank, broadcast_attrs->rhs_axis);

  // If zero point and scale are scalar then axis doesn't matter.
  bool lhs_scale_is_scalar = (types[2].as<TensorTypeNode>())->shape.size() == 0;
  bool lhs_zp_is_scalar = (types[3].as<TensorTypeNode>())->shape.size() == 0;
  bool rhs_scale_is_scalar = (types[4].as<TensorTypeNode>())->shape.size() == 0;
  bool rhs_zp_is_scalar = (types[5].as<TensorTypeNode>())->shape.size() == 0;

  if (!(lhs_scale_is_scalar && lhs_zp_is_scalar)) {
    ICHECK_LT(lhs_axis, lhs_rank > 0 ? lhs_rank : 1)
        << "lhs_axis " << broadcast_attrs->lhs_axis << " is out of range";
    ICHECK_GE(lhs_axis, 0) << "lhs_axis " << broadcast_attrs->lhs_axis << " is out of range";
  }

  if (!(rhs_scale_is_scalar && rhs_zp_is_scalar)) {
    ICHECK_LT(rhs_axis, rhs_rank > 0 ? rhs_rank : 1)
        << "rhs_axis " << broadcast_attrs->rhs_axis << " is out of range";
    ICHECK_GE(rhs_axis, 0) << "rhs_axis " << broadcast_attrs->rhs_axis << " is out of range";
  }

  PrimExpr lhs_axis_shape;
  if (lhs_rank > 0) {
    lhs_axis_shape = lhs_data->shape[lhs_axis];
  } else {
    lhs_axis_shape = Integer(1);
  }

  PrimExpr rhs_axis_shape;
  if (rhs_rank > 0) {
    rhs_axis_shape = rhs_data->shape[rhs_axis];
  } else {
    rhs_axis_shape = Integer(1);
  }

  // Check and assign types for scale and zero points.
  AssignType(types[2], DataType::Float(32), lhs_axis_shape, reporter);  // lhs_scale
  AssignType(types[3], DataType::Int(32), lhs_axis_shape, reporter);    // lhs_zero_point
  AssignType(types[4], DataType::Float(32), rhs_axis_shape, reporter);  // rhs_scale
  AssignType(types[5], DataType::Int(32), rhs_axis_shape, reporter);    // rhs_zero_point

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
                         Expr rhs_zero_point, Expr output_scale, Expr output_zero_point,           \
                         int lhs_axis, int rhs_axis) {                                             \
        static const Op& op = Op::Get("qnn." OpName);                                              \
        auto attrs = make_object<BroadcastAttrs>();                                                \
        attrs->lhs_axis = lhs_axis;                                                                \
        attrs->rhs_axis = rhs_axis;                                                                \
        return Call(op,                                                                            \
                    {lhs, rhs, lhs_scale, lhs_zero_point, rhs_scale, rhs_zero_point, output_scale, \
                     output_zero_point},                                                           \
                    Attrs(attrs), {});                                                             \
      });                                                                                          \
  RELAY_REGISTER_OP("qnn." OpName)                                                                 \
      .set_attrs_type<BroadcastAttrs>()                                                            \
      .set_num_inputs(kNumQnnBinaryOpInputs)                                                       \
      .add_argument("lhs", "Tensor", "The left hand side quantized tensor.")                       \
      .add_argument("rhs", "Tensor", "The right hand side quantized tensor.")                      \
      .add_argument("lhs_scale", "Tensor", "The scale of the lhs tensor.")                         \
      .add_argument("lhs_zero_point", "Tensor", "The zero_point of the lhs tensor.")               \
      .add_argument("rhs_scale", "Tensor", "The scale of the rhs tensor.")                         \
      .add_argument("rhs_zero_point", "Tensor", "The zero_point of the rhs tensor.")               \
      .add_argument("output_scale", "Tensor", "The scale of the output tensor.")                   \
      .add_argument("output_zero_point", "Tensor", "The zero_point of the output tensor.")         \
      .add_argument("lhs_axis", "Tensor", "The channel quantization of the lhs tensor.")           \
      .add_argument("rhs_axis", "Tensor", "The channel quantization of the rhs tensor.")           \
      .add_type_rel("QnnBroadcast", QnnBroadcastRel)                                               \
      .set_attr<TNonComputational>("TNonComputational", true)                                      \
      .set_attr<FInferCorrectLayout>("FInferCorrectLayout", QnnBinaryBroadcastLayout)

static inline bool QnnElementwiseUnaryFuncRel(const Array<Type>& types, int num_inputs,
                                              const Attrs& attrs, const TypeReporter& reporter) {
  // Expected Types: data, scale, zero_point, output_scale, output_zero_point
  ICHECK_EQ(types.size(), 6);
  const auto* x = types[0].as<TensorTypeNode>();
  if (x == nullptr) return false;
  ICHECK(x->dtype == DataType::Int(8) || x->dtype == DataType::UInt(8))
      << "Expected quantized type(int8, uint8) for input but was " << x->dtype;

  // Check the types of scale and zero points.
  for (size_t i = 1; i < 5; ++i) {
    if (types[i].as<IncompleteTypeNode>()) {
      return false;
    }
  }
  ICHECK(IsScalarType(types[1], DataType::Float(32)));  // scale
  ICHECK(IsScalarType(types[2], DataType::Int(32)));    // zero_point
  ICHECK(IsScalarType(types[3], DataType::Float(32)));  // output_scale
  ICHECK(IsScalarType(types[4], DataType::Int(32)));    // output_zero_point

  // Assign types for scale and zero points.
  reporter->Assign(types[1], TensorType({}, DataType::Float(32)));  // scale
  reporter->Assign(types[2], TensorType({}, DataType::Int(32)));    // zero_point
  reporter->Assign(types[3], TensorType({}, DataType::Float(32)));  // output_scale
  reporter->Assign(types[4], TensorType({}, DataType::Int(32)));    // output_zero_point

  // Collect the input tensor and output tensor devoid of scale and zero points to reuse Relay
  // IdentityRel infer type function.
  Array<Type> tensor_types = {types[0], types[5]};
  return IdentityRel(tensor_types, 2, attrs, reporter);
}

static inline Expr LegalizeExpr(const Expr& expr) {
  // Canonicalizations should not contain qnn ops, so use this
  // to lower expressions automatically after using things like qnn.dequantize
  // in the lowering process.
  auto mod = IRModule::FromExpr(expr);
  mod = transform::Legalize()(mod);
  if (expr.as<FunctionNode>()) {
    return mod->Lookup("main");
  } else {
    return mod->Lookup("main").as<FunctionNode>()->body;
  }
}

/*! Quick helper macro
 * - Expose a positional make function to construct the node.
 * - Register op to the registry.
 *
 * For Unary Operators which also take in QParams.
 *
 * \param OpName the name of registry.
 */
#define QNN_CREATE_UNARY_ELEMENTWISE_OP(OpName)                                                 \
  TVM_REGISTER_GLOBAL("relay.qnn.op._make." OpName)                                             \
      .set_body_typed(                                                                          \
          [](Expr x, Expr scale, Expr zero_point, Expr output_scale, Expr output_zero_point) {  \
            return Call(Op::Get("qnn." OpName),                                                 \
                        {x, scale, zero_point, output_scale, output_zero_point}, Attrs(), {});  \
          });                                                                                   \
                                                                                                \
  RELAY_REGISTER_OP("qnn." OpName)                                                              \
      .describe("Elementwise " OpName " for quantized tensors.")                                \
      .set_num_inputs(5)                                                                        \
      .add_argument("data", "Quantized Tensor", "The input data.")                              \
      .add_argument("scale", "Tensor", "The quantization scale of the input tensor.")           \
      .add_argument("zero_point", "Tensor", "The quantization zero_point of the input tensor.") \
      .add_argument("output_scale", "Tensor", "The quantization scale of the output tensor.")   \
      .add_argument("output_zero_point", "Tensor",                                              \
                    "The quantization zero_point of the output tensor.")                        \
      .set_support_level(11)                                                                    \
      .add_type_rel("qnn." OpName, QnnElementwiseUnaryFuncRel)                                  \
      .set_attr<TNonComputational>("TNonComputational", true)

/*! Quick helper macro
 * Create a default canonicalization for a QNN operator, which dequantizes the operator
 * runs the calculation using the provided Call func, and then requantizes.
 *
 * FloatingPointFunc is usually a handle from "src/relay/transforms/pattern_utils.h"
 *
 * \param FloatingPointFunc the floating point function with function signature `Expr Erf(Expr e)`
 */
#define QNN_UNARY_OP_DEFAULT_CANONICALIZATION(FloatingPointFunc)                                  \
  [](const Attrs& attrs, const Array<Expr>& new_args, const Array<tvm::relay::Type>& arg_types) { \
    QnnUnaryOpArguments args(new_args);                                                           \
    QnnUnaryOpTensorType input_type(arg_types, 0);                                                \
    Expr dequantized_arg = MakeDequantize(args.x, args.scale, args.zero_point, -1);               \
    Expr output = FloatingPointFunc(dequantized_arg);                                             \
    Expr result =                                                                                 \
        MakeQuantize(output, args.output_scale, args.output_zero_point, -1, input_type.dtype);    \
    return LegalizeExpr(result);                                                                  \
  }
}  // namespace qnn
}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_QNN_OP_OP_COMMON_H_
