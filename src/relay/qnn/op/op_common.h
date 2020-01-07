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

namespace tvm {
namespace relay {
namespace qnn {

static inline bool QnnBroadcastRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                                   const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 9);

  // Check the scale and zero point types
  CHECK(IsScalarType(types[2], DataType::Float(32)));  // lhs_scale
  CHECK(IsScalarType(types[3], DataType::Int(32)));    // lhs_zero_point
  CHECK(IsScalarType(types[4], DataType::Float(32)));  // rhs_scale
  CHECK(IsScalarType(types[5], DataType::Int(32)));    // rhs_zero_point
  CHECK(IsScalarType(types[6], DataType::Float(32)));  // output_scale
  CHECK(IsScalarType(types[7], DataType::Int(32)));    // output_zero_point

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
#define QNN_REGISTER_BINARY_OP(OpName)                                  \
  TVM_REGISTER_GLOBAL("relay.qnn.op._make." OpName)                     \
  .set_body_typed([](Expr lhs, Expr rhs, Expr lhs_scale, Expr lhs_zero_point, Expr rhs_scale, \
                     Expr rhs_zero_point, Expr output_scale, Expr output_zero_point) { \
    static const Op& op = Op::Get("qnn." OpName);                       \
    return CallNode::make(op, {lhs, rhs,                                \
                               lhs_scale, lhs_zero_point,               \
                               rhs_scale, rhs_zero_point,               \
                               output_scale, output_zero_point}, Attrs(), {}); \
  });                                                                   \
  RELAY_REGISTER_OP("qnn." OpName)                                      \
  .set_num_inputs(8)                                                    \
    .add_argument("lhs", "Tensor", "The left hand side quantized tensor.")                 \
    .add_argument("rhs", "Tensor", "The right hand side quantized tensor.")                \
    .add_argument("lhs_scale", "Tensor", "The scale of the lhs tensor.")                   \
    .add_argument("lhs_zero_point", "Tensor", "The zero_point of the lhs tensor.")         \
    .add_argument("rhs_scale", "Tensor", "The scale of the rhs tensor.")                   \
    .add_argument("rhs_zero_point", "Tensor", "The zero_point of the rhs tensor.")         \
    .add_argument("output_scale", "Tensor", "The scale of the output tensor.")             \
    .add_argument("output_zero_point", "Tensor", "The zero_point of the output tensor.")   \
    .add_type_rel("QnnBroadcast", QnnBroadcastRel)

}  // namespace qnn
}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_QNN_OP_OP_COMMON_H_
