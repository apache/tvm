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
#define QNN_REGISTER_BINARY_OP(OpName)                                                     \
  TVM_REGISTER_API("relay.qnn.op._make." OpName)                                           \
    .set_body_typed<Expr(Expr, Expr, double, int32_t, double, int32_t, double, int32_t)>(  \
        [](Expr lhs, Expr rhs, double lhs_scale, int32_t lhs_zero_point, double rhs_scale, \
           int32_t rhs_zero_point, double output_scale, int32_t output_zero_point) {       \
          auto attrs = make_node<QnnBinaryOpAttrs>();                                      \
          attrs->lhs_scale = lhs_scale;                                                    \
          attrs->lhs_zero_point = lhs_zero_point;                                          \
          attrs->rhs_scale = rhs_scale;                                                    \
          attrs->rhs_zero_point = rhs_zero_point;                                          \
          attrs->output_scale = output_scale;                                              \
          attrs->output_zero_point = output_zero_point;                                    \
          static const Op& op = Op::Get("qnn." OpName);                                    \
          return CallNode::make(op, {lhs, rhs}, Attrs(attrs), {});                         \
        });                                                                                \
  RELAY_REGISTER_OP("qnn." OpName)                                                         \
    .set_num_inputs(2)                                                                     \
    .add_argument("lhs", "Tensor", "The left hand side quantized tensor.")                 \
    .add_argument("rhs", "Tensor", "The right hand side quantized tensor.")                \
    .add_type_rel("Broadcast", BroadcastRel)

}  // namespace qnn
}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_QNN_OP_OP_COMMON_H_
