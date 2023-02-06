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
 * \file binary.h
 * \brief The functions to make Relax binary arithmetic and comparison operator calls.
 */
#ifndef TVM_RELAX_OP_TENSOR_BINARY_H_
#define TVM_RELAX_OP_TENSOR_BINARY_H_

#include "../op_common.h"

namespace tvm {
namespace relax {

/*!
 * \brief Quick helper macro
 * - Expose a make function to construct the node.
 * - Register op to the registry.
 * \param OpName The name of operator to register. The name passed in will
 *  1. be prepended with a prefix "relax.op." as the FFI identifier string for the make function,
 *  2. be prepended with a prefix "relax." as the identifier string in the operator registry.
 */
#define RELAX_REGISTER_BINARY_OP_AND_IMPL(OpName)                  \
  Expr OpName(Expr x1, Expr x2) {                                  \
    static const Op& op = Op::Get("relax." #OpName);               \
    return Call(op, {x1, x2}, Attrs(), {});                        \
  }                                                                \
  TVM_REGISTER_GLOBAL("relax.op." #OpName).set_body_typed(OpName); \
  TVM_REGISTER_OP("relax." #OpName)                                \
      .set_num_inputs(2)                                           \
      .add_argument("x1", "Tensor", "The first input tensor.")     \
      .add_argument("x2", "Tensor", "The second input tensor.")

#define RELAX_REGISTER_BINARY_BROADCAST_OP_AND_IMPL(OpName)             \
  RELAX_REGISTER_BINARY_OP_AND_IMPL(OpName).set_attr<FInferStructInfo>( \
      "FInferStructInfo", InferStructInfoBroadcastArith)

#define RELAX_REGISTER_CMP_OP_AND_IMPL(OpName)                          \
  RELAX_REGISTER_BINARY_OP_AND_IMPL(OpName).set_attr<FInferStructInfo>( \
      "FInferStructInfo", InferStructInfoBroadcastCMP)

/***************** Arithmetic operators *****************/

/*! \brief Addition with numpy-style broadcasting. */
Expr add(Expr x1, Expr x2);

/*! \brief Multiplication with numpy-style broadcasting. */
Expr multiply(Expr x1, Expr x2);


}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_OP_TENSOR_BINARY_H_
