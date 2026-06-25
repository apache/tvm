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

#include "binary.h"

namespace tvm {
namespace relax {
namespace distributed {

Type InferDistTypeBroadcastArith(const Call& call, const BlockBuilder& ctx) {
  return InferDistTypeBroadcast(call, ctx, InferBinaryArithOpOutDtype);
}

Type InferDistTypeBroadcastCMP(const Call& call, const BlockBuilder& ctx) {
  return InferDistTypeBroadcast(
      call, ctx,
      [](const Call& call, const BlockBuilder& ctx, const TensorType& x1_ty,
         const TensorType& x2_ty) { return PrimType::Bool(); });
}

/***************** Arithmetic operators *****************/

RELAX_REGISTER_BINARY_BROADCAST_DIST_INFER_TYPE(add);
RELAX_REGISTER_BINARY_BROADCAST_DIST_INFER_TYPE(divide);
RELAX_REGISTER_BINARY_BROADCAST_DIST_INFER_TYPE(floor_divide);
RELAX_REGISTER_BINARY_BROADCAST_DIST_INFER_TYPE(multiply);
RELAX_REGISTER_BINARY_BROADCAST_DIST_INFER_TYPE(power);
RELAX_REGISTER_BINARY_BROADCAST_DIST_INFER_TYPE(subtract);
RELAX_REGISTER_BINARY_BROADCAST_DIST_INFER_TYPE(mod);
RELAX_REGISTER_BINARY_BROADCAST_DIST_INFER_TYPE(floor_mod);

/***************** Comparison operators *****************/

RELAX_REGISTER_CMP_DIST_INFER_TYPE(equal);
RELAX_REGISTER_CMP_DIST_INFER_TYPE(greater);
RELAX_REGISTER_CMP_DIST_INFER_TYPE(greater_equal);
RELAX_REGISTER_CMP_DIST_INFER_TYPE(less);
RELAX_REGISTER_CMP_DIST_INFER_TYPE(less_equal);
RELAX_REGISTER_CMP_DIST_INFER_TYPE(not_equal);

/***************** Min/Max operators *****************/

RELAX_REGISTER_BINARY_BROADCAST_DIST_INFER_TYPE(minimum);
RELAX_REGISTER_BINARY_BROADCAST_DIST_INFER_TYPE(maximum);

/***************** Logical operators *****************/

RELAX_REGISTER_BINARY_BROADCAST_DIST_INFER_TYPE(logical_and);
RELAX_REGISTER_BINARY_BROADCAST_DIST_INFER_TYPE(logical_or);
RELAX_REGISTER_BINARY_BROADCAST_DIST_INFER_TYPE(logical_xor);

/***************** Bitwise operators *****************/

RELAX_REGISTER_BINARY_BROADCAST_DIST_INFER_TYPE(bitwise_and);
RELAX_REGISTER_BINARY_BROADCAST_DIST_INFER_TYPE(bitwise_or);
RELAX_REGISTER_BINARY_BROADCAST_DIST_INFER_TYPE(bitwise_xor);
RELAX_REGISTER_BINARY_BROADCAST_DIST_INFER_TYPE(left_shift);
RELAX_REGISTER_BINARY_BROADCAST_DIST_INFER_TYPE(right_shift);

}  // namespace distributed
}  // namespace relax
}  // namespace tvm
