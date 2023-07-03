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

StructInfo InferDistStructInfoBroadcastArith(const Call& call, const BlockBuilder& ctx) {
  return InferDistStructInfoBroadcast(call, ctx, InferBinaryArithOpOutDtype);
}

StructInfo InferDistStructInfoBroadcastCMP(const Call& call, const BlockBuilder& ctx) {
  return InferDistStructInfoBroadcast(
      call, ctx,
      [](const Call& call, const BlockBuilder& ctx, const TensorStructInfo& x1_sinfo,
         const TensorStructInfo& x2_sinfo) { return DataType::Bool(); });
}

/***************** Arithmetic operators *****************/

RELAX_REGISTER_BINARY_BROADCAST_DIST_INFER_STRUCT_INFO(add);
RELAX_REGISTER_BINARY_BROADCAST_DIST_INFER_STRUCT_INFO(divide);
RELAX_REGISTER_BINARY_BROADCAST_DIST_INFER_STRUCT_INFO(floor_divide);
RELAX_REGISTER_BINARY_BROADCAST_DIST_INFER_STRUCT_INFO(multiply);
RELAX_REGISTER_BINARY_BROADCAST_DIST_INFER_STRUCT_INFO(power);
RELAX_REGISTER_BINARY_BROADCAST_DIST_INFER_STRUCT_INFO(subtract);

/***************** Comparison operators *****************/

RELAX_REGISTER_CMP_DIST_INFER_STRUCT_INFO(equal);
RELAX_REGISTER_CMP_DIST_INFER_STRUCT_INFO(greater);
RELAX_REGISTER_CMP_DIST_INFER_STRUCT_INFO(greater_equal);
RELAX_REGISTER_CMP_DIST_INFER_STRUCT_INFO(less);
RELAX_REGISTER_CMP_DIST_INFER_STRUCT_INFO(less_equal);
RELAX_REGISTER_CMP_DIST_INFER_STRUCT_INFO(not_equal);

/***************** Min/Max operators *****************/

RELAX_REGISTER_BINARY_BROADCAST_DIST_INFER_STRUCT_INFO(minimum);
RELAX_REGISTER_BINARY_BROADCAST_DIST_INFER_STRUCT_INFO(maximum);

/***************** Logical operators *****************/

RELAX_REGISTER_BINARY_BROADCAST_DIST_INFER_STRUCT_INFO(logical_and);
RELAX_REGISTER_BINARY_BROADCAST_DIST_INFER_STRUCT_INFO(logical_or);
RELAX_REGISTER_BINARY_BROADCAST_DIST_INFER_STRUCT_INFO(logical_xor);

/***************** Bitwise operators *****************/

RELAX_REGISTER_BINARY_BROADCAST_DIST_INFER_STRUCT_INFO(bitwise_and);
RELAX_REGISTER_BINARY_BROADCAST_DIST_INFER_STRUCT_INFO(bitwise_or);
RELAX_REGISTER_BINARY_BROADCAST_DIST_INFER_STRUCT_INFO(bitwise_xor);

}  // namespace distributed
}  // namespace relax
}  // namespace tvm
