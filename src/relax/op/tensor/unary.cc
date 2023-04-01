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
 * \file unary.cc
 * \brief Relax unary arithmetic operators.
 */

#include "unary.h"

#include <utility>

namespace tvm {
namespace relax {

StructInfo InferStructInfoUnaryCheck(const Call& call, const BlockBuilder& ctx) {
  return InferStructInfoUnary<false>(
      call, ctx, [](const TensorStructInfo& input_sinfo) { return DataType::Bool(); });
}

/***************** Arithmetic operators *****************/

RELAX_REGISTER_UNARY_ARITH_OP_AND_IMPL(sigmoid, /*require_float_dtype=*/true);
RELAX_REGISTER_UNARY_ARITH_OP_AND_IMPL(sign, /*require_float_dtype=*/false);

// relax.clip
TVM_REGISTER_OP("relax.clip")
    .set_num_inputs(3)
    .add_argument("x", "Tensor", "The input tensor.")
    .add_argument("min", "PrimValue", "The lower-bound of the range to be clipped to")
    .add_argument("max", "PrimValue", "The upper-bound of the range to be clipped to")
    .set_attr<FInferStructInfo>("FInferStructInfo", ReturnStructInfoFromArg<0>);

Expr clip(Expr x, Expr min, Expr max) {
  CHECK(min->IsInstance<PrimValueNode>())
      << "The argument `min` of relax.clip is expected to be a PrimValue, but got"
      << min->GetTypeKey();
  CHECK(max->IsInstance<PrimValueNode>())
      << "The argument `max` of relax.clip is expected to be a PrimValue, but got"
      << max->GetTypeKey();
  static const Op& op = Op::Get("relax.clip");
  return Call(op, {std::move(x), std::move(min), std::move(max)});
}

TVM_REGISTER_GLOBAL("relax.op.clip").set_body_typed(clip);

}  // namespace relax
}  // namespace tvm
