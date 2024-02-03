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

RELAX_REGISTER_UNARY_ARITH_OP_AND_IMPL(abs, /*require_float_dtype=*/false);
RELAX_REGISTER_UNARY_ARITH_OP_AND_IMPL(acos, /*require_float_dtype=*/true);
RELAX_REGISTER_UNARY_ARITH_OP_AND_IMPL(acosh, /*require_float_dtype=*/true);
RELAX_REGISTER_UNARY_ARITH_OP_AND_IMPL(asin, /*require_float_dtype=*/true);
RELAX_REGISTER_UNARY_ARITH_OP_AND_IMPL(asinh, /*require_float_dtype=*/true);
RELAX_REGISTER_UNARY_ARITH_OP_AND_IMPL(atan, /*require_float_dtype=*/true);
RELAX_REGISTER_UNARY_ARITH_OP_AND_IMPL(atanh, /*require_float_dtype=*/true);
RELAX_REGISTER_UNARY_ARITH_OP_AND_IMPL(bitwise_not, /*require_float_dtype=*/false);
RELAX_REGISTER_UNARY_ARITH_OP_AND_IMPL(ceil, /*require_float_dtype=*/false);
RELAX_REGISTER_UNARY_ARITH_OP_AND_IMPL(cos, /*require_float_dtype=*/true);
RELAX_REGISTER_UNARY_ARITH_OP_AND_IMPL(cosh, /*require_float_dtype=*/true);
RELAX_REGISTER_UNARY_ARITH_OP_AND_IMPL(exp, /*require_float_dtype=*/true);
RELAX_REGISTER_UNARY_ARITH_OP_AND_IMPL(floor, /*require_float_dtype=*/false);
RELAX_REGISTER_UNARY_ARITH_OP_AND_IMPL(log, /*require_float_dtype=*/true);
RELAX_REGISTER_UNARY_ARITH_OP_AND_IMPL(logical_not, /*require_float_dtype=*/false);
RELAX_REGISTER_UNARY_ARITH_OP_AND_IMPL(negative, /*require_float_dtype=*/false);
RELAX_REGISTER_UNARY_ARITH_OP_AND_IMPL(round, /*require_float_dtype=*/false);
RELAX_REGISTER_UNARY_ARITH_OP_AND_IMPL(rsqrt, /*require_float_dtype=*/true);
RELAX_REGISTER_UNARY_ARITH_OP_AND_IMPL(sigmoid, /*require_float_dtype=*/true);
RELAX_REGISTER_UNARY_ARITH_OP_AND_IMPL(sign, /*require_float_dtype=*/false);
RELAX_REGISTER_UNARY_ARITH_OP_AND_IMPL(sin, /*require_float_dtype=*/true);
RELAX_REGISTER_UNARY_ARITH_OP_AND_IMPL(sinh, /*require_float_dtype=*/true);
RELAX_REGISTER_UNARY_ARITH_OP_AND_IMPL(square, /*require_float_dtype=*/false);
RELAX_REGISTER_UNARY_ARITH_OP_AND_IMPL(sqrt, /*require_float_dtype=*/true);
RELAX_REGISTER_UNARY_ARITH_OP_AND_IMPL(tan, /*require_float_dtype=*/true);
RELAX_REGISTER_UNARY_ARITH_OP_AND_IMPL(tanh, /*require_float_dtype=*/true);
RELAX_REGISTER_UNARY_ARITH_OP_AND_IMPL(erf, /*require_float_dtype=*/true);

// relax.clip
TVM_REGISTER_OP("relax.clip")
    .set_num_inputs(3)
    .add_argument("x", "Tensor", "The input tensor.")
    .add_argument("min", "PrimValue", "The lower-bound of the range to be clipped to")
    .add_argument("max", "PrimValue", "The upper-bound of the range to be clipped to")
    .set_attr<FInferStructInfo>("FInferStructInfo", ReturnStructInfoFromArg<0>)
    .set_attr<Bool>("FPurity", Bool(true));

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

/***************** Check operators *****************/

RELAX_REGISTER_UNARY_CHECK_OP_AND_IMPL(isfinite);
RELAX_REGISTER_UNARY_CHECK_OP_AND_IMPL(isinf);
RELAX_REGISTER_UNARY_CHECK_OP_AND_IMPL(isnan);

}  // namespace relax
}  // namespace tvm
