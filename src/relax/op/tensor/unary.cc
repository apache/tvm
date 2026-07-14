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

#include <tvm/ffi/reflection/registry.h>

#include <utility>

namespace tvm {
namespace relax {

Type InferTypeUnaryCheck(const Call& call, const BlockBuilder& ctx) {
  return InferTypeUnary<false>(call, ctx,
                               [](const TensorType& input_ty) { return PrimType::Bool(); });
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
RELAX_REGISTER_UNARY_ARITH_OP_AND_IMPL(trunc, /*require_float_dtype=*/false);
RELAX_REGISTER_UNARY_ARITH_OP_AND_IMPL(erf, /*require_float_dtype=*/true);

// relax.clip
TVM_REGISTER_OP("relax.clip")
    .set_num_inputs(3)
    .add_argument("x", "Tensor", "The input tensor.")
    .add_argument("min", "PrimExpr", "The lower-bound of the range to be clipped to")
    .add_argument("max", "PrimExpr", "The upper-bound of the range to be clipped to")
    .set_attr<FInferType>("FInferType", ReturnTypeFromArg<0>)
    .set_attr<bool>("FPurity", true);

Expr clip(Expr x, Expr min, Expr max) {
  TVM_FFI_ICHECK(min.as<PrimExpr>())
      << "The argument `min` of relax.clip is expected to be a PrimExpr, but got "
      << min->GetTypeKey();
  TVM_FFI_ICHECK(max.as<PrimExpr>())
      << "The argument `max` of relax.clip is expected to be a PrimExpr, but got "
      << max->GetTypeKey();
  static const Op& op = Op::Get("relax.clip");
  return Call(Type::Missing(), op, {std::move(x), std::move(min), std::move(max)});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.clip", clip);
}

/***************** Check operators *****************/

RELAX_REGISTER_UNARY_CHECK_OP_AND_IMPL(isfinite);
RELAX_REGISTER_UNARY_CHECK_OP_AND_IMPL(isinf);
RELAX_REGISTER_UNARY_CHECK_OP_AND_IMPL(isnan);

}  // namespace relax
}  // namespace tvm
