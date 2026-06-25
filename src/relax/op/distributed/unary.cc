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

#include "unary.h"

namespace tvm {
namespace relax {
namespace distributed {

Type InferDistTypeUnaryCheck(const Call& call, const BlockBuilder& ctx) {
  return InferDistTypeUnary<false>(call, ctx,
                                   [](const TensorType& input_ty) { return PrimType::Bool(); });
}

RELAX_REGISTER_UNARY_ARITH_DIST_INFER_TYPE(abs, /*require_float_dtype=*/false);
RELAX_REGISTER_UNARY_ARITH_DIST_INFER_TYPE(acos, /*require_float_dtype=*/true);
RELAX_REGISTER_UNARY_ARITH_DIST_INFER_TYPE(acosh, /*require_float_dtype=*/true);
RELAX_REGISTER_UNARY_ARITH_DIST_INFER_TYPE(asin, /*require_float_dtype=*/true);
RELAX_REGISTER_UNARY_ARITH_DIST_INFER_TYPE(asinh, /*require_float_dtype=*/true);
RELAX_REGISTER_UNARY_ARITH_DIST_INFER_TYPE(atan, /*require_float_dtype=*/true);
RELAX_REGISTER_UNARY_ARITH_DIST_INFER_TYPE(atanh, /*require_float_dtype=*/true);
RELAX_REGISTER_UNARY_ARITH_DIST_INFER_TYPE(bitwise_not, /*require_float_dtype=*/false);
RELAX_REGISTER_UNARY_ARITH_DIST_INFER_TYPE(ceil, /*require_float_dtype=*/false);
RELAX_REGISTER_UNARY_ARITH_DIST_INFER_TYPE(cos, /*require_float_dtype=*/true);
RELAX_REGISTER_UNARY_ARITH_DIST_INFER_TYPE(cosh, /*require_float_dtype=*/true);
RELAX_REGISTER_UNARY_ARITH_DIST_INFER_TYPE(exp, /*require_float_dtype=*/true);
RELAX_REGISTER_UNARY_ARITH_DIST_INFER_TYPE(floor, /*require_float_dtype=*/false);
RELAX_REGISTER_UNARY_ARITH_DIST_INFER_TYPE(log, /*require_float_dtype=*/true);
RELAX_REGISTER_UNARY_ARITH_DIST_INFER_TYPE(logical_not, /*require_float_dtype=*/false);
RELAX_REGISTER_UNARY_ARITH_DIST_INFER_TYPE(negative, /*require_float_dtype=*/false);
RELAX_REGISTER_UNARY_ARITH_DIST_INFER_TYPE(round, /*require_float_dtype=*/false);
RELAX_REGISTER_UNARY_ARITH_DIST_INFER_TYPE(rsqrt, /*require_float_dtype=*/true);
RELAX_REGISTER_UNARY_ARITH_DIST_INFER_TYPE(sigmoid, /*require_float_dtype=*/true);
RELAX_REGISTER_UNARY_ARITH_DIST_INFER_TYPE(sign, /*require_float_dtype=*/false);
RELAX_REGISTER_UNARY_ARITH_DIST_INFER_TYPE(sin, /*require_float_dtype=*/true);
RELAX_REGISTER_UNARY_ARITH_DIST_INFER_TYPE(sinh, /*require_float_dtype=*/true);
RELAX_REGISTER_UNARY_ARITH_DIST_INFER_TYPE(square, /*require_float_dtype=*/false);
RELAX_REGISTER_UNARY_ARITH_DIST_INFER_TYPE(sqrt, /*require_float_dtype=*/true);
RELAX_REGISTER_UNARY_ARITH_DIST_INFER_TYPE(tan, /*require_float_dtype=*/true);
RELAX_REGISTER_UNARY_ARITH_DIST_INFER_TYPE(tanh, /*require_float_dtype=*/true);
RELAX_REGISTER_UNARY_ARITH_DIST_INFER_TYPE(erf, /*require_float_dtype=*/true);

RELAX_REGISTER_UNARY_CHECK_DIST_INFER_TYPE(isfinite);
RELAX_REGISTER_UNARY_CHECK_DIST_INFER_TYPE(isinf);
RELAX_REGISTER_UNARY_CHECK_DIST_INFER_TYPE(isnan);

}  // namespace distributed
}  // namespace relax
}  // namespace tvm
