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

StructInfo InferDistStructInfoUnaryCheck(const Call& call, const BlockBuilder& ctx) {
  return InferDistStructInfoUnary<false>(
      call, ctx, [](const TensorStructInfo& input_sinfo) { return DataType::Bool(); });
}

RELAX_REGISTER_UNARY_ARITH_DIST_INFER_STRUCT_INFO(abs, /*require_float_dtype=*/false);
RELAX_REGISTER_UNARY_ARITH_DIST_INFER_STRUCT_INFO(acos, /*require_float_dtype=*/true);
RELAX_REGISTER_UNARY_ARITH_DIST_INFER_STRUCT_INFO(acosh, /*require_float_dtype=*/true);
RELAX_REGISTER_UNARY_ARITH_DIST_INFER_STRUCT_INFO(asin, /*require_float_dtype=*/true);
RELAX_REGISTER_UNARY_ARITH_DIST_INFER_STRUCT_INFO(asinh, /*require_float_dtype=*/true);
RELAX_REGISTER_UNARY_ARITH_DIST_INFER_STRUCT_INFO(atan, /*require_float_dtype=*/true);
RELAX_REGISTER_UNARY_ARITH_DIST_INFER_STRUCT_INFO(atanh, /*require_float_dtype=*/true);
RELAX_REGISTER_UNARY_ARITH_DIST_INFER_STRUCT_INFO(bitwise_not, /*require_float_dtype=*/false);
RELAX_REGISTER_UNARY_ARITH_DIST_INFER_STRUCT_INFO(ceil, /*require_float_dtype=*/false);
RELAX_REGISTER_UNARY_ARITH_DIST_INFER_STRUCT_INFO(cos, /*require_float_dtype=*/true);
RELAX_REGISTER_UNARY_ARITH_DIST_INFER_STRUCT_INFO(cosh, /*require_float_dtype=*/true);
RELAX_REGISTER_UNARY_ARITH_DIST_INFER_STRUCT_INFO(exp, /*require_float_dtype=*/true);
RELAX_REGISTER_UNARY_ARITH_DIST_INFER_STRUCT_INFO(floor, /*require_float_dtype=*/false);
RELAX_REGISTER_UNARY_ARITH_DIST_INFER_STRUCT_INFO(log, /*require_float_dtype=*/true);
RELAX_REGISTER_UNARY_ARITH_DIST_INFER_STRUCT_INFO(logical_not, /*require_float_dtype=*/false);
RELAX_REGISTER_UNARY_ARITH_DIST_INFER_STRUCT_INFO(negative, /*require_float_dtype=*/false);
RELAX_REGISTER_UNARY_ARITH_DIST_INFER_STRUCT_INFO(round, /*require_float_dtype=*/false);
RELAX_REGISTER_UNARY_ARITH_DIST_INFER_STRUCT_INFO(rsqrt, /*require_float_dtype=*/true);
RELAX_REGISTER_UNARY_ARITH_DIST_INFER_STRUCT_INFO(sigmoid, /*require_float_dtype=*/true);
RELAX_REGISTER_UNARY_ARITH_DIST_INFER_STRUCT_INFO(sign, /*require_float_dtype=*/false);
RELAX_REGISTER_UNARY_ARITH_DIST_INFER_STRUCT_INFO(sin, /*require_float_dtype=*/true);
RELAX_REGISTER_UNARY_ARITH_DIST_INFER_STRUCT_INFO(sinh, /*require_float_dtype=*/true);
RELAX_REGISTER_UNARY_ARITH_DIST_INFER_STRUCT_INFO(square, /*require_float_dtype=*/false);
RELAX_REGISTER_UNARY_ARITH_DIST_INFER_STRUCT_INFO(sqrt, /*require_float_dtype=*/true);
RELAX_REGISTER_UNARY_ARITH_DIST_INFER_STRUCT_INFO(tan, /*require_float_dtype=*/true);
RELAX_REGISTER_UNARY_ARITH_DIST_INFER_STRUCT_INFO(tanh, /*require_float_dtype=*/true);
RELAX_REGISTER_UNARY_ARITH_DIST_INFER_STRUCT_INFO(erf, /*require_float_dtype=*/true);

RELAX_REGISTER_UNARY_CHECK_DIST_INFER_STRUCT_INFO(isfinite);
RELAX_REGISTER_UNARY_CHECK_DIST_INFER_STRUCT_INFO(isinf);
RELAX_REGISTER_UNARY_CHECK_DIST_INFER_STRUCT_INFO(isnan);

}  // namespace distributed
}  // namespace relax
}  // namespace tvm
