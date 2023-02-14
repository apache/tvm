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
 * \file binary.cc
 * \brief binary broadcast operators.
 */

#include "binary.h"

#include <algorithm>

namespace tvm {
namespace relax {

template <typename FType>
StructInfo InferStructInfoBroadcast(const Call& call, const BlockBuilder& ctx,
                                    FType f_compute_out_dtype) {
  Array<TensorStructInfo> input_sinfo = GetInputTensorStructInfo(call, ctx);
  TensorStructInfo x1_sinfo = input_sinfo[0];
  TensorStructInfo x2_sinfo = input_sinfo[1];

  // DateType
  DataType output_dtype = f_compute_out_dtype(call, ctx, x1_sinfo, x2_sinfo);

  // ndims
  int output_ndim;
  if (x1_sinfo->IsUnknownNdim() || x2_sinfo->IsUnknownNdim()) {
    output_ndim = kUnknownNDim;
  } else {
    output_ndim = std::max(x1_sinfo->ndim, x2_sinfo->ndim);
  }

  const auto* x1_shape = x1_sinfo->shape.as<ShapeExprNode>();
  const auto* x2_shape = x2_sinfo->shape.as<ShapeExprNode>();
  // Shapes and ndims
  if (x1_shape && x2_shape) {
    // If all inputs have shapes, directly infer shapes
    Optional<Array<PrimExpr>> output_shape =
        InferBinaryBroadcastShape(call, ctx, x1_shape->values, x2_shape->values);
    if (!output_shape.defined()) {
      return TensorStructInfo(output_dtype, /*ndim=*/output_ndim);
    } else {
      ICHECK_EQ(static_cast<int>(output_shape.value().size()), output_ndim);
      return TensorStructInfo(ShapeExpr(output_shape.value()), output_dtype);
    }
  } else if (x1_sinfo->shape.defined() && x1_sinfo->shape.same_as(x2_sinfo->shape)) {
    return TensorStructInfo(x1_sinfo->shape.value(), output_dtype);
  } else {
    return TensorStructInfo(output_dtype, /*ndim=*/output_ndim);
  }
}

StructInfo InferStructInfoBroadcastArith(const Call& call, const BlockBuilder& ctx) {
  return InferStructInfoBroadcast(call, ctx, InferBinaryArithOpOutDtype);
}

StructInfo InferStructInfoBroadcastCMP(const Call& call, const BlockBuilder& ctx) {
  return InferStructInfoBroadcast(
      call, ctx,
      [](const Call& call, const BlockBuilder& ctx, const TensorStructInfo& x1_sinfo,
         const TensorStructInfo& x2_sinfo) { return DataType::Bool(); });
}

/***************** Arithmetic operators *****************/

RELAX_REGISTER_BINARY_BROADCAST_OP_AND_IMPL(add);
RELAX_REGISTER_BINARY_BROADCAST_OP_AND_IMPL(divide);
RELAX_REGISTER_BINARY_BROADCAST_OP_AND_IMPL(floor_divide);
RELAX_REGISTER_BINARY_BROADCAST_OP_AND_IMPL(multiply);
RELAX_REGISTER_BINARY_BROADCAST_OP_AND_IMPL(subtract);

/***************** Comparison operators *****************/

RELAX_REGISTER_CMP_OP_AND_IMPL(equal);
RELAX_REGISTER_CMP_OP_AND_IMPL(greater);
RELAX_REGISTER_CMP_OP_AND_IMPL(greater_equal);
RELAX_REGISTER_CMP_OP_AND_IMPL(less);
RELAX_REGISTER_CMP_OP_AND_IMPL(less_equal);
RELAX_REGISTER_CMP_OP_AND_IMPL(not_equal);

}  // namespace relax
}  // namespace tvm
