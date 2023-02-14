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
 * \file search.cc
 * \brief Searching operators.
 */

#include "search.h"

#include <algorithm>
#include <utility>

namespace tvm {
namespace relax {

/* relax.where */
Expr where(Expr condition, Expr x1, Expr x2) {
  static const Op& op = Op::Get("relax.where");
  return Call(op, {std::move(condition), std::move(x1), std::move(x2)}, Attrs(), {});
}

TVM_REGISTER_GLOBAL("relax.op.where").set_body_typed(where);

StructInfo InferStructInfoWhere(const Call& call, const BlockBuilder& ctx) {
  Array<TensorStructInfo> input_sinfo = GetInputTensorStructInfo(call, ctx);
  TensorStructInfo cond_sinfo = input_sinfo[0];
  TensorStructInfo x1_sinfo = input_sinfo[1];
  TensorStructInfo x2_sinfo = input_sinfo[2];

  if (!cond_sinfo->dtype.is_bool()) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "Where requires the input condition tensor to have boolean dtype. However, "
                        "the given condition dtype is "
                     << cond_sinfo->dtype);
  }
  DataType output_dtype = InferBinaryArithOpOutDtype(call, ctx, x1_sinfo, x2_sinfo);

  int output_ndim;
  if (cond_sinfo->IsUnknownNdim() || x1_sinfo->IsUnknownNdim() || x2_sinfo->IsUnknownNdim()) {
    output_ndim = kUnknownNDim;
  } else {
    output_ndim = std::max(cond_sinfo->ndim, std::max(x1_sinfo->ndim, x2_sinfo->ndim));
  }

  const auto* cond_shape = cond_sinfo->shape.as<ShapeExprNode>();
  const auto* x1_shape = x1_sinfo->shape.as<ShapeExprNode>();
  const auto* x2_shape = x2_sinfo->shape.as<ShapeExprNode>();
  if (cond_shape && x1_shape && x2_shape) {
    // Step 1. Compute the broadcasted shape of x1's and x2's
    Optional<Array<PrimExpr>> broadcasted_shape =
        InferBinaryBroadcastShape(call, ctx, x1_shape->values, x2_shape->values);
    if (!broadcasted_shape.defined()) {
      return TensorStructInfo(output_dtype, output_ndim);
    }
    // Step 2. Compute the broadcasted shape of cond's and the previous broadcasted shape.
    broadcasted_shape =
        InferBinaryBroadcastShape(call, ctx, cond_shape->values, broadcasted_shape.value());
    if (!broadcasted_shape.defined()) {
      return TensorStructInfo(output_dtype, output_ndim);
    }
    ICHECK_EQ(static_cast<int>(broadcasted_shape.value().size()), output_ndim);
    return TensorStructInfo(ShapeExpr(broadcasted_shape.value()), output_dtype);
  } else if (cond_sinfo->shape.defined() &&                 //
             x1_sinfo->shape.defined() &&                   //
             x2_sinfo->shape.defined() &&                   //
             cond_sinfo->shape.same_as(x1_sinfo->shape) &&  //
             cond_sinfo->shape.same_as(x2_sinfo->shape)) {
    return TensorStructInfo(cond_sinfo->shape.value(), output_dtype);
  } else {
    return TensorStructInfo(output_dtype, output_ndim);
  }
}

TVM_REGISTER_OP("relax.where")
    .set_num_inputs(3)
    .add_argument("condition", "Tensor", "When True, yield `x1`; otherwise, yield `x2`.")
    .add_argument("x1", "Tensor", "The first input tensor.")
    .add_argument("x2", "Tensor", "The second input tensor.")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoWhere);

}  // namespace relax
}  // namespace tvm
