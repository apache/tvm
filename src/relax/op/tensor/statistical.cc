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
 * \file statistical.cc
 * \brief Statistical operators.
 */

#include "statistical.h"

#include <vector>

namespace tvm {
namespace relax {

StructInfo InferStructInfoStatistical(const Call& call, const BlockBuilder& ctx) {
  TensorStructInfo data_sinfo = GetUnaryInputTensorStructInfo(call, ctx);
  const auto* attrs = call->attrs.as<StatisticalAttrs>();

  std::vector<int> axes;
  if (!data_sinfo->IsUnknownNdim() && attrs->axis.defined()) {
    axes = NormalizeAxes(call, ctx, data_sinfo->ndim, attrs->axis.value());
  }

  int out_ndim;
  if (attrs->keepdims) {
    out_ndim = data_sinfo->ndim;
  } else if (!attrs->axis.defined()) {
    out_ndim = 0;
  } else if (data_sinfo->IsUnknownNdim()) {
    out_ndim = kUnknownNDim;
  } else {
    out_ndim = data_sinfo->ndim - axes.size();
    ICHECK_GE(out_ndim, 0);
  }

  // The inference rule for reduction operator output shapes:
  // - axes is None, keepdims is false -> return the zero-rank shape;
  // - axes is None, keepdims is true -> return the shape whose ndim is the same as input and every
  // value is 1.
  // - axes is not None, keepdims is false -> the returned shape does not contain the input axes.
  // - axes is not None, keepdims is true -> the returned shape has value 1 at the positions of the
  // input axes
  const auto* data_shape = data_sinfo->shape.as<ShapeExprNode>();
  if (data_shape == nullptr) {
    if (!attrs->axis.defined() && attrs->keepdims && out_ndim != kUnknownNDim) {
      return TensorStructInfo(
          ShapeExpr(Array<PrimExpr>(out_ndim, IntImm(DataType::Int(64), /*value=*/1))),
          data_sinfo->dtype);
    } else {
      return out_ndim == 0 ? TensorStructInfo(ShapeExpr(Array<PrimExpr>()), data_sinfo->dtype)
                           : TensorStructInfo(data_sinfo->dtype, out_ndim);
    }
  }

  Array<PrimExpr> out_shape;
  out_shape.reserve(out_ndim);
  for (int i = 0; i < data_sinfo->ndim; ++i) {
    if (attrs->axis.defined() && std::find(axes.begin(), axes.end(), i) == axes.end()) {
      out_shape.push_back(data_shape->values[i]);
    } else if (attrs->keepdims) {
      out_shape.push_back(IntImm(DataType::Int(64), /*value=*/1));
    }
  }
  ICHECK_EQ(static_cast<int>(out_shape.size()), out_ndim);
  return TensorStructInfo(ShapeExpr(out_shape), data_sinfo->dtype);
}

TVM_REGISTER_NODE_TYPE(StatisticalAttrs);

RELAX_REGISTER_STATISTICAL_OP_INTERFACE(max);
RELAX_REGISTER_STATISTICAL_OP_INTERFACE(mean);
RELAX_REGISTER_STATISTICAL_OP_INTERFACE(min);
RELAX_REGISTER_STATISTICAL_OP_INTERFACE(prod);
RELAX_REGISTER_STATISTICAL_OP_INTERFACE(std);
RELAX_REGISTER_STATISTICAL_OP_INTERFACE(sum);
RELAX_REGISTER_STATISTICAL_OP_INTERFACE(variance);

}  // namespace relax
}  // namespace tvm
