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
 * \file set.cc
 * \brief Relax set operators.
 */

#include "set.h"

#include <utility>
#include <vector>

namespace tvm {
namespace relax {

/* relax.unique */
TVM_REGISTER_NODE_TYPE(UniqueAttrs);

Expr unique(Expr x, bool sorted, bool return_index, bool return_inverse, bool return_counts,
            Optional<Integer> axis) {
  ObjectPtr<UniqueAttrs> attrs = make_object<UniqueAttrs>();
  attrs->sorted = sorted;
  attrs->return_index = return_index;
  attrs->return_inverse = return_inverse;
  attrs->return_counts = return_counts;
  attrs->axis = std::move(axis);

  static const Op& op = Op::Get("relax.unique");
  return Call(op, {std::move(x)}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relax.op.unique").set_body_typed(unique);

StructInfo InferStructInfoUnique(const Call& call, const BlockBuilder& ctx) {
  TensorStructInfo data_sinfo = GetUnaryInputTensorStructInfo(call, ctx);
  const auto* attrs = call->attrs.as<UniqueAttrs>();
  if (!data_sinfo->IsUnknownNdim() && attrs->axis.defined()) {
    // Normalize the axis for sanity check purpose.
    NormalizeAxis(call, ctx, data_sinfo->ndim, attrs->axis.value()->value);
  }

  int n_int_return = static_cast<int>(attrs->return_index) +
                     static_cast<int>(attrs->return_inverse) +
                     static_cast<int>(attrs->return_counts);

  std::vector<StructInfo> output_sinfo;
  output_sinfo.reserve(1 + n_int_return);

  // unique values
  if (data_sinfo->ndim == 0) {
    output_sinfo.push_back(
        TensorStructInfo(ShapeExpr({IntImm(DataType::Int(64), /*value=*/1)}), data_sinfo->dtype));
  } else if (attrs->axis.defined()) {
    output_sinfo.push_back(TensorStructInfo(data_sinfo->dtype, data_sinfo->ndim));
  } else {
    output_sinfo.push_back(TensorStructInfo(data_sinfo->dtype, /*ndim=*/1));
  }

  // index, reverse and counts
  TensorStructInfo int_return{nullptr};
  if (data_sinfo->ndim == 0) {
    int_return =
        TensorStructInfo(ShapeExpr({IntImm(DataType::Int(64), /*value=*/1)}), DataType::Int(64));
  } else {
    int_return = TensorStructInfo(DataType::Int(64), /*ndim=*/1);
  }
  for (int i = 0; i < n_int_return; ++i) {
    output_sinfo.push_back(int_return);
  }

  if (output_sinfo.size() == 1) {
    return output_sinfo[0];
  } else {
    return TupleStructInfo(output_sinfo);
  }
}

TVM_REGISTER_OP("relax.unique")
    .set_attrs_type<UniqueAttrs>()
    .set_num_inputs(1)
    .add_argument("x", "Tensor", "The input tensor")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoUnique)
    .set_attr<FCallPacked>("FCallPacked", "relax.run.unique");

}  // namespace relax
}  // namespace tvm
