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
 * \file sort.cc
 * \brief sorting operators.
 */

#include "sort.h"

#include <algorithm>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

namespace tvm {
namespace relax {

/* relax.sort */
TVM_REGISTER_NODE_TYPE(SortAttrs);

Expr sort(Expr data, Optional<Integer> axis, Optional<Bool> descending) {
  auto attrs = make_object<SortAttrs>();
  attrs->axis = std::move(axis);
  attrs->descending = std::move(descending);

  static const Op& op = Op::Get("relax.sort");
  return Call(op, {std::move(data)}, Attrs{attrs}, {});
}

TVM_REGISTER_GLOBAL("relax.op.sort").set_body_typed(sort);

StructInfo InferStructInfoSort(const Call& call, const BlockBuilder& ctx) {
  TensorStructInfo data_sinfo = GetUnaryInputTensorStructInfo(call, ctx);
  const auto* attrs = call->attrs.as<SortAttrs>();

  if (attrs->axis.defined()) return data_sinfo;

  // flattened
  const auto* data_shape = data_sinfo->shape.as<ShapeExprNode>();
  if (data_shape == nullptr) {
    return TensorStructInfo(data_sinfo->dtype, data_sinfo->ndim, data_sinfo->vdevice);
  } else {
    PrimExpr flattened_d = 1;
    for (const auto v : data_shape->values) {
      flattened_d *= v;
    }
    return TensorStructInfo(ShapeExpr(Array<PrimExpr>({flattened_d})), data_sinfo->dtype,
                            data_sinfo->vdevice);
  }
}

TVM_REGISTER_OP("relax.sort")
    .set_attrs_type<SortAttrs>()
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor.")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoSort)
    .set_attr<Bool>("FPurity", Bool(true));

}  // namespace relax
}  // namespace tvm
