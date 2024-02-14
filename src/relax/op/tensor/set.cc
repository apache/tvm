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

Expr unique(Expr x, PrimValue sorted, PrimValue return_index, PrimValue return_inverse,
            PrimValue return_counts, Optional<PrimValue> axis) {
  static const Op& op = Op::Get("relax.unique");
  Call call;
  if (!axis) {
    call = Call(op, {std::move(x), sorted, return_index, return_inverse, return_counts});
  } else {
    PrimValue pv_axis = axis.value();
    call = Call(op, {std::move(x), sorted, return_index, return_inverse, return_counts, pv_axis});
  }
  return call;
}

TVM_REGISTER_GLOBAL("relax.op.unique").set_body_typed(unique);

StructInfo InferStructInfoUnique(const Call& call, const BlockBuilder& ctx) {
  TensorStructInfo data_sinfo = Downcast<TensorStructInfo>(call->args[0]->struct_info_);
  PrimValue axis, return_index, return_inverse, return_counts;
  if (call->args.size() == 6) {
    if (auto* prim_value_node = call->args[5].as<PrimValueNode>()) {
      axis = GetRef<PrimValue>(prim_value_node);
    }
  }
  if (!data_sinfo->IsUnknownNdim() && axis.defined()) {
    // Normalize the axis for sanity check purpose.
    if (const auto* axis_int = axis->value.as<IntImmNode>()) {
      NormalizeAxis(call, ctx, data_sinfo->ndim, axis_int->value);
    }
  }
  ICHECK(call->args[2]->IsInstance<PrimValueNode>());
  ICHECK(call->args[3]->IsInstance<PrimValueNode>());
  ICHECK(call->args[4]->IsInstance<PrimValueNode>());

  return_index = Downcast<PrimValue>(call->args[2]);
  return_inverse = Downcast<PrimValue>(call->args[3]);
  return_counts = Downcast<PrimValue>(call->args[4]);

  auto f_convert_to_int64 = [](const PrimExpr& value) {
    CHECK(value->IsInstance<IntImmNode>())
        << value << " expects to be IntImm, but gets " << value->GetTypeKey();
    const auto* val_node = value.as<IntImmNode>();
    auto val_imm = GetRef<IntImm>(val_node);
    return val_imm->value;
  };

  int64_t n_int_return = f_convert_to_int64(return_index->value) +
                         f_convert_to_int64(return_inverse->value) +
                         f_convert_to_int64(return_counts->value);

  std::vector<StructInfo> output_sinfo;
  output_sinfo.reserve(1 + n_int_return);

  // unique values
  if (data_sinfo->ndim == 0) {
    output_sinfo.push_back(TensorStructInfo(ShapeExpr({IntImm(DataType::Int(64), /*value=*/1)}),
                                            data_sinfo->dtype, data_sinfo->vdevice));
  } else if (axis.defined()) {
    output_sinfo.push_back(
        TensorStructInfo(data_sinfo->dtype, data_sinfo->ndim, data_sinfo->vdevice));
  } else {
    output_sinfo.push_back(TensorStructInfo(data_sinfo->dtype, /*ndim=*/1, data_sinfo->vdevice));
  }

  // index, reverse and counts
  TensorStructInfo int_return{nullptr};
  if (data_sinfo->ndim == 0) {
    int_return = TensorStructInfo(ShapeExpr({IntImm(DataType::Int(64), /*value=*/1)}),
                                  DataType::Int(64), data_sinfo->vdevice);
  } else {
    int_return = TensorStructInfo(DataType::Int(64), /*ndim=*/1, data_sinfo->vdevice);
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
    .set_num_inputs(6)
    .add_argument("x", "Tensor", "The input tensor")
    .add_argument(
        "sorted", "Tensor",
        "Whether to sort the unique elements in ascending order before returning as output.")
    .add_argument(
        "return_index", "Tensor",
        "Whether to return an additional tensor with indices for where elements in the unique "
        "tensor come from the original input.")
    .add_argument("return_inverse", "Tensor",
                  "Whether to return an additional tensor with indices for where elements in the "
                  "original input ended up in the returned unique list.")
    .add_argument("return_counts", "Tensor",
                  "Whether to return an additional tensor with counts of each unique elements")
    .add_argument(
        "axis", "Tensor",
        "The dimension to apply unique. If it is NullOpt, the unique values of the flattened input "
        "are returned.")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoUnique)
    .set_attr<FCallPacked>("FCallPacked", "relax.run.unique")
    .set_attr<Bool>("FPurity", Bool(true));

}  // namespace relax
}  // namespace tvm
