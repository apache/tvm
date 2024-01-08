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
 * \file sorting.cc
 * \brief sorting operators.
 */

#include "sorting.h"

#include <vector>

namespace tvm {
namespace relax {

/* relax.sort */
TVM_REGISTER_NODE_TYPE(SortAttrs);

Expr sort(Expr data, int axis, bool descending) {
  auto attrs = make_object<SortAttrs>();
  attrs->axis = std::move(axis);
  attrs->descending = std::move(descending);

  static const Op& op = Op::Get("relax.sort");
  return Call(op, {std::move(data)}, Attrs{attrs}, {});
}

TVM_REGISTER_GLOBAL("relax.op.sort").set_body_typed(sort);

StructInfo InferStructInfoSort(const Call& call, const BlockBuilder& ctx) {
  return GetUnaryInputTensorStructInfo(call, ctx);
}

TVM_REGISTER_OP("relax.sort")
    .set_attrs_type<SortAttrs>()
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor.")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoSort)
    .set_attr<Bool>("FPurity", Bool(true));

/* relax.argsort */
TVM_REGISTER_NODE_TYPE(ArgsortAttrs);

Expr argsort(Expr data, int axis, bool descending, DataType dtype) {
  auto attrs = make_object<ArgsortAttrs>();
  attrs->axis = std::move(axis);
  attrs->descending = std::move(descending);
  attrs->dtype = std::move(dtype);

  static const Op& op = Op::Get("relax.argsort");
  return Call(op, {std::move(data)}, Attrs{attrs}, {});
}

TVM_REGISTER_GLOBAL("relax.op.argsort").set_body_typed(argsort);

StructInfo InferStructInfoArgsort(const Call& call, const BlockBuilder& ctx) {
  TensorStructInfo data_sinfo = GetUnaryInputTensorStructInfo(call, ctx);
  const auto* attrs = call->attrs.as<ArgsortAttrs>();
  DataType out_type = attrs->dtype.is_void() ? data_sinfo->dtype : attrs->dtype;
  if (data_sinfo->shape.defined()) {
    return TensorStructInfo(data_sinfo->shape.value(), out_type, data_sinfo->vdevice);
  }
  return TensorStructInfo(out_type, data_sinfo->ndim, data_sinfo->vdevice);
}

TVM_REGISTER_OP("relax.argsort")
    .set_attrs_type<ArgsortAttrs>()
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor.")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoArgsort)
    .set_attr<Bool>("FPurity", Bool(true));

/* relax.topk */
TVM_REGISTER_NODE_TYPE(TopKAttrs);

Expr topk(Expr data, int k, int axis, String ret_type, bool largest, DataType dtype) {
  auto attrs = make_object<TopKAttrs>();
  attrs->k = std::move(k);
  attrs->axis = std::move(axis);
  attrs->ret_type = std::move(ret_type);
  attrs->largest = std::move(largest);
  attrs->dtype = std::move(dtype);

  static const Op& op = Op::Get("relax.topk");
  return Call(op, {std::move(data)}, Attrs{attrs}, {});
}

TVM_REGISTER_GLOBAL("relax.op.topk").set_body_typed(topk);

StructInfo InferStructInfoTopK(const Call& call, const BlockBuilder& ctx) {
  TensorStructInfo data_sinfo = GetUnaryInputTensorStructInfo(call, ctx);
  const auto* data_shape = data_sinfo->shape.as<ShapeExprNode>();
  const auto* attrs = call->attrs.as<TopKAttrs>();
  DataType indices_type = attrs->dtype.is_void() ? data_sinfo->dtype : attrs->dtype;
  int ndim = data_sinfo->ndim;
  int k = attrs->k;
  String ret_type = attrs->ret_type;
  int axis = attrs->axis;
  if (axis < 0 && ndim > 0) {
    axis += ndim;
  }

  std::vector<StructInfo> output_sinfos;
  output_sinfos.reserve(2);
  if (data_shape == nullptr) {
    output_sinfos.push_back(
        TensorStructInfo(data_sinfo->dtype, data_sinfo->ndim, data_sinfo->vdevice));
    output_sinfos.push_back(TensorStructInfo(indices_type, data_sinfo->ndim, data_sinfo->vdevice));
  } else {
    Array<PrimExpr> out_shape = data_shape->values;
    const auto* int_dim = out_shape[axis].as<IntImmNode>();
    if (k > 0 && (int_dim == nullptr || k < int_dim->value)) {
      out_shape.Set(axis, k);
    }
    output_sinfos.push_back(
        TensorStructInfo(ShapeExpr(out_shape), data_sinfo->dtype, data_sinfo->vdevice));
    output_sinfos.push_back(
        TensorStructInfo(ShapeExpr(out_shape), indices_type, data_sinfo->vdevice));
  }

  if (ret_type == "both") {
    return TupleStructInfo(output_sinfos);
  } else if (ret_type == "values") {
    return output_sinfos[0];
  } else if (ret_type == "indices") {
    return output_sinfos[1];
  }
  LOG(FATAL) << "Unsupported ret type: " << ret_type;
}

TVM_REGISTER_OP("relax.topk")
    .set_attrs_type<TopKAttrs>()
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor.")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoTopK)
    .set_attr<Bool>("FPurity", Bool(true));

}  // namespace relax
}  // namespace tvm
