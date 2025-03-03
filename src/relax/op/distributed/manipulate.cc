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

#include "manipulate.h"

#include <algorithm>
#include <numeric>
#include <string>
#include <utility>
#include <vector>
namespace tvm {
namespace relax {
namespace distributed {

StructInfo InferDistStructInfoPermuteDims(const Call& call, const BlockBuilder& ctx) {
  Array<distributed::DTensorStructInfo> input_dtensor_sinfos = GetInputDTensorStructInfo(call, ctx);
  TensorStructInfo data_sinfo = input_dtensor_sinfos[0]->tensor_sinfo;

  const auto* attrs = call->attrs.as<PermuteDimsAttrs>();

  // Todo(relax-team): revisit here for better check on if the input tensor has
  // ndim same as the number of input axes.
  if (!attrs->axes.defined() && data_sinfo->IsUnknownNdim()) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "Input of distributed operator must have known ndim");
  }

  if (attrs->axes.defined()) {
    int n_axis = attrs->axes.value().size();
    if (!data_sinfo->IsUnknownNdim() && n_axis != data_sinfo->ndim) {
      ctx->ReportFatal(Diagnostic::Error(call)
                       << "PermuteDims expects the number of input axes to equal the ndim of the "
                          "input tensor. However, the tensor ndim is "
                       << data_sinfo->ndim << " while the given number of axes is " << n_axis);
    }
  }

  std::vector<int> axes;
  if (attrs->axes.defined()) {
    axes = NormalizeAxes(call, ctx, data_sinfo->ndim, attrs->axes.value());
  } else {
    // Construct the reverse permutation via std::iota
    axes.resize(data_sinfo->ndim);
    std::iota(axes.rbegin(), axes.rend(), 0);
  }
  if (IsIdentityPermutation(axes)) {
    return input_dtensor_sinfos[0];
  }

  const auto* data_shape = data_sinfo->shape.as<ShapeExprNode>();
  if (data_shape == nullptr) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "Input of distributed operator must have known shape");
  }
  std::vector<PrimExpr> new_shape;
  new_shape.reserve(data_sinfo->ndim);
  for (int i = 0; i < data_sinfo->ndim; ++i) {
    new_shape.push_back(data_shape->values[axes[i]]);
  }
  TensorStructInfo output_tensor_sinfo(ShapeExpr(new_shape), data_sinfo->dtype);
  return InferShardingSpec(call, ctx, output_tensor_sinfo, distributed::BuildAxisGraphPermuteDims);
}

TVM_REGISTER_OP("relax.permute_dims")
    .set_attr<FInferStructInfo>("dist.FInferStructInfo", InferDistStructInfoPermuteDims);

StructInfo InferDistStructInfoReshape(const Call& call, const BlockBuilder& ctx) {
  if (call->args.size() != 2) {
    ctx->ReportFatal(Diagnostic::Error(call) << "Reshape op should take 2 arguments");
  }
  Array<distributed::DTensorStructInfo> input_dtensor_sinfos = GetInputDTensorStructInfo(call, ctx);
  TensorStructInfo data_sinfo = input_dtensor_sinfos[0]->tensor_sinfo;

  const auto* new_shape_sinfo = GetStructInfoAs<ShapeStructInfoNode>(call->args[1]);
  if (!data_sinfo.defined()) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "Reshape requires the input data to be Tensor. However, the given one is "
                     << call->args[0]->struct_info_->GetTypeKey());
  }
  if (new_shape_sinfo == nullptr) {
    ctx->ReportFatal(
        Diagnostic::Error(call)
        << "Reshape requires the input new shape to be Shape. However, the given one is "
        << call->args[1]->struct_info_->GetTypeKey());
  }

  Optional<Array<PrimExpr>> old_shape_values;
  if (data_sinfo->shape.defined()) {
    const auto* old_shape_sinfo = GetStructInfoAs<ShapeStructInfoNode>(data_sinfo->shape.value());
    ICHECK_NOTNULL(old_shape_sinfo);
    old_shape_values = old_shape_sinfo->values;
  }

  if (new_shape_sinfo->values.defined() && old_shape_values.defined()) {
    PrimExpr new_shape_prod = ComputeShapeProduct(new_shape_sinfo->values.value());
    PrimExpr old_shape_prod = ComputeShapeProduct(old_shape_values.value());
    if (ctx->GetAnalyzer()->CanProve(old_shape_prod != new_shape_prod)) {
      ctx->ReportFatal(Diagnostic::Error(call)
                       << "Reshape expects the new shape to be convertible from the old shape. "
                          "However, the old shape is "
                       << data_sinfo->shape << ", with product " << old_shape_prod
                       << ", while the new shape is " << call->args[1] << ", with product "
                       << new_shape_prod);
    }
  }
  Expr target_shape = call->args[1];
  TensorStructInfo output_tensor_sinfo;
  // If shape values are defined, use them
  if (target_shape->IsInstance<VarNode>() && new_shape_sinfo->values.defined()) {
    output_tensor_sinfo =
        TensorStructInfo(ShapeExpr(new_shape_sinfo->values.value()), data_sinfo->dtype);
  } else {
    output_tensor_sinfo = TensorStructInfo(target_shape, data_sinfo->dtype);
  }
  return InferShardingSpec(call, ctx, output_tensor_sinfo, distributed::BuildAxisGraphReshape);
}

TVM_REGISTER_OP("relax.reshape")
    .set_attr<FInferStructInfo>("dist.FInferStructInfo", InferDistStructInfoReshape);

}  // namespace distributed
}  // namespace relax
}  // namespace tvm
