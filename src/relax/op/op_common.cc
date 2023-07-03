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

#include "op_common.h"

#include <algorithm>

namespace tvm {
namespace relax {

Array<Expr> GetCallArgs(const Call& call) {
  static const Op& call_tir_op = Op::Get("relax.call_tir");
  Array<Expr> args;
  if (call->op.same_as(call_tir_op)) {
    args = Downcast<Tuple>(call->args[1])->fields;
  } else {
    args = call->args;
  }
  return args;
}

Array<TensorStructInfo> GetInputTensorStructInfo(const Call& call, const BlockBuilder& ctx) {
  Op op = Downcast<Op>(call->op);
  int n_input = op->arguments.size();
  if (static_cast<int>(call->args.size()) != n_input) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << op << " op should have " << n_input << " arguments");
  }
  Array<TensorStructInfo> input_tensor_sinfo;
  input_tensor_sinfo.reserve(n_input);
  for (int i = 0; i < n_input; ++i) {
    const auto* sinfo = GetStructInfoAs<TensorStructInfoNode>(call->args[i]);
    if (sinfo == nullptr) {
      ctx->ReportFatal(Diagnostic::Error(call)
                       << op << " requires the input " << op->arguments[i]->name
                       << " to be Tensor. However, the given one has a "
                       << call->args[i]->struct_info_->GetTypeKey());
    }
    input_tensor_sinfo.push_back(GetRef<TensorStructInfo>(sinfo));
  }
  return input_tensor_sinfo;
}

Array<TensorStructInfo> GetTensorStructInfoFromTuple(const Call& call, const BlockBuilder& ctx,
                                                     const Expr& tup) {
  const auto* tuple_sinfo = GetStructInfoAs<TupleStructInfoNode>(tup);
  if (tuple_sinfo == nullptr) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << call->op
                     << " expects the input to be a Tuple of Tensors. However, the given input is "
                     << tup->struct_info_->GetTypeKey());
  }

  Array<TensorStructInfo> tensor_sinfo;
  tensor_sinfo.reserve(tuple_sinfo->fields.size());
  for (StructInfo field_sinfo : tuple_sinfo->fields) {
    const auto* field_tensor_sinfo = field_sinfo.as<TensorStructInfoNode>();
    if (field_tensor_sinfo == nullptr) {
      ctx->ReportFatal(
          Diagnostic::Error(call)
          << call->op << " expects the input to be a Tuple of Tensors. However, the given input is "
          << tup->struct_info_);
    }
    tensor_sinfo.push_back(GetRef<TensorStructInfo>(field_tensor_sinfo));
  }
  return tensor_sinfo;
}

Optional<Array<PrimExpr>> InferBinaryBroadcastShape(const Call& call, const BlockBuilder& ctx,
                                                    const Array<PrimExpr>& x1_shape,
                                                    const Array<PrimExpr>& x2_shape) {
  arith::Analyzer* analyzer = ctx->GetAnalyzer();
  int x1_ndim = x1_shape.size();
  int x2_ndim = x2_shape.size();
  int max_ndim = std::max(x1_ndim, x2_ndim);

  std::vector<PrimExpr> output_shape;
  output_shape.reserve(max_ndim);

  int i = 1;
  for (; i <= std::min(x1_ndim, x2_ndim); ++i) {
    const PrimExpr& dim0 = x1_shape[x1_ndim - i];
    const PrimExpr& dim1 = x2_shape[x2_ndim - i];
    const auto* int_dim0 = dim0.as<IntImmNode>();
    const auto* int_dim1 = dim1.as<IntImmNode>();
    if (int_dim0 != nullptr && int_dim0->value == 1) {
      output_shape.push_back(dim1);
    } else if (int_dim1 != nullptr && int_dim1->value == 1) {
      output_shape.push_back(dim0);
    } else if (analyzer->CanProveEqual(dim0, dim1)) {
      output_shape.push_back(dim0);
    } else if (int_dim0 && int_dim1 && int_dim0->value != int_dim1->value) {
      ctx->ReportFatal(Diagnostic::Error(call)
                       << "In " << call->op << ", the first input shape at dim " << x1_ndim - i
                       << " is " << dim0 << " and the second input shape at dim " << x2_ndim - i
                       << " is " << dim1 << ", which are not broadcastable.");
    } else {
      // Use simple fallback when shape mismatch.
      return NullOpt;
    }
  }
  auto& longer_shape = (x1_ndim > x2_ndim) ? x1_shape : x2_shape;
  for (; i <= max_ndim; ++i) {
    output_shape.push_back(longer_shape[max_ndim - i]);
  }
  return Array<PrimExpr>(output_shape.rbegin(), output_shape.rend());
}

std::vector<int> NormalizeAxes(const Call& call, const BlockBuilder& ctx, int ndim,
                               const Array<Integer>& axes) {
  ICHECK_NE(ndim, kUnknownNDim) << "The ndim is required to be known for this function.";
  std::vector<bool> appeared_dims_set;
  std::vector<int> axes_non_neg;
  appeared_dims_set.resize(ndim, /*value=*/false);
  axes_non_neg.reserve(axes.size());
  for (const Integer& axis : axes) {
    int _axis = axis->value;
    if (_axis < -ndim || _axis >= ndim) {
      ctx->ReportFatal(Diagnostic::Error(call) << "In " << call->op << ", the input axis " << _axis
                                               << " is out of range. The input tensor has " << ndim
                                               << " dimensions, so axis should be in range ["
                                               << -ndim << ", " << ndim << ").");
    } else if (_axis < 0) {
      _axis = ndim + _axis;
    }

    if (appeared_dims_set[_axis]) {
      ctx->ReportFatal(Diagnostic::Error(call)
                       << "In " << call->op
                       << ", the input axes is required to be non-repetitive. However, there are "
                          "multiple given axes referring to axis "
                       << _axis);
    }
    appeared_dims_set[_axis] = true;
    axes_non_neg.push_back(_axis);
  }
  return axes_non_neg;
}

InferLayoutOutput InferLayoutUnaryEwise(const Call& call,
                                        const Map<String, Array<String>>& desired_layouts,
                                        const VarLayoutMap& var_layout_map) {
  ICHECK(NoDesiredLayout(call, desired_layouts));
  LayoutDecision layout = GetLayoutDecision(var_layout_map, call->args[0]);
  return InferLayoutOutput({layout}, {layout}, Attrs(call->attrs));
}

}  // namespace relax
}  // namespace tvm
