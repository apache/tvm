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
 * \file manipulate.cc
 * \brief Manipulation operators.
 */

#include "manipulate.h"

#include <algorithm>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "tvm/runtime/data_type.h"

namespace tvm {
namespace relax {

/* relax.broadcast_to */
Expr broadcast_to(Expr x, Expr shape) {
  static const Op& op = Op::Get("relax.broadcast_to");
  return Call(op, {std::move(x), std::move(shape)}, Attrs(), {});
}

TVM_REGISTER_GLOBAL("relax.op.broadcast_to").set_body_typed(broadcast_to);

StructInfo InferStructInfoBroadcastTo(const Call& call, const BlockBuilder& ctx) {
  if (call->args.size() != 2) {
    ctx->ReportFatal(Diagnostic::Error(call) << "broadcast_to should take 2 arguments.");
  }
  const auto* data_sinfo = GetStructInfoAs<TensorStructInfoNode>(call->args[0]);
  const auto* tgt_shape_sinfo = GetStructInfoAs<ShapeStructInfoNode>(call->args[1]);
  if (data_sinfo == nullptr) {
    ctx->ReportFatal(
        Diagnostic::Error(call)
        << "broadcast_to requires the input data to be Tensor. However, the given one is "
        << call->args[0]->struct_info_->GetTypeKey());
  }
  if (tgt_shape_sinfo == nullptr) {
    ctx->ReportFatal(
        Diagnostic::Error(call)
        << "broadcast_to requires the input new shape to be Shape. However, the given one is "
        << call->args[1]->struct_info_->GetTypeKey());
  }

  if (!data_sinfo->IsUnknownNdim() && !tgt_shape_sinfo->IsUnknownNdim() &&
      tgt_shape_sinfo->ndim < data_sinfo->ndim) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "broadcast_to expects the input shape to have the number of ndim at least "
                        "as the input tensor's. However, the given tensor has ndim "
                     << data_sinfo->ndim << " while the target shape has ndim "
                     << tgt_shape_sinfo->ndim);
  }

  // Trust the input target shape when there is no possibility to do any compile-time check.
  if (!data_sinfo->shape.defined()) {
    return TensorStructInfo(/*shape=*/call->args[1], data_sinfo->dtype, data_sinfo->vdevice);
  }
  ShapeStructInfo shape_sinfo = Downcast<ShapeStructInfo>(data_sinfo->shape.value()->struct_info_);
  if (!shape_sinfo->values.defined() || !tgt_shape_sinfo->values.defined()) {
    return TensorStructInfo(/*shape=*/call->args[1], data_sinfo->dtype, data_sinfo->vdevice);
  }

  arith::Analyzer* analyzer = ctx->GetAnalyzer();
  Array<PrimExpr> old_shape_value = shape_sinfo->values.value();
  Array<PrimExpr> tgt_shape_value = tgt_shape_sinfo->values.value();
  int old_ndim = old_shape_value.size();
  int tgt_ndim = tgt_shape_value.size();
  for (int i = 0; i < old_ndim; ++i) {
    PrimExpr old_len = old_shape_value[old_ndim - i - 1];
    PrimExpr tgt_len = tgt_shape_value[tgt_ndim - i - 1];
    const auto* old_len_int = old_len.as<IntImmNode>();
    if (old_len_int != nullptr && old_len_int->value == 1) {
      continue;
    } else if (analyzer->CanProve(old_len != tgt_len)) {
      ctx->ReportFatal(
          Diagnostic::Error(call)
          << "broadcast_to expects the input tensor shape is broadcastable to the target shape. "
             "The target shape at dim "
          << tgt_ndim - i - 1 << " is " << tgt_len << " while the input tensor shape at dim "
          << old_ndim - i - 1 << " is " << old_len << ", which are not equal.");
    }
    // Todo(relax-team): revisit here for better check on if the tensor length
    // is consistent with the length in the given shape.
  }
  return TensorStructInfo(/*shape=*/call->args[1], data_sinfo->dtype, data_sinfo->vdevice);
}

TVM_REGISTER_OP("relax.broadcast_to")
    .set_num_inputs(2)
    .add_argument("x", "Tensor", "The input tensor.")
    .add_argument("shape", "Shape", "The target shape.")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoBroadcastTo)
    .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kFollow)
    .set_attr<Bool>("FPurity", Bool(true));

/* relax.concat */
TVM_REGISTER_NODE_TYPE(ConcatAttrs);

Expr concat(Expr tensors, Optional<Integer> axis) {
  ObjectPtr<ConcatAttrs> attrs = make_object<ConcatAttrs>();
  attrs->axis = std::move(axis);

  static const Op& op = Op::Get("relax.concat");
  return Call(op, {std::move(tensors)}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relax.op.concat").set_body_typed(concat);

Optional<Array<PrimExpr>> CheckConcatOutputShape(const Call& call, const BlockBuilder& ctx,
                                                 const std::vector<Array<PrimExpr>>& shape_values,
                                                 int axis) {
  bool shape_unknown = false;
  arith::Analyzer* analyzer = ctx->GetAnalyzer();
  PrimExpr concat_sum = [&]() {
    // For the specified axis, we compute the sum of shape value over each tensor.

    // Special case, if all concatenated values have the same shape
    StructuralEqual structural_equal;
    PrimExpr first_concat_dim = shape_values[0][axis];
    bool all_same = std::all_of(shape_values.begin(), shape_values.end(), [&](const auto& a) {
      return structural_equal(a[axis], first_concat_dim);
    });
    if (all_same) {
      return first_concat_dim * IntImm(DataType::Int(64), shape_values.size());
    }

    // General case, add up the dimensions along the specified axis.
    PrimExpr concat_sum = IntImm(DataType::Int(64), 0);
    for (Array<PrimExpr> shape_value : shape_values) {
      concat_sum += shape_value[axis];
    }
    return concat_sum;
  }();

  // For other axes, we check the equality of all tensors' shape values, to ensure safety.
  for (int d = 0; d < static_cast<int>(shape_values[0].size()); ++d) {
    if (d == axis) {
      continue;
    }
    for (int i = 1; i < static_cast<int>(shape_values.size()); ++i) {
      if (analyzer->CanProve(shape_values[i][d] != shape_values[0][d])) {
        ctx->ReportFatal(Diagnostic::Error(call)
                         << "Concat expects the input tensors to have the same shape on every "
                            "dimension except the one indicated by the input axis. However, the "
                            "input contains tensors whose shapes on dimension "
                         << d << " is " << shape_values[0][d] << " and " << shape_values[i][d]);
      } else if (!analyzer->CanProveEqual(shape_values[i][d], shape_values[0][d])) {
        shape_unknown = true;
      }
    }
  }

  if (shape_unknown) {
    return NullOpt;
  }
  Array<PrimExpr> output_shape = shape_values[0];
  output_shape.Set(axis, concat_sum);
  return output_shape;
}

StructInfo InferStructInfoConcat(const Call& call, const BlockBuilder& ctx) {
  if (call->args.size() != 1) {
    ctx->ReportFatal(Diagnostic::Error(call) << "Concat op should have 1 argument");
  }
  Array<TensorStructInfo> tensor_sinfo = GetTensorStructInfoFromTuple(call, ctx, call->args[0]);
  if (tensor_sinfo.empty()) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "Concat op expects at least one tensor in the input Tuple. However, the "
                        "given input Tuple is empty.");
  }

  const auto* attrs = call->attrs.as<ConcatAttrs>();
  int output_ndim = attrs->axis.defined() ? kUnknownNDim : 1;
  DataType output_dtype = DataType::Void();
  Optional<VDevice> vdev = NullOpt;
  bool shape_unknown = false;
  bool is_void_dtype = false;
  bool vdevice_unknown = false;
  std::vector<Array<PrimExpr>> shape_values;
  shape_values.reserve(tensor_sinfo.size());

  for (TensorStructInfo sinfo : tensor_sinfo) {
    // Update the output dtype.
    if (sinfo->dtype.is_void()) {
      is_void_dtype = true;
    } else if (output_dtype.is_void()) {
      output_dtype = sinfo->dtype;
    } else if (sinfo->dtype != output_dtype) {
      ctx->ReportFatal(Diagnostic::Error(call)
                       << "Concat expects all input tensors to have the same dtype. However, the "
                          "input contains tensors with dtype "
                       << output_dtype << " and " << sinfo->dtype);
    }

    // Update the output ndim.
    // Todo(relax-team): revisit here for better check on if the input tensor has
    // ndim 1 when the input axis is undefined.
    if (output_ndim == kUnknownNDim) {
      output_ndim = sinfo->ndim;
    } else if (sinfo->ndim != kUnknownNDim && sinfo->ndim != output_ndim) {
      ctx->ReportFatal(Diagnostic::Error(call)
                       << "Concat expects all input tensors to have same ndim. However, the "
                          "input contains tensors with ndim "
                       << output_ndim << " and " << sinfo->ndim);
    }

    // Update the virtual device.
    if (!vdevice_unknown) {
      if (sinfo->vdevice.defined()) {
        if (!vdev.defined()) {
          vdev = sinfo->vdevice.value();
        } else if (sinfo->vdevice.value()->target.defined()) {
          // mismatch
          if (sinfo->vdevice.value() != vdev) {
            vdevice_unknown = true;
          }
        }
      }
    }

    // Update the shape values for best effort check.
    const auto* shape_expr = sinfo->shape.as<ShapeExprNode>();
    if (shape_expr != nullptr) {
      shape_values.push_back(shape_expr->values);
      continue;
    }
    shape_unknown = true;

    if (!sinfo->shape.defined()) {
      continue;
    }
    // Keep the shape value for equality check.
    ShapeStructInfo shape_sinfo = Downcast<ShapeStructInfo>(sinfo->shape.value()->struct_info_);
    if (shape_sinfo->values.defined()) {
      shape_values.push_back(shape_sinfo->values.value());
    }
  }

  if (is_void_dtype) {
    output_dtype = DataType::Void();
  }
  if (vdevice_unknown) {
    vdev = NullOpt;
  }

  if (output_ndim == kUnknownNDim) {
    return tensor_sinfo.size() == 1 ? tensor_sinfo[0]
                                    : TensorStructInfo(output_dtype, output_ndim, vdev);
  }

  int axis =
      attrs->axis.defined() ? NormalizeAxis(call, ctx, output_ndim, attrs->axis.value()->value) : 0;
  // If there is only one input tensor, no action is needed.
  if (tensor_sinfo.size() == 1) {
    return tensor_sinfo[0];
  }
  if (shape_values.empty()) {
    if (!vdevice_unknown) {
      return TensorStructInfo(output_dtype, output_ndim, vdev);
    }
    return TensorStructInfo(output_dtype, output_ndim);
  }

  // As long as the there is known shape value, we will do the best effort check to ensure safety.
  Optional<Array<PrimExpr>> output_shape = CheckConcatOutputShape(call, ctx, shape_values, axis);

  if (shape_unknown || !output_shape.defined()) {
    if (!vdevice_unknown) {
      return TensorStructInfo(output_dtype, output_ndim, vdev);
    }
    return TensorStructInfo(output_dtype, output_ndim);
  } else {
    if (!vdevice_unknown) {
      return TensorStructInfo(ShapeExpr(output_shape.value()), output_dtype, vdev);
    }
    return TensorStructInfo(ShapeExpr(output_shape.value()), output_dtype);
  }
}

InferLayoutOutput InferLayoutConcat(const Call& call,
                                    const Map<String, Array<String>>& desired_layouts,
                                    const VarLayoutMap& var_layout_map) {
  ICHECK(NoDesiredLayout(call, desired_layouts));

  const auto* attrs = call->attrs.as<ConcatAttrs>();
  ICHECK(attrs != nullptr) << "Invalid Call";
  NLayout nlayout = GetNLayout(var_layout_map, call->args[0]);
  ICHECK(nlayout.IsNested());
  ICHECK(nlayout.NestedArray()[0].IsLeaf());

  int n_tensor = nlayout.NestedArray().size();
  LayoutDecision layout = nlayout.NestedArray()[0].LeafValue();
  Array<NLayout> input_layouts, output_layouts;
  for (int i = 0; i < n_tensor; ++i) {
    input_layouts.push_back(layout);
  }
  output_layouts.push_back(layout);
  ObjectPtr<ConcatAttrs> new_attrs = make_object<ConcatAttrs>(*attrs);
  new_attrs->axis = Integer(FindAxis(layout->layout, attrs->axis.value_or(0)->value));
  return InferLayoutOutput({NLayout(input_layouts)}, output_layouts, Attrs(new_attrs));
}

TVM_REGISTER_OP("relax.concat")
    .set_attrs_type<ConcatAttrs>()
    .set_num_inputs(1)
    .add_argument("tensors", "Tuple of Tensors", "The input list of tensors.")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoConcat)
    .set_attr<FRelaxInferLayout>("FRelaxInferLayout", InferLayoutConcat)
    .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kFollow)
    .set_attr<Bool>("FPurity", Bool(true));

/* relax.expand_dims */
TVM_REGISTER_NODE_TYPE(ExpandDimsAttrs);

Expr expand_dims(Expr x, Array<Integer> axis) {
  ObjectPtr<ExpandDimsAttrs> attrs = make_object<ExpandDimsAttrs>();
  attrs->axis = std::move(axis);

  static const Op& op = Op::Get("relax.expand_dims");
  return Call(op, {std::move(x)}, Attrs{attrs}, {});
}

TVM_REGISTER_GLOBAL("relax.op.expand_dims").set_body_typed(expand_dims);

StructInfo InferStructInfoExpandDims(const Call& call, const BlockBuilder& ctx) {
  TensorStructInfo data_sinfo = GetUnaryInputTensorStructInfo(call, ctx);
  const auto* attrs = call->attrs.as<ExpandDimsAttrs>();
  if (attrs->axis.empty()) {
    return data_sinfo;
  }

  if (data_sinfo->IsUnknownNdim()) {
    return TensorStructInfo(data_sinfo->dtype, kUnknownNDim, data_sinfo->vdevice);
  }

  int n_new_dim = attrs->axis.size();
  int output_ndim = data_sinfo->ndim + n_new_dim;
  std::vector<int> axes = NormalizeAxes(call, ctx, output_ndim, attrs->axis);

  const auto* data_shape = data_sinfo->shape.as<ShapeExprNode>();
  if (data_shape == nullptr) {
    return TensorStructInfo(data_sinfo->dtype, output_ndim, data_sinfo->vdevice);
  }

  std::vector<PrimExpr> output_shape;
  output_shape.resize(output_ndim, PrimExpr());
  for (int i = 0; i < n_new_dim; ++i) {
    output_shape[axes[i]] = IntImm(DataType::Int(64), 1);
  }

  int i_data_shape = 0;
  for (int i = 0; i < output_ndim; ++i) {
    if (output_shape[i].defined()) {
      continue;
    }
    ICHECK_LT(i_data_shape, data_sinfo->ndim);
    output_shape[i] = data_shape->values[i_data_shape];
    ++i_data_shape;
  }
  ICHECK_EQ(i_data_shape, data_sinfo->ndim);
  return TensorStructInfo(ShapeExpr(output_shape), data_sinfo->dtype, data_sinfo->vdevice);
}

InferLayoutOutput InferLayoutExpandDims(const Call& call,
                                        const Map<String, Array<String>>& desired_layouts,
                                        const VarLayoutMap& var_layout_map) {
  ICHECK(NoDesiredLayout(call, desired_layouts));
  const auto* attrs = call->attrs.as<ExpandDimsAttrs>();
  ICHECK(attrs != nullptr) << "Invalid Call";
  const auto* tensor_sinfo = GetStructInfoAs<TensorStructInfoNode>(call->args[0]);
  ICHECK(tensor_sinfo != nullptr) << "Invalid Call";
  ICHECK(!tensor_sinfo->IsUnknownNdim()) << "Only support static ndim for now";

  LayoutDecision existing_layout = GetLayoutDecision(var_layout_map, call->args[0]);
  int ndim = tensor_sinfo->ndim;
  // Can't handle sub indexed layouts.
  if (existing_layout->layout.ndim() != existing_layout->layout.ndim_primal()) {
    existing_layout = LayoutDecision(InitialLayout(ndim));
  }
  int n_new_dim = attrs->axis.size();
  int output_ndim = ndim + n_new_dim;
  std::vector<bool> is_new_dim(output_ndim, false);
  for (const auto& axis : attrs->axis) {
    is_new_dim[(axis->value + output_ndim) % output_ndim] = true;
  }
  std::string new_layout;
  for (int i = 0; i < output_ndim; ++i) {
    if (!is_new_dim[i]) {
      new_layout.push_back('A' + i);
    }
  }
  new_layout = TransposeStrLike(new_layout, InitialLayout(ndim), existing_layout->layout);
  std::string output_layout;
  for (int i = 0, j = 0; i < output_ndim; ++i) {
    if (is_new_dim[i]) {
      output_layout.push_back('A' + i);
    } else {
      output_layout.push_back(new_layout.at(j++));
    }
  }
  return InferLayoutOutput({existing_layout}, {LayoutDecision(Layout(output_layout))},
                           Attrs(call->attrs));
}

TVM_REGISTER_OP("relax.expand_dims")
    .set_num_inputs(1)
    .set_attrs_type<ExpandDimsAttrs>()
    .add_argument("x", "Tensor", "The input tensor.")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoExpandDims)
    .set_attr<FRelaxInferLayout>("FRelaxInferLayout", InferLayoutExpandDims)
    .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kFollow)
    .set_attr<Bool>("FPurity", Bool(true));

// Helper function for flatten and reshape.
PrimExpr ComputeShapeProduct(const Array<PrimExpr>& shape_values) {
  PrimExpr shape_prod = IntImm(DataType::Int(64), 1);
  for (PrimExpr value : shape_values) {
    shape_prod *= value;
  }
  return shape_prod;
}

/* relax.flatten */
Expr flatten(Expr x) {
  static const Op& op = Op::Get("relax.flatten");
  return Call(op, {std::move(x)}, {}, {});
}

TVM_REGISTER_GLOBAL("relax.op.flatten").set_body_typed(flatten);

StructInfo InferStructInfoFlatten(const Call& call, const BlockBuilder& ctx) {
  TensorStructInfo data_sinfo = GetUnaryInputTensorStructInfo(call, ctx);
  if (data_sinfo->IsUnknownNdim()) {
    return TensorStructInfo(data_sinfo->dtype, /*ndim=*/1, data_sinfo->vdevice);
  } else if (data_sinfo->ndim == 0) {
    return TensorStructInfo(ShapeExpr({1}), data_sinfo->dtype, data_sinfo->vdevice);
  } else if (data_sinfo->ndim == 1) {
    return data_sinfo;
  }

  const auto* data_shape = data_sinfo->shape.as<ShapeExprNode>();
  if (data_shape == nullptr) {
    return TensorStructInfo(data_sinfo->dtype, /*ndim=*/1, data_sinfo->vdevice);
  }
  PrimExpr shape_prod = ComputeShapeProduct(data_shape->values);
  return TensorStructInfo(ShapeExpr({std::move(shape_prod)}), data_sinfo->dtype,
                          data_sinfo->vdevice);
}

TVM_REGISTER_OP("relax.flatten")
    .set_num_inputs(1)
    .add_argument("x", "Tensor", "The input tensor.")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoFlatten)
    .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kFollow)
    .set_attr<Bool>("FPurity", Bool(true));

/* relax.layout_transform */
TVM_REGISTER_NODE_TYPE(LayoutTransformAttrs);

Expr layout_transform(Expr x, tir::IndexMap index_map, Optional<PrimValue> pad_value,
                      Optional<Array<IntImm>> axis_separators,
                      Optional<Array<IntImm>> input_axis_separators) {
  ObjectPtr<LayoutTransformAttrs> attrs = make_object<LayoutTransformAttrs>();
  attrs->index_map = std::move(index_map);
  attrs->pad_value = std::move(pad_value);
  attrs->axis_separators = std::move(axis_separators);
  attrs->input_axis_separators = std::move(input_axis_separators);

  static const Op& op = Op::Get("relax.layout_transform");
  return Call(op, {std::move(x)}, Attrs{attrs}, {});
}

TVM_REGISTER_GLOBAL("relax.op.layout_transform").set_body_typed(layout_transform);

StructInfo InferStructInfoLayoutTransform(const Call& call, const BlockBuilder& ctx) {
  TensorStructInfo data_sinfo = GetUnaryInputTensorStructInfo(call, ctx);
  const auto* attrs = call->attrs.as<LayoutTransformAttrs>();
  tir::IndexMap index_map = attrs->index_map;
  Optional<PrimValue> optional_pad_value = attrs->pad_value;

  // Check pad_value has same dtype as input.
  if (optional_pad_value.defined()) {
    PrimExpr padded_value = optional_pad_value.value()->value;
    if (padded_value->dtype != data_sinfo->dtype) {
      ctx->ReportFatal(Diagnostic::Error(call)
                       << "layout_transform pad_value dtype (" << padded_value->dtype
                       << ") and input dtype (" << data_sinfo->dtype << ") must be the same");
    }
  }

  if (data_sinfo->IsUnknownNdim()) {
    // Todo(relax-team): revisit here for better check on if the input tensor has desired ndim.
    return TensorStructInfo(data_sinfo->dtype, /*ndim=*/index_map->final_indices.size(),
                            data_sinfo->vdevice);
  }

  // If rank is known, check that it is compatible with the index_map, i.e., #dims match.
  if (index_map->initial_indices.size() != static_cast<size_t>(data_sinfo->ndim)) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "number of dimensions in input must match the number of source dimensions "
                        "in index map, but got "
                     << data_sinfo->ndim << " != " << index_map->initial_indices.size());
  }

  if (!data_sinfo->shape.defined()) {
    return TensorStructInfo(data_sinfo->dtype, /*ndim=*/index_map->final_indices.size(),
                            data_sinfo->vdevice);
  }

  ShapeStructInfo shape_sinfo = Downcast<ShapeStructInfo>(data_sinfo->shape.value()->struct_info_);
  if (!shape_sinfo->values.defined()) {
    return TensorStructInfo(data_sinfo->dtype, /*ndim=*/index_map->final_indices.size(),
                            data_sinfo->vdevice);
  }

  arith::Analyzer analyzer;
  Array<PrimExpr> output_shape = index_map->MapShape(shape_sinfo->values.value(), &analyzer);
  return TensorStructInfo(ShapeExpr(output_shape), data_sinfo->dtype, data_sinfo->vdevice);
}

TVM_REGISTER_OP("relax.layout_transform")
    .set_num_inputs(1)
    .set_attrs_type<LayoutTransformAttrs>()
    .add_argument("x", "Tensor", "The input tensor.")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoLayoutTransform)
    .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kFollow)
    .set_attr<Bool>("FPurity", Bool(true));

/* relax.permute_dims */
TVM_REGISTER_NODE_TYPE(PermuteDimsAttrs);

Expr permute_dims(Expr x, Optional<Array<Integer>> axes) {
  ObjectPtr<PermuteDimsAttrs> attrs = make_object<PermuteDimsAttrs>();
  attrs->axes = std::move(axes);

  static const Op& op = Op::Get("relax.permute_dims");
  return Call(op, {std::move(x)}, Attrs{attrs}, {});
}

TVM_REGISTER_GLOBAL("relax.op.permute_dims").set_body_typed(permute_dims);

bool IsIdentityPermutation(const std::vector<int>& permutation) {
  for (int i = 0; i < static_cast<int>(permutation.size()); ++i) {
    if (permutation[i] != i) {
      return false;
    }
  }
  return true;
}

StructInfo InferStructInfoPermuteDims(const Call& call, const BlockBuilder& ctx) {
  TensorStructInfo data_sinfo = GetUnaryInputTensorStructInfo(call, ctx);

  const auto* attrs = call->attrs.as<PermuteDimsAttrs>();

  // Todo(relax-team): revisit here for better check on if the input tensor has
  // ndim same as the number of input axes.
  if (!attrs->axes.defined() && data_sinfo->IsUnknownNdim()) {
    return TensorStructInfo(data_sinfo->dtype, kUnknownNDim, data_sinfo->vdevice);
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
    return data_sinfo;
  }

  const auto* data_shape = data_sinfo->shape.as<ShapeExprNode>();
  if (data_shape == nullptr) {
    return TensorStructInfo(data_sinfo->dtype, data_sinfo->ndim, data_sinfo->vdevice);
  }
  std::vector<PrimExpr> new_shape;
  new_shape.reserve(data_sinfo->ndim);
  for (int i = 0; i < data_sinfo->ndim; ++i) {
    new_shape.push_back(data_shape->values[axes[i]]);
  }
  return TensorStructInfo(ShapeExpr(new_shape), data_sinfo->dtype, data_sinfo->vdevice);
}

InferLayoutOutput InferLayoutPermuteDims(const Call& call,
                                         const Map<String, Array<String>>& desired_layouts,
                                         const VarLayoutMap& var_layout_map) {
  ICHECK(NoDesiredLayout(call, desired_layouts));

  const auto* attrs = call->attrs.as<PermuteDimsAttrs>();
  ICHECK(attrs != nullptr) << "Invalid Call";
  const auto* tensor_sinfo = GetStructInfoAs<TensorStructInfoNode>(call->args[0]);
  ICHECK(tensor_sinfo != nullptr) << "Invalid Call";
  ICHECK(!tensor_sinfo->IsUnknownNdim()) << "Only support static ndim for now";
  int ndim = tensor_sinfo->ndim;

  LayoutDecision existing_layout = GetLayoutDecision(var_layout_map, call->args[0]);

  // permute_dims can't handle sub indexed layouts.
  if (existing_layout->layout.ndim() != existing_layout->layout.ndim_primal()) {
    existing_layout = LayoutDecision(InitialLayout(ndim));
  }

  Array<Integer> order;
  if (attrs->axes.defined()) {
    order = attrs->axes.value();
  } else {
    order.reserve(ndim);
    for (int i = 0; i < ndim; ++i) {
      order.push_back(Integer(ndim - i - 1));
    }
  }
  std::string order_str;
  for (const auto& axis : order) {
    order_str.push_back(axis->value + 'A');
  }
  String new_axes =
      TransposeStrLike(InitialLayout(ndim).name(), existing_layout->layout, order_str);
  Array<Integer> new_order;
  for (size_t i = 0; i < new_axes.size(); ++i) {
    new_order.push_back(Integer(new_axes.at(i) - 'A'));
  }
  ObjectPtr<PermuteDimsAttrs> new_attrs = make_object<PermuteDimsAttrs>(*attrs);
  new_attrs->axes = new_order;
  return InferLayoutOutput({existing_layout}, {InitialLayoutDecision(ndim)}, Attrs(new_attrs));
}

TVM_REGISTER_OP("relax.permute_dims")
    .set_attrs_type<PermuteDimsAttrs>()
    .set_num_inputs(1)
    .add_argument("x", "Tensor", "The input tensor.")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoPermuteDims)
    .set_attr<FRelaxInferLayout>("FRelaxInferLayout", InferLayoutPermuteDims)
    .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kFollow)
    .set_attr<Bool>("FPurity", Bool(true));

/* relax.reshape */
Expr ConvertNewShapeToExpr(const Expr& data, const Variant<Expr, Array<PrimExpr>>& shape) {
  const ArrayNode* array;
  // Treat shape expressions as constant arrays to handle special values.
  if (const auto* e = shape.as<ShapeExprNode>()) {
    array = e->values.as<ArrayNode>();
    // Other non-shape expressions are used directly.
  } else if (const auto* e = shape.as<ExprNode>()) {
    return GetRef<Expr>(e);
    // Process special values in constants and produce an expression.
  } else {
    array = shape.as<ArrayNode>();
  }
  CHECK(array != nullptr) << "Reshape only expects the input new shape to be either an Expr or an "
                             "Array of PrimExprs. However, the given new shape is "
                          << shape;
  int dim_to_infer = -1;
  // Keep track of which dimensions should be copied from input.
  std::vector<int> zero_dims;
  for (int i = 0; i < static_cast<int>(array->size()); ++i) {
    const auto* _len = array->at(i).as<PrimExprNode>();
    CHECK(_len != nullptr) << "Reshape only expects the input new shape to be either an Expr or an "
                              "Array of PrimExprs. However, the given new shape is "
                           << shape;
    PrimExpr len = GetRef<PrimExpr>(_len);
    CHECK(len->dtype.is_int()) << "Reshape requires the new shape values to be all "
                                  "integers. However, the give new shape is "
                               << shape;
    const auto* int_len = len.as<IntImmNode>();
    if (int_len != nullptr && int_len->value == 0) {
      // Note that this dimension should be copied from the original shape.
      zero_dims.push_back(i);
    } else if (int_len != nullptr && int_len->value == -1) {
      CHECK_EQ(dim_to_infer, -1) << "Reshape accepts at most one \"-1\" in the new shape. However, "
                                    "there are multiple \"-1\" in the given new shape  "
                                 << shape;
      dim_to_infer = i;
    } else {
      CHECK(int_len == nullptr || int_len->value > 0)
          << "Reshape requires all values in the new shape to be positive except a single \"-1\". "
             "However, the given new shape is "
          << shape;
    }
  }

  Array<PrimExpr> array_ref = GetRef<Array<PrimExpr>>(array);
  // When there is no dimension to infer, just return the input array as ShapeExpr.
  if (dim_to_infer == -1 && zero_dims.empty()) {
    return ShapeExpr(array_ref);
  }

  // Otherwise, we require the input tensor to have known shape value for inference.
  const auto* data_sinfo = GetStructInfoAs<TensorStructInfoNode>(data);
  CHECK(data_sinfo != nullptr)
      << "Reshape expects the input data to be a Tensor. However, the given input is "
      << data->struct_info_->GetTypeKey();
  CHECK(data_sinfo->shape.defined())
      << "Reshape expects the input tensor to have known shape when there is some dimension length "
         "to infer. However, the given input has no shape.";
  const auto* shape_sinfo = GetStructInfoAs<ShapeStructInfoNode>(data_sinfo->shape.value());
  CHECK(shape_sinfo != nullptr && shape_sinfo->values.defined())
      << "Reshape expects the input tensor to have known shape when there is some dimension length "
         "to infer. However, the given input shape is "
      << data_sinfo->shape << " whose shape value is unknown.";

  // Set any 0 valued dimensions to match the corresponding input shape.
  if (!zero_dims.empty()) {
    for (int i : zero_dims) {
      array_ref.Set(i, shape_sinfo->values.value()[i]);
    }
  }

  // Set any -1 dimensions to complete the number of appropriate elements.
  // Start by computing the shape product of all positive indices.
  PrimExpr new_shape_prod = IntImm(DataType::Int(64), 1);
  for (int i = 0; i < static_cast<int>(array_ref.size()); ++i) {
    PrimExpr new_dim = array_ref[i];
    const auto* int_dim = new_dim.as<IntImmNode>();
    // We expect any symbolic not to signal the intent of -1, and therefore do no check for
    // symbolic value here.
    if (int_dim == nullptr || int_dim->value > 0) {
      new_shape_prod = new_shape_prod * new_dim;
    }
  }

  // Assign appropriate value to -1 dimension.
  if (dim_to_infer != -1) {
    arith::Analyzer analyzer;
    PrimExpr old_shape_prod = ComputeShapeProduct(shape_sinfo->values.value());
    array_ref.Set(dim_to_infer, analyzer.Simplify(floordiv(old_shape_prod, new_shape_prod)));
  }
  return ShapeExpr(array_ref);
}

Expr reshape(Expr x, Variant<Expr, Array<PrimExpr>> shape) {
  Expr shape_in_expr = ConvertNewShapeToExpr(x, shape);
  static const Op& op = Op::Get("relax.reshape");
  return Call(op, {std::move(x), std::move(shape_in_expr)}, Attrs(), {});
}

TVM_REGISTER_GLOBAL("relax.op.reshape").set_body_typed(reshape);

StructInfo InferStructInfoReshape(const Call& call, const BlockBuilder& ctx) {
  if (call->args.size() != 2) {
    ctx->ReportFatal(Diagnostic::Error(call) << "Reshape op should take 2 arguments");
  }
  const auto* data_sinfo = GetStructInfoAs<TensorStructInfoNode>(call->args[0]);
  const auto* new_shape_sinfo = GetStructInfoAs<ShapeStructInfoNode>(call->args[1]);
  if (data_sinfo == nullptr) {
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
  // If shape values are defined, use them
  if (target_shape->IsInstance<VarNode>() && new_shape_sinfo->values.defined()) {
    return TensorStructInfo(ShapeExpr(new_shape_sinfo->values.value()), data_sinfo->dtype,
                            data_sinfo->vdevice);
  }
  return TensorStructInfo(target_shape, data_sinfo->dtype, data_sinfo->vdevice);
}

TVM_REGISTER_OP("relax.reshape")
    .set_num_inputs(2)
    .add_argument("x", "Tensor", "The input tensor.")
    .add_argument("shape", "Shape", "The input new shape.")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoReshape)
    .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kFollow)
    .set_attr<Bool>("FPurity", Bool(true));

/* relax.split */
TVM_REGISTER_NODE_TYPE(SplitAttrs);

Expr split(Expr x, Variant<IntImm, Array<IntImm>> indices_or_sections, int axis) {
  ObjectPtr<SplitAttrs> attrs = make_object<SplitAttrs>();
  if (const auto* indices = indices_or_sections.as<ArrayNode>()) {
    for (int i = 0; i < static_cast<int>(indices->size()); ++i) {
      const auto* idx = indices->at(i).as<IntImmNode>();
      CHECK(idx != nullptr) << "Split op only accepts an array of integers as the indices. "
                               "However, the given indices "
                            << indices_or_sections << " contains some non-integer.";
    }
    indices_or_sections = ConvertIntImmToInt64(GetRef<Array<IntImm>>(indices));
  } else if (const auto* n_section = indices_or_sections.as<IntImmNode>()) {
    CHECK_GT(n_section->value, 0) << "Split op expects the input number of sections to be a "
                                     "positive integer. However, the given number of sections is "
                                  << n_section->value;
    indices_or_sections = IntImm(DataType::Int(64), n_section->value);
  } else {
    LOG(FATAL) << "Split op expects the input indices_or_sections to be either an Array of "
                  "PrimExpr or an integer. However, the given one is "
               << indices_or_sections->GetTypeKey();
  }
  attrs->indices_or_sections = indices_or_sections;
  attrs->axis = axis;

  static const Op& op = Op::Get("relax.split");
  return Call(op, {std::move(x)}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relax.op.split").set_body_typed(split);

StructInfo InferStructInfoSplit(const Call& call, const BlockBuilder& ctx) {
  TensorStructInfo data_sinfo = GetUnaryInputTensorStructInfo(call, ctx);
  const auto* attrs = call->attrs.as<SplitAttrs>();
  const auto* data_shape = data_sinfo->shape.as<ShapeExprNode>();
  int axis =
      data_sinfo->IsUnknownNdim() ? -1 : NormalizeAxis(call, ctx, data_sinfo->ndim, attrs->axis);

  if (auto opt_indices = attrs->indices_or_sections.as<Array<IntImm>>()) {
    auto p_indices = opt_indices.value();
    // When there is not index, return the input tensor's struct info.
    if (p_indices.size() == 0) {
      return TupleStructInfo({data_sinfo});
    }
    // Fall back to unknown shape when the input tensor doesn't have ShapeExpr as shape.
    if (data_shape == nullptr) {
      return TupleStructInfo(Array<StructInfo>(
          p_indices.size() + 1,
          TensorStructInfo(data_sinfo->dtype, data_sinfo->ndim, data_sinfo->vdevice)));
    }

    ICHECK_NE(axis, -1);

    IntImm zero(DataType::Int(64), /*value=*/0);

    std::vector<StructInfo> output_sinfo;
    for (size_t i = 0; i < p_indices.size() + 1; i++) {
      PrimExpr left;
      if (i == 0) {
        left = zero;
      } else {
        left = p_indices[i - 1];
      }

      PrimExpr right;
      if (i < p_indices.size()) {
        right = p_indices[i];
      } else {
        right = data_shape->values[axis];
      }

      left = tvm::min(tvm::max(left, 0), data_shape->values[axis]);
      right = tvm::min(tvm::max(right, 0), data_shape->values[axis]);

      PrimExpr split_dim = right - left;
      split_dim = tvm::max(split_dim, 0);
      split_dim = ctx->GetAnalyzer()->Simplify(split_dim);

      Array<PrimExpr> shape = data_shape->values;
      shape.Set(axis, split_dim);
      output_sinfo.push_back(
          TensorStructInfo(ShapeExpr(shape), data_sinfo->dtype, data_sinfo->vdevice));
    }
    return TupleStructInfo(output_sinfo);
  } else if (const auto* p_n_section = attrs->indices_or_sections.as<IntImmNode>()) {
    ICHECK_GT(p_n_section->value, 0);
    int n_section = p_n_section->value;
    // When the number of section is one, return the input tensor's struct info.
    if (n_section == 1) {
      return TupleStructInfo({data_sinfo});
    }
    // Fall back to unknown shape when the input tensor doesn't have ShapeExpr as shape.
    if (data_shape == nullptr) {
      return TupleStructInfo(Array<StructInfo>(
          n_section, TensorStructInfo(data_sinfo->dtype, data_sinfo->ndim, data_sinfo->vdevice)));
    }
    ICHECK_NE(axis, -1);
    PrimExpr split_len = ceildiv(data_shape->values[axis], n_section);
    split_len = ctx->GetAnalyzer()->Simplify(split_len);

    // Construct struct info for tensors except the last one.
    Array<PrimExpr> shape = data_shape->values;
    shape.Set(axis, split_len);
    std::vector<StructInfo> output_sinfo(
        n_section - 1, TensorStructInfo(ShapeExpr(shape), data_sinfo->dtype, data_sinfo->vdevice));

    // Construct struct info for the last tensor.
    PrimExpr last_split_len = data_shape->values[axis] - split_len * (n_section - 1);
    last_split_len = ctx->GetAnalyzer()->Simplify(last_split_len);
    shape.Set(axis, last_split_len);
    output_sinfo.push_back(
        TensorStructInfo(ShapeExpr(shape), data_sinfo->dtype, data_sinfo->vdevice));
    return TupleStructInfo(output_sinfo);
  }
  ICHECK(false) << "Cannot reach here.";
  throw;
}

InferLayoutOutput InferLayoutSplit(const Call& call,
                                   const Map<String, Array<String>>& desired_layouts,
                                   const VarLayoutMap& var_layout_map) {
  ICHECK(NoDesiredLayout(call, desired_layouts));

  const auto* attrs = call->attrs.as<SplitAttrs>();
  ICHECK(attrs != nullptr) << "Invalid Call";
  const auto* tensor_sinfo = GetStructInfoAs<TensorStructInfoNode>(call->args[0]);
  ICHECK(tensor_sinfo != nullptr) << "Invalid Call";
  ICHECK(!tensor_sinfo->IsUnknownNdim()) << "Only support known ndim";

  LayoutDecision existing_layout = GetLayoutDecision(var_layout_map, call->args[0]);
  StructInfo out_sinfo = InferStructInfoSplit(call, BlockBuilder::Create(IRModule()));
  const auto* out_tuple = out_sinfo.as<TupleStructInfoNode>();

  /*
   * Fallback if the outputs can't be represented in input sub indexed layout
   * This can happen after sub indexing, if we can't split the corresponding primal axis
   */
  if (existing_layout->layout.ndim() != existing_layout->layout.ndim_primal()) {
    for (const auto& si : out_tuple->fields) {
      ICHECK(si->IsInstance<TensorStructInfoNode>())
          << "Fields of TupleStructInfo must be TensorStructInfo"
             "output structinfo, but got "
          << si;
      auto sinfo = Downcast<TensorStructInfo>(si);
      Optional<ShapeExpr> shape_expr = GetRef<ShapeExpr>(sinfo->shape.as<ShapeExprNode>());
      CHECK(shape_expr.defined());
      auto shape_arr = shape_expr.value();
      if (!CanProveLayoutTransform(InitialLayout(tensor_sinfo->ndim), existing_layout->layout,
                                   shape_arr->values)) {
        existing_layout = InitialLayout(tensor_sinfo->ndim);
        break;
      }
    }
  }

  ObjectPtr<SplitAttrs> new_attrs = make_object<SplitAttrs>(*attrs);
  new_attrs->axis = FindAxis(existing_layout->layout, attrs->axis);
  ICHECK(out_tuple != nullptr) << "Invalid Call";
  NLayout tuple_layouts(Array<NLayout>(out_tuple->fields.size(), existing_layout));
  return InferLayoutOutput({existing_layout}, {tuple_layouts}, Attrs(new_attrs));
}

TVM_REGISTER_OP("relax.split")
    .set_attrs_type<SplitAttrs>()
    .set_num_inputs(1)
    .add_argument("x", "Tensor", "The input tensor.")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoSplit)
    .set_attr<FRelaxInferLayout>("FRelaxInferLayout", InferLayoutSplit)
    .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kFollow)
    .set_attr<Bool>("FPurity", Bool(true));

/* relax.squeeze */
TVM_REGISTER_NODE_TYPE(SqueezeAttrs);

Expr squeeze(Expr x, Optional<Array<Integer>> axis) {
  ObjectPtr<SqueezeAttrs> attrs = make_object<SqueezeAttrs>();
  attrs->axis = std::move(axis);

  static const Op& op = Op::Get("relax.squeeze");
  return Call(op, {std::move(x)}, Attrs{attrs}, {});
}

TVM_REGISTER_GLOBAL("relax.op.squeeze").set_body_typed(squeeze);

StructInfo InferStructInfoSqueeze(const Call& call, const BlockBuilder& ctx) {
  TensorStructInfo data_sinfo = GetUnaryInputTensorStructInfo(call, ctx);
  const auto* attrs = call->attrs.as<SqueezeAttrs>();
  if (attrs->axis.defined() && attrs->axis.value().empty()) {
    return data_sinfo;
  }

  if (data_sinfo->IsUnknownNdim()) {
    return TensorStructInfo(data_sinfo->dtype, kUnknownNDim, data_sinfo->vdevice);
  }

  Optional<Array<PrimExpr>> shape_value;
  if (data_sinfo->shape.defined()) {
    shape_value = Downcast<ShapeStructInfo>(data_sinfo->shape.value()->struct_info_)->values;
  }

  std::vector<bool> axis_removal_mask;
  axis_removal_mask.resize(data_sinfo->ndim, /*value=*/false);

  if (attrs->axis.defined()) {
    std::vector<int> axes = NormalizeAxes(call, ctx, data_sinfo->ndim, attrs->axis.value());

    if (!shape_value.defined()) {
      return TensorStructInfo(data_sinfo->dtype, data_sinfo->ndim - axes.size(),
                              data_sinfo->vdevice);
    }
    for (int i = 0; i < static_cast<int>(axes.size()); ++i) {
      // Todo(relax-team): revisit here for better check on if the axis being squeezed has length 1.
      // When `axis` is given, the dim lengths at the axes must be integer 1 when it is not symbolic
      const auto* int_len = shape_value.value()[axes[i]].as<IntImmNode>();
      if (int_len != nullptr && int_len->value != 1) {
        ctx->ReportFatal(Diagnostic::Error(call)
                         << "Squeeze expects the input tensor shape values at the given axis "
                            "positions to be all 1. However, the tensor shape at axis "
                         << axes[i] << " is " << shape_value.value()[axes[i]]
                         << " which is not 1. If it is symbolic, please use MatchCast to cast it "
                            "to 1 before doing Squeeze.");
      }
      axis_removal_mask[axes[i]] = true;
    }
  } else {
    // When `axis` is not defined, squeeze all unit-length dimensions.
    // Note: This is a less well-defined path in Array API standard's squeeze
    // (https://data-apis.org/array-api/latest/API_specification/generated/array_api.squeeze.html).
    // Consider discourage usage later.
    if (!shape_value.defined()) {
      return TensorStructInfo(data_sinfo->dtype, kUnknownNDim, data_sinfo->vdevice);
    }
    for (int i = 0; i < data_sinfo->ndim; ++i) {
      // Whenever a dimension length is symbolic, fall back to unknown ndim.
      const auto* int_len = shape_value.value()[i].as<IntImmNode>();
      if (int_len == nullptr) {
        return TensorStructInfo(data_sinfo->dtype, kUnknownNDim, data_sinfo->vdevice);
      }
      if (int_len->value == 1) {
        axis_removal_mask[i] = true;
      }
    }
  }

  std::vector<PrimExpr> output_shape;
  output_shape.reserve(data_sinfo->ndim - axis_removal_mask.size());
  for (int i = 0; i < data_sinfo->ndim; ++i) {
    if (!axis_removal_mask[i]) {
      output_shape.push_back(shape_value.value()[i]);
    }
  }

  if (data_sinfo->shape.value()->IsInstance<VarNode>()) {
    if (static_cast<int>(output_shape.size()) == data_sinfo->ndim) {
      return data_sinfo;
    } else if (attrs->axis.defined()) {
      return TensorStructInfo(data_sinfo->dtype, output_shape.size(), data_sinfo->vdevice);
    } else {
      return TensorStructInfo(data_sinfo->dtype, kUnknownNDim, data_sinfo->vdevice);
    }
  } else {
    return TensorStructInfo(ShapeExpr(output_shape), data_sinfo->dtype, data_sinfo->vdevice);
  }
}

InferLayoutOutput InferLayoutSqueeze(const Call& call,
                                     const Map<String, Array<String>>& desired_layouts,
                                     const VarLayoutMap& var_layout_map) {
  ICHECK(NoDesiredLayout(call, desired_layouts));

  const auto* attrs = call->attrs.as<SqueezeAttrs>();
  ICHECK(attrs != nullptr) << "Invalid Call";
  const auto* tensor_sinfo = GetStructInfoAs<TensorStructInfoNode>(call->args[0]);
  ICHECK(tensor_sinfo != nullptr) << "Invalid Call";
  ICHECK(!tensor_sinfo->IsUnknownNdim()) << "Only support static ndim for now";
  ICHECK(tensor_sinfo->shape.defined()) << "Only support static shape for now";
  int ndim = tensor_sinfo->ndim;
  const auto* shape = tensor_sinfo->shape.as<ShapeExprNode>();
  ICHECK(shape != nullptr) << "Only support static shape for now";

  Array<Integer> axis;
  if (attrs->axis.defined()) {
    axis = attrs->axis.value();
  } else {
    axis.reserve(ndim);
    for (int i = 0; i < ndim; ++i) {
      if (tir::is_one(shape->values[i])) {
        axis.push_back(Integer(i));
      }
    }
  }

  std::string axis_str(ndim, '0');
  for (const auto& iter : axis) {
    axis_str[iter->value] = '1';
  }
  for (int i = 0, j = 0; i < ndim; ++i) {
    if (axis_str[i] != '1') {
      axis_str[i] = 'A' + j++;
    }
  }

  LayoutDecision existing_layout = GetLayoutDecision(var_layout_map, call->args[0]);
  // Can't handle sub indexed layouts.
  if (existing_layout->layout.ndim() != existing_layout->layout.ndim_primal()) {
    existing_layout = LayoutDecision(InitialLayout(ndim));
  }
  String new_axis_str = TransposeStrLike(axis_str, InitialLayout(ndim), existing_layout->layout);
  Array<Integer> new_axis;
  for (size_t i = 0; i < new_axis_str.size(); ++i) {
    if (new_axis_str.at(i) == '1') {
      new_axis.push_back(Integer(i));
    }
  }
  std::string output_layout = new_axis_str;
  output_layout.erase(std::remove(output_layout.begin(), output_layout.end(), '1'),
                      output_layout.end());

  ObjectPtr<SqueezeAttrs> new_attrs = make_object<SqueezeAttrs>(*attrs);
  new_attrs->axis = new_axis;
  return InferLayoutOutput({existing_layout}, {LayoutDecision(Layout(output_layout))},
                           Attrs(new_attrs));
}

TVM_REGISTER_OP("relax.squeeze")
    .set_num_inputs(1)
    .set_attrs_type<SqueezeAttrs>()
    .add_argument("x", "Tensor", "The input tensor.")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoSqueeze)
    .set_attr<FRelaxInferLayout>("FRelaxInferLayout", InferLayoutSqueeze)
    .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kFollow)
    .set_attr<Bool>("FPurity", Bool(true));

void CheckCollapseShape(const Call& call, const BlockBuilder& ctx,
                        const Array<PrimExpr>& data_shape, const Array<PrimExpr>& target_shape) {
  arith::Analyzer* analyzer = ctx->GetAnalyzer();

  int data_ndim = data_shape.size();
  int target_ndim = target_shape.size();

  int data_ax = data_ndim - 1;
  int target_ax = target_ndim - 1;
  for (; data_ax >= 0; --data_ax) {
    if (target_ax < 0) {
      continue;
    }
    const PrimExpr& dim0 = data_shape[data_ax];
    const PrimExpr& dim1 = target_shape[target_ax];
    const auto* int_dim0 = dim0.as<IntImmNode>();
    const auto* int_dim1 = dim1.as<IntImmNode>();

    if (analyzer->CanProveEqual(dim0, dim1) || (int_dim1 != nullptr && int_dim1->value == 1)) {
      --target_ax;
    } else if (int_dim0 && int_dim1 && int_dim0->value != int_dim1->value) {
      ctx->ReportFatal(Diagnostic::Error(call)
                       << "In " << call->op << ", the data shape at dim " << data_ax << " is "
                       << dim0 << " and the target shape at dim " << target_ax << " is " << dim1
                       << ", which do not match the rule of collapse sum.");
    } else {
      // Todo(relax-team): At this moment, enforcing MatchCast is fine. But we may need to revisit
      // this requirement to reduce the workload of importers and better support dynamic shapes.
      ctx->ReportFatal(Diagnostic::Error(call)
                       << call->op
                       << " fails to match the axes because of unknown dim or symbolic"
                          " shape. In this position the dim of data shape is "
                       << dim0 << " while the dim of target shape is " << dim1
                       << ". If it is symbolic, consider use MatchCast first.");
    }
  }
}

/* relax.collapse_sum_like */
Expr collapse_sum_like(Expr data, Expr collapse_target) {
  static const Op& op = Op::Get("relax.collapse_sum_like");
  return Call(op, {std::move(data), std::move(collapse_target)}, Attrs(), {});
}

TVM_REGISTER_GLOBAL("relax.op.collapse_sum_like").set_body_typed(collapse_sum_like);

StructInfo InferStructInfoCollapseSumLike(const Call& call, const BlockBuilder& ctx) {
  Array<TensorStructInfo> input_sinfo = GetInputTensorStructInfo(call, ctx);
  TensorStructInfo data_sinfo = input_sinfo[0];
  TensorStructInfo collapse_target_sinfo = input_sinfo[1];

  DataType output_dtype = data_sinfo->dtype;

  Optional<Array<PrimExpr>> data_shape_value;
  if (data_sinfo->shape.defined()) {
    data_shape_value = GetStructInfoAs<ShapeStructInfoNode>(data_sinfo->shape.value())->values;
  }
  Optional<Array<PrimExpr>> collapse_target_shape_value;
  if (collapse_target_sinfo->shape.defined()) {
    collapse_target_shape_value =
        GetStructInfoAs<ShapeStructInfoNode>(collapse_target_sinfo->shape.value())->values;
  }

  if (data_shape_value.defined() && collapse_target_shape_value.defined()) {
    CheckCollapseShape(call, ctx, data_shape_value.value(), collapse_target_shape_value.value());
  }

  if (collapse_target_sinfo->shape.defined()) {
    return TensorStructInfo(collapse_target_sinfo->shape.value(), output_dtype,
                            collapse_target_sinfo->vdevice);
  } else {
    return TensorStructInfo(output_dtype, collapse_target_sinfo->ndim,
                            collapse_target_sinfo->vdevice);
  }
}

TVM_REGISTER_OP("relax.collapse_sum_like")
    .set_num_inputs(2)
    .add_argument("data", "Tensor", "The input tensor.")
    .add_argument("collapse_target", "Tensor",
                  "The tensor whose shape is the shape to collapse to.")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoCollapseSumLike)
    .set_attr<Bool>("FPurity", Bool(true));

/* relax.collapse_sum_to */
Expr collapse_sum_to(Expr data, Expr shape) {
  static const Op& op = Op::Get("relax.collapse_sum_to");
  return Call(op, {std::move(data), std::move(shape)}, Attrs(), {});
}

TVM_REGISTER_GLOBAL("relax.op.collapse_sum_to").set_body_typed(collapse_sum_to);

StructInfo InferStructInfoCollapseSumTo(const Call& call, const BlockBuilder& ctx) {
  if (call->args.size() != 2) {
    ctx->ReportFatal(Diagnostic::Error(call) << "CollapseSumTo should have 2 arguments");
  }

  const auto* data_sinfo = GetStructInfoAs<TensorStructInfoNode>(call->args[0]);
  const auto* shape_sinfo = GetStructInfoAs<ShapeStructInfoNode>(call->args[1]);

  if (data_sinfo == nullptr) {
    ctx->ReportFatal(
        Diagnostic::Error(call)
        << "CollapseSumTo requires the input data to be a Tensor. However, the given one is "
        << call->args[0]->struct_info_->GetTypeKey());
  }
  if (shape_sinfo == nullptr) {
    ctx->ReportFatal(
        Diagnostic::Error(call)
        << "CollapseSumTo requires the input shape to be a Shape. However, the given one is "
        << call->args[1]->struct_info_->GetTypeKey());
  }

  DataType output_dtype = data_sinfo->dtype;

  Optional<Array<PrimExpr>> data_shape_value;
  if (data_sinfo->shape.defined()) {
    data_shape_value = GetStructInfoAs<ShapeStructInfoNode>(data_sinfo->shape.value())->values;
  }

  if (data_shape_value.defined() && shape_sinfo->values.defined()) {
    CheckCollapseShape(call, ctx, data_shape_value.value(), shape_sinfo->values.value());
  }
  return TensorStructInfo(/*shape=*/call->args[1], output_dtype, data_sinfo->vdevice);
}

TVM_REGISTER_OP("relax.collapse_sum_to")
    .set_num_inputs(2)
    .add_argument("data", "Tensor", "The input tensor.")
    .add_argument("shape", "Shape", "The shape to collapse to.")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoCollapseSumTo)
    .set_attr<Bool>("FPurity", Bool(true));

/* relax.repeat */
TVM_REGISTER_NODE_TYPE(RepeatAttrs);

Expr repeat(Expr data, int repeats, Optional<Integer> axis) {
  auto attrs = make_object<RepeatAttrs>();
  attrs->repeats = std::move(repeats);
  attrs->axis = std::move(axis);

  static const Op& op = Op::Get("relax.repeat");
  return Call(op, {std::move(data)}, Attrs{attrs}, {});
}

TVM_REGISTER_GLOBAL("relax.op.repeat").set_body_typed(repeat);

StructInfo InferStructInfoRepeat(const Call& call, const BlockBuilder& ctx) {
  arith::Analyzer* analyzer = ctx->GetAnalyzer();
  TensorStructInfo data_sinfo = GetUnaryInputTensorStructInfo(call, ctx);
  const auto* attrs = call->attrs.as<RepeatAttrs>();
  const auto* data_shape = data_sinfo->shape.as<ShapeExprNode>();

  if (attrs->axis.defined() && !data_sinfo->IsUnknownNdim()) {
    int axis = attrs->axis.value()->value;
    int ndim = data_sinfo->ndim;
    if (axis < -ndim || axis >= ndim) {
      ctx->ReportFatal(
          Diagnostic::Error(call)
          << "Repeat requires the input axis belongs range "
             "[-data.struct_info.ndim, data.struct_info.ndim - 1]. However, the input axis is "
          << axis << ", while ndim is " << ndim);
    }
  }

  if (data_shape == nullptr) {
    if (attrs->axis.defined()) {
      if (analyzer->CanProveEqual(attrs->repeats, 1)) {
        // the shape does not changes
        return data_sinfo;
      } else {
        return TensorStructInfo(data_sinfo->dtype, data_sinfo->ndim, data_sinfo->vdevice);
      }
    } else {
      return TensorStructInfo(data_sinfo->dtype, 1, data_sinfo->vdevice);
    }
  }

  if (!attrs->axis.defined()) {
    PrimExpr new_shape =
        analyzer->Simplify(ComputeShapeProduct(data_shape->values) * attrs->repeats);
    return TensorStructInfo(ShapeExpr(Array<PrimExpr>({new_shape})), data_sinfo->dtype,
                            data_sinfo->vdevice);
  }

  int axis = NormalizeAxis(call, ctx, data_sinfo->ndim, attrs->axis.value()->value);
  auto shape_array = data_shape->values;
  shape_array.Set(axis, analyzer->Simplify(shape_array[axis] * attrs->repeats));
  return TensorStructInfo(ShapeExpr(shape_array), data_sinfo->dtype, data_sinfo->vdevice);
}

// TODO(relax-team): implement FRelaxInferLayout for repeat
TVM_REGISTER_OP("relax.repeat")
    .set_attrs_type<RepeatAttrs>()
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor.")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoRepeat)
    .set_attr<Bool>("FPurity", Bool(true));

/* relax.tile */
TVM_REGISTER_NODE_TYPE(TileAttrs);

Expr tile(Expr data, Array<Integer> repeats) {
  auto attrs = make_object<TileAttrs>();
  attrs->repeats = std::move(repeats);

  static const Op& op = Op::Get("relax.tile");
  return Call(op, {std::move(data)}, Attrs{attrs}, {});
}

TVM_REGISTER_GLOBAL("relax.op.tile").set_body_typed(tile);

StructInfo InferStructInfoTile(const Call& call, const BlockBuilder& ctx) {
  arith::Analyzer* analyzer = ctx->GetAnalyzer();
  TensorStructInfo data_sinfo = GetUnaryInputTensorStructInfo(call, ctx);
  const auto* attrs = call->attrs.as<TileAttrs>();
  const auto* data_shape = data_sinfo->shape.as<ShapeExprNode>();
  int l = attrs->repeats.size();
  int ndim = data_sinfo->ndim;

  if (data_shape == nullptr) {
    if (data_sinfo->IsUnknownNdim()) {
      return TensorStructInfo(data_sinfo->dtype, kUnknownNDim, data_sinfo->vdevice);
    }
    if (l > ndim) {
      return TensorStructInfo(data_sinfo->dtype, l, data_sinfo->vdevice);
    } else {
      for (auto i : attrs->repeats) {
        if (!analyzer->CanProveEqual(i, 1)) {
          return TensorStructInfo(data_sinfo->dtype, data_sinfo->ndim, data_sinfo->vdevice);
        }
      }
      // if control reaches here, the shape should not be changed
      return data_sinfo;
    }
  }

  int out_ndim = std::max(l, ndim);
  int l_delta = out_ndim - l;
  int ndim_delta = out_ndim - ndim;
  Array<PrimExpr> out_shape;
  for (int i = 0; i < out_ndim; ++i) {
    if (i < l_delta) {
      out_shape.push_back(data_shape->values[i - ndim_delta]);
    } else if (i < ndim_delta) {
      out_shape.push_back(attrs->repeats[i - l_delta]);
    } else {
      out_shape.push_back(
          analyzer->Simplify(data_shape->values[i - ndim_delta] * attrs->repeats[i - l_delta]));
    }
  }

  return TensorStructInfo(ShapeExpr(out_shape), data_sinfo->dtype, data_sinfo->vdevice);
}

// TODO(relax-team): implement FRelaxInferLayout for tile
TVM_REGISTER_OP("relax.tile")
    .set_attrs_type<TileAttrs>()
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor.")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoTile)
    .set_attr<Bool>("FPurity", Bool(true));

/* relax.flip */
TVM_REGISTER_NODE_TYPE(FlipAttrs);

Expr flip(Expr data, Integer axis) {
  auto attrs = make_object<FlipAttrs>();
  attrs->axis = std::move(axis);
  static const Op& op = Op::Get("relax.flip");
  return Call(op, {std::move(data)}, Attrs{attrs}, {});
}

TVM_REGISTER_GLOBAL("relax.op.flip").set_body_typed(flip);

StructInfo InferStructInfoFlip(const Call& call, const BlockBuilder& ctx) {
  if (call->args.size() != 1) {
    ctx->ReportFatal(Diagnostic::Error(call) << "Flip op should take 1 argument");
  }
  TensorStructInfo data_sinfo = GetUnaryInputTensorStructInfo(call, ctx);
  const auto* attrs = call->attrs.as<FlipAttrs>();
  int axis = attrs->axis.IntValue();
  if (!data_sinfo->IsUnknownNdim()) {
    int ndim = data_sinfo->ndim;
    if (axis < -ndim || axis >= ndim) {
      ctx->ReportFatal(Diagnostic::Error(call) << "Flip requires the input axis belongs range "
                                                  "[-ndim, ndim - 1]. However, the input axis is "
                                               << axis << ", while ndim is " << ndim);
    }
  }
  return data_sinfo;
}

TVM_REGISTER_OP("relax.flip")
    .set_attrs_type<FlipAttrs>()
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor.")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoFlip)
    .set_attr<Bool>("FPurity", Bool(true));

/* relax.gather_elements */
TVM_REGISTER_NODE_TYPE(GatherElementsAttrs);

Expr gather_elements(Expr data, Expr indices, int axis) {
  auto attrs = make_object<GatherElementsAttrs>();
  attrs->axis = Integer(axis);
  static const Op& op = Op::Get("relax.gather_elements");
  return Call(op, {data, indices}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relax.op.gather_elements").set_body_typed(gather_elements);

StructInfo InferStructInfoGatherElements(const Call& call, const BlockBuilder& ctx) {
  const auto* data_sinfo = GetStructInfoAs<TensorStructInfoNode>(call->args[0]);
  const auto* indices_sinfo = GetStructInfoAs<TensorStructInfoNode>(call->args[1]);
  const auto* attrs = call->attrs.as<GatherElementsAttrs>();

  if (data_sinfo == nullptr) {
    ctx->ReportFatal(
        Diagnostic::Error(call)
        << "GatherElements requires the input data to be a Tensor. However, the given one is "
        << call->args[0]->struct_info_->GetTypeKey());
  }
  if (indices_sinfo == nullptr) {
    ctx->ReportFatal(
        Diagnostic::Error(call)
        << "GatherElements requires the input indices to be a Tensor. However, the given one is "
        << call->args[1]->struct_info_->GetTypeKey());
  }

  if (!indices_sinfo->IsUnknownDtype() && !indices_sinfo->dtype.is_int()) {
    ctx->ReportFatal(
        Diagnostic::Error(call)
        << "GatherElements requires the input indices to have int64 dtype. However, the "
        << "given indices dtype is " << indices_sinfo->dtype);
  }

  if (data_sinfo->IsUnknownNdim() || indices_sinfo->IsUnknownNdim()) {
    return TensorStructInfo(data_sinfo->dtype, kUnknownNDim, data_sinfo->vdevice);
  }

  int axis = attrs->axis.IntValue();
  if (axis < -data_sinfo->ndim || axis >= data_sinfo->ndim) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "GatherElements requires axis to be within the input dimension range ["
                     << -data_sinfo->ndim << ", " << data_sinfo->ndim - 1 << "]. However, the "
                     << "given axis is " << axis);
  }

  if (data_sinfo->ndim != indices_sinfo->ndim) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "GatherElements requires data and indices to have the same rank. However, "
                     << "data rank is " << data_sinfo->ndim << " while indices rank is "
                     << indices_sinfo->ndim);
  }
  if (indices_sinfo->shape.defined()) {
    return TensorStructInfo(indices_sinfo->shape.value(), data_sinfo->dtype, data_sinfo->vdevice);
  }
  return TensorStructInfo(data_sinfo->dtype, indices_sinfo->ndim, data_sinfo->vdevice);
}

TVM_REGISTER_OP("relax.gather_elements")
    .set_attrs_type<GatherElementsAttrs>()
    .set_num_inputs(2)
    .add_argument("data", "Tensor", "The input tensor.")
    .add_argument("indices", "Tensor", "The indices tensor.")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoGatherElements)
    .set_attr<Bool>("FPurity", Bool(true));

/* relax.gather_nd */
TVM_REGISTER_NODE_TYPE(GatherNDAttrs);

Expr gather_nd(Expr data, Expr indices, int batch_dims) {
  auto attrs = make_object<GatherNDAttrs>();
  attrs->batch_dims = Integer(batch_dims);
  static const Op& op = Op::Get("relax.gather_nd");
  return Call(op, {data, indices}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relax.op.gather_nd").set_body_typed(gather_nd);

StructInfo InferStructInfoGatherND(const Call& call, const BlockBuilder& ctx) {
  const auto* data_sinfo = GetStructInfoAs<TensorStructInfoNode>(call->args[0]);
  const auto* indices_sinfo = GetStructInfoAs<TensorStructInfoNode>(call->args[1]);
  const auto* attrs = call->attrs.as<GatherNDAttrs>();

  if (data_sinfo == nullptr) {
    ctx->ReportFatal(
        Diagnostic::Error(call)
        << "GatherND requires the input data to be a Tensor. However, the given one is "
        << call->args[0]->struct_info_->GetTypeKey());
  }
  if (indices_sinfo == nullptr) {
    ctx->ReportFatal(
        Diagnostic::Error(call)
        << "GatherND requires the input indices to be a Tensor. However, the given one is "
        << call->args[1]->struct_info_->GetTypeKey());
  }
  ICHECK_GE(attrs->batch_dims.IntValue(), 0);
  int batch_dims = attrs->batch_dims.IntValue();
  int input_dims = data_sinfo->ndim;
  if (!indices_sinfo->IsUnknownDtype() && indices_sinfo->dtype != DataType::Int(64)) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "GatherND requires the input indices to have int64 dtype. However, the "
                     << "given indices dtype is " << indices_sinfo->dtype);
  }

  if (data_sinfo->IsUnknownNdim() || indices_sinfo->IsUnknownNdim()) {
    return TensorStructInfo(data_sinfo->dtype, kUnknownNDim, data_sinfo->vdevice);
  }

  if (batch_dims < 0 || batch_dims > data_sinfo->ndim) {
    ctx->ReportFatal(
        Diagnostic::Error(call)
        << "GatherND batch_dims must be in range [0, data.ndim]. However, got batch_dims="
        << batch_dims << ", data.ndim=" << input_dims);
  }

  if (batch_dims > indices_sinfo->ndim - 1) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "GatherND batch_dims cannot exceed indices.ndim-1. However, got batch_dims="
                     << batch_dims << ", indices.ndim=" << indices_sinfo->ndim);
  }

  // Check if indices shape is known
  const auto* indices_shape = indices_sinfo->shape.as<ShapeExprNode>();
  const auto* data_shape = data_sinfo->shape.as<ShapeExprNode>();
  if (!indices_shape || !indices_shape->values.back()->IsInstance<IntImmNode>()) {
    return TensorStructInfo(data_sinfo->dtype, kUnknownNDim, data_sinfo->vdevice);
  }
  int l = indices_shape->values.back().as<IntImmNode>()->value;
  int output_ndim = indices_sinfo->ndim + input_dims - l - 1 - batch_dims;
  if (!data_shape) {
    return TensorStructInfo(data_sinfo->dtype, output_ndim, data_sinfo->vdevice);
  }

  // In this condition, all input shapes are known
  Array<PrimExpr> out_shape;
  if (l > input_dims - batch_dims) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "GatherND requires the last dimension of indices to be less than or "
                        "equal to the rank of data minus batch_dims. However, the given shapes are "
                     << "indices: " << ShapeExpr(indices_shape->values) << ", data: "
                     << ShapeExpr(data_shape->values) << ", with batch_dims=" << batch_dims);
  }
  for (int i = 0; i < indices_sinfo->ndim - 1; ++i) {
    out_shape.push_back(indices_shape->values[i]);
  }
  for (int i = batch_dims + l; i < input_dims; ++i) {
    out_shape.push_back(data_shape->values[i]);
  }
  ICHECK_EQ(out_shape.size(), output_ndim);
  return TensorStructInfo(ShapeExpr(out_shape), data_sinfo->dtype, data_sinfo->vdevice);
}

TVM_REGISTER_OP("relax.gather_nd")
    .set_attrs_type<GatherNDAttrs>()
    .set_num_inputs(2)
    .add_argument("data", "Tensor", "The input tensor.")
    .add_argument("indices", "Tensor", "The indices tensor.")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoGatherND)
    .set_attr<Bool>("FPurity", Bool(true));

/* relax.scatter_elements */
TVM_REGISTER_NODE_TYPE(ScatterElementsAttrs);

Expr scatter_elements(Expr data, Expr indices, Expr updates, int axis, String reduction) {
  auto attrs = make_object<ScatterElementsAttrs>();
  attrs->axis = std::move(axis);
  attrs->reduction = std::move(reduction);
  static const Op& op = Op::Get("relax.scatter_elements");
  return Call(op, {data, indices, updates}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relax.op.scatter_elements").set_body_typed(scatter_elements);

StructInfo InferStructInfoScatterElements(const Call& call, const BlockBuilder& ctx) {
  arith::Analyzer* analyzer = ctx->GetAnalyzer();
  const auto* data_sinfo = GetStructInfoAs<TensorStructInfoNode>(call->args[0]);
  const auto* indices_sinfo = GetStructInfoAs<TensorStructInfoNode>(call->args[1]);
  const auto* updates_sinfo = GetStructInfoAs<TensorStructInfoNode>(call->args[2]);

  auto diag_def = [&](const TensorStructInfoNode* sinfo, String name, String type_key) {
    if (sinfo == nullptr) {
      ctx->ReportFatal(Diagnostic::Error(call)
                       << "ScatterElements requires the input " << name
                       << " to be a Tensor. However, the given one is " << type_key);
    }
  };

  diag_def(data_sinfo, "data", call->args[0]->struct_info_->GetTypeKey());
  diag_def(indices_sinfo, "indices", call->args[1]->struct_info_->GetTypeKey());
  diag_def(updates_sinfo, "updates", call->args[2]->struct_info_->GetTypeKey());

  if (data_sinfo->IsUnknownNdim()) {
    // When `data` has unknown rank, assume rest of arguments are correct and proceed.
    // If the assumption turns out to be wrong, runtime error will be triggered.
    return TensorStructInfo(data_sinfo->dtype, kUnknownNDim, data_sinfo->vdevice);
  }

  if (!indices_sinfo->IsUnknownNdim() && !updates_sinfo->IsUnknownNdim()) {
    if (data_sinfo->ndim != indices_sinfo->ndim) {
      ctx->ReportFatal(Diagnostic::Error(call)
                       << "ScatterElements op requires the data tensor to have the same rank with "
                          "indices tensor. However, the given dimensions are "
                       << "indices: " << indices_sinfo->ndim << ", data: " << data_sinfo->ndim);
    }

    if (indices_sinfo->ndim != updates_sinfo->ndim) {
      ctx->ReportFatal(
          Diagnostic::Error(call)
          << "ScatterElements op requires the indices tensor to have the same rank with "
             "updates tensor. However, the given dimensions are "
          << "indices: " << indices_sinfo->ndim << ", updates: " << updates_sinfo->ndim);
    }
  }

  if (data_sinfo->IsUnknownDtype() || updates_sinfo->IsUnknownDtype()) {
    auto diag_dtype = [&](const TensorStructInfoNode* sinfo, String name) {
      if (sinfo->IsUnknownDtype()) {
        // TODO(tvm-team): Do we have an equivalent of `ctx->ReportFatal` for warning?
        LOG(WARNING) << "Data type of " << name
                     << " has not been specified. Assume it has an integer type.";
      }
    };
    diag_dtype(data_sinfo, "data");
    diag_dtype(data_sinfo, "updates");
  } else {
    if (data_sinfo->dtype != updates_sinfo->dtype) {
      ctx->ReportFatal(Diagnostic::Error(call)
                       << "ScatterElements op requires the input data to have same type with "
                          "updates. However, the given types are "
                       << "data: " << data_sinfo->dtype << ", updates: " << updates_sinfo->dtype);
    }
  }

  if (indices_sinfo->IsUnknownDtype()) {
    // TODO(tvm-team): Do we have an equivalent of `ctx->ReportFatal` for warning?
    LOG(WARNING) << "Data type of indice has not been specified. Assume it has an integer type.";
  } else if (!(indices_sinfo->dtype.is_int() || indices_sinfo->dtype.is_uint())) {
    ctx->ReportFatal(
        Diagnostic::Error(call)
        << "ScatterElements op requires the input indices to have integer dtype. However, the "
           "given indices dtype is "
        << indices_sinfo->dtype);
  }

  const auto* indices_shape = indices_sinfo->shape.as<ShapeExprNode>();
  const auto* updates_shape = updates_sinfo->shape.as<ShapeExprNode>();
  if (indices_shape && updates_shape) {
    for (int i = 0; i < indices_sinfo->ndim; i++) {
      if (analyzer->CanProve(indices_shape->values[i] != updates_shape->values[i])) {
        ctx->ReportFatal(
            Diagnostic::Error(call)
            << "ScatterElements op requires the indices tensor to have the same shape with "
               "updates tensor. However, the given shapes are "
            << "indices: " << ShapeExpr(indices_shape->values)
            << ", updates: " << ShapeExpr(updates_shape->values));
      }
    }
  }
  const auto* data_shape = data_sinfo->shape.as<ShapeExprNode>();
  if (data_shape) {
    return TensorStructInfo(ShapeExpr(data_shape->values), data_sinfo->dtype, data_sinfo->vdevice);
  }
  return TensorStructInfo(data_sinfo->dtype, data_sinfo->ndim, data_sinfo->vdevice);
}

// TODO(relax-team): implement FRelaxInferLayout for scatter_elements
TVM_REGISTER_OP("relax.scatter_elements")
    .set_attrs_type<ScatterElementsAttrs>()
    .set_num_inputs(3)
    .add_argument("data", "Tensor", "The input tensor.")
    .add_argument("indices", "Tensor", "The indices tensor.")
    .add_argument("updates", "Tensor", "The input tensor of updates.")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoScatterElements)
    .set_attr<Bool>("FPurity", Bool(true));

/* relax.scatter_nd */
TVM_REGISTER_NODE_TYPE(ScatterNDAttrs);

Expr scatter_nd(Expr data, Expr indices, Expr updates, String reduction) {
  auto attrs = make_object<ScatterNDAttrs>();
  attrs->reduction = std::move(reduction);
  static const Op& op = Op::Get("relax.scatter_nd");
  return Call(op, {data, indices, updates}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relax.op.scatter_nd").set_body_typed(scatter_nd);

StructInfo InferStructInfoScatterND(const Call& call, const BlockBuilder& ctx) {
  // `call->args` contains: [data, indices, updates]
  arith::Analyzer* analyzer = ctx->GetAnalyzer();
  ICHECK_EQ(call->args.size(), 3);
  const auto* data_sinfo = GetStructInfoAs<TensorStructInfoNode>(call->args[0]);
  const auto* indices_sinfo = GetStructInfoAs<TensorStructInfoNode>(call->args[1]);
  const auto* updates_sinfo = GetStructInfoAs<TensorStructInfoNode>(call->args[2]);

  if (data_sinfo == nullptr) {
    ctx->ReportFatal(
        Diagnostic::Error(call)
        << "ScatterND op requires the input data to be a tensor. However, the given type is "
        << call->args[0]->GetTypeKey());
  }
  if (indices_sinfo == nullptr) {
    ctx->ReportFatal(
        Diagnostic::Error(call)
        << "ScatterND op requires the input indices to be a tensor. However, the given type is "
        << call->args[1]->GetTypeKey());
  }
  if (updates_sinfo == nullptr) {
    ctx->ReportFatal(
        Diagnostic::Error(call)
        << "ScatterND op requires the input updates to be a tensor. However, the given type is "
        << call->args[2]->GetTypeKey());
  }

  if (data_sinfo->IsUnknownDtype() || updates_sinfo->IsUnknownDtype()) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "ScatterND op requires the input data and updates to have known dtype. "
                        "However, the given types are "
                     << "data: " << data_sinfo->dtype << ", updates: " << updates_sinfo->dtype);
  }

  if (data_sinfo->dtype != updates_sinfo->dtype) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "ScatterND op requires the input data to have same type with updates. "
                        "However, the given types are "
                     << "data: " << data_sinfo->dtype << ", updates: " << updates_sinfo->dtype);
  }

  if (indices_sinfo->IsUnknownDtype()) {
    LOG(WARNING) << "Data type of indices has not been specified. Assume it has an integer type.";
  } else if (!(indices_sinfo->dtype.is_int() || indices_sinfo->dtype.is_uint())) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "ScatterND op requires the input indices to have integer dtype. However, "
                        "the given indices dtype is "
                     << indices_sinfo->dtype);
  }

  const auto* data_shape = data_sinfo->shape.as<ShapeExprNode>();
  const auto* indices_shape = indices_sinfo->shape.as<ShapeExprNode>();
  const auto* updates_shape = updates_sinfo->shape.as<ShapeExprNode>();

  if (data_shape && indices_shape && updates_shape) {
    const IntImmNode* k_dim = indices_shape->values[indices_sinfo->ndim - 1].as<IntImmNode>();
    if (!k_dim) {
      ctx->ReportFatal(Diagnostic::Error(call)
                       << "ScatterND needs a static shape for the last axis of indices, got "
                       << indices_shape->values);
    }
    const size_t data_ndim = data_sinfo->ndim;
    const size_t indices_ndim = indices_sinfo->ndim;
    const size_t updates_ndim = updates_sinfo->ndim;
    if (data_ndim + indices_ndim - k_dim->value - 1 != updates_ndim) {
      ctx->ReportFatal(Diagnostic::Error(call)
                       << "ScatterND op requires the updates tensor to have the rank of "
                          "`data tensor + indices tensor - last axis of indices tensor - 1`. "
                          "However, the given shapes are "
                       << "data: " << ShapeExpr(data_shape->values)
                       << ", indices: " << ShapeExpr(indices_shape->values)
                       << ", updates: " << ShapeExpr(updates_shape->values));
    }
    if (k_dim->value > static_cast<int>(data_ndim)) {
      ctx->ReportFatal(Diagnostic::Error(call)
                       << "ScatterND op requires the last axis of indices tensor to be less than "
                          "or equal to the rank of data tensor. However, the given shapes are "
                       << "data: " << ShapeExpr(data_shape->values)
                       << ", indices: " << ShapeExpr(indices_shape->values));
    }
    Array<PrimExpr> expected_updates_shape;
    for (size_t i = 0; i < indices_ndim - 1; i++) {
      expected_updates_shape.push_back(indices_shape->values[i]);
    }
    for (size_t i = k_dim->value; i < data_ndim; i++) {
      expected_updates_shape.push_back(data_shape->values[i]);
    }
    auto check_shape = [&](const Array<PrimExpr>& expected, const Array<PrimExpr>& actual) {
      if (expected.size() != actual.size()) {
        return false;
      }
      for (size_t i = 0; i < expected.size(); i++) {
        if (!analyzer->CanProve(expected[i] == actual[i])) {
          return false;
        }
      }
      return true;
    };
    if (!check_shape(expected_updates_shape, updates_shape->values)) {
      ctx->ReportFatal(
          Diagnostic::Error(call)
          << "ScatterND op requires the updates tensor to have the shape with constraint: "
          << "`updates.shape = indices.shape[:-1] + data.shape[K:]`, but got "
          << "updates.shape: " << ShapeExpr(updates_shape->values) << ", indices.shape: "
          << ShapeExpr(indices_shape->values) << ", data.shape: " << ShapeExpr(data_shape->values));
    }
  }
  if (data_shape) {
    return TensorStructInfo(ShapeExpr(data_shape->values), data_sinfo->dtype, data_sinfo->vdevice);
  }
  return TensorStructInfo(data_sinfo->dtype, data_sinfo->ndim, data_sinfo->vdevice);
}

TVM_REGISTER_OP("relax.scatter_nd")
    .set_attrs_type<ScatterNDAttrs>()
    .set_num_inputs(3)
    .add_argument("data", "Tensor", "The input tensor.")
    .add_argument("indices", "Tensor", "The indices tensor.")
    .add_argument("updates", "Tensor", "The input tensor of updates.")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoScatterND)
    .set_attr<Bool>("FPurity", Bool(true));

/* relax.one_hot */
TVM_REGISTER_NODE_TYPE(OneHotAttrs);
Expr one_hot(Expr indices, PrimValue on_value, PrimValue off_value, int depth, int axis) {
  ObjectPtr<OneHotAttrs> attrs = make_object<OneHotAttrs>();
  attrs->depth = depth;
  attrs->axis = axis;

  // Check if on_value and off_value have the same dtype
  DataType on_dtype = on_value->value->dtype;
  DataType off_dtype = off_value->value->dtype;
  ICHECK(on_dtype == off_dtype) << "one_hot: on_value and off_value must have the same dtype, "
                                << "but got " << on_dtype << " and " << off_dtype;

  ICHECK(depth > 0) << "one_hot: depth must be positive, but got " << depth;

  static const Op& op = Op::Get("relax.one_hot");
  return Call(op, {indices, on_value, off_value}, Attrs(attrs), {});
}  // namespace relax

TVM_REGISTER_GLOBAL("relax.op.one_hot").set_body_typed(one_hot);

StructInfo InferStructInfoOneHot(const Call& call, const BlockBuilder& ctx) {
  TensorStructInfo indices_sinfo = GetInputTensorStructInfo(call, 0, ctx);
  const auto* attrs = call->attrs.as<OneHotAttrs>();
  PrimValue on_value = Downcast<PrimValue>(call->args[1]);
  PrimValue off_value = Downcast<PrimValue>(call->args[2]);
  // Check if on_value and off_value have the same dtype
  ICHECK(on_value->value->dtype == off_value->value->dtype)
      << "one_hot: on_value and off_value must have the same dtype, "
      << "but got " << on_value->value->dtype << " and " << off_value->value->dtype;
  DataType dtype = on_value->value->dtype;

  // Check if indices has an integer dtype
  if (indices_sinfo->IsUnknownDtype()) {
    LOG(WARNING) << "Data type of indices has not been specified. Assume it has an integer type.";
  } else if (!(indices_sinfo->dtype.is_int() || indices_sinfo->dtype.is_uint())) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "one_hot op requires the input indices to have integer dtype. However, the "
                        "given indices dtype is "
                     << indices_sinfo->dtype);
  }
  // Check if indices has unknown dimension
  if (indices_sinfo->IsUnknownNdim()) {
    return TensorStructInfo(dtype, kUnknownNDim, indices_sinfo->vdevice);
  }
  // Get the shape of indices
  const auto* indices_shape = indices_sinfo->shape.as<ShapeExprNode>();
  if (indices_shape == nullptr) {
    return TensorStructInfo(dtype, indices_sinfo->ndim + 1, indices_sinfo->vdevice);
  }

  Array<PrimExpr> output_shape = indices_shape->values;
  int axis = attrs->axis;
  if (axis < 0) {
    axis += output_shape.size() + 1;
  }
  ICHECK(0 <= axis && axis <= static_cast<int>(output_shape.size()))
      << "one_hot: axis must be in the range of [0, " << output_shape.size() << "], "
      << "but got " << axis;
  output_shape.insert(output_shape.begin() + axis, attrs->depth);

  return TensorStructInfo(ShapeExpr(output_shape), dtype, indices_sinfo->vdevice);
}

TVM_REGISTER_OP("relax.one_hot")
    .set_attrs_type<OneHotAttrs>()
    .set_num_inputs(3)
    .add_argument("indices", "Tensor", "The indices tensor.")
    .add_argument("on_value", "PrimValue", "The value to fill at specified indices.")
    .add_argument("off_value", "PrimValue", "The value to fill at other indices.")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoOneHot)
    .set_attr<Bool>("FPurity", Bool(true));

}  // namespace relax
}  // namespace tvm
