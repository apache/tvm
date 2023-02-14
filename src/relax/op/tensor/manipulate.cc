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
#include <utility>
#include <vector>

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
    return TensorStructInfo(/*shape=*/call->args[1], data_sinfo->dtype);
  }
  ShapeStructInfo shape_sinfo = Downcast<ShapeStructInfo>(data_sinfo->shape.value()->struct_info_);
  if (!shape_sinfo->values.defined() || !tgt_shape_sinfo->values.defined()) {
    return TensorStructInfo(/*shape=*/call->args[1], data_sinfo->dtype);
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
  return TensorStructInfo(/*shape=*/call->args[1], data_sinfo->dtype);
}

TVM_REGISTER_OP("relax.broadcast_to")
    .set_num_inputs(2)
    .add_argument("x", "Tensor", "The input tensor.")
    .add_argument("shape", "Shape", "The target shape.")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoBroadcastTo);

/* relax.concat */
TVM_REGISTER_NODE_TYPE(ConcatAttrs);

Expr concat(Expr tensors, Optional<Integer> axis) {
  ObjectPtr<ConcatAttrs> attrs = make_object<ConcatAttrs>();
  attrs->axis = std::move(axis);

  static const Op& op = Op::Get("relax.concat");
  return Call(op, {std::move(tensors)}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relax.op.concat").set_body_typed(concat);

Array<TensorStructInfo> GetTensorSInfoFromTuple(const Call& call, const BlockBuilder& ctx,
                                                const Expr& expr) {
  const auto* tuple_sinfo = GetStructInfoAs<TupleStructInfoNode>(expr);
  if (tuple_sinfo == nullptr) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << call->op
                     << " expects the input to be a Tuple of Tensors. However, the given input is "
                     << expr->struct_info_->GetTypeKey());
  }

  Array<TensorStructInfo> tensor_sinfo;
  tensor_sinfo.reserve(tuple_sinfo->fields.size());
  for (StructInfo field_sinfo : tuple_sinfo->fields) {
    const auto* field_tensor_sinfo = field_sinfo.as<TensorStructInfoNode>();
    if (field_tensor_sinfo == nullptr) {
      ctx->ReportFatal(
          Diagnostic::Error(call)
          << call->op << " expects the input to be a Tuple of Tensors. However, the given input is "
          << expr->struct_info_);
    }
    tensor_sinfo.push_back(GetRef<TensorStructInfo>(field_tensor_sinfo));
  }
  return tensor_sinfo;
}

Optional<Array<PrimExpr>> CheckConcatOutputShape(const Call& call, const BlockBuilder& ctx,
                                                 const std::vector<Array<PrimExpr>>& shape_values,
                                                 int axis) {
  bool shape_unknown = false;
  arith::Analyzer* analyzer = ctx->GetAnalyzer();
  PrimExpr concat_sum = IntImm(DataType::Int(64), 0);
  for (int d = 0; d < static_cast<int>(shape_values[0].size()); ++d) {
    // For the specified axis, we compute the sum of shape value over each tensor.
    if (d == axis) {
      for (Array<PrimExpr> shape_value : shape_values) {
        concat_sum += shape_value[d];
      }
      continue;
    }

    // For other axes, we check the equality of all tensors' shape values, to ensure safety.
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
  Array<TensorStructInfo> tensor_sinfo = GetTensorSInfoFromTuple(call, ctx, call->args[0]);
  if (tensor_sinfo.empty()) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "Concat op expects at least one tensor in the input Tuple. However, the "
                        "given input Tuple is empty.");
  }

  const auto* attrs = call->attrs.as<ConcatAttrs>();
  int output_ndim = attrs->axis.defined() ? kUnknownNDim : 1;
  DataType output_dtype = DataType::Void();
  bool shape_unknown = false;
  bool is_void_dtype = false;
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
  if (output_ndim == kUnknownNDim) {
    return tensor_sinfo.size() == 1 ? tensor_sinfo[0] : TensorStructInfo(output_dtype, output_ndim);
  }

  int axis =
      attrs->axis.defined() ? NormalizeAxis(call, ctx, output_ndim, attrs->axis.value()->value) : 0;
  // If there is only one input tensor, no action is needed.
  if (tensor_sinfo.size() == 1) {
    return tensor_sinfo[0];
  }
  if (shape_values.empty()) {
    return TensorStructInfo(output_dtype, output_ndim);
  }

  // As long as the there is known shape value, we will do the best effort check to ensure safety.
  Optional<Array<PrimExpr>> output_shape = CheckConcatOutputShape(call, ctx, shape_values, axis);

  if (shape_unknown || !output_shape.defined()) {
    return TensorStructInfo(output_dtype, output_ndim);
  } else {
    return TensorStructInfo(ShapeExpr(output_shape.value()), output_dtype);
  }
}

TVM_REGISTER_OP("relax.concat")
    .set_attrs_type<ConcatAttrs>()
    .set_num_inputs(1)
    .add_argument("tensors", "Tuple of Tensors", "The input list of tensors.")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoConcat);

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
    return TensorStructInfo(data_sinfo->dtype, kUnknownNDim);
  }

  int n_new_dim = attrs->axis.size();
  int output_ndim = data_sinfo->ndim + n_new_dim;
  std::vector<int> axes = NormalizeAxes(call, ctx, output_ndim, attrs->axis);

  const auto* data_shape = data_sinfo->shape.as<ShapeExprNode>();
  if (data_shape == nullptr) {
    return TensorStructInfo(data_sinfo->dtype, output_ndim);
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
  return TensorStructInfo(ShapeExpr(output_shape), data_sinfo->dtype);
}

TVM_REGISTER_OP("relax.expand_dims")
    .set_num_inputs(1)
    .set_attrs_type<ExpandDimsAttrs>()
    .add_argument("x", "Tensor", "The input tensor.")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoExpandDims);

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
    return TensorStructInfo(data_sinfo->dtype, /*ndim=*/1);
  } else if (data_sinfo->ndim == 0) {
    return TensorStructInfo(ShapeExpr({1}), data_sinfo->dtype);
  } else if (data_sinfo->ndim == 1) {
    return data_sinfo;
  }

  const auto* data_shape = data_sinfo->shape.as<ShapeExprNode>();
  if (data_shape == nullptr) {
    return TensorStructInfo(data_sinfo->dtype, /*ndim=*/1);
  }
  PrimExpr shape_prod = ComputeShapeProduct(data_shape->values);
  return TensorStructInfo(ShapeExpr({std::move(shape_prod)}), data_sinfo->dtype);
}

TVM_REGISTER_OP("relax.flatten")
    .set_num_inputs(1)
    .add_argument("x", "Tensor", "The input tensor.")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoFlatten);

/* relax.layout_transform */
TVM_REGISTER_NODE_TYPE(LayoutTransformAttrs);

Expr layout_transform(Expr x, tir::IndexMap index_map, Optional<PrimValue> pad_value) {
  ObjectPtr<LayoutTransformAttrs> attrs = make_object<LayoutTransformAttrs>();
  attrs->index_map = std::move(index_map);
  attrs->pad_value = std::move(pad_value);

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
    return TensorStructInfo(data_sinfo->dtype, /*ndim=*/index_map->final_indices.size());
  }

  // If rank is known, check that it is compatible with the index_map, i.e., #dims match.
  if (index_map->initial_indices.size() != static_cast<size_t>(data_sinfo->ndim)) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "number of dimensions in input must match the number of source dimensions "
                        "in index map, but got "
                     << data_sinfo->ndim << " != " << index_map->initial_indices.size());
  }

  if (!data_sinfo->shape.defined()) {
    return TensorStructInfo(data_sinfo->dtype, /*ndim=*/index_map->final_indices.size());
  }

  ShapeStructInfo shape_sinfo = Downcast<ShapeStructInfo>(data_sinfo->shape.value()->struct_info_);
  if (!shape_sinfo->values.defined()) {
    return TensorStructInfo(data_sinfo->dtype, /*ndim=*/index_map->final_indices.size());
  }

  Array<PrimExpr> output_shape = index_map->MapShape(shape_sinfo->values.value());
  return TensorStructInfo(ShapeExpr(output_shape), data_sinfo->dtype);
}

TVM_REGISTER_OP("relax.layout_transform")
    .set_num_inputs(1)
    .set_attrs_type<LayoutTransformAttrs>()
    .add_argument("x", "Tensor", "The input tensor.")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoLayoutTransform);

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
    return TensorStructInfo(data_sinfo->dtype, kUnknownNDim);
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
    return TensorStructInfo(data_sinfo->dtype, data_sinfo->ndim);
  }
  std::vector<PrimExpr> new_shape;
  new_shape.reserve(data_sinfo->ndim);
  for (int i = 0; i < data_sinfo->ndim; ++i) {
    new_shape.push_back(data_shape->values[axes[i]]);
  }
  return TensorStructInfo(ShapeExpr(new_shape), data_sinfo->dtype);
}

TVM_REGISTER_OP("relax.permute_dims")
    .set_attrs_type<PermuteDimsAttrs>()
    .set_num_inputs(1)
    .add_argument("x", "Tensor", "The input tensor.")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoPermuteDims);

/* relax.reshape */
Expr ConvertNewShapeToExpr(const Expr& data, const ObjectRef& shape) {
  if (const auto* e = shape.as<ExprNode>()) {
    return GetRef<Expr>(e);
  }

  const auto* array = shape.as<ArrayNode>();
  CHECK(array != nullptr) << "Reshape only expects the input new shape to be either an Expr or an "
                             "Array of PrimExprs. However, the given new shape is "
                          << shape;
  int dim_to_infer = -1;
  PrimExpr new_shape_prod = IntImm(DataType::Int(64), 1);
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
    if (int_len != nullptr && int_len->value == -1) {
      CHECK_EQ(dim_to_infer, -1) << "Reshape accepts at most one \"-1\" in the new shape. However, "
                                    "there are multiple \"-1\" in the given new shape  "
                                 << shape;
      dim_to_infer = i;
    } else {
      CHECK(int_len == nullptr || int_len->value > 0)
          << "Reshape requires all values in the new shape to be positive except a single \"-1\". "
             "However, the given new shape is "
          << shape;
      // We expect any symbolic not to signal the intent of -1, and therefore do no check for
      // symbolic value here.
      new_shape_prod = new_shape_prod * len;
    }
  }

  Array<PrimExpr> array_ref = GetRef<Array<PrimExpr>>(array);
  // When there is no dimension to infer, just return the input array as ShapeExpr.
  if (dim_to_infer == -1) {
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

  arith::Analyzer analyzer;
  PrimExpr old_shape_prod = ComputeShapeProduct(shape_sinfo->values.value());
  array_ref.Set(dim_to_infer, analyzer.Simplify(floordiv(old_shape_prod, new_shape_prod)));
  return ShapeExpr(array_ref);
}

Expr reshape(Expr x, ObjectRef shape) {
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
  return TensorStructInfo(call->args[1], data_sinfo->dtype);
}

TVM_REGISTER_OP("relax.reshape")
    .set_num_inputs(2)
    .add_argument("x", "Tensor", "The input tensor.")
    .add_argument("shape", "Shape", "The input new shape.")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoReshape);

/* relax.split */
TVM_REGISTER_NODE_TYPE(SplitAttrs);

Expr split(Expr x, ObjectRef indices_or_sections, int axis) {
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

  if (const auto* p_indices = attrs->indices_or_sections.as<ArrayNode>()) {
    // When there is not index, return the input tensor's struct info.
    if (p_indices->size() == 0) {
      return TupleStructInfo({data_sinfo});
    }
    // Fall back to unknown shape when the input tensor doesn't have ShapeExpr as shape.
    if (data_shape == nullptr) {
      return TupleStructInfo(Array<StructInfo>(
          p_indices->size() + 1, TensorStructInfo(data_sinfo->dtype, data_sinfo->ndim)));
    }

    ICHECK_NE(axis, -1);
    const auto* axis_length = data_shape->values[axis].as<IntImmNode>();
    // Fall back to unknown shape when the input tensor shape at the given axis is symbolic.
    if (axis_length == nullptr) {
      return TupleStructInfo(Array<StructInfo>(
          p_indices->size() + 1, TensorStructInfo(data_sinfo->dtype, data_sinfo->ndim)));
    }

    // Only do output shape inference when all the indices and the total length are integers.
    Array<IntImm> indices = GetRef<Array<IntImm>>(p_indices);
    IntImm zero(DataType::Int(64), /*value=*/0);
    indices.insert(indices.begin(), zero);
    indices.insert(indices.end(), Downcast<IntImm>(data_shape->values[axis]));

    std::vector<StructInfo> output_sinfo;
    output_sinfo.reserve(indices.size() - 1);
    for (int i = 0; i + 1 < static_cast<int>(indices.size()); ++i) {
      PrimExpr l = tvm::max(zero, indices[i]);
      PrimExpr r = tvm::min(data_shape->values[axis], indices[i + 1]);

      Array<PrimExpr> shape = data_shape->values;
      shape.Set(axis, tvm::max(zero, r - l));
      output_sinfo.push_back(TensorStructInfo(ShapeExpr(shape), data_sinfo->dtype));
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
      return TupleStructInfo(
          Array<StructInfo>(n_section, TensorStructInfo(data_sinfo->dtype, data_sinfo->ndim)));
    }
    ICHECK_NE(axis, -1);
    PrimExpr split_len = ceildiv(data_shape->values[axis], n_section);

    // Construct struct info for tensors except the last one.
    Array<PrimExpr> shape = data_shape->values;
    shape.Set(axis, split_len);
    std::vector<StructInfo> output_sinfo(n_section - 1,
                                         TensorStructInfo(ShapeExpr(shape), data_sinfo->dtype));

    // Construct struct info for the last tensor.
    shape.Set(axis, data_shape->values[axis] - split_len * (n_section - 1));
    output_sinfo.push_back(TensorStructInfo(ShapeExpr(shape), data_sinfo->dtype));
    return TupleStructInfo(output_sinfo);
  }
  ICHECK(false) << "Cannot reach here.";
  throw;
}

TVM_REGISTER_OP("relax.split")
    .set_attrs_type<SplitAttrs>()
    .set_num_inputs(1)
    .add_argument("x", "Tensor", "The input tensor.")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoSplit);

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
    return TensorStructInfo(data_sinfo->dtype, kUnknownNDim);
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
      return TensorStructInfo(data_sinfo->dtype, data_sinfo->ndim - axes.size());
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
      return TensorStructInfo(data_sinfo->dtype, kUnknownNDim);
    }
    for (int i = 0; i < data_sinfo->ndim; ++i) {
      // Whenever a dimension length is symbolic, fall back to unknown ndim.
      const auto* int_len = shape_value.value()[i].as<IntImmNode>();
      if (int_len == nullptr) {
        return TensorStructInfo(data_sinfo->dtype, kUnknownNDim);
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
      return TensorStructInfo(data_sinfo->dtype, output_shape.size());
    } else {
      return TensorStructInfo(data_sinfo->dtype, kUnknownNDim);
    }
  } else {
    return TensorStructInfo(ShapeExpr(output_shape), data_sinfo->dtype);
  }
}

TVM_REGISTER_OP("relax.squeeze")
    .set_num_inputs(1)
    .set_attrs_type<SqueezeAttrs>()
    .add_argument("x", "Tensor", "The input tensor.")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoSqueeze);

}  // namespace relax
}  // namespace tvm
