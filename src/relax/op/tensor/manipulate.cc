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

namespace tvm {
namespace relax {

/* relax.concat */
TVM_REGISTER_NODE_TYPE(ConcatAttrs);

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

// Helper function for flatten and reshape.
PrimExpr ComputeShapeProduct(const Array<PrimExpr>& shape_values) {
  PrimExpr shape_prod = IntImm(DataType::Int(64), 1);
  for (PrimExpr value : shape_values) {
    shape_prod *= value;
  }
  return shape_prod;
}

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

/* relax.split */
TVM_REGISTER_NODE_TYPE(SplitAttrs);

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
    return TensorStructInfo(collapse_target_sinfo->shape.value(), output_dtype);
  } else {
    return TensorStructInfo(output_dtype, collapse_target_sinfo->ndim);
  }
}

TVM_REGISTER_OP("relax.collapse_sum_like")
    .set_num_inputs(2)
    .add_argument("data", "Tensor", "The input tensor.")
    .add_argument("collapse_target", "Tensor",
                  "The tensor whose shape is the shape to collapse to.")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoCollapseSumLike);

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

  return TensorStructInfo(/*shape=*/call->args[1], output_dtype);
}

TVM_REGISTER_OP("relax.collapse_sum_to")
    .set_num_inputs(2)
    .add_argument("data", "Tensor", "The input tensor.")
    .add_argument("shape", "Shape", "The shape to collapse to.")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoCollapseSumTo);

/* relax.cumsum */
TVM_REGISTER_NODE_TYPE(CumsumAttrs);

Expr cumsum(Expr data, Optional<Integer> axis, DataType dtype) {
  auto attrs = make_object<CumsumAttrs>();
  attrs->axis = std::move(axis);
  attrs->dtype = std::move(dtype);

  static const Op& op = Op::Get("relax.cumsum");
  return Call(op, {std::move(data)}, Attrs{attrs}, {});
}

TVM_REGISTER_GLOBAL("relax.op.cumsum").set_body_typed(cumsum);

StructInfo InferStructInfoCumsum(const Call& call, const BlockBuilder& ctx) {
  TensorStructInfo data_sinfo = GetUnaryInputTensorStructInfo(call, ctx);
  const auto* attrs = call->attrs.as<CumsumAttrs>();

  DataType out_type = attrs->dtype.is_void() ? data_sinfo->dtype : attrs->dtype;

  if (!attrs->axis.defined()) {
    // flattened
    const auto* data_shape = data_sinfo->shape.as<ShapeExprNode>();
    if (data_shape == nullptr) {
      return TensorStructInfo(out_type, data_sinfo->ndim);
    } else {
      PrimExpr flattened_d = 1;
      for (const auto v : data_shape->values) {
        flattened_d *= v;
      }
      return TensorStructInfo(ShapeExpr(Array<PrimExpr>({flattened_d})), out_type);
    }
  }

  if (data_sinfo->shape.defined()) {
    return TensorStructInfo(data_sinfo->shape.value(), out_type);
  } else {
    return TensorStructInfo(out_type, data_sinfo->ndim);
  }
}

TVM_REGISTER_OP("relax.cumsum")
    .set_attrs_type<CumsumAttrs>()
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor.")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoCumsum);

}  // namespace relax
}  // namespace tvm
