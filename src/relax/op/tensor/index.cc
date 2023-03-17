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
 * \file index.cc
 * \brief indexing operators.
 */

#include "index.h"

#include <algorithm>
#include <utility>
#include <vector>

namespace tvm {
namespace relax {

/* relax.take */
TVM_REGISTER_NODE_TYPE(TakeAttrs);

Expr take(Expr x, Expr indices, Optional<Integer> axis) {
  ObjectPtr<TakeAttrs> attrs = make_object<TakeAttrs>();
  attrs->axis = std::move(axis);

  static const Op& op = Op::Get("relax.take");
  return Call(op, {std::move(x), std::move(indices)}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relax.op.take").set_body_typed(take);

StructInfo InferStructInfoTake(const Call& call, const BlockBuilder& ctx) {
  Array<TensorStructInfo> input_sinfo = GetInputTensorStructInfo(call, ctx);
  TensorStructInfo data_sinfo = input_sinfo[0];
  TensorStructInfo indices_sinfo = input_sinfo[1];
  if (indices_sinfo->ndim != 1) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "Take op requires the input indices to be 1-dimensional tensor. However, "
                        "the given indices ndim is "
                     << indices_sinfo->ndim);
  } else if (!indices_sinfo->IsUnknownDtype() &&
             !(indices_sinfo->dtype.is_int() || indices_sinfo->dtype.is_uint())) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "Take op requires the input indices to have integer dtype. However, the "
                        "given indices dtype is "
                     << indices_sinfo->dtype);
  }

  const auto* attrs = call->attrs.as<TakeAttrs>();
  if (!attrs->axis.defined() && data_sinfo->ndim != 1) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "Take op expects the input data to be 1-dimensional tensor when the axis "
                        "is not specified. However, the given data tensor has ndim "
                     << data_sinfo->ndim);
  }
  if (data_sinfo->IsUnknownNdim()) {
    return TensorStructInfo(data_sinfo->dtype, kUnknownNDim);
  }

  int axis = attrs->axis.defined()
                 ? NormalizeAxis(call, ctx, data_sinfo->ndim, attrs->axis.value()->value)
                 : 0;
  const auto* data_shape = data_sinfo->shape.as<ShapeExprNode>();
  const auto* indices_shape = indices_sinfo->shape.as<ShapeExprNode>();
  if (data_shape == nullptr || indices_shape == nullptr) {
    return TensorStructInfo(data_sinfo->dtype, data_sinfo->ndim);
  }

  Array<PrimExpr> output_shape = data_shape->values;
  output_shape.Set(axis, indices_shape->values[0]);
  return TensorStructInfo(ShapeExpr(output_shape), data_sinfo->dtype);
}

TVM_REGISTER_OP("relax.take")
    .set_attrs_type<TakeAttrs>()
    .set_num_inputs(2)
    .add_argument("x", "Tensor", "The source tensor.")
    .add_argument("indices", "Tensor", "The indices of the values to extract.")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoTake);

/* relax.strided_slice */
TVM_REGISTER_NODE_TYPE(StridedSliceAttrs);

Expr strided_slice(Expr x,                 //
                   Array<Integer> axes,    //
                   Array<PrimExpr> begin,  //
                   Array<PrimExpr> end,    //
                   Optional<Array<PrimExpr>> strides) {
  int n_axis = axes.size();
  CHECK_EQ(static_cast<int>(begin.size()), n_axis)
      << "StridedSlice requires the number of begin indices to equal the number of axes.";
  CHECK_EQ(static_cast<int>(end.size()), n_axis)
      << "StridedSlice requires the number of end indices to equal the number of axes.";
  if (strides.defined()) {
    CHECK_EQ(static_cast<int>(strides.value().size()), n_axis)
        << "StridedSlice requires the number of strides to equal the number of axes.";
  }

  // Todo(relax-team): We are going to support dynamic strided slice, where
  // begin/end/stride can be not static at compile time. Therefore, begin/end/stride
  // should not be part of StridedSliceAttrs, as we only allow static values to
  // reside in attributes. However, using ShapeExpr to represent these
  // arrays is not conceptually right, because they are not describing a
  // concrete shape. The proper way to support dynamic strided slice is to use
  // Tuple of PrimValue to represent begin/end/stride. Since at this moment
  // we have no support for PrimValue, we store begin/end/stride as attribute
  // fields as a workaround.
  // Will switch to Tuple of PrimValue after introducing PrimValue.
  auto f_convert_to_int64 = [](const PrimExpr& value) {
    if (value->IsInstance<IntImmNode>()) {
      return cast(DataType::Int(64), value);
    }
    CHECK(value.dtype() == DataType::Int(64)) << "strided_slice expects the input begin/end/stride "
                                                 "values to be all int64. However, the given "
                                              << value << " has dtype " << value->dtype;
    return value;
  };

  ObjectPtr<StridedSliceAttrs> attrs = make_object<StridedSliceAttrs>();
  attrs->axes = std::move(axes);
  attrs->begin = begin.Map(f_convert_to_int64);
  attrs->end = end.Map(f_convert_to_int64);
  attrs->strides = strides.defined() ? strides.value().Map(f_convert_to_int64) : strides;

  static const Op& op = Op::Get("relax.strided_slice");
  return Call(op, {std::move(x)}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relax.op.strided_slice").set_body_typed(strided_slice);

inline int64_t CanonicalizeIndex(int64_t index, int64_t extent, int64_t stride) {
  int64_t begin_range = stride < 0 ? -1 : 0;
  int64_t end_range = stride < 0 ? extent - 1 : extent;
  if (index < 0) {
    index += extent;
  }
  return std::min(std::max(index, begin_range), end_range);
}

PrimExpr GetLength(const PrimExpr& begin_expr, const PrimExpr& end_expr, const int stride,
                   const PrimExpr& ndim_expr) {
  ICHECK_NE(stride, 0) << "Stride cannot be 0.";
  if (!begin_expr->IsInstance<tvm::IntImmNode>() || !end_expr->IsInstance<tvm::IntImmNode>() ||
      !ndim_expr->IsInstance<tvm::IntImmNode>()) {
    if (stride < 0)
      return ceildiv(begin_expr - end_expr, -stride);
    else
      return ceildiv(end_expr - begin_expr, stride);
  }
  int64_t begin = begin_expr.as<IntImmNode>()->value;
  int64_t end = end_expr.as<IntImmNode>()->value;
  int64_t ndim = ndim_expr.as<IntImmNode>()->value;

  begin = CanonicalizeIndex(begin, ndim, stride);
  end = CanonicalizeIndex(end, ndim, stride);

  if (stride < 0) {
    return ceildiv(IntImm(DataType::Int(64), begin - end), -stride);
  } else {
    return ceildiv(IntImm(DataType::Int(64), end - begin), stride);
  }
}

StructInfo InferStructInfoStridedSlice(const Call& call, const BlockBuilder& ctx) {
  TensorStructInfo data_sinfo = GetUnaryInputTensorStructInfo(call, ctx);
  const auto* attrs = call->attrs.as<StridedSliceAttrs>();
  if (attrs->axes.empty()) {
    return data_sinfo;
  }

  if (data_sinfo->IsUnknownNdim()) {
    return TensorStructInfo(data_sinfo->dtype, kUnknownNDim);
  }

  std::vector<int> axes = NormalizeAxes(call, ctx, data_sinfo->ndim, attrs->axes);
  const auto* data_shape = data_sinfo->shape.as<ShapeExprNode>();
  if (data_shape == nullptr) {
    return TensorStructInfo(data_sinfo->dtype, data_sinfo->ndim);
  }

  int n_axis = axes.size();
  Array<PrimExpr> strides = attrs->strides.defined()
                                ? attrs->strides.value()
                                : Array<PrimExpr>(n_axis, IntImm(DataType::Int(64), 1));
  std::vector<int> int_strides;
  int_strides.reserve(n_axis);
  // Only do output shape inference when all the begin/end/stride values are integers.
  for (int i = 0; i < n_axis; ++i) {
    const auto* int_begin = attrs->begin[i].as<IntImmNode>();
    const auto* int_end = attrs->end[i].as<IntImmNode>();
    const auto* int_stride = strides[i].as<IntImmNode>();
    if (!int_begin || !int_end || !int_stride) {
      return TensorStructInfo(data_sinfo->dtype, data_sinfo->ndim);
    }
    int_strides.push_back(int_stride->value);
  }

  Array<PrimExpr> output_shape = data_shape->values;
  for (int i = 0; i < n_axis; ++i) {
    ICHECK_NE(int_strides[i], 0) << "strided_slice requires stride to be non-zero.";
    output_shape.Set(axes[i], GetLength(attrs->begin[i], attrs->end[i], int_strides[i],
                                        data_shape->values[axes[i]]));
  }
  return TensorStructInfo(ShapeExpr(output_shape), data_sinfo->dtype);
}

TVM_REGISTER_OP("relax.strided_slice")
    .set_attrs_type<StridedSliceAttrs>()
    .set_num_inputs(1)
    .add_argument("x", "Tensor", "The source tensor to be sliced.")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoStridedSlice);

}  // namespace relax
}  // namespace tvm
