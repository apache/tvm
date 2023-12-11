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

  if (indices_sinfo->IsUnknownDtype()) {
    // TODO(tvm-team): Do we have an equivalent of `ctx->ReportFatal` for warning?
    LOG(WARNING) << "Data type of indice has not been specified. Assume it has an integer type.";
  } else if (!(indices_sinfo->dtype.is_int() || indices_sinfo->dtype.is_uint())) {
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
  if (data_sinfo->IsUnknownNdim() || indices_sinfo->IsUnknownNdim()) {
    return TensorStructInfo(data_sinfo->dtype, kUnknownNDim, data_sinfo->vdevice);
  }

  int axis = attrs->axis.defined()
                 ? NormalizeAxis(call, ctx, data_sinfo->ndim, attrs->axis.value()->value)
                 : 0;
  const auto* data_shape = data_sinfo->shape.as<ShapeExprNode>();
  const auto* indices_shape = indices_sinfo->shape.as<ShapeExprNode>();
  if (data_shape == nullptr || indices_shape == nullptr) {
    return TensorStructInfo(data_sinfo->dtype, indices_sinfo->ndim + data_sinfo->ndim - 1,
                            data_sinfo->vdevice);
  }

  Array<PrimExpr> output_shape;
  for (int i = 0; i < data_sinfo->ndim; i++) {
    if (i == axis) {
      for (int j = 0; j < indices_sinfo->ndim; j++)
        output_shape.push_back(indices_shape->values[j]);
    } else {
      output_shape.push_back(data_shape->values[i]);
    }
  }
  return TensorStructInfo(ShapeExpr(output_shape), data_sinfo->dtype, data_sinfo->vdevice);
}

TVM_REGISTER_OP("relax.take")
    .set_attrs_type<TakeAttrs>()
    .set_num_inputs(2)
    .add_argument("x", "Tensor", "The source tensor.")
    .add_argument("indices", "Tensor", "The indices of the values to extract.")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoTake)
    .set_attr<Bool>("FPurity", Bool(true));

/* relax.strided_slice */
TVM_REGISTER_NODE_TYPE(StridedSliceAttrs);

Expr strided_slice(Expr x,                             //
                   Array<Integer> axes,                //
                   Array<PrimExpr> begin,              //
                   Array<PrimExpr> end,                //
                   Optional<Array<PrimExpr>> strides,  //
                   bool assume_inbound) {
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
  attrs->assume_inbound = assume_inbound;

  static const Op& op = Op::Get("relax.strided_slice");
  return Call(op, {std::move(x)}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relax.op.strided_slice").set_body_typed(strided_slice);

inline PrimExpr CanonicalizeIndex(PrimExpr index, PrimExpr extent, int64_t stride,
                                  bool assume_inbound) {
  // Same as topi strided slice CanonicalizeIndex function in
  // include/tvm/topi/detail/strided_slice.h
  PrimExpr begin_range = stride < 0 ? -1 : 0;
  PrimExpr end_range = stride < 0 ? extent - 1 : extent;
  index = if_then_else(index < 0, index + extent, index);
  return assume_inbound ? index : min(max(index, begin_range), end_range);  // NOLINT
}

PrimExpr GetLength(PrimExpr begin, PrimExpr end, const int64_t stride, const PrimExpr& length,
                   bool assume_inbound) {
  begin = CanonicalizeIndex(begin, length, stride, assume_inbound);
  end = CanonicalizeIndex(end, length, stride, assume_inbound);
  arith::Analyzer ana;
  if (stride < 0) {
    return ana.Simplify(ceildiv(begin - end, IntImm(DataType::Int(64), -stride)));
  } else {
    return ana.Simplify(ceildiv(end - begin, IntImm(DataType::Int(64), stride)));
  }
}

StructInfo InferStructInfoStridedSlice(const Call& call, const BlockBuilder& ctx) {
  TensorStructInfo data_sinfo = GetUnaryInputTensorStructInfo(call, ctx);
  const auto* attrs = call->attrs.as<StridedSliceAttrs>();
  if (attrs->axes.empty()) {
    return data_sinfo;
  }

  if (data_sinfo->IsUnknownNdim()) {
    return TensorStructInfo(data_sinfo->dtype, kUnknownNDim, data_sinfo->vdevice);
  }

  std::vector<int> axes = NormalizeAxes(call, ctx, data_sinfo->ndim, attrs->axes);
  const auto* data_shape = data_sinfo->shape.as<ShapeExprNode>();
  if (data_shape == nullptr) {
    return TensorStructInfo(data_sinfo->dtype, data_sinfo->ndim, data_sinfo->vdevice);
  }

  int n_axis = axes.size();
  Array<PrimExpr> strides = attrs->strides.defined()
                                ? attrs->strides.value()
                                : Array<PrimExpr>(n_axis, IntImm(DataType::Int(64), 1));
  std::vector<int64_t> int_strides;
  int_strides.reserve(n_axis);
  // Only do output shape inference when all the begin/end/strides values are integers.
  for (int i = 0; i < n_axis; ++i) {
    const auto* int_stride = strides[i].as<IntImmNode>();
    if (!int_stride) {
      return TensorStructInfo(data_sinfo->dtype, data_sinfo->ndim, data_sinfo->vdevice);
    }
    int_strides.push_back(int_stride->value);
  }

  Array<PrimExpr> output_shape = data_shape->values;
  for (int i = 0; i < n_axis; ++i) {
    ICHECK_NE(int_strides[i], 0)
        << "Strided slice requires strides to be non-zero but got 0 for axis " << axes[i] << ".";
    output_shape.Set(axes[i], GetLength(attrs->begin[i], attrs->end[i], int_strides[i],
                                        data_shape->values[axes[i]], attrs->assume_inbound));
  }
  return TensorStructInfo(ShapeExpr(output_shape), data_sinfo->dtype, data_sinfo->vdevice);
}

InferLayoutOutput InferLayoutStridedSlice(const Call& call,
                                          const Map<String, Array<String>>& desired_layouts,
                                          const VarLayoutMap& var_layout_map) {
  ICHECK(NoDesiredLayout(call, desired_layouts));

  const auto* attrs = call->attrs.as<StridedSliceAttrs>();
  ICHECK(attrs != nullptr) << "Invalid Call";
  const auto* tensor_sinfo = GetStructInfoAs<TensorStructInfoNode>(call->args[0]);
  ICHECK(tensor_sinfo != nullptr) << "Invalid Call";
  ICHECK(!tensor_sinfo->IsUnknownNdim()) << "Only support known ndim";
  LayoutDecision existing_layout = GetLayoutDecision(var_layout_map, call->args[0]);
  std::vector<Integer> new_axes;
  for (const auto& axis : attrs->axes) {
    new_axes.push_back(FindAxis(existing_layout->layout, axis->value));
  }
  ObjectPtr<StridedSliceAttrs> new_attrs = make_object<StridedSliceAttrs>(*attrs);
  new_attrs->axes = std::move(new_axes);
  return InferLayoutOutput({existing_layout}, {existing_layout}, Attrs(new_attrs));
}

TVM_REGISTER_OP("relax.strided_slice")
    .set_attrs_type<StridedSliceAttrs>()
    .set_num_inputs(1)
    .add_argument("x", "Tensor", "The source tensor to be sliced.")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoStridedSlice)
    .set_attr<FRelaxInferLayout>("FRelaxInferLayout", InferLayoutStridedSlice)
    .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kFollow)
    .set_attr<Bool>("FPurity", Bool(true));

/* relax.dynamic_strided_slice */
Expr dynamic_strided_slice(Expr x,      //
                           Expr begin,  //
                           Expr end,    //
                           Expr strides) {
  static const Op& op = Op::Get("relax.dynamic_strided_slice");
  return Call(op, {std::move(x), std::move(begin), std::move(end), std::move(strides)}, {});
}

TVM_REGISTER_GLOBAL("relax.op.dynamic_strided_slice").set_body_typed(dynamic_strided_slice);

StructInfo InferStructInfoDynStridedSlice(const Call& call, const BlockBuilder& ctx) {
  const auto* data_sinfo = GetStructInfoAs<TensorStructInfoNode>(call->args[0]);
  const auto* begin_sinfo = GetStructInfoAs<TensorStructInfoNode>(call->args[1]);
  const auto* end_sinfo = GetStructInfoAs<TensorStructInfoNode>(call->args[2]);
  const auto* strides_sinfo = GetStructInfoAs<TensorStructInfoNode>(call->args[3]);

  ICHECK(data_sinfo);
  if (data_sinfo->IsUnknownNdim()) {
    LOG(WARNING) << "When data rank is unknown, dynamic strided slice assumes begin/end/strides "
                    "tensors are well-formed. It could produce runtime error when this assumption "
                    "turns out to be wrong.";
    return TensorStructInfo(data_sinfo->dtype, kUnknownNDim, data_sinfo->vdevice);
  }
  if (data_sinfo->IsUnknownDtype()) {
    LOG(WARNING) << "When data type is unknown, dynamic strided slice assumes to have a valid "
                    "dtype. It could produce runtime error when this assumption "
                    "turns out to be wrong.";
  }

  int n_axis = data_sinfo->ndim;
  auto diag_def = [&](const TensorStructInfoNode* sinfo, String name) {
    ICHECK(sinfo) << "Dynamic strided slice requires the input " << name
                  << " to be have the struct info. Please try normalizing the inputs.";
    CHECK_EQ(sinfo->ndim, 1) << "Dynamic strided slice requires " << name
                             << " to be 1d tensor (list of values).";
    const auto* shape = sinfo->shape.as<ShapeExprNode>();
    ICHECK(shape) << "Dynamic strided slice requires the input " << name
                  << " to have well-defined shape.";
    // NOTE(tvm-team): This strong restriction seems necessary for now until we have a generic
    // solution in converting 1d Tensor with unknown num_elem to Array<PrimExpr>.
    const auto* num_elem = shape->values[0].as<IntImmNode>();
    ICHECK(num_elem) << "Dynamic strided slice requires the input " << name
                     << " to have a known integer shape value.";
    CHECK_EQ(num_elem->value, n_axis) << "Dynamic strided slice requires the number of indices in "
                                      << name << " to equal the number of axes.";
    if (sinfo->IsUnknownDtype()) {
      LOG(WARNING) << "Dynamic strided slice assumes " << name
                   << " to be int64 when it is not specified.";
    } else {
      CHECK(sinfo->dtype == DataType::Int(64))
          << "Dynamic strided_slice expects the input " << name
          << "values to be all int64. However, " << name << " has dtype " << sinfo->dtype << ".";
    }
  };
  diag_def(begin_sinfo, "begin");
  diag_def(end_sinfo, "end");
  diag_def(strides_sinfo, "strides");

  // The output shape will depend on the runtime value in begin/end/strides tensors.
  // TODO(tvm-team): Currently, it is unable to express partially-static shape. Revisit when
  // PrimValue lands.
  return TensorStructInfo(data_sinfo->dtype, n_axis, data_sinfo->vdevice);
}  // namespace relax

// TODO(tvm-team): Register FRelaxInferLayout, TMixedPrecisionPolicy
TVM_REGISTER_OP("relax.dynamic_strided_slice")
    .set_num_inputs(4)
    .add_argument("x", "Tensor", "The source tensor to be sliced.")
    .add_argument("begin", "Tensor", "The indices to begin with in the slicing.")
    .add_argument("end", "Tensor", "Indices indicating end of the slice.")
    .add_argument("strides", "Tensor", "The stride values.")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoDynStridedSlice)
    .set_attr<Bool>("FPurity", Bool(true));

}  // namespace relax
}  // namespace tvm
