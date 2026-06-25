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
 * \file statistical.cc
 * \brief Statistical operators.
 */

#include "statistical.h"

#include <tvm/ffi/reflection/registry.h>

#include <string>
#include <vector>

namespace tvm {
namespace relax {

TVM_FFI_STATIC_INIT_BLOCK() {
  StatisticalAttrs::RegisterReflection();
  ScanopAttrs::RegisterReflection();
}

Type InferTypeStatistical(const Call& call, const BlockBuilder& ctx) {
  TensorType data_ty = GetUnaryInputTensorType(call, ctx);
  const auto* attrs = call->attrs.as<StatisticalAttrs>();

  std::vector<int> axes;
  if (!data_ty->IsUnknownNdim() && attrs->axis.defined()) {
    axes = NormalizeAxes(call, ctx, data_ty->ndim, attrs->axis.value());
  }

  int out_ndim;
  if (attrs->keepdims) {
    out_ndim = data_ty->ndim;
  } else if (!attrs->axis.defined()) {
    out_ndim = 0;
  } else if (data_ty->IsUnknownNdim()) {
    out_ndim = kUnknownNDim;
  } else {
    out_ndim = data_ty->ndim - axes.size();
    TVM_FFI_ICHECK_GE(out_ndim, 0);
  }

  // The inference rule for reduction operator output shapes:
  // - axes is None, keepdims is false -> return the zero-rank shape;
  // - axes is None, keepdims is true -> return the shape whose ndim is the same as input and every
  // value is 1.
  // - axes is not None, keepdims is false -> the returned shape does not contain the input axes.
  // - axes is not None, keepdims is true -> the returned shape has value 1 at the positions of the
  // input axes
  const auto* data_shape = data_ty->shape.as<ShapeExprNode>();
  if (data_shape == nullptr) {
    if (!attrs->axis.defined() && attrs->keepdims && out_ndim != kUnknownNDim) {
      return TensorType(ShapeExpr(ffi::Array<PrimExpr>(out_ndim, IntImm::Int64(/*value=*/1))),
                        data_ty->dtype, data_ty->vdevice);
    } else {
      return out_ndim == 0
                 ? TensorType(ShapeExpr(ffi::Array<PrimExpr>()), data_ty->dtype, data_ty->vdevice)
                 : TensorType(data_ty->dtype, out_ndim, data_ty->vdevice);
    }
  }

  ffi::Array<PrimExpr> out_shape;
  out_shape.reserve(out_ndim);
  for (int i = 0; i < data_ty->ndim; ++i) {
    if (attrs->axis.defined() && std::find(axes.begin(), axes.end(), i) == axes.end()) {
      out_shape.push_back(data_shape->values[i]);
    } else if (attrs->keepdims) {
      out_shape.push_back(IntImm::Int64(/*value=*/1));
    }
  }
  TVM_FFI_ICHECK_EQ(static_cast<int>(out_shape.size()), out_ndim);
  return TensorType(ShapeExpr(out_shape), data_ty->dtype, data_ty->vdevice);
}

InferLayoutOutput InferLayoutStatistical(
    const Call& call, const ffi::Map<ffi::String, ffi::Array<ffi::String>>& desired_layouts,
    const VarLayoutMap& var_layout_map) {
  TVM_FFI_ICHECK(NoDesiredLayout(call, desired_layouts));

  const auto* attrs = call->attrs.as<StatisticalAttrs>();
  TVM_FFI_ICHECK(attrs != nullptr) << "Invalid Call";
  const auto* tensor_ty = GetTypeAs<TensorTypeNode>(call->args[0]);
  TVM_FFI_ICHECK(tensor_ty != nullptr) << "Invalid Call";
  TVM_FFI_ICHECK(!tensor_ty->IsUnknownNdim()) << "Only support known ndim";
  int ndim = tensor_ty->ndim;

  ffi::Array<int64_t> axis;
  if (attrs->axis.defined()) {
    axis = attrs->axis.value();
  } else {
    axis.reserve(ndim);
    for (int i = 0; i < ndim; ++i) {
      axis.push_back(i);
    }
  }

  std::string axis_str(ndim, '0');
  for (int64_t iter : axis) {
    axis_str[(iter + ndim) % ndim] = '#';
  }
  for (int i = 0, j = 0; i < ndim; ++i) {
    if (axis_str[i] != '#') {
      axis_str[i] = 'A' + j++;
    }
  }

  LayoutDecision exisiting_layout = GetLayoutDecision(var_layout_map, call->args[0]);
  auto new_axis_str = TransposeSubLayoutStrLike(axis_str, InitialLayout(ndim).name(),
                                                exisiting_layout->layout.name());
  std::string output_layout_ref = new_axis_str;
  new_axis_str.erase(std::remove_if(new_axis_str.begin(), new_axis_str.end(),
                                    [](unsigned char c) { return std::isdigit(c); }),
                     new_axis_str.end());

  ffi::Array<int64_t> new_axis;
  for (size_t i = 0; i < new_axis_str.size(); ++i) {
    if (new_axis_str.at(i) == '#') {
      new_axis.push_back(static_cast<int64_t>(i));
    }
  }
  std::string output_layout;
  for (size_t i = 0; i < output_layout_ref.length(); ++i) {
    if ((isdigit(output_layout_ref[i]) && (output_layout_ref[i + 1] == '#')) ||
        (output_layout_ref[i] == '#'))
      continue;
    output_layout.push_back(output_layout_ref[i]);
  }

  ffi::ObjectPtr<StatisticalAttrs> new_attrs = ffi::make_object<StatisticalAttrs>(*attrs);
  new_attrs->axis = new_axis;
  return InferLayoutOutput({exisiting_layout},
                           {attrs->keepdims ? exisiting_layout : SLayout(output_layout)},
                           Attrs(new_attrs));
}

Type InferTypeScan(const Call& call, const BlockBuilder& ctx) {
  TensorType data_ty = GetUnaryInputTensorType(call, ctx);
  const auto* attrs = call->attrs.as<ScanopAttrs>();

  PrimType out_type =
      attrs->dtype == DLDataType{kDLOpaqueHandle, 0, 0} ? data_ty->dtype : PrimType(attrs->dtype);

  if (!attrs->axis.has_value()) {
    // flattened
    const auto* data_shape = data_ty->shape.as<ShapeExprNode>();
    if (data_shape == nullptr) {
      return TensorType(out_type, data_ty->ndim, data_ty->vdevice);
    } else {
      PrimExpr flattened_d = 1;
      for (const auto v : data_shape->values) {
        flattened_d *= v;
      }
      return TensorType(ShapeExpr(ffi::Array<PrimExpr>({flattened_d})), out_type, data_ty->vdevice);
    }
  }

  if (data_ty->shape.defined()) {
    return TensorType(data_ty->shape.value(), out_type, data_ty->vdevice);
  } else {
    return TensorType(out_type, data_ty->ndim, data_ty->vdevice);
  }
}

Type InferTypeStatisticalExtension(const Call& call, const BlockBuilder& ctx) {
  TensorType data_ty = GetUnaryInputTensorType(call, ctx);
  const auto* attrs = call->attrs.as<StatisticalAttrs>();

  std::vector<int> axes;
  if (!data_ty->IsUnknownNdim() && attrs->axis.defined()) {
    axes = NormalizeAxes(call, ctx, data_ty->ndim, attrs->axis.value());
  }

  int out_ndim;
  if (attrs->keepdims) {
    out_ndim = data_ty->ndim;
  } else if (!attrs->axis.defined()) {
    out_ndim = 0;
  } else if (data_ty->IsUnknownNdim()) {
    out_ndim = kUnknownNDim;
  } else {
    out_ndim = data_ty->ndim - axes.size();
    TVM_FFI_ICHECK_GE(out_ndim, 0);
  }

  // The inference rule for median operator output shapes:
  // - axes is None || len(axes) > 1, keepdims is false -> return the zero-rank shape;
  // - axes is None || len(axes) > 1, keepdims is true -> return the shape whose ndim
  // is the same as input and every value is 1.
  // - len(axes) == 1, keepdims is false -> the returned shape does not contain the input axis.
  // - len(axes) == 1, keepdims is true -> the returned shape has value 1 at the positions of the
  // input axis
  const auto* data_shape = data_ty->shape.as<ShapeExprNode>();
  if (data_shape == nullptr) {
    if (!attrs->axis.defined() && attrs->keepdims && out_ndim != kUnknownNDim) {
      return TensorType(ShapeExpr(ffi::Array<PrimExpr>(out_ndim, IntImm::Int64(/*value=*/1))),
                        data_ty->dtype, data_ty->vdevice);
    }
    if (out_ndim == 0) {
      return TensorType(ShapeExpr(ffi::Array<PrimExpr>()), data_ty->dtype, data_ty->vdevice);
    }
    return TupleType({TensorType(data_ty->dtype, out_ndim, data_ty->vdevice),
                      TensorType(PrimType::Int(64), out_ndim, data_ty->vdevice)});
  }

  ffi::Array<PrimExpr> out_shape;
  out_shape.reserve(out_ndim);
  for (int i = 0; i < data_ty->ndim; ++i) {
    if (attrs->axis.defined() && std::find(axes.begin(), axes.end(), i) == axes.end()) {
      out_shape.push_back(data_shape->values[i]);
    } else if (attrs->keepdims) {
      out_shape.push_back(IntImm::Int64(/*value=*/1));
    }
  }
  TVM_FFI_ICHECK_EQ(static_cast<int>(out_shape.size()), out_ndim);

  if (!attrs->axis.defined() || axes.size() > 1)
    return TensorType(ShapeExpr(out_shape), data_ty->dtype, data_ty->vdevice);
  else
    return TupleType({TensorType(ShapeExpr(out_shape), data_ty->dtype, data_ty->vdevice),
                      TensorType(ShapeExpr(out_shape), PrimType::Int(64), data_ty->vdevice)});
}

/* relax.cumprod */
Expr cumprod(Expr data, ffi::Optional<int64_t> axis, ffi::Optional<DLDataType> dtype,
             bool exclusive) {
  auto attrs = ffi::make_object<ScanopAttrs>();
  attrs->axis = std::move(axis);
  attrs->dtype = dtype.value_or((DLDataType{kDLOpaqueHandle, 0, 0}));
  attrs->exclusive = exclusive;

  static const Op& op = Op::Get("relax.cumprod");
  return Call(op, {std::move(data)}, Attrs{attrs}, {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.cumprod", cumprod);
}

TVM_REGISTER_OP("relax.cumprod")
    .set_attrs_type<ScanopAttrs>()
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor.")
    .set_attr<FInferType>("FInferType", InferTypeScan)
    .set_attr<bool>("FPurity", true);

/* relax.cumsum */
Expr cumsum(Expr data, ffi::Optional<int64_t> axis, ffi::Optional<DLDataType> dtype,
            bool exclusive) {
  auto attrs = ffi::make_object<ScanopAttrs>();
  attrs->axis = std::move(axis);
  attrs->dtype = dtype.value_or((DLDataType{kDLOpaqueHandle, 0, 0}));
  attrs->exclusive = exclusive;

  static const Op& op = Op::Get("relax.cumsum");
  return Call(op, {std::move(data)}, Attrs{attrs}, {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.cumsum", cumsum);
}

TVM_REGISTER_OP("relax.cumsum")
    .set_attrs_type<ScanopAttrs>()
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor.")
    .set_attr<FInferType>("FInferType", InferTypeScan)
    .set_attr<bool>("FPurity", true);

/* relax.median */
Expr median(Expr data, ffi::Optional<ffi::Array<int64_t>> axis, bool keepdims) {
  ffi::ObjectPtr<StatisticalAttrs> attrs = ffi::make_object<StatisticalAttrs>();
  attrs->axis = std::move(axis);
  attrs->keepdims = keepdims;
  static const Op& op = Op::Get("relax.median");
  return Call(op, {std::move(data)}, Attrs{attrs}, {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.median", median);
}

TVM_REGISTER_OP("relax.median")
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor.")
    .set_attr<FInferType>("FInferType", InferTypeStatisticalExtension)
    .set_attr<bool>("FPurity", true);

RELAX_REGISTER_STATISTICAL_OP_INTERFACE(max);
RELAX_REGISTER_STATISTICAL_OP_INTERFACE(mean);
RELAX_REGISTER_STATISTICAL_OP_INTERFACE(min);
RELAX_REGISTER_STATISTICAL_OP_INTERFACE(prod);
RELAX_REGISTER_STATISTICAL_OP_INTERFACE(std);
RELAX_REGISTER_STATISTICAL_OP_INTERFACE(sum);
RELAX_REGISTER_STATISTICAL_OP_INTERFACE(variance);

}  // namespace relax
}  // namespace tvm
