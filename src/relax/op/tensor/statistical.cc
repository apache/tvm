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

#include <string>
#include <vector>

namespace tvm {
namespace relax {

StructInfo InferStructInfoStatistical(const Call& call, const BlockBuilder& ctx) {
  TensorStructInfo data_sinfo = GetUnaryInputTensorStructInfo(call, ctx);
  const auto* attrs = call->attrs.as<StatisticalAttrs>();

  std::vector<int> axes;
  if (!data_sinfo->IsUnknownNdim() && attrs->axis.defined()) {
    axes = NormalizeAxes(call, ctx, data_sinfo->ndim, attrs->axis.value());
  }

  int out_ndim;
  if (attrs->keepdims) {
    out_ndim = data_sinfo->ndim;
  } else if (!attrs->axis.defined()) {
    out_ndim = 0;
  } else if (data_sinfo->IsUnknownNdim()) {
    out_ndim = kUnknownNDim;
  } else {
    out_ndim = data_sinfo->ndim - axes.size();
    ICHECK_GE(out_ndim, 0);
  }

  // The inference rule for reduction operator output shapes:
  // - axes is None, keepdims is false -> return the zero-rank shape;
  // - axes is None, keepdims is true -> return the shape whose ndim is the same as input and every
  // value is 1.
  // - axes is not None, keepdims is false -> the returned shape does not contain the input axes.
  // - axes is not None, keepdims is true -> the returned shape has value 1 at the positions of the
  // input axes
  const auto* data_shape = data_sinfo->shape.as<ShapeExprNode>();
  if (data_shape == nullptr) {
    if (!attrs->axis.defined() && attrs->keepdims && out_ndim != kUnknownNDim) {
      return TensorStructInfo(
          ShapeExpr(Array<PrimExpr>(out_ndim, IntImm(DataType::Int(64), /*value=*/1))),
          data_sinfo->dtype, data_sinfo->vdevice);
    } else {
      return out_ndim == 0 ? TensorStructInfo(ShapeExpr(Array<PrimExpr>()), data_sinfo->dtype,
                                              data_sinfo->vdevice)
                           : TensorStructInfo(data_sinfo->dtype, out_ndim, data_sinfo->vdevice);
    }
  }

  Array<PrimExpr> out_shape;
  out_shape.reserve(out_ndim);
  for (int i = 0; i < data_sinfo->ndim; ++i) {
    if (attrs->axis.defined() && std::find(axes.begin(), axes.end(), i) == axes.end()) {
      out_shape.push_back(data_shape->values[i]);
    } else if (attrs->keepdims) {
      out_shape.push_back(IntImm(DataType::Int(64), /*value=*/1));
    }
  }
  ICHECK_EQ(static_cast<int>(out_shape.size()), out_ndim);
  return TensorStructInfo(ShapeExpr(out_shape), data_sinfo->dtype, data_sinfo->vdevice);
}

InferLayoutOutput InferLayoutStatistical(const Call& call,
                                         const Map<String, Array<String>>& desired_layouts,
                                         const VarLayoutMap& var_layout_map) {
  ICHECK(NoDesiredLayout(call, desired_layouts));

  const auto* attrs = call->attrs.as<StatisticalAttrs>();
  ICHECK(attrs != nullptr) << "Invalid Call";
  const auto* tensor_sinfo = GetStructInfoAs<TensorStructInfoNode>(call->args[0]);
  ICHECK(tensor_sinfo != nullptr) << "Invalid Call";
  ICHECK(!tensor_sinfo->IsUnknownNdim()) << "Only support known ndim";
  int ndim = tensor_sinfo->ndim;

  Array<Integer> axis;
  if (attrs->axis.defined()) {
    axis = attrs->axis.value();
  } else {
    axis.reserve(ndim);
    for (int i = 0; i < ndim; ++i) {
      axis.push_back(Integer(i));
    }
  }

  std::string axis_str(ndim, '0');
  for (const auto& iter : axis) {
    axis_str[(iter->value + ndim) % ndim] = '#';
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

  Array<Integer> new_axis;
  for (size_t i = 0; i < new_axis_str.size(); ++i) {
    if (new_axis_str.at(i) == '#') {
      new_axis.push_back(Integer(i));
    }
  }
  std::string output_layout;
  for (size_t i = 0; i < output_layout_ref.length(); ++i) {
    if ((isdigit(output_layout_ref[i]) && (output_layout_ref[i + 1] == '#')) ||
        (output_layout_ref[i] == '#'))
      continue;
    output_layout.push_back(output_layout_ref[i]);
  }

  ObjectPtr<StatisticalAttrs> new_attrs = make_object<StatisticalAttrs>(*attrs);
  new_attrs->axis = new_axis;
  return InferLayoutOutput({exisiting_layout},
                           {attrs->keepdims ? exisiting_layout : Layout(output_layout)},
                           Attrs(new_attrs));
}

TVM_REGISTER_NODE_TYPE(ScanopAttrs);

StructInfo InferStructInfoScan(const Call& call, const BlockBuilder& ctx) {
  TensorStructInfo data_sinfo = GetUnaryInputTensorStructInfo(call, ctx);
  const auto* attrs = call->attrs.as<ScanopAttrs>();

  DataType out_type = attrs->dtype.is_void() ? data_sinfo->dtype : attrs->dtype;

  if (!attrs->axis.defined()) {
    // flattened
    const auto* data_shape = data_sinfo->shape.as<ShapeExprNode>();
    if (data_shape == nullptr) {
      return TensorStructInfo(out_type, data_sinfo->ndim, data_sinfo->vdevice);
    } else {
      PrimExpr flattened_d = 1;
      for (const auto v : data_shape->values) {
        flattened_d *= v;
      }
      return TensorStructInfo(ShapeExpr(Array<PrimExpr>({flattened_d})), out_type,
                              data_sinfo->vdevice);
    }
  }

  if (data_sinfo->shape.defined()) {
    return TensorStructInfo(data_sinfo->shape.value(), out_type, data_sinfo->vdevice);
  } else {
    return TensorStructInfo(out_type, data_sinfo->ndim, data_sinfo->vdevice);
  }
}

/* relax.cumprod */
Expr cumprod(Expr data, Optional<Integer> axis, DataType dtype, Bool exclusive) {
  auto attrs = make_object<ScanopAttrs>();
  attrs->axis = std::move(axis);
  attrs->dtype = std::move(dtype);
  attrs->exclusive = std::move(exclusive);

  static const Op& op = Op::Get("relax.cumprod");
  return Call(op, {std::move(data)}, Attrs{attrs}, {});
}

TVM_REGISTER_GLOBAL("relax.op.cumprod").set_body_typed(cumprod);

TVM_REGISTER_OP("relax.cumprod")
    .set_attrs_type<ScanopAttrs>()
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor.")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoScan)
    .set_attr<Bool>("FPurity", Bool(true));

/* relax.cumsum */
Expr cumsum(Expr data, Optional<Integer> axis, DataType dtype, Bool exclusive) {
  auto attrs = make_object<ScanopAttrs>();
  attrs->axis = std::move(axis);
  attrs->dtype = std::move(dtype);
  attrs->exclusive = std::move(exclusive);

  static const Op& op = Op::Get("relax.cumsum");
  return Call(op, {std::move(data)}, Attrs{attrs}, {});
}

TVM_REGISTER_GLOBAL("relax.op.cumsum").set_body_typed(cumsum);

TVM_REGISTER_OP("relax.cumsum")
    .set_attrs_type<ScanopAttrs>()
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor.")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoScan)
    .set_attr<Bool>("FPurity", Bool(true));

TVM_REGISTER_NODE_TYPE(StatisticalAttrs);

RELAX_REGISTER_STATISTICAL_OP_INTERFACE(max);
RELAX_REGISTER_STATISTICAL_OP_INTERFACE(mean);
RELAX_REGISTER_STATISTICAL_OP_INTERFACE(min);
RELAX_REGISTER_STATISTICAL_OP_INTERFACE(prod);
RELAX_REGISTER_STATISTICAL_OP_INTERFACE(std);
RELAX_REGISTER_STATISTICAL_OP_INTERFACE(sum);
RELAX_REGISTER_STATISTICAL_OP_INTERFACE(variance);

}  // namespace relax
}  // namespace tvm
