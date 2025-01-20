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

#include "nn.h"

#include <utility>
#include <vector>

namespace tvm {
namespace relax {

/* relax.nn.relu */
RELAX_REGISTER_UNARY_NN_OP_AND_IMPL(relu, "nn.relu", /*require_float_dtype=*/false);

/* relax.nn.gelu */
RELAX_REGISTER_UNARY_NN_OP_AND_IMPL(gelu, "nn.gelu", /*require_float_dtype=*/true);

/* relax.nn.gelu_tanh */
RELAX_REGISTER_UNARY_NN_OP_AND_IMPL(gelu_tanh, "nn.gelu_tanh", /*require_float_dtype=*/true);

/* relax.nn.silu */
RELAX_REGISTER_UNARY_NN_OP_AND_IMPL(silu, "nn.silu", /*require_float_dtype=*/true);

/* relax.nn.leakyrelu */
TVM_REGISTER_NODE_TYPE(LeakyReluAttrs);

Expr leakyrelu(Expr data, double alpha) {
  auto attrs = make_object<LeakyReluAttrs>();
  attrs->alpha = alpha;
  static const Op& op = Op::Get("relax.nn.leakyrelu");
  return Call(op, {data}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relax.op.nn.leakyrelu").set_body_typed(leakyrelu);

TVM_REGISTER_OP("relax.nn.leakyrelu")
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor.")
    .set_attrs_type<LeakyReluAttrs>()
    .set_attr<FInferStructInfo>("FInferStructInfo",
                                InferStructInfoUnaryArith</*require_float_dtype=*/true>)
    .set_attr<Bool>("FPurity", Bool(true));

/* relax.nn.softmax */
TVM_REGISTER_NODE_TYPE(SoftmaxAttrs);

Expr softmax(Expr data, int axis) {
  auto attrs = make_object<SoftmaxAttrs>();
  attrs->axis = axis;
  static const Op& op = Op::Get("relax.nn.softmax");
  return Call(op, {data}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relax.op.nn.softmax").set_body_typed(softmax);

StructInfo InferStructInfoSoftmax(const Call& call, const BlockBuilder& ctx) {
  TensorStructInfo data_sinfo = GetUnaryInputTensorStructInfo(call, ctx);
  if (data_sinfo->IsUnknownNdim()) {
    return data_sinfo;
  }
  if (!data_sinfo->IsUnknownDtype() && !data_sinfo->dtype.is_float()) {
    ctx->ReportFatal(Diagnostic::Error(call) << "Softmax requires the input tensor to have float "
                                                "dtype. However, the given input dtype is "
                                             << data_sinfo->dtype);
  }
  const auto* attrs = call->attrs.as<SoftmaxAttrs>();
  NormalizeAxis(call, ctx, data_sinfo->ndim, attrs->axis);

  return data_sinfo;
}

InferLayoutOutput InferLayoutSoftmax(const Call& call,
                                     const Map<String, Array<String>>& desired_layouts,
                                     const VarLayoutMap& var_layout_map) {
  ICHECK(NoDesiredLayout(call, desired_layouts));
  const auto* attrs = call->attrs.as<SoftmaxAttrs>();
  ICHECK(attrs) << "Invalid Call";

  LayoutDecision layout = GetLayoutDecision(var_layout_map, call->args[0]);

  // TODO(Siva): We could handle if the axis is not the sub indexed one.
  if (layout->layout.ndim() != layout->layout.ndim_primal()) {
    const auto* tensor_sinfo = GetStructInfoAs<TensorStructInfoNode>(call->args[0]);
    ICHECK(tensor_sinfo != nullptr) << "Invalid Call";
    ICHECK(!tensor_sinfo->IsUnknownNdim()) << "Only support static ndim for now";
    int ndim = tensor_sinfo->ndim;
    layout = LayoutDecision(InitialLayout(ndim));
  }

  ObjectPtr<SoftmaxAttrs> new_attrs = make_object<SoftmaxAttrs>(*attrs);
  new_attrs->axis = FindAxis(layout->layout, attrs->axis);
  return InferLayoutOutput({layout}, {layout}, Attrs(new_attrs));
}

TVM_REGISTER_OP("relax.nn.softmax")
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor.")
    .set_attrs_type<SoftmaxAttrs>()
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoSoftmax)
    .set_attr<FRelaxInferLayout>("FRelaxInferLayout", InferLayoutSoftmax)
    .set_attr<Bool>("FPurity", Bool(true));

/* relax.nn.log_softmax */
Expr log_softmax(Expr data, int axis) {
  auto attrs = make_object<SoftmaxAttrs>();
  attrs->axis = axis;
  static const Op& op = Op::Get("relax.nn.log_softmax");
  return Call(op, {data}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relax.op.nn.log_softmax").set_body_typed(log_softmax);

TVM_REGISTER_OP("relax.nn.log_softmax")
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor.")
    .set_attrs_type<SoftmaxAttrs>()
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoSoftmax)
    .set_attr<Bool>("FPurity", Bool(true));

/* relax.nn.pad */
TVM_REGISTER_NODE_TYPE(PadAttrs);

Expr pad(Expr data, Array<Integer> pad_width, Expr pad_value, String pad_mode) {
  auto attrs = make_object<PadAttrs>();
  attrs->pad_width = std::move(pad_width);
  attrs->pad_mode = std::move(pad_mode);
  static const Op& op = Op::Get("relax.nn.pad");
  return Call(op, {data, pad_value}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relax.op.nn.pad").set_body_typed(pad);

StructInfo InferStructInfoPad(const Call& call, const BlockBuilder& ctx) {
  Array<TensorStructInfo> input_sinfo = GetInputTensorStructInfo(call, ctx);
  const auto* attrs = call->attrs.as<PadAttrs>();
  int ndim = input_sinfo[0]->ndim;
  Array<Integer> pad_width = attrs->pad_width;
  ICHECK(static_cast<int>(pad_width.size()) == 2 * ndim) << "Illegal pad_width";

  Array<PrimExpr> out_shape;
  if (input_sinfo[0]->shape.defined()) {
    // Compute output shape by adding corresponding pad width to each axis.
    const auto* data_shape = input_sinfo[0]->shape.as<ShapeExprNode>();
    for (int i = 0; i < ndim; i++) {
      // Sum pad width for this axis.
      PrimExpr added_width = pad_width[2 * i] + pad_width[(2 * i) + 1];
      const PrimExpr current_width = data_shape->values[i];
      out_shape.push_back(current_width + added_width);
    }
  } else {
    // Shape isnt defined, best we can do is return ndim and dtype.
    return TensorStructInfo(input_sinfo[0]->dtype, ndim);
  }
  return TensorStructInfo(ShapeExpr(out_shape), input_sinfo[0]->dtype);
}

TVM_REGISTER_OP("relax.nn.pad")
    .set_num_inputs(2)
    .add_argument("data", "Tensor", "The input tensor.")
    .add_argument("pad_value", "Tensor", "The value to fill in padded area with.")
    .set_attrs_type<PadAttrs>()
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoPad)
    .set_attr<Bool>("FPurity", Bool(true));

/* relax.nn.batchnorm */
bool NormCheckDtypeAndShape(const Call& call, const BlockBuilder& ctx,
                            const Array<TensorStructInfo>& input_sinfo, Array<Integer> axes) {
  Op op = Downcast<Op>(call->op);
  int n_input = op->arguments.size();

  TensorStructInfo data_sinfo = input_sinfo[0];

  std::vector<int> axes_non_neg;
  if (!data_sinfo->IsUnknownNdim()) {
    axes_non_neg = NormalizeAxes(call, ctx, data_sinfo->ndim, axes);
  }
  int n_axis = axes.size();
  if (!data_sinfo->IsUnknownDtype() && !data_sinfo->dtype.is_float()) {
    ctx->ReportFatal(
        Diagnostic::Error(call)
        << op << " requires the input data to have float dtype. However, the given data dtype is "
        << data_sinfo->dtype);
  }
  for (int i = 1; i < n_input; ++i) {
    if (input_sinfo[i]->dtype != data_sinfo->dtype) {
      ctx->ReportFatal(Diagnostic::Error(call)
                       << op
                       << " requires all the input tensors to have the same dtype. However, the "
                       << op->arguments[i]->name << " has dtype " << input_sinfo[i]->dtype
                       << " which is other than the input data's dtype " << data_sinfo->dtype);
    } else if (input_sinfo[i]->ndim != n_axis) {
      ctx->ReportFatal(Diagnostic::Error(call)
                       << op << " requires the input " << op->arguments[i]->name
                       << " to have as many dimensions as the length of input axes. However, the "
                          "given one has ndim "
                       << input_sinfo[i]->ndim << ", which is other than the length of axes "
                       << n_axis);
    }
  }

  std::vector<Array<PrimExpr>> axis_lengths;
  axis_lengths.reserve(n_input);
  if (const auto* data_shape = data_sinfo->shape.as<ShapeExprNode>()) {
    std::vector<PrimExpr> lengths;
    lengths.reserve(n_axis);
    for (int d = 0; d < n_axis; ++d) {
      lengths.push_back(data_shape->values[axes_non_neg[d]]);
    }
    axis_lengths.push_back(lengths);
  }
  for (int i = 1; i < n_input; ++i) {
    if (const auto* shape = input_sinfo[i]->shape.as<ShapeExprNode>()) {
      axis_lengths.push_back(shape->values);
    }
  }

  arith::Analyzer* analyzer = ctx->GetAnalyzer();
  for (int i = 1; i < static_cast<int>(axis_lengths.size()); ++i) {
    for (int d = 0; d < n_axis; ++d) {
      if (analyzer->CanProve(axis_lengths[0][d] != axis_lengths[i][d])) {
        ctx->ReportFatal(Diagnostic::Error(call)
                         << op
                         << " requires the input gamma, beta, etc., to have size same as the "
                            "lengths of the data on the given axes. However, there exists "
                         << axis_lengths[0] << " and " << axis_lengths[i] << " that are unequal.");
      } else if (!analyzer->CanProveEqual(axis_lengths[0][d], axis_lengths[i][d])) {
        return true;
      }
    }
  }
  return false;
}

/* relax.nn.batch_norm */
TVM_REGISTER_NODE_TYPE(BatchNormAttrs);

Expr batch_norm(Expr data, Expr gamma, Expr beta, Expr moving_mean, Expr moving_var,  //
                int axis, double epsilon, bool center, bool scale, double momentum) {
  ObjectPtr<BatchNormAttrs> attrs = make_object<BatchNormAttrs>();
  attrs->axis = axis;
  attrs->epsilon = epsilon;
  attrs->center = center;
  attrs->scale = scale;
  attrs->momentum = momentum;

  static const Op& op = Op::Get("relax.nn.batch_norm");
  return Call(op,
              {std::move(data), std::move(gamma), std::move(beta), std::move(moving_mean),
               std::move(moving_var)},
              Attrs{attrs}, {});
}

TVM_REGISTER_GLOBAL("relax.op.nn.batch_norm").set_body_typed(batch_norm);

StructInfo InferStructInfoBatchNorm(const Call& call, const BlockBuilder& ctx) {
  Array<TensorStructInfo> input_sinfo = GetInputTensorStructInfo(call, ctx);

  const auto* attrs = call->attrs.as<BatchNormAttrs>();
  bool unknown_shape = NormCheckDtypeAndShape(call, ctx, input_sinfo, {attrs->axis});

  DataType dtype = input_sinfo[0]->dtype;
  if (unknown_shape) {
    auto vdev = input_sinfo[0]->vdevice;
    return TupleStructInfo({TensorStructInfo(dtype, input_sinfo[0]->ndim, vdev),
                            TensorStructInfo(dtype, /*ndim=*/1, vdev),
                            TensorStructInfo(dtype, /*ndim=*/1, vdev)});
  } else {
    return TupleStructInfo({input_sinfo[0], input_sinfo[3], input_sinfo[4]});
  }
}

InferLayoutOutput InferLayoutBatchNorm(const Call& call,
                                       const Map<String, Array<String>>& desired_layouts,
                                       const VarLayoutMap& var_layout_map) {
  ICHECK(NoDesiredLayout(call, desired_layouts));
  std::vector<NLayout> initial_layouts;
  for (size_t i = 0; i < 5; ++i) {
    const auto* tensor_sinfo = GetStructInfoAs<TensorStructInfoNode>(call->args[i]);
    ICHECK(tensor_sinfo != nullptr) << "Invalid Call";
    ICHECK(!tensor_sinfo->IsUnknownNdim()) << "Only support known ndim";
    initial_layouts.push_back(InitialLayoutDecision(tensor_sinfo->ndim));
  }
  const auto* attrs = call->attrs.as<BatchNormAttrs>();
  ICHECK(attrs) << "Invalid Call";

  LayoutDecision layout = GetLayoutDecision(var_layout_map, call->args[0]);

  // While dealing with sub layouts, its adviced to deal with batchnorm
  // on other ways like decomposing or fusion methods.
  // This handling is fail safe fallback.
  const auto* input_sinfo = GetStructInfoAs<TensorStructInfoNode>(call->args[0]);
  int ndim = input_sinfo->ndim;
  if (layout->layout.ndim() != layout->layout.ndim_primal()) {
    layout = LayoutDecision(InitialLayout(ndim));
  }

  ObjectPtr<BatchNormAttrs> new_attrs = make_object<BatchNormAttrs>(*attrs);
  new_attrs->axis = FindAxis(layout->layout, (attrs->axis + ndim) % ndim);
  return InferLayoutOutput(
      {layout, initial_layouts[1], initial_layouts[2], initial_layouts[3], initial_layouts[4]},
      {{layout, initial_layouts[3], initial_layouts[4]}}, Attrs(new_attrs));
}

TVM_REGISTER_OP("relax.nn.batch_norm")
    .set_attrs_type<BatchNormAttrs>()
    .set_num_inputs(5)
    .add_argument("data", "Tensor", "Input to which batch_norm will be applied.")
    .add_argument("gamma", "Tensor", "The gamma scale factor.")
    .add_argument("beta", "Tensor", "The beta offset factor.")
    .add_argument("moving_mean", "Tensor", "Running mean of input.")
    .add_argument("moving_var", "Tensor", "Running variance of input.")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoBatchNorm)
    .set_attr<FRelaxInferLayout>("FRelaxInferLayout", InferLayoutBatchNorm)
    .set_attr<Bool>("FPurity", Bool(true));

/* relax.nn.layer_norm */
TVM_REGISTER_NODE_TYPE(LayerNormAttrs);

Expr layer_norm(Expr data, Expr gamma, Expr beta, Array<Integer> axes, double epsilon, bool center,
                bool scale) {
  ObjectPtr<LayerNormAttrs> attrs = make_object<LayerNormAttrs>();
  attrs->axes = std::move(axes);
  attrs->epsilon = epsilon;
  attrs->center = center;
  attrs->scale = scale;

  static const Op& op = Op::Get("relax.nn.layer_norm");
  return Call(op, {std::move(data), std::move(gamma), std::move(beta)}, Attrs{attrs}, {});
}

TVM_REGISTER_GLOBAL("relax.op.nn.layer_norm").set_body_typed(layer_norm);

StructInfo InferStructInfoLayerNorm(const Call& call, const BlockBuilder& ctx) {
  Array<TensorStructInfo> input_sinfo = GetInputTensorStructInfo(call, ctx);

  const auto* attrs = call->attrs.as<LayerNormAttrs>();
  bool unknown_shape = NormCheckDtypeAndShape(call, ctx, input_sinfo, attrs->axes);

  return unknown_shape ? TensorStructInfo(input_sinfo[0]->dtype, input_sinfo[0]->ndim,
                                          input_sinfo[0]->vdevice)
                       : input_sinfo[0];
}

InferLayoutOutput InferLayoutLayerNorm(const Call& call,
                                       const Map<String, Array<String>>& desired_layouts,
                                       const VarLayoutMap& var_layout_map) {
  ICHECK(NoDesiredLayout(call, desired_layouts));
  std::vector<NLayout> initial_layouts;
  for (size_t i = 0; i < 3; ++i) {
    const auto* tensor_sinfo = GetStructInfoAs<TensorStructInfoNode>(call->args[i]);
    ICHECK(tensor_sinfo != nullptr) << "Invalid Call";
    ICHECK(!tensor_sinfo->IsUnknownNdim()) << "Only support known ndim";
    initial_layouts.push_back(InitialLayoutDecision(tensor_sinfo->ndim));
  }
  const auto* attrs = call->attrs.as<LayerNormAttrs>();
  ICHECK(attrs) << "Invalid Call";

  LayoutDecision layout = GetLayoutDecision(var_layout_map, call->args[0]);
  ObjectPtr<LayerNormAttrs> new_attrs = make_object<LayerNormAttrs>(*attrs);
  const auto* input_sinfo = GetStructInfoAs<TensorStructInfoNode>(call->args[0]);
  int ndim = input_sinfo->ndim;
  std::vector<Integer> new_axis;
  for (const auto& axis : attrs->axes) {
    new_axis.push_back(FindAxis(layout->layout, (axis->value + ndim) % ndim));
  }
  new_attrs->axes = std::move(new_axis);
  return InferLayoutOutput({layout, initial_layouts[1], initial_layouts[2]}, {layout},
                           Attrs(new_attrs));
}

TVM_REGISTER_OP("relax.nn.layer_norm")
    .set_attrs_type<LayerNormAttrs>()
    .set_num_inputs(3)
    .add_argument("data", "Tensor", "Input to which batch_norm will be applied.")
    .add_argument("gamma", "Tensor", "The gamma scale factor.")
    .add_argument("beta", "Tensor", "The beta offset factor.")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoLayerNorm)
    .set_attr<FRelaxInferLayout>("FRelaxInferLayout", InferLayoutLayerNorm)
    .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kFollow)
    .set_attr<Bool>("FPurity", Bool(true));

/* relax.nn.group_norm */
TVM_REGISTER_NODE_TYPE(GroupNormAttrs);

Expr group_norm(Expr data, Expr gamma, Expr beta, int num_groups, int channel_axis,
                Array<Integer> axes, double epsilon, bool center, bool scale) {
  ObjectPtr<GroupNormAttrs> attrs = make_object<GroupNormAttrs>();
  attrs->num_groups = num_groups;
  attrs->channel_axis = channel_axis;
  attrs->axes = std::move(axes);
  attrs->epsilon = epsilon;
  attrs->center = center;
  attrs->scale = scale;

  static const Op& op = Op::Get("relax.nn.group_norm");
  return Call(op, {std::move(data), std::move(gamma), std::move(beta)}, Attrs{attrs}, {});
}

TVM_REGISTER_GLOBAL("relax.op.nn.group_norm").set_body_typed(group_norm);

StructInfo InferStructInfoGroupNorm(const Call& call, const BlockBuilder& ctx) {
  Op op = Downcast<Op>(call->op);
  Array<TensorStructInfo> input_sinfo = GetInputTensorStructInfo(call, ctx);
  const auto* attrs = call->attrs.as<GroupNormAttrs>();

  TensorStructInfo data_sinfo = input_sinfo[0];
  int channel_axis = -1;
  if (!data_sinfo->IsUnknownNdim()) {
    channel_axis = NormalizeAxis(call, ctx, data_sinfo->ndim, attrs->channel_axis);
    std::vector<int> axes = NormalizeAxes(call, ctx, data_sinfo->ndim, attrs->axes);
    // channel_axis must be in axes.
    if (std::find(axes.begin(), axes.end(), channel_axis) != axes.end()) {
      ctx->ReportFatal(Diagnostic::Error(call)
                       << op
                       << " expects that channel_axis must not be in axes, but got channel_axis: "
                       << channel_axis << ", axes: " << attrs->axes);
    }
  }
  if (!data_sinfo->IsUnknownDtype() && !data_sinfo->dtype.is_float()) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << op << " expects that data must be float, but got " << data_sinfo->dtype);
  }
  arith::Analyzer* analyzer = ctx->GetAnalyzer();
  const auto* data_shape = data_sinfo->shape.as<ShapeExprNode>();
  if (data_shape != nullptr && channel_axis != -1 &&
      analyzer->CanProve(floormod(data_shape->values[channel_axis], attrs->num_groups) != 0)) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << op << " expects that the size of channel_axis must be divisible by "
                     << attrs->num_groups << ", but got " << data_shape->values[channel_axis]);
  }
  for (int i = 1; i < static_cast<int>(op->arguments.size()); ++i) {
    if (input_sinfo[i]->dtype != data_sinfo->dtype) {
      ctx->ReportFatal(Diagnostic::Error(call)
                       << op << " expects that all inputs must have the same dtype, but got "
                       << input_sinfo[i]->dtype << " and " << data_sinfo->dtype);
    } else if (input_sinfo[i]->ndim != 1) {
      ctx->ReportFatal(Diagnostic::Error(call)
                       << op << " expects that all inputs must have ndim=1, but got "
                       << input_sinfo[i]->ndim);
    } else if (channel_axis != -1) {
      const auto* shape = input_sinfo[i]->shape.as<ShapeExprNode>();
      if (shape != nullptr && data_shape != nullptr) {
        PrimExpr channel_size = data_shape->values[channel_axis];
        PrimExpr input_size = shape->values[0];
        if (analyzer->CanProve(channel_size != input_size)) {
          ctx->ReportFatal(Diagnostic::Error(call)
                           << op << " expects that the size of input " << i
                           << " must be equal to the size of channel_axis, but got " << input_size
                           << " and " << channel_size);
        }
      }
    }
  }
  return data_sinfo;
}

InferLayoutOutput InferLayoutGroupNorm(const Call& call,
                                       const Map<String, Array<String>>& desired_layouts,
                                       const VarLayoutMap& var_layout_map) {
  ICHECK(NoDesiredLayout(call, desired_layouts));
  std::vector<NLayout> initial_layouts;
  for (size_t i = 0; i < 3; ++i) {
    const auto* tensor_sinfo = GetStructInfoAs<TensorStructInfoNode>(call->args[i]);
    ICHECK(tensor_sinfo != nullptr) << "Invalid Call";
    ICHECK(!tensor_sinfo->IsUnknownNdim()) << "Only support known ndim";
    initial_layouts.push_back(InitialLayoutDecision(tensor_sinfo->ndim));
  }
  const auto* attrs = call->attrs.as<GroupNormAttrs>();
  ICHECK(attrs) << "Invalid Call";

  LayoutDecision layout = GetLayoutDecision(var_layout_map, call->args[0]);
  ObjectPtr<GroupNormAttrs> new_attrs = make_object<GroupNormAttrs>(*attrs);
  std::vector<Integer> new_axes;
  for (const auto& axis : attrs->axes) {
    new_axes.push_back(FindAxis(layout->layout, axis->value));
  }
  new_attrs->axes = std::move(new_axes);
  new_attrs->channel_axis = FindAxis(layout->layout, attrs->channel_axis);
  return InferLayoutOutput({layout, initial_layouts[1], initial_layouts[2]}, {layout},
                           Attrs(new_attrs));
}

TVM_REGISTER_OP("relax.nn.group_norm")
    .set_attrs_type<GroupNormAttrs>()
    .set_num_inputs(3)
    .add_argument("data", "Tensor", "Input to which batch_norm will be applied.")
    .add_argument("gamma", "Tensor", "The gamma scale factor.")
    .add_argument("beta", "Tensor", "The beta offset factor.")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoGroupNorm)
    .set_attr<FRelaxInferLayout>("FRelaxInferLayout", InferLayoutGroupNorm)
    .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kFollow)
    .set_attr<Bool>("FPurity", Bool(true));

/* relax.nn.rms_norm */
TVM_REGISTER_NODE_TYPE(RMSNormAttrs);

Expr rms_norm(Expr data, Expr weight, Array<Integer> axes, double epsilon) {
  ObjectPtr<RMSNormAttrs> attrs = make_object<RMSNormAttrs>();
  attrs->axes = std::move(axes);
  attrs->epsilon = epsilon;

  static const Op& op = Op::Get("relax.nn.rms_norm");
  return Call(op, {std::move(data), std::move(weight)}, Attrs{attrs}, {});
}

TVM_REGISTER_GLOBAL("relax.op.nn.rms_norm").set_body_typed(rms_norm);

StructInfo InferStructInfoRMSNorm(const Call& call, const BlockBuilder& ctx) {
  Array<TensorStructInfo> input_sinfo = GetInputTensorStructInfo(call, ctx);

  const auto* attrs = call->attrs.as<RMSNormAttrs>();
  bool unknown_shape = NormCheckDtypeAndShape(call, ctx, input_sinfo, attrs->axes);

  return unknown_shape ? TensorStructInfo(input_sinfo[0]->dtype, input_sinfo[0]->ndim,
                                          input_sinfo[0]->vdevice)
                       : input_sinfo[0];
}

InferLayoutOutput InferLayoutRMSNorm(const Call& call,
                                     const Map<String, Array<String>>& desired_layouts,
                                     const VarLayoutMap& var_layout_map) {
  ICHECK(NoDesiredLayout(call, desired_layouts));
  std::vector<NLayout> initial_layouts;
  for (size_t i = 0; i < 2; ++i) {
    const auto* tensor_sinfo = GetStructInfoAs<TensorStructInfoNode>(call->args[i]);
    ICHECK(tensor_sinfo != nullptr) << "Invalid Call";
    ICHECK(!tensor_sinfo->IsUnknownNdim()) << "Only support known ndim";
    initial_layouts.push_back(InitialLayoutDecision(tensor_sinfo->ndim));
  }
  const auto* attrs = call->attrs.as<RMSNormAttrs>();
  ICHECK(attrs) << "Invalid Call";

  LayoutDecision layout = GetLayoutDecision(var_layout_map, call->args[0]);
  ObjectPtr<RMSNormAttrs> new_attrs = make_object<RMSNormAttrs>(*attrs);
  std::vector<Integer> new_axis;
  for (const auto& axis : attrs->axes) {
    new_axis.push_back(FindAxis(layout->layout, axis->value));
  }
  new_attrs->axes = std::move(new_axis);
  return InferLayoutOutput({layout, initial_layouts[1], initial_layouts[2]}, {layout},
                           Attrs(new_attrs));
}

TVM_REGISTER_OP("relax.nn.rms_norm")
    .set_attrs_type<RMSNormAttrs>()
    .set_num_inputs(2)
    .add_argument("data", "Tensor", "Input to which rms_norm will be applied.")
    .add_argument("weight", "Tensor", "The scale factor.")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoRMSNorm)
    .set_attr<FRelaxInferLayout>("FRelaxInferLayout", InferLayoutRMSNorm)
    .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kFollow)
    .set_attr<Bool>("FPurity", Bool(true));

/* relax.nn.dropout */
TVM_REGISTER_NODE_TYPE(DropoutAttrs);

Expr dropout(Expr data, double rate) {
  ObjectPtr<DropoutAttrs> attrs = make_object<DropoutAttrs>();
  attrs->rate = rate;

  static const Op& op = Op::Get("relax.nn.dropout");
  return Call(op, {std::move(data)}, Attrs{attrs}, {});
}

TVM_REGISTER_GLOBAL("relax.op.nn.dropout").set_body_typed(dropout);

StructInfo InferStructInfoDropout(const Call& call, const BlockBuilder& ctx) {
  TensorStructInfo data_sinfo = GetUnaryInputTensorStructInfo(call, ctx);
  return TupleStructInfo({data_sinfo, data_sinfo});
}

TVM_REGISTER_OP("relax.nn.dropout")
    .set_attrs_type<DropoutAttrs>()
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "Input to which dropout will be applied.")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoDropout)
    .set_attr<FRelaxInferLayout>("FRelaxInferLayout", InferLayoutUnaryEwise)
    .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kFollow)
    .set_attr<Bool>("FPurity", Bool(true));

/* relax.nn.cross_entropy_with_logits */
StructInfo InferStructInfoCrossEntropy(const Call& call, const BlockBuilder& ctx) {
  Array<TensorStructInfo> input_sinfo = GetInputTensorStructInfo(call, ctx);
  TensorStructInfo pred_sinfo = input_sinfo[0];
  TensorStructInfo label_sinfo = input_sinfo[1];

  // infer dtype
  DataType dtype = InferBinaryArithOpOutDtype(call, ctx, pred_sinfo, label_sinfo);

  // infer vdevice
  Optional<VDevice> vdevice = InferBinaryArithOpOutVDevice(call, ctx, pred_sinfo, label_sinfo);

  // infer ndim
  if (!pred_sinfo->IsUnknownNdim() && !label_sinfo->IsUnknownNdim() &&
      pred_sinfo->ndim != label_sinfo->ndim) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "CrossEntropy requires predictions and labels to have the same ndim. "
                        "However, the ndim of predictions is "
                     << pred_sinfo->ndim << " while the ndim of labels is " << label_sinfo->ndim);
  }

  Optional<Array<PrimExpr>> pred_shape_value;
  if (pred_sinfo->shape.defined()) {
    pred_shape_value = GetStructInfoAs<ShapeStructInfoNode>(pred_sinfo->shape.value())->values;
  }

  Optional<Array<PrimExpr>> label_shape_value;
  if (label_sinfo->shape.defined()) {
    label_shape_value = GetStructInfoAs<ShapeStructInfoNode>(label_sinfo->shape.value())->values;
  }

  if (pred_shape_value.defined() && label_shape_value.defined()) {
    arith::Analyzer* analyzer = ctx->GetAnalyzer();
    for (size_t i = 0; i < pred_shape_value.value().size(); ++i) {
      if (analyzer->CanProve(pred_shape_value.value()[i] != label_shape_value.value()[i])) {
        ctx->ReportFatal(Diagnostic::Error(call)
                         << "CrossEntropy requires the predictions and labels to have "
                            "the same shape. However, the shape of predictions at dim "
                         << i << " is" << pred_shape_value.value()[i]
                         << " while the shape of labels at this dim is "
                         << label_shape_value.value()[i]);
      }
    }
  }
  return TensorStructInfo(ShapeExpr(Array<PrimExpr>()), dtype, vdevice);
}

Expr cross_entropy_with_logits(Expr predictions, Expr labels) {
  static const Op& op = Op::Get("relax.nn.cross_entropy_with_logits");
  return Call(op, {std::move(predictions), std::move(labels)}, {}, {});
}

TVM_REGISTER_GLOBAL("relax.op.nn.cross_entropy_with_logits")
    .set_body_typed(cross_entropy_with_logits);

TVM_REGISTER_OP("relax.nn.cross_entropy_with_logits")
    .set_num_inputs(2)
    .add_argument("predictions", "Tensor", "The predictions.")
    .add_argument("labels", "Tensor", "The labels.")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoCrossEntropy)
    .set_attr<Bool>("FPurity", Bool(true));

/* relax.nn.nll_loss */
TVM_REGISTER_NODE_TYPE(NLLLossAttrs);

Expr nll_loss(Expr predictions, Expr targets, Optional<Expr> weights, String reduction,
              int ignore_index) {
  ObjectPtr<NLLLossAttrs> attrs = make_object<NLLLossAttrs>();

  ICHECK(reduction == "none" || reduction == "sum" || reduction == "mean")
      << "The argument reduction of NLLLoss should be one of the following "
         "values: none, mean, sum. However, the given value is "
      << reduction;

  attrs->reduction = std::move(reduction);
  attrs->ignore_index = ignore_index;

  static const Op& op = Op::Get("relax.nn.nll_loss");
  if (weights.defined()) {
    return Call(op, {std::move(predictions), std::move(targets), std::move(weights.value())},
                Attrs{attrs}, {});
  } else {
    return Call(op, {std::move(predictions), std::move(targets)}, Attrs{attrs}, {});
  }
}

TVM_REGISTER_GLOBAL("relax.op.nn.nll_loss").set_body_typed(nll_loss);

StructInfo InferStructInfoNLLLoss(const Call& call, const BlockBuilder& ctx) {
  if (call->args.size() < 2 || call->args.size() > 3) {
    ctx->ReportFatal(Diagnostic::Error(call) << "NLLLoss op should take 2 or 3 arguments");
  }

  const auto* pred_sinfo = GetStructInfoAs<TensorStructInfoNode>(call->args[0]);
  const auto* tgt_sinfo = GetStructInfoAs<TensorStructInfoNode>(call->args[1]);
  const TensorStructInfoNode* wgt_sinfo = nullptr;
  if (call->args.size() == 3) {
    wgt_sinfo = GetStructInfoAs<TensorStructInfoNode>(call->args[2]);
    if (wgt_sinfo == nullptr) {
      ctx->ReportFatal(
          Diagnostic::Error(call)
          << "NLLLoss requires the argument weights to be Tensor. However, the given one is "
          << call->args[2]->struct_info_->GetTypeKey());
    }
  }

  if (pred_sinfo == nullptr) {
    ctx->ReportFatal(
        Diagnostic::Error(call)
        << "NLLLoss requires the argument preditions to be Tensor. However, the given one is "
        << call->args[0]->struct_info_->GetTypeKey());
  }
  if (tgt_sinfo == nullptr) {
    ctx->ReportFatal(
        Diagnostic::Error(call)
        << "NLLLoss requires the argument targets to be Tensor. However, the given one is "
        << call->args[1]->struct_info_->GetTypeKey());
  }

  // infer dtype, vdevice
  DataType output_dtype;
  Optional<VDevice> vdevice;
  if (wgt_sinfo != nullptr) {
    output_dtype = InferBinaryArithOpOutDtype(call, ctx, GetRef<TensorStructInfo>(pred_sinfo),
                                              GetRef<TensorStructInfo>(wgt_sinfo));
    vdevice = InferBinaryArithOpOutVDevice(call, ctx, GetRef<TensorStructInfo>(pred_sinfo),
                                           GetRef<TensorStructInfo>(wgt_sinfo));
  } else {
    output_dtype = pred_sinfo->dtype;
    vdevice = pred_sinfo->vdevice;
  }

  // the type of targets must be int/uint.
  if (!tgt_sinfo->IsUnknownDtype() && !tgt_sinfo->dtype.is_int() && !tgt_sinfo->dtype.is_uint()) {
    ctx->ReportFatal(
        Diagnostic::Error(call)
        << "NLLLoss expects the dtype of targets to be int/uint. However, the dtype of targets is "
        << tgt_sinfo->dtype);
  }

  // infer ndim
  int K = kUnknownNDim;  // k dim
  if (!pred_sinfo->IsUnknownNdim()) {
    if (pred_sinfo->ndim < 1) {
      ctx->ReportFatal(
          Diagnostic::Error(call)
          << "NLLLoss expects the ndim of predictions >= 1. However, the ndim of predictions is "
          << pred_sinfo->ndim);
    }
    K = pred_sinfo->ndim <= 2 ? 0 : pred_sinfo->ndim - 2;
  }
  if (!tgt_sinfo->IsUnknownNdim()) {
    int K_tgt = tgt_sinfo->ndim <= 1 ? 0 : tgt_sinfo->ndim - 1;
    if (K != kUnknownNDim && K != K_tgt) {
      ctx->ReportFatal(Diagnostic::Error(call)
                       << "NLLLoss expects number of dimensions K inferred from different "
                          "arguments to be equal. However, K from predictions is "
                       << K << " while K from targets is " << K_tgt);
    }
  }
  if (wgt_sinfo != nullptr && !wgt_sinfo->IsUnknownNdim() && wgt_sinfo->ndim != 1) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "NLLLoss expects the ndim of weights == 1. However, the ndim of weights is "
                     << wgt_sinfo->ndim);
  }

  arith::Analyzer* analyzer = ctx->GetAnalyzer();
  Optional<PrimExpr> N;
  Optional<PrimExpr> C;
  Array<PrimExpr> output_shape;  // N, d1, d2, ..., dk

  Optional<Array<PrimExpr>> pred_shape_value;
  if (pred_sinfo->shape.defined()) {
    pred_shape_value = GetStructInfoAs<ShapeStructInfoNode>(pred_sinfo->shape.value())->values;
  }
  if (pred_shape_value.defined()) {
    if (pred_shape_value.value().size() == 1) {
      // (C,)
      ICHECK(pred_sinfo->ndim == 1);
      C = pred_shape_value.value()[0];
    } else {
      // (N, C, d1, d2, ..., dk)
      ICHECK(pred_shape_value.value().size() >= 2);
      ICHECK(pred_sinfo->ndim == static_cast<int>(pred_shape_value.value().size()));
      N = pred_shape_value.value()[0];
      C = pred_shape_value.value()[1];
      output_shape = Array<PrimExpr>();
      output_shape.push_back(N.value());
      for (size_t i = 2; i < pred_shape_value.value().size(); ++i) {
        output_shape.push_back(pred_shape_value.value()[i]);
      }
    }
  }

  Optional<Array<PrimExpr>> tgt_shape_value;
  if (tgt_sinfo->shape.defined()) {
    tgt_shape_value = GetStructInfoAs<ShapeStructInfoNode>(tgt_sinfo->shape.value())->values;
  }
  if (tgt_shape_value.defined()) {
    if (tgt_shape_value.value().empty()) {
      // ()
      ICHECK(tgt_sinfo->ndim == 0);
      if (N.defined()) {
        ctx->ReportFatal(Diagnostic::Error(call)
                         << "Shape mismatch for NLLLoss. Predictions shape is "
                            "(N, C, ...) while targets is a scalar");
      }
    } else {
      // (N,) or (N, d1, d2, ..., dk)
      // check N
      const PrimExpr& N_tgt = tgt_shape_value.value()[0];
      if (N.defined() && analyzer->CanProve(N.value() != N_tgt)) {
        ctx->ReportFatal(Diagnostic::Error(call)
                         << "NLLLoss expects minibatch size N inferred from different "
                            "arguments to be equal. However, N from predictions is "
                         << N << " while N from targets is " << N_tgt);
      }
      // only C case
      if (!N.defined() && C.defined()) {
        ctx->ReportFatal(Diagnostic::Error(call)
                         << "Shape mismatch for NLLLoss. Predictions shape is "
                            "(C,) while targets is not a scalar");
      }

      if (tgt_shape_value.value().size() == 1) {
        // (N,)
        ICHECK(tgt_sinfo->IsUnknownNdim() || tgt_sinfo->ndim == 1);
      } else {
        // (N, d1, d2, ..., dk)
        ICHECK(tgt_shape_value.value().size() >= 2);
        ICHECK(tgt_sinfo->IsUnknownNdim() ||
               tgt_sinfo->ndim == static_cast<int>(tgt_shape_value.value().size()));

        if (pred_shape_value.defined()) {
          // check (d1, d2, ..., dk)
          for (size_t i = 1; i < tgt_shape_value.value().size(); ++i) {
            if (analyzer->CanProve(output_shape[i] != tgt_shape_value.value()[i])) {
              ctx->ReportFatal(Diagnostic::Error(call)
                               << "Shape mismatch for NLLLoss. The prediction shape at this dim is "
                               << output_shape[i] << " while the target shape at this dim is "
                               << tgt_shape_value.value()[i]);
            }
          }
        }
      }
    }
  }

  if (wgt_sinfo != nullptr) {
    Optional<Array<PrimExpr>> wgt_shape_value;
    if (wgt_sinfo->shape.defined()) {
      wgt_shape_value = GetStructInfoAs<ShapeStructInfoNode>(wgt_sinfo->shape.value())->values;
    }
    if (wgt_shape_value.defined()) {
      ICHECK(wgt_shape_value.value().size() == 1);
      ICHECK(wgt_sinfo->IsUnknownNdim() || wgt_sinfo->ndim == 1);
      const PrimExpr& C_wgt = wgt_shape_value.value()[0];
      if (C.defined() && analyzer->CanProve(C.value() != C_wgt)) {
        ctx->ReportFatal(Diagnostic::Error(call)
                         << "NLLLoss expects number of classes C inferred from different "
                            "arguments to be equal. However, C from predictions is "
                         << C << " while C from weights is " << C_wgt);
      }
    }
  }

  const auto* attrs = call->attrs.as<NLLLossAttrs>();
  String reduction = attrs->reduction;

  if (reduction == "none") {
    // () or (N,) or (N, d1, d2, ..., dk)
    if (pred_sinfo->shape.as<ShapeExprNode>()) {
      return TensorStructInfo(ShapeExpr(output_shape), output_dtype, vdevice);
    } else {
      int output_ndim = pred_sinfo->ndim == kUnknownNDim ? kUnknownNDim : pred_sinfo->ndim - 1;
      return TensorStructInfo(output_dtype, /*ndim=*/output_ndim, vdevice);
    }
  } else {
    // sum or mean. output is scalar
    return TensorStructInfo(/*shape=*/ShapeExpr(Array<PrimExpr>()), output_dtype, vdevice);
  }
}

TVM_REGISTER_OP("relax.nn.nll_loss")
    .set_attrs_type<NLLLossAttrs>()
    .set_num_inputs(3)
    .add_argument("predictions", "Tensor", "The prediction tensor.")
    .add_argument("targets", "Tensor", "The target tensor.")
    .add_argument("weights", "Optional<Tensor>", "The weight of each target values.")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoNLLLoss)
    .set_attr<Bool>("FPurity", Bool(true));

}  // namespace relax
}  // namespace tvm
