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

/* relax.nn.silu */
RELAX_REGISTER_UNARY_NN_OP_AND_IMPL(silu, "nn.silu", /*require_float_dtype=*/true);

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

TVM_REGISTER_OP("relax.nn.softmax")
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor.")
    .set_attrs_type<SoftmaxAttrs>()
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoSoftmax);

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
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoSoftmax);

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
                int axis, double epsilon, bool center, bool scale) {
  ObjectPtr<BatchNormAttrs> attrs = make_object<BatchNormAttrs>();
  attrs->axis = axis;
  attrs->epsilon = epsilon;
  attrs->center = center;
  attrs->scale = scale;

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
    return TupleStructInfo({TensorStructInfo(dtype, input_sinfo[0]->ndim),
                            TensorStructInfo(dtype, /*ndim=*/1),
                            TensorStructInfo(dtype, /*ndim=*/1)});
  } else {
    return TupleStructInfo({input_sinfo[0], input_sinfo[3], input_sinfo[4]});
  }
}

TVM_REGISTER_OP("relax.nn.batch_norm")
    .set_attrs_type<BatchNormAttrs>()
    .set_num_inputs(5)
    .add_argument("data", "Tensor", "Input to which batch_norm will be applied.")
    .add_argument("gamma", "Tensor", "The gamma scale factor.")
    .add_argument("beta", "Tensor", "The beta offset factor.")
    .add_argument("moving_mean", "Tensor", "Running mean of input.")
    .add_argument("moving_var", "Tensor", "Running variance of input.")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoBatchNorm);

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

  return unknown_shape ? TensorStructInfo(input_sinfo[0]->dtype, input_sinfo[0]->ndim)
                       : input_sinfo[0];
}

TVM_REGISTER_OP("relax.nn.layer_norm")
    .set_attrs_type<LayerNormAttrs>()
    .set_num_inputs(3)
    .add_argument("data", "Tensor", "Input to which batch_norm will be applied.")
    .add_argument("gamma", "Tensor", "The gamma scale factor.")
    .add_argument("beta", "Tensor", "The beta offset factor.")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoLayerNorm);

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
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoDropout);

/* relax.nn.cross_entropy_with_logits */
StructInfo InferStructInfoCrossEntropy(const Call& call, const BlockBuilder& ctx) {
  Array<TensorStructInfo> input_sinfo = GetInputTensorStructInfo(call, ctx);
  TensorStructInfo pred_sinfo = input_sinfo[0];
  TensorStructInfo label_sinfo = input_sinfo[1];

  // infer dtype
  DataType dtype = InferBinaryArithOpOutDtype(call, ctx, pred_sinfo, label_sinfo);

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
  return TensorStructInfo(ShapeExpr(Array<PrimExpr>()), dtype);
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
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoCrossEntropy);

}  // namespace relax
}  // namespace tvm
