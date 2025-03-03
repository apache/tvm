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
 * \file src/contrib/msc/core/transform/set_expr_layout.cc
 * \brief Pass for setting layout for expr and constant.
 */

#include <tvm/relax/expr.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/transform.h>

#include "../utils.h"
#include "layout_utils.h"

namespace tvm {
namespace relax {

using namespace tvm::contrib::msc;

std::tuple<int64_t, int64_t> AccumulateMatch(const Array<PrimExpr>& input_shape,
                                             const Array<PrimExpr>& output_shape, size_t in_start,
                                             size_t out_start) {
  // find input position in_pos and output position out_pos
  // cumsum(in_shape[in_start:in_pos])==cumsum(out_shape[out_start:out_pos])
  std::vector<int64_t> in_shape, out_shape;
  for (const auto& s : input_shape) {
    in_shape.push_back(Downcast<Integer>(s)->value);
  }
  for (const auto& s : output_shape) {
    out_shape.push_back(Downcast<Integer>(s)->value);
  }
  int64_t in_size = static_cast<int64_t>(in_shape.size());
  int64_t out_size = static_cast<int64_t>(out_shape.size());
  int64_t in_pos = in_start;
  int64_t out_pos = out_start;
  int64_t in_accumulate = in_shape[in_pos];
  int64_t out_accumulate = out_shape[out_pos];
  while (in_accumulate != out_accumulate) {
    if (in_accumulate > out_accumulate) {
      out_pos += 1;
      if (out_pos >= out_size) {
        return std::make_tuple(-1, -1);
      }
      out_accumulate *= out_shape[out_pos];
    } else {
      in_pos += 1;
      if (in_pos >= in_size) {
        return std::make_tuple(-1, -1);
      }
      in_accumulate *= in_shape[in_pos];
    }
  }
  if (in_accumulate != out_accumulate) {
    return std::make_tuple(-1, -1);
  }
  // append tailing
  if (in_pos >= 0) {
    while (in_pos < in_size - 1 && in_shape[in_pos + 1] == 1) {
      in_pos++;
    }
    while (out_pos < out_size - 1 && out_shape[out_pos + 1] == 1) {
      out_pos++;
    }
  }
  return std::make_tuple(in_pos - in_start, out_pos - out_start);
}

std::tuple<std::vector<size_t>, std::vector<size_t>> InferReshapeAxes(
    const Array<PrimExpr>& input_shape, const Array<PrimExpr>& output_shape, int batch_dim) {
  std::vector<size_t> expand_axes, reduce_axes;
  size_t in_start = 0;
  while (in_start < input_shape.size()) {
    size_t out_start = in_start + expand_axes.size() - reduce_axes.size();
    int64_t in_dist, out_dist;
    std::tie(in_dist, out_dist) = AccumulateMatch(input_shape, output_shape, in_start, out_start);
    if (in_dist == -1) {
      return std::make_tuple(std::vector<size_t>(), std::vector<size_t>());
    }
    if (out_dist >= in_dist) {
      for (size_t i = 0; i < static_cast<size_t>(out_dist - in_dist); i++) {
        if (batch_dim >= 0 && (out_start + i) == static_cast<size_t>(batch_dim)) {
          expand_axes.push_back(out_start + i + 1);
        } else {
          expand_axes.push_back(out_start + i);
        }
      }
    } else {
      for (size_t i = 0; i < static_cast<size_t>(in_dist - out_dist); i++) {
        if (batch_dim >= 0 && (in_start + i) == static_cast<size_t>(batch_dim)) {
          reduce_axes.push_back(in_start + i + 1);
        } else {
          reduce_axes.push_back(in_start + i);
        }
      }
    }
    in_start += in_dist + 1;
  }
  if (input_shape.size() + expand_axes.size() - reduce_axes.size() != output_shape.size()) {
    return std::make_tuple(std::vector<size_t>(), std::vector<size_t>());
  }
  return std::make_tuple(expand_axes, reduce_axes);
}

// Forward and Backward infer
InferLayoutOutput MSCInferLayoutConv(const Call& call,
                                     const Map<String, Array<String>>& desired_layouts,
                                     const VarLayoutMap& var_layout_map) {
  LayoutDecision data_layout, kernel_layout, out_layout;
  const String& op_name = Downcast<Op>(call->op)->name;
  if (op_name == "relax.nn.conv1d") {
    const auto* attrs = call->attrs.as<Conv1DAttrs>();
    data_layout = LayoutDecision(attrs->data_layout);
    kernel_layout = LayoutDecision(attrs->kernel_layout);
    out_layout = LayoutDecision(attrs->out_layout);
  } else if (op_name == "relax.nn.conv2d") {
    const auto* attrs = call->attrs.as<Conv2DAttrs>();
    data_layout = LayoutDecision(attrs->data_layout);
    kernel_layout = LayoutDecision(attrs->kernel_layout);
    out_layout = LayoutDecision(attrs->out_layout);
  } else if (op_name == "relax.nn.conv2d_transpose") {
    const auto* attrs = call->attrs.as<Conv2DTransposeAttrs>();
    data_layout = LayoutDecision(attrs->data_layout);
    kernel_layout = LayoutDecision(attrs->kernel_layout);
    out_layout = LayoutDecision(attrs->out_layout);
  }
  return InferLayoutOutput({data_layout, kernel_layout}, {out_layout}, Attrs());
}

InferLayoutOutput MSCInferLayoutPool2d(const Call& call,
                                       const Map<String, Array<String>>& desired_layouts,
                                       const VarLayoutMap& var_layout_map) {
  LayoutDecision layout, out_layout;
  const String& op_name = Downcast<Op>(call->op)->name;
  if (op_name == "relax.nn.adaptive_avg_pool2d") {
    const auto* attrs = call->attrs.as<AdaptivePool2DAttrs>();
    layout = LayoutDecision(attrs->layout);
    out_layout = LayoutDecision(attrs->out_layout);
  } else {
    const auto* attrs = call->attrs.as<Pool2DAttrs>();
    layout = LayoutDecision(attrs->layout);
    out_layout = LayoutDecision(attrs->out_layout);
  }
  return InferLayoutOutput({layout}, {out_layout}, Attrs());
}

InferLayoutOutput MSCInferLayoutResize2d(const Call& call,
                                         const Map<String, Array<String>>& desired_layouts,
                                         const VarLayoutMap& var_layout_map) {
  const auto* attrs = call->attrs.as<Resize2DAttrs>();
  const auto& data_layout = LayoutDecision(attrs->layout);
  const auto& shape_layout = LayoutDecision("O");
  return InferLayoutOutput({data_layout, shape_layout}, {data_layout}, Attrs());
}

// Forward Infer
InferLayoutOutput ForwardInferLayoutCommon(const Call& call,
                                           const Map<String, Array<String>>& desired_layouts,
                                           const VarLayoutMap& var_layout_map) {
  Array<NLayout> input_layouts;
  LayoutDecision layout_hint;
  for (const auto& arg : call->args) {
    const auto& in_layout = LayoutUtils::InferLayoutDecision(arg, var_layout_map);
    if (in_layout->layout.defined()) {
      layout_hint = in_layout;
    }
    input_layouts.push_back(in_layout);
  }
  if (!layout_hint.defined()) {
    return InferLayoutOutput();
  }
  const auto& sinfo = GetStructInfo(call);
  if (sinfo->IsInstance<TensorStructInfoNode>()) {
    return InferLayoutOutput(input_layouts, {layout_hint}, Attrs());
  }
  Array<NLayout> output_layouts;
  if (const auto* tuple_sinfo = sinfo.as<TupleStructInfoNode>()) {
    for (size_t i = 0; i < tuple_sinfo->fields.size(); i++) {
      output_layouts.push_back(layout_hint);
    }
    return InferLayoutOutput(input_layouts, {output_layouts}, Attrs());
  }
  return InferLayoutOutput();
}

InferLayoutOutput ForwardInferLayoutBroadcast(const Call& call,
                                              const Map<String, Array<String>>& desired_layouts,
                                              const VarLayoutMap& var_layout_map) {
  Array<NLayout> input_layouts;
  LayoutDecision layout_hint;
  for (const auto& arg : call->args) {
    const auto& in_layout = LayoutUtils::InferLayoutDecision(arg, var_layout_map);
    if (in_layout->layout.defined()) {
      if (!layout_hint.defined() || layout_hint->layout.ndim() < in_layout->layout.ndim()) {
        layout_hint = in_layout;
      }
    }
    input_layouts.push_back(in_layout);
  }
  if (!layout_hint.defined()) {
    return InferLayoutOutput();
  }
  const auto& sinfo = GetStructInfo(call);
  if (sinfo->IsInstance<TensorStructInfoNode>()) {
    return InferLayoutOutput(input_layouts, {layout_hint}, Attrs());
  }
  return InferLayoutOutput();
}

InferLayoutOutput ForwardInferLayoutInplace(const Call& call,
                                            const Map<String, Array<String>>& desired_layouts,
                                            const VarLayoutMap& var_layout_map) {
  return ForwardInferLayoutCommon(call, desired_layouts, var_layout_map);
}

InferLayoutOutput ForwardInferLayoutBinary(const Call& call,
                                           const Map<String, Array<String>>& desired_layouts,
                                           const VarLayoutMap& var_layout_map) {
  const auto& output = ForwardInferLayoutCommon(call, desired_layouts, var_layout_map);
  if (!output.defined()) {
    return output;
  }
  std::vector<NLayout> input_layouts;
  for (size_t i = 0; i < call->args.size(); i++) {
    const auto& sinfo = GetStructInfo(call->args[i]);
    if (const auto* t_info = sinfo.as<TensorStructInfoNode>()) {
      if (t_info->ndim == 0) {
        input_layouts.push_back(LayoutDecision(""));
      } else if (t_info->ndim == 1) {
        const auto& ref_layout = output->output_layouts[0].LeafValue()->layout;
        input_layouts.push_back(LayoutDecision(ref_layout[ref_layout.ndim() - 1].name()));
      } else {
        input_layouts.push_back(output->input_layouts[i]);
      }
    } else {
      LOG(FATAL) << "Binary input should be tensor, get " << sinfo->GetTypeKey();
    }
  }
  return InferLayoutOutput(input_layouts, output->output_layouts, Attrs());
}

InferLayoutOutput ForwardInferLayoutArgMaxMin(const Call& call,
                                              const Map<String, Array<String>>& desired_layouts,
                                              const VarLayoutMap& var_layout_map) {
  LayoutDecision input_layout = LayoutUtils::InferLayoutDecision(call->args[0], var_layout_map);
  if (!input_layout->layout.defined()) {
    return InferLayoutOutput();
  }
  const auto* attrs = call->attrs.as<ArgmaxArgminAttrs>();
  if (attrs->keepdims) {
    return InferLayoutOutput({input_layout}, {input_layout}, Attrs());
  }
  if (!attrs->axis.defined()) {
    return InferLayoutOutput({input_layout}, {LayoutDecision("")}, Attrs());
  }
  const auto& input_shape = ExprUtils::GetShape(call->args[0]);
  if (input_shape.size() == 0) {
    return InferLayoutOutput();
  }
  std::vector<size_t> axes;
  axes.push_back(CommonUtils::GetIndex(Downcast<Integer>(attrs->axis)->value, input_shape.size()));
  LayoutDecision output_layout = LayoutUtils::ReduceLayout(input_layout, axes);
  return InferLayoutOutput({input_layout}, {output_layout}, Attrs());
}

InferLayoutOutput ForwardInferLayoutBatchNorm(const Call& call,
                                              const Map<String, Array<String>>& desired_layouts,
                                              const VarLayoutMap& var_layout_map) {
  const auto& input_shape = ExprUtils::GetShape(call->args[0]);
  if (input_shape.size() == 0) {
    return InferLayoutOutput();
  }
  LayoutDecision in_layout = LayoutUtils::InferLayoutDecision(call->args[0], var_layout_map);
  if (!in_layout->layout.defined()) {
    if (input_shape.size() == 4) {
      in_layout = LayoutDecision("NCHW");
    } else if (input_shape.size() == 3) {
      in_layout = LayoutDecision("NCD");
    }
  }
  LayoutDecision g_layout = LayoutDecision("O");
  return InferLayoutOutput({in_layout, g_layout, g_layout, g_layout, g_layout},
                           {{in_layout, g_layout, g_layout}}, Attrs());
}

InferLayoutOutput ForkwardInferLayoutExpandDims(const Call& call,
                                                const Map<String, Array<String>>& desired_layouts,
                                                const VarLayoutMap& var_layout_map) {
  LayoutDecision input_layout = LayoutUtils::InferLayoutDecision(call->args[0], var_layout_map);
  if (!input_layout->layout.defined()) {
    return InferLayoutOutput();
  }
  const auto& input_shape = ExprUtils::GetShape(call->args[0]);
  if (input_shape.size() == 0) {
    return InferLayoutOutput();
  }
  const auto* attrs = call->attrs.as<ExpandDimsAttrs>();
  std::vector<size_t> expand_axes;
  for (const auto& s : attrs->axis) {
    expand_axes.push_back(CommonUtils::GetIndex(s->value, input_shape.size()));
  }
  LayoutDecision output_layout = LayoutUtils::ExpandLayout(input_layout, expand_axes);
  return InferLayoutOutput({input_layout}, {output_layout}, Attrs());
}

InferLayoutOutput ForwardInferLayoutNormalize(const Call& call,
                                              const Map<String, Array<String>>& desired_layouts,
                                              const VarLayoutMap& var_layout_map) {
  const auto& input_shape = ExprUtils::GetShape(call->args[0]);
  if (input_shape.size() == 0) {
    return InferLayoutOutput();
  }
  LayoutDecision in_layout = LayoutUtils::InferLayoutDecision(call->args[0], var_layout_map);
  if (!in_layout->layout.defined()) {
    if (input_shape.size() == 4) {
      in_layout = LayoutDecision("NCHW");
    } else if (input_shape.size() == 3) {
      in_layout = LayoutDecision("NCD");
    }
  }
  LayoutDecision g_layout = LayoutDecision("O");
  return InferLayoutOutput({in_layout, g_layout, g_layout}, {in_layout}, Attrs());
}

InferLayoutOutput ForwardInferLayoutMatmul(const Call& call,
                                           const Map<String, Array<String>>& desired_layouts,
                                           const VarLayoutMap& var_layout_map) {
  const auto& a_shape = ExprUtils::GetShape(call->args[0]);
  const auto& b_shape = ExprUtils::GetShape(call->args[1]);
  if (a_shape.size() == 0) {
    return InferLayoutOutput();
  }
  LayoutDecision a_layout = LayoutUtils::InferLayoutDecision(call->args[0], var_layout_map);
  if (!a_layout->layout.defined()) {
    if (a_shape.size() == 4) {
      a_layout = LayoutDecision("NCHW");
    } else if (a_shape.size() == 3) {
      a_layout = LayoutDecision("NCD");
    } else if (a_shape.size() == 2) {
      a_layout = LayoutDecision("NC");
    }
  }
  size_t start = a_layout->layout.ndim() - b_shape.size();
  String pre_layout;
  for (size_t i = start; i < a_layout->layout.ndim() - 2; i++) {
    pre_layout = pre_layout + a_layout->layout[i].name();
  }
  LayoutDecision b_layout = LayoutDecision(pre_layout + "IO");
  return InferLayoutOutput({a_layout, b_layout}, {a_layout}, Attrs());
}

InferLayoutOutput ForwardInferLayoutPermute(const Call& call,
                                            const Map<String, Array<String>>& desired_layouts,
                                            const VarLayoutMap& var_layout_map) {
  LayoutDecision input_layout = LayoutUtils::InferLayoutDecision(call->args[0], var_layout_map);
  if (!input_layout->layout.defined()) {
    return InferLayoutOutput();
  }
  std::vector<size_t> permute_axes;
  const auto* attrs = call->attrs.as<PermuteDimsAttrs>();
  if (!attrs->axes.defined()) {
    for (size_t i = input_layout->layout.ndim(); i > 0; i--) {
      permute_axes.push_back(i - 1);
    }
  } else {
    for (const auto& a : attrs->axes.value()) {
      permute_axes.push_back(a->value);
    }
  }
  LayoutDecision output_layout = LayoutUtils::PermuteLayout(input_layout, permute_axes);
  return InferLayoutOutput({input_layout}, {output_layout}, Attrs());
}

InferLayoutOutput ForwardInferLayoutReduceAxis(const Call& call,
                                               const Map<String, Array<String>>& desired_layouts,
                                               const VarLayoutMap& var_layout_map) {
  LayoutDecision input_layout = LayoutUtils::InferLayoutDecision(call->args[0], var_layout_map);
  if (!input_layout->layout.defined()) {
    return InferLayoutOutput();
  }
  const auto* attrs = call->attrs.as<StatisticalAttrs>();
  if (attrs->keepdims) {
    return InferLayoutOutput({input_layout}, {input_layout}, Attrs());
  }
  if (!attrs->axis.defined()) {
    return InferLayoutOutput({input_layout}, {LayoutDecision("")}, Attrs());
  }
  const auto& input_shape = ExprUtils::GetShape(call->args[0]);
  if (input_shape.size() == 0) {
    return InferLayoutOutput();
  }
  std::vector<size_t> axes;
  for (const auto& s : attrs->axis.value()) {
    axes.push_back(CommonUtils::GetIndex(s->value, input_shape.size()));
  }
  LayoutDecision output_layout = LayoutUtils::ReduceLayout(input_layout, axes);
  return InferLayoutOutput({input_layout}, {output_layout}, Attrs());
}

InferLayoutOutput ForwardInferLayoutReshape(const Call& call,
                                            const Map<String, Array<String>>& desired_layouts,
                                            const VarLayoutMap& var_layout_map) {
  LayoutDecision input_layout = LayoutUtils::InferLayoutDecision(call->args[0], var_layout_map);
  if (!input_layout->layout.defined()) {
    return InferLayoutOutput();
  }
  const auto& input_shape = ExprUtils::GetShape(call->args[0]);
  const auto& output_shape = ExprUtils::GetShape(call);
  if (input_shape.size() == 0 || output_shape.size() == 0) {
    return InferLayoutOutput();
  }
  LayoutDecision output_layout = input_layout;
  if (input_shape.size() != output_shape.size()) {
    int batch_dim = LayoutUtils::InferBatchDim(input_layout);
    std::vector<size_t> expand_axes, reduce_axes;
    std::tie(expand_axes, reduce_axes) = InferReshapeAxes(input_shape, output_shape, batch_dim);
    if (reduce_axes.size() == 0 && expand_axes.size() == 0) {
      return InferLayoutOutput();
    }
    if (reduce_axes.size() > 0) {
      output_layout = LayoutUtils::ReduceLayout(output_layout, reduce_axes);
    }
    if (expand_axes.size() > 0) {
      output_layout = LayoutUtils::ExpandLayout(output_layout, expand_axes);
    }
  }
  return InferLayoutOutput({input_layout, LayoutDecision("O")}, {output_layout}, Attrs());
}

InferLayoutOutput ForwardInferLayoutSqueeze(const Call& call,
                                            const Map<String, Array<String>>& desired_layouts,
                                            const VarLayoutMap& var_layout_map) {
  LayoutDecision input_layout = LayoutUtils::InferLayoutDecision(call->args[0], var_layout_map);
  if (!input_layout->layout.defined()) {
    return InferLayoutOutput();
  }
  const auto& input_shape = ExprUtils::GetShape(call->args[0]);
  if (input_shape.size() == 0) {
    return InferLayoutOutput();
  }
  const auto* attrs = call->attrs.as<SqueezeAttrs>();
  std::vector<size_t> reduce_axes;
  if (attrs->axis.defined()) {
    for (const auto& s : attrs->axis.value()) {
      size_t v_index = CommonUtils::GetIndex(s->value, input_shape.size());
      if (Downcast<Integer>(input_shape[v_index])->value == 1) {
        reduce_axes.push_back(v_index);
      }
    }
  } else {
    for (size_t i = 0; i < input_shape.size(); i++) {
      if (Downcast<Integer>(input_shape[i])->value == 1) {
        reduce_axes.push_back(i);
      }
    }
  }
  LayoutDecision output_layout = LayoutUtils::ReduceLayout(input_layout, reduce_axes);
  return InferLayoutOutput({input_layout}, {output_layout}, Attrs());
}

InferLayoutOutput ForwardInferLayoutTake(const Call& call,
                                         const Map<String, Array<String>>& desired_layouts,
                                         const VarLayoutMap& var_layout_map) {
  LayoutDecision input_layout = LayoutUtils::InferLayoutDecision(call->args[0], var_layout_map);
  LayoutDecision indices_layout = LayoutUtils::InferLayoutDecision(call->args[1], var_layout_map);
  const auto& input_shape = ExprUtils::GetShape(call->args[0]);
  const auto& output_shape = ExprUtils::GetShape(call);
  if (input_shape.size() == 0) {
    return InferLayoutOutput();
  }
  if (input_layout->layout.defined()) {
    if (input_shape.size() == output_shape.size()) {
      return InferLayoutOutput({input_layout, indices_layout}, {input_layout}, Attrs());
    }
    LayoutDecision output_layout = LayoutUtils::ReduceLayout(input_layout, std::vector<size_t>{0});
    return InferLayoutOutput({input_layout, indices_layout}, {output_layout}, Attrs());
  }
  if (indices_layout->layout.defined()) {
    std::vector<size_t> expand_axes;
    for (size_t i = indices_layout->layout.ndim(); i < output_shape.size(); i++) {
      expand_axes.push_back(i);
    }
    LayoutDecision output_layout;
    if (expand_axes.size() == 0) {
      output_layout = indices_layout;
    } else {
      output_layout = LayoutUtils::ExpandLayout(indices_layout, expand_axes);
    }
    return InferLayoutOutput({input_layout, indices_layout}, {output_layout}, Attrs());
  }
  return InferLayoutOutput();
}

InferLayoutOutput ForwardInferLayoutPlugin(const Call& call,
                                           const Map<String, Array<String>>& desired_layouts,
                                           const VarLayoutMap& var_layout_map) {
  if (!call->args[0]->IsInstance<ExternFuncNode>()) {
    return InferLayoutOutput();
  }
  const auto& name = Downcast<ExternFunc>(call->args[0])->global_symbol;
  const auto* pf = runtime::Registry::Get("msc.plugin.op.InferLayout" + name);
  if (pf == nullptr) {
    return InferLayoutOutput();
  }
  const auto& args = Downcast<Tuple>(call->args[1]);
  return (*pf)(args->fields, var_layout_map);
}

// nn ops
TVM_REGISTER_OP("relax.nn.avg_pool2d")
    .set_attr<FRelaxInferLayout>("FMSCForwardInferLayout", MSCInferLayoutPool2d);
TVM_REGISTER_OP("relax.nn.adaptive_avg_pool2d")
    .set_attr<FRelaxInferLayout>("FMSCForwardInferLayout", MSCInferLayoutPool2d);
TVM_REGISTER_OP("relax.nn.batch_norm")
    .set_attr<FRelaxInferLayout>("FMSCForwardInferLayout", ForwardInferLayoutBatchNorm);
TVM_REGISTER_OP("relax.nn.conv1d")
    .set_attr<FRelaxInferLayout>("FMSCForwardInferLayout", MSCInferLayoutConv);
TVM_REGISTER_OP("relax.nn.conv2d")
    .set_attr<FRelaxInferLayout>("FMSCForwardInferLayout", MSCInferLayoutConv);
TVM_REGISTER_OP("relax.nn.conv2d_transpose")
    .set_attr<FRelaxInferLayout>("FMSCForwardInferLayout", MSCInferLayoutConv);
TVM_REGISTER_OP("relax.nn.dropout")
    .set_attr<FRelaxInferLayout>("FMSCForwardInferLayout", ForwardInferLayoutCommon);
TVM_REGISTER_OP("relax.nn.group_norm")
    .set_attr<FRelaxInferLayout>("FMSCForwardInferLayout", ForwardInferLayoutNormalize);
TVM_REGISTER_OP("relax.nn.layer_norm")
    .set_attr<FRelaxInferLayout>("FMSCForwardInferLayout", ForwardInferLayoutNormalize);
TVM_REGISTER_OP("relax.nn.max_pool2d")
    .set_attr<FRelaxInferLayout>("FMSCForwardInferLayout", MSCInferLayoutPool2d);

// reduce axis ops
TVM_REGISTER_OP("relax.argmax")
    .set_attr<FRelaxInferLayout>("FMSCForwardInferLayout", ForwardInferLayoutArgMaxMin);
TVM_REGISTER_OP("relax.argmin")
    .set_attr<FRelaxInferLayout>("FMSCForwardInferLayout", ForwardInferLayoutArgMaxMin);
TVM_REGISTER_OP("relax.max")
    .set_attr<FRelaxInferLayout>("FMSCForwardInferLayout", ForwardInferLayoutReduceAxis);
TVM_REGISTER_OP("relax.min")
    .set_attr<FRelaxInferLayout>("FMSCForwardInferLayout", ForwardInferLayoutReduceAxis);
TVM_REGISTER_OP("relax.mean")
    .set_attr<FRelaxInferLayout>("FMSCForwardInferLayout", ForwardInferLayoutReduceAxis);
TVM_REGISTER_OP("relax.sum")
    .set_attr<FRelaxInferLayout>("FMSCForwardInferLayout", ForwardInferLayoutReduceAxis);
TVM_REGISTER_OP("relax.prod")
    .set_attr<FRelaxInferLayout>("FMSCForwardInferLayout", ForwardInferLayoutReduceAxis);
TVM_REGISTER_OP("relax.std")
    .set_attr<FRelaxInferLayout>("FMSCForwardInferLayout", ForwardInferLayoutReduceAxis);

// binary ops
TVM_REGISTER_OP("relax.add")
    .set_attr<FRelaxInferLayout>("FMSCForwardInferLayout", ForwardInferLayoutBinary);
TVM_REGISTER_OP("relax.divide")
    .set_attr<FRelaxInferLayout>("FMSCForwardInferLayout", ForwardInferLayoutBinary);
TVM_REGISTER_OP("relax.floor_divide")
    .set_attr<FRelaxInferLayout>("FMSCForwardInferLayout", ForwardInferLayoutBinary);
TVM_REGISTER_OP("relax.multiply")
    .set_attr<FRelaxInferLayout>("FMSCForwardInferLayout", ForwardInferLayoutBinary);
TVM_REGISTER_OP("relax.power")
    .set_attr<FRelaxInferLayout>("FMSCForwardInferLayout", ForwardInferLayoutBinary);
TVM_REGISTER_OP("relax.subtract")
    .set_attr<FRelaxInferLayout>("FMSCForwardInferLayout", ForwardInferLayoutBinary);
TVM_REGISTER_OP("relax.equal")
    .set_attr<FRelaxInferLayout>("FMSCForwardInferLayout", ForwardInferLayoutBinary);
TVM_REGISTER_OP("relax.greater")
    .set_attr<FRelaxInferLayout>("FMSCForwardInferLayout", ForwardInferLayoutBinary);
TVM_REGISTER_OP("relax.greater_equal")
    .set_attr<FRelaxInferLayout>("FMSCForwardInferLayout", ForwardInferLayoutBinary);
TVM_REGISTER_OP("relax.less")
    .set_attr<FRelaxInferLayout>("FMSCForwardInferLayout", ForwardInferLayoutBinary);
TVM_REGISTER_OP("relax.less_equal")
    .set_attr<FRelaxInferLayout>("FMSCForwardInferLayout", ForwardInferLayoutBinary);
TVM_REGISTER_OP("relax.not_equal")
    .set_attr<FRelaxInferLayout>("FMSCForwardInferLayout", ForwardInferLayoutBinary);
TVM_REGISTER_OP("relax.maximum")
    .set_attr<FRelaxInferLayout>("FMSCForwardInferLayout", ForwardInferLayoutBinary);
TVM_REGISTER_OP("relax.minimum")
    .set_attr<FRelaxInferLayout>("FMSCForwardInferLayout", ForwardInferLayoutBinary);
TVM_REGISTER_OP("relax.logical_and")
    .set_attr<FRelaxInferLayout>("FMSCForwardInferLayout", ForwardInferLayoutBinary);
TVM_REGISTER_OP("relax.logical_or")
    .set_attr<FRelaxInferLayout>("FMSCForwardInferLayout", ForwardInferLayoutBinary);
TVM_REGISTER_OP("relax.logical_xor")
    .set_attr<FRelaxInferLayout>("FMSCForwardInferLayout", ForwardInferLayoutBinary);
TVM_REGISTER_OP("relax.bitwise_and")
    .set_attr<FRelaxInferLayout>("FMSCForwardInferLayout", ForwardInferLayoutBinary);
TVM_REGISTER_OP("relax.bitwise_or")
    .set_attr<FRelaxInferLayout>("FMSCForwardInferLayout", ForwardInferLayoutBinary);
TVM_REGISTER_OP("relax.bitwise_xor")
    .set_attr<FRelaxInferLayout>("FMSCForwardInferLayout", ForwardInferLayoutBinary);

// math ops
TVM_REGISTER_OP("relax.expand_dims")
    .set_attr<FRelaxInferLayout>("FMSCForwardInferLayout", ForkwardInferLayoutExpandDims);
TVM_REGISTER_OP("relax.matmul")
    .set_attr<FRelaxInferLayout>("FMSCForwardInferLayout", ForwardInferLayoutMatmul);
TVM_REGISTER_OP("relax.permute_dims")
    .set_attr<FRelaxInferLayout>("FMSCForwardInferLayout", ForwardInferLayoutPermute);
TVM_REGISTER_OP("relax.reshape")
    .set_attr<FRelaxInferLayout>("FMSCForwardInferLayout", ForwardInferLayoutReshape);
TVM_REGISTER_OP("relax.squeeze")
    .set_attr<FRelaxInferLayout>("FMSCForwardInferLayout", ForwardInferLayoutSqueeze);
TVM_REGISTER_OP("relax.take")
    .set_attr<FRelaxInferLayout>("FMSCForwardInferLayout", ForwardInferLayoutTake);
TVM_REGISTER_OP("relax.image.resize2d")
    .set_attr<FRelaxInferLayout>("FMSCForwardInferLayout", MSCInferLayoutResize2d);

// plugin op
TVM_REGISTER_OP("relax.call_dps_packed")
    .set_attr<FRelaxInferLayout>("FMSCForwardInferLayout", ForwardInferLayoutPlugin);

// Backward Infer
InferLayoutOutput BackwardInferLayoutCommon(const Call& call,
                                            const Map<String, Array<String>>& desired_layouts,
                                            const VarLayoutMap& var_layout_map) {
  NLayout output_layout = LayoutUtils::InferNLayout(call, var_layout_map);
  LayoutDecision layout_hint;
  if (output_layout.IsLeaf()) {
    layout_hint = output_layout.LeafValue();
  } else {
    for (const auto& l : output_layout.NestedArray()) {
      if (l.IsLeaf() && l.LeafValue()->layout.defined()) {
        layout_hint = l.LeafValue();
      }
    }
  }
  if (!layout_hint->layout.defined()) {
    return InferLayoutOutput();
  }
  Array<NLayout> input_layouts;
  for (const auto& arg : call->args) {
    const auto& saved_layout = LayoutUtils::InferLayoutDecision(arg, var_layout_map);
    if (saved_layout->layout.defined()) {
      input_layouts.push_back(saved_layout);
    } else {
      input_layouts.push_back(layout_hint);
    }
  }
  return InferLayoutOutput(input_layouts, {output_layout}, Attrs());
}

InferLayoutOutput BackwardInferLayoutBinary(const Call& call,
                                            const Map<String, Array<String>>& desired_layouts,
                                            const VarLayoutMap& var_layout_map) {
  const auto& output = BackwardInferLayoutCommon(call, desired_layouts, var_layout_map);
  if (!output.defined()) {
    return output;
  }
  std::vector<NLayout> input_layouts;
  for (size_t i = 0; i < call->args.size(); i++) {
    const auto& sinfo = GetStructInfo(call->args[i]);
    if (const auto* t_info = sinfo.as<TensorStructInfoNode>()) {
      if (t_info->ndim == 0) {
        input_layouts.push_back(LayoutDecision(""));
      } else if (t_info->ndim == 1) {
        const auto& ref_layout = output->output_layouts[0].LeafValue()->layout;
        input_layouts.push_back(LayoutDecision(ref_layout[ref_layout.ndim() - 1].name()));
      } else {
        input_layouts.push_back(output->input_layouts[i]);
      }
    } else {
      LOG(FATAL) << "Binary input should be tensor, get " << sinfo->GetTypeKey();
    }
  }
  return InferLayoutOutput(input_layouts, output->output_layouts, Attrs());
}

InferLayoutOutput BackwardInferLayoutInplace(const Call& call,
                                             const Map<String, Array<String>>& desired_layouts,
                                             const VarLayoutMap& var_layout_map) {
  return BackwardInferLayoutCommon(call, desired_layouts, var_layout_map);
}

InferLayoutOutput BackwardInferLayoutArgMaxMin(const Call& call,
                                               const Map<String, Array<String>>& desired_layouts,
                                               const VarLayoutMap& var_layout_map) {
  LayoutDecision output_layout = LayoutUtils::InferLayoutDecision(call, var_layout_map);
  if (!output_layout->layout.defined()) {
    return InferLayoutOutput();
  }
  const auto* attrs = call->attrs.as<ArgmaxArgminAttrs>();
  if (attrs->keepdims) {
    return InferLayoutOutput({output_layout}, {output_layout}, Attrs());
  }
  const auto& input_shape = ExprUtils::GetShape(call->args[0]);
  if (input_shape.size() == 0) {
    return InferLayoutOutput();
  }
  std::vector<size_t> axes;
  axes.push_back(CommonUtils::GetIndex(Downcast<Integer>(attrs->axis)->value, input_shape.size()));
  LayoutDecision input_layout = LayoutUtils::ExpandLayout(output_layout, axes);
  return InferLayoutOutput({input_layout}, {output_layout}, Attrs());
}

InferLayoutOutput BackwardInferLayoutBatchNorm(const Call& call,
                                               const Map<String, Array<String>>& desired_layouts,
                                               const VarLayoutMap& var_layout_map) {
  LayoutDecision output_layout = LayoutUtils::InferLayoutDecisionAt(call, var_layout_map, 0);
  if (!output_layout->layout.defined()) {
    return InferLayoutOutput();
  }
  LayoutDecision g_layout = LayoutDecision("O");
  return InferLayoutOutput({output_layout, g_layout, g_layout, g_layout, g_layout},
                           {{output_layout, g_layout, g_layout}}, Attrs());
}

InferLayoutOutput BackwardInferLayoutExpandDims(const Call& call,
                                                const Map<String, Array<String>>& desired_layouts,
                                                const VarLayoutMap& var_layout_map) {
  LayoutDecision output_layout = LayoutUtils::InferLayoutDecision(call, var_layout_map);
  if (!output_layout->layout.defined()) {
    return InferLayoutOutput();
  }
  const auto& input_shape = ExprUtils::GetShape(call->args[0]);
  if (input_shape.size() == 0) {
    return InferLayoutOutput();
  }
  const auto* attrs = call->attrs.as<ExpandDimsAttrs>();
  std::vector<size_t> expand_axes;
  for (const auto& s : attrs->axis) {
    expand_axes.push_back(CommonUtils::GetIndex(s->value, input_shape.size()));
  }
  LayoutDecision input_layout = LayoutUtils::ReduceLayout(output_layout, expand_axes);
  return InferLayoutOutput({input_layout}, {output_layout}, Attrs());
}

InferLayoutOutput BackwardInferLayoutNormalize(const Call& call,
                                               const Map<String, Array<String>>& desired_layouts,
                                               const VarLayoutMap& var_layout_map) {
  LayoutDecision output_layout = LayoutUtils::InferLayoutDecisionAt(call, var_layout_map, 0);
  if (!output_layout->layout.defined()) {
    return InferLayoutOutput();
  }
  LayoutDecision g_layout = LayoutDecision("O");
  return InferLayoutOutput({output_layout, g_layout, g_layout}, {output_layout}, Attrs());
}

InferLayoutOutput BackwardInferLayoutMatmul(const Call& call,
                                            const Map<String, Array<String>>& desired_layouts,
                                            const VarLayoutMap& var_layout_map) {
  LayoutDecision output_layout = LayoutUtils::InferLayoutDecision(call, var_layout_map);
  if (!output_layout->layout.defined()) {
    return InferLayoutOutput();
  }
  const auto& b_shape = ExprUtils::GetShape(call->args[1]);
  if (b_shape.size() == 0) {
    return InferLayoutOutput();
  }
  size_t start = output_layout->layout.ndim() - b_shape.size();
  String pre_layout;
  for (size_t i = start; i < output_layout->layout.ndim() - 2; i++) {
    pre_layout = pre_layout + output_layout->layout[i].name();
  }
  LayoutDecision b_layout = LayoutDecision(pre_layout + "IO");
  return InferLayoutOutput({output_layout, b_layout}, {output_layout}, Attrs());
}

InferLayoutOutput BackwardInferLayoutPermute(const Call& call,
                                             const Map<String, Array<String>>& desired_layouts,
                                             const VarLayoutMap& var_layout_map) {
  LayoutDecision output_layout = LayoutUtils::InferLayoutDecision(call, var_layout_map);
  if (!output_layout->layout.defined()) {
    return InferLayoutOutput();
  }
  std::vector<size_t> permute_axes;
  const auto* attrs = call->attrs.as<PermuteDimsAttrs>();
  if (!attrs->axes.defined()) {
    for (size_t i = output_layout->layout.ndim(); i > 0; i--) {
      permute_axes.push_back(i - 1);
    }
  } else {
    std::vector<int> attr_axes;
    for (const auto& s : attrs->axes.value()) {
      attr_axes.push_back(s->value);
    }
    for (size_t i = 0; i < output_layout->layout.ndim(); i++) {
      int pos = ArrayUtils::IndexOf(attr_axes, static_cast<int>(i));
      if (pos >= 0) {
        permute_axes.push_back(pos);
      } else {
        permute_axes.push_back(i);
      }
    }
  }
  LayoutDecision input_layout = LayoutUtils::PermuteLayout(output_layout, permute_axes);
  return InferLayoutOutput({input_layout}, {output_layout}, Attrs());
}

InferLayoutOutput BackwardInferLayoutReduceAxis(const Call& call,
                                                const Map<String, Array<String>>& desired_layouts,
                                                const VarLayoutMap& var_layout_map) {
  LayoutDecision output_layout = LayoutUtils::InferLayoutDecision(call, var_layout_map);
  if (!output_layout->layout.defined()) {
    return InferLayoutOutput();
  }
  const auto* attrs = call->attrs.as<StatisticalAttrs>();
  if (attrs->keepdims) {
    return InferLayoutOutput({output_layout}, {output_layout}, Attrs());
  }
  const auto& input_shape = ExprUtils::GetShape(call->args[0]);
  if (input_shape.size() == 0) {
    return InferLayoutOutput();
  }
  std::vector<size_t> axes;
  for (const auto& s : attrs->axis.value()) {
    axes.push_back(CommonUtils::GetIndex(s->value, input_shape.size()));
  }
  LayoutDecision input_layout = LayoutUtils::ExpandLayout(output_layout, axes);
  return InferLayoutOutput({input_layout}, {output_layout}, Attrs());
}

InferLayoutOutput BackwardInferLayoutReshape(const Call& call,
                                             const Map<String, Array<String>>& desired_layouts,
                                             const VarLayoutMap& var_layout_map) {
  LayoutDecision output_layout = LayoutUtils::InferLayoutDecision(call, var_layout_map);
  if (!output_layout->layout.defined()) {
    return InferLayoutOutput();
  }
  const auto& input_shape = ExprUtils::GetShape(call->args[0]);
  const auto& output_shape = ExprUtils::GetShape(call);
  if (input_shape.size() == 0 || output_shape.size() == 0) {
    return InferLayoutOutput();
  }
  LayoutDecision input_layout = output_layout;
  if (input_shape.size() != output_shape.size()) {
    int batch_dim = LayoutUtils::InferBatchDim(output_layout);
    std::vector<size_t> expand_axes, reduce_axes;
    std::tie(expand_axes, reduce_axes) = InferReshapeAxes(input_shape, output_shape, batch_dim);
    if (reduce_axes.size() == 0 && expand_axes.size() == 0) {
      return InferLayoutOutput();
    }
    if (expand_axes.size() > 0) {
      input_layout = LayoutUtils::ReduceLayout(input_layout, expand_axes);
    }
    if (reduce_axes.size() > 0) {
      input_layout = LayoutUtils::ExpandLayout(input_layout, reduce_axes);
    }
  }
  return InferLayoutOutput({input_layout, LayoutDecision("O")}, {output_layout}, Attrs());
}

InferLayoutOutput BackwardInferLayoutSqueeze(const Call& call,
                                             const Map<String, Array<String>>& desired_layouts,
                                             const VarLayoutMap& var_layout_map) {
  LayoutDecision output_layout = LayoutUtils::InferLayoutDecision(call, var_layout_map);
  if (!output_layout->layout.defined()) {
    return InferLayoutOutput();
  }
  const auto& input_shape = ExprUtils::GetShape(call->args[0]);
  if (input_shape.size() == 0) {
    return InferLayoutOutput();
  }
  const auto* attrs = call->attrs.as<SqueezeAttrs>();
  std::vector<size_t> reduce_axes;
  if (attrs->axis.defined()) {
    for (const auto& s : attrs->axis.value()) {
      size_t v_index = CommonUtils::GetIndex(s->value, input_shape.size());
      if (Downcast<Integer>(input_shape[v_index])->value == 1) {
        reduce_axes.push_back(v_index);
      }
    }
  } else {
    for (size_t i = 0; i < input_shape.size(); i++) {
      if (Downcast<Integer>(input_shape[i])->value == 1) {
        reduce_axes.push_back(i);
      }
    }
  }
  LayoutDecision input_layout = LayoutUtils::ExpandLayout(output_layout, reduce_axes);
  return InferLayoutOutput({input_layout}, {output_layout}, Attrs());
}

InferLayoutOutput BackwardInferLayoutTake(const Call& call,
                                          const Map<String, Array<String>>& desired_layouts,
                                          const VarLayoutMap& var_layout_map) {
  LayoutDecision output_layout = LayoutUtils::InferLayoutDecision(call, var_layout_map);
  LayoutDecision input_layout = LayoutUtils::InferLayoutDecision(call->args[0], var_layout_map);
  LayoutDecision indices_layout = LayoutUtils::InferLayoutDecision(call->args[1], var_layout_map);
  const auto& input_shape = ExprUtils::GetShape(call->args[0]);
  const auto& output_shape = ExprUtils::GetShape(call);
  if (!output_layout->layout.defined()) {
    return InferLayoutOutput();
  }
  if (input_shape.size() == 0) {
    return InferLayoutOutput();
  }
  if (!indices_layout.defined()) {
    indices_layout = LayoutUtils::ReduceLayout(output_layout, std::vector<size_t>{0});
  }
  if (input_shape.size() == output_shape.size()) {
    return InferLayoutOutput({output_layout, indices_layout}, {output_layout}, Attrs());
  }
  if (!input_layout.defined()) {
    input_layout = LayoutUtils::ExpandLayout(output_layout, std::vector<size_t>{0});
  }
  return InferLayoutOutput({input_layout, indices_layout}, {output_layout}, Attrs());
}

InferLayoutOutput BackwardInferLayoutTupleInputs(const Call& call,
                                                 const Map<String, Array<String>>& desired_layouts,
                                                 const VarLayoutMap& var_layout_map) {
  LayoutDecision output_layout = LayoutUtils::InferLayoutDecision(call, var_layout_map);
  if (!output_layout->layout.defined()) {
    return InferLayoutOutput();
  }
  std::vector<NLayout> input_layouts;
  if (const auto* t_node = GetStructInfo(call->args[0]).as<TupleStructInfoNode>()) {
    for (size_t i = 0; i < t_node->fields.size(); i++) {
      input_layouts.push_back(output_layout);
    }
  } else {
    LOG_FATAL << "Expected input as tuple, get " << call->args[0];
  }
  return InferLayoutOutput(input_layouts, {output_layout}, Attrs());
}

// nn ops
TVM_REGISTER_OP("relax.nn.avg_pool2d")
    .set_attr<FRelaxInferLayout>("FMSCBackwardInferLayout", MSCInferLayoutPool2d);
TVM_REGISTER_OP("relax.nn.adaptive_avg_pool2d")
    .set_attr<FRelaxInferLayout>("FMSCBackwardInferLayout", MSCInferLayoutPool2d);
TVM_REGISTER_OP("relax.nn.batch_norm")
    .set_attr<FRelaxInferLayout>("FMSCBackwardInferLayout", BackwardInferLayoutBatchNorm);
TVM_REGISTER_OP("relax.nn.conv1d")
    .set_attr<FRelaxInferLayout>("FMSCBackwardInferLayout", MSCInferLayoutConv);
TVM_REGISTER_OP("relax.nn.conv2d")
    .set_attr<FRelaxInferLayout>("FMSCBackwardInferLayout", MSCInferLayoutConv);
TVM_REGISTER_OP("relax.nn.conv2d_transpose")
    .set_attr<FRelaxInferLayout>("FMSCBackwardInferLayout", MSCInferLayoutConv);
TVM_REGISTER_OP("relax.nn.group_norm")
    .set_attr<FRelaxInferLayout>("FMSCBackwardInferLayout", BackwardInferLayoutNormalize);
TVM_REGISTER_OP("relax.nn.layer_norm")
    .set_attr<FRelaxInferLayout>("FMSCBackwardInferLayout", BackwardInferLayoutNormalize);
TVM_REGISTER_OP("relax.nn.max_pool2d")
    .set_attr<FRelaxInferLayout>("FMSCBackwardInferLayout", MSCInferLayoutPool2d);

// reduce axis ops
TVM_REGISTER_OP("relax.argmax")
    .set_attr<FRelaxInferLayout>("FMSCBackwardInferLayout", BackwardInferLayoutArgMaxMin);
TVM_REGISTER_OP("relax.argmin")
    .set_attr<FRelaxInferLayout>("FMSCBackwardInferLayout", BackwardInferLayoutArgMaxMin);
TVM_REGISTER_OP("relax.max")
    .set_attr<FRelaxInferLayout>("FMSCBackwardInferLayout", BackwardInferLayoutReduceAxis);
TVM_REGISTER_OP("relax.min")
    .set_attr<FRelaxInferLayout>("FMSCBackwardInferLayout", BackwardInferLayoutReduceAxis);
TVM_REGISTER_OP("relax.mean")
    .set_attr<FRelaxInferLayout>("FMSCBackwardInferLayout", BackwardInferLayoutReduceAxis);
TVM_REGISTER_OP("relax.sum")
    .set_attr<FRelaxInferLayout>("FMSCBackwardInferLayout", BackwardInferLayoutReduceAxis);
TVM_REGISTER_OP("relax.prod")
    .set_attr<FRelaxInferLayout>("FMSCBackwardInferLayout", BackwardInferLayoutReduceAxis);
TVM_REGISTER_OP("relax.std")
    .set_attr<FRelaxInferLayout>("FMSCBackwardInferLayout", BackwardInferLayoutReduceAxis);

// binary ops
TVM_REGISTER_OP("relax.add")
    .set_attr<FRelaxInferLayout>("FMSCBackwardInferLayout", BackwardInferLayoutBinary);
TVM_REGISTER_OP("relax.divide")
    .set_attr<FRelaxInferLayout>("FMSCBackwardInferLayout", BackwardInferLayoutBinary);
TVM_REGISTER_OP("relax.floor_divide")
    .set_attr<FRelaxInferLayout>("FMSCBackwardInferLayout", BackwardInferLayoutBinary);
TVM_REGISTER_OP("relax.multiply")
    .set_attr<FRelaxInferLayout>("FMSCBackwardInferLayout", BackwardInferLayoutBinary);
TVM_REGISTER_OP("relax.power")
    .set_attr<FRelaxInferLayout>("FMSCBackwardInferLayout", BackwardInferLayoutBinary);
TVM_REGISTER_OP("relax.subtract")
    .set_attr<FRelaxInferLayout>("FMSCBackwardInferLayout", BackwardInferLayoutBinary);
TVM_REGISTER_OP("relax.equal")
    .set_attr<FRelaxInferLayout>("FMSCBackwardInferLayout", BackwardInferLayoutBinary);
TVM_REGISTER_OP("relax.greater")
    .set_attr<FRelaxInferLayout>("FMSCBackwardInferLayout", BackwardInferLayoutBinary);
TVM_REGISTER_OP("relax.greater_equal")
    .set_attr<FRelaxInferLayout>("FMSCBackwardInferLayout", BackwardInferLayoutBinary);
TVM_REGISTER_OP("relax.less")
    .set_attr<FRelaxInferLayout>("FMSCBackwardInferLayout", BackwardInferLayoutBinary);
TVM_REGISTER_OP("relax.less_equal")
    .set_attr<FRelaxInferLayout>("FMSCBackwardInferLayout", BackwardInferLayoutBinary);
TVM_REGISTER_OP("relax.not_equal")
    .set_attr<FRelaxInferLayout>("FMSCBackwardInferLayout", BackwardInferLayoutBinary);
TVM_REGISTER_OP("relax.maximum")
    .set_attr<FRelaxInferLayout>("FMSCBackwardInferLayout", BackwardInferLayoutBinary);
TVM_REGISTER_OP("relax.minimum")
    .set_attr<FRelaxInferLayout>("FMSCBackwardInferLayout", BackwardInferLayoutBinary);
TVM_REGISTER_OP("relax.logical_and")
    .set_attr<FRelaxInferLayout>("FMSCBackwardInferLayout", BackwardInferLayoutBinary);
TVM_REGISTER_OP("relax.logical_or")
    .set_attr<FRelaxInferLayout>("FMSCBackwardInferLayout", BackwardInferLayoutBinary);
TVM_REGISTER_OP("relax.logical_xor")
    .set_attr<FRelaxInferLayout>("FMSCBackwardInferLayout", BackwardInferLayoutBinary);
TVM_REGISTER_OP("relax.bitwise_and")
    .set_attr<FRelaxInferLayout>("FMSCBackwardInferLayout", BackwardInferLayoutBinary);
TVM_REGISTER_OP("relax.bitwise_or")
    .set_attr<FRelaxInferLayout>("FMSCBackwardInferLayout", BackwardInferLayoutBinary);
TVM_REGISTER_OP("relax.bitwise_xor")
    .set_attr<FRelaxInferLayout>("FMSCBackwardInferLayout", BackwardInferLayoutBinary);

// math ops
TVM_REGISTER_OP("relax.concat")
    .set_attr<FRelaxInferLayout>("FMSCBackwardInferLayout", BackwardInferLayoutTupleInputs);
TVM_REGISTER_OP("relax.expand_dims")
    .set_attr<FRelaxInferLayout>("FMSCBackwardInferLayout", BackwardInferLayoutExpandDims);
TVM_REGISTER_OP("relax.matmul")
    .set_attr<FRelaxInferLayout>("FMSCBackwardInferLayout", BackwardInferLayoutMatmul);
TVM_REGISTER_OP("relax.permute_dims")
    .set_attr<FRelaxInferLayout>("FMSCBackwardInferLayout", BackwardInferLayoutPermute);
TVM_REGISTER_OP("relax.reshape")
    .set_attr<FRelaxInferLayout>("FMSCBackwardInferLayout", BackwardInferLayoutReshape);
TVM_REGISTER_OP("relax.squeeze")
    .set_attr<FRelaxInferLayout>("FMSCBackwardInferLayout", BackwardInferLayoutSqueeze);
TVM_REGISTER_OP("relax.take")
    .set_attr<FRelaxInferLayout>("FMSCBackwardInferLayout", BackwardInferLayoutTake);
TVM_REGISTER_OP("relax.image.resize2d")
    .set_attr<FRelaxInferLayout>("FMSCBackwardInferLayout", MSCInferLayoutResize2d);

class LayoutInfer : public ExprVisitor {
 public:
  explicit LayoutInfer(const IRModule& ref_module) : ref_module_(ref_module) { Reset(); }

  void Reset() {
    infered_ = false;
    var_map_.clear();
    ordered_exprs_.clear();
  }

  void RecordExpr(const Var& var, const Expr& expr) {
    var_map_.Set(var, expr);
    ordered_exprs_.push_back(expr);
  }

  Expr Infer(const Expr& expr) {
    Reset();
    ForwardInfer(expr);
    BackwardInfer();
    return expr;
  }

  void ForwardInfer(const Expr& expr) { ExprVisitor::VisitExpr(expr); }

  void BackwardInfer() {
    for (size_t e_idx = ordered_exprs_.size(); e_idx > 0; e_idx--) {
      const Expr& expr = ordered_exprs_[e_idx - 1];
      if (expr->IsInstance<TupleNode>()) {
        continue;
      }
      if (expr->IsInstance<TupleGetItemNode>()) {
        continue;
      }
      if (expr->IsInstance<ShapeExprNode>()) {
        continue;
      }
      if (!expr->IsInstance<CallNode>()) {
        continue;
      }
      const Call& call = Downcast<Call>(expr);
      if (const auto* v_node = call->op.as<GlobalVarNode>()) {
        const auto& func = Downcast<Function>(ref_module_->Lookup(v_node->name_hint));
        BackwardInferFunc(func, call);
        continue;
      } else if (call->op->IsInstance<VarNode>() && local_funcs_.count(call->op)) {
        BackwardInferFunc(local_funcs_[call->op], call);
        continue;
      }
      size_t infered_num = 0;
      for (const auto& arg : call->args) {
        if (IsArgInfered(arg)) {
          infered_num++;
        }
      }
      if (call->args.size() == 0 || infered_num == call->args.size() ||
          !call->op->IsInstance<OpNode>() || LayoutUtils::HasUnknownDimTensor(call->args)) {
        continue;
      }
      const OpNode* op_node = call->op.as<OpNode>();
      if (op_node == nullptr) {
        continue;
      }
      // Infer by op_node
      Op op = Downcast<Op>(GetRef<Op>(op_node));
      InferLayoutOutput infered_layout;
      const auto& msc_infer_map = Op::GetAttrMap<FRelaxInferLayout>("FMSCBackwardInferLayout");
      try {
        if (msc_infer_map.count(op)) {
          FRelaxInferLayout f = msc_infer_map[op];
          infered_layout = f(call, Map<String, Array<String>>(), var_layout_map_);
        } else {
          infered_layout =
              BackwardInferLayoutCommon(call, Map<String, Array<String>>(), var_layout_map_);
        }
      } catch (runtime::InternalError& err) {
        LOG(WARNING) << "Failed to backward infer layout " << expr << " : " << err.message();
        infered_layout = InferLayoutOutput();
      }
      try {
        if (infered_layout.defined()) {
          SetInputLayouts(call, infered_layout->input_layouts);
        }
      } catch (runtime::InternalError& err) {
        LOG(WARNING) << "Failed to backward set inputs layout for " << call << " : "
                     << err.message();
      }
    }
  }

  void VisitBinding_(const VarBindingNode* binding, const CallNode* call_node) final {
    ExprVisitor::VisitBinding_(binding, call_node);
    const auto& call = GetRef<Call>(call_node);
    if (const auto* v_node = call->op.as<GlobalVarNode>()) {
      const auto& func = Downcast<Function>(ref_module_->Lookup(v_node->name_hint));
      RecordExpr(binding->var, call);
      ForwardInferFunc(func, call, binding->var);
    } else if (call->op->IsInstance<VarNode>() && local_funcs_.count(call->op)) {
      RecordExpr(binding->var, call);
      ForwardInferFunc(local_funcs_[call->op], call, binding->var);
    } else {
      // infer call
      bool infer_outputs = true;
      RecordExpr(binding->var, call);
      if (LayoutUtils::LayoutInfered(call)) {
        infer_outputs = false;
      }
      if (call->args.size() == 0 || !call->op->IsInstance<OpNode>() ||
          LayoutUtils::HasUnknownDimTensor(call->args)) {
        infer_outputs = false;
      }
      const OpNode* op_node = call->op.as<OpNode>();
      if (op_node == nullptr) {
        infer_outputs = false;
      }
      if (infer_outputs) {
        // infer layouts
        Op op = Downcast<Op>(GetRef<Op>(op_node));
        InferLayoutOutput infered_layout;
        const auto& msc_infer_map = Op::GetAttrMap<FRelaxInferLayout>("FMSCForwardInferLayout");
        const auto& relax_infer_map = Op::GetAttrMap<FRelaxInferLayout>("FRelaxInferLayout");
        bool set_inputs = true;
        try {
          if (msc_infer_map.count(op)) {
            FRelaxInferLayout f = msc_infer_map[op];
            infered_layout = f(call, Map<String, Array<String>>(), var_layout_map_);
          } else if (!relax_infer_map.count(op)) {
            infered_layout =
                ForwardInferLayoutCommon(call, Map<String, Array<String>>(), var_layout_map_);
          }
          if (relax_infer_map.count(op) && !infered_layout.defined()) {
            FRelaxInferLayout f = relax_infer_map[op];
            infered_layout = f(call, Map<String, Array<String>>(), var_layout_map_);
            set_inputs = false;
          }
        } catch (runtime::InternalError& err) {
          LOG(WARNING) << "Failed to forward infer layout for " << binding->var << " : "
                       << binding->value << ", reason: " << err.message();
          infered_layout = InferLayoutOutput();
        }
        if (infered_layout.defined() && infered_layout->output_layouts.size() == 1) {
          try {
            SetExprLayout(binding->var, infered_layout->output_layouts[0]);
          } catch (runtime::InternalError& err) {
            LOG(WARNING) << "Failed to forward set output layout for " << binding->var << " : "
                         << binding->value << ", reason: " << err.message();
          }
        }
        if (set_inputs && infered_layout.defined()) {
          try {
            SetInputLayouts(call, infered_layout->input_layouts);
          } catch (runtime::InternalError& err) {
            LOG(WARNING) << "Failed to forward set inputs layout for " << call << " : "
                         << err.message();
          }
        }
      }
    }
  }

  void VisitBinding_(const VarBindingNode* binding, const FunctionNode* val) final {
    local_funcs_.Set(binding->var, GetRef<Function>(val));
  }

  void VisitBinding_(const VarBindingNode* binding, const TupleNode* val) final {
    ExprVisitor::VisitBinding_(binding, val);
    RecordExpr(binding->var, GetRef<Tuple>(val));
    if (IsNestedTensor(binding->var)) {
      Array<NLayout> input_layouts;
      for (const auto& field : val->fields) {
        input_layouts.push_back(LayoutUtils::InferLayoutDecision(field, var_layout_map_));
      }
      SetExprLayout(binding->var, input_layouts);
    }
  }

  void VisitBinding_(const VarBindingNode* binding, const TupleGetItemNode* val) final {
    ExprVisitor::VisitBinding_(binding, val);
    RecordExpr(binding->var, GetRef<TupleGetItem>(val));
    const auto& out_layout = LayoutUtils::InferLayoutDecisionAt(GetRef<TupleGetItem>(val)->tuple,
                                                                var_layout_map_, val->index);
    SetExprLayout(binding->var, out_layout);
  }

  void VisitBinding_(const VarBindingNode* binding, const ShapeExprNode* val) final {
    ExprVisitor::VisitBinding_(binding, val);
    RecordExpr(binding->var, GetRef<ShapeExpr>(val));
    SetExprLayout(binding->var, LayoutDecision("O"));
  }

  bool infered() { return infered_; }

 private:
  bool IsArgInfered(const Expr& arg) {
    if (arg->IsInstance<VarNode>() && var_map_.count(Downcast<Var>(arg))) {
      if (LayoutUtils::LayoutInfered(var_map_[Downcast<Var>(arg)]) > 0) {
        return true;
      }
    } else if (const auto* tuple_node = arg.as<TupleNode>()) {
      for (const auto& field : tuple_node->fields) {
        if (!IsArgInfered(field)) {
          return false;
        }
      }
      return true;
    } else if (LayoutUtils::LayoutInfered(arg)) {
      return true;
    }
    return false;
  }

  void SetExprLayout(const Expr& expr, const NLayout& layout) {
    if (expr->IsInstance<VarNode>()) {
      const auto& var = Downcast<Var>(expr);
      var_layout_map_[var] = layout;
      if (LayoutUtils::SetLayout(var, layout)) {
        infered_ = true;
      }
      if (var_map_.count(var) && LayoutUtils::SetLayout(var_map_[var], layout)) {
        infered_ = true;
      }
    } else if (LayoutUtils::SetLayout(expr, layout)) {
      infered_ = true;
    }
  }

  void SetInputLayouts(const Call& call, const Array<NLayout>& input_layouts) {
    if (input_layouts.size() == call->args.size()) {
      for (size_t i = 0; i < input_layouts.size(); i++) {
        SetExprLayout(call->args[i], input_layouts[i]);
      }
    }
  }

  void ForwardInferFunc(const Function& func, const Call& call, const Var& ret) {
    for (size_t i = 0; i < call->args.size(); i++) {
      if (call->args[i]->IsInstance<VarNode>() &&
          var_layout_map_.count(Downcast<Var>(call->args[i]))) {
        SetExprLayout(func->params[i], var_layout_map_[Downcast<Var>(call->args[i])]);
      }
    }
    ForwardInfer(func);
    for (size_t i = 0; i < func->params.size(); i++) {
      if (var_layout_map_.count(func->params[i])) {
        SetExprLayout(call->args[i], var_layout_map_[func->params[i]]);
      }
    }
    if (const auto* b_node = func->body.as<relax::SeqExprNode>()) {
      if (b_node->body->IsInstance<VarNode>() &&
          var_layout_map_.count(Downcast<Var>(b_node->body))) {
        SetExprLayout(ret, var_layout_map_[Downcast<Var>(b_node->body)]);
      }
    } else {
      LOG(FATAL) << "Function body should be SeqExpr, get " << func->body;
    }
  }

  void BackwardInferFunc(const Function& func, const Call& call) {
    for (size_t i = 0; i < func->params.size(); i++) {
      if (var_layout_map_.count(func->params[i])) {
        const auto& param_layout = var_layout_map_[func->params[i]];
        SetExprLayout(call->args[i], param_layout);
        if (call->args[i]->IsInstance<VarNode>() && var_map_.count(Downcast<Var>(call->args[i]))) {
          const auto& producer = var_map_[Downcast<Var>(call->args[i])];
          if (producer->IsInstance<CallNode>() &&
              local_funcs_.count(Downcast<Call>(producer)->op)) {
            const auto& caller = local_funcs_[Downcast<Call>(producer)->op];
            if (const auto* b_node = caller->body.as<relax::SeqExprNode>()) {
              if (b_node->body->IsInstance<VarNode>() &&
                  var_map_.count(Downcast<Var>(b_node->body))) {
                SetExprLayout(b_node->body, param_layout);
              }
            } else {
              LOG(FATAL) << "Caller body should be SeqExpr, get " << caller->body;
            }
          }
        }
      }
    }
  }

  IRModule ref_module_;
  bool infered_;
  Map<Var, Expr> var_map_;
  Array<Expr> ordered_exprs_;
  std::unordered_map<Var, NLayout, ObjectPtrHash, ObjectPtrEqual> var_layout_map_;
  Map<Expr, Function> local_funcs_;
};  // class LayoutInfer

class LayoutChecker : public ExprVisitor {
 public:
  LayoutChecker() { missing_num_ = 0; }

  void Check(const Expr& expr) {
    ExprVisitor::VisitExpr(expr);
    ICHECK_EQ(missing_num_, 0) << "Some layout is missing";
  }

  void VisitExpr_(const CallNode* call) final {
    ExprVisitor::VisitExpr_(call);
    if (!LayoutUtils::LayoutInfered(GetRef<Call>(call))) {
      missing_num_++;
    }
  }

  void VisitExpr_(const ConstantNode* cn) final {
    ExprVisitor::VisitExpr_(cn);
    if (!LayoutUtils::LayoutInfered(GetRef<Constant>(cn))) {
      missing_num_++;
    }
  }

 private:
  size_t missing_num_;
};  // class LayoutChecker

void SetExprLayout(const IRModule& ref_module, const Expr& func, bool allow_missing) {
  auto layout_infer = LayoutInfer(ref_module);
  auto new_func = layout_infer.Infer(func);
  if (!allow_missing) {
    LayoutChecker().Check(new_func);
  }
}

namespace transform {

Pass SetExprLayout(bool allow_missing, const String& entry_name) {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func = [=](IRModule m,
                                                                            PassContext pc) {
    relax::SetExprLayout(m, m->Lookup(entry_name), allow_missing);
    return m;
  };
  return CreateModulePass(pass_func, 0, "SetExprLayout", {});
}

TVM_REGISTER_GLOBAL("relax.transform.SetExprLayout").set_body_typed(SetExprLayout);

}  // namespace transform
}  // namespace relax
}  // namespace tvm
