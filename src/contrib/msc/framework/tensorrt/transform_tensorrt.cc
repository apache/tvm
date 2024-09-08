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
 * \file src/contrib/msc/framework/tensorrt/transform_tensorrt.cc
 * \brief Pass for transform the function to tensorrt.
 */

#include <tvm/relax/attrs/sorting.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/transform.h>

#include "../../../../relax/transform/utils.h"
#include "../../../../support/scalars.h"
#include "../../core/transform/rewrite_utils.h"
#include "../../core/utils.h"

namespace tvm {
namespace relax {
using namespace tvm::contrib::msc;

struct TensorRTTransConfig {
  // Whether to cast linear to conv
  bool linear_to_conv{true};
  std::vector<size_t> version{0, 0, 0};

  void Load(dmlc::JSONReader* reader) {
    std::string key;
    reader->BeginObject();
    while (reader->NextObjectItem(&key)) {
      if (key == "linear_to_conv") {
        reader->Read(&linear_to_conv);
      } else if (key == "version") {
        reader->Read(&version);
      } else {
        LOG(FATAL) << "Do not support key " << key;
      }
    }
  }
};

const TensorRTTransConfig ParseConfig(const String& config_str) {
  TensorRTTransConfig config;
  if (config_str.size() > 0) {
    std::istringstream is(config_str);
    dmlc::JSONReader reader(&is);
    reader.Read(&config);
  }
  return config;
}

using FRewriteTensorRT =
    runtime::TypedPackedFunc<Expr(BlockBuilder builder, const Var& var, const Call& src_call,
                                  const Map<Expr, Call>& new_calls, const String& config)>;

const Array<PrimExpr> BroadcastShape(const Array<PrimExpr>& src_shape,
                                     const Array<PrimExpr>& out_shape) {
  size_t diff = out_shape.size() - src_shape.size();
  Array<PrimExpr> leading_shape, tailing_shape;
  for (size_t i = 0; i < diff; i++) {
    leading_shape.push_back(Integer(1));
  }
  for (const auto& s : src_shape) {
    tailing_shape.push_back(s);
    leading_shape.push_back(s);
  }
  for (size_t i = 0; i < diff; i++) {
    tailing_shape.push_back(Integer(1));
  }
  if (ArrayUtils::Broadcastable(tailing_shape, out_shape)) {
    return tailing_shape;
  }
  ICHECK(ArrayUtils::Broadcastable(leading_shape, out_shape))
      << "Only support elemwise ops with leading or tailing expand";
  return leading_shape;
}

Expr RewriteElemwise(BlockBuilder builder, const Var& var, const Call& src_call,
                     const Map<Expr, Call>& new_calls, const String& config) {
  const auto& call = new_calls.count(src_call) ? new_calls[src_call] : src_call;
  const auto& shape_a = ExprUtils::GetShape(call->args[0]);
  const auto& shape_b = ExprUtils::GetShape(call->args[1]);
  const auto& shape_out = ExprUtils::GetShape(var);
  static const Op& reshape_op = Op::Get("relax.reshape");
  if (shape_a.size() > shape_b.size()) {
    const auto& exp_shape = BroadcastShape(shape_b, shape_out);
    const auto& expand_b =
        RewriteUtils::MakeCall(builder, ExprUtils::GetSpanName(call, "expand_b"), reshape_op,
                               {call->args[1], ShapeExpr(exp_shape)});
    return Call(call->op, {call->args[0], expand_b}, call->attrs, call->sinfo_args, call->span);
  } else if (shape_a.size() < shape_b.size()) {
    const auto& exp_shape = BroadcastShape(shape_a, shape_out);
    const auto& expand_a =
        RewriteUtils::MakeCall(builder, ExprUtils::GetSpanName(call, "expand_a"), reshape_op,
                               {call->args[0], ShapeExpr(exp_shape)});
    return Call(call->op, {expand_a, call->args[1]}, call->attrs, call->sinfo_args, call->span);
  }
  return call;
}

Expr RewriteAdd(BlockBuilder builder, const Var& var, const Call& src_call,
                const Map<Expr, Call>& new_calls, const String& config) {
  const auto& call = new_calls.count(src_call) ? new_calls[src_call] : src_call;
  if (new_calls.count(call->args[0]) &&
      new_calls[call->args[0]]->op == Op::Get("relax.nn.conv1d")) {
    const auto& reshape = Downcast<Call>(builder->LookupBinding(Downcast<Var>(call->args[0])));
    if (reshape->op != Op::Get("relax.reshape")) {
      return call;
    }
    const auto& conv2d = Downcast<Call>(builder->LookupBinding(Downcast<Var>(reshape->args[0])));
    if (conv2d->op != Op::Get("relax.nn.conv2d")) {
      return call;
    }
    const auto& input_shape = ExprUtils::GetShape(call->args[0]);
    const auto& bias_shape = ExprUtils::GetShape(call->args[1]);
    const auto* conv_attrs = conv2d->attrs.as<Conv2DAttrs>();
    if (conv_attrs->data_layout == "NCHW") {
      // expand bias reshape
      Array<PrimExpr> exp_bias_shape{bias_shape[0], bias_shape[1], Integer(1), bias_shape[2]};
      static const Op& reshape_op = Op::Get("relax.reshape");
      const auto& exp_bias =
          RewriteUtils::MakeCall(builder, ExprUtils::GetSpanName(call, "exp_bias"), reshape_op,
                                 {call->args[1], ShapeExpr(exp_bias_shape)});
      // redirect to conv2d
      static const Op& add_op = Op::Get("relax.add");
      const auto& exp_add = RewriteUtils::MakeCall(builder, ExprUtils::GetSpanName(call, "exp_add"),
                                                   add_op, {reshape->args[0], exp_bias});
      // reduce output
      return Call(reshape_op, {exp_add, ShapeExpr(input_shape)}, Attrs(), call->sinfo_args,
                  call->span);
    } else {
      LOG_FATAL << "Unexpected data layout " << conv_attrs->data_layout;
    }
  }
  return RewriteElemwise(builder, var, call, new_calls, config);
}

Expr RewriteArgmaxmin(BlockBuilder builder, const Var& var, const Call& src_call,
                      const Map<Expr, Call>& new_calls, const String& config) {
  const auto& call = new_calls.count(src_call) ? new_calls[src_call] : src_call;
  const auto& out_dtype = ExprUtils::GetDataType(var);
  const auto* src_attrs = src_call->attrs.as<ArgmaxArgminAttrs>();
  ICHECK(out_dtype == DataType::Int(32) || out_dtype == DataType::Int(64))
      << "Unexpected out dtype " << out_dtype;
  static const Op& topk_op = Op::Get("relax.topk");
  auto topk_attrs = make_object<TopKAttrs>();
  topk_attrs->k = 1;
  if (src_attrs->axis.defined()) {
    topk_attrs->axis = src_attrs->axis.value()->value;
  }
  topk_attrs->largest = call->op == Op::Get("relax.argmax");
  topk_attrs->ret_type = "both";
  topk_attrs->dtype = out_dtype;
  // change to topk
  const auto& topk = RewriteUtils::MakeCall(builder, ExprUtils::GetSpanName(call, "topk"), topk_op,
                                            {call->args[0]}, Attrs(topk_attrs));
  const auto& get_name = ExprUtils::GetSpanName(call, ".1");
  const auto& get_item =
      TupleGetItem(topk, 1, SpanUtils::CreateWithAttr(msc_attr::kName, get_name));
  if (src_attrs->keepdims) {
    return get_item;
  }
  const auto& get_item_var = builder->Emit(get_item, get_name);
  static const Op& reshape_op = Op::Get("relax.reshape");
  const auto& output_shape = ExprUtils::GetShape(var);
  return Call(reshape_op, {get_item_var, ShapeExpr(output_shape)}, Attrs(), call->sinfo_args,
              call->span);
}

Expr RewriteAttention(BlockBuilder builder, const Var& var, const Call& src_call,
                      const Map<Expr, Call>& new_calls, const String& config) {
  const auto& call = new_calls.count(src_call) ? new_calls[src_call] : src_call;
  const auto& in_dtype = ExprUtils::GetDataType(call->args[0]);
  const auto* src_attrs = src_call->attrs.as<AttentionAttrs>();

  // define dims
  const auto& in_q_shape = ExprUtils::GetShape(call->args[0]);
  const auto& in_v_shape = ExprUtils::GetShape(call->args[2]);
  const auto& batch_size = in_q_shape[0];
  const auto& seq_len = in_q_shape[1];
  const auto& num_head = in_q_shape[2];
  const auto& head_dim = in_q_shape[3];
  const auto& seq_len_kv = in_v_shape[1];
  const auto& head_dim_v = in_v_shape[3];

  // create ops
  static const Op& permute_dims_op = Op::Get("relax.permute_dims");
  static const Op& reshape_op = Op::Get("relax.reshape");
  static const Op& matmul_op = Op::Get("relax.matmul");
  static const Op& multiply_op = Op::Get("relax.multiply");
  static const Op& add_op = Op::Get("relax.add");
  static const Op& divide_op = Op::Get("relax.divide");
  static const Op& sqrt_op = Op::Get("relax.sqrt");
  static const Op& softmax_op = Op::Get("relax.nn.softmax");
  static const Op& tril_op = Op::Get("relax.tril");
  static const Op& max_op = Op::Get("relax.max");
  static const Op& sum_op = Op::Get("relax.sum");
  static const Op& subtract_op = Op::Get("relax.subtract");
  static const Op& exp_op = Op::Get("relax.exp");

  // prepare q,k,v
  auto permute_attrs = make_object<PermuteDimsAttrs>();
  Array<Integer> axes{Integer(0), Integer(2), Integer(1), Integer(3)};
  permute_attrs->axes = axes;
  const auto& q_trans =
      RewriteUtils::MakeCall(builder, ExprUtils::GetSpanName(call, "q_trans"), permute_dims_op,
                             {call->args[0]}, Attrs(permute_attrs));
  const auto& k_trans =
      RewriteUtils::MakeCall(builder, ExprUtils::GetSpanName(call, "k_trans"), permute_dims_op,
                             {call->args[1]}, Attrs(permute_attrs));
  const auto& v_trans =
      RewriteUtils::MakeCall(builder, ExprUtils::GetSpanName(call, "v_trans"), permute_dims_op,
                             {call->args[2]}, Attrs(permute_attrs));
  Array<PrimExpr> q_shape({batch_size * num_head, seq_len, head_dim});
  const auto& q_reshape = RewriteUtils::MakeCall(builder, ExprUtils::GetSpanName(call, "q_reshape"),
                                                 reshape_op, {q_trans, ShapeExpr(q_shape)});
  Array<PrimExpr> k_shape({batch_size * num_head, seq_len_kv, head_dim});
  const auto& k_reshape = RewriteUtils::MakeCall(builder, ExprUtils::GetSpanName(call, "k_reshape"),
                                                 reshape_op, {k_trans, ShapeExpr(k_shape)});
  Array<PrimExpr> v_shape({batch_size * num_head, seq_len_kv, head_dim_v});
  const auto& v_reshape = RewriteUtils::MakeCall(builder, ExprUtils::GetSpanName(call, "v_reshape"),
                                                 reshape_op, {v_trans, ShapeExpr(v_shape)});
  auto reduce_permute_attrs = make_object<PermuteDimsAttrs>();
  Array<Integer> v_axes{Integer(0), Integer(2), Integer(1)};
  reduce_permute_attrs->axes = v_axes;
  // transpose for batch_matmul
  const auto& k_reshape_trans =
      RewriteUtils::MakeCall(builder, ExprUtils::GetSpanName(call, "k_reshape_trans"),
                             permute_dims_op, {k_reshape}, Attrs(reduce_permute_attrs));

  // calculate product
  auto matmul_attrs = make_object<MatmulAttrs>();
  matmul_attrs->out_dtype = in_dtype;
  const auto& qk_prod =
      RewriteUtils::MakeCall(builder, ExprUtils::GetSpanName(call, "qk_prod"), matmul_op,
                             {q_reshape, k_reshape_trans}, Attrs(matmul_attrs));
  Expr p_scale;
  if (src_attrs->scale.defined()) {
    double value = static_cast<double>(src_attrs->scale.value()->value);
    const auto& scale = RewriteUtils::MakeConstant(builder, ExprUtils::GetSpanName(call, "scale"),
                                                   value, in_dtype, 3);
    p_scale = RewriteUtils::MakeCall(builder, ExprUtils::GetSpanName(call, "p_scale"), multiply_op,
                                     {qk_prod, scale});
  } else {
    double value = static_cast<double>(Downcast<Integer>(head_dim)->value);
    const auto& scale = RewriteUtils::MakeConstant(builder, ExprUtils::GetSpanName(call, "scale"),
                                                   value, in_dtype, 3);
    const auto& sqrt_scale = RewriteUtils::MakeCall(
        builder, ExprUtils::GetSpanName(call, "sqrt_scale"), sqrt_op, {scale});
    p_scale = RewriteUtils::MakeCall(builder, ExprUtils::GetSpanName(call, "p_scale"), divide_op,
                                     {qk_prod, sqrt_scale});
  }

  // bias
  Expr prod = p_scale;
  if (call->args.size() == 4) {
    Array<PrimExpr> exp_shape{batch_size, num_head, seq_len, seq_len_kv};
    Array<PrimExpr> reduce_shape{batch_size * num_head, seq_len, seq_len_kv};
    const auto& prod_exp = RewriteUtils::MakeCall(builder, ExprUtils::GetSpanName(call, "prod_exp"),
                                                  reshape_op, {prod, ShapeExpr(exp_shape)});
    const auto& prod_add = RewriteUtils::MakeCall(builder, ExprUtils::GetSpanName(call, "prod_add"),
                                                  add_op, {prod_exp, call->args[3]});
    prod = RewriteUtils::MakeCall(builder, ExprUtils::GetSpanName(call, "prod_reduce"), reshape_op,
                                  {prod_add, ShapeExpr(reduce_shape)});
  }

  // causal_mask
  Expr s_value;
  if (!src_attrs->causal_mask.defined()) {
    auto softmax_attrs = make_object<SoftmaxAttrs>();
    softmax_attrs->axis = 2;
    s_value = RewriteUtils::MakeCall(builder, ExprUtils::GetSpanName(call, "act"), softmax_op,
                                     {prod}, Attrs(softmax_attrs));
  } else {
    const auto& causal_mask = src_attrs->causal_mask.value();
    PrimValue tril_k;
    if (causal_mask == "TopLeft") {
      tril_k = PrimValue(Integer(0));
    } else if (causal_mask == "BottomRight") {
      tril_k = PrimValue(seq_len - seq_len_kv);
    } else {
      LOG_FATAL << "Unexpected causal_mask " << causal_mask;
    }
    const auto& p_masked = RewriteUtils::MakeCall(builder, ExprUtils::GetSpanName(call, "p_masked"),
                                                  tril_op, {prod, tril_k});
    auto reduce_attrs = make_object<StatisticalAttrs>();
    Array<Integer> axis{Integer(2)};
    reduce_attrs->axis = axis;
    reduce_attrs->keepdims = true;
    const auto& p_max = RewriteUtils::MakeCall(builder, ExprUtils::GetSpanName(call, "p_max"),
                                               max_op, {prod}, Attrs(reduce_attrs));
    const auto& p_diff = RewriteUtils::MakeCall(builder, ExprUtils::GetSpanName(call, "p_diff"),
                                                subtract_op, {p_masked, p_max});
    const auto& p_exp =
        RewriteUtils::MakeCall(builder, ExprUtils::GetSpanName(call, "p_exp"), exp_op, {p_diff});
    const auto& p_masked_exp = RewriteUtils::MakeCall(
        builder, ExprUtils::GetSpanName(call, "p_masked_exp"), tril_op, {p_exp, tril_k});
    const auto& p_masked_sum =
        RewriteUtils::MakeCall(builder, ExprUtils::GetSpanName(call, "p_masked_sum"), sum_op,
                               {p_masked_exp}, Attrs(reduce_attrs));
    s_value = RewriteUtils::MakeCall(builder, ExprUtils::GetSpanName(call, "act"), divide_op,
                                     {p_masked_exp, p_masked_sum});
  }

  // final calculation
  const auto& o_prod = RewriteUtils::MakeCall(builder, ExprUtils::GetSpanName(call, "o_prod"),
                                              matmul_op, {s_value, v_reshape}, Attrs(matmul_attrs));
  Array<PrimExpr> o_shape{batch_size, num_head, seq_len, head_dim_v};
  return Call(reshape_op, {o_prod, ShapeExpr(o_shape)}, Attrs(), call->sinfo_args, call->span);
}

Expr RewriteBatchNorm(BlockBuilder builder, const Var& var, const Call& src_call,
                      const Map<Expr, Call>& new_calls, const String& config) {
  const auto& call = new_calls.count(src_call) ? new_calls[src_call] : src_call;
  const auto& input_shape = ExprUtils::GetShape(call->args[0]);
  const auto& in_dtype = ExprUtils::GetDataType(call->args[0]);
  const auto* src_attrs = src_call->attrs.as<BatchNormAttrs>();
  // define expand shape
  Array<PrimExpr> exp_shape(input_shape.size(), Integer(1));
  exp_shape.Set(src_attrs->axis, input_shape[src_attrs->axis]);

  // create eps constant
  const auto& eps = RewriteUtils::MakeConstant(builder, ExprUtils::GetSpanName(call, "eps"),
                                               src_attrs->epsilon, in_dtype);

  // create ops
  static const Op& add_op = Op::Get("relax.add");
  static const Op& divide_op = Op::Get("relax.divide");
  static const Op& multiply_op = Op::Get("relax.multiply");
  static const Op& reshape_op = Op::Get("relax.reshape");
  static const Op& sqrt_op = Op::Get("relax.sqrt");
  static const Op& subtract_op = Op::Get("relax.subtract");

  // scale factor: gamma/sqrt(var + epsilon)
  const auto& eps_add = RewriteUtils::MakeCall(builder, ExprUtils::GetSpanName(call, "eps_add"),
                                               add_op, {call->args[4], eps});
  const auto& sqrt =
      RewriteUtils::MakeCall(builder, ExprUtils::GetSpanName(call, "sqrt"), sqrt_op, {eps_add});
  const auto& scale_factor = RewriteUtils::MakeCall(
      builder, ExprUtils::GetSpanName(call, "scale_factor"), divide_op, {call->args[1], sqrt});
  Expr res = call->args[0];
  // scale
  if (src_attrs->scale) {
    const auto& exp_scale =
        RewriteUtils::MakeCall(builder, ExprUtils::GetSpanName(call, "exp_scale"), reshape_op,
                               {scale_factor, ShapeExpr(exp_shape)});
    res = RewriteUtils::MakeCall(builder, ExprUtils::GetSpanName(call, "scale"), multiply_op,
                                 {res, exp_scale});
  }
  // offset
  if (src_attrs->center) {
    // offset factor: beta-mean*scale_factor
    const auto& average = RewriteUtils::MakeCall(builder, ExprUtils::GetSpanName(call, "average"),
                                                 multiply_op, {call->args[3], scale_factor});
    const auto& offset_factor =
        RewriteUtils::MakeCall(builder, ExprUtils::GetSpanName(call, "offset_factor"), subtract_op,
                               {call->args[2], average});
    const auto& exp_offset =
        RewriteUtils::MakeCall(builder, ExprUtils::GetSpanName(call, "exp_offset"), reshape_op,
                               {offset_factor, ShapeExpr(exp_shape)});
    res = RewriteUtils::MakeCall(builder, ExprUtils::GetSpanName(call, "offset"), add_op,
                                 {res, exp_offset});
  }
  return Tuple(Array<Expr>{res}, call->span);
}

Expr RewriteBroadcastTo(BlockBuilder builder, const Var& var, const Call& src_call,
                        const Map<Expr, Call>& new_calls, const String& config) {
  const auto& call = new_calls.count(src_call) ? new_calls[src_call] : src_call;
  const auto& input_shape = ExprUtils::GetShape(call->args[0]);
  const auto& output_shape = ExprUtils::GetShape(var);
  Expr concat_input = call->args[0];
  static const Op& concat_op = Op::Get("relax.concat");
  for (size_t i = 0; i < input_shape.size(); i++) {
    int64_t in_dim = Downcast<Integer>(input_shape[i])->value;
    int64_t out_dim = Downcast<Integer>(output_shape[i])->value;
    if (in_dim != out_dim) {
      Array<Expr> concat_inputs(out_dim / in_dim, concat_input);
      auto concat_attrs = make_object<ConcatAttrs>();
      concat_attrs->axis = Integer(i);
      concat_input = RewriteUtils::MakeCall(
          builder, ExprUtils::GetSpanName(call, "concat_" + std::to_string(i)), concat_op,
          {Tuple(concat_inputs)}, Attrs(concat_attrs));
    }
  }
  return concat_input;
}

Expr RewriteConv1d(BlockBuilder builder, const Var& var, const Call& src_call,
                   const Map<Expr, Call>& new_calls, const String& config) {
  const auto& call = new_calls.count(src_call) ? new_calls[src_call] : src_call;
  const auto* src_attrs = src_call->attrs.as<Conv1DAttrs>();
  const auto& input_shape = ExprUtils::GetShape(call->args[0]);
  const auto& weight_shape = ExprUtils::GetShape(call->args[1]);
  const auto& output_shape = ExprUtils::GetShape(var);
  if (src_attrs->data_layout == "NCW") {
    Array<Expr> new_args;
    // expand inputs
    Array<PrimExpr> exp_input_shape{input_shape[0], input_shape[1], Integer(1), input_shape[2]};
    Array<PrimExpr> exp_weight_shape{weight_shape[0], weight_shape[1], Integer(1), weight_shape[2]};
    static const Op& reshape_op = Op::Get("relax.reshape");
    new_args.push_back(RewriteUtils::MakeCall(builder, ExprUtils::GetSpanName(call, "exp_input"),
                                              reshape_op,
                                              {call->args[0], ShapeExpr(exp_input_shape)}));
    new_args.push_back(RewriteUtils::MakeCall(builder, ExprUtils::GetSpanName(call, "exp_weight"),
                                              reshape_op,
                                              {call->args[1], ShapeExpr(exp_weight_shape)}));
    // change to conv2d
    static const Op& conv2d_op = Op::Get("relax.nn.conv2d");
    auto conv_attrs = make_object<Conv2DAttrs>();
    conv_attrs->strides = Array<IntImm>{src_attrs->strides[0], Integer(1)};
    conv_attrs->padding =
        Array<IntImm>{Integer(0), src_attrs->padding[0], Integer(0), src_attrs->padding[1]};
    conv_attrs->dilation = Array<IntImm>{src_attrs->dilation[0], Integer(1)};
    conv_attrs->groups = src_attrs->groups;
    conv_attrs->data_layout = "NCHW";
    conv_attrs->kernel_layout = "OIHW";
    conv_attrs->out_layout = "NCHW";
    conv_attrs->out_dtype = src_attrs->out_dtype;
    const auto& conv2d = RewriteUtils::MakeCall(builder, ExprUtils::GetSpanName(call, "exp"),
                                                conv2d_op, new_args, Attrs(conv_attrs));
    // reduce output
    return Call(reshape_op, {conv2d, ShapeExpr(output_shape)}, Attrs(), call->sinfo_args,
                call->span);
  } else {
    LOG_FATAL << "Unexpected data layout " << src_attrs->data_layout;
  }
  return call;
}

Expr RewriteGelu(BlockBuilder builder, const Var& var, const Call& src_call,
                 const Map<Expr, Call>& new_calls, const String& config) {
  // 0.5 * x * (1 + erf(sqrt(0.5) * x))
  const auto& call = new_calls.count(src_call) ? new_calls[src_call] : src_call;
  size_t in_dim = ExprUtils::GetShape(call->args[0]).size();
  const auto& in_dtype = ExprUtils::GetDataType(call->args[0]);
  // create ops
  static const Op& add_op = Op::Get("relax.add");
  static const Op& multiply_op = Op::Get("relax.multiply");
  static const Op& erf_op = Op::Get("relax.erf");

  const auto& factor = RewriteUtils::MakeConstant(builder, ExprUtils::GetSpanName(call, "factor"),
                                                  std::sqrt(0.5), in_dtype, in_dim);
  const auto& mul = RewriteUtils::MakeCall(builder, ExprUtils::GetSpanName(call, "mul"),
                                           multiply_op, {factor, call->args[0]});
  const auto& erf =
      RewriteUtils::MakeCall(builder, ExprUtils::GetSpanName(call, "erf"), erf_op, {mul});
  const auto& one =
      RewriteUtils::MakeConstant(builder, ExprUtils::GetSpanName(call, "one"), 1, in_dtype, in_dim);
  const auto& add =
      RewriteUtils::MakeCall(builder, ExprUtils::GetSpanName(call, "add"), add_op, {one, erf});
  const auto& mul2 = RewriteUtils::MakeCall(builder, ExprUtils::GetSpanName(call, "mul2"),
                                            multiply_op, {call->args[0], add});
  const auto& half = RewriteUtils::MakeConstant(builder, ExprUtils::GetSpanName(call, "one"), 0.5,
                                                in_dtype, in_dim);
  return Call(multiply_op, {half, mul2}, Attrs(), call->sinfo_args, call->span);
}

Expr RewriteGeluTanh(BlockBuilder builder, const Var& var, const Call& src_call,
                     const Map<Expr, Call>& new_calls, const String& config) {
  // 0.5 * x * (1 + tanh(sqrt(2/pi) * (0.044715F * pow(x, 3) + x)))
  const auto& call = new_calls.count(src_call) ? new_calls[src_call] : src_call;
  size_t in_dim = ExprUtils::GetShape(call->args[0]).size();
  const auto& in_dtype = ExprUtils::GetDataType(call->args[0]);

  // create ops
  static const Op& add_op = Op::Get("relax.add");
  static const Op& multiply_op = Op::Get("relax.multiply");
  static const Op& pow_op = Op::Get("relax.power");
  static const Op& tanh_op = Op::Get("relax.tanh");

  const auto& pow_factor = RewriteUtils::MakeConstant(
      builder, ExprUtils::GetSpanName(call, "pow_factor"), 3, in_dtype, in_dim);
  const auto& mul_factor = RewriteUtils::MakeConstant(
      builder, ExprUtils::GetSpanName(call, "mul_factor"), 0.044715, in_dtype, in_dim);
  const auto& pi_factor = RewriteUtils::MakeConstant(
      builder, ExprUtils::GetSpanName(call, "pi_factor"), std::sqrt(2 / M_PI), in_dtype, in_dim);

  const auto& pow = RewriteUtils::MakeCall(builder, ExprUtils::GetSpanName(call, "pow"), pow_op,
                                           {call->args[0], pow_factor});
  const auto& mul = RewriteUtils::MakeCall(builder, ExprUtils::GetSpanName(call, "mul"),
                                           multiply_op, {mul_factor, pow});
  const auto& add = RewriteUtils::MakeCall(builder, ExprUtils::GetSpanName(call, "add"), add_op,
                                           {mul, call->args[0]});
  const auto& mul2 = RewriteUtils::MakeCall(builder, ExprUtils::GetSpanName(call, "mul2"),
                                            multiply_op, {pi_factor, add});
  const auto& tanh =
      RewriteUtils::MakeCall(builder, ExprUtils::GetSpanName(call, "tanh"), tanh_op, {mul2});
  const auto& one =
      RewriteUtils::MakeConstant(builder, ExprUtils::GetSpanName(call, "one"), 1, in_dtype, in_dim);
  const auto& add2 =
      RewriteUtils::MakeCall(builder, ExprUtils::GetSpanName(call, "add"), add_op, {one, tanh});
  const auto& mul3 = RewriteUtils::MakeCall(builder, ExprUtils::GetSpanName(call, "mul3"),
                                            multiply_op, {call->args[0], add2});
  const auto& half = RewriteUtils::MakeConstant(builder, ExprUtils::GetSpanName(call, "one"), 0.5,
                                                in_dtype, in_dim);
  return Call(multiply_op, {half, mul3}, Attrs(), call->sinfo_args, call->span);
}

Expr RewriteGroupNorm(BlockBuilder builder, const Var& var, const Call& src_call,
                      const Map<Expr, Call>& new_calls, const String& config) {
  const auto& call = new_calls.count(src_call) ? new_calls[src_call] : src_call;
  const auto& input_shape = ExprUtils::GetShape(call->args[0]);
  const auto& in_dtype = ExprUtils::GetDataType(call->args[0]);
  const auto* src_attrs = src_call->attrs.as<GroupNormAttrs>();
  Array<PrimExpr> group_shape = input_shape;
  Array<PrimExpr> exp_shape(input_shape.size(), Integer(1));
  size_t axis = CommonUtils::GetIndex(src_attrs->channel_axis, input_shape.size());
  int64_t channel_dim = Downcast<Integer>(input_shape[axis])->value *
                        Downcast<Integer>(input_shape[axis + 1])->value / src_attrs->num_groups;
  group_shape.Set(axis, Integer(src_attrs->num_groups));
  group_shape.Set(axis + 1, Integer(channel_dim));
  exp_shape.Set(axis, Integer(src_attrs->num_groups));

  // create eps constant
  const auto& eps = RewriteUtils::MakeConstant(builder, ExprUtils::GetSpanName(call, "eps"),
                                               src_attrs->epsilon, in_dtype);

  // create ops
  static const Op& add_op = Op::Get("relax.add");
  static const Op& divide_op = Op::Get("relax.divide");
  static const Op& mean_op = Op::Get("relax.mean");
  static const Op& multiply_op = Op::Get("relax.multiply");
  static const Op& square_op = Op::Get("relax.square");
  static const Op& reshape_op = Op::Get("relax.reshape");
  static const Op& sqrt_op = Op::Get("relax.sqrt");
  static const Op& subtract_op = Op::Get("relax.subtract");

  // reshape input
  const auto& reshape_in =
      RewriteUtils::MakeCall(builder, ExprUtils::GetSpanName(call, "reshape_in"), reshape_op,
                             {call->args[0], ShapeExpr(group_shape)});

  // mean(input)
  auto mean_attrs = make_object<StatisticalAttrs>();
  mean_attrs->axis = src_attrs->axes;
  mean_attrs->keepdims = true;
  const auto& mean = RewriteUtils::MakeCall(builder, ExprUtils::GetSpanName(call, "mean"), mean_op,
                                            {reshape_in}, Attrs(mean_attrs));

  // variance: mean((input-mean)*(input-mean))
  const auto& diff = RewriteUtils::MakeCall(builder, ExprUtils::GetSpanName(call, "diff"),
                                            subtract_op, {reshape_in, mean});
  const auto& square =
      RewriteUtils::MakeCall(builder, ExprUtils::GetSpanName(call, "square"), square_op, {diff});
  const auto& variance = RewriteUtils::MakeCall(builder, ExprUtils::GetSpanName(call, "variance"),
                                                mean_op, {square}, Attrs(mean_attrs));

  // sqrt(var + epsilon)
  Array<PrimExpr> exp_eps_shape(input_shape.size(), Integer(1));
  const auto& exp_eps = RewriteUtils::MakeCall(builder, ExprUtils::GetSpanName(call, "exp_eps"),
                                               reshape_op, {eps, ShapeExpr(exp_eps_shape)});
  const auto& eps_add = RewriteUtils::MakeCall(builder, ExprUtils::GetSpanName(call, "eps_add"),
                                               add_op, {variance, exp_eps});
  const auto& sqrt =
      RewriteUtils::MakeCall(builder, ExprUtils::GetSpanName(call, "sqrt"), sqrt_op, {eps_add});

  // diff/sqrt
  Expr res = RewriteUtils::MakeCall(builder, ExprUtils::GetSpanName(call, "divide"), divide_op,
                                    {diff, sqrt});

  // scale
  if (src_attrs->scale) {
    const auto& exp_gamma =
        RewriteUtils::MakeCall(builder, ExprUtils::GetSpanName(call, "exp_gamma"), reshape_op,
                               {call->args[1], ShapeExpr(exp_shape)});
    res = RewriteUtils::MakeCall(builder, ExprUtils::GetSpanName(call, "scale"), multiply_op,
                                 {res, exp_gamma});
  }
  // offset
  if (src_attrs->center) {
    const auto& exp_beta =
        RewriteUtils::MakeCall(builder, ExprUtils::GetSpanName(call, "exp_beta"), reshape_op,
                               {call->args[2], ShapeExpr(exp_shape)});
    res = RewriteUtils::MakeCall(builder, ExprUtils::GetSpanName(call, "offset"), add_op,
                                 {res, exp_beta});
  }
  // reshape output
  return Call(reshape_op, {res, ShapeExpr(input_shape)}, Attrs(), call->sinfo_args, call->span);
}

Expr RewriteLayerNorm(BlockBuilder builder, const Var& var, const Call& src_call,
                      const Map<Expr, Call>& new_calls, const String& config) {
  const auto& call = new_calls.count(src_call) ? new_calls[src_call] : src_call;
  const auto& input_shape = ExprUtils::GetShape(call->args[0]);
  const auto& in_dtype = ExprUtils::GetDataType(call->args[0]);
  const auto* src_attrs = src_call->attrs.as<LayerNormAttrs>();
  Array<PrimExpr> exp_shape(input_shape.size(), Integer(1));
  for (const auto& a : src_attrs->axes) {
    size_t index = CommonUtils::GetIndex(static_cast<int>(a->value), input_shape.size());
    exp_shape.Set(index, input_shape[index]);
  }
  // create eps constant
  const auto& eps = RewriteUtils::MakeConstant(builder, ExprUtils::GetSpanName(call, "eps"),
                                               src_attrs->epsilon, in_dtype);

  // create ops
  static const Op& add_op = Op::Get("relax.add");
  static const Op& divide_op = Op::Get("relax.divide");
  static const Op& mean_op = Op::Get("relax.mean");
  static const Op& multiply_op = Op::Get("relax.multiply");
  static const Op& square_op = Op::Get("relax.square");
  static const Op& reshape_op = Op::Get("relax.reshape");
  static const Op& sqrt_op = Op::Get("relax.sqrt");
  static const Op& subtract_op = Op::Get("relax.subtract");

  // mean(input)
  auto mean_attrs = make_object<StatisticalAttrs>();
  mean_attrs->axis = src_attrs->axes;
  mean_attrs->keepdims = true;
  const auto& mean = RewriteUtils::MakeCall(builder, ExprUtils::GetSpanName(call, "mean"), mean_op,
                                            {call->args[0]}, Attrs(mean_attrs));

  // variance: mean((input-mean)*(input-mean))
  const auto& diff = RewriteUtils::MakeCall(builder, ExprUtils::GetSpanName(call, "diff"),
                                            subtract_op, {call->args[0], mean});
  const auto& square =
      RewriteUtils::MakeCall(builder, ExprUtils::GetSpanName(call, "square"), square_op, {diff});
  const auto& variance = RewriteUtils::MakeCall(builder, ExprUtils::GetSpanName(call, "variance"),
                                                mean_op, {square}, Attrs(mean_attrs));

  // sqrt(var + epsilon)
  Array<PrimExpr> exp_eps_shape(input_shape.size(), Integer(1));
  const auto& exp_eps = RewriteUtils::MakeCall(builder, ExprUtils::GetSpanName(call, "exp_eps"),
                                               reshape_op, {eps, ShapeExpr(exp_eps_shape)});
  const auto& eps_add = RewriteUtils::MakeCall(builder, ExprUtils::GetSpanName(call, "eps_add"),
                                               add_op, {variance, exp_eps});
  const auto& sqrt =
      RewriteUtils::MakeCall(builder, ExprUtils::GetSpanName(call, "sqrt"), sqrt_op, {eps_add});

  // diff/sqrt
  Call res = Call(divide_op, {diff, sqrt}, Attrs(), call->sinfo_args, call->span);

  // scale
  if (src_attrs->scale) {
    const auto& exp_gamma =
        RewriteUtils::MakeCall(builder, ExprUtils::GetSpanName(call, "exp_gamma"), reshape_op,
                               {call->args[1], ShapeExpr(exp_shape)});
    const auto& res_var =
        RewriteUtils::ReEmit(builder, ExprUtils::GetSpanName(call, "pre_scale"), res);
    if (src_attrs->center) {
      res = Call(multiply_op, {res_var, exp_gamma});
    } else {
      res = Call(multiply_op, {res_var, exp_gamma}, Attrs(), call->sinfo_args, call->span);
    }
  }
  // offset
  if (src_attrs->center) {
    const auto& exp_beta =
        RewriteUtils::MakeCall(builder, ExprUtils::GetSpanName(call, "exp_beta"), reshape_op,
                               {call->args[2], ShapeExpr(exp_shape)});
    const auto& res_var =
        RewriteUtils::ReEmit(builder, ExprUtils::GetSpanName(call, "pre_offset"), res);
    res = Call(add_op, {res_var, exp_beta}, Attrs(), call->sinfo_args, call->span);
  }
  return res;
}

Expr RewriteMatmul(BlockBuilder builder, const Var& var, const Call& src_call,
                   const Map<Expr, Call>& new_calls, const String& config) {
  const auto& trt_config = ParseConfig(config);
  const auto& call = new_calls.count(src_call) ? new_calls[src_call] : src_call;
  const auto& shape_a = ExprUtils::GetShape(call->args[0]);
  const auto& shape_b = ExprUtils::GetShape(call->args[1]);
  static const Op& reshape_op = Op::Get("relax.reshape");
  if (call->args[1]->IsInstance<ConstantNode>() && shape_b.size() == 2 &&
      trt_config.linear_to_conv) {
    const auto& out_shape = ExprUtils::GetShape(var);
    PrimExpr accumulate = ArrayUtils::Accumulate(shape_a, shape_a.size() - 1);
    Array<PrimExpr> exp_shape{accumulate, shape_a[shape_a.size() - 1], Integer(1), Integer(1)};
    const auto& exp_in = RewriteUtils::MakeCall(builder, ExprUtils::GetSpanName(call, "exp_in"),
                                                reshape_op, {call->args[0], ShapeExpr(exp_shape)});
    // transpose and expand weight to OIHW
    static const Op& permute_dims_op = Op::Get("relax.permute_dims");
    auto permute_attrs = make_object<PermuteDimsAttrs>();
    Array<Integer> axes{Integer(1), Integer(0)};
    permute_attrs->axes = axes;
    const auto& trans_weight =
        RewriteUtils::MakeCall(builder, ExprUtils::GetSpanName(call, "trans_weight"),
                               permute_dims_op, {call->args[1]}, Attrs(permute_attrs));
    Array<PrimExpr> weight_shape{shape_b[1], shape_b[0], Integer(1), Integer(1)};
    const auto& exp_weight =
        RewriteUtils::MakeCall(builder, ExprUtils::GetSpanName(call, "exp_weight"), reshape_op,
                               {trans_weight, ShapeExpr(weight_shape)});
    // to conv2d
    static const Op& conv2d_op = Op::Get("relax.nn.conv2d");
    auto conv_attrs = make_object<Conv2DAttrs>();
    conv_attrs->strides = Array<IntImm>{Integer(1), Integer(1)};
    conv_attrs->padding = Array<IntImm>{Integer(0), Integer(0), Integer(0), Integer(0)};
    conv_attrs->dilation = Array<IntImm>{Integer(1), Integer(1)};
    conv_attrs->groups = 1;
    conv_attrs->data_layout = "NCHW";
    conv_attrs->kernel_layout = "OIHW";
    conv_attrs->out_layout = "NCHW";
    conv_attrs->out_dtype = ExprUtils::GetDataType(var);
    const auto& conv2d = RewriteUtils::MakeCall(builder, ExprUtils::GetSpanName(call, "conv2d"),
                                                conv2d_op, {exp_in, exp_weight}, Attrs(conv_attrs));
    return Call(reshape_op, {conv2d, ShapeExpr(out_shape)}, Attrs(), call->sinfo_args, call->span);
  }
  if (shape_a.size() > shape_b.size()) {
    Array<PrimExpr> exp_shape(shape_a.size(), Integer(1));
    size_t diff = shape_a.size() - shape_b.size();
    for (size_t i = diff; i < shape_a.size(); i++) {
      exp_shape.Set(i, shape_b[i - diff]);
    }
    const auto& expand_b =
        RewriteUtils::MakeCall(builder, ExprUtils::GetSpanName(call, "expand_b"), reshape_op,
                               {call->args[1], ShapeExpr(exp_shape)});
    return Call(call->op, {call->args[0], expand_b}, call->attrs, call->sinfo_args, call->span);
  }
  if (shape_a.size() < shape_b.size()) {
    Array<PrimExpr> exp_shape(shape_b.size(), Integer(1));
    size_t diff = shape_b.size() - shape_a.size();
    for (size_t i = diff; i < shape_b.size(); i++) {
      exp_shape.Set(i, shape_a[i - diff]);
    }
    const auto& expand_a =
        RewriteUtils::MakeCall(builder, ExprUtils::GetSpanName(call, "expand_a"), reshape_op,
                               {call->args[0], ShapeExpr(exp_shape)});
    return Call(call->op, {expand_a, call->args[1]}, call->attrs, call->sinfo_args, call->span);
  }
  return call;
}

Expr RewriteRsqrt(BlockBuilder builder, const Var& var, const Call& src_call,
                  const Map<Expr, Call>& new_calls, const String& config) {
  const auto& call = new_calls.count(src_call) ? new_calls[src_call] : src_call;
  const auto& input_shape = ExprUtils::GetShape(call->args[0]);
  const auto& in_dtype = ExprUtils::GetDataType(call->args[0]);
  // create 1 constant
  const auto& one = RewriteUtils::MakeConstant(builder, ExprUtils::GetSpanName(call, "eps"), 1,
                                               in_dtype, input_shape.size());

  // create ops
  static const Op& divide_op = Op::Get("relax.divide");
  static const Op& sqrt_op = Op::Get("relax.sqrt");

  // expand and divide
  const auto& sqrt = RewriteUtils::MakeCall(builder, ExprUtils::GetSpanName(call, "sqrt"), sqrt_op,
                                            {call->args[0]});
  return Call(divide_op, {one, sqrt}, Attrs(), call->sinfo_args, call->span);
}

Expr RewriteSilu(BlockBuilder builder, const Var& var, const Call& src_call,
                 const Map<Expr, Call>& new_calls, const String& config) {
  const auto& call = new_calls.count(src_call) ? new_calls[src_call] : src_call;
  // create ops
  static const Op& multiply_op = Op::Get("relax.multiply");
  static const Op& sigmoid_op = Op::Get("relax.sigmoid");
  // silu=input*sigmoid(input)
  const auto& sigmoid = RewriteUtils::MakeCall(builder, ExprUtils::GetSpanName(call, "sigmoid"),
                                               sigmoid_op, {call->args[0]});
  return Call(multiply_op, {call->args[0], sigmoid}, Attrs(), call->sinfo_args, call->span);
}

Expr RewriteShapeLike(BlockBuilder builder, const Var& var, const Call& src_call,
                      const Map<Expr, Call>& new_calls, const String& config) {
  const auto& call = new_calls.count(src_call) ? new_calls[src_call] : src_call;
  const auto& output_shape = ExprUtils::GetShape(var);
  static const Op& reshape_op = Op::Get("relax.reshape");
  return Call(reshape_op, {call->args[0], ShapeExpr(output_shape)}, Attrs(), call->sinfo_args,
              call->span);
}

Expr RewriteSplit(BlockBuilder builder, const Var& var, const Call& src_call,
                  const Map<Expr, Call>& new_calls, const String& config) {
  const auto& call = new_calls.count(src_call) ? new_calls[src_call] : src_call;
  const auto& input_shape = ExprUtils::GetShape(call->args[0]);
  const auto* src_attrs = src_call->attrs.as<SplitAttrs>();
  size_t axis = CommonUtils::GetIndex(src_attrs->axis, input_shape.size());
  std::vector<int64_t> split_begins, split_ends;
  // get split begins and ends
  if (src_attrs->indices_or_sections->IsInstance<PrimExprNode>()) {
    int64_t sections = Downcast<Integer>(src_attrs->indices_or_sections)->value;
    int64_t size = Downcast<Integer>(input_shape[axis])->value / sections;
    for (int64_t i = 0; i < sections; i++) {
      split_begins.push_back(i * size);
      split_ends.push_back(i * size + size);
    }
  } else if (src_attrs->indices_or_sections->IsInstance<ArrayNode>()) {
    const auto& indices = Downcast<Array<Integer>>(src_attrs->indices_or_sections);
    int64_t last_index = 0;
    for (size_t i = 0; i < indices.size(); ++i) {
      split_begins.push_back(last_index);
      last_index = indices[i]->value;
      split_ends.push_back(last_index);
    }
    split_begins.push_back(last_index);
    split_ends.push_back(Downcast<Integer>(input_shape[axis])->value);
  } else {
    LOG_FATAL << "Unexpected indices_or_sections " << src_attrs->indices_or_sections << "("
              << src_attrs->indices_or_sections->GetTypeKey() << ")";
  }
  // create strided_slices
  Array<Expr> outputs;
  for (size_t i = 0; i < split_begins.size(); i++) {
    static const Op& strided_slice_op = Op::Get("relax.strided_slice");
    const auto& axes = Tuple(Array<Expr>{PrimValue(IntImm(DataType::Int(64), axis))});
    const auto& begin = Tuple(Array<Expr>{PrimValue(IntImm(DataType::Int(64), split_begins[i]))});
    const auto& end = Tuple(Array<Expr>{PrimValue(IntImm(DataType::Int(64), split_ends[i]))});
    const auto& strides = Tuple(Array<Expr>{PrimValue(IntImm(DataType::Int(64), 1))});
    auto attrs = make_object<StridedSliceAttrs>();
    attrs->assume_inbound = true;
    const auto& slice = RewriteUtils::MakeCall(
        builder, ExprUtils::GetSpanName(call, "slice_" + std::to_string(i)), strided_slice_op,
        {call->args[0], axes, begin, end, strides}, Attrs(attrs));
    outputs.push_back(slice);
  }
  return Tuple(outputs, call->span);
}

// nn ops
TVM_REGISTER_OP("relax.nn.attention")
    .set_attr<FRewriteTensorRT>("FRewriteTensorRT", RewriteAttention);
TVM_REGISTER_OP("relax.nn.attention_bias")
    .set_attr<FRewriteTensorRT>("FRewriteTensorRT", RewriteAttention);
TVM_REGISTER_OP("relax.nn.batch_norm")
    .set_attr<FRewriteTensorRT>("FRewriteTensorRT", RewriteBatchNorm);
TVM_REGISTER_OP("relax.nn.conv1d").set_attr<FRewriteTensorRT>("FRewriteTensorRT", RewriteConv1d);
TVM_REGISTER_OP("relax.nn.group_norm")
    .set_attr<FRewriteTensorRT>("FRewriteTensorRT", RewriteGroupNorm);
TVM_REGISTER_OP("relax.nn.gelu").set_attr<FRewriteTensorRT>("FRewriteTensorRT", RewriteGelu);
TVM_REGISTER_OP("relax.nn.gelu_tanh")
    .set_attr<FRewriteTensorRT>("FRewriteTensorRT", RewriteGeluTanh);
TVM_REGISTER_OP("relax.nn.layer_norm")
    .set_attr<FRewriteTensorRT>("FRewriteTensorRT", RewriteLayerNorm);
TVM_REGISTER_OP("relax.nn.silu").set_attr<FRewriteTensorRT>("FRewriteTensorRT", RewriteSilu);

// elemwise ops
TVM_REGISTER_OP("relax.add").set_attr<FRewriteTensorRT>("FRewriteTensorRT", RewriteAdd);
TVM_REGISTER_OP("relax.divide").set_attr<FRewriteTensorRT>("FRewriteTensorRT", RewriteElemwise);
TVM_REGISTER_OP("relax.floor_divide")
    .set_attr<FRewriteTensorRT>("FRewriteTensorRT", RewriteElemwise);
TVM_REGISTER_OP("relax.greater").set_attr<FRewriteTensorRT>("FRewriteTensorRT", RewriteElemwise);
TVM_REGISTER_OP("relax.less").set_attr<FRewriteTensorRT>("FRewriteTensorRT", RewriteElemwise);
TVM_REGISTER_OP("relax.maximum").set_attr<FRewriteTensorRT>("FRewriteTensorRT", RewriteElemwise);
TVM_REGISTER_OP("relax.minimum").set_attr<FRewriteTensorRT>("FRewriteTensorRT", RewriteElemwise);
TVM_REGISTER_OP("relax.multiply").set_attr<FRewriteTensorRT>("FRewriteTensorRT", RewriteElemwise);
TVM_REGISTER_OP("relax.power").set_attr<FRewriteTensorRT>("FRewriteTensorRT", RewriteElemwise);
TVM_REGISTER_OP("relax.subtract").set_attr<FRewriteTensorRT>("FRewriteTensorRT", RewriteElemwise);

// math ops
TVM_REGISTER_OP("relax.argmax").set_attr<FRewriteTensorRT>("FRewriteTensorRT", RewriteArgmaxmin);
TVM_REGISTER_OP("relax.argmin").set_attr<FRewriteTensorRT>("FRewriteTensorRT", RewriteArgmaxmin);
TVM_REGISTER_OP("relax.broadcast_to")
    .set_attr<FRewriteTensorRT>("FRewriteTensorRT", RewriteBroadcastTo);
TVM_REGISTER_OP("relax.expand_dims")
    .set_attr<FRewriteTensorRT>("FRewriteTensorRT", RewriteShapeLike);
TVM_REGISTER_OP("relax.matmul").set_attr<FRewriteTensorRT>("FRewriteTensorRT", RewriteMatmul);
TVM_REGISTER_OP("relax.rsqrt").set_attr<FRewriteTensorRT>("FRewriteTensorRT", RewriteRsqrt);
TVM_REGISTER_OP("relax.squeeze").set_attr<FRewriteTensorRT>("FRewriteTensorRT", RewriteShapeLike);
TVM_REGISTER_OP("relax.split").set_attr<FRewriteTensorRT>("FRewriteTensorRT", RewriteSplit);

class TensorRTTransformer : public ExprMutator {
 public:
  explicit TensorRTTransformer(IRModule ctx_module, const String& config)
      : ExprMutator(ctx_module) {
    config_ = config;
  }

  void VisitBinding_(const VarBindingNode* binding, const CallNode* call_node) final {
    if (const auto* op_node = call_node->op.as<OpNode>()) {
      const auto& op = Downcast<Op>(GetRef<Op>(op_node));
      const auto& rewrite_map = Op::GetAttrMap<FRewriteTensorRT>("FRewriteTensorRT");
      if (rewrite_map.count(op)) {
        const auto& call = GetRef<Call>(call_node);
        FRewriteTensorRT f = rewrite_map[op];
        const auto& new_call = f(builder_, binding->var, call, new_calls_, config_);
        if (new_call != call) {
          ReEmitBinding(binding, builder_->Normalize(new_call));
          new_calls_.Set(binding->var, call);
        }
      }
    }
    if (!new_calls_.count(binding->var)) {
      ExprMutator::VisitBinding_(binding, call_node);
    }
  }

 private:
  Map<Expr, Call> new_calls_;
  String config_;
};

Function TransformTensorRT(const Function& func, const IRModule& module, const String& config) {
  return Downcast<Function>(TensorRTTransformer(module, config).VisitExpr(func));
}

namespace transform {

Pass TransformTensorRT(const String& config) {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        return relax::TransformTensorRT(f, m, config);
      };
  return CreateFunctionPass(pass_func, 0, "TransformTensorRT", {});
}

TVM_REGISTER_GLOBAL("relax.transform.TransformTensorRT").set_body_typed(TransformTensorRT);

}  // namespace transform
}  // namespace relax
}  // namespace tvm
