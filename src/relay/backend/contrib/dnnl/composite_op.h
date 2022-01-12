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

#ifndef TVM_RELAY_BACKEND_CONTRIB_DNNL_COMPOSITE_OP_H_
#define TVM_RELAY_BACKEND_CONTRIB_DNNL_COMPOSITE_OP_H_

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "codegen_tools.h"

namespace tvm {
namespace relay {
namespace contrib {

using KernelAttrs = std::unordered_map<std::string, dmlc::any>;
using KernelRequisites = std::pair<std::vector<Expr>, KernelAttrs>;

/**
 * Rule to replace relay graph with DNNL supported pattern.
 *  Conv+Bias+Requantize+Clip(cast_int8 or relu)+Sum
 *
 * @note
 *  assume wgh_zp == 0. Only symmetric weight are supported right now.
 *
 * Original relay representation:
 * %1 = conv(SRC, WGH) - conv(src_zp, WGH) + BIAS
 * %2 = (%1 - rq_in_zp) * rq_in_scl / rq_out_scl + rq_out_zp
 * %3 = clip(%2, 0, 255)
 * %4 = ((%3 - sum_lh_zp) * sum_lh_scl + (SRC2 - sum_rh_zp) * sum_rh_scl)/sum_out_scl + sum_out_zp
 *
 * DNNL implemented patern:
 * %1 = clip((conv(SRC, WGH + wgh_shft) + (BIAS + bias_shft)) * o_scl, clip_low, clip_high)
 *      * clip_scl + SRC2 * sum_scl + dst_zp
 *
 * @note
 *  dst_zp can be moved into bias_shft
 *  clip_scl can be moved into o_scl
 *
 * Possible solution #0:
 *   clip_scl = sum_lh_scl /sum_out_scl
 *   clip_low  = 0 - sum_lh_zp - sum_rh_zp * sum_rh_scl / sum_lh_scl
 *   clip_high = 255 - sum_lh_zp - sum_rh_zp * sum_rh_scl / sum_lh_scl
 *   o_scl = rq_in_scl / rq_out_scl
 *   bias_shft = - conv(src_zp, WGH) - rq_in_zp + (rq_out_zp - sum_lh_zp - sum_rh_zp * sum_rh_scl /
 *               sum_lh_scl) * rq_out_scl / rq_in_scl
 *   wgh_shft = 0
 *   sum_scl = sum_rh_scl / sum_out_scl
 *   dst_zp = sum_out_zp
 *
 *
 * Possible solution #1 (dst_zp == 0):
 *   new_clip_low = clip_low + dst_zp / clip_scl
 *   new_clip_high = clip_high + dst_zp / clip_scl
 *   new_bias_shft = bias_shft + dst_zp / clip_scl / o_scl
 *
 *   clip_scl = sum_lh_scl /sum_out_scl
 *   clip_low  = 0 - sum_lh_zp - sum_rh_zp * sum_rh_scl / sum_lh_scl +
 *               sum_out_zp * sum_out_scl / sum_lh_scl
 *   clip_high = 255 - sum_lh_zp - sum_rh_zp * sum_rh_scl / sum_lh_scl +
 *               sum_out_zp * sum_out_scl / sum_lh_scl
 *   o_scl = rq_in_scl / rq_out_scl
 *   bias_shft = - conv(src_zp, WGH) - rq_in_zp + (rq_out_zp - sum_lh_zp - sum_rh_zp * sum_rh_scl /
 *               sum_lh_scl) * rq_out_scl / rq_in_scl + sum_out_zp * sum_out_scl / sum_lh_scl *
 *               rq_out_scl / rq_in_scl
 *   sum_scl = sum_rh_scl / sum_out_scl
 *   dst_zp = 0
 *
 *
 * Possible solution #2 (clip_scl == 1.f):
 *   new_clip_low = clip_low * clip_scl
 *   new_clip_high = clip_high * clip_scl
 *   new_o_scl = o_scl * clip_scl
 *
 *   clip_scl = 1.f
 *   clip_low  = (0 - sum_lh_zp - sum_rh_zp * sum_rh_scl / sum_lh_scl + sum_out_zp * sum_out_scl /
 *               sum_lh_scl) * sum_lh_scl /sum_out_scl
 *   clip_high = (255 - sum_lh_zp - sum_rh_zp * sum_rh_scl / sum_lh_scl + sum_out_zp * sum_out_scl /
 *               sum_lh_scl) * sum_lh_scl /sum_out_scl
 *   o_scl = rq_in_scl / rq_out_scl * sum_lh_scl /sum_out_scl
 *   bias_shft = - conv(src_zp, WGH) - rq_in_zp + (rq_out_zp - sum_lh_zp - sum_rh_zp * sum_rh_scl /
 *               sum_lh_scl) * rq_out_scl / rq_in_scl + sum_out_zp * sum_out_scl / sum_lh_scl *
 *               rq_out_scl / rq_in_scl
 *   sum_scl = sum_rh_scl / sum_out_scl
 *   dst_zp = 0
 */
struct qnn_arg_set_relay {
  Expr wgh, bias, src_zp, wgh_zp, src_scl, wgh_scl, rq_in_zp, rq_in_scl, rq_out_zp, rq_out_scl,
      sum_lh_zp, sum_lh_scl, sum_rh_zp, sum_rh_scl, sum_out_scl, sum_out_zp;
};

struct qnn_arg_set_dnnl {
  Expr bias, clip_scl, clip_low, clip_high, o_scl, sum_scl, dst_zp;

  /** Evaluate contained expressions and collapse to scalar if it's broadcast scalar */
  qnn_arg_set_dnnl evalAndCollapseToScalar() const {
    qnn_arg_set_dnnl res;

    res.bias = collapse_to_scalar(EvalExpr(this->bias));
    res.clip_low = collapse_to_scalar(EvalExpr(this->clip_low));
    res.clip_high = collapse_to_scalar(EvalExpr(this->clip_high));
    res.clip_scl = collapse_to_scalar(EvalExpr(this->clip_scl));
    res.dst_zp = collapse_to_scalar(EvalExpr(this->dst_zp));
    res.o_scl = collapse_to_scalar(EvalExpr(this->o_scl));
    res.sum_scl = collapse_to_scalar(EvalExpr(this->sum_scl));

    return res;
  }
};

qnn_arg_set_dnnl qnnReformulate(const qnn_arg_set_relay &origin) {
  auto &r = origin;  // short alias "relay"
  using namespace tensor_arithmetic;
  ICHECK(is_const_scalar_eq(r.wgh_zp, 0)) << "Doesn't support patterns with not zero kernel_zp";

  // Convolution on zp filled data. Also applicable for dense and grouped conv.
  auto conv_zp = [] (const Expr &zp, const Expr &wgh) -> Expr {
    if (is_const_scalar_eq<int32_t>(zp, 0)) return constant(0);
    ICHECK(is_scalar(zp)) << "Only scalar data_zp is supported for qnn primitives";

    // reduce kernel {OC, IC, KH, KW} -> {OC} in case of group that is still correct
    auto reduced_kernel =
        MakeReduce(cast<int32_t>(wgh), {0}, false /*keepdims*/, true /*exclude*/, "sum");
    return zp * reduced_kernel;
  };

  // If there is no bias will use zero value
  auto bias = r.bias.defined() ? r.bias : constant(0);

  // Will use formulas #2 (dst_zp == 0, clip_scl == 1.0f)
  qnn_arg_set_dnnl res;
  res.dst_zp = constant(0);
  res.o_scl = r.rq_in_scl / r.rq_out_scl * r.sum_lh_scl / r.sum_out_scl;
  res.sum_scl = r.sum_rh_scl / r.sum_out_scl;
  res.clip_scl = constant(1.0f);
  res.clip_low = (cast<float>(constant(0) - r.sum_lh_zp)
                  - cast<float>(r.sum_rh_zp) * r.sum_rh_scl / r.sum_lh_scl
                  + cast<float>(r.sum_out_zp) * r.sum_out_scl / r.sum_lh_scl)
                 * r.sum_lh_scl / r.sum_out_scl;
  res.clip_high = (cast<float>(constant(255) - r.sum_lh_zp)
                   - cast<float>(r.sum_rh_zp) * r.sum_rh_scl / r.sum_lh_scl
                   + cast<float>(r.sum_out_zp) * r.sum_out_scl / r.sum_lh_scl)
                  * r.sum_lh_scl / r.sum_out_scl;
  res.bias = cast<float>(bias) - cast<float>(conv_zp(r.src_zp, r.wgh) + r.rq_in_zp)
      + cast<float>(r.rq_out_zp - r.sum_lh_zp) * r.rq_out_scl / r.rq_in_scl
      - cast<float>(r.sum_rh_zp) * r.sum_rh_scl / r.sum_lh_scl * r.rq_out_scl / r.rq_in_scl
      + cast<float>(r.sum_out_zp) * r.sum_out_scl / r.sum_lh_scl * r.rq_out_scl / r.rq_in_scl;

  return res.evalAndCollapseToScalar();
}

/*!
 * @brief Specify optional QNN args and attrs if required
 *
 * @param wgh weight node
 * @param bias bias node (constant node)
 * @param base base action node (conv or dense)
 * @param rq requantize node (optional)
 * @param sum sum node (optional)
 * @param inputs resulting input collection (will append to it)
 * @param attrs resulting attribute collection (will append to it)
 */
void optQnnArgsForRqSumPattern(const Expr &wgh, const Expr &bias, const OpSeq::Layer &base,
                               const OpSeq::Layer &rq, const OpSeq::Layer &sum,
                               std::vector<Expr> *inputs, KernelAttrs *attrs) {
  ICHECK(wgh.defined());
  ICHECK(base);
  ICHECK(inputs);
  ICHECK(attrs);

  qnn_arg_set_relay args_relay;
  args_relay.wgh = wgh;
  args_relay.bias = bias;

  args_relay.src_zp  = base.extern_args_[2];
  args_relay.wgh_zp  = base.extern_args_[3];
  args_relay.src_scl = base.extern_args_[4];
  args_relay.wgh_scl = base.extern_args_[5];

  // Requantize is optional
  args_relay.rq_in_scl  = rq ? rq.extern_args_[0] : constant(1.f);
  args_relay.rq_in_zp   = rq ? rq.extern_args_[1] : constant(0);
  args_relay.rq_out_scl = rq ? rq.extern_args_[2] : constant(1.f);
  args_relay.rq_out_zp  = rq ? rq.extern_args_[3] : constant(0);

  // Sum is optional
  args_relay.sum_lh_scl  = sum ? sum.extern_args_[1] : constant(1.f);
  args_relay.sum_lh_zp   = sum ? sum.extern_args_[2] : constant(0);
  args_relay.sum_rh_scl  = sum ? sum.extern_args_[3] : constant(0.f);
  args_relay.sum_rh_zp   = sum ? sum.extern_args_[4] : constant(0);
  args_relay.sum_out_scl = sum ? sum.extern_args_[5] : constant(1.f);
  args_relay.sum_out_zp  = sum ? sum.extern_args_[6] : constant(0);

  // Recalculate QNN specific arguments
  auto args_dnnl = qnnReformulate(args_relay);

  // Helper to register optional qnn args
  auto put_arg = [&attrs, &inputs] (const Expr &expr, std::string name, auto skip_value) {
    if (expr.defined() && !is_const_scalar_eq(expr, skip_value)) {
      (*attrs)[name] = dmlc_attr(inputs->size());
      inputs->push_back(expr);
    }
  };

  // Bias should be a vector {OC}, even if it's scalar
  if (is_scalar(args_dnnl.bias) && !is_const_scalar_eq(args_dnnl.bias, 0)) {
    int OC = shape_of(wgh)[0];
    args_dnnl.bias = EvalExpr(broadcast(args_dnnl.bias, {OC}));
  }

  put_arg(args_dnnl.bias, "bias_idx", 0);
  put_arg(args_dnnl.o_scl, "o_scl_idx", 1);
  put_arg(args_dnnl.dst_zp, "dst_zp_idx", 0);

  if (!is_const_scalar_eq(args_dnnl.sum_scl, 0.f)) {
    put_arg(sum.extern_args_[0], "sum_idx", std::nanf(""));
    put_arg(args_dnnl.sum_scl, "sum_scl_idx", 0);
  }

  if (args_dnnl.clip_scl.defined()) {
    ICHECK(is_scalar(args_dnnl.clip_low));
    ICHECK(is_scalar(args_dnnl.clip_high));

    std::vector<std::string> clip_attr{"clip"};
    clip_attr.push_back(std::to_string(inputs->size()));
    inputs->push_back(args_dnnl.clip_scl);
    clip_attr.push_back(std::to_string(inputs->size()));
    inputs->push_back(args_dnnl.clip_low);
    clip_attr.push_back(std::to_string(inputs->size()));
    inputs->push_back(args_dnnl.clip_high);

    (*attrs)["activation"] = dmlc_attr(clip_attr);
  }
}

/*!
 * Legalize bias shape to 1D form
 *
 * @param orig_bias
 * @return 1D version of original bias expr
 */
Expr legalizeBiasShape(const Expr& orig_bias) {
  return EvalExpr(squeeze(orig_bias));
}

/**
 * Parse qnn.conv2d based fused patterns
 * @param fn function to parse
 */
KernelRequisites parseQnnConv2dComposite(const FunctionNode* fn) {
  OpSeq ops;
  ops(fn->body);

  std::vector<std::string>
      qnn_conv_sum_pat{"qnn.conv2d", "add", "qnn.requantize", "clip", "cast", "qnn.add", "clip"},
      qnn_conv_sum_no_bias_pat{"qnn.conv2d", "qnn.requantize", "clip", "cast", "qnn.add", "clip"},
      qnn_conv_pat{"qnn.conv2d", "add", "qnn.requantize", "clip", "cast"},
      qnn_conv_no_bias_pat{"qnn.conv2d", "qnn.requantize", "clip", "cast"};

  auto layer_names = ops.getOpNames();
  ICHECK(layer_names == qnn_conv_sum_pat ||
         layer_names == qnn_conv_pat ||
         layer_names == qnn_conv_no_bias_pat ||
         layer_names == qnn_conv_sum_no_bias_pat)
      << "Unsupported patter for DNNL code generator. Looks like some discrepancy "
         "between DNNL partitioner pass and code generator.";

  auto conv = ops.getOpLayer("qnn.conv2d");
  auto bs = ops.getOpLayer("add");
  auto rq = ops.getOpLayer("qnn.requantize");
  auto sum = ops.getOpLayer("qnn.add");

  auto data = conv.extern_args_[0];
  auto wgh = conv.extern_args_[1];
  auto bias = bs ? legalizeBiasShape(bs.extern_args_[0]) : Expr{};

  // make regular wights layout
  auto wgh_layout = conv.call_node_->attrs.as<Conv2DAttrs>()->kernel_layout;
  auto oihw_wgh = permute(wgh, permutation(wgh_layout, "OIHW"));

  auto attrs = extractAttrs(conv.call_node_);  // extract original attrs
  std::vector<Expr> inputs = {data, wgh};  // args with fixed positions

  optQnnArgsForRqSumPattern(oihw_wgh, bias, conv, rq, sum, &inputs, &attrs);
  return {inputs, attrs};
}

KernelRequisites parseQnnDenseComposite(const FunctionNode* fn) {
  OpSeq ops;
  ops(fn->body);

  std::vector<std::string>
      qnn_dense_sum_pat{"qnn.dense", "add", "qnn.requantize", "clip", "cast", "qnn.add", "clip"},
      qnn_dense_sum_no_bias_pat{"qnn.dense", "qnn.requantize", "clip", "cast", "qnn.add", "clip"},
      qnn_dense_pat{"qnn.dense", "add", "qnn.requantize", "clip", "cast"},
      qnn_dense_no_bias_pat{"qnn.dense", "qnn.requantize", "clip", "cast"};;

  auto layer_names = ops.getOpNames();
  ICHECK(layer_names == qnn_dense_sum_pat ||
         layer_names == qnn_dense_sum_no_bias_pat ||
         layer_names == qnn_dense_pat ||
         layer_names == qnn_dense_no_bias_pat)
      << "Unsupported patter for DNNL code generator. Looks like some discrepancy "
         "between DNNL partitioner pass and code generator.";

  auto dense = ops.getOpLayer("qnn.dense");
  auto bs = ops.getOpLayer("add");
  auto rq = ops.getOpLayer("qnn.requantize");
  auto sum = ops.getOpLayer("qnn.add");

  auto data = dense.extern_args_[0];
  auto wgh = dense.extern_args_[1];
  auto bias = bs ? legalizeBiasShape(bs.extern_args_[0]) : Expr{};

  auto attrs = extractAttrs(dense.call_node_);  // extract original attrs
  std::vector<Expr> inputs = {data, wgh};  // args with fixed positions

  optQnnArgsForRqSumPattern(wgh, bias, dense, rq, sum, &inputs, &attrs);
  return {inputs, attrs};
}

KernelRequisites parseBaseOpComposite(const FunctionNode* fn, const std::string &base_op_name) {
  ICHECK(base_op_name == "nn.conv2d" || base_op_name == "nn.dense");
  OpSeq ops;
  ops(fn->body);

  auto conv = ops.getOpLayer(base_op_name);
  auto bias = ops.getOpLayer("add");
  auto relu = ops.getOpLayer("nn.relu");
  auto tanh = ops.getOpLayer("tanh");
  auto sigm = ops.getOpLayer("sigmoid");

  auto act = relu ? relu : tanh ? tanh : sigm ? sigm : OpSeq::Layer{};

  auto attrs = extractAttrs(conv.call_node_);
  std::vector<Expr> inputs = {
      conv.extern_args_[0],  // data
      conv.extern_args_[1]   // kernel
  };

  if (bias) {
    attrs["bias_idx"] = dmlc_attr(inputs.size());
    inputs.push_back(bias.extern_args_[0]);
  }

  if (act) {
    auto act_name = act.call_node_->op.as<OpNode>()->name;
    std::vector<std::string> act_attr = {act_name};
    act_attr.push_back(std::to_string(inputs.size()));
    inputs.push_back(InferType(constant(1.0f)));
    act_attr.push_back(std::to_string(inputs.size()));
    inputs.push_back(InferType(constant(0.0f)));
    act_attr.push_back(std::to_string(inputs.size()));
    inputs.push_back(InferType(constant(0.0f)));

    attrs["activation"] = dmlc_attr(act_attr);
  }

  return {inputs, attrs};
}

KernelRequisites DNNLCompositeFunctionsParser(const FunctionNode* fn) {
  auto comp = fn->GetAttr<String>(attr::kComposite);
  ICHECK(comp.defined());
  std::string name = comp.value();

  if (name == "dnnl.qnn.conv2d_sum" ||
      name == "dnnl.qnn.conv2d") {
    return parseQnnConv2dComposite(fn);
  } else if (name == "dnnl.qnn.dense_sum" ||
             name == "dnnl.qnn.dense") {
    return parseQnnDenseComposite(fn);
  } else if (name == "dnnl.conv2d_bias_relu" ||
             name == "dnnl.conv2d_bias_tanh" ||
             name == "dnnl.conv2d_bias_sigmoid" ||
             name == "dnnl.conv2d_bias" ||
             name == "dnnl.conv2d_relu" ||
             name == "dnnl.conv2d_tanh" ||
             name == "dnnl.conv2d_sigmoid") {
    return parseBaseOpComposite(fn, "nn.conv2d");
  } else if (name == "dnnl.dense_bias_relu" ||
             name == "dnnl.dense_bias_tanh" ||
             name == "dnnl.dense_bias_sigmoid" ||
             name == "dnnl.dense_bias" ||
             name == "dnnl.dense_relu" ||
             name == "dnnl.dense_tanh" ||
             name == "dnnl.dense_sigmoid") {
    return parseBaseOpComposite(fn, "nn.dense");
  } else {
    LOG(FATAL) << "Unknown composite function " << name;
  }
  return {};
}

}  // namespace contrib
}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_BACKEND_CONTRIB_DNNL_COMPOSITE_OP_H_
