
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
 * \file fuse_pads.cc
 * \brief Fuses pads that precede qnn.conv2d ops inside CMSIS-NN composite functions.
 */

#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/attrs/transform.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>
#include <tvm/runtime/ndarray.h>

#include "../../../op/make_op.h"
#include "../../../qnn/utils.h"
#include "../../../transforms/pattern_utils.h"
#include "convolutions.h"

namespace tvm {
namespace relay {
namespace contrib {
namespace cmsisnn {

/*!
 * \brief This Mutator will find all partitioned functions meant for CMSIS-NN Conv2D.
 * Then, it will fuse preceding pads with qnn.conv2d.
 */
class FusePadsMutator : public MixedModeMutator {
 public:
  explicit FusePadsMutator(const IRModule& mod) : mod_(mod) {}

 private:
  /*!  * \brief In order to eliminate preceding nn.pad op, pad_width of nn.pad is passed onto
   * convolution layer to update Conv2DAttrs's padding attribute. */
  void UpdateConv2DPadding(const CallNode* conv2d_call, const Array<Array<Integer>>& pad_width,
                           const Conv2DAttrs* conv2d_attrs, Attrs* new_attrs) {
    auto attrs = make_object<Conv2DAttrs>();
    attrs->strides = std::move(conv2d_attrs->strides);
    attrs->dilation = std::move(conv2d_attrs->dilation);
    attrs->groups = conv2d_attrs->groups;
    attrs->channels = std::move(conv2d_attrs->channels);
    attrs->kernel_size = std::move(conv2d_attrs->kernel_size);
    attrs->data_layout = std::move(conv2d_attrs->data_layout);
    attrs->kernel_layout = std::move(conv2d_attrs->kernel_layout);
    attrs->out_layout = std::move(conv2d_attrs->out_layout);
    attrs->out_dtype = std::move(conv2d_attrs->out_dtype);

    // pad_width: ((), (top, bottom), (left, right), ()) for NHWC layout
    // conv2d_attrs->padding: (top, left, bottom, right)
    std::string data_layout = conv2d_attrs->data_layout.c_str();
    int pos_h = data_layout.find("H");
    int pos_w = data_layout.find("W");

    int pad_top =
        qnn::get_const_int(conv2d_attrs->padding[0]) + qnn::get_const_int(pad_width[pos_h][0]);
    int pad_left =
        qnn::get_const_int(conv2d_attrs->padding[1]) + qnn::get_const_int(pad_width[pos_w][0]);
    int pad_bottom =
        qnn::get_const_int(conv2d_attrs->padding[2]) + qnn::get_const_int(pad_width[pos_h][1]);
    int pad_right =
        qnn::get_const_int(conv2d_attrs->padding[3]) + qnn::get_const_int(pad_width[pos_w][1]);

    int pad_diff_w = pad_right - pad_left;
    int pad_diff_h = pad_bottom - pad_top;
    bool can_pad_be_fused =
        ((pad_diff_w == 0 || pad_diff_w == 1) && (pad_diff_h == 0 || pad_diff_h == 1));
    std::string error = "Difference on each side of a dimension should be either 0 or 1. ";
    error += "Effective padding in this case: (pad_top, pad_left, pad_bottom, pad_right)=(";
    error += std::to_string(pad_top);
    error += ", ";
    error += std::to_string(pad_left);
    error += ", ";
    error += std::to_string(pad_bottom);
    error += ", ";
    error += std::to_string(pad_right);
    error += ")";
    ICHECK(can_pad_be_fused) << error;

    attrs->padding = {pad_top, pad_left, pad_bottom, pad_right};
    *new_attrs = tvm::Attrs{attrs};
  }

  /*!
   * \brief Identifies the sequence for qnn.conv2D and fuses the preceding nn.pad present within the
   * CMSIS-NN partitioned function. */
  Expr FusePadConv2d(const Expr& expr) {
    const CallNode* clip_call = nullptr;
    const CallNode* requantize_call = nullptr;
    const CallNode* bias_add_call = nullptr;
    const CallNode* conv2d_call = nullptr;
    const CallNode* pad_call = nullptr;
    auto* final_call = expr.as<CallNode>();
    auto* final_op = final_call->op.as<OpNode>();
    if (final_op->name == "clip") {
      clip_call = final_call;
      requantize_call = clip_call->args[0].as<CallNode>();
    } else {
      requantize_call = final_call;
    }
    auto* requantize_input = requantize_call->args[0].as<CallNode>();
    auto* requantize_input_op = requantize_input->op.as<OpNode>();
    if (requantize_input_op->name == "nn.bias_add") {
      bias_add_call = requantize_input;
      conv2d_call = bias_add_call->args[0].as<CallNode>();
    } else {
      conv2d_call = requantize_input;
    }
    Array<Array<Integer>> pad_width;
    if (auto* conv2d_input = conv2d_call->args[0].as<CallNode>()) {
      if (auto* conv2d_input_op = conv2d_input->op.as<OpNode>()) {
        if (conv2d_input_op->name == "nn.pad") {
          pad_call = conv2d_input;
          pad_width = pad_call->attrs.as<PadAttrs>()->pad_width;
        }
      }
    }

    auto* conv2d_attrs = conv2d_call->attrs.as<Conv2DAttrs>();
    tvm::Attrs new_conv2d_attrs = conv2d_call->attrs;

    // create new paddings for qnn.conv2d
    conv2d_attrs = new_conv2d_attrs.as<Conv2DAttrs>();
    Expr new_conv2d_input = conv2d_call->args[0];
    if (pad_call) {
      new_conv2d_input = pad_call->args[0];
      UpdateConv2DPadding(conv2d_call, pad_width, conv2d_attrs, &new_conv2d_attrs);
    }

    // Conv2D arguments: pad's input + rest of the origin args
    auto new_conv2d_args = conv2d_call->args;
    new_conv2d_args.erase(new_conv2d_args.begin());
    new_conv2d_args.insert(new_conv2d_args.begin(), new_conv2d_input);
    Call ret_call = Call(conv2d_call->op, new_conv2d_args, new_conv2d_attrs, {});
    if (bias_add_call) {
      ret_call =
          Call(bias_add_call->op, {ret_call, bias_add_call->args[1]}, bias_add_call->attrs, {});
    }
    auto new_requantize_args = requantize_call->args;
    new_requantize_args.erase(new_requantize_args.begin());
    new_requantize_args.insert(new_requantize_args.begin(), ret_call);
    ret_call = Call(requantize_call->op, new_requantize_args, requantize_call->attrs, {});
    if (clip_call) {
      ret_call = Call(clip_call->op, {ret_call}, clip_call->attrs, {});
    }
    return std::move(ret_call);
  }

  Expr Rewrite_(const CallNode* call, const Expr& post) final {
    Expr ret_call = post;
    // Identify qnn.conv2d partitioned function
    if (call->op.as<FunctionNode>()) {
      auto* func = call->op.as<FunctionNode>();
      auto func_name = func->GetAttr<String>(attr::kComposite);
      if (func_name.defined() && func_name == "cmsis-nn.qnn_conv2d") {
        Expr new_body = FusePadConv2d(func->body);
        Function new_func = Function(FreeVars(new_body), new_body, func->ret_type,
                                     FreeTypeVars(new_body, mod_), func->attrs);
        auto* post_call = post.as<CallNode>();
        ret_call = Call(new_func, post_call->args);
      }
    }

    return ret_call;
  }

 private:
  IRModule mod_;
};

IRModule FusePads(const IRModule& mod) {
  for (auto gv : mod->GetGlobalVars()) {
    Function func = Downcast<Function>(mod->Lookup(gv));

    // only mutate CMSIS-NN partitioned functions
    auto compiler_name = func->GetAttr<String>(attr::kCompiler);
    if (!compiler_name.defined() || compiler_name != "cmsis-nn") {
      continue;
    }

    auto fuse_pads_mutator = FusePadsMutator(mod);
    auto new_func_body = fuse_pads_mutator.VisitExpr(func->body);
    if (!new_func_body.same_as(func->body)) {
      Function new_func =
          Function(func->params, new_func_body, func->ret_type, func->type_params, func->attrs);
      mod->Update(gv, new_func);
    }
  }
  return mod;
}

transform::Pass CMSISNNFusePads() {
  runtime::TypedPackedFunc<IRModule(IRModule, transform::PassContext)> pass_func =
      [=](IRModule m, transform::PassContext pc) { return FusePads(m); };
  return tvm::transform::CreateModulePass(pass_func, 0, "CMSISNNFusePads", {});
}

TVM_REGISTER_GLOBAL("relay.ext.cmsisnn.transform.CMSISNNFusePads").set_body_typed(CMSISNNFusePads);

}  // namespace cmsisnn
}  // namespace contrib
}  // namespace relay
}  // namespace tvm
