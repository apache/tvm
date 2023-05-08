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
 * \file src/relay/backend/contrib/cmsisnn/fuse_pads.cc
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

inline IntImm ToIntImm(int32_t value) { return IntImm(DataType::Int(32), value); }

/*!
 * \brief From padding attributes of nn.pad and qnn.conv2d, calculates effective padding along H
 * and W dimensions.
 */
Array<IntImm> GetEffectiveConv2DPadding(Expr conv2d, Expr pad) {
  // pad_width: ((), (top, bottom), (left, right), ()) for NHWC layout
  // conv2d_attrs->padding: (top, left, bottom, right)
  auto* conv2d_call = conv2d.as<CallNode>();
  auto* conv2d_attrs = conv2d_call->attrs.as<Conv2DAttrs>();
  std::string data_layout = conv2d_attrs->data_layout.c_str();
  int pos_h = data_layout.find("H");
  int pos_w = data_layout.find("W");

  auto* pad_call = pad.as<CallNode>();
  Array<Array<Integer>> pad_width = pad_call->attrs.as<PadAttrs>()->pad_width;
  int pad_top =
      qnn::get_const_int(conv2d_attrs->padding[0]) + qnn::get_const_int(pad_width[pos_h][0]);
  int pad_left =
      qnn::get_const_int(conv2d_attrs->padding[1]) + qnn::get_const_int(pad_width[pos_w][0]);
  int pad_bottom =
      qnn::get_const_int(conv2d_attrs->padding[2]) + qnn::get_const_int(pad_width[pos_h][1]);
  int pad_right =
      qnn::get_const_int(conv2d_attrs->padding[3]) + qnn::get_const_int(pad_width[pos_w][1]);

  return {ToIntImm(pad_top), ToIntImm(pad_left), ToIntImm(pad_bottom), ToIntImm(pad_right)};
}

/*!
 * \brief This Mutator will find all partitioned functions meant for CMSIS-NN Conv2D.
 * Then, it will fuse preceding pads with qnn.conv2d.
 */
class FusePadsMutator : public MixedModeMutator {
 public:
  explicit FusePadsMutator(const IRModule& mod) : mod_(mod) {}

 private:
  /*!
   * \brief In order to eliminate preceding nn.pad op, pad_width of nn.pad is passed onto
   * convolution layer to update Conv2DAttrs's padding attribute. */
  void UpdateConv2DPadding(const CallNode* conv2d_call, const CallNode* pad_call,
                           Attrs* new_attrs) {
    Array<IntImm> effective_padding =
        GetEffectiveConv2DPadding(GetRef<Call>(conv2d_call), GetRef<Call>(pad_call));
    int pad_top = effective_padding[0]->value;
    int pad_left = effective_padding[1]->value;
    int pad_bottom = effective_padding[2]->value;
    int pad_right = effective_padding[3]->value;
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

    // Prepare new attrs as padding has changed
    auto* conv2d_attrs = conv2d_call->attrs.as<Conv2DAttrs>();
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
    attrs->padding = {pad_top, pad_left, pad_bottom, pad_right};
    *new_attrs = tvm::Attrs{attrs};
  }

  /*!
   * \brief Identifies the sequence for qnn.conv2D and fuses the preceding nn.pad present within the
   * CMSIS-NN partitioned function. */
  Expr FusePadConv2d(const CallNode* conv2d_call) {
    // create new paddings for qnn.conv2d
    tvm::Attrs new_conv2d_attrs = conv2d_call->attrs;
    Expr new_conv2d_input = conv2d_call->args[0];
    if (auto* pad_call = conv2d_call->args[0].as<CallNode>()) {
      if (auto* pad_call_op = pad_call->op.as<OpNode>()) {
        if (pad_call_op->name == "nn.pad") {
          new_conv2d_input = pad_call->args[0];
          UpdateConv2DPadding(conv2d_call, pad_call, &new_conv2d_attrs);
        }
      }
    }

    // Conv2D arguments: pad's input + rest of the origin args
    auto new_conv2d_args = conv2d_call->args;
    new_conv2d_args.erase(new_conv2d_args.begin());
    new_conv2d_args.insert(new_conv2d_args.begin(), new_conv2d_input);
    Call ret_call = Call(conv2d_call->op, new_conv2d_args, new_conv2d_attrs, {}, conv2d_call->span);
    return std::move(ret_call);
  }

  Expr Rewrite_(const CallNode* call, const Expr& post) final {
    Expr ret_call = post;
    auto* post_call = post.as<CallNode>();

    // Fuse nn.pad and qnn.conv2d
    if (auto* conv2d_op = post_call->op.as<OpNode>()) {
      if (conv2d_op->name == "qnn.conv2d") {
        ret_call = FusePadConv2d(post_call);
      }
    }

    // Identify qnn.conv2d partitioned function
    if (post_call->op.as<FunctionNode>()) {
      auto* func = call->op.as<FunctionNode>();
      auto func_name = func->GetAttr<String>(attr::kComposite);
      if (func_name.defined() && func_name == "cmsis-nn.qnn_conv2d") {
        Expr new_body = VisitExpr(func->body);
        Function new_func = Function(FreeVars(new_body), new_body, func->ret_type,
                                     FreeTypeVars(new_body, mod_), func->attrs);
        ret_call = Call(new_func, post_call->args);
        ret_call->span = call->span;
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
TVM_REGISTER_GLOBAL("relay.ext.cmsisnn.transform.GetEffectiveConv2DPadding")
    .set_body_typed(GetEffectiveConv2DPadding);

}  // namespace cmsisnn
}  // namespace contrib
}  // namespace relay
}  // namespace tvm
