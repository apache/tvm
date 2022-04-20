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
 * \file src/relay/transforms/flatten_atrous_conv.cc
 * \brief This transform flattens atrous convolution, which corresponds to the sequence of
 * operations: "space_to_batch_nd"->"conv2d"->"batch_to_space_nd".
 */

#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/dataflow_matcher.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/qnn/attrs.h>
#include <tvm/relay/transform.h>
#include <tvm/topi/broadcast.h>

#include <array>
#include <set>
#include <unordered_map>

#include "../qnn/utils.h"
#include "pattern_utils.h"

namespace tvm {
namespace relay {

/* Description of FlattenAtrousConv
 *
 * The purpose of this pass is to find a sequence of space_to_batch_nd-conv2d-batch_to_space_nd
 * operations:
 *
 *   x     w
 *   |     |
 *   s2b   |
 *    \   /
 *     conv2d
 *      |
 *      b2s
 *
 * and convert them into subgraphs with a convolution with the modified "dilation" and
 * recalculated "padding" parameters.
 */

using ExprSet = std::unordered_set<Expr, ObjectPtrHash, ObjectPtrEqual>;

class FlattenAtrousConvSubgraphMutator {
 public:
  Expr MutateSubgraph(const Expr& expr) {
    try {
      const CallNode* b2s_node_ = expr.as<CallNode>();
      const CallNode* conv2d_node_ = b2s_node_->args[0].as<CallNode>();
      const CallNode* s2b_node_ = conv2d_node_->args[0].as<CallNode>();

      ICHECK(b2s_node_ != nullptr);
      const auto* b2s_attrs = b2s_node_->attrs.as<BatchToSpaceNDAttrs>();
      ICHECK(b2s_attrs != nullptr);

      Array<PrimExpr> dilation = {b2s_attrs->block_shape[0], b2s_attrs->block_shape[1]};

      ICHECK(conv2d_node_ != nullptr);
      const auto* conv2d_attrs = conv2d_node_->attrs.as<Conv2DAttrs>();
      ICHECK(conv2d_attrs != nullptr);

      Array<PrimExpr> kernel_shape = conv2d_attrs->kernel_size;
      PrimExpr kernel_h = kernel_shape[0];
      PrimExpr kernel_w = kernel_shape[1];

      ICHECK(s2b_node_ != nullptr);
      const auto* s2b_attrs = s2b_node_->attrs.as<SpaceToBatchNDAttrs>();
      ICHECK(s2b_attrs != nullptr);

      Expr data = s2b_node_->args[0];
      ICHECK(conv2d_attrs->data_layout == "NHWC");
      Array<PrimExpr> data_shape = transform::InferTypeLocal(data).as<TensorTypeNode>()->shape;
      PrimExpr in_h = data_shape[1];
      PrimExpr in_w = data_shape[2];

      PrimExpr dilation_h = dilation[0];
      PrimExpr dilation_w = dilation[1];

      PrimExpr dilated_kernel_h = (kernel_h - 1) * dilation_h + 1;
      PrimExpr dilated_kernel_w = (kernel_w - 1) * dilation_w + 1;

      Array<PrimExpr> strides = {1, 1};
      PrimExpr stride_h = strides[0];
      PrimExpr stride_w = strides[1];

      auto _get_pad_pair = [](PrimExpr input1d, PrimExpr kernel1d,
                              PrimExpr stride1d) -> Array<PrimExpr> {
        PrimExpr out1d = truncdiv((input1d + stride1d - 1), stride1d);
        PrimExpr pad = topi::maximum(((out1d - 1) * stride1d + kernel1d - input1d), 0);
        PrimExpr pad_before = truncdiv(pad, 2);
        PrimExpr pad_after = pad - pad_before;
        return {pad_before, pad_after};
      };

      Array<PrimExpr> pad_v = _get_pad_pair(in_h, dilated_kernel_h, stride_h);
      Array<PrimExpr> pad_h = _get_pad_pair(in_w, dilated_kernel_w, stride_w);

      Array<IndexExpr> padding = {pad_v[0], pad_h[0], pad_v[1], pad_h[1]};

      Expr weight = conv2d_node_->args[1];

      if (conv2d_node_->op == Op::Get("nn.conv2d")) {
        return Conv2D(data, weight, strides, padding, dilation, conv2d_attrs->groups,
                      conv2d_attrs->channels, conv2d_attrs->kernel_size, conv2d_attrs->data_layout,
                      conv2d_attrs->kernel_layout, conv2d_attrs->out_layout,
                      conv2d_attrs->out_dtype);
      }

      if (conv2d_node_->op == Op::Get("qnn.conv2d")) {
        Expr input_zero_point = conv2d_node_->args[2];
        Expr kernel_zero_point = conv2d_node_->args[3];
        Expr input_scale = conv2d_node_->args[4];
        Expr kernel_scale = conv2d_node_->args[5];
        return qnn::MakeQnnConv2D(data, weight, input_zero_point, kernel_zero_point, input_scale,
                                  kernel_scale, strides, padding, dilation, conv2d_attrs->groups,
                                  conv2d_attrs->channels, conv2d_attrs->kernel_size,
                                  conv2d_attrs->data_layout, conv2d_attrs->kernel_layout,
                                  conv2d_attrs->out_layout, conv2d_attrs->out_dtype);
      }

      DLOG(INFO) << "Ran into an unhandled convolution, skipping " << expr << std::endl;
      return expr;
    } catch (std::exception& e) {
      DLOG(INFO) << "Ran into an error rewriting a subgraph, skipping " << expr << " with "
                 << e.what() << std::endl;
      return expr;
    }
  }
};

class FlattenAtrousConvRewriter : public MixedModeMutator {
 protected:
  Expr Rewrite_(const CallNode* pre, const Expr& post) override {
    if (const CallNode* call_node = post.as<CallNode>()) {
      if (ops_[op_iter_].count(call_node->op)) {
        ++op_iter_;
        if (op_iter_ == ops_.size()) {
          op_iter_ = 0;
          return FlattenAtrousConvSubgraphMutator().MutateSubgraph(post);
        }
      } else {
        op_iter_ = 0;
      }
    }
    return post;
  }

 private:
  size_t op_iter_ = 0;
  const std::array<ExprSet, 3> ops_ = {
      ExprSet{Op::Get("nn.space_to_batch_nd")},
      ExprSet{Op::Get("nn.conv2d"), Op::Get("qnn.conv2d")},
      ExprSet{Op::Get("nn.batch_to_space_nd")},
  };
};

Expr FlattenAtrousConv(const Expr& expr, const IRModule& mod) {
  return FlattenAtrousConvRewriter().Mutate(expr);
}

namespace transform {

Pass FlattenAtrousConv() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        return Downcast<Function>(FlattenAtrousConv(f, m));
      };
  return CreateFunctionPass(pass_func, 0, "FlattenAtrousConv", {"InferType"});
}

TVM_REGISTER_GLOBAL("relay._transform.FlattenAtrousConv").set_body_typed(FlattenAtrousConv);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
