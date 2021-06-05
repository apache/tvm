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
 * \file src/relay/transforms/fold_explicit_padding.cc
 * \brief A pass for folding explicit pads into other ops.
 */

#include <tvm/relay/dataflow_matcher.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>
#include <tvm/runtime/logging.h>

#include "../op/tensor/transform.h"
#include "pattern_utils.h"

namespace tvm {
namespace relay {

/*!
 * \brief SimplifyConvPad matches a pad followed by a conv/convtranspose/pool/etc
 * with a pad attribute and merges the padding into the kernel.
 */
class SimplifyConvPad {
 public:
  DFPattern pattern() const { return pattern_; }

  SimplifyConvPad() {
    x_ = IsWildcard();
    w_ = IsWildcard();
    pad_ = IsOp("nn.pad")({x_, IsWildcard()});
    conv1d_ = IsOp("nn.conv1d");
    conv2d_ = IsOp("nn.conv2d");
    conv3d_ = IsOp("nn.conv3d");
    conv_ = (conv1d_ || conv2d_ || conv3d_)({pad_, w_});
    pattern_ = conv_;
  }

  template <typename T>
  Attrs MakeConvAttrs(const T* old_attrs, const Array<PrimExpr> padding) const {
    ICHECK(old_attrs);
    ICHECK(padding.size() == old_attrs->padding.size())
        << "Number of dimensions to pad and convolution padding attributes should have the same "
           "extent";

    auto new_attrs = make_object<T>();
    Array<PrimExpr> combined_padding;
    for (size_t i = 0; i < padding.size(); ++i) {
      combined_padding.push_back(padding[i] + old_attrs->padding[i]);
    }
    new_attrs->strides = old_attrs->strides;
    new_attrs->padding = combined_padding;
    new_attrs->dilation = old_attrs->dilation;
    new_attrs->groups = old_attrs->groups;
    new_attrs->channels = old_attrs->channels;
    new_attrs->kernel_size = old_attrs->kernel_size;
    new_attrs->data_layout = old_attrs->data_layout;
    new_attrs->kernel_layout = old_attrs->kernel_layout;
    new_attrs->out_layout = old_attrs->out_layout;
    new_attrs->out_dtype = old_attrs->out_dtype;
    return Attrs(new_attrs);
  }

  template <typename T>
  Attrs GetAttrs(const PadAttrs* param, const T* attrs) const {
    ICHECK(param);
    ICHECK(attrs);
    ICHECK(attrs->data_layout.size() == param->pad_width.size())
        << "Data Layout and padding attributes should have the same extent";

    std::string data_layout = attrs->data_layout;
    std::set<char> image_dims({'H', 'W', 'D'});
    Array<PrimExpr> padding;
    // If we're padding a non-spatial dimension, don't simplify
    // Convolution can only pad on spatial axes
    for (size_t i = 0; i < param->pad_width.size(); ++i) {
      if (!image_dims.count(data_layout[i])) {
        for (size_t j = 0; j < param->pad_width[i].size(); ++j) {
          if (param->pad_width[i][j] != 0) {
            return Attrs();
          }
        }
      }
    }
    for (size_t j = 0; j < param->pad_width[0].size(); ++j) {
      for (size_t i = 0; i < param->pad_width.size(); ++i) {
        if (image_dims.count(data_layout[i])) {
          padding.push_back(param->pad_width[i][j]);
        }
      }
    }

    return MakeConvAttrs(attrs, padding);
  }

  Expr callback(const Expr& pre, const Expr& post,
                const Map<DFPattern, Array<Expr>>& node_map) const {
    const CallNode* call_node = post.as<CallNode>();
    ICHECK(call_node);
    auto pad = node_map[pad_][0];
    const CallNode* pad_node = pad.as<CallNode>();
    ICHECK(pad_node);
    const PadAttrs* param = pad_node->attrs.as<PadAttrs>();
    ICHECK(param);
    Array<Expr> args = pad_node->args;

    // Possibly perform more optimizations if the pad_value is 0
    const ConstantNode* pad_value = args[1].as<ConstantNode>();
    if (param->pad_mode == "constant" && pad_value && ToScalar(pad_value->data) == 0.0) {
      Attrs attrs;
      if (node_map.count(conv1d_)) {
        attrs = GetAttrs(param, call_node->attrs.as<Conv1DAttrs>());
      } else if (node_map.count(conv2d_)) {
        attrs = GetAttrs(param, call_node->attrs.as<Conv2DAttrs>());
      } else if (node_map.count(conv3d_)) {
        attrs = GetAttrs(param, call_node->attrs.as<Conv3DAttrs>());
      } else {
        return post;
      }
      if (!attrs.defined()) {
        return post;
      }
      auto x = node_map[x_][0];
      auto w = node_map[w_][0];
      return Call(call_node->op, {x, w}, attrs, call_node->type_args, call_node->span);
    }
    return post;
  }

 private:
  /*! \brief Pattern for rewriting */
  DFPattern pattern_;
  /*! \brief Pattern input */
  DFPattern x_;
  /*! \brief Pattern input weight */
  DFPattern w_;
  /*! \brief Pattern pad */
  DFPattern pad_;
  /*! \brief Pattern conv */
  DFPattern conv_;
  DFPattern conv1d_;
  DFPattern conv2d_;
  DFPattern conv3d_;
};

class SimplifyExplicitPadding {
 public:
  explicit SimplifyExplicitPadding(IRModule mod) : mod_(mod) {
    CreateCallback(SimplifyConvPad());
    // TODO(mbrookhart): ConvTranspose(Pad(x)), Pool(Pad(x))
  }
  template <typename T>
  void CreateCallback(const T& pattern) {
    auto func = [pattern](TVMArgs args, TVMRetValue* rv) {
      Expr pre = args[0];
      Expr post = args[1];
      Map<DFPattern, Array<Expr>> node_map = args[2];
      *rv = pattern.callback(pre, post, node_map);
    };
    callbacks_.push_back(DFPatternCallback(pattern.pattern(), PackedFunc(func), true));
  }

  Expr Simplify(const Expr& expr) { return RewritePatterns(callbacks_, expr, mod_); }

 private:
  IRModule mod_;
  /*! \brief Callbacks for expr simplification */
  Array<DFPatternCallback> callbacks_;
};

/*!
 * \brief FoldExplicitPadding finds explict padding before an op that can
 * support implicit padding and fuses them.
 */
Expr FoldExplicitPadding(const Expr& expr, const IRModule& mod) {
  return SimplifyExplicitPadding(mod).Simplify(expr);
}

namespace transform {

Pass FoldExplicitPadding() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        return Downcast<Function>(FoldExplicitPadding(f, m));
      };
  return CreateFunctionPass(pass_func, 0, " FoldExplicitPadding", {"InferType"});
}

TVM_REGISTER_GLOBAL("relay._transform.FoldExplicitPadding").set_body_typed(FoldExplicitPadding);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
