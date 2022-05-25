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
#include <tvm/runtime/data_type.h>
#include <tvm/runtime/logging.h>
#include <tvm/tir/op.h>
#include <tvm/topi/nn/pooling.h>

#include "../op/tensor/transform.h"
#include "pattern_utils.h"

namespace tvm {
namespace relay {

/*!
 * \brief SimplifyConvPad matches a pad followed by a conv
 * with a pad attribute and merges the padding into the kernel.
 */
class SimplifyConvPad {
 public:
  DFPattern pattern() const { return pattern_; }

  SimplifyConvPad() {
    x_ = IsWildcard();
    pad_ = IsOp("nn.pad")({x_, IsWildcard()});

    // pad->conv patterns
    w_ = IsWildcard();
    conv1d_ = IsOp("nn.conv1d");
    conv2d_ = IsOp("nn.conv2d");
    conv3d_ = IsOp("nn.conv3d");
    contrib_conv2d_nchwc_ = IsOp("nn.contrib_conv2d_NCHWc");
    conv_ = (conv1d_ || conv2d_ || conv3d_ || contrib_conv2d_nchwc_)({pad_, w_});

    input_zero_point_ = IsWildcard();
    kernel_zero_point_ = IsWildcard();
    input_scale_ = IsWildcard();
    kernel_scale_ = IsWildcard();
    qconv2d_ = IsOp("qnn.conv2d")(
        {pad_, w_, input_zero_point_, kernel_zero_point_, input_scale_, kernel_scale_});

    // pad->pool patterns
    avg_pool1d_ = IsOp("nn.avg_pool1d");
    avg_pool2d_ = IsOp("nn.avg_pool2d");
    avg_pool3d_ = IsOp("nn.avg_pool3d");
    avg_pool_ = avg_pool1d_ || avg_pool2d_ || avg_pool3d_;
    max_pool1d_ = IsOp("nn.max_pool1d");
    max_pool2d_ = IsOp("nn.max_pool2d");
    max_pool3d_ = IsOp("nn.max_pool3d");
    max_pool_ = max_pool1d_ || max_pool2d_ || max_pool3d_;
    pool_ = (avg_pool_ || max_pool_)({pad_});

    pattern_ = conv_ || qconv2d_ || pool_;
  }

  template <typename T>
  Array<PrimExpr> get_combined_padding(const T* old_attrs, Array<PrimExpr> padding) const {
    ICHECK(padding.size() == old_attrs->padding.size())
        << "Number of dimensions to pad and convolution padding attributes should have the same "
           "extent";

    auto new_attrs = make_object<T>();
    Array<PrimExpr> combined_padding;
    for (size_t i = 0; i < padding.size(); ++i) {
      combined_padding.push_back(padding[i] + old_attrs->padding[i]);
    }
    return combined_padding;
  }

  template <typename T>
  Attrs MakeConvAttrs(const T* old_attrs, const Array<PrimExpr> padding) const {
    // Creates attrs from old_attrs with fields shared by 1D, 2D, 3D pool attrs flavors
    ICHECK(old_attrs);
    auto combined_padding = get_combined_padding(old_attrs, padding);
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
  Attrs MakeConv2D3DAttrs(const T* old_attrs, const Array<PrimExpr> padding) const {
    // Propagate additional Conv2D- and Conv3D-specific attrs
    auto attrs = MakeConvAttrs(old_attrs, padding);
    attrs->auto_scheduler_rewritten_layout = old_attrs->auto_scheduler_rewritten_layout;
    return attrs;
  }

  template <typename T>
  Attrs MakePoolAttrs(const T* old_attrs, const Array<PrimExpr> padding) const {
    // Creates attrs from old_attrs with fields shared by 1D, 2D, 3D pool attrs flavors
    ICHECK(old_attrs);
    auto combined_padding = get_combined_padding(old_attrs, padding);
    new_attrs->pool_size = old_attrs->pool_size;
    new_attrs->strides = old_attrs->strides;
    new_attrs->dilation = old_attrs->dilation;
    new_attrs->padding = combined_padding;
    new_attrs->layout = old_attrs->layout;
    new_attrs->out_layout = old_attrs->out_layout;
    new_attrs->ceil_mode = old_attrs->ceil_mode;
    return Attrs(new_attrs);
  }

  template <typename T>
  Attrs MakeAvgPoolAttrs(const T* old_attrs, const Array<PrimExpr> padding) const {
    // Propagate additional AvgPool-specific attrs
    auto attrs = MakePoolAttrs(old_attrs, padding);
    attrs->count_include_pad = old_attrs->count_include_pad;
    return attrs;
  }

  template <typename T>
  bool GetAvgPoolAttrsCountIncludePad(const T* attrs) const {
    // Helper function to get the `count_include_pad` field from 1D, 2D, 3D avg pool attrs
    return attrs->count_include_pad;
  }

  template <typename T>
  Attrs GetAttrs(const PadAttrs* param, const T* attrs, DFPattern pattern) const {
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

    if (pattern == conv1d_) {
      return MakeConvAttrs(attrs, padding);
    } else if (pattern == conv2d_ || pattern == qconv2d_ || pattern == conv3d_) {
      return MakeConv2D3DAttrs(attrs, padding);
    } else if (pattern == max_pool_) {
      return MakePoolAttrs(attrs, padding);
    } else if (pattern == avg_pool_) {
      return MakeAvgPoolAttrs(attrs, padding);
    }
    ICHECK(false) << "Unsupported fold explicit padding case, where given pattern is not mapped to attrs";
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

    auto x = node_map[x_][0];
    auto w = node_map[w_][0];

    // Possibly perform more optimizations if the pad_value is 0
    const Expr& pv = pad_node->args[1];
    const ConstantNode* pad_value = pv.as<ConstantNode>();
    if (node_map.find(qconv2d_) != node_map.end()) {
      Attrs attrs = GetAttrs(param, call_node->attrs.as<Conv2DAttrs>(), qconv2d_);
      auto input_zero_point = node_map[input_zero_point_][0];
      auto kernel_zero_point = node_map[kernel_zero_point_][0];
      auto input_scale = node_map[input_scale_][0];
      auto kernel_scale = node_map[kernel_scale_][0];
      // Fold Padding and QNN Convolution only if pad value == input zero point.
      if (IsEqualScalar(input_zero_point, pv)) {
        return Call(call_node->op,
                    {x, w, input_zero_point, kernel_zero_point, input_scale, kernel_scale}, attrs,
                    call_node->type_args, call_node->span);
      }
      return post;
    }
    

    if (param->pad_mode == "constant" && pad_value && ToScalar(pad_value->data) == 0.0) {
      Attrs attrs;
      if (node_map.count(conv1d_)) {
        attrs = GetAttrs(param, call_node->attrs.as<Conv1DAttrs>(), conv1d_);
      } else if (node_map.count(conv2d_)) {
        attrs = GetAttrs(param, call_node->attrs.as<Conv2DAttrs>(), conv2d_);
      } else if (node_map.count(conv3d_)) {
        attrs = GetAttrs(param, call_node->attrs.as<Conv3DAttrs>(), conv3d_);
      } else {
        return post;
      }
      if (!attrs.defined()) {
        return post;
      }
      return Call(call_node->op, {x, w}, attrs, call_node->type_args, call_node->span);
    }
    // The default pad constant for max pool is the min possible value for the dtype
    if (param->pad_mode == "constant" && pad_value) {
      Attrs attrs;

      auto min_value = tvm::min_value(tvm::runtime::DataType(pad_value->data->dtype));
      const FloatImmNode* maybe_min_float = min_value.as<FloatImmNode>();
      const IntImmNode* maybe_min_int = min_value.as<IntImmNode>();

      auto pad_scalar = ToScalar(pad_value->data);

      if (node_map.count(max_pool_)
          && ((maybe_min_float && pad_scalar == maybe_min_float->value)
          || (maybe_min_int && pad_scalar == maybe_min_int->value))) {
          // When the pad value in the preceding pad op matches the default max pool pad value
          // (minimum possible value for the dtype), fold.
          if (node_map.count(max_pool1d_)) {
            attrs = GetAttrs(param, call_node->attrs.as<MaxPool1DAttrs>(), max_pool1d_);
          } else if (node_map.count(max_pool2d_)) {
            attrs = GetAttrs(param, call_node->attrs.as<MaxPool2DAttrs>(), max_pool2d_);
          } else if (node_map.count(max_pool3d_)) {
            attrs = GetAttrs(param, call_node->attrs.as<MaxPool3DAttrs>(), max_pool3d_);
          }
      } else if (node_map.count(avg_pool_) && pad_scalar == 0) {
          // When the pad value in the preceding pad op matches the default avg pool pad value (0), fold.
          auto obj = AvgPool1DAttrs;
          if (node_map.count(avg_pool1d_)) {
            auto old_attrs = call_node->attrs.as<AvgPool1DAttrs>();
            if (old_attrs->count_include_pad) {
              attrs = GetAttrs(param, old_attrs, avg_pool1d_);
            }
          } else if (node_map.count(avg_pool2d_)) {
            attrs = GetAttrs(param, call_node->attrs.as<AvgPool2DAttrs>());
          } else if (node_map.count(avg_pool3d_)) {
            attrs = GetAttrs(param, call_node->attrs.as<AvgPool3DAttrs>());
          }
      }

      if (!attrs.defined()) {
        return post;
      }
      return Call(call_node->op, {x}, attrs, call_node->type_args, call_node->span);
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
  DFPattern contrib_conv2d_nchwc_;
  DFPattern qconv2d_;
  DFPattern input_zero_point_;
  DFPattern kernel_zero_point_;
  DFPattern input_scale_;
  DFPattern kernel_scale_;
  /*! \brief Pattern pool */
  DFPattern pool_;
  DFPattern avg_pool1d_;
  DFPattern avg_pool2d_;
  DFPattern avg_pool3d_;
  DFPattern avg_pool_;
  DFPattern max_pool1d_;
  DFPattern max_pool2d_;
  DFPattern max_pool3d_;
  DFPattern max_pool_;
};


/*!
 * \brief SimplifyPoolPad matches a pad followed by a pool
 * with a pad attribute and merges the padding into the kernel.
 */
class SimplifyPoolPad {
 public:
  DFPattern pattern() const { return pattern_; }

  SimplifyPoolPad() {
    x_ = IsWildcard();
    pad_ = IsOp("nn.pad")({x_, IsWildcard()});
      
    avg_pool1d_ = IsOp("nn.avg_pool1d");
    avg_pool2d_ = IsOp("nn.avg_pool2d");
    avg_pool3d_ = IsOp("nn.avg_pool3d");
    avg_pool_ = avg_pool1d_ || avg_pool2d_ || avg_pool3d_;

    max_pool1d_ = IsOp("nn.max_pool1d");
    max_pool2d_ = IsOp("nn.max_pool2d");
    max_pool3d_ = IsOp("nn.max_pool3d");
    max_pool_ = max_pool1d_ || max_pool2d_ || max_pool3d_;

    pool_ = (avg_pool_ || max_pool_)({pad_});

    pattern_ = pool_;
  }

  template <typename T>
  Attrs MakePoolAttrs(const T* old_attrs, const Array<PrimExpr> padding) const {
    ICHECK(old_attrs);
    ICHECK(padding.size() == old_attrs->padding.size())
        << "Number of dimensions to pad and pool padding attributes should have the same "
           "extent";

    auto new_attrs = make_object<T>();
    Array<PrimExpr> combined_padding;
    std::cout << "padding length " << padding.size() << std::endl;
    for (size_t i = 0; i < padding.size(); ++i) {
      auto added_item = padding[i] + old_attrs->padding[i];
      std::cout << "adding item to padding " << added_item << std::endl;
      combined_padding.push_back(added_item);
    }

    // TODO @anwang: generalize for avg pool as well, or split out?
    new_attrs->pool_size = old_attrs->pool_size;
    new_attrs->strides = old_attrs->strides;
    new_attrs->dilation = old_attrs->dilation;
    new_attrs->padding = combined_padding;
    new_attrs->layout = old_attrs->layout;
    new_attrs->out_layout = old_attrs->out_layout;
    new_attrs->ceil_mode = old_attrs->ceil_mode;

    // TODO count_include_pad
    // new_attrs->count_include_pad = true;

    return Attrs(new_attrs);
  }

  template <typename T>
  Attrs GetAttrs(const PadAttrs* param, const T* attrs) const {
    ICHECK(param);
    ICHECK(attrs);
    ICHECK(attrs->layout.size() == param->pad_width.size())
        << "Data Layout and padding attributes should have the same extent";

    std::string data_layout = attrs->layout;
    std::set<char> image_dims({'H', 'W', 'D'});
    Array<PrimExpr> padding;
    // If we're padding a non-spatial dimension, don't simplify
    // Pool can only pad on spatial axes
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

    return MakePoolAttrs(attrs, padding);
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

    auto x = node_map[x_][0];

    // Possibly perform more optimizations if the pad_value is 0
    const ConstantNode* pad_value = args[1].as<ConstantNode>();

    // The default pad constant for max pool is the min possible value for the dtype
    if (param->pad_mode == "constant" && pad_value) {
      Attrs attrs;

      auto min_value = tvm::min_value(tvm::runtime::DataType(pad_value->data->dtype));
      const FloatImmNode* maybe_min_float = min_value.as<FloatImmNode>();
      const IntImmNode* maybe_min_int = min_value.as<IntImmNode>();

      auto pad_scalar = ToScalar(pad_value->data);

      if (node_map.count(max_pool_)
          && ((maybe_min_float && pad_scalar == maybe_min_float->value)
          || (maybe_min_int && pad_scalar == maybe_min_int->value))) {
          // When the pad value in the preceding pad op matches the default max pool pad value
          // (minimum possible value for the dtype), fold.
          if (node_map.count(max_pool1d_)) {
            attrs = GetAttrs(param, call_node->attrs.as<MaxPool1DAttrs>());
          } else if (node_map.count(max_pool2d_)) {
            attrs = GetAttrs(param, call_node->attrs.as<MaxPool2DAttrs>());
          } else if (node_map.count(max_pool3d_)) {
            attrs = GetAttrs(param, call_node->attrs.as<MaxPool3DAttrs>());
          }
      } else if (node_map.count(avg_pool_) && pad_scalar == 0) {
          // When the pad value in the preceding pad op matches the default avg pool pad value (0), fold.
          if (node_map.count(avg_pool1d_)) {
            attrs = GetAttrs(param, call_node->attrs.as<AvgPool1DAttrs>());
          } else if (node_map.count(avg_pool2d_)) {
            attrs = GetAttrs(param, call_node->attrs.as<AvgPool2DAttrs>());
          } else if (node_map.count(avg_pool3d_)) {
            attrs = GetAttrs(param, call_node->attrs.as<AvgPool3DAttrs>());
          }
      }

      if (!attrs.defined()) {
        return post;
      }
      return Call(call_node->op, {x}, attrs, call_node->type_args, call_node->span);
    }
    return post;
  }

 private:
  /*! \brief Pattern for rewriting */
  DFPattern pattern_;
  /*! \brief Pattern input */
  DFPattern x_;
  /*! \brief Pattern pad */
  DFPattern pad_;
  /*! \brief Pattern pool */
  DFPattern pool_;
  DFPattern avg_pool1d_;
  DFPattern avg_pool2d_;
  DFPattern avg_pool3d_;
  DFPattern avg_pool_;
  DFPattern max_pool1d_;
  DFPattern max_pool2d_;
  DFPattern max_pool3d_;
  DFPattern max_pool_;
};

class SimplifyExplicitPadding {
 public:
  explicit SimplifyExplicitPadding(IRModule mod) : mod_(mod) {
    CreateCallback(SimplifyPoolPad());
    CreateCallback(SimplifyConvPad());
    // TODO(mbrookhart): ConvTranspose(Pad(x))
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
