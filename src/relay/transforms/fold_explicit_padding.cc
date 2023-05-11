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

#include <optional>
#include <set>
#include <string>

#include "../op/tensor/transform.h"
#include "pattern_utils.h"

namespace tvm {
namespace relay {

/*!
 * \brief SimplifyExplicitPad matches a pad followed by a conv/maxpool/avgpool
 * with a pad attribute and merges the padding into the kernel.
 */
class SimplifyExplicitPad {
 public:
  DFPattern pattern() const { return pattern_; }

  SimplifyExplicitPad() {
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
    max_pool1d_ = IsOp("nn.max_pool1d");
    max_pool2d_ = IsOp("nn.max_pool2d");
    max_pool3d_ = IsOp("nn.max_pool3d");
    max_pool_ = max_pool1d_ || max_pool2d_ || max_pool3d_;
    pool_ = (max_pool_ || avg_pool1d_ || avg_pool2d_ || avg_pool3d_)({pad_});

    pattern_ = conv_ || qconv2d_ || pool_;
  }

  template <typename T>
  Array<PrimExpr> get_combined_padding(const T* old_attrs, Array<PrimExpr> padding) const {
    ICHECK(padding.size() == old_attrs->padding.size())
        << "Number of dimensions to pad and convolution padding attributes should have the same "
           "extent";

    Array<PrimExpr> combined_padding;
    for (size_t i = 0; i < padding.size(); ++i) {
      combined_padding.push_back(padding[i] + old_attrs->padding[i]);
    }
    return combined_padding;
  }

  template <typename T>
  Attrs MakeConvAttrs(const PadAttrs* param, const T* old_attrs) const {
    // Creates attrs from old_attrs with fields shared by 1D, 2D, 3D conv attrs
    ICHECK(old_attrs);
    ICHECK(param);
    auto padding = get_padding(param, old_attrs->data_layout);
    if (!padding) {
      return Attrs();
    }
    auto combined_padding = get_combined_padding(old_attrs, padding.value());

    auto new_attrs = make_object<T>();
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
  Attrs MakeConv2D3DAttrs(const PadAttrs* param, const T* old_attrs) const {
    // Propagate additional Conv2D- and Conv3D-specific attrs
    auto attrs = MakeConvAttrs(param, old_attrs);
    if (!attrs.defined()) {
      return Attrs();
    }

    T* new_attrs = const_cast<T*>(attrs.template as<T>());
    new_attrs->auto_scheduler_rewritten_layout = old_attrs->auto_scheduler_rewritten_layout;
    new_attrs->meta_schedule_original_shape = old_attrs->meta_schedule_original_shape;
    return attrs;
  }

  template <typename T>
  Attrs MakePoolAttrs(const PadAttrs* param, const T* old_attrs) const {
    // Creates attrs from old_attrs with fields shared by 1D, 2D, 3D pool attrs
    ICHECK(old_attrs);
    ICHECK(param);
    auto padding = get_padding(param, old_attrs->layout);
    if (!padding) {
      return Attrs();
    }
    auto combined_padding = get_combined_padding(old_attrs, padding.value());

    auto new_attrs = make_object<T>();
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
  Attrs MakeAvgPoolAttrs(const PadAttrs* param, const T* old_attrs) const {
    // Propagate additional AvgPool-specific attrs
    auto attrs = MakePoolAttrs(param, old_attrs);
    if (!attrs.defined()) {
      return attrs;
    }

    T* new_attrs = const_cast<T*>(attrs.template as<T>());
    new_attrs->count_include_pad = old_attrs->count_include_pad;
    if (!new_attrs->count_include_pad) {
      // AvgPool's divisor doesn't include padding, so don't fold the explicit pad
      // unless all original pad items are 0.
      for (IndexExpr pad : old_attrs->padding) {
        const IntImmNode* maybe_int_imm = pad.as<IntImmNode>();
        if (!maybe_int_imm || maybe_int_imm->value != 0) {
          // Return undefined attrs to signal that we don't want to fold explicit pad
          return Attrs();
        }
      }
      // Turn on `count_include_pad` to preserve original pad first, then pool behavior
      // where AvgPool's divisor implicitly includes padding.
      new_attrs->count_include_pad = true;
    }

    return attrs;
  }

  static const std::optional<Array<PrimExpr>> get_padding(const PadAttrs* param,
                                                          std::string data_layout) {
    // Gets spatial axes padding from the given PadAttrs `param`. If padding
    // is non-zero on non-spatial axes, return std::nullopt.
    ICHECK(param);
    ICHECK(data_layout.size() == param->pad_width.size())
        << "Data Layout and padding attributes should have the same extent";

    std::set<char> image_dims({'H', 'W', 'D'});
    Array<PrimExpr> padding;
    // If we're padding a non-spatial dimension, don't simplify
    // Convolution/Pool can only pad on spatial axes
    for (size_t i = 0; i < param->pad_width.size(); ++i) {
      if (!image_dims.count(data_layout[i])) {
        for (size_t j = 0; j < param->pad_width[i].size(); ++j) {
          if (param->pad_width[i][j] != 0) {
            return std::nullopt;
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
    return padding;
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

    const Expr& pv = pad_node->args[1];
    const ConstantNode* pad_value = pv.as<ConstantNode>();

    if (node_map.find(qconv2d_) != node_map.end()) {
      Attrs attrs = MakeConv2D3DAttrs(param, call_node->attrs.as<Conv2DAttrs>());
      if (!attrs.defined()) {
        return post;
      }
      auto input_zero_point = node_map[input_zero_point_][0];
      auto kernel_zero_point = node_map[kernel_zero_point_][0];
      auto input_scale = node_map[input_scale_][0];
      auto kernel_scale = node_map[kernel_scale_][0];
      // Fold Padding and QNN Convolution only if pad value == input zero point.
      if (IsEqualScalar(input_zero_point, pv)) {
        auto w = node_map[w_][0];
        return Call(call_node->op,
                    {x, w, input_zero_point, kernel_zero_point, input_scale, kernel_scale}, attrs,
                    call_node->type_args, call_node->span);
      }
      return post;
    }

    if (param->pad_mode == "constant" && pad_value) {
      Attrs attrs;
      auto pad_scalar = ToScalar(pad_value->data);
      if (pad_scalar == 0.0) {
        // Fold Padding and Conv/AvgPool only if pad_value == 0.
        if (node_map.count(conv_)) {
          if (node_map.count(conv1d_)) {
            attrs = MakeConvAttrs(param, call_node->attrs.as<Conv1DAttrs>());
          } else if (node_map.count(conv2d_)) {
            attrs = MakeConv2D3DAttrs(param, call_node->attrs.as<Conv2DAttrs>());
          } else if (node_map.count(conv3d_)) {
            attrs = MakeConv2D3DAttrs(param, call_node->attrs.as<Conv3DAttrs>());
          }
          if (!attrs.defined()) {
            return post;
          }
          auto w = node_map[w_][0];
          return Call(call_node->op, {x, w}, attrs, call_node->type_args, call_node->span);
        } else if (node_map.count(avg_pool1d_)) {
          attrs = MakeAvgPoolAttrs(param, call_node->attrs.as<AvgPool1DAttrs>());
        } else if (node_map.count(avg_pool2d_)) {
          attrs = MakeAvgPoolAttrs(param, call_node->attrs.as<AvgPool2DAttrs>());
        } else if (node_map.count(avg_pool3d_)) {
          attrs = MakeAvgPoolAttrs(param, call_node->attrs.as<AvgPool3DAttrs>());
        }
      }
      if (node_map.count(max_pool_)) {
        // Fold Padding and MaxPool only if pad_value is the min possible value for the dtype
        auto min_value = tvm::min_value(tvm::runtime::DataType(pad_value->data->dtype));
        const FloatImmNode* maybe_min_float = min_value.as<FloatImmNode>();
        const IntImmNode* maybe_min_int = min_value.as<IntImmNode>();

        if ((maybe_min_float && pad_scalar == maybe_min_float->value) ||
            (maybe_min_int && pad_scalar == maybe_min_int->value)) {
          if (node_map.count(max_pool1d_)) {
            attrs = MakePoolAttrs(param, call_node->attrs.as<MaxPool1DAttrs>());
          } else if (node_map.count(max_pool2d_)) {
            attrs = MakePoolAttrs(param, call_node->attrs.as<MaxPool2DAttrs>());
          } else if (node_map.count(max_pool3d_)) {
            attrs = MakePoolAttrs(param, call_node->attrs.as<MaxPool3DAttrs>());
          }
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
  DFPattern max_pool1d_;
  DFPattern max_pool2d_;
  DFPattern max_pool3d_;
  DFPattern max_pool_;
};

class SimplifyExplicitPadding {
 public:
  explicit SimplifyExplicitPadding(IRModule mod) : mod_(mod) {
    CreateCallback(SimplifyExplicitPad());
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
