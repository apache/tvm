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
 * \file src/relay/transforms/simplify_expr.cc
 * \brief A pass for simplifying the Relay expression.
 */

#include "simplify_expr.h"

#include <tvm/relay/dataflow_matcher.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>
#include <tvm/runtime/logging.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <stack>
#include <string>
#include <utility>

#include "../op/tensor/transform.h"
#include "fold_constant.h"
#include "pattern_utils.h"

namespace tvm {
namespace relay {

/*!
 * \brief SimplifyReshape matches the pattern of consecutive reshape or reverse_reshape ops,
 *   and merges into one reshape op.
 */
class SimplifyReshape : public DFPatternRewrite {
 public:
  SimplifyReshape() {
    x_ = IsWildcard();
    auto reshape1 = IsOp("reshape") || IsOp("contrib_reverse_reshape");
    auto reshape2 = IsOp("reshape") || IsOp("contrib_reverse_reshape");
    pattern_ = reshape1({reshape2({x_})});
  }

  Expr Callback(const Expr& pre, const Expr& post,
                const Map<DFPattern, Array<Expr>>& node_map) const override {
    auto x = node_map[x_][0];
    bool const_shape = true;
    Array<Integer> newshape;
    for (auto dim : Downcast<TensorType>(pre->checked_type())->shape) {
      if (dim.as<IntImmNode>() == nullptr) {
        const_shape = false;
        break;
      }
      newshape.push_back(Downcast<Integer>(dim));
    }
    if (const_shape) {
      return MakeReshape(x, newshape);
    }
    return post;
  }

 private:
  /*! \brief Pattern input */
  DFPattern x_;
};

double max_value_of_dtype(DataType dtype) {
  double max_value;
  if (!(dtype.is_float() || dtype.is_bfloat16() || dtype.is_int() || dtype.is_uint())) {
    LOG(INFO) << "The max_value of DataType cannot be decided, return "
                 "std::numeric_limits<double>::max().";
    return std::numeric_limits<double>::max();
  }

  auto prim_expr = tvm::max_value(dtype);
  if (prim_expr.as<IntImmNode>()) {
    max_value = static_cast<double>(prim_expr.as<IntImmNode>()->value);
  } else if (prim_expr.as<FloatImmNode>()) {
    max_value = prim_expr.as<FloatImmNode>()->value;
  } else {
    max_value = std::numeric_limits<double>::max();
  }
  return max_value;
}

double min_value_of_dtype(DataType dtype) {
  double min_value;
  if (!(dtype.is_float() || dtype.is_bfloat16() || dtype.is_int() || dtype.is_uint())) {
    LOG(INFO) << "The min_value of DataType cannot be decided, return "
                 "std::numeric_limits<double>::min().";
    return std::numeric_limits<double>::min();
  }
  auto prim_expr = tvm::min_value(dtype);
  if (prim_expr.as<IntImmNode>()) {
    min_value = static_cast<double>(prim_expr.as<IntImmNode>()->value);
  } else if (prim_expr.as<FloatImmNode>()) {
    min_value = prim_expr.as<FloatImmNode>()->value;
  } else {
    min_value = std::numeric_limits<double>::min();
  }
  return min_value;
}

/*!
 * \brief SimplifyClipAndCast removes redundant Clip and Cast.
 *
 * Example:
 *   %1 = cast(%0, dtype="uint8") [type=uint8]
 *   %2 = clip(%1, a_min=0f, a_max=255f) [type=int8]
 *
 * Optimized to (remove Clip):
 *   %1 = cast(%0, dtype="uint8") [type=uint8]
 *
 * Example:
 *   %0 == [type=int32]
 *   %1 = clip(%0, a_min=0f, a_max=255f) [type=int32]
 *   %2 = cast(%1, dtype="uint8") [type=uint8]
 *   %3 = cast(%2, dtype="int32") [type=int32]
 *
 * Optimized to (both casts can be removed):
 *   %1 = clip(%0, a_min=0f, a_max=255f) [type=int32]
 */
class SimplifyClipAndCast : public DFPatternRewrite {
 public:
  SimplifyClipAndCast() {
    x_ = IsWildcard();

    round_ = IsOp("round");
    round_ = round_ || IsOp("floor");
    round_ = round_ || IsOp("ceil");
    round_ = round_ || IsOp("trunc");
    round_pat_ = round_({x_}) || x_;

    add_ = (IsOp("add") || IsOp("subtract"));
    add_weight_ = IsConstant();
    add_pat_ = add_({round_pat_, add_weight_}) || round_pat_;

    clip_pat_ = (IsOp("clip") || IsOp("cast"))({add_pat_});
    clip_pat_ = clip_pat_ || IsOp("cast_like")({add_pat_, IsWildcard()});
    clip_pat_ = clip_pat_ || IsOp("reshape")({clip_pat_});

    ObjectPtr<AltPatternNode> alt_pat_ptr_ = make_object<AltPatternNode>();
    alt_pat_ = AltPattern(alt_pat_ptr_);
    alt_pat_ptr_->right = clip_pat_;

    cast_pat_ = IsOp("cast")({alt_pat_}) || IsOp("cast_like")({alt_pat_, IsWildcard()});
    cast_pat_ = cast_pat_ || IsOp("reshape")({cast_pat_});
    alt_pat_ptr_->left = cast_pat_;

    pattern_ = alt_pat_;
  }

  /* In this pass, it is important to distinguish if a tensor is a floating number tensor or an
   * integer tensor, which we name the SimpleDataType of a tensor. We consider the SimpleDataType
   * of a tensor in two different senses. One is nominal datatype, that is the datatype annotated
   * by the computational graph. The other one is essential datatype, a tensor has an integral
   * essential_datatype iif if can converted integer without loss of precision. For example, 5.0
   * has floating_point nominal_datatype, and integral essential datatype. In general, a tensor
   * after "round" OP has integral essential_datatype; a tensor x after x --> cast(int) -->
   * cast(float) has integral_essential datatype
   */
  enum class SimpleDataType : int {
    floating_point = 0,
    integral = 1,
  };

  Expr Callback(const Expr& pre, const Expr& post,
                const Map<DFPattern, Array<Expr>>& node_map) const override {
    auto is_floating_point = [&](Expr expr) {
      DataType dtype = Downcast<TensorType>(expr->checked_type())->dtype;
      return dtype.is_float() || dtype.is_bfloat16();
    };

    Expr x = node_map[x_][0];

    // A sequence of cast operators effectively has the following effects:
    // 1. convert tensor to the last cast operator's DataType.
    // 2. set the max_value/min_value of the tensor.
    // 3. set the essential_datatype of the tensor.
    // The following code determine the above three effects of the sequence of cast operators.
    DataType target_datatype = Downcast<TensorType>(post->checked_type())->dtype;
    std::vector<int> target_shape{};
    for (size_t i = 0; i < Downcast<TensorType>(post->checked_type())->shape.size(); ++i) {
      PrimExpr shape_i = Downcast<TensorType>(post->checked_type())->shape[i];
      if (shape_i.as<AnyNode>()) {
        target_shape.push_back(-1);
      } else {
        ICHECK(shape_i.as<IntImmNode>());
        target_shape.push_back(shape_i.as<IntImmNode>()->value);
      }
    }

    double target_max_value = max_value_of_dtype(target_datatype);
    double target_min_value = min_value_of_dtype(target_datatype);
    SimpleDataType target_essential_datatype = SimpleDataType::floating_point;

    std::vector<DataType> all_cast_dtypes{};
    Expr current_expr = post;
    while (current_expr != x && current_expr.as<CallNode>() &&
           (current_expr.as<CallNode>()->op.as<OpNode>()->name == "cast" ||
            current_expr.as<CallNode>()->op.as<OpNode>()->name == "cast_like" ||
            current_expr.as<CallNode>()->op.as<OpNode>()->name == "reshape")) {
      if (current_expr.as<CallNode>()->op.as<OpNode>()->name != "reshape") {
        DataType current_datatype = Downcast<TensorType>(current_expr->checked_type())->dtype;
        all_cast_dtypes.push_back(current_datatype);
        target_max_value = std::min(target_max_value, max_value_of_dtype(current_datatype));
        target_min_value = std::max(target_min_value, min_value_of_dtype(current_datatype));
        if (!is_floating_point(current_expr)) {
          target_essential_datatype = SimpleDataType::integral;
        }
      }
      current_expr = current_expr.as<CallNode>()->args[0];
    }

    // push all CalllNode before "cast" into a stack
    std::stack<Expr> expr_stack{};
    while (current_expr != x) {
      expr_stack.push(current_expr);
      if ((Downcast<Op>(current_expr.as<CallNode>()->op) == Op::Get("add") ||
           Downcast<Op>(current_expr.as<CallNode>()->op) == Op::Get("subtract")) &&
          current_expr.as<CallNode>()->args[0].as<ConstantNode>() &&
          !current_expr.as<CallNode>()->args[1].as<ConstantNode>()) {
        current_expr = current_expr.as<CallNode>()->args[1];
      } else {
        current_expr = current_expr.as<CallNode>()->args[0];
      }
    }

    // the datatype of current_expr.
    SimpleDataType essential_datatype;
    // TODO(kfeng123) This InferType is required. But why?
    DataType x_dtype = Downcast<TensorType>(InferType(x)->checked_type())->dtype;
    if (x_dtype.is_float() || x_dtype.is_bfloat16()) {
      essential_datatype = SimpleDataType::floating_point;
    } else {
      essential_datatype = SimpleDataType::integral;
    }

    // the min/max possible value of current_expr
    double max_value{max_value_of_dtype(x_dtype)}, min_value{min_value_of_dtype(x_dtype)};

    // rewrite the OP's before cast
    current_expr = x;
    // LOG(WARNING) << "post: " << AsText(post, false);
    // LOG(WARNING) << "x: " << AsText(x, false);
    while (!expr_stack.empty()) {
      Expr the_expr = expr_stack.top();
      expr_stack.pop();
      ICHECK(the_expr.as<CallNode>());
      ICHECK(the_expr.as<CallNode>()->op.as<OpNode>());
      // LOG(WARNING) << "the_expr: " << AsText(the_expr, false);

      // String op_name = the_expr.as<CallNode>()->op.as<OpNode>()->name;

      Op op = Downcast<Op>(the_expr.as<CallNode>()->op);

      if (op == Op::Get("round") || op == Op::Get("floor") || op == Op::Get("ceil") ||
          op == Op::Get("trunc")) {
        if (essential_datatype == SimpleDataType::integral) {
          // In this case, the op can be removed
        } else {
          current_expr = relay::Call(the_expr.as<CallNode>()->op, {current_expr},
                                     the_expr.as<CallNode>()->attrs);
        }
        essential_datatype = SimpleDataType::integral;
      } else if (op == Op::Get("add") || op == Op::Get("subtract")) {
        //  true: left matches pattern, right is constant. false: left is constant, right matches
        //  pattern.
        LOG(INFO) << "yaya x";
        bool if_left_matches_pattern;
        if (the_expr.as<CallNode>()->args[0].as<ConstantNode>() &&
            !the_expr.as<CallNode>()->args[1].as<ConstantNode>()) {
          if_left_matches_pattern = false;
        } else {
          if_left_matches_pattern = true;
        }
        LOG(INFO) << "yaya xx";
        Expr constant_expr = if_left_matches_pattern ? the_expr.as<CallNode>()->args[1]
                                                     : the_expr.as<CallNode>()->args[0];

        if (if_left_matches_pattern) {
          LOG(INFO) << "yaya l";
          current_expr = relay::Call(the_expr.as<CallNode>()->op, {current_expr, constant_expr},
                                     the_expr.as<CallNode>()->attrs);
        } else {
          LOG(INFO) << "yaya r";
          current_expr = relay::Call(the_expr.as<CallNode>()->op, {constant_expr, current_expr},
                                     the_expr.as<CallNode>()->attrs);
        }
        if (essential_datatype == SimpleDataType::integral && is_floating_point(constant_expr)) {
          LOG(INFO) << "yaya r";
          // in this case, the essential_datatype of current_expr is integral iff args[1] has
          // integral essential_datatype
          TensorType const_tensor_type = Downcast<TensorType>(constant_expr->checked_type());
          DataType dtype = const_tensor_type->dtype;
          size_t data_size = const_tensor_type->Size().as<tir::IntImmNode>()->value;
          if (dtype.is_float() && dtype.bits() == 32) {  // float32
            LOG(INFO) << "yaya r1";
            void* ptr = constant_expr.as<ConstantNode>()->data->data;
            float* the_const_ptr = reinterpret_cast<float*>(ptr);
            if (data_size > 1) {  // Too slow to examine a large tensor
              essential_datatype = SimpleDataType::floating_point;
            } else {
              LOG(INFO) << "yaya r2";
              for (size_t i = 0; i < data_size; ++i) {
                if (the_const_ptr[i] != std::round(the_const_ptr[i])) {
                  essential_datatype = SimpleDataType::floating_point;
                  break;
                }
              }
            }
          } else {  // TODO(kfeng123) deal with the essential_datatype in the cases of float16 and
                    // bfloat16
            essential_datatype = SimpleDataType::floating_point;
          }
        }
      } else if (op == Op::Get("clip")) {
        double a_max = the_expr.as<CallNode>()->attrs.as<ClipAttrs>()->a_max;
        double a_min = the_expr.as<CallNode>()->attrs.as<ClipAttrs>()->a_min;
        if (a_max < max_value || a_min > min_value) {
          current_expr = relay::Call(the_expr.as<CallNode>()->op, {current_expr},
                                     the_expr.as<CallNode>()->attrs);
          max_value = std::min(max_value, a_max);
          min_value = std::max(min_value, a_min);
        }  // else this clip op can be removed
      }
    }

    LOG(INFO) << "yaya 1";
    // rewrite "cast" OPs
    std::vector<DataType> preserved_casts{};

    if (target_max_value < max_value_of_dtype(target_datatype) && target_max_value < max_value) {
      for (const DataType& dtype : all_cast_dtypes) {
        if (target_max_value == max_value_of_dtype(dtype)) {
          max_value = target_max_value;
          min_value = std::max(min_value, min_value_of_dtype(dtype));
          if (dtype.is_int() || dtype.is_uint()) {
            essential_datatype = SimpleDataType::integral;
          }
          preserved_casts.push_back(dtype);
          break;
        }
      }
    }

    LOG(INFO) << "yaya 2";
    if (target_min_value > min_value_of_dtype(target_datatype) && target_min_value > min_value) {
      for (const DataType& dtype : all_cast_dtypes) {
        if (target_min_value == min_value_of_dtype(dtype)) {
          min_value = target_min_value;
          max_value = std::min(max_value, max_value_of_dtype(dtype));
          if (dtype.is_int() || dtype.is_uint()) {
            essential_datatype = SimpleDataType::integral;
          }
          if (std::find(preserved_casts.begin(), preserved_casts.end(), dtype) ==
              std::end(preserved_casts)) {  // This if will always be true
            preserved_casts.push_back(dtype);
          }
          break;
        }
      }
    }

    LOG(INFO) << "yaya 3";
    if (target_datatype.is_int() || target_datatype.is_uint()) {
      essential_datatype = SimpleDataType::integral;
    }

    LOG(INFO) << "yaya 4";
    if (essential_datatype != SimpleDataType::integral &&
        target_essential_datatype == SimpleDataType::integral) {
      for (const DataType& dtype : all_cast_dtypes) {
        if (dtype.is_int() || dtype.is_uint()) {
          if (std::find(preserved_casts.begin(), preserved_casts.end(), dtype) ==
              std::end(preserved_casts)) {
            preserved_casts.push_back(dtype);
          }
          break;
        }
      }
    }

    LOG(INFO) << "yaya 5";
    for (const auto& dtype : preserved_casts) {
      if (dtype != target_datatype) {
        auto attrs = make_object<CastAttrs>();
        attrs->dtype = dtype;
        current_expr = relay::Call(Op::Get("cast"), {current_expr}, Attrs(attrs));
      }
    }

    LOG(INFO) << "yaya 6";
    TensorType last_tensor_type = Downcast<TensorType>(InferType(current_expr)->checked_type());
    if (last_tensor_type->dtype != target_datatype) {
      auto attrs = make_object<CastAttrs>();
      attrs->dtype = target_datatype;
      current_expr = relay::Call(Op::Get("cast"), {current_expr}, Attrs(attrs));
    }

    LOG(INFO) << "yaya 7";
    std::vector<int> last_shape{};
    for (size_t i = 0; i < last_tensor_type->shape.size(); ++i) {
      if (last_tensor_type->shape[i].as<AnyNode>()) {
        last_shape.push_back(-1);
      } else {
        last_shape.push_back(last_tensor_type->shape[i].as<IntImmNode>()->value);
      }
    }

    LOG(INFO) << "yaya 8";
    if (last_shape != target_shape) {
      auto attrs = make_object<ReshapeAttrs>();
      for (const auto& shape_item : target_shape) {
        attrs->newshape.push_back(shape_item);
      }
      current_expr = relay::Call(Op::Get("reshape"), {current_expr}, Attrs(attrs));
    }

    LOG(INFO) << "end";
    return current_expr;
  }

 protected:
  DFPattern x_, round_, round_pat_, add_, add_weight_, add_pat_, clip_pat_, cast_, cast_pat_,
      alt_pat_;
};

class SimplifyCastAndTranspose : public DFPatternRewrite {
 public:
  SimplifyCastAndTranspose() {
    x_ = IsWildcard();
    cast_ = IsOp("cast");
    cast_pat_ = cast_({x_});

    ObjectPtr<AltPatternNode> alt_pat_ptr_ = make_object<AltPatternNode>();
    alt_pat_ = DFPattern(alt_pat_ptr_);
    alt_pat_ptr_->left = cast_pat_;

    add_weight_ = IsConstant();
    uni_pat_ = IsOp("clip")({alt_pat_});
    uni_pat_ = uni_pat_ || IsOp("add")({alt_pat_, add_weight_});
    uni_pat_ = uni_pat_ || IsOp("subtract")({alt_pat_, add_weight_});

    alt_pat_ptr_->right = uni_pat_;

    pattern_ = (IsOp("transpose") || IsOp("layout_transform"))({alt_pat_});
  }

  Expr Callback(const Expr& pre, const Expr& post,
                const Map<DFPattern, Array<Expr>>& node_map) const override {
    Expr x = node_map[x_][0];

    int input_bits = Downcast<TensorType>(x->checked_type())->dtype.bits();
    int output_bits = Downcast<TensorType>(post->checked_type())->dtype.bits();
    if (output_bits <= input_bits) {
      return post;
    }

    Expr current_expr = post;
    std::stack<Expr> expr_stack{};
    while (current_expr != x) {
      expr_stack.push(current_expr);
      if (node_map.count(add_weight_) != 0 &&
          std::find(node_map[add_weight_].begin(), node_map[add_weight_].end(),
                    current_expr.as<CallNode>()->args[0]) != node_map[add_weight_].end()) {
        current_expr = current_expr.as<CallNode>()->args[1];
      } else {
        current_expr = current_expr.as<CallNode>()->args[0];
      }
    }

    // rewrite the OP's
    current_expr = x;
    current_expr = relay::Call(node_map[pattern_][0].as<CallNode>()->op, {x},
                               Attrs(node_map[pattern_][0].as<CallNode>()->attrs));
    while (!expr_stack.empty()) {
      Expr the_expr = expr_stack.top();
      expr_stack.pop();
      Op op = Downcast<Op>(the_expr.as<CallNode>()->op);

      if (op == Op::Get("add") || op == Op::Get("subtract")) {
        if (std::find(node_map[add_weight_].begin(), node_map[add_weight_].end(),
                      the_expr.as<CallNode>()->args[0]) != node_map[add_weight_].end()) {
          current_expr = relay::Call(the_expr.as<CallNode>()->op,
                                     {the_expr.as<CallNode>()->args[0], current_expr},
                                     the_expr.as<CallNode>()->attrs);
        } else {
          current_expr = relay::Call(the_expr.as<CallNode>()->op,
                                     {current_expr, the_expr.as<CallNode>()->args[1]},
                                     the_expr.as<CallNode>()->attrs);
        }
      } else if (op != Op::Get("transpose") && op != Op::Get("layout_transform")) {
        current_expr = relay::Call(the_expr.as<CallNode>()->op, {current_expr},
                                   the_expr.as<CallNode>()->attrs);
      }
    }

    return current_expr;
  }

 protected:
  DFPattern x_, cast_, cast_pat_, uni_pat_, add_weight_, add_pat_, clip_pat_, alt_pat_;
};

/*!
 * \brief Return the axis order for layout transform and transpose
 * ops.
 */
static std::vector<int> GetTransposeAxisOrder(const Call& call, int ndim) {
  std::vector<int> attr_axes;
  if (auto attr = call->attrs.as<TransposeAttrs>()) {
    if (attr->axes.defined()) {
      for (int i = 0; i < ndim; ++i) {
        int64_t axis = attr->axes[i].IntValue();
        axis += (axis < 0) ? ndim : 0;
        attr_axes.push_back(axis);
      }
    } else {
      // Empty axes means reverse
      for (int i = ndim - 1; i >= 0; --i) {
        attr_axes.push_back(i);
      }
    }
  } else if (auto attr = call->attrs.as<LayoutTransformAttrs>()) {
    Layout src_layout(attr->src_layout);
    Layout dst_layout(attr->dst_layout);
    for (int i = 0; i < ndim; ++i) {
      attr_axes.push_back(src_layout.IndexOf(dst_layout[i]));
    }
  } else {
    CHECK(false) << "Expected transpose or layout_transform, but got "
                 << Downcast<Op>(call->op)->name;
  }
  return std::move(attr_axes);
}

/*!
 * \brief SimplifyTranspose matches the pattern of consecutive transpose op,
 *   and merges or cancels them.
 */
class SimplifyTranspose : public DFPatternRewrite {
 public:
  SimplifyTranspose() {
    x_ = IsWildcard();
    auto trans1 = IsOp("transpose") || IsOp("layout_transform");
    auto trans2 = IsOp("transpose") || IsOp("layout_transform");
    pattern_ = trans1({trans2({x_})});
  }

  Expr Callback(const Expr& pre, const Expr& post,
                const Map<DFPattern, Array<Expr>>& node_map) const override {
    auto x = node_map[x_][0];

    Call trans_call = Downcast<Call>(post);

    // Try to fuse any rank changing layout transformations
    if (auto layout_trans = FoldRankChangingLayoutTrans(x, trans_call)) {
      if (auto attr = layout_trans.value()->attrs.as<LayoutTransformAttrs>()) {
        // Prune any trivial layout transformation
        if (attr->src_layout == attr->dst_layout) {
          return x;
        }
      }
      return layout_trans.value();
    }

    // Initialize axes
    int ndim = Downcast<TensorType>(pre->checked_type())->shape.size();
    Array<Integer> axes;
    for (int i = 0; i < ndim; ++i) {
      axes.push_back(i);
    }

    // Collect axes changes from the matched pattern, including two consecutive transposes.
    std::vector<std::vector<int>> interm_axes;
    interm_axes.push_back(GetTransposeAxisOrder(trans_call, ndim));
    trans_call = Downcast<Call>(trans_call->args[0]);
    interm_axes.push_back(GetTransposeAxisOrder(trans_call, ndim));

    // Calculate the final axes in reverse order (from root to output)
    auto it = interm_axes.rbegin();
    while (it != interm_axes.rend()) {
      auto interm = *it;

      Array<Integer> new_axes;
      for (int i = 0; i < ndim; ++i) {
        new_axes.push_back(axes[interm[i]]);
      }
      axes = new_axes;
      it++;
    }

    return MakeTranspose(x, axes);
  }

  String PermuteLayout(const String& layout, std::vector<int> axes_order) const {
    std::string new_layout{};
    std::string old_layout{layout};
    ICHECK_EQ(axes_order.size(), layout.size())
        << "Number of axes must match the number of named axes in the layout to permute: length("
        << old_layout << ") != " << axes_order.size();
    std::stringstream order;
    for (auto axis : axes_order) {
      new_layout += old_layout[axis];
      order << axis << ", ";
    }
    DLOG(INFO) << "Using transpose axes order {" << order.str()
               << "} to permute layout: " << old_layout << " to " << new_layout;
    return new_layout;
  }

  struct RankChangingLayoutDescriptor {
    Layout src_layout;
    Layout dst_layout;
    // Either a rank changing layout transform or a transpose
    Call other_transform;
  };

  std::unique_ptr<RankChangingLayoutDescriptor> GetRankChangeDescriptor(const Call& call) const {
    std::unique_ptr<RankChangingLayoutDescriptor> desc{nullptr};
    if (auto attr = call->attrs.as<LayoutTransformAttrs>()) {
      if (attr->src_layout.length() != attr->dst_layout.length()) {
        desc = std::make_unique<RankChangingLayoutDescriptor>();
        desc->src_layout = Layout(attr->src_layout);
        desc->dst_layout = Layout(attr->dst_layout);
        desc->other_transform = Downcast<Call>(call->args[0]);
      }
    }
    if (auto attr = Downcast<Call>(call->args[0])->attrs.as<LayoutTransformAttrs>()) {
      if (attr->src_layout.length() != attr->dst_layout.length()) {
        if (!desc) {
          desc = std::make_unique<RankChangingLayoutDescriptor>();
          desc->src_layout = Layout(attr->src_layout);
          desc->dst_layout = Layout(attr->dst_layout);
          desc->other_transform = call;
        } else {
          ICHECK(desc->src_layout->name == attr->dst_layout)
              << "Back-to-back layout transforms must have the same intermediate layout: "
              << desc->src_layout->name << " != " << attr->dst_layout;
          desc->src_layout = Layout(attr->src_layout);
        }
      }
    }
    return desc;
  }

  /*
   * \brief Fuse call and it's argument into a single layout_transform operator
   * when either call or it's argument is a rang changing layout_transform, e.g.,
   *
   *  Simplify
   *
   *  [N, H, W, C] -> Transpose -> [N, C, H, W] -> LayoutTrans -> [N, C, H, W, 4c]
   *
   *  to,
   *
   *  [N, H, W, C] -> LayoutTrans -> [N, C, H, W, 4c].
   *
   * \param The input expression to the matched pattern
   * \param The pattern root; the second of two consecutive Transpose/LayoutTransform ops
   */
  Optional<Call> FoldRankChangingLayoutTrans(const Expr& data, const Call& call) const {
    // Check to see if either the first or second call in matched pattern
    // is a rank changing layout transform. If so, return a descriptor containing
    // the layouts and any additional transpose or layout transform op.
    auto desc = GetRankChangeDescriptor(call);
    if (desc == nullptr) {
      // No rank changing layout transform
      return Optional<Call>{nullptr};
    }

    Optional<Expr> output_layout_trans;
    // Fuse a rank increasing layout transform and a preceeding transpose
    if (desc->src_layout->axes.size() < desc->dst_layout->axes.size()) {
      auto axes = GetTransposeAxisOrder(desc->other_transform, desc->src_layout->axes.size());
      // Calculate the reverse axis order and apply to the source layout
      std::vector<int> inverse(axes.size());
      for (size_t i = 0; i < axes.size(); i++) {
        inverse[axes[i]] = i;
      }
      String new_layout = PermuteLayout(desc->src_layout->name, inverse);
      output_layout_trans = MakeLayoutTransform(data, new_layout, desc->dst_layout->name);
      // Fuse a rank descreasing layout transform followed by a transpose
    } else if (desc->src_layout->axes.size() > desc->dst_layout->axes.size()) {
      auto axes = GetTransposeAxisOrder(desc->other_transform, desc->dst_layout->axes.size());
      String new_layout = PermuteLayout(desc->dst_layout->name, axes);
      output_layout_trans = MakeLayoutTransform(data, desc->src_layout->name, new_layout);
      // Fuse two back-to-back layout transformations which change rank
    } else if (desc->other_transform->attrs.as<LayoutTransformAttrs>()) {
      output_layout_trans =
          MakeLayoutTransform(data, desc->src_layout->name, desc->dst_layout->name);
    }
    return Downcast<Call>(output_layout_trans);
  }

 private:
  /*! \brief Pattern input */
  DFPattern x_;
};

/*!
 * \brief SimplifyNoOpTranspose matches the pattern of transpose or
 *  layout transform ops which do not change the layout or rank and
 *  removes the op.
 */
class SimplifyNoOpTranspose : public DFPatternRewrite {
 public:
  SimplifyNoOpTranspose() {
    x_ = IsWildcard();
    auto trans1 = IsOp("transpose") || IsOp("layout_transform");
    pattern_ = trans1({x_});
  }

  Expr Callback(const Expr& pre, const Expr& post,
                const Map<DFPattern, Array<Expr>>& node_map) const override {
    auto x = node_map[x_][0];
    Call trans_call = Downcast<Call>(post);

    // Do not remove ops which change rank
    if (auto attr = trans_call->attrs.as<LayoutTransformAttrs>()) {
      if (attr->src_layout != attr->dst_layout) {
        return post;
      }
    }

    int ndim = Downcast<TensorType>(pre->checked_type())->shape.size();
    auto axes = GetTransposeAxisOrder(trans_call, ndim);

    bool need_transpose = false;
    for (int i = 0; i < ndim; ++i) {
      if (axes[i] != i) {
        need_transpose = true;
        break;
      }
    }

    if (!need_transpose) return x;

    return post;
  }

 private:
  /*! \brief Pattern input */
  DFPattern x_;
};

/*!
 * \brief FullElementwise finds full like ops followed by broadcasting ops, and eliminates
 * the full op by directly passing the fill value into the broadcasting op.
 */
class FullElementwise : public DFPatternRewrite {
 public:
  FullElementwise() {
    x_ = IsWildcard();
    data_ = IsWildcard();
    value_ = IsConstant();

    full_ = IsOp("full")({value_}) || IsOp("full_like")({data_, value_});
    ones_ = IsOp("ones")({}) || IsOp("ones_like")({data_});
    zeros_ = IsOp("zeros")({}) || IsOp("zeros_like")({data_});

    Map<String, ObjectRef> attrs;
    attrs.Set("TOpPattern", Integer(static_cast<int>(kBroadcast)));
    DFPattern op = IsWildcard().HasAttr(attrs);
    DFPattern full = full_ || ones_ || zeros_;
    pattern_ = op({full, x_}) || op({x_, full});
  }

  Expr Callback(const Expr& pre, const Expr& post,
                const Map<DFPattern, Array<Expr>>& node_map) const override {
    const CallNode* call = pre.as<CallNode>();
    ICHECK(call);
    Type pre_type = pre->checked_type_;
    ICHECK(pre_type.as<TensorTypeNode>());
    auto dtype = pre_type.as<TensorTypeNode>()->dtype;
    auto x = node_map[x_][0];
    bool is_left = post.as<CallNode>()->args[1] == x;
    Type x_type;
    if (is_left) {
      x_type = call->args[1]->checked_type_;
    } else {
      x_type = call->args[0]->checked_type_;
    }

    if (StructuralEqual()(x_type, pre_type)) {
      Expr value;
      if (node_map.count(full_)) {
        value = node_map[value_][0];
        ICHECK(IsConstScalar(value));
      } else if (node_map.count(ones_)) {
        value = MakeConstantScalar(dtype, 1);
      } else if (node_map.count(zeros_)) {
        value = MakeConstantScalar(dtype, 0);
      } else {
        ICHECK(false) << "Didn't find a full op while matching full + elementwise";
      }
      if (is_left) {
        return Call(call->op, {value, x}, call->attrs, call->type_args, call->span);
      } else {
        return Call(call->op, {x, value}, call->attrs, call->type_args, call->span);
      }
    }
    return post;
  }

 private:
  /*! \brief binary argument */
  DFPattern x_;
  /*! \brief data ops get shape from */
  DFPattern data_;
  /*! \brief constant input */
  DFPattern value_;
  /*! \brief full op */
  DFPattern full_;
  /*! \brief ones op */
  DFPattern ones_;
  /*! \brief zeros op */
  DFPattern zeros_;
};

/*!
 * \brief Converts `*_like` operators to their explicit shape equivalent (e.g. `zeros_like(x, y)` to
 * `zeros(x, y.shape)`), when the target shape is concrete. This removes unnecessary dependencies
 * and can enable more opportunities for operator fusion.
 */
class ConcretizeLikeRewrite : public DFPatternRewrite {
 public:
  explicit ConcretizeLikeRewrite(const Op& op) {
    ICHECK(op->num_inputs == 1 || op->num_inputs == 2)
        << "ConcretizeLike does not handle operators that aren't unary or binary, got: " << op;
    like_pat_ = IsWildcard();
    data_pat_ = IsWildcard();
    if (op->num_inputs == 1) {
      pattern_ = IsExpr(op)({like_pat_});
    } else {
      pattern_ = IsExpr(op)({data_pat_, like_pat_});
    }
  }

  virtual bool Check(const Expr& pre, const Expr& post,
                     const Map<DFPattern, Array<Expr>>& node_map) const {
    const CallNode* call_node = pre.as<CallNode>();
    ICHECK(call_node);

    if (!call_node->checked_type().as<TensorTypeNode>()) {
      return false;
    }

    return true;
  }

  virtual Expr Concretize(const Map<DFPattern, Array<Expr>>& node_map, Array<Integer> shape,
                          DataType dtype) const = 0;

  Expr Callback(const Expr& pre, const Expr& post,
                const Map<DFPattern, Array<Expr>>& node_map) const override {
    if (!Check(pre, post, node_map)) {
      return post;
    }

    const TensorTypeNode* like_ty = pre->checked_type().as<TensorTypeNode>();
    Array<Integer> cshape;
    for (const auto& dim : like_ty->shape) {
      if (auto imm = dim.as<IntImm>()) {
        cshape.push_back(Integer(imm.value()));
      } else {
        // shape is not static, don't concretize
        return post;
      }
    }

    return Concretize(node_map, cshape, like_ty->dtype);
  }

 protected:
  DFPattern data_pat_;
  DFPattern like_pat_;
};

class ConcretizeZerosLikeRewrite : public ConcretizeLikeRewrite {
 public:
  ConcretizeZerosLikeRewrite() : ConcretizeLikeRewrite(Op::Get("zeros_like")) {}

  Expr Concretize(const Map<DFPattern, Array<Expr>>& node_map, Array<Integer> shape,
                  DataType dtype) const override {
    return MakeZeros(shape, dtype);
  }
};

class ConcretizeOnesLikeRewrite : public ConcretizeLikeRewrite {
 public:
  ConcretizeOnesLikeRewrite() : ConcretizeLikeRewrite(Op::Get("ones_like")) {}

  Expr Concretize(const Map<DFPattern, Array<Expr>>& node_map, Array<Integer> shape,
                  DataType dtype) const override {
    return MakeOnes(shape, dtype);
  }
};

class ConcretizeFullLikeRewrite : public ConcretizeLikeRewrite {
 public:
  ConcretizeFullLikeRewrite() : ConcretizeLikeRewrite(Op::Get("full_like")) {}

  Expr Concretize(const Map<DFPattern, Array<Expr>>& node_map, Array<Integer> shape,
                  DataType dtype) const override {
    // `like_pat_` here is `fill_value`
    return MakeFull(node_map[like_pat_][0], shape, dtype);
  }
};

class ConcretizeReshapeLikeRewrite : public ConcretizeLikeRewrite {
 public:
  ConcretizeReshapeLikeRewrite() : ConcretizeLikeRewrite(Op::Get("reshape_like")) {}

  Expr Concretize(const Map<DFPattern, Array<Expr>>& node_map, Array<Integer> shape,
                  DataType dtype) const override {
    return MakeReshape(node_map[data_pat_][0], shape);
  }
};

class ConcretizeCollapseSumLikeRewrite : public ConcretizeLikeRewrite {
 public:
  ConcretizeCollapseSumLikeRewrite() : ConcretizeLikeRewrite(Op::Get("collapse_sum_like")) {}

  Expr Concretize(const Map<DFPattern, Array<Expr>>& node_map, Array<Integer> shape,
                  DataType dtype) const override {
    ICHECK_LE(shape.size(), std::numeric_limits<int64_t>::max());
    static const Op& op = Op::Get("collapse_sum_to");
    auto attrs = make_object<InitOpAttrs>();
    attrs->shape = shape;
    std::vector<int64_t> s;
    std::transform(shape.begin(), shape.end(), std::back_inserter(s),
                   [](Integer i) { return i.IntValue(); });
    auto cshape = MakeConstantTensor(DataType::Int(32), {static_cast<int64_t>(shape.size())}, s);
    return Call(op, {node_map[data_pat_][0], cshape}, Attrs(attrs));
  }
};

class ConcretizeBroadcastToLikeRewrite : public ConcretizeLikeRewrite {
 public:
  ConcretizeBroadcastToLikeRewrite() : ConcretizeLikeRewrite(Op::Get("broadcast_to_like")) {}

  Expr Concretize(const Map<DFPattern, Array<Expr>>& node_map, Array<Integer> shape,
                  DataType dtype) const override {
    return MakeBroadCastTo(node_map[data_pat_][0], shape);
  }
};

/*!
 * \brief Converts cast_like operator to cast. Not inheriting from ConcretizeLikeRewrite
 * because even if shape is not static, still can concretize.
 */
class ConcretizeCastLikeRewrite : public DFPatternRewrite {
 public:
  ConcretizeCastLikeRewrite() {
    data_pat_ = IsWildcard();
    like_pat_ = IsWildcard();
    pattern_ = IsOp("cast_like")({data_pat_, like_pat_});
  }

  Expr Callback(const Expr& pre, const Expr& post,
                const Map<DFPattern, Array<Expr>>& node_map) const override {
    const CallNode* call_node = pre.as<CallNode>();
    ICHECK(call_node);

    if (!call_node->checked_type().as<TensorTypeNode>()) {
      return post;
    }

    const TensorTypeNode* like_ty = pre->checked_type().as<TensorTypeNode>();
    return MakeCast(node_map[data_pat_][0], like_ty->dtype);
  }

 protected:
  DFPattern data_pat_;
  DFPattern like_pat_;
};

/*! \brief Eliminates expressions that are equivalent to identity. */
class EliminateIdentityRewrite : public DFPatternRewrite {
 public:
  EliminateIdentityRewrite() {
    x_ = IsWildcard();
    const_ = IsConstant();

    DFPattern add_op = IsOp("add");
    DFPattern mul_op = IsOp("multiply");
    DFPattern zeros_expr = IsOp("zeros")({}) || IsOp("zeros_like")({IsWildcard()}) || const_;
    DFPattern ones_expr = IsOp("ones")({}) || IsOp("ones_like")({IsWildcard()}) || const_;

    // add and multiply are commutative so we don't need another pattern for reversed args
    DFPattern add_id = add_op({x_, zeros_expr});
    DFPattern mul_id = mul_op({x_, ones_expr});

    DFPattern sub_id = IsOp("subtract")({x_, zeros_expr});
    DFPattern div_id = IsOp("divide")({x_, ones_expr});

    pattern_ = add_id || mul_id || sub_id || div_id;
  }

  bool CheckConstant(const OpNode* op, const ConstantNode* constant) const {
    if (!IsScalar(GetRef<Expr>(constant))) {
      return false;
    }
    auto value = TryToScalar(constant->data, 0);
    if (!value) {
      // unsupported dtype
      return false;
    }
    if (op->name == "add" || op->name == "subtract") {
      return value.value() == 0.0;
    } else if (op->name == "multiply" || op->name == "divide") {
      return value.value() == 1.0;
    }
    return false;
  }

  Expr Callback(const Expr& pre, const Expr& post,
                const Map<DFPattern, Array<Expr>>& node_map) const override {
    const CallNode* call = pre.as<CallNode>();
    ICHECK(call);
    Type pre_type = pre->checked_type_;
    ICHECK(pre_type.as<TensorTypeNode>());
    auto x = node_map[x_][0];
    bool is_left = post.as<CallNode>()->args[1] == x;
    Type x_type;
    if (is_left) {
      x_type = call->args[1]->checked_type_;
    } else {
      x_type = call->args[0]->checked_type_;
    }

    if (node_map.count(const_)) {
      // the other argument is a Constant in this case
      const ConstantNode* constant = node_map[const_][0].as<ConstantNode>();
      const OpNode* op = call->op.as<OpNode>();
      ICHECK(constant);
      ICHECK(op);
      if (!CheckConstant(op, constant)) {
        return post;
      }
    }

    if (StructuralEqual()(x_type, pre_type)) {
      return x;
    }

    return post;
  }

 private:
  DFPattern x_;
  DFPattern const_;
};

/*! \brief Switch adjacent add-mul with constants to mul-add.
 * As mul-add pattern is more friendly to FoldScaleAxis.
 */
class SwitchAddMultiply : public DFPatternRewrite {
 public:
  SwitchAddMultiply() {
    x_ = IsWildcard();
    c1_ = IsConstant();
    c2_ = IsConstant();
    pattern_ = (x_ + c1_) * c2_;
  }

  Expr Callback(const Expr& pre, const Expr& post,
                const Map<DFPattern, Array<Expr>>& node_map) const override {
    auto x = node_map[x_][0];
    auto c1 = node_map[c1_][0];
    auto c2 = node_map[c2_][0];

    if (x.as<ConstantNode>()) {
      return post;
    }

    Expr const_expr = Call(Op::Get("multiply"), {c1, c2});
    Expr const_val = transform::FoldConstantExpr(const_expr);

    return Call(Op::Get("add"), {Call(Op::Get("multiply"), {x, c2}), const_val});
  }

 private:
  DFPattern x_;
  DFPattern c1_;
  DFPattern c2_;
};

/*! \brief Simplify two adjacent multiply or add with constants for further constant folding.
 * The pattern matching supports commutative property.
 */
class SimplifyAdjacentMultiplyOrAdd : public DFPatternRewrite {
 public:
  SimplifyAdjacentMultiplyOrAdd() {
    x_ = IsWildcard();
    c1_ = IsConstant();
    c2_ = IsConstant();
    pattern_ = (x_ * c1_ * c2_) || (x_ + c1_ + c2_);
  }

  Expr Callback(const Expr& pre, const Expr& post,
                const Map<DFPattern, Array<Expr>>& node_map) const override {
    const CallNode* call = pre.as<CallNode>();
    auto x = node_map[x_][0];
    auto c1 = node_map[c1_][0];
    auto c2 = node_map[c2_][0];

    if (x.as<ConstantNode>()) {
      return post;
    }

    Expr const_expr = Call(call->op, {c1, c2});
    Expr const_val = transform::FoldConstantExpr(const_expr);

    return Call(call->op, {x, const_val});
  }

 private:
  DFPattern x_;
  DFPattern c1_;
  DFPattern c2_;
};

/*! \brief Simplifying x+x to x*2 */
class SimplifyAdd : public DFPatternRewrite {
 public:
  SimplifyAdd() {
    x_ = IsWildcard();
    y_ = IsWildcard();
    pattern_ = IsOp("add")({x_, y_});
  }

  Expr Callback(const Expr& pre, const Expr& post,
                const Map<DFPattern, Array<Expr>>& node_map) const override {
    Type pre_type = pre->checked_type_;
    auto dtype = pre_type.as<TensorTypeNode>()->dtype;
    auto x = node_map[x_][0];
    auto y = node_map[y_][0];
    auto data_type = Downcast<TensorType>(x->checked_type());

    if (x == y) {
      Expr value;
      value = MakeConstantScalar(dtype, 2);
      return InferType(Call(Op::Get("multiply"), {x, value}));
    }
    return post;
  }

 private:
  /*! \brief Pattern input */
  DFPattern x_;
  DFPattern y_;
};

/*! \brief Simplifying a * x * x + b * x * y + c * y * y to a * (x + p * y) * (x + q * y) */
class SimplifyBinomial : public DFPatternRewrite {
 public:
  SimplifyBinomial() {
    x_ = IsWildcard();
    y_ = IsWildcard();
    a_ = IsConstant();
    b_ = IsConstant();
    c_ = IsConstant();
    DFPattern add = IsOp("add");
    DFPattern mul = IsOp("multiply");
    DFPattern x_sq = mul({a_, mul({x_, x_})}) || mul({x_, mul({a_, x_})}) || mul({x_, x_});
    DFPattern xy = mul({b_, mul({x_, y_})}) || mul({x_, mul({b_, y_})}) ||
                   mul({y_, mul({b_, x_})}) || mul({x_, y_});
    DFPattern y_sq = mul({c_, mul({y_, y_})}) || mul({y_, mul({c_, y_})}) || mul({y_, y_});

    pattern_ = add({add({xy, x_sq}), y_sq}) || add({add({xy, y_sq}), x_sq}) ||
               add({add({x_sq, y_sq}), xy});
  }

  Expr Callback(const Expr& pre, const Expr& post,
                const Map<DFPattern, Array<Expr>>& node_map) const override {
    Type pre_type = pre->checked_type_;
    auto dtype = pre_type.as<TensorTypeNode>()->dtype;
    auto x = node_map[x_][0];
    auto y = node_map[y_][0];
    double a_val = 1;
    double b_val = 1;
    double c_val = 1;
    double* vals[] = {&a_val, &b_val, &c_val};
    DFPattern nodes[] = {a_, b_, c_};
    for (int i = 0; i < 3; i++) {
      if (node_map.count(nodes[i]) > 0) {
        if (dtype == DataType::Int(32, 1))
          *vals[i] = static_cast<int*>(
              transform::FoldConstantExpr(node_map[nodes[i]][0]).as<ConstantNode>()->data->data)[0];
        else if (dtype == DataType::Float(32, 1))
          *vals[i] = static_cast<float*>(
              transform::FoldConstantExpr(node_map[nodes[i]][0]).as<ConstantNode>()->data->data)[0];
        else if (dtype == DataType::Float(64, 1))
          *vals[i] = static_cast<double*>(
              transform::FoldConstantExpr(node_map[nodes[i]][0]).as<ConstantNode>()->data->data)[0];
      }
    }
    if (c_val == 1 && a_val > 1) {
      auto temp_exp = x;
      x = y;
      y = temp_exp;
      float temp_val = a_val;
      a_val = c_val;
      c_val = temp_val;
    }

    double sub_value = b_val * b_val - 4 * a_val * c_val;
    if (sub_value < 0) return pre;
    bool same_multiplicands = sub_value < 10e-5;

    double discriminant = std::sqrt(sub_value);
    Expr first_val = MakeConstantScalar(dtype, (b_val + discriminant) / (2 * a_val));
    Expr second_val = same_multiplicands
                          ? first_val
                          : MakeConstantScalar(dtype, (b_val - discriminant) / (2 * a_val));

    Expr first_multiplicand = Call(Op::Get("add"), {x, Call(Op::Get("multiply"), {y, first_val})});
    Expr second_multiplicand =
        same_multiplicands ? first_multiplicand
                           : Call(Op::Get("add"), {x, Call(Op::Get("multiply"), {y, second_val})});
    Expr a = MakeConstantScalar(dtype, a_val);
    return Call(Op::Get("multiply"),
                {a, Call(Op::Get("multiply"), {first_multiplicand, second_multiplicand})});
  }

 private:
  /*! \brief Pattern input */
  DFPattern a_;
  DFPattern b_;
  DFPattern c_;
  DFPattern x_;
  DFPattern y_;
};

/*! \brief Simplifying x/sqrt to x*rsqrt */
class SimplifyRSqrt : public DFPatternRewrite {
 public:
  SimplifyRSqrt() {
    x_ = IsWildcard();
    numerator_ = IsWildcard();
    auto sqrt = IsOp("sqrt");
    pattern_ = IsOp("divide")({numerator_, sqrt({x_})});
  }

  Expr Callback(const Expr& pre, const Expr& post,
                const Map<DFPattern, Array<Expr>>& node_map) const override {
    static const Op& op = Op::Get("rsqrt");
    auto x = node_map[x_][0];
    auto numerator = node_map[numerator_][0];
    return Call(Op::Get("multiply"), {numerator, Call(op, {x})});
  }

 private:
  /*! \brief Pattern input */
  DFPattern x_;
  DFPattern numerator_;
};

/*! \brief Base class for simplifying dequantize followed by arg ops */
class SimplifyDQArgFunc : public DFPatternRewrite {
 public:
  explicit SimplifyDQArgFunc(std::string op) : op_(op) {
    x_ = IsWildcard();
    dq_ = IsOp("qnn.dequantize")({x_, IsWildcard(), IsWildcard()});
    pattern_ = IsOp(op_)({dq_});
  }

  Expr Callback(const Expr& pre, const Expr& post,
                const Map<DFPattern, Array<Expr>>& node_map) const override {
    const CallNode* call = pre.as<CallNode>();
    ICHECK(call);
    auto x = node_map[x_][0];
    return Call(Op::Get(op_), {x}, call->attrs);
  }

 protected:
  /*! \brief Pattern input */
  DFPattern x_;
  /*! \brief dequantize op */
  DFPattern dq_;
  /*! \brief Name of op to simplify */
  String op_;
};

/*! \brief Simplify dequantize follwed by argmax */
class SimplifyDQArgMax : public SimplifyDQArgFunc {
 public:
  SimplifyDQArgMax() : SimplifyDQArgFunc("argmax") {}
};

/*! \brief Simplify dequantize follwed by argmin */
class SimplifyDQArgMin : public SimplifyDQArgFunc {
 public:
  SimplifyDQArgMin() : SimplifyDQArgFunc("argmin") {}
};

/*! \brief Simplify dequantize follwed by argsort */
class SimplifyDQArgSort : public SimplifyDQArgFunc {
 public:
  SimplifyDQArgSort() : SimplifyDQArgFunc("argsort") {}
};

Expr SimplifyExpr(const Expr& expr, const IRModule& mod) {
  // the rewrites will be applied in the given order, and repeated until fixed point
  DFPatternRewriteComposer composer;
  composer.AddRewrite<ConcretizeZerosLikeRewrite>();
  composer.AddRewrite<ConcretizeOnesLikeRewrite>();
  composer.AddRewrite<ConcretizeFullLikeRewrite>();
  composer.AddRewrite<ConcretizeReshapeLikeRewrite>();
  composer.AddRewrite<ConcretizeCollapseSumLikeRewrite>();
  composer.AddRewrite<ConcretizeBroadcastToLikeRewrite>();
  composer.AddRewrite<ConcretizeCastLikeRewrite>();
  composer.AddRewrite<SimplifyAdd>();
  composer.AddRewrite<SimplifyRSqrt>();
  composer.AddRewrite<EliminateIdentityRewrite>();
  composer.AddRewrite<SimplifyReshape>();
  composer.AddRewrite<SimplifyTranspose>();
  composer.AddRewrite<SimplifyNoOpTranspose>();
  composer.AddRewrite<FullElementwise>();
  composer.AddRewrite<SwitchAddMultiply>();
  composer.AddRewrite<SimplifyAdjacentMultiplyOrAdd>();
  composer.AddRewrite<SimplifyDQArgMax>();
  composer.AddRewrite<SimplifyDQArgMin>();
  composer.AddRewrite<SimplifyDQArgSort>();
  composer.AddRewrite<SimplifyClipAndCast>();
  composer.AddRewrite<SimplifyCastAndTranspose>();
  composer.AddRewrite<SimplifyBinomial>();
  return RewritePatterns(composer.MakeCallbacks(), expr, mod);
}

Expr SimplifyExprPostAlterOp(const Expr& expr, const IRModule& mod) {
  // stripped-down version of AlterOp that cleans up some patterns
  // often left by the AlterOpLayout pass.
  DFPatternRewriteComposer composer;
  composer.AddRewrite<EliminateIdentityRewrite>();
  composer.AddRewrite<SimplifyReshape>();
  composer.AddRewrite<SimplifyClipAndCast>();
  composer.AddRewrite<SimplifyCastAndTranspose>();
  return RewritePatterns(composer.MakeCallbacks(), expr, mod);
}

namespace transform {

Pass SimplifyExpr() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        auto res = Downcast<Function>(SimplifyExpr(f, m));
        return res;
      };
  return CreateFunctionPass(pass_func, 0, "SimplifyExpr", {"InferType"});
}

Pass SimplifyExprPostAlterOp() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        auto res = Downcast<Function>(SimplifyExprPostAlterOp(f, m));
        return res;
      };
  return CreateFunctionPass(pass_func, 0, "SimplifyExprPostAlterOp", {"InferType"});
}

TVM_REGISTER_GLOBAL("relay._transform.SimplifyExpr").set_body_typed(SimplifyExpr);
TVM_REGISTER_GLOBAL("relay._transform.SimplifyExprPostAlterOp")
    .set_body_typed(SimplifyExprPostAlterOp);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
