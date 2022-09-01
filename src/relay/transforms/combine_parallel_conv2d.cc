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
 *
 * \file combine_parallel_conv2d.cc
 * \brief Combine parallel 2d convolutions into a single convolution.
 *
 * This pass replaces convolutions that share the same input node and the same
 * arguments (except that the number of output channels can be different) with a
 * single convolution. The weight of the new 2d convolution is the concatenation
 * of the original weights. Elemwise and broadcast ops following conv2d are also
 * combined if possible.
 *
 * This prevents launching multiple kernels in networks with multiple
 * convolution branches, such as Inception block.
 */

#include <tvm/relay/analysis.h>
#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/attrs/transform.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/transform.h>

#include <unordered_map>
#include <unordered_set>

#include "./combine_parallel_op.h"
#include "./expr_subst.h"
#include "pattern_utils.h"

namespace tvm {
namespace relay {

class ParallelConv2DCombiner : public ParallelOpCombiner {
 public:
  explicit ParallelConv2DCombiner(uint64_t min_num_branches)
      : ParallelOpCombiner("nn.conv2d", min_num_branches) {}

 protected:
  bool IsSupportedOp(const CallNode* n) { return n->attrs.as<Conv2DAttrs>()->groups == 1; }

  bool CanOpsBeCombined(const CallNode* a, const CallNode* b) {
    StructuralEqual eq;
    const Layout kOIHW("OIHW");
    const auto* attrs_a = a->attrs.as<Conv2DAttrs>();
    const auto* attrs_b = b->attrs.as<Conv2DAttrs>();
    ICHECK(attrs_a);
    ICHECK(attrs_b);
    const auto* tweight_a = a->args[1]->type_as<TensorTypeNode>();
    const auto* tweight_b = b->args[1]->type_as<TensorTypeNode>();
    const auto shape_a =
        tir::BijectiveLayout(Layout(attrs_a->kernel_layout), kOIHW).ForwardShape(tweight_a->shape);
    const auto shape_b =
        tir::BijectiveLayout(Layout(attrs_b->kernel_layout), kOIHW).ForwardShape(tweight_b->shape);

    return eq(attrs_a->strides, attrs_b->strides) && eq(attrs_a->padding, attrs_b->padding) &&
           eq(attrs_a->dilation, attrs_b->dilation) && eq(attrs_a->groups, attrs_b->groups) &&
           eq(attrs_a->data_layout, attrs_b->data_layout) &&
           eq(attrs_a->kernel_layout, attrs_b->kernel_layout) &&
           eq(attrs_a->out_dtype, attrs_b->out_dtype) &&
           eq(attrs_a->out_layout, attrs_b->out_layout) && eq(shape_a[2], shape_b[2]) &&
           eq(shape_a[3], shape_b[3]);
  }

  Call MakeCombinedOp(const Group& branches) {
    const Op& conv2d = Op::Get("nn.conv2d");
    Expr data = branches[0][0]->args[0];
    auto [new_weight, new_channels] = TransformWeight(branches);

    const CallNode* group_root = branches[0][0];
    const auto* attrs = group_root->attrs.as<Conv2DAttrs>();
    ICHECK(attrs);
    const auto new_attrs = make_object<Conv2DAttrs>();
    new_attrs->strides = attrs->strides;
    new_attrs->padding = attrs->padding;
    new_attrs->dilation = attrs->dilation;
    new_attrs->groups = attrs->groups;
    new_attrs->kernel_size = attrs->kernel_size;
    new_attrs->data_layout = attrs->data_layout;
    new_attrs->kernel_layout = attrs->kernel_layout;
    new_attrs->out_layout = attrs->out_layout;
    new_attrs->out_dtype = attrs->out_dtype;
    new_attrs->channels = new_channels;

    const std::string& layout =
        new_attrs->out_layout == "" ? new_attrs->data_layout : new_attrs->out_layout;
    channel_pos_ = layout.find('C');
    ICHECK_NE(channel_pos_, std::string::npos);

    return Call(conv2d, {data, new_weight}, Attrs{new_attrs}, {});
  }

  bool IsArgCompatible(const CallNode* a, const CallNode* b, size_t index) {
    StructuralEqual eq;
    auto ta = a->args[index]->type_as<TensorTypeNode>();
    auto tb = b->args[index]->type_as<TensorTypeNode>();
    auto toutput_a = a->type_as<TensorTypeNode>();
    auto toutput_b = b->type_as<TensorTypeNode>();

    if (!eq(ta->dtype, tb->dtype) || ta->shape.size() != tb->shape.size()) return false;

    // Position of the 'C' dimension in the argument
    size_t arg_channel_pos = channel_pos_ - toutput_a->shape.size() + ta->shape.size();

    // Channel super-dimension shoule be present and not broadcasted
    if ((arg_channel_pos > channel_pos_) ||  // size_t overflow
        !eq(ta->shape[arg_channel_pos], toutput_a->shape[channel_pos_]) ||
        !eq(tb->shape[arg_channel_pos], toutput_b->shape[channel_pos_]))
      return false;

    for (size_t i = 0; i < ta->shape.size(); i++) {
      if (i == arg_channel_pos) continue;
      if (!eq(ta->shape[i], tb->shape[i])) return false;
    }
    return true;
  }

  Call MakeCombinedCallFromFollowingOps(const Expr& data, const Group& branches, size_t depth,
                                        size_t parent_index) {
    Array<Expr> new_args;
    const CallNode* call = branches[0][depth];
    size_t ndim = call->type_as<TensorTypeNode>()->shape.size();

    for (size_t i = 0; i < call->args.size(); i++) {
      if (i == parent_index) {
        new_args.push_back(data);
        continue;
      }

      size_t arg_ndim = call->args[i]->type_as<TensorTypeNode>()->shape.size();
      size_t arg_channel_pos = channel_pos_ - ndim + arg_ndim;
      Array<Expr> tuple;
      for (const auto& branch : branches) {
        tuple.push_back(branch[depth]->args[i]);
      }

      auto concat = MakeConcatenate(Tuple(tuple), arg_channel_pos);
      new_args.push_back(std::move(concat));
    }

    return Call(call->op, new_args, call->attrs, {});
  }

  void UpdateGroupOutput(const Expr& data, const Group& branches, size_t depth,
                         ExprSubstMap* subst_map) {
    int64_t index = 0;

    for (const auto& branch : branches) {
      const CallNode* conv2d = branch[0];
      int64_t channels = GetConv2DSuperChannelsDim(conv2d);
      Array<Integer> begin;
      Array<Integer> end;
      for (size_t i = 0; i < channel_pos_; i++) {
        begin.push_back(0);
        end.push_back(-1);
      }
      begin.push_back(index);
      index += channels;
      end.push_back(channels);
      Array<Integer> strides(begin.size(), 1);
      auto slice = MakeStridedSlice(data, begin, end, strides, "size");
      subst_map->insert({GetRef<Expr>(branch[depth]), slice});
    }
  }

 private:
  /* \brief index of channel dimension */
  size_t channel_pos_;

  std::tuple<Expr, IndexExpr> TransformWeight(const Group& branches) {
    int64_t num_filters = 0;  // number of filters of the transformed weight
    Array<Expr> weights;
    for (const auto& branch : branches) {
      auto conv2d = branch[0];
      weights.push_back(conv2d->args[1]);
      auto channels = GetConv2DSuperChannelsDim(conv2d);
      num_filters += channels;
    }
    auto index =
        branches[0][0]->attrs.as<Conv2DAttrs>()->kernel_layout.operator std::string().find('O');
    ICHECK_NE(index, std::string::npos);
    return std::make_tuple(MakeConcatenate(Tuple(weights), index),
                           tir::make_const(DataType::Int(32), num_filters));
  }
};

/*! \brief Combine parallel conv2d if number of branches >= min_num_branches */
Expr CombineParallelConv2D(const Expr& expr, uint64_t min_num_branches) {
  return ParallelConv2DCombiner(min_num_branches).Combine(expr);
}

namespace transform {

Pass CombineParallelConv2D(uint64_t min_num_branches) {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        return Downcast<Function>(CombineParallelConv2D(f, min_num_branches));
      };
  return CreateFunctionPass(pass_func, 4, "CombineParallelConv2d", {"InferType"});
}

TVM_REGISTER_GLOBAL("relay._transform.CombineParallelConv2D").set_body_typed(CombineParallelConv2D);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
