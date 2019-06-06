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
 * Copyright (c) 2018 by Contributors
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
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/attrs/transform.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/transform.h>
#include <unordered_map>
#include <unordered_set>
#include "./expr_subst.h"
#include "./pattern_util.h"


namespace tvm {
namespace relay {

using Branch = std::vector<const CallNode*>;
using Group = std::vector<Branch>;

/*
  Find parallel branches starting with conv2d as shown below and then group branches by kernel
  shape and attributes of conv2d. Conv2d can be followed by zero or more elemwise or broadcast ops.
  Intermediate nodes have exactly one successor. It is possible that branches meet at a point,
  which should be handled in ParallelConv2DCombiner.

         data
        /    \
    conv2d   conv2d
      |        |
      op       op
      |        |
*/
class BranchGroupFinder : private ExprVisitor {
 public:
  std::vector<Group> Find(const Expr& expr) {
    static const Op& conv2d = Op::Get("nn.conv2d");

    this->VisitExpr(expr);

    std::vector<Group> groups;
    for (const auto& root : conv_roots_) {
      const auto& children = children_map_.at(root);
      size_t ngroups = groups.size();
      for (const CallNode* child : children) {
        if (!child->op.same_as(conv2d)) continue;

        auto&& branch = CreateBranch(child);
        // add the branch to a group, or create a new group
        auto it = std::find_if(groups.begin() + ngroups, groups.end(), [&](const Group& group) {
          CHECK(!group.empty() && !group[0].empty());
          return IsCompatibleConv2D(child, group[0][0]);
        });
        if (it != groups.end()) {
          it->push_back(branch);
        } else {
          groups.emplace_back();
          // each group has at least one branch
          groups.back().push_back(branch);
        }
      }
    }
    return groups;
  }

 private:
  std::unordered_set<Expr, NodeHash, NodeEqual> conv_roots_;
  std::unordered_map<Expr, std::vector<const CallNode*>, NodeHash, NodeEqual> children_map_;

  // Two 2d convolutions can be combined if they have the same attributes or
  // only have different output channels.
  bool IsCompatibleConv2D(const CallNode* a, const CallNode* b) {
    AttrsEqual eq;
    static const Layout kOIHW("OIHW");
    const auto* attrs_a = a->attrs.as<Conv2DAttrs>();
    const auto* attrs_b = b->attrs.as<Conv2DAttrs>();
    CHECK(attrs_a);
    CHECK(attrs_b);
    const auto* tweight_a = a->args[1]->type_as<TensorTypeNode>();
    const auto* tweight_b = b->args[1]->type_as<TensorTypeNode>();
    const auto shape_a = BijectiveLayoutNode::make(
      Layout(attrs_a->kernel_layout), kOIHW).ForwardShape(tweight_a->shape);
    const auto shape_b = BijectiveLayoutNode::make(
      Layout(attrs_b->kernel_layout), kOIHW).ForwardShape(tweight_b->shape);

    return eq(attrs_a->strides, attrs_b->strides) && eq(attrs_a->padding, attrs_b->padding) &&
           eq(attrs_a->dilation, attrs_b->dilation) && eq(attrs_a->groups, attrs_b->groups) &&
           eq(attrs_a->data_layout, attrs_b->data_layout) &&
           eq(attrs_a->kernel_layout, attrs_b->kernel_layout) &&
           eq(attrs_a->out_dtype, attrs_b->out_dtype) &&
           eq(attrs_a->out_layout, attrs_b->out_layout) && eq(shape_a[2], shape_b[2]) &&
           eq(shape_a[3], shape_b[3]);
  }

  // Create a branch starting from conv2d.
  Branch CreateBranch(const CallNode* conv) {
    static auto fpattern = Op::GetAttr<TOpPattern>("TOpPattern");
    // each branch has at least one element, the first element is always conv2d
    Branch branch{conv};
    auto it = children_map_.find(GetRef<Expr>(branch.back()));
    while (it != children_map_.end() && it->second.size() == 1) {
      const CallNode* call = it->second[0];
      auto pattern = fpattern[Downcast<Op>(call->op)];
      if (pattern <= kBroadcast) {
        branch.push_back(call);
        it = children_map_.find(GetRef<Expr>(branch.back()));
      } else {
        break;
      }
    }
    return branch;
  }

  void VisitExpr_(const CallNode* n) final {
    static const Op& conv2d = Op::Get("nn.conv2d");
    ExprVisitor::VisitExpr_(n);
    if (n->op.same_as(conv2d) && n->attrs.as<Conv2DAttrs>()->groups == 1) {
      conv_roots_.insert(n->args[0]);
      children_map_[n->args[0]].push_back(n);
    } else {
      for (size_t i = 0; i < n->args.size(); i++) {
        children_map_[n->args[i]].push_back(n);
      }
    }
  }
};

class ParallelConv2DCombiner {
 public:
  explicit ParallelConv2DCombiner(uint64_t min_num_branches) : min_num_branches_(min_num_branches) {
  }

  Expr Combine(const Expr& expr) {
    auto groups = BranchGroupFinder().Find(expr);
    for (const Group& group : groups) {
      if (group.size() < min_num_branches_) {
        continue;
      }
      CombineBranches(group);
    }
    return ExprSubst(expr, std::move(subst_map_));
  }

 private:
  std::unordered_map<Expr, Expr, NodeHash, NodeEqual> subst_map_;
  uint64_t min_num_branches_;

  std::tuple<Expr, IndexExpr> TransformWeight(const Group& branches) {
    int64_t num_filters = 0;  // number of filters of the transformed weight
    Array<Expr> weights;
    for (const auto& branch : branches) {
      auto conv2d = branch[0];
      weights.push_back(conv2d->args[1]);
      auto channels = GetConv2DSuperChannelsDim(conv2d);
      num_filters += channels;
    }
    auto index = branches[0][0]->attrs.as<Conv2DAttrs>()->kernel_layout.find('O');
    CHECK_NE(index, std::string::npos);
    return std::make_tuple(MakeConcatenate(TupleNode::make(weights), index),
                           MakeConstScalar(Int(32), num_filters));
  }

  Call MakeCombinedConv2D(const Group& branches) {
    static const Op& conv2d = Op::Get("nn.conv2d");
    Expr data = branches[0][0]->args[0];
    Expr new_weight;
    IndexExpr new_channels;
    std::tie(new_weight, new_channels) = TransformWeight(branches);

    const CallNode* group_root = branches[0][0];
    const auto* attrs = group_root->attrs.as<Conv2DAttrs>();
    CHECK(attrs);
    const auto new_attrs = make_node<Conv2DAttrs>();
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

    return CallNode::make(conv2d, {data, new_weight}, Attrs{new_attrs}, {});
  }

  bool IsArgCompatible(const CallNode* a, const CallNode* b, size_t index, size_t channel_pos) {
    AttrsEqual eq;
    auto ta = a->args[index]->type_as<TensorTypeNode>();
    auto tb = b->args[index]->type_as<TensorTypeNode>();
    auto toutput_a = a->type_as<TensorTypeNode>();
    auto toutput_b = b->type_as<TensorTypeNode>();

    if (!eq(ta->dtype, tb->dtype) || ta->shape.size() != tb->shape.size())
      return false;

    // Position of the 'C' dimension in the argument
    size_t arg_channel_pos = channel_pos - toutput_a->shape.size() + ta->shape.size();

    // Channel super-dimension shoule be present and not broadcasted
    if ((arg_channel_pos > channel_pos) ||  // size_t overflow
        !eq(ta->shape[arg_channel_pos], toutput_a->shape[channel_pos]) ||
        !eq(tb->shape[arg_channel_pos], toutput_b->shape[channel_pos]))
      return false;

    for (size_t i = 0; i < ta->shape.size(); i++) {
      if (i == arg_channel_pos) continue;
      if (!eq(ta->shape[i], tb->shape[i]))
        return false;
    }
    return true;
  }

  // Check if ops in depth-th level can be combined
  bool CheckLevel(const Group& branches, size_t depth, size_t channel_pos, size_t parent_index) {
    const CallNode* call = branches[0][depth];
    AttrsEqual attrs_equal;
    // check if all branches in current depth can be combined
    for (auto it = branches.begin() + 1; it != branches.end(); it++) {
      const Branch& branch = *it;
      if (!branch[depth]->op.same_as(call->op) ||
          !attrs_equal(branch[depth]->attrs, call->attrs) ||
          branch[depth]->args.size() != call->args.size()) {
        return false;
      }

      if (branch[depth]->args[parent_index].get() != branch[depth - 1])
        return false;

      // Check args
      for (size_t i = 0; i < call->args.size(); i++) {
        if (i == parent_index) continue;

        if (!IsArgCompatible(call, branch[depth], i, channel_pos) ||
            !attrs_equal(call->attrs, branch[depth]->attrs)) {
          return false;
        }
      }
    }
    return true;
  }

  // Combine args and make the combined CallNode
  Call MakeCombinedCall(const Expr& data, const Group& branches, size_t depth, size_t channel_pos,
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
      size_t arg_channel_pos = channel_pos - ndim + arg_ndim;
      Array<Expr> tuple;
      for (const auto& branch : branches) {
        tuple.push_back(branch[depth]->args[i]);
      }
      auto concat = MakeConcatenate(TupleNode::make(tuple), arg_channel_pos);
      new_args.push_back(std::move(concat));
    }
    return CallNode::make(call->op, new_args, call->attrs, {});
  }

  // Replace output of each branch with slices of the combined output
  void UpdateGroupOutput(const Expr& data, const Group& branches, size_t depth,
                         size_t channel_pos) {
    int64_t index = 0;
    for (const auto& branch : branches) {
      const CallNode* conv2d = branch[0];
      int64_t channels = GetConv2DSuperChannelsDim(conv2d);
      Array<Integer> begin;
      Array<Integer> end;
      for (size_t i = 0; i < channel_pos; i++) {
        begin.push_back(0);
        end.push_back(NullValue<Integer>());
      }
      begin.push_back(index);
      index += channels;
      end.push_back(index);
      auto slice = MakeStridedSlice(data, std::move(begin), std::move(end), Array<Integer>{});
      subst_map_[GetRef<Expr>(branch[depth])] = slice;
    }
  }

  // Combine branches in a group. Conv2d in different branches in the same group are safe to
  // combine. Subsequent ops may or may not be combined. We start from conv2d and try to
  // combine ops from all branches in the same depth.
  void CombineBranches(const Group& branches) {
    Call combined = MakeCombinedConv2D(branches);
    auto conv_param = combined->attrs.as<Conv2DAttrs>();
    const std::string& layout =
        conv_param->out_layout == "" ? conv_param->data_layout : conv_param->out_layout;
    size_t channel_pos = layout.find('C');
    CHECK_NE(channel_pos, std::string::npos);
    auto it = std::min_element(branches.begin(), branches.end(),
                               [](const Branch& branch_a,
                                  const Branch& branch_b) {
                                    return branch_a.size() < branch_b.size();
                                  });
    size_t depth = it->size();
    size_t i;
    // starting from 1 to skip the conv2d
    for (i = 1; i < depth; i++) {
      size_t parent_index;
      for (parent_index = 0; parent_index < branches[0][i]->args.size(); parent_index++) {
        if (branches[0][i]->args[parent_index].get() == branches[0][i - 1]) break;
      }
      CHECK_NE(parent_index, branches[0][i]->args.size());
      if (!CheckLevel(branches, i, channel_pos, parent_index)) break;
      combined = MakeCombinedCall(combined, branches, i, channel_pos, parent_index);
    }
    UpdateGroupOutput(combined, branches, i - 1, channel_pos);
  }
};

/*! \brief Combine parallel conv2d if number of branches >= min_num_branches */
Expr CombineParallelConv2D(const Expr& expr, uint64_t min_num_branches) {
  return ParallelConv2DCombiner(min_num_branches).Combine(expr);
}

namespace transform {

Pass CombineParallelConv2D(uint64_t min_num_branches) {
  runtime::TypedPackedFunc<Function(Function, Module, PassContext)> pass_func =
    [=](Function f, Module m, PassContext pc) {
      return Downcast<Function>(CombineParallelConv2D(f, min_num_branches));
  };
  return CreateFunctionPass(pass_func, 4, "CombineParallelConv2d",
                            {ir::StringImm::make("InferType")});
}

TVM_REGISTER_API("relay._transform.CombineParallelConv2D")
.set_body_typed(CombineParallelConv2D);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
