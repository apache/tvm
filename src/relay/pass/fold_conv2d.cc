/*!
 * Copyright (c) 2018 by Contributors
 *
 * \file fold_conv2d.cc
 *
 * \brief Fold multiple 2d convolutions into a single convolution.
 *
 * This pass replaces convolutions that share the same input node and the same arguments (except
 * that the number of output channels can be different) with a single convolution. The weight of
 * the new 2d convolution is the concatenation of the original weights.
 *
 * This prevents launching multiple kernels in networks with multiple convolution branches, such
 * as Inception block.
 */

#include <tvm/relay/pass.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/attrs/transform.h>
#include "./expr_subst.h"
#include "./pattern_util.h"

namespace tvm {
namespace relay {

class SiblingConv2DFinder : public ExprVisitor {
 public:
  std::unordered_map<Expr, std::vector<const CallNode*>, NodeHash, NodeEqual> Find(
      const Expr& expr) {
    this->VisitExpr(expr);
    return std::move(children_map_);
  }

  void VisitExpr_(const CallNode* n) final {
    static const Op& conv2d = Op::Get("nn.conv2d");
    ExprVisitor::VisitExpr_(n);
    if (n->op.same_as(conv2d) && n->attrs.as<Conv2DAttrs>()->groups == 1) {
      children_map_[n->args[0]].push_back(n);
    }
  }

 private:
  std::unordered_map<Expr, std::vector<const CallNode*>, NodeHash, NodeEqual> children_map_;
};

std::tuple<Expr, IndexExpr> TransformWeight(std::vector<const CallNode*> convolutions) {
  int64_t num_filters = 0;  // number of filters of the transformed weight
  Array<Expr> weights;
  for (const CallNode* n : convolutions) {
    weights.push_back(n->args[1]);
    auto channels = as_const_int(n->attrs.as<Conv2DAttrs>()->channels);
    CHECK(channels);
    num_filters += *channels;
  }
  auto index = convolutions[0]->attrs.as<Conv2DAttrs>()->weight_layout.find('O');
  CHECK_NE(index, std::string::npos);
  return std::make_tuple(MakeConcatenate(TupleNode::make(weights), index),
                         MakeConstScalar(Int(32), num_filters));
}

// Two 2d convolutions can be combined if they have the same attributes or only have
// different output channels.
bool IsCompatibleConv2D(const Conv2DAttrs& a, const Conv2DAttrs& b) {
  AttrsEqual eq;
  return eq(a.strides, b.strides) &&
         eq(a.padding, b.padding) &&
         eq(a.dilation, b.dilation) &&
         eq(a.groups, b.groups) &&
         eq(a.kernel_size, b.kernel_size) &&
         eq(a.data_layout, b.data_layout) &&
         eq(a.weight_layout, b.weight_layout) &&
         eq(a.out_dtype, b.out_dtype) &&
         eq(a.out_layout, b.out_layout);
}

Expr MakeFoldedConv2D(const Expr& data, const std::vector<const CallNode*>& convolutions) {
  static const Op& conv2d = Op::Get("nn.conv2d");

  Expr new_weight;
  IndexExpr new_channels;
  std::tie(new_weight, new_channels) = TransformWeight(convolutions);

  const CallNode* group_root = convolutions[0];
  auto attrs = group_root->attrs.as<Conv2DAttrs>();
  auto new_attrs = make_node<Conv2DAttrs>();
  new_attrs->strides = attrs->strides;
  new_attrs->padding = attrs->padding;
  new_attrs->dilation = attrs->dilation;
  new_attrs->groups = attrs->groups;
  new_attrs->kernel_size = attrs->kernel_size;
  new_attrs->data_layout = attrs->data_layout;
  new_attrs->weight_layout = attrs->weight_layout;
  new_attrs->out_layout = attrs->out_layout;
  new_attrs->out_dtype = attrs->out_dtype;
  new_attrs->channels = new_channels;

  return CallNode::make(conv2d, {data, new_weight}, Attrs{new_attrs}, {});
}

Expr FoldConv2D(const Expr& expr) {
  // data -> array of conv2d with the same input
  auto children_map = SiblingConv2DFinder().Find(expr);
  Map<Expr, Expr> subst_map;

  for (const auto& pair : children_map) {
    Expr data = pair.first;
    std::vector<const CallNode*> children = pair.second;

    if (children.size() < 2) continue;

    std::vector<size_t> group_ids(children.size());
    std::vector<std::vector<const CallNode*>> groups;

    for (size_t i = 0; i < children.size(); i++) {
      const CallNode* n = children[i];
      auto args = n->attrs.as<Conv2DAttrs>();

      // assign a group id or create a new group for each conv2d
      auto it = std::find_if(groups.begin(), groups.end(),
                             [&](const std::vector<const CallNode*>& group) {
                               const CallNode* group_root = *(group.begin());
                               auto group_args = group_root->attrs.as<Conv2DAttrs>();
                               return IsCompatibleConv2D(*args, *group_args);
                             });

      if (it != groups.end()) {
        auto group_id = std::distance(groups.begin(), it);
        group_ids[i] = group_id;
        groups[group_id].push_back(n);
      } else {
        group_ids[i] = groups.size();
        groups.emplace_back(std::vector<const CallNode*>{n});
      }
    }

    for (const auto& convs : groups) {
      if (convs.size() < 2) {
        continue;
      }
      auto new_conv2d = MakeFoldedConv2D(data, convs);

      int64_t start = 0;
      // replace original conv2d with slice of output of the new conv2d
      for (const auto& conv2d : convs) {
        auto params = conv2d->attrs.as<Conv2DAttrs>();
        auto channels = as_const_int(params->channels);
        CHECK(channels);
        auto indices = MakeConstantArrayFromRange(Int(64), start, start + *channels);
        auto channel_index = params->data_layout.find('C');
        CHECK_NE(channel_index, std::string::npos);
        auto take = MakeTake(new_conv2d, indices, channel_index);
        start += *channels;
        subst_map.Set(GetRef<Call>(conv2d), take);
      }
    }
  }

  return ExprSubst(expr, std::move(subst_map));
}

TVM_REGISTER_API("relay._ir_pass.FoldConv2D")
.set_body([](TVMArgs args, TVMRetValue* ret) {
    *ret = FoldConv2D(args[0]);
  });

}  // namespace relay
}  // namespace tvm
