/*!
 * Copyright (c) 2018 by Contributors
 *
 * \file combine_parallel_conv2d.cc
 *
 * \brief Combine parallel 2d convolutions into a single convolution.
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
    auto channels = GetConv2DSuperChannelsDim(n);
    num_filters += channels;
  }
  auto index = convolutions[0]->attrs.as<Conv2DAttrs>()->weight_layout.find('O');
  CHECK_NE(index, std::string::npos);
  return std::make_tuple(MakeConcatenate(TupleNode::make(weights), index),
                         MakeConstScalar(Int(32), num_filters));
}

// Two 2d convolutions can be combined if they have the same attributes or only have
// different output channels.
bool IsCompatibleConv2D(const CallNode* a, const CallNode* b) {
  AttrsEqual eq;
  static const Layout kOIHW("OIHW");
  auto attrs_a = a->attrs.as<Conv2DAttrs>();
  auto attrs_b = b->attrs.as<Conv2DAttrs>();
  auto tweight_a = a->args[1]->type_as<TensorTypeNode>();
  auto tweight_b = b->args[1]->type_as<TensorTypeNode>();
  auto shape_a = ConvertLayout(tweight_a->shape, attrs_a->weight_layout, kOIHW);
  auto shape_b = ConvertLayout(tweight_b->shape, attrs_b->weight_layout, kOIHW);

  return eq(attrs_a->strides, attrs_b->strides) &&
         eq(attrs_a->padding, attrs_b->padding) &&
         eq(attrs_a->dilation, attrs_b->dilation) &&
         eq(attrs_a->groups, attrs_b->groups) &&
         eq(attrs_a->data_layout, attrs_b->data_layout) &&
         eq(attrs_a->weight_layout, attrs_b->weight_layout) &&
         eq(attrs_a->out_dtype, attrs_b->out_dtype) &&
         eq(attrs_a->out_layout, attrs_b->out_layout) &&
         eq(shape_a[2], shape_b[2]) &&
         eq(shape_a[3], shape_b[3]);
}

Expr MakeCombinedConv2D(const Expr& data, const std::vector<const CallNode*>& convolutions) {
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

Expr CombineParallelConv2D(const Expr& expr) {
  // data -> array of conv2d with the same input
  auto children_map = SiblingConv2DFinder().Find(expr);
  std::unordered_map<Expr, Expr, NodeHash, NodeEqual> subst_map;

  for (const auto& pair : children_map) {
    Expr data = pair.first;
    std::vector<const CallNode*> children = pair.second;

    if (children.size() < 2) continue;

    std::vector<size_t> group_ids(children.size());
    std::vector<std::vector<const CallNode*>> groups;

    for (size_t i = 0; i < children.size(); i++) {
      const CallNode* n = children[i];

      // assign a group id or create a new group for each conv2d
      auto it = std::find_if(groups.begin(), groups.end(),
                             [&](const std::vector<const CallNode*>& group) {
                               return IsCompatibleConv2D(n, group[0]);
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
      if (convs.size() < 2) continue;

      auto new_conv2d = MakeCombinedConv2D(data, convs);
      int64_t index = 0;
      // replace original conv2d with slice of output of the new conv2d
      for (const CallNode* conv2d : convs) {
        auto params = conv2d->attrs.as<Conv2DAttrs>();
        int64_t channels = GetConv2DSuperChannelsDim(conv2d);
        size_t channel_pos = params->data_layout.find('C');
        CHECK_NE(channel_pos, std::string::npos);
        Array<Integer> begin;
        Array<Integer> end;
        for (size_t i = 0; i < channel_pos; i++) {
          begin.push_back(0);
          end.push_back(NullValue<Integer>());
        }
        begin.push_back(index);
        index += channels;
        end.push_back(index);
        auto slice = MakeStridedSlice(new_conv2d, std::move(begin), std::move(end),
                                      Array<Integer>{});
        subst_map[GetRef<Call>(conv2d)] = slice;
      }
    }
  }

  return ExprSubst(expr, std::move(subst_map));
}

TVM_REGISTER_API("relay._ir_pass.CombineParallelConv2D")
.set_body([](TVMArgs args, TVMRetValue* ret) {
    *ret = CombineParallelConv2D(args[0]);
  });

}  // namespace relay
}  // namespace tvm
