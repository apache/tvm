/*!
 *  Copyright (c) 2016 by Contributors
 * \file gradients.cc
 * \brief Passes that takes gradient of the graph
 * This code code was modified based on mxnet codebase by Min Lin
 */
#include <nnvm/pass.h>
#include <nnvm/op_attr_types.h>
#include <algorithm>
#include <functional>

namespace nnvm {
namespace pass {
namespace {

// default aggregate gradient function
// require operator __zero__ and __ewise_sum__ to be presented.
NodeEntry DefaultAggregateGradient(std::vector<NodeEntry>&& v) {
  if (v.size() == 1) {
    return std::move(v[0]);
  } else if (v.size() == 0) {
    NodePtr zero_node = Node::Create();
    zero_node->attrs.op = Op::Get("__zero__");
    return NodeEntry{zero_node, 0, 0};
  } else {
    NodePtr sum_node = Node::Create();
    sum_node->attrs.op = Op::Get("__ewise_sum__");
    sum_node->inputs = std::move(v);
    return NodeEntry{sum_node, 0, 0};
  }
}

// helper entry
struct GradEntry {
#ifdef _MSC_VER
  NodeEntry sum = NodeEntry{nullptr, 0, 0};
#else
  NodeEntry sum{nullptr, 0, 0};
#endif
  std::vector<NodeEntry> grads;
  bool need_attr_hint{true};
};

Graph Gradient(Graph src) {
  using nnvm::FGradient;
  using MirrorFun = std::function<int (const Node& node)>;
  using AttrHintFun = std::function<NodeEntry (const NodeEntry& src, const NodeEntry &like)>;

  CHECK_NE(src.attrs.count("grad_ys"), 0)
      << "Gradient require grad_ys to be presented.";
  CHECK_NE(src.attrs.count("grad_ys_out_grad"), 0)
      << "Gradient require grad_ys_out_grad to be presented.";
  CHECK_NE(src.attrs.count("grad_xs"), 0)
      << "Gradient require grad_xs to be presented.";
  const std::vector<NodeEntry>& ys =
      src.GetAttr<std::vector<NodeEntry> >("grad_ys");
  const std::vector<NodeEntry>& ys_out_grad =
      src.GetAttr<std::vector<NodeEntry> >("grad_ys_out_grad");
  const std::vector<NodeEntry>& xs =
      src.GetAttr<std::vector<NodeEntry> >("grad_xs");
  using AggFun = std::function<NodeEntry (std::vector<NodeEntry>&& inputs)>;
  AggFun agg_fun = DefaultAggregateGradient;
  if (src.attrs.count("grad_aggregate_fun") != 0) {
    agg_fun = src.GetAttr<AggFun>("grad_aggregate_fun");
  }
  MirrorFun mirror_fun = nullptr;
  if (src.attrs.count("grad_mirror_fun") != 0) {
    mirror_fun = src.GetAttr<MirrorFun>("grad_mirror_fun");
  }
  AttrHintFun attr_hint_fun = nullptr;
  if (src.attrs.count("attr_hint_fun") != 0) {
    attr_hint_fun = src.GetAttr<AttrHintFun>("attr_hint_fun");
  }

  // topo sort
  std::vector<NodePtr> topo_order;
  std::unordered_map<Node*, std::vector<GradEntry> > output_grads;

  DFSVisit(ys, [&](const NodePtr& node) {
      if (output_grads.count(node.get()) == 0) {
        output_grads[node.get()].resize(node->num_outputs());
      }
      topo_order.push_back(node);
    });

  CHECK_EQ(ys.size(), ys_out_grad.size());
  for (size_t i = 0; i < ys.size(); ++i) {
    NodeEntry ograd = ys_out_grad[i];
    output_grads[ys[i].node.get()][ys[i].index].grads = { ograd };
  }

  // construct mirror reduece memory strategy if needed
  std::unordered_map<Node*, NodePtr> mirror_map;
  if (mirror_fun != nullptr) {
    for (const NodePtr& n : topo_order) {
      if (mirror_fun(*n)) {
        NodePtr new_node = Node::Create();
        *new_node = *n;
        new_node->attrs.name += "_mirror";
        for (auto& e : new_node->inputs) {
          e.node = mirror_map.at(e.node.get());
        }
        for (auto& n : new_node->control_deps) {
          n = mirror_map.at(n.get());
        }
        mirror_map[n.get()] = std::move(new_node);
      } else {
        mirror_map[n.get()] = n;
      }
    }
  }

  // traverse backward
  static auto& grad_fun_map = Op::GetAttr<FGradient>("FGradient");
  static auto& finfer_shape = Op::GetAttr<FInferShape>("FInferShape");

  std::vector<NodeEntry> out_agg_grads;
  for (auto rit = topo_order.rbegin(); rit != topo_order.rend(); ++rit) {
    const NodePtr& ptr = *rit;
    if (ptr->is_variable()) continue;
    out_agg_grads.clear();
    auto& out_grad_vec = output_grads.at(ptr.get());
    for (uint32_t i = 0; i < out_grad_vec.size(); ++i) {
      GradEntry& e = out_grad_vec[i];
      e.sum = agg_fun(std::move(e.grads));
      if (e.need_attr_hint && attr_hint_fun != nullptr) {
        e.sum = attr_hint_fun(e.sum, NodeEntry{ptr, 0, i});
      }
      out_agg_grads.push_back(e.sum);
    }
    if ((*rit)->inputs.size() != 0) {
      NodePtr fwd_node = (mirror_map.size() == 0 ? ptr : mirror_map.at(ptr.get()));
      std::vector<NodeEntry> input_grads = grad_fun_map[ptr->op()](
          fwd_node, out_agg_grads);
      CHECK_EQ((*rit)->inputs.size(), input_grads.size())
          << "Gradient function not returning enough gradient";
      auto git = input_grads.begin();
      for (auto it = (*rit)->inputs.begin(); it != (*rit)->inputs.end(); ++it, ++git) {
        auto& ge = output_grads[it->node.get()][it->index];
        // if any of the backward op can do shape inference, the hint is not necessary.
        if (finfer_shape.count(git->node->op())) {
          ge.need_attr_hint = false;
        }
        ge.grads.emplace_back(std::move(*git));
      }
    }
  }
  // take out the xs' grads
  Graph ret;
  ret.outputs.reserve(xs.size());
  for (const NodeEntry& e : xs) {
    GradEntry& entry = output_grads[e.node.get()][e.index];
    // aggregate sum if there haven't been
    if (entry.sum.node.get() == nullptr) {
      entry.sum = agg_fun(std::move(entry.grads));
      if (entry.need_attr_hint && attr_hint_fun != nullptr) {
        entry.sum = attr_hint_fun(entry.sum, e);
      }
    }
    ret.outputs.emplace_back(std::move(entry.sum));
  }
  return ret;
}

// register pass
NNVM_REGISTER_PASS(Gradient)
.describe("Return a gradient graph of src.attrs[\"ys\"] wrt src.attrs[\"xs\"]")
.set_body(Gradient)
.set_change_graph(true)
.depend_graph_attr("grad_ys")
.depend_graph_attr("grad_xs")
.depend_graph_attr("grad_ys_out_grad");

}  // namespace
}  // namespace pass
}  // namespace nnvm
