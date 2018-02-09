/*!
 * Copyright (c) 2017 by Contributors
 * \file fold_scale_axis.cc
 * \author Fold scaling parameter of axis into weight of conv/dense
*/
#include <nnvm/graph.h>
#include <nnvm/op_attr_types.h>
#include <nnvm/graph_attr_types.h>
#include <nnvm/pass.h>
#include <nnvm/compiler/op_attr_types.h>
#include <nnvm/top/nn.h>
#include "./pattern_util.h"
#include "./graph_transform.h"

namespace nnvm {
namespace compiler {

enum FoldScaleKind {
  // No folding is applied
  kNone,
  // The folding decision is pending
  kPending,
  // The original operator that contains the scale.
  kProvider,
  // Pass through the scale to parent/child to the first axis.
  kPassTroughFirst,
  // The final conumer of axis scale using multiply
  // Likely be a conv or dense operator.
  kMulConsumer,
  // The final conumer of axis scale using division
  kDivConsumer
};

// Input fold information
struct FoldScaleInput {
  uint32_t index;
  int axis;
};

// The entry of folding chains on which
// we should perform folding on
struct FoldChainEntry {
  // Entry kind
  FoldScaleKind kind{kNone};
  // The output axis to be folded
  int axis{0};
  // Source node in the fold chain
  int source{0};
  // Following field only used by provider.
  // The input index
  int fold_input_index{1};
  // The scale entry
  NodeEntry scale_entry;
};

// Try to pass axis scaling to backward,
// Given that we we know the status of current fold axis.
using FScaleAxisBackward = std::function<
  FoldScaleKind(const NodeAttrs& attrs,
                int axis,
                const std::vector<TShape>& in_shape,
                const std::vector<TShape>& out_shape,
                std::vector<std::pair<uint32_t, int> >* in_axis)>;

// Detect if there is a scaling axis happening
bool DetectScaleAxis(const IndexedGraph& idx,
                     uint32_t nid,
                     const ShapeVector& shape_vec,
                     const std::vector<uint32_t>& ref_count,
                     bool is_forward,
                     std::vector<FoldChainEntry>* chain) {
  const IndexedGraph::Node& inode = idx[nid];
  static const Op* bcast_mul = Op::Get("broadcast_mul");
  static const Op* expand_dims = Op::Get("expand_dims");
  if (inode.source->op() != bcast_mul) return false;
  const TShape& oshape = shape_vec[idx.entry_id(nid, 0)];
  CHECK_NE(oshape.ndim(), 0);
  if (oshape.ndim() <= 1) return false;
  for (int i = 0; i < 2; ++i) {
    const IndexedGraph::NodeEntry& a = inode.inputs[i];
    const IndexedGraph::NodeEntry& b = inode.inputs[1 - i];
    std::pair<int, int> axis =
        MatchBroadcast1DAxis(oshape, shape_vec[idx.entry_id(a)]);
    if (axis.first != -1 &&
        shape_vec[idx.entry_id(b)] == oshape) {
      if (ref_count[a.node_id] != 1) return false;
      if (is_forward && ref_count[nid] != 1) return false;
      if (!is_forward && ref_count[b.node_id] != 1) return false;
      const IndexedGraph::Node& anode = idx[a.node_id];
      // mark the current entry.
      FoldChainEntry& e = (*chain)[nid];
      if (anode.source->is_variable()) {
        e.fold_input_index = 1 - i;
        e.scale_entry = inode.source->inputs[1 - i];
      } else if (anode.source->op()  == expand_dims &&
                 shape_vec[idx.entry_id(anode.source->inputs[0])].ndim() == 1) {
        e.fold_input_index = 1 - i;
        e.scale_entry = anode.source->inputs[0];
      } else {
        return false;
      }
      e.axis = axis.first;
      e.kind = kPending;
      e.source = nid;
      if (!is_forward) {
        // pass message to another input
        FoldChainEntry& enext = (*chain)[b.node_id];
        enext.axis = e.axis;
        enext.kind = kPending;
        enext.source = nid;
      }
      return true;
    }
  }
  return false;
}

Graph FoldScaleAxis(Graph src) {
  // Operator pattern
  static auto& fbackward =
      nnvm::Op::GetAttr<FScaleAxisBackward>("FScaleAxisBackward");
  const IndexedGraph& idx = src.indexed_graph();
  const ShapeVector& shape_vec = src.GetAttr<ShapeVector>("shape");
  std::vector<uint32_t> ref_count = GetNodeRefCounts(idx);
  std::vector<FoldChainEntry> bwd_chain(idx.num_nodes());
  // shape hint for the inference.
  std::vector<TShape> in_shape, out_shape;
  // perform backward folding.
  for (uint32_t i = idx.num_nodes(); i != 0; --i) {
    uint32_t nid = i - 1;
    const auto& inode = idx[nid];
    if (inode.source->is_variable()) continue;
    if (DetectScaleAxis(idx, nid, shape_vec,
                        ref_count, false, &bwd_chain)) continue;
    if (bwd_chain[nid].kind != kPending) continue;
    if (ref_count[nid] != 1 || !fbackward.count(inode.source->op())) {
      bwd_chain[nid].kind = kNone; continue;
    }
    // get input shape and output shape.
    in_shape.clear(); out_shape.clear();
    for (const IndexedGraph::NodeEntry& e : inode.inputs) {
      in_shape.push_back(shape_vec[idx.entry_id(e)]);
    }
    for (uint32_t i = 0; i < inode.source->num_outputs(); ++i) {
      out_shape.push_back(shape_vec[idx.entry_id(nid, i)]);
    }
    std::vector<std::pair<uint32_t, int> > in_axis;
    FoldScaleKind kind =
        fbackward[inode.source->op()](
            inode.source->attrs, bwd_chain[nid].axis,
            in_shape, out_shape, &in_axis);
    bwd_chain[nid].kind = kind;
    if (kind == kNone) continue;
    CHECK_GE(in_axis.size(), 1U);
    CHECK(kind == kPassTroughFirst || kind == kMulConsumer);
    // propagate back.
    bool can_prop = true;
    for (size_t i = 0; i < in_axis.size(); ++i) {
      const IndexedGraph::NodeEntry& e = inode.inputs[in_axis[0].first];
      if (ref_count[e.node_id] != 1 ||
          idx[e.node_id].source->num_outputs() != 1) {
        can_prop = false; break;
      }
    }
    if (!can_prop) continue;
    for (size_t i = 0; i < in_axis.size(); ++i) {
      const IndexedGraph::NodeEntry& e = inode.inputs[in_axis[i].first];
      if (kind == kPassTroughFirst && i == 0) {
        bwd_chain[e.node_id].kind = kPending;
      } else {
        bwd_chain[nid].kind = kNone;
        bwd_chain[e.node_id].kind = kMulConsumer;
      }
      bwd_chain[e.node_id].axis = in_axis[i].second;
      bwd_chain[e.node_id].source = bwd_chain[nid].source;
    }
    if (kind == kMulConsumer) {
      bwd_chain[bwd_chain[nid].source].kind = kProvider;
    }
  }
  auto transform = [&](uint32_t nid, const NodePtr& n, std::vector<NodeEntry>* ret) {
    const FoldChainEntry& e = bwd_chain[nid];
    if (e.kind == kMulConsumer && bwd_chain[e.source].kind == kProvider) {
      const FoldChainEntry& se = bwd_chain[e.source];
      CHECK_EQ(n->num_outputs(), 1);
      NodeEntry scale = ExpandBiasToMatchAxis(
          se.scale_entry,
          shape_vec[idx.entry_id(nid, 0)].ndim(),
          shape_vec[idx.entry_id(se.scale_entry)].ndim(),
          e.axis);
      *ret = {MakeNode("broadcast_mul", n->attrs.name + "_sc",
                       {NodeEntry{n, 0, 0}, scale})};
      return true;
    } else if (e.kind == kProvider) {
      *ret = {n->inputs[e.fold_input_index]};
      return true;
    } else {
      return false;
    }
  };
  return GraphTransform(src, transform);
}

NNVM_REGISTER_PASS(FoldScaleAxis)
.set_body(FoldScaleAxis);

// property registration.
FoldScaleKind ReluScaleAxisBackward(
    const NodeAttrs& attrs,
    int axis,
    const std::vector<TShape>& in_shape,
    const std::vector<TShape>& out_shape,
    std::vector<std::pair<uint32_t, int> >* in_axis) {
  in_axis->emplace_back(0, axis);
  return kPassTroughFirst;
}

NNVM_REGISTER_OP(relu)
.set_attr<FScaleAxisBackward>("FScaleAxisBackward", ReluScaleAxisBackward);

NNVM_REGISTER_OP(leaky_relu)
.set_attr<FScaleAxisBackward>("FScaleAxisBackward", ReluScaleAxisBackward);

FoldScaleKind BroadcastAddSubScaleAxisBackward(
    const NodeAttrs& attrs,
    int axis,
    const std::vector<TShape>& in_shape,
    const std::vector<TShape>& out_shape,
    std::vector<std::pair<uint32_t, int> >* in_axis) {
  for (int i = 0; i < 2; ++i) {
    std::pair<int, int> m = MatchBroadcast1DAxis(out_shape[0], in_shape[i]);
    if (m.second != -1 && in_shape[1 - i] == out_shape[0]) {
      in_axis->emplace_back(i, axis);
      in_axis->emplace_back(1 - i, m.second);
      return kPassTroughFirst;
    }
  }
  return kNone;
}

NNVM_REGISTER_OP(broadcast_add)
.set_attr<FScaleAxisBackward>("FScaleAxisBackward", BroadcastAddSubScaleAxisBackward);

NNVM_REGISTER_OP(broadcast_sub)
.set_attr<FScaleAxisBackward>("FScaleAxisBackward", BroadcastAddSubScaleAxisBackward);

FoldScaleKind Conv2DScaleAxisBackward(
    const NodeAttrs& attrs,
    int axis,
    const std::vector<TShape>& in_shape,
    const std::vector<TShape>& out_shape,
    std::vector<std::pair<uint32_t, int> >* in_axis) {
  using top::Conv2DParam;
  const Conv2DParam& param = nnvm::get<Conv2DParam>(attrs.parsed);
  // only optimize for nchw for now
  if (param.layout == top::kNCHW) {
    in_axis->emplace_back(1, 0);
    if (param.use_bias) {
      in_axis->emplace_back(2, 0);
    }
    return kMulConsumer;
  } else {
    return kNone;
  }
}

NNVM_REGISTER_OP(conv2d)
.set_attr<FScaleAxisBackward>("FScaleAxisBackward", Conv2DScaleAxisBackward);

}  // namespace compiler
}  // namespace nnvm
