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
#include "pattern_util.h"
#include "graph_transform.h"

namespace nnvm {
namespace compiler {

enum FoldScaleKind {
  // No folding is applied
  kNone,
  // The folding decision is pending, we can fold on a state.
  kPending,
  // The original operator that contains the scale.
  kProvider,
  // The final conumer of axis scale using multiply
  // Likely be a conv or dense operator.
  kMulConsumer,
  // The final conumer of axis scale using division
  kDivConsumer
};

struct FoldChainInfo {
  // Entry kind
  FoldScaleKind kind{kNone};
  // The output axis to be folded
  int axis{0};
  // Source node in the fold chain
  int source{0};
};

// The entry of folding chains on which
// we should perform folding on
struct FoldChainEntry {
  // Fold information
  FoldChainInfo info;
  // Number of outgoing fork count
  // in forward propagation.
  int fork_count{0};
  // Following field only used by provider.
  // The input index
  int fold_input_index{1};
  // The scale entry
  NodeEntry scale_entry;
};

// Try to pass axis scaling to backward,
// Given that we we know the status of current fold axis.
// return whether the forward signal is consumed.
using FScaleAxisBackward = std::function<
  bool(const NodeAttrs& attrs,
       const std::vector<TShape>& in_shape,
       const std::vector<TShape>& out_shape,
       const FoldChainInfo& out_info,
       std::vector<FoldChainInfo>* in_info)>;


// Try to pass axis scaling to forward,
// Given that we we know the status of one of its input to be pending
// also update other input info
// return whether the forward signal is consumed.
using FScaleAxisForward = std::function<
  bool(const NodeAttrs& attrs,
       const std::vector<TShape>& in_shape,
       const std::vector<TShape>& out_shape,
       std::vector<FoldChainInfo>* in_info,
       FoldChainInfo* out_info)>;


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
      e.info.axis = axis.first;
      e.info.kind = kPending;
      e.info.source = nid;
      e.fork_count = 1;
      // In the backward message passing
      // We need to eagerly pass it to the input
      // In the forward message passing
      // we will "pull" the message from input.
      if (!is_forward) {
        FoldChainEntry& enext = (*chain)[b.node_id];
        enext.info.axis = e.info.axis;
        enext.info.kind = kPending;
        enext.info.source = nid;
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
  static auto& fforward =
      nnvm::Op::GetAttr<FScaleAxisForward>("FScaleAxisForward");
  const IndexedGraph& idx = src.indexed_graph();
  const ShapeVector& shape_vec = src.GetAttr<ShapeVector>("shape");
  std::vector<uint32_t> ref_count = GetNodeRefCounts(idx);
  std::vector<FoldChainEntry> bwd_chain(idx.num_nodes());
  std::vector<FoldChainEntry> fwd_chain(idx.num_nodes());
  // shape hint for the inference.
  std::vector<TShape> in_shape, out_shape;

  // perform backward folding.
  for (uint32_t i = idx.num_nodes(); i != 0; --i) {
    uint32_t nid = i - 1;
    const auto& inode = idx[nid];
    if (inode.source->is_variable()) continue;
    if (DetectScaleAxis(idx, nid, shape_vec,
                        ref_count, false, &bwd_chain)) continue;
    if (bwd_chain[nid].info.kind != kPending) continue;
    // if referred by multiple node, cannot do propagation
    if (ref_count[nid] != 1 || !fbackward.count(inode.source->op())) {
      bwd_chain[nid].info.kind = kNone; continue;
    }
    // get input shape and output shape.
    in_shape.clear(); out_shape.clear();
    for (const IndexedGraph::NodeEntry& e : inode.inputs) {
      in_shape.push_back(shape_vec[idx.entry_id(e)]);
    }
    for (uint32_t i = 0; i < inode.source->num_outputs(); ++i) {
      out_shape.push_back(shape_vec[idx.entry_id(nid, i)]);
    }
    std::vector<FoldChainInfo> in_info(in_shape.size(), FoldChainInfo());
    bool consumed = fbackward[inode.source->op()](
        inode.source->attrs,
        in_shape,
        out_shape,
        bwd_chain[nid].info,
        &in_info);
    CHECK_EQ(in_info.size(), in_shape.size());
    // propagate back.
    bool can_prop = true;
    for (size_t i = 0; i < in_info.size(); ++i) {
      const IndexedGraph::NodeEntry& e = inode.inputs[i];
      if (ref_count[e.node_id] != 1 ||
          idx[e.node_id].source->num_outputs() != 1) {
        can_prop = false; break;
      }
    }
    if (!can_prop) continue;
    for (size_t i = 0; i < in_info.size(); ++i) {
      const IndexedGraph::NodeEntry& e = inode.inputs[i];
      bwd_chain[e.node_id].info = in_info[i];
    }
    // mark consumed by making the source as provider.
    if (consumed) {
      bwd_chain[bwd_chain[nid].info.source].info.kind = kProvider;
    }
  }


  // perform forward folding.
  for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
    const auto& inode = idx[nid];
    if (inode.source->is_variable()) continue;
    // skip scales that are already folded in backward.
    if (bwd_chain[nid].info.kind == kProvider) continue;
    if (DetectScaleAxis(idx, nid, shape_vec,
                        ref_count, true, &fwd_chain)) continue;
    if (inode.source->num_outputs() != 1) continue;
    // Do state update
    // get input shape and output shape.
    std::vector<FoldChainInfo> in_info;
    FoldChainInfo out_info;
    int num_inpending = 0;
    in_shape.clear(); out_shape.clear();
    for (const IndexedGraph::NodeEntry& e : inode.inputs) {
      in_shape.push_back(shape_vec[idx.entry_id(e)]);
      // input information
      in_info.push_back(fwd_chain[e.node_id].info);
      if (fwd_chain[e.node_id].info.kind == kPending) {
        ++num_inpending;
      }
    }
    for (uint32_t i = 0; i < inode.source->num_outputs(); ++i) {
      out_shape.push_back(shape_vec[idx.entry_id(nid, i)]);
    }
    if (num_inpending != 1 ||
        !fforward.count(inode.source->op())) continue;
    bool consumed = fforward[inode.source->op()](
        inode.source->attrs,
        in_shape,
        out_shape,
        &in_info,
        &out_info);
    // update input info
    for (size_t i = 0; i < in_info.size(); ++i) {
      fwd_chain[inode.inputs[i].node_id].info = in_info[i];
    }
    if (consumed) {
      fwd_chain[nid].info = out_info;
      for (size_t i = 0; i < in_info.size(); ++i) {
        if (in_info[i].kind == kPending) {
          if (--fwd_chain[in_info[i].source].fork_count == 0) {
            fwd_chain[in_info[i].source].info.kind = kProvider;
          }
        }
      }
    } else {
      // can propagate condition
      if (inode.source->num_outputs() == 1) {
        fwd_chain[nid].info = out_info;
        if (out_info.kind == kPending) {
          // When there is multiple reference to input
          // every path have to be consumed
          fwd_chain[out_info.source].fork_count += ref_count[nid] - 1;
        }
      }
    }
  }

  auto transform = [&](uint32_t nid, const NodePtr& n, std::vector<NodeEntry>* ret) {
    NodeEntry rvalue = NodeEntry{n, 0, 0};
    {
      // Backward chain
      const FoldChainEntry& e = bwd_chain[nid];
      if (e.info.kind == kMulConsumer &&
          bwd_chain[e.info.source].info.kind == kProvider) {
        const FoldChainEntry& se = bwd_chain[e.info.source];
        CHECK_EQ(n->num_outputs(), 1);
        NodeEntry scale = ExpandBiasToMatchAxis(
            se.scale_entry,
            shape_vec[idx.entry_id(nid, 0)].ndim(),
            shape_vec[idx.entry_id(se.scale_entry)].ndim(),
            e.info.axis);
        rvalue = MakeNode("broadcast_mul", n->attrs.name + "_sc",
                          {rvalue, scale});
      } else if (e.info.kind == kProvider) {
        rvalue = n->inputs[e.fold_input_index];
      }
    }
    // Note that the value might get transformed twice if it
    // folds value from both fwd and backward chain.
    {
      // forward chain
      const FoldChainEntry& e = fwd_chain[nid];
      if (e.info.kind == kMulConsumer &&
          fwd_chain[e.info.source].info.kind == kProvider) {
        const FoldChainEntry& se = fwd_chain[e.info.source];
        CHECK_EQ(n->num_outputs(), 1);
        NodeEntry scale = ExpandBiasToMatchAxis(
            se.scale_entry,
            shape_vec[idx.entry_id(nid, 0)].ndim(),
            shape_vec[idx.entry_id(se.scale_entry)].ndim(),
            e.info.axis);
        rvalue = MakeNode("broadcast_mul", n->attrs.name + "_sc",
                          {rvalue, scale});
      } else if (e.info.kind == kDivConsumer &&
                 fwd_chain[e.info.source].info.kind == kProvider) {
        const FoldChainEntry& se = fwd_chain[e.info.source];
        CHECK_EQ(n->num_outputs(), 1);
        NodeEntry scale = ExpandBiasToMatchAxis(
            se.scale_entry,
            shape_vec[idx.entry_id(nid, 0)].ndim(),
            shape_vec[idx.entry_id(se.scale_entry)].ndim(),
            e.info.axis);
        rvalue = MakeNode("broadcast_div", n->attrs.name + "_sc",
                          {rvalue, scale});
      } else if (e.info.kind == kProvider) {
        rvalue = n->inputs[e.fold_input_index];
      }
    }
    if (rvalue.node == n) {
      return false;
    } else {
      *ret = {rvalue};
      return true;
    }
  };
  return GraphTransform(src, transform);
}

NNVM_REGISTER_PASS(FoldScaleAxis)
.set_body(FoldScaleAxis);

// property registration.
bool ReluScaleAxisBackward(
    const NodeAttrs& attrs,
    const std::vector<TShape>& in_shape,
    const std::vector<TShape>& out_shape,
    const FoldChainInfo& out_info,
    std::vector<FoldChainInfo>* in_axis) {
  (*in_axis)[0] = out_info;
  return false;
}

bool ReluScaleAxisForward(
    const NodeAttrs& attrs,
    const std::vector<TShape>& in_shape,
    const std::vector<TShape>& out_shape,
    std::vector<FoldChainInfo>* in_info,
    FoldChainInfo* out_info) {
  *out_info = (*in_info)[0];
  return false;
}

NNVM_REGISTER_OP(relu)
.set_attr<FScaleAxisBackward>("FScaleAxisBackward", ReluScaleAxisBackward);

NNVM_REGISTER_OP(leaky_relu)
.set_attr<FScaleAxisBackward>("FScaleAxisBackward", ReluScaleAxisBackward);

NNVM_REGISTER_OP(relu)
.set_attr<FScaleAxisForward>("FScaleAxisForward", ReluScaleAxisForward);

NNVM_REGISTER_OP(leaky_relu)
.set_attr<FScaleAxisForward>("FScaleAxisForward", ReluScaleAxisForward);

// property registration.
template <typename T>
bool Pool2DBackward(
    const NodeAttrs& attrs,
    const std::vector<TShape>& in_shape,
    const std::vector<TShape>& out_shape,
    const FoldChainInfo& out_info,
    std::vector<FoldChainInfo>* in_axis) {
  const T& param = nnvm::get<T>(attrs.parsed);
  if (out_info.axis == 1 && param.layout == "NCHW") {
    (*in_axis)[0] = out_info;
  }
  return false;
}

template <typename T>
bool Pool2DForward(
    const NodeAttrs& attrs,
    const std::vector<TShape>& in_shape,
    const std::vector<TShape>& out_shape,
    std::vector<FoldChainInfo>* in_info,
    FoldChainInfo* out_info) {
  const T& param = nnvm::get<T>(attrs.parsed);
  if ((*in_info)[0].axis == 1 && param.layout == "NCHW") {
    *out_info = (*in_info)[0];
  }
  return false;
}

NNVM_REGISTER_OP(max_pool2d)
.set_attr<FScaleAxisBackward>("FScaleAxisBackward", Pool2DBackward<top::MaxPool2DParam>);

NNVM_REGISTER_OP(avg_pool2d)
.set_attr<FScaleAxisBackward>("FScaleAxisBackward", Pool2DBackward<top::AvgPool2DParam>);

NNVM_REGISTER_OP(max_pool2d)
.set_attr<FScaleAxisForward>("FScaleAxisForward", Pool2DForward<top::MaxPool2DParam>);

NNVM_REGISTER_OP(avg_pool2d)
.set_attr<FScaleAxisForward>("FScaleAxisForward", Pool2DForward<top::AvgPool2DParam>);



bool BroadcastAddSubScaleAxisBackward(
    const NodeAttrs& attrs,
    const std::vector<TShape>& in_shape,
    const std::vector<TShape>& out_shape,
    const FoldChainInfo& out_info,
    std::vector<FoldChainInfo>* in_axis) {
  if (out_info.kind != kPending) return false;
  for (int i = 0; i < 2; ++i) {
    std::pair<int, int> m = MatchBroadcast1DAxis(out_shape[0], in_shape[1 - i]);
    if (m.second != -1 &&
        in_shape[i] == out_shape[0] &&
        m.first == out_info.axis) {
      (*in_axis)[i].kind = kPending;
      (*in_axis)[i].axis = out_info.axis;
      (*in_axis)[i].source = out_info.source;
      (*in_axis)[1 - i].kind = kMulConsumer;
      (*in_axis)[1 - i].axis = m.second;
      (*in_axis)[1 - i].source = out_info.source;
      return false;
    }
  }
  return false;
}

bool BroadcastAddSubScaleAxisForward(
    const NodeAttrs& attrs,
    const std::vector<TShape>& in_shape,
    const std::vector<TShape>& out_shape,
    std::vector<FoldChainInfo>* in_info,
    FoldChainInfo* out_info) {
  for (int i = 0; i < 2; ++i) {
    if ((*in_info)[i].kind == kPending) {
      std::pair<int, int> m = MatchBroadcast1DAxis(out_shape[0], in_shape[1 - i]);
      if (m.second != -1 &&
          in_shape[i] == out_shape[0] &&
          m.first == (*in_info)[i].axis) {
        out_info->kind = kPending;
        out_info->axis = m.first;
        out_info->source = (*in_info)[i].source;
        (*in_info)[1 - i].kind = kDivConsumer;
        (*in_info)[1 - i].axis = m.second;
        (*in_info)[1 - i].source = (*in_info)[i].source;
        return false;
      }
    }
  }
  return false;
}

NNVM_REGISTER_OP(broadcast_add)
.set_attr<FScaleAxisBackward>("FScaleAxisBackward", BroadcastAddSubScaleAxisBackward);

NNVM_REGISTER_OP(broadcast_sub)
.set_attr<FScaleAxisBackward>("FScaleAxisBackward", BroadcastAddSubScaleAxisBackward);

NNVM_REGISTER_OP(broadcast_add)
.set_attr<FScaleAxisForward>("FScaleAxisForward", BroadcastAddSubScaleAxisForward);

NNVM_REGISTER_OP(broadcast_sub)
.set_attr<FScaleAxisForward>("FScaleAxisForward", BroadcastAddSubScaleAxisForward);

bool Conv2DScaleAxisBackward(
    const NodeAttrs& attrs,
    const std::vector<TShape>& in_shape,
    const std::vector<TShape>& out_shape,
    const FoldChainInfo& out_info,
    std::vector<FoldChainInfo>* in_axis) {
  using top::Conv2DParam;
  const Conv2DParam& param = nnvm::get<Conv2DParam>(attrs.parsed);
  if (out_info.kind != kPending) return false;
  // only optimize for kernel layout OIHW for now
  if (param.kernel_layout == "OIHW" && out_info.axis == 1) {
    (*in_axis)[1].kind = kMulConsumer;
    (*in_axis)[1].axis = 0;
    (*in_axis)[1].source = out_info.source;
    if (param.use_bias) {
      (*in_axis)[2].kind = kMulConsumer;
      (*in_axis)[2].axis = 0;
      (*in_axis)[2].source = out_info.source;
    }
    return true;
  } else {
    return false;
  }
}

bool Conv2DScaleAxisForward(
    const NodeAttrs& attrs,
    const std::vector<TShape>& in_shape,
    const std::vector<TShape>& out_shape,
    std::vector<FoldChainInfo>* in_info,
    FoldChainInfo* out_info) {
  using top::Conv2DParam;
  const Conv2DParam& param = nnvm::get<Conv2DParam>(attrs.parsed);
  if ((*in_info)[0].kind != kPending) return false;
  // only optimize for nchw for now
  if (param.kernel_layout == "OIHW" && (*in_info)[0].axis == 1) {
    // Check whether it is depthwise conv2d
    if (param.use_bias) {
      CHECK_EQ(in_shape.size(), 3U) << "Input:[data, weight, bias]";
    } else {
      CHECK_EQ(in_shape.size(), 2U) << "Input:[data, weight]";
    }

    auto dshape = in_shape.at(0);
    CHECK_EQ(dshape.ndim(), 4U) << "Input data shape should be 4D";

    // TODO(FrozenGene): Currently, we don't support conv2d's groups != in channels.
    if (param.groups > 1 && dshape[1] != param.groups) {
      LOG(WARNING) << "FoldScaleAxis optimization doesn't support conv2d "
                   << "with groups != in channels. We will skip FoldScaleAxis "
                   << "optimization for this op.";
      return false;
    }


    // input channel equals to groups, which means depthwise conv2d
    bool is_depthwise_conv2d = (dshape[1] == param.groups);

    // if it is depthwise convolution, the weight fold axis should along to axis 0.
    // For example:
    // data shape [1,54,63,127] weights shape [54,1,3,3], scale shape [54]
    // depthwise convolution's weights shape means we have divided the data shape's channel
    // to groups parties. Here, we divide 54 channels into 54 parties. Every part size is 1.
    // weights shape's first dimision means how many parties we have divided (mapping to
    // input shape's channel). So, in the depthwise convolution, we shouldn't do like
    // traditional convolution(i.e. OIHW)

    // Backgroud of this algorithm:

    // Original Graph:
    //    Graph(%x,
    //          %in_scale,
    //          %weight,
    //          %bias,
    //          %out_scale) {
    //      %1 = __add_scalar__(%x, scalar='1')
    //      %3 = expand_dims(%in_scale, num_newaxis='2', axis='1')
    //      %4 = broadcast_mul(%1, %3)
    //      %7 = conv2d(%4, %weight, %bias, padding='(1, 1)', kernel_size='(3, 3)', channels='2')
    //      %8 = relu(%7)
    //      %10 = expand_dims(%out_scale, num_newaxis='2', axis='1')
    //      %11 = broadcast_mul(%8, %10)
    //      ret %11
    //    }

    // Optimized Graph:
    //    Graph(%x,
    //          %weight,
    //          %out_scale,
    //          %in_scale,
    //          %bias) {
    //      %1 = __add_scalar__(%x, scalar='1')
    //      %4 = expand_dims(%out_scale, num_newaxis='3', axis='1')
    //      %5 = broadcast_mul(%weight, %4)
    //      %7 = expand_dims(%in_scale, num_newaxis='2', axis='1')
    //      %8 = broadcast_mul(%5, %7)
    //      %10 = broadcast_mul(%bias, %out_scale)
    //      %11 = conv2d(%1, %8, %10, padding='(1, 1)', kernel_size='(3, 3)', channels='2')
    //      %12 = relu(%11)
    //      ret %12
    //    }

    // Conv2DScaleAxisForward will need in_scale. Conv2DScaleAxisBackward will need out_scale.
    // in_scale will apply into input data's channel (in_channel). out_scale will apply in
    // conv2d's result, which will apply in weight's output channel.
    // So, default Conv2DScaleAxisForward will fold axis 1 (weights' input channel).
    // Conv2DScaleAxisBackward will fold axis 0 (weights' output channel).
    // But depthwise convolution is another story as said previously.
    (*in_info)[1].kind = kMulConsumer;
    (*in_info)[1].axis = is_depthwise_conv2d ? 0 : 1;
    (*in_info)[1].source = (*in_info)[0].source;
    return true;
  } else {
    return false;
  }
}

NNVM_REGISTER_OP(conv2d)
.set_attr<FScaleAxisBackward>("FScaleAxisBackward", Conv2DScaleAxisBackward);

NNVM_REGISTER_OP(conv2d)
.set_attr<FScaleAxisForward>("FScaleAxisForward", Conv2DScaleAxisForward);

}  // namespace compiler
}  // namespace nnvm
