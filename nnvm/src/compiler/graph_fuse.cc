/*!
 *  Copyright (c) 2017 by Contributors
 * \file graph_fuse.cc
 * \brief Fuse the operators together.
 */
#include <dmlc/parameter.h>
#include <nnvm/compiler/packed_func_ext.h>
#include <nnvm/graph.h>
#include <nnvm/graph_attr_types.h>
#include <nnvm/node.h>
#include <nnvm/op_attr_types.h>
#include <nnvm/pass.h>
#include <nnvm/pass_functions.h>
#include <nnvm/tuple.h>
#include <tvm/lowered_func.h>
#include <tvm/runtime/packed_func.h>

#include "./graph_fuse.h"
#include "./graph_runtime.h"
#include "./pattern_util.h"

namespace nnvm {
namespace compiler {
using namespace tvm;

// Partition the graph into segments
// Each segment will be compiled into one operator.
// Need also mark the property of the segment.
nnvm::Graph GraphFusePartition(nnvm::Graph g) {
  // setup ref counter
  const IndexedGraph& idx = g.indexed_graph();
  int opt_level = 2;
  if (g.attrs.count("opt_level") != 0) {
    opt_level = g.MoveCopyAttr<int>("opt_level");
  }

  // Get attributes from the graph
  const ShapeVector& shape_vec = g.GetAttr<ShapeVector>("shape");

  // Reference counter of each op node
  // For now, always store result when an op is referred more than once.
  std::vector<uint32_t> ref_count = GetNodeRefCounts(idx);
  for (const auto& e : idx.outputs()) {
    // this line will realize all the outputs
    ref_count[e.node_id] += 1;
  }
  // Pattern for the subgraph
  PatternVec pattern_vec(idx.num_nodes(),  kOpaque);
  // Whether node can be fused to parent.
  std::vector<FuseRule> fuse_vec(idx.num_nodes(), FuseRule::kUknown);
  // Master node id of fusion segment.
  std::vector<int> master_vec(idx.num_nodes(), -1);
  // Operator pattern
  static auto& op_pattern = nnvm::Op::GetAttr<TOpPattern>("TOpPattern");

  for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
    const auto& inode = idx[nid];
    if (inode.source->is_variable()) {
      fuse_vec[nid] = FuseRule::kRealize; continue;
    }
    TOpPattern pt = op_pattern.get(inode.source->op(), kOpaque);

    if (pt <= kBroadcast) {
      // Try to check if we can fuse to the master.
      int chosen_master = -1;
      bool ewise = inode.source->num_outputs() == 1;
      for (const auto& e : inode.inputs) {
        if (fuse_vec[e.node_id] == FuseRule::kUknown) {
          TOpPattern ipt = pattern_vec[e.node_id];
          if (ipt != kElemWise) ewise = false;
          if (ipt <= kInjective) {
            fuse_vec[e.node_id] = FuseRule::kFuseToMaster;
          } else if (ipt == kOutEWiseFusable &&
                     chosen_master == -1 &&
                     shape_vec[idx.entry_id(nid, 0)] == shape_vec[idx.entry_id(e)]) {
            chosen_master = master_vec[e.node_id];
            fuse_vec[e.node_id] = FuseRule::kFuseToMaster;
          } else {
            fuse_vec[e.node_id] = FuseRule::kRealize;
          }
        }
        if (ewise) {
          if (shape_vec[idx.entry_id(nid, 0)] != shape_vec[idx.entry_id(e)]) {
            ewise = false;
          }
        }
      }
      master_vec[nid] = chosen_master;
      if (chosen_master != -1) {
        pt = kOutEWiseFusable;
      } else {
        pt = ewise ? kElemWise : kBroadcast;
      }
    } else if (pt == kInjective || pt == kCommReduce) {
      // fuse to the comm reduce or injective
      for (const auto& e : inode.inputs) {
        if (fuse_vec[e.node_id] == FuseRule::kUknown) {
          TOpPattern ipt = pattern_vec[e.node_id];
          if (ipt <= kInjective) {
            fuse_vec[e.node_id] = FuseRule::kFuseToMaster;
          } else {
            fuse_vec[e.node_id] = FuseRule::kRealize;
          }
        }
      }
      if (pt == kCommReduce) {
        master_vec[nid] = nid;
      }
    } else {
      // realize
      master_vec[nid] = nid;
      for (const auto& e : inode.inputs) {
        if (fuse_vec[e.node_id] == FuseRule::kUknown) {
          fuse_vec[e.node_id] = FuseRule::kRealize;
          if (master_vec[e.node_id] == -1) {
            master_vec[e.node_id] = e.node_id;
          }
        }
      }
    }

    pattern_vec[nid] = pt;
    if (ref_count[nid] > 1 || opt_level < 1) {
      fuse_vec[nid] = FuseRule::kRealize;
      if (master_vec[nid] == -1) {
        master_vec[nid] = nid;
      }
    }
  }

  // point to the group root id of each node
  GroupVec group_vec(idx.num_nodes(), -1);
  for (uint32_t i = idx.num_nodes(); i != 0; --i) {
    uint32_t nid = i - 1;
    const auto& inode = idx[nid];
    if (group_vec[nid] == -1) {
      group_vec[nid] = nid;
    }
    // propagate the group id.
    for (const auto& e : inode.inputs) {
      if (fuse_vec[e.node_id] == FuseRule::kFuseToMaster) {
        CHECK(group_vec[e.node_id] == -1||
              group_vec[e.node_id] == group_vec[nid]);
        group_vec[e.node_id] = group_vec[nid];
      }
    }
  }
  g.attrs["group_root"] = std::make_shared<any>(std::move(group_vec));
  g.attrs["group_master"] = std::make_shared<any>(std::move(master_vec));
  g.attrs["pattern"] = std::make_shared<any>(std::move(pattern_vec));
  return g;
}


NNVM_REGISTER_PASS(GraphFusePartition)
.set_body(GraphFusePartition)
.depend_graph_attr("shape")
.depend_graph_attr("dtype");

// Fuse the partitioned graph into segments.
// Create a new graph with fused noded.
// Also inheritate attribute shape, dltype from previous graph.
nnvm::Graph GraphFuse(nnvm::Graph&& g) {
  CHECK(g.HasAttr("group_root") && g.HasAttr("pattern"))
      << "GraphFusePartition pass hasn't been applied yet.";

  const IndexedGraph& idx = g.indexed_graph();
  // Get attributes from the graph
  const ShapeVector& shape_vec = g.GetAttr<ShapeVector>("shape");
  const DTypeVector& dtype_vec = g.GetAttr<DTypeVector>("dtype");
  const GroupVec& group_vec = g.GetAttr<GroupVec>("group_root");
  const PatternVec &pattern_vec = g.GetAttr<PatternVec>("pattern");

  // specially handle assign
  const nnvm::Op* assign_op = nnvm::Op::Get("_assign");

  FuseVec fuse_vec(idx.num_nodes());
  // setup inputs and placeholder.
  for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
    const auto& inode = idx[nid];
    if (inode.source->is_variable()) continue;
    CHECK_GE(group_vec[nid], 0);
    int root_id = group_vec[nid];
    FuseEntry& fe = fuse_vec[root_id];
    fe.flatten_data = (pattern_vec[root_id] == kElemWise ||
                       inode.source->op() == assign_op);
    for (const auto& e : inode.inputs) {
      if (group_vec[e.node_id] != root_id && fe.imap.count(e) == 0) {
        Array<Expr> shape;
        if (fe.flatten_data) {
          // elementwise support flatten
          int64_t prod = 1;
          for (int64_t x : shape_vec[idx.entry_id(e)]) {
            prod *= x;
          }
          CHECK_LE(prod, static_cast<int64_t>(std::numeric_limits<int>::max()));
          shape.push_back(make_const(Int(32), prod));
        } else {
          for (int64_t x : shape_vec[idx.entry_id(e)]) {
            CHECK_LE(x, static_cast<int64_t>(std::numeric_limits<int>::max()));
            shape.push_back(make_const(Int(32), x));
          }
        }
        std::ostringstream os_name;
        os_name << "input" << fe.imap.size();
        Tensor data = placeholder(
            shape, TVMType2Type(GetDLType(dtype_vec[idx.entry_id(e)])),
            os_name.str());
        NodeEntry garg = Symbol::CreateVariable(os_name.str()).outputs[0];
        fe.imap[e] = garg;
        fe.reverse_imap[garg.node.get()] = e;
        fe.input_info[garg.node.get()] = std::move(data);
      }
    }
  }
  // Setup the Subgraph
  std::vector<NodeEntry> subgraph_vec(idx.num_node_entries());
  for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
    const auto& inode = idx[nid];
    if (inode.source->is_variable()) continue;
    int root_id = group_vec[nid];
    FuseEntry& fe = fuse_vec[root_id];
    // copy and create subgraph node.
    NodePtr gnode = Node::Create();
    gnode->attrs = inode.source->attrs;
    // input loading
    for (const auto& e : inode.inputs) {
      if (group_vec[e.node_id] != root_id) {
        auto it = fe.imap.find(e);
        CHECK(it != fe.imap.end());
        gnode->inputs.push_back(it->second);
      } else {
        const NodeEntry& ne = subgraph_vec[idx.entry_id(e)];
        CHECK(!idx[e.node_id].source->is_variable());
        CHECK(ne.node != nullptr);
        gnode->inputs.push_back(ne);
      }
    }
    // schedule on root node, and use master's schedule
    if (static_cast<int>(nid) != root_id) {
      for (uint32_t index = 0; index < inode.source->num_outputs(); ++index) {
        uint32_t eid = idx.entry_id(nid, index);
        subgraph_vec[eid] = NodeEntry{gnode, index, 0};
      }
    } else {
      for (uint32_t index = 0; index < inode.source->num_outputs(); ++index) {
        fe.subgraph.outputs.push_back(NodeEntry{gnode, index, 0});
      }
    }
  }
  g.attrs["fused_entries"] = std::make_shared<any>(std::move(fuse_vec));
  return g;
}

NNVM_REGISTER_PASS(GraphFuse)
    .set_body(GraphFuse)
    .set_change_graph(true)
    .provide_graph_attr("fused_entries")
    .depend_graph_attr("shape")
    .depend_graph_attr("dtype")
    .depend_graph_attr("group_root")
    .depend_graph_attr("group_master");

}  // namespace compiler
}  // namespace nnvm
