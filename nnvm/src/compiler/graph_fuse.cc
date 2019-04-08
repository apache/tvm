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
#include <memory>
#include <utility>
#include <limits>
#include <unordered_map>

#include "graph_fuse.h"
#include "graph_runtime.h"
#include "pattern_util.h"

namespace nnvm {
namespace compiler {
using namespace tvm;

// Partition the graph into segments
// Each segment will be compiled into one operator.
// Also mark the property of the segment.
nnvm::Graph GraphFindFusibleGroups(nnvm::Graph g) {
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
      // Check if we can fuse to the master.
      int chosen_master = -1;
      bool ewise = inode.source->num_outputs() == 1;
      bool mark_as_injective = false;
      for (const auto& e : inode.inputs) {
        if (fuse_vec[e.node_id] == FuseRule::kUknown) {
          TOpPattern ipt = pattern_vec[e.node_id];
          if (ipt != kElemWise) ewise = false;
          if (ipt <= kBroadcast) {
            fuse_vec[e.node_id] = FuseRule::kFuseToMaster;
          } else if (ipt == kInjective) {
            fuse_vec[e.node_id] = FuseRule::kFuseToMaster;
            mark_as_injective = true;
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
      } else if (mark_as_injective) {
        pt = kInjective;
      } else {
        pt = ewise ? kElemWise : kBroadcast;
      }
    } else if (pt == kInjective || pt == kCommReduce) {
      // Fuse to the comm reduce or injective
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
      // Realize
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

  // Point to the group root id of each node.
  GroupVec group_vec(idx.num_nodes(), -1);
  std::vector<std::vector<uint32_t> > node_ids_per_group(idx.num_nodes());
  for (uint32_t i = idx.num_nodes(); i != 0; --i) {
    uint32_t nid = i - 1;
    const auto& inode = idx[nid];
    bool is_root = false;
    if (group_vec[nid] == -1) {
      group_vec[nid] = nid;
      node_ids_per_group[nid].push_back(nid);
      is_root = true;
    }

    // Check if injective op and out_ewise_fusable op (e.g. conv2d) are in the same group.
    bool parent_out_ewise = false;
    bool parent_injective = false;
    for (const auto& e : inode.inputs) {
      if (fuse_vec[e.node_id] != FuseRule::kFuseToMaster) continue;
      TOpPattern pt = pattern_vec[e.node_id];
      if (pt == kOutEWiseFusable) {
        parent_out_ewise = true;
      } else if (pt == kInjective) {
        parent_injective = true;
      }
    }
    // Change the master node from out_ewise_fusable op to itself
    if (parent_injective && parent_out_ewise) {
      master_vec[nid] = nid;
      if (!is_root) {
        // Children nodes in the same group might be pointing to a master node in a different group.
        for (uint32_t j : node_ids_per_group[group_vec[nid]]) {
          master_vec[j] = nid;
        }
      }
    }

    // Propagate the group id.
    for (const auto& e : inode.inputs) {
      TOpPattern pt = pattern_vec[e.node_id];
      if (parent_out_ewise && parent_injective) {
        if (pt == kOutEWiseFusable) {
          continue;  // Do not fuse out_ewise_fusable op
        } else if (pt == kInjective) {
          master_vec[e.node_id] = nid;
        }
      }
      if (fuse_vec[e.node_id] == FuseRule::kFuseToMaster) {
        CHECK(group_vec[e.node_id] == -1||
              group_vec[e.node_id] == group_vec[nid]);
        group_vec[e.node_id] = group_vec[nid];
        node_ids_per_group[group_vec[nid]].push_back(e.node_id);
      }
    }
  }

  /*
     Above algorithm will not fuse a node whose output is fed to more than one
     child node. This is because in general, it does not make sense to fuse multiple
     children branches with their parent, as in the following example.

            conv2d
            /  |  \
           /   |   \
         op    op   op
          |    |    |
          |    |    |

     However, when all children branches meet at a certain node, there is a possibility for
     further operator fusion. For example, all nodes in the following subgraph can be fused
     into a single node, if three 'in-between' nodes and the bottom node are all element wise
     operation.

            conv2d
            /  |  \
           /   |   \
         op    op   op
          \    |    /
           \   |   /
          elemwise add
               |

     This pattern is not uncommon. For example, it arises when conv2d op is followed by exponential
     linear unit. If bias add and batch normalization are also present, they can be fused as well.

     In fact, above fusion algorithm already fuses three in-between nodes and the element wise
     add node in the figure above. The following code fuses the conv2d node with the already
     fused children nodes. The following patterns are supported.

     * Any number of child nodes from the top node
     * The path from the top node to bottom node can contain any number of element wise ops.

     The only restriction is that in-between nodes cannot have more than one child.

     The overview of the algorithm below is as follows:

     1. Check if all children nodes are fused into a single op by the existing fusion algorithm
     2. Fuse the parent node to children nodes, and update its group id to be the children's group id
     3. If the parent node originally belongs to another group (for example, conv + batch norm),
        propagate the new group id to a grand parent and upward
  */
  if (opt_level >= 1) {
    std::vector<std::vector<uint32_t> > children_group_ids(idx.num_nodes());
    for (uint32_t nid = idx.num_nodes() - 1; nid != 0; --nid) {
      const auto& inode = idx[nid];
      if (inode.source->is_variable()) continue;
      CHECK_NE(group_vec[nid], -1);
      if (inode.inputs.size() != 1) continue;
      const uint32_t parent_nid = inode.inputs[0].node_id;
      // if parent node has more than one child, record each child's group id.
      if (ref_count[parent_nid] > 1) children_group_ids[parent_nid].push_back(group_vec[nid]);
    }

    std::vector<int> new_group_id(idx.num_nodes(), -1);
    for (uint32_t nid = idx.num_nodes() - 1; nid != 0; --nid) {
      if (new_group_id[group_vec[nid]] != -1) {
        // propagate new group id from child
        group_vec[nid] = new_group_id[group_vec[nid]];
      }
      TOpPattern pt = op_pattern.get(idx[nid].source->op(), kOpaque);
      if (pt == kOpaque) continue;
      const auto& group_ids = children_group_ids[nid];
      if (group_ids.size() <= 1) continue;
      const uint32_t child_group_id = group_ids[0];
      const auto& children_node_ids = node_ids_per_group[child_group_id];

      auto is_same_group_id = [child_group_id](uint32_t id) {
          return id == child_group_id;
      };
      auto is_fusible_pattern = [&idx](uint32_t child_nid) {
        TOpPattern child_pt = op_pattern.get(idx[child_nid].source->op(), kOpaque);
        return child_pt  <= kBroadcast;
      };
      // fuse this node with children if
      // all children belong to the same group and
      // all nodes in the group are element wise or broadcast op.
      const bool can_be_fused = std::all_of(group_ids.begin(), group_ids.end(), is_same_group_id) &&
        std::all_of(children_node_ids.begin(), children_node_ids.end(), is_fusible_pattern);

      if (can_be_fused) {
        new_group_id[group_vec[nid]] = child_group_id;
        group_vec[nid] = child_group_id;
        for (uint32_t nid2 : node_ids_per_group[child_group_id]) {
          pattern_vec[nid2] = pattern_vec[nid];
          master_vec[nid2] = master_vec[nid];
        }
      }
    }
  }

  g.attrs["group_root"] = std::make_shared<any>(std::move(group_vec));
  g.attrs["group_master"] = std::make_shared<any>(std::move(master_vec));
  g.attrs["pattern"] = std::make_shared<any>(std::move(pattern_vec));
  return g;
}

NNVM_REGISTER_PASS(GraphFindFusibleGroups)
.set_body(GraphFindFusibleGroups)
.depend_graph_attr("shape")
.depend_graph_attr("dtype");

// Fuse the partitioned graph into segments.
// Create a new graph with fused nodes.
// Also inherit attribute shape, dltype from the previous graph.
nnvm::Graph GraphFuse(nnvm::Graph g) {
  CHECK(g.HasAttr("group_root") && g.HasAttr("pattern"))
      << "GraphFindFusibleGroups pass hasn't been applied yet.";

  const IndexedGraph& idx = g.indexed_graph();
  // Get attributes from the graph
  const ShapeVector& shape_vec = g.GetAttr<ShapeVector>("shape");
  const DTypeVector& dtype_vec = g.GetAttr<DTypeVector>("dtype");
  const GroupVec& group_vec = g.GetAttr<GroupVec>("group_root");
  const PatternVec& pattern_vec = g.GetAttr<PatternVec>("pattern");

  // Specially handle assign op.
  const nnvm::Op* assign_op = nnvm::Op::Get("_assign");

  FuseEntryVec fuse_entries(idx.num_nodes());
  // Setup inputs and placeholder.
  for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
    const auto& inode = idx[nid];
    if (inode.source->is_variable()) continue;
    CHECK_GE(group_vec[nid], 0);
    int root_id = group_vec[nid];
    FuseEntry& fe = fuse_entries[root_id];
    fe.flatten_data = (pattern_vec[root_id] == kElemWise ||
                       inode.source->op() == assign_op);
    for (const auto& e : inode.inputs) {
      if (group_vec[e.node_id] != root_id && fe.imap.count(e) == 0) {
        Array<Expr> shape;
        if (fe.flatten_data) {
          // Elementwise support flatten
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
    FuseEntry& fe = fuse_entries[root_id];
    // Create a subgraph node.
    NodePtr gnode = Node::Create();
    gnode->attrs = inode.source->attrs;
    // Set input entries for the subgraph node.
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
    // Schedule on the root node and use the master's schedule
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
  g.attrs["fused_entry"] = std::make_shared<any>(std::move(fuse_entries));
  return g;
}

NNVM_REGISTER_PASS(GraphFuse)
    .set_body(GraphFuse)
    .set_change_graph(true)
    .provide_graph_attr("fused_entry")
    .depend_graph_attr("shape")
    .depend_graph_attr("dtype")
    .depend_graph_attr("group_root")
    .depend_graph_attr("group_master");

}  // namespace compiler
}  // namespace nnvm
