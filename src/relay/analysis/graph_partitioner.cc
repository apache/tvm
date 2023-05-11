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

#include "./graph_partitioner.h"

#include <vector>

namespace tvm {
namespace relay {

DominatorTree DominatorTree::PostDom(support::Arena* arena, const IndexedForwardGraph& graph) {
  DominatorTree tree;
  tree.nodes.resize(graph.post_dfs_order.size(), nullptr);
  // reverse topo order
  for (size_t i = graph.post_dfs_order.size(); i != 0; --i) {
    size_t index = i - 1;
    tree.nodes[index] = tree.GetNode(arena, graph.post_dfs_order[index]);
  }
  return tree;
}

DominatorTree::Node* DominatorTree::LeastCommonAncestor(Node* lhs, Node* rhs,
                                                        OpPatternKind* edge_pattern) {
  while (lhs != rhs) {
    if (lhs == nullptr) return nullptr;
    if (rhs == nullptr) return nullptr;
    if (lhs->depth < rhs->depth) {
      edge_pattern[0] = CombinePattern(edge_pattern[0], rhs->pattern);
      rhs = rhs->parent;
    } else if (rhs->depth < lhs->depth) {
      edge_pattern[0] = CombinePattern(edge_pattern[0], lhs->pattern);
      lhs = lhs->parent;
    } else {
      edge_pattern[0] = CombinePattern(edge_pattern[0], lhs->pattern);
      edge_pattern[0] = CombinePattern(edge_pattern[0], rhs->pattern);
      lhs = lhs->parent;
      rhs = rhs->parent;
    }
  }
  return lhs;
}

DominatorTree::Node* DominatorTree::LeastCommonAncestor(
    const LinkedList<IndexedForwardGraph::Edge>& input_nodes, OpPatternKind* edge_pattern) {
  auto link = input_nodes.head;
  if (link == nullptr) {
    return nullptr;
  }
  auto get_node = [&](const IndexedForwardGraph::Edge& edge) {
    size_t oindex = edge.node->index;
    ICHECK_LT(oindex, nodes.size());
    Node* onode = nodes[oindex];
    ICHECK(onode != nullptr);
    return onode;
  };
  Node* parent = get_node(link->value);
  *edge_pattern = CombinePattern(*edge_pattern, link->value.pattern);
  link = link->next;
  for (; link != nullptr; link = link->next) {
    parent = LeastCommonAncestor(parent, get_node(link->value), edge_pattern);
    *edge_pattern = CombinePattern(*edge_pattern, link->value.pattern);
  }
  return parent;
}

DominatorTree::Node* DominatorTree::GetNode(support::Arena* arena,
                                            IndexedForwardGraph::Node* gnode) {
  Node* tnode = arena->make<Node>();
  tnode->gnode = gnode;
  if (gnode->extern_ref) {
    tnode->depth = 1;
    tnode->parent = nullptr;
    tnode->pattern = kOpaque;
  } else {
    // find the LCAs of all outputs.
    OpPatternKind pattern = kElemWise;
    Node* parent = LeastCommonAncestor(gnode->outputs, &pattern);
    tnode->depth = parent ? parent->depth + 1 : 1;
    tnode->parent = parent;
    tnode->pattern = pattern;
  }
  return tnode;
}

std::vector<GraphPartitioner::Group*> GraphPartitioner::Partition(
    const IndexedForwardGraph& graph) {
  this->InitGroups(graph);
  if (opt_level_ == 0) return std::move(groups_);
  // get post dominator tree
  auto post_dom_tree = DominatorTree::PostDom(arena_, graph);
  // run fusion algorithm.
  for (int phase = 0; phase < 3; ++phase) {
    this->RunFuse(graph, post_dom_tree, phase);
  }
  return std::move(groups_);
}

GraphPartitioner::Group* GraphPartitioner::Group::FindRoot() {
  // fast path
  if (this->parent == nullptr) return this;
  // slow path with path compression.
  Group* root = this;
  while (root->parent != nullptr) {
    root = root->parent;
  }
  for (Group* p = this; p != root;) {
    Group* parent = p->parent;
    p->parent = root;
    p = parent;
  }
  return root;
}

template <typename F>
bool GraphPartitioner::CheckPath_(IndexedForwardGraph::Node* src, IndexedForwardGraph::Node* sink,
                                  F fcond) {
  if (visited_.count(src)) return true;
  visited_.insert(src);
  Group* gnode = groups_[src->index];
  ICHECK(gnode != nullptr);
  gnode = gnode->FindRoot();
  if (!fcond(gnode->pattern, src == sink)) return false;
  if (src == sink) return true;
  for (auto link = src->outputs.head; link != nullptr; link = link->next) {
    if (!CheckPath_(link->value.node, sink, fcond)) return false;
  }
  return true;
}

template <typename F>
bool GraphPartitioner::CheckPath(IndexedForwardGraph::Node* src, IndexedForwardGraph::Node* sink,
                                 F fcond) {
  ICHECK(!src->extern_ref);
  visited_.clear();
  ICHECK(src != sink);
  for (auto link = src->outputs.head; link != nullptr; link = link->next) {
    if (!CheckPath_(link->value.node, sink, fcond)) return false;
  }
  return true;
}

OpPatternKind CombinePattern(OpPatternKind lhs, OpPatternKind rhs) {
  if (lhs > relay::kBroadcast && rhs > relay::kBroadcast) {
    LOG(FATAL) << "Cannot merge two complex group together";
  }
  if (lhs > rhs) return lhs;
  return rhs;
}

void GraphPartitioner::MergeFromTo(Group* child, Group* parent) {
  child = child->FindRoot();
  parent = parent->FindRoot();
  if (child == parent) return;
  // update the number of nodes of the parent group
  parent->num_nodes += child->num_nodes;
  child->parent = parent;
  // update anchor ref and pattern
  if (child->anchor_ref != nullptr) {
    ICHECK(parent->anchor_ref == nullptr);
    parent->anchor_ref = child->anchor_ref;
    parent->pattern = CombinePattern(child->pattern, parent->pattern);
  }
}

void GraphPartitioner::CommitFuse_(IndexedForwardGraph::Node* src, IndexedForwardGraph::Node* sink,
                                   Group* target) {
  if (src == sink) return;
  if (visited_.count(src)) return;
  visited_.insert(src);
  Group* gnode = groups_[src->index];
  ICHECK(gnode != nullptr);
  // merge the current group to the parent if possible.
  MergeFromTo(gnode, target);
  for (auto link = src->outputs.head; link != nullptr; link = link->next) {
    CommitFuse_(link->value.node, sink, target);
  }
}

void GraphPartitioner::CommitFuse(IndexedForwardGraph::Node* src, IndexedForwardGraph::Node* sink) {
  Group* target = groups_[sink->index];
  visited_.clear();
  ICHECK(src != sink);
  CommitFuse_(src, sink, target);
}

size_t GraphPartitioner::CountNodesUptoSink_(IndexedForwardGraph::Node* src,
                                             IndexedForwardGraph::Node* sink) {
  if (src == sink || visited_.count(src)) return 0;
  visited_.insert(src);
  Group* gnode = groups_[src->index];
  ICHECK(gnode != nullptr);
  auto sum = gnode->num_nodes;
  for (auto link = src->outputs.head; link != nullptr; link = link->next) {
    sum += CountNodesUptoSink_(link->value.node, sink);
  }
  return sum;
}

size_t GraphPartitioner::CountFusedNodesWithNewChild(IndexedForwardGraph::Node* child,
                                                     IndexedForwardGraph::Node* dom_parent) {
  Group* target = groups_[dom_parent->index];
  visited_.clear();
  ICHECK(child != dom_parent);
  return target->FindRoot()->num_nodes + CountNodesUptoSink_(child, dom_parent);
}

void GraphPartitioner::InitGroups(const IndexedForwardGraph& graph) {
  groups_.resize(graph.post_dfs_order.size());
  for (size_t nid = 0; nid < groups_.size(); ++nid) {
    const auto* graph_node = graph.post_dfs_order[nid];
    auto* group_node = arena_->make<Group>();
    group_node->pattern = graph_node->pattern;
    group_node->root_ref = graph_node->ref;
    // set anchor ref if necessary.
    if (group_node->pattern == relay::kOutEWiseFusable) {
      group_node->anchor_ref = graph_node->ref;
    }
    groups_[nid] = group_node;
  }
}

void GraphPartitioner::RunFuse(const IndexedForwardGraph& graph,    //
                               const DominatorTree& post_dom_tree,  //
                               int phase) {
  for (size_t nid = 0; nid < groups_.size(); ++nid) {
    // the group of current node has been specified already.
    auto* graph_node = graph.post_dfs_order[nid];
    auto* dom_node = post_dom_tree.nodes[nid];
    Group* group_node = groups_[nid];
    ICHECK(group_node != nullptr);
    // no actions for opaque nodes
    if (group_node->pattern == kOpaque) continue;
    // no actions needed if the current node have no dominator
    if (dom_node->parent == nullptr) continue;
    ICHECK(!graph_node->extern_ref);
    size_t dom_parent_gindex = dom_node->parent->gnode->index;

    // refuse the fusion if too many ops are going to be fused together
    if (CountFusedNodesWithNewChild(graph_node, dom_node->parent->gnode) > max_fuse_depth_)
      continue;

    if (phase == 2) {
      // Fuse injective ops into intermediate tuples, if any
      if (group_node->pattern > relay::kInjective) continue;
      Group* dom_parent_group = groups_[dom_parent_gindex];
      Group* dom_root_group = dom_parent_group->FindRoot();
      // If dom node group has a tuple as its root, we do not fuse tuple fields into it
      if (dom_root_group->pattern == relay::kTuple) continue;
      if (dom_parent_group->pattern == kTuple && dom_root_group->pattern <= relay::kInjective) {
        // Now we know the tuple has been fused into subsequent injective ops
        auto fcond = [](OpPatternKind kind, bool is_sink) { return kind <= kInjective; };
        // dom_root_group can also be tuple, as in inception layers
        // CheckPath is needed to avoid fusing two intermediate tuples
        if (CheckPath(graph_node, dom_node->parent->gnode, fcond)) {
          CommitFuse(graph_node, dom_node->parent->gnode);
        }
      }
      continue;
    }

    // Skip if current node is already fused to the parent.
    if (groups_[dom_parent_gindex] != nullptr &&
        group_node->FindRoot() == groups_[dom_parent_gindex]->FindRoot()) {
      continue;
    }
    // Do not fuse into tuple for now
    if (groups_[dom_parent_gindex]->pattern == kTuple) continue;
    // Try to fuse current node to its post-dominator.
    if (group_node->pattern == kOutEWiseFusable) {
      if (phase != 0) continue;
      // Path for OutEWiseFusable: conv2d
      // Check if the dominator relation is elemwise.
      if (dom_node->parent != nullptr && dom_node->pattern == kElemWise) {
        ICHECK(dom_node->parent->gnode != nullptr);
        // The fuse can be executed if all the intermediate ops are still broadcast.
        auto fcond = [](OpPatternKind kind, bool is_sink) { return kind <= kBroadcast; };
        if (CheckPath(graph_node, dom_node->parent->gnode, fcond)) {
          CommitFuse(graph_node, dom_node->parent->gnode);
        }
      }
    } else if (group_node->pattern <= kBroadcast) {
      // Pre-condition: can only be fused to parent which is injective or reduction.
      if (dom_node->parent != nullptr &&
          (dom_node->pattern <= kInjective || dom_node->pattern == kCommReduce)) {
        // Check if all the intermediate ops are still broadcast.
        // The final terminal node can already be fused to a OutEWiseFusable group.
        auto fcond = [](OpPatternKind kind, bool is_sink) {
          if (!is_sink) {
            // Elemwise, broadcast, and injective ops on the parallel branches
            // are allowed be fused to the elemwise/broadcast anchor.
            return kind <= kInjective;
          } else {
            return (kind <= kBroadcast || kind == kCommReduce || kind == kInjective ||
                    kind == kOutEWiseFusable);
          }
        };
        if (CheckPath(graph_node, dom_node->parent->gnode, fcond)) {
          CommitFuse(graph_node, dom_node->parent->gnode);
        }
      }
    } else if (group_node->pattern == kInjective || group_node->pattern == kTuple) {
      // defer injective fusion to second phase.
      // so conv2d always finishes fusing.
      if (phase != 1) continue;
      // Check if all path are injective.
      auto fcond = [](OpPatternKind kind, bool is_sink) { return kind <= kInjective; };
      if (CheckPath(graph_node, dom_node->parent->gnode, fcond)) {
        CommitFuse(graph_node, dom_node->parent->gnode);
      }
    } else {
      // do nothing.
      ICHECK(group_node->pattern == kCommReduce);
    }
  }
}

}  // namespace relay
}  // namespace tvm
