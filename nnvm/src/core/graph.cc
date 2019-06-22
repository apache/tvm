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
 *  Copyright (c) 2016 by Contributors
 * \file graph_attr_types.cc
 * \brief Graph node data structure.
 */
#include <nnvm/graph.h>
#include <nnvm/op_attr_types.h>
#include <limits>

namespace nnvm {

const IndexedGraph& Graph::indexed_graph() const {
  if (indexed_graph_ == nullptr) {
    indexed_graph_.reset(new IndexedGraph(*this));
  }
  return *indexed_graph_;
}

// a subgraph should not refer to any nodes with higher level
// where "level" refers to the nested depth of the subgraph
// e.g. the main graph is level 0
// subgraphs of the main graph is level 1
// subgraphs of the subgraphs of the main graph is level 2
static void SubgraphSanityCheck(const std::vector<std::shared_ptr<Symbol>> &subgraphs) {
  std::vector<const std::vector<nnvm::NodeEntry>*> curr_level;
  std::vector<const std::vector<nnvm::NodeEntry>*> next_level;
  std::unordered_map<nnvm::Node*, uint32_t> node2level;
  for (auto &subgraph : subgraphs)
    next_level.push_back(&subgraph->outputs);
  for (uint32_t level = 0; !next_level.empty(); ++level) {
    curr_level.swap(next_level);
    next_level.clear();
    for (const std::vector<NodeEntry> *graph_ptr : curr_level) {
      const std::vector<NodeEntry> &graph = *graph_ptr;
      DFSVisit(graph, [&next_level, &node2level, level](const NodePtr& n) {
        nnvm::Node *node = n.get();
        // if the node is visited, but on a different level, then check failed
        // if check failed here or before, we stop doing anything, but raise an error
        CHECK(!node2level.count(node) || node2level[node] == level)
          << "A subgraph should not depend on the outputs of nodes on higher levels";
        // otherwise, this node belongs to the current level
        node2level[node] = level;
        // subgraphs of current node belongs to next level
        for (const auto& subgraph : n->attrs.subgraphs) {
          next_level.push_back(&subgraph->outputs);
        }
      });
    }
  }
}

// implement constructor from graph
IndexedGraph::IndexedGraph(const Graph &g) {
  entry_rptr_.push_back(0);
  std::vector<size_t> inputs_rptr{0}, control_rptr{0};
  std::vector<std::shared_ptr<Symbol>> subgraphs;

  DFSVisit(g.outputs, [this, &inputs_rptr, &control_rptr, &subgraphs]
             (const NodePtr& n) {
      const auto& is_ghost = Op::GetAttr<TIsGhost>("TIsGhost");
      if (!n->is_variable() && is_ghost.get(n->op(), false)) return;
      CHECK_LT(nodes_.size(), std::numeric_limits<uint32_t>::max());
      uint32_t nid = static_cast<uint32_t>(nodes_.size());
      CHECK(n);
      for (const auto &subgraph : n->attrs.subgraphs)
        subgraphs.push_back(subgraph);
      // nodes_
      IndexedGraph::Node new_node;
      new_node.source = n.get();
      new_node.weak_ref = n;
      nodes_.emplace_back(std::move(new_node));
      // arg_nodes_
      if (n->is_variable()) {
        input_nodes_.push_back(nid);
      }
      // node2index_
      node2index_[n.get()] = nid;
      // entry rptr
      entry_rptr_.push_back(entry_rptr_.back() + n->num_outputs());
      // input entries
      for (const auto& e : n->inputs) {
        auto it = node2index_.find(e.node.get());
        CHECK(it != node2index_.end() && it->first == e.node.get());
        input_entries_.emplace_back(NodeEntry{it->second, e.index, e.version});
      }
      inputs_rptr.push_back(input_entries_.size());
      // control deps
      for (const auto& nptr : n->control_deps) {
        if (!nptr->is_variable() && is_ghost.get(nptr->op(), false)) continue;
        auto it = node2index_.find(nptr.get());
        CHECK(it != node2index_.end()) << "control dep not found in graph";
        control_deps_.push_back(it->second);
      }
      control_rptr.push_back(control_deps_.size());
  });
  if (!subgraphs.empty())
    SubgraphSanityCheck(subgraphs);

  for (const auto& e : g.outputs) {
    outputs_.emplace_back(NodeEntry{
        node2index_.at(e.node.get()), e.index, e.version});
  }

  static auto& fmutate_inputs = Op::GetAttr<FMutateInputs>("FMutateInputs");
  // setup array view
  // input_entries_ and control_rptr must not change after this step.
  const NodeEntry* iptr = dmlc::BeginPtr(input_entries_);
  for (size_t nid = 0; nid < nodes_.size(); ++nid) {
    nodes_[nid].inputs = array_view<NodeEntry>(
        iptr + inputs_rptr[nid], iptr + inputs_rptr[nid + 1]);
    if (nodes_[nid].source->op() != nullptr &&
        fmutate_inputs.count(nodes_[nid].source->op())) {
      for (uint32_t i : fmutate_inputs[nodes_[nid].source->op()](nodes_[nid].source->attrs)) {
        mutable_input_nodes_.insert(nodes_[nid].inputs[i].node_id);
      }
    }
  }
  const uint32_t* cptr = dmlc::BeginPtr(control_deps_);
  for (size_t nid = 0; nid < nodes_.size(); ++nid) {
    nodes_[nid].control_deps = array_view<uint32_t>(
        cptr + control_rptr[nid], cptr + control_rptr[nid + 1]);
  }
}

}  // namespace nnvm
