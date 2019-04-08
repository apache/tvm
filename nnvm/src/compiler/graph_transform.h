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
 * Copyright (c) 2017 by Contributors
 * \file graph_transform.h
 * \brief A mutator class that does local pattern matching and mutates a node.
*/
#ifndef NNVM_COMPILER_GRAPH_TRANSFORM_H_
#define NNVM_COMPILER_GRAPH_TRANSFORM_H_

#include <nnvm/graph.h>
#include <vector>
#include <utility>
#include <unordered_map>

namespace nnvm {
namespace compiler {

/*!
 * \brief Transform the graph to build a new Graph, in post DFS order.
 *
 *  Automatically copies node when some of its children or control_deps changed.
 *  This function won't be called in Variable.
 *
 * \param graph The original graph
 *
 * \param ftransform Function of (int nid, const NodePtr& node, std::vector<NodeEntry>* out) -> bool
 *
 *      If empty vector is returned, it means original entries should be kept.
 *
 * \tparam FTransform The transformation function.
 */
template<typename FTransform>
Graph GraphTransform(Graph graph, FTransform ftransform) {
  const IndexedGraph& idx = graph.indexed_graph();
  // new nodes
  std::vector<NodeEntry> new_entry_map(idx.num_node_entries());
  std::vector<bool> updated(idx.num_node_entries(), false);

  // setup inputs and placeholder.
  for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
    const auto& inode = idx[nid];
    bool need_copy = false;
    for (const IndexedGraph::NodeEntry& e : inode.inputs) {
      if (updated[idx.entry_id(e)]) {
        need_copy = true; break;
      }
    }
    if (!need_copy) {
      for (const uint32_t cid : inode.control_deps) {
        const auto& cnode = idx[cid];
        for (uint32_t i = 0 ; i < cnode.source->num_outputs(); ++i) {
          if (updated[idx.entry_id(cid, i)]) {
            need_copy = true;
          }
        }
        if (need_copy) break;
      }
    }

    if (!need_copy) {
      std::vector<NodeEntry> ret;
      if (ftransform(nid, inode.weak_ref.lock(), &ret)) {
        CHECK_EQ(ret.size(), static_cast<size_t>(inode.source->num_outputs()));
        for (uint32_t i = 0 ; i < inode.source->num_outputs(); ++i) {
          updated[idx.entry_id(nid, i)] = true;
          new_entry_map[idx.entry_id(nid, i)] = ret[i];
        }
      }
    } else {
      NodePtr node = Node::Create();
      node->attrs = inode.source->attrs;
      for (size_t i = 0; i < inode.inputs.size(); ++i) {
        const IndexedGraph::NodeEntry& e = inode.inputs[i];
        if (updated[idx.entry_id(e)]) {
          node->inputs.push_back(new_entry_map[idx.entry_id(e)]);
        } else {
          node->inputs.push_back(inode.source->inputs[i]);
        }
      }
      for (size_t i = 0; i < inode.control_deps.size(); ++i) {
        const uint32_t cid = inode.control_deps[i];
        const auto& cnode = idx[cid];
        CHECK_NE(cnode.source->num_outputs(), 0U);
        NodePtr selected_ptr;
        for (uint32_t j = 0 ; j < cnode.source->num_outputs(); ++j) {
          NodePtr cptr = updated[idx.entry_id(cid, j)] ?
              new_entry_map[idx.entry_id(cid, j)].node : inode.source->control_deps[i];
          if (selected_ptr == nullptr) {
            selected_ptr = std::move(cptr);
          } else {
            CHECK(selected_ptr.get() == cptr.get())
                << "Control dependency node changed to more than one node";
          }
        }
        node->control_deps.push_back(selected_ptr);
      }
      std::vector<NodeEntry> ret;
      if (ftransform(nid, node, &ret)) {
        CHECK_EQ(ret.size(), static_cast<size_t>(inode.source->num_outputs()));
        for (uint32_t i = 0 ; i < inode.source->num_outputs(); ++i) {
          updated[idx.entry_id(nid, i)] = true;
          new_entry_map[idx.entry_id(nid, i)] = ret[i];
        }
      } else {
        for (uint32_t i = 0 ; i < inode.source->num_outputs(); ++i) {
          updated[idx.entry_id(nid, i)] = true;
          new_entry_map[idx.entry_id(nid, i)] = NodeEntry{node, i, 0};
        }
      }
    }
  }
  Graph ret;
  for (size_t i = 0; i < idx.outputs().size(); ++i) {
    const IndexedGraph::NodeEntry& e = idx.outputs()[i];
    if (updated[idx.entry_id(e)]) {
      ret.outputs.push_back(new_entry_map[idx.entry_id(e)]);
    } else {
      ret.outputs.push_back(graph.outputs[i]);
    }
  }
  return ret;
}

}  // namespace compiler
}  // namespace nnvm

#endif  // NNVM_COMPILER_GRAPH_TRANSFORM_H_
