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
 * \file src/relay/collage/dataflow_graph.cc
 * \brief A representation of the dataflow for an overall Relay expression.
 */

#include "./dataflow_graph.h"

namespace tvm {
namespace relay {
namespace collage {

DataflowGraph::DataflowGraph(Expr expr) : expr_(std::move(expr)) {
  indexed_graph_ = CreateIndexedGraph(expr_);
  downstream_map_.reserve(indexed_graph_->size());
  for (PostDfsIndex index = 0; index < indexed_graph_->size(); ++index) {
    const Node* node = indexed_graph_->index_to_node(index);
    std::unordered_set<const Node*> downstream_nodes;
    node->AccumulateDownstreamNodes(&downstream_nodes);
    IndexSet index_set(indexed_graph_->size());
    for (const Node* downstream_node : downstream_nodes) {
      index_set.Add(downstream_node->index_);
    }
    downstream_map_.emplace_back(std::move(index_set));
  }
}

}  // namespace collage
}  // namespace relay
}  // namespace tvm
