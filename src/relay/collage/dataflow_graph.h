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
 * \file src/relay/collage/dataflow_graph.h
 * \brief A representation of the dataflow for an overall Relay expression.
 */
#ifndef TVM_RELAY_COLLAGE_DATAFLOW_GRAPH_H_
#define TVM_RELAY_COLLAGE_DATAFLOW_GRAPH_H_

#include <tvm/relay/expr.h>

#include <memory>
#include <vector>

#include "../ir/indexed_graph.h"
#include "./index_set.h"

namespace tvm {
namespace relay {
namespace collage {

/*!
 * \brief Represents the dataflow of an overall Relay expression.
 */
class DataflowGraph {
 public:
  using Node = IndexedGraph<Expr>::Node;

  explicit DataflowGraph(Expr expr);

  size_t size() const { return indexed_graph_->size(); }
  const Node* index_to_node(PostDfsIndex index) const {
    return indexed_graph_->index_to_node(index);
  }
  const Node* item_to_node(const Expr& expr) const { return indexed_graph_->item_to_node(expr); }
  const Node* item_to_node(const ExprNode* expr_node) const {
    return indexed_graph_->item_to_node(expr_node);
  }
  const Expr& expr() const { return expr_; }
  const IndexedGraph<Expr>& indexed_graph() const { return *indexed_graph_; }

  const IndexSet& downstream_of(PostDfsIndex index) const {
    ICHECK_LT(index, indexed_graph_->size());
    return downstream_map_[index];
  }

 private:
  /*! \brief The overall expression. */
  Expr expr_;
  /*! \brief The indexed graph which captures the main dataflow. */
  std::unique_ptr<IndexedGraph<Expr>> indexed_graph_;
  /*! \brief Map from a node's PostDfsIndex to the set of its downstream dataflow node indexes. */
  std::vector<IndexSet> downstream_map_;
};

}  // namespace collage
}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_COLLAGE_DATAFLOW_GRAPH_H_
