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
 * \file src/relay/analysis/dependency_graph.h
 * \brief create a dependency graph.
 */
#ifndef TVM_RELAY_ANALYSIS_DEPENDENCY_GRAPH_H_
#define TVM_RELAY_ANALYSIS_DEPENDENCY_GRAPH_H_

#include <tvm/relay/expr.h>

#include <unordered_map>
#include <vector>

#include "../../support/arena.h"
#include "../transforms/let_list.h"

namespace tvm {
namespace relay {

using support::LinkedList;
using support::LinkNode;

/* DependencyGraph track input and output of an Expr.
 * Additionally, dummy scope is created to model scope.
 * It allow us to traverse the graph in reverse order.
 */
class DependencyGraph {
 public:
  /*! \brief A node in the graph. */
  struct Node {
    // Determine scope boundaries. Used for calculating scopes, not for
    // constructing dependency graph.
    bool new_scope = false;
    // incoming edges
    LinkedList<Node*> children;
    // outgoing edges
    LinkedList<Node*> parents;
  };

  /*! \brief Maps a Relay Expr to its node in the dependency graph. */
  std::unordered_map<Expr, Node*, ObjectPtrHash, ObjectPtrEqual> expr_node;

  /*! \brief The dependency graph in post DFS order. */
  std::vector<Node*> post_dfs_order;

  /*!
   * \brief Create a dependency graph.
   * \param arena The arena used for data allocation.
   * \param body The body of the expression to create a graph.
   */
  static DependencyGraph Create(support::Arena* arena, const Expr& body);

 private:
  class Creator;
};

}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_ANALYSIS_DEPENDENCY_GRAPH_H_
