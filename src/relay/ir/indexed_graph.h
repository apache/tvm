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
 * \file src/relay/ir/indexed_graph.h
 * \brief A pattern matcher for matching dataflow properties.
 */
#ifndef TVM_RELAY_IR_INDEXED_GRAPH_H_
#define TVM_RELAY_IR_INDEXED_GRAPH_H_

#include <tvm/relay/dataflow_pattern.h>

#include <memory>
#include <stack>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace tvm {
namespace relay {

/*!
 * \brief A Wrapper around a templated graph type
 *  Holds a forward-backward indexed representation of the graph and a dominator tree representation
 * of the graph
 *
 *  This class is templated and the implementaiton is in the header file so we can analyze both
 * DFPattern and Expr with the same infrastructure.
 *
 *  IndexedGraph should be instantiated through the CreateIndexedGraph utilities.
 */
template <typename T>
class IndexedGraph {
 public:
  /*! \brief A Node that wraps the input type and represents the indexed graph and dominator tree */
  struct Node {
    /*! \brief Node Constructor
     *  \param ref The input graph node
     *  \param index The index of the node in toplogical order
     */
    Node(const T& ref, const size_t index) : ref_(ref), index_(index) {}

    /*! \brief The input node */
    const T ref_;
    /*! \brief The topological order index */
    const size_t index_;

    /*! \brief A boolean to determine if this node is external to the graph */
    bool is_external_ = false;
    /*! \brief The forward inputs of the node */
    std::vector<Node*> inputs_;
    /*! \brief The forward outputs/users of the node */
    std::vector<Node*> outputs_;

    /*! \brief The depth of the node in the dominator tree */
    size_t depth_ = 0;
    /*! \brief The dominator parent/final user of the outputs of this node */
    Node* dominator_parent_;
    /*! \brief The nodes this node dominates */
    std::vector<Node*> dominator_children_;

    bool Dominates(const Node* other) {
      std::stack<const Node*> stack;
      std::unordered_set<const Node*> visited;
      stack.push(this);
      while (!stack.empty()) {
        const Node* current = stack.top();
        stack.pop();
        for (auto node : current->dominator_children_) {
          if (visited.count(node) == 0) {
            if (other == node) {
              return true;
            } else {
              stack.push(node);
            }
            visited.insert(node);
          }
        }
      }
      return false;
    }
  };
  /*! \brief Construct the domination tree inside IndexedGraph */
  void PostDom() {
    for (size_t i = topological_order_.size(); i != 0; --i) {
      size_t index = i - 1;
      auto* current = topological_order_[index].get();
      if (current->is_external_) {
        current->depth_ = 1;
        current->dominator_parent_ = nullptr;
      } else {
        auto parent = LeastCommonAncestor(current->outputs_);
        current->depth_ = parent ? parent->depth_ + 1 : 1;
        current->dominator_parent_ = parent;
        parent->dominator_children_.push_back(current);
      }
    }
  }
  /*! \brief Map of input nodes to IndexedGraph Nodes */
  std::unordered_map<T, std::shared_ptr<Node>, ObjectPtrHash, ObjectPtrEqual> node_map_;
  /*! \brief Topological IndexedGraph Nodes */
  std::vector<std::shared_ptr<Node>> topological_order_;

 protected:
  /*! \brief Find the least common ancestor of all outputs of a node */
  Node* LeastCommonAncestor(const std::vector<Node*>& outputs) {
    if (outputs.size() == 0) {
      return nullptr;
    }
    auto parent = outputs.at(0);
    for (size_t i = 1; i < outputs.size(); ++i) {
      parent = LeastCommonAncestor(parent, outputs.at(i));
    }
    return parent;
  }

  /*! \brief Find the least common ancestor of two nodes */
  Node* LeastCommonAncestor(Node* lhs, Node* rhs) {
    if (lhs == nullptr || rhs == nullptr) {
      return nullptr;
    }
    while (lhs != rhs) {
      ICHECK(lhs);
      ICHECK(rhs);
      if (lhs->depth_ < rhs->depth_) {
        rhs = rhs->dominator_parent_;
      } else if (lhs->depth_ > rhs->depth_) {
        lhs = lhs->dominator_parent_;
      } else {
        rhs = rhs->dominator_parent_;
        lhs = lhs->dominator_parent_;
      }
    }
    return lhs;
  }
};

/*! \brief Create an Indexed Graph based on an Expr */
IndexedGraph<Expr> CreateIndexedGraph(const Expr& expr);
/*! \brief Create an Indexed Graph based on an DFPattern */
IndexedGraph<DFPattern> CreateIndexedGraph(const DFPattern& pattern);

}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_IR_INDEXED_GRAPH_H_
