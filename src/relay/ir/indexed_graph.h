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
 * \brief A graph representation of the dataflow in a Relay expression.
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

/*! \brief The index of a node in the post-dfs traversal of overall expression. */
using PostDfsIndex = size_t;

/*!
 * \brief Represents the dataflow of an expression (or dataflow pattern) as a graph which is
 * overlaid on the underlying expression (or dataflow pattern) graph.
 *
 * Each graph node references the corresponding sub-expression (or dataflow sub-pattern) node,
 * and captures:
 *  - dataflow inputs
 *  - dataflow outputs (or a flag indicating the node is an implied output)
 *  - dominator parent
 *  - dominator children
 *  - basic block
 *
 * This class is templated so we can analyze both DFPatterns and Exprs with the same infrastructure.
 *
 * IndexedGraph should be instantiated through the CreateIndexedGraph utilities.
 */
template <typename T>
class IndexedGraph {
 public:
  using TNode = typename T::ContainerType;

  /*! \brief A Node in the dataflow graph. */
  struct Node {
    /*! \brief Node Constructor
     *  \param node_ref The input graph node
     *  \param index The index of the node in toplogical order
     */
    Node(const TNode* node_ref, PostDfsIndex index) : node_ref_(node_ref), index_(index) {}

    /*! \brief The underlying expression or pattern node. */
    const TNode* node_ref_;

    T ref() const {
      ICHECK(node_ref_ != nullptr);
      return GetRef<T>(node_ref_);
    }

    /*!
     * \brief The index of this node in post-dfs order. If left.index_ > right.index_ then
     * left does not flow into right. If left.index_ = right.index_ then left and right are
     * the same node.
     */
    const PostDfsIndex index_;

    /*! \brief If true this node has implicit outputs, for example as the result of a function. */
    bool is_external_ = false;
    /*! \brief Immediate dataflow inputs to this node. */
    std::vector<Node*> inputs_;
    /*! \brief Immediate dataflow outputs of this node -- may be empty if is_external_ is true. */
    std::vector<Node*> outputs_;

    /*!
     * \brief The node representing the 'basic block' containing this node:
     *  - Function bodies start a new basic block for their bodies.
     *  - The true and false branches of an if start their own blocks.
     *  - The arms of a match each have their own blocks.
     */
    Node* basic_block_ = nullptr;

    /*! \brief The depth of this node in the dominator tree */
    size_t depth_ = 0;
    /*!
     * \brief The dominator parent of this node. This is the node N with least index such that
     * all possible dataflows from this node pass through N.
     */
    Node* dominator_parent_ = nullptr;
    /*! \brief The nodes this node dominates. */
    std::vector<Node*> dominator_children_;

    /*!
     * Add to \p nodes all the nodes which are strictly downstream of \p this, ie can be
     * reached by following output paths.
     */
    void AccumulateDownstreamNodes(std::unordered_set<const Node*> nodes) const {
      std::stack<const Node*> stack;
      stack.push(this);
      while (!stack.empty()) {
        const Node* current = stack.top();
        stack.pop();
        for (auto node : current->outputs_) {
          if (nodes.count(node) == 0) {
            stack.push(node);
            nodes.insert(node);
          }
        }
      }
    }

    /*!
     * \brief Returns true if \p this is a dominator of \p other. Ie all dataflow paths from \p
     * other pass through \p this.
     */
    bool Dominates(const Node* other) const {
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
    for (PostDfsIndex i = topological_order_.size(); i != 0; --i) {
      PostDfsIndex index = i - 1;
      auto* current = topological_order_[index].get();
      if (current->is_external_) {
        current->depth_ = 1;
        current->dominator_parent_ = nullptr;
      } else {
        auto parent = LeastCommonAncestor(current->outputs_);
        current->depth_ = parent ? parent->depth_ + 1 : 1;
        current->dominator_parent_ = parent;
        if (parent) {
          parent->dominator_children_.push_back(current);
        }
      }
    }
  }

  PostDfsIndex size() const { return topological_order_.size(); }

  Node* item_to_node(const T& item) { return item_to_node(item.get()); }
  const Node* item_to_node(const T& item) const { return item_to_node(item.get()); }

  Node* item_to_node(const TNode* item) {
    auto itr = node_map_.find(item);
    ICHECK(itr != node_map_.end()) << PrettyPrint(GetRef<T>(item));
    return itr->second;
  }

  const Node* item_to_node(const TNode* item) const {
    auto itr = node_map_.find(item);
    ICHECK(itr != node_map_.end()) << PrettyPrint(GetRef<T>(item));
    return itr->second;
  }

  Node* index_to_node(PostDfsIndex index) {
    ICHECK_LT(index, topological_order_.size()) << index;
    return topological_order_[index].get();
  }

  const Node* index_to_node(PostDfsIndex index) const {
    ICHECK_LT(index, topological_order_.size()) << index;
    return topological_order_[index].get();
  }

 private:
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

  void AddNode(const T& item) {
    PostDfsIndex index = topological_order_.size();
    VLOG(2) << "node index " << index << " is:\n" << PrettyPrint(item);
    auto node = std::make_unique<Node>(item.get(), index);
    node_map_[item.get()] = node.get();
    topological_order_.emplace_back(std::move(node));
  }

  /*! \brief Map from underlying sub-expressions or dataflow sub-pattern graph nodes. */
  std::unordered_map<const TNode*, Node*> node_map_;
  /*! \brief All nodes in increasing post-dfs index order. This vector owns all the nodes. */
  std::vector<std::unique_ptr<Node>> topological_order_;

  friend std::unique_ptr<IndexedGraph<Expr>> CreateIndexedGraph(const Expr& expr);
  friend std::unique_ptr<IndexedGraph<DFPattern>> CreateIndexedGraph(const DFPattern& pattern);
};

/*! \brief Returns an Indexed Graph for \p expr, which much outlive the result. */
std::unique_ptr<IndexedGraph<Expr>> CreateIndexedGraph(const Expr& expr);

/*!
 * \brief Returns an Indexed Graph for \p pattern, which must outlive the result.
 * The dataflow for a pattern mimics the dataflow for the expression which would match
 * that pattern.
 */
std::unique_ptr<IndexedGraph<DFPattern>> CreateIndexedGraph(const DFPattern& pattern);

}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_IR_INDEXED_GRAPH_H_
