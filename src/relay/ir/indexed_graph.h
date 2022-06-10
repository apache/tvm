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
 * \brief A graph representation of the dataflow in a Relay expression or Relay (dataflow)
 * pattern. Each 'indexed graph' node is 1:1 with an expression/pattern 'node', hence the
 * term 'IndexedGraph'. Dataflow is captured in a generic representation which is convenient
 * for analysis, particularly pattern matching and partitioning.
 *
 * TODO(mbs): Copied from fuse_ops.cc, consider refactoring to share implementation.
 */
#ifndef TVM_RELAY_IR_INDEXED_GRAPH_H_
#define TVM_RELAY_IR_INDEXED_GRAPH_H_

#include <tvm/relay/dataflow_pattern.h>

#include <memory>
#include <stack>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace tvm {
namespace relay {

/*! \brief The index of a node in the post-dfs traversal of overall expression. */
using PostDfsIndex = size_t;

/*!
 * \brief Returns a brief summary of the 'reference' expression or pattern. Only used by
 * IndexedGraph::ToString() for debugging.
 */
std::string RefToSummary(const Expr& expr);
std::string RefToSummary(const DFPattern& pattern);

/*!
 * \brief Represents the implied dataflow of an expression or (dataflow) pattern as a DAG who's
 * nodes are 1:1 with those in the underlying expression/pattern.
 *
 * Each indexed graph node captures:
 *  - Dataflow inputs.
 *  - Dataflow outputs (or a flag indicating the node is an implied output).
 *  - Dominator parent (ie closest node at which all outputs of the current node re-combine).
 *  - Dominator children (inverse of above).
 *  - Basic block (ie node representing the body of a function, arm of an if, etc).
 *
 * This class is templated so we can analyze both DFPatterns and Exprs with the same infrastructure.
 *
 * IndexedGraph should be instantiated through the CreateIndexedGraph utilities below.
 */
template <typename T>
class IndexedGraph {
 public:
  using TNode = typename T::ContainerType;

  /*! \brief A Node in the graph. */
  struct Node {
    /*! \brief Node Constructor
     *  \param ref The expression or dataflow pattern node this indexed graph node is augmenting.
     *  \param index The index of this node in the topological order
     */
    Node(const TNode* ref, PostDfsIndex index) : node_ref_(ref), index_(index) {}

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
    void AccumulateDownstreamNodes(std::unordered_set<const Node*>* nodes) const {
      std::stack<const Node*> stack;
      stack.push(this);
      while (!stack.empty()) {
        const Node* current = stack.top();
        stack.pop();
        for (auto node : current->outputs_) {
          if (nodes->count(node) == 0) {
            stack.push(node);
            nodes->insert(node);
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

  /*!
   * \brief (For debugging only) Returns description of indexed graph with hints as to the
   * sub-expressions or sub-patterns corresponding to each indexed graph node.
   */
  std::string ToString() const {
    std::ostringstream os;
    os << "IndexedGraph(size = " << topological_order_.size() << ") {" << std::endl;
    for (PostDfsIndex index = 0; index < topological_order_.size(); ++index) {
      const Node* node = topological_order_[index].get();
      ICHECK_EQ(index, node->index_);
      os << "  " << index << " (" << RefToSummary(node->ref()) << "): inputs=[";
      for (const auto* sub_node : node->inputs_) {
        os << sub_node->index_ << ",";
      }
      os << "], outputs=[";
      for (const auto* sub_node : node->outputs_) {
        os << sub_node->index_ << ",";
      }
      os << "]";
      if (node->is_external_) {
        os << ", external";
      }
      if (node->basic_block_) {
        os << ", basic_block=" << node->basic_block_->index_;
      }
      if (node->depth_ > 0) {
        os << ", depth=" << node->depth_;
      }
      if (node->dominator_parent_) {
        os << ", dom_parent=" << node->dominator_parent_->index_;
      }
      os << ", dom_children=[";
      for (const auto* sub_node : node->dominator_children_) {
        os << sub_node->index_ << ",";
      }
      os << "]" << std::endl;
    }
    os << "}";
    return os.str();
  }

  /*!
   * Check-fails if the graph is ill-formed. For debugging only.
   */
  void CheckValid() const {
    ICHECK_GT(topological_order_.size(), 0);
    for (PostDfsIndex index = 0; index < topological_order_.size(); ++index) {
      const Node* node = topological_order_[index].get();
      // We have a node.
      ICHECK(node);
      // Bijections with post-dfs indexes and expressions/patterns are correct.
      ICHECK_EQ(node->index_, index);
      ICHECK(node->node_ref_);
      auto itr = node_map_.find(node->node_ref_);
      ICHECK(itr != node_map_.end());
      ICHECK_EQ(itr->second, node) << "at index " << index << " in:" << std::endl << ToString();
      // Inputs come before.
      for (size_t i = 0; i < node->inputs_.size(); ++i) {
        const Node* input = node->inputs_[i];
        ICHECK(input);
        ICHECK_LT(input->index_, index);
        ICHECK(std::find(input->outputs_.begin(), input->outputs_.end(), node) !=
               input->outputs_.end());
      }
      // Outputs come after.
      for (size_t i = 0; i < node->outputs_.size(); ++i) {
        const Node* output = node->outputs_[i];
        ICHECK(output);
        ICHECK_GT(output->index_, index);
        ICHECK(std::find(output->inputs_.begin(), output->inputs_.end(), node) !=
               output->inputs_.end());
      }
      ICHECK_GT(node->depth_, 0);
      // Dominator children come before.
      for (size_t i = 0; i < node->dominator_children_.size(); ++i) {
        const Node* child = node->dominator_children_[i];
        ICHECK(child);
        ICHECK_LT(child->index_, index);
      }
      if (node->dominator_parent_) {
        // Dominator comes after.
        ICHECK_GT(node->dominator_parent_->index_, index);
      }
    }
  }

 private:
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
    PostDfsIndex lhs_index = lhs->index_;
    PostDfsIndex rhs_index = rhs->index_;
    while (lhs != rhs) {
      ICHECK(lhs && rhs) << "LCA(" << lhs_index << ", " << rhs_index << ") on graph:" << std::endl
                         << ToString();
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

  /*!
   * \brief Appends a node corresponding to \p ref, and maintains the sub-expression/sub-pattern to
   * node bijection. The insertion index will be the node's PostDfsIndex. All other node properties
   * are accumulated in-place.
   */
  void AddNode(const T& ref) {
    PostDfsIndex index = topological_order_.size();
    auto node = std::make_unique<Node>(ref.get(), index);
    node_map_[ref.get()] = node.get();
    topological_order_.emplace_back(std::move(node));
  }

  /*!
   * \brief Map from underlying sub-expression or sub-pattern nodes to their indexed graph nodes.
   */
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
