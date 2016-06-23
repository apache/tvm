/*!
 *  Copyright (c) 2016 by Contributors
 * \file base.h
 * \brief Configuation of nngraph as well as basic data structure.
 */
#ifndef NNGRAPH_NODE_H_
#define NNGRAPH_NODE_H_

#include <memory>
#include <unordered_map>
#include "./op_prop.h"

namespace nngraph {
// Forward declare node.
struct Node;

/*! \brief an entry that represents output data from a node */
struct NodeEntry {
  /*! \brief the source node of this data */
  std::shared_ptr<Node> node;
  /*! \brief index of output from the source. */
  uint32_t index;
};

/*!
 * \brief Node represents an operation in a computation graph.
 */
struct Node {
  /*! \brief name of the node */
  std::string name;
  /*! \brief the operator this node is pointing at */
  const OpProperty *op;
  /*! \brief inputs to this node */
  std::vector<NodeEntry> inputs;
  /*!
   * \brief additional attributes about the node,
   *  Use pointer to save space, as attr can be accessed in a slow way,
   *  not every node will have attributes.
   */
  std::unordered_map<std::string, std::string> attr;

  ~Node() {
    if (inputs.size() != 0) {
      // explicit deletion via DFS
      // this is used to avoid stackoverflow caused by chain of deletions
      std::vector<Node*> stack{this};
      std::vector<std::shared_ptr<Node> > to_delete;
      while (!stack.empty()) {
        Node* n = stack.back();
        stack.pop_back();
        for (NodeEntry& e: n->inputs) {
          if (e.node.unique()) {
            stack.push_back(e.node.get());
            to_delete.emplace_back(std::move(e.node));
          } else {
            e.node.reset();
          }
        }
        n->inputs.clear();
      }
    }
  }
};

}  // namespace nngraph

#endif  // NNGRAPH_NODE_H_
