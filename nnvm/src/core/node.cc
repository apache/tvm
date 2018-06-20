/*!
 *  Copyright (c) 2016 by Contributors
 * \file node.cc
 * \brief Graph node data structure.
 */
#include <nnvm/node.h>

namespace nnvm {

Node::~Node() {
  if (inputs.size() != 0) {
    // explicit deletion via DFS
    // this is used to avoid stackoverflow caused by chain of deletions
    std::vector<Node*> stack{this};
    std::vector<NodePtr> to_delete;
    while (!stack.empty()) {
      Node* n = stack.back();
      stack.pop_back();
      for (NodeEntry& e : n->inputs) {
        if (e.node.unique()) {
          stack.push_back(e.node.get());
          to_delete.emplace_back(std::move(e.node));
        } else {
          e.node.reset();
        }
      }
      for (NodePtr& sp : n->control_deps) {
        if (sp.unique()) {
          stack.push_back(sp.get());
          to_delete.emplace_back(std::move(sp));
        } else {
          sp.reset();
        }
      }
      n->inputs.clear();
    }
  }
}

NodePtr Node::Create() {
  return std::make_shared<Node>();
}

}  // namespace nnvm
