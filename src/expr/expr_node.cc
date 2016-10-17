/*!
 *  Copyright (c) 2016 by Contributors
 * \file expr_node.cc
 */
#include <tvm/expr_node.h>
#include <memory>

namespace dmlc {
DMLC_REGISTRY_ENABLE(::tvm::NodeFactoryReg);
}  // namespace dmlc

namespace tvm {

void Node::Destroy() {
  bool safe = true;
  this->VisitNodeRefFields([&safe](const char* k, NodeRef* r) {
      if (r->node_.get() != nullptr) safe = false;
    });

  if (!safe) {
    // explicit deletion via DFS
    // this is used to avoid stackoverflow caused by chain of deletions
    std::vector<Node*> stack{this};
    std::vector<std::shared_ptr<Node> > to_delete;
    while (!stack.empty()) {
      Node* n = stack.back();
      stack.pop_back();
      n->VisitNodeRefFields([&safe, &stack, &to_delete](const char* k, NodeRef* r) {
          if (r->node_.unique()) {
            stack.push_back(r->node_.get());
            to_delete.emplace_back(std::move(r->node_));
          } else {
            r->node_.reset();
          }
        });
    }
  }
}

TVM_REGISTER_NODE_TYPE(VarNode);
TVM_REGISTER_NODE_TYPE(IntNode);
TVM_REGISTER_NODE_TYPE(FloatNode);
TVM_REGISTER_NODE_TYPE(UnaryOpNode);
TVM_REGISTER_NODE_TYPE(BinaryOpNode);

}  // namespace tvm
