/*!
 *  Copyright (c) 2018 by Contributors
 * \file op.cc
 * \brief Relay's representation of operators.
 */
#include "tvm/relay/op.h"
#include "tvm/ir_functor.h"

namespace tvm {
namespace relay {

using tvm::IRPrinter;
using namespace runtime;

Operator OperatorNode::make(Type op_type) {
  std::shared_ptr<OperatorNode> n = std::make_shared<OperatorNode>();
  n->op_type = std::move(op_type);
  return Operator(n);
}

TVM_REGISTER_API("relay._make.Operator").set_body([](TVMArgs args, TVMRetValue *ret) {
  *ret = OperatorNode::make(args[0]);
});

TVM_STATIC_IR_FUNCTOR_REGISTER(IRPrinter, vtable)
    .set_dispatch<OperatorNode>([](const OperatorNode *node, tvm::IRPrinter *p) {
      p->stream << "OperatorNode(" << node->op_type << ")";
    });

}  // namespace relay
}  // namespace tvm
