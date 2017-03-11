/*!
 *  Copyright (c) 2017 by Contributors
 * \file channel.cc
 */
#include <tvm/channel.h>

namespace tvm {

Channel ChannelNode::make(Var handle_var, Type dtype) {
  auto n = std::make_shared<ChannelNode>();
  n->handle_var = handle_var;
  n->dtype = dtype;
  return Channel(n);
}

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<ChannelNode>([](const ChannelNode *op, IRPrinter *p) {
    p->stream << "channel(" << op->handle_var << ", " << op->dtype << ")";
});

TVM_REGISTER_NODE_TYPE(ChannelNode);
}  // namespace tvm
