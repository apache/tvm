/*!
 *  Copyright (c) 2017 by Contributors
 * \file graph_handle.cc
 */
#include <tvm/packed_func_ext.h>
#include "./graph_handle.h"

namespace tvm {

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<GraphHandleNode>([](const GraphHandleNode *op, IRPrinter *p) {
    p->stream << "graph-handle("
              << "handle=0x" << std::hex
              << reinterpret_cast<uint64_t>(op->graph_handle) << ")";
});

TVM_REGISTER_NODE_TYPE(GraphHandleNode);

}  // namespace tvm
