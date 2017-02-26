/*!
 *  Copyright (c) 2017 by Contributors
 * \file lowered_func.cc
 */
#include <tvm/lowered_func.h>

namespace tvm {

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<LoweredFuncNode>([](const LoweredFuncNode *op, IRPrinter *p) {
    p->stream << "LoweredFunc(" << op->name << ", " << op << ")";
});

TVM_REGISTER_NODE_TYPE(LoweredFuncNode);

}  // namespace tvm
