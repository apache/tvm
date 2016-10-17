/*!
 *  Copyright (c) 2016 by Contributors
 * \file op.cc
 */
#include <tvm/op.h>
#include <tvm/expr_node.h>

namespace tvm {

Expr BinaryOp::operator()(Expr lhs, Expr rhs) const {
  auto nptr = std::make_shared<BinaryOpNode>(
      this, std::move(lhs), std::move(rhs));
  nptr->Verify();
  return Expr(std::move(nptr));
}

#define DEFINE_SINGLETON_GET(TypeName)           \
  TypeName* TypeName::Get() {                    \
    static TypeName inst;                        \
    return &inst;                                \
  }

DEFINE_SINGLETON_GET(AddOp);
DEFINE_SINGLETON_GET(SubOp);
DEFINE_SINGLETON_GET(MulOp);
DEFINE_SINGLETON_GET(DivOp);
DEFINE_SINGLETON_GET(MaxOp);
DEFINE_SINGLETON_GET(MinOp);

}  // namespace tvm
