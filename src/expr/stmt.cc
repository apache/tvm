/*!
 *  Copyright (c) 2016 by Contributors
 * \file stmt.cc
 */
#include <tvm/expr.h>
#include <tvm/stmt.h>
#include <tvm/stmt_node.h>

namespace tvm {

Stmt Store(Var buffer, Expr offset, Expr src) {
  auto nptr = std::make_shared<StoreNode>();
  nptr->buffer = std::move(buffer);
  nptr->offset = std::move(offset);
  nptr->src = std::move(src);
  nptr->Verify();
  return Stmt(std::move(nptr));
}

Stmt ForRange(Var loop_var, Range range, Stmt body) {
  auto nptr = std::make_shared<ForRangeNode>();
  nptr->loop_var = std::move(loop_var);
  nptr->range = std::move(range);
  nptr->body = std::move(body);
  nptr->Verify();
  return Stmt(std::move(nptr));
}

Stmt IfThenElse(Expr cond, Stmt then_body, Stmt else_body) {
  auto nptr = std::make_shared<IfThenElseNode>();
  nptr->cond = std::move(cond);
  nptr->then_body = std::move(then_body);
  nptr->else_body = std::move(else_body);
  nptr->Verify();
  return Stmt(std::move(nptr));
}

}  // namespace tvm
