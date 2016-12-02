/*!
 *  Copyright (c) 2016 by Contributors
 * \file ir.cc
 */
#include <tvm/base.h>
#include <tvm/expr.h>
#include <tvm/ir.h>
#include <ir/IR.h>
#include <ir/IRPrinter.h>
#include <memory>

namespace Halide {
namespace Internal {

using tvm::ir::Reduce;
using tvm::ir::AttrStmt;

template<>
void ExprNode<Reduce>::accept(IRVisitor *v, const Expr&) const {
  LOG(FATAL) << "Reduce do not work with old Visitor, use IRFunctor style visitor";
}

template<>
void StmtNode<AttrStmt>::accept(IRVisitor *v, const Stmt&) const {
  LOG(FATAL) << "AttrStmt do not work with old Visitor, use IRFunctor style visitor";
}

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<Reduce>([](const Reduce *op, IRPrinter *p) {
  p->stream << "reduce("
            << op->op
            << ", ";
  p->print(op->source);
  p->stream << ", rdom=" << op->rdom << ")";
});

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<AttrStmt>([](const AttrStmt *op, IRPrinter *p) {
    p->stream << "attr " << op->type_key << " = ";
    p->print(op->value);
    p->stream << '\n';
    p->print(op->body);
});

}  // namespace Internal
}  // namespace Halide

namespace tvm {
namespace ir {

Expr Reduce::make(std::string op, Expr source, Array<IterVar> rdom) {
  auto n = std::make_shared<Reduce>();
  CHECK(source.defined());
  for (size_t i = 0; i < rdom.size(); ++i) {
    CHECK(rdom[i].defined());
  }
  n->type = source.type();
  n->source = source;
  n->op = op;
  n->rdom = rdom;
  return Expr(n);
}

Stmt AttrStmt::make(NodeRef node, std::string type_key, Expr value, Stmt body) {
  auto n = std::make_shared<AttrStmt>();
  n->node = node;
  n->type_key = type_key;
  n->value = value;
  n->body = body;
  return Stmt(n);
}

TVM_REGISTER_NODE_TYPE(Reduce);
TVM_REGISTER_NODE_TYPE(AttrStmt);

TVM_REGISTER_NODE_TYPE(FloatImm);
TVM_REGISTER_NODE_TYPE(IntImm);
TVM_REGISTER_NODE_TYPE(UIntImm);
TVM_REGISTER_NODE_TYPE(StringImm);
TVM_REGISTER_NODE_TYPE(Cast);
TVM_REGISTER_NODE_TYPE(Variable);
TVM_REGISTER_NODE_TYPE(Add);
TVM_REGISTER_NODE_TYPE(Sub);
TVM_REGISTER_NODE_TYPE(Mul);
TVM_REGISTER_NODE_TYPE(Div);
TVM_REGISTER_NODE_TYPE(Mod);
TVM_REGISTER_NODE_TYPE(Min);
TVM_REGISTER_NODE_TYPE(Max);
TVM_REGISTER_NODE_TYPE(EQ);
TVM_REGISTER_NODE_TYPE(NE);
TVM_REGISTER_NODE_TYPE(LT);
TVM_REGISTER_NODE_TYPE(LE);
TVM_REGISTER_NODE_TYPE(GT);
TVM_REGISTER_NODE_TYPE(GE);
TVM_REGISTER_NODE_TYPE(And);
TVM_REGISTER_NODE_TYPE(Or);
TVM_REGISTER_NODE_TYPE(Not);
TVM_REGISTER_NODE_TYPE(Select);
TVM_REGISTER_NODE_TYPE(Load);
TVM_REGISTER_NODE_TYPE(Ramp);
TVM_REGISTER_NODE_TYPE(Broadcast);
TVM_REGISTER_NODE_TYPE(Call);
TVM_REGISTER_NODE_TYPE(Let);
TVM_REGISTER_NODE_TYPE(LetStmt);
TVM_REGISTER_NODE_TYPE(AssertStmt);
TVM_REGISTER_NODE_TYPE(ProducerConsumer);
TVM_REGISTER_NODE_TYPE(For);
TVM_REGISTER_NODE_TYPE(Store);
TVM_REGISTER_NODE_TYPE(Provide);
TVM_REGISTER_NODE_TYPE(Allocate);
TVM_REGISTER_NODE_TYPE(Free);
TVM_REGISTER_NODE_TYPE(Realize);
TVM_REGISTER_NODE_TYPE(Block);
TVM_REGISTER_NODE_TYPE(IfThenElse);
TVM_REGISTER_NODE_TYPE(Evaluate);
}  // namespace ir
}  // namespace tvm
