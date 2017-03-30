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

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<Reduce>([](const Reduce *op, IRPrinter *p) {
  p->stream << "reduce("
            << op->op
            << ", ";
  p->print(op->source);
  p->stream << ", axis=" << op->axis;
  if (!is_const(op->condition, 1)) {
    p->stream << ", where=" << op->condition;
  }
  p->stream << ")";
});

}  // namespace Internal
}  // namespace Halide

namespace tvm {
namespace ir {

Expr Reduce::make(std::string op, Expr source,
                  Array<IterVar> axis, Expr condition) {
  for (size_t i = 0; i < axis.size(); ++i) {
    CHECK_EQ(axis[i]->iter_type, kCommReduce)
        << "Can only take axis created by reduce_axis";
  }
  if (!condition.defined()) {
    condition = const_true();
  }
  auto n = std::make_shared<Reduce>();
  CHECK(source.defined());
  for (size_t i = 0; i < axis.size(); ++i) {
    CHECK(axis[i].defined());
  }
  n->type = source.type();
  n->source = source;
  n->op = op;
  n->axis = axis;
  n->condition = condition;
  return Expr(n);
}

Expr Reduce::InitValue(const std::string& op, Type type) {
  if (op == "Add") {
    return make_zero(type);
  } else if (op == "Max") {
    return type.min();
  } else if (op == "Min") {
    return type.max();
  } else {
    LOG(FATAL) << "Unsupported reduction " << op;
    return Expr();
  }
}

Expr Reduce::Combine(const std::string& op, Expr a, Expr b) {
  if (op == "Add") {
    return Add::make(a, b);
  } else if (op == "Max") {
    return Max::make(a, b);
  } else if (op == "Min") {
    return Min::make(a, b);
  } else {
    LOG(FATAL) << "Unsupported reduction " << op;
    return Expr();
  }
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
