/*!
 *  Copyright (c) 2016 by Contributors
 * \file ir_visitor.cc
 */
#include <tvm/ir.h>
#include <tvm/ir_visitor.h>
#include <unordered_set>

namespace tvm {
namespace ir {
// visitor to implement apply
class IRApplyVisit : public IRVisitor {
 public:
  explicit IRApplyVisit(std::function<void(const NodeRef&)> f) : f_(f) {}

  void Visit(const NodeRef& node) final {
    if (visited_.count(node.get()) != 0) return;
    visited_.insert(node.get());
    IRVisitor::Visit(node);
    f_(node);
  }

 private:
  std::function<void(const NodeRef&)> f_;
  std::unordered_set<const Node*> visited_;
};


void PostOrderVisit(const NodeRef& node, std::function<void(const NodeRef&)> fvisit) {
  IRApplyVisit(fvisit).Visit(node);
}

IRVisitor::FVisit& IRVisitor::vtable() {  // NOLINT(*)
  static FVisit inst; return inst;
}

void NoOp(const NodeRef& n, IRVisitor* v) {
}

inline void VisitArray(const Array<Expr>& arr, IRVisitor* v) {
  for (size_t i = 0; i < arr.size(); i++) {
    v->Visit(arr[i]);
  }
}

inline void VisitRDom(const Array<IterVar>& rdom, IRVisitor* v) {
  for (size_t i = 0; i < rdom.size(); i++) {
    Range r = rdom[i]->dom;
    v->Visit(r->min);
    v->Visit(r->extent);
  }
}

#define DISPATCH_TO_VISIT(OP)                       \
  set_dispatch<OP>([](const OP* op, IRVisitor* v) { \
      v->Visit_(op);                                \
    })

TVM_STATIC_IR_FUNCTOR(IRVisitor, vtable)
.DISPATCH_TO_VISIT(Variable)
.DISPATCH_TO_VISIT(LetStmt)
.DISPATCH_TO_VISIT(AttrStmt)
.DISPATCH_TO_VISIT(IfThenElse)
.DISPATCH_TO_VISIT(For)
.DISPATCH_TO_VISIT(Allocate)
.DISPATCH_TO_VISIT(Load)
.DISPATCH_TO_VISIT(Store)
.DISPATCH_TO_VISIT(Let)
.DISPATCH_TO_VISIT(Call)
.DISPATCH_TO_VISIT(Free);

void IRVisitor::Visit_(const Variable* op) {}

void IRVisitor::Visit_(const LetStmt *op) {
  this->Visit(op->value);
  this->Visit(op->body);
}

void IRVisitor::Visit_(const AttrStmt* op) {
  this->Visit(op->value);
  this->Visit(op->body);
}

void IRVisitor::Visit_(const For *op) {
  IRVisitor* v = this;
  v->Visit(op->min);
  v->Visit(op->extent);
  v->Visit(op->body);
}

void IRVisitor::Visit_(const Allocate *op) {
  IRVisitor* v = this;
  for (size_t i = 0; i < op->extents.size(); i++) {
    v->Visit(op->extents[i]);
  }
  v->Visit(op->body);
  v->Visit(op->condition);
  if (op->new_expr.defined()) {
    v->Visit(op->new_expr);
  }
}

void IRVisitor::Visit_(const Load *op) {
  this->Visit(op->index);
}

void IRVisitor::Visit_(const Store *op) {
  this->Visit(op->value);
  this->Visit(op->index);
}

void IRVisitor::Visit_(const IfThenElse *op) {
  this->Visit(op->condition);
  this->Visit(op->then_case);
  if (op->else_case.defined()) {
    this->Visit(op->else_case);
  }
}

void IRVisitor::Visit_(const Let *op) {
  this->Visit(op->value);
  this->Visit(op->body);
}

void IRVisitor::Visit_(const Free* op) {}

void IRVisitor::Visit_(const Call *op) {
  VisitArray(op->args, this);
}

TVM_STATIC_IR_FUNCTOR(IRVisitor, vtable)
.set_dispatch<Reduce>([](const Reduce* op, IRVisitor* v) {
    VisitRDom(op->axis, v);
    v->Visit(op->source);
  })
.set_dispatch<IntImm>(NoOp)
.set_dispatch<UIntImm>(NoOp)
.set_dispatch<FloatImm>(NoOp)
.set_dispatch<StringImm>(NoOp);

TVM_STATIC_IR_FUNCTOR(IRVisitor, vtable)
.set_dispatch<Cast>([](const Cast* op, IRVisitor* v) {
    v->Visit(op->value);
  });

// binary operator
template<typename T>
inline void Binary(const T* op, IRVisitor* v) {
  v->Visit(op->a);
  v->Visit(op->b);
}

TVM_STATIC_IR_FUNCTOR(IRVisitor, vtable)
.set_dispatch<Add>(Binary<Add>)
.set_dispatch<Sub>(Binary<Sub>)
.set_dispatch<Mul>(Binary<Mul>)
.set_dispatch<Div>(Binary<Div>)
.set_dispatch<Mod>(Binary<Mod>)
.set_dispatch<Min>(Binary<Min>)
.set_dispatch<Max>(Binary<Max>)
.set_dispatch<EQ>(Binary<EQ>)
.set_dispatch<NE>(Binary<NE>)
.set_dispatch<LT>(Binary<LT>)
.set_dispatch<LE>(Binary<LE>)
.set_dispatch<GT>(Binary<GT>)
.set_dispatch<GE>(Binary<GE>)
.set_dispatch<And>(Binary<And>)
.set_dispatch<Or>(Binary<Or>);

TVM_STATIC_IR_FUNCTOR(IRVisitor, vtable)
.set_dispatch<Not>([](const Not* op, IRVisitor* v) {
    v->Visit(op->a);
  })
.set_dispatch<Select>([](const Select *op, IRVisitor* v) {
    v->Visit(op->condition);
    v->Visit(op->true_value);
    v->Visit(op->false_value);
  })
.set_dispatch<Ramp>([](const Ramp *op, IRVisitor* v) {
    v->Visit(op->base);
    v->Visit(op->stride);
  })
.set_dispatch<Broadcast>([](const Broadcast *op, IRVisitor* v) {
    v->Visit(op->value);
  });

TVM_STATIC_IR_FUNCTOR(IRVisitor, vtable)
.set_dispatch<AssertStmt>([](const AssertStmt *op, IRVisitor* v) {
    v->Visit(op->condition);
    v->Visit(op->message);
  })
.set_dispatch<ProducerConsumer>([](const ProducerConsumer *op, IRVisitor* v) {
    v->Visit(op->body);
  })
.set_dispatch<Provide>([](const Provide *op, IRVisitor* v) {
    VisitArray(op->args, v);
    v->Visit(op->value);
  })
.set_dispatch<Realize>([](const Realize *op, IRVisitor* v) {
    // Mutate the bounds
    for (size_t i = 0; i < op->bounds.size(); i++) {
      v->Visit(op->bounds[i]->min);
      v->Visit(op->bounds[i]->extent);
    }

    v->Visit(op->body);
    v->Visit(op->condition);
  })
.set_dispatch<Block>([](const Block *op, IRVisitor* v) {
    v->Visit(op->first);
    v->Visit(op->rest);
  })
.set_dispatch<Evaluate>([](const Evaluate *op, IRVisitor* v) {
    v->Visit(op->value);
  });

}  // namespace ir
}  // namespace tvm
