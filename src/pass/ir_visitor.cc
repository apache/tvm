/*!
 *  Copyright (c) 2016 by Contributors
 * \file ir_visitor.cc
 */
#include <tvm/ir.h>
#include <tvm/ir_visitor.h>
#include <unordered_set>

namespace tvm {
namespace ir {
namespace {
// visitor to implement apply
class IRApplyVisit : public IRVisitor {
 public:
  explicit IRApplyVisit(std::function<void(const IRNodeRef&)> f) : f_(f) {}

  void visit(const IRNodeRef& node) final {
    if (visited_.count(node.get()) != 0) return;
    visited_.insert(node.get());
    IRVisitor::visit(node);
    f_(node);
  }

 private:
  std::function<void(const IRNodeRef&)> f_;
  std::unordered_set<const Node*> visited_;
};
}  // namespace

IRVisitor::FVisit& IRVisitor::vtable() {  // NOLINT(*)
  static FVisit inst; return inst;
}


void PostOrderVisit(const IRNodeRef& node, std::function<void(const IRNodeRef&)> fvisit) {
  IRApplyVisit v(fvisit);
  v.visit(node);
}

// namespace to register the functors.
namespace {

using namespace Halide::Internal;

void NoOp(const IRNodeRef& n, IRVisitor* v) {
}

inline void VisitArray(Array<Expr> arr, IRVisitor* v) {
  for (size_t i = 0; i < arr.size(); i++) {
    v->visit(arr[i]);
  }
}

inline void VisitRDom(RDomain rdom, IRVisitor* v) {
  for (size_t i = 0; i < rdom->domain.size(); i++) {
    Range r = rdom->domain[i];
    v->visit(r->min);
    v->visit(r->extent);
  }
}

TVM_STATIC_IR_FUNCTOR(IRVisitor, vtable)
.set_dispatch<Reduce>([](const Reduce* op, IRVisitor* v) {
    VisitRDom(op->rdom, v);
    v->visit(op->source);
  });

TVM_STATIC_IR_FUNCTOR(IRVisitor, vtable)
.set_dispatch<IntImm>(NoOp)
.set_dispatch<UIntImm>(NoOp)
.set_dispatch<FloatImm>(NoOp)
.set_dispatch<StringImm>(NoOp)
.set_dispatch<Variable>(NoOp);

TVM_STATIC_IR_FUNCTOR(IRVisitor, vtable)
.set_dispatch<Cast>([](const Cast* op, IRVisitor* v) {
    v->visit(op->value);
  });

// binary operator
template<typename T>
inline void Binary(const T* op, IRVisitor* v) {
  v->visit(op->a);
  v->visit(op->b);
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
    v->visit(op->a);
  })
.set_dispatch<Select>([](const Select *op, IRVisitor* v) {
    v->visit(op->condition);
    v->visit(op->true_value);
    v->visit(op->false_value);
  })
.set_dispatch<Load>([](const Load *op, IRVisitor* v) {
    v->visit(op->index);
  })
.set_dispatch<Ramp>([](const Ramp *op, IRVisitor* v) {
    v->visit(op->base);
    v->visit(op->stride);
  })
.set_dispatch<Broadcast>([](const Broadcast *op, IRVisitor* v) {
    v->visit(op->value);
  })
.set_dispatch<Call>([](const Call *op, IRVisitor* v) {
    VisitArray(op->args, v);
  })
.set_dispatch<Let>([](const Let *op, IRVisitor* v) {
    v->visit(op->value);
    v->visit(op->body);
  });

TVM_STATIC_IR_FUNCTOR(IRVisitor, vtable)
.set_dispatch<LetStmt>([](const LetStmt *op, IRVisitor* v) {
    v->visit(op->value);
    v->visit(op->body);
  })
.set_dispatch<AssertStmt>([](const AssertStmt *op, IRVisitor* v) {
    v->visit(op->condition);
    v->visit(op->message);
  })
.set_dispatch<ProducerConsumer>([](const ProducerConsumer *op, IRVisitor* v) {
    v->visit(op->body);
  })
.set_dispatch<For>([](const For *op, IRVisitor* v) {
    v->visit(op->min);
    v->visit(op->extent);
    v->visit(op->body);
  })
.set_dispatch<Store>([](const Store *op, IRVisitor* v) {
    v->visit(op->value);
    v->visit(op->index);
  })
.set_dispatch<Provide>([](const Provide *op, IRVisitor* v) {
    VisitArray(op->args, v);
    VisitArray(op->values, v);
  })
.set_dispatch<Allocate>([](const Allocate *op, IRVisitor* v) {
    for (size_t i = 0; i < op->extents.size(); i++) {
      v->visit(op->extents[i]);
    }
    v->visit(op->body);
    v->visit(op->condition);
    if (op->new_expr.defined()) {
      v->visit(op->new_expr);
    }
  })
.set_dispatch<Free>(NoOp)
.set_dispatch<Realize>([](const Realize *op, IRVisitor* v) {
    // Mutate the bounds
    for (size_t i = 0; i < op->bounds.size(); i++) {
      v->visit(op->bounds[i]->min);
      v->visit(op->bounds[i]->extent);
    }

    v->visit(op->body);
    v->visit(op->condition);
  })
.set_dispatch<Block>([](const Block *op, IRVisitor* v) {
    v->visit(op->first);
    v->visit(op->rest);
  })
.set_dispatch<IfThenElse>([](const IfThenElse *op, IRVisitor* v) {
    v->visit(op->condition);
    v->visit(op->then_case);
    v->visit(op->else_case);
  })
.set_dispatch<Evaluate>([](const Evaluate *op, IRVisitor* v) {
    v->visit(op->value);
  });

}  // namespace
}  // namespace ir
}  // namespace tvm
