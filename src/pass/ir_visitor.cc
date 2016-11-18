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

  void Visit(const IRNodeRef& node) final {
    if (visited_.count(node.get()) != 0) return;
    visited_.insert(node.get());
    IRVisitor::Visit(node);
    f_(node);
  }

 private:
  std::function<void(const IRNodeRef&)> f_;
  std::unordered_set<const Node*> visited_;
};

}  // namespace

void PostOrderVisit(const IRNodeRef& node, std::function<void(const IRNodeRef&)> fvisit) {
  IRApplyVisit(fvisit).Visit(node);
}

IRVisitor::FVisit& IRVisitor::vtable() {  // NOLINT(*)
  static FVisit inst; return inst;
}


// namespace to register the functors.
namespace {

using namespace Halide::Internal;

void NoOp(const IRNodeRef& n, IRVisitor* v) {
}

inline void VisitArray(Array<Expr> arr, IRVisitor* v) {
  for (size_t i = 0; i < arr.size(); i++) {
    v->Visit(arr[i]);
  }
}

inline void VisitRDom(RDomain rdom, IRVisitor* v) {
  for (size_t i = 0; i < rdom->domain.size(); i++) {
    Range r = rdom->domain[i];
    v->Visit(r->min);
    v->Visit(r->extent);
  }
}

TVM_STATIC_IR_FUNCTOR(IRVisitor, vtable)
.set_dispatch<Reduce>([](const Reduce* op, IRVisitor* v) {
    VisitRDom(op->rdom, v);
    v->Visit(op->source);
  });

TVM_STATIC_IR_FUNCTOR(IRVisitor, vtable)
.set_dispatch<IntImm>(NoOp)
.set_dispatch<UIntImm>(NoOp)
.set_dispatch<FloatImm>(NoOp)
.set_dispatch<StringImm>(NoOp)
.set_dispatch<Variable>(NoOp);

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
.set_dispatch<Load>([](const Load *op, IRVisitor* v) {
    v->Visit(op->index);
  })
.set_dispatch<Ramp>([](const Ramp *op, IRVisitor* v) {
    v->Visit(op->base);
    v->Visit(op->stride);
  })
.set_dispatch<Broadcast>([](const Broadcast *op, IRVisitor* v) {
    v->Visit(op->value);
  })
.set_dispatch<Call>([](const Call *op, IRVisitor* v) {
    VisitArray(op->args, v);
  })
.set_dispatch<Let>([](const Let *op, IRVisitor* v) {
    v->Visit(op->value);
    v->Visit(op->body);
  });

TVM_STATIC_IR_FUNCTOR(IRVisitor, vtable)
.set_dispatch<LetStmt>([](const LetStmt *op, IRVisitor* v) {
    v->Visit(op->value);
    v->Visit(op->body);
  })
.set_dispatch<AssertStmt>([](const AssertStmt *op, IRVisitor* v) {
    v->Visit(op->condition);
    v->Visit(op->message);
  })
.set_dispatch<ProducerConsumer>([](const ProducerConsumer *op, IRVisitor* v) {
    v->Visit(op->body);
  })
.set_dispatch<For>([](const For *op, IRVisitor* v) {
    v->Visit(op->min);
    v->Visit(op->extent);
    v->Visit(op->body);
  })
.set_dispatch<Store>([](const Store *op, IRVisitor* v) {
    v->Visit(op->value);
    v->Visit(op->index);
  })
.set_dispatch<Provide>([](const Provide *op, IRVisitor* v) {
    VisitArray(op->args, v);
    VisitArray(op->values, v);
  })
.set_dispatch<Allocate>([](const Allocate *op, IRVisitor* v) {
    for (size_t i = 0; i < op->extents.size(); i++) {
      v->Visit(op->extents[i]);
    }
    v->Visit(op->body);
    v->Visit(op->condition);
    if (op->new_expr.defined()) {
      v->Visit(op->new_expr);
    }
  })
.set_dispatch<Free>(NoOp)
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
.set_dispatch<IfThenElse>([](const IfThenElse *op, IRVisitor* v) {
    v->Visit(op->condition);
    v->Visit(op->then_case);
    v->Visit(op->else_case);
  })
.set_dispatch<Evaluate>([](const Evaluate *op, IRVisitor* v) {
    v->Visit(op->value);
  });

}  // namespace
}  // namespace ir
}  // namespace tvm
