/*!
 *  Copyright (c) 2016 by Contributors
 * \file ir_mutator.cc
 */
#include <tvm/ir.h>
#include <tvm/ir_mutator.h>

namespace tvm {
namespace ir {

IRMutator::FMutateExpr& IRMutator::vtable_expr() {  // NOLINT(*)
  static FMutateExpr inst; return inst;
}

IRMutator::FMutateStmt& IRMutator::vtable_stmt() {  // NOLINT(*)
  static FMutateStmt inst; return inst;
}

// const expr
inline Expr ReturnSelfExpr(const NodeRef&, const Expr& e, IRMutator*) {
  return e;
}

inline Array<Expr> MutateArray(Array<Expr> arr, IRMutator *m) {
  std::vector<Expr> new_arr(arr.size());
  bool changed = false;
  for (size_t i = 0; i < arr.size(); i++) {
    Expr old_elem = arr[i];
    Expr new_elem = m->Mutate(old_elem);
    if (!new_elem.same_as(old_elem)) changed = true;
    new_arr[i] = new_elem;
  }
  if (!changed) {
    return arr;
  } else {
    return Array<Expr>(new_arr);
  }
}

inline Array<IterVar> MutateIterVarArr(Array<IterVar> rdom, IRMutator *m) {
  std::vector<IterVar> new_dom(rdom.size());
  bool changed = false;
  for (size_t i = 0; i < rdom.size(); i++) {
    IterVar v = rdom[i];
    Range r = v->dom;
    Expr new_min = m->Mutate(r->min);
    Expr new_extent = m->Mutate(r->extent);
    if (!r->min.same_as(new_min)) changed = true;
    if (!r->extent.same_as(new_extent)) changed = true;
    new_dom[i] = IterVarNode::make(
        Range::make_with_min_extent(new_min, new_extent),
        v->var, v->thread_tag);
  }
  if (!changed) {
    return rdom;
  } else {
    return Array<IterVar>(new_dom);
  }
}

#define DISPATCH_TO_MUTATE_STMT(OP)                                 \
  set_dispatch<OP>([](const OP* op, const Stmt& s, IRMutator* m) {  \
      return m->Mutate_(op, s);                                     \
    })

#define DISPATCH_TO_MUTATE_EXPR(OP)                                 \
  set_dispatch<OP>([](const OP* op, const Expr& e, IRMutator* m) {  \
      return m->Mutate_(op, e);                                     \
    })

TVM_STATIC_IR_FUNCTOR(IRMutator, vtable_stmt)
.DISPATCH_TO_MUTATE_STMT(LetStmt)
.DISPATCH_TO_MUTATE_STMT(AttrStmt)
.DISPATCH_TO_MUTATE_STMT(Provide)
.DISPATCH_TO_MUTATE_STMT(Realize)
.DISPATCH_TO_MUTATE_STMT(Store)
.DISPATCH_TO_MUTATE_STMT(For)
.DISPATCH_TO_MUTATE_STMT(Allocate)
.DISPATCH_TO_MUTATE_STMT(Free);

Stmt IRMutator::Mutate_(const LetStmt *op, const Stmt& s) {
  Expr value = this->Mutate(op->value);
  Stmt body = this->Mutate(op->body);
  if (value.same_as(op->value) &&
      body.same_as(op->body)) {
    return s;
  } else {
    return LetStmt::make(op->var, value, body);
  }
}

Stmt IRMutator::Mutate_(const AttrStmt* op, const Stmt& s) {
  Expr value = this->Mutate(op->value);
  Stmt body = this->Mutate(op->body);
  if (value.same_as(op->value) &&
      body.same_as(op->body)) {
    return s;
  } else {
    return AttrStmt::make(op->node, op->type_key, value, body);
  }
}

Stmt IRMutator::Mutate_(const For *op, const Stmt& s) {
  Expr min = this->Mutate(op->min);
  Expr extent = this->Mutate(op->extent);
  Stmt body = this->Mutate(op->body);
  if (min.same_as(op->min) &&
      extent.same_as(op->extent) &&
      body.same_as(op->body)) {
    return s;
  } else {
    return For::make(
        op->loop_var, min, extent, op->for_type, op->device_api, body);
  }
}

Stmt IRMutator::Mutate_(const Allocate* op, const Stmt& s) {
  IRMutator* m = this;
  std::vector<Expr> new_extents;
  bool all_extents_unmodified = true;
  for (size_t i = 0; i < op->extents.size(); i++) {
    new_extents.push_back(m->Mutate(op->extents[i]));
    all_extents_unmodified &= new_extents[i].same_as(op->extents[i]);
  }
  Stmt body = m->Mutate(op->body);
  Expr condition = m->Mutate(op->condition);
  Expr new_expr;
  if (op->new_expr.defined()) {
    new_expr = m->Mutate(op->new_expr);
  }
  if (all_extents_unmodified &&
      body.same_as(op->body) &&
      condition.same_as(op->condition) &&
      new_expr.same_as(op->new_expr)) {
    return s;
  } else {
    return Allocate::make(
        op->buffer_var, op->type,
        new_extents, condition, body,
        new_expr, op->free_function);
  }
}

Stmt IRMutator::Mutate_(const Provide* op, const Stmt& s) {
  auto new_args = MutateArray(op->args, this);
  auto new_value = this->Mutate(op->value);
  if (op->args.same_as(new_args) && op->value.same_as(new_value)) {
    return s;
  } else {
    return Provide::make(op->func, op->value_index, new_value, new_args);
  }
}

Stmt IRMutator::Mutate_(const Realize* op, const Stmt& s) {
  IRMutator* m = this;
  Halide::Internal::Region new_bounds;
  bool bounds_changed = false;

  // Mutate the bounds
  for (size_t i = 0; i < op->bounds.size(); i++) {
    Expr old_min = op->bounds[i]->min;
    Expr old_extent = op->bounds[i]->extent;
    Expr new_min = m->Mutate(old_min);
    Expr new_extent = m->Mutate(old_extent);
    if (!new_min.same_as(old_min))  bounds_changed = true;
    if (!new_extent.same_as(old_extent)) bounds_changed = true;
    new_bounds.push_back(
        Range::make_by_min_extent(new_min, new_extent));
  }

  Stmt body = m->Mutate(op->body);
  Expr condition = m->Mutate(op->condition);
  if (!bounds_changed &&
      body.same_as(op->body) &&
      condition.same_as(op->condition)) {
    return s;
  } else {
    return Realize::make(op->func, op->value_index,
                         op->type, new_bounds,
                         condition, body);
  }
}

Stmt IRMutator::Mutate_(const Store *op, const Stmt& s) {
  Expr value = this->Mutate(op->value);
  Expr index = this->Mutate(op->index);
  if (value.same_as(op->value) && index.same_as(op->index)) {
    return s;
  } else {
    return Store::make(op->buffer_var, value, index);
  }
}

Stmt IRMutator::Mutate_(const Free *op, const Stmt& s) {
  return s;
}

TVM_STATIC_IR_FUNCTOR(IRMutator, vtable_expr)
.DISPATCH_TO_MUTATE_EXPR(Call)
.DISPATCH_TO_MUTATE_EXPR(Let)
.DISPATCH_TO_MUTATE_EXPR(Load)
.DISPATCH_TO_MUTATE_EXPR(Variable);

Expr IRMutator::Mutate_(const Call* op, const Expr& e) {
  auto new_args = MutateArray(op->args, this);
  if (op->args.same_as(new_args)) {
    return e;
  } else {
    return Call::make(op->type, op->name, new_args, op->call_type,
                      op->func, op->value_index);
  }
}

Expr IRMutator::Mutate_(const Load *op, const Expr& e) {
  Expr index = this->Mutate(op->index);
  if (index.same_as(op->index)) {
    return e;
  } else {
    return Load::make(op->type, op->buffer_var, index);
  }
}


Expr IRMutator::Mutate_(const Variable *op, const Expr& e) {
  return e;
}

Expr IRMutator::Mutate_(const Let *op, const Expr& e) {
  Expr value = this->Mutate(op->value);
  Expr body = this->Mutate(op->body);
  if (value.same_as(op->value) &&
      body.same_as(op->body)) {
    return e;
  } else {
    return Let::make(op->var, value, body);
  }
}

TVM_STATIC_IR_FUNCTOR(IRMutator, vtable_expr)
.set_dispatch<Reduce>([](const Reduce* op, const Expr& e, IRMutator* m) {
    Array<IterVar> new_axis  = MutateIterVarArr(op->axis, m);
    Expr new_source  = m->Mutate(op->source);
    if (op->axis.same_as(new_axis) &&
        op->source.same_as(new_source)) {
      return e;
    } else {
      return Reduce::make(op->op, new_source, new_axis);
    }
  });

TVM_STATIC_IR_FUNCTOR(IRMutator, vtable_expr)
.set_dispatch<IntImm>(ReturnSelfExpr)
.set_dispatch<UIntImm>(ReturnSelfExpr)
.set_dispatch<FloatImm>(ReturnSelfExpr)
.set_dispatch<StringImm>(ReturnSelfExpr);

TVM_STATIC_IR_FUNCTOR(IRMutator, vtable_expr)
.set_dispatch<Cast>([](const Cast* op, const Expr& e, IRMutator* m) {
    Expr value = m->Mutate(op->value);
    if (value.same_as(op->value)) {
      return e;
    } else {
      return Cast::make(op->type, value);
    }
  });

// binary operator
template<typename T>
inline Expr Binary(const T* op, const Expr& e, IRMutator* m) {
  Expr a = m->Mutate(op->a);
  Expr b = m->Mutate(op->b);
  if (a.same_as(op->a) &&
      b.same_as(op->b)) {
    return e;
  } else {
    return T::make(a, b);
  }
}

TVM_STATIC_IR_FUNCTOR(IRMutator, vtable_expr)
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

TVM_STATIC_IR_FUNCTOR(IRMutator, vtable_expr)
.set_dispatch<Not>([](const Not* op, const Expr& e, IRMutator* m) {
    Expr a = m->Mutate(op->a);
    if (a.same_as(op->a)) {
      return e;
    } else {
      return Not::make(a);
    }
  })
.set_dispatch<Select>([](const Select *op, const Expr& e, IRMutator* m) {
    Expr cond = m->Mutate(op->condition);
    Expr t = m->Mutate(op->true_value);
    Expr f = m->Mutate(op->false_value);
    if (cond.same_as(op->condition) &&
        t.same_as(op->true_value) &&
        f.same_as(op->false_value)) {
      return e;
    } else {
      return Select::make(cond, t, f);
    }
  })
.set_dispatch<Ramp>([](const Ramp *op, const Expr& e, IRMutator* m) {
    Expr base = m->Mutate(op->base);
    Expr stride = m->Mutate(op->stride);
    if (base.same_as(op->base) &&
        stride.same_as(op->stride)) {
      return e;
    } else {
      return Ramp::make(base, stride, op->lanes);
    }
  })
.set_dispatch<Broadcast>([](const Broadcast *op, const Expr& e, IRMutator* m) {
    Expr value = m->Mutate(op->value);
    if (value.same_as(op->value)) {
      return e;
    } else {
      return Broadcast::make(value, op->lanes);
    }
  });

TVM_STATIC_IR_FUNCTOR(IRMutator, vtable_stmt)
.set_dispatch<AssertStmt>([](const AssertStmt *op, const Stmt& s, IRMutator* m) {
    Expr condition = m->Mutate(op->condition);
    Expr message = m->Mutate(op->message);

    if (condition.same_as(op->condition) && message.same_as(op->message)) {
      return s;
    } else {
      return AssertStmt::make(condition, message);
    }
  })
.set_dispatch<ProducerConsumer>([](const ProducerConsumer *op, const Stmt& s, IRMutator* m) {
    Stmt body = m->Mutate(op->body);
    if (body.same_as(op->body)) {
      return s;
    } else {
      return ProducerConsumer::make(op->func, op->is_producer, body);
    }
  })
.set_dispatch<Block>([](const Block *op, const Stmt& s, IRMutator* m) {
    Stmt first = m->Mutate(op->first);
    Stmt rest = m->Mutate(op->rest);
    if (first.same_as(op->first) &&
        rest.same_as(op->rest)) {
      return s;
    } else {
      return Block::make(first, rest);
    }
  })
.set_dispatch<IfThenElse>([](const IfThenElse *op, const Stmt& s, IRMutator* m) {
    Expr condition = m->Mutate(op->condition);
    Stmt then_case = m->Mutate(op->then_case);
    Stmt else_case;
    if (else_case.defined()) {
      else_case = m->Mutate(op->else_case);
    }
    if (condition.same_as(op->condition) &&
        then_case.same_as(op->then_case) &&
        else_case.same_as(op->else_case)) {
      return s;
    } else {
      return IfThenElse::make(condition, then_case, else_case);
    }
  })
.set_dispatch<Evaluate>([](const Evaluate *op, const Stmt& s, IRMutator* m) {
    Expr v = m->Mutate(op->value);
    if (v.same_as(op->value)) {
      return s;
    } else {
      return Evaluate::make(v);
    }
  });
}  // namespace ir
}  // namespace tvm
