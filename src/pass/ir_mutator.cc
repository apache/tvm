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

// namespace to register the functors.
namespace {

using namespace Halide::Internal;

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

inline Array<IterVar> MutateRDom(Array<IterVar> rdom, IRMutator *m) {
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

TVM_STATIC_IR_FUNCTOR(IRMutator, vtable_expr)
.set_dispatch<Reduce>([](const Reduce* op, const Expr& e, IRMutator* m) {
    Array<IterVar> new_rdom  = MutateRDom(op->rdom, m);
    Expr new_source  = m->Mutate(op->source);
    if (op->rdom.same_as(new_rdom) &&
        op->source.same_as(new_source)) {
      return e;
    } else {
      return Reduce::make(op->op, new_source, new_rdom);
    }
  });

TVM_STATIC_IR_FUNCTOR(IRMutator, vtable_stmt)
.set_dispatch<AttrStmt>([](const AttrStmt* op, const Stmt& s, IRMutator* m) {
    Expr value = m->Mutate(op->value);
    Stmt body = m->Mutate(op->body);
    if (value.same_as(op->value) &&
        body.same_as(op->body)) {
      return s;
    } else {
      return AttrStmt::make(op->node, op->type_key, op->value, op->body);
    }
  });

TVM_STATIC_IR_FUNCTOR(IRMutator, vtable_expr)
.set_dispatch<IntImm>(ReturnSelfExpr)
.set_dispatch<UIntImm>(ReturnSelfExpr)
.set_dispatch<FloatImm>(ReturnSelfExpr)
.set_dispatch<StringImm>(ReturnSelfExpr)
.set_dispatch<Variable>(ReturnSelfExpr);

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
.set_dispatch<Load>([](const Load *op, const Expr& e, IRMutator* m) {
    Expr index = m->Mutate(op->index);
    if (index.same_as(op->index)) {
      return e;
    } else {
      return Load::make(op->type, op->buffer_var, index);
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
  })
.set_dispatch<Call>([](const Call *op, const Expr& e, IRMutator* m) {
    auto new_args = MutateArray(op->args, m);
    if (op->args.same_as(new_args)) {
      return e;
    } else {
      return Call::make(op->type, op->name, new_args, op->call_type,
                        op->func, op->value_index);
    }
  })
.set_dispatch<Let>([](const Let *op, const Expr& e, IRMutator* m) {
    Expr value = m->Mutate(op->value);
    Expr body = m->Mutate(op->body);
    if (value.same_as(op->value) &&
        body.same_as(op->body)) {
      return e;
    } else {
      return Let::make(op->var, value, body);
    }
  });

TVM_STATIC_IR_FUNCTOR(IRMutator, vtable_stmt)
.set_dispatch<LetStmt>([](const LetStmt *op, const Stmt& s, IRMutator* m) {
    Expr value = m->Mutate(op->value);
    Stmt body = m->Mutate(op->body);
    if (value.same_as(op->value) &&
        body.same_as(op->body)) {
      return s;
    } else {
      return LetStmt::make(op->var, value, body);
    }
  })
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
.set_dispatch<For>([](const For *op, const Stmt& s, IRMutator* m) {
    Expr min = m->Mutate(op->min);
    Expr extent = m->Mutate(op->extent);
    Stmt body = m->Mutate(op->body);
    if (min.same_as(op->min) &&
        extent.same_as(op->extent) &&
        body.same_as(op->body)) {
      return s;
    } else {
      return For::make(
          op->loop_var, min, extent, op->for_type, op->device_api, body);
    }
  })
.set_dispatch<Store>([](const Store *op, const Stmt& s, IRMutator* m) {
    Expr value = m->Mutate(op->value);
    Expr index = m->Mutate(op->index);
    if (value.same_as(op->value) && index.same_as(op->index)) {
      return s;
    } else {
      return Store::make(op->buffer_var, value, index);
    }
  })
.set_dispatch<Provide>([](const Provide *op, const Stmt& s, IRMutator* m) {
    auto new_args = MutateArray(op->args, m);
    auto new_value = m->Mutate(op->value);
    if (op->args.same_as(new_args) && op->value.same_as(new_value)) {
      return s;
    } else {
      return Provide::make(op->func, op->value_index, new_value, new_args);
    }
  })
.set_dispatch<Allocate>([](const Allocate *op, const Stmt& s, IRMutator* m) {
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
  })
.set_dispatch<Free>([](const Free *op, const Stmt& s, IRMutator* m) {
  return s;
  })
.set_dispatch<Realize>([](const Realize *op, const Stmt& s, IRMutator* m) {
    Region new_bounds;
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

}  // namespace
}  // namespace ir
}  // namespace tvm
