/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 *  Copyright (c) 2016 by Contributors
 * \file ir_mutator.cc
 */
#include <tvm/ir.h>
#include <tvm/ir_mutator.h>
#include <tvm/packed_func_ext.h>
#include "ir_util.h"

namespace tvm {
namespace ir {

class IRTransformer final : public IRMutator {
 public:
  IRTransformer(const runtime::PackedFunc& f_preorder,
                const runtime::PackedFunc& f_postorder,
                const std::unordered_set<uint32_t>& only_enable)
      : f_preorder_(f_preorder),
        f_postorder_(f_postorder),
        only_enable_(only_enable) {
  }
  Stmt Mutate(Stmt stmt) final {
    return MutateInternal<Stmt>(stmt);
  }
  Expr Mutate(Expr expr) final {
    return MutateInternal<Expr>(expr);
  }

 private:
  template<typename T>
  T MutateInternal(T node) {
    if (only_enable_.size() &&
        !only_enable_.count(node->type_index())) {
      return IRMutator::Mutate(node);
    }
    if (f_preorder_ != nullptr) {
      T pre = f_preorder_(node);
      if (pre.defined()) return pre;
    }
    node = IRMutator::Mutate(node);
    if (f_postorder_ != nullptr) {
      T post = f_postorder_(node);
      if (post.defined()) return post;
    }
    return node;
  }
  // The functions
  const runtime::PackedFunc& f_preorder_;
  const runtime::PackedFunc& f_postorder_;
  // type indices enabled.
  const std::unordered_set<uint32_t>& only_enable_;
};

Stmt IRTransform(const Stmt& ir_node,
                 const runtime::PackedFunc& f_preorder,
                 const runtime::PackedFunc& f_postorder,
                 const Array<Expr>& only_enable) {
  std::unordered_set<uint32_t> only_type_index;
  for (Expr s : only_enable) {
    only_type_index.insert(Node::TypeKey2Index(s.as<StringImm>()->value.c_str()));
  }
  return IRTransformer(f_preorder, f_postorder, only_type_index)
      .Mutate(ir_node);
}

IRMutator::FMutateExpr& IRMutator::vtable_expr() {  // NOLINT(*)
  static FMutateExpr inst; return inst;
}

IRMutator::FMutateStmt& IRMutator::vtable_stmt() {  // NOLINT(*)
  static FMutateStmt inst; return inst;
}

inline Array<Expr> MutateArray(Array<Expr> arr, IRMutator *m) {
  return UpdateArray(arr, [&m] (const Expr& e) { return m->Mutate(e); });
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
        Range::make_by_min_extent(new_min, new_extent),
        v->var, v->iter_type, v->thread_tag);
  }
  if (!changed) {
    return rdom;
  } else {
    return Array<IterVar>(new_dom);
  }
}


// Mutate Stmt

#define DISPATCH_TO_MUTATE_STMT(OP)                                 \
  set_dispatch<OP>([](const OP* op, const Stmt& s, IRMutator* m) {  \
      return m->Mutate_(op, s);                                     \
    })

Stmt IRMutator::Mutate_(const AttrStmt* op, const Stmt& s) {
  Expr value = this->Mutate(op->value);
  Stmt body = this->Mutate(op->body);
  if (value.same_as(op->value) &&
      body.same_as(op->body)) {
    return s;
  } else {
    return AttrStmt::make(op->node, op->attr_key, value, body);
  }
}

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

Stmt IRMutator::Mutate_(const IfThenElse *op, const Stmt& s) {
  Expr condition = this->Mutate(op->condition);
  Stmt then_case = this->Mutate(op->then_case);
  Stmt else_case;
  if (op->else_case.defined()) {
    else_case = this->Mutate(op->else_case);
  }
  if (condition.same_as(op->condition) &&
      then_case.same_as(op->then_case) &&
      else_case.same_as(op->else_case)) {
    return s;
  } else {
    return IfThenElse::make(condition, then_case, else_case);
  }
}

Stmt IRMutator::Mutate_(const Store *op, const Stmt& s) {
  Expr value = this->Mutate(op->value);
  Expr index = this->Mutate(op->index);
  Expr pred = this->Mutate(op->predicate);
  if (value.same_as(op->value) && index.same_as(op->index) && pred.same_as(op->predicate)) {
    return s;
  } else {
    return Store::make(op->buffer_var, value, index, pred);
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
}

Stmt IRMutator::Mutate_(const Prefetch* op, const Stmt& s) {
  IRMutator* m = this;
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

  if (!bounds_changed) {
    return s;
  } else {
    return Prefetch::make(op->func, op->value_index,
                          op->type, new_bounds);
  }
}

Stmt IRMutator::Mutate_(const Block* op, const Stmt& s) {
  Stmt first = this->Mutate(op->first);
  Stmt rest = this->Mutate(op->rest);
  if (first.same_as(op->first) &&
      rest.same_as(op->rest)) {
    return s;
  } else {
    return Block::make(first, rest);
  }
}

Stmt IRMutator::Mutate_(const AssertStmt *op, const Stmt& s) {
  Expr condition = this->Mutate(op->condition);
  Expr message = this->Mutate(op->message);
  Stmt body = this->Mutate(op->body);

  if (condition.same_as(op->condition) &&
      message.same_as(op->message) &&
      body.same_as(op->body)) {
    return s;
  } else {
    return AssertStmt::make(condition, message, body);
  }
}

Stmt IRMutator::Mutate_(const ProducerConsumer *op, const Stmt& s) {
  Stmt body = this->Mutate(op->body);
  if (body.same_as(op->body)) {
    return s;
  } else {
    return ProducerConsumer::make(op->func, op->is_producer, body);
  }
}

Stmt IRMutator::Mutate_(const Evaluate *op, const Stmt& s) {
  Expr v = this->Mutate(op->value);
  if (v.same_as(op->value)) {
    return s;
  } else {
    return Evaluate::make(v);
  }
}

Stmt IRMutator::Mutate_(const Free *op, const Stmt& s) {
  return s;
}

TVM_STATIC_IR_FUNCTOR(IRMutator, vtable_stmt)
.DISPATCH_TO_MUTATE_STMT(LetStmt)
.DISPATCH_TO_MUTATE_STMT(AttrStmt)
.DISPATCH_TO_MUTATE_STMT(IfThenElse)
.DISPATCH_TO_MUTATE_STMT(For)
.DISPATCH_TO_MUTATE_STMT(Allocate)
.DISPATCH_TO_MUTATE_STMT(Store)
.DISPATCH_TO_MUTATE_STMT(Free)
.DISPATCH_TO_MUTATE_STMT(AssertStmt)
.DISPATCH_TO_MUTATE_STMT(ProducerConsumer)
.DISPATCH_TO_MUTATE_STMT(Provide)
.DISPATCH_TO_MUTATE_STMT(Realize)
.DISPATCH_TO_MUTATE_STMT(Block)
.DISPATCH_TO_MUTATE_STMT(Evaluate)
.DISPATCH_TO_MUTATE_STMT(Prefetch);


// Mutate Expr

#define DISPATCH_TO_MUTATE_EXPR(OP)                                 \
  set_dispatch<OP>([](const OP* op, const Expr& e, IRMutator* m) {  \
      return m->Mutate_(op, e);                                     \
    })

Expr IRMutator::Mutate_(const Variable *op, const Expr& e) {
  return e;
}

Expr IRMutator::Mutate_(const Load *op, const Expr& e) {
  Expr index = this->Mutate(op->index);
  Expr pred = this->Mutate(op->predicate);
  if (index.same_as(op->index) && pred.same_as(op->predicate)) {
    return e;
  } else {
    return Load::make(op->type, op->buffer_var, index, pred);
  }
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

Expr IRMutator::Mutate_(const Call* op, const Expr& e) {
  auto new_args = MutateArray(op->args, this);
  if (op->args.same_as(new_args)) {
    return e;
  } else {
    return Call::make(op->type, op->name, new_args, op->call_type,
                      op->func, op->value_index);
  }
}

#define DEFINE_BIOP_EXPR_MUTATE_(OP)                        \
  Expr IRMutator::Mutate_(const OP* op, const Expr& e) {    \
    Expr a = this->Mutate(op->a);                           \
    Expr b = this->Mutate(op->b);                           \
    if (a.same_as(op->a) &&                                 \
        b.same_as(op->b)) {                                 \
      return e;                                             \
    } else {                                                \
      return OP::make(a, b);                                 \
    }                                                       \
  }

DEFINE_BIOP_EXPR_MUTATE_(Add)
DEFINE_BIOP_EXPR_MUTATE_(Sub)
DEFINE_BIOP_EXPR_MUTATE_(Mul)
DEFINE_BIOP_EXPR_MUTATE_(Div)
DEFINE_BIOP_EXPR_MUTATE_(Mod)
DEFINE_BIOP_EXPR_MUTATE_(FloorDiv)
DEFINE_BIOP_EXPR_MUTATE_(FloorMod)
DEFINE_BIOP_EXPR_MUTATE_(Min)
DEFINE_BIOP_EXPR_MUTATE_(Max)
DEFINE_BIOP_EXPR_MUTATE_(EQ)
DEFINE_BIOP_EXPR_MUTATE_(NE)
DEFINE_BIOP_EXPR_MUTATE_(LT)
DEFINE_BIOP_EXPR_MUTATE_(LE)
DEFINE_BIOP_EXPR_MUTATE_(GT)
DEFINE_BIOP_EXPR_MUTATE_(GE)
DEFINE_BIOP_EXPR_MUTATE_(And)
DEFINE_BIOP_EXPR_MUTATE_(Or)

Expr IRMutator::Mutate_(const Reduce *op, const Expr& e) {
  Array<IterVar> new_axis  = MutateIterVarArr(op->axis, this);
  Array<Expr> new_source = MutateArray(op->source, this);
  Expr new_cond = this->Mutate(op->condition);
  if (op->axis.same_as(new_axis) &&
      op->source.same_as(new_source) &&
      op->condition.same_as(new_cond)) {
    return e;
  } else {
    return Reduce::make(
      op->combiner, new_source, new_axis, new_cond, op->value_index);
  }
}

Expr IRMutator::Mutate_(const Cast *op, const Expr& e) {
  Expr value = this->Mutate(op->value);
  if (value.same_as(op->value)) {
    return e;
  } else {
    return Cast::make(op->type, value);
  }
}

Expr IRMutator::Mutate_(const Not *op, const Expr& e) {
  Expr a = this->Mutate(op->a);
  if (a.same_as(op->a)) {
    return e;
  } else {
    return Not::make(a);
  }
}

Expr IRMutator::Mutate_(const Select *op, const Expr& e) {
  Expr cond = this->Mutate(op->condition);
  Expr t = this->Mutate(op->true_value);
  Expr f = this->Mutate(op->false_value);
  if (cond.same_as(op->condition) &&
      t.same_as(op->true_value) &&
      f.same_as(op->false_value)) {
    return e;
  } else {
    return Select::make(cond, t, f);
  }
}

Expr IRMutator::Mutate_(const Ramp *op, const Expr& e) {
  Expr base = this->Mutate(op->base);
  Expr stride = this->Mutate(op->stride);
  if (base.same_as(op->base) &&
      stride.same_as(op->stride)) {
    return e;
  } else {
    return Ramp::make(base, stride, op->lanes);
  }
}

Expr IRMutator::Mutate_(const Broadcast *op, const Expr& e) {
  Expr value = this->Mutate(op->value);
  if (value.same_as(op->value)) {
    return e;
  } else {
    return Broadcast::make(value, op->lanes);
  }
}

Expr IRMutator::Mutate_(const Shuffle *op, const Expr& e) {
  auto new_vec = MutateArray(op->vectors, this);
  if (new_vec.same_as(op->vectors)) {
    return e;
  } else {
    return Shuffle::make(new_vec, op->indices);
  }
}

#define DEFINE_OP_RETURN_SELF_EXPR_MUTATE_(OP)              \
  Expr IRMutator::Mutate_(const OP *op, const Expr& e) {    \
    return e;                                               \
  }

DEFINE_OP_RETURN_SELF_EXPR_MUTATE_(IntImm)
DEFINE_OP_RETURN_SELF_EXPR_MUTATE_(UIntImm)
DEFINE_OP_RETURN_SELF_EXPR_MUTATE_(FloatImm)
DEFINE_OP_RETURN_SELF_EXPR_MUTATE_(StringImm)

TVM_STATIC_IR_FUNCTOR(IRMutator, vtable_expr)
.DISPATCH_TO_MUTATE_EXPR(Variable)
.DISPATCH_TO_MUTATE_EXPR(Load)
.DISPATCH_TO_MUTATE_EXPR(Let)
.DISPATCH_TO_MUTATE_EXPR(Call)
.DISPATCH_TO_MUTATE_EXPR(Add)
.DISPATCH_TO_MUTATE_EXPR(Sub)
.DISPATCH_TO_MUTATE_EXPR(Mul)
.DISPATCH_TO_MUTATE_EXPR(Div)
.DISPATCH_TO_MUTATE_EXPR(Mod)
.DISPATCH_TO_MUTATE_EXPR(FloorDiv)
.DISPATCH_TO_MUTATE_EXPR(FloorMod)
.DISPATCH_TO_MUTATE_EXPR(Min)
.DISPATCH_TO_MUTATE_EXPR(Max)
.DISPATCH_TO_MUTATE_EXPR(EQ)
.DISPATCH_TO_MUTATE_EXPR(NE)
.DISPATCH_TO_MUTATE_EXPR(LT)
.DISPATCH_TO_MUTATE_EXPR(LE)
.DISPATCH_TO_MUTATE_EXPR(GT)
.DISPATCH_TO_MUTATE_EXPR(GE)
.DISPATCH_TO_MUTATE_EXPR(And)
.DISPATCH_TO_MUTATE_EXPR(Or)
.DISPATCH_TO_MUTATE_EXPR(Reduce)
.DISPATCH_TO_MUTATE_EXPR(Cast)
.DISPATCH_TO_MUTATE_EXPR(Not)
.DISPATCH_TO_MUTATE_EXPR(Select)
.DISPATCH_TO_MUTATE_EXPR(Ramp)
.DISPATCH_TO_MUTATE_EXPR(Broadcast)
.DISPATCH_TO_MUTATE_EXPR(IntImm)
.DISPATCH_TO_MUTATE_EXPR(UIntImm)
.DISPATCH_TO_MUTATE_EXPR(FloatImm)
.DISPATCH_TO_MUTATE_EXPR(StringImm)
.DISPATCH_TO_MUTATE_EXPR(Shuffle);

}  // namespace ir
}  // namespace tvm
