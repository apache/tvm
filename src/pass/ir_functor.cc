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
 * \file ir_functor.cc
 */
#include <tvm/ir_functor_ext.h>
#include <tvm/packed_func_ext.h>

namespace tvm {
namespace ir {

// visitor to implement apply
class IRApplyVisit :
      public StmtExprVisitor {
 public:
  explicit IRApplyVisit(std::function<void(const ObjectRef&)> f) : f_(f) {}

  void VisitExpr(const Expr& node) final {
    if (visited_.count(node.get()) != 0) return;
    visited_.insert(node.get());
    ExprVisitor::VisitExpr(node);
    f_(node);
  }

  void VisitStmt(const Stmt& node) final {
    if (visited_.count(node.get()) != 0) return;
    visited_.insert(node.get());
    StmtVisitor::VisitStmt(node);
    f_(node);
  }

 private:
  std::function<void(const ObjectRef&)> f_;
  std::unordered_set<const Object*> visited_;
};

void PostOrderVisit(const ObjectRef& node,
                    std::function<void(const ObjectRef&)> fvisit) {
  if (node.as<StmtNode>()) {
    IRApplyVisit visitor(fvisit);
    visitor(Downcast<Stmt>(node));
  } else {
    IRApplyVisit visitor(fvisit);
    visitor(Downcast<Expr>(node));
  }
}

class IRTransformer final :
      public StmtExprMutator {
 public:
  IRTransformer(const runtime::PackedFunc& f_preorder,
                const runtime::PackedFunc& f_postorder,
                const std::unordered_set<uint32_t>& only_enable)
      : f_preorder_(f_preorder),
        f_postorder_(f_postorder),
        only_enable_(only_enable) {
  }
  Stmt VisitStmt(const Stmt& stmt) final {
    return MutateInternal<Stmt>(stmt, [this](const Stmt& s) {
      return StmtMutator::VisitStmt(s);
    });
  }
  Expr VisitExpr(const Expr& expr) final {
    return MutateInternal<Expr>(expr, [this](const Expr& e) {
      return ExprMutator::VisitExpr(e);
    });
  }

 private:
  template <typename T, typename F>
  T MutateInternal(const T& node, F fmutate) {
    if (only_enable_.size() &&
        !only_enable_.count(node->type_index())) {
      return fmutate(node);
    }
    if (f_preorder_ != nullptr) {
      T pre = f_preorder_(node);
      if (pre.defined()) return pre;
    }
    T new_node = fmutate(node);
    if (f_postorder_ != nullptr) {
      T post = f_postorder_(new_node);
      if (post.defined()) return post;
    }
    return new_node;
  }
  // The functions
  const runtime::PackedFunc& f_preorder_;
  const runtime::PackedFunc& f_postorder_;
  // type indices enabled.
  const std::unordered_set<uint32_t>& only_enable_;
};

Stmt IRTransform(Stmt ir_node,
                 const runtime::PackedFunc& f_preorder,
                 const runtime::PackedFunc& f_postorder,
                 const Array<Expr>& only_enable) {
  std::unordered_set<uint32_t> only_type_index;
  for (Expr s : only_enable) {
    only_type_index.insert(Object::TypeKey2Index(s.as<StringImm>()->value.c_str()));
  }
  IRTransformer transform(f_preorder, f_postorder, only_type_index);
  return transform(std::move(ir_node));
}

// Implementation of Visitors
template<typename T, typename F>
inline void VisitArray(const Array<T>& arr, F fvisit) {
  for (size_t i = 0; i < arr.size(); i++) {
    fvisit(arr[i]);
  }
}

void StmtVisitor::VisitStmt_(const LetStmt* op) {
  this->VisitExpr(op->value);
  this->VisitStmt(op->body);
}

void StmtVisitor::VisitStmt_(const AttrStmt* op) {
  this->VisitExpr(op->value);
  this->VisitStmt(op->body);
}

void StmtVisitor::VisitStmt_(const For* op) {
  this->VisitExpr(op->min);
  this->VisitExpr(op->extent);
  this->VisitStmt(op->body);
}

void StmtVisitor::VisitStmt_(const Allocate* op) {
  VisitArray(op->extents, [this](const Expr& e) { this->VisitExpr(e); });
  this->VisitStmt(op->body);
  this->VisitExpr(op->condition);
  if (op->new_expr.defined()) {
    this->VisitExpr(op->new_expr);
  }
}

void StmtVisitor::VisitStmt_(const Store* op) {
  this->VisitExpr(op->value);
  this->VisitExpr(op->index);
  this->VisitExpr(op->predicate);
}

void StmtVisitor::VisitStmt_(const IfThenElse* op) {
  this->VisitExpr(op->condition);
  this->VisitStmt(op->then_case);
  if (op->else_case.defined()) {
    this->VisitStmt(op->else_case);
  }
}

void StmtVisitor::VisitStmt_(const Free* op) {}

void StmtVisitor::VisitStmt_(const AssertStmt* op) {
  this->VisitExpr(op->condition);
  this->VisitExpr(op->message);
  this->VisitStmt(op->body);
}

void StmtVisitor::VisitStmt_(const ProducerConsumer* op) {
  this->VisitStmt(op->body);
}

void StmtVisitor::VisitStmt_(const Provide* op) {
  VisitArray(op->args, [this](const Expr& e) { this->VisitExpr(e); });
  this->VisitExpr(op->value);
}

void StmtVisitor::VisitStmt_(const Realize* op) {
  VisitArray(op->bounds, [this](const Range& r) {
      this->VisitExpr(r->min);
      this->VisitExpr(r->extent);
    });
  this->VisitStmt(op->body);
  this->VisitExpr(op->condition);
}

void StmtVisitor::VisitStmt_(const Prefetch* op) {
  VisitArray(op->bounds, [this](const Range& r) {
      this->VisitExpr(r->min);
      this->VisitExpr(r->extent);
    });
}

void StmtVisitor::VisitStmt_(const Block* op) {
  this->VisitStmt(op->first);
  this->VisitStmt(op->rest);
}

void StmtVisitor::VisitStmt_(const Evaluate* op) {
  this->VisitExpr(op->value);
}

void ExprVisitor::VisitExpr_(const Variable* op) {}

void ExprVisitor::VisitExpr_(const Load* op) {
  this->VisitExpr(op->index);
  this->VisitExpr(op->predicate);
}

void ExprVisitor::VisitExpr_(const Let* op) {
  this->VisitExpr(op->value);
  this->VisitExpr(op->body);
}

void ExprVisitor::VisitExpr_(const Call* op) {
  VisitArray(op->args, [this](const Expr& e) { this->VisitExpr(e); });
}

#define DEFINE_BINOP_VISIT_(OP)                           \
  void ExprVisitor::VisitExpr_(const OP* op) {            \
    this->VisitExpr(op->a);                               \
    this->VisitExpr(op->b);                               \
  }

DEFINE_BINOP_VISIT_(Add);
DEFINE_BINOP_VISIT_(Sub);
DEFINE_BINOP_VISIT_(Mul);
DEFINE_BINOP_VISIT_(Div);
DEFINE_BINOP_VISIT_(Mod);
DEFINE_BINOP_VISIT_(FloorDiv);
DEFINE_BINOP_VISIT_(FloorMod);
DEFINE_BINOP_VISIT_(Min);
DEFINE_BINOP_VISIT_(Max);
DEFINE_BINOP_VISIT_(EQ);
DEFINE_BINOP_VISIT_(NE);
DEFINE_BINOP_VISIT_(LT);
DEFINE_BINOP_VISIT_(LE);
DEFINE_BINOP_VISIT_(GT);
DEFINE_BINOP_VISIT_(GE);
DEFINE_BINOP_VISIT_(And);
DEFINE_BINOP_VISIT_(Or);

void ExprVisitor::VisitExpr_(const IntImm* op) {}
void ExprVisitor::VisitExpr_(const UIntImm* op) {}
void ExprVisitor::VisitExpr_(const FloatImm* op) {}
void ExprVisitor::VisitExpr_(const StringImm* op) {}

void ExprVisitor::VisitExpr_(const Reduce* op) {
  VisitArray(op->axis, [this](const IterVar& r) {
      this->VisitExpr(r->dom->min);
      this->VisitExpr(r->dom->extent);
    });
  VisitArray(op->source, [this](const Expr& e) { this->VisitExpr(e); });
  this->VisitExpr(op->condition);
}

void ExprVisitor::VisitExpr_(const Cast* op) {
  this->VisitExpr(op->value);
}

void ExprVisitor::VisitExpr_(const Not* op) {
  this->VisitExpr(op->a);
}

void ExprVisitor::VisitExpr_(const Select* op) {
  this->VisitExpr(op->condition);
  this->VisitExpr(op->true_value);
  this->VisitExpr(op->false_value);
}

void ExprVisitor::VisitExpr_(const Ramp* op) {
  this->VisitExpr(op->base);
  this->VisitExpr(op->stride);
}

void ExprVisitor::VisitExpr_(const Shuffle* op) {
  VisitArray(op->indices, [this](const Expr& e) { this->VisitExpr(e); });
  VisitArray(op->vectors, [this](const Expr& e) { this->VisitExpr(e); });
}

void ExprVisitor::VisitExpr_(const Broadcast* op) {
  this->VisitExpr(op->value);
}

// Implementation of mutators
template<typename T, typename F>
inline Array<T> MutateArray(const Array<T>& arr,
                            F fmutate,
                            bool allow_copy_on_write = false) {
  if (allow_copy_on_write) {
    // if we allow copy on write, we can directly
    // call the inplace mutate function.
    const_cast<Array<T>&>(arr).MutateByApply(fmutate);
    return arr;
  } else {
    Array<T> copy = arr;
    copy.MutateByApply(fmutate);
    return copy;
  }
}

class StmtMutator::Internal {
 public:
  static Array<Expr> Mutate(StmtMutator* self, const Array<Expr>& arr) {
    auto fmutate = [self](const Expr& e) { return self->VisitExpr(e); };
    return MutateArray(arr, fmutate, self->allow_copy_on_write_);
  }

  static Array<Stmt> Mutate(StmtMutator* self, const Array<Stmt>& arr) {
    auto fmutate = [self](const Stmt& s) { return self->VisitStmt(s); };
    return MutateArray(arr, fmutate, self->allow_copy_on_write_);
  }

  static Array<Range> Mutate(StmtMutator* self, const Array<Range>& arr) {
    auto fmutate = [self](const Range& r) {
      Expr min = self->VisitExpr(r->min);
      Expr extent = self->VisitExpr(r->extent);
      if (min.same_as(r->min) && extent.same_as(r->extent)) {
        return r;
      } else {
        return Range::make_by_min_extent(min, extent);
      }
    };
    return MutateArray(arr, fmutate, self->allow_copy_on_write_);
  }
};

Stmt StmtMutator::VisitStmt_(const AttrStmt* op) {
  Expr value = this->VisitExpr(op->value);
  Stmt body = this->VisitStmt(op->body);
  if (value.same_as(op->value) &&
      body.same_as(op->body)) {
    return GetRef<Stmt>(op);
  } else {
    auto n = CopyOnWrite(op);
    n->value = std::move(value);
    n->body = std::move(body);
    return Stmt(n);
  }
}

Stmt StmtMutator::VisitStmt_(const LetStmt* op) {
  Expr value = this->VisitExpr(op->value);
  Stmt body = this->VisitStmt(op->body);
  if (value.same_as(op->value) &&
      body.same_as(op->body)) {
    return GetRef<Stmt>(op);
  } else {
    auto n = CopyOnWrite(op);
    n->value = std::move(value);
    n->body = std::move(body);
    return Stmt(n);
  }
}

Stmt StmtMutator::VisitStmt_(const For* op) {
  Expr min = this->VisitExpr(op->min);
  Expr extent = this->VisitExpr(op->extent);
  Stmt body = this->VisitStmt(op->body);
  if (min.same_as(op->min) &&
      extent.same_as(op->extent) &&
      body.same_as(op->body)) {
    return GetRef<Stmt>(op);
  } else {
    auto n = CopyOnWrite(op);
    n->min = std::move(min);
    n->extent = std::move(extent);
    n->body = std::move(body);
    return Stmt(n);
  }
}

Stmt StmtMutator::VisitStmt_(const Allocate* op) {
  Array<Expr> extents = Internal::Mutate(this, op->extents);
  Stmt body = this->VisitStmt(op->body);
  Expr condition = this->VisitExpr(op->condition);
  Expr new_expr;
  if (op->new_expr.defined()) {
    new_expr = this->VisitExpr(op->new_expr);
  }
  if (extents.same_as(op->extents) &&
      body.same_as(op->body) &&
      condition.same_as(op->condition) &&
      new_expr.same_as(op->new_expr)) {
    return GetRef<Stmt>(op);
  } else {
    auto n = CopyOnWrite(op);
    n->extents = std::move(extents);
    n->body = std::move(body);
    n->condition = std::move(condition);
    n->new_expr = std::move(new_expr);
    return Stmt(n);
  }
}

Stmt StmtMutator::VisitStmt_(const IfThenElse* op) {
  Expr condition = this->VisitExpr(op->condition);
  Stmt then_case = this->VisitStmt(op->then_case);
  Stmt else_case;
  if (op->else_case.defined()) {
    else_case = this->VisitStmt(op->else_case);
  }
  if (condition.same_as(op->condition) &&
      then_case.same_as(op->then_case) &&
      else_case.same_as(op->else_case)) {
    return GetRef<Stmt>(op);
  } else {
    auto n = CopyOnWrite(op);
    n->condition = std::move(condition);
    n->then_case = std::move(then_case);
    n->else_case = std::move(else_case);
    return Stmt(n);
  }
}

Stmt StmtMutator::VisitStmt_(const Store* op) {
  Expr value = this->VisitExpr(op->value);
  Expr index = this->VisitExpr(op->index);
  Expr predicate = this->VisitExpr(op->predicate);
  if (value.same_as(op->value) &&
      index.same_as(op->index) &&
      predicate.same_as(op->predicate)) {
    return GetRef<Stmt>(op);
  } else {
    auto n = CopyOnWrite(op);
    n->value = std::move(value);
    n->index = std::move(index);
    n->predicate = std::move(predicate);
    return Stmt(n);
  }
}

Stmt StmtMutator::VisitStmt_(const Provide* op) {
  Array<Expr> args = Internal::Mutate(this, op->args);
  Expr value = this->VisitExpr(op->value);
  if (args.same_as(op->args) &&
      value.same_as(op->value)) {
    return GetRef<Stmt>(op);
  } else {
    auto n = CopyOnWrite(op);
    n->args = std::move(args);
    n->value = std::move(value);
    return Stmt(n);
  }
}

Stmt StmtMutator::VisitStmt_(const Realize* op) {
  Region bounds = Internal::Mutate(this, op->bounds);
  Stmt body = this->VisitStmt(op->body);
  Expr condition = this->VisitExpr(op->condition);
  if (bounds.same_as(op->bounds) &&
      body.same_as(op->body) &&
      condition.same_as(op->condition)) {
    return GetRef<Stmt>(op);
  } else {
    auto n = CopyOnWrite(op);
    n->bounds = std::move(bounds);
    n->body = std::move(body);
    n->condition = std::move(condition);
    return Stmt(n);
  }
}

Stmt StmtMutator::VisitStmt_(const Prefetch* op) {
  Region bounds = Internal::Mutate(this, op->bounds);
  if (bounds.same_as(op->bounds)) {
    return GetRef<Stmt>(op);
  } else {
    auto n = CopyOnWrite(op);
    n->bounds = std::move(bounds);
    return Stmt(n);
  }
}

Stmt StmtMutator::VisitStmt_(const Block* op) {
  Stmt first = this->VisitStmt(op->first);
  Stmt rest = this->VisitStmt(op->rest);
  if (first.same_as(op->first) &&
      rest.same_as(op->rest)) {
    return GetRef<Stmt>(op);
  } else {
    auto n = CopyOnWrite(op);
    n->first = std::move(first);
    n->rest = std::move(rest);
    return Stmt(n);
  }
}

Stmt StmtMutator::VisitStmt_(const AssertStmt* op) {
  Expr condition = this->VisitExpr(op->condition);
  Expr message = this->VisitExpr(op->message);
  Stmt body = this->VisitStmt(op->body);

  if (condition.same_as(op->condition) &&
      message.same_as(op->message) &&
      body.same_as(op->body)) {
    return GetRef<Stmt>(op);
  } else {
    auto n = CopyOnWrite(op);
    n->condition = std::move(condition);
    n->message = std::move(message);
    n->body = std::move(body);
    return Stmt(n);
  }
}

Stmt StmtMutator::VisitStmt_(const ProducerConsumer* op) {
  Stmt body = this->VisitStmt(op->body);
  if (body.same_as(op->body)) {
    return GetRef<Stmt>(op);
  } else {
    auto n = CopyOnWrite(op);
    n->body = std::move(body);
    return Stmt(n);
  }
}

Stmt StmtMutator::VisitStmt_(const Evaluate* op) {
  Expr value = this->VisitExpr(op->value);
  if (value.same_as(op->value)) {
    return GetRef<Stmt>(op);
  } else {
    auto n = CopyOnWrite(op);
    n->value = std::move(value);
    return Stmt(n);
  }
}

Stmt StmtMutator::VisitStmt_(const Free* op) {
  return GetRef<Stmt>(op);
}


Expr ExprMutator::VisitExpr_(const Variable* op) {
  return GetRef<Expr>(op);
}

Expr ExprMutator::VisitExpr_(const Load* op) {
  Expr index = this->VisitExpr(op->index);
  Expr predicate = this->VisitExpr(op->predicate);
  if (index.same_as(op->index) && predicate.same_as(op->predicate)) {
    return GetRef<Expr>(op);
  } else {
    return Load::make(op->dtype, op->buffer_var, index, predicate);
  }
}

Expr ExprMutator::VisitExpr_(const Let* op) {
  Expr value = this->VisitExpr(op->value);
  Expr body = this->VisitExpr(op->body);
  if (value.same_as(op->value) &&
      body.same_as(op->body)) {
    return GetRef<Expr>(op);
  } else {
    return Let::make(op->var, value, body);
  }
}

Expr ExprMutator::VisitExpr_(const Call* op) {
  auto fmutate = [this](const Expr& e) { return this->VisitExpr(e); };
  Array<Expr> args = MutateArray(op->args, fmutate);

  if (args.same_as(op->args)) {
    return GetRef<Expr>(op);
  } else {
    return Call::make(op->dtype,
                      op->name,
                      args,
                      op->call_type,
                      op->func,
                      op->value_index);
  }
}

#define DEFINE_OP_RETURN_SELF_EXPR_MUTATE_(OP)                    \
  Expr ExprMutator::VisitExpr_(const OP *op) {                    \
    return GetRef<Expr>(op);                                      \
  }

DEFINE_OP_RETURN_SELF_EXPR_MUTATE_(IntImm)
DEFINE_OP_RETURN_SELF_EXPR_MUTATE_(UIntImm)
DEFINE_OP_RETURN_SELF_EXPR_MUTATE_(FloatImm)
DEFINE_OP_RETURN_SELF_EXPR_MUTATE_(StringImm)

#define DEFINE_BIOP_EXPR_MUTATE_(OP)                                    \
  Expr ExprMutator::VisitExpr_(const OP* op) {                          \
    Expr a = this->VisitExpr(op->a);                                    \
    Expr b = this->VisitExpr(op->b);                                    \
    if (a.same_as(op->a) &&                                             \
        b.same_as(op->b)) {                                             \
      return GetRef<Expr>(op);                                          \
    } else {                                                            \
      return OP::make(a, b);                                            \
    }                                                                   \
  }

DEFINE_BIOP_EXPR_MUTATE_(Add);
DEFINE_BIOP_EXPR_MUTATE_(Sub);
DEFINE_BIOP_EXPR_MUTATE_(Mul);
DEFINE_BIOP_EXPR_MUTATE_(Div);
DEFINE_BIOP_EXPR_MUTATE_(Mod);
DEFINE_BIOP_EXPR_MUTATE_(FloorDiv);
DEFINE_BIOP_EXPR_MUTATE_(FloorMod);
DEFINE_BIOP_EXPR_MUTATE_(Min);
DEFINE_BIOP_EXPR_MUTATE_(Max);
DEFINE_BIOP_EXPR_MUTATE_(EQ);
DEFINE_BIOP_EXPR_MUTATE_(NE);
DEFINE_BIOP_EXPR_MUTATE_(LT);
DEFINE_BIOP_EXPR_MUTATE_(LE);
DEFINE_BIOP_EXPR_MUTATE_(GT);
DEFINE_BIOP_EXPR_MUTATE_(GE);
DEFINE_BIOP_EXPR_MUTATE_(And);
DEFINE_BIOP_EXPR_MUTATE_(Or);

Expr ExprMutator::VisitExpr_(const Reduce* op) {
  auto fitervar =  [this](const IterVar& v) {
    Range r = v->dom;
    Expr min = this->VisitExpr(r->min);
    Expr extent = this->VisitExpr(r->extent);
    if (min.same_as(r->min) &&
        extent.same_as(r->extent)) {
      return v;
    } else {
      return IterVarNode::make(
          Range::make_by_min_extent(min, extent),
          v->var, v->iter_type, v->thread_tag);
    }
  };
  Array<IterVar> axis = MutateArray(op->axis, fitervar);

  auto fexpr = [this](const Expr& e) { return this->VisitExpr(e); };
  Array<Expr> source = MutateArray(op->source, fexpr);

  Expr condition = this->VisitExpr(op->condition);

  if (axis.same_as(op->axis) &&
      source.same_as(op->source) &&
      condition.same_as(op->condition)) {
    return GetRef<Expr>(op);
  } else {
    return Reduce::make(
      op->combiner, source, axis, condition, op->value_index);
  }
}

Expr ExprMutator::VisitExpr_(const Cast* op) {
  Expr value = this->VisitExpr(op->value);
  if (value.same_as(op->value)) {
    return GetRef<Expr>(op);
  } else {
    return Cast::make(op->dtype, value);
  }
}

Expr ExprMutator::VisitExpr_(const Not* op) {
  Expr a = this->VisitExpr(op->a);
  if (a.same_as(op->a)) {
    return GetRef<Expr>(op);
  } else {
    return Not::make(a);
  }
}

Expr ExprMutator::VisitExpr_(const Select* op) {
  Expr condition = this->VisitExpr(op->condition);
  Expr true_value = this->VisitExpr(op->true_value);
  Expr false_value = this->VisitExpr(op->false_value);
  if (condition.same_as(op->condition) &&
      true_value.same_as(op->true_value) &&
      false_value.same_as(op->false_value)) {
    return GetRef<Expr>(op);
  } else {
    return Select::make(condition, true_value, false_value);
  }
}

Expr ExprMutator::VisitExpr_(const Ramp* op) {
  Expr base = this->VisitExpr(op->base);
  Expr stride = this->VisitExpr(op->stride);
  if (base.same_as(op->base) &&
      stride.same_as(op->stride)) {
    return GetRef<Expr>(op);
  } else {
    return Ramp::make(base, stride, op->lanes);
  }
}

Expr ExprMutator::VisitExpr_(const Broadcast* op) {
  Expr value = this->VisitExpr(op->value);
  if (value.same_as(op->value)) {
    return GetRef<Expr>(op);
  } else {
    return Broadcast::make(value, op->lanes);
  }
}

Expr ExprMutator::VisitExpr_(const Shuffle* op) {
  auto fexpr = [this](const Expr& e) { return this->VisitExpr(e); };
  auto vectors = MutateArray(op->vectors, fexpr);
  if (vectors.same_as(op->vectors)) {
    return GetRef<Expr>(op);
  } else {
    return Shuffle::make(vectors, op->indices);
  }
}

}  // namespace ir
}  // namespace tvm
