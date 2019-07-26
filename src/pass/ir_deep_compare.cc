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
 *  Copyright (c) 2017 by Contributors
 * \file ir_deep_compare.cc
 */
#include <tvm/ir_pass.h>
#include <tvm/ir_functor_ext.h>

namespace tvm {
namespace ir {

using ExprComparator = ExprFunctor<void(const Expr& n, const Expr &other)>;
using StmtComparator = StmtFunctor<void(const Stmt& n, const Stmt &other)>;

#define DEFINE_BIOP_EXPR_CMP_(OP)                                 \
  void VisitExpr_(const OP* op, const Expr& other) final {        \
    const OP* rhs = other.as<OP>();                               \
    if (CompareExpr(op->a, rhs->a) != 0) return;                      \
    if (CompareExpr(op->b, rhs->b) != 0) return;                      \
  }

// Deep comparison to check if two IR graph are equivalent
class IRDeepCompare :
      public ExprComparator, public StmtComparator {
 public:
  // Equality comparison
  bool Equal(const Stmt& lhs, const Stmt& rhs) {
    tie_def_ = true;
    VisitStmt(lhs, rhs);
    return order_ == 0;
  }

  bool Equal(const Expr& lhs, const Expr& rhs) {
    tie_def_ = true;
    VisitExpr(lhs, rhs);
    return order_ == 0;
  }

  int Compare(const Expr& lhs, const Expr& rhs) {
    tie_def_ = false;
    VisitExpr(lhs, rhs);
    return order_;
  }

  void VisitExpr(const Expr& n, const Expr& other) override {
    if (order_ != 0) return;
    if (n.same_as(other)) return;
    if (CompareValue(n->type_index(), other->type_index()) != 0) return;
    if (CompareType(n.type(), other.type()) != 0) return;
    ExprComparator::VisitExpr(n, other);
  }

  void VisitStmt(const Stmt& n, const Stmt& other) override {
    if (order_ != 0) return;
    if (n.same_as(other)) return;
    if (CompareValue(n->type_index(), other->type_index()) != 0) return;
    StmtComparator::VisitStmt(n, other);
  }
  // Stmt
  void VisitStmt_(const LetStmt* op, const Stmt& other) final {
    const LetStmt* rhs = other.as<LetStmt>();
    if (CompareExpr(op->value, rhs->value) != 0) return;
    if (tie_def_) {
      vmap_[op->var.get()] = rhs->var.get();
    } else {
      if (CompareExpr(op->var, rhs->var) != 0) return;
    }
    if (CompareStmt(op->body, rhs->body) != 0) return;
  }

  void VisitStmt_(const AttrStmt* op, const Stmt& other) final {
    const AttrStmt* rhs = other.as<AttrStmt>();
    if (CompareString(op->attr_key, rhs->attr_key) != 0) return;
    if (CompareNodeRef(op->node, rhs->node) != 0) return;
    if (CompareExpr(op->value, rhs->value) != 0) return;
    if (CompareStmt(op->body, rhs->body) != 0) return;
  }

  void VisitStmt_(const IfThenElse* op, const Stmt& other) final {
    const IfThenElse* rhs = other.as<IfThenElse>();
    if (CompareExpr(op->condition, rhs->condition) != 0) return;
    if (CompareStmt(op->then_case, rhs->then_case) != 0) return;
    if (CompareStmt(op->else_case, rhs->else_case) != 0) return;
  }

  void VisitStmt_(const For* op, const Stmt& other) final {
    const For* rhs = other.as<For>();
    if (CompareExpr(op->min, rhs->min) != 0) return;
    if (CompareExpr(op->extent, rhs->extent) != 0) return;
    if (tie_def_) {
      vmap_[op->loop_var.get()] = rhs->loop_var.get();
    } else {
      if (CompareExpr(op->loop_var, rhs->loop_var) != 0) return;
    }
    if (CompareStmt(op->body, rhs->body) != 0) return;
  }

  void VisitStmt_(const Allocate* op, const Stmt& other) final {
    const Allocate* rhs = other.as<Allocate>();
    if (tie_def_) {
      vmap_[op->buffer_var.get()] = rhs->buffer_var.get();
    } else {
      if (CompareExpr(op->buffer_var, rhs->buffer_var) != 0) return;
    }
    if (CompareType(op->type, rhs->type) != 0) return;
    if (CompareArray(op->extents, rhs->extents) != 0) return;
    if (CompareExpr(op->condition, rhs->condition) != 0) return;
    if (CompareStmt(op->body, rhs->body) != 0) return;
    if (CompareExpr(op->new_expr, rhs->new_expr) != 0) return;
    if (CompareString(op->free_function, rhs->free_function) != 0) return;
  }

  void VisitStmt_(const Store* op, const Stmt& other) final {
    const Store* rhs = other.as<Store>();
    if (CompareExpr(op->buffer_var, rhs->buffer_var) != 0) return;
    if (CompareExpr(op->value, rhs->value) != 0) return;
    if (CompareExpr(op->index, rhs->index) != 0) return;
    if (CompareExpr(op->predicate, rhs->predicate) != 0) return;
  }

  void VisitStmt_(const Free* op, const Stmt& other) final {
    const Free* rhs = other.as<Free>();
    if (CompareExpr(op->buffer_var, rhs->buffer_var) != 0) return;
  }

  void VisitStmt_(const AssertStmt* op, const Stmt& other) final {
    const AssertStmt* rhs = other.as<AssertStmt>();
    if (CompareExpr(op->condition, rhs->condition) != 0) return;
    if (CompareExpr(op->message, rhs->message) != 0) return;
    if (CompareStmt(op->body, rhs->body) != 0) return;
  }

  void VisitStmt_(const ProducerConsumer* op, const Stmt& other) final {
    const ProducerConsumer* rhs = other.as<ProducerConsumer>();
    if (CompareNodeRef(op->func, rhs->func) != 0) return;
    if (CompareValue(op->is_producer, rhs->is_producer) != 0) return;
    if (CompareStmt(op->body, rhs->body) != 0) return;
  }

  void VisitStmt_(const Provide* op, const Stmt& other) final {
    const Provide* rhs = other.as<Provide>();
    if (CompareNodeRef(op->func, rhs->func) != 0) return;
    if (CompareValue(op->value_index, rhs->value_index) != 0) return;
    if (CompareExpr(op->value, rhs->value) != 0) return;
    if (CompareArray(op->args, rhs->args) != 0) return;
  }

  void VisitStmt_(const Realize* op, const Stmt& other) final {
    const Realize* rhs = other.as<Realize>();
    if (CompareNodeRef(op->func, rhs->func) != 0) return;
    if (CompareValue(op->value_index, rhs->value_index) != 0) return;
    if (CompareType(op->type, rhs->type) != 0) return;
    if (CompareRegion(op->bounds, rhs->bounds) != 0) return;
    if (CompareStmt(op->body, rhs->body) != 0) return;
  }

  void VisitStmt_(const Prefetch* op, const Stmt& other) final {
    const Prefetch* rhs = other.as<Prefetch>();
    if (CompareNodeRef(op->func, rhs->func) != 0) return;
    if (CompareValue(op->value_index, rhs->value_index) != 0) return;
    if (CompareType(op->type, rhs->type) != 0) return;
    if (CompareRegion(op->bounds, rhs->bounds) != 0) return;
  }

  void VisitStmt_(const Block* op, const Stmt& other) final {
    const Block* rhs = other.as<Block>();
    if (CompareStmt(op->first, rhs->first) != 0) return;
    if (CompareStmt(op->rest, rhs->rest) != 0) return;
  }

  void VisitStmt_(const Evaluate* op, const Stmt& other) final {
    const Evaluate* rhs = other.as<Evaluate>();
    CompareExpr(op->value, rhs->value);
  }

  // Exprs
  void VisitExpr_(const Variable* op, const Expr& other) final {
    const Variable* rhs = other.as<Variable>();
    auto it = vmap_.find(op);
    if (it != vmap_.end()) op = it->second;
    if (op < rhs) {
      order_ = -1;
    } else if (op > rhs) {
      order_ = +1;
    }
  }
  void VisitExpr_(const Load* op, const Expr& other) final {
    const Load* rhs = other.as<Load>();
    if (CompareExpr(op->buffer_var, rhs->buffer_var) != 0) return;
    if (CompareExpr(op->index, rhs->index) != 0) return;
    if (CompareExpr(op->predicate, rhs->predicate) != 0) return;
  }

  void VisitExpr_(const Let* op, const Expr& other) final {
    const Let* rhs = other.as<Let>();
    if (tie_def_) {
      vmap_[op->var.get()] = rhs->var.get();
    } else {
      if (CompareExpr(op->var, rhs->var) != 0) return;
    }
    if (CompareExpr(op->value, rhs->value) != 0) return;
    if (CompareExpr(op->body, rhs->body) != 0) return;
  }

  void VisitExpr_(const Call* op, const Expr& other) final {
    const Call* rhs = other.as<Call>();
    if (CompareString(op->name, rhs->name)) return;
    if (CompareArray(op->args, rhs->args)) return;
    if (CompareValue(op->call_type, rhs->call_type) != 0) return;
    if (CompareNodeRef(op->func, rhs->func) != 0) return;
    if (CompareValue(op->value_index, rhs->value_index) != 0) return;
  }

  void VisitExpr_(const Reduce *op, const Expr& other) final {
    const Reduce* rhs = other.as<Reduce>();
    if (CompareCommReducer(op->combiner, rhs->combiner) != 0) return;
    if (CompareValue(op->axis.size(), rhs->axis.size()) != 0) return;
    if (CompareValue(op->value_index, rhs->value_index) != 0) return;
    for (size_t i = 0; i < op->axis.size(); ++i) {
      if (CompareExpr(op->axis[i]->dom->min, rhs->axis[i]->dom->min) != 0) return;
      if (CompareExpr(op->axis[i]->dom->extent, rhs->axis[i]->dom->extent) != 0) return;
      if (tie_def_) {
        vmap_[op->axis[i]->var.get()] = rhs->axis[i]->var.get();
      } else {
        if (CompareExpr(op->axis[i]->var, rhs->axis[i]->var) != 0) return;
      }
    }
    if (CompareExpr(op->condition, rhs->condition) != 0) return;
    if (CompareArray(op->source, rhs->source) != 0) return;
  }

  void VisitExpr_(const IntImm *op, const Expr& other) final {
    CompareValue(op->value, other.as<IntImm>()->value);
  }

  void VisitExpr_(const UIntImm *op, const Expr& other) final {
    CompareValue(op->value, other.as<UIntImm>()->value);
  }

  void VisitExpr_(const FloatImm *op, const Expr& other) final {
    CompareValue(op->value, other.as<FloatImm>()->value);
  }

  void VisitExpr_(const StringImm *op, const Expr& other) final {
    CompareString(op->value, other.as<StringImm>()->value);
  }

  void VisitExpr_(const Cast *op, const Expr& other) final {
    CompareExpr(op->value, other.as<Cast>()->value);
  }

  void VisitExpr_(const Not *op, const Expr& other) final {
    CompareExpr(op->a, other.as<Not>()->a);
  }

  void VisitExpr_(const Select *op, const Expr& other) final {
    const Select* rhs = other.as<Select>();
    if (CompareExpr(op->condition, rhs->condition) != 0) return;
    if (CompareExpr(op->true_value, rhs->true_value) != 0) return;
    if (CompareExpr(op->false_value, rhs->false_value) != 0) return;
  }

  void VisitExpr_(const Ramp *op, const Expr& other) final {
    const Ramp* rhs = other.as<Ramp>();
    if (CompareExpr(op->base, rhs->base) != 0) return;
    if (CompareExpr(op->stride, rhs->stride) != 0) return;
    if (CompareValue(op->lanes, rhs->lanes) != 0) return;
  }

  void VisitExpr_(const Broadcast *op, const Expr& other) final {
    const Broadcast* rhs = other.as<Broadcast>();
    if (CompareExpr(op->value, rhs->value) != 0) return;
    if (CompareValue(op->lanes, rhs->lanes) != 0) return;
  }

  void VisitExpr_(const Shuffle *op, const Expr& other) final {
    const Shuffle* rhs = other.as<Shuffle>();
    if (CompareArray(op->vectors, rhs->vectors) != 0) return;
    if (CompareArray(op->indices, rhs->indices) != 0) return;
  }

  DEFINE_BIOP_EXPR_CMP_(Add)
  DEFINE_BIOP_EXPR_CMP_(Sub)
  DEFINE_BIOP_EXPR_CMP_(Mul)
  DEFINE_BIOP_EXPR_CMP_(Div)
  DEFINE_BIOP_EXPR_CMP_(Mod)
  DEFINE_BIOP_EXPR_CMP_(FloorDiv)
  DEFINE_BIOP_EXPR_CMP_(FloorMod)
  DEFINE_BIOP_EXPR_CMP_(Min)
  DEFINE_BIOP_EXPR_CMP_(Max)
  DEFINE_BIOP_EXPR_CMP_(EQ)
  DEFINE_BIOP_EXPR_CMP_(NE)
  DEFINE_BIOP_EXPR_CMP_(LT)
  DEFINE_BIOP_EXPR_CMP_(LE)
  DEFINE_BIOP_EXPR_CMP_(GT)
  DEFINE_BIOP_EXPR_CMP_(GE)
  DEFINE_BIOP_EXPR_CMP_(And)
  DEFINE_BIOP_EXPR_CMP_(Or)

 private:
  int CompareExpr(const Expr& lhs, const Expr& rhs) {
    if (order_ != 0) return order_;
    if (!lhs.defined() && rhs.defined()) {
      order_ = -1; return order_;
    }
    if (!rhs.defined() && lhs.defined()) {
      order_ = +1; return order_;
    }
    VisitExpr(lhs, rhs);
    return order_;
  }

  int CompareStmt(const Stmt& lhs, const Stmt& rhs) {
    if (order_ != 0) return order_;
    if (!lhs.defined() && rhs.defined()) {
      order_ = -1; return order_;
    }
    if (!rhs.defined() && lhs.defined()) {
      order_ = +1; return order_;
    }
    VisitStmt(lhs, rhs);
    return order_;
  }

  int CompareArray(const Array<Expr>& lhs, const Array<Expr>& rhs) {
    if (order_ != 0) return order_;
    if (CompareValue(lhs.size(), rhs.size()) != 0) return order_;
    for (size_t i = 0; i < lhs.size(); ++i) {
      if (CompareExpr(lhs[i], rhs[i]) != 0) return order_;
    }
    return order_;
  }

  int CompareRegion(const Region& lhs, const Region& rhs) {
    if (order_ != 0) return order_;
    if (CompareValue(lhs.size(), rhs.size()) != 0) return order_;
    for (size_t i = 0; i < lhs.size(); ++i) {
      if (CompareExpr(lhs[i]->min, rhs[i]->min) != 0) return order_;
      if (CompareExpr(lhs[i]->extent, rhs[i]->extent) != 0) return order_;
    }
    return order_;
  }

  int CompareNodeRef(const NodeRef& lhs, const NodeRef& rhs) {
    if (order_ != 0) return order_;
    if (lhs.get() < rhs.get()) {
      order_ = -1; return order_;
    }
    if (lhs.get() > rhs.get()) {
      order_ = +1; return order_;
    }
    return order_;
  }

  int CompareType(const Type& lhs, const Type& rhs) {
    if (order_ != 0) return order_;
    if (lhs == rhs) return order_;
    if (CompareValue(lhs.code(), rhs.code()) != 0) return order_;
    if (CompareValue(lhs.bits(), rhs.bits()) != 0) return order_;
    if (CompareValue(lhs.lanes(), rhs.lanes()) != 0) return order_;
    return order_;
  }

  int CompareString(const std::string& lhs, const std::string& rhs) {
    if (order_ != 0) return order_;
    order_ = lhs.compare(rhs);
    return order_;
  }

  template<typename T>
  int CompareValue(const T& lhs, const T& rhs) {
    if (order_ != 0) return order_;
    if (lhs < rhs) {
      order_ = -1; return order_;
    } else if (lhs > rhs) {
      order_ = +1; return order_;
    }
    return order_;
  }

  int CompareCommReducer(const CommReducer& lhs, const CommReducer& rhs) {
    if (order_ != 0) return order_;
    if (lhs == rhs) return order_;
    if (CompareValue(lhs->lhs.size(), rhs->lhs.size()) != 0) return order_;
    if (CompareValue(lhs->rhs.size(), rhs->rhs.size()) != 0) return order_;
    IRDeepCompare cmp;
    if (tie_def_) {
      for (size_t i = 0; i < lhs->lhs.size(); ++i) {
        cmp.vmap_[lhs->lhs[i].get()] = rhs->lhs[i].get();
      }
      for (size_t i = 0; i < lhs->rhs.size(); ++i) {
        cmp.vmap_[lhs->rhs[i].get()] = rhs->rhs[i].get();
      }
    } else {
      for (size_t i = 0; i < lhs->lhs.size(); ++i) {
        if (CompareExpr(lhs->lhs[i], rhs->lhs[i]) != 0) return order_;
      }
      for (size_t i = 0; i < lhs->lhs.size(); ++i) {
        if (CompareExpr(lhs->rhs[i], rhs->rhs[i]) != 0) return order_;
      }
    }
    order_ = cmp.CompareArray(lhs->result, rhs->result);
    return order_;
  }
  // The order flag, smaller, -1, bigger: +1, equal: 0
  int order_{0};
  // Whether tie intermediate definitions.
  // This allows use to tie definitions of two variables together.
  // This enables us to assert equal between (let x in x + 1),  (let y in y + 1)
  // However, the comparison is no longer in total order.
  // Only equality/non-equality information is valid.
  bool tie_def_{false};
  // varaible remap if any
  std::unordered_map<const Variable*, const Variable*> vmap_;
};


bool Equal(const Stmt& lhs, const Stmt& rhs) {
  return IRDeepCompare().Equal(lhs, rhs);
}

bool Equal(const Expr& lhs, const Expr& rhs) {
  // quick pass for constant expressions.
  if (const int64_t *a = as_const_int(lhs)) {
    if (const int64_t *b = as_const_int(rhs)) {
      return a[0] == b[0];
    }
  }
  if (!lhs.defined()) {
    if (rhs.defined()) return false;
    if (!rhs.defined()) return true;
  } else {
    if (!rhs.defined()) return false;
  }
  // deep comparison.
  return IRDeepCompare().Equal(lhs, rhs);
}

int Compare(const Expr& lhs, const Expr& rhs) {
  return IRDeepCompare().Compare(lhs, rhs);
}

}  // namespace ir
}  // namespace tvm
