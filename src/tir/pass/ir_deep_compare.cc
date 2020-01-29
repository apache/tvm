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
 * \file ir_deep_compare.cc
 */
#include <tvm/tir/ir_pass.h>
#include <tvm/tir/expr_functor.h>
#include <tvm/tir/stmt_functor.h>

namespace tvm {
namespace tir {

using ExprComparator = ExprFunctor<void(const PrimExpr& n, const PrimExpr &other)>;
using StmtComparator = StmtFunctor<void(const Stmt& n, const Stmt &other)>;

#define DEFINE_BIOP_EXPR_CMP_(OP)                                 \
  void VisitExpr_(const OP* op, const PrimExpr& other) final {    \
    const OP* rhs = other.as<OP>();                               \
    if (CompareExpr(op->a, rhs->a) != 0) return;                  \
    if (CompareExpr(op->b, rhs->b) != 0) return;                  \
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

  bool Equal(const PrimExpr& lhs, const PrimExpr& rhs) {
    tie_def_ = true;
    VisitExpr(lhs, rhs);
    return order_ == 0;
  }

  int Compare(const PrimExpr& lhs, const PrimExpr& rhs) {
    tie_def_ = false;
    VisitExpr(lhs, rhs);
    return order_;
  }

  void VisitExpr(const PrimExpr& n, const PrimExpr& other) override {
    if (order_ != 0) return;
    if (n.same_as(other)) return;
    if (CompareValue(n->type_index(), other->type_index()) != 0) return;
    if (CompareType(n.dtype(), other.dtype()) != 0) return;
    ExprComparator::VisitExpr(n, other);
  }

  void VisitStmt(const Stmt& n, const Stmt& other) override {
    if (order_ != 0) return;
    if (n.same_as(other)) return;
    if (CompareValue(n->type_index(), other->type_index()) != 0) return;
    StmtComparator::VisitStmt(n, other);
  }
  // Stmt
  void VisitStmt_(const LetStmtNode* op, const Stmt& other) final {
    const LetStmtNode* rhs = other.as<LetStmtNode>();
    if (CompareExpr(op->value, rhs->value) != 0) return;
    if (tie_def_) {
      vmap_[op->var.get()] = rhs->var.get();
    } else {
      if (CompareExpr(op->var, rhs->var) != 0) return;
    }
    if (CompareStmt(op->body, rhs->body) != 0) return;
  }

  void VisitStmt_(const AttrStmtNode* op, const Stmt& other) final {
    const AttrStmtNode* rhs = other.as<AttrStmtNode>();
    if (CompareString(op->attr_key, rhs->attr_key) != 0) return;
    if (CompareNodeRef(op->node, rhs->node) != 0) return;
    if (CompareExpr(op->value, rhs->value) != 0) return;
    if (CompareStmt(op->body, rhs->body) != 0) return;
  }

  void VisitStmt_(const IfThenElseNode* op, const Stmt& other) final {
    const IfThenElseNode* rhs = other.as<IfThenElseNode>();
    if (CompareExpr(op->condition, rhs->condition) != 0) return;
    if (CompareStmt(op->then_case, rhs->then_case) != 0) return;
    if (CompareStmt(op->else_case, rhs->else_case) != 0) return;
  }

  void VisitStmt_(const ForNode* op, const Stmt& other) final {
    const ForNode* rhs = other.as<ForNode>();
    if (CompareExpr(op->min, rhs->min) != 0) return;
    if (CompareExpr(op->extent, rhs->extent) != 0) return;
    if (tie_def_) {
      vmap_[op->loop_var.get()] = rhs->loop_var.get();
    } else {
      if (CompareExpr(op->loop_var, rhs->loop_var) != 0) return;
    }
    if (CompareStmt(op->body, rhs->body) != 0) return;
  }

  void VisitStmt_(const AllocateNode* op, const Stmt& other) final {
    const AllocateNode* rhs = other.as<AllocateNode>();
    if (tie_def_) {
      vmap_[op->buffer_var.get()] = rhs->buffer_var.get();
    } else {
      if (CompareExpr(op->buffer_var, rhs->buffer_var) != 0) return;
    }
    if (CompareType(op->dtype, rhs->dtype) != 0) return;
    if (CompareArray(op->extents, rhs->extents) != 0) return;
    if (CompareExpr(op->condition, rhs->condition) != 0) return;
    if (CompareStmt(op->body, rhs->body) != 0) return;
    if (CompareExpr(op->new_expr, rhs->new_expr) != 0) return;
    if (CompareString(op->free_function, rhs->free_function) != 0) return;
  }

  void VisitStmt_(const StoreNode* op, const Stmt& other) final {
    const StoreNode* rhs = other.as<StoreNode>();
    if (CompareExpr(op->buffer_var, rhs->buffer_var) != 0) return;
    if (CompareExpr(op->value, rhs->value) != 0) return;
    if (CompareExpr(op->index, rhs->index) != 0) return;
    if (CompareExpr(op->predicate, rhs->predicate) != 0) return;
  }

  void VisitStmt_(const FreeNode* op, const Stmt& other) final {
    const FreeNode* rhs = other.as<FreeNode>();
    if (CompareExpr(op->buffer_var, rhs->buffer_var) != 0) return;
  }

  void VisitStmt_(const AssertStmtNode* op, const Stmt& other) final {
    const AssertStmtNode* rhs = other.as<AssertStmtNode>();
    if (CompareExpr(op->condition, rhs->condition) != 0) return;
    if (CompareExpr(op->message, rhs->message) != 0) return;
    if (CompareStmt(op->body, rhs->body) != 0) return;
  }

  void VisitStmt_(const ProducerConsumerNode* op, const Stmt& other) final {
    const ProducerConsumerNode* rhs = other.as<ProducerConsumerNode>();
    if (CompareNodeRef(op->func, rhs->func) != 0) return;
    if (CompareValue(op->is_producer, rhs->is_producer) != 0) return;
    if (CompareStmt(op->body, rhs->body) != 0) return;
  }

  void VisitStmt_(const ProvideNode* op, const Stmt& other) final {
    const ProvideNode* rhs = other.as<ProvideNode>();
    if (CompareNodeRef(op->func, rhs->func) != 0) return;
    if (CompareValue(op->value_index, rhs->value_index) != 0) return;
    if (CompareExpr(op->value, rhs->value) != 0) return;
    if (CompareArray(op->args, rhs->args) != 0) return;
  }

  void VisitStmt_(const RealizeNode* op, const Stmt& other) final {
    const RealizeNode* rhs = other.as<RealizeNode>();
    if (CompareNodeRef(op->func, rhs->func) != 0) return;
    if (CompareValue(op->value_index, rhs->value_index) != 0) return;
    if (CompareType(op->dtype, rhs->dtype) != 0) return;
    if (CompareRegion(op->bounds, rhs->bounds) != 0) return;
    if (CompareStmt(op->body, rhs->body) != 0) return;
  }

  void VisitStmt_(const PrefetchNode* op, const Stmt& other) final {
    const PrefetchNode* rhs = other.as<PrefetchNode>();
    if (CompareNodeRef(op->func, rhs->func) != 0) return;
    if (CompareValue(op->value_index, rhs->value_index) != 0) return;
    if (CompareType(op->dtype, rhs->dtype) != 0) return;
    if (CompareRegion(op->bounds, rhs->bounds) != 0) return;
  }

  void VisitStmt_(const SeqStmtNode* op, const Stmt& other) final {
    const SeqStmtNode* rhs = other.as<SeqStmtNode>();
    if (CompareValue(op->size(), rhs->size()) != 0) return;
    for (size_t i = 0; i < op->size(); ++i) {
      if (CompareStmt(op->seq[i], rhs->seq[i]) != 0) return;
    }
  }

  void VisitStmt_(const EvaluateNode* op, const Stmt& other) final {
    const EvaluateNode* rhs = other.as<EvaluateNode>();
    CompareExpr(op->value, rhs->value);
  }

  // Exprs
  void VisitExpr_(const VarNode* op, const PrimExpr& other) final {
    const VarNode* rhs = other.as<VarNode>();
    auto it = vmap_.find(op);
    if (it != vmap_.end()) op = it->second;
    if (op < rhs) {
      order_ = -1;
    } else if (op > rhs) {
      order_ = +1;
    }
  }
  void VisitExpr_(const LoadNode* op, const PrimExpr& other) final {
    const LoadNode* rhs = other.as<LoadNode>();
    if (CompareExpr(op->buffer_var, rhs->buffer_var) != 0) return;
    if (CompareExpr(op->index, rhs->index) != 0) return;
    if (CompareExpr(op->predicate, rhs->predicate) != 0) return;
  }

  void VisitExpr_(const LetNode* op, const PrimExpr& other) final {
    const LetNode* rhs = other.as<LetNode>();
    if (tie_def_) {
      vmap_[op->var.get()] = rhs->var.get();
    } else {
      if (CompareExpr(op->var, rhs->var) != 0) return;
    }
    if (CompareExpr(op->value, rhs->value) != 0) return;
    if (CompareExpr(op->body, rhs->body) != 0) return;
  }

  void VisitExpr_(const CallNode* op, const PrimExpr& other) final {
    const CallNode* rhs = other.as<CallNode>();
    if (CompareString(op->name, rhs->name)) return;
    if (CompareArray(op->args, rhs->args)) return;
    if (CompareValue(op->call_type, rhs->call_type) != 0) return;
    if (CompareNodeRef(op->func, rhs->func) != 0) return;
    if (CompareValue(op->value_index, rhs->value_index) != 0) return;
  }

  void VisitExpr_(const ReduceNode *op, const PrimExpr& other) final {
    const ReduceNode* rhs = other.as<ReduceNode>();
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

  void VisitExpr_(const IntImmNode *op, const PrimExpr& other) final {
    CompareValue(op->value, other.as<IntImmNode>()->value);
  }

  void VisitExpr_(const FloatImmNode *op, const PrimExpr& other) final {
    CompareValue(op->value, other.as<FloatImmNode>()->value);
  }

  void VisitExpr_(const StringImmNode *op, const PrimExpr& other) final {
    CompareString(op->value, other.as<StringImmNode>()->value);
  }

  void VisitExpr_(const CastNode *op, const PrimExpr& other) final {
    CompareExpr(op->value, other.as<CastNode>()->value);
  }

  void VisitExpr_(const NotNode *op, const PrimExpr& other) final {
    CompareExpr(op->a, other.as<NotNode>()->a);
  }

  void VisitExpr_(const SelectNode *op, const PrimExpr& other) final {
    const SelectNode* rhs = other.as<SelectNode>();
    if (CompareExpr(op->condition, rhs->condition) != 0) return;
    if (CompareExpr(op->true_value, rhs->true_value) != 0) return;
    if (CompareExpr(op->false_value, rhs->false_value) != 0) return;
  }

  void VisitExpr_(const RampNode *op, const PrimExpr& other) final {
    const RampNode* rhs = other.as<RampNode>();
    if (CompareExpr(op->base, rhs->base) != 0) return;
    if (CompareExpr(op->stride, rhs->stride) != 0) return;
    if (CompareValue(op->lanes, rhs->lanes) != 0) return;
  }

  void VisitExpr_(const BroadcastNode *op, const PrimExpr& other) final {
    const BroadcastNode* rhs = other.as<BroadcastNode>();
    if (CompareExpr(op->value, rhs->value) != 0) return;
    if (CompareValue(op->lanes, rhs->lanes) != 0) return;
  }

  void VisitExpr_(const ShuffleNode *op, const PrimExpr& other) final {
    const ShuffleNode* rhs = other.as<ShuffleNode>();
    if (CompareArray(op->vectors, rhs->vectors) != 0) return;
    if (CompareArray(op->indices, rhs->indices) != 0) return;
  }

  DEFINE_BIOP_EXPR_CMP_(AddNode)
  DEFINE_BIOP_EXPR_CMP_(SubNode)
  DEFINE_BIOP_EXPR_CMP_(MulNode)
  DEFINE_BIOP_EXPR_CMP_(DivNode)
  DEFINE_BIOP_EXPR_CMP_(ModNode)
  DEFINE_BIOP_EXPR_CMP_(FloorDivNode)
  DEFINE_BIOP_EXPR_CMP_(FloorModNode)
  DEFINE_BIOP_EXPR_CMP_(MinNode)
  DEFINE_BIOP_EXPR_CMP_(MaxNode)
  DEFINE_BIOP_EXPR_CMP_(EQNode)
  DEFINE_BIOP_EXPR_CMP_(NENode)
  DEFINE_BIOP_EXPR_CMP_(LTNode)
  DEFINE_BIOP_EXPR_CMP_(LENode)
  DEFINE_BIOP_EXPR_CMP_(GTNode)
  DEFINE_BIOP_EXPR_CMP_(GENode)
  DEFINE_BIOP_EXPR_CMP_(AndNode)
  DEFINE_BIOP_EXPR_CMP_(OrNode)

 private:
  int CompareExpr(const PrimExpr& lhs, const PrimExpr& rhs) {
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

  int CompareArray(const Array<PrimExpr>& lhs, const Array<PrimExpr>& rhs) {
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

  int CompareNodeRef(const ObjectRef& lhs, const ObjectRef& rhs) {
    if (order_ != 0) return order_;
    if (lhs.get() < rhs.get()) {
      order_ = -1; return order_;
    }
    if (lhs.get() > rhs.get()) {
      order_ = +1; return order_;
    }
    return order_;
  }

  int CompareType(const DataType& lhs, const DataType& rhs) {
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
  std::unordered_map<const VarNode*, const VarNode*> vmap_;
};


bool Equal(const Stmt& lhs, const Stmt& rhs) {
  return IRDeepCompare().Equal(lhs, rhs);
}

bool Equal(const PrimExpr& lhs, const PrimExpr& rhs) {
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

int Compare(const PrimExpr& lhs, const PrimExpr& rhs) {
  return IRDeepCompare().Compare(lhs, rhs);
}

}  // namespace tir
}  // namespace tvm
