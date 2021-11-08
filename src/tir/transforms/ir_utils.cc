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
 * \file ir_utils.cc
 * \brief Helper functions to construct and compose IR nodes.
 */
#include "ir_utils.h"

#include <tvm/arith/analyzer.h>
#include <tvm/arith/int_solver.h>
#include <tvm/tir/stmt_functor.h>

#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace tvm {
namespace tir {

Stmt MergeNest(const std::vector<Stmt>& nest, Stmt body) {
  // use reverse iteration
  for (auto ri = nest.rbegin(); ri != nest.rend(); ++ri) {
    Stmt s = *ri;
    if (const auto* for_ = s.as<ForNode>()) {
      auto n = make_object<ForNode>(*for_);
      ICHECK(is_no_op(n->body));
      n->body = body;
      body = Stmt(n);
    } else if (const auto* let = s.as<LetStmtNode>()) {
      auto n = make_object<LetStmtNode>(*let);
      ICHECK(is_no_op(n->body));
      n->body = body;
      body = Stmt(n);
    } else if (const auto* attr = s.as<AttrStmtNode>()) {
      auto n = make_object<AttrStmtNode>(*attr);
      ICHECK(is_no_op(n->body));
      n->body = body;
      body = Stmt(n);
    } else if (const auto* ite = s.as<IfThenElseNode>()) {
      auto n = make_object<IfThenElseNode>(*ite);
      ICHECK(is_no_op(n->then_case));
      ICHECK(!n->else_case.defined());
      n->then_case = body;
      body = Stmt(n);
    } else if (const auto* seq = s.as<SeqStmtNode>()) {
      auto n = make_object<SeqStmtNode>(*seq);
      ICHECK(n->size() != 0 && is_no_op(n->seq[n->size() - 1]));
      n->seq.Set(n->size() - 1, body);
      body = Stmt(n);
    } else if (const auto* assert_ = s.as<AssertStmtNode>()) {
      auto n = make_object<AssertStmtNode>(*assert_);
      ICHECK(is_no_op(n->body));
      n->body = body;
      body = Stmt(n);
    } else if (const auto* alloc = s.as<AllocateNode>()) {
      auto n = make_object<AllocateNode>(*alloc);
      ICHECK(is_no_op(n->body));
      n->body = body;
      body = Stmt(n);
    } else {
      LOG(FATAL) << "not supported nest type";
    }
  }
  return body;
}

Stmt MergeNest(const std::vector<std::vector<Stmt>>& nest, Stmt body) {
  for (auto ri = nest.rbegin(); ri != nest.rend(); ++ri) {
    body = MergeNest(*ri, body);
  }
  return body;
}

class IRConvertSSA final : public StmtExprMutator {
 public:
  PrimExpr VisitExpr_(const VarNode* op) final {
    if (scope_.count(op) && !scope_[op].empty()) {
      return scope_[op].back();
    } else {
      return GetRef<PrimExpr>(op);
    }
  }
  PrimExpr VisitExpr_(const LetNode* op) final {
    const Var& v = op->var;
    if (defined_.count(v.get())) {
      PrimExpr value = this->VisitExpr(op->value);
      Var new_var(v->name_hint, v.dtype());
      scope_[v.get()].push_back(new_var);
      PrimExpr body = this->VisitExpr(op->body);
      scope_[v.get()].pop_back();
      return Let(new_var, value, body);
    } else {
      defined_.insert(v.get());
      return StmtExprMutator::VisitExpr_(op);
    }
  }
  PrimExpr VisitExpr_(const LoadNode* op) final {
    PrimExpr expr = StmtExprMutator::VisitExpr_(op);
    op = expr.as<LoadNode>();
    const VarNode* v = op->buffer_var.get();
    if (scope_.count(v) && !scope_[v].empty()) {
      return Load(op->dtype, scope_[v].back(), op->index, op->predicate);
    } else {
      return expr;
    }
  }
  Stmt VisitStmt_(const StoreNode* op) final {
    Stmt stmt = StmtExprMutator::VisitStmt_(op);
    op = stmt.as<StoreNode>();
    const VarNode* v = op->buffer_var.get();
    if (scope_.count(v) && !scope_[v].empty()) {
      return Store(scope_[v].back(), op->value, op->index, op->predicate);
    } else {
      return stmt;
    }
  }
  Stmt VisitStmt_(const LetStmtNode* op) final {
    const Var& v = op->var;
    if (defined_.count(v.get())) {
      PrimExpr value = this->VisitExpr(op->value);
      Var new_var(v->name_hint, v.dtype());
      scope_[v.get()].push_back(new_var);
      Stmt body = this->VisitStmt(op->body);
      scope_[v.get()].pop_back();
      return LetStmt(new_var, value, body);
    } else {
      defined_.insert(v.get());
      return StmtExprMutator::VisitStmt_(op);
    }
  }
  Stmt VisitStmt_(const ForNode* op) final {
    const Var& v = op->loop_var;
    if (defined_.count(v.get())) {
      Var new_var(v->name_hint, v.dtype());
      scope_[v.get()].push_back(new_var);
      Stmt stmt = StmtExprMutator::VisitStmt_(op);
      scope_[v.get()].pop_back();
      op = stmt.as<ForNode>();
      return For(new_var, op->min, op->extent, op->kind, op->body, op->thread_binding,
                 op->annotations);
    } else {
      defined_.insert(v.get());
      return StmtExprMutator::VisitStmt_(op);
    }
  }
  Stmt VisitStmt_(const AllocateNode* op) final {
    const Var& v = op->buffer_var;
    if (defined_.count(v.get())) {
      Var new_var(v->name_hint, v->type_annotation);
      scope_[v.get()].push_back(new_var);
      Stmt stmt = StmtExprMutator::VisitStmt_(op);
      scope_[v.get()].pop_back();
      op = stmt.as<AllocateNode>();
      return Allocate(new_var, op->dtype, op->extents, op->condition, op->body);
    } else {
      defined_.insert(v.get());
      return StmtExprMutator::VisitStmt_(op);
    }
  }
  Stmt VisitStmt_(const AttrStmtNode* op) final {
    if (const VarNode* v = op->node.as<VarNode>()) {
      Stmt stmt = StmtExprMutator::VisitStmt_(op);
      op = stmt.as<AttrStmtNode>();
      if (scope_.count(v) && scope_[v].size() != 0) {
        return AttrStmt(scope_[v].back(), op->attr_key, op->value, op->body);
      } else {
        return stmt;
      }
    } else {
      return StmtExprMutator::VisitStmt_(op);
    }
  }

 private:
  std::unordered_map<const VarNode*, std::vector<Var>> scope_;
  std::unordered_set<const VarNode*> defined_;
};

Stmt ConvertSSA(Stmt stmt) { return IRConvertSSA()(std::move(stmt)); }

String GetPtrStorageScope(Var buffer_var) {
  const auto* ptr_type = buffer_var->type_annotation.as<PointerTypeNode>();
  ICHECK(ptr_type) << "The provided variable is not of pointer type";
  return ptr_type->storage_scope;
}

Array<PrimExpr> ConvertIndices(const MatchBufferRegion& match_buffer,
                               const Array<PrimExpr>& indices) {
  const Buffer& target = match_buffer->buffer;
  const BufferRegion& source = match_buffer->source;
  ICHECK_EQ(indices.size(), target->shape.size());

  arith::Analyzer analyzer;
  Array<PrimExpr> result;
  result.reserve(source->region.size());
  size_t offset = source->region.size() - indices.size();
  for (size_t i = 0; i < offset; ++i) {
    const Range& range = source->region[i];
    ICHECK(analyzer.CanProve(range->extent == 1));
    result.push_back(range->min);
  }
  for (size_t i = 0; i < indices.size(); ++i) {
    const Range& range = source->region[i + offset];
    const PrimExpr& index = indices[i];
    result.push_back(range->min + index);
  }
  return result;
}

Region ConvertRegion(const MatchBufferRegion& match_buffer, const Region& region) {
  const Buffer& target = match_buffer->buffer;
  const BufferRegion& source = match_buffer->source;
  ICHECK_EQ(region.size(), target->shape.size());

  arith::Analyzer analyzer;
  Region result;
  result.reserve(source->region.size());
  size_t offset = source->region.size() - region.size();
  for (size_t i = 0; i < offset; ++i) {
    const Range& source_range = source->region[i];
    ICHECK(analyzer.CanProve(source_range->extent == 1));
    result.push_back(Range::FromMinExtent(source_range->min, 1));
  }
  for (size_t i = 0; i < region.size(); ++i) {
    const Range& source_range = source->region[i + offset];
    const Range& target_range = region[i];
    result.push_back(
        Range::FromMinExtent(source_range->min + target_range->min, target_range->extent));
  }
  return result;
}

Bool IsFromLegacyTESchedule(PrimFunc f) {
  Optional<Bool> from_legacy_te_schedule = f->GetAttr("from_legacy_te_schedule", Bool(false));
  return from_legacy_te_schedule.value();
}

Map<Var, Range> ConditionalBoundsContext::GetVarBoundsFromCondition() {
  // extract equations and related vars from condition expression.
  // currently only extract simple integral equations which could be solvable.
  arith::Analyzer analyzer;
  PrimExpr condition = is_true_branch_ ? condition_ : analyzer.Simplify(!condition_);
  Array<PrimExpr> equations;
  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> var_set;
  std::function<void(const PrimExpr&)> fvisit = [&equations, &var_set, &fvisit](const PrimExpr& e) {
    if (e->IsInstance<GENode>() || e->IsInstance<GTNode>() || e->IsInstance<LENode>() ||
        e->IsInstance<LTNode>() || e->IsInstance<EQNode>() || e->IsInstance<NENode>()) {
      bool is_simple = true;
      std::vector<Var> cand_vars;
      PostOrderVisit(e, [&cand_vars, &is_simple, &e](const ObjectRef& obj) {
        if (obj.same_as(e)) {
          return;
        } else if (const VarNode* var = obj.as<VarNode>()) {
          if (var->dtype.is_int() || var->dtype.is_uint()) {
            cand_vars.push_back(GetRef<Var>(var));
          }
        } else {
          is_simple &= obj->IsInstance<AddNode>() || obj->IsInstance<SubNode>() ||
                       obj->IsInstance<MulNode>() || obj->IsInstance<FloorDivNode>() ||
                       obj->IsInstance<FloorModNode>() || obj->IsInstance<IntImmNode>();
        }
      });
      if (is_simple && !cand_vars.empty()) {
        for (const Var& var : cand_vars) var_set.insert(var);
        equations.push_back(Downcast<PrimExpr>(e));
      }
    } else if (e->IsInstance<AndNode>()) {
      And op = Downcast<And>(e);
      fvisit(op->a);
      fvisit(op->b);
    } else if (e->IsInstance<CallNode>()) {
      Call op = Downcast<Call>(e);
      if (op->op.same_as(builtin::likely())) {
        fvisit(op->args[0]);
      }
    }
  };
  fvisit(condition);
  if (equations.empty() || var_set.empty()) {
    return Map<Var, Range>();
  }
  // build dom ranges for related vars
  Array<Var> vars = Array<Var>(var_set.begin(), var_set.end());
  Map<Var, Range> ranges;
  for (const Var& v : vars) {
    auto it = dom_map_->find(v.get());
    if (it != dom_map_->end()) {
      const auto& int_set = it->second;
      ranges.Set(v, Range::FromMinExtent(int_set.min(),
                                         analyzer.Simplify(int_set.max() - int_set.min() + 1)));
    }
  }
  // solve constraints
  arith::IntConstraints constraint(vars, ranges, equations);
  auto result = arith::SolveInequalitiesToRange(constraint);
  return result->ranges;
}

ConditionalBoundsContext::ConditionalBoundsContext(
    const PrimExpr& condition, std::unordered_map<const VarNode*, arith::IntSet>* dom_map,
    bool is_true_branch)
    : condition_(condition), dom_map_(dom_map), is_true_branch_(is_true_branch) {}

void ConditionalBoundsContext::EnterWithScope() {
  for (const auto& p : GetVarBoundsFromCondition()) {
    const auto* var = p.first.get();
    auto it = dom_map_->find(var);
    if (it != dom_map_->end()) {
      origin_map_.emplace(var, it->second);
      it->second = arith::Intersect({it->second, arith::IntSet::FromRange(p.second)});
    }
  }
}

void ConditionalBoundsContext::ExitWithScope() {
  for (const auto& p : origin_map_) {
    (*dom_map_)[p.first] = p.second;
  }
}

}  // namespace tir
}  // namespace tvm
