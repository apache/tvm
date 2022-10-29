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
      ICHECK(!n->else_case);
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
      ScopedRedefine redefine(this, v);
      PrimExpr body = this->VisitExpr(op->body);
      return Let(redefine.new_var, value, body);
    } else {
      defined_.insert(v.get());
      return StmtExprMutator::VisitExpr_(op);
    }
  }

  PrimExpr VisitExpr_(const LoadNode* op) final {
    LOG(FATAL) << "Unexpected use of deprecated LoadNode.  Please use BufferLoadNode instead.";
    return PrimExpr();
  }

  Stmt VisitStmt_(const StoreNode* op) final {
    LOG(FATAL) << "Unexpected use of deprecated StoreNode.  Please use BufferStoreNode instead.";
    return Stmt();
  }

  PrimExpr VisitExpr_(const BufferLoadNode* op) final {
    auto node = Downcast<BufferLoad>(StmtExprMutator::VisitExpr_(op));
    auto output = VisitBufferAccess(std::move(node));
    return std::move(output);
  }

  Stmt VisitStmt_(const BufferStoreNode* op) final {
    auto node = Downcast<BufferStore>(StmtExprMutator::VisitStmt_(op));
    auto output = VisitBufferAccess(std::move(node));
    return std::move(output);
  }

  Stmt VisitStmt_(const DeclBufferNode* op) final {
    DeclBuffer decl = Downcast<DeclBuffer>(StmtExprMutator::VisitStmt_(op));
    Buffer new_buffer = GetRemappedBuffer(decl->buffer);
    if (!new_buffer.same_as(decl->buffer)) {
      decl.CopyOnWrite()->buffer = std::move(new_buffer);
    }
    return std::move(decl);
  }

  template <typename Node>
  Node VisitBufferAccess(Node node) {
    Buffer new_buf = GetRemappedBuffer(node->buffer);
    if (!new_buf.same_as(node->buffer)) {
      auto writer = node.CopyOnWrite();
      writer->buffer = new_buf;
    }

    return node;
  }

  Buffer GetRemappedBuffer(Buffer buf) {
    // Determine the buffer var that should be in the updated buffer,
    // given the current scope.  If no redefines are present, then the
    // buffer var is unchanged.
    Var new_buffer_var = buf->data;
    auto var_it = scope_.find(buf->data.get());
    if (var_it != scope_.end() && !var_it->second.empty()) {
      new_buffer_var = var_it->second.back();
    }

    // If no mapping is required, return the original buffer.
    if (new_buffer_var.same_as(buf->data)) {
      return buf;
    }

    // If the current scope already has a mapping of this buffer, use
    // the mapped buffer.
    auto key = buf.get();
    std::vector<Buffer>& buffers = buf_remap_[key];
    if (buffers.size() && buffers.back()->data.same_as(new_buffer_var)) {
      return buffers.back();
    }

    // Otherwise, make and return a new buffer object that uses the
    // new buffer, pushing it onto the scoped stack of existing
    // buffers.  This will be popped when the new_buffer_var
    // redefinition is popped.
    Buffer new_buf(new_buffer_var, buf->dtype, buf->shape, buf->strides, buf->elem_offset,
                   buf->name, buf->data_alignment, buf->offset_factor, buf->buffer_type,
                   buf->axis_separators, buf->span);
    buffers.push_back(new_buf);
    return new_buf;
  }

  Stmt VisitStmt_(const LetStmtNode* op) final {
    const Var& v = op->var;
    if (defined_.count(v.get())) {
      PrimExpr value = this->VisitExpr(op->value);
      ScopedRedefine redefine(this, v);
      Stmt body = this->VisitStmt(op->body);
      return LetStmt(redefine.new_var, value, body);
    } else {
      defined_.insert(v.get());
      return StmtExprMutator::VisitStmt_(op);
    }
  }
  Stmt VisitStmt_(const ForNode* op) final {
    const Var& v = op->loop_var;
    if (defined_.count(v.get())) {
      ScopedRedefine redefine(this, v);
      Stmt stmt = StmtExprMutator::VisitStmt_(op);
      op = stmt.as<ForNode>();
      return For(redefine.new_var, op->min, op->extent, op->kind, op->body, op->thread_binding,
                 op->annotations);
    } else {
      defined_.insert(v.get());
      return StmtExprMutator::VisitStmt_(op);
    }
  }
  Stmt VisitStmt_(const AllocateNode* op) final {
    const Var& v = op->buffer_var;
    if (defined_.count(v.get())) {
      ScopedRedefine redefine(this, v);
      Stmt stmt = StmtExprMutator::VisitStmt_(op);
      op = stmt.as<AllocateNode>();
      return Allocate(redefine.new_var, op->dtype, op->extents, op->condition, op->body);
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
  struct ScopedRedefine {
    ScopedRedefine(IRConvertSSA* parent, Var old_var) : parent(parent), old_var(old_var) {
      if (old_var->type_annotation.defined()) {
        new_var = Var(old_var->name_hint, old_var->type_annotation);
      } else {
        new_var = Var(old_var->name_hint, old_var->dtype);
      }
      parent->scope_[old_var.get()].push_back(new_var);
    }

    ~ScopedRedefine() {
      parent->scope_[old_var.get()].pop_back();
      for (auto& kv : parent->buf_remap_) {
        std::vector<Buffer>& buffers = kv.second;
        if (buffers.size() && (buffers.back()->data.get() == new_var.get())) {
          buffers.pop_back();
        }
      }
    }

    IRConvertSSA* parent;
    Var old_var;
    Var new_var;
  };

  std::unordered_map<const VarNode*, std::vector<Var>> scope_;
  std::unordered_set<const VarNode*> defined_;
  std::unordered_map<const BufferNode*, std::vector<Buffer>> buf_remap_;
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
  Array<Var> vars;
  std::function<void(const PrimExpr&)> fvisit = [&equations, &vars, &fvisit](const PrimExpr& e) {
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
        for (const Var& new_var : cand_vars) {
          if (!std::any_of(vars.begin(), vars.end(),
                           [&new_var](const Var& v) { return v.same_as(new_var); })) {
            vars.push_back(new_var);
          }
        }
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
  if (equations.empty() || vars.empty()) {
    return Map<Var, Range>();
  }
  // build dom ranges for related vars
  Map<Var, Range> ranges;
  for (const Var& v : vars) {
    arith::IntSet dom;
    auto relax_it = relax_map_->find(v.get());
    if (relax_it != relax_map_->end()) {
      dom = relax_it->second;
    } else {
      auto hint_it = hint_map_->find(v.get());
      if (hint_it != hint_map_->end()) {
        dom = hint_it->second;
      }
    }
    if (dom.defined()) {
      ranges.Set(v, Range::FromMinExtent(dom.min(), analyzer.Simplify(dom.max() - dom.min() + 1)));
    }
  }
  // solve constraints
  arith::IntConstraints constraint(vars, ranges, equations);
  auto result = arith::SolveInequalitiesToRange(constraint);
  return result->ranges;
}

ConditionalBoundsContext::ConditionalBoundsContext(
    const PrimExpr& condition, std::unordered_map<const VarNode*, arith::IntSet>* relax_map,
    std::unordered_map<const VarNode*, arith::IntSet>* hint_map, bool is_true_branch)
    : condition_(condition),
      relax_map_(relax_map),
      hint_map_(hint_map),
      is_true_branch_(is_true_branch) {}

void ConditionalBoundsContext::EnterWithScope() {
  for (const auto& p : GetVarBoundsFromCondition()) {
    const auto* var = p.first.get();
    arith::IntSet new_dom = arith::IntSet::FromRange(p.second);
    auto relax_it = relax_map_->find(var);
    if (relax_it != relax_map_->end()) {
      // this is a bound for relaxed var
      origin_map_.emplace(var, relax_it->second);
      relax_it->second = arith::Intersect({relax_it->second, new_dom});
    } else {
      // this is a bound for free var
      auto hint_it = hint_map_->find(var);
      if (hint_it != hint_map_->end()) {
        origin_map_.emplace(var, hint_it->second);
        hint_it->second = arith::Intersect({hint_it->second, new_dom});
      } else {
        origin_map_.emplace(var, arith::IntSet::Nothing());
        hint_map_->insert(hint_it, {var, new_dom});
      }
    }
  }
}

void ConditionalBoundsContext::ExitWithScope() {
  for (const auto& p : origin_map_) {
    const auto* var = p.first;
    auto relax_it = relax_map_->find(var);
    if (relax_it != relax_map_->end()) {
      // recover bound for relaxed var
      relax_it->second = p.second;
    } else {
      // recover bound for free var
      auto hint_it = hint_map_->find(var);
      ICHECK(hint_it != hint_map_->end());
      if (p.second.IsNothing()) {
        hint_map_->erase(hint_it);
      } else {
        hint_it->second = p.second;
      }
    }
  }
}

std::pair<PrimExpr, PrimExpr> GetAsyncWaitAttributes(const AttrStmtNode* op) {
  ICHECK(op && op->attr_key == tir::attr::async_wait_queue_scope);
  auto inner = op->body.as<AttrStmtNode>();
  ICHECK(inner && inner->attr_key == tir::attr::async_wait_inflight_count);
  return std::make_pair(op->value, inner->value);
}

}  // namespace tir
}  // namespace tvm
