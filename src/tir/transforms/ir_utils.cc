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
#include <tvm/tir/transform.h>

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
    } else if (const auto* alloc = s.as<AllocateConstNode>()) {
      auto n = make_object<AllocateConstNode>(*alloc);
      ICHECK(is_no_op(n->body));
      n->body = body;
      body = Stmt(n);
    } else if (const auto* decl_buffer = s.as<DeclBufferNode>()) {
      auto n = make_object<DeclBufferNode>(*decl_buffer);
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
  PrimFunc VisitPrimFunc(PrimFunc func) {
    std::vector<ScopedRedefine> redefines;

    // Remap parameters, if they were used in another function
    auto params = func->params.Map([&](const tir::Var& var) -> tir::Var {
      if (defined_.count(var.get())) {
        const ScopedRedefine& redefine = redefines.emplace_back(this, var);
        return redefine.new_var;
      } else {
        defined_.insert(var.get());
        return var;
      }
    });

    // Remap implicitly defined buffer parameters
    {
      std::unordered_set<const VarNode*> defined_params;
      for (const auto& var : func->params) {
        defined_params.insert(var.get());
      }
      for (const auto& [var, buffer] : func->buffer_map) {
        static_cast<void>(var);  // gcc 7.x bug, https://gcc.gnu.org/bugzilla/show_bug.cgi?id=81767
        auto check_expr = [&](const PrimExpr& expr) {
          auto* var_ptr = expr.as<VarNode>();
          if (!var_ptr) return;
          if (defined_params.count(var_ptr)) return;

          if (defined_.count(var_ptr)) {
            auto var = GetRef<Var>(var_ptr);
            redefines.emplace_back(this, var);
          } else {
            defined_.insert(var_ptr);
          }
        };
        for (const auto& dim : buffer->shape) {
          check_expr(dim);
        }
        for (const auto& stride : buffer->strides) {
          check_expr(stride);
        }
        check_expr(buffer->elem_offset);
      }
    }

    // Update the buffer map, based on the redefined parameters
    auto buffer_map = [&]() {
      Map<Var, Buffer> buffer_map;
      bool made_change = false;
      for (const auto& [var, buffer] : func->buffer_map) {
        auto new_var = GetRemappedVar(var);
        if (defined_.count(buffer->data.get())) {
          redefines.emplace_back(this, buffer->data);
        } else {
          defined_.insert(buffer->data.get());
        }
        auto new_buf = GetRemappedBuffer(buffer);

        made_change = made_change || !var.same_as(new_var) || !buffer.same_as(new_buf);
        buffer_map.Set(new_var, new_buf);
      }
      if (made_change) {
        return buffer_map;
      } else {
        return func->buffer_map;
      }
    }();

    auto attrs = [&]() -> DictAttrs {
      if (!func->attrs.defined()) {
        return DictAttrs();
      }

      Map<String, ObjectRef> dict;
      bool made_change = false;

      for (const auto& [key, old_value] : func->attrs->dict) {
        auto value = old_value;
        if (auto* expr = value.as<PrimExprNode>()) {
          value = VisitExpr(GetRef<PrimExpr>(expr));
        } else if (auto* stmt = value.as<StmtNode>()) {
          value = VisitStmt(GetRef<Stmt>(stmt));
        }

        made_change = made_change || !value.same_as(old_value);
        dict.Set(key, value);
      }

      if (made_change) {
        return DictAttrs(dict);
      } else {
        return func->attrs;
      }
    }();

    auto body = VisitStmt(func->body);

    // If anything changed, update the returned function
    if (!params.same_as(func->params) || !buffer_map.same_as(func->buffer_map) ||
        !attrs.same_as(func->attrs) || !body.same_as(func->body)) {
      func = PrimFunc(params, body, func->ret_type, buffer_map, attrs);
    }

    // Pop the redefines in reverse order of creation
    while (redefines.size()) {
      redefines.pop_back();
    }
    function_scope_var_remap_.clear();
    return func;
  }

  PrimExpr VisitExpr_(const VarNode* op) final { return GetRemappedVar(GetRef<Var>(op)); }
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

  Var GetRemappedVar(Var var) {
    if (auto it = scope_.find(var.get()); it != scope_.end() && it->second.size()) {
      return it->second.back();
    } else if (auto it = function_scope_var_remap_.find(var.get());
               it != function_scope_var_remap_.end()) {
      return it->second;
    } else {
      return var;
    }
  }

  Buffer GetRemappedBuffer(Buffer buf) {
    // Determine the buffer var that should be in the updated buffer,
    // given the current scope.  If no redefines are present, then the
    // buffer var is unchanged.
    Var new_buffer_var = GetRemappedVar(buf->data);
    PrimExpr elem_offset = VisitExpr(buf->elem_offset);
    auto visit_expr = [this](const PrimExpr& expr) { return VisitExpr(expr); };
    Array<PrimExpr> shape = buf->shape.Map(visit_expr);
    Array<PrimExpr> strides = buf->strides.Map(visit_expr);

    // If no mapping is required, return the original buffer.
    if (new_buffer_var.same_as(buf->data) && elem_offset.same_as(buf->elem_offset) &&
        shape.same_as(buf->shape) && strides.same_as(buf->strides)) {
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
    Buffer new_buf = buf;
    {
      auto write_ptr = new_buf.CopyOnWrite();
      write_ptr->data = new_buffer_var;
      write_ptr->shape = shape;
      write_ptr->strides = strides;
      write_ptr->elem_offset = elem_offset;
    }
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
      return Allocate(redefine.new_var, op->dtype, op->extents, op->condition, op->body,
                      op->annotations);
    } else {
      defined_.insert(v.get());
      return StmtExprMutator::VisitStmt_(op);
    }
  }
  Stmt VisitStmt_(const AttrStmtNode* op) final {
    if (const IterVarNode* iter_var = op->node.as<IterVarNode>()) {
      Range dom = iter_var->dom;
      if (dom.defined()) {
        auto min = VisitExpr(dom->min);
        auto extent = VisitExpr(dom->extent);
        if (!min.same_as(iter_var->dom->min) || !extent.same_as(iter_var->dom->extent)) {
          dom = Range::FromMinExtent(min, extent);
        }
      }

      Var var = iter_var->var;
      if (auto it = function_scope_var_remap_.find(var.get());
          it != function_scope_var_remap_.end()) {
        var = it->second;
      } else if (defined_.count(var.get())) {
        Var new_var = [&]() {
          if (var->type_annotation.defined()) {
            return Var(var->name_hint, var->type_annotation);
          } else {
            return Var(var->name_hint, var->dtype);
          }
        }();

        function_scope_var_remap_.insert({var.get(), new_var});
        var = new_var;
      } else {
        function_scope_var_remap_.insert({var.get(), var});
        defined_.insert(var.get());
      }

      IterVar new_iter_var;
      if (dom.same_as(iter_var->dom) && var.same_as(iter_var->var)) {
        new_iter_var = GetRef<IterVar>(iter_var);
      } else {
        new_iter_var = IterVar(dom, var, iter_var->iter_type, iter_var->thread_tag, iter_var->span);
      }

      auto value = VisitExpr(op->value);
      auto body = VisitStmt(op->body);

      if (new_iter_var.get() == iter_var && body.same_as(op->body) && value.same_as(op->value)) {
        return GetRef<Stmt>(op);
      } else {
        return AttrStmt(new_iter_var, op->attr_key, value, body, iter_var->span);
      }

    } else if (const VarNode* v = op->node.as<VarNode>()) {
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
      if (parent) {
        parent->scope_[old_var.get()].pop_back();
        for (auto& kv : parent->buf_remap_) {
          std::vector<Buffer>& buffers = kv.second;
          if (buffers.size() && (buffers.back()->data.get() == new_var.get())) {
            buffers.pop_back();
          }
        }
      }
    }

    ScopedRedefine& operator=(const ScopedRedefine&) = delete;
    ScopedRedefine(const ScopedRedefine&) = delete;

    ScopedRedefine& operator=(ScopedRedefine&& other) {
      swap(other);
      return *this;
    }
    ScopedRedefine(ScopedRedefine&& other) { swap(other); }

    void swap(ScopedRedefine& other) {
      std::swap(parent, other.parent);
      std::swap(old_var, other.old_var);
      std::swap(new_var, other.new_var);
    }

    IRConvertSSA* parent{nullptr};
    Var old_var;
    Var new_var;
  };

  std::unordered_map<const VarNode*, std::vector<Var>> scope_;
  std::unordered_set<const VarNode*> defined_;
  std::unordered_map<const BufferNode*, std::vector<Buffer>> buf_remap_;

  std::unordered_map<const VarNode*, Var> function_scope_var_remap_;
};

Stmt ConvertSSA(Stmt stmt) { return IRConvertSSA()(std::move(stmt)); }

String GetPtrStorageScope(Var buffer_var) {
  const auto* ptr_type = buffer_var->type_annotation.as<PointerTypeNode>();
  ICHECK(ptr_type) << "The provided variable is not of pointer type";
  return ptr_type->storage_scope;
}

Array<PrimExpr> GetBufferAllocationShape(const Buffer& buffer) {
  Array<PrimExpr> alloc_shape = buffer->shape;
  if (buffer->strides.size()) {
    ICHECK_EQ(buffer->shape.size(), buffer->strides.size());
    for (size_t i = buffer->strides.size() - 1; i > 0; --i) {
      ICHECK(
          arith::Analyzer().CanProveEqual(floormod(buffer->strides[i - 1], buffer->strides[i]), 0));
      alloc_shape.Set(i, buffer->strides[i - 1] / buffer->strides[i]);
    }
  }
  return alloc_shape;
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

Optional<arith::IntConstraints> ConditionalBoundsContext::TrySolveCondition() {
  // extract equations and related vars from condition expression.
  // currently only extract simple integral equations which could be solvable.
  arith::Analyzer analyzer;
  PrimExpr condition = analyzer.Simplify(condition_);
  if (is_const_int(condition)) {
    return NullOpt;
  }
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
    return NullOpt;
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
  arith::IntConstraints result = arith::SolveInequalitiesToRange(constraint);
  if (!result->relations.empty()) {
    return NullOpt;
  }
  return std::move(result);
}

ConditionalBoundsContext::ConditionalBoundsContext(
    const PrimExpr& condition, std::unordered_map<const VarNode*, arith::IntSet>* relax_map,
    std::unordered_map<const VarNode*, arith::IntSet>* hint_map,
    std::vector<PrimExpr>* pending_conditions)
    : condition_(condition),
      relax_map_(relax_map),
      hint_map_(hint_map),
      pending_conditions_(pending_conditions),
      origin_pending_conditions_num_(pending_conditions->size()) {}

void ConditionalBoundsContext::EnterWithScope() {
  Optional<arith::IntConstraints> constraints = TrySolveCondition();
  if (!constraints.defined()) {
    // fail to process the condition, add to unresolved
    pending_conditions_->push_back(condition_);
    return;
  }
  // update solved var ranges
  for (const auto& kv : constraints.value()->ranges) {
    const VarNode* var = kv.first.get();
    arith::IntSet new_dom = arith::IntSet::FromRange(kv.second);
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
  pending_conditions_->resize(origin_pending_conditions_num_);
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

/*! \brief Collect storage alignment information from annotations. */
class StorageAlignCollector : public StmtVisitor {
 private:
  friend std::unordered_map<Var, StorageAlignAnnotation, ObjectPtrHash, ObjectPtrEqual>
  CollectStorageAlignAnnotation(const Stmt& body);

  /*! \brief For s-stir, the alignment annotations reside in block annotations. */
  void VisitStmt_(const BlockNode* op) final {
    auto it = op->annotations.find(attr::buffer_dim_align);
    if (it != op->annotations.end()) {
      auto storage_align_annotation = Downcast<StorageAlignAnnotation>((*it).second);
      for (const auto& storage_align_tuple : storage_align_annotation) {
        int buffer_index = storage_align_tuple[0]->value;
        const Buffer& buffer = op->writes[buffer_index]->buffer;
        storage_align_[buffer->data].push_back(storage_align_tuple);
      }
    }
    StmtVisitor::VisitStmt_(op);
  }

  /*! \brief For lowered tir, the alignment annotations reside in allocate annotations. */
  void VisitStmt_(const AllocateNode* op) final {
    auto it = op->annotations.find(attr::buffer_dim_align);
    if (it != op->annotations.end()) {
      auto storage_align_annotation = Downcast<StorageAlignAnnotation>((*it).second);
      for (const auto& storage_align_tuple : storage_align_annotation) {
        int buffer_index = storage_align_tuple[0]->value;
        // the first buffer idx info is meaningless for allocate
        // stmt and should set as negative intentionally.
        ICHECK_EQ(buffer_index, -1);
        storage_align_[op->buffer_var].push_back(storage_align_tuple);
      }
    }
    StmtVisitor::VisitStmt_(op);
  }

  /*! \brief The map from buffer var to its storage alignment information. */
  std::unordered_map<Var, StorageAlignAnnotation, ObjectPtrHash, ObjectPtrEqual> storage_align_;
};

std::unordered_map<Var, StorageAlignAnnotation, ObjectPtrHash, ObjectPtrEqual>
CollectStorageAlignAnnotation(const Stmt& body) {
  StorageAlignCollector collector;
  collector(body);
  return std::move(collector.storage_align_);
}

int Stoi(const std::string& str) {
  try {
    return std::stoi(str);
  } catch (std::invalid_argument& e) {
    LOG(FATAL) << "Cannot convert \"" << str << "\" to int";
    throw;
  }
}

std::pair<int32_t, int32_t> GetWmmaFragmentDimSize(const std::string& shape_str,
                                                   const std::string& scope) {
  size_t m, n, k;
  size_t last_pos = 0, pos = 0;
  pos = shape_str.find(", ", last_pos);
  m = Stoi(shape_str.substr(last_pos, pos - last_pos));
  last_pos = pos + 2;
  pos = shape_str.find(", ", last_pos);
  n = Stoi(shape_str.substr(last_pos, pos - last_pos));
  last_pos = pos + 2;
  k = Stoi(shape_str.substr(last_pos, shape_str.length() - last_pos));
  if (scope == "wmma.matrix_a") {
    return std::pair<int32_t, int32_t>(m, k);
  } else if (scope == "wmma.matrix_b") {
    return std::pair<int32_t, int32_t>(k, n);
  } else if (scope == "wmma.accumulator") {
    return std::pair<int32_t, int32_t>(m, n);
  }
  return std::pair<int32_t, int32_t>(0, 0);
}

std::optional<bool> IsHostFunc(const PrimFunc& func) {
  if (func->HasNonzeroAttr(tvm::tir::attr::kIsHostFunc)) {
    return true;
  } else if (auto target = func->GetAttr<Target>(tvm::attr::kTarget)) {
    return target.value()->HasKey("cpu");
  } else {
    return std::nullopt;
  }
}

namespace transform {
Pass ConvertSSA() {
  auto pass_func = [](IRModule mod, PassContext ctx) {
    tir::IRConvertSSA converter;
    Map<GlobalVar, BaseFunc> functions;
    bool made_change = false;
    for (auto [gvar, base_func] : mod->functions) {
      if (auto* ptr = base_func.as<tir::PrimFuncNode>()) {
        auto updated = converter.VisitPrimFunc(GetRef<tir::PrimFunc>(ptr));
        if (!updated.same_as(base_func)) {
          made_change = true;
          base_func = updated;
        }
      }
      functions.Set(gvar, base_func);
    }
    if (made_change) {
      mod.CopyOnWrite()->functions = std::move(functions);
    }
    return mod;
  };
  return tvm::transform::CreateModulePass(pass_func, 0, "tir.ConvertSSA", {});
}

TVM_REGISTER_GLOBAL("tir.transform.ConvertSSA").set_body_typed(ConvertSSA);

}  // namespace transform
}  // namespace tir
}  // namespace tvm
