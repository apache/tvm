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
 * \file inject_virtual_thread.cc
 */
#include <tvm/ffi/cast.h>
#include <tvm/ffi/extra/structural_mutate.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/prim/builtin.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/s_tir/stmt_functor.h>
#include <tvm/s_tir/transform.h>
#include <tvm/tirx/builtin.h>

#include <unordered_set>

#include "../../s_tir/ir/ir_mutator_with_analyzer.h"
#include "ir_utils.h"

namespace tvm {
namespace s_tir {
using namespace tvm::prim;
using namespace tvm::tirx;

namespace {

ffi::Optional<Var> GetBufferDataVar(const ffi::Any& data) {
  if (auto var = data.as<Var>()) {
    return var;
  }
  if (const auto* call = data.as<CallNode>();
      call && call->op.same_as(tirx::builtin::buffer_data()) && call->args.size() == 1) {
    return call->args[0].as<Var>();
  }
  return std::nullopt;
}

}  // namespace

// If expression is touched by var.
class ExprTouched final : public StmtExprVisitor {
 public:
  using StmtExprVisitor::Visit_;
  explicit ExprTouched(const std::unordered_set<const VarNode*>& touched) : touched_var_(touched) {}

  void Reset(bool check_write) {
    expr_touched_ = false;
    used_vars_.clear();
    write_vars_.clear();
    check_write_ = check_write;
  }

  ffi::Optional<VisitInterrupt> Visit(ffi::AnyView n) final {
    // early stopping
    if (expr_touched_ && !check_write_) return std::nullopt;
    return StmtExprVisitor::Visit(n);
  }

  ffi::Optional<VisitInterrupt> Visit_(const TensorLoadNode* op) final {
    HandleUseVar(op->source.as_or_throw<tvm::tirx::BufferVar>().get());
    return StmtExprVisitor::Visit_(op);
  }
  ffi::Optional<VisitInterrupt> Visit_(const VarNode* op) final {
    HandleUseVar(op);
    return std::nullopt;
  }
  ffi::Optional<VisitInterrupt> Visit_(const CallNode* op) final {
    if (op->op.same_as(tirx::builtin::masked_load()) ||
        op->op.same_as(tirx::builtin::masked_store())) {
      bool is_load = op->op.same_as(tirx::builtin::masked_load());
      const VarNode* buffer = op->args[0].as_or_throw<Var>().get();
      if (is_load) {
        HandleUseVar(buffer);
      } else {
        HandleWriteVar(buffer);
      }
      for (size_t i = 1; i < op->args.size(); ++i) {
        TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(op->args[i]));
      }
    } else if (op->op.same_as(tirx::builtin::tvm_access_ptr())) {
      const auto* rw_mask = op->args[4].as<IntImmNode>();
      auto buffer = GetBufferDataVar(op->args[1]);
      if (!buffer.has_value()) {
        // Nested access pointers are valid pointer expressions.  Visit the
        // inner pointer and this access's offset instead of assuming a raw
        // buffer Var at every level.
        TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(op->args[1]));
        return this->Visit(op->args[2].as_or_throw<PrimExpr>());
      }
      const VarNode* buffer_var = buffer.value().get();
      TVM_FFI_ICHECK(rw_mask);
      // read
      if (rw_mask->value & 1) {
        HandleUseVar(buffer_var);
      }
      if (rw_mask->value & 2) {
        HandleWriteVar(buffer_var);
      }
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(op->args[2].as_or_throw<PrimExpr>()));
    } else {
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(StmtExprVisitor::Visit_(op));
    }
    return std::nullopt;
  }
  void HandleUseVar(const VarNode* var) {
    auto it = touched_var_.find(var);
    if (it != touched_var_.end()) {
      expr_touched_ = true;
    }
    // rember the used vars
    // in case the var get touched later in a loop.
    if (!expr_touched_) {
      used_vars_.push_back(var);
    }
  }
  void HandleWriteVar(const VarNode* var) { write_vars_.push_back(var); }
  // the fields.
  bool expr_touched_{false};
  std::vector<const VarNode*> used_vars_;
  std::vector<const VarNode*> write_vars_;
  const std::unordered_set<const VarNode*>& touched_var_;
  bool check_write_{false};
};

// Analyze if the buffers are invariant to value of var
class VarTouchedAnalysis : public StmtExprVisitor {
 public:
  using StmtExprVisitor::Visit_;
  ffi::Optional<VisitInterrupt> Visit(ffi::AnyView value) override {
    if (value.as<ExprNode>()) return std::nullopt;
    return StmtExprVisitor::Visit(value);
  }
  ffi::Optional<VisitInterrupt> Visit_(const BindNode* op) final {
    expr_touched_->Reset(false);
    expr_touched_->Visit(op->value);
    Record(op->var.get(), *expr_touched_);
    return std::nullopt;
  }

  ffi::Optional<VisitInterrupt> Visit_(const BufferStoreNode* op) final {
    expr_touched_->Reset(false);
    expr_touched_->Visit(op->value);
    for (const auto& index : op->indices) {
      expr_touched_->Visit(index);
    }
    Record(op->buffer.get(), *expr_touched_);
    return std::nullopt;
  }
  ffi::Optional<VisitInterrupt> Visit_(const ForNode* op) final {
    expr_touched_->Reset(false);
    expr_touched_->Visit(op->min);
    expr_touched_->Visit(op->extent);
    Record(op->loop_var.get(), *expr_touched_);
    return this->Visit(op->body);
  }
  // external function call
  ffi::Optional<VisitInterrupt> Visit_(const EvaluateNode* op) final {
    expr_touched_->Reset(true);
    expr_touched_->Visit(op->value);
    for (const VarNode* var : expr_touched_->write_vars_) {
      Record(var, *expr_touched_);
    }
    return std::nullopt;
  }
  ffi::Optional<VisitInterrupt> Visit_(const AllocBufferNode* op) final {
    expr_touched_->Reset(false);
    for (size_t i = 0; i < op->buffer->shape.size(); ++i) {
      expr_touched_->Visit(op->buffer->shape[i]);
    }
    Record(op->buffer.get(), *expr_touched_);
    return StmtExprVisitor::Visit_(op);
  }
  void Record(const VarNode* var, const ExprTouched& tc) {
    if (touched_var_.count(var)) return;
    if (tc.expr_touched_) {
      touched_var_.insert(var);
    } else {
      for (const VarNode* r : tc.used_vars_) {
        if (r != var) {
          affect_[r].push_back(var);
        }
      }
    }
  }

  std::unordered_set<const VarNode*> TouchedVar(const Stmt& stmt, const VarNode* var) {
    touched_var_.insert(var);
    this->Visit(stmt);
    // do a DFS to push affect around dependency.
    std::vector<const VarNode*> pending(touched_var_.begin(), touched_var_.end());
    while (!pending.empty()) {
      const VarNode* v = pending.back();
      pending.pop_back();
      for (const VarNode* r : affect_[v]) {
        if (!touched_var_.count(r)) {
          touched_var_.insert(r);
          pending.push_back(r);
        }
      }
    }
    return std::move(touched_var_);
  }

 private:
  // Whether variable is touched by the thread variable.
  std::unordered_set<const VarNode*> touched_var_;
  ffi::ObjectPtr<ExprTouched> expr_touched_ = ffi::make_object<ExprTouched>(touched_var_);
  // x -> all the buffers x read from
  std::unordered_map<const VarNode*, std::vector<const VarNode*>> affect_;
};

// Inject virtual thread loop
// rewrite the buffer access pattern when necessary.
class VTInjector : public s_tir::IRMutatorWithAnalyzer {
 public:
  using s_tir::IRMutatorWithAnalyzer::Mutate;
  using s_tir::IRMutatorWithAnalyzer::Mutate_;

  // constructor
  VTInjector(sym::AnalyzerObj* analyzer, Var var, int num_threads,
             const std::unordered_set<const VarNode*>& touched_var, bool allow_share)
      : IRMutatorWithAnalyzer(analyzer),
        var_(var),
        num_threads_(num_threads),
        touched_var_(touched_var),
        allow_share_(allow_share) {}
  // Inject VTLoop when needed.
  UnchangedOr<ffi::Any> Mutate(ffi::AnyView value,
                               InplaceMode inplace_mode = InplaceMode::kDisallow) final {
    if (!value.as<StmtNode>()) return StmtExprMutator::Mutate(value, inplace_mode);
    TVM_FFI_ICHECK(!visit_touched_var_);
    auto result = StmtExprMutator::Mutate(value, inplace_mode);
    if (visit_touched_var_ || trigger_base_inject_) {
      if (!vt_loop_injected_) {
        Stmt stmt = std::move(result).ValueOrUnchanged(value).as_or_throw<Stmt>();
        return ffi::Any(InjectVTLoop(stmt, false));
      }
      visit_touched_var_ = false;
      trigger_base_inject_ = false;
    }
    return result;
  }
  // Variable
  UnchangedOr<Expr> Mutate_(const TensorRegionNode* op, InplaceMode inplace_mode) final {
    if (!op->source.as<BufferVar>()) {
      return StmtExprMutator::Mutate_(op, inplace_mode);
    }
    auto region = Mutate(op->region).as_or_throw<UnchangedOr<ffi::Array<Range>>>();
    if (region.UnchangedOrSameAs(op->region)) return ffi::Unchanged();
    TensorRegion node = ffi::GetRef<TensorRegion>(op);
    node.CopyOnWrite()->region = std::move(region).ValueUnchecked();
    return node;
  }

  UnchangedOr<Expr> Mutate_(const VarNode* op, InplaceMode inplace_mode) final {
    if (def_region_kind() == kTVMFFIDefRegionKindNone) {
      TVM_FFI_ICHECK(!alloc_remap_.count(op))
          << "BufferVar address may get rewritten in virtual thread";
      if (touched_var_.count(op)) {
        visit_touched_var_ = true;
      }
    }
    return StmtExprMutator::Mutate_(op, inplace_mode);
  }
  PrimExpr RewriteIndex(PrimExpr index, PrimExpr alloc_extent) const {
    return analyzer_->Simplify(index + var_.as_or_throw<PrimExpr>() * alloc_extent);
  }
  // Expression.
  UnchangedOr<Expr> Mutate_(const CallNode* op, InplaceMode inplace_mode) final {
    if (op->op.same_as(tirx::builtin::masked_load()) ||
        op->op.same_as(tirx::builtin::masked_store())) {
      bool is_load = op->op.same_as(tirx::builtin::masked_load());
      BufferVar buffer(op->args[0].as_or_throw<Var>());
      PrimExpr value;
      if (!is_load)
        value = Mutate(op->args[1]).ValueOrUnchanged(op->args[1]).as_or_throw<PrimExpr>();
      ffi::Array<PrimExpr> indices;
      for (size_t i = is_load ? 1 : 2; i + 1 < op->args.size(); ++i) {
        indices.push_back(
            Mutate(op->args[i]).ValueOrUnchanged(op->args[i]).as_or_throw<PrimExpr>());
      }
      PrimExpr predicate = Mutate(op->args[op->args.size() - 1])
                               .ValueOrUnchanged(op->args[op->args.size() - 1])
                               .as_or_throw<PrimExpr>();
      if (is_load) {
        TensorLoad access = VisitBufferAccess(BufferLoad(buffer, indices, op->span));
        ffi::Array<Expr> args{access->source.as_or_throw<tvm::tirx::BufferVar>().var()};
        for (const PrimExpr& index : access->indices) args.push_back(index);
        args.push_back(predicate);
        return Call(op->ty, op->op, args, op->attrs, op->ty_args, op->span);
      }
      BufferStore access = VisitBufferAccess(BufferStore(buffer, value, indices, op->span));
      ffi::Array<Expr> args{access->buffer.var(), access->value};
      for (const PrimExpr& index : access->indices) args.push_back(index);
      args.push_back(predicate);
      return Call(op->ty, op->op, args, op->attrs, op->ty_args, op->span);
    } else if (op->op.same_as(tirx::builtin::buffer_data())) {
      auto buffer = GetBufferDataVar(ffi::GetRef<Call>(op)).value();
      auto it = alloc_remap_.find(buffer.get());
      if (it == alloc_remap_.end()) {
        return StmtExprMutator::Mutate_(op, inplace_mode);
      }
      visit_touched_var_ = true;
      return GetRemappedBuffer(BufferVar(buffer), it->second).data();
    } else if (op->op.same_as(tirx::builtin::tvm_access_ptr())) {
      TVM_FFI_ICHECK_EQ(op->args.size(), 5U);
      PrimType dtype = op->args[0].as_or_throw<PrimExpr>().ty();
      auto buffer = GetBufferDataVar(op->args[1]);
      if (!buffer.has_value()) {
        return StmtExprMutator::Mutate_(op, inplace_mode);
      }
      auto it = alloc_remap_.find(buffer.value().get());
      if (it == alloc_remap_.end()) return StmtExprMutator::Mutate_(op, inplace_mode);
      visit_touched_var_ = true;
      PrimExpr offset = Mutate(op->args[2]).ValueOrUnchanged(op->args[2]).as_or_throw<PrimExpr>();
      PrimExpr extent = Mutate(op->args[3]).ValueOrUnchanged(op->args[3]).as_or_throw<PrimExpr>();
      PrimExpr stride = it->second / prim::MakeConst(offset.ty(), dtype.lanes());
      offset = RewriteIndex(offset, stride);
      Expr data = buffer.value()->ty.as<BufferTypeNode>()
                      ? GetRemappedBuffer(BufferVar(buffer.value()), it->second).data()
                      : op->args[1];

      return Call(op->ty, op->op, {op->args[0], data, offset, extent, op->args[4]});
    } else {
      return StmtExprMutator::Mutate_(op, inplace_mode);
    }
  }
  UnchangedOr<Stmt> Mutate_(const EvaluateNode* op, InplaceMode inplace_mode) final {
    trigger_base_inject_ = !allow_share_;
    return StmtExprMutator::Mutate_(op, inplace_mode);
  }
  // BufferLoad
  UnchangedOr<PrimExpr> Mutate_(const TensorLoadNode* op, InplaceMode inplace_mode) final {
    auto indices = Mutate(op->indices).as_or_throw<UnchangedOr<ffi::Array<PrimExpr>>>();
    TensorLoad node = ffi::GetRef<TensorLoad>(op);
    if (!indices.UnchangedOrSameAs(op->indices)) {
      node.CopyOnWrite()->indices = std::move(indices).ValueUnchecked();
    }
    return VisitBufferAccess(std::move(node));
  }
  // BufferStore
  UnchangedOr<Stmt> Mutate_(const BufferStoreNode* op, InplaceMode inplace_mode) final {
    auto value = Mutate(op->value);
    auto indices = Mutate(op->indices).as_or_throw<UnchangedOr<ffi::Array<PrimExpr>>>();
    BufferStore node = ffi::GetRef<BufferStore>(op);
    if (!value.UnchangedOrSameAs(op->value) || !indices.UnchangedOrSameAs(op->indices)) {
      auto* n = node.CopyOnWrite();
      n->value = std::move(value).ValueOrUnchanged(op->value);
      n->indices = std::move(indices).ValueOrUnchanged(op->indices);
    }
    trigger_base_inject_ = !allow_share_;
    return VisitBufferAccess(std::move(node));
  }

  template <typename Node>
  Node VisitBufferAccess(Node node) {
    if (touched_var_.count(node->buffer.get())) {
      visit_touched_var_ = true;
    }

    auto it = alloc_remap_.find(node->buffer.get());
    if (it != alloc_remap_.end()) {
      TVM_FFI_ICHECK_EQ(node->indices.size(), 1)
          << "InjectVirtualThread expects rewritten allocations to be flat memory.";
      auto writer = node.CopyOnWrite();
      writer->buffer = GetRemappedBuffer(node->buffer, it->second);
      writer->indices = {RewriteIndex(node->indices[0], it->second)};
    }

    return node;
  }

  TensorLoad VisitBufferAccess(TensorLoad node) {
    BufferVar buffer = node->source.as_or_throw<tvm::tirx::BufferVar>();
    if (touched_var_.count(buffer.get())) {
      visit_touched_var_ = true;
    }
    auto it = alloc_remap_.find(buffer.get());
    if (it == alloc_remap_.end()) return node;
    TVM_FFI_ICHECK_EQ(node->indices.size(), 1)
        << "InjectVirtualThread expects rewritten allocations to be flat memory.";
    auto* writer = node.CopyOnWrite();
    writer->source = GetRemappedBuffer(buffer, it->second);
    writer->indices = {RewriteIndex(node->indices[0], it->second)};
    return node;
  }

  BufferVar GetRemappedBuffer(BufferVar buf, PrimExpr alloc_extent) {
    if (auto replacement = VarRemapGet(buf).as<BufferVar>()) return replacement.value();
    BufferVar original = buf;

    TVM_FFI_ICHECK_EQ(buf->shape.size(), 1)
        << "Expected buffers being rewritten to already be flattened.";
    auto writer = CopyBufferType(buf);
    writer->shape = {buf->shape[0] * alloc_extent};
    buf = RebuildBufferVar(buf, std::move(writer));

    VarRemapSet(original, buf);
    return buf;
  }

  // Attribute
  UnchangedOr<Stmt> Mutate_(const AttrStmtNode* op, InplaceMode inplace_mode) final {
    auto value_result = this->Mutate(op->value, inplace_mode);
    bool value_unchanged = value_result.UnchangedOrSameAs(op->value);
    Expr value = std::move(value_result).ValueOrUnchanged(op->value);
    if (visit_touched_var_ && !vt_loop_injected_) {
      return InjectVTLoop(ffi::GetRef<Stmt>(op), true);
    } else {
      auto body_result = this->Mutate(op->body, inplace_mode);
      bool body_unchanged = body_result.UnchangedOrSameAs(op->body);
      Stmt body = std::move(body_result).ValueOrUnchanged(op->body);
      if (value_unchanged && body_unchanged) {
        return ffi::Unchanged();
      } else {
        return AttrStmt(op->node, op->attr_key, value, body);
      }
    }
  }
  // Bind
  UnchangedOr<Stmt> Mutate_(const BindNode* op, InplaceMode inplace_mode) final {
    auto value_result = this->Mutate(op->value, inplace_mode);
    bool value_unchanged = value_result.UnchangedOrSameAs(op->value);
    Expr value = std::move(value_result).ValueOrUnchanged(op->value);
    if (visit_touched_var_ && !vt_loop_injected_) {
      return InjectVTLoop(ffi::GetRef<Stmt>(op), true);
    }
    visit_touched_var_ = false;
    if (value_unchanged) {
      return ffi::Unchanged();
    } else {
      return Bind(op->var, value);
    }
  }
  // For
  UnchangedOr<Stmt> Mutate_(const ForNode* op, InplaceMode inplace_mode) final {
    TVM_FFI_ICHECK(is_zero(op->min));
    auto extent_result = this->Mutate(op->extent, inplace_mode);
    bool extent_unchanged = extent_result.UnchangedOrSameAs(op->extent);
    PrimExpr extent = std::move(extent_result).ValueOrUnchanged(op->extent);
    if (visit_touched_var_ && !vt_loop_injected_) {
      Stmt stmt = InjectVTLoop(ffi::GetRef<Stmt>(op), true);
      ++max_loop_depth_;
      return stmt;
    }
    visit_touched_var_ = false;
    auto body_result = this->Mutate(op->body, inplace_mode);
    bool body_unchanged = body_result.UnchangedOrSameAs(op->body);
    Stmt body = std::move(body_result).ValueOrUnchanged(op->body);
    ++max_loop_depth_;
    if (extent_unchanged && body_unchanged) {
      return ffi::Unchanged();
    } else {
      if (inplace_mode == InplaceMode::kAllow) {
        auto* writable = const_cast<ForNode*>(op);
        writable->extent = std::move(extent);
        writable->body = std::move(body);
        return ffi::Unchanged();
      } else {
        auto copy = ffi::make_object<ForNode>(*op);
        copy->extent = std::move(extent);
        copy->body = std::move(body);
        return For(std::move(copy));
      }
    }
  }
  // IfThenElse
  UnchangedOr<Stmt> Mutate_(const IfThenElseNode* op, InplaceMode inplace_mode) final {
    auto condition_result = this->Mutate(op->condition, inplace_mode);
    bool condition_unchanged = condition_result.UnchangedOrSameAs(op->condition);
    PrimExpr condition = std::move(condition_result).ValueOrUnchanged(op->condition);
    if (visit_touched_var_ && !vt_loop_injected_) {
      return InjectVTLoop(ffi::GetRef<Stmt>(op), true);
    }
    visit_touched_var_ = false;
    TVM_FFI_ICHECK_EQ(max_loop_depth_, 0);
    auto then_case_result = this->Mutate(op->then_case, inplace_mode);
    bool then_case_unchanged = then_case_result.UnchangedOrSameAs(op->then_case);
    Stmt then_case = std::move(then_case_result).ValueOrUnchanged(op->then_case);
    ffi::Optional<Stmt> else_case = std::nullopt;
    if (op->else_case) {
      int temp = max_loop_depth_;
      max_loop_depth_ = 0;
      else_case =
          this->Mutate(op->else_case.value(), inplace_mode).ValueOrUnchanged(op->else_case.value());
      max_loop_depth_ = std::max(temp, max_loop_depth_);
    }
    if (condition_unchanged && then_case_unchanged && else_case.same_as(op->else_case)) {
      return ffi::Unchanged();
    } else {
      return IfThenElse(condition, then_case, else_case);
    }
  }

  // While
  UnchangedOr<Stmt> Mutate_(const WhileNode* op, InplaceMode inplace_mode) final {
    // TODO(masahi): What should we do for While nodes?
    TVM_FFI_THROW(InternalError) << "WhileNode in InjectVirtualThread not supported yet";
    TVM_FFI_UNREACHABLE();
  }

  // Seq
  // When a Bind child triggers VT injection, we group the Bind together with
  // all remaining siblings (which may reference the bound variable) and wrap
  // them as a single unit in the VT loop.  This preserves the semantics that
  // With flat Bind (no body), a Bind whose value touches vt_var must be
  // grouped with all remaining siblings and wrapped in a VT loop together.
  // This preserves the scoping that was implicit when Bind carried a body.
  UnchangedOr<Stmt> Mutate_(const SeqStmtNode* op, InplaceMode inplace_mode) final {
    TVM_FFI_ICHECK_EQ(max_loop_depth_, 0);
    ffi::Array<Stmt> new_seq;
    bool changed = false;
    for (size_t i = 0; i < op->seq.size(); ++i) {
      int temp = max_loop_depth_;
      max_loop_depth_ = 0;
      // For Bind children, pre-check if the value touches vt_var before
      // visiting.  If so, group the Bind with all remaining siblings and
      // wrap the group with InjectVTLoop.
      if (const auto* bind = op->seq[i].as<BindNode>(); bind && !vt_loop_injected_) {
        // Visit just the value expression to probe for vt_var dependency.
        TVM_FFI_ICHECK(!visit_touched_var_);
        // This is a probe: the original Bind is revisited when forming the group.
        this->Mutate(bind->value, InplaceMode::kDisallow);
        if (visit_touched_var_) {
          // Reset flag (InjectVTLoop will handle it).
          visit_touched_var_ = false;
          // Gather the original Bind + all remaining original siblings.
          ffi::Array<Stmt> group;
          for (size_t j = i; j < op->seq.size(); ++j) {
            group.push_back(op->seq[j]);
          }
          Stmt grouped = group.size() == 1 ? group[0] : SeqStmt(group);
          // before_mutation=true: InjectVTLoop will re-visit the entire group
          // with vt_loop_injected_=true, properly substituting vt_var.
          Stmt wrapped = InjectVTLoop(grouped, true);
          new_seq.push_back(wrapped);
          max_loop_depth_ = std::max(max_loop_depth_, temp);
          changed = true;
          // All remaining siblings consumed — exit loop.
          break;
        }
        // Value did not touch vt_var.  Reset and visit the Bind normally.
        visit_touched_var_ = false;
      }
      // Non-Bind child or Bind that does not touch vt_var: visit normally.
      auto child_result = this->Mutate(op->seq[i]);
      bool child_unchanged = child_result.UnchangedOrSameAs(op->seq[i]);
      Stmt child = std::move(child_result).ValueOrUnchanged(op->seq[i]).as_or_throw<Stmt>();
      max_loop_depth_ = std::max(max_loop_depth_, temp);
      if (!child_unchanged) changed = true;
      new_seq.push_back(child);
    }
    if (!changed) return ffi::Unchanged();
    if (new_seq.size() == 1) return new_seq[0];
    return SeqStmt(new_seq);
  }
  // Allocate
  // AllocBuffer
  UnchangedOr<Stmt> Mutate_(const AllocBufferNode* op, InplaceMode inplace_mode) final {
    AllocBuffer node = ffi::GetRef<AllocBuffer>(op);

    ffi::Array<PrimExpr> shape = op->buffer->shape.Map([this](const PrimExpr& s) {
      // Keep the retained allocation and its buffer type unchanged.
      return Mutate(s, InplaceMode::kDisallow).ValueOrUnchanged(s);
    });

    if (visit_touched_var_ && !vt_loop_injected_) {
      return InjectVTLoop(ffi::GetRef<Stmt>(op), true);
    }

    visit_touched_var_ = false;

    if (touched_var_.count(op->buffer.get()) || !allow_share_) {
      TVM_FFI_ICHECK_EQ(shape.size(), 1)
          << "InjectVirtualThread expects rewritten allocations to be flat memory.";
      PrimExpr stride = shape[0];
      shape = {stride * num_threads_};
      alloc_remap_[op->buffer.get()] = stride;
    }

    if (shape.same_as(op->buffer->shape)) {
      return ffi::Unchanged();
    } else {
      auto type = CopyBufferType(op->buffer);
      type->shape = shape;
      BufferVar new_buffer = RebuildBufferVar(op->buffer, std::move(type));
      VarRemapSet(op->buffer, new_buffer);
      return AllocBuffer(new_buffer, op->annotations);
    }
  }

  // inject vthread loop
  Stmt InjectVTLoop(Stmt stmt, bool before_mutation) {
    TVM_FFI_ICHECK(!vt_loop_injected_);
    // reset the flags
    visit_touched_var_ = false;
    trigger_base_inject_ = false;
    vt_loop_injected_ = true;
    if (before_mutation) {
      stmt = this->Mutate(stmt, InplaceMode::kDisallow).ValueOrUnchanged(stmt);
    }
    // reset the flags after processing.
    vt_loop_injected_ = false;
    visit_touched_var_ = false;
    // only unroll if number of vthreads are small
    if (max_loop_depth_ == 0 && num_threads_ < 16) {
      // do unrolling if it is inside innermost content.
      ffi::Array<Stmt> seq;
      for (int i = 0; i < num_threads_; ++i) {
        PrimType var_ty = var_->ty.as_or_throw<PrimType>();
        auto f_substitute = [this, i,
                             var_ty](const Var& var) -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
          if (var.same_as(var_)) return ffi::Any(IntImm(var_ty, i));
          return ffi::Unchanged();
        };
        seq.push_back(
            ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(stmt, f_substitute).as_or_throw<Stmt>());
      }
      return SeqStmt::Flatten(seq);
    } else {
      // insert a for loop
      Var idx(var_->name + ".s", var_->ty);
      auto f_substitute = [this,
                           &idx](const Var& var) -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
        if (var.same_as(var_)) return ffi::Any(idx.as_or_throw<PrimExpr>());
        return ffi::Unchanged();
      };
      stmt = ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(stmt, f_substitute).as_or_throw<Stmt>();
      PrimType idx_dtype = idx->ty.as_or_throw<PrimType>();
      return For(idx.as_or_throw<PrimVar>(), IntImm(idx_dtype, 0),
                 prim::MakeConst(idx_dtype, num_threads_), ForKind::kSerial, stmt);
    }
  }

 private:
  // vthread variable
  Var var_;
  // the threads/lanes
  int num_threads_;
  // whether the loop is already injected.
  bool vt_loop_injected_{false};
  // whether current expression get touched.
  bool visit_touched_var_{false};
  // Trigger base stmt
  bool trigger_base_inject_{false};
  // the counter of loops in after mutation.
  int max_loop_depth_{0};
  // The variables that get touched.
  const std::unordered_set<const VarNode*>& touched_var_;
  // Whether allow shareding.
  bool allow_share_;
  /* \brief The allocations that get touched -> extent
   *
   * Maps from the buffer_var of an allocate node to the original
   * extent of the allocation.  Used when rewriting the indices of
   * BufferLoad/BufferStore.
   */
  std::unordered_map<const VarNode*, PrimExpr> alloc_remap_;
  /*! \brief Map of buffers that are modified.
   *
   * Buffers allocated or written to within the virtual thread loop
   * must have one copy per virtual thread.  This is done by enlarging
   * the allocated buffer size, then modifying the indices at which
   * each virtual thread accesses the buffer.
   */
};

class VirtualThreadInjector : public s_tir::IRMutatorWithAnalyzer {
 public:
  using s_tir::IRMutatorWithAnalyzer::Mutate;
  using s_tir::IRMutatorWithAnalyzer::Mutate_;

  using IRMutatorWithAnalyzer::IRMutatorWithAnalyzer;

  UnchangedOr<Stmt> Mutate_(const AttrStmtNode* op, InplaceMode inplace_mode) final {
    Stmt stmt = StmtExprMutator::Mutate_(op, inplace_mode).ValueOrUnchanged(ffi::GetRef<Stmt>(op));
    op = stmt.as<AttrStmtNode>();
    if (op->attr_key == s_tir::attr::virtual_thread) {
      IterVar iv = op->node.as_or_throw<IterVar>();
      bool allow_share = std::string(iv->thread_tag).substr(0, 7) == "vthread";
      int nthread = op->value.as<IntImmNode>()->value.as<int>().value();
      auto vs = ffi::make_object<VarTouchedAnalysis>();
      auto touched = vs->TouchedVar(op->body, iv->var.get());
      auto injector =
          ffi::make_object<VTInjector>(analyzer_, iv->var, nthread, touched, allow_share);
      return injector->Mutate(op->body).ValueOrUnchanged(op->body);
    } else {
      return stmt;
    }
  }
};

namespace transform {

Pass InjectVirtualThread() {
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    auto* n = f.CopyOnWrite();

    sym::Analyzer analyzer;

    n->body = ffi::make_object<VirtualThreadInjector>(analyzer)
                  ->Mutate(n->body, InplaceMode::kAllow)
                  .ValueOrUnchanged(std::move(n->body));
    n->body = s_tir::ConvertSSA(std::move(n->body));
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "s_tir.InjectVirtualThread", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("s_tir.transform.InjectVirtualThread", InjectVirtualThread);
}

}  // namespace transform

}  // namespace s_tir
}  // namespace tvm
