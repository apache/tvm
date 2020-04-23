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
#include <tvm/runtime/registry.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>
#include <unordered_set>
#include "ir_util.h"
#include "../../arith/compute_expr.h"

namespace tvm {
namespace tir {

// If expression is touched by var.
class ExprTouched final : public StmtExprVisitor {
 public:
  explicit ExprTouched(const std::unordered_set<const VarNode*> &touched,
                       bool check_write)
      : touched_var_(touched), check_write_(check_write) {}

  void VisitExpr(const PrimExpr& n) final {
    // early stopping
    if (expr_touched_ && !check_write_) return;
    StmtExprVisitor::VisitExpr(n);
  }
    void VisitStmt(const Stmt& n) final {
    // early stopping
    if (expr_touched_ && !check_write_) return;
    StmtExprVisitor::VisitStmt(n);
  }
  void VisitExpr_(const LoadNode *op) final {
    HandleUseVar(op->buffer_var.get());
    StmtExprVisitor::VisitExpr_(op);
  }
  void VisitExpr_(const VarNode *op) final {
    HandleUseVar(op);
  }
  void VisitExpr_(const CallNode *op) final {
    if (op->is_intrinsic(intrinsic::tvm_access_ptr)) {
      const auto* rw_mask = op->args[4].as<IntImmNode>();
      const VarNode* buffer_var = op->args[1].as<VarNode>();
      CHECK(buffer_var);
      CHECK(rw_mask);
      // read
      if (rw_mask->value & 1) {
        HandleUseVar(buffer_var);
      }
      if (rw_mask->value & 2) {
        HandleWriteVar(buffer_var);
      }
      this->VisitExpr(op->args[2]);
    } else {
      StmtExprVisitor::VisitExpr_(op);
    }
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
  void HandleWriteVar(const VarNode* var) {
    write_vars_.push_back(var);
  }
  // the fields.
  bool expr_touched_{false};
  std::vector<const VarNode*> used_vars_;
  std::vector<const VarNode*> write_vars_;
  const std::unordered_set<const VarNode*>& touched_var_;
  bool check_write_;
};

// Analyze if the buffers are invariant to value of var
class VarTouchedAnalysis : public StmtVisitor {
 public:
  void VisitStmt_(const LetStmtNode* op) final {
    ExprTouched tc(touched_var_, false);
    tc(op->value);
    Record(op->var.get(), tc);
    this->VisitStmt(op->body);
  }
  void VisitStmt_(const StoreNode* op) final {
    ExprTouched tc(touched_var_, false);
    tc(op->value);
    tc(op->index);
    Record(op->buffer_var.get(), tc);
  }
  void VisitStmt_(const ForNode* op) final {
    ExprTouched tc(touched_var_, false);
    tc(op->min);
    tc(op->extent);
    Record(op->loop_var.get(), tc);
    this->VisitStmt(op->body);
  }
  // external function call
  void VisitStmt_(const EvaluateNode* op) final {
    ExprTouched tc(touched_var_, true);
    tc(op->value);
    for (const VarNode* var : tc.write_vars_) {
      Record(var, tc);
    }
  }
  void VisitStmt_(const AllocateNode* op) final {
    ExprTouched tc(touched_var_, false);
    for (size_t i = 0; i < op->extents.size(); ++i) {
      tc(op->extents[i]);
    }
    tc.VisitExpr(op->condition);
    Record(op->buffer_var.get(), tc);
    this->VisitStmt(op->body);
  }
  void Record(const VarNode* var,
              const ExprTouched& tc) {
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

  std::unordered_set<const VarNode*>
  TouchedVar(const Stmt& stmt,
             const VarNode* var) {
    touched_var_.insert(var);
    this->VisitStmt(stmt);
    // do a DFS to push affect around dependency.
    std::vector<const VarNode*> pending(
        touched_var_.begin(), touched_var_.end());
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
  // x -> all the buffers x read from
  std::unordered_map<const VarNode*,
                     std::vector<const VarNode*> > affect_;
};


// Inject virtual thread loop
// rewrite the buffer access pattern when necessary.
class VTInjector : public StmtExprMutator {
 public:
  // constructor
  VTInjector(Var var,
             int num_threads,
             const std::unordered_set<const VarNode*>& touched_var,
             bool allow_share)
      : var_(var), num_threads_(num_threads),
        touched_var_(touched_var), allow_share_(allow_share) {
  }
  // Inject VTLoop when needed.
  Stmt VisitStmt(const Stmt& s) final {
    CHECK(!visit_touched_var_);
    auto stmt = StmtExprMutator::VisitStmt(s);
    if (visit_touched_var_ || trigger_base_inject_) {
      if (!vt_loop_injected_)  {
        return InjectVTLoop(stmt, false);
      }
      visit_touched_var_ = false;
      trigger_base_inject_ = false;
    }
    return stmt;
  }
  // Variable
  PrimExpr VisitExpr_(const VarNode* op) final {
    CHECK(!alloc_remap_.count(op))
        << "Buffer address may get rewritten in virtual thread";
    if (touched_var_.count(op)) {
      visit_touched_var_ = true;
    }
    return GetRef<PrimExpr>(op);
  }
  PrimExpr RewriteIndex(PrimExpr index, PrimExpr alloc_extent) const {
    return index + var_ * alloc_extent;
  }
  // Load
  PrimExpr VisitExpr_(const LoadNode* op) final {
    PrimExpr expr = StmtExprMutator::VisitExpr_(op);
    op = expr.as<LoadNode>();
    if (touched_var_.count(op->buffer_var.get())) {
      visit_touched_var_ = true;
    }
    auto it = alloc_remap_.find(op->buffer_var.get());
    if (it != alloc_remap_.end()) {
      return LoadNode::make(op->dtype, op->buffer_var,
                        RewriteIndex(op->index, it->second),
                        op->predicate);
    } else {
      return expr;
    }
  }
  // Expression.
  PrimExpr VisitExpr_(const CallNode* op) final {
    if (op->is_intrinsic(intrinsic::tvm_access_ptr)) {
      CHECK_EQ(op->args.size(), 5U);
      DataType dtype = op->args[0].dtype();
      const VarNode* buffer = op->args[1].as<VarNode>();
      auto it = alloc_remap_.find(buffer);
      if (it == alloc_remap_.end()) return StmtExprMutator::VisitExpr_(op);
      visit_touched_var_ = true;
      PrimExpr offset = this->VisitExpr(op->args[2]);
      PrimExpr extent = this->VisitExpr(op->args[3]);
      PrimExpr stride =
          it->second / make_const(offset.dtype(), dtype.lanes());
      offset = stride * var_ + offset;
      return CallNode::make(
          op->dtype, op->name,
          {op->args[0], op->args[1], offset, extent, op->args[4]},
          op->call_type);
    } else if (op->is_intrinsic(intrinsic::tvm_context_id)) {
      return allow_share_ ? GetRef<PrimExpr>(op) : var_;
    } else {
      return StmtExprMutator::VisitExpr_(op);
    }
  }
  Stmt VisitStmt_(const EvaluateNode* op) final {
    trigger_base_inject_ = !allow_share_;
    return StmtExprMutator::VisitStmt_(op);
  }
  // Store
  Stmt VisitStmt_(const StoreNode* op) final {
    Stmt stmt = StmtExprMutator::VisitStmt_(op);
    op = stmt.as<StoreNode>();
    if (touched_var_.count(op->buffer_var.get())) {
      visit_touched_var_ = true;
    }
    trigger_base_inject_ = !allow_share_;
    auto it = alloc_remap_.find(op->buffer_var.get());
    if (it != alloc_remap_.end()) {
      return StoreNode::make(op->buffer_var,
                         op->value,
                         RewriteIndex(op->index, it->second),
                         op->predicate);
    } else {
      return stmt;
    }
  }
  // Attribute
  Stmt VisitStmt_(const AttrStmtNode* op) final {
    PrimExpr value = this->VisitExpr(op->value);
    if (visit_touched_var_ && !vt_loop_injected_) {
      return InjectVTLoop(GetRef<Stmt>(op), true);
    } else if (!allow_share_ && !vt_loop_injected_ &&
               (op->attr_key == attr::coproc_uop_scope ||
                op->attr_key == attr::coproc_scope)) {
      return InjectVTLoop(GetRef<Stmt>(op), true);
    } else {
      Stmt body = this->VisitStmt(op->body);
      if (value.same_as(op->value) &&
          body.same_as(op->body)) {
        return GetRef<Stmt>(op);
      } else {
        return AttrStmtNode::make(op->node, op->attr_key, value, body);
      }
    }
  }
  // LetStmt
  Stmt VisitStmt_(const LetStmtNode* op) final {
    PrimExpr value = this->VisitExpr(op->value);
    if (visit_touched_var_ && !vt_loop_injected_) {
      return InjectVTLoop(GetRef<Stmt>(op), true);
    }
    visit_touched_var_ = false;
    Stmt body = this->VisitStmt(op->body);
    if (value.same_as(op->value) &&
        body.same_as(op->body)) {
      return GetRef<Stmt>(op);
    } else {
      return LetStmtNode::make(op->var, value, body);
    }
  }
  // For
  Stmt VisitStmt_(const ForNode* op) final {
    CHECK(is_zero(op->min));
    PrimExpr extent = this->VisitExpr(op->extent);
    if (visit_touched_var_ && !vt_loop_injected_) {
      Stmt stmt = InjectVTLoop(GetRef<Stmt>(op), true);
      ++max_loop_depth_;
      return stmt;
    }
    visit_touched_var_ = false;
    Stmt body = this->VisitStmt(op->body);
    ++max_loop_depth_;
    if (extent.same_as(op->extent) &&
        body.same_as(op->body)) {
      return GetRef<Stmt>(op);
    } else {
      return ForNode::make(
          op->loop_var, op->min, extent, op->for_type, op->device_api, body);
    }
  }
  // IfThenElse
  Stmt VisitStmt_(const IfThenElseNode* op) final {
    PrimExpr condition = this->VisitExpr(op->condition);
    if (visit_touched_var_ && !vt_loop_injected_) {
      return InjectVTLoop(GetRef<Stmt>(op), true);
    }
    visit_touched_var_ = false;
    CHECK_EQ(max_loop_depth_, 0);
    Stmt then_case = this->VisitStmt(op->then_case);
    Stmt else_case;
    if (op->else_case.defined()) {
      int temp = max_loop_depth_;
      max_loop_depth_ = 0;
      else_case = this->VisitStmt(op->else_case);
      max_loop_depth_ = std::max(temp, max_loop_depth_);
    }
    if (condition.same_as(op->condition) &&
        then_case.same_as(op->then_case) &&
        else_case.same_as(op->else_case)) {
      return GetRef<Stmt>(op);
    } else {
      return IfThenElseNode::make(condition, then_case, else_case);
    }
  }

  // Seq
  Stmt VisitStmt_(const SeqStmtNode* op) final {
    CHECK_EQ(max_loop_depth_, 0);
    auto fmutate = [this](const Stmt& s) {
      int temp = max_loop_depth_;
      max_loop_depth_ = 0;
      Stmt ret = this->VisitStmt(s);
      max_loop_depth_ = std::max(max_loop_depth_, temp);
      return ret;
    };
    return StmtMutator::VisitSeqStmt_(op, false, fmutate);
  }
  // Allocate
  Stmt VisitStmt_(const AllocateNode* op) final {
    PrimExpr condition = this->VisitExpr(op->condition);
    if (visit_touched_var_ && !vt_loop_injected_) {
      return InjectVTLoop(GetRef<Stmt>(op), true);
    }

    bool changed = false;
    Array<PrimExpr> extents;
    for (size_t i = 0; i < op->extents.size(); i++) {
      PrimExpr new_ext = this->VisitExpr(op->extents[i]);
      if (visit_touched_var_ && !vt_loop_injected_) {
        return InjectVTLoop(GetRef<Stmt>(op), true);
      }
      if (!new_ext.same_as(op->extents[i])) changed = true;
      extents.push_back(new_ext);
    }
    visit_touched_var_ = false;

    Stmt body;
    // always rewrite if not allow sharing.
    if (touched_var_.count(op->buffer_var.get()) || !allow_share_) {
      // place v on highest dimension.
      PrimExpr stride = arith::ComputeReduce<MulNode>(
          op->extents, PrimExpr()) * op->dtype.lanes();
      Array<PrimExpr> other;
      other.push_back(make_const(op->extents[0].dtype(), num_threads_));
      for (PrimExpr e : extents) {
        other.push_back(e);
      }
      extents = other;
      changed = true;
      // mark this buffer get touched.
      alloc_remap_[op->buffer_var.get()] = stride;
      // Mutate the body.
      body = this->VisitStmt(op->body);
    } else {
      // Mutate the body.
      body = this->VisitStmt(op->body);
    }
    if (!changed &&
        body.same_as(op->body) &&
        condition.same_as(op->condition)) {
      return GetRef<Stmt>(op);
    } else {
      return AllocateNode::make(
          op->buffer_var, op->dtype,
          extents, condition, body);
    }
  }

  // inject vthread loop
  Stmt InjectVTLoop(Stmt stmt, bool before_mutation) {
    CHECK(!vt_loop_injected_);
    // reset the flags
    visit_touched_var_ = false;
    trigger_base_inject_ = false;
    vt_loop_injected_ = true;
    if (before_mutation) {
      stmt = this->VisitStmt(stmt);
    }
    // reset the flags after processing.
    vt_loop_injected_ = false;
    visit_touched_var_ = false;
    // only unroll if number of vthreads are small
    if (max_loop_depth_ == 0 && num_threads_ < 16) {
      // do unrolling if it is inside innermost content.
      Array<Stmt> seq;
      for (int i = 0; i < num_threads_; ++i) {
        seq.push_back(Substitute(stmt, {{var_, make_const(var_.dtype(), i)}}));
      }
      return SeqStmt::Flatten(seq);
    } else {
      // insert a for loop
      Var idx(var_->name_hint + ".s", var_->dtype);
      Map<Var, PrimExpr> values{{var_, idx}};
      stmt = Substitute(stmt, values);
      return ForNode::make(idx, make_zero(idx.dtype()),
                       make_const(idx.dtype(), num_threads_),
                       ForType::Serial, DeviceAPI::None, stmt);
    }
  }

 private:
  // vthread variable
  Var var_;
  // the threads/lanes
  int num_threads_;
  // whethe the loop is already injected.
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
  // The allocations that get touched -> extent
  std::unordered_map<const VarNode*, PrimExpr> alloc_remap_;
};


class VirtualThreadInjector : public StmtMutator {
 public:
  Stmt VisitStmt_(const AttrStmtNode* op) final {
    Stmt stmt = StmtMutator::VisitStmt_(op);
    op = stmt.as<AttrStmtNode>();
    if (op->attr_key == attr::virtual_thread) {
      IterVar iv = Downcast<IterVar>(op->node);
      bool allow_share = iv->thread_tag == "vthread";
      int nthread = static_cast<int>(op->value.as<IntImmNode>()->value);
      VarTouchedAnalysis vs;
      auto touched = vs.TouchedVar(op->body, iv->var.get());
      VTInjector injecter(iv->var, nthread, touched, allow_share);
      return injecter(op->body);
    } else {
      return stmt;
    }
  }

  Stmt VisitStmt_(const ProvideNode* op) final {
    LOG(FATAL) << "Need to call StorageFlatten first";
    return GetRef<Stmt>(op);
  }
};

Stmt InjectVirtualThread(Stmt stmt) {
  stmt = VirtualThreadInjector()(std::move(stmt));
  return ConvertSSA(std::move(stmt));
}

namespace transform {

Pass InjectVirtualThread() {
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    auto* n = f.CopyOnWrite();
    n->body = ConvertSSA(VirtualThreadInjector()(std::move(n->body)));
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.InjectVirtualThread", {});
}

TVM_REGISTER_GLOBAL("tir.transform.InjectVirtualThread")
.set_body_typed(InjectVirtualThread);

}  // namespace transform

}  // namespace tir
}  // namespace tvm
