/*!
 *  Copyright (c) 2017 by Contributors
 * \file inject_virtual_thread.cc
 */
#include <tvm/ir.h>
#include <tvm/ir_visitor.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_pass.h>
#include <unordered_set>
#include "../arithmetic/compute_expr.h"

namespace tvm {
namespace ir {

// If expression is touched by var.
class ExprTouched final : public IRVisitor {
 public:
  explicit ExprTouched(const std::unordered_set<const Variable*> &touched)
      : touched_var_(touched) {}
  void Visit(const NodeRef& n) final {
    // early stopping
    if (expr_touched_) return;
    IRVisitor::Visit(n);
  }
  void Visit_(const Load *op) final {
    HandleUseVar(op->buffer_var.get());
    IRVisitor::Visit_(op);
  }
  void Visit_(const Variable *op) final {
    HandleUseVar(op);
  }
  void HandleUseVar(const Variable* var) {
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
  // the fields.
  bool expr_touched_{false};
  std::vector<const Variable*> used_vars_;
  const std::unordered_set<const Variable*>& touched_var_;
};

// Analyze if the buffers are invariant to value of var
class VarTouchedAnalysis : public IRVisitor {
 public:
  void Visit_(const LetStmt *op) {
    ExprTouched tc(touched_var_);
    tc.Visit(op->value);
    Record(op->var.get(), tc);
    this->Visit(op->body);
  }
  void Visit_(const Store *op) {
    ExprTouched tc(touched_var_);
    tc.Visit(op->value);
    tc.Visit(op->index);
    Record(op->buffer_var.get(), tc);
  }
  void Visit_(const For *op) {
    ExprTouched tc(touched_var_);
    tc.Visit(op->min);
    tc.Visit(op->extent);
    Record(op->loop_var.get(), tc);
    this->Visit(op->body);
  }
  void Visit_(const Allocate *op) {
    ExprTouched tc(touched_var_);
    for (size_t i = 0; i < op->extents.size(); ++i) {
      tc.Visit(op->extents[i]);
    }
    tc.Visit(op->condition);
    if (op->new_expr.defined()) {
      tc.Visit(op->new_expr);
    }
    Record(op->buffer_var.get(), tc);
    this->Visit(op->body);
  }
  void Record(const Variable* var,
              const ExprTouched& tc) {
    if (touched_var_.count(var)) return;
    if (tc.expr_touched_) {
      touched_var_.insert(var);
    } else {
      for (const Variable* r : tc.used_vars_) {
        affect_[r].push_back(var);
      }
    }
  }

  std::unordered_set<const Variable*>
  TouchedVar(const Stmt& stmt,
             const Variable* var) {
    touched_var_.insert(var);
    this->Visit(stmt);
    // do a DFS to push affect around dependency.
    std::vector<const Variable*> pending(
        touched_var_.begin(), touched_var_.end());
    while (!pending.empty()) {
      const Variable* v = pending.back();
      pending.pop_back();
      for (const Variable* r : affect_[v]) {
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
  std::unordered_set<const Variable*> touched_var_;
  // x -> all the buffers x read from
  std::unordered_map<const Variable*,
                     std::vector<const Variable*> > affect_;
};


// Inject virtual thread loop
// rewrite the buffer access pattern when necessary.
class VTInjector : public IRMutator {
 public:
  using IRMutator::Mutate;
  // constructor
  VTInjector(Var var,
             int num_threads,
             std::unordered_set<const Variable*> touched_var)
      : var_(var), num_threads_(num_threads), touched_var_(touched_var) {
  }
  // Inject VTLoop when needed.
  Stmt Mutate(Stmt stmt) final {
    CHECK(!visit_touched_var_)
        << stmt->type_key() << stmt;
    stmt = IRMutator::Mutate(stmt);
    if (visit_touched_var_) {
      if (!vt_loop_injected_) return InjectVTLoop(stmt, false);
      visit_touched_var_ = false;
    }
    return stmt;
  }
  // Variable
  Expr Mutate_(const Variable *op, const Expr& e) final {
    if (touched_var_.count(op)) {
      visit_touched_var_ = true;
    }
    return e;
  }
  Expr RewriteIndex(Expr index, Expr alloc_extent) const {
    return index + var_ * alloc_extent;
  }
  // Load
  Expr Mutate_(const Load* op, const Expr& e) final {
    Expr expr = IRMutator::Mutate_(op, e);
    op = expr.as<Load>();
    if (touched_var_.count(op->buffer_var.get())) {
      visit_touched_var_ = true;
    }
    auto it = touched_alloc_.find(op->buffer_var.get());
    if (it != touched_alloc_.end()) {
      return Load::make(op->type, op->buffer_var,
                        RewriteIndex(op->index, it->second),
                        op->predicate);
    } else {
      return expr;
    }
  }
  // Store
  Stmt Mutate_(const Store* op, const Stmt& s) final {
    Stmt stmt = IRMutator::Mutate_(op, s);
    op = stmt.as<Store>();
    if (touched_var_.count(op->buffer_var.get())) {
      visit_touched_var_ = true;
    }
    auto it = touched_alloc_.find(op->buffer_var.get());
    if (it != touched_alloc_.end()) {
      return Store::make(op->buffer_var,
                         op->value,
                         RewriteIndex(op->index, it->second),
                         op->predicate);
    } else {
      return stmt;
    }
  }
  // Attribute
  Stmt Mutate_(const AttrStmt* op, const Stmt& s) final {
    Expr value = Mutate(op->value);
    if (visit_touched_var_) {
      return InjectVTLoop(s, true);
    } else {
      Stmt body = Mutate(op->body);
      if (value.same_as(op->value) &&
          body.same_as(op->body)) {
        return s;
      } else {
        return AttrStmt::make(op->node, op->attr_key, value, body);
      }
    }
  }
  // LetStmt
  Stmt Mutate_(const LetStmt* op, const Stmt& s) final {
    Expr value = this->Mutate(op->value);
    if (visit_touched_var_ && !vt_loop_injected_) {
      return InjectVTLoop(s, true);
    }
    visit_touched_var_ = false;
    Stmt body = Mutate(op->body);
    if (value.same_as(op->value) &&
        body.same_as(op->body)) {
      return s;
    } else {
      return LetStmt::make(op->var, value, body);
    }
  }
  // For
  Stmt Mutate_(const For* op, const Stmt& s) final {
    CHECK(is_zero(op->min));
    Expr extent = Mutate(op->extent);
    if (visit_touched_var_ && !vt_loop_injected_) {
      Stmt stmt = InjectVTLoop(s, true);
      ++max_loop_depth_;
      return stmt;
    }
    visit_touched_var_ = false;
    Stmt body = Mutate(op->body);
    ++max_loop_depth_;
    if (extent.same_as(op->extent) &&
        body.same_as(op->body)) {
      return s;
    } else {
      return For::make(
          op->loop_var, op->min, extent, op->for_type, op->device_api, body);
    }
  }
  // IfThenElse
  Stmt Mutate_(const IfThenElse* op, const Stmt& s) final {
    Expr condition = this->Mutate(op->condition);
    if (visit_touched_var_ && !vt_loop_injected_) {
      return InjectVTLoop(s, true);
    }
    visit_touched_var_ = false;
    CHECK_EQ(max_loop_depth_, 0);
    Stmt then_case = this->Mutate(op->then_case);
    Stmt else_case;
    if (else_case.defined()) {
      int temp = max_loop_depth_;
      max_loop_depth_ = 0;
      else_case = this->Mutate(op->else_case);
      max_loop_depth_ = std::max(temp, max_loop_depth_);
    }
    if (condition.same_as(op->condition) &&
        then_case.same_as(op->then_case) &&
        else_case.same_as(op->else_case)) {
      return s;
    } else {
      return IfThenElse::make(condition, then_case, else_case);
    }
  }
  // Block
  Stmt Mutate_(const Block* op, const Stmt& s) final {
    CHECK_EQ(max_loop_depth_, 0);
    Stmt first = this->Mutate(op->first);
    int temp = max_loop_depth_;
    max_loop_depth_ = 0;
    Stmt rest = this->Mutate(op->rest);
    max_loop_depth_ = std::max(max_loop_depth_, temp);
    if (first.same_as(op->first) &&
        rest.same_as(op->rest)) {
      return s;
    } else {
      return Block::make(first, rest);
    }
  }
  // Allocate
  Stmt Mutate_(const Allocate* op, const Stmt& s) final {
    if (op->new_expr.defined() && !vt_loop_injected_) {
      return InjectVTLoop(s, true);
    }
    Expr condition = Mutate(op->condition);
    if (visit_touched_var_ && !vt_loop_injected_) {
      return InjectVTLoop(s, true);
    }

    bool changed = false;
    Array<Expr> extents;
    for (size_t i = 0; i < op->extents.size(); i++) {
      Expr new_ext = Mutate(op->extents[i]);
      if (visit_touched_var_ && !vt_loop_injected_) {
        return InjectVTLoop(s, true);
      }
      if (!new_ext.same_as(op->extents[i])) changed = true;
      extents.push_back(new_ext);
    }
    visit_touched_var_ = false;

    Stmt body;
    if (touched_var_.count(op->buffer_var.get())) {
      // place v on highest dimension.
      Expr stride = extents[0];
      for (size_t i = 1; i < extents.size(); ++i) {
        stride = arith::ComputeExpr<Mul>(stride, extents[i]);
      }
      if (op->type.lanes() != 0) {
        stride = stride * op->type.lanes();
      }
      Array<Expr> other;
      other.push_back(num_threads_);
      for (Expr e : extents) {
        other.push_back(e);
      }
      extents = other;
      changed = true;
      // mark this buffer get touched.
      touched_alloc_[op->buffer_var.get()] = stride;
      // Mutate the body.
      body = Mutate(op->body);
    } else {
      // Mutate the body.
      body = Mutate(op->body);
    }
    if (!changed &&
        body.same_as(op->body) &&
        condition.same_as(op->condition)) {
      return s;
    } else {
      return Allocate::make(
          op->buffer_var, op->type,
          extents, condition, body,
          op->new_expr, op->free_function);
    }
  }

  // inject vthread loop
  Stmt InjectVTLoop(Stmt stmt, bool before_mutation) {
    CHECK(!vt_loop_injected_);
    // reset the flags
    visit_touched_var_ = false;
    vt_loop_injected_ = true;
    if (before_mutation) {
      stmt = this->Mutate(stmt);
    }
    // reset the flags after processing.
    vt_loop_injected_ = false;
    visit_touched_var_ = false;
    if (max_loop_depth_ == 0) {
      // do unrolling if it is inside innermost content.
      Stmt blk = Substitute(stmt, {{var_, make_zero(var_.type())}});
      for (int i = 1; i < num_threads_; ++i) {
        blk = Block::make(
            blk, Substitute(stmt, {{var_, make_const(var_.type(), i)}}));
      }
      return blk;
    } else {
      // insert a for loop
      Var idx(var_->name_hint + ".s", var_->type);
      stmt = Substitute(stmt, {{var_, idx}});
      return For::make(idx, 0, num_threads_,
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
  // the counter of loops in after mutation.
  int max_loop_depth_{0};
  // The variables that get touched.
  std::unordered_set<const Variable*> touched_var_;
  // The allocations that get touched -> extent
  std::unordered_map<const Variable*, Expr> touched_alloc_;
};


class VirtualThreadInjector : public IRMutator {
 public:
  Stmt Mutate_(const AttrStmt* op, const Stmt& s) final {
    Stmt stmt = IRMutator::Mutate_(op, s);
    op = stmt.as<AttrStmt>();
    if (op->attr_key == attr::virtual_thread) {
      IterVar iv(op->node.node_);
      int nthread = static_cast<int>(op->value.as<IntImm>()->value);
      VarTouchedAnalysis vs;
      auto touched = vs.TouchedVar(op->body, iv->var.get());
      VTInjector injecter(iv->var, nthread, touched);
      return injecter.Mutate(op->body);
    } else {
      return stmt;
    }
  }

  Stmt Mutate_(const Provide* op, const Stmt& s) final {
    LOG(FATAL) << "Need to call StorageFlatten first";
    return s;
  }
};

Stmt InjectVirtualThread(Stmt stmt) {
  stmt = VirtualThreadInjector().Mutate(stmt);
  return ConvertSSA(stmt);
}

}  // namespace ir
}  // namespace tvm
