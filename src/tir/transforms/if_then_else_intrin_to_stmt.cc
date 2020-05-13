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
 * \file if_then_else_intrin_to_stmt.cc
 */
#include <tvm/runtime/registry.h>
#include <tvm/tir/transform.h>
#include <tvm/tir/stmt_functor.h>
#include <vector>
#include "../../runtime/thread_storage_scope.h"

namespace tvm {
namespace tir {

/*
 * Base class of IfThenElseFinder and SyncChecker
 * Visit sub-expressions of a statement only, not to visit its
 * sub-statements
 */
template <class RetType>
class SubExprVisitor : public ExprVisitor {
 public:
  RetType Check(const IfThenElseNode *op) {
    operator()(op->condition);
    return ret_;
  }

  RetType Check(const LetStmtNode *op) {
    operator()(op->value);
    return ret_;
  }

  RetType Check(const ForNode *op) {
    operator()(op->min);
    operator()(op->extent);
    return ret_;
  }

  RetType Check(const AllocateNode *op) {
    for (const auto &item : op->extents) {
      operator()(item);
    }
    operator()(op->condition);
    return ret_;
  }

  RetType Check(const StoreNode *op) {
    operator()(op->value);
    operator()(op->index);
    operator()(op->predicate);
    return ret_;
  }

  RetType Check(const ProvideNode *op) {
    operator()(op->value);
    for (const auto &item : op->args) {
      operator()(item);
    }
    return ret_;
  }

  RetType Check(const RealizeNode *op) {
    operator()(op->condition);
    return ret_;
  }

  RetType Check(const EvaluateNode *op) {
    operator()(op->value);
    return ret_;
  }

 protected:
  explicit SubExprVisitor(const RetType &ret)
    : ret_(ret) {}

  RetType ret_;
};

typedef Map<Var, ObjectRef> VarSet;  // Actually a set. The value is not used.

/*
 * Check if there are any Load or Store node of shared memory.
 * If so, we should skip transforming the statment to avoid
 * thread synchronizing issues
 */
class SyncChecker : public SubExprVisitor<bool> {
 public:
  explicit SyncChecker(const VarSet &shared_vars)
    : SubExprVisitor(false), shared_vars_(shared_vars) {}

  template <class Node>
  bool Check(const Node *op) {
    return SubExprVisitor::Check(op);
  }

  bool Check(const StoreNode *op) {
    if (shared_vars_.count(op->buffer_var)) {
      return true;
    }
    return SubExprVisitor::Check(op);
  }

 protected:
  void VisitExpr_(const LoadNode *op) override {
    if (shared_vars_.count(op->buffer_var)) {
      ret_ = true;
    }
    SubExprVisitor::VisitExpr_(op);
  }

 private:
  const VarSet &shared_vars_;
};

/*
 * Find the condition of the out-most if_then_else intrinsic and
 * count the total number of if_then_else intrinsics
 */
class IfThenElseFinder : public SubExprVisitor<std::pair<PrimExpr, int>> {
 public:
  IfThenElseFinder()
    : SubExprVisitor(std::pair<PrimExpr, int>(nullptr, 0)) {}

 protected:
  void VisitExpr_(const CallNode *op) override {
    if (op->is_intrinsic(tir::intrinsic::tvm_if_then_else)) {
      if (!ret_.first.defined())
        ret_.first = op->args[0];  // condition
      ret_.second++;  // cnt
    }
    SubExprVisitor::VisitExpr_(op);
  }
};

// Remove the out-most IfThenElse intrinsic and leave one branch
class IfThenElseOutMostRemover : public ExprMutator {
 public:
  explicit IfThenElseOutMostRemover(bool branch_to_keep)
      : branch_to_keep_(branch_to_keep) {}

  Stmt Mutate(const IfThenElseNode *op) {
    auto condition = operator()(std::move(op->condition));
    return IfThenElseNode::make(condition, op->then_case, op->else_case);
  }

  Stmt Mutate(const LetStmtNode *op) {
    auto value = operator()(std::move(op->value));
    return LetStmtNode::make(op->var, value, op->body);
  }

  Stmt Mutate(const ForNode *op) {
    auto min = operator()(std::move(op->min));
    auto extent = operator()(std::move(op->extent));
    return ForNode::make(op->loop_var, min, extent,
        op->for_type, op->device_api, op->body);
  }

  Stmt Mutate(const AllocateNode *op) {
    std::vector<PrimExpr> extents(op->extents.size());
    for (size_t i = 0, n = op->extents.size(); i < n; i++) {
      extents[i] = operator()(op->extents[i]);
    }
    auto condition = operator()(std::move(op->condition));
    return AllocateNode::make(op->buffer_var, op->dtype,
        extents, condition, op->body);
  }

  Stmt Mutate(const StoreNode *op) {
    auto value = operator()(std::move(op->value));
    auto index = operator()(std::move(op->index));
    auto predicate = operator()(std::move(op->predicate));
    return StoreNode::make(op->buffer_var, value, index, predicate);
  }

  Stmt Mutate(const ProvideNode *op) {
    auto value = operator()(std::move(op->value));
    std::vector<PrimExpr> args(op->args.size());
    for (size_t i = 0, n = op->args.size(); i < n; i++) {
      args[i] = operator()(op->args[i]);
    }
    return ProvideNode::make(op->func, op->value_index, value, args);
  }

  Stmt Mutate(const RealizeNode *op) {
    auto condition = operator()(std::move(op->condition));
    return RealizeNode::make(op->func, op->value_index, op->dtype,
        op->bounds, condition, op->body);
  }

  Stmt Mutate(const EvaluateNode *op) {
    auto value = operator()(std::move(op->value));
    return EvaluateNode::make(value);
  }

 protected:
  PrimExpr VisitExpr_(const CallNode *op) override {
    if (!found_ && op->is_intrinsic(tir::intrinsic::tvm_if_then_else)) {
      found_ = true;
      if (branch_to_keep_) {
        return op->args[1];
      } else {
        return op->args[2];
      }
      // no recursion
    } else {
      return ExprMutator::VisitExpr_(op);
    }
  }

 private:
  bool branch_to_keep_;
  bool found_ = false;
};

/*
 * Replace if_then_else intrinsics to be If statements
 *
 * Algorithm:
 * 1. For each statement.
 * 2. Stop if there are too many tvm_if_then_else intrinsics.
 * 3. Find the out-most tvm_if_then_else intrinsic in it.
 * 4. Put the statement into a new IfThenElse statement, and remove the out-most tvm_if_then_else intrinsic.
 * 5. Recurse until no more tvm_if_then_else.
 */
class IfThenElseReplacer : public StmtMutator {
 public:
  explicit IfThenElseReplacer(int max_cascading_intrin)
      : max_cascading_intrin_(max_cascading_intrin) {}

 protected:
  Stmt VisitStmt_(const AttrStmtNode *op) override {
    using runtime::StorageRank;
    using runtime::StorageScope;

    if (op->attr_key == attr::storage_scope) {
      StorageScope scope = StorageScope::make(op->value.as<StringImmNode>()->value);
      if (scope.rank == StorageRank::kShared) {
        shared_vars_.Set(Downcast<Var>(op->node), ObjectRef{});
      }
    }
    return StmtMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const IfThenElseNode *op) override {
    return Replace_(op);
  }

  Stmt VisitStmt_(const LetStmtNode *op) override {
    return Replace_(op);
  }

  Stmt VisitStmt_(const ForNode *op) override {
    return Replace_(op);
  }

  Stmt VisitStmt_(const AllocateNode *op) override {
    return Replace_(op);
  }

  Stmt VisitStmt_(const StoreNode *op) override {
    return Replace_(op);
  }

  Stmt VisitStmt_(const ProvideNode *op) override {
    return Replace_(op);
  }

  Stmt VisitStmt_(const RealizeNode *op) override {
    return Replace_(op);
  }

  Stmt VisitStmt_(const EvaluateNode *op) override {
    return Replace_(op);
  }

 private:
  template <class Node>
  Stmt Replace_(const Node *op) {
    if (SyncChecker(shared_vars_).Check(op)) {
      return StmtMutator::VisitStmt_(op);
    }
    PrimExpr condition;
    int cnt;
    std::tie(condition, cnt) = IfThenElseFinder().Check(op);
    if (cnt > 0 && cnt <= max_cascading_intrin_) {
      IfThenElseOutMostRemover keep_true(true);
      IfThenElseOutMostRemover keep_false(false);
      auto true_branch = keep_true.Mutate(CopyOnWrite(op).get());
      auto false_branch = keep_false.Mutate(CopyOnWrite(op).get());
      Stmt ret = IfThenElseNode::make(condition,
          Downcast<Stmt>(true_branch), Downcast<Stmt>(false_branch));
      return VisitStmt_(ret.as<IfThenElseNode>());
      // Note that we transform one if_then_else at a time, so we are using
      // pre-order recursion
    } else {
      return StmtMutator::VisitStmt_(op);
    }
  }

 private:
  int max_cascading_intrin_;
  VarSet shared_vars_;
};


namespace transform {

Pass IfThenElseIntrinToStmt(int max_cascading_intrin) {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    auto* n = f.CopyOnWrite();
    n->body = IfThenElseReplacer(max_cascading_intrin)(std::move(n->body));
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.IfThenElseIntrinToStmt", {});
}

TVM_REGISTER_GLOBAL("tir.transform.IfThenElseIntrinToStmt")
.set_body_typed(IfThenElseIntrinToStmt);

}  // namespace transform

}  // namespace tir
}  // namespace tvm
