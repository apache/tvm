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
 * \file remove_store_undef.cc
 * \brief Remove stores of tir::builtin::undef
 */
#include <tvm/runtime/registry.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

namespace tvm {
namespace tir {

class StoreUndefLocator : public StmtExprVisitor {
 public:
  static std::unordered_set<const BufferStoreNode*> Locate(Stmt stmt) {
    StoreUndefLocator locator;
    locator(std::move(stmt));
    return locator.undef_stores_;
  }

 private:
  StoreUndefLocator() = default;

  void VisitStmt_(const BufferStoreNode* op) final {
    bool stash_undef = false;
    std::swap(has_undef_, stash_undef);
    StmtExprVisitor::VisitExpr(op->value);
    std::swap(has_undef_, stash_undef);
    if (stash_undef) {
      ICHECK(SideEffect(op->value) <= CallEffectKind::kReadState)
          << "Error: T.undef() used in BufferStore expressions "
          << "must not have other side effects";
      undef_stores_.insert(op);
    }
  }

  void VisitExpr_(const BufferLoadNode* op) final {
    // This function left deliberately empty.  builtin::undef()
    // shouldn't occur in the indices of BufferLoad.  Avoiding
    // visiting the indices catches the builtin::undef in
    // ValidateAllUndefRemoved.
  }

  void VisitStmt_(const LetStmtNode* op) final {
    bool stash_undef = false;
    std::swap(has_undef_, stash_undef);
    StmtExprVisitor::VisitExpr(op->value);
    std::swap(has_undef_, stash_undef);
    if (stash_undef) {
      ICHECK(SideEffect(op->value) <= CallEffectKind::kReadState)
          << "Error: T.undef() used in Let expressions "
          << "must not have other side effects";
      var_bindings_with_undef_.insert(op->var.get());
    }

    StmtExprVisitor::VisitStmt(op->body);
  }

  void VisitExpr_(const VarNode* op) final {
    if (var_bindings_with_undef_.count(op)) {
      has_undef_ = true;
    }
  }

  void VisitExpr_(const CallNode* op) final {
    if (op->op.same_as(builtin::undef())) {
      has_undef_ = true;
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  bool has_undef_{false};

  std::unordered_set<const VarNode*> var_bindings_with_undef_;
  std::unordered_set<const BufferStoreNode*> undef_stores_;
};

// Remove any BufferStores whose value depends on T.undef
class StoreUndefRemover : public StmtExprMutator {
 public:
  static Stmt Apply(Stmt stmt) {
    auto to_remove = StoreUndefLocator::Locate(stmt);
    StoreUndefRemover mutator(to_remove);
    return mutator(std::move(stmt));
  }

 private:
  using Parent = StmtExprMutator;

  explicit StoreUndefRemover(const std::unordered_set<const BufferStoreNode*>& to_remove)
      : to_remove_(to_remove) {}

  Stmt VisitStmt_(const BufferStoreNode* op) final {
    if (to_remove_.count(op)) {
      return Evaluate(0);
    } else {
      return Parent::VisitStmt_(op);
    }
  }

  const std::unordered_set<const BufferStoreNode*>& to_remove_;
};

// Remove any BufferStores whose value depends on T.undef
class ContainsUndefChecker : public StmtExprVisitor {
 public:
  static bool Check(const Stmt& stmt) {
    ContainsUndefChecker checker;
    checker(stmt);
    return checker.contains_undef;
  }

 private:
  void VisitExpr_(const CallNode* op) final {
    if (op->op.same_as(builtin::undef())) {
      contains_undef = true;
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  bool contains_undef{false};
};

namespace transform {
Pass RemoveStoreUndefInternal() {
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    auto* n = f.CopyOnWrite();
    n->body = StoreUndefRemover::Apply(std::move(n->body));
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.RemoveStoreUndefInternal", {});
}

Pass ValidateAllUndefRemoved() {
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    bool contains_undef = ContainsUndefChecker::Check(f->body);
    ICHECK(!contains_undef) << "Expected removal of BufferStore containing builtin::undef() "
                            << "to remove all instances of builtin::undef().  "
                            << "Instead, result was"
                            << "\n"
                            << f;
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.ValidateAllUndefRemoved", {});
}

Pass RemoveStoreUndef() {
  return Sequential({RemoveStoreUndefInternal(), RemoveNoOp(), ValidateAllUndefRemoved()},
                    "tir.RemoveStoreUndef");
}

TVM_REGISTER_GLOBAL("tir.transform.RemoveStoreUndef").set_body_typed(RemoveStoreUndef);

}  // namespace transform

}  // namespace tir
}  // namespace tvm
