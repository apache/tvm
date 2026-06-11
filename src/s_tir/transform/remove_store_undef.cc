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
 * \brief Remove stores of tirx::builtin::undef
 */
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/s_tir/transform.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>

namespace tvm {
namespace s_tir {
using namespace tvm::tirx;

struct UndefInfo {
  std::unordered_set<const BufferStoreNode*> undef_stores;
  std::unordered_set<const VarNode*> undef_bind_vars;
};

class StoreUndefLocator : public StmtExprVisitor {
 public:
  static UndefInfo Locate(Stmt stmt) {
    StoreUndefLocator locator;
    locator(std::move(stmt));
    return {locator.undef_stores_, locator.var_bindings_with_undef_};
  }

 private:
  StoreUndefLocator() = default;

  void VisitStmt_(const BufferStoreNode* op) final {
    // Check the value for undef.
    bool stash_undef = false;
    std::swap(has_undef_, stash_undef);
    StmtExprVisitor::VisitExpr(op->value);
    std::swap(has_undef_, stash_undef);
    if (stash_undef) {
      TVM_FFI_ICHECK(SideEffect(op->value) <= CallEffectKind::kReadState)
          << "Error: T.undef() used in BufferStore expressions "
          << "must not have other side effects";
      undef_stores_.insert(op);
    }

    // Check indices for undef.  Undef in buffer indices is always an
    // error (there is no valid lowering).  With flat Bind, we must
    // check indices eagerly because the Bind node is a sibling rather
    // than an ancestor and may be removed before post-validation.
    bool idx_undef = false;
    std::swap(has_undef_, idx_undef);
    for (const auto& idx : op->indices) {
      StmtExprVisitor::VisitExpr(idx);
    }
    std::swap(has_undef_, idx_undef);
    TVM_FFI_ICHECK(!idx_undef) << "Error: T.undef() may not be used in buffer indices";
  }

  void VisitExpr_(const BufferLoadNode* op) final {
    // Check indices for undef.  Undef in buffer indices is always an error.
    bool idx_undef = false;
    std::swap(has_undef_, idx_undef);
    for (const auto& idx : op->indices) {
      StmtExprVisitor::VisitExpr(idx);
    }
    std::swap(has_undef_, idx_undef);
    TVM_FFI_ICHECK(!idx_undef) << "Error: T.undef() may not be used in buffer indices";
  }

  void VisitStmt_(const BindNode* op) final {
    bool stash_undef = false;
    std::swap(has_undef_, stash_undef);
    StmtExprVisitor::VisitExpr(op->value);
    std::swap(has_undef_, stash_undef);
    if (stash_undef) {
      TVM_FFI_ICHECK(SideEffect(op->value) <= CallEffectKind::kReadState)
          << "Error: T.undef() used in Let expressions "
          << "must not have other side effects";
      var_bindings_with_undef_.insert(op->var.get());
    }
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

// Remove BufferStores whose value depends on T.undef, and also
// remove Bind nodes whose value contains undef.  Undef in buffer
// indices is already caught eagerly in the locator phase.
class StoreUndefRemover : public StmtExprMutator {
 public:
  static Stmt Apply(Stmt stmt) {
    auto info = StoreUndefLocator::Locate(stmt);
    StoreUndefRemover mutator(info);
    return mutator(std::move(stmt));
  }

 private:
  using Parent = StmtExprMutator;

  explicit StoreUndefRemover(const UndefInfo& info)
      : stores_to_remove_(info.undef_stores), bind_vars_to_remove_(info.undef_bind_vars) {}

  Stmt VisitStmt_(const BufferStoreNode* op) final {
    if (stores_to_remove_.count(op)) {
      return Evaluate(0);
    } else {
      return Parent::VisitStmt_(op);
    }
  }

  Stmt VisitStmt_(const BindNode* op) final {
    if (bind_vars_to_remove_.count(op->var.get())) {
      return Evaluate(0);
    } else {
      return Parent::VisitStmt_(op);
    }
  }

  const std::unordered_set<const BufferStoreNode*>& stores_to_remove_;
  const std::unordered_set<const VarNode*>& bind_vars_to_remove_;
};

// Check that no builtin::undef() remains in the IR.
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
  auto pass_func = [](PrimFunc f, IRModule m, tvm::transform::PassContext ctx) {
    auto* n = f.CopyOnWrite();
    n->body = StoreUndefRemover::Apply(std::move(n->body));
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "s_tir.RemoveStoreUndefInternal", {});
}

Pass ValidateAllUndefRemoved() {
  auto pass_func = [](PrimFunc f, IRModule m, tvm::transform::PassContext ctx) {
    bool contains_undef = ContainsUndefChecker::Check(f->body);
    TVM_FFI_ICHECK(!contains_undef)
        << "Expected removal of BufferStore containing builtin::undef() "
        << "to remove all instances of builtin::undef().  "
        << "Instead, result was"
        << "\n"
        << f;
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "s_tir.ValidateAllUndefRemoved", {});
}

Pass RemoveStoreUndef() {
  return tvm::transform::Sequential(
      {RemoveStoreUndefInternal(), tirx::transform::RemoveNoOp(), ValidateAllUndefRemoved()},
      "s_tir.RemoveStoreUndef");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("s_tir.transform.RemoveStoreUndef", RemoveStoreUndef);
}

}  // namespace transform

}  // namespace s_tir
}  // namespace tvm
