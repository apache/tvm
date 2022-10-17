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
 * \brief Infer TensorCore metadata from tensor intrinsic.
 * \file tensorcore_fragment.cc
 */
#include <tvm/runtime/registry.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <unordered_map>
#include <unordered_set>

#include "../../runtime/thread_storage_scope.h"
#include "ir_utils.h"
#include "storage_access.h"

namespace tvm {
namespace tir {

// Get fragment information from tensor intrinsics
class FragmentGetter : public StmtExprVisitor {
 public:
  void VisitExpr_(const CallNode* op) final {
    StmtExprVisitor::VisitExpr_(op);

    if (op->op.same_as(builtin::tvm_load_matrix_sync()) ||
        op->op.same_as(builtin::tvm_store_matrix_sync())) {
      // Get shape and layout information from load and store intrinsic
      ICHECK_EQ(op->args.size(), 8U);
      const VarNode* buffer_var = op->args[0].as<VarNode>();
      ICHECK(buffer_var);
      // Get shape
      const IntImmNode* m = op->args[1].as<IntImmNode>();
      const IntImmNode* n = op->args[2].as<IntImmNode>();
      const IntImmNode* k = op->args[3].as<IntImmNode>();
      const StringImmNode* layout = op->args[7].as<StringImmNode>();
      ICHECK(m);
      ICHECK(n);
      ICHECK(k);
      ICHECK(layout);

      std::string scope = GetPtrStorageScope(GetRef<Var>(buffer_var));
      if (fragments.count(buffer_var)) {
        // check if the fragment has met before
        FragmentInfo info = fragments[buffer_var];
        ICHECK_EQ(m->value, info.m);
        ICHECK_EQ(n->value, info.n);
        ICHECK_EQ(k->value, info.k);
        if (scope == "wmma.matrix_a" || scope == "wmma.matrix_b") {
          ICHECK_EQ(layout->value, info.layout);
        }
      } else {
        // store metadata
        FragmentInfo info;
        if (scope == "wmma.matrix_a" || scope == "wmma.matrix_b") {
          info = FragmentInfo(m->value, n->value, k->value, layout->value, scope);
        } else if (scope == "wmma.accumulator") {
          info = FragmentInfo(m->value, n->value, k->value, "", scope);
        }
        fragments[buffer_var] = info;
      }
    } else if (op->op.same_as(builtin::tvm_fill_fragment())) {
      // Get shape information from fill intrinsic
      ICHECK_EQ(op->args.size(), 6U);
      const VarNode* buffer_var = op->args[0].as<VarNode>();
      ICHECK(buffer_var);
      // Get shape
      const IntImmNode* m = op->args[1].as<IntImmNode>();
      const IntImmNode* n = op->args[2].as<IntImmNode>();
      const IntImmNode* k = op->args[3].as<IntImmNode>();
      ICHECK(m);
      ICHECK(n);
      ICHECK(k);

      std::string scope = GetPtrStorageScope(GetRef<Var>(buffer_var));
      if (fragments.count(buffer_var)) {
        FragmentInfo info = fragments[buffer_var];
        ICHECK_EQ(m->value, info.m);
        ICHECK_EQ(n->value, info.n);
        ICHECK_EQ(k->value, info.k);
      } else {
        // default to row major ordering
        FragmentInfo info(m->value, n->value, k->value, "row_major", scope);
        fragments[buffer_var] = info;
      }
    }
  }

  // Get memory scope
  void VisitStmt_(const AttrStmtNode* op) final { StmtExprVisitor::VisitStmt_(op); }

  // Fragment metadata for all fragments
  std::unordered_map<const VarNode*, FragmentInfo> fragments;
};

std::unordered_map<const VarNode*, FragmentInfo> GetTensorCoreFragmentInfo(const Stmt& stmt) {
  FragmentGetter getter;
  getter(stmt);
  return std::move(getter.fragments);
}

// Check shape of fragment making sure it is a valid shape for tvm_mma_sync
class FragmentChecker : public StmtExprVisitor {
 public:
  explicit FragmentChecker(const FragmentGetter& getter) : fragment_getter(getter) {}

  void VisitExpr_(const CallNode* op) final {
    StmtExprVisitor::VisitExpr_(op);
    // Check shape when calling tvm_mma_sync
    if (op->op.same_as(builtin::tvm_mma_sync()) || op->op.same_as(builtin::tvm_bmma_sync())) {
      ICHECK_EQ(op->args.size(), 8U);
      const VarNode* buffer_var_d = op->args[0].as<VarNode>();
      const VarNode* buffer_var_a = op->args[2].as<VarNode>();
      const VarNode* buffer_var_b = op->args[4].as<VarNode>();
      const VarNode* buffer_var_c = op->args[6].as<VarNode>();
      ICHECK(buffer_var_d);
      ICHECK(buffer_var_a);
      ICHECK(buffer_var_b);
      ICHECK(buffer_var_c);

      // Check all fragment A, B, C and D have the same shape
      ICHECK(CheckShape(buffer_var_d, buffer_var_a));
      ICHECK(CheckShape(buffer_var_d, buffer_var_b));
      ICHECK(CheckShape(buffer_var_d, buffer_var_c));
    }
  }

 private:
  // A tool for checking shapes of two fragments
  bool CheckShape(const VarNode* buffer1, const VarNode* buffer2) {
    CHECK(fragment_getter.fragments.count(buffer1))
        << "Tensorecore fragment " << buffer1->name_hint
        << " must be filled (with tvm_fill_fragment) or loaded (with tvm_load_matrix_sync) before "
           "use.";
    CHECK(fragment_getter.fragments.count(buffer2))
        << "Tensorecore fragment " << buffer2->name_hint
        << " must be filled (with tvm_fill_fragment) or loaded (with tvm_load_matrix_sync) before "
           "use.";
    FragmentInfo info1 = fragment_getter.fragments.at(buffer1);
    FragmentInfo info2 = fragment_getter.fragments.at(buffer2);
    return info1.m == info2.m && info1.n == info2.n && info1.k == info2.k;
  }
  // Fragment infomation
  const FragmentGetter& fragment_getter;
};

// Store the metadata into attributes
class InferFragmenter : public StmtMutator {
 public:
  explicit InferFragmenter(const FragmentGetter& getter) : fragment_getter(getter) {}

  Stmt VisitStmt_(const AllocateNode* op) final {
    Stmt stmt = StmtMutator::VisitStmt_(op);
    const VarNode* buffer = op->buffer_var.get();
    if (fragment_getter.fragments.count(buffer)) {
      // Add attribute to fragments allocation
      FragmentInfo info = fragment_getter.fragments.at(buffer);

      // Add shape attribute to all fragments
      std::string shape =
          std::to_string(info.m) + ", " + std::to_string(info.n) + ", " + std::to_string(info.k);
      PrimExpr shape_expr = StringImm(shape);
      Stmt shape_attr = AttrStmt(op->buffer_var, attr::fragment_shape, shape_expr, stmt);
      if (info.layout != "") {
        // Add shape attribute to matrix_a and matrix_b
        Stmt layout_attr =
            AttrStmt(op->buffer_var, attr::fragment_layout, StringImm(info.layout), shape_attr);
        return layout_attr;
      } else {
        return shape_attr;
      }
    }
    return stmt;
  }

 private:
  // Fragment infomation
  const FragmentGetter& fragment_getter;
};

Stmt InferFragment(Stmt stmt) {
  FragmentGetter getter;
  getter(stmt);
  FragmentChecker checker(getter);
  checker(stmt);
  stmt = InferFragmenter(getter)(std::move(stmt));
  return stmt;
}

namespace transform {

Pass InferFragment() {
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    auto* n = f.CopyOnWrite();
    n->body = InferFragment(std::move(n->body));
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.InferFragment", {});
}

TVM_REGISTER_GLOBAL("tir.transform.InferFragment").set_body_typed(InferFragment);

}  // namespace transform
}  // namespace tir
}  // namespace tvm
