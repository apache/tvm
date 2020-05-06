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
#include <tvm/tir/expr.h>
#include <tvm/tir/transform.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/runtime/registry.h>

#include <unordered_map>
#include <unordered_set>

#include "storage_access.h"
#include "ir_util.h"
#include "../../runtime/thread_storage_scope.h"

namespace tvm {
namespace tir {

// Get fragment information from tensor intrinsics
class FragmentGetter : public StmtExprVisitor {
 public:
  // fragment metadata
  struct FragmentInfo {
    // fragment shape
    int m, n, k;
    // fragment layout (row-major or column-major)
    std::string layout;
    FragmentInfo() = default;
    FragmentInfo(int _m, int _n, int _k, const std::string& _layout)
      : m(_m), n(_n), k(_k), layout(_layout) {}
  };

  void VisitExpr_(const CallNode* op) final {
    StmtExprVisitor::VisitExpr_(op);

    if (op->is_intrinsic(intrinsic::tvm_load_matrix_sync) ||
        op->is_intrinsic(intrinsic::tvm_store_matrix_sync)) {
      // Get shape and layout information from load and store intrinsic
      CHECK_EQ(op->args.size(), 8U);
      const VarNode* buffer_var = op->args[0].as<VarNode>();
      CHECK(buffer_var);
      // Get shape
      const IntImmNode* m = op->args[1].as<IntImmNode>();
      const IntImmNode* n = op->args[2].as<IntImmNode>();
      const IntImmNode* k = op->args[3].as<IntImmNode>();
      const StringImmNode* layout = op->args[7].as<StringImmNode>();
      CHECK(m);
      CHECK(n);
      CHECK(k);
      CHECK(layout);

      std::string scope = scopes[buffer_var];
      if (fragments.count(buffer_var)) {
        // check if the fragment has met before
        FragmentInfo info = fragments[buffer_var];
        CHECK_EQ(m->value, info.m);
        CHECK_EQ(n->value, info.n);
        CHECK_EQ(k->value, info.k);
        if (scope == "wmma.matrix_a" || scope == "wmma.matrix_b") {
          CHECK_EQ(layout->value, info.layout);
        }
      } else {
        // store metadata
        FragmentInfo info;
        if (scope == "wmma.matrix_a" || scope == "wmma.matrix_b") {
          info = FragmentInfo(m->value, n->value, k->value, layout->value);
        } else if (scope == "wmma.accumulator") {
          info = FragmentInfo(m->value, n->value, k->value, "");
        }
        fragments[buffer_var] = info;
      }
    } else if (op->is_intrinsic(intrinsic::tvm_fill_fragment)) {
      // Get shape information from fill intrinsic
      CHECK_EQ(op->args.size(), 6U);
      const VarNode* buffer_var = op->args[0].as<VarNode>();
      CHECK(buffer_var);
      // Get shape
      const IntImmNode* m = op->args[1].as<IntImmNode>();
      const IntImmNode* n = op->args[2].as<IntImmNode>();
      const IntImmNode* k = op->args[3].as<IntImmNode>();
      CHECK(m);
      CHECK(n);
      CHECK(k);

      std::string scope = scopes[buffer_var];
      // Only wmma.accumulator can use tvm_fill_fragment
      CHECK_EQ(scope, "wmma.accumulator");
      if (fragments.count(buffer_var)) {
        FragmentInfo info = fragments[buffer_var];
        CHECK_EQ(m->value, info.m);
        CHECK_EQ(n->value, info.n);
        CHECK_EQ(k->value, info.k);
      } else {
        FragmentInfo info(m->value, n->value, k->value, "");
        fragments[buffer_var] = info;
      }
    }
  }

  // Get memory scope
  void VisitStmt_(const AttrStmtNode* op) final {
    if (op->attr_key == attr::storage_scope) {
      const VarNode* buffer = op->node.as<VarNode>();
      CHECK(buffer);
      scopes[buffer] = op->value.as<StringImmNode>()->value;
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  // Memory scope for allocations
  std::unordered_map<const VarNode*, std::string> scopes;
  // Fragment metadata for all fragments
  std::unordered_map<const VarNode*, FragmentInfo> fragments;
};

// Check shape of fragment making sure it is a valid shape for tvm_mma_sync
class FragmentChecker : public StmtExprVisitor {
 public:
  explicit FragmentChecker(const FragmentGetter &getter) : fragment_getter(getter) {}

  void VisitExpr_(const CallNode* op) final {
    StmtExprVisitor::VisitExpr_(op);
    // Check shape when calling tvm_mma_sync
    if (op->is_intrinsic(intrinsic::tvm_mma_sync) ||
        op->is_intrinsic(intrinsic::tvm_bmma_sync)) {
      CHECK_EQ(op->args.size(), 8U);
      const VarNode* buffer_var_d = op->args[0].as<VarNode>();
      const VarNode* buffer_var_a = op->args[2].as<VarNode>();
      const VarNode* buffer_var_b = op->args[4].as<VarNode>();
      const VarNode* buffer_var_c = op->args[6].as<VarNode>();
      CHECK(buffer_var_d);
      CHECK(buffer_var_a);
      CHECK(buffer_var_b);
      CHECK(buffer_var_c);

      // Check all fragment A, B, C and D have the same shape
      CHECK(CheckShape(buffer_var_d, buffer_var_a));
      CHECK(CheckShape(buffer_var_d, buffer_var_b));
      CHECK(CheckShape(buffer_var_d, buffer_var_c));
    }
  }

 private:
  // A tool for checking shapes of two fragments
  bool CheckShape(const VarNode* buffer1, const VarNode* buffer2) {
    CHECK(fragment_getter.fragments.count(buffer1));
    CHECK(fragment_getter.fragments.count(buffer2));
    FragmentGetter::FragmentInfo info1 = fragment_getter.fragments.at(buffer1);
    FragmentGetter::FragmentInfo info2 = fragment_getter.fragments.at(buffer2);
    return info1.m == info2.m && info1.n == info2.n && info1.k == info2.k;
  }
  // Fragment infomation
  const FragmentGetter &fragment_getter;
};

// Store the metadata into attributes
class InferFragmenter : public StmtMutator {
 public:
  explicit InferFragmenter(const FragmentGetter &getter) : fragment_getter(getter) {}

  Stmt VisitStmt_(const AllocateNode* op) final {
    Stmt stmt = StmtMutator::VisitStmt_(op);
    const VarNode* buffer = op->buffer_var.get();
    if (fragment_getter.fragments.count(buffer)) {
      // Add attribute to fragments allocation
      FragmentGetter::FragmentInfo info = fragment_getter.fragments.at(buffer);

      // Add shape attribute to all fragments
      std::string shape = std::to_string(info.m) + ", " +
                          std::to_string(info.n) + ", " +
                          std::to_string(info.k);
      PrimExpr shape_expr = StringImmNode::make(shape);
      Stmt shape_attr = AttrStmtNode::make(op->buffer_var, attr::fragment_shape, shape_expr, stmt);
      if (info.layout != "") {
        // Add shape attribute to matrix_a and matrix_b
        Stmt layout_attr = AttrStmtNode::make(op->buffer_var, attr::fragment_layout,
                                          StringImmNode::make(info.layout), shape_attr);
        return layout_attr;
      } else {
        return shape_attr;
      }
    }
    return stmt;
  }

 private:
  // Fragment infomation
  const FragmentGetter &fragment_getter;
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

TVM_REGISTER_GLOBAL("tir.transform.InferFragment")
.set_body_typed(InferFragment);

}  // namespace transform
}  // namespace tir
}  // namespace tvm
