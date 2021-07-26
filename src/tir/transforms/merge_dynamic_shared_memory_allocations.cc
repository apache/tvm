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
 * \file merge_dynamic_shared_memory_allocations.cc
 */
#include <tvm/runtime/registry.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <unordered_map>
#include <unordered_set>

#include "../../runtime/thread_storage_scope.h"
#include "ir_utils.h"

namespace tvm {
namespace tir {

class AllocateCollector : public StmtExprVisitor {
 public:
  void VisitStmt_(const AllocateNode* op) final {
    auto storage_scope = runtime::StorageScope::Create(GetPtrStorageScope(op->buffer_var));
    if (storage_scope.rank == runtime::StorageRank::kShared && storage_scope.tag == ".dyn") {
      dyn_shmem_allocs_.insert(op);
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  std::unordered_set<const AllocateNode*> dyn_shmem_allocs_;
};

class DynamicSharedMemoryRewriter : public StmtExprMutator {
 public:
  DynamicSharedMemoryRewriter(const std::unordered_set<const AllocateNode*>& dyn_shmem_allocs)
      : dyn_shmem_allocs_{dyn_shmem_allocs} {}

  Stmt Rewrite(Stmt stmt) { return stmt; }

  PrimExpr VisitExpr_(const LoadNode* op) final { return StmtExprMutator::VisitExpr_(op); }

  Stmt VisitStmt_(const AllocateNode* op) final { return StmtExprMutator::VisitStmt_(op); }

  Stmt VisitStmt_(const StoreNode* op) final { return StmtExprMutator::VisitStmt_(op); }

 private:
  Var merged_buf_var_{"buf_dyn_shmem", PointerType(PrimType(DataType::UInt(8)), "shared.dyn")};
  std::unordered_set<const AllocateNode*> dyn_shmem_allocs_;
};

Stmt MergeDynamicSharedMemoryAllocations(Stmt stmt) {
  AllocateCollector collector;
  collector(stmt);
  return DynamicSharedMemoryRewriter(collector.dyn_shmem_allocs_).Rewrite(std::move(stmt));
}

namespace transform {

Pass MergeDynamicSharedMemoryAllocations() {
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    auto* n = f.CopyOnWrite();
    n->body = MergeDynamicSharedMemoryAllocations(std::move(n->body));
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.MergeDynamicSharedMemoryAllocations", {});
}

TVM_REGISTER_GLOBAL("tir.transform.MergeDynamicSharedMemoryAllocations")
    .set_body_typed(MergeDynamicSharedMemoryAllocations);

}  // namespace transform
}  // namespace tir
}  // namespace tvm
