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
 * \brief Each GPU kernel is allowed to have only one dynamic shared memory allocation.
 * This pass merges multiple TIR-level dynamic shared memory allocations into one allocation.
 */
#include <tvm/runtime/registry.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <unordered_map>
#include <unordered_set>

#include "../../runtime/thread_storage_scope.h"
#include "ir_utils.h"

namespace tvm {
namespace tir {

bool IsDynamicSharedMemory(Var buffer_var) {
  auto storage_scope = runtime::StorageScope::Create(GetPtrStorageScope(buffer_var));
  return storage_scope.rank == runtime::StorageRank::kShared && storage_scope.tag == ".dyn";
}

class AllocateCollector : public StmtExprVisitor {
 public:
  void VisitStmt_(const AllocateNode* op) final {
    if (IsDynamicSharedMemory(op->buffer_var)) {
      dyn_shmem_allocs_.insert(op);
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  std::unordered_set<const AllocateNode*> dyn_shmem_allocs_;
};

class DynamicSharedMemoryRewriter : public StmtExprMutator {
 public:
  explicit DynamicSharedMemoryRewriter(
      const std::unordered_set<const AllocateNode*>& dyn_shmem_allocs)
      : dyn_shmem_allocs_{dyn_shmem_allocs} {}

  Stmt VisitStmt_(const AttrStmtNode* op) final {
    if (op->attr_key == attr::thread_extent && !allocated) {
      // Allocate one dynamic shared memory allocation at the beginning of thread scope
      int align = 1;
      for (const auto& alloc : dyn_shmem_allocs_) {
        ICHECK_EQ(alloc->dtype.lanes(), 1) << "vector dtype allocation not supported.";
        align = std::max(align, alloc->dtype.bytes());
      }
      for (const auto& alloc : dyn_shmem_allocs_) {
        buffer_byte_offsets_[alloc->buffer_var.get()] = merged_alloc_size_;
        merged_alloc_size_ += alloc->extent * align;
      }

      allocated = true;
      auto new_body = Allocate(merged_buf_var_, DataType::UInt(8), merged_alloc_size_, const_true(),
                               StmtExprMutator::VisitStmt(op->body));
      return AttrStmt(op->node, op->attr_key, op->value, new_body, op->span);
    }
    return StmtMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const AllocateNode* op) final {
    if (IsDynamicSharedMemory(op->buffer_var)) {
      return StmtExprMutator::VisitStmt(op->body);
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  PrimExpr VisitExpr_(const LoadNode* op) final {
    if (IsDynamicSharedMemory(op->buffer_var)) {
      auto offset = GetBufferOffset(op->buffer_var, op->dtype);
      auto index = StmtExprMutator::VisitExpr(op->index);
      return Load(op->dtype, merged_buf_var_, offset + index, op->predicate, op->span);
    }
    return StmtExprMutator::VisitExpr_(op);
  }

  Stmt VisitStmt_(const StoreNode* op) final {
    if (IsDynamicSharedMemory(op->buffer_var)) {
      auto offset = GetBufferOffset(op->buffer_var, op->value->dtype);
      auto index = StmtExprMutator::VisitExpr(op->index);
      auto value = StmtExprMutator::VisitExpr(op->value);
      return Store(merged_buf_var_, value, offset + index, op->predicate, op->span);
    }
    return StmtExprMutator::VisitStmt_(op);
  }

 private:
  PrimExpr GetBufferOffset(Var buffer_var, DataType dtype) {
    auto it = buffer_byte_offsets_.find(buffer_var.get());
    ICHECK(it != buffer_byte_offsets_.end());
    return indexdiv(it->second, dtype.bytes());
  }

  Var merged_buf_var_{"buf_dyn_shmem", PointerType(PrimType(DataType::UInt(8)), "shared.dyn")};
  std::unordered_set<const AllocateNode*> dyn_shmem_allocs_;
  PrimExpr merged_alloc_size_{0};
  std::unordered_map<const VarNode*, PrimExpr> buffer_byte_offsets_;
  bool allocated{false};
};

Stmt MergeDynamicSharedMemoryAllocations(Stmt stmt) {
  AllocateCollector collector;
  collector(stmt);
  if (collector.dyn_shmem_allocs_.size() > 1) {
    return DynamicSharedMemoryRewriter(collector.dyn_shmem_allocs_)(std::move(stmt));
  }
  return stmt;
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
