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
 * \file lower_device_storage_access.cc
 * \brief Lower the special device storage access.
 */
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>
#include <tvm/tir/buffer.h>
#include <tvm/target/target_info.h>
#include <tvm/runtime/registry.h>

#include <tvm/tir/ir_pass.h>

#include "../pass/ir_util.h"
#include "../../runtime/thread_storage_scope.h"

namespace tvm {
namespace tir {

using runtime::StorageScope;
using runtime::StorageRank;

class StorageAccessInfoLower : public StmtExprMutator {
 public:
  Stmt VisitStmt_(const AllocateNode* op) final {
    // Lower allocate to device allocate when needed.
    Stmt stmt = StmtExprMutator::VisitStmt_(op);
    op = stmt.as<AllocateNode>();
    // For special memory, remove allocate, or use head expr
    auto it = storage_info_.find(op->buffer_var.get());
    if (it != storage_info_.end() && it->second.info.defined()) {
      const MemoryInfo& info = it->second.info;
      ++it->second.alloc_count;
      CHECK_LE(it->second.alloc_count, 1)
          << "Double allocation of " << it->second.scope.to_string();

      if (info->head_address.defined()) {
        return LetStmtNode::make(
          op->buffer_var, info->head_address, op->body);
      } else {
        return op->body;
      }
    } else {
      return stmt;
    }
  }
  Stmt VisitStmt_(const AttrStmtNode* op) final {
    if (op->attr_key == attr::storage_scope) {
      const VarNode* buf = op->node.as<VarNode>();
      StorageScope scope = StorageScope::make(op->value.as<StringImmNode>()->value);
      StorageEntry e;
      e.scope = scope;
      if (scope.tag.length() != 0) {
        e.info = GetMemoryInfo(op->value.as<StringImmNode>()->value);
        CHECK(e.info.defined()) << "Cannot find memory info of " << scope.to_string();
      }
      storage_info_[buf] = e;
      return StmtExprMutator::VisitStmt_(op);

    } else {
      return StmtExprMutator::VisitStmt_(op);
    }
  }

  PrimExpr VisitExpr_(const CallNode* op) final {
    if (op->is_intrinsic(intrinsic::tvm_access_ptr)) {
      return MakeAccessPtr(op);
    } else {
      return StmtExprMutator::VisitExpr_(op);
    }
  }

 private:
  // tvm_access_ptr
  PrimExpr MakeAccessPtr(const CallNode* op) {
    // Specially handle the buffer packed intrinsic
    PrimExpr expr = StmtExprMutator::VisitExpr_(op);
    op = expr.as<CallNode>();
    CHECK_EQ(op->args.size(), 5U);
    DataType dtype = op->args[0].dtype();
    const VarNode* buffer = op->args[1].as<VarNode>();
    Var buffer_var = Downcast<Var>(op->args[1]);
    PrimExpr offset = op->args[2];
    auto it = storage_info_.find(buffer);
    if (it != storage_info_.end() && it->second.info.defined()) {
      return MakeTaggedAccessPtr(
          op->dtype, buffer_var, dtype, offset,
          it->second.info);
    }
    CHECK(op->dtype.is_handle());
    // Change to address_of
    return AddressOffset(buffer_var, dtype, offset);
  }

  PrimExpr MakeTaggedAccessPtr(DataType ptr_type,
                               Var buffer_var,
                               DataType dtype,
                               PrimExpr offset,
                               const MemoryInfo& info) {
    if (ptr_type.is_handle()) {
      CHECK(info->head_address.defined())
          << buffer_var << " is not adddressable.";
      return AddressOffset(buffer_var, dtype, offset);
    }
    int dtype_bits = dtype.bits() * dtype.lanes();
    CHECK_EQ(info->unit_bits % dtype_bits, 0);
    return cast(ptr_type,
                   tir::Simplify(offset / make_const(
                       offset.dtype(), info->unit_bits / dtype_bits)));
  }
  // The storage entry.
  struct StorageEntry {
    // Whether it is tagged memory.
    StorageScope scope;
    // The memory info if any.
    MemoryInfo info;
    // Allocation counter
    int alloc_count{0};
  };
  // The storage scope of each buffer
  std::unordered_map<const VarNode*, StorageEntry> storage_info_;
};

Stmt LowerStorageAccessInfo(Stmt stmt) {
  return StorageAccessInfoLower()(std::move(stmt));
}


namespace transform {

Pass LowerDeviceStorageAccessInfo() {
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    auto* n = f.CopyOnWrite();
    n->body = StorageAccessInfoLower()(std::move(n->body));
    return f;
  };
  return CreatePrimFuncPass(
      pass_func, 0, "tir.LowerDeviceStorageAccessInfo", {});
}

TVM_REGISTER_GLOBAL("tir.transform.LowerDeviceStorageAccessInfo")
.set_body_typed(LowerDeviceStorageAccessInfo);

}  // namespace transform
}  // namespace tir
}  // namespace tvm
