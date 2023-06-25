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
#include <tvm/arith/analyzer.h>
#include <tvm/runtime/registry.h>
#include <tvm/target/target_info.h>
#include <tvm/tir/buffer.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include "../../runtime/thread_storage_scope.h"
#include "ir_utils.h"

namespace tvm {
namespace tir {

using runtime::StorageRank;
using runtime::StorageScope;

class StorageAccessInfoLower : public StmtExprMutator {
 public:
  Stmt VisitStmt_(const AllocateNode* op) final {
    auto scope = StorageScope::Create(GetPtrStorageScope(op->buffer_var));
    if (scope.tag.length() != 0 && scope.tag != ".dyn") {
      auto info = GetMemoryInfo(GetPtrStorageScope(op->buffer_var));
      ICHECK(info.defined()) << "Cannot find memory info of " << scope.to_string();
      ICHECK(storage_info_.find(op->buffer_var.get()) == storage_info_.end())
          << "Double allocation of " << scope.to_string();
      storage_info_[op->buffer_var.get()] = info;

      // Lower allocate to device allocate when needed.
      Stmt stmt = StmtExprMutator::VisitStmt_(op);
      op = stmt.as<AllocateNode>();
      if (info->head_address.defined()) {
        return LetStmt(op->buffer_var, info->head_address, op->body);
      } else {
        return op->body;
      }
    } else {
      return StmtExprMutator::VisitStmt_(op);
    }
  }

  Stmt VisitStmt_(const DeclBufferNode* op) final {
    auto node = Downcast<DeclBuffer>(StmtExprMutator::VisitStmt_(op));
    if (auto it = storage_info_.find(node->buffer->data.get());
        it != storage_info_.end() && !it->second->head_address.defined()) {
      return node->body;
    } else {
      return std::move(node);
    }
  }

  PrimExpr VisitExpr_(const CallNode* op) final {
    if (op->op.same_as(builtin::tvm_access_ptr())) {
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
    ICHECK_EQ(op->args.size(), 5U);
    DataType dtype = op->args[0].dtype();
    const VarNode* buffer = op->args[1].as<VarNode>();
    Var buffer_var = Downcast<Var>(op->args[1]);
    PrimExpr offset = op->args[2];
    auto it = storage_info_.find(buffer);
    if (it != storage_info_.end() && it->second.defined()) {
      return MakeTaggedAccessPtr(op->dtype, buffer_var, dtype, offset, it->second);
    }
    ICHECK(op->dtype.is_handle());
    // Change to address_of
    return AddressOffset(buffer_var, dtype, offset);
  }

  PrimExpr MakeTaggedAccessPtr(DataType ptr_type, Var buffer_var, DataType dtype, PrimExpr offset,
                               const MemoryInfo& info) {
    if (ptr_type.is_handle()) {
      ICHECK(info->head_address.defined()) << buffer_var << " is not adddressable.";
      return AddressOffset(buffer_var, dtype, offset);
    }
    int dtype_bits = dtype.bits() * dtype.lanes();
    ICHECK_EQ(info->unit_bits % dtype_bits, 0);
    return cast(ptr_type, analyzer_.Simplify(
                              offset / make_const(offset.dtype(), info->unit_bits / dtype_bits)));
  }
  // The storage scope of each buffer
  std::unordered_map<const VarNode*, MemoryInfo> storage_info_;
  // analyzer
  arith::Analyzer analyzer_;
};

Stmt LowerStorageAccessInfo(Stmt stmt) { return StorageAccessInfoLower()(std::move(stmt)); }

namespace transform {

Pass LowerDeviceStorageAccessInfo() {
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    auto* n = f.CopyOnWrite();
    n->body = StorageAccessInfoLower()(std::move(n->body));
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.LowerDeviceStorageAccessInfo", {});
}

TVM_REGISTER_GLOBAL("tir.transform.LowerDeviceStorageAccessInfo")
    .set_body_typed(LowerDeviceStorageAccessInfo);

}  // namespace transform
}  // namespace tir
}  // namespace tvm
