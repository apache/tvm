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
 * \file storage_access.cc
 */
#include <tvm/tir/ir_pass.h>
#include <tvm/target/target_info.h>
#include <string>
#include <utility>
#include "storage_access.h"
#include "ir_util.h"
#include "../../arith/compute_expr.h"

namespace tvm {
namespace tir {

void StorageAccessVisitor::VisitExpr_(const LoadNode* op) {
  const VarNode* buf = op->buffer_var.as<VarNode>();
  StorageScope scope = GetScope(buf);
  if (Enabled(buf, scope)) {
    CHECK(allow_append_);
    AccessEntry e;
    e.threads = env_threads();
    e.buffer = op->buffer_var;
    e.dtype = op->dtype.element_of();
    e.touched = arith::IntSet::vector(op->index);
    e.type = kRead;
    e.scope = scope;
    curr_stmt_.access.emplace_back(std::move(e));
  }
  // traverse child
  StmtExprVisitor::VisitExpr_(op);
}

void StorageAccessVisitor::VisitStmt_(const StoreNode* op) {
  allow_append_ = true;
  CHECK_EQ(curr_stmt_.access.size(), 0U);
  curr_stmt_.stmt = op;
  const VarNode* buf = op->buffer_var.as<VarNode>();
  StorageScope scope = GetScope(buf);
  if (Enabled(buf, scope)) {
    AccessEntry e;
    e.threads = env_threads();
    e.buffer = op->buffer_var;
    e.dtype = op->value.dtype().element_of();
    e.touched = arith::IntSet::vector(op->index);
    e.type = kWrite;
    e.scope = scope;
    curr_stmt_.access.emplace_back(std::move(e));
  }
  // traverse child
  StmtExprVisitor::VisitStmt_(op);
  // push to the scope
  scope_.back().push_back(curr_stmt_);
  // clear access entry.
  curr_stmt_.access.clear();
  allow_append_ = false;
}

void StorageAccessVisitor::VisitStmt_(const EvaluateNode* op) {
  allow_append_ = true;
  CHECK_EQ(curr_stmt_.access.size(), 0U);
  curr_stmt_.stmt = op;
  StmtExprVisitor::VisitStmt_(op);
  // push to the scope
  if (curr_stmt_.access.size() != 0) {
    scope_.back().push_back(curr_stmt_);
    curr_stmt_.access.clear();
  }
  allow_append_ = false;
}

void StorageAccessVisitor::VisitStmt_(const AttrStmtNode* op) {
  if (op->attr_key == attr::storage_scope) {
    const VarNode* buf = op->node.as<VarNode>();
    storage_scope_[buf] =
        StorageScope::make(op->value.as<StringImmNode>()->value);
    StmtExprVisitor::VisitStmt_(op);
  } else if (op->attr_key == attr::double_buffer_write) {
    CHECK(double_buffer_write_ == nullptr);
    double_buffer_write_ = op->node.as<VarNode>();
    scope_.push_back(std::vector<StmtEntry>());
    StmtExprVisitor::VisitStmt_(op);
    StmtEntry s;
    s.stmt = op;
    s.access = Summarize(std::move(scope_.back()), nullptr);
    scope_.pop_back();
    if (!s.access.empty()) {
      for (AccessEntry& e : s.access) {
        if (e.type == kWrite && e.buffer.get() == double_buffer_write_) {
          e.double_buffer_write = true;
        }
      }
      scope_.back().emplace_back(std::move(s));
    }
    double_buffer_write_ = nullptr;
  } else if (op->attr_key == attr::coproc_scope) {
    IterVar iv = Downcast<IterVar>(op->node);
    env_threads_.push_back(iv);
    StmtExprVisitor::VisitStmt_(op);
    env_threads_.CopyOnWrite()->data.pop_back();
  } else if (op->attr_key == attr::thread_extent) {
    IterVar iv = Downcast<IterVar>(op->node);
    env_threads_.push_back(iv);
    if (!in_device_env_) {
      in_device_env_ = true;
      scope_.push_back(std::vector<StmtEntry>());
      StmtExprVisitor::VisitStmt_(op);
      // no need to take the result as the thread barrier automatically syncs.
      Summarize(std::move(scope_.back()), nullptr);
      in_device_env_ = false;
      scope_.pop_back();
    } else {
      StmtExprVisitor::VisitStmt_(op);
    }
    env_threads_.CopyOnWrite()->data.pop_back();
  } else {
    StmtExprVisitor::VisitStmt_(op);
  }
}

void StorageAccessVisitor::VisitStmt_(const ForNode* op) {
  scope_.push_back(std::vector<StmtEntry>());
  StmtExprVisitor::VisitStmt_(op);
  StmtEntry s;
  s.stmt = op;
  s.access = Summarize(std::move(scope_.back()), op);
  scope_.pop_back();
  if (s.access.size() != 0) {
    // relax the touched set to contain all ranges in the loop.
    std::unordered_map<const VarNode*, arith::IntSet> relax_map;
    relax_map[op->loop_var.get()] = arith::IntSet::range(
        Range::make_by_min_extent(op->min, op->extent));
    for (AccessEntry& e : s.access) {
      if (e.buffer.defined()) {
        CHECK(e.touched.defined());
        e.touched = arith::EvalSet(e.touched, relax_map);
      }
    }
  }
  if (!s.access.empty()) {
    scope_.back().emplace_back(std::move(s));
  }
}

void StorageAccessVisitor::VisitStmt_(const IfThenElseNode* op) {
  ++condition_counter_;
  this->VisitExpr(op->condition);
  scope_.push_back(std::vector<StmtEntry>());
  this->VisitStmt(op->then_case);
  StmtEntry s;
  s.stmt = op;
  s.access = Summarize(std::move(scope_.back()), nullptr);
  scope_.pop_back();
  if (op->else_case.defined()) {
    scope_.push_back(std::vector<StmtEntry>());
    auto v = Summarize(std::move(scope_.back()), nullptr);
    scope_.pop_back();
    s.access.insert(s.access.end(), v.begin(), v.end());
  }
  scope_.back().emplace_back(std::move(s));
  --condition_counter_;
}

void StorageAccessVisitor::VisitExpr_(const CallNode* op) {
  if (op->is_intrinsic(intrinsic::tvm_address_of)) {
    const LoadNode *l = op->args[0].as<LoadNode>();
    StmtExprVisitor::VisitExpr_(l);
  } else if (op->is_intrinsic(intrinsic::tvm_access_ptr)) {
    CHECK_EQ(op->args.size(), 5U);
    DataType dtype = op->args[0].dtype();
    const VarNode* buffer = op->args[1].as<VarNode>();
    PrimExpr offset = op->args[2];
    PrimExpr extent = op->args[3];
    const IntImmNode* flag = op->args[4].as<IntImmNode>();
    StorageScope scope = GetScope(buffer);
    // The buffer scope.
    if (Enabled(buffer, scope)) {
      CHECK(allow_append_);
      AccessEntry e;
      e.threads = env_threads();
      e.dtype = dtype;
      e.buffer = Downcast<Var>(op->args[1]);
      e.touched = arith::IntSet::range(
          Range::make_by_min_extent(offset, extent));
      e.scope = scope;
      if (flag->value & 1) {
        e.type = kRead;
        curr_stmt_.access.emplace_back(e);
      }
      if (flag->value & 2) {
        e.type = kWrite;
        curr_stmt_.access.emplace_back(e);
      }
    }
    StmtExprVisitor::VisitExpr_(op);
  } else if (op->is_intrinsic(intrinsic::tvm_storage_sync)) {
    CHECK(allow_append_);
    const std::string& s = op->args[0].as<StringImmNode>()->value;
    if (s != "warp") {
      StorageScope scope = StorageScope::make(s);
      AccessEntry e;
      e.threads = env_threads();
      e.type = kSync;
      e.scope = StorageScope::make(s);
      curr_stmt_.access.emplace_back(std::move(e));
    }
  } else {
    StmtExprVisitor::VisitExpr_(op);
  }
}

StorageScope StorageAccessVisitor::GetScope(const VarNode* buf) const {
  auto it = storage_scope_.find(buf);
  StorageScope s;
  s.rank = StorageRank::kGlobal;
  if (it == storage_scope_.end()) return s;
  return it->second;
}


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
        return AllocateNode::make(
            op->buffer_var, op->dtype, op->extents, op->condition,
            op->body, info->head_address, "nop");
      }
      return op->body;
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

LoweredFunc LowerDeviceStorageAccessInfo(LoweredFunc f) {
  auto n = make_object<LoweredFuncNode>(*f.operator->());
  n->body = LowerStorageAccessInfo(f->body);
  return LoweredFunc(n);
}

}  // namespace tir
}  // namespace tvm
