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
#include "storage_access.h"

#include <tvm/target/target_info.h>
#include <tvm/tir/op.h>

#include <string>
#include <utility>

#include "ir_utils.h"

namespace tvm {
namespace tir {

void StorageAccessVisitor::VisitExpr_(const BufferLoadNode* op) {
  Var buf = op->buffer->data;
  StorageScope scope = GetScope(buf);
  if (Enabled(buf.get(), scope)) {
    ICHECK(allow_append_) << op << " " << scope.to_string();
    AccessEntry e;
    e.threads = env_threads();
    e.buffer = buf;
    e.dtype = op->dtype.element_of();
    for (const auto& index : op->indices) {
      e.touched.push_back(arith::IntSet::Vector(index));
    }
    e.type = kRead;
    e.scope = scope;
    curr_stmt_.access.emplace_back(std::move(e));
  }
  // traverse child
  StmtExprVisitor::VisitExpr_(op);
}

void StorageAccessVisitor::VisitStmt_(const BufferStoreNode* op) {
  allow_append_ = true;
  ICHECK_EQ(curr_stmt_.access.size(), 0U);
  curr_stmt_.stmt = op;

  Var buf = op->buffer->data;
  StorageScope scope = GetScope(buf);
  if (Enabled(buf.get(), scope)) {
    AccessEntry e;
    e.threads = env_threads();
    e.buffer = buf;
    e.dtype = op->value.dtype().element_of();
    for (const auto& index : op->indices) {
      e.touched.push_back(arith::IntSet::Vector(index));
    }
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
  ICHECK_EQ(curr_stmt_.access.size(), 0U);
  curr_stmt_.stmt = op;
  StmtExprVisitor::VisitStmt_(op);
  // push to the scope
  if (curr_stmt_.access.size() != 0) {
    scope_.back().push_back(curr_stmt_);
    curr_stmt_.access.clear();
  }
  allow_append_ = false;
}

void StorageAccessVisitor::VisitStmt_(const LetStmtNode* op) {
  allow_append_ = true;
  ICHECK_EQ(curr_stmt_.access.size(), 0U);
  curr_stmt_.stmt = op;
  this->VisitExpr(op->value);
  // push to the scope
  scope_.back().push_back(curr_stmt_);
  // clear access entry.
  curr_stmt_.access.clear();
  allow_append_ = false;
  // traverse body block
  this->VisitStmt(op->body);
}

void StorageAccessVisitor::VisitStmt_(const AttrStmtNode* op) {
  if (op->attr_key == attr::double_buffer_write) {
    ICHECK(double_buffer_write_ == nullptr);
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
    env_threads_.pop_back();
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
    env_threads_.pop_back();
  } else if (op->attr_key == attr::hand_threaded) {
    // skip this pass on blocks that were hand_threaded
    // this avoids control flow and read/write conflicts
    // between hand-threaded kernels and automatic threading
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
    relax_map[op->loop_var.get()] =
        arith::IntSet::FromRange(Range::FromMinExtent(op->min, op->extent));
    for (AccessEntry& e : s.access) {
      if (e.buffer.defined()) {
        ICHECK(e.touched.size());
        Array<arith::IntSet> new_touched;
        for (const auto& touched : e.touched) {
          new_touched.push_back(arith::EvalSet(touched, relax_map));
        }
        e.touched = std::move(new_touched);
      }
    }
  }
  if (!s.access.empty()) {
    scope_.back().emplace_back(std::move(s));
  }
}

bool IsThreadInvariant(const PrimExpr& cond) {
  if (auto call = cond.as<CallNode>()) {
    if (auto opt_call_op = call->op.as<Op>()) {
      auto call_op = opt_call_op.value();
      if (call_op.same_as(builtin::tvm_thread_invariant())) {
        return true;
      }
    }
  }
  return false;
}

void StorageAccessVisitor::VisitStmt_(const IfThenElseNode* op) {
  bool is_thread_invariant = IsThreadInvariant(op->condition);
  if (!is_thread_invariant) {
    ++condition_counter_;
  }
  this->VisitExpr(op->condition);
  scope_.push_back(std::vector<StmtEntry>());
  this->VisitStmt(op->then_case);
  StmtEntry s;
  s.stmt = op;
  s.access = Summarize(std::move(scope_.back()), nullptr);
  scope_.pop_back();
  if (op->else_case) {
    scope_.push_back(std::vector<StmtEntry>());
    this->VisitStmt(op->else_case.value());
    auto v = Summarize(std::move(scope_.back()), nullptr);
    scope_.pop_back();
    s.access.insert(s.access.end(), v.begin(), v.end());
  }
  scope_.back().emplace_back(std::move(s));
  if (!is_thread_invariant) {
    --condition_counter_;
  }
}

void StorageAccessVisitor::VisitStmt_(const WhileNode* op) {
  bool is_thread_invariant = IsThreadInvariant(op->condition);
  if (!is_thread_invariant) {
    ++condition_counter_;
  }
  this->VisitExpr(op->condition);
  scope_.push_back(std::vector<StmtEntry>());
  this->VisitStmt(op->body);
  StmtEntry s;
  s.stmt = op;
  s.access = Summarize(std::move(scope_.back()), nullptr);
  scope_.pop_back();
  scope_.back().emplace_back(std::move(s));
  if (!is_thread_invariant) {
    --condition_counter_;
  }
}

void StorageAccessVisitor::VisitExpr_(const CallNode* op) {
  if (op->op.same_as(builtin::address_of())) {
    const BufferLoadNode* load = op->args[0].as<BufferLoadNode>();
    StmtExprVisitor::VisitExpr_(load);
  } else if (op->op.same_as(builtin::tvm_access_ptr())) {
    ICHECK_EQ(op->args.size(), 5U);
    DataType dtype = op->args[0].dtype();
    const VarNode* buffer = op->args[1].as<VarNode>();
    PrimExpr offset = op->args[2];
    PrimExpr extent = op->args[3];
    const IntImmNode* flag = op->args[4].as<IntImmNode>();
    StorageScope scope = GetScope(GetRef<Var>(buffer));
    // The buffer scope.
    if (Enabled(buffer, scope)) {
      ICHECK(allow_append_);
      AccessEntry e;
      e.threads = env_threads();
      e.dtype = dtype;
      e.buffer = Downcast<Var>(op->args[1]);
      e.touched = {arith::IntSet::FromRange(Range::FromMinExtent(offset, extent))};
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
  } else if (op->op.same_as(builtin::tvm_storage_sync())) {
    ICHECK(allow_append_);
    const std::string& s = op->args[0].as<StringImmNode>()->value;
    if (s != "warp") {
      StorageScope scope = StorageScope::Create(s);
      AccessEntry e;
      e.threads = env_threads();
      e.type = kSync;
      e.scope = StorageScope::Create(s);
      curr_stmt_.access.emplace_back(std::move(e));
    }
  } else {
    StmtExprVisitor::VisitExpr_(op);
  }
}

StorageScope StorageAccessVisitor::GetScope(Var buffer_var) const {
  if (buffer_var->type_annotation.as<PointerTypeNode>()) {
    return StorageScope::Create(GetPtrStorageScope(buffer_var));
  }
  return StorageScope();  // global by default
}

}  // namespace tir
}  // namespace tvm
