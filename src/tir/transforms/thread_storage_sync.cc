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
 * \file thread_storage_sync.cc
 */
#include <tvm/tir/expr.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/transform.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>
#include <tvm/runtime/registry.h>

#include <unordered_map>
#include <unordered_set>

#include "ir_util.h"
#include "storage_access.h"
#include "../../runtime/thread_storage_scope.h"

namespace tvm {
namespace tir {

class ThreadSyncPlanner : public StorageAccessVisitor {
 public:
  explicit ThreadSyncPlanner(StorageScope sync_scope)
      : sync_scope_(sync_scope) {}

    // The syncs inserted before each statement
  std::unordered_set<const Object*> syncs_inserted_;

 protected:
  bool Enabled(const VarNode* buf,
               const StorageScope& scope) const final {
    return in_device_env() && scope == sync_scope_;
  }
  // Plan the sync
  std::vector<AccessEntry> Summarize(
      std::vector<StmtEntry> seq, const ForNode* loop) final {
    // Unsynced reads and writes
    std::vector<AccessEntry> reads;
    std::vector<AccessEntry> writes;
    // if it is a loop, rotate two times to consider effect of loop.
    // simulation based approach to find dependenceies
    for (size_t i = 0; i < seq.size(); ++i) {
      const StmtEntry& s = seq[i];
      // check if sync before statement is needed.
      bool sync_before_stmt = (syncs_inserted_.count(s.stmt) != 0);
      // Apply the syncs added already.
      if (sync_before_stmt) {
        reads.clear();
        writes.clear();
      }
      for (const AccessEntry& acc : s.access) {
        if (acc.type == kRead) {
          if (FindConflict(writes, acc, false)) {
            sync_before_stmt = true; break;
          }
        } else if (acc.type == kWrite) {
          if (FindConflict(reads, acc, false)) {
            sync_before_stmt = true; break;
          }
        } else if (acc.type == kSync) {
          reads.clear(); writes.clear();
        }
      }
      // If sync is inserted. remove the irrelevant things.
      if (sync_before_stmt) {
        reads.clear(); writes.clear();
      }
      // Add the read/write of current statement
      for (const AccessEntry& acc : s.access) {
        if (acc.type == kRead) {
          reads.push_back(acc);
        } else if (acc.type == kWrite) {
          writes.push_back(acc);
        } else if (acc.type == kSync) {
          reads.clear(); writes.clear();
        }
      }
      if (sync_before_stmt) {
        CHECK_EQ(condition_counter(), 0)
            << "Cannot insert syncs inside condition";
        syncs_inserted_.insert(s.stmt);
      }
    }
    if (loop != nullptr) {
      for (size_t i = 0; i < seq.size(); ++i) {
        const StmtEntry& s = seq[i];
        if (syncs_inserted_.count(s.stmt) != 0) break;
        if (reads.empty() && writes.empty()) break;
        bool sync_before_stmt = false;
        for (const AccessEntry& acc : s.access) {
          if (acc.type == kRead) {
            if (FindConflict(writes, acc, true)) {
              sync_before_stmt = true; break;
            }
          } else if (acc.type == kWrite) {
            if (FindConflict(reads, acc, true)) {
              sync_before_stmt = true; break;
            }
          } else if (acc.type == kSync) {
            reads.clear(); writes.clear();
          }
        }
        if (sync_before_stmt) {
          CHECK_EQ(condition_counter(), 0)
              << "Cannot insert syncs inside condition";
          syncs_inserted_.insert(s.stmt);
          break;
        }
      }
    }
    // return the exposed entries, remove unecessary ones.
    int sync_count = 0;
    // head are before first sync, tail are after last sync
    std::vector<AccessEntry> head, tail;
    AccessEntry esync;
    esync.threads = this->env_threads();
    esync.type = kSync;
    esync.scope = sync_scope_;

    for (const StmtEntry& s : seq) {
      if (syncs_inserted_.count(s.stmt)) {
        if (sync_count != 0) {
          tail.clear();
        } else {
          head.push_back(esync);
        }
        ++sync_count;
      }
      for (const AccessEntry& acc : s.access) {
        if (acc.type == kSync) {
          if (sync_count != 0) {
            tail.clear();
          } else {
            head.push_back(esync);
          }
          ++sync_count;
        } else {
          if (sync_count != 0) {
            tail.push_back(acc);
          } else {
            head.push_back(acc);
          }
        }
      }
    }
    head.insert(head.end(), tail.begin(), tail.end());
    if (loop != nullptr) {
      // clear double buffer flag after a loop is finished.
      for (AccessEntry& e : head) {
        e.double_buffer_write = false;
      }
    }
    return head;
  }

 private:
  // find conflicting entry in vec.
  bool FindConflict(const std::vector<AccessEntry>& vec,
                    const AccessEntry& e,
                    bool loop_carry) {
    for (const AccessEntry& x : vec) {
      if (x.buffer.same_as(e.buffer)) {
        // Assumes no race between threads
        // Same index value means no conflicts
        // TODO(tqchen) more standard set based testing.
        if (e.touched.is_single_point() &&
            x.touched.is_single_point()) {
          if (ExprDeepEqual()(e.touched.point_value(),
                              x.touched.point_value())) continue;
        }
        if (x.double_buffer_write &&
            e.type == kRead &&
            !loop_carry) continue;
        return true;
      }
    }
    return false;
  }

 private:
  // synchronization scope
  StorageScope sync_scope_;
};

class ThreadSyncInserter : public StmtExprMutator {
 public:
  ThreadSyncInserter(StorageScope sync_scope,
                     const std::unordered_set<const Object*>& syncs)
      : sync_scope_(sync_scope), syncs_(syncs) {}

  Stmt VisitStmt(const Stmt& stmt) final {
    if (syncs_.size() == 0) return stmt;
    if (syncs_.count(stmt.get())) {
      Stmt barrier;
      if (sync_scope_.rank == StorageRank::kGlobal) {
        barrier = MakeGlobalBarrier();
      } else {
        barrier = EvaluateNode::make(
                CallNode::make(DataType::Int(32), intrinsic::tvm_storage_sync,
                           {StringImmNode::make(sync_scope_.to_string())},
                           CallNode::Intrinsic));
      }
      // Mutate after query, to avoid stmt change.
      auto ret = StmtExprMutator::VisitStmt(stmt);
      ret = SeqStmt({barrier, ret});
      return ret;
    } else {
      return StmtExprMutator::VisitStmt(stmt);
    }
  }
  PrimExpr VisitExpr_(const LoadNode* op) final {
    if (sync_scope_.rank == StorageRank::kGlobal &&
        GetScope(op->buffer_var.get()).rank == StorageRank::kGlobal) {
      ++rw_stats_[op->buffer_var].read_count;
    }
    return StmtExprMutator::VisitExpr_(op);
  }
  Stmt VisitStmt_(const StoreNode* op) final {
    if (sync_scope_.rank == StorageRank::kGlobal &&
        GetScope(op->buffer_var.get()).rank == StorageRank::kGlobal) {
      ++rw_stats_[op->buffer_var].write_count;
    }
    return StmtExprMutator::VisitStmt_(op);
  }
  Stmt VisitStmt_(const AttrStmtNode* op) final {
    if (op->attr_key == attr::thread_extent) {
      bool temp = true;
      std::swap(temp, in_thread_env_);
      thread_extents_.push_back(op);
      Stmt ret = StmtExprMutator::VisitStmt_(op);
      thread_extents_.pop_back();
      std::swap(temp, in_thread_env_);
      // first thread scope.
      if (!in_thread_env_ && sync_scope_.rank == StorageRank::kGlobal) {
        ret = InitGlobalBarrier(ret.as<AttrStmtNode>());
        num_blocks_ = PrimExpr();
        is_lead_ = PrimExpr();
      }
      return ret;
    } else if (op->attr_key == attr::storage_scope) {
      const VarNode* buf = op->node.as<VarNode>();
      storage_scope_[buf] =
          StorageScope::make(op->value.as<StringImmNode>()->value);
      return StmtExprMutator::VisitStmt_(op);
    } else {
      return StmtExprMutator::VisitStmt_(op);
    }
  }

  PrimExpr VisitExpr_(const CallNode* op) final {
    if (op->is_intrinsic(intrinsic::tvm_access_ptr)) {
      PrimExpr expr = StmtExprMutator::VisitExpr_(op);
      op = expr.as<CallNode>();
      CHECK_EQ(op->args.size(), 5U);
      const VarNode* buffer_var = op->args[1].as<VarNode>();
      Var var(GetRef<Var>(buffer_var));
      const IntImmNode* flag = op->args[4].as<IntImmNode>();
      if ((flag->value & 1) && sync_scope_.rank == StorageRank::kGlobal &&
          GetScope(buffer_var).rank == StorageRank::kGlobal) {
        ++rw_stats_[var].read_count;
      }
      if (flag->value & 2 && sync_scope_.rank == StorageRank::kGlobal &&
          GetScope(buffer_var).rank == StorageRank::kGlobal) {
        ++rw_stats_[var].write_count;
      }
      return expr;
    } else {
      return StmtExprMutator::VisitExpr_(op);
    }
  }

 private:
  // RW statistics about data
  struct Entry {
    int read_count{0};
    int write_count{0};
  };
  // Get current storage scope.
  StorageScope GetScope(const VarNode* buf) const {
    auto it = storage_scope_.find(buf);
    StorageScope s;
    s.rank = StorageRank::kGlobal;
    if (it == storage_scope_.end()) return s;
    return it->second;
  }
  // private functions.
  Stmt InitGlobalBarrier(const AttrStmtNode* op) {
    CHECK(op != nullptr);
    Array<PrimExpr> pargs = {StringImmNode::make(runtime::symbol::tvm_prepare_global_barrier)};
    Stmt prep = EvaluateNode::make(
        CallNode::make(DataType::Int(32), intrinsic::tvm_call_packed, pargs, CallNode::Intrinsic));
    Stmt body = op->body;
    for (const auto& kv : rw_stats_) {
      const auto& e = kv.second;
      if (e.read_count != 0 && e.write_count != 0) {
        body = AttrStmtNode::make(kv.first, attr::volatile_scope, 1, body);
      }
    }
    rw_stats_.clear();
    Stmt kinit = EvaluateNode::make(
        CallNode::make(
            DataType::Int(32),
            intrinsic::tvm_global_barrier_kinit, {}, CallNode::Intrinsic));
    body = SeqStmt({kinit, body});
    body = AttrStmtNode::make(
        op->node, op->attr_key, op->value, body);
    return SeqStmt({prep, body});
  }
  Stmt MakeGlobalBarrier() {
    CHECK(sync_scope_.rank == StorageRank::kGlobal);
    if (!num_blocks_.defined()) {
      CHECK(!is_lead_.defined());
      num_work_dim_ = thread_extents_.size();
      for (const AttrStmtNode* attr : thread_extents_) {
        IterVar iv = Downcast<IterVar>(attr->node);
        runtime::ThreadScope s = runtime::ThreadScope::make(iv->thread_tag);
        if (s.rank == 0) {
          num_blocks_ = (num_blocks_.defined() ?
                         attr->value * num_blocks_ : attr->value);
        } else if (s.rank == 1) {
          PrimExpr cond = iv->var == make_zero(iv->var.dtype());
          is_lead_ = is_lead_.defined() ? (is_lead_ && cond) : cond;
        }
      }
    } else {
      CHECK_EQ(num_work_dim_, thread_extents_.size());
    }
    return EvaluateNode::make(
        CallNode::make(DataType::Int(32), intrinsic::tvm_storage_sync,
                   {StringImmNode::make(sync_scope_.to_string()),
                    is_lead_, num_blocks_},
                   CallNode::Intrinsic));
  }
  // data structure.
  StorageScope sync_scope_;
  const std::unordered_set<const Object*>& syncs_;
  // The storage scope of each buffer
  std::unordered_map<const VarNode*, StorageScope> storage_scope_;
  // The read write statistics of storage
  std::unordered_map<Var, Entry, ObjectHash, ObjectEqual> rw_stats_;
  // The statistics for global barrier
  bool in_thread_env_{false};
  // memorized results
  std::vector<const AttrStmtNode*> thread_extents_;
  size_t num_work_dim_{0};
  PrimExpr num_blocks_;
  PrimExpr is_lead_;
};

Stmt ThreadSync(Stmt stmt, std::string storage_scope) {
  StorageScope sync_scope = StorageScope::make(storage_scope);
  ThreadSyncPlanner planner(sync_scope);
  planner(stmt);
  return ThreadSyncInserter(sync_scope, planner.syncs_inserted_)(std::move(stmt));
}

namespace transform {

Pass ThreadSync(std::string storage_scope) {
  auto pass_func = [storage_scope](PrimFunc f, IRModule m, PassContext ctx) {
    auto* n = f.CopyOnWrite();
    n->body = ThreadSync(std::move(n->body), storage_scope);
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.ThreadSync", {});
}

TVM_REGISTER_GLOBAL("tir.transform.ThreadSync")
.set_body_typed(ThreadSync);

}  // namespace transform
}  // namespace tir
}  // namespace tvm
