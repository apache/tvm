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
#include <tvm/runtime/registry.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <unordered_map>
#include <unordered_set>

#include "../../runtime/thread_storage_scope.h"
#include "../../tl/op/builtin.h"
#include "ir_utils.h"
#include "storage_access.h"

namespace tvm {
namespace tir {

class ThreadPartialSyncPlanner : public StorageAccessVisitor {
 public:
  explicit ThreadPartialSyncPlanner(StorageScope sync_scope) : sync_scope_(sync_scope) {}

  // The syncs inserted before each statement
  std::unordered_set<const Object*> syncs_inserted_;
  std::unordered_map<const Object*, int> partial_syncs_inserted_;

 protected:
  bool Enabled(const VarNode* buf, const StorageScope& scope) const final {
    return in_device_env() && scope == sync_scope_;
  }
  // Plan the sync
  std::vector<AccessEntry> Summarize(std::vector<StmtEntry> seq, const ForNode* loop) final {
    // Redirect all "shared.dyn" buffer access to the same buffer var
    // so that the accesses can be planned together.
    Var shared_dyn_buf;
    // for (StmtEntry& entry : seq) {
    //   for (AccessEntry& access : entry.access) {
    //     if (access.scope.rank == StorageRank::kShared && access.scope.tag == ".dyn" &&
    //         access.buffer.defined()) {
    //       if (!shared_dyn_buf.defined()) {
    //         shared_dyn_buf = access.buffer;
    //       } else {
    //         access.buffer = shared_dyn_buf;
    //       }
    //     }
    //   }
    // }

    // Unsynced reads and writes
    std::vector<AccessEntry> reads;
    std::vector<AccessEntry> writes;
    // if it is a loop, rotate two times to consider effect of loop.
    // simulation based approach to find dependencies
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
            sync_before_stmt = true;
            break;
          }
        } else if (acc.type == kWrite) {
          if (FindConflict(reads, acc, false)) {
            sync_before_stmt = true;
            break;
          }
        } else if (acc.type == kSync) {
          reads.clear();
          writes.clear();
        }
      }
      // If sync is inserted. remove the irrelevant things.
      if (sync_before_stmt) {
        reads.clear();
        writes.clear();
      }
      // Add the read/write of current statement
      for (const AccessEntry& acc : s.access) {
        if (acc.type == kRead) {
          reads.push_back(acc);
        } else if (acc.type == kWrite) {
          writes.push_back(acc);
        } else if (acc.type == kSync) {
          reads.clear();
          writes.clear();
        }
      }
      if (sync_before_stmt) {
        insert_syncs(s.stmt);
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
              sync_before_stmt = true;
              break;
            }
          } else if (acc.type == kWrite) {
            if (FindConflict(reads, acc, true)) {
              sync_before_stmt = true;
              break;
            }
          } else if (acc.type == kSync) {
            reads.clear();
            writes.clear();
          }
        }
        if (sync_before_stmt) {
          insert_syncs(s.stmt);
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
  bool FindConflict(const std::vector<AccessEntry>& prev, const AccessEntry& curr,
                    bool loop_carry) {
    for (const AccessEntry& x : prev) {
      if (FindConflict(x, curr, loop_carry)) {
        return true;
      }
    }
    return false;
  }

  bool FindConflict(const AccessEntry& prev, const AccessEntry& curr, bool loop_carry) {
    // Access to different buffers does not conflict.
    if (!prev.buffer.same_as(curr.buffer)) {
      return false;
    }

    // Assumes no race between threads
    // Same index value means no conflicts
    // TODO(tqchen) more standard set based testing.
    bool has_same_index = true;
    // Even if access has the same index, those indices need to
    // depend on the innermost thread id to avoid race condition
    bool depends_on_thread_index = true;
    const VarNode* thread_index_var = nullptr;
    if (!curr.threads.empty()) {
      thread_index_var = curr.threads.back()->var.get();
    }

    for (size_t i = 0; i < prev.touched.size(); i++) {
      const auto& prev_intset = prev.touched[i];
      const auto& curr_intset = curr.touched[i];

      if (prev_intset.IsSinglePoint() && curr_intset.IsSinglePoint()) {
        PrimExpr prev_index = prev_intset.PointValue();
        PrimExpr curr_index = curr_intset.PointValue();
        has_same_index = ExprDeepEqual()(prev_index, curr_index);
        if (thread_index_var != nullptr) {
          auto f_uses_thread_index = [=](const tvm::tir::VarNode* parameter) {
            return parameter == thread_index_var;
          };
          depends_on_thread_index = depends_on_thread_index &&
                                    UsesVar(curr_index, f_uses_thread_index) &&
                                    UsesVar(prev_index, f_uses_thread_index);
        }
      } else {
        has_same_index = false;
      }

      if (!(has_same_index && depends_on_thread_index)) {
        break;
      }
    }
    if (has_same_index && depends_on_thread_index) {
      return false;
    }

    // If this is a read into a double buffer that was previously
    // swapped out, then it doesn't conflict.
    if (prev.double_buffer_write && curr.type == kRead && !loop_carry) {
      return false;
    }

    // If nothing else allows sharing the same buffer, then they are
    // in conflict.
    return true;
  }

  void VisitStmt_(const AttrStmtNode* op) final {
    if (op->attr_key == "kWarpSpecializationScope") {
      IfThenElse body = Downcast<IfThenElse>(op->body);
      auto partitions = Downcast<Array<IntImm>>(op->node);
      ICHECK(partitions.size() == 2);

      scope_.push_back(std::vector<StmtEntry>());
      num_partial_threads_ = partitions[0];
      this->VisitStmt(body->then_case);
      StmtEntry s;
      s.stmt = op;
      s.access = Summarize(std::move(scope_.back()), nullptr);
      scope_.pop_back();

      num_partial_threads_ = partitions[1];
      scope_.push_back(std::vector<StmtEntry>());
      VisitStmt(body->else_case.value());
      auto v = Summarize(std::move(scope_.back()), nullptr);
      scope_.pop_back();
      s.access.insert(s.access.end(), v.begin(), v.end());

      num_partial_threads_ = NullOpt;
    } else {
      StorageAccessVisitor::VisitStmt_(op);
    }
  }

  void insert_syncs(const Object* obj) {
    // ICHECK_EQ(condition_counter(), 0) << "Cannot insert syncs inside condition";
    if (syncs_inserted_.count(obj)) return;
    if (num_partial_threads_.defined()) {
      syncs_inserted_.insert(obj);
      partial_syncs_inserted_[obj] = static_cast<int>(num_partial_threads_.value()->value);
    } else {
      syncs_inserted_.insert(obj);
    }
  }

 private:
  Optional<IntImm> num_partial_threads_;
  // synchronization scope
  StorageScope sync_scope_;
};

// There are cases where necessary syncthreads is not inserted by ThreadPartialSyncInserter.
// For example, syncthreads is needed after async_wait_queue in the second loop below,
// but since ThreadPartialSyncInserter is not aware of the asynchronous semantics, it cannot tell
// that the syncthreads is needed there.
//
// // Pipeline prologue
// for i in range(125):
//    async_commit_queue(0):
//       async_scope:
//          shared[(i + 3) % 4] = ...
// ...
//
// // Pipeline Epilogue
// for i in range(3):
//    async_wait_queue(0, 2 - i):
//       local[...] = shared[(i + 125) % 4]


class ThreadPartialSyncInserter : public StmtExprMutator {
 public:
  ThreadPartialSyncInserter(StorageScope sync_scope, const std::unordered_set<const Object*>& syncs,
                     std::unordered_map<const Object*, int> partial_syncs)
      : sync_scope_(sync_scope), syncs_(syncs), partial_syncs_(partial_syncs) {}

  Stmt VisitStmt(const Stmt& stmt) final {
    if (syncs_.size() == 0) return stmt;
    if (syncs_.count(stmt.get())) {
      Stmt barrier;
      if (partial_syncs_.count(stmt.get())) {
        auto iter = partial_syncs_.find(stmt.get());
        ICHECK(sync_scope_.rank == StorageRank::kShared);
        barrier = Evaluate(Call(DataType::Int(32), tl::SyncThreadsPartialOp(), {iter->second}));
      } else {
        return StmtExprMutator::VisitStmt(stmt);
      }
      // Mutate after query, to avoid stmt change.
      auto ret = StmtExprMutator::VisitStmt(stmt);
      ret = SeqStmt({barrier, ret});
      return ret;
    } else {
      return StmtExprMutator::VisitStmt(stmt);
    }
  }

 private:
  // data structure.
  StorageScope sync_scope_;
  const std::unordered_set<const Object*>& syncs_;
  const std::unordered_map<const Object*, int>& partial_syncs_;
};

Stmt ThreadPartialSync(Stmt stmt, std::string storage_scope) {
  StorageScope sync_scope = StorageScope::Create(storage_scope);
  ThreadPartialSyncPlanner planner(sync_scope);
  planner(stmt);
  return ThreadPartialSyncInserter(sync_scope, planner.syncs_inserted_,
                            planner.partial_syncs_inserted_)(std::move(stmt));
}

namespace transform {

Pass ThreadPartialSync(String storage_scope) {
  auto pass_func = [storage_scope](PrimFunc f, IRModule m, PassContext ctx) {
    auto* n = f.CopyOnWrite();
    n->body = ThreadPartialSync(std::move(n->body), storage_scope);
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.ThreadPartialSync", {});
}

TVM_REGISTER_GLOBAL("tir.transform.ThreadPartialSync").set_body_typed(ThreadPartialSync);

}  // namespace transform
}  // namespace tir
}  // namespace tvm
