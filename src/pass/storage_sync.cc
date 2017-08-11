/*!
 *  Copyright (c) 2017 by Contributors
 * \file storage_sync.cc
 */
#include <tvm/ir.h>
#include <tvm/ir_pass.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_visitor.h>
#include <unordered_map>
#include <unordered_set>
#include "./ir_util.h"
#include "./storage_access.h"
#include "../runtime/thread_storage_scope.h"

namespace tvm {
namespace ir {

class ThreadSyncPlanner : public StorageAccessVisitor {
 public:
  explicit ThreadSyncPlanner(StorageScope sync_scope)
      : sync_scope_(sync_scope) {}

    // The syncs inserted before each statement
  std::unordered_set<const Node*> syncs_inserted_;

 protected:
  bool Enabled(const Variable* buf,
               const StorageScope& scope) const final {
    return in_device_env() && scope == sync_scope_;
  }
  // Plan the sync
  std::vector<AccessEntry> Summarize(
      std::vector<StmtEntry> seq, const For* loop) final {
    // Unsynced reads and writes
    std::vector<AccessEntry> reads;
    std::vector<AccessEntry> writes;

    // if it is a loop, rotate two times to consider effect of loop.
    size_t max_seq = seq.size();
    if (loop != nullptr) max_seq *= 2;
    // simulation based approach to find dependenceies
    for (size_t i = 0; i < max_seq; ++i) {
      const StmtEntry& s = seq[i % seq.size()];
      // check if sync before statement is needed.
      bool sync_before_stmt = (syncs_inserted_.count(s.stmt) != 0);
      // Apply the syncs added already.
      if (sync_before_stmt) {
        reads.clear();
        writes.clear();
      }
      for (const AccessEntry& acc : s.access) {
        if (acc.type == kRead) {
          if (FindConflict(writes, acc)) {
            sync_before_stmt = true; break;
          }
        } else if (acc.type == kWrite) {
          if (FindConflict(reads, acc)) {
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
    return head;
  }

 private:
  // find conflicting entry in vec.
  bool FindConflict(const std::vector<AccessEntry>& vec,
                    const AccessEntry& e) {
    for (const AccessEntry& x : vec) {
      if (x.buffer.same_as(e.buffer)) {
        // Assumes no race between threads
        // Same index value means no conflicts
        // TODO(tqchen) more standard set based testing.
        if (e.touched.is_single_point() &&
            x.touched.is_single_point()) {
          if (Equal(e.touched.point_value(),
                    x.touched.point_value())) continue;
        }
        return true;
      }
    }
    return false;
  }

 private:
  // synchronization scope
  StorageScope sync_scope_;
};

class ThreadSyncInserter : public IRMutator {
 public:
  ThreadSyncInserter(StorageScope sync_scope,
                     const std::unordered_set<const Node*>& syncs)
      : sync_scope_(sync_scope), syncs_(syncs) {}

  Stmt Mutate(Stmt stmt) final {
    if (syncs_.size() == 0) return stmt;
    stmt = IRMutator::Mutate(stmt);
    if (syncs_.count(stmt.get())) {
      Stmt barrier;
      if (sync_scope_.rank == 0) {
        barrier = MakeGlobalBarrier();
      } else {
        barrier = Evaluate::make(
                Call::make(Int(32), intrinsic::tvm_storage_sync,
                           {StringImm::make(sync_scope_.to_string())},
                           Call::Intrinsic));
      }
      stmt = Block::make(barrier, stmt);
    }
    return stmt;
  }
  Expr Mutate_(const Load* op, const Expr& e) final {
    if (sync_scope_.rank == 0 &&
        GetScope(op->buffer_var.get()).rank == 0) {
      ++rw_stats_[op->buffer_var].read_count;
    }
    return IRMutator::Mutate_(op, e);
  }
  Stmt Mutate_(const Store* op, const Stmt& s) final {
    if (sync_scope_.rank == 0 &&
        GetScope(op->buffer_var.get()).rank == 0) {
      ++rw_stats_[op->buffer_var].write_count;
    }
    return IRMutator::Mutate_(op, s);
  }
  Stmt Mutate_(const AttrStmt* op, const Stmt& s) final {
    if (op->attr_key == attr::thread_extent) {
      bool temp = true;
      std::swap(temp, in_thread_env_);
      thread_extents_.push_back(op);
      Stmt ret = IRMutator::Mutate_(op, s);
      thread_extents_.pop_back();
      std::swap(temp, in_thread_env_);
      // first thread scope.
      if (!in_thread_env_ && sync_scope_.rank == 0) {
        ret = InitGlobalBarrier(ret.as<AttrStmt>());
        num_blocks_ = Expr();
        is_lead_ = Expr();
      }
      return ret;
    } else if (op->attr_key == attr::storage_scope) {
      const Variable* buf = op->node.as<Variable>();
      storage_scope_[buf] =
          StorageScope::make(op->value.as<StringImm>()->value);
      return IRMutator::Mutate_(op, s);
    } else {
      return IRMutator::Mutate_(op, s);
    }
  }

 private:
  // RW statistics about data
  struct Entry {
    int read_count{0};
    int write_count{0};
  };
  // Get current storage scope.
  StorageScope GetScope(const Variable* buf) const {
    auto it = storage_scope_.find(buf);
    StorageScope s; s.rank = 0;
    if (it == storage_scope_.end()) return s;
    return it->second;
  }
  // private functions.
  Stmt InitGlobalBarrier(const AttrStmt* op) {
    CHECK(op != nullptr);
    Array<Expr> pargs = {StringImm::make(runtime::symbol::tvm_prepare_global_barrier)};
    Stmt prep = Evaluate::make(
        Call::make(Int(32), intrinsic::tvm_call_packed, pargs, Call::Intrinsic));
    Stmt body = op->body;
    for (const auto& kv : rw_stats_) {
      const auto& e = kv.second;
      if (e.read_count != 0 && e.write_count != 0) {
        body = AttrStmt::make(kv.first, attr::volatile_scope, 1, body);
      }
    }
    rw_stats_.clear();
    Stmt kinit = Evaluate::make(
        Call::make(Int(32), intrinsic::tvm_global_barrier_kinit, {}, Call::Intrinsic));
    body = Block::make(kinit, body);
    body = AttrStmt::make(
        op->node, op->attr_key, op->value, body);
    return Block::make(prep, body);
  }
  Stmt MakeGlobalBarrier() {
    CHECK_EQ(sync_scope_.rank, 0);
    if (!num_blocks_.defined()) {
      CHECK(!is_lead_.defined());
      num_work_dim_ = thread_extents_.size();
      for (const AttrStmt* attr : thread_extents_) {
        IterVar iv(attr->node.node_);
        runtime::ThreadScope s = runtime::ThreadScope::make(iv->thread_tag);
        if (s.rank == 0) {
          num_blocks_ = (num_blocks_.defined() ?
                         attr->value * num_blocks_ : attr->value);
        } else if (s.rank == 1) {
          Expr cond = iv->var == make_zero(iv->var.type());
          is_lead_ = is_lead_.defined() ? (is_lead_ && cond) : cond;
        }
      }
    } else {
      CHECK_EQ(num_work_dim_, thread_extents_.size());
    }
    return Evaluate::make(
        Call::make(Int(32), intrinsic::tvm_storage_sync,
                   {StringImm::make(sync_scope_.to_string()),
                    is_lead_, num_blocks_},
                   Call::Intrinsic));
  }
  // data structure.
  StorageScope sync_scope_;
  const std::unordered_set<const Node*>& syncs_;
  // The storage scope of each buffer
  std::unordered_map<const Variable*, StorageScope> storage_scope_;
  // The read write statistics of storage
  std::unordered_map<VarExpr, Entry, NodeHash, NodeEqual> rw_stats_;
  // The statistics for global barrier
  bool in_thread_env_{false};
  // memorized results
  std::vector<const AttrStmt*> thread_extents_;
  size_t num_work_dim_{0};
  Expr num_blocks_;
  Expr is_lead_;
};

Stmt ThreadSync(Stmt stmt, std::string storage_scope) {
  StorageScope sync_scope = StorageScope::make(storage_scope);
  ThreadSyncPlanner planner(sync_scope);
  planner.Visit(stmt);
  return ThreadSyncInserter(sync_scope, planner.syncs_inserted_).Mutate(stmt);
}

LoweredFunc ThreadSync(LoweredFunc f, std::string storage_scope) {
  CHECK_NE(f->func_type, kHostFunc);
  auto n = std::make_shared<LoweredFuncNode>(*f.operator->());
  n->body = ThreadSync(f->body, storage_scope);
  return LoweredFunc(n);
}

// Visitor to find touched set by co-processor scope.
class CoProcTouchedBuffer : public IRVisitor {
 public:
  void Visit_(const Load* op) final {
    if (in_scope_) {
      touched_.insert(op->buffer_var.get());
    }
    IRVisitor::Visit_(op);
  }
  void Visit_(const Store* op) final {
    if (in_scope_) {
      touched_.insert(op->buffer_var.get());
    }
    IRVisitor::Visit_(op);
  }
  void Visit_(const Call* op) final {
    if (op->is_intrinsic(intrinsic::tvm_access_ptr)) {
      const Variable* buffer = op->args[1].as<Variable>();
      touched_.insert(buffer);
    }
    IRVisitor::Visit_(op);
  }
  void Visit_(const AttrStmt* op) final {
    if (op->attr_key == attr::coproc_scope && !in_scope_) {
      in_scope_ = true;
      IterVar iv(op->node.node_);
      coproc_.insert(iv);
      IRVisitor::Visit_(op);
      in_scope_ = false;
    } else {
      IRVisitor::Visit_(op);
    }
  }

  std::unordered_set<const Variable*> touched_;
  std::unordered_set<IterVar> coproc_;

 private:
  bool in_scope_{false};
};

// Synchronization planning with co-processor.
class CoProcSyncPlanner : public StorageAccessVisitor {
 public:
  void Plan(const Stmt& stmt) {
    CoProcTouchedBuffer visitor;
    visitor.Visit(stmt);
    touched_ = std::move(visitor.touched_);
    if (!touched_.empty()) {
      this->Visit(stmt);
      PlanWriteSync(scope_.back(), nullptr, true);
      CHECK_EQ(visitor.coproc_.size(), 1U);
      if (write_sync_.size() == 0) {
        write_sync_[stmt.get()] = GetWriteSync(
            (*visitor.coproc_.begin())->var->name_hint + ".coproc_sync");
      }
    }
  }

  // Write synchronization to be inserted before or after stmt.
  std::unordered_map<const Node*, std::vector<Stmt> > write_sync_;

 protected:
  bool Enabled(const Variable* buf,
               const StorageScope& scope) const final {
    return touched_.count(buf) && scope == global_scope_;
  }

  // Plan the sync
  std::vector<AccessEntry> Summarize(
      std::vector<StmtEntry> seq, const For* loop) final {
    return PlanWriteSync(seq, loop, false);
  }

 private:
  // Plan write synchronization if write is not coherent
  std::vector<AccessEntry> PlanWriteSync(
      std::vector<StmtEntry> seq, const For* loop,
      bool force_sync_at_end) {
    // detect write barriers
    // access by the co-processor.
    std::vector<AccessEntry> co_access;
    bool contain_sync = false;

    auto find_conflict = [&](const AccessEntry& acc) {
      for (const AccessEntry& x : co_access) {
        if (x.buffer.same_as(acc.buffer) &&
            ((acc.type == kRead && x.type == kWrite) ||
             acc.type == kWrite)) {
          return true;
        }
      }
      return false;
    };
    for (size_t i = 0; i < seq.size(); ++i) {
      const StmtEntry& s = seq[i];
      bool sync_write = false;
      for (const AccessEntry& acc : s.access) {
        if (acc.threads.size() == 0 && find_conflict(acc)) {
          sync_write = true; break;
        }
        if (acc.type == kSync) {
          co_access.clear();
          contain_sync = true;
        }
      }
      if (sync_write) {
        CHECK_NE(i, 0U);
        write_sync_[seq[i - 1].stmt] = GetWriteSync(co_access);
        co_access.clear();
        contain_sync = true;
      }
      for (const AccessEntry& acc : s.access) {
        if (acc.threads.size() != 0) {
          co_access.push_back(acc);
        }
      }
    }
    bool sync_at_end = force_sync_at_end;
    if (loop != nullptr && !sync_at_end) {
      // loop carray dependency
      for (size_t i = 0; i < seq.size(); ++i) {
        const StmtEntry& s = seq[i];
        for (const AccessEntry& acc : s.access) {
          if (acc.threads.size() == 0 && find_conflict(acc)) {
            sync_at_end = true; break;
          }
        }
        if (write_sync_.count(s.stmt) || sync_at_end) break;
      }
    }
    if (sync_at_end && co_access.size() != 0) {
      CHECK_NE(seq.size(), 0);
      contain_sync = true;
      write_sync_[seq.back().stmt] = GetWriteSync(co_access);
      co_access.clear();
    }
    if (contain_sync) {
      AccessEntry e;
      e.type = kSync;
      e.scope = global_scope_;
      co_access.insert(co_access.begin(), e);
    }
    return co_access;
  }
  // Add write Synchronization
  std::vector<Stmt> GetWriteSync(const std::vector<AccessEntry>& co_access) {
    // Does not consider memory coherence, need runtime.
    CHECK_NE(co_access.size(), 0U);
    CHECK_EQ(co_access[0].threads.size(), 1U);
    return GetWriteSync(co_access[0].threads[0]->var->name_hint + ".coproc_sync");
  }

  std::vector<Stmt> GetWriteSync(std::string sync_name) {
    std::vector<Stmt> stmts;
    stmts.emplace_back(
      Evaluate::make(Call::make(
          Int(32),
          sync_name,
          {}, Call::Intrinsic)));
    return stmts;
  }

  std::unordered_set<const Variable*> touched_;
  StorageScope global_scope_ = StorageScope::make("global");
};

class CoProcSyncInserter : public IRMutator {
 public:
  explicit CoProcSyncInserter(
      const std::unordered_map<const Node*, std::vector<Stmt> >& write_sync)
      : write_sync_(write_sync) {}

  Stmt Mutate(Stmt stmt) final {
    stmt = IRMutator::Mutate(stmt);
    auto it = write_sync_.find(stmt.get());
    if (it != write_sync_.end()) {
      stmt = Block::make(stmt, MergeSeq(it->second));
    }
    return stmt;
  }

 private:
  const std::unordered_map<const Node*, std::vector<Stmt> >& write_sync_;
};

Stmt CoProcSync(Stmt stmt) {
  CoProcSyncPlanner planner;
  planner.Plan(stmt);
  if (planner.write_sync_.size() != 0) {
    return CoProcSyncInserter(planner.write_sync_).Mutate(stmt);
  } else {
    return stmt;
  }
}

}  // namespace ir
}  // namespace tvm
