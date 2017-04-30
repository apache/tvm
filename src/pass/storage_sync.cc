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

using namespace storage;

class StorageSyncPlanner : public IRVisitor {
 public:
  explicit StorageSyncPlanner(StorageScope sync_scope)
    : sync_scope_(sync_scope) {}
  void Visit_(const Load* op) final {
    if (!in_device_env_) return;
    CHECK(allow_load_);
    const Variable* buf = op->buffer_var.as<Variable>();
    StorageScope s = GetScope(buf);
    if (s == sync_scope_) {
      curr_stmt_.access.emplace_back(
          AccessEntry(buf, op->index, kRead, s));
    }
  }
  void Visit_(const Store* op) final {
    if (!in_device_env_) return;
    allow_load_ = true;
    CHECK_EQ(curr_stmt_.access.size(), 0U);
    curr_stmt_.stmt = op;
    const Variable* buf = op->buffer_var.as<Variable>();
    StorageScope s = GetScope(buf);
    if (s == sync_scope_) {
      curr_stmt_.access.emplace_back(
          AccessEntry(buf, op->index, kWrite, s));
    }
    // traverse child
    IRVisitor::Visit_(op);
    // push to the scope
    scope_.back().push_back(curr_stmt_);
    // clear access entry.
    curr_stmt_.access.clear();
    allow_load_ = false;
  }
  void Visit_(const Evaluate* op) final {
    if (!in_device_env_) return;
    if (const Call* call = op->value.as<Call>()) {
      if (call->is_intrinsic(intrinsic::tvm_storage_sync)) {
        const std::string& s = call->args[0].as<StringImm>()->value;
        if (s != "warp") {
          StorageScope scope = StorageScope::make(s);
          if (scope.rank <= sync_scope_.rank) {
            CHECK_EQ(curr_stmt_.access.size(), 0U);
            curr_stmt_.access.emplace_back(
                AccessEntry(nullptr, Expr(), kSync, scope));
            // push to the scope
            scope_.back().push_back(curr_stmt_);
            curr_stmt_.access.clear();
          }
        }
      }
    }
  }
  void Visit_(const AttrStmt* op) final {
    if (op->attr_key == attr::storage_scope) {
      const Variable* buf = op->node.as<Variable>();
      storage_scope_[buf] =
          StorageScope::make(op->value.as<StringImm>()->value);
      IRVisitor::Visit_(op);
    } else if (op->attr_key == attr::thread_extent && !in_device_env_) {
      in_device_env_ = true;
      CHECK_EQ(scope_.size(), 0U);
      scope_.push_back(std::vector<StmtEntry>());
      IRVisitor::Visit_(op);
      this->PlanSync(false);
      in_device_env_ = false;
      scope_.pop_back();
    } else {
      IRVisitor::Visit_(op);
    }
  }
  void Visit_(const For* op) final {
    if (in_device_env_) {
      scope_.push_back(std::vector<StmtEntry>());
      IRVisitor::Visit_(op);
      StmtEntry s; s.stmt = op;
      s.access = PlanSync(true);
      scope_.pop_back();
      scope_.back().emplace_back(std::move(s));
    } else {
      IRVisitor::Visit_(op);
    }
  }
  void Visit_(const Call* op) final {
    if (op->is_intrinsic(Call::address_of)) {
      const Load *l = op->args[0].as<Load>();
      IRVisitor::Visit_(l);
    } else {
      IRVisitor::Visit_(op);
    }
  }
  void Visit_(const IfThenElse* op) final {
    if (in_device_env_) {
      ++condition_counter_;
      this->Visit(op->condition);
      scope_.push_back(std::vector<StmtEntry>());
      this->Visit(op->then_case);

      StmtEntry s; s.stmt = op;
      s.access = PlanSync(false);
      scope_.pop_back();
      if (op->else_case.defined()) {
        scope_.push_back(std::vector<StmtEntry>());
        auto v = PlanSync(false);
        scope_.pop_back();
        s.access.insert(s.access.end(), v.begin(), v.end());
      }
      scope_.back().emplace_back(std::move(s));
      --condition_counter_;
    } else {
      IRVisitor::Visit_(op);
    }
  }

  // The syncs inserted before each statement
  std::unordered_set<const Node*> syncs_inserted_;

 private:
  // Get storage scope of buffer.
  StorageScope GetScope(const Variable* buf) const {
    auto it = storage_scope_.find(buf);
    StorageScope s; s.rank = 0;
    if (it == storage_scope_.end()) return s;
    return it->second;
  }
  // Plan the sync
  std::vector<AccessEntry> PlanSync(bool is_loop) {
    // unsynced reads and writes
    std::vector<AccessEntry> reads;
    std::vector<AccessEntry> writes;
    const std::vector<StmtEntry>& seq = scope_.back();

    // if it is a loop, rotate two times to consider effect of loop.
    size_t max_seq = seq.size();
    if (is_loop) max_seq *= 2;
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
        CHECK_EQ(condition_counter_, 0)
            << "Cannot insert syncs inside condition";
        syncs_inserted_.insert(s.stmt);
      }
    }
    // return the exposed entries, remove unecessary ones.
    int sync_count = 0;
    // head are before first sync, tail are after last sync
    std::vector<AccessEntry> head, tail;
    for (const StmtEntry& s : seq) {
      if (syncs_inserted_.count(s.stmt)) {
        if (sync_count != 0) {
          tail.clear();
        } else {
          head.push_back(AccessEntry(nullptr, Expr(), kSync, sync_scope_));
        }
        ++sync_count;
      }
      for (const AccessEntry& acc : s.access) {
        if (acc.type == kSync) {
          if (sync_count != 0) {
            tail.clear();
          } else {
            head.push_back(AccessEntry(nullptr, Expr(), kSync, sync_scope_));
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
  // find conflicting entry in vec.
  bool FindConflict(const std::vector<AccessEntry>& vec,
                    const AccessEntry& e) {
    for (const AccessEntry& x : vec) {
      if (x.buffer == e.buffer &&
          !e.index.same_as(x.index)) return true;
    }
    return false;
  }
  // Whether we are inside condition.
  int condition_counter_{0};
  // whether load is enabled.
  bool in_device_env_{false};
  // whether load is enabled.
  bool allow_load_{false};
  // the current free stmt entry.
  StmtEntry curr_stmt_;
  // access scope
  std::vector<std::vector<StmtEntry> > scope_;
  // The storage scope of each buffer
  std::unordered_map<const Variable*, StorageScope> storage_scope_;
  // The sync scope we care about.
  StorageScope sync_scope_;
};

class StorageSyncInserter : public IRMutator {
 public:
  StorageSyncInserter(StorageScope sync_scope,
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

Stmt StorageSync(Stmt stmt, std::string storage_scope) {
  StorageScope sync_scope = StorageScope::make(storage_scope);
  StorageSyncPlanner planner(sync_scope);
  planner.Visit(stmt);
  return StorageSyncInserter(sync_scope, planner.syncs_inserted_).Mutate(stmt);
}

LoweredFunc StorageSync(LoweredFunc f, std::string storage_scope) {
  auto n = std::make_shared<LoweredFuncNode>(*f.operator->());
  n->body = StorageSync(f->body, storage_scope);
  return LoweredFunc(n);
}

}  // namespace ir
}  // namespace tvm
