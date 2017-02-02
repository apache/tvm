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
#include "../runtime/thread_storage_scope.h"

namespace tvm {
namespace ir {

using runtime::StorageScope;

class StorageSyncPlanner : public IRVisitor {
 public:
  explicit StorageSyncPlanner(StorageScope sync_scope)
    : sync_scope_(sync_scope) {}
  // only intended to be used once.
  // The syncs inserted before each statement
  std::unordered_set<const Node*> Plan(Stmt stmt) {
    CHECK_EQ(scope_.size(), 0U);
    scope_.push_back(std::vector<StmtEntry>());
    this->Visit(stmt);
    this->PlanSync(false);
    return std::move(syncs_inserted_);
  }
  void Visit_(const Load* op) final {
    CHECK(allow_load_);
    const Variable* buf = op->buffer_var.as<Variable>();
    StorageScope s = GetScope(buf);
    if (s == sync_scope_) {
      curr_stmt_.access.emplace_back(
          AccessEntry(buf, kRead, s));
    }
  }
  void Visit_(const Store* op) final {
    allow_load_ = true;
    CHECK_EQ(curr_stmt_.access.size(), 0U);
    curr_stmt_.stmt = op;
    const Variable* buf = op->buffer_var.as<Variable>();
    StorageScope s = GetScope(buf);
    if (s == sync_scope_) {
      curr_stmt_.access.emplace_back(
          AccessEntry(buf, kWrite, s));
    }
    // traverse child
    IRVisitor::Visit_(op);
    // push to the scope
    scope_.back().push_back(curr_stmt_);
    // clear access entry.
    curr_stmt_.access.clear();
    allow_load_ = false;
  }
  void Visit_(const AttrStmt* op) final {
    if (op->type_key == "storage_scope") {
      const Variable* buf = op->node.as<Variable>();
      storage_scope_[buf] =
          StorageScope::make(op->value.as<StringImm>()->value);
    }
    IRVisitor::Visit_(op);
  }
  void Visit_(const For* op) final {
    scope_.push_back(std::vector<StmtEntry>());
    IRVisitor::Visit_(op);
    StmtEntry s; s.stmt = op;
    s.access = PlanSync(true);
    scope_.pop_back();
    scope_.back().emplace_back(std::move(s));
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
  }

 private:
  // Storage access type
  enum AccessType {
    kRead,
    kWrite,
    kSync
  };
  // The access entry
  struct AccessEntry {
    /*! \brief The buffer variable, if any */
    const Variable* buffer{nullptr};
    /*! \brief The type of access */
    AccessType type;
    /*! \brief The storage scope */
    StorageScope scope;
    // constructor
    AccessEntry() {}
    AccessEntry(const Variable* buffer,
                AccessType type,
                StorageScope scope)
        : buffer(buffer), type(type), scope(scope) {}
  };
  // The statment entry
  struct StmtEntry {
    // the associated statement.
    const Node* stmt;
    std::vector<AccessEntry> access;
  };
  // Get current storage scope.
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
          head.push_back(AccessEntry(nullptr, kSync, sync_scope_));
        }
        ++sync_count;
      }
      for (const AccessEntry& acc : s.access) {
        if (acc.type == kSync) {
          if (sync_count != 0) {
            tail.clear();
          } else {
            head.push_back(AccessEntry(nullptr, kSync, sync_scope_));
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
  static bool FindConflict(const std::vector<AccessEntry>& vec,
                           const AccessEntry& e) {
    for (const AccessEntry& x : vec) {
      if (x.buffer == e.buffer) return true;
    }
    return false;
  }
  // Whether we are inside condition.
  int condition_counter_{0};
  // whether load is enabled.
  bool allow_load_{false};
  // the current free stmt entry.
  StmtEntry curr_stmt_;
  // access scope
  std::vector<std::vector<StmtEntry> > scope_;
  // The storage scope of each buffer
  std::unordered_map<const Variable*, StorageScope> storage_scope_;
  // The syncs inserted before each statement
  std::unordered_set<const Node*> syncs_inserted_;
  // The sync scope we care about.
  StorageScope sync_scope_;
};

class StorageSyncInserter : public IRMutator {
 public:
  StorageSyncInserter(StorageScope sync_scope,
                      std::unordered_set<const Node*> syncs)
      : sync_scope_(sync_scope), syncs_(syncs) {}

  Stmt Mutate(Stmt stmt) final {
    stmt = IRMutator::Mutate(stmt);
    if (syncs_.count(stmt.get())) {
      stmt = Block::make(
          Evaluate::make(
              Call::make(Int(32), intrinsic::tvm_storage_sync,
                         {StringImm::make(sync_scope_.to_string())},
                         Call::Intrinsic)),
          stmt);
    }
    return stmt;
  }

  StorageScope sync_scope_;
  std::unordered_set<const Node*> syncs_;
};

Stmt StorageSync(Stmt stmt, std::string storage_scope) {
  StorageScope sync_scope = StorageScope::make(storage_scope);
  auto syncs = StorageSyncPlanner(sync_scope).Plan(stmt);
  return StorageSyncInserter(sync_scope, syncs).Mutate(stmt);
}

LoweredFunc StorageSync(LoweredFunc f, std::string storage_scope) {
  auto n = std::make_shared<LoweredFuncNode>(*f.operator->());
  n->body = StorageSync(f->body, storage_scope);
  return LoweredFunc(n);
}

}  // namespace ir
}  // namespace tvm
