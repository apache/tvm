/*!
 *  Copyright (c) 2017 by Contributors
 * \file lift_allocate.cc
 */
#include <tvm/ir.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_pass.h>
#include <unordered_map>
#include "./ir_util.h"
#include "../runtime/thread_storage_scope.h"

namespace tvm {
namespace ir {

using runtime::StorageScope;
using runtime::ThreadScope;

class AllocateLifter : public IRMutator {
 public:
  Stmt Lift(Stmt stmt) {
    stmt = this->Mutate(stmt);
    StorageScope key; key.rank = 0;
    stmt = MergeNest(allocs_[key], stmt);
    return stmt;
  }

  Stmt Mutate_(const AttrStmt* op, const Stmt& s) final {
    CHECK(op->type_key != attr::virtual_thread)
        << "InjectVirtualThread before LiftStorageAlloc";
    if (op->type_key == attr::storage_scope) {
      StorageScope sc = StorageScope::make(op->value.as<StringImm>()->value);
      allocs_[sc].emplace_back(
          AttrStmt::make(
          op->node, attr::storage_scope,
            op->value, Evaluate::make(0)));
       storage_scope_[op->node.get()] = sc;
      return this->Mutate(op->body);
    } else if (op->type_key == attr::thread_extent) {
      IterVar iv(op->node.node_);
      ThreadScope ts = ThreadScope::make(iv->thread_tag);
      curr_thread_scope_.push_back(ts);
      Stmt stmt = IRMutator::Mutate_(op, s);
      curr_thread_scope_.pop_back();
      op = stmt.as<AttrStmt>();

      bool first_scope = true;
      for (const ThreadScope& t : curr_thread_scope_) {
        if (t.rank == ts.rank) first_scope = false;
      }
      if (first_scope) {
        StorageScope key;
        key.rank = ts.rank + 1;
        std::vector<Stmt>& vec = allocs_[key];
        if (vec.size() != 0) {
          Stmt body = MergeNest(vec, op->body);
          vec.clear();
          return AttrStmt::make(
              op->node, op->type_key, op->value, body);
          }
        }
      return stmt;
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const For* op, const Stmt& s) final {
    CHECK(op->for_type != ForType::Vectorized)
        << "VectorizeLoop before LiftStorageAlloc";
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const Allocate* op, const Stmt& s) final {
    auto it = storage_scope_.find(op->buffer_var.get());
    CHECK(it != storage_scope_.end());
    allocs_[it->second].emplace_back(
        Allocate::make(
            op->buffer_var, op->type, op->extents, op->condition,
            Evaluate::make(0)));
    return this->Mutate(op->body);
  }

 private:
  // storage scope of internal allocation.
  std::unordered_map<const Node*, StorageScope> storage_scope_;
  // The current thread scope.
  std::vector<ThreadScope> curr_thread_scope_;
  // The allocations by rank
  std::unordered_map<StorageScope, std::vector<Stmt> > allocs_;
};

Stmt LiftAllocate(Stmt stmt) {
  return AllocateLifter().Lift(stmt);
}

}  // namespace ir
}  // namespace tvm
