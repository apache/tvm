/*!
 * Copyright (c) 2017 by Contributors
 * \file storage_rewrite.cc
 * \brief Memory access pattern analysis and optimization.
 *  Re-write data access to enable memory sharing when possible.
 */
#include <tvm/ir.h>
#include <tvm/ir_pass.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_visitor.h>
#include <map>
#include <unordered_set>
#include <unordered_map>
#include "./ir_util.h"
#include "./storage_access.h"

namespace tvm {
namespace ir {

using namespace storage;
// Find a linear pattern of storage acess
// Composite scopes(loop/thread_launch/IfThen) is represented by two points:
// before_scope -> scope_body -> after_scope
//
// The linear_seq_ stores before_scope and after_scope.
// The access to the arrays are stored at the after_scope point.
//
// Define "scope" as the body of For/thread_launch/IfThenElse
// This pass tries to detect last point that we need to keep memory
// alive under the same scope as allocate.
// The storage need to be kept alive between allocate and last access.
// The free point is only inserted at the same scope of allocate.
//
class StorageAccessPatternFinder final : public IRVisitor {
 public:
  // Get linear access pattern.
  std::vector<StmtEntry> GetLinearSeq(const Stmt& s) {
    this->Visit(s);
    return std::move(linear_seq_);
  }
  void Visit_(const Allocate* op) final {
    CHECK(!in_parallel_env_)
        << "Allocation inside parallel is not yet handled.";
    size_t level = scope_.size();
    const Variable* buf = op->buffer_var.get();
    CHECK(!alloc_scope_level_.count(buf));
    alloc_scope_level_[buf] = level;
    StmtEntry e;
    e.stmt = op;
    e.access.emplace_back(
        AccessEntry(buf, Expr(), kAlloc, GetScope(buf)));
    linear_seq_.emplace_back(std::move(e));
    IRVisitor::Visit_(op);
  }
  void Visit_(const Store* op) final {
    scope_.push_back(StmtEntry());
    // visit subexpr
    IRVisitor::Visit_(op);
    // Add write access.
    const Variable* buf = op->buffer_var.get();
    auto it = alloc_scope_level_.find(buf);
    if (it != alloc_scope_level_.end()) {
      scope_[it->second].access.emplace_back(
        AccessEntry(buf, op->index, kWrite, GetScope(buf)));
    }
    StmtEntry e = scope_.back();
    scope_.pop_back();
    if (e.access.size() != 0) {
      e.stmt = op;
      linear_seq_.push_back(e);
    }
  }
  void Visit_(const Evaluate* op) final {
    scope_.push_back(StmtEntry());
    // visit subexpr
    IRVisitor::Visit_(op);
    StmtEntry e = scope_.back();
    scope_.pop_back();
    if (e.access.size() != 0) {
      e.stmt = op;
      linear_seq_.push_back(e);
    }
  }
  void Visit_(const Load* op) final {
    // Add write access.
    IRVisitor::Visit_(op);
    const Variable* buf = op->buffer_var.get();
    auto it = alloc_scope_level_.find(buf);
    if (it != alloc_scope_level_.end()) {
      CHECK_LT(it->second, scope_.size())
          << "Load memory in places other than store.";
      scope_[it->second].access.emplace_back(
          AccessEntry(buf, op->index, kRead, GetScope(buf)));
    }
  }
  void Visit_(const Variable* buf) final {
    // Directly reference to the variable count as a read.
    auto it = alloc_scope_level_.find(buf);
    if (it != alloc_scope_level_.end()) {
      CHECK_LT(it->second, scope_.size()) << " buf=" << buf->name_hint;
      scope_[it->second].access.emplace_back(
          AccessEntry(buf, Expr(), kOpaque, GetScope(buf)));
    }
  }
  template<typename T>
  void VisitNewScope(const T* op) {
    scope_.push_back(StmtEntry());
    StmtEntry e;
    e.stmt = op;
    // before scope.
    linear_seq_.push_back(e);
    IRVisitor::Visit_(op);
    // after scope.
    e.access = std::move(scope_.back().access);
    scope_.pop_back();
    linear_seq_.push_back(e);
  }
  void Visit_(const AttrStmt* op) final {
    // Only record the outer most thread extent.
    if (op->attr_key == attr::thread_extent && !in_thread_env_) {
      in_thread_env_ = true;
      VisitNewScope(op);
      in_thread_env_ = false;
    } else if (op->attr_key == attr::storage_scope) {
      const Variable* buf = op->node.as<Variable>();
      storage_scope_[buf] =
          StorageScope::make(op->value.as<StringImm>()->value);
      IRVisitor::Visit_(op);
    } else {
      IRVisitor::Visit_(op);
    }
  }
  void Visit_(const For* op) final {
    if (op->for_type == ForType::Parallel) {
      bool in_par = in_parallel_env_;
      in_parallel_env_ = true;
      VisitNewScope(op);
      in_parallel_env_ = in_par;
    } else {
      VisitNewScope(op);
    }
  }
  void Visit_(const IfThenElse* op) final {
    VisitNewScope(op);
  }

 private:
  // Get storage scope of buffer.
  StorageScope GetScope(const Variable* buf) const {
    auto it = storage_scope_.find(buf);
    CHECK(it != storage_scope_.end());
    return it->second;
  }
  // Whether already in thread env.
  bool in_thread_env_{false};
  // Whether already in parallel env.
  bool in_parallel_env_{false};
  // linearized access sequence.
  std::vector<StmtEntry> linear_seq_;
  // The scope stack.
  std::vector<StmtEntry> scope_;
  // The storage scope of each buffer
  std::unordered_map<const Variable*, StorageScope> storage_scope_;
  // buffer -> allocated scope level in the IR.
  std::unordered_map<const Variable*, size_t> alloc_scope_level_;
};

// Planner to plan and rewrite memory allocation.
class StoragePlanRewriter : public IRMutator {
 public:
  Stmt Rewrite(Stmt stmt) {
    std::vector<StmtEntry> seq =
        StorageAccessPatternFinder().GetLinearSeq(stmt);
    this->FindFreeLocation(seq);
    this->PlanMemory(seq);
    this->PrepareNewAlloc();
    stmt = this->Mutate(stmt);
    if (attach_map_.count(nullptr)) {
      std::vector<Stmt> nest;
      for (StorageEntry* e : attach_map_.at(nullptr)) {
        CHECK_EQ(e->scope.rank, 0);
        nest.emplace_back(AttrStmt::make(
            e->alloc_var, attr::storage_scope,
            StringImm::make(e->scope.to_string()),
            Evaluate::make(0)));
        nest.push_back(e->new_alloc);
      }
      stmt = MergeNest(nest, stmt);
    }
    return stmt;
  }
  Stmt Mutate_(const Store* op, const Stmt& s) final {
    Stmt stmt = IRMutator::Mutate_(op, s);
    op = stmt.as<Store>();
    auto it = alloc_map_.find(op->buffer_var.get());
    if (it == alloc_map_.end()) return stmt;
    return Store::make(it->second->alloc_var, op->value, op->index, op->predicate);
  }
  Expr Mutate_(const Load* op, const Expr& e) final {
    Expr expr = IRMutator::Mutate_(op, e);
    op = expr.as<Load>();
    auto it = alloc_map_.find(op->buffer_var.get());
    if (it == alloc_map_.end()) return expr;
    return Load::make(op->type, it->second->alloc_var, op->index, op->predicate);
  }
  Expr Mutate_(const Variable* op, const Expr& e) final {
    auto it = alloc_map_.find(op);
    if (it != alloc_map_.end()) {
      return it->second->alloc_var;
    } else {
      return e;
    }
  }
  Stmt Mutate_(const AttrStmt* op, const Stmt& s) final {
    CHECK(op->attr_key != attr::virtual_thread)
        << "InjectVirtualThread before StoragePlan";
    if (op->attr_key == attr::storage_scope) {
      return this->Mutate(op->body);
    } else if (op->attr_key == attr::thread_extent) {
      // remake all the allocation at the thread extent.
      if (attach_map_.count(op)) {
        std::vector<Stmt> nest;
        for (StorageEntry* e : attach_map_.at(op)) {
          nest.emplace_back(AttrStmt::make(
              e->alloc_var, attr::storage_scope,
              StringImm::make(e->scope.to_string()),
              Evaluate::make(0)));
          nest.push_back(e->new_alloc);
        }
        Stmt stmt = IRMutator::Mutate_(op, s);
        op = stmt.as<AttrStmt>();
        Stmt body = MergeNest(nest, op->body);
        return AttrStmt::make(
            op->node, op->attr_key, op->value, body);
      } else {
        return IRMutator::Mutate_(op, s);
      }
    } else if (op->attr_key == attr::volatile_scope) {
      Stmt stmt = IRMutator::Mutate_(op, s);
      op = stmt.as<AttrStmt>();
      auto it = alloc_map_.find(op->node.as<Variable>());
      if (it == alloc_map_.end()) return stmt;
      return AttrStmt::make(
          it->second->alloc_var, op->attr_key, op->value, op->body);
    } else {
      return IRMutator::Mutate_(op, s);
    }
  }
  Stmt Mutate_(const For* op, const Stmt& s) final {
    CHECK(op->for_type != ForType::Vectorized)
        << "VectorizeLoop before LiftStorageAlloc";
    return IRMutator::Mutate_(op, s);
  }
  Stmt Mutate_(const Allocate* op, const Stmt& s) final {
    return this->Mutate(op->body);
  }

 private:
  // Alllocate entry of node.
  struct StorageEntry {
    // The scope that this alloc attaches after
    // For shared/local memory it is beginning of the thread extent.
    // for global memory it is nullptr, means beginning of everything.
    const Node* attach_scope_{nullptr};
    // The constant size of the buffer in bytes, only used if it is constant.
    size_t const_size{0};
    // The storage scope.
    StorageScope scope;
    // Allocs that shares this entry.
    std::vector<const Allocate*> allocs;
    // The var expr of new allocation.
    VarExpr alloc_var;
    // The replacement allocation
    Stmt new_alloc;
  };
  // Prepare the new allocations
  void PrepareNewAlloc() {
    for (size_t i = 0; i < alloc_vec_.size(); ++i) {
      StorageEntry* e = alloc_vec_[i].get();
      // find the element with the most amount of bytes.
      Type t = e->allocs[0]->type;
      for (const Allocate* op : e->allocs) {
        if (op->type.bytes() * op->type.lanes() > t.bytes() * t.lanes()) {
          t = op->type;
        }
      }
      // Get the allocation size;
      e->alloc_var = e->allocs[0]->buffer_var;
      if (e->allocs.size() == 1) {
        // simply use the original allocation.
        e->new_alloc = Allocate::make(
            e->alloc_var, t, e->allocs[0]->extents,
            e->allocs[0]->condition, Evaluate::make(0));
      } else {
        // Build a merged allocation.
        int alloc_unit = t.bytes() * t.lanes();
        Expr combo_size;
        for (const Allocate* op : e->allocs) {
          // Get the size
          Expr sz = op->extents[0];
          for (size_t i = 1; i < op->extents.size(); ++i) {
            sz = sz * op->extents[i];
          }
          int bytes = op->type.bytes() * op->type.lanes();
          if (alloc_unit != bytes) {
            sz = (sz * make_const(sz.type(), bytes) +
                  make_const(sz.type(), alloc_unit - 1)) /
                make_const(sz.type(), alloc_unit);
          }
          if (combo_size.defined()) {
            combo_size = max(combo_size, sz);
          } else {
            combo_size = sz;
          }
        }
        combo_size = ir::Simplify(combo_size);
        e->new_alloc = Allocate::make(
            e->alloc_var, t, {combo_size}, const_true(),
            Evaluate::make(0));
      }
      attach_map_[e->attach_scope_].push_back(e);
    }
  }
  // Find the free location of each varaible.
  // Just do a reverse linear scan.
  void FindFreeLocation(const std::vector<StmtEntry>& seq) {
    std::unordered_set<const Variable*> touched;
    for (size_t i = seq.size(); i != 0; --i) {
      const StmtEntry& s = seq[i - 1];
      for (const AccessEntry& e : s.access) {
        if (!touched.count(e.buffer)) {
          touched.insert(e.buffer);
          free_loc_[i - 1].push_back(e.buffer);
        }
      }
    }
  }
  // Memory plan algorithm
  void PlanMemory(const std::vector<StmtEntry>& seq) {
    for (size_t i = 0; i < seq.size(); ++i) {
      const StmtEntry& s = seq[i];
      if (s.stmt->is_type<AttrStmt>()) {
        const auto* op = static_cast<const AttrStmt*>(s.stmt);
        CHECK_EQ(op->attr_key, attr::thread_extent);
        if (thread_scope_ != nullptr) {
          CHECK(thread_scope_ == op);
          // erase all non-global memory from constant free map.
          for (auto it = const_free_map_.begin();
               it != const_free_map_.end();) {
            if (it->second->scope.rank != 0) {
              it = const_free_map_.erase(it);
            } else {
              ++it;
            }
          }
          thread_scope_ = nullptr;
        } else {
          thread_scope_ = op;
        }
      } else if (s.stmt->is_type<Allocate>()) {
        const auto* op = static_cast<const Allocate*>(s.stmt);
        StorageEntry* e = this->FindAlloc(op, s.access[0].scope);
        e->allocs.emplace_back(op);
        alloc_map_[op->buffer_var.get()] = e;
      }
      // free list
      if (free_loc_.count(i)) {
        for (const Variable* var : free_loc_.at(i)) {
          this->Free(var);
        }
      }
    }
  }
  // Allocate new storage entry.
  StorageEntry* NewAlloc(const Allocate* op,
                         const StorageScope& scope,
                         size_t const_size) {
    // Re-use not successful, allocate a new buffer.
    std::unique_ptr<StorageEntry> entry(new StorageEntry());
    entry->attach_scope_ = thread_scope_;
    entry->scope = scope;
    entry->const_size = const_size;
    StorageEntry* e = entry.get();
    alloc_vec_.emplace_back(std::move(entry));
    return e;
  }
  StorageEntry* FindAlloc(const Allocate* op,
                          const StorageScope& scope) {
    // skip plan for local variable,
    // compiler can do a better job with register allocation.
    const size_t match_range = 16;
    size_t const_size = static_cast<size_t>(
        op->constant_allocation_size()) * op->type.bytes() * op->type.lanes();
    if (scope.rank > 1 || op->type.is_handle()) {
      return NewAlloc(op, scope, const_size);
    }
    // disable reuse of small arrays
    if (const_size > 0  && const_size <= 32) {
      return NewAlloc(op, scope, const_size);
    }
    if (const_size != 0) {
      // constant allocation.
      auto begin = const_free_map_.lower_bound(const_size / match_range);
      auto mid = const_free_map_.lower_bound(const_size);
      auto end = const_free_map_.upper_bound(const_size * match_range);
      for (auto it = mid; it != end; ++it) {
        StorageEntry *e = it->second;
        if (it->second->scope != scope) continue;
        e->const_size = std::max(const_size, e->const_size);
        const_free_map_.erase(it);
        return e;
      }
      for (auto it = mid; it != begin;) {
        --it;
        StorageEntry *e = it->second;
        if (it->second->scope != scope) continue;
        const_free_map_.erase(it);
        return e;
      }
    } else {
      // Simple strategy: round roubin.
      for (auto it = sym_free_list_.begin();
           it != sym_free_list_.end(); ++it) {
        StorageEntry* e = *it;
        if (e->scope != scope) continue;
        sym_free_list_.erase(it);
        return e;
      }
    }
    return NewAlloc(op, scope, const_size);
  }
  // simulated free.
  void Free(const Variable* var) {
    auto it = alloc_map_.find(var);
    CHECK(it != alloc_map_.end());
    StorageEntry* e = it->second;
    // Disable sharing of local memory.
    if (e->scope.rank > 1 || e->allocs[0]->type.is_handle()) return;
    // disable reuse of small arrays
    if (e->const_size > 0 && e->const_size <= 32) return;
    // normal free.
    if (e->const_size != 0) {
      const_free_map_.insert({e->const_size, e});
    } else {
      sym_free_list_.push_back(e);
    }
  }
  // thread scope.
  const Node* thread_scope_{nullptr};
  // Locations of free ops.
  std::unordered_map<size_t,
                     std::vector<const Variable*> > free_loc_;
  // The allocation attach map
  std::unordered_map<const Node*, std::vector<StorageEntry*> > attach_map_;
  // The allocation assign map
  std::unordered_map<const Variable*, StorageEntry*> alloc_map_;
  // constant size free map.
  std::multimap<size_t, StorageEntry*> const_free_map_;
  // symbolic free list, for non constant items.
  std::list<StorageEntry*> sym_free_list_;
  // The allocations
  std::vector<std::unique_ptr<StorageEntry> > alloc_vec_;
};

Stmt StorageRewrite(Stmt stmt) {
  return StoragePlanRewriter().Rewrite(stmt);
}
}  // namespace ir
}  // namespace tvm
