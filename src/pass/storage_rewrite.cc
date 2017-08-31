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
#include <tvm/target_info.h>
#include <map>
#include <unordered_set>
#include <unordered_map>
#include "./ir_util.h"
#include "../arithmetic/compute_expr.h"
#include "../runtime/thread_storage_scope.h"

namespace tvm {
namespace ir {

using runtime::StorageScope;

// Find a linear pattern of storage acess
// Used for liveness analysis.
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
class LinearAccessPatternFinder final : public IRVisitor {
 public:
  /*! \brief record the touch hist of statment. */
  struct StmtEntry {
    // The statment
    const Node* stmt;
    // Scope used for allocation.
    StorageScope alloc_scope;
    // The buffer variables this statment touched.
    std::vector<const Variable*> touched;
  };

  // Get linear access pattern.
  std::vector<StmtEntry> GetLinearSeq(const Stmt& s) {
    this->Visit(s);
    return std::move(linear_seq_);
  }
  void Visit_(const Allocate* op) final {
    size_t level = scope_.size();
    const Variable* buf = op->buffer_var.get();
    CHECK(!alloc_scope_level_.count(buf));
    alloc_scope_level_[buf] = level;
    StmtEntry e;
    e.stmt = op;
    e.alloc_scope = GetScope(buf);
    e.touched.push_back(buf);
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
      scope_[it->second].touched.push_back(buf);
    }
    StmtEntry e = scope_.back();
    scope_.pop_back();
    if (e.touched.size() != 0) {
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
    if (e.touched.size() != 0) {
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
      scope_[it->second].touched.push_back(buf);
    }
  }
  void Visit_(const Call* op) final {
    if (op->is_intrinsic(intrinsic::tvm_address_of)) {
      const Load* l = op->args[0].as<Load>();
      this->Visit(l->index);
    } else {
      IRVisitor::Visit_(op);
    }
  }
  void Visit_(const Variable* buf) final {
    // Directly reference to the variable count as a read.
    auto it = alloc_scope_level_.find(buf);
    if (it != alloc_scope_level_.end()) {
      CHECK_LT(it->second, scope_.size()) << " buf=" << buf->name_hint;
      scope_[it->second].touched.push_back(buf);
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
    e.touched = std::move(scope_.back().touched);
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
  void Visit_(const IfThenElse* op) final {
    VisitNewScope(op);
  }

  void Visit_(const For* op) final {
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
  using StmtEntry = LinearAccessPatternFinder::StmtEntry;

  Stmt Rewrite(Stmt stmt) {
    std::vector<StmtEntry> seq =
       LinearAccessPatternFinder().GetLinearSeq(stmt);
    this->FindFreeLocation(seq);
    this->PlanMemory(seq);
    this->PrepareNewAlloc();
    stmt = this->Mutate(stmt);
    if (attach_map_.count(nullptr)) {
      std::vector<Stmt> nest;
      for (StorageEntry* e : attach_map_.at(nullptr)) {
        CHECK_EQ(e->scope.rank, 0);
        if (e->new_alloc.defined()) {
          nest.emplace_back(AttrStmt::make(
              e->alloc_var, attr::storage_scope,
              StringImm::make(e->scope.to_string()),
              Evaluate::make(0)));
          nest.push_back(e->new_alloc);
        }
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
    return Store::make(it->second->alloc_var,
                       op->value,
                       RemapIndex(op->value.type(), op->index, it->second),
                       op->predicate);
  }
  Expr Mutate_(const Load* op, const Expr& e) final {
    Expr expr = IRMutator::Mutate_(op, e);
    op = expr.as<Load>();
    auto it = alloc_map_.find(op->buffer_var.get());
    if (it == alloc_map_.end()) return expr;
    return Load::make(op->type,
                      it->second->alloc_var,
                      RemapIndex(op->type, op->index, it->second),
                      op->predicate);
  }
  Expr Mutate_(const Variable* op, const Expr& e) final {
    auto it = alloc_map_.find(op);
    if (it != alloc_map_.end()) {
      if (it->second->elem_offset != 0) {
        LOG(WARNING) << "Use a merged buffer variable address, could cause error";
      }
      return it->second->alloc_var;
    } else {
      return e;
    }
  }
  Expr Mutate_(const Call* op, const Expr& e) final {
    if (op->is_intrinsic(intrinsic::tvm_access_ptr)) {
      CHECK_EQ(op->args.size(), 5U);
      Type dtype = op->args[0].type();
      const Variable* buffer = op->args[1].as<Variable>();
      auto it = alloc_map_.find(buffer);
       if (it == alloc_map_.end()) return IRMutator::Mutate_(op, e);
       const StorageEntry* se = it->second;
       Expr offset = Mutate(op->args[2]);
       Expr extent = Mutate(op->args[3]);
       CHECK_EQ(se->elem_type, dtype.element_of())
           << " buffer=" << buffer->name_hint;
       CHECK_EQ(se->elem_offset % dtype.lanes(), 0);
       if (se->elem_offset != 0) {
         offset = make_const(offset.type(), se->elem_offset / dtype.lanes()) + offset;
       }
       return Call::make(
           op->type, op->name,
           {op->args[0], se->alloc_var, offset, extent, op->args[4]},
           op->call_type);
    } else {
      return IRMutator::Mutate_(op, e);
    }
  }

  Stmt Mutate_(const AttrStmt* op, const Stmt& s) final {
    CHECK(op->attr_key != attr::virtual_thread)
        << "InjectVirtualThread before StoragePlan";
    if (op->attr_key == attr::storage_scope) {
      return this->Mutate(op->body);
    } else if (op->attr_key == attr::thread_extent ||
               op->attr_key == attr::pragma_scope) {
      // remake all the allocation at the attach scope.
      if (attach_map_.count(op)) {
        auto& svec = attach_map_[op];
        Stmt stmt = IRMutator::Mutate_(op, s);
        op = stmt.as<AttrStmt>();
        return AttrStmt::make(
            op->node, op->attr_key, op->value,
            MakeAttach(svec, op->body));
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
    // remake all the allocation at the attach scope.
    if (attach_map_.count(op)) {
      auto& svec = attach_map_[op];
      Stmt stmt = IRMutator::Mutate_(op, s);
      op = stmt.as<For>();
      return For::make(
          op->loop_var, op->min, op->extent, op->for_type, op->device_api,
          MakeAttach(svec, op->body));
    } else {
      return IRMutator::Mutate_(op, s);
    }
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
    // The constant size of the buffer in bits, only used if it is constant
    uint64_t const_nbits{0};
    // The storage scope.
    StorageScope scope;
    // Allocs that shares this entry.
    std::vector<const Allocate*> allocs;
    // The children of this entry, not including itself.
    std::vector<StorageEntry*> merged_children;
    // The replacement allocation, if any.
    Stmt new_alloc;
    // The var expr of new allocation.
    VarExpr alloc_var;
    // The allocation element type.
    Type elem_type;
    // This is non-zero if this allocate is folded into another one
    // the address becomes alloc_var + sizeof(elem_type) * elem_offset;
    uint64_t elem_offset{0};
  };
  Stmt MakeAttach(const std::vector<StorageEntry*>& svec,
                  Stmt body) {
    std::vector<Stmt> nest;
    for (StorageEntry* e : svec) {
      nest.emplace_back(AttrStmt::make(
          e->alloc_var, attr::storage_scope,
          StringImm::make(e->scope.to_string()),
          Evaluate::make(0)));
      nest.push_back(e->new_alloc);
    }
    return MergeNest(nest, body);
  }
  // Remap the index
  Expr RemapIndex(Type dtype, Expr index, StorageEntry* e) {
    CHECK_EQ(dtype.element_of(), e->elem_type);
    if (e->elem_offset == 0) return index;
    return make_const(index.type(), e->elem_offset) + index;
  }
  // Prepare the new allocations
  void PrepareNewAlloc() {
    for (size_t i = 0; i < alloc_vec_.size(); ++i) {
      StorageEntry* e = alloc_vec_[i].get();
      attach_map_[e->attach_scope_].push_back(e);
    }
    // find allocation via attach map.
    for (auto &kv : attach_map_) {
      // find the element with the most amount of bytes.
      std::vector<StorageEntry*>& vec = kv.second;
      // try to find merge, for tagged memory
      for (size_t i = 0; i < vec.size(); ++i) {
        StorageEntry* e = vec[i];
        if (e->scope.tag.length() != 0) {
          CHECK_NE(e->const_nbits, 0U)
              << "Special tagged memory must be const size";
          for (size_t j = 0; j < i; ++j) {
            if (e->scope == vec[j]->scope) {
              vec[j]->merged_children.push_back(e);
              break;
            }
          }
        }
      }
      // Start allocation
      for (size_t i = 0; i < vec.size(); ++i) {
        StorageEntry* e = vec[i];
        // already merged
        if (e->elem_offset != 0) continue;
        if (e->merged_children.size() != 0) {
          NewAllocTagMerged(e); continue;
        }
        // Get the allocation size;
        e->alloc_var = e->allocs[0]->buffer_var;
        Type alloc_type = e->allocs[0]->type;
        for (const Allocate* op : e->allocs) {
          if (op->type.lanes() > alloc_type.lanes()) {
            alloc_type = op->type;
          }
        }
        if (e->allocs.size() == 1) {
          // simply use the original allocation.
          e->new_alloc = Allocate::make(
              e->alloc_var, alloc_type, e->allocs[0]->extents,
              e->allocs[0]->condition, Evaluate::make(0));
        } else {
          // Build a merged allocation
          Expr combo_size;
          for (const Allocate* op : e->allocs) {
            Expr sz = arith::ComputeReduce<Mul>(op->extents);
            if (alloc_type.lanes() != op->type.lanes()) {
              sz = (sz * make_const(sz.type(), op->type.lanes()) +
                    make_const(sz.type(), alloc_type.lanes() - 1)) /
                  make_const(sz.type(), alloc_type.lanes());
            }
            if (combo_size.defined()) {
              combo_size = max(combo_size, sz);
            } else {
              combo_size = sz;
            }
          }
          combo_size = ir::Simplify(combo_size);
          e->new_alloc = Allocate::make(
              e->alloc_var, alloc_type, {combo_size}, const_true(),
              Evaluate::make(0));
        }
      }
    }
  }
  // New allocation for merged data
  void NewAllocTagMerged(StorageEntry* e) {
    CHECK_NE(e->scope.tag.length(), 0U);
    // allocate with element type.
    CHECK_NE(e->const_nbits, 0U);
    MemoryInfo info = GetMemoryInfo(e->scope.to_string());
    size_t align = 1;
    if (info.defined()) {
      align = (info->max_simd_bits + e->elem_type.bits() - 1) / e->elem_type.bits();
    }
    uint64_t total_elem = e->const_nbits / e->elem_type.bits();
    if (total_elem % align != 0) {
      total_elem += align  - (total_elem % align);
    }
    e->alloc_var = e->allocs[0]->buffer_var;
    for (StorageEntry* child : e->merged_children) {
      CHECK_NE(e->const_nbits, 0U);
      CHECK_NE(total_elem, 0U);
      size_t num_elem = child->const_nbits / child->elem_type.bits();
      child->elem_offset = total_elem;
      child->alloc_var = e->alloc_var;
      total_elem += num_elem;
      if (total_elem % align != 0) {
        total_elem += align  - (total_elem % align);
      }
    }
    Expr alloc_size = make_const(e->allocs[0]->extents[0].type(),
                                 total_elem);
    e->new_alloc = Allocate::make(
        e->alloc_var, e->elem_type, {alloc_size}, const_true(),
        Evaluate::make(0));
    if (info.defined()) {
      CHECK_LE(total_elem * e->elem_type.bits(), info->max_num_bits)
          << "Allocation exceed bound of memory tag " << e->scope.to_string();
    }
  }
  // Find the free location of each varaible.
  // Just do a reverse linear scan.
  void FindFreeLocation(const std::vector<StmtEntry>& seq) {
    std::unordered_set<const Variable*> touched;
    for (size_t i = seq.size(); i != 0; --i) {
      const StmtEntry& s = seq[i - 1];
      for (const Variable* buffer : s.touched) {
        if (!touched.count(buffer)) {
          touched.insert(buffer);
          free_loc_[i - 1].push_back(buffer);
        }
      }
    }
  }
  void PlanNewScope(const Node* op) {
    if (thread_scope_ != nullptr) {
      CHECK(thread_scope_ == op);
      // erase all memory atatched to this scope.
      for (auto it = const_free_map_.begin(); it != const_free_map_.end();) {
        if (it->second->attach_scope_ == op) {
          it = const_free_map_.erase(it);
        } else {
          ++it;
        }
      }
      for (auto it = sym_free_list_.begin(); it != sym_free_list_.end();) {
        if ((*it)->attach_scope_ == op) {
          it = sym_free_list_.erase(it);
        } else {
          ++it;
        }
      }
      thread_scope_ = nullptr;
    } else {
      thread_scope_ = op;
    }
  }

  // Memory plan algorithm
  void PlanMemory(const std::vector<StmtEntry>& seq) {
    for (size_t i = 0; i < seq.size(); ++i) {
      const StmtEntry& s = seq[i];
      if (s.stmt->is_type<AttrStmt>()) {
        const auto* op = static_cast<const AttrStmt*>(s.stmt);
        CHECK(op->attr_key == attr::thread_extent ||
              op->attr_key == attr::pragma_scope);
        PlanNewScope(op);
      } else if (s.stmt->is_type<For>()) {
        const auto* op = static_cast<const For*>(s.stmt);
        if (op->for_type == ForType::Parallel) {
          if (thread_scope_ == nullptr || thread_scope_ == op) {
            PlanNewScope(op);
          }
        }
      } else if (s.stmt->is_type<Allocate>()) {
        const auto* op = static_cast<const Allocate*>(s.stmt);
        StorageEntry* e = this->FindAlloc(op, thread_scope_, s.alloc_scope);
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
                         const Node* attach_scope,
                         const StorageScope& scope,
                         size_t const_nbits) {
    // Re-use not successful, allocate a new buffer.
    std::unique_ptr<StorageEntry> entry(new StorageEntry());
    entry->attach_scope_ = attach_scope;
    entry->scope = scope;
    entry->elem_type = op->type.element_of();
    entry->const_nbits = const_nbits;
    StorageEntry* e = entry.get();
    alloc_vec_.emplace_back(std::move(entry));
    return e;
  }
  StorageEntry* FindAlloc(const Allocate* op,
                          const Node* attach_scope,
                          const StorageScope& scope) {
    // skip plan for local variable,
    // compiler can do a better job with register allocation.
    const uint64_t match_range = 16;
    uint64_t const_nbits = static_cast<uint64_t>(
        op->constant_allocation_size() * op->type.bits() * op->type.lanes());
    if (scope.rank > 1 || op->type.is_handle()) {
      return NewAlloc(op, attach_scope, scope, const_nbits);
    }
    // disable reuse of small arrays, they will be lowered to registers in LLVM
    if (const_nbits > 0  &&
        const_nbits <= 32 &&
        scope.tag.length() == 0) {
      return NewAlloc(op, attach_scope, scope, const_nbits);
    }
    if (const_nbits != 0) {
      // constant allocation.
      auto begin = const_free_map_.lower_bound(const_nbits / match_range);
      auto mid = const_free_map_.lower_bound(const_nbits);
      auto end = const_free_map_.upper_bound(const_nbits * match_range);
      for (auto it = mid; it != end; ++it) {
        StorageEntry *e = it->second;
        if (e->attach_scope_ != attach_scope) continue;
        if (e->scope != scope) continue;
        if (e->elem_type != op->type.element_of()) continue;
        e->const_nbits = std::max(const_nbits, e->const_nbits);
        const_free_map_.erase(it);
        return e;
      }
      for (auto it = mid; it != begin;) {
        --it;
        StorageEntry *e = it->second;
        if (e->attach_scope_ != attach_scope) continue;
        if (e->scope != scope) continue;
        if (e->elem_type != op->type.element_of()) continue;
        const_free_map_.erase(it);
        return e;
      }
    } else {
      // Simple strategy: round roubin.
      for (auto it = sym_free_list_.begin();
           it != sym_free_list_.end(); ++it) {
        StorageEntry* e = *it;
        if (e->attach_scope_ != attach_scope) continue;
        if (e->scope != scope) continue;
        if (e->elem_type != op->type.element_of()) continue;
        sym_free_list_.erase(it);
        return e;
      }
    }
    return NewAlloc(op, attach_scope, scope, const_nbits);
  }
  // simulated free.
  void Free(const Variable* var) {
    auto it = alloc_map_.find(var);
    CHECK(it != alloc_map_.end());
    StorageEntry* e = it->second;
    // Disable sharing of local memory.
    if (e->scope.rank > 1 || e->allocs[0]->type.is_handle()) return;
    // disable reuse of small arrays
    if (e->const_nbits > 0 && e->const_nbits <= 32) return;
    // normal free.
    if (e->const_nbits != 0) {
      const_free_map_.insert({e->const_nbits, e});
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
  std::multimap<uint64_t, StorageEntry*> const_free_map_;
  // symbolic free list, for non constant items.
  std::list<StorageEntry*> sym_free_list_;
  // The allocations
  std::vector<std::unique_ptr<StorageEntry> > alloc_vec_;
};

// Turn alloc into vector alloc
// if all its access is the same vector type.
class VectorAllocRewriter : public IRMutator {
 public:
  Expr Mutate_(const Load* op, const Expr& e) final {
    UpdateTypeMap(op->buffer_var.get(), op->type);
    return IRMutator::Mutate_(op, e);
  }

  Stmt Mutate_(const Store* op, const Stmt& s) final {
    UpdateTypeMap(op->buffer_var.get(), op->value.type());
    return IRMutator::Mutate_(op, s);
  }
  Expr Mutate_(const Call* op, const Expr& e) final {
    if (op->is_intrinsic(intrinsic::tvm_access_ptr)) {
      Type dtype = op->args[0].type();
      const Variable* buffer = op->args[1].as<Variable>();
      UpdateTypeMap(buffer, dtype);
    }
    return IRMutator::Mutate_(op, e);
  }

  Stmt Mutate_(const Allocate* op, const Stmt& s) final {
    Stmt stmt = IRMutator::Mutate_(op, s);
    op = stmt.as<Allocate>();
    const auto& tvec = acc_map_[op->buffer_var.get()];

    if (tvec.size() == 1 &&
        tvec[0].element_of() == op->type.element_of() &&
        tvec[0].lanes() % op->type.lanes() == 0 &&
        tvec[0].lanes() != op->type.lanes()) {
      int factor = tvec[0].lanes() / op->type.lanes();
      Array<Expr> extents = op->extents;
      arith::ModularEntry me = EvalModular(
          extents[extents.size() - 1],
          std::unordered_map<const Variable*, arith::ModularEntry>());
      if (me.base % factor == 0 && me.coeff % factor == 0) {
        extents.Set(extents.size() - 1,
                    extents[extents.size() - 1] / make_const(extents[0].type(), factor));
        return Allocate::make(
            op->buffer_var, tvec[0], extents,
            op->condition, op->body);
      }
    }
    return stmt;
  }


 private:
  void UpdateTypeMap(const Variable* buffer, Type t) {
    auto& tvec = acc_map_[buffer];
    if (std::find(tvec.begin(), tvec.end(), t) == tvec.end()) {
      tvec.push_back(t);
    }
  }
  // Internal access map
  std::unordered_map<const Variable*,
                     std::vector<Type> > acc_map_;
};


Stmt StorageRewrite(Stmt stmt) {
  stmt = StoragePlanRewriter().Rewrite(stmt);
  return VectorAllocRewriter().Mutate(stmt);
}
}  // namespace ir
}  // namespace tvm
