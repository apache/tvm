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
 * \file merge_shared_memory_allocations.cc
 * \brief Each GPU kernel is allowed to have only one dynamic or static shared memory allocation.
 * This pass merges multiple TIR-level dynamic or static shared memory allocations into one
 * allocation.
 */
#include <tvm/runtime/registry.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <unordered_map>
#include <unordered_set>

#include "../../runtime/thread_storage_scope.h"
#include "../../support/arena.h"
#include "ir_utils.h"

namespace tvm {
namespace tir {

using runtime::StorageRank;
using runtime::StorageScope;

bool IsDynamicSharedMemory(Var buffer_var) {
  StorageScope storage_scope = runtime::StorageScope::Create(GetPtrStorageScope(buffer_var));
  return storage_scope.rank == runtime::StorageRank::kShared && storage_scope.tag == ".dyn";
}

bool IsStaticSharedMemory(Var buffer_var) {
  StorageScope storage_scope = runtime::StorageScope::Create(GetPtrStorageScope(buffer_var));
  return storage_scope.rank == runtime::StorageRank::kShared && storage_scope.tag == "";
}

/*!
 * \brief collect the mapping from the buffer var to its allocate
 */
class AllocateCollector : public StmtExprVisitor {
 public:
  void VisitStmt_(const AllocateNode* op) final {
    if (IsDynamicSharedMemory(op->buffer_var)) {
      dyn_shmem_allocs_[op->buffer_var.get()] = op;
    } else if (IsStaticSharedMemory(op->buffer_var)) {
      static_shmem_allocs_[op->buffer_var.get()] = op;
    }
    StmtExprVisitor::VisitStmt_(op);
  }
  // The dynamic mapping from the original buffer var to its allocate
  std::unordered_map<const VarNode*, const AllocateNode*> dyn_shmem_allocs_;
  // The static mapping from the original buffer var to its allocate
  std::unordered_map<const VarNode*, const AllocateNode*> static_shmem_allocs_;
};

// Find a linear pattern of storage access
// Used for liveness analysis.
// "linear" means fitting a complex access pattern into an array of StmtEntry
//
// Define "scope" as the body of For/thread_launch/IfThenElse
// Composite scopes(loop/thread_launch/IfThen) is represented by three StmtEntry:
// before_scope -> scope_body -> after_scope
//
// This pass tries to detect last point that we need to keep memory
// alive under the same scope as Allocate.
// The storage need to be kept alive between Allocate and last access.
// The free point is only inserted at the same scope of Allocate.
//
class SharedMemLinearAccessPatternFinder final : public StmtExprVisitor {
 public:
  explicit SharedMemLinearAccessPatternFinder(bool is_dynamic = true) : is_dynamic_(is_dynamic) {}
  /*! \brief record the touch list of statement. */
  struct StmtEntry {
    // The statement
    const Object* stmt;
    // The index in the linear_seq_ to point to end of the nested scope.
    // This is only set to non-zero if stmt is a nested scope.
    // if offset > 0, means this is the begin, the end entry is current_index + offset
    // if offset < 0, means this is the end, the begin entry is current_index + offset
    int64_t scope_pair_offset{0};
    // The buffer variables this statement touched.
    std::vector<const VarNode*> touched;
  };
  // The scope of each allocation
  struct AllocEntry {
    // the level in the scope stack
    size_t level{0};
    // allocation stmt
    const AllocateNode* alloc{nullptr};
  };

  void VisitStmt_(const AllocateNode* op) final {
    size_t level = scope_.size();
    const VarNode* buf = op->buffer_var.get();
    alloc_info_[buf].alloc = op;
    alloc_info_[buf].level = level;
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const BufferStoreNode* op) final {
    scope_.push_back(StmtEntry());
    // visit subexpr
    StmtExprVisitor::VisitStmt_(op);
    // Add write access.
    const VarNode* buf = op->buffer->data.get();
    auto it = alloc_info_.find(buf);
    if (it != alloc_info_.end() && it->second.alloc) {
      ICHECK_LT(it->second.level, scope_.size());
      if (IsAppropriateSharedMemory(GetRef<Var>(buf))) {
        scope_[it->second.level].touched.push_back(buf);
      }
    }
    StmtEntry e = scope_.back();
    scope_.pop_back();
    if (e.touched.size() != 0) {
      e.stmt = op;
      linear_seq_.push_back(e);
    }
  }

  void VisitStmt_(const EvaluateNode* op) final {
    scope_.push_back(StmtEntry());
    // visit subexpr
    StmtExprVisitor::VisitStmt_(op);
    StmtEntry e = scope_.back();
    scope_.pop_back();
    if (e.touched.size() != 0) {
      e.stmt = op;
      linear_seq_.push_back(e);
    }
  }

  void VisitExpr_(const BufferLoadNode* op) final {
    // Add write access.
    StmtExprVisitor::VisitExpr_(op);
    const VarNode* buf = op->buffer->data.get();
    auto it = alloc_info_.find(buf);
    if (it != alloc_info_.end() && it->second.alloc) {
      ICHECK_LT(it->second.level, scope_.size()) << "Load memory in places other than store.";
      if (IsAppropriateSharedMemory(GetRef<Var>(buf))) {
        scope_[it->second.level].touched.push_back(buf);
      }
    }
  }

  void VisitExpr_(const CallNode* op) final {
    if (op->op.same_as(builtin::address_of())) {
      const BufferLoadNode* load = op->args[0].as<BufferLoadNode>();
      for (const auto& index : load->indices) {
        this->VisitExpr(index);
      }
    } else {
      StmtExprVisitor::VisitExpr_(op);
    }
  }
  void VisitExpr_(const VarNode* buf) final {
    // Directly reference to the variable count as a read.
    auto it = alloc_info_.find(buf);
    if (it != alloc_info_.end() && it->second.alloc) {
      ICHECK_LT(it->second.level, scope_.size());
      if (IsAppropriateSharedMemory(GetRef<Var>(buf))) {
        scope_[it->second.level].touched.push_back(buf);
      }
    }
  }
  template <typename T>
  void VisitNewScope(const T* op) {
    scope_.push_back(StmtEntry());
    StmtEntry e;
    e.stmt = op;
    int64_t begin_index = static_cast<int64_t>(linear_seq_.size());
    // before scope.
    linear_seq_.push_back(e);
    StmtExprVisitor::VisitStmt_(op);
    // after scope.
    e.touched = std::move(scope_.back().touched);
    scope_.pop_back();
    int64_t end_index = static_cast<int64_t>(linear_seq_.size());
    ICHECK_GT(end_index, begin_index);
    e.scope_pair_offset = begin_index - end_index;
    linear_seq_.push_back(e);
    // record the pointer to end index.
    ICHECK_NE(end_index, 0U);
    linear_seq_[begin_index].scope_pair_offset = end_index - begin_index;
  }
  void VisitStmt_(const AttrStmtNode* op) final {
    // Only record the outer most thread extent.
    if (op->attr_key == attr::thread_extent && !in_thread_env_) {
      in_thread_env_ = true;
      VisitNewScope(op);
      in_thread_env_ = false;
    } else if (op->attr_key == attr::extern_scope) {
      VisitNewScope(op);
    } else if (op->attr_key == attr::virtual_thread) {
      VisitNewScope(op);
    } else {
      StmtExprVisitor::VisitStmt_(op);
    }
  }
  void VisitStmt_(const IfThenElseNode* op) final { VisitNewScope(op); }

  void VisitStmt_(const ForNode* op) final { VisitNewScope(op); }

  void VisitStmt_(const WhileNode* op) final { VisitNewScope(op); }

  void VisitStmt_(const AssertStmtNode* op) final { VisitNewScope(op); }

  // linearized access sequence.
  std::vector<StmtEntry> linear_seq_;
  // The storage scope of each buffer
  std::unordered_map<const VarNode*, AllocEntry> alloc_info_;

 private:
  // Wrapper function to determine if the shared memory allocation for a variable is appropriate.
  bool IsAppropriateSharedMemory(const Var& var) {
    return is_dynamic_ ? IsDynamicSharedMemory(var) : IsStaticSharedMemory(var);
  }
  // Whether do dyanmic analysis.
  bool is_dynamic_{true};
  // Whether already in thread env.
  bool in_thread_env_{false};
  // The scope stack.
  std::vector<StmtEntry> scope_;
};

/*!
 * \brief merge the buffers whose live range has no intersection and rewrite the body
 */
class SharedMemoryRewriter : public StmtExprMutator {
 public:
  explicit SharedMemoryRewriter(
      const std::unordered_map<const VarNode*, const AllocateNode*>& shmem_allocs,
      bool is_dynamic = true)
      : is_dynamic_{is_dynamic}, shmem_allocs_{shmem_allocs} {
    if (!is_dynamic) {
      merged_buf_var_ = Var("buf_shmem", PointerType(PrimType(DataType::UInt(8)), "shared"));
    }
  }

  /*!
   * \brief plan the memory reuse for all the buffer allocated in the statement
   * \param stmt the statement
   */
  void PlanReuse(const Stmt& stmt, bool is_dynamic = true) {
    SharedMemLinearAccessPatternFinder finder(is_dynamic);
    finder(stmt);
    this->LivenessAnalysis(finder.linear_seq_);
    this->PlanMemory(finder.linear_seq_);
  }

 private:
  Stmt VisitStmt_(const AttrStmtNode* op) final {
    if (op->attr_key == attr::thread_extent && !allocated_) {
      // Allocate one dynamic shared memory allocation at the beginning of thread scope
      int max_layer_num = 0;
      std::vector<const StorageEntry*> all_entry;
      for (const auto& e : const_free_map_) {
        all_entry.push_back(e.second);
      }
      for (const StorageEntry* e : sym_free_list_) {
        all_entry.push_back(e);
      }
      for (const StorageEntry* e : all_entry) {
        max_layer_num = std::max(max_layer_num, static_cast<int>(e->allocs.size()));
      }
      // calculate align for each layer of each storage entry.
      std::vector<int> align(max_layer_num, 0);
      for (const StorageEntry* e : all_entry) {
        for (int i = 0; i < static_cast<int>(e->allocs.size()); i++) {
          for (const VarNode* buffer : e->allocs[i]) {
            const AllocateNode* alloc = shmem_allocs_[buffer];
            align[i] = std::max(align[i], alloc->dtype.bytes());
          }
        }
      }
      // calculate offset for each buffer based on the align of each layer
      for (const StorageEntry* e : all_entry) {
        PrimExpr max_inner_offset = 0;
        for (int i = 0; i < static_cast<int>(e->allocs.size()); i++) {
          PrimExpr inner_offset = 0;
          for (const VarNode* buffer : e->allocs[i]) {
            const AllocateNode* alloc = shmem_allocs_[buffer];
            buffer_byte_offsets_[buffer] = merged_alloc_size_ + inner_offset;
            inner_offset += alloc->extents[0] * alloc->dtype.bytes();
            inner_offset += indexmod(align[i] - indexmod(inner_offset, align[i]), align[i]);
          }
          max_inner_offset = max(max_inner_offset, inner_offset);
        }
        merged_alloc_size_ += max_inner_offset;
      }

      allocated_ = true;
      Allocate new_body(merged_buf_var_, DataType::UInt(8), {merged_alloc_size_}, const_true(),
                        StmtExprMutator::VisitStmt(op->body));
      return AttrStmt(op->node, op->attr_key, op->value, new_body, op->span);
    }
    return StmtMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const AllocateNode* op) final {
    if (IsAppropriateSharedMemory(op->buffer_var)) {
      return StmtExprMutator::VisitStmt(op->body);
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const DeclBufferNode* op) final {
    auto node = Downcast<DeclBuffer>(StmtExprMutator::VisitStmt_(op));
    if (auto new_buf = GetUpdatedBuffer(node->buffer); !new_buf.same_as(node->buffer)) {
      node.CopyOnWrite()->buffer = new_buf;
    }
    return std::move(node);
  }

  PrimExpr VisitExpr_(const BufferLoadNode* op) final {
    auto node = Downcast<BufferLoad>(StmtExprMutator::VisitExpr_(op));
    return VisitBufferAccess(std::move(node));
  }

  Stmt VisitStmt_(const BufferStoreNode* op) final {
    auto node = Downcast<BufferStore>(StmtExprMutator::VisitStmt_(op));
    return VisitBufferAccess(std::move(node));
  }

  template <typename Node>
  Node VisitBufferAccess(Node node) {
    if (IsAppropriateSharedMemory(node->buffer->data)) {
      ICHECK_EQ(node->indices.size(), 1)
          << "MergeSharedMemoryAllocations expects flat memory buffers, "
          << "and is to be run after "
          << "StorageFlatten (TE schedules) or FlattenBuffer (TIR schedules)";
      Array<PrimExpr> indices = {node->indices[0] +
                                 this->GetBufferOffset(node->buffer->data, node->buffer->dtype)};

      auto writer = node.CopyOnWrite();
      writer->buffer = GetUpdatedBuffer(node->buffer);
      writer->indices = indices;
    }

    return node;
  }

  Buffer GetUpdatedBuffer(Buffer buffer) {
    auto key = buffer.get();
    auto it = buffer_remap_.find(key);
    if (it != buffer_remap_.end()) {
      return it->second;
    }

    if (IsAppropriateSharedMemory(buffer->data)) {
      ICHECK_EQ(buffer->shape.size(), 1)
          << "Buffer " << buffer << " has shape " << buffer->shape << ".  "
          << "MergeSharedMemoryAllocations expects flat memory buffers, "
          << "and is to be run after "
          << "StorageFlatten (TE schedules) or FlattenBuffer (TIR schedules)";
      auto writer = buffer.CopyOnWrite();
      writer->data = merged_buf_var_;
    }

    buffer_remap_[key] = buffer;
    return buffer;
  }

  PrimExpr VisitExpr_(const CallNode* op) final {
    if (op->op.same_as(builtin::tvm_access_ptr())) {
      ICHECK_EQ(op->args.size(), 5U);
      DataType dtype = op->args[0].dtype();
      Var buffer = Downcast<Var>(op->args[1]);
      if (!IsAppropriateSharedMemory(buffer)) {
        return StmtExprMutator::VisitExpr_(op);
      }
      PrimExpr extra_offset = GetBufferOffset(buffer, dtype);

      PrimExpr offset = this->VisitExpr(op->args[2]);
      PrimExpr extent = this->VisitExpr(op->args[3]);
      return Call(op->dtype, op->op,
                  {op->args[0], merged_buf_var_, extra_offset + offset, extent, op->args[4]});
    } else {
      return StmtExprMutator::VisitExpr_(op);
    }
  }

  PrimExpr GetBufferOffset(Var buffer_var, DataType dtype) {
    auto it = buffer_byte_offsets_.find(buffer_var.get());
    ICHECK(it != buffer_byte_offsets_.end());
    return indexdiv(it->second, dtype.bytes());
  }

  // Wrapper function to determine if the shared memory allocation for a variable is appropriate.
  bool IsAppropriateSharedMemory(const Var& var) {
    return is_dynamic_ ? IsDynamicSharedMemory(var) : IsStaticSharedMemory(var);
  }

  using StmtEntry = SharedMemLinearAccessPatternFinder::StmtEntry;
  struct StorageEntry {
    // The constant size of the buffer in bits, only used if it is constant
    uint64_t const_nbits{0};
    // Allocs that shares this entry.
    // The inner vector means a "layer"
    // For example, it we need to allocate C in the memory of A and B:
    // |  A: 4096 bytes |  B: 4096 bytes |
    // |            C: 8192 bytes        |
    // Then the allocs = {{A, B}, {C}}
    std::vector<std::vector<const VarNode*>> allocs;
  };

  // Event entry in liveness analysis
  struct EventEntry {
    // variables we generate
    std::vector<const VarNode*> gen;
    // variables we kill
    std::vector<const VarNode*> kill;
  };

  /*!
   * \brief Liveness analysis to find gen and kill point of each variable.
   * \param seq the linear pattern of storage access
   */
  void LivenessAnalysis(const std::vector<StmtEntry>& seq) {
    // find kill point, do a reverse linear scan.
    std::unordered_set<const VarNode*> touched;
    for (size_t i = seq.size(); i != 0; --i) {
      const StmtEntry& s = seq[i - 1];
      for (const VarNode* buffer : s.touched) {
        if (!touched.count(buffer)) {
          touched.insert(buffer);
          event_map_[s.stmt].kill.push_back(buffer);
        }
      }
    }
    // find gen point, do forward scan
    touched.clear();
    for (size_t i = 0; i < seq.size(); ++i) {
      int64_t offset = seq[i].scope_pair_offset;
      if (offset < 0) continue;
      const StmtEntry& s = seq[i + offset];
      for (const VarNode* buffer : s.touched) {
        if (!touched.count(buffer)) {
          touched.insert(buffer);
          event_map_[s.stmt].gen.push_back(buffer);
        }
      }
    }
  }

  /*!
   * \brief Memory plan algorithm
   * \param seq the linear pattern of storage access
   * \param alloc_info
   */
  void PlanMemory(const std::vector<StmtEntry>& seq) {
    std::unordered_set<const VarNode*> inplace_flag;

    for (size_t i = 0; i < seq.size(); ++i) {
      auto it = event_map_.find(seq[i].stmt);
      // scope_pair_offset <= 0 means it is either
      // - leaf stmt(offset = 0)
      // - end of scope(offset < 0)
      // In both cases, we need to handle the kill event correctly
      auto is_leaf_alloc = [&](const VarNode* var) {
        return seq[i].scope_pair_offset == 0 &&
               std::find(it->second.gen.begin(), it->second.gen.end(), var) != it->second.gen.end();
      };
      if (it != event_map_.end() && seq[i].scope_pair_offset <= 0) {
        for (const VarNode* var : it->second.kill) {
          if (!is_leaf_alloc(var)) this->Free(var);
        }
      }
      // scope_pair_offset >= 0 means it is either
      // - leaf stmt(offset = 0)
      // - beginning of scope(offset < 0)
      // In both cases, we need to handle the gen event correctly
      if (it != event_map_.end() && seq[i].scope_pair_offset >= 0) {
        for (const VarNode* var : it->second.gen) {
          ICHECK(shmem_allocs_.count(var));
          const AllocateNode* alloc = shmem_allocs_[var];
          StorageEntry* dst_entry = FindAlloc(alloc);
          alloc_map_[var] = dst_entry;
        }
      }
      if (it != event_map_.end() && seq[i].scope_pair_offset <= 0) {
        for (const VarNode* var : it->second.kill) {
          if (is_leaf_alloc(var)) this->Free(var);
        }
      }
    }
  }
  /*!
   * \brief Allocate new storage entry.
   * \param op the allocate node
   * \param the size of the allocation in bits
   * \return the new storage entry
   */
  StorageEntry* NewAlloc(const AllocateNode* op, size_t const_nbits) {
    ICHECK(op != nullptr);
    // Re-use not successful, allocate a new buffer.
    StorageEntry* entry = arena_.make<StorageEntry>();
    entry->allocs.push_back({op->buffer_var.get()});
    entry->const_nbits = const_nbits;
    return entry;
  }
  /*!
   * \brief find the storage entry in the free list for the allocate
   * \param op the allocate node
   * \return the storage entry
   */
  StorageEntry* FindAlloc(const AllocateNode* op) {
    ICHECK(op != nullptr);
    // skip plan for local variable,
    // compiler can do a better job with register allocation.
    const uint64_t match_range = 16;
    uint64_t op_elem_bits = op->dtype.bits() * op->dtype.lanes();
    uint64_t const_nbits = static_cast<uint64_t>(op->ConstantAllocationSize() * op_elem_bits);
    // disable reuse of small arrays, they will be lowered to registers in LLVM
    // This rules only apply if we are using non special memory
    if (const_nbits > 0 && const_nbits <= 32) {
      return NewAlloc(op, const_nbits);
    }

    if (const_nbits != 0) {
      // constant allocation.
      auto begin = const_free_map_.lower_bound(0);
      auto mid = const_free_map_.lower_bound(const_nbits);
      auto end = const_free_map_.upper_bound(const_nbits * match_range);
      // Start looking at the buffer that is bigger than the required size first.
      // If we find one, directly allocate the buffer in its location and remove its entry in the
      // free list
      for (auto it = mid; it != end; ++it) {
        StorageEntry* e = it->second;
        e->const_nbits = std::max(const_nbits, e->const_nbits);
        const_free_map_.erase(it);
        it->second->allocs.push_back({op->buffer_var.get()});
        return e;
      }
      // Then start looking at smaller buffers.
      // Keep collecting the buffer until the sum of their size exceeds the buffer to allocate
      // and finally free all these entry in the free list
      std::vector<std::multimap<uint64_t, StorageEntry*>::iterator> delete_it;
      // the alloc list for the new entry
      std::vector<std::vector<const VarNode*>> reuse_allocs;
      uint64_t mem_ct = 0;
      for (auto it = mid; it != begin;) {
        --it;
        delete_it.push_back(it);
        mem_ct += it->second->const_nbits;
        int n = it->second->allocs.size();
        if (n > static_cast<int>(reuse_allocs.size())) {
          reuse_allocs.resize(n, {});
        }
        for (int i = 0; i < n; i++) {
          for (const VarNode* alloc : it->second->allocs[i]) {
            reuse_allocs[i].push_back(alloc);
          }
        }
        if (mem_ct >= const_nbits) {
          break;
        }
      }
      reuse_allocs.push_back({op->buffer_var.get()});
      if (mem_ct != 0) {
        StorageEntry* e = arena_.make<StorageEntry>();
        e->const_nbits = std::max(const_nbits, mem_ct);
        e->allocs = reuse_allocs;
        for (auto it : delete_it) {
          const_free_map_.erase(it);
        }
        return e;
      }
    } else {
      // if its symbolic allocation, just arbitrarily choose one entry to fit in because we don't
      // know its actual size
      for (auto it = sym_free_list_.begin(); it != sym_free_list_.end(); ++it) {
        StorageEntry* e = *it;
        sym_free_list_.erase(it);
        return e;
      }
    }
    return NewAlloc(op, const_nbits);
  }

  /*!
   * \brief add the storage entry to the buffer var into the free list.
   * \param var the buffer var
   */
  void Free(const VarNode* var) {
    auto it = alloc_map_.find(var);
    ICHECK(it != alloc_map_.end());
    StorageEntry* e = it->second;
    ICHECK_NE(e->allocs.size(), 0U);

    // disable reuse of small arrays
    if (e->const_nbits > 0 && e->const_nbits <= 32) return;

    // normal free.
    if (e->const_nbits != 0) {
      const_free_map_.insert({e->const_nbits, e});
    } else {
      sym_free_list_.push_back(e);
    }
  }
  // Wheather enable dyanmic analysis.
  bool is_dynamic_{true};
  // The var for the merged buffer
  Var merged_buf_var_{"buf_dyn_shmem", PointerType(PrimType(DataType::UInt(8)), "shared.dyn")};
  // The mapping from the original buffer var to its allocate
  std::unordered_map<const VarNode*, const AllocateNode*> shmem_allocs_;
  // The size of the merged buffer
  PrimExpr merged_alloc_size_{0};
  // The mapping from the original buffer var to its offset in the merged buffer
  std::unordered_map<const VarNode*, PrimExpr> buffer_byte_offsets_;
  // The mapping from the original buffer objects to their location in the merged buffer.
  std::unordered_map<const BufferNode*, Buffer> buffer_remap_;
  // The flag indicating whether the merged buffer has been allocated
  bool allocated_{false};
  // Locations of free ops.
  std::unordered_map<const Object*, EventEntry> event_map_;
  // constant size free map.
  std::multimap<uint64_t, StorageEntry*> const_free_map_;
  // symbolic free list, for non constant items.
  std::list<StorageEntry*> sym_free_list_;
  // The allocation assign map
  std::unordered_map<const VarNode*, StorageEntry*> alloc_map_;
  /*! \brief allocator of all the StorageEntry*/
  support::Arena arena_;
};

Stmt MergeSharedMemoryAllocations(Stmt stmt, bool merge_static_smem) {
  AllocateCollector collector;
  collector(stmt);
  if (collector.dyn_shmem_allocs_.size() > 1) {
    SharedMemoryRewriter rewriter(collector.dyn_shmem_allocs_);
    rewriter.PlanReuse(stmt);
    stmt = rewriter(std::move(stmt));
  }
  if (merge_static_smem && collector.static_shmem_allocs_.size() > 1) {
    SharedMemoryRewriter rewriter(collector.static_shmem_allocs_, false);
    rewriter.PlanReuse(stmt, false);
    stmt = rewriter(std::move(stmt));
  }
  return stmt;
}

namespace transform {

Pass MergeSharedMemoryAllocations() {
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    bool merge_static_smem = ctx->GetConfig<Bool>("tir.merge_static_smem", Bool(false)).value();
    auto* n = f.CopyOnWrite();
    n->body = MergeSharedMemoryAllocations(std::move(n->body), merge_static_smem);
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.MergeSharedMemoryAllocations", {});
}

TVM_REGISTER_GLOBAL("tir.transform.MergeSharedMemoryAllocations")
    .set_body_typed(MergeSharedMemoryAllocations);

}  // namespace transform
}  // namespace tir
}  // namespace tvm
