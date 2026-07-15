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
#include <tvm/ffi/cast.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/op.h>
#include <tvm/runtime/logging.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/s_tir/transform.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>

#include <list>
#include <map>
#include <unordered_map>
#include <unordered_set>

#include "../../runtime/thread_storage_scope.h"
#include "../../support/arena.h"
#include "../../tirx/transform/ir_utils.h"

namespace tvm {
namespace s_tir {
using namespace tvm::tirx;

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
 * \brief Compute constant allocation size from buffer's allocation shape.
 * \return Product of extents if all constant, 0 otherwise.
 */
static int64_t ConstantAllocationSize(const ffi::Array<PrimExpr>& extents) {
  int64_t result = 1;
  for (size_t i = 0; i < extents.size(); ++i) {
    if (const IntImmNode* int_size = extents[i].as<IntImmNode>()) {
      result *= int_size->value;
      if (result > std::numeric_limits<int64_t>::max()) return 0;
    } else {
      return 0;
    }
  }
  return result;
}

/*!
 * \brief collect the mapping from the buffer var to its Buffer within a subtree
 */
class AllocateCollector : public StmtExprVisitor {
 public:
  explicit AllocateCollector(bool is_dynamic) : is_dynamic_(is_dynamic) {}

  void VisitStmt_(const AllocBufferNode* op) final {
    if (is_dynamic_ && IsDynamicSharedMemory(op->buffer->data)) {
      shmem_allocs_[op->buffer->data.get()] = op->buffer;
    } else if (!is_dynamic_ && IsStaticSharedMemory(op->buffer->data)) {
      shmem_allocs_[op->buffer->data.get()] = op->buffer;
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  // The mapping from the original buffer var to its Buffer
  std::unordered_map<const VarNode*, Buffer> shmem_allocs_;

 private:
  bool is_dynamic_;
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
// alive under the same scope as AllocBuffer.
// The storage need to be kept alive between AllocBuffer and last access.
// The free point is only inserted at the same scope of AllocBuffer.
//
class SharedMemLinearAccessPatternFinder final : public StmtExprVisitor {
 public:
  explicit SharedMemLinearAccessPatternFinder(bool is_dynamic = true) : is_dynamic_(is_dynamic) {}
  /*! \brief record the touch list of statement. */
  struct StmtEntry {
    // The statement
    const ffi::Object* stmt;
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
    // The buffer object
    Buffer buffer;
  };

  void VisitStmt_(const AllocBufferNode* op) final {
    size_t level = scope_.size();
    const VarNode* buf = op->buffer->data.get();
    alloc_info_[buf].buffer = op->buffer;
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
    if (it != alloc_info_.end() && it->second.buffer.defined()) {
      TVM_FFI_ICHECK_LT(it->second.level, scope_.size());
      if (IsAppropriateSharedMemory(ffi::GetRef<Var>(buf))) {
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
    // Add read access.
    StmtExprVisitor::VisitExpr_(op);
    const VarNode* buf = op->buffer->data.get();
    auto it = alloc_info_.find(buf);
    if (it != alloc_info_.end() && it->second.buffer.defined()) {
      TVM_FFI_ICHECK_LT(it->second.level, scope_.size())
          << "Load memory in places other than store.";
      if (IsAppropriateSharedMemory(ffi::GetRef<Var>(buf))) {
        scope_[it->second.level].touched.push_back(buf);
      }
    }
  }

  void VisitExpr_(const CallNode* op) final {
    if (op->op.same_as(builtin::address_of())) {
      if (const auto* load = op->args[0].as<BufferLoadNode>()) {
        for (const auto& index : load->indices) {
          this->VisitExpr(index);
        }
      } else {
        this->VisitExpr(op->args[0]);
      }
    } else {
      StmtExprVisitor::VisitExpr_(op);
    }
  }

  void VisitExpr_(const VarNode* buf) final {
    // Directly reference to the variable count as a read.
    auto it = alloc_info_.find(buf);
    if (it != alloc_info_.end() && it->second.buffer.defined()) {
      TVM_FFI_ICHECK_LT(it->second.level, scope_.size());
      if (IsAppropriateSharedMemory(ffi::GetRef<Var>(buf))) {
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
    TVM_FFI_ICHECK_GT(end_index, begin_index);
    e.scope_pair_offset = begin_index - end_index;
    linear_seq_.push_back(e);
    // record the pointer to end index.
    TVM_FFI_ICHECK_NE(end_index, 0U);
    linear_seq_[begin_index].scope_pair_offset = end_index - begin_index;
  }

  void VisitStmt_(const AttrStmtNode* op) final {
    // Only record the outer most thread extent.
    if (op->attr_key == tirx::attr::thread_extent && !in_thread_env_) {
      in_thread_env_ = true;
      VisitNewScope(op);
      in_thread_env_ = false;
    } else if (op->attr_key == tirx::attr::extern_scope) {
      VisitNewScope(op);
    } else if (op->attr_key == s_tir::attr::virtual_thread) {
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
  // Whether do dynamic analysis.
  bool is_dynamic_{true};
  // Whether already in thread env.
  bool in_thread_env_{false};
  // The scope stack.
  std::vector<StmtEntry> scope_;
};

/*!
 * \brief merge the buffers whose live range has no intersection and rewrite the body
 *
 * Uses a scope-stack design: each thread_extent block (kernel launch) gets its
 * own KernelScope that owns the merged buffer var and all per-launch bookkeeping.
 * This correctly handles PrimFuncs with multiple sibling thread_extent blocks.
 */
class SharedMemoryRewriter : public StmtExprMutator {
 public:
  explicit SharedMemoryRewriter(bool is_dynamic = true) : is_dynamic_{is_dynamic} {}

 private:
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
   * \brief Per-kernel-launch scope holding all state for one thread_extent block.
   */
  struct KernelScope {
    // The merged buffer var for THIS kernel launch.
    Var merged_buf_var;
    // Total byte size of THIS kernel's merged buffer.
    PrimExpr merged_alloc_size{0};
    // Allocations from THIS kernel's subtree.
    std::unordered_map<const VarNode*, Buffer> shmem_allocs;
    // Per-buffer byte offset into merged_buf_var.
    std::unordered_map<const VarNode*, PrimExpr> buffer_byte_offsets;
    // Buffer-object remap: original Buffer -> merged-data-var Buffer.
    std::unordered_map<const BufferNode*, Buffer> buffer_remap;
    // Has any original alloc in this scope been marked volatile?
    bool has_volatile_alloc{false};
    // Liveness data (event_map, alloc_map, const_free_map, sym_free_list) — all per-scope.
    std::unordered_map<const ffi::Object*, EventEntry> event_map;
    std::multimap<uint64_t, StorageEntry*> const_free_map;
    std::list<StorageEntry*> sym_free_list;
    std::unordered_map<const VarNode*, StorageEntry*> alloc_map;
  };

  /*!
   * \brief Create a fresh merged buffer Var for a new kernel scope.
   *        Same name string is fine — Var identity is by pointer, not name.
   */
  Var MakeMergedBufferVar() {
    if (is_dynamic_) {
      return Var("buf_dyn_shmem", PointerType(PrimType::UInt(8), "shared.dyn"));
    } else {
      return Var("buf_shmem", PointerType(PrimType::UInt(8), "shared"));
    }
  }

  Stmt VisitStmt_(const AttrStmtNode* op) final {
    if (op->attr_key == tirx::attr::thread_extent && !in_thread_env_) {
      in_thread_env_ = true;

      // 1. Push a fresh scope.
      scope_stack_.emplace_back();
      KernelScope& scope = scope_stack_.back();
      scope.merged_buf_var = MakeMergedBufferVar();

      // 2. Collect shmem allocs that belong to THIS subtree.
      AllocateCollector collector(is_dynamic_);
      collector(op->body);
      scope.shmem_allocs = std::move(collector.shmem_allocs_);

      // Per-scope early bail-out: if this thread_extent block has ≤1 shmem
      // allocation, there is nothing to merge.  Skip liveness analysis,
      // memory planning, and rewriting entirely.
      if (scope.shmem_allocs.size() <= 1) {
        scope_stack_.pop_back();
        in_thread_env_ = false;
        return StmtExprMutator::VisitStmt_(op);
      }

      // 3. Liveness + reuse plan over this subtree only.
      // Run the finder on the full AttrStmt (not just op->body) so that
      // VisitNewScope creates the proper scope pair entry for the thread_extent.
      SharedMemLinearAccessPatternFinder finder(is_dynamic_);
      finder(ffi::GetRef<Stmt>(op));
      this->LivenessAnalysis(finder.linear_seq_, scope);
      this->PlanMemory(finder.linear_seq_, scope);

      // 4. Compute byte offsets / merged_alloc_size.
      this->ComputeOffsets(scope);

      // 5. Recursively mutate the body — reads scope_stack_.back() for all rewrites.
      Stmt visited_body = StmtExprMutator::VisitStmt(op->body);

      in_thread_env_ = false;

      // 6. If this scope has no shmem allocs, skip the wrapper.
      if (scope.shmem_allocs.empty()) {
        scope_stack_.pop_back();
        return AttrStmt(op->node, op->attr_key, op->value, visited_body, op->span);
      }

      // 7. Wrap with the merged-buffer AllocBuffer.
      Buffer merged_buf(scope.merged_buf_var, PrimType::UInt(8), {scope.merged_alloc_size}, {},
                        PrimExpr(), scope.merged_buf_var->name_hint, 0, 0, BufferType::kDefault);
      ffi::Map<ffi::String, ffi::Any> annotations;
      if (scope.has_volatile_alloc) {
        annotations.Set(tirx::attr::kVolatile, true);
      }
      Stmt alloc_stmt = AllocBuffer(merged_buf, annotations);
      Stmt new_body = SeqStmt::Flatten(alloc_stmt, visited_body);

      // 8. Pop the scope.
      scope_stack_.pop_back();

      return AttrStmt(op->node, op->attr_key, op->value, new_body, op->span);
    }
    return StmtMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const AllocBufferNode* op) final {
    if (IsAppropriateSharedMemory(op->buffer->data)) {
      if (!scope_stack_.empty()) {
        KernelScope& scope = scope_stack_.back();
        if (scope.shmem_allocs.count(op->buffer->data.get())) {
          if (op->annotations.count(tirx::attr::kVolatile)) {
            scope.has_volatile_alloc = true;
          }
          return Evaluate(0);
        }
      }
      // Outside any thread_extent scope — leave as-is.
      return StmtExprMutator::VisitStmt_(op);
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const DeclBufferNode* op) final {
    auto node = StmtExprMutator::VisitStmt_(op).as_or_throw<DeclBuffer>();
    if (auto new_buf = GetUpdatedBuffer(node->buffer); !new_buf.same_as(node->buffer)) {
      node.CopyOnWrite()->buffer = new_buf;
    }
    return node;
  }

  Expr VisitExpr_(const BufferLoadNode* op) final {
    auto node = StmtExprMutator::VisitExpr_(op).as_or_throw<BufferLoad>();
    return VisitBufferAccess(std::move(node));
  }

  Stmt VisitStmt_(const BufferStoreNode* op) final {
    auto node = StmtExprMutator::VisitStmt_(op).as_or_throw<BufferStore>();
    return VisitBufferAccess(std::move(node));
  }

  template <typename Node>
  Node VisitBufferAccess(Node node) {
    if (IsAppropriateSharedMemory(node->buffer->data) && !scope_stack_.empty() &&
        scope_stack_.back().shmem_allocs.count(node->buffer->data.get())) {
      TVM_FFI_ICHECK_EQ(node->indices.size(), 1)
          << "MergeSharedMemoryAllocations expects flat memory buffers, "
          << "and is to be run after "
          << "FlattenBuffer";
      ffi::Array<PrimExpr> indices = {
          node->indices[0] + this->GetBufferOffset(node->buffer->data, node->buffer->dtype->dtype)};

      auto writer = node.CopyOnWrite();
      writer->buffer = GetUpdatedBuffer(node->buffer);
      writer->indices = indices;
    }

    return node;
  }

  Buffer GetUpdatedBuffer(Buffer buffer) {
    if (scope_stack_.empty()) return buffer;
    KernelScope& scope = scope_stack_.back();
    if (!scope.shmem_allocs.count(buffer->data.get())) return buffer;

    auto key = buffer.get();
    auto it = scope.buffer_remap.find(key);
    if (it != scope.buffer_remap.end()) {
      return it->second;
    }

    if (IsAppropriateSharedMemory(buffer->data)) {
      TVM_FFI_ICHECK_EQ(buffer->shape.size(), 1)
          << "Buffer " << buffer << " has shape " << buffer->shape << ".  "
          << "MergeSharedMemoryAllocations expects flat memory buffers, "
          << "and is to be run after "
          << "FlattenBuffer";
      auto writer = buffer.CopyOnWrite();
      writer->data = scope.merged_buf_var;
    }

    scope.buffer_remap[key] = buffer;
    return buffer;
  }

  Expr VisitExpr_(const CallNode* op) final {
    static const Op& ptx_cp_async_op = Op::Get("tirx.ptx.cp_async_raw");
    if (op->op.same_as(builtin::tvm_access_ptr())) {
      TVM_FFI_ICHECK_EQ(op->args.size(), 5U);
      DLDataType dtype = op->args[0].as_or_throw<PrimExpr>().ty()->dtype;
      auto buffer_opt = op->args[1].as<Var>();
      if (!buffer_opt.has_value()) {
        return StmtExprMutator::VisitExpr_(op);
      }
      Var buffer = buffer_opt.value();
      if (!IsAppropriateSharedMemory(buffer) || scope_stack_.empty() ||
          !scope_stack_.back().shmem_allocs.count(buffer.get())) {
        return StmtExprMutator::VisitExpr_(op);
      }
      PrimExpr extra_offset = GetBufferOffset(buffer, dtype);

      PrimExpr offset = this->VisitPrimExpr(op->args[2].as_or_throw<PrimExpr>());
      PrimExpr extent = this->VisitPrimExpr(op->args[3].as_or_throw<PrimExpr>());
      return Call(op->ty, op->op,
                  {op->args[0], scope_stack_.back().merged_buf_var, extra_offset + offset, extent,
                   op->args[4]});
    } else if (op->op.same_as(ptx_cp_async_op)) {
      TVM_FFI_ICHECK((op->args.size() == 5U) || (op->args.size() == 6U));
      Var buffer = op->args[0].as_or_throw<Var>();
      const auto* ptr_type = buffer->ty.as<PointerTypeNode>();
      TVM_FFI_ICHECK(ptr_type) << "The buffer should be a pointer type.";
      const auto* prim_type = ptr_type->element_type.as<PrimTypeNode>();
      TVM_FFI_ICHECK(prim_type) << "The buffer should be a pointer to a primitive type.";
      DLDataType dtype = prim_type->dtype;
      if (!IsAppropriateSharedMemory(buffer) || scope_stack_.empty() ||
          !scope_stack_.back().shmem_allocs.count(buffer.get())) {
        return StmtExprMutator::VisitExpr_(op);
      }
      PrimExpr extra_offset = GetBufferOffset(buffer, dtype);
      PrimExpr offset = this->VisitPrimExpr(op->args[1].as_or_throw<PrimExpr>());
      // the dst shared memory is a byte buffer generated by merging shared memory.
      // we need to multiply the offset index by the byte size of the original value dtype, to get
      // the correct offset of merged shared buffer.
      int index_factor = (static_cast<int>(dtype.bits) * static_cast<int>(dtype.lanes) + 7) / 8;
      if (op->args.size() == 5)
        return Call(op->ty.as_or_throw<PrimType>(), op->op,
                    {scope_stack_.back().merged_buf_var,
                     mul(extra_offset + offset, PrimExpr(index_factor)), op->args[2],
                     op->args[3].as_or_throw<PrimExpr>(), op->args[4].as_or_throw<PrimExpr>()})
            .as_or_throw<PrimExpr>();
      else
        return Call(op->ty.as_or_throw<PrimType>(), op->op,
                    {scope_stack_.back().merged_buf_var,
                     mul(extra_offset + offset, PrimExpr(index_factor)), op->args[2],
                     op->args[3].as_or_throw<PrimExpr>(), op->args[4].as_or_throw<PrimExpr>(),
                     op->args[5].as_or_throw<PrimExpr>()})
            .as_or_throw<PrimExpr>();
    } else {
      return StmtExprMutator::VisitExpr_(op);
    }
  }

  PrimExpr GetBufferOffset(Var buffer_var, DLDataType dtype) {
    TVM_FFI_ICHECK(!scope_stack_.empty());
    KernelScope& scope = scope_stack_.back();
    auto it = scope.buffer_byte_offsets.find(buffer_var.get());
    TVM_FFI_ICHECK(it != scope.buffer_byte_offsets.end());
    int elem_bytes = (static_cast<int>(dtype.bits) * static_cast<int>(dtype.lanes) + 7) / 8;
    return indexdiv(it->second, elem_bytes);
  }

  // Wrapper function to determine if the shared memory allocation for a variable is appropriate.
  bool IsAppropriateSharedMemory(const Var& var) {
    return is_dynamic_ ? IsDynamicSharedMemory(var) : IsStaticSharedMemory(var);
  }

  /*!
   * \brief Liveness analysis to find gen and kill point of each variable.
   * \param seq the linear pattern of storage access
   * \param scope the kernel scope to write results into
   */
  void LivenessAnalysis(const std::vector<StmtEntry>& seq, KernelScope& scope) {
    // find kill point, do a reverse linear scan.
    std::unordered_set<const VarNode*> touched;
    for (size_t i = seq.size(); i != 0; --i) {
      const StmtEntry& s = seq[i - 1];
      for (const VarNode* buffer : s.touched) {
        if (!touched.count(buffer)) {
          touched.insert(buffer);
          scope.event_map[s.stmt].kill.push_back(buffer);
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
          scope.event_map[s.stmt].gen.push_back(buffer);
        }
      }
    }
  }

  /*!
   * \brief Memory plan algorithm
   * \param seq the linear pattern of storage access
   * \param scope the kernel scope to write results into
   */
  void PlanMemory(const std::vector<StmtEntry>& seq, KernelScope& scope) {
    std::unordered_set<const VarNode*> inplace_flag;

    for (size_t i = 0; i < seq.size(); ++i) {
      auto it = scope.event_map.find(seq[i].stmt);
      // scope_pair_offset <= 0 means it is either
      // - leaf stmt(offset = 0)
      // - end of scope(offset < 0)
      // In both cases, we need to handle the kill event correctly
      auto is_leaf_alloc = [&](const VarNode* var) {
        return seq[i].scope_pair_offset == 0 &&
               std::find(it->second.gen.begin(), it->second.gen.end(), var) != it->second.gen.end();
      };
      if (it != scope.event_map.end() && seq[i].scope_pair_offset <= 0) {
        for (const VarNode* var : it->second.kill) {
          if (!is_leaf_alloc(var)) this->Free(var, scope);
        }
      }
      // scope_pair_offset >= 0 means it is either
      // - leaf stmt(offset = 0)
      // - beginning of scope(offset < 0)
      // In both cases, we need to handle the gen event correctly
      if (it != scope.event_map.end() && seq[i].scope_pair_offset >= 0) {
        for (const VarNode* var : it->second.gen) {
          TVM_FFI_ICHECK(scope.shmem_allocs.count(var));
          const Buffer& buf = scope.shmem_allocs.at(var);
          StorageEntry* dst_entry = FindAlloc(buf, scope);
          scope.alloc_map[var] = dst_entry;
        }
      }
      if (it != scope.event_map.end() && seq[i].scope_pair_offset <= 0) {
        for (const VarNode* var : it->second.kill) {
          if (is_leaf_alloc(var)) this->Free(var, scope);
        }
      }
    }
  }

  /*!
   * \brief Compute byte offsets for all entries in the scope after PlanMemory.
   * \param scope the kernel scope whose offset map to fill
   */
  void ComputeOffsets(KernelScope& scope) {
    int max_layer_num = 0;
    std::vector<const StorageEntry*> all_entry;
    for (const auto& e : scope.const_free_map) {
      all_entry.push_back(e.second);
    }
    for (const StorageEntry* e : scope.sym_free_list) {
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
          const Buffer& buf = scope.shmem_allocs.at(buffer);
          int elem_bytes = static_cast<int>(buf->dtype.StorageBytes());
          align[i] = std::max(align[i], elem_bytes);
        }
      }
    }
    // calculate offset for each buffer based on the align of each layer
    for (const StorageEntry* e : all_entry) {
      PrimExpr max_inner_offset = 0;
      for (int i = 0; i < static_cast<int>(e->allocs.size()); i++) {
        PrimExpr inner_offset = 0;
        for (const VarNode* buffer : e->allocs[i]) {
          const Buffer& buf = scope.shmem_allocs.at(buffer);
          ffi::Array<PrimExpr> alloc_shape = GetBufferAllocationShape(buf);
          int elem_bytes = static_cast<int>(buf->dtype.StorageBytes());
          int align_bytes = std::max(align[i], elem_bytes);
          if (buf->data_alignment > 0) {
            TVM_FFI_ICHECK(buf->data_alignment % align_bytes == 0)
                << "The alignment of the buffer is not a multiple of the data type size.";
            align_bytes = buf->data_alignment;
          }
          PrimExpr buffer_bytes = alloc_shape[0] * elem_bytes;
          inner_offset +=
              indexmod(align_bytes - indexmod(scope.merged_alloc_size + inner_offset, align_bytes),
                       align_bytes);
          scope.buffer_byte_offsets[buffer] = scope.merged_alloc_size + inner_offset;
          inner_offset += buffer_bytes;
        }
        max_inner_offset = max(max_inner_offset, inner_offset);
      }
      scope.merged_alloc_size = scope.merged_alloc_size + max_inner_offset;
    }
  }

  /*!
   * \brief Allocate new storage entry.
   * \param buf the buffer object
   * \param const_nbits the size of the allocation in bits
   * \return the new storage entry
   */
  StorageEntry* NewAlloc(const Buffer& buf, size_t const_nbits) {
    // Re-use not successful, allocate a new buffer.
    StorageEntry* entry = arena_.make<StorageEntry>();
    entry->allocs.push_back({buf->data.get()});
    entry->const_nbits = const_nbits;
    return entry;
  }

  /*!
   * \brief find the storage entry in the free list for the buffer
   * \param buf the buffer object
   * \param scope the kernel scope whose free lists to search
   * \return the storage entry
   */
  StorageEntry* FindAlloc(const Buffer& buf, KernelScope& scope) {
    // skip plan for local variable,
    // compiler can do a better job with register allocation.
    const uint64_t match_range = 16;
    ffi::Array<PrimExpr> alloc_shape = GetBufferAllocationShape(buf);
    DLDataType dtype = buf->dtype->dtype;
    uint64_t op_elem_bits = static_cast<uint64_t>(dtype.bits) * dtype.lanes;
    uint64_t const_nbits =
        static_cast<uint64_t>(ConstantAllocationSize(alloc_shape) * op_elem_bits);
    // disable reuse of small arrays, they will be lowered to registers in LLVM
    // This rules only apply if we are using non special memory
    if (const_nbits > 0 && const_nbits <= 32) {
      return NewAlloc(buf, const_nbits);
    }

    if (const_nbits != 0) {
      // constant allocation.
      auto begin = scope.const_free_map.lower_bound(0);
      auto mid = scope.const_free_map.lower_bound(const_nbits);
      auto end = scope.const_free_map.upper_bound(const_nbits * match_range);
      // Start looking at the buffer that is bigger than the required size first.
      // If we find one, directly allocate the buffer in its location and remove its entry in the
      // free list
      for (auto it = mid; it != end; ++it) {
        StorageEntry* e = it->second;
        e->const_nbits = std::max(const_nbits, e->const_nbits);
        scope.const_free_map.erase(it);
        e->allocs.push_back({buf->data.get()});
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
      reuse_allocs.push_back({buf->data.get()});
      if (mem_ct != 0) {
        StorageEntry* e = arena_.make<StorageEntry>();
        e->const_nbits = std::max(const_nbits, mem_ct);
        e->allocs = reuse_allocs;
        for (auto it : delete_it) {
          scope.const_free_map.erase(it);
        }
        return e;
      }
    } else {
      // if its symbolic allocation, just arbitrarily choose one entry to fit in because we don't
      // know its actual size
      for (auto it = scope.sym_free_list.begin(); it != scope.sym_free_list.end(); ++it) {
        StorageEntry* e = *it;
        scope.sym_free_list.erase(it);
        return e;
      }
    }
    return NewAlloc(buf, const_nbits);
  }

  /*!
   * \brief add the storage entry to the buffer var into the free list.
   * \param var the buffer var
   * \param scope the kernel scope whose free lists to update
   */
  void Free(const VarNode* var, KernelScope& scope) {
    auto it = scope.alloc_map.find(var);
    TVM_FFI_ICHECK(it != scope.alloc_map.end());
    StorageEntry* e = it->second;
    TVM_FFI_ICHECK_NE(e->allocs.size(), 0U);

    // disable reuse of small arrays
    if (e->const_nbits > 0 && e->const_nbits <= 32) return;

    // normal free.
    if (e->const_nbits != 0) {
      scope.const_free_map.insert({e->const_nbits, e});
    } else {
      scope.sym_free_list.push_back(e);
    }
  }

  // Whether enable dynamic analysis.
  bool is_dynamic_{true};
  // Whether already inside a thread_extent (outermost only).
  bool in_thread_env_{false};
  // Stack of per-kernel-launch scopes. Pushed on thread_extent entry, popped on exit.
  std::vector<KernelScope> scope_stack_;
  /*! \brief allocator of all the StorageEntry (shared across all scopes) */
  support::Arena arena_;
};

Stmt MergeSharedMemoryAllocations(Stmt stmt, bool merge_static_smem) {
  // Function-level early-out: skip the rewriter entirely if the PrimFunc
  // has ≤1 dynamic shared-memory allocation (nothing to merge).
  {
    AllocateCollector dyn_probe(/*is_dynamic=*/true);
    dyn_probe(stmt);
    if (dyn_probe.shmem_allocs_.size() > 1) {
      SharedMemoryRewriter dyn_rewriter(/*is_dynamic=*/true);
      stmt = dyn_rewriter(std::move(stmt));
    }
  }
  if (merge_static_smem) {
    // Similarly skip the static rewriter if there is ≤1 static shmem alloc.
    AllocateCollector static_probe(/*is_dynamic=*/false);
    static_probe(stmt);
    if (static_probe.shmem_allocs_.size() > 1) {
      SharedMemoryRewriter static_rewriter(/*is_dynamic=*/false);
      stmt = static_rewriter(std::move(stmt));
    }
  }
  return stmt;
}

namespace transform {

Pass MergeSharedMemoryAllocations() {
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    bool merge_static_smem = ctx->GetConfig<bool>("tirx.merge_static_smem", false).value();
    auto* n = f.CopyOnWrite();
    n->body = s_tir::MergeSharedMemoryAllocations(std::move(n->body), merge_static_smem);
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "s_tir.MergeSharedMemoryAllocations", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("s_tir.transform.MergeSharedMemoryAllocations",
                        static_cast<Pass (*)()>(MergeSharedMemoryAllocations));
}

}  // namespace transform
}  // namespace s_tir
}  // namespace tvm
