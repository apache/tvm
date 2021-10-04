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
 * \file storage_rewrite.cc
 * \brief Memory access pattern analysis and optimization.
 *  Re-write data access to enable memory sharing when possible.
 */
#include <tvm/arith/analyzer.h>
#include <tvm/ir/type.h>
#include <tvm/runtime/registry.h>
#include <tvm/target/target_info.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <map>
#include <unordered_map>
#include <unordered_set>

#include "../../runtime/thread_storage_scope.h"
#include "../ir/buffer_common.h"
#include "ir_utils.h"

namespace tvm {
namespace tir {

using runtime::StorageRank;
using runtime::StorageScope;

// Find a linear pattern of storage access
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
class LinearAccessPatternFinder final : public StmtExprVisitor {
 public:
  /*! \brief record the touch hist of statment. */
  struct StmtEntry {
    // The statment
    const Object* stmt;
    // The index in the linear_seq_ to point to end of the nested scope.
    // This is only set to non-zero if stmt is a nested scope.
    // if offset > 0, means this is the begin, the end entry is current_index + offset
    // if offset < 0, means this is the end, the begin entry is current_index + offset
    int64_t scope_pair_offset{0};
    // The buffer variables this statment touched.
    std::vector<const VarNode*> touched;
  };
  // The scope of each allocation
  struct AllocEntry {
    // scope level
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
  void VisitStmt_(const StoreNode* op) final {
    scope_.push_back(StmtEntry());
    // visit subexpr
    StmtExprVisitor::VisitStmt_(op);
    // Add write access.
    const VarNode* buf = op->buffer_var.get();
    auto it = alloc_info_.find(buf);
    if (it != alloc_info_.end() && it->second.alloc) {
      ICHECK_LT(it->second.level, scope_.size());
      scope_[it->second.level].touched.push_back(buf);
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
  void VisitExpr_(const LoadNode* op) final {
    // Add write access.
    StmtExprVisitor::VisitExpr_(op);
    const VarNode* buf = op->buffer_var.get();
    auto it = alloc_info_.find(buf);
    if (it != alloc_info_.end() && it->second.alloc) {
      ICHECK_LT(it->second.level, scope_.size()) << "Load memory in places other than store.";
      scope_[it->second.level].touched.push_back(buf);
    }
  }
  void VisitExpr_(const CallNode* op) final {
    if (op->op.same_as(builtin::address_of())) {
      const LoadNode* l = op->args[0].as<LoadNode>();
      this->VisitExpr(l->index);
    } else {
      StmtExprVisitor::VisitExpr_(op);
    }
  }
  void VisitExpr_(const VarNode* buf) final {
    // Directly reference to the variable count as a read.
    auto it = alloc_info_.find(buf);
    if (it != alloc_info_.end() && it->second.alloc) {
      ICHECK_LT(it->second.level, scope_.size()) << " buf=" << buf->name_hint;
      scope_[it->second.level].touched.push_back(buf);
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
  // Whether already in thread env.
  bool in_thread_env_{false};
  // The scope stack.
  std::vector<StmtEntry> scope_;
};

// Verify if the statement can be run safely via inplace fashion
//
// Detect pattern: dst[index] = f(src[index])
//
// WARNING: the current detection algorithm cannot handle the case
// when a location in an array is written multiple times
//
// For example, the following program will pass the check,
// but we cannot make A and B to be the same array.
//
//  A[0] = B[0] + 1
//  A[0] = B[0] + 1
//
// The high level code generator needs to ensure that the generated
// code only write each location of the target array once.
//
// This is the case with IR generated by the current compute schedule.
// We explicitly return false if we find there is an extern block
// which can be arbitrary IR.
//
// Neve-the-less, inplace detector should be used with care in mind.
// We may also consider introduce a condition checker that checks
// if every index only visited once for an absolute sufficient condition.
//
// The code after inplace transformation is no longer idempotent.
//
class InplaceOpVerifier : public StmtExprVisitor {
 public:
  bool Check(const Object* stmt, const VarNode* dst, const VarNode* src) {
    dst_ = dst;
    src_ = src;
    result_ = true;
    if (stmt->IsInstance<AttrStmtNode>()) {
      VisitStmt_(static_cast<const AttrStmtNode*>(stmt));
    } else if (stmt->IsInstance<ForNode>()) {
      VisitStmt_(static_cast<const ForNode*>(stmt));
    } else if (stmt->IsInstance<IfThenElseNode>()) {
      VisitStmt_(static_cast<const IfThenElseNode*>(stmt));
    } else if (stmt->IsInstance<WhileNode>()) {
      VisitStmt_(static_cast<const WhileNode*>(stmt));
    } else if (stmt->IsInstance<StoreNode>()) {
      VisitStmt_(static_cast<const StoreNode*>(stmt));
    } else {
      return false;
    }
    return result_;
  }

  using StmtExprVisitor::VisitStmt_;

  void VisitStmt(const Stmt& n) final {
    if (!result_) return;
    StmtExprVisitor::VisitStmt(n);
  }
  void VisitExpr(const PrimExpr& n) final {
    if (!result_) return;
    StmtExprVisitor::VisitExpr(n);
  }

  void VisitExpr_(const VarNode* op) final {
    // assume all opaque access is unsafe
    if (op == dst_ || op == src_) {
      result_ = false;
      return;
    }
  }

  void VisitStmt_(const StoreNode* op) final {
    ++mem_nest_;
    this->VisitExpr(op->index);
    --mem_nest_;
    if (op->buffer_var.get() == dst_) {
      store_ = op;
      this->VisitExpr(op->value);
      this->VisitExpr(op->predicate);
      store_ = nullptr;
    } else {
      this->VisitExpr(op->value);
      this->VisitExpr(op->predicate);
    }
  }

  void VisitStmt_(const AttrStmtNode* op) final {
    // always reject extern code
    if (op->attr_key == attr::extern_scope || op->attr_key == attr::volatile_scope) {
      result_ = false;
      return;
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const LoadNode* op) final {
    const VarNode* buf = op->buffer_var.get();
    // cannot read from dst_ (no reduction)
    if (buf == dst_) {
      result_ = false;
      return;
    }
    // do not allow indirect memory load
    if (mem_nest_ != 0) {
      result_ = false;
      return;
    }
    if (src_ == buf) {
      if (store_ == nullptr || store_->value.dtype() != op->dtype ||
          !tir::ExprDeepEqual()(store_->index, op->index)) {
        result_ = false;
        return;
      }
    }
    ++mem_nest_;
    StmtExprVisitor::VisitExpr_(op);
    --mem_nest_;
  }

 private:
  // result of the check
  bool result_{true};
  // destination memory
  const VarNode* dst_;
  // source variable
  const VarNode* src_;
  // counter of load,
  // it is not safe to inplace when there is nested load like A[B[i]]
  int mem_nest_{0};
  // The current store to be inspected
  const StoreNode* store_{nullptr};
};

/* \brief Rewrite and merge memory allocation.
 *
 * Using LinearAccessPatternFinder, determines which buffers could share an
 * allocation.  This includes both sequential usage of the same buffer and
 * merging small allocations at the same scope into a single larger allocation.
 * The merging of small allocations requires the codegen to cast the resulting
 * value from the storage type to the output type after access.
 */
class StoragePlanRewriter : public StmtExprMutator {
 public:
  using StmtEntry = LinearAccessPatternFinder::StmtEntry;
  using AllocEntry = LinearAccessPatternFinder::AllocEntry;

  Stmt Rewrite(Stmt stmt, bool detect_inplace) {
    detect_inplace_ = detect_inplace;
    // plan the rewrite
    LinearAccessPatternFinder finder;
    finder(stmt);
    this->LivenessAnalysis(finder.linear_seq_);
    this->PlanMemory(finder.linear_seq_, finder.alloc_info_);
    this->PrepareNewAlloc();
    // start rewrite
    stmt = operator()(std::move(stmt));
    if (attach_map_.count(nullptr)) {
      return MakeAttach(attach_map_.at(nullptr), stmt);
    }
    return stmt;
  }
  Stmt VisitStmt_(const StoreNode* op) final {
    Stmt stmt = StmtExprMutator::VisitStmt_(op);
    op = stmt.as<StoreNode>();
    auto it = alloc_map_.find(op->buffer_var.get());
    if (it == alloc_map_.end()) return stmt;
    return Store(it->second->alloc_var, op->value,
                 RemapIndex(op->value.dtype(), op->index, it->second), op->predicate);
  }
  PrimExpr VisitExpr_(const LoadNode* op) final {
    PrimExpr expr = StmtExprMutator::VisitExpr_(op);
    op = expr.as<LoadNode>();
    auto it = alloc_map_.find(op->buffer_var.get());
    if (it == alloc_map_.end()) return expr;
    return Load(op->dtype, it->second->alloc_var, RemapIndex(op->dtype, op->index, it->second),
                op->predicate);
  }
  PrimExpr VisitExpr_(const VarNode* op) final {
    auto it = alloc_map_.find(op);
    if (it != alloc_map_.end()) {
      if (it->second->bits_offset != 0) {
        LOG(WARNING) << "Use a merged buffer variable address, could cause error";
      }
      return it->second->alloc_var;
    } else {
      return GetRef<PrimExpr>(op);
    }
  }
  PrimExpr VisitExpr_(const CallNode* op) final {
    if (op->op.same_as(builtin::tvm_access_ptr())) {
      ICHECK_EQ(op->args.size(), 5U);
      DataType dtype = op->args[0].dtype();
      const VarNode* buffer = op->args[1].as<VarNode>();
      auto it = alloc_map_.find(buffer);
      if (it == alloc_map_.end()) {
        return StmtExprMutator::VisitExpr_(op);
      }
      const StorageEntry* se = it->second;
      PrimExpr offset = this->VisitExpr(op->args[2]);
      PrimExpr extent = this->VisitExpr(op->args[3]);
      uint64_t elem_bits = dtype.bits() * dtype.lanes();
      ICHECK_EQ(se->bits_offset % elem_bits, 0U);
      if (se->bits_offset != 0) {
        offset = make_const(offset.dtype(), se->bits_offset / elem_bits) + offset;
      }
      return Call(op->dtype, op->op, {op->args[0], se->alloc_var, offset, extent, op->args[4]});
    } else {
      return StmtExprMutator::VisitExpr_(op);
    }
  }

  Stmt VisitStmt_(const AttrStmtNode* op) final {
    if (op->attr_key == attr::thread_extent || op->attr_key == attr::virtual_thread ||
        attr::IsPragmaKey(op->attr_key)) {
      // remake all the allocation at the attach scope.
      if (attach_map_.count(op)) {
        auto& svec = attach_map_[op];
        Stmt stmt = StmtExprMutator::VisitStmt_(op);
        op = stmt.as<AttrStmtNode>();
        return AttrStmt(op->node, op->attr_key, op->value, MakeAttach(svec, op->body));
      } else {
        return StmtExprMutator::VisitStmt_(op);
      }
    } else if (op->attr_key == attr::volatile_scope) {
      Stmt stmt = StmtExprMutator::VisitStmt_(op);
      op = stmt.as<AttrStmtNode>();
      auto it = alloc_map_.find(op->node.as<VarNode>());
      if (it == alloc_map_.end()) return stmt;
      return AttrStmt(it->second->alloc_var, op->attr_key, op->value, op->body);
    } else {
      return StmtExprMutator::VisitStmt_(op);
    }
  }

  Stmt VisitStmt_(const ForNode* op) final {
    ICHECK(op->kind != ForKind::kVectorized) << "VectorizeLoop before LiftStorageAlloc";
    // remake all the allocation at the attach scope.
    if (attach_map_.count(op)) {
      auto& svec = attach_map_[op];
      Stmt stmt = StmtExprMutator::VisitStmt_(op);
      op = stmt.as<ForNode>();
      return For(op->loop_var, op->min, op->extent, op->kind, MakeAttach(svec, op->body),
                 op->thread_binding, op->annotations);
    } else {
      return StmtExprMutator::VisitStmt_(op);
    }
  }

  Stmt VisitStmt_(const AllocateNode* op) final { return this->VisitStmt(op->body); }

 private:
  struct StorageEntry {
    // The scope that this alloc attaches after
    // For shared/local memory it is beginning of the thread extent.
    // for global memory it is nullptr, means beginning of everything.
    const Object* attach_scope_{nullptr};
    // The constant size of the buffer in bits, only used if it is constant
    uint64_t const_nbits{0};
    // The storage scope.
    StorageScope scope;
    // Allocs that shares this entry.
    std::vector<const AllocateNode*> allocs;
    // The children of this entry, not including itself.
    std::vector<StorageEntry*> merged_children;
    // The replacement allocation, if any.
    Stmt new_alloc;
    // The var expr of new allocation.
    Var alloc_var;
    // The allocation element type.
    DataType elem_type;
    // This is non-zero if this allocate is folded into another one
    // the address(in bits) becomes alloc_var + bits_offset;
    // can be effectively converted to the element type.
    // We need to convert bit_offset to offset of specific element type later.
    //
    // We use bits(instead of bytes) to support non-conventional indexing in hardware.
    // When we are merging buffer together, the bits_offset are set to be aligned
    // to certain value given by the max_simd_bits property of the special memory.
    //
    // This allows effective sharing among different types as long as their alignment
    // requirement fits into the max_simd_bits.
    uint64_t bits_offset{0};
  };

  // Checks whether the storage_scope is especially tagged for a specific memory.
  bool IsSpecialTaggedMemory(const StorageScope& scope) {
    return scope.tag.length() != 0 && scope.tag != ".dyn" && scope.tag != ".workspace";
  }

  // Alllocate entry of node.
  // Event entry in liveness analysis
  struct EventEntry {
    // variables we generate
    std::vector<const VarNode*> gen;
    // variables we kill
    std::vector<const VarNode*> kill;
  };

  Stmt MakeAttach(const std::vector<StorageEntry*>& svec, Stmt body) {
    std::vector<Stmt> nest;
    for (StorageEntry* e : svec) {
      if (e->new_alloc.defined()) {
        nest.push_back(e->new_alloc);
      }
    }
    return MergeNest(nest, body);
  }
  // Remap the index
  PrimExpr RemapIndex(DataType dtype, PrimExpr index, StorageEntry* e) {
    if (e->bits_offset == 0) return index;
    uint64_t elem_bits = dtype.bits();
    ICHECK_EQ(e->bits_offset % elem_bits, 0U);
    return make_const(index.dtype(), e->bits_offset / elem_bits) + index;
  }
  // Prepare the new allocations
  void PrepareNewAlloc() {
    for (size_t i = 0; i < alloc_vec_.size(); ++i) {
      StorageEntry* e = alloc_vec_[i].get();
      attach_map_[e->attach_scope_].push_back(e);
    }
    // find allocation via attach map.
    for (auto& kv : attach_map_) {
      // find the element with the most amount of bytes.
      std::vector<StorageEntry*>& vec = kv.second;
      // try to find merge, for tagged memory
      for (size_t i = 0; i < vec.size(); ++i) {
        StorageEntry* e = vec[i];
        if (IsSpecialTaggedMemory(e->scope)) {
          ICHECK_NE(e->const_nbits, 0U) << "Special tagged memory must be const size";
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
        if (e->bits_offset != 0) continue;
        if (e->merged_children.size() != 0) {
          NewAllocTagMerged(e);
          continue;
        }
        // Get the allocation size;
        e->alloc_var = e->allocs[0]->buffer_var;
        DataType alloc_type = e->allocs[0]->dtype;
        for (const AllocateNode* op : e->allocs) {
          if (op->dtype.lanes() > alloc_type.lanes()) {
            alloc_type = op->dtype;
          }
        }

        if (e->allocs.size() == 1) {
          // simply use the original allocation.
          e->new_alloc = Allocate(e->alloc_var, alloc_type, e->allocs[0]->extent,
                                  e->allocs[0]->condition, Evaluate(0));
          if (IsSpecialTaggedMemory(e->scope)) {
            MemoryInfo info = GetMemoryInfo(e->scope.to_string());
            uint64_t total_elem = e->const_nbits / e->elem_type.bits();
            ICHECK_LE(total_elem * e->elem_type.bits(), info->max_num_bits)
                << "Allocation exceed bound of memory tag " << e->scope.to_string();
          }
        } else {
          // Build a merged allocation
          PrimExpr combo_size;
          for (const AllocateNode* op : e->allocs) {
            PrimExpr sz = op->extent;
            auto nbits = op->dtype.bits() * op->dtype.lanes();
            if (const auto* imm = sz.as<IntImmNode>()) {
              if (imm->value > std::numeric_limits<int>::max() / nbits) {
                LOG(WARNING) << "The allocation requires : " << imm->value << " * " << nbits
                             << " bits, which is greater than the maximum of"
                                " int32. The size is cast to int64."
                             << "\n";
                sz = make_const(DataType::Int(64), imm->value);
              }
            }
            // transform to bits
            auto sz_nbits = sz * nbits;
            if (combo_size.defined()) {
              combo_size = max(combo_size, sz_nbits);
            } else {
              combo_size = sz_nbits;
            }
          }
          // transform to alloc bytes
          auto type_bits = alloc_type.bits() * alloc_type.lanes();
          bool divided = analyzer_.CanProve(indexmod(combo_size, type_bits) == 0);
          combo_size = indexdiv(combo_size, type_bits);
          // round up for can not divided
          if (!divided) {
            combo_size = combo_size + make_const(DataType::Int(32), 1);
          }
          combo_size = analyzer_.Simplify(combo_size);
          e->new_alloc = Allocate(e->alloc_var, alloc_type, combo_size, const_true(), Evaluate(0));
          if (IsSpecialTaggedMemory(e->scope)) {
            MemoryInfo info = GetMemoryInfo(e->scope.to_string());
            uint64_t total_elem = e->const_nbits / e->elem_type.bits();
            ICHECK_LE(total_elem * e->elem_type.bits(), info->max_num_bits)
                << "Allocation exceed bound of memory tag " << e->scope.to_string();
          }
        }
      }
    }
  }
  // New allocation for merged data
  void NewAllocTagMerged(StorageEntry* e) {
    ICHECK_NE(e->scope.tag.length(), 0U);
    // allocate with element type.
    ICHECK_NE(e->const_nbits, 0U);
    MemoryInfo info = GetMemoryInfo(e->scope.to_string());
    uint64_t total_bits = e->const_nbits;
    // By default, align to 32 bits.
    size_t align = 32;
    if (info.defined()) {
      align = info->max_simd_bits;
    }
    // Always align to max_simd_bits
    // so we can remap types by keeping this property
    if (total_bits % align != 0) {
      total_bits += align - (total_bits % align);
    }
    e->alloc_var = e->allocs[0]->buffer_var;
    for (StorageEntry* child : e->merged_children) {
      ICHECK_NE(child->const_nbits, 0U);
      ICHECK_NE(total_bits, 0U);
      child->bits_offset = total_bits;
      child->alloc_var = e->alloc_var;
      total_bits += child->const_nbits;
      if (total_bits % align != 0) {
        total_bits += align - (total_bits % align);
      }
    }
    uint64_t type_bits = e->elem_type.bits() * e->elem_type.lanes();
    PrimExpr alloc_size =
        make_const(e->allocs[0]->extent.dtype(), (total_bits + type_bits - 1) / type_bits);
    e->new_alloc = Allocate(e->alloc_var, e->elem_type, alloc_size, const_true(), Evaluate(0));
    if (info.defined()) {
      ICHECK_LE(total_bits, info->max_num_bits)
          << "Allocation exceed bound of memory tag " << e->scope.to_string();
    }
  }
  // Liveness analysis to find gen and kill point of each variable.
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
  void PlanNewScope(const Object* op) {
    if (thread_scope_ != nullptr) {
      ICHECK(thread_scope_ == op);
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
  void PlanMemory(const std::vector<StmtEntry>& seq,
                  const std::unordered_map<const VarNode*, AllocEntry>& alloc_info) {
    std::unordered_set<const VarNode*> inplace_flag;

    for (size_t i = 0; i < seq.size(); ++i) {
      const StmtEntry& s = seq[i];
      auto it = event_map_.find(seq[i].stmt);

      // scope_pair_offset >= 0 means it is either
      // - leaf stmt(offset = 0)
      // - beginning of scope(offset < 0)
      // In both cases, we need to handle the gen event correctly
      if (it != event_map_.end() && seq[i].scope_pair_offset >= 0) {
        // Inplace operation detection
        // specially handle this
        bool detect_inplace = detect_inplace_ && (it->second.gen.size() <= 2);

        for (const VarNode* var : it->second.gen) {
          ICHECK(alloc_info.count(var));
          const AllocateNode* alloc = alloc_info.at(var).alloc;
          auto storage_scope = StorageScope::Create(GetPtrStorageScope(GetRef<Var>(var)));
          StorageEntry* dst_entry = nullptr;
          // inplace detection
          if (detect_inplace) {
            // only one inplace var for s.stmt
            bool inplace_found = false;
            for (const VarNode* src : it->second.kill) {
              if (!inplace_flag.count(src) && alloc_map_.count(src)) {
                InplaceOpVerifier visitor;
                StorageEntry* src_entry = alloc_map_.at(src);
                if (src_entry->scope == storage_scope &&
                    src_entry->attach_scope_ == thread_scope_ &&
                    src_entry->elem_type == alloc->dtype.element_of() &&
                    visitor.Check(s.stmt, var, src)) {
                  uint64_t const_nbits = static_cast<uint64_t>(alloc->constant_allocation_size()) *
                                         alloc->dtype.bits() * alloc->dtype.lanes();
                  if (src_entry->const_nbits == const_nbits && !inplace_found) {
                    // successfully inplace
                    dst_entry = src_entry;
                    inplace_flag.insert(src);
                    inplace_found = true;
                  }
                }
              }
            }
          }
          if (dst_entry == nullptr) {
            dst_entry = FindAlloc(alloc, thread_scope_, storage_scope);
          }
          dst_entry->allocs.emplace_back(alloc);
          alloc_map_[var] = dst_entry;
        }
      }
      // enter/exit new scope
      if (s.stmt->IsInstance<AttrStmtNode>()) {
        const auto* op = static_cast<const AttrStmtNode*>(s.stmt);
        if (op->attr_key == attr::thread_extent || op->attr_key == attr::virtual_thread ||
            attr::IsPragmaKey(op->attr_key)) {
          PlanNewScope(op);
        } else {
          ICHECK(op->attr_key == attr::extern_scope);
        }
      } else if (s.stmt->IsInstance<ForNode>()) {
        const auto* op = static_cast<const ForNode*>(s.stmt);
        if (op->kind == ForKind::kParallel) {
          if (thread_scope_ == nullptr || thread_scope_ == op) {
            PlanNewScope(op);
          }
        }
      }
      // scope_pair_offset <= 0 means it is either
      // - leaf stmt(offset = 0)
      // - end of scope(offset < 0)
      // In both cases, we need to handle the kill event correctly
      if (it != event_map_.end() && seq[i].scope_pair_offset <= 0) {
        for (const VarNode* var : it->second.kill) {
          // skip space which are already replaced by inplace
          if (!inplace_flag.count(var)) {
            this->Free(var);
          }
        }
      }
    }
  }
  // Allocate new storage entry.
  StorageEntry* NewAlloc(const AllocateNode* op, const Object* attach_scope,
                         const StorageScope& scope, size_t const_nbits) {
    ICHECK(op != nullptr);
    // Re-use not successful, allocate a new buffer.
    std::unique_ptr<StorageEntry> entry(new StorageEntry());
    entry->attach_scope_ = attach_scope;
    entry->scope = scope;
    entry->elem_type = op->dtype.element_of();
    entry->const_nbits = const_nbits;
    StorageEntry* e = entry.get();
    alloc_vec_.emplace_back(std::move(entry));
    return e;
  }

  StorageEntry* FindAlloc(const AllocateNode* op, const Object* attach_scope,
                          const StorageScope& scope) {
    ICHECK(op != nullptr);
    // skip plan for local variable,
    // compiler can do a better job with register allocation.
    const uint64_t match_range = 16;
    uint64_t op_elem_bits = op->dtype.bits() * op->dtype.lanes();
    uint64_t const_nbits = static_cast<uint64_t>(op->constant_allocation_size() * op_elem_bits);
    // disable reuse of small arrays, they will be lowered to registers in LLVM
    // This rules only apply if we are using non special memory
    if (scope.tag.length() == 0) {
      if (scope.rank >= StorageRank::kWarp || op->dtype.is_handle()) {
        return NewAlloc(op, attach_scope, scope, const_nbits);
      }
      if (const_nbits > 0 && const_nbits <= 32) {
        return NewAlloc(op, attach_scope, scope, const_nbits);
      }
    }
    if (const_nbits != 0) {
      // constant allocation.
      auto begin = const_free_map_.lower_bound(const_nbits / match_range);
      auto mid = const_free_map_.lower_bound(const_nbits);
      auto end = const_free_map_.upper_bound(const_nbits * match_range);
      // start looking at the buffer that is bigger than the required size first
      for (auto it = mid; it != end; ++it) {
        StorageEntry* e = it->second;
        if (e->attach_scope_ != attach_scope) continue;
        if (e->scope != scope) continue;
        // when not divided, no reuse, eg, float4 vs float3
        if (e->bits_offset % op_elem_bits != 0) continue;
        e->const_nbits = std::max(const_nbits, e->const_nbits);
        const_free_map_.erase(it);
        return e;
      }
      // then start looking at smaller buffers.
      for (auto it = mid; it != begin;) {
        --it;
        StorageEntry* e = it->second;
        if (e->attach_scope_ != attach_scope) continue;
        if (e->scope != scope) continue;
        if (e->elem_type != op->dtype.element_of()) continue;
        e->const_nbits = std::max(const_nbits, e->const_nbits);
        const_free_map_.erase(it);
        return e;
      }
    } else {
      // Simple strategy: round roubin.
      for (auto it = sym_free_list_.begin(); it != sym_free_list_.end(); ++it) {
        StorageEntry* e = *it;
        if (e->attach_scope_ != attach_scope) continue;
        if (e->scope != scope) continue;
        if (e->elem_type != op->dtype.element_of()) continue;
        sym_free_list_.erase(it);
        return e;
      }
    }
    return NewAlloc(op, attach_scope, scope, const_nbits);
  }
  // simulated free.
  void Free(const VarNode* var) {
    auto it = alloc_map_.find(var);
    ICHECK(it != alloc_map_.end());
    StorageEntry* e = it->second;
    ICHECK_NE(e->allocs.size(), 0U);

    // disable reuse of small arrays, they will be lowered to registers in LLVM
    // This rules only apply if we are using non special memory
    if (e->scope.tag.length() == 0) {
      // Disable sharing of local memory.
      if (e->scope.rank >= StorageRank::kWarp || e->allocs[0]->dtype.is_handle()) return;
      // disable reuse of small arrays
      if (e->const_nbits > 0 && e->const_nbits <= 32) return;
    }
    // normal free.
    if (e->const_nbits != 0) {
      const_free_map_.insert({e->const_nbits, e});
    } else {
      sym_free_list_.push_back(e);
    }
  }
  // thread scope.
  const Object* thread_scope_{nullptr};
  // whether enable inplace detection.
  bool detect_inplace_{false};
  // Locations of free ops.
  std::unordered_map<const Object*, EventEntry> event_map_;
  // constant size free map.
  std::multimap<uint64_t, StorageEntry*> const_free_map_;
  // symbolic free list, for non constant items.
  std::list<StorageEntry*> sym_free_list_;
  // The allocation attach map
  std::unordered_map<const Object*, std::vector<StorageEntry*> > attach_map_;
  // The allocation assign map
  std::unordered_map<const VarNode*, StorageEntry*> alloc_map_;
  // The allocations
  std::vector<std::unique_ptr<StorageEntry> > alloc_vec_;
  // analyzer
  arith::Analyzer analyzer_;
};

/* Helper struct containing information on how a buffer is declared and used
 *
 */
struct BufferVarInfo {
  enum DeclarationLocation {
    kPrimFuncParam = (1 << 0),
    kPrimFuncBufferMap = (1 << 1),
    kAllocateNode = (1 << 2),
    kLetNode = (1 << 3),
  };

  // The tir::Var that represents this buffer.
  Var var;

  // The data type of an element of the buffer.
  DataType element_dtype;

  /* The extent of the buffer.
   *
   * If multidimensional, the extent of the last dimension of the buffer.  If the
   * size is unknown (e.g. pointer arguments to PrimFunc with no corresponding
   * entry in buffer_map), then extent is zero.
   */
  PrimExpr extent;

  // Where the buffer was declared
  DeclarationLocation declaration_location;

  // When accessed, which element type is it accessed as.  This may
  // differ both in base type (e.g. int32* cast to float32* after
  // packing in StorageRewrite) or in number of lanes (e.g. float16*
  // cast to float16x4*).
  std::unordered_set<DataType> access_dtype;

  DataType get_preferred_dtype() const {
    std::unordered_set<DataType> base_access_dtype;
    for (auto dtype : access_dtype) {
      base_access_dtype.insert(dtype.element_of());
    }
    // If the array is accessed as multiple base types within a
    // function, no point in changing the declared type.  CodeGenC can
    // handle this with a type-cast prior to indexing.  Vulkan will
    // raise an error at code-gen time, if a later pass doesn't split
    // it out.
    if (base_access_dtype.size() != 1) {
      return element_dtype;
    }

    DataType preferred_base_type = *base_access_dtype.begin();

    // If there is only one vectorizable size used to access the
    // buffer, and if that access size is compatible with the array
    // size, then the buffer is vectorizable.  In the future, this
    // could be improved to allow vectorized buffer access of size
    // GCD(*lanes_used), if necessary.
    int preferred_lanes = element_dtype.lanes();
    if ((element_dtype.lanes() == 1) && (access_dtype.size() == 1)) {
      arith::Analyzer analyzer_;
      arith::ModularSet me = analyzer_.modular_set(extent);

      int lanes = access_dtype.begin()->lanes();
      if ((me->coeff % lanes == 0) && (me->base % lanes == 0)) {
        preferred_lanes = lanes;
      }
    }

    return preferred_base_type.with_lanes(preferred_lanes);
  }
};

/* Checks whether buffers are accessed as scalar or vector parameters in a
 * function.
 *
 */
class VectorTypeAccessChecker : public StmtExprVisitor {
 public:
  /* Constructor
   *
   * @param params The parameters passed to a PrimFunc
   *
   * @param buffer_map The buffer_map associated with a PrimFunc
   *
   * @param allow_untyped_handles If a buffer or pointer variable is
   * missing a type annotation, assume that it has the same underlying
   * type as it is later accessed, with scalar element types.
   */
  VectorTypeAccessChecker(const Array<tir::Var>& params, const Map<Var, Buffer>& buffer_map,
                          bool allow_untyped_pointers = false)
      : allow_untyped_pointers_(allow_untyped_pointers) {
    // If a parameter is in the buffer map, we want to track the
    // version in the map.
    for (auto it : buffer_map) {
      Buffer& buffer = it.second;
      Var buffer_var = buffer->data;
      DataType dtype = buffer->dtype;
      PrimExpr extent = buffer->shape.size() ? buffer->shape[buffer->shape.size() - 1] : 0;
      OnArrayDeclaration(buffer_var, dtype, extent, BufferVarInfo::kPrimFuncParam);
    }

    // If a pointer parameter isn't in the buffer map, then we want to
    // track the parameter itself.
    for (Var buffer_var : params) {
      auto pointer_type = GetPointerType(buffer_var->type_annotation);
      if (pointer_type.first && (buffer_map.count(buffer_var) == 0)) {
        DataType dtype = pointer_type.second;
        PrimExpr extent = 0;
        OnArrayDeclaration(buffer_var, dtype, extent, BufferVarInfo::kPrimFuncBufferMap);
      }
    }
  }

  void VisitExpr_(const LoadNode* op) final {
    OnArrayAccess(op->dtype, op->buffer_var.get(), op->index, op->predicate);
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const StoreNode* op) final {
    OnArrayAccess(op->value.dtype(), op->buffer_var.get(), op->index, op->predicate);
    StmtExprVisitor::VisitStmt_(op);
  }
  void VisitExpr_(const CallNode* op) final {
    if (op->op.same_as(builtin::tvm_access_ptr())) {
      DataType dtype = op->args[0].dtype();
      const VarNode* buffer = op->args[1].as<VarNode>();
      PrimExpr index = op->args[2];
      OnArrayAccess(dtype, buffer, index, const_true(dtype.lanes()));
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const AllocateNode* op) final {
    OnArrayDeclaration(op->buffer_var, op->dtype, op->extent, BufferVarInfo::kAllocateNode);

    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const LetNode* op) final {
    HandleLetNode(op->var);
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const LetStmtNode* op) final {
    HandleLetNode(op->var);
    StmtExprVisitor::VisitStmt_(op);
  }

  void HandleLetNode(Var let_var) {
    if (let_var->dtype.is_handle()) {
      auto pointer_type = GetPointerType(let_var->type_annotation);
      if (pointer_type.first) {
        OnArrayDeclaration(let_var, pointer_type.second, 0, BufferVarInfo::kLetNode);
      } else if (allow_untyped_pointers_) {
        OnArrayDeclaration(let_var, let_var->dtype, 0, BufferVarInfo::kLetNode);
      } else {
        LOG(FATAL) << "Let statement of variable " << let_var->name_hint
                   << " is missing a type annotation, "
                   << "or type annotation is not a pointer to primitive";
      }
    }
  }

  /* Update the type map for a buffer based on its declaration
   *
   * @param buffer The VarNode representing the buffer.
   *
   * @param element_dtype The dtype of a single element of the buffer.
   * If unknown, when used with the allow_untyped_handles option,
   * should be a handle dtype.
   *
   * @param extent The extent of the buffer.  Zero if size is unknown.
   *
   * @param declaration_location How the buffer was allocated, so that
   * some locations can be rewritten without others.
   */
  void OnArrayDeclaration(Var buffer, DataType element_dtype, PrimExpr extent,
                          BufferVarInfo::DeclarationLocation declaration_location) {
    ICHECK(info_map_.find(buffer.get()) == info_map_.end())
        << "Array declaration of " << buffer->name_hint << " occurred multiple times.";

    if (element_dtype == DataType::Bool()) {
      element_dtype = DataType::Int(8).with_lanes(element_dtype.lanes());
    }

    info_map_[buffer.get()] = {buffer, element_dtype, extent, declaration_location};
  }

  /* Update the type map for a buffer based on its usage
   *
   * @param value_dtype The dtype of the value being stored to or
   * loaded from the buffer.
   *
   * @param buffer The VarNode representing the buffer.
   *
   * @param index The index at which the value is being stored/loaded.
   *
   * @param predicate The predicate used for the store/load.
   */
  void OnArrayAccess(DataType value_dtype, const VarNode* buffer, const PrimExpr& index,
                     const PrimExpr& predicate) {
    auto it = info_map_.find(buffer);
    ICHECK(it != info_map_.end()) << "Load/Store of buffer " << buffer->name_hint << " (" << buffer
                                  << ") occurred before its declaration.";
    BufferVarInfo& var_info = it->second;

    if (value_dtype.element_of() == DataType::Bool()) {
      value_dtype = DataType::Int(8).with_lanes(value_dtype.lanes());
    }

    if (var_info.element_dtype.is_handle()) {
      ICHECK(allow_untyped_pointers_) << "Variable " << buffer->name_hint
                                      << " was missing a type annotation in its declaration";
      var_info.element_dtype = value_dtype.element_of();
    }

    DataType access_dtype = value_dtype;

    int lanes_used = var_info.element_dtype.lanes();

    // This can happen due to a previous pass that had rewrite_store_load =
    // false.  This occurs from the StorageRewrite in tvm::lower, followed by the
    // PointerValueTypeRewrite in BuildSPIRV.  The rewrite_store_load = false is
    // necessary because the C-based codegens do not yet support vectorized
    // pointer types (e.g. float16x4*).  Once they do, this if statement should
    // instead be replaced by the below ICHECK_EQ.
    if (index.dtype().lanes() * var_info.element_dtype.lanes() != value_dtype.lanes()) {
      ICHECK_EQ(index.dtype().lanes(), value_dtype.lanes());
      lanes_used = 1;
      var_info.element_dtype = var_info.element_dtype.with_lanes(1);
    }

    // TODO(Lunderberg): Uncomment this check once it can be applied.
    // See https://discuss.tvm.apache.org/t/pre-rfc-vectorized-tir-buffers/10615
    // for discussion.

    // ICHECK_EQ(index.dtype().lanes() * var_info.element_dtype.lanes(), value_dtype.lanes())
    //     << "Attempting to retrieve " << value_dtype.lanes() << " lanes of data with "
    //     << index.dtype().lanes() << " indices into an array whose elements have "
    //     << var_info.element_dtype.lanes() << " lanes.  "
    //     << "Expected output with " << index.dtype().lanes() * var_info.element_dtype.lanes()
    //     << " lanes.";

    // If the index is a RampNode with stride of 1 and offset
    // divisible by the number of number of lanes, and the predicate
    // does not apply any masking, then this array access could be
    // vectorized.
    const RampNode* ramp_index = index.as<RampNode>();
    if (ramp_index && is_one(ramp_index->stride) && is_one(predicate)) {
      arith::ModularSet me = analyzer_.modular_set(ramp_index->base);
      if ((me->coeff % ramp_index->lanes == 0) && (me->base % ramp_index->lanes == 0)) {
        lanes_used = ramp_index->lanes;
      }
    }

    var_info.access_dtype.insert(access_dtype.with_lanes(lanes_used));
  }

  // Map of buffer variable information determined
  std::unordered_map<const VarNode*, BufferVarInfo> info_map_;

  //
  bool allow_untyped_pointers_{false};

  // internal analyzer
  arith::Analyzer analyzer_;
};

/* \brief Rewrites buffer/pointer variables from scalar types to vectorized
 * types.
 *
 * Some runtimes do not allow casting between composite types and the underlying
 * base type (e.g. Vulkan, casting from 1-lane float16* to 4-lane float16x4*).
 * In these cases, in order to have vectorized load/store on an array, the
 * element type of that array must be vectorized.  This is in contrast to C-style
 * runtimes, in which `float16x4* vec = *(float16x4*)(float_arr + offset)` is
 * valid.
 *
 * By default, VectorTypeRewriter will attempt to rewrite all buffer variables to
 * vectorized access, if the load/store occurring in the PrimFunc are all
 * vectorized.  This includes adjusting the indices being used to access the
 * array.  (e.g. If `float16* scalar_arr` is being converted to `float16x4*
 * vec_arr`, then `scalar_arr[Ramp(offset, 1, 4)]` will be converted to
 * `vec_arr[offset/4]`.)
 *
 * Currently, several of the C-style runtimes do not support buffers whose
 * elements are vectorized types, or rely on the presence of the Ramp nodes to
 * identify vectorized loads.  The boolean parameters in the constructor are to
 * mimic the previous behavior of VectorTypeRewriter, to avoid breaking these
 * runtimes.  Once all runtimes support vectorized buffer elements, these
 * parameters can be removed.
 */
class VectorTypeRewriter : public StmtExprMutator {
 public:
  /* Constructor
   *
   * @param checker The VectorTypeAccessChecker that has previously read out
   * information from the PrimFunc
   *
   * @param rewrite_params Whether pointer-type parameters passed into the
   * function should be rewritten from scalar types to vectorized types.
   *
   * @param rewrite_buffer_map Whether buffers present in the buffer_map should
   * have their data variable be rewritten from scalar types to vectorized types.
   *
   * @param rewrite_allocate_node Whether the buffer variable associated with
   * AllocateNodes should be rewritten from scalar types to vectorized types.
   *
   * @param rewrite_indices Whether the indices to the Load and Store nodes
   * should be rewritten to correspond to the new buffer_var type.
   *
   * @param rewrite_let_node Whether pointer declarations in let nodes
   * should be re-written.
   */
  VectorTypeRewriter(const std::unordered_map<const VarNode*, BufferVarInfo>& info_map,
                     bool rewrite_params = true, bool rewrite_buffer_map = true,
                     bool rewrite_allocate_node = true, bool rewrite_indices = true,
                     bool rewrite_let_node = true)
      : rewrite_indices_(rewrite_indices) {
    int rewrite_mask = 0;
    if (rewrite_params) {
      rewrite_mask |= BufferVarInfo::kPrimFuncParam;
    }
    if (rewrite_buffer_map) {
      rewrite_mask |= BufferVarInfo::kPrimFuncBufferMap;
    }
    if (rewrite_allocate_node) {
      rewrite_mask |= BufferVarInfo::kAllocateNode;
    }
    if (rewrite_let_node) {
      rewrite_mask |= BufferVarInfo::kLetNode;
    }

    // Rewrite any buffer variables whose preferred type isn't their current type.
    for (const auto& pair : info_map) {
      const auto& var_info = pair.second;
      DataType preferred = var_info.get_preferred_dtype();
      if (preferred != var_info.element_dtype && (rewrite_mask & var_info.declaration_location)) {
        Var old_buffer_var = var_info.var;
        Var new_buffer_var(old_buffer_var->name_hint,
                           PointerType(PrimType(preferred), GetPtrStorageScope(old_buffer_var)),
                           old_buffer_var->span);

        rewrite_map_[var_info.var.get()] = {var_info.var, new_buffer_var, var_info.element_dtype,
                                            preferred};
      }
    }
  }

  PrimExpr VisitExpr_(const LoadNode* op) final {
    PrimExpr expr = StmtExprMutator::VisitExpr_(op);
    op = expr.as<LoadNode>();

    if (!rewrite_indices_) {
      return expr;
    }

    auto it = rewrite_map_.find(op->buffer_var.get());
    if (it == rewrite_map_.end()) {
      return expr;
    }
    const auto& info = it->second;

    DataType out_dtype_base = info.new_element_dtype.element_of();

    const RampNode* ramp_index = op->index.as<RampNode>();
    if (ramp_index && is_one(ramp_index->stride)) {
      PrimExpr new_index =
          ramp_index->base / make_const(ramp_index->base.dtype(), ramp_index->lanes);
      return Load(out_dtype_base.with_lanes(op->dtype.lanes()), info.new_buffer_var, new_index,
                  const_true(new_index.dtype().lanes()), op->span);
    } else {
      return Load(out_dtype_base, info.new_buffer_var, op->index, op->predicate);
    }
  }

  Stmt VisitStmt_(const StoreNode* op) final {
    Stmt stmt = StmtExprMutator::VisitStmt_(op);
    op = stmt.as<StoreNode>();

    if (!rewrite_indices_) {
      return stmt;
    }

    auto it = rewrite_map_.find(op->buffer_var.get());
    if (it == rewrite_map_.end()) {
      return stmt;
    }
    const auto& info = it->second;

    const RampNode* ramp_index = op->index.as<RampNode>();
    if (ramp_index && is_one(ramp_index->stride)) {
      PrimExpr new_index =
          ramp_index->base / make_const(ramp_index->base.dtype(), ramp_index->lanes);
      return Store(info.new_buffer_var, op->value, new_index, const_true(new_index.dtype().lanes()),
                   op->span);
    } else {
      return Store(info.new_buffer_var, op->value, op->index, op->predicate, op->span);
    }
  }

  PrimExpr VisitExpr_(const CallNode* op) final {
    if (op->op.same_as(builtin::tvm_access_ptr())) {
      PrimExpr expr = StmtExprMutator::VisitExpr_(op);
      op = expr.as<CallNode>();

      if (!rewrite_indices_) {
        return expr;
      }

      const VarNode* buffer_var = op->args[1].as<VarNode>();
      auto it = rewrite_map_.find(buffer_var);
      if (it == rewrite_map_.end()) {
        return expr;
      }
      const auto& info = it->second;

      PrimExpr index = op->args[2];
      PrimExpr extent = op->args[3];
      PrimExpr flag = op->args[4];

      PrimExpr e_dtype = tir::TypeAnnotation(info.new_element_dtype);
      PrimExpr factor = make_const(extent.dtype(), info.new_element_dtype.lanes());
      extent = extent / factor;
      index = index / factor;
      Array<PrimExpr> acc_args{e_dtype, info.new_buffer_var, index, extent, flag};
      return Call(info.new_element_dtype, builtin::tvm_access_ptr(), acc_args);

    } else {
      return StmtExprMutator::VisitExpr_(op);
    }
  }

  Stmt VisitStmt_(const AllocateNode* op) final {
    Stmt stmt = StmtExprMutator::VisitStmt_(op);
    op = stmt.as<AllocateNode>();

    auto it = rewrite_map_.find(op->buffer_var.get());
    if (it == rewrite_map_.end()) {
      return stmt;
    }

    const auto& info = it->second;

    Var new_buffer_var = info.new_buffer_var;

    int factor = info.new_element_dtype.lanes() / op->dtype.lanes();

    PrimExpr extent = op->extent / make_const(op->extent.dtype(), factor);
    return Allocate(new_buffer_var, info.new_element_dtype, extent, op->condition, op->body);
  }

  /* Update the parameters and all remaining variable references
   *
   * Should be called after calling operator() on the body of the
   * function.
   *
   * @param func A pointer to the PrimFunc being modified.
   */
  void Finalize(PrimFunc* func_ptr) const {
    ICHECK(func_ptr) << "Finalize expects a non-null pointer";
    auto& func = *func_ptr;
    auto* n = func.CopyOnWrite();

    // Remap any remaining references to the old buffer variables
    Map<Var, PrimExpr> var_remap;
    for (const auto& pair : rewrite_map_) {
      const auto& info = pair.second;
      var_remap.Set(info.old_buffer_var, info.new_buffer_var);
    }
    n->body = Substitute(n->body, var_remap);

    // Remap the argument list to use the new buffer variables.
    Array<Var> new_params;
    for (const auto& old_param : n->params) {
      auto it = rewrite_map_.find(old_param.get());
      if (it == rewrite_map_.end()) {
        new_params.push_back(old_param);
      } else {
        const auto& info = it->second;
        new_params.push_back(info.new_buffer_var);
      }
    }
    n->params = new_params;

    // Remap the Buffer objects in so that the buffers use the new buffer variables
    Map<Var, Buffer> new_buffer_map;
    for (const auto& pair : n->buffer_map) {
      Var key = pair.first;
      Buffer old_buffer = pair.second;
      Var old_var = old_buffer->data;

      auto it = rewrite_map_.find(old_var.get());
      if (it == rewrite_map_.end()) {
        new_buffer_map.Set(key, old_buffer);
      } else {
        auto& info = it->second;
        int factor = info.new_element_dtype.lanes() / info.old_element_dtype.lanes();
        ICHECK_EQ(factor * info.new_element_dtype.lanes(), info.old_element_dtype.lanes());

        auto* buffer_cow = old_buffer.CopyOnWrite();
        buffer_cow->data = info.new_buffer_var;
        buffer_cow->dtype = info.new_element_dtype;
        size_t ndim = buffer_cow->shape.size();
        const auto& last_dim = buffer_cow->shape[ndim - 1];
        buffer_cow->shape.Set(ndim - 1, last_dim / make_const(last_dim.dtype(), factor));
        new_buffer_map.Set(key, old_buffer);
      }
    }
    n->buffer_map = new_buffer_map;
  }

 private:
  struct RewriteInfo {
    Var old_buffer_var;
    Var new_buffer_var;
    DataType old_element_dtype;
    DataType new_element_dtype;
  };

  bool rewrite_indices_{true};
  std::unordered_map<const VarNode*, RewriteInfo> rewrite_map_;
};

// Rewrite allocates, pointer parameters, and buffer map into vectorized versions
// if each access into a buffer is the same vector type.
PrimFunc PointerValueTypeRewrite(PrimFunc f, bool allow_untyped_pointers = false,
                                 bool rewrite_params = true, bool rewrite_buffer_map = true,
                                 bool rewrite_allocate_node = true, bool rewrite_indices = true,
                                 bool rewrite_let_node = true) {
  VectorTypeAccessChecker checker(f->params, f->buffer_map, allow_untyped_pointers);
  checker(f->body);

  VectorTypeRewriter rewriter(checker.info_map_, rewrite_params, rewrite_buffer_map,
                              rewrite_allocate_node, rewrite_indices, rewrite_let_node);
  PrimFuncNode* n = f.CopyOnWrite();
  n->body = rewriter(std::move(n->body));
  rewriter.Finalize(&f);

  return f;
}

namespace transform {

Pass StorageRewrite() {
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    auto* n = f.CopyOnWrite();
    n->body = StoragePlanRewriter().Rewrite(std::move(n->body), true);
    return PointerValueTypeRewrite(std::move(f), true, false, false, true, false, true);
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.StorageRewrite", {});
}

TVM_REGISTER_GLOBAL("tir.transform.StorageRewrite").set_body_typed(StorageRewrite);

Pass PointerValueTypeRewrite() {
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    return PointerValueTypeRewrite(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.PointerValueTypeRewrite", {});
}

TVM_REGISTER_GLOBAL("tir.transform.PointerValueTypeRewrite")
    .set_body_typed(PointerValueTypeRewrite);

}  // namespace transform

}  // namespace tir
}  // namespace tvm
