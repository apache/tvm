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
#include "ir_util.h"
#include "../arithmetic/compute_expr.h"
#include "../runtime/thread_storage_scope.h"

namespace tvm {
namespace ir {

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
class LinearAccessPatternFinder final : public IRVisitor {
 public:
  /*! \brief record the touch hist of statment. */
  struct StmtEntry {
    // The statment
    const Node* stmt;
    // The index in the linear_seq_ to point to end of the nested scope.
    // This is only set to non-zero if stmt is a nested scope.
    // if offset > 0, means this is the begin, the end entry is current_index + offset
    // if offset < 0, means this is the end, the begin entry is current_index + offset
    int64_t scope_pair_offset{0};
    // The buffer variables this statment touched.
    std::vector<const Variable*> touched;
  };
  // The scope of each allocation
  struct AllocEntry {
    // Scope used for allocation.
    StorageScope storage_scope;
    // scope level
    size_t level{0};
    // allocation stmt
    const Allocate* alloc{nullptr};
  };

  void Visit_(const Allocate* op) final {
    size_t level = scope_.size();
    const Variable* buf = op->buffer_var.get();
    auto it = alloc_info_.find(buf);
    CHECK(it != alloc_info_.end());
    CHECK(it->second.alloc == nullptr);
    it->second.alloc = op;
    it->second.level = level;
    IRVisitor::Visit_(op);
  }
  void Visit_(const Store* op) final {
    scope_.push_back(StmtEntry());
    // visit subexpr
    IRVisitor::Visit_(op);
    // Add write access.
    const Variable* buf = op->buffer_var.get();
    auto it = alloc_info_.find(buf);
    if (it != alloc_info_.end() && it->second.alloc) {
      CHECK_LT(it->second.level, scope_.size());
      scope_[it->second.level].touched.push_back(buf);
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
    auto it = alloc_info_.find(buf);
    if (it != alloc_info_.end() && it->second.alloc) {
      CHECK_LT(it->second.level, scope_.size())
          << "Load memory in places other than store.";
      scope_[it->second.level].touched.push_back(buf);
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
    auto it = alloc_info_.find(buf);
    if (it != alloc_info_.end() && it->second.alloc) {
      CHECK_LT(it->second.level, scope_.size())
          << " buf=" << buf->name_hint;
      scope_[it->second.level].touched.push_back(buf);
    }
  }
  template<typename T>
  void VisitNewScope(const T* op) {
    scope_.push_back(StmtEntry());
    StmtEntry e;
    e.stmt = op;
    int64_t begin_index =  static_cast<int64_t>(linear_seq_.size());
    // before scope.
    linear_seq_.push_back(e);
    IRVisitor::Visit_(op);
    // after scope.
    e.touched = std::move(scope_.back().touched);
    scope_.pop_back();
    int64_t end_index =  static_cast<int64_t>(linear_seq_.size());
    CHECK_GT(end_index, begin_index);
    e.scope_pair_offset = begin_index - end_index;
    linear_seq_.push_back(e);
    // record the pointer to end index.
    CHECK_NE(end_index, 0U);
    linear_seq_[begin_index].scope_pair_offset = end_index - begin_index;
  }
  void Visit_(const AttrStmt* op) final {
    // Only record the outer most thread extent.
    if (op->attr_key == attr::thread_extent && !in_thread_env_) {
      in_thread_env_ = true;
      VisitNewScope(op);
      in_thread_env_ = false;
    } else if (op->attr_key == attr::extern_scope) {
      VisitNewScope(op);
    } else if (op->attr_key == attr::virtual_thread) {
      VisitNewScope(op);
    } else if (op->attr_key == attr::storage_scope) {
      const Variable* buf = op->node.as<Variable>();
      alloc_info_[buf].storage_scope =
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

  void Visit_(const AssertStmt* op) final {
    VisitNewScope(op);
  }

  // linearized access sequence.
  std::vector<StmtEntry> linear_seq_;
  // The storage scope of each buffer
  std::unordered_map<const Variable*, AllocEntry> alloc_info_;

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
class InplaceOpVerifier : public IRVisitor {
 public:
  bool Check(const Node* stmt,
             const Variable* dst,
             const Variable* src) {
    dst_ = dst;
    src_ = src;
    result_ = true;
    if (stmt->is_type<AttrStmt>()) {
      Visit_(static_cast<const AttrStmt*>(stmt));
    } else if (stmt->is_type<For>()) {
      Visit_(static_cast<const For*>(stmt));
    } else if (stmt->is_type<IfThenElse>()) {
      Visit_(static_cast<const IfThenElse*>(stmt));
    } else if (stmt->is_type<Store>()) {
      Visit_(static_cast<const Store*>(stmt));
    } else {
      return false;
    }
    return result_;
  }

  using IRVisitor::Visit_;

  void Visit(const NodeRef& e) final {
    if (!result_) return;
    IRVisitor::Visit(e);
  }

  void Visit_(const Variable* op) final {
    // assume all opaque access is unsafe
    if (op == dst_ || op == src_) {
      result_ = false; return;
    }
  }

  void Visit_(const Store* op) final {
    ++mem_nest_;
    this->Visit(op->index);
    --mem_nest_;
    if (op->buffer_var.get() == dst_) {
      store_ = op;
      this->Visit(op->value);
      this->Visit(op->predicate);
      store_ = nullptr;
    } else {
      this->Visit(op->value);
      this->Visit(op->predicate);
    }
  }

  void Visit_(const AttrStmt* op) final {
    // always reject extern code
    if (op->attr_key == attr::extern_scope ||
        op->attr_key == attr::volatile_scope) {
      result_ = false; return;
    }
    IRVisitor::Visit_(op);
  }

  void Visit_(const Load* op) final {
    const Variable* buf = op->buffer_var.get();
    // cannot read from dst_ (no reduction)
    if (buf == dst_) {
      result_ = false; return;
    }
    // do not allow indirect memory load
    if (mem_nest_ != 0) {
      result_ = false; return;
    }
    if (src_ == buf) {
      if (store_ == nullptr ||
          store_->value.type() != op->type ||
          !ir::Equal(store_->index, op->index)) {
        result_ = false; return;
      }
    }
    ++mem_nest_;
    IRVisitor::Visit_(op);
    --mem_nest_;
  }


 private:
  // result of the check
  bool result_{true};
  // destination memory
  const Variable* dst_;
  // source variable
  const Variable* src_;
  // counter of load,
  // it is not safe to inplace when there is nested load like A[B[i]]
  int mem_nest_{0};
  // The current store to be inspected
  const Store* store_{nullptr};
};

// Planner to plan and rewrite memory allocation.
class StoragePlanRewriter : public IRMutator {
 public:
  using StmtEntry = LinearAccessPatternFinder::StmtEntry;
  using AllocEntry = LinearAccessPatternFinder::AllocEntry;

  Stmt Rewrite(Stmt stmt, bool detect_inplace) {
    detect_inplace_ = detect_inplace;
    // plan the rewrite
    LinearAccessPatternFinder finder;
    finder.Visit(stmt);
    this->LivenessAnalysis(finder.linear_seq_);
    this->PlanMemory(finder.linear_seq_, finder.alloc_info_);
    this->PrepareNewAlloc();
    // start rewrite
    stmt = this->Mutate(stmt);
    if (attach_map_.count(nullptr)) {
      std::vector<Stmt> nest;
      for (StorageEntry* e : attach_map_.at(nullptr)) {
        // CHECK_EQ(e->scope.rank, 0);
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
      if (it->second->bits_offset != 0) {
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
       uint64_t elem_bits = dtype.bits() * dtype.lanes();
       CHECK_EQ(se->bits_offset % elem_bits, 0U);
       if (se->bits_offset != 0) {
         offset = make_const(offset.type(), se->bits_offset / elem_bits) + offset;
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
    if (op->attr_key == attr::storage_scope) {
      return this->Mutate(op->body);
    } else if (op->attr_key == attr::thread_extent ||
               op->attr_key == attr::virtual_thread ||
               attr::IsPragmaKey(op->attr_key)) {
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

  // Alllocate entry of node.
  // Event entry in liveness analysis
  struct EventEntry {
    // variables we generate
    std::vector<const Variable*> gen;
    // variables we kill
    std::vector<const Variable*> kill;
  };

  Stmt MakeAttach(const std::vector<StorageEntry*>& svec,
                  Stmt body) {
    std::vector<Stmt> nest;
    for (StorageEntry* e : svec) {
      if (e->new_alloc.defined()) {
        nest.emplace_back(AttrStmt::make(
            e->alloc_var, attr::storage_scope,
            StringImm::make(e->scope.to_string()),
            Evaluate::make(0)));
        nest.push_back(e->new_alloc);
      }
    }
    return MergeNest(nest, body);
  }
  // Remap the index
  Expr RemapIndex(Type dtype, Expr index, StorageEntry* e) {
    if (e->bits_offset == 0) return index;
    uint64_t elem_bits = dtype.bits() * dtype.lanes();
    CHECK_EQ(e->bits_offset % elem_bits, 0U);
    return make_const(index.type(), e->bits_offset / elem_bits) + index;
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
        if (e->bits_offset != 0) continue;
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
          Expr sz = arith::ComputeReduce<Mul>(e->allocs[0]->extents,
                                              make_const(Int(32), 1));
          e->new_alloc = Allocate::make(
              e->alloc_var, alloc_type, {sz},
              e->allocs[0]->condition, Evaluate::make(0));
          if (e->scope.tag.length() != 0) {
            MemoryInfo info = GetMemoryInfo(e->scope.to_string());
            uint64_t total_elem = e->const_nbits / e->elem_type.bits();
            CHECK_LE(total_elem * e->elem_type.bits(), info->max_num_bits)
                << "Allocation exceed bound of memory tag " << e->scope.to_string();
          }
        } else {
          // Build a merged allocation
          Expr combo_size;
          for (const Allocate* op : e->allocs) {
            Expr sz = arith::ComputeReduce<Mul>(op->extents, make_const(Int(32), 1));
            auto nbits = op->type.bits() * op->type.lanes();
            if (const auto* imm = sz.as<IntImm>()) {
              if (imm->value > std::numeric_limits<int>::max() / nbits) {
                LOG(WARNING) << "The allocation requires : " << imm->value
                             << " * " << nbits
                             << " bits, which is greater than the maximum of"
                                " int32. The size is cast to int64."
                             << "\n";
                sz = make_const(Int(64), imm->value);
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
            combo_size = combo_size + make_const(Int(32), 1);
          }
          combo_size = ir::Simplify(combo_size);
          e->new_alloc = Allocate::make(
              e->alloc_var, alloc_type, {combo_size}, const_true(),
              Evaluate::make(0));
          if (e->scope.tag.length() != 0) {
            MemoryInfo info = GetMemoryInfo(e->scope.to_string());
            uint64_t total_elem = e->const_nbits / e->elem_type.bits();
            CHECK_LE(total_elem * e->elem_type.bits(), info->max_num_bits)
                << "Allocation exceed bound of memory tag " << e->scope.to_string();
          }
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
    uint64_t total_bits = e->const_nbits;
    // By default, align to 32 bits.
    size_t align = 32;
    if (info.defined()) {
      align = info->max_simd_bits;
    }
    // Always align to max_simd_bits
    // so we can remap types by keeping this property
    if (total_bits % align != 0) {
      total_bits += align  - (total_bits % align);
    }
    e->alloc_var = e->allocs[0]->buffer_var;
    for (StorageEntry* child : e->merged_children) {
      CHECK_NE(child->const_nbits, 0U);
      CHECK_NE(total_bits, 0U);
      child->bits_offset = total_bits;
      child->alloc_var = e->alloc_var;
      total_bits += child->const_nbits;
      if (total_bits % align != 0) {
        total_bits += align  - (total_bits % align);
      }
    }
    uint64_t type_bits = e->elem_type.bits() * e->elem_type.lanes();
    Expr alloc_size = make_const(e->allocs[0]->extents[0].type(),
                                 (total_bits + type_bits - 1) / type_bits);
    e->new_alloc = Allocate::make(
        e->alloc_var, e->elem_type, {alloc_size}, const_true(),
        Evaluate::make(0));
    if (info.defined()) {
      CHECK_LE(total_bits, info->max_num_bits)
          << "Allocation exceed bound of memory tag " << e->scope.to_string();
    }
  }
  // Liveness analysis to find gen and kill point of each variable.
  void LivenessAnalysis(const std::vector<StmtEntry>& seq) {
    // find kill point, do a reverse linear scan.
    std::unordered_set<const Variable*> touched;
    for (size_t i = seq.size(); i != 0; --i) {
      const StmtEntry& s = seq[i - 1];
      for (const Variable* buffer : s.touched) {
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
      for (const Variable* buffer : s.touched) {
        if (!touched.count(buffer)) {
          touched.insert(buffer);
          event_map_[s.stmt].gen.push_back(buffer);
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
  void PlanMemory(const std::vector<StmtEntry>& seq,
                  const std::unordered_map<const Variable*, AllocEntry>& alloc_info) {
    std::unordered_set<const Variable*> inplace_flag;

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

        for (const Variable* var : it->second.gen) {
          CHECK(alloc_info.count(var));
          const AllocEntry& ae = alloc_info.at(var);
          StorageEntry* dst_entry = nullptr;
          // inplace detection
          if (detect_inplace) {
            // only one inplace var for s.stmt
            bool inplace_found = false;
            for (const Variable* src : it->second.kill) {
              if (!inplace_flag.count(src) && alloc_map_.count(src)) {
                InplaceOpVerifier visitor;
                StorageEntry* src_entry = alloc_map_.at(src);
                if (src_entry->scope == ae.storage_scope &&
                    src_entry->attach_scope_ == thread_scope_ &&
                    src_entry->elem_type == ae.alloc->type.element_of() &&
                    visitor.Check(s.stmt, var, src)) {
                  uint64_t const_nbits =
                      static_cast<uint64_t>(ae.alloc->constant_allocation_size()) *
                      ae.alloc->type.bits() *
                      ae.alloc->type.lanes();
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
            dst_entry = FindAlloc(ae.alloc, thread_scope_, ae.storage_scope);
          }
          dst_entry->allocs.emplace_back(ae.alloc);
          alloc_map_[var] = dst_entry;
        }
      }
      // enter/exit new scope
      if (s.stmt->is_type<AttrStmt>()) {
        const auto* op = static_cast<const AttrStmt*>(s.stmt);
        if (op->attr_key == attr::thread_extent ||
            op->attr_key == attr::virtual_thread ||
            attr::IsPragmaKey(op->attr_key)) {
          PlanNewScope(op);
        } else {
          CHECK(op->attr_key == attr::extern_scope);
        }
      } else if (s.stmt->is_type<For>()) {
        const auto* op = static_cast<const For*>(s.stmt);
        if (op->for_type == ForType::Parallel) {
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
        for (const Variable* var : it->second.kill) {
          // skip space which are already replaced by inplace
          if (!inplace_flag.count(var)) {
            this->Free(var);
          }
        }
      }
    }
  }
  // Allocate new storage entry.
  StorageEntry* NewAlloc(const Allocate* op,
                         const Node* attach_scope,
                         const StorageScope& scope,
                         size_t const_nbits) {
    CHECK(op != nullptr);
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
    CHECK(op != nullptr);
    // skip plan for local variable,
    // compiler can do a better job with register allocation.
    const uint64_t match_range = 16;
    uint64_t op_elem_bits = op->type.bits() * op->type.lanes();
    uint64_t const_nbits = static_cast<uint64_t>(
        op->constant_allocation_size() * op_elem_bits);
    // disable reuse of small arrays, they will be lowered to registers in LLVM
    // This rules only apply if we are using non special memory
    if (scope.tag.length() == 0) {
      if (scope.rank >= StorageRank::kWarp || op->type.is_handle()) {
        return NewAlloc(op, attach_scope, scope, const_nbits);
      }
      if (const_nbits > 0  &&  const_nbits <= 32) {
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
        StorageEntry *e = it->second;
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
        StorageEntry *e = it->second;
        if (e->attach_scope_ != attach_scope) continue;
        if (e->scope != scope) continue;
        if (e->elem_type != op->type.element_of()) continue;
        e->const_nbits = std::max(const_nbits, e->const_nbits);
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
    CHECK_NE(e->allocs.size(), 0U);

    // disable reuse of small arrays, they will be lowered to registers in LLVM
    // This rules only apply if we are using non special memory
    if (e->scope.tag.length() == 0) {
      // Disable sharing of local memory.
      if (e->scope.rank >= StorageRank::kWarp ||
          e->allocs[0]->type.is_handle()) return;
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
  const Node* thread_scope_{nullptr};
  // whether enable inplace detection.
  bool detect_inplace_{false};
  // Locations of free ops.
  std::unordered_map<const Node*, EventEntry> event_map_;
  // constant size free map.
  std::multimap<uint64_t, StorageEntry*> const_free_map_;
  // symbolic free list, for non constant items.
  std::list<StorageEntry*> sym_free_list_;
  // The allocation attach map
  std::unordered_map<const Node*, std::vector<StorageEntry*> > attach_map_;
  // The allocation assign map
  std::unordered_map<const Variable*, StorageEntry*> alloc_map_;
  // The allocations
  std::vector<std::unique_ptr<StorageEntry> > alloc_vec_;
  // analyzer
  arith::Analyzer analyzer_;
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
      arith::ModularSet me = analyzer_.modular_set(extents[extents.size() - 1]);
      if (me->base % factor == 0 && me->coeff % factor == 0) {
        extents.Set(extents.size() - 1,
                    extents[extents.size() - 1] / make_const(extents[0].type(), factor));
        return Allocate::make(
            op->buffer_var, tvec[0], extents,
            op->condition, op->body);
      }
    }
    return stmt;
  }

  void UpdateTypeMap(const Variable* buffer, Type t) {
    auto& tvec = acc_map_[buffer];
    if (std::find(tvec.begin(), tvec.end(), t) == tvec.end()) {
      tvec.push_back(t);
    }
  }

  // Internal access map
  std::unordered_map<const Variable*, std::vector<Type> > acc_map_;
  // internal analyzer
  arith::Analyzer analyzer_;
};


LoweredFunc PointerValueTypeRewrite(LoweredFunc f) {
  auto n = make_node<LoweredFuncNode>(*f.operator->());
  VectorAllocRewriter rewriter;
  n->body = rewriter.Mutate(n->body);
  for (Var arg : f->args) {
    if (arg.type().is_handle()) {
      const auto& tvec = rewriter.acc_map_[arg.get()];
      if (tvec.size() == 1) {
        Expr dtype = make_const(tvec[0], 0);
        n->handle_data_type.Set(arg, dtype);
      } else {
        // always set data type to be non vectorized so
        // load/store can still work via scalarization
        if (tvec.size() != 0 && !n->handle_data_type.count(arg)) {
          Expr dtype = make_const(tvec[0].with_lanes(1), 0);
          n->handle_data_type.Set(arg, dtype);
        }
      }
    }
  }
  return LoweredFunc(n);
}

Stmt StorageRewrite(Stmt stmt) {
  stmt = StoragePlanRewriter().Rewrite(stmt, true);
  return VectorAllocRewriter().Mutate(stmt);
}
}  // namespace ir
}  // namespace tvm
