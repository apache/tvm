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

#include <tvm/ir/type.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <algorithm>
#include <array>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "../../runtime/thread_storage_scope.h"
#include "ir_utils.h"

namespace tvm {
namespace tir {
namespace transform {

using BufferRegionSet = std::unordered_set<BufferRegion, ObjectPtrHash, ObjectPtrEqual>;
template <typename ValueType>
using BlockRealizeMap = std::unordered_map<BlockRealize, ValueType, ObjectPtrHash, ObjectPtrEqual>;

/*!
 * \brief Analyze the read and write accesses of the body statements, used by `LocalPadder`.
 */
class StorageAccessAnalyzer : public StmtExprVisitor {
 private:
  enum class StorageType : int32_t { kGlobal = 0, kShared = 1, kLocal = 2, kOthers = 3 };
  enum class RWMode { kRead, kWrite, kUnset };

  void VisitExpr_(const VarNode* op) final {
    if (rw_mode_ == RWMode::kRead) {
      read_marker_.SetStorageAccessMarker(GetRef<Var>(op));
    }
    if (rw_mode_ == RWMode::kWrite) {
      write_marker_.SetStorageAccessMarker(GetRef<Var>(op));
    }
  }
  class ReadScope {
   public:
    explicit ReadScope(StorageAccessAnalyzer* analyzer) : analyzer_(analyzer) {}
    void EnterWithScope() { analyzer_->rw_mode_ = RWMode::kRead; }
    void ExitWithScope() { analyzer_->rw_mode_ = RWMode::kUnset; }

   private:
    StorageAccessAnalyzer* analyzer_;
  };
  class WriteScope {
   public:
    explicit WriteScope(StorageAccessAnalyzer* analyzer) : analyzer_(analyzer) {}
    void EnterWithScope() { analyzer_->rw_mode_ = RWMode::kWrite; }
    void ExitWithScope() { analyzer_->rw_mode_ = RWMode::kUnset; }

   private:
    StorageAccessAnalyzer* analyzer_;
  };

  class AccessMarker {
   public:
    void SetStorageAccessMarker(const Var& var) {
      using runtime::StorageScope;

      const PointerTypeNode* ptr_type = var->type_annotation.as<PointerTypeNode>();
      if (ptr_type == nullptr) {
        return;
      }
      if (StorageScope::Create(ptr_type->storage_scope) == StorageScope::Create("global")) {
        bit_vector_[static_cast<int>(StorageType::kGlobal)] = true;
      } else if (StorageScope::Create(ptr_type->storage_scope) == StorageScope::Create("shared")) {
        bit_vector_[static_cast<int>(StorageType::kShared)] = true;
      } else if (StorageScope::Create(ptr_type->storage_scope) == StorageScope::Create("local") ||
                 StorageScope::Create(ptr_type->storage_scope) ==
                     StorageScope::Create("wmma.matrix_a") ||
                 StorageScope::Create(ptr_type->storage_scope) ==
                     StorageScope::Create("wmma.matrix_b") ||
                 StorageScope::Create(ptr_type->storage_scope) ==
                     StorageScope::Create("wmma.accumulator")) {
        bit_vector_[static_cast<int>(StorageType::kLocal)] = true;
      } else {
        bit_vector_[static_cast<int>(StorageType::kOthers)] = true;
      }
    }
    bool NoAccesses() const {
      return !(bit_vector_[static_cast<int>(StorageType::kGlobal)] ||
               bit_vector_[static_cast<int>(StorageType::kShared)] ||
               bit_vector_[static_cast<int>(StorageType::kLocal)] ||
               bit_vector_[static_cast<int>(StorageType::kOthers)]);
    }
    bool OnlyGlobalAccesses() const {
      return !(bit_vector_[static_cast<int>(StorageType::kShared)] ||
               bit_vector_[static_cast<int>(StorageType::kLocal)] ||
               bit_vector_[static_cast<int>(StorageType::kOthers)]) &&
             bit_vector_[static_cast<int>(StorageType::kGlobal)];
    }
    bool OnlyLocalAccesses() const {
      return !(bit_vector_[static_cast<int>(StorageType::kGlobal)] ||
               bit_vector_[static_cast<int>(StorageType::kShared)] ||
               bit_vector_[static_cast<int>(StorageType::kOthers)]) &&
             bit_vector_[static_cast<int>(StorageType::kLocal)];
    }
    bool OnlyLocalOrSharedAccesses() const {
      return !(bit_vector_[static_cast<int>(StorageType::kGlobal)] ||
               bit_vector_[static_cast<int>(StorageType::kOthers)]) &&
             (bit_vector_[static_cast<int>(StorageType::kShared)] ||
              bit_vector_[static_cast<int>(StorageType::kLocal)]);
    }

   private:
    std::array<bool, static_cast<int>(StorageType::kOthers) + 1> bit_vector_ = {false};
  };
  RWMode rw_mode_;
  AccessMarker read_marker_, write_marker_;
  std::pair<AccessMarker, AccessMarker> Analyze(const BlockRealizeNode* op) {
    {
      With<ReadScope> read_scope(this);
      for (const BufferRegion& read_buffer : op->block->reads) {
        VisitExpr(read_buffer->buffer->data);
      }
    }
    {
      With<WriteScope> write_scope(this);
      for (const BufferRegion& write_buffer : op->block->writes) {
        VisitExpr(write_buffer->buffer->data);
      }
    }
    return std::make_pair(read_marker_, write_marker_);
  }

  friend class LocalPadder;
};

class LocalPadder : public StmtExprMutator {
 public:
  explicit LocalPadder(std::vector<BlockRealize>&& block_realizes)
      : block_realizes_(std::move(block_realizes)) {}

 private:
  Stmt VisitStmt_(const BlockRealizeNode* op) final {
    if (is_one(op->predicate) || is_zero(op->predicate)) {
      return StmtExprMutator::VisitStmt_(op);
    }
    StorageAccessAnalyzer::AccessMarker read_marker, write_marker;
    std::tie(read_marker, write_marker) = StorageAccessAnalyzer().Analyze(op);

    // Remove and/or Inline the predicates, while preserving the correctness, where by "inline", we
    // refer to the following transformation:
    //
    //    if (predicate) A = ...;
    //    |
    //    A = predicate ? ... : init_constexpr;
    //
    // and by "correctness", we refer to:
    // - The padded value does not affect the computed results in the global memory.
    // - There is no out-of-boundary accesses.
    // - There is no race condition.

    // First, decompose the condition. Since a predicate is usually in the form of
    //
    //     a1 < c1 && a2 < c2 ...
    std::vector<PrimExpr> predicates = DecomposePredicate(op->predicate);
    std::vector<PrimExpr> residual_subexprs;

    BlockRealize ret = Downcast<BlockRealize>(StmtExprMutator::VisitStmt_(op));

    for (size_t i = 0; i < predicates.size(); ++i) {
      // In order to prove that `predicate` is directly removable, we have to show that if it is
      // evaluated to false, the same local memory location will be rejected by the one that is
      // structurally similar in the write-back block anyway.
      PrimExpr predicate = predicates[i];

      bool affect_region_size = false;
      // If the predicate sub-expression has been marked as affecting the region size by the
      // `CompactBufferRegion` pass, mark it as not removable.
      if (const CallNode* op = predicate.as<CallNode>()) {
        if (op->op.same_as(builtin::affect_region_size())) {
          affect_region_size = true;
          predicate = op->args[0];
        }
      }
      if ((!write_marker.OnlyLocalAccesses()) || affect_region_size) {
        residual_subexprs.push_back(predicate);
        continue;
      }

      // For each write buffer, get all of its consumers. In order to directly remove the predicate,
      // it is requested that all predicates should share the same guard.
      bool all_writes_are_guarded = true;
      for (const BufferRegion& write_buffer : op->block->writes) {
        bool all_consumers_have_same_predicate = true;
        for (const std::pair<BlockRealize, BufferRegionSet>& consumer_reads_pair :
             GetConsumers(GetRef<BlockRealize>(op), write_buffer)) {
          const BlockRealize& consumer = consumer_reads_pair.first;
          const BufferRegionSet& reads = consumer_reads_pair.second;
          bool consumer_has_same_predicate = false;

          CHECK(!reads.empty());
          if (reads.size() > 1) {
            // Unable to check the equivalence relationship in the case of multiple consumers.
            break;
          }

          for (PrimExpr consumer_predicate : DecomposePredicate(consumer->predicate)) {
            if (const CallNode* op = predicate.as<CallNode>()) {
              if (op->op.same_as(builtin::affect_region_size())) {
                consumer_predicate = op->args[0];
              }
            }
            if (CheckPredicateEquivalence(predicate, write_buffer, consumer_predicate,
                                          *reads.begin())) {
              consumer_has_same_predicate = true;
              break;
            }
          }  // for (consumer_predicate in DecomposePredicate(consumer->predicate))
          all_consumers_have_same_predicate &= consumer_has_same_predicate;
        }  // for (consumer in GetConsumers(write_buffer))
        all_writes_are_guarded &= all_consumers_have_same_predicate;
      }  // for (write_buffer in op->block->writes)

      if (!all_writes_are_guarded) {
        residual_subexprs.push_back(predicate);
      }
    }  // for (i in [0, predicates.size()))
    if (residual_subexprs.empty()) {
      return BlockRealize(ret->iter_values, Bool(1), ret->block);
    }
    return BlockRealize(ret->iter_values, FlattenPredicateSubExprs(residual_subexprs), ret->block);
  }

  /*! \brief Get all consumers of a buffer. */
  BlockRealizeMap<BufferRegionSet> GetConsumers(const BlockRealize& this_block,
                                                const BufferRegion& buffer_region) const {
    BlockRealizeMap<BufferRegionSet> consumers;
    bool this_block_encountered = false;
    for (const BlockRealize& block_realize : block_realizes_) {
      if (block_realize.same_as(this_block)) {
        this_block_encountered = true;
        continue;
      }
      if (!this_block_encountered || !IsLeafBlock(block_realize)) {
        continue;
      }
      for (const BufferRegion& read_buffer : block_realize->block->reads) {
        if (read_buffer->buffer->data.same_as(buffer_region->buffer->data)) {
          consumers[block_realize].insert(read_buffer);
        }
      }
    }
    return consumers;
  }

  /*! \brief Check whether a block is a leaf block or not. */
  bool IsLeafBlock(const BlockRealize& this_block) const {
    bool is_leaf_block = true;
    PostOrderVisit(this_block->block->body, [&is_leaf_block](const ObjectRef& obj_ref) {
      if (!is_leaf_block) {
        return;
      }
      if (obj_ref.as<BlockRealizeNode>()) {
        is_leaf_block = false;
        return;
      }
    });
    return is_leaf_block;
  }

  /*!
   * \brief Reverse the addition order. Imagine that we have an array `A[a, b, c]`. In most cases,
   *        the predicate is in the form of `(a * B + b * C) + c`. However, we would like it to be
   *        in the format of `a * B + (b * C + c)` for better substitution.
   */
  class ReassociateAdd : public ExprMutator {
   public:
    static PrimExpr Mutate(const PrimExpr& expr) {
      ReassociateAdd mutator;
      PrimExpr ret = expr;
      do {
        mutator.changed_ = false;
        ret = mutator(ret);
      } while (mutator.changed_);
      return ret;
    }

   private:
    PrimExpr VisitExpr_(const AddNode* op) final {
      if (const AddNode* lhs = op->a.as<AddNode>()) {
        changed_ = true;
        return Add(ExprMutator::VisitExpr(lhs->a),
                   Add(ExprMutator::VisitExpr(lhs->b), ExprMutator::VisitExpr(op->b)));
      }
      return ExprMutator::VisitExpr_(op);
    }
    bool changed_ = false;
  };

  class ExprSubstitutor : public ExprMutator {
   public:
    explicit ExprSubstitutor(const Map<PrimExpr, PrimExpr>& expr_substitute_map)
        : expr_substitute_map_(expr_substitute_map) {}

    PrimExpr VisitExpr(const PrimExpr& expr) final {
      PrimExpr ret = ExprMutator::VisitExpr(expr);
      for (const std::pair<const PrimExpr, PrimExpr> kv : expr_substitute_map_) {
        if (StructuralEqual()(kv.first, ret)) {
          return kv.second;
        }
      }
      return ret;
    }

   private:
    Map<PrimExpr, PrimExpr> expr_substitute_map_;
  };

  /**
   * \brief Check the equivalence relationship between the producer and the consumer predicate by
   *        substituting the read indices of the latter with the write indices.
   *
   */
  bool CheckPredicateEquivalence(const PrimExpr& write_predicate,
                                 const BufferRegion& write_buffer_region,
                                 const PrimExpr& read_predicate,
                                 const BufferRegion& read_buffer_region) {
    if (write_buffer_region->region.size() != read_buffer_region->region.size()) {
      return false;
    }
    Map<PrimExpr, PrimExpr> substitute_map;
    for (size_t region_idx = 0; region_idx < write_buffer_region->region.size(); ++region_idx) {
      Range write_range = write_buffer_region->region[region_idx];
      Range read_range = read_buffer_region->region[region_idx];
      if (!StructuralEqual()(write_range->extent, read_range->extent)) {
        return false;
      }
      substitute_map.Set(read_range->min, write_range->min);
    }
    return StructuralEqual()(
        ReassociateAdd::Mutate(write_predicate),
        ExprSubstitutor(substitute_map)(ReassociateAdd::Mutate(read_predicate)));
  }
  std::vector<BlockRealize> block_realizes_;
};

Stmt LocalPadTransform(Stmt stmt) {
  // Record all the blocks, used for tracing producer-consumer relationship.
  std::vector<BlockRealize> block_realizes_rev_post_order;
  PreOrderVisit(stmt, [&block_realizes_rev_post_order](const ObjectRef& obj_ref) -> bool {
    if (const BlockRealizeNode* op = obj_ref.as<BlockRealizeNode>()) {
      block_realizes_rev_post_order.push_back(GetRef<BlockRealize>(op));
    }
    return true;
  });
  LocalPadder local_padder(std::move(block_realizes_rev_post_order));
  stmt = local_padder(std::move(stmt));
  return stmt;
}

Pass LocalPad(bool enable_local_pad) {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    if (!enable_local_pad) {
      return f;
    }
    PrimFuncNode* mutable_func_node = f.CopyOnWrite();
    mutable_func_node->body = LocalPadTransform(std::move(mutable_func_node->body));
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.LocalPad", {});
}

TVM_REGISTER_GLOBAL("tir.transform.LocalPad").set_body_typed(LocalPad);

}  // namespace transform
}  // namespace tir
}  // namespace tvm
