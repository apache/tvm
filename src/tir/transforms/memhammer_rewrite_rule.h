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
#ifndef TVM_TIR_TRANSFORMS_MEMHAMMER_REWRITE_RULE_H_
#define TVM_TIR_TRANSFORMS_MEMHAMMER_REWRITE_RULE_H_

#include <tvm/arith/iter_affine_map.h>
#include <tvm/runtime/registry.h>
#include <tvm/target/target.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <utility>

#include "../schedule/utils.h"

namespace tvm {
namespace tir {

/*! \brief The set containing all possible constraints of a data copy */
struct ConstraintSet {
  /*! \brief The extents of the thread binding loops */
  Map<String, Integer> thread_extent;
  /*! \brief The outer loops surrounding the data copy */
  Array<For> outer_loops;
  /*! \brief The read region of the data copy */
  BufferRegion read_region;
  /*! \brief The write region of the data copy */
  BufferRegion write_region;
  /*! \brief The dtype size in bits */
  int data_bits;
  /*! \brief Whether to insert a local stage in the data copy */
  int add_local_stage = 0;
  /*! \brief The vectorization length in bytes */
  int vector_bytes = 1;

  explicit ConstraintSet(Map<String, Integer> thread_extent,  //
                         Array<For> outer_loops,              //
                         BufferRegion read_region,            //
                         BufferRegion write_region,           //
                         int data_bits,                       //
                         const Map<String, ObjectRef>& ann)
      : thread_extent(thread_extent),
        outer_loops(outer_loops),
        read_region(read_region),
        write_region(write_region),
        data_bits(data_bits) {
    if (Optional<ObjectRef> add_local_stage = ann.Get("local_stage")) {
      this->add_local_stage = Downcast<Integer>(add_local_stage.value())->value;
    }
    if (Optional<ObjectRef> vector_bytes = ann.Get("vector_bytes")) {
      this->vector_bytes = Downcast<Integer>(vector_bytes.value())->value;
    }
  }
};

/*! \brief The set containing all possible outputs of a rewrite rule */
struct OutputSet {
  /*! \brief New buffers allocated after rewrite */
  Array<Buffer> alloc_buffer;
  /*! \brief The minimal padding size of a buffer in base 2 logarithm */
  Map<Buffer, Integer> padding_min;
};

/*!
 * \brief Rules to rewrite a data copy.
 */
class RewriteRule {
 protected:
  /* RewriteRule() = default; */
  /*!
   * \brief Rewrite the stmt under certain constraints
   * \param stmt The stmt
   * \param constraints The constraints of the rewrite
   * \param output Some additional information that the rewrite rule produces. (including the new
   *               buffer to be allocated, etc.)
   * \return the stmt after rewrite
   */
  virtual Stmt Rewrite(const Stmt& stmt, const ConstraintSet& constraints,
                       OutputSet* output) const = 0;
  /*!
   * \brief Whether the rewrite rule can be applied to the stmt under certain constraints
   * \param stmt The stmt
   * \param constraints The constraints of the rewrite
   * \return A boolean flag indicating whether the rule can be applied
   */
  virtual bool CanApply(const Stmt& stmt, const ConstraintSet& constraints) const { return true; }

 public:
  inline Stmt Apply(const Stmt& stmt, const ConstraintSet& constraints, OutputSet* output) const {
    if (CanApply(stmt, constraints)) {
      return Rewrite(stmt, constraints, output);
    } else {
      return stmt;
    }
  }
};

inline bool IsCopyBetweenScope(const Buffer& src_buffer, const Buffer& tgt_buffer,
                               runtime::StorageRank src_rank, runtime::StorageRank tgt_rank) {
  runtime::StorageScope src_scope = runtime::StorageScope::Create(src_buffer.scope());
  runtime::StorageScope tgt_scope = runtime::StorageScope::Create(tgt_buffer.scope());
  return src_scope.rank == src_rank && tgt_scope.rank == tgt_rank;
}

inline bool IsScope(const Buffer& src_buffer, runtime::StorageRank src_rank) {
  runtime::StorageScope src_scope = runtime::StorageScope::Create(src_buffer.scope());
  return src_scope.rank == src_rank;
}

/*!
 * \brief Coalesce and vectorize memory access.
 */
class CoalescedAccess : public RewriteRule {
 public:
  CoalescedAccess() = default;
  Stmt Rewrite(const Stmt& stmt, const ConstraintSet& constraints, OutputSet* output) const final;
  bool CanApply(const Stmt& stmt, const ConstraintSet& constraints) const final {
    Buffer src_buffer = constraints.read_region->buffer;
    Buffer tgt_buffer = constraints.write_region->buffer;
    return IsCopyBetweenScope(src_buffer, tgt_buffer, runtime::StorageRank::kGlobal,
                              runtime::StorageRank::kShared) ||
           IsCopyBetweenScope(src_buffer, tgt_buffer, runtime::StorageRank::kShared,
                              runtime::StorageRank::kGlobal);
  }
};

/*!
 * \brief Transform from A[f(i,j)] = B[i,j] to A[i,j] = B[f^{-1}(i,j)]
 */
class InverseMapping : public RewriteRule {
 public:
  InverseMapping() = default;
  Stmt Rewrite(const Stmt& stmt, const ConstraintSet& constraints, OutputSet* output) const final;
  bool CanApply(const Stmt& stmt, const ConstraintSet& constraints) const final {
    Buffer src_buffer = constraints.read_region->buffer;
    Buffer tgt_buffer = constraints.write_region->buffer;
    return IsCopyBetweenScope(src_buffer, tgt_buffer, runtime::StorageRank::kShared,
                              runtime::StorageRank::kGlobal);
  }
};

/*!
 * \brief Create a local stage when loading from global memory to shared memory.
 */
class CreateLocalStage : public RewriteRule {
 public:
  CreateLocalStage() = default;
  Stmt Rewrite(const Stmt& stmt, const ConstraintSet& constraints, OutputSet* output) const final;
  bool CanApply(const Stmt& stmt, const ConstraintSet& constraints) const final {
    Buffer src_buffer = constraints.read_region->buffer;
    Buffer tgt_buffer = constraints.write_region->buffer;
    return IsCopyBetweenScope(src_buffer, tgt_buffer, runtime::StorageRank::kGlobal,
                              runtime::StorageRank::kShared) &&
           is_one(constraints.add_local_stage);
  }
};

/*!
 * \brief Add a cache stage in shared memory. Perform tensor core rewrite for wmma->shared, and
 *  perform coalescing and vectorizing for shared->global.
 */
class WmmaToGlobal : public RewriteRule {
 public:
  WmmaToGlobal() = default;
  Stmt Rewrite(const Stmt& stmt, const ConstraintSet& constraints, OutputSet* output) const final;
  bool CanApply(const Stmt& stmt, const ConstraintSet& constraints) const final {
    Buffer src_buffer = constraints.read_region->buffer;
    Buffer tgt_buffer = constraints.write_region->buffer;
    return IsCopyBetweenScope(src_buffer, tgt_buffer, runtime::StorageRank::kWMMAAccumulator,
                              runtime::StorageRank::kGlobal);
  }
};

/*!
 * \brief Add a cache stage in shared memory. Perform tensor core rewrite for mma->shared, and
 *  perform coalescing and vectorizing for shared->global.
 */
class MmaToGlobal : public RewriteRule {
 public:
  MmaToGlobal() = default;
  Stmt Rewrite(const Stmt& stmt, const ConstraintSet& constraints, OutputSet* output) const final;
  bool CanApply(const Stmt& stmt, const ConstraintSet& constraints) const final {
    Buffer src_buffer = constraints.read_region->buffer;
    Buffer tgt_buffer = constraints.write_region->buffer;
    return IsCopyBetweenScope(src_buffer, tgt_buffer, runtime::StorageRank::kMMAMatrixC,
                              runtime::StorageRank::kGlobal);
  }
};

/*!
 * \brief Rewrite shared->wmma data copy with load_matrix_sync
 */
class SharedToWmma : public RewriteRule {
 public:
  SharedToWmma() = default;
  Stmt Rewrite(const Stmt& stmt, const ConstraintSet& constraints, OutputSet* output) const final;
  bool CanApply(const Stmt& stmt, const ConstraintSet& constraints) const final {
    Buffer src_buffer = constraints.read_region->buffer;
    Buffer tgt_buffer = constraints.write_region->buffer;
    return IsCopyBetweenScope(src_buffer, tgt_buffer, runtime::StorageRank::kShared,
                              runtime::StorageRank::kWMMAMatrixA) ||
           IsCopyBetweenScope(src_buffer, tgt_buffer, runtime::StorageRank::kShared,
                              runtime::StorageRank::kWMMAMatrixB);
  }
};

/*!
 * \brief Rewrite wmma->shared data copy with store_matrix_sync
 */
class WmmaToShared : public RewriteRule {
 public:
  WmmaToShared() = default;
  Stmt Rewrite(const Stmt& stmt, const ConstraintSet& constraints, OutputSet* output) const final;
  bool CanApply(const Stmt& stmt, const ConstraintSet& constraints) const final {
    Buffer src_buffer = constraints.read_region->buffer;
    Buffer tgt_buffer = constraints.write_region->buffer;
    return IsCopyBetweenScope(src_buffer, tgt_buffer, runtime::StorageRank::kWMMAAccumulator,
                              runtime::StorageRank::kShared);
  }
};

/*!
 * \brief Insert a cache stage to the compute location
 * \param stmt the stmt
 * \param is_write_cache whether to write a read cache or write cache
 * \param storage_scope the storage scope of the new cache
 * \param compute_location the compute location.
 * \param outer_loops the outer loops of this stmt
 * \param alloc_buffer the new cache block
 * \return a pair. The first is the stmt after transformation.
 *         The second is the SeqStmt that contains 2 stages (one original and another inserted).
 */
std::pair<Stmt, SeqStmt> InsertCacheStage(Stmt stmt, bool is_write_cache, String storage_scope,
                                          Optional<For> compute_location,
                                          const Array<For>& outer_loops, Buffer* alloc_buffer);

}  // namespace tir
}  // namespace tvm

#endif  // TVM_TIR_TRANSFORMS_MEMHAMMER_REWRITE_RULE_H_
