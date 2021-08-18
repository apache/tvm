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
#ifndef TVM_TIR_SCHEDULE_SCHEDULE_H_
#define TVM_TIR_SCHEDULE_SCHEDULE_H_

#include <tvm/tir/schedule/state.h>
#include <tvm/tir/schedule/trace.h>

namespace tvm {
namespace tir {

/*! \brief The level of detailed error message rendering */
enum class ScheduleErrorRenderLevel : int32_t {
  /*! \brief Render a detailed error message */
  kDetail = 0,
  /*! \brief Render the error in fast mode */
  kFast = 1,
  /*! \brief No error message at all */
  kNone = 2,
};

/**************** Random variable: BlockRV ****************/

/*! \brief A random variable that evaluates to a TensorIR block */
class BlockRVNode : public runtime::Object {
 public:
  void VisitAttrs(tvm::AttrVisitor* v) {}
  static constexpr const char* _type_key = "tir.BlockRV";
  TVM_DECLARE_FINAL_OBJECT_INFO(BlockRVNode, runtime::Object);
};

/*!
 * \brief Managed reference to BlockRVNode
 * \sa BlockRVNode
 */
class BlockRV : public runtime::ObjectRef {
 public:
  /*! \brief Constructor */
  TVM_DLL BlockRV();
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(BlockRV, runtime::ObjectRef, BlockRVNode);
};

/**************** Random variable: LoopRV ****************/

/*! \brief A random variable that evaluates to a TensorIR for loop */
class LoopRVNode : public runtime::Object {
 public:
  void VisitAttrs(tvm::AttrVisitor* v) {}
  static constexpr const char* _type_key = "tir.LoopRV";
  TVM_DECLARE_FINAL_OBJECT_INFO(LoopRVNode, runtime::Object);
};

/*!
 * \brief Managed reference to LoopRVNode
 * \sa LoopRVNode
 */
class LoopRV : public runtime::ObjectRef {
 public:
  /*! \brief Constructor */
  TVM_DLL LoopRV();
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(LoopRV, runtime::ObjectRef, LoopRVNode);
};

/**************** Random variable: ExprRV ****************/

/*! \brief An expr random variable */
using ExprRV = PrimExpr;

using ExprRVNode = PrimExprNode;

/**************** The Schedule class ****************/

class Schedule;

/*! \brief The user-facing schedule class */
class ScheduleNode : public runtime::Object {
  friend class Schedule;

 public:
  virtual ~ScheduleNode() = default;

  static constexpr const char* _type_key = "tir.Schedule";
  TVM_DECLARE_FINAL_OBJECT_INFO(ScheduleNode, runtime::Object);

 public:
  /*! \brief Get the IRModule associated with this schedule. */
  virtual IRModule mod() const { return state()->mod; }
  /*! \return The internal state of scheduling */
  virtual ScheduleState state() const = 0;
  /*! \return The internally maintained trace of scheduling program execution */
  virtual Optional<Trace> trace() const = 0;
  /*!
   * \brief Returns a copy of the schedule, including both its state and its symbol table,
   * guaranteeing that
   * 1) SRef tree is completely reconstructed;
   * 2) The IRModule being scheduled is not modified;
   * 3) All the random variables are valid in the copy, pointing to the corresponding sref
   * reconstructed
   */
  virtual Schedule Copy() const = 0;
  /*!
   * \brief Seed the randomness
   * \param seed The new random seed, -1 if use device random, otherwise non-negative
   */
  virtual void Seed(int64_t seed = -1) {
    LOG(FATAL) << "ValueError: The schedule cannot be seeded because no randomness is allowed";
  }

 public:
  /******** Lookup/Remove random variables ********/
  /*!
   * \brief Get the block corresponding to the specific BlockRV
   * \param block_rv The BlockRV to be looked up
   * \return The corresponding block
   */
  virtual Block Get(const BlockRV& block_rv) const = 0;
  /*!
   * \brief Get the for loop corresponding to the specific LoopRV
   * \param loop_rv The LoopRV to be looked up
   * \return The corresponding for loop
   */
  virtual For Get(const LoopRV& loop_rv) const = 0;
  /*!
   * \brief Get the expr corresponding to the specific random variable
   * \param expr_rv The random variable to be looked up
   * \return The corresponding expr
   */
  virtual PrimExpr Get(const ExprRV& expr_rv) const = 0;
  /*!
   * \brief Get the block sref corresponding to the specific BlockRV
   * \param block_rv The BlockRV to be looked up
   * \return The corresponding block sref
   */
  virtual StmtSRef GetSRef(const BlockRV& block_rv) const = 0;
  /*!
   * \brief Get the loop sref corresponding to the specific LoopRV
   * \param loop_rv The LoopRV to be looked up
   * \return The corresponding loop sref
   */
  virtual StmtSRef GetSRef(const LoopRV& loop_rv) const = 0;
  /*!
   * \brief Get the block/loop sref corresponding to the specific statement
   * \param stmt The statement to be looked up
   * \return The corresponding block/loop sref
   */
  virtual StmtSRef GetSRef(const StmtNode* stmt) const;
  /*!
   * \brief Get the block/loop sref corresponding to the specific statement
   * \param stmt The statement to be looked up
   * \return The corresponding block/loop sref
   */
  StmtSRef GetSRef(const Stmt& stmt) const { return this->GetSRef(stmt.get()); }
  /*!
   * \brief Remove a block random variable from the symbol table
   * \param block_rv The random variable to be removed
   */
  virtual void RemoveRV(const BlockRV& block_rv) = 0;
  /*!
   * \brief Remove a loop random variable from the symbol table
   * \param loop_rv The random variable to be removed
   */
  virtual void RemoveRV(const LoopRV& loop_rv) = 0;
  /*!
   * \brief Remove an integer random variable from the symbol table
   * \param expr_rv The random variable to be removed
   */
  virtual void RemoveRV(const ExprRV& expr_rv) = 0;

 public:
  /******** Schedule: Sampling ********/
  /******** Schedule: Get blocks & loops ********/
  /*!
   * \brief Retrieve a block in a specific function with its name
   * \param name The name of the block to be retrieved
   * \param func_name The name of the function
   * \return The block retrieved
   * \note Indexing error is raised if 0 or multiple blocks exist with the specific name
   */
  virtual BlockRV GetBlock(const String& name, const String& func_name = "main") = 0;
  /*!
   * \brief Get the parent loops of the block in its scope, from outer to inner
   * \param block_rv The query block
   * \return A list of loops above the given block in its scope, from outer to inner
   */
  virtual Array<LoopRV> GetLoops(const BlockRV& block_rv) = 0;
  /******** Schedule: Transform loops ********/
  /*!
   * \brief Fuse a list of consecutive loops into one. It requires:
   * 1) The loops can't have annotations or thread bindings.
   * 2) The (i+1)-th loop must be the only child of the i-th loop.
   * 3) All loops must start with 0.
   * \param loop_rvs The loops to be fused
   * \return The new loop after fusion
   */
  virtual LoopRV Fuse(const Array<LoopRV>& loop_rvs) = 0;
  /*!
   * \brief Split a loop into a list of consecutive loops. It requires:
   * 1) The loop can't have annotation or thread binding.
   * 2) The loop must start with 0.
   * \param loop_rv The loop to be split
   * \param factors The tiling factors, and at most one of which is -1, which means that
   * factor is inferred.
   * \return The new loops after split
   */
  virtual Array<LoopRV> Split(const LoopRV& loop_rv, const Array<Optional<ExprRV>>& factors) = 0;
  /******** Schedule: Manipulate ForKind ********/
  /*!
   * \brief Parallelize the input loop. It requires:
   * 1) The scope block that the loop is in should have stage-pipeline property
   * 2) All the blocks under the loop are complete blocks or reduction blocks, and have affine
   * bindings
   * 3) For each block under the loop, the loop can only be contained in data-parallel block iters'
   * bindings
   * \param loop_rv The loop to be parallelized
   */
  virtual void Parallel(const LoopRV& loop_rv) = 0;
  /*!
   * \brief Vectorize the input loop. It requires:
   * 1) The scope block that the loop is in should have stage-pipeline property
   * 2) All the blocks under the loop are complete blocks or reduction blocks, and have affine
   * bindings
   * 3) For each block under the loop, the loop can only be contained in data-parallel block iters'
   * bindings
   * \param loop_rv The loop to be vectorized
   */
  virtual void Vectorize(const LoopRV& loop_rv) = 0;
  /*!
   * \brief Bind the input loop to the given thread axis. It requires:
   * 1) The scope block that the loop is in should have stage-pipeline property
   * 2) All the blocks under the loop are complete blocks or reduction blocks, and have affine
   * bindings
   * 3) For each block under the loop, if the thread axis starts with "threadIdx`, the loop can only
   * be contained in data-parallel block iter and reduction block iters' bindings. Otherwise the
   * loop can only be contained in data-parallel block iters' bindings
   * \param loop_rv The loop to be bound to the thread axis
   * \param thread_axis The thread axis to be bound to the loop
   */
  virtual void Bind(const LoopRV& loop_rv, const String& thread_axis) = 0;
  /*!
   * \brief Unroll the input loop. It requires nothing
   * \param loop_rv The loop to be unrolled
   */
  virtual void Unroll(const LoopRV& loop_rv) = 0;
  /******** Schedule: Insert cache stages ********/
  /******** Schedule: Compute location ********/
  /*!
   * \brief Inline a block into its consumer(s). It requires:
   * 1) The block is a complete non-root block, which only produces one buffer
   * 2) The block must not be the only leaf in the scope.
   * 3) The body of the block must be a BufferStore statement in the form of,
   *    A[i, j, k, ...] = ...
   * where the indices of the LHS are all distinct atomic variables,
   * and no variables other than those indexing variables are allowed in the statement.
   * \param block The block to be inlined to its consumer(s)
   */
  virtual void ComputeInline(const BlockRV& block) = 0;
  /*!
   * \brief Inline a block into its only producer. It requires:
   * 1) The block is a complete non-root block, which only produces and consumers one buffer
   * 2) The block must not be the only leaf in the scope.
   * 3) The only producer of the block is a read-after-write producer and a complete non-root block
   * 4) The body of the block must be a BufferStore statement in the form of,
   *    B[f(i, j, k, ...)] = g(i, j, k, A[i, j, k, ...] ...)
   * where the indices of each `BufferLoad` on the RHS are all distinct atomic variables,
   * and no variables other than those indexing variables are allowed in the statement.
   * \param block The block to be inlined to its producer
   */
  virtual void ReverseComputeInline(const BlockRV& block) = 0;
  /******** Schedule: Reduction ********/
  /*!
   * \brief Factorize an associative reduction block by the specified loop.
   * \details An associative reduction cannot be parallelized directly,
   * because it leads to potential race condition during accumulation.
   * Alternatively, the reduction could be factorized on a loop with the following steps:
   * - Step 1: evenly slice the reduction into `n` separate chunks, where `n` is the loop extent
   * - Step 2: compute the chunks separately and write the result into `n` intermediate buffers;
   * - Step 3: accumulate the `n` separate buffer into the result buffer.
   * Note that the Step 2 above introduces opportunities for parallelization.
   * RFactor is a schedule primitive that implements the transformation described above.
   * \param loop_rv The loop outside block we want to do rfactor
   * \param factor_axis The position where the new dimension is placed in the new introduced rfactor
   *                    buffer. Suppose the original reduction block writes to buffer `B` with
   *                    ndim(B) dimensions, then `factor_axis` should be in range `[-ndim(B) - 1,
   *                    ndim(B)]`, and the negative index will be normalized to a non-negative one
   * \return The rfactor block
   */
  virtual BlockRV RFactor(const LoopRV& loop_rv, int factor_axis) = 0;
  /******** Schedule: Block annotation ********/
  /*!
   * \brief Set alignment requirement for specific dimension such that
   *        stride[axis] == k * factor + offset for some k. This is useful to set memory layout for
   *        more friendly memory access pattern. For example, we can set alignment to be factor=2,
   *        offset=1 to avoid bank conflict for thread access on higher dimension in GPU shared
   *        memory.
   * \param block_rv The producer block of the buffer
   * \param buffer_index The index of the buffer in block's write region
   * \param axis The dimension to be specified for alignment
   * \param factor The factor multiple of alignment
   * \param offset The required offset factor
   */
  virtual void StorageAlign(const BlockRV& block_rv, int buffer_index, int axis, int factor,
                            int offset) = 0;
  /******** Schedule: Blockize & Tensorize ********/
  /******** Schedule: Annotation ********/
  /******** Schedule: Misc ********/
  /*! \brief A no-op that marks the start of postprocessing phase of scheduling */
  virtual void EnterPostproc() = 0;
};

/*!
 * \brief Managed reference to ScheduleNode
 *
 * A schedule is a set of transformations that change the order of computation but
 * preserve the semantics of computation. Some example of schedules:
 * 1) Split a loop into two;
 * 2) Reorder two loops;
 * 3) Inline the computation of a specific buffer into its consumer
 *
 * The schedule class stores auxiliary information to schedule correctly and efficiently.
 *
 * Link to tutorial: https://tvm.apache.org/docs/tutorials/language/schedule_primitives.html
 *
 * \sa ScheduleNode
 */
class Schedule : public runtime::ObjectRef {
 public:
  /*!
   * \brief Construct a concrete TensorIR schedule from an IRModule
   * \param mod The IRModule to be scheduled
   * \param debug_mask Do extra correctness checking after the class creation
   * and each time after calling the Replace method.
   * \param error_render_level The level of error rendering
   * \return The concrete schedule created
   * \sa ScheduleDebugMask
   * \note The checks performed includes:
   * 1) VerifySRefTree
   * 2) VerifyCachedFlags
   */
  TVM_DLL static Schedule Concrete(IRModule mod, int debug_mask,
                                   ScheduleErrorRenderLevel error_render_level);
  /*!
   * \brief Construct a traced concrete TensorIR schedule from an IRModule
   * \param mod The IRModule to be scheduled
   * \param debug_mask Do extra correctness checking after the class creation
   * and each time after calling the Replace method.
   * \param error_render_level The level of error rendering
   * \return The concrete schedule created
   * \sa ScheduleDebugMask
   * \note The checks performed include:
   * 1) VerifySRefTree
   * 2) VerifyCachedFlags
   */
  TVM_DLL static Schedule Traced(IRModule mod, int debug_mask,
                                 ScheduleErrorRenderLevel error_render_level);
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(Schedule, runtime::ObjectRef, ScheduleNode);
};

}  // namespace tir
}  // namespace tvm

#endif  // TVM_TIR_SCHEDULE_SCHEDULE_H_
