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
#ifndef TVM_TIR_SCHEDULE_ANALYSIS_H_
#define TVM_TIR_SCHEDULE_ANALYSIS_H_

#include <tvm/arith/analyzer.h>
#include <tvm/ir/op.h>
#include <tvm/tir/index_map.h>
#include <tvm/tir/schedule/schedule.h>
#include <tvm/tir/schedule/state.h>

#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "../../runtime/thread_storage_scope.h"

namespace tvm {
namespace tir {

/******** Verification ********/
/*!
 * \brief Verifies the sref tree state is consistent with the IR
 * \param self The schedule state containing the sref to be verified
 * \throw An exception will be thrown if the sref tree is not valid
 */
void VerifySRefTree(const ScheduleState& self);
/*!
 * \brief Verifies the cached flags in the schedule state, including:
 * - affine_binding
 * - region_cover
 * - stage_pipeline
 * \param self The schedule state to be verified
 * \throw An exception will be thrown if some srefs are not valid
 */
void VerifyCachedFlags(const ScheduleState& self);

/******** IR Module ********/
/*!
 * \brief Get PrimFunc and GlobalVar that the root block belongs to
 * \param mod The IRModule
 * \param root_block The root block of the PrimFunc
 * \param result_g_var The result GlobalVar
 * \return The result PrimFunc where the root block belongs to
 * \note This function returns the pointer instead of ObjectRef to avoid later copy-on-write
 */
const PrimFuncNode* GetRootPrimFunc(const IRModule& mod, const StmtNode* root_block,
                                    GlobalVar* result_g_var);

/*!
 * \brief Get the root node of the sref tree, which is the root block of the PrimFunc.
 * \param sref The given sref.
 * \return The root node of the sref tree which contains the given node.
 */
StmtSRef GetSRefTreeRoot(const StmtSRef& sref);

/******** Scope ********/
/*!
 * \brief Checks if scope the specified sref is in is a stage-pipeline and return it
 * \param self The schedule state
 * \param sref The sref whose scope is to be checked
 * \param require_stage_pipeline A boolean indicating whether to check stage pipeline
 * \throw ScheduleError if
 * 1) the sref has been the root of the AST (so it has no scope root), or
 * 2) require_stage_pipeline = true, but its scope root is not a stage pipeline
 * \return The block sref to the scope root
 */
StmtSRef GetScopeRoot(const ScheduleState& self, const StmtSRef& sref, bool require_stage_pipeline);

/*!
 * \brief The information of a block scope, including the leaf blocks,
 * as well as the loop types (spatial, reduction) for each loop in the scope.
 */
struct ScopeBlockLoopInfo {
  /*! \brief A list of the leaf blocks, from left to right */
  std::vector<BlockRealize> realizes;
  /*! \brief The loop vars bound to spatial block iters */
  std::unordered_set<const VarNode*> spatial_vars;
  /*! \brief The loop vars bound to non-spatial block iters */
  std::unordered_set<const VarNode*> non_spatial_vars;
};

/*!
 * \brief Inspect the scope of the given sref
 * \param scope_block The root block of the scope
 * \return The information of the scope
 */
ScopeBlockLoopInfo GetScopeBlockLoopInfo(const Block& scope_block);

/*!
 * \brief Checks whether the block is a complete block under the scope
 * \param self The schedule state
 * \param block_sref The block to be checked
 * \param scope_root_sref The sref to the root block of the scope that `block_sref` is in
 * \return A boolean indicating if the block is a complete block
 * \note Definition of a complete block:
 * 1) All block vars are data parallel
 * 2) Dominant: the block is the only writer of its output,
 * dominating the reader of its output buffers
 * 3) No overlap between the buffers the block reads and writes
 */
bool IsCompleteBlock(const ScheduleState& self, const StmtSRef& block_sref,
                     const StmtSRef& scope_root_sref);

/*!
 * \brief Check if the block is a complete block under the scope
 * \param self The schedule state
 * \param block_sref The sref to the block whose completeness is to be checked
 * \param scope_root_sref The scope root of the block
 * \throw ScheduleError If the block is not a complete block
 */
void CheckCompleteBlock(const ScheduleState& self, const StmtSRef& block_sref,
                        const StmtSRef& scope_root_sref);

/*!
 * \brief Check whether the block is a reduction block under the scope
 * \param self The schedule state
 * \param block_sref The block to be checked
 * \param scope_root_sref The sref to the root block of the scope that `block_sref` is in
 * \return A boolean indicating if the block is a reduction block
 * \note Definition of a reduction block:
 * 1) The block has the `init` statement
 * 2) All the block bindings are quasi-affine expressions
 * 3) All block vars are either data parallel block vars or reduction block vars
 * 4) Dominant: the block is the only writer of its output, dominating the reader of its output
 * buffers
 * 5) The reduction block vars are not used to index the output buffers
 */
bool IsReductionBlock(const ScheduleState& self, const StmtSRef& block_sref,
                      const StmtSRef& scope_root_sref);

/*!
 * \brief Check if the block is a reduction block under the scope
 * \param self The schedule state
 * \param block_sref The sref of the block to be checked
 * \param scope_root_sref The scope root of the block
 * \throw ScheduleError If the block is not a reduction block
 */
void CheckReductionBlock(const ScheduleState& self, const StmtSRef& block_sref,
                         const StmtSRef& scope_root_sref);

/*!
 * \brief Check if the block is a complete block or a reduction block under the scope
 * \param self The schedule state
 * \param block_sref The sref of the block to be checked
 * \param scope_root_sref The scope root of the block
 * \throw ScheduleError If the block is neither a complete block nor a reduction block
 */
void CheckCompleteOrReductionBlock(const ScheduleState& self, const StmtSRef& block_sref,
                                   const StmtSRef& scope_root_sref);

/*!
 * \brief Check the subtree compact dataflow property. The scope root may have one or more subtrees
 *        rooted at its direct children, and this property requires all the blocks of the subtree
 *        that the specified sref is in to be local complete block or local reduction block.
 * \param self The schedule state
 * \param subtree_root The sref of the subtree root to be checked
 */
void CheckSubtreeCompactDataflow(const ScheduleState& self, const StmtSRef& subtree_root);
/*!
 * \brief Check if the block is an output block, i.e. the block writes to at least a buffer that is
 * not allocated under the current scope
 * \param self The schedule state
 * \param block_sref The block to be checked
 * \param scope_root_sref The scope root of the block
 * \return A boolean flag indicating if the block is an output block
 */
bool IsOutputBlock(const ScheduleState& self, const StmtSRef& block_sref,
                   const StmtSRef& scope_root_sref);

/*!
 * \brief Check if the block is not an output block, i.e. all the buffers the block writes to
 * are allocated under the current scope
 * \param self The schedule state
 * \param block_sref The block to be checked
 * \param scope_root_sref The scope root of the block
 * \throw ScheduleError if the block is an output block
 */
void CheckNotOutputBlock(const ScheduleState& self, const StmtSRef& block_sref,
                         const StmtSRef& scope_root_sref);

/*!
 * \brief Extracts the types of the block vars
 * \param block_sref The block to be checked
 * \return A vector of types of the block vars
 */
std::vector<IterVarType> GetBlockVarTypes(const StmtSRef& block_sref);

/*!
 * \brief Checks if a block could be considered as a "write cache"
 * \param block_sref The block to be checked
 * \return A boolean flag indicating if the block is a write cache
 */
bool IsWriteCache(const StmtSRef& block_sref);

/******** Binding ********/
/*!
 * \brief Verifies if the block binding in a specific BlockRealize is an affine binding.
 * The binding can be represented as an injective affine map from the loop iterators.
 * \param realize The BlockRealize to be analyzed
 * \param loop_var_ranges The ranges of the loop variables
 * \param analyzer The analyzer
 * \return A boolean flag indicating if the binding is affine
 */
bool IsAffineBinding(const BlockRealize& realize, const Map<Var, Range>& loop_var_ranges,
                     arith::Analyzer* analyzer);

/*!
 * \brief Check whether a block has an affine binding using the cached flag, and throw an exception
 * if the block does not have an affine binding.
 * \param self The schedule state
 * \param block The block to be checked
 * \throw ScheduleError If the input block does not have an affine binding
 */
void CheckAffineBinding(const ScheduleState& self, Block block);

/*!
 * \brief Check whether a block has an affine binding under the high exclusive sref node,
 * throw an exception if the block does not have an affine binding.
 * \param self The schedule state
 * \param block The block to be checked
 * \param high_exclusive The highest sref node
 * \throw ScheduleError If the input block does not have an affine binding
 */
void CheckPartialAffineBinding(const ScheduleState& self, Block block,
                               const Optional<StmtSRef>& high_exclusive);

/*!
 * \brief Extracts the ranges of loop variables in a path of the sref tree
 * \param low_inclusive The lowest node in the path
 * \param high_exclusive The highest node in the path, defaults to the scope root if not specified
 * \param extra_relax_scope If the scope is not global, the method will look beyond the limit and
 * retrieve extra domains. For example,
 * - if the storage scope is warp, it will look upwards for threadIdx.x
 * - if the storage scope is shared, it will look for threadIdx.x/y/z
 * \return The loop domain
 */
Map<Var, Range> LoopDomainOfSRefTreePath(const StmtSRef& low_inclusive,
                                         const Optional<StmtSRef>& high_exclusive = NullOpt,
                                         const runtime::StorageScope& extra_relax_scope =  //
                                         runtime::StorageScope{runtime::StorageRank::kGlobal, ""});

/*!
 * \brief Returns the block var binding
 * \param realize The BlockRealize to be analyzed
 * \return The block var binding
 */
Map<Var, PrimExpr> GetBindings(const BlockRealize& realize);

/*!
 * \brief Get the vars involved in the bindings of data parallel block vars and reduction block
 * vars, respectively
 * \param block_realize The BlockRealize to be analyzed
 * \param data_par_vars The vars that appear in the binding of any data parallel block iter
 * \param reduce_vars The vars that appear in the binding of any reduction block iter
 * \return A boolean indicating whether the block has block iters that is neither a data parallel
 * block iter nor a reduction block iter
 */
bool GetVarsTouchedByBlockIters(const BlockRealize& block_realize,
                                std::unordered_set<const VarNode*>* data_par_vars,
                                std::unordered_set<const VarNode*>* reduce_vars);

/******** Loop properties ********/
/*!
 * \brief Check the loop starts with zero.
 * \param self The schedule state
 * \param loop_sref The StmtSRef that points to the loop to be checked
 * \param analyzer The arithmetic analyzer
 * \throw ScheduleError If the loop doesn't starts with zero.
 */
void CheckLoopStartsWithZero(const ScheduleState& self, const StmtSRef& loop_sref,
                             arith::Analyzer* analyzer);

/*!
 * \brief Check whether a block has a trivial binding, i.e. each block var is bound to a outer loop,
 * from outer to inner.
 * \param self The schedule state
 * \param block_sref The block to be checked
 * \throw ScheduleError If the block does not have trivial bindings
 */
void CheckBlockHasTrivialBinding(const ScheduleState& self, const StmtSRef& block_sref);

/******** Block-loop relation ********/

/*!
 * \brief Gets StmtSRefs of leaf blocks of a scope where a specific block/loop is in
 * \param self The schedule state
 * \param parent_sref The StmtSRef that points to the parent block/loop
 * \return A list of StmtSRefs of leaf block
 */
Array<StmtSRef> GetChildBlockSRefOnSRefTree(const ScheduleState& self, const StmtSRef& parent_sref);

/*!
 * \brief Gets the BlockRealize of the leaf blocks of a scope where a specific block/loop is in
 * \param parent_sref The StmtSRef that points to the parent block/loop
 * \return A list of leaf BlockRealize
 */
Array<BlockRealize> GetChildBlockRealizeOnSRefTree(const StmtSRef& parent_sref);

/*!
 * \brief Get the BlockRealize of the single child block of the block or loop specified by
 * `parent_sref` on SRef tree, or throw an exception if there is 0 or multiple child blocks
 * \param self The schedule state
 * \param parent_sref The StmtSRef that points to the parent block/loop
 * \return The BlockRealize of the single child block
 * \throw ScheduleError If there is 0 or multiple child blocks
 */
BlockRealize CheckGetSingleChildBlockRealizeOnSRefTree(const ScheduleState& self,
                                                       const StmtSRef& parent_sref);

/*!
 * \brief Get the BlockRealize of the input block
 * \param self The schedule state
 * \param block_sref The StmtSRef of the queried block
 * \return The BlockRealize of the input block
 */
BlockRealize GetBlockRealize(const ScheduleState& self, const StmtSRef& block_sref);

/*!
 * \brief Get the IterVarType of the specific loop, according to the blocks it's bound to
 * \param loop_sref The loop to be checked
 * \return The IterVarType of the specific loop
 */
IterVarType GetLoopIterType(const StmtSRef& loop_sref);

/*!
 * \brief Get the lowest common ancestor of an array of blocks or loops on the sref tree
 * \param srefs The block srefs or loop srefs whose lowest common ancestor is to be queried
 * \return The lowest common ancestor of the input block srefs or loop srefs
 * \note The input array is required to have at least one sref
 */
StmtSRef GetSRefLowestCommonAncestor(const Array<StmtSRef>& srefs);

/*!
 * \brief Checks if the given block has been applied by multi-level tiling. We check this by
 *        examine the block's annotation.
 * \param block_sref The block to be checked
 * \return A boolean indicating whether the block has been multi-level tiled.
 */
bool HasBeenMultiLevelTiled(const StmtSRef& block_sref);

/*!
 * \brief Collect all the feasible compute-at locations of the input block
 * \param self The schedule state
 * \param block_sref The block whose compute-at locations are to be collected
 * \return All the feasible compute-at locations of the input block, given as an array of loop srefs
 *         and an array of their indices among the outer loops of the input block
 */
std::pair<Array<StmtSRef>, std::vector<int>> CollectComputeLocation(const ScheduleState& self,
                                                                    const StmtSRef& block_sref);

/******** Producer-consumer relation ********/

/*!
 * \brief Get the producer blocks to the given block under the given scope
 * \param block_sref The block whose producers are to be retrieved
 * \param scope The block scope where the given block is in
 * \return The producer blocks of the specified block
 */
Array<StmtSRef> GetProducers(const StmtSRef& block_sref, const BlockScope& scope);

/*!
 * \brief Get the consumer blocks to the given block under the given scope
 * \param block_sref The block whose consumers are to be retrieved
 * \param scope The block scope where the given block is in
 * \return The consumer blocks of the specified block
 */
Array<StmtSRef> GetConsumers(const StmtSRef& block_sref, const BlockScope& scope);

/*!
 * \brief A solution to split a ordered list of subtrees into two parts,
 * where producers are on the LHS and consumers are on the RHS.
 * For example, subtree[0, 3) are on the LHS, and subtree[3, 6) are on the RHS.
 */
struct ProducerConsumerSplit {
  /*! \brief Indicates that all producers fall into `subtrees[0, last_producer_position]` */
  int last_producer_position;
  /*! \brief Indicates that all consumers fall into `subtrees[first_consumer_position, ...)` */
  int first_consumer_position;
  /*! \brief The number of given producers visited in `subtrees` */
  int n_producers_visited;
  /*! \brief The number of given consumers visited in `subtrees` */
  int n_consumers_visited;
  /*!
   * \brief Find a split among the given `subtree`
   * \param state The schedule state
   * \param subtrees The ordered list of subtrees to be split
   * \param producer_block_srefs The producers
   * \param consumer_block_srefs The consumers
   * \param block2realize If not null, the corresponding BlockRealize to each block in the scope
   * will be saved in this map
   * \return The valid split points are (last_producer_position, first_consumer_position]
   * \throw ScheduleError is not valid split is found
   */
  static ProducerConsumerSplit Find(
      const ScheduleState& state, const Array<Stmt>& subtrees,
      const Array<StmtSRef>& producer_block_srefs, const Array<StmtSRef>& consumer_block_srefs,
      std::unordered_map<const BlockNode*, const BlockRealizeNode*>* block2realize);
};

/******** Block-buffer relation ********/

/*!
 * \brief Get the n-th read or write buffer of the given block.
 * \param self The schedule state.
 * \param block The queried block.
 * \param n The index of the queried buffer.
 * \param index_type The type of the buffer index, kRead or kWrite.
 * \return The buffer of the n-th read/write region of the block.
 * \throw ScheduleError If the buffer index is out of bound.
 */
Buffer GetNthAccessBuffer(const ScheduleState& self, const Block& block, int n,
                          BufferIndexType index_type);

/*!
 * \brief Get the n-th read or write buffer of the given block.
 * \param self The schedule state.
 * \param block The queried block.
 * \param n The index of the queried buffer.
 * \param index_type The type of the buffer index, kRead or kWrite.
 * \return The n-th read/write region of the block.
 * \throw ScheduleError If the buffer index is out of bound.
 */
BufferRegion GetNthAccessBufferRegion(const ScheduleState& self, const Block& block, int n,
                                      BufferIndexType index_type);

/*!
 * \brief Find the defining site of the buffer in the given block and its ancestors
 * \param block_sref The block sref
 * \param buffer The buffer
 * \return The defining site of the buffer and whether the buffer is allocated (otherwise the
 *         buffer is from match_buffer).
 */
std::pair<Optional<StmtSRef>, bool> GetBufferDefiningSite(const StmtSRef& block_sref,
                                                          const Buffer& buffer);

/******** Reduction Block Related ********/

/*!
 * \brief Get the init values and the BufferStore updates from the input reduction block
 * \param self The schedule state, used for error reporting
 * \param block The block from which the init values and BufferStore updates are extracted from
 * \return The extracted init values and BufferStore updates
 * \throw ScheduleError If rfactor or cross-thread reduction cannot be applied to the block
 */
std::pair<Array<PrimExpr>, Array<BufferStore>> GetInitValuesAndUpdatesFromReductionBlock(
    const Optional<ScheduleState>& self, Block block);

/*!
 * \brief Check whether the input array of IterVars only contains data-parallel and reduction block
 * iters
 * \param iters The input array of IterVars to be checked
 * \return A boolean indicating whether the input array of IterVars only contains data-parallel and
 * reduction block iters
 */
bool ContainsOnlyDataParAndReductionBlockIter(const Array<IterVar>& iters);

/*!
 * \brief Check whether the block's reduction block iters are not used to index the block's output
 * buffers
 * \param block The block to be checked
 * \return A boolean indicating whether the block's reduction block iters are not used to index the
 * block's output buffer
 */
bool ReductionIterNotIndexOutputBuffer(const Block& block);

/*!
 * \brief Given a list of reduction identities and a list of reduction combiners, detect the
 * corresponding commutative reducer, and extract the combiner LHS values and combiner RHS values
 * \param self The schedule state
 * \param identities The reduction identities to be analyzed
 * \param combiners The reduction combiners to be analyzed
 * \return The corresponding CommReducer, combiner LHS values and combiner RHS values
 * \throw ScheduleError If no corresponding commutative reducer can be matched
 */
std::tuple<CommReducer, Array<PrimExpr>, Array<PrimExpr>> GetReducerAndCombinerLhsRhs(
    const Optional<ScheduleState>& self, const Array<PrimExpr>& identities,
    const Array<BufferStore>& combiners);

/******** Commutative Reducer ********/

/*!
 * \brief Get the list of the registered reducer-getter functions
 * \return The list of the registered reducer-getter functions
 * \sa ReducerRegistry
 */
std::vector<runtime::TypedPackedFunc<Optional<CommReducer>(Array<PrimExpr>)>> GetReducerGetters();

/*!
 * \brief Given the input identities and the combiner BufferStores of a reduction, extract the
 * corresponding commutative reducer, LHS values and RHS values, if possible.
 * \param identities The identities of the reduction
 * \param combiners The combiners of the reduction
 * \param result_reducer The extracted CommReducer
 * \param lhs The extracted LHS values of the reducer
 * \param rhs The extracted RHS values of the reducer
 * \return A boolean indicating whether a corresponding commutative reducer is found
 */
bool FromIdentityCombiner(const Array<PrimExpr>& identities, const Array<BufferStore>& combiners,
                          CommReducer* result_reducer, Array<PrimExpr>* lhs, Array<PrimExpr>* rhs);

/******** Misc ********/

/*!
 * \brief Check whether the input storage scope string is valid. Throw an error if not.
 * \param self The schedule state
 * \param storage_scope The storage scope string to be checked
 * \throw ScheduleError If the input storage scope is not valid
 */
void CheckStorageScope(const ScheduleState& self, String storage_scope);

/*!
 * \brief Checks if a block could be successfully computed inline into its consumer
 * \param self The schedule state
 * \param block_sref The block to be checked
 * \return A boolean indicating whether the block could be successfully computed inline
 */
bool CanComputeInline(const ScheduleState& self, const StmtSRef& block_sref);

/*!
 * \brief Checks if a block could be successfully computed inline into its producer
 * \param self The schedule state
 * \param block_sref The block to be checked
 * \return A boolean indicating whether the block could be successfully computed inline
 */
bool CanReverseComputeInline(const ScheduleState& self, const StmtSRef& block_sref);

/*!
 * \brief Checks if a producer block could be successfully computed at the specific loop.
 * \param self The schedule state
 * \param block_sref The block to be moved
 * \param loop_sref The loop where the block to be moved to
 * \param preserve_unit_loops Whether to keep the trivial loops whose extents are 1
 * \return A boolean indicating whether the block could be successfully compute at the specific loop
 */
bool CanComputeAt(const ScheduleState& self, const StmtSRef& block_sref, const StmtSRef& loop_sref,
                  bool preserve_unit_loops);

/*!
 * \brief Checks if a consumer block could be successfully computed at the specific loop.
 * \param self The schedule state
 * \param block_sref The block to be moved
 * \param loop_sref The loop where the block to be moved to
 * \param preserve_unit_loops Whether to keep the trivial loops whose extents are 1
 * \return A boolean indicating whether the block could be successfully reverse compute at the
 * specific loop
 */
bool CanReverseComputeAt(const ScheduleState& self, const StmtSRef& block_sref,
                         const StmtSRef& loop_sref, bool preserve_unit_loops);

/*!
 * \brief Provided the access pattern to a buffer, suggest one of the possible layout
 * transformation to minimize the locality of the access pattern.
 * \param buffer The buffer to be transformed
 * \param indices The access pattern to the buffer
 * \param loops The loops above the buffer
 * \param predicate The predicate of the access
 * \param analyzer Arithmetic analyzer
 */
Optional<IndexMap> SuggestIndexMap(const Buffer& buffer, const Array<PrimExpr>& indices,
                                   const Array<For>& loops, const PrimExpr& predicate,
                                   arith::Analyzer* analyzer);

/*!
 * \brief Checks if the given AST contains the specific operators
 * \param stmt The AST statement to be checked
 * \param ops The list of operators to be checked
 * \return A boolean indicating whether the AST contains the specific operators
 */
bool HasOp(const Stmt& stmt, const Array<Op>& ops);

/*!
 * \brief Checks if the given AST statement contains if-then-else, including
 * 1) IfThenElse statement
 * 2) Select expression
 * 3) The operator `tir.if_then_else`
 * 4) non-constant-true Block predicates
 * \param stmt The AST statement to be checked
 * \return A boolean indicating whether the statement contains the if-then-else pattern
 */
bool HasIfThenElse(const Stmt& stmt);

/*!
 * \brief Given the read/write region, extract the pattern of their index correspondence
 * namely, the mapping from read index to the write index.
 * \param read_region The read region
 * \param write_region The write region
 * \return A tuple of booleans, the extracted pattern
 * 0) exists: if the pattern is found
 * 1) surjective: if the pattern is surjective, i.e. each write index is mapped at least once
 *    e.g. A[i, j] = B[i, i, j]
 * 2) injective: if the pattern is injective, i.e. each write index is mapped at most once.
 *    e.g. A[i, j] = B[i]
 * 3) ordered: if the mapping is ordered
 * 4) no_const_read: if there is no constant indexing in the read indices,
 *    e.g. A[i, j] = B[0, i, j]
 * 5) no_shift_read: if there is no constant shift in the read indices,
 *    e.g. A[i, j] = B[i + 1, j]
 */
std::tuple</*exists=*/bool,
           /*surjective=*/bool,
           /*injective=*/bool,
           /*ordered=*/bool,
           /*no_const_read=*/bool,
           /*no_shift_read=*/bool>
AnalyzeReadWritePattern(const BufferRegion& read_region, const BufferRegion& write_region);

/*!
 * \brief Check if the block is a data parallel block, i.e. all the block vars are data parallel
 * \param block_sref The block to be checked
 * \return A boolean flag indicating if the block is a data parallel block
 */
bool IsSpatial(const StmtSRef& block_sref);

/*!
 * \brief Check whether a block has a trivial binding, i.e. each block var is bound to a outer loop,
 * from outer to inner.
 * \param self The schedule state
 * \param block_sref The block to be checked
 * \return A boolean flag indicating if the block has a trivial binding
 */
bool IsTrivialBinding(const ScheduleState& self, const StmtSRef& block_sref);

/*!
 * \brief Checks if the given block has data reuse opportunity and thus multi-level tiling is
 * beneficial.
 * \param self The schedule state
 * \param block_sref The block to be checked
 * \return A boolean indicating whether the block has data reuse opportunity
 */
bool NeedsMultiLevelTiling(const ScheduleState& self, const StmtSRef& block_sref);

/*!
 * \brief Checks if all the blocks in the PrimFunc is spatial
 * \param func The PrimFunc to be checked
 * \return A boolean indicating whether all the blocks in the PrimFunc is spatial
 */
bool IsSpatialPrimFunc(const PrimFunc& func);

/*!
 * \brief Checks if the rfactor or cross thread reduction is beneficial to the given block.
 * \param self The schedule state.
 * \param block_sref The block to be checked.
 * \param max_parallel_extent The maximum parallel jobs on the target.
 * \param max_parallel_basic The maximum cores on the target.
 * \return A boolean indicating whether the operation is beneficial.
 */
bool NeedsRFactorOrCrossThreadReduction(const tir::ScheduleState& self,   //
                                        const tir::StmtSRef& block_sref,  //
                                        int64_t max_parallel_extent,      //
                                        int64_t max_parallel_basic);

/*!
 * \brief Analyze the buffer region under the sref tree path [dom_low_inclusive, dom_high_exclusive)
 * Relaxation of the region may be used in upper-bound analysis, i.e. some extra region may be added
 * to the result.
 * \param region The buffer region to be analyzed
 * \param dom_low_inclusive The lowest node in the sref tree path
 * \param dom_high_exclusive The highest node in the sref tree path
 * \return An n-dimensional integer set
 */
Array<arith::IntSet> AnalyzeRegionUpperBound(const BufferRegion& region, const PrimExpr& predicate,
                                             const StmtSRef& dom_low_inclusive,
                                             const StmtSRef& dom_high_exclusive,
                                             arith::Analyzer* analyzer);

/*!
 * \brief Analyze the buffer region under the sref tree path [dom_low_inclusive, dom_high_exclusive)
 * Some subregion may be discarded during the lower-bound analysis.
 * \param realize The block realize that touches the buffer region
 * \param region The buffer region to be analyzed
 * \param dom_low_inclusive The lowest node in the sref tree path
 * \param dom_high_exclusive The highest node in the sref tree path
 * \param analyzer The analyzer
 * \return An n-dimensional integer set
 */
Array<arith::IntSet> AnalyzeRegionLowerBound(const BufferRegion& region, const PrimExpr& predicate,
                                             const StmtSRef& dom_low_inclusive,
                                             const StmtSRef& dom_high_exclusive,
                                             arith::Analyzer* analyzer);

/*!
 * \brief Check if buffer indices are all Vars and extr
 * \param buffer_access The BufferLoad or BufferStore
 * \return The indices if the indices are all Vars, otherwise NullOpt
 */
template <typename T>
Optional<Array<Var>> CheckTrivialBufferIndices(const T& buffer_access) {
  Array<Var> indices;
  for (const PrimExpr& index : buffer_access->indices) {
    const VarNode* var = index.as<VarNode>();
    if (var == nullptr) {
      return NullOpt;
    }
    indices.push_back(GetRef<Var>(var));
  }
  return indices;
}

/*!
 * \brief Simplify non-trivial expressions
 * \param expr The expression to be simplified
 * \param analyzer The analyzer
 * \return The simplified expression
 *
 * During scheduling, we often need preserve block iters in trivial expressions that can be
 * simplified to constant values for further scheduling and analysis because simplifing away the
 * block iters may result in loss of information for further analysis.
 */
PrimExpr SimplifyNonTrivialExpr(const PrimExpr& expr, arith::Analyzer* analyzer);

/*! \brief Necessary information used for tensorization */
class TensorizeInfoNode : public Object {
 public:
  /*! \brief Maps loops in a target block to the ones in an intrinsic description */
  Map<tir::StmtSRef, tir::For> loop_map;
  /*! \brief Maps loops in an intrinsic description to its index, outer to inner */
  Map<tir::For, Integer> desc_loop_indexer;
  /*! \brief Optional padded extents of the block iters when padding is needed to match the
   * intrinsic description
   */
  Optional<Array<Integer>> block_iter_paddings;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("loop_map", &loop_map);
    v->Visit("desc_loop_indexer", &desc_loop_indexer);
    v->Visit("block_iter_paddings", &block_iter_paddings);
  }

  static constexpr const char* _type_key = "tir.schedule.TensorizeInfo";
  TVM_DECLARE_FINAL_OBJECT_INFO(TensorizeInfoNode, Object);
};

class TensorizeInfo : public ObjectRef {
 public:
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(TensorizeInfo, ObjectRef, TensorizeInfoNode);
};

/*!
 * \brief Establish a mapping between loops in a target block and an intrinsic description
 * \param self The schedule state to be tensorized
 * \param block_sref The target block to match against
 * \param desc_func The prim func describing the computation to be tensorized
 * \param allow_padding Whether to allow padding the block iters to match the intrinsic description
 * \return TensorizeInfo structure if a valid mapping is found, NullOpt otherwise
 */
Optional<TensorizeInfo> GetTensorizeLoopMapping(const tir::ScheduleState& self,
                                                const tir::StmtSRef& block_sref,
                                                const tir::PrimFunc& desc_func, bool allow_padding);

/*！\brief Necessary information used to perform transformations for tensorization */
class AutoTensorizeMappingInfoNode : public Object {
 public:
  /*! \brief Possible mappings to apply to block iters */
  Array<IndexMap> mappings;

  /* Additional information from AutoTensorizeComparator */

  /*! \brief Mapping from LHS buffer to RHS buffer */
  Map<Buffer, Buffer> lhs_buffer_map;
  /*! \brief Buffer indices on RHS */
  Map<Buffer, Array<PrimExpr>> rhs_buffer_indices;
  /*! \brief Block iters on LHS */
  Array<IterVar> lhs_iters;
  /*! \brief Block iters on RHS */
  Array<IterVar> rhs_iters;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("mappings", &mappings);
    v->Visit("lhs_buffer_map", &lhs_buffer_map);
    v->Visit("rhs_buffer_indices", &rhs_buffer_indices);
    v->Visit("lhs_iters", &lhs_iters);
    v->Visit("rhs_iters", &rhs_iters);
  }

  static constexpr const char* _type_key = "tir.schedule.AutoTensorizeMappingInfo";
  TVM_DECLARE_FINAL_OBJECT_INFO(AutoTensorizeMappingInfoNode, Object);
};

class AutoTensorizeMappingInfo : public ObjectRef {
 public:
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(AutoTensorizeMappingInfo, ObjectRef,
                                            AutoTensorizeMappingInfoNode);
};

/*!
 * \brief Get mapping info between a target block and an intrinsic description including layout
 * transformations to apply.
 * \param self The schedule state
 * \param block_sref The compute block for auto tensorization
 * \param desc_func The prim func describing the computation to be tensorized
 * \return AutoTensorizeMappingInfo structure if a potential mapping is found, NullOpt otherwise.
 * \note Returning a valid AutoTensorizeMappingInfo doesn't guarantee the block can be tensorized.
 * We will need to apply the suggested layout transformations and then match against the tensor
 * intrinsics.
 */
Optional<AutoTensorizeMappingInfo> GetAutoTensorizeMappingInfo(const ScheduleState& self,
                                                               const StmtSRef& block_sref,
                                                               const PrimFunc& desc_func);

/*!
 * \brief Perform basic checks for auto tensorization applicability, such as the structure of
 * arithmetic operations and data types.
 * \param sch The schedule to be tensorized
 * \param block_rv The compute block for auto tensorization
 * \param desc_func The prim func describing the computation to be tensorized
 * \return true if basic conditions are met.
 */
bool CheckAutoTensorizeApplicable(const tir::Schedule& sch, const tir::BlockRV& block_rv,
                                  const tir::PrimFunc& desc_func);
}  // namespace tir
}  // namespace tvm

#endif  // TVM_TIR_SCHEDULE_ANALYSIS_H_
