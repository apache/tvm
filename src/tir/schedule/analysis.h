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

#include <tvm/tir/schedule/state.h>

#include <unordered_map>
#include <unordered_set>
#include <vector>

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
 * \param require_subtree_compact_dataflow A boolean indicating whether to check
 * subtree compact dataflow property. The scope root may have one or more subtrees rooted at
 * its direct children, and this property requires all the blocks of the subtree
 * that the specified sref is in to be complete block or reduction block.
 * \throw ScheduleError if
 * 1) the sref has been the root of the AST (so it has no scope root), or
 * 2) require_stage_pipeline = true, but its scope root is not a stage pipeline
 * 3) require_subtree_compact_dataflow = true, but the subtree that the sref is in doesn't satisfy
 * the compact dataflow condition, i.e. a block in the subtree is neither complete block nor
 * reduction block
 * \return The block sref to the scope root
 */
StmtSRef GetScopeRoot(const ScheduleState& self, const StmtSRef& sref, bool require_stage_pipeline,
                      bool require_subtree_compact_dataflow);

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
 * \throw ScheduleError If the block is not a reduction block
 */
void CheckCompleteOrReductionBlock(const ScheduleState& self, const StmtSRef& block_sref,
                                   const StmtSRef& scope_root_sref);

/*!
 * \brief Check if the block is an output block, i.e. the buffer regions written by the block are
 * allocated under the current scope
 * \param self The schedule state
 * \param block_sref The block to be checked
 * \param scope_root_sref The scope root of the block
 * \return A boolean flag indicating if the block is an output block
 */
bool IsOutputBlock(const ScheduleState& self, const StmtSRef& block_sref,
                   const StmtSRef& scope_root_sref);

/*!
 * \brief Check if the block is an output block, i.e. the buffer regions written by the block are
 * allocated under the current scope
 * \param self The schedule state
 * \param block_sref The block to be checked
 * \param scope_root_sref The scope root of the block
 * \throw ScheduleError if the block is an output block
 */
void CheckNotOutputBlock(const ScheduleState& self, const StmtSRef& block_sref,
                         const StmtSRef& scope_root_sref);

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
 * \param is_write A boolean flag to indicate querying write buffer or read buffer.
 * \return The buffer of the n-th read/write region of the block.
 * \throw ScheduleError If the buffer index is out of bound.
 */
Buffer GetNthAccessBuffer(const ScheduleState& self, const Block& block, int n, bool is_write);

/******** Commutative Reducer ********/

/*!
 * \brief Get the list of the registered reducer-getter functions
 * \return The list of the registered reducer-getter functions
 * \sa ReducerRegistry
 */
std::vector<TypedPackedFunc<CommReducer(DataType)>> GetReducerGetters();

/*!
 * \brief Given the input identity and the combiner BufferStore of a reduction, extract the
 * corresponding commutative reducer and its lhs, rhs if possible.
 * \param identity The identity of the reduction
 * \param combiner The combiner of the reduction
 * \param result_reducer The extracted CommReducer
 * \param lhs The extracted lhs of the reducer
 * \param rhs The extracted rhs of the reducer
 * \return A boolean indicating whether a corresponding commutative reducer is found
 */
bool FromIdentityCombiner(const PrimExpr& identity, const BufferStore& combiner,
                          CommReducer* result_reducer, PrimExpr* lhs, PrimExpr* rhs);

}  // namespace tir
}  // namespace tvm

#endif  // TVM_TIR_SCHEDULE_ANALYSIS_H_
