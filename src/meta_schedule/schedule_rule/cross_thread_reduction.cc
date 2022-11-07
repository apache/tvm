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
#include "../utils.h"

namespace tvm {
namespace meta_schedule {

class CrossThreadReductionNode : public ScheduleRuleNode {
 public:
  // Inherited from ScheduleRuleNode
  void InitializeWithTuneContext(const TuneContext& context) final {
    ICHECK(context->target.defined());
    Target target = context->target.value();

    Optional<Integer> opt_max_threads_per_block = target->GetAttr<Integer>("max_threads_per_block");
    Optional<Integer> opt_warp_size = target->GetAttr<Integer>("thread_warp_size");

    if (!opt_max_threads_per_block.defined()) {
      TVM_PY_LOG(WARNING, context->logger)
          << "Target does not have attribute \"max_threads_per_block\", therefore the "
             "rule CrossThreadReduction will not be applied";
    }
    if (!opt_warp_size.defined()) {
      TVM_PY_LOG(WARNING, context->logger)
          << "Target does not have attribute \"thread_warp_size\", therefore the rule "
             "CrossThreadReduction will not be applied";
    }
    max_threads_per_block = opt_max_threads_per_block.value_or(Integer(-1))->value;
    warp_size = opt_warp_size.value_or(Integer(-1))->value;
  }

  // Inherited from ScheduleRuleNode
  Array<tir::Schedule> Apply(const tir::Schedule& sch, const tir::BlockRV& block_rv) final {
    // Step 0. Check the conditions of this rule.
    if (max_threads_per_block == -1 || warp_size == -1) {
      return {sch};
    }
    const tir::StmtSRef& block_sref = sch->GetSRef(block_rv);
    if (!NeedsRFactorOrCrossThreadReduction(sch->state(), block_sref, max_threads_per_block,
                                            warp_size)) {
      return {sch};
    }

    // Step 1. Make a copy of the original schedule. The new copy is used for scheduling.
    tir::Schedule tmp_sch = sch->Copy();
    tmp_sch->Seed(sch->ForkSeed());

    // Step 2. Check the opportunity for block fusion. We say "fusible", if we can compute-at the
    // block to its consumers. We want to fuse as much as possible because it results in
    // significantly faster schedule.
    // `target_loop` is the loop position where the input block will be computed at.
    // `target_block` is the consumer block that we want to compute-at the input block to.
    // `tgt_block_innermost_loop` is the innermost loop outside the target block.

    auto [fusible, target_loop, target_block, tgt_block_innermost_loop] =
        GetComputeTargetLoopAndBlock(tmp_sch, block_rv);

    // Step 3. Try block fusion.
    int n_candidate = static_cast<int>(thread_extents.size());
    Array<FloatImm> probs(n_candidate, FloatImm(DataType::Float(64), 1.0 / n_candidate));
    tir::ExprRV thread_extent = tmp_sch->SampleCategorical(thread_extents, probs);
    if (fusible) {
      ICHECK(target_block.defined());
      ICHECK(target_loop.defined());

      // Step 3.1.
      // - If the outer loops of `target_block` haven't been bound to "threadIdx.x", we should first
      //   bound the innermost outer loop of `target_block` to threadIdx. Possibly we need to split
      //   the loop before binding.
      // - Otherwise, we search for the extent of "threadIdx.x" and use it as the split factor.
      if (!InThreadScope(tmp_sch, target_block)) {
        const Array<tir::LoopRV>& split_res =
            tmp_sch->Split(tgt_block_innermost_loop, {NullOpt, thread_extent});
        tmp_sch->Bind(split_res[1], "threadIdx.x");
        if (tgt_block_innermost_loop.same_as(target_loop)) {
          target_loop = split_res[0];
        }
      } else {
        thread_extent = GetThreadIdxExtentFromTrace(tmp_sch->trace().value());
      }
      // Step 3.2. Do the compute-at.
      tmp_sch->ComputeAt(block_rv, target_loop, /*preserve_unit_loops=*/true);
      // Step 3.3. Set the storage scope of the output buffer to shared memory.
      tmp_sch->SetScope(block_rv, /*buffer_index=*/0, /*storage_scope=*/"shared");
    }

    // Step 4. Reorder the loop axes if reduction loops are not innermost. After the reordering,
    // fuse all the reduction loops.
    size_t num_spatial_loops;
    tir::LoopRV fused_reduce_loop;
    ReorderAndFuseReductionLoops(tmp_sch, block_rv, &fused_reduce_loop, &num_spatial_loops);
    // Step 5. Split the fused reduction loop and bind the inner one to threadIdx.
    const Array<tir::LoopRV>& split_res =
        tmp_sch->Split(fused_reduce_loop, {NullOpt, thread_extent});
    tmp_sch->Bind(split_res[1], "threadIdx.x");

    return {tmp_sch, sch};
  }

  // Inherited from ScheduleRuleNode
  ScheduleRule Clone() const final {
    ObjectPtr<CrossThreadReductionNode> n = make_object<CrossThreadReductionNode>(*this);
    return ScheduleRule(n);
  }

 private:
  /*!
   * \brief Check whether the input block is in thread scope, i.e., some of its outer loop is
   * bound to threadIdx.
   * \param sch The TensorIR schedule
   * \param block The block to be checked
   * \return A boolean indicating whether the block is in thread scope.
   */
  bool InThreadScope(const tir::Schedule& sch, const tir::BlockRV& block) {
    const Array<tir::LoopRV>& axes = sch->GetLoops(block);
    for (const tir::LoopRV& loop_rv : axes) {
      const tir::For& loop = sch->Get(loop_rv);
      runtime::ThreadScope thread_scope = tir::GetThreadScope(loop.get());
      if (tir::IsThreadIdx(thread_scope)) {
        return true;
      }
    }
    return false;
  }

  /*!
   * \brief Get the ExprRV which used to define the extent of a given loop.
   * \param trace The trace of the schedule, where the extent is to be found
   * \param loop The loop whose extent is to be found
   * \param extent The finding result
   * \return Whether the find is successful.
   */
  bool GetLoopRVExtentSource(const tir::Trace& trace, const tir::LoopRV& loop,
                             tir::ExprRV* extent) {
    for (const tir::Instruction& inst : trace->insts) {
      if (inst->kind->name == "Split") {
        int i = std::find(inst->outputs.begin(), inst->outputs.end(), loop) - inst->outputs.begin();
        CHECK(inst->inputs[1 + i].defined())
            << "ValueError: Extracting an extent which needs inference is not supported so far";
        *extent = Downcast<tir::ExprRV>(inst->inputs[1 + i]);
        return true;
      }
    }
    return false;
  }

  /*!
   * \brief Get the ExprRV extent of "threadIdx.x" in the given schedule trace.
   * \param trace The trace of the schedule, where the extent is to be found
   * \return The extent of "threadIdx.x" in the input schedule
   */
  tir::ExprRV GetThreadIdxExtentFromTrace(const tir::Trace& trace) {
    tir::ExprRV extent{nullptr};
    for (const tir::Instruction& inst : trace->insts) {
      if (inst->kind->name == "Bind" && Downcast<String>(inst->attrs[0]) == "threadIdx.x") {
        if (GetLoopRVExtentSource(trace, Downcast<tir::LoopRV>(inst->inputs[0]), &extent)) {
          return extent;
        }
      }
    }
    CHECK(false) << "ValueError: Unable to get the extent of \"threadIdx.x\"";
    throw;
  }

  /*!
   * \brief Get the compute-at target loop and the first block under the target loop.
   * \param sch The TensorIR schedule
   * \param block_rv The block whose compute-at target loop is queried
   * \return A tuple consisting of
   * 1. a boolean indicating whether the block can be computed at some target loop (a.k.a. fusible);
   * 2. the compute-at target loop when fusible, or a null loop random variable;
   * 3. the first block under the target loop when fusible, or a null block random variable;
   * 4. the innermost loop outside the target block when fusible, or a null block random variable.
   */
  std::tuple<bool, tir::LoopRV, tir::BlockRV, tir::LoopRV> GetComputeTargetLoopAndBlock(
      const tir::Schedule& sch, const tir::BlockRV& block_rv) {
    // Step 0. Due to technical reason of some primitives (e.g., compute-at), if the block is doing
    // a tuple reduction, fusion is temporarily not supported.
    if (sch->Get(block_rv)->writes.size() != 1) {
      return std::make_tuple(false, tir::LoopRV{nullptr}, tir::BlockRV{nullptr},
                             tir::LoopRV{nullptr});
    }

    // Step 1. Get all the consumers of the input block.
    Array<tir::BlockRV> consumers = sch->GetConsumers(block_rv);

    // Step 2. If the block has no consumer or the first consumer needs multi-level tiling, it is
    // not fusible.
    if (consumers.empty() || tir::NeedsMultiLevelTiling(sch->state(), sch->GetSRef(consumers[0]))) {
      return std::make_tuple(false, tir::LoopRV{nullptr}, tir::BlockRV{nullptr},
                             tir::LoopRV{nullptr});
    }

    // Step 3. Calculate the lowest common ancestor of all the consumers.
    // - If the lowest common ancestor is a block:
    //   - if there is only one consumer, the target block is that consumer;
    //   - if there are multiple consumers, they must not share a common loop, and the case is not
    //     fusible;
    // - If the lowest common ancestor is a loop, the target block is also the first consumer.
    const tir::StmtSRef& lca_sref =
        tir::GetSRefLowestCommonAncestor(tir::BlockRVs2StmtSRefs(sch, consumers));
    if (consumers.size() > 1 && lca_sref->StmtAs<tir::BlockNode>() != nullptr) {
      return std::make_tuple(false, tir::LoopRV{nullptr}, tir::BlockRV{nullptr},
                             tir::LoopRV{nullptr});
    }

    // Step 4. Get the outer loops of the target block, and get the compute-at position index.
    Array<tir::LoopRV> tgt_block_loops = sch->GetLoops(consumers[0]);
    int pos = GetComputePosition(sch, sch->GetLoops(block_rv), tgt_block_loops, lca_sref);

    // Step 5. A negative position index means not fusible, and vice-versa.
    if (pos < 0) {
      return std::make_tuple(false, tir::LoopRV{nullptr}, tir::BlockRV{nullptr},
                             tir::LoopRV{nullptr});
    } else {
      return std::make_tuple(true, tgt_block_loops[pos], consumers[0], tgt_block_loops.back());
    }
  }

  /*!
   * \brief Get the compute-at position index of the input block, according to
   * 1. the loops outside the input block;
   * 2. the loops outside the target block;
   * 3. the lowest common ancestor of all the consumers of the input block.
   * \param sch The TensorIR schedule
   * \param block_loops The loops outside the input block
   * \param tgt_block_loops The loops outside the target block
   * \param lca_sref The lowest common ancestor of all the consumers of the input block
   * \return The compute-at position index of the input block
   */
  int GetComputePosition(const tir::Schedule& sch, const Array<tir::LoopRV>& block_loops,
                         const Array<tir::LoopRV>& tgt_block_loops, const tir::StmtSRef& lca_sref) {
    int n_block_loop = static_cast<int>(block_loops.size());
    int n_tgt_block_loop = static_cast<int>(tgt_block_loops.size());

    for (int i = 0; i < n_block_loop && i < n_tgt_block_loop; ++i) {
      if (tir::GetLoopIterType(sch->GetSRef(block_loops[i])) != tir::IterVarType::kDataPar) {
        return i - 1;
      } else if (sch->GetSRef(tgt_block_loops[i]).same_as(lca_sref)) {
        // If the lowest common ancestor is a loop, the compute location of the input block should
        // not be deeper than the LCA loop.
        return i;
      }
    }
    return std::min(n_block_loop, n_tgt_block_loop) - 1;
  }

 public:
  /*! \brief The maximum number of threads allowed in a thread block */
  int max_threads_per_block;
  /*! \brief The number of threads per warp */
  int warp_size;
  /*! \brief Candidates of thread axis extent (values are required to be positive). */
  Array<Integer> thread_extents;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("max_threads_per_block", &max_threads_per_block);
    v->Visit("warp_size", &warp_size);
    v->Visit("thread_extents", &thread_extents);
  }

  static constexpr const char* _type_key = "meta_schedule.CrossThreadReduction";
  TVM_DECLARE_FINAL_OBJECT_INFO(CrossThreadReductionNode, ScheduleRuleNode);
};

ScheduleRule ScheduleRule::CrossThreadReduction(Array<Integer> thread_extents) {
  for (const Integer& extent : thread_extents) {
    CHECK(extent->value > 0) << "ValueError: The candidates of thread extent must be positive";
  }
  ObjectPtr<CrossThreadReductionNode> n = make_object<CrossThreadReductionNode>();
  n->thread_extents = std::move(thread_extents);
  return ScheduleRule(n);
}

TVM_REGISTER_NODE_TYPE(CrossThreadReductionNode);
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleRuleCrossThreadReduction")
    .set_body_typed(ScheduleRule::CrossThreadReduction);

}  // namespace meta_schedule
}  // namespace tvm
