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

class AddRFactorNode : public ScheduleRuleNode {
 public:
  // Inherited from ScheduleRuleNode
  void InitializeWithTuneContext(const TuneContext& context) final {
    ICHECK(context->target.defined());
    Target target = context->target.value();
    this->max_parallel_basic_ = GetTargetNumCores(target);
    if (this->max_jobs_per_core != -1) {
      this->max_parallel_extent_ = max_parallel_basic_ * max_jobs_per_core;
    }
  }

  // Inherited from ScheduleRuleNode
  Array<tir::Schedule> Apply(const tir::Schedule& sch, const tir::BlockRV& block_rv);

  // Inherited from ScheduleRuleNode
  ScheduleRule Clone() const final {
    ObjectPtr<AddRFactorNode> n = make_object<AddRFactorNode>(*this);
    return ScheduleRule(n);
  }

 public:
  /*!
   * \brief The maximum number of jobs to be launched per core.
   * It sets the uplimit of parallelism, i.e. `num_cores * max_jobs_per_core`.
   * Use -1 to disable parallelism.
   */
  int max_jobs_per_core;
  /*! \brief The maximum size of the innermost factor */
  int max_innermost_factor;
  /*! \brief The number of uplimit of parallelism. */
  int max_parallel_extent_;
  /*! \brief The number of cores. */
  int max_parallel_basic_;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("max_jobs_per_core", &max_jobs_per_core);
    v->Visit("max_innermost_factor", &max_innermost_factor);
    // `max_parallel_extent_` is not visited
    // `max_parallel_basic_` is not visited
  }

  static constexpr const char* _type_key = "meta_schedule.AddRFactor";
  TVM_DECLARE_FINAL_OBJECT_INFO(AddRFactorNode, ScheduleRuleNode);
};

ScheduleRule ScheduleRule::AddRFactor(int max_jobs_per_core,
                                      Optional<Integer> max_innermost_factor) {
  ObjectPtr<AddRFactorNode> n = make_object<AddRFactorNode>();
  n->max_jobs_per_core = max_jobs_per_core;
  n->max_innermost_factor = max_innermost_factor.value_or(Integer(-1))->value;
  n->max_parallel_extent_ = -1;
  n->max_parallel_basic_ = -1;
  return ScheduleRule(n);
}

Array<tir::Schedule> AddRFactorNode::Apply(const tir::Schedule& sch, const tir::BlockRV& block_rv) {
  tir::StmtSRef block_sref = sch->GetSRef(block_rv);
  if (!NeedsRFactorOrCrossThreadReduction(sch->state(), block_sref, max_parallel_extent_,
                                          max_parallel_basic_)) {
    return {sch};
  }

  // Make a copy of the original schedule.
  tir::Schedule ori_sch = sch->Copy();
  ori_sch->Seed(sch->ForkSeed());

  // Reorder the loop axes if reduction loops are not innermost.
  // After the reordering, fuse all the reduction loops.
  size_t num_spatial_loops;
  tir::LoopRV fused_reduce_loop;
  ReorderAndFuseReductionLoops(sch, block_rv, &fused_reduce_loop, &num_spatial_loops);

  // Split the fused reduction loop.
  Array<tir::ExprRV> factors = sch->SamplePerfectTile(fused_reduce_loop, 2, max_innermost_factor);
  Array<tir::LoopRV> split_loops = sch->Split(fused_reduce_loop, {factors.begin(), factors.end()});

  Array<tir::Schedule> res;
  for (const tir::LoopRV& split_loop : split_loops) {
    tir::Schedule sch_tmp = sch->Copy();
    sch_tmp->Seed(sch->ForkSeed());
    try {
      const tir::BlockRV& block_rf = sch_tmp->RFactor(split_loop, num_spatial_loops);
      Array<tir::LoopRV> axes = sch_tmp->GetLoops(block_rf);
      ICHECK_GT(axes.size(), num_spatial_loops);

      // Annotate that the rfactor block, which is now the producer of the original block, needs to
      // be considered by the rule Random-Compute-Location.
      sch_tmp->Annotate(block_rv, tir::attr::meta_schedule_random_compute_producer, Integer(1));
      res.push_back(sch_tmp);
    } catch (const tvm::runtime::Error& e) {
    }
  }

  res.push_back(ori_sch);
  return res;
}

TVM_REGISTER_NODE_TYPE(AddRFactorNode);
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleRuleAddRFactor")
    .set_body_typed(ScheduleRule::AddRFactor);

}  // namespace meta_schedule
}  // namespace tvm
