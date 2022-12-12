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
namespace tir {

bool IsRootBlock(const Schedule& sch, const BlockRV& block_rv) {
  StmtSRef block_sref = sch->GetSRef(block_rv);
  return block_sref->parent == nullptr;
}

bool CheckSpatialPrimFunc(const Schedule& sch, const BlockRV& root_block_rv) {
  return IsSpatialPrimFunc(
      GetRef<PrimFunc>(GetRootPrimFunc(sch->mod(), sch->Get(root_block_rv).get(), nullptr)));
}

}  // namespace tir
}  // namespace tvm

namespace tvm {
namespace meta_schedule {

class ParallelizeVectorizeUnrollNode : public ScheduleRuleNode {
 public:
  // Inherited from ScheduleRuleNode
  void InitializeWithTuneContext(const TuneContext& context) final {
    ICHECK(context->target.defined());
    if (this->max_jobs_per_core != -1) {
      Target target = context->target.value();
      this->max_parallel_extent_ = GetTargetNumCores(target) * max_jobs_per_core;
    }
  }

  // Inherited from ScheduleRuleNode
  Array<tir::Schedule> Apply(const tir::Schedule& sch, const tir::BlockRV& root_rv) {
    // Currently only mark the root block with annotations.
    if (!tir::IsRootBlock(sch, root_rv)) {
      return {sch};
    }

    // Parallelization
    if (max_jobs_per_core != -1) {
      sch->Annotate(root_rv, tir::attr::meta_schedule_parallel,
                    Integer(this->max_parallel_extent_));
    }
    // Vectorization
    if (max_vectorize_extent != -1) {
      sch->Annotate(root_rv, tir::attr::meta_schedule_vectorize, Integer(max_vectorize_extent));
    }
    // Unroll
    if (!unroll_max_steps.empty() && !tir::CheckSpatialPrimFunc(sch, root_rv)) {
      int n = unroll_max_steps.size();
      double prob = 1.0 / n;
      Array<FloatImm> probs(n, FloatImm(DataType::Float(64), prob));
      PrimExpr max_step = sch->SampleCategorical(unroll_max_steps, probs);
      if (unroll_explicit) {
        sch->Annotate(root_rv, tir::attr::meta_schedule_unroll_explicit, max_step);
      } else {
        sch->Annotate(root_rv, tir::attr::meta_schedule_unroll_implicit, max_step);
      }
    }
    return {sch};
  }

  // Inherited from ScheduleRuleNode
  ScheduleRule Clone() const final {
    ObjectPtr<ParallelizeVectorizeUnrollNode> n =
        make_object<ParallelizeVectorizeUnrollNode>(*this);
    return ScheduleRule(n);
  }

 public:
  /*!
   * \brief The maximum number of jobs to be launched per CPU core. It sets the
   * upper limit of CPU parallelism, i.e. `num_cores * max_jobs_per_core`. Use -1 to disable
   * parallelism.
   */
  int64_t max_jobs_per_core;
  /*!
   * \brief The maximum extent to be vectorized.
   * It sets the upper limit of the hardware target vectorization. Use -1 to disable vectorization.
   */
  int max_vectorize_extent;
  /*!
   * \brief The options of the maximum number of unroll steps to be done.
   * Use an empty array to disable unroll.
   */
  Array<Integer> unroll_max_steps;
  /*! \brief Whether to explicitly unroll the loop, or just add an "unroll" pragma. */
  bool unroll_explicit;
  /*! \brief The number of maximum available jobs in CPU. */
  int64_t max_parallel_extent_;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("max_jobs_per_core", &max_jobs_per_core);
    v->Visit("max_vectorize_extent", &max_vectorize_extent);
    v->Visit("unroll_max_steps", &unroll_max_steps);
    v->Visit("unroll_explicit", &unroll_explicit);
    // `max_parallel_extent_` is not visited
  }

  static constexpr const char* _type_key = "meta_schedule.ParallelizeVectorizeUnroll";
  TVM_DECLARE_FINAL_OBJECT_INFO(ParallelizeVectorizeUnrollNode, ScheduleRuleNode);
};

ScheduleRule ScheduleRule::ParallelizeVectorizeUnroll(int max_jobs_per_core,
                                                      int max_vectorize_extent,
                                                      Array<Integer> unroll_max_steps,
                                                      bool unroll_explicit) {
  ObjectPtr<ParallelizeVectorizeUnrollNode> n = make_object<ParallelizeVectorizeUnrollNode>();
  n->max_jobs_per_core = max_jobs_per_core;
  n->max_vectorize_extent = max_vectorize_extent;
  n->unroll_max_steps = unroll_max_steps;
  n->unroll_explicit = unroll_explicit;
  n->max_parallel_extent_ = -1;
  return ScheduleRule(n);
}

TVM_REGISTER_NODE_TYPE(ParallelizeVectorizeUnrollNode);
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleRuleParallelizeVectorizeUnroll")
    .set_body_typed(ScheduleRule::ParallelizeVectorizeUnroll);

}  // namespace meta_schedule
}  // namespace tvm
