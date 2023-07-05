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
#include <tvm/meta_schedule/schedule/cuda/thread_bind.h>

#include <algorithm>
#include <limits>

#include "../utils.h"

namespace tvm {
namespace meta_schedule {

class AutoBindNode : public ScheduleRuleNode {
 public:
  // Inherited from ScheduleRuleNode
  void InitializeWithTuneContext(const TuneContext& context) final {
    CHECK(context->target.defined()) << "ValueError: target is not defined";
    Optional<Integer> max_threads_per_block =
        context->target.value()->GetAttr<Integer>("max_threads_per_block");
    CHECK(max_threads_per_block.defined())
        << "ValueError: missing attribute `max_threads_per_block` in the target";
    this->max_threads_per_block_ = max_threads_per_block.value().IntValue();
  }

  // Inherited from ScheduleRuleNode
  Array<tir::Schedule> Apply(const tir::Schedule& sch, const tir::BlockRV& block_rv) final;

  // Inherited from ScheduleRuleNode
  ScheduleRule Clone() const final {
    ObjectPtr<AutoBindNode> n = make_object<AutoBindNode>(*this);
    return ScheduleRule(n);
  }

 public:
  /*! \brief The max number of threads per block from Target */
  int64_t max_threads_per_block_ = -1;
  /*! \brief The max number of threadblocks in the cuda device */
  int64_t max_threadblocks_ = -1;
  /*! \brief thread_extents Candidates of thread axis extent. */
  Array<Integer> thread_extents_;

  void VisitAttrs(tvm::AttrVisitor* v) {
    // `max_threads_per_block_` is not visited
    // `max_threadblocks_` is not visited
    // `thread_extents_` is not visited
  }

  static constexpr const char* _type_key = "meta_schedule.AutoBind";
  TVM_DECLARE_FINAL_OBJECT_INFO(AutoBindNode, ScheduleRuleNode);
};

Array<tir::Schedule> AutoBindNode::Apply(const tir::Schedule& sch, const tir::BlockRV& block_rv) {
  ICHECK_NE(this->max_threads_per_block_, -1);
  auto get_factor = MakeFactorSampler(sch, this->thread_extents_);
  BindBlockThreadIdx(sch, block_rv, max_threadblocks_, max_threads_per_block_, get_factor);
  return {sch};
}

ScheduleRule ScheduleRule::AutoBind(int max_threadblocks, Array<Integer> thread_extents,
                                    int max_threads_per_block) {
  ObjectPtr<AutoBindNode> n = make_object<AutoBindNode>();
  n->max_threadblocks_ = max_threadblocks;
  n->max_threads_per_block_ = max_threads_per_block;
  n->thread_extents_ = std::move(thread_extents);
  return ScheduleRule(n);
}

TVM_REGISTER_NODE_TYPE(AutoBindNode);
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleRuleAutoBind").set_body_typed(ScheduleRule::AutoBind);

}  // namespace meta_schedule
}  // namespace tvm
