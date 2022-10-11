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

/*!
 * \brief Parse instruction: sch.bind(..., axis)
 * \param sch The schedule
 * \param inst The instruction to be parsed
 * \param axis The axis name expected
 * \return NullOpt if parsing fails; Otherwise, the extent of thread axis
 */
Optional<Integer> ParseThreadBinding(const Schedule& sch, const Instruction& inst, String axis) {
  static InstructionKind inst_kind_bind = InstructionKind::Get("Bind");
  if (!inst->kind.same_as(inst_kind_bind)) {
    return NullOpt;
  }
  ICHECK_EQ(inst->inputs.size(), 1);
  ICHECK_EQ(inst->attrs.size(), 1);
  String thread_axis = Downcast<String>(inst->attrs[0]);
  if (thread_axis != axis) {
    return NullOpt;
  }
  return Downcast<Integer>(sch->Get(Downcast<LoopRV>(inst->inputs[0]))->extent);
}

/*!
 * \brief Parse instruction: sch.annotate(..., attr::meta_schedule_cooperative_fetch)
 * \param sch The schedule
 * \param inst The instruction to be parsed
 * \param vector_lane The number of vector lane in vectorized cooperative fetching
 * \return NullOpt if parsing fails; Otherwise, the annotated block
 */
Optional<BlockRV> ParseAnnotate(const Schedule& sch, const Instruction& inst,
                                int64_t* vector_lane) {
  static InstructionKind inst_kind_annotate = InstructionKind::Get("Annotate");
  if (!inst->kind.same_as(inst_kind_annotate)) {
    return NullOpt;
  }
  ICHECK_EQ(inst->inputs.size(), 2);
  ICHECK_EQ(inst->attrs.size(), 1);
  String ann_key = Downcast<String>(inst->attrs[0]);
  if (ann_key != attr::meta_schedule_cooperative_fetch) {
    return NullOpt;
  }
  *vector_lane = Downcast<Integer>(sch->Get(Downcast<ExprRV>(inst->inputs[1])))->value;
  return Downcast<BlockRV>(inst->inputs[0]);
}

/*!
 * \brief Parse instruction: sch.annotate(..., attr::warp_execution)
 * \param sch The schedule
 * \param inst The instruction to be parsed
 * \return Whether ths parsing is successful
 */
bool ParseWarpExecutionAnn(const Schedule& sch, const Instruction& inst) {
  static InstructionKind inst_kind_annotate = InstructionKind::Get("Annotate");
  if (!inst->kind.same_as(inst_kind_annotate)) {
    return false;
  }
  ICHECK_EQ(inst->inputs.size(), 2);
  ICHECK_EQ(inst->attrs.size(), 1);
  String ann_key = Downcast<String>(inst->attrs[0]);
  return ann_key == attr::warp_execution;
}

}  // namespace tir

namespace meta_schedule {

/*!
 * \brief Rewrite the cooperative fetch annotation to actual vectorized cooperative fetching
 * in loop bindings.
 */
class RewriteCooperativeFetchNode : public PostprocNode {
 public:
  // Inherited from PostprocNode
  void InitializeWithTuneContext(const TuneContext& context) final {
    if (Optional<Integer> v = context->target.value()->GetAttr<Integer>("thread_warp_size")) {
      this->thread_warp_size_ = v.value()->value;
    } else {
      TVM_PY_LOG(INFO, context->logger) << "'thread_warp_size' is not defined in the target";
    }
  }

  // Inherited from PostprocNode
  bool Apply(const tir::Schedule& sch) final;

  Postproc Clone() const {
    ObjectPtr<RewriteCooperativeFetchNode> n = make_object<RewriteCooperativeFetchNode>(*this);
    return Postproc(n);
  }

  void VisitAttrs(tvm::AttrVisitor* v) {}

  static constexpr const char* _type_key = "meta_schedule.RewriteCooperativeFetch";
  TVM_DECLARE_FINAL_OBJECT_INFO(RewriteCooperativeFetchNode, PostprocNode);

 private:
  int thread_warp_size_ = -1;
};

bool RewriteCooperativeFetchNode::Apply(const tir::Schedule& sch) {
  tir::Trace trace = sch->trace().value();
  int64_t thread_extent_x = -1;
  int64_t thread_extent_y = -1;
  int64_t vector_lane = 1;
  std::vector<std::function<void()>> tasks;
  for (const tir::Instruction& inst : trace->insts) {
    if (Optional<Integer> new_thread_extent = tir::ParseThreadBinding(sch, inst, "threadIdx.x")) {
      thread_extent_x = new_thread_extent.value()->value;
      continue;
    }
    if (Optional<Integer> new_thread_extent = tir::ParseThreadBinding(sch, inst, "threadIdx.y")) {
      thread_extent_y = new_thread_extent.value()->value;
      continue;
    }
    if (tir::ParseWarpExecutionAnn(sch, inst)) {
      thread_extent_x = thread_warp_size_;
      continue;
    }
    Optional<tir::BlockRV> opt_block_rv = tir::ParseAnnotate(sch, inst, &vector_lane);
    if (!opt_block_rv.defined()) {
      continue;
    }
    auto task = [thread_extent_x, thread_extent_y, vector_lane, sch,
                 block = opt_block_rv.value()]() mutable -> void {
      sch->Unannotate(block, tir::attr::meta_schedule_cooperative_fetch);
      tir::LoopRV fused = sch->GetLoops(block).back();
      int64_t fused_extent = -1;
      if (const int64_t* extent = tir::GetLoopIntExtent(sch->Get(fused).get())) {
        fused_extent = *extent;
      } else {
        return;
      }
      if (fused_extent % vector_lane != 0) {
        vector_lane = 1;
      }
      if (thread_extent_y != -1) {
        if (vector_lane > 1) {
          Array<tir::LoopRV> split = sch->Split(fused, {NullOpt,                   //
                                                        Integer(thread_extent_y),  //
                                                        Integer(thread_extent_x),  //
                                                        Integer(vector_lane)});
          sch->Vectorize(split[3]);
          sch->Bind(split[2], "threadIdx.x");
          sch->Bind(split[1], "threadIdx.y");
        } else {
          Array<tir::LoopRV> split = sch->Split(fused, {NullOpt,                   //
                                                        Integer(thread_extent_y),  //
                                                        Integer(thread_extent_x)});
          sch->Bind(split[2], "threadIdx.x");
          sch->Bind(split[1], "threadIdx.y");
        }
      } else {
        if (vector_lane > 1) {
          Array<tir::LoopRV> split = sch->Split(fused, {NullOpt,                   //
                                                        Integer(thread_extent_x),  //
                                                        Integer(vector_lane)});
          sch->Vectorize(split[2]);
          sch->Bind(split[1], "threadIdx.x");
        } else {
          Array<tir::LoopRV> split = sch->Split(fused, {NullOpt, Integer(thread_extent_x)});
          sch->Bind(split[1], "threadIdx.x");
        }
      }
    };
    tasks.push_back(task);
  }
  for (auto&& task : tasks) {
    task();
  }
  return true;
}

Postproc Postproc::RewriteCooperativeFetch() {
  ObjectPtr<RewriteCooperativeFetchNode> n = make_object<RewriteCooperativeFetchNode>();
  return Postproc(n);
}

TVM_REGISTER_NODE_TYPE(RewriteCooperativeFetchNode);
TVM_REGISTER_GLOBAL("meta_schedule.PostprocRewriteCooperativeFetch")
    .set_body_typed(Postproc::RewriteCooperativeFetch);

}  // namespace meta_schedule
}  // namespace tvm
