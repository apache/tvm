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
#include <tvm/ffi/reflection/registry.h>
#include <tvm/s_tir/stmt.h>

#include "../utils.h"

namespace tvm {
namespace s_tir {
using namespace tvm::tir;

/*!
 * \brief Parse instruction: sch.bind(..., axis)
 * \param sch The schedule
 * \param inst The instruction to be parsed
 * \param axis The axis name expected
 * \return std::nullopt if parsing fails; Otherwise, the extent of thread axis
 */
ffi::Optional<Integer> ParseThreadBinding(const Schedule& sch, const Instruction& inst,
                                          ffi::String axis) {
  static InstructionKind inst_kind_bind = InstructionKind::Get("Bind");
  if (!inst->kind.same_as(inst_kind_bind)) {
    return std::nullopt;
  }
  TVM_FFI_ICHECK_EQ(inst->inputs.size(), 1);
  TVM_FFI_ICHECK_EQ(inst->attrs.size(), 1);
  ffi::String thread_axis = Downcast<ffi::String>(inst->attrs[0]);
  if (thread_axis != axis) {
    return std::nullopt;
  }
  return Downcast<Integer>(sch->Get(Downcast<LoopRV>(inst->inputs[0]))->extent);
}

/*!
 * \brief Parse instruction: sch.annotate(..., attr::meta_schedule_cooperative_fetch)
 * \param sch The schedule
 * \param inst The instruction to be parsed
 * \param vector_lane The number of vector lane in vectorized cooperative fetching
 * \return std::nullopt if parsing fails; Otherwise, the annotated block
 */
ffi::Optional<SBlockRV> ParseAnnotate(const Schedule& sch, const Instruction& inst,
                                      int64_t* vector_lane) {
  static InstructionKind inst_kind_annotate = InstructionKind::Get("Annotate");
  if (!inst->kind.same_as(inst_kind_annotate)) {
    return std::nullopt;
  }
  TVM_FFI_ICHECK_EQ(inst->inputs.size(), 2);
  TVM_FFI_ICHECK_EQ(inst->attrs.size(), 1);
  ffi::String ann_key = Downcast<ffi::String>(inst->attrs[0]);
  if (ann_key != s_tir::attr::meta_schedule_cooperative_fetch) {
    return std::nullopt;
  }
  *vector_lane = Downcast<Integer>(sch->Get(Downcast<ExprRV>(inst->inputs[1])))->value;
  return Downcast<SBlockRV>(inst->inputs[0]);
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
  TVM_FFI_ICHECK_EQ(inst->inputs.size(), 2);
  TVM_FFI_ICHECK_EQ(inst->attrs.size(), 1);
  ffi::String ann_key = Downcast<ffi::String>(inst->attrs[0]);
  return ann_key == s_tir::attr::warp_execution;
}

size_t GetMaxUsedDtypeBytes(SBlock block) {
  size_t max_bytes = 1;
  static auto q_multiply_shift_per_axis = Op::Get("tir.q_multiply_shift_per_axis");
  static auto q_multiply_shift = Op::Get("tir.q_multiply_shift");

  tir::PostOrderVisit(block->body, [&](const ObjectRef& obj) {
    if (const auto* store = obj.as<tir::BufferStoreNode>()) {
      max_bytes = std::max(max_bytes, static_cast<size_t>(store->value->dtype.bytes()));
    } else if (const auto* load = obj.as<tir::BufferLoadNode>()) {
      max_bytes = std::max(max_bytes, static_cast<size_t>(load->dtype.bytes()));
    } else if (const auto* call = obj.as<tir::CallNode>()) {
      if (call->op.same_as(q_multiply_shift_per_axis) || call->op.same_as(q_multiply_shift)) {
        // q_multiply_shift uses 64 bit multiply
        max_bytes = std::max<size_t>(max_bytes, 8);
      }
    } else if (const auto* cast = obj.as<tir::CastNode>()) {
      max_bytes = std::max<size_t>(max_bytes, cast->dtype.bytes());
    }
  });

  return max_bytes;
}

}  // namespace s_tir

namespace s_tir {
namespace meta_schedule {

/*!
 * \brief Rewrite the cooperative fetch annotation to actual vectorized cooperative fetching
 * in loop bindings.
 */
class RewriteCooperativeFetchNode : public PostprocNode {
 public:
  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<RewriteCooperativeFetchNode>();
  }

  // Inherited from PostprocNode
  void InitializeWithTuneContext(const TuneContext& context) final {
    if (ffi::Optional<Integer> v = context->target.value()->GetAttr<Integer>("thread_warp_size")) {
      this->thread_warp_size_ = v.value()->value;
    } else {
      TVM_PY_LOG(INFO, context->logger) << "'thread_warp_size' is not defined in the target";
    }
  }

  // Inherited from PostprocNode
  bool Apply(const s_tir::Schedule& sch) final;

  Postproc Clone() const {
    ObjectPtr<RewriteCooperativeFetchNode> n = ffi::make_object<RewriteCooperativeFetchNode>(*this);
    return Postproc(n);
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("s_tir.meta_schedule.RewriteCooperativeFetch",
                                    RewriteCooperativeFetchNode, PostprocNode);

 private:
  int thread_warp_size_ = -1;
};

bool RewriteCooperativeFetchNode::Apply(const s_tir::Schedule& sch) {
  s_tir::Trace trace = sch->trace().value();
  int64_t thread_extent_x = -1;
  int64_t thread_extent_y = -1;
  int64_t vector_lane = 1;
  std::vector<std::function<void()>> tasks;
  for (const s_tir::Instruction& inst : trace->insts) {
    if (ffi::Optional<Integer> new_thread_extent =
            s_tir::ParseThreadBinding(sch, inst, "threadIdx.x")) {
      thread_extent_x = new_thread_extent.value()->value;
      continue;
    }
    if (ffi::Optional<Integer> new_thread_extent =
            s_tir::ParseThreadBinding(sch, inst, "threadIdx.y")) {
      thread_extent_y = new_thread_extent.value()->value;
      continue;
    }
    if (s_tir::ParseWarpExecutionAnn(sch, inst)) {
      thread_extent_x = thread_warp_size_;
      continue;
    }
    ffi::Optional<s_tir::SBlockRV> opt_block_rv = s_tir::ParseAnnotate(sch, inst, &vector_lane);
    if (!opt_block_rv.defined()) {
      continue;
    }
    auto task = [thread_extent_x, thread_extent_y, vector_lane, sch,
                 block = opt_block_rv.value()]() mutable -> void {
      sch->Unannotate(block, s_tir::attr::meta_schedule_cooperative_fetch);
      s_tir::LoopRV fused = sch->GetLoops(block).back();
      int64_t fused_extent = -1;
      if (const int64_t* extent = s_tir::GetLoopIntExtent(sch->Get(fused).get())) {
        fused_extent = *extent;
      } else {
        return;
      }
      if (fused_extent % vector_lane != 0) {
        vector_lane = 1;
      }
      // If the block involves 64 bit values, disable vectorization for now since
      // vectorization of 64 bit values does not work well on CUDA.
      // TODO(masahi, vinx13): Decouple epilogue fusion computation and shared to global store, so
      // that we can always vectorize the latter.
      if (s_tir::GetMaxUsedDtypeBytes(sch->Get(block)) > 4) {
        vector_lane = 1;
      }
      if (thread_extent_y != -1) {
        if (vector_lane > 1) {
          ffi::Array<s_tir::LoopRV> split = sch->Split(fused, {std::nullopt,              //
                                                               Integer(thread_extent_y),  //
                                                               Integer(thread_extent_x),  //
                                                               Integer(vector_lane)});
          sch->Vectorize(split[3]);
          sch->Bind(split[2], "threadIdx.x");
          sch->Bind(split[1], "threadIdx.y");
        } else {
          ffi::Array<s_tir::LoopRV> split = sch->Split(fused, {std::nullopt,              //
                                                               Integer(thread_extent_y),  //
                                                               Integer(thread_extent_x)});
          sch->Bind(split[2], "threadIdx.x");
          sch->Bind(split[1], "threadIdx.y");
        }
      } else {
        if (vector_lane > 1) {
          ffi::Array<s_tir::LoopRV> split = sch->Split(fused, {std::nullopt,              //
                                                               Integer(thread_extent_x),  //
                                                               Integer(vector_lane)});
          sch->Vectorize(split[2]);
          sch->Bind(split[1], "threadIdx.x");
        } else {
          ffi::Array<s_tir::LoopRV> split =
              sch->Split(fused, {std::nullopt, Integer(thread_extent_x)});
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
  ObjectPtr<RewriteCooperativeFetchNode> n = ffi::make_object<RewriteCooperativeFetchNode>();
  return Postproc(n);
}

TVM_FFI_STATIC_INIT_BLOCK() { RewriteCooperativeFetchNode::RegisterReflection(); }

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("s_tir.meta_schedule.PostprocRewriteCooperativeFetch",
                        Postproc::RewriteCooperativeFetch);
}

}  // namespace meta_schedule
}  // namespace s_tir
}  // namespace tvm
