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

void PyScheduleRuleNode::InitializeWithTuneContext(const TuneContext& context) {
  ICHECK(f_initialize_with_tune_context != nullptr)
      << "PyScheduleRule's InitializeWithTuneContext method not implemented!";
  f_initialize_with_tune_context(context);
}

Array<tir::Schedule> PyScheduleRuleNode::Apply(const tir::Schedule& sch,
                                               const tir::BlockRV& block) {
  ICHECK(f_apply != nullptr) << "PyScheduleRule's Apply method not implemented!";
  return f_apply(sch, block);
}

ScheduleRule PyScheduleRuleNode::Clone() const {
  ICHECK(f_clone != nullptr) << "PyScheduleRule's Clone method not implemented!";
  return f_clone();
}

ScheduleRule ScheduleRule::PyScheduleRule(
    PyScheduleRuleNode::FInitializeWithTuneContext f_initialize_with_tune_context,  //
    PyScheduleRuleNode::FApply f_apply,                                             //
    PyScheduleRuleNode::FClone f_clone,                                             //
    PyScheduleRuleNode::FAsString f_as_string) {
  ObjectPtr<PyScheduleRuleNode> n = make_object<PyScheduleRuleNode>();
  n->f_initialize_with_tune_context = std::move(f_initialize_with_tune_context);
  n->f_apply = std::move(f_apply);
  n->f_clone = std::move(f_clone);
  n->f_as_string = std::move(f_as_string);
  return ScheduleRule(n);
}

Array<ScheduleRule> ScheduleRule::DefaultLLVM() {
  return {
      ScheduleRule::ApplyCustomRule(),
      ScheduleRule::InlineConstantScalars(),
      ScheduleRule::AutoInline(
          /*into_producer=*/false,
          /*into_consumer=*/true,
          /*inline_const_tensor=*/true,
          /*disallow_if_then_else=*/true,
          /*require_injective=*/true,
          /*require_ordered=*/true,
          /*disallow_op=*/Array<String>{"tir.exp"}),
      ScheduleRule::AddRFactor(
          /*max_jobs_per_core=*/16,
          /*max_innermost_factor=*/Integer(64)),
      ScheduleRule::MultiLevelTiling(
          /*structure=*/"SSRSRS",
          /*tile_binds=*/NullOpt,
          /*max_innermost_factor=*/Integer(64),
          /*vector_load_lens=*/NullOpt,
          /*reuse_read=*/NullOpt,
          /*reuse_write=*/
          Map<String, ObjectRef>{{"req", String("may")},
                                 {"levels", Array<Integer>{1, 2}},
                                 {"scope", String("global")}}),
      ScheduleRule::ParallelizeVectorizeUnroll(
          /*max_jobs_per_core=*/16,
          /*max_vectorize_extent=*/64,
          /*unroll_max_steps=*/Array<Integer>{0, 16, 64, 512},
          /*unroll_explicit=*/true),
      ScheduleRule::RandomComputeLocation(),
  };
}

Array<ScheduleRule> ScheduleRule::DefaultVNNI() {
  return {
      ScheduleRule::ApplyCustomRule(),
      ScheduleRule::InlineConstantScalars(),
      ScheduleRule::AutoInline(
          /*into_producer=*/false,
          /*into_consumer=*/true,
          /*inline_const_tensor=*/true,
          /*disallow_if_then_else=*/true,
          /*require_injective=*/true,
          /*require_ordered=*/true,
          /*disallow_op=*/Array<String>{"tir.exp"}),
      ScheduleRule::AddRFactor(
          /*max_jobs_per_core=*/16,
          /*max_innermost_factor=*/Integer(64)),
      ScheduleRule::MultiLevelTilingWithIntrin(
          /*intrin_name=*/"dot_16x4_vnni",
          /*structure=*/"SSRSRS",
          /*tile_binds=*/NullOpt,
          /*max_innermost_factor=*/Integer(64),
          /*vector_load_lens=*/NullOpt,
          /*reuse_read=*/NullOpt,
          /*reuse_write=*/
          Map<String, ObjectRef>{{"req", String("may")},
                                 {"levels", Array<Integer>{1, 2}},
                                 {"scope", String("global")}}),
      ScheduleRule::MultiLevelTiling(
          /*structure=*/"SSRSRS",
          /*tile_binds=*/NullOpt,
          /*max_innermost_factor=*/Integer(64),
          /*vector_load_lens=*/NullOpt,
          /*reuse_read=*/NullOpt,
          /*reuse_write=*/
          Map<String, ObjectRef>{{"req", String("may")},
                                 {"levels", Array<Integer>{1, 2}},
                                 {"scope", String("global")}}),
      ScheduleRule::ParallelizeVectorizeUnroll(
          /*max_jobs_per_core=*/16,
          /*max_vectorize_extent=*/64,
          /*unroll_max_steps=*/Array<Integer>{0, 16, 64, 512},
          /*unroll_explicit=*/true),
      ScheduleRule::RandomComputeLocation(),
  };
}

Array<ScheduleRule> ScheduleRule::DefaultCUDA() {
  return {
      ScheduleRule::ApplyCustomRule(),
      ScheduleRule::MultiLevelTiling(
          /*structure=*/"SSSRRSRS",
          /*tile_binds=*/Array<String>{"blockIdx.x", "vthread.x", "threadIdx.x"},
          /*max_innermost_factor=*/Integer(64),
          /*vector_load_lens=*/Array<Integer>{1, 2, 3, 4, 8, 16},
          /*reuse_read=*/
          Map<String, ObjectRef>{{"req", String("must")},
                                 {"levels", Array<Integer>{4}},  //
                                 {"scope", String("shared")}},
          /*reuse_write=*/
          Map<String, ObjectRef>{{"req", String("must")},
                                 {"levels", Array<Integer>{3}},  //
                                 {"scope", String("local")}}),
      ScheduleRule::InlineConstantScalars(),
      ScheduleRule::AutoInline(
          /*into_producer=*/true,
          /*into_consumer=*/true,
          /*inline_const_tensor=*/true,
          /*disallow_if_then_else=*/false,
          /*require_injective=*/false,
          /*require_ordered=*/false,
          /*disallow_op=*/Array<String>{}),
      ScheduleRule::CrossThreadReduction(
          /*thread_extents=*/Array<Integer>{4, 8, 16, 32, 64, 128, 256, 512}),
      ScheduleRule::ParallelizeVectorizeUnroll(
          /*max_jobs_per_core=*/-1,
          /*max_vectorize_extent=*/-1,
          /*unroll_max_steps=*/Array<Integer>{0, 16, 64, 512, 1024},
          /*unroll_explicit=*/true),
      ScheduleRule::AutoBind(
          /*max_threadblocks=*/256,
          /*thread_extents*/ Array<Integer>{32, 64, 128, 256, 512, 1024}),
  };
}

Array<ScheduleRule> ScheduleRule::DefaultCUDATensorCore() {
  Array<Map<String, String>> intrin_groups = {
      {
          {"init", "wmma_fill_16x16x16_f16"},
          {"load_a", "wmma_load_16x16x16_f16_a"},
          {"load_b", "wmma_load_16x16x16_f16_b"},
          {"compute", "wmma_sync_16x16x16_f16f16f16"},
          {"store", "wmma_store_16x16x16_f16_shared"},
      },
      {
          {"init", "wmma_fill_16x16x16_f16"},
          {"load_a", "wmma_load_16x16x16_f16_a"},
          {"load_b", "wmma_load_16x16x16_f16_b_trans"},
          {"compute", "wmma_sync_16x16x16_f16f16f16_trans"},
          {"store", "wmma_store_16x16x16_f16_shared"},
      },
      {
          {"init", "wmma_fill_16x16x16_s32"},
          {"load_a", "wmma_load_16x16x16_s8_a"},
          {"load_b", "wmma_load_16x16x16_s8_b"},
          {"compute", "wmma_sync_16x16x16_s8s8s32"},
          {"store", "wmma_store_16x16x16_s32_shared"},
      },
      {
          {"init", "wmma_fill_16x16x16_s32"},
          {"load_a", "wmma_load_16x16x16_s8_a"},
          {"load_b", "wmma_load_16x16x16_s8_b_trans"},
          {"compute", "wmma_sync_16x16x16_s8s8s32_trans"},
          {"store", "wmma_store_16x16x16_s32_shared"},
      },
  };
  Array<ScheduleRule> results{
      ScheduleRule::ApplyCustomRule(),
      ScheduleRule::MultiLevelTilingTensorCore(
          /*intrin_groups=*/intrin_groups,
          /*structure=*/"SSSRRSRS",
          /*tile_binds=*/Array<String>{"blockIdx.y", "blockIdx.x", "threadIdx.y"},
          /*max_innermost_factor=*/Integer(4),
          /*vector_load_lens=*/Array<Integer>{1, 2, 3, 4, 8, 16},
          /*reuse_read=*/
          Map<String, ObjectRef>{{"req", String("must")},
                                 {"levels", Array<Integer>{4}},  //
                                 {"scope", String("shared")}},
          /*reuse_write=*/
          Map<String, ObjectRef>{{"req", String("must")},
                                 {"levels", Array<Integer>{2}},  //
                                 {"scope", String("shared")}},
          /*use_software_pipeline=*/false)  //
  };
  Array<ScheduleRule> append = ScheduleRule::DefaultCUDA();
  results.insert(results.end(), append.begin() + 1, append.end());
  return results;
}

Array<ScheduleRule> ScheduleRule::DefaultHexagon() {
  return {
      ScheduleRule::ApplyCustomRule(),
      ScheduleRule::InlineConstantScalars(),
      ScheduleRule::AutoInline(
          /*into_producer=*/false,
          /*into_consumer=*/true,
          /*inline_const_tensor=*/true,
          /*disallow_if_then_else=*/true,
          /*require_injective=*/true,
          /*require_ordered=*/true,
          /*disallow_op=*/Array<String>{"tir.exp"}),
      ScheduleRule::MultiLevelTilingWideVector(
          /*structure=*/"SRSRS",
          /*vector_length_in_bits=*/1024,
          /*max_innermost_factor=*/Integer(128),
          /*reuse_read=*/NullOpt,
          /*reuse_write=*/
          Map<String, ObjectRef>{{"req", String("may")},
                                 {"levels", Array<Integer>{1, 2}},
                                 {"scope", String("global")}}),
      ScheduleRule::ParallelizeVectorizeUnroll(
          /*max_jobs_per_core=*/16,
          /*max_vectorize_extent=*/128,
          /*unroll_max_steps=*/Array<Integer>{0, 16, 64, 512},
          /*unroll_explicit=*/true),
  };
}

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<PyScheduleRuleNode>([](const ObjectRef& n, ReprPrinter* p) {
      const auto* self = n.as<PyScheduleRuleNode>();
      ICHECK(self);
      PyScheduleRuleNode::FAsString f_as_string = (*self).f_as_string;
      ICHECK(f_as_string != nullptr) << "PyScheduleRule's AsString method not implemented!";
      p->stream << f_as_string();
    });

TVM_REGISTER_OBJECT_TYPE(ScheduleRuleNode);
TVM_REGISTER_NODE_TYPE(PyScheduleRuleNode);

TVM_REGISTER_GLOBAL("meta_schedule.ScheduleRuleInitializeWithTuneContext")
    .set_body_method<ScheduleRule>(&ScheduleRuleNode::InitializeWithTuneContext);
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleRuleApply")
    .set_body_method<ScheduleRule>(&ScheduleRuleNode::Apply);
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleRuleClone")
    .set_body_method<ScheduleRule>(&ScheduleRuleNode::Clone);
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleRulePyScheduleRule")
    .set_body_typed(ScheduleRule::PyScheduleRule);
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleRuleDefaultLLVM")
    .set_body_typed(ScheduleRule::DefaultLLVM);
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleRuleDefaultCUDA")
    .set_body_typed(ScheduleRule::DefaultCUDA);
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleRuleDefaultCUDATensorCore")
    .set_body_typed(ScheduleRule::DefaultCUDATensorCore);
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleRuleDefaultHexagon")
    .set_body_typed(ScheduleRule::DefaultHexagon);

}  // namespace meta_schedule
}  // namespace tvm
