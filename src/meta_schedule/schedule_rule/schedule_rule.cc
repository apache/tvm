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
#include <tvm/runtime/data_type.h>

#include "../utils.h"

namespace tvm {
namespace meta_schedule {

void PyScheduleRuleNode::InitializeWithTuneContext(const TuneContext& context) {
  ICHECK(f_initialize_with_tune_context != nullptr)
      << "PyScheduleRule's InitializeWithTuneContext method not implemented!";
  f_initialize_with_tune_context(context);
}

ffi::Array<tir::Schedule> PyScheduleRuleNode::Apply(const tir::Schedule& sch,
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
  ObjectPtr<PyScheduleRuleNode> n = ffi::make_object<PyScheduleRuleNode>();
  n->f_initialize_with_tune_context = std::move(f_initialize_with_tune_context);
  n->f_apply = std::move(f_apply);
  n->f_clone = std::move(f_clone);
  n->f_as_string = std::move(f_as_string);
  return ScheduleRule(n);
}

ffi::Array<ScheduleRule> ScheduleRule::DefaultLLVM() {
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
          /*disallow_op=*/ffi::Array<ffi::String>{"tir.exp"}),
      ScheduleRule::AddRFactor(
          /*max_jobs_per_core=*/16,
          /*max_innermost_factor=*/Integer(64)),
      ScheduleRule::MultiLevelTiling(
          /*structure=*/"SSRSRS",
          /*tile_binds=*/std::nullopt,
          /*max_innermost_factor=*/Integer(64),
          /*vector_load_lens=*/std::nullopt,
          /*reuse_read=*/std::nullopt,
          /*reuse_write=*/
          ffi::Map<ffi::String, ffi::Any>{{"req", ffi::String("may")},
                                          {"levels", ffi::Array<Integer>{1, 2}},
                                          {"scope", ffi::String("global")}}),
      ScheduleRule::ParallelizeVectorizeUnroll(
          /*max_jobs_per_core=*/16,
          /*max_vectorize_extent=*/64,
          /*unroll_max_steps=*/ffi::Array<Integer>{0, 16, 64, 512},
          /*unroll_explicit=*/true),
      ScheduleRule::RandomComputeLocation(),
  };
}

ffi::Array<ScheduleRule> ScheduleRule::DefaultX86(const ffi::String& type) {
  static const ffi::Map<ffi::String, ffi::String> intrins = {{"vnni", "dot_16x4_vnni"},
                                                             {"avx512", "dot_16x4_avx512"}};
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
          /*disallow_op=*/ffi::Array<ffi::String>{"tir.exp"}),
      ScheduleRule::AddRFactor(
          /*max_jobs_per_core=*/16,
          /*max_innermost_factor=*/Integer(64)),
      ScheduleRule::MultiLevelTilingWithIntrin(
          /*intrin_name=*/intrins[type],
          /*structure=*/"SSRSRS",
          /*tile_binds=*/std::nullopt,
          /*max_innermost_factor=*/Integer(64),
          /*vector_load_lens=*/std::nullopt,
          /*reuse_read=*/std::nullopt,
          /*reuse_write=*/
          ffi::Map<ffi::String, ffi::Any>{{"req", ffi::String("may")},
                                          {"levels", ffi::Array<Integer>{1, 2}},
                                          {"scope", ffi::String("global")}}),
      ScheduleRule::MultiLevelTiling(
          /*structure=*/"SSRSRS",
          /*tile_binds=*/std::nullopt,
          /*max_innermost_factor=*/Integer(64),
          /*vector_load_lens=*/std::nullopt,
          /*reuse_read=*/std::nullopt,
          /*reuse_write=*/
          ffi::Map<ffi::String, ffi::Any>{{"req", ffi::String("may")},
                                          {"levels", ffi::Array<Integer>{1, 2}},
                                          {"scope", ffi::String("global")}}),
      ScheduleRule::ParallelizeVectorizeUnroll(
          /*max_jobs_per_core=*/16,
          /*max_vectorize_extent=*/64,
          /*unroll_max_steps=*/ffi::Array<Integer>{0, 16, 64, 512},
          /*unroll_explicit=*/true),
      ScheduleRule::RandomComputeLocation(),
  };
}

ffi::Array<ScheduleRule> ScheduleRule::DefaultCUDA() {
  return {
      ScheduleRule::ApplyCustomRule(),
      ScheduleRule::MultiLevelTiling(
          /*structure=*/"SSSRRSRS",
          /*tile_binds=*/ffi::Array<ffi::String>{"blockIdx.x", "vthread.x", "threadIdx.x"},
          /*max_innermost_factor=*/Integer(64),
          /*vector_load_lens=*/ffi::Array<Integer>{1, 2, 3, 4, 8, 16},
          /*reuse_read=*/
          ffi::Map<ffi::String, ffi::Any>{{"req", ffi::String("must")},
                                          {"levels", ffi::Array<Integer>{4}},  //
                                          {"scope", ffi::String("shared")}},
          /*reuse_write=*/
          ffi::Map<ffi::String, ffi::Any>{{"req", ffi::String("must")},
                                          {"levels", ffi::Array<Integer>{3}},  //
                                          {"scope", ffi::String("local")}}),
      ScheduleRule::InlineConstantScalars(),
      ScheduleRule::AutoInline(
          /*into_producer=*/true,
          /*into_consumer=*/true,
          /*inline_const_tensor=*/true,
          /*disallow_if_then_else=*/false,
          /*require_injective=*/false,
          /*require_ordered=*/false,
          /*disallow_op=*/ffi::Array<ffi::String>{}),
      ScheduleRule::CrossThreadReduction(
          /*thread_extents=*/ffi::Array<Integer>{4, 8, 16, 32, 64, 128, 256, 512}),
      ScheduleRule::ParallelizeVectorizeUnroll(
          /*max_jobs_per_core=*/-1,
          /*max_vectorize_extent=*/-1,
          /*unroll_max_steps=*/ffi::Array<Integer>{0, 16, 64, 512, 1024},
          /*unroll_explicit=*/true),
      ScheduleRule::AutoBind(
          /*max_threadblocks=*/256,
          /*thread_extents*/ ffi::Array<Integer>{32, 64, 128, 256, 512, 1024}),
  };
}

ffi::Array<ScheduleRule> ScheduleRule::DefaultCUDATensorCore() {
  ffi::Array<ffi::Map<ffi::String, ffi::String>> wmma_intrin_groups = {
      // Tensor Cores f32 += f16 * f16
      {
          {"init", "wmma_fill_16x16x16_f32"},
          {"load_a", "wmma_load_16x16x16_f16_a_shared_dyn"},
          {"load_b", "wmma_load_16x16x16_f16_b_shared_dyn"},
          {"compute", "wmma_sync_16x16x16_f16f16f32"},
          {"store", "wmma_store_16x16x16_f32_shared_dyn"},
      },
      {
          {"init", "wmma_fill_16x16x16_f32"},
          {"load_a", "wmma_load_16x16x16_f16_a_shared_dyn"},
          {"load_b", "wmma_load_16x16x16_f16_b_trans_shared_dyn"},
          {"compute", "wmma_sync_16x16x16_f16f16f32_trans"},
          {"store", "wmma_store_16x16x16_f32_shared_dyn"},
      },
      // Tensor Cores f16 += f16 * f16
      {
          {"init", "wmma_fill_16x16x16_f16"},
          {"load_a", "wmma_load_16x16x16_f16_a_shared_dyn"},
          {"load_b", "wmma_load_16x16x16_f16_b_shared_dyn"},
          {"compute", "wmma_sync_16x16x16_f16f16f16"},
          {"store", "wmma_store_16x16x16_f16_shared_dyn"},
      },
      {
          {"init", "wmma_fill_16x16x16_f16"},
          {"load_a", "wmma_load_16x16x16_f16_a_shared_dyn"},
          {"load_b", "wmma_load_16x16x16_f16_b_trans_shared_dyn"},
          {"compute", "wmma_sync_16x16x16_f16f16f16_trans"},
          {"store", "wmma_store_16x16x16_f16_shared_dyn"},
      },
      // Tensor Cores s32 += s8 * s8
      {
          {"init", "wmma_fill_16x16x16_s32"},
          {"load_a", "wmma_load_16x16x16_s8_a_shared_dyn"},
          {"load_b", "wmma_load_16x16x16_s8_b_shared_dyn"},
          {"compute", "wmma_sync_16x16x16_s8s8s32"},
          {"store", "wmma_store_16x16x16_s32_shared_dyn"},
      },
      {
          {"init", "wmma_fill_16x16x16_s32"},
          {"load_a", "wmma_load_16x16x16_s8_a_shared_dyn"},
          {"load_b", "wmma_load_16x16x16_s8_b_trans_shared_dyn"},
          {"compute", "wmma_sync_16x16x16_s8s8s32_trans"},
          {"store", "wmma_store_16x16x16_s32_shared_dyn"},
      },
  };
  ffi::Array<ffi::Map<ffi::String, ffi::String>> mma_intrin_groups = {
      // Tensor Core MMA
      {
          {"init", "mma_init_m16n8k8_f16"},
          {"load_a", "mma_load_m16n8k8_f16_A_shared_dyn"},
          {"load_b", "mma_load_m16n8k8_f16_B_shared_dyn"},
          {"compute", "mma_sync_m16n8k8_f16f16f16"},
          {"store", "mma_store_m16n8k8_f16_global"},
      },
      {
          {"init", "mma_init_m16n8k8_f32"},
          {"load_a", "mma_load_m16n8k8_f16_A_shared_dyn"},
          {"load_b", "mma_load_m16n8k8_f16_B_shared_dyn"},
          {"compute", "mma_sync_m16n8k8_f16f16f32"},
          {"store", "mma_store_m16n8k8_f32_global"},
      },
  };
  ffi::Array<ScheduleRule> results{
      ScheduleRule::ApplyCustomRule(),
      ScheduleRule::MultiLevelTilingTensorCore(
          /*intrin_groups=*/wmma_intrin_groups,
          /*structure=*/"SSSRRSRS",
          /*tile_binds=*/ffi::Array<ffi::String>{"blockIdx.y", "blockIdx.x", "threadIdx.y"},
          /*max_innermost_factor=*/Integer(4),
          /*vector_load_lens=*/ffi::Array<Integer>{1, 2, 3, 4, 8, 16},
          /*reuse_read=*/
          ffi::Map<ffi::String, ffi::Any>{{"req", ffi::String("must")},
                                          {"levels", ffi::Array<Integer>{4}},  //
                                          {"scope", ffi::String("shared.dyn")}},
          /*reuse_write=*/
          ffi::Map<ffi::String, ffi::Any>{{"req", ffi::String("must")},
                                          {"levels", ffi::Array<Integer>{2}},  //
                                          {"scope", ffi::String("shared.dyn")}},
          /*use_software_pipeline=*/false),  //
      ScheduleRule::MultiLevelTilingTensorCore(
          /*intrin_groups=*/mma_intrin_groups,
          /*structure=*/"SSSRRSRS",
          /*tile_binds=*/ffi::Array<ffi::String>{"blockIdx.y", "blockIdx.x", "threadIdx.y"},
          /*max_innermost_factor=*/Integer(4),
          /*vector_load_lens=*/ffi::Array<Integer>{1, 2, 3, 4, 8, 16},
          /*reuse_read=*/
          ffi::Map<ffi::String, ffi::Any>{{"req", ffi::String("must")},
                                          {"levels", ffi::Array<Integer>{4}},  //
                                          {"scope", ffi::String("shared.dyn")}},
          /*reuse_write=*/
          ffi::Map<ffi::String, ffi::Any>{{"req", ffi::String("no")},
                                          {"levels", ffi::Array<Integer>{2}},  //
                                          {"scope", ffi::String("shared.dyn")}},
          /*use_software_pipeline=*/true)  //
  };
  ffi::Array<ScheduleRule> append = ScheduleRule::DefaultCUDA();
  results.insert(results.end(), append.begin() + 1, append.end());
  return results;
}

ffi::Array<ScheduleRule> ScheduleRule::DefaultHexagon() {
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
          /*disallow_op=*/ffi::Array<ffi::String>{"tir.exp"}),
      ScheduleRule::MultiLevelTilingWideVector(
          /*structure=*/"SRSRS",
          /*vector_length_in_bits=*/1024,
          /*max_innermost_factor=*/Integer(128),
          /*reuse_read=*/std::nullopt,
          /*reuse_write=*/
          ffi::Map<ffi::String, ffi::Any>{{"req", ffi::String("may")},
                                          {"levels", ffi::Array<Integer>{1, 2}},
                                          {"scope", ffi::String("global")}}),
      ScheduleRule::ParallelizeVectorizeUnroll(
          /*max_jobs_per_core=*/16,
          /*max_vectorize_extent=*/128,
          /*unroll_max_steps=*/ffi::Array<Integer>{0, 16, 64, 512},
          /*unroll_explicit=*/true),
  };
}

ffi::Array<ScheduleRule> ScheduleRule::DefaultRISCV(const int vlen) {
  ffi::Array<ScheduleRule> rules;
  rules.push_back(ScheduleRule::ApplyCustomRule());
  rules.push_back(ScheduleRule::InlineConstantScalars());
  rules.push_back(ScheduleRule::AutoInline(
      /*into_producer=*/false,
      /*into_consumer=*/true,
      /*inline_const_tensor=*/true,
      /*disallow_if_then_else=*/true,
      /*require_injective=*/true,
      /*require_ordered=*/true,
      /*disallow_op=*/ffi::Array<ffi::String>{"tir.exp"}));
  rules.push_back(ScheduleRule::AddRFactor(
      /*max_jobs_per_core=*/16,
      /*max_innermost_factor=*/Integer(64)));
  auto current_target = tvm::Target::Current();
  const auto reg_rvv_intrinsics =
      tvm::ffi::Function::GetGlobalRequired("tir.tensor_intrin.register_rvv_isa_intrinsics");
  const auto rvv_kernels_inventory = reg_rvv_intrinsics(current_target, /* inventory_only */ true)
                                         .cast<ffi::Map<ffi::String, int>>();
  for (const auto& intrin : rvv_kernels_inventory) {
    if (!tir::TensorIntrin::Get(intrin.first, /*allow_missing*/ true)) {
      // on demand intrinsic register
      reg_rvv_intrinsics(current_target, /* inventory_only */ false);
    }
    rules.push_back(ScheduleRule::MultiLevelTilingWithIntrin(
        /*intrin_name=*/intrin.first,
        /*structure=*/"SSRSRS",
        /*tile_binds=*/std::nullopt,
        /*max_innermost_factor=*/Integer(intrin.second),
        /*vector_load_lens=*/std::nullopt,
        /*reuse_read=*/std::nullopt,
        /*reuse_write=*/
        ffi::Map<ffi::String, ffi::Any>{{"req", ffi::String("may")},
                                        {"levels", ffi::Array<Integer>{1, 2}},
                                        {"scope", ffi::String("global")}}));
  }
  rules.push_back(ScheduleRule::MultiLevelTiling(
      /*structure=*/"SSRSRS",
      /*tile_binds=*/std::nullopt,
      /*max_innermost_factor=*/Integer(64),
      /*vector_load_lens=*/std::nullopt,
      /*reuse_read=*/std::nullopt,
      /*reuse_write=*/
      ffi::Map<ffi::String, ffi::Any>{{"req", ffi::String("may")},
                                      {"levels", ffi::Array<Integer>{1, 2}},
                                      {"scope", ffi::String("global")}}));
  rules.push_back(ScheduleRule::ParallelizeVectorizeUnroll(
      /*max_jobs_per_core=*/16,
      /*max_vectorize_extent=*/64,
      /*unroll_max_steps=*/ffi::Array<Integer>{0, 16, 64, 512},
      /*unroll_explicit=*/true));
  rules.push_back(ScheduleRule::RandomComputeLocation());

  return rules;
}

ffi::Array<ScheduleRule> GetARMNeonSpecificRules() {
  return {
      ScheduleRule::MultiLevelTilingWithIntrin(
          /*intrin_name=*/ffi::String("dot_4x4_i8i8s32_neon"),
          /*structure=*/"SSRSRS",
          /*tile_binds=*/std::nullopt,
          /*max_innermost_factor=*/Integer(32),
          /*vector_load_lens=*/std::nullopt,
          /*reuse_read=*/std::nullopt,
          /*reuse_write=*/
          ffi::Map<ffi::String, ffi::Any>{{"req", ffi::String("may")},
                                          {"levels", ffi::Array<Integer>{1, 2}},
                                          {"scope", ffi::String("global")}}),
  };
}

ffi::Array<ScheduleRule> GetARMDotprodSpecificRules() {
  return {
      ScheduleRule::MultiLevelTilingWithIntrin(
          /*intrin_name=*/ffi::String("dot_4x4_i8i8s32_sdot"),
          /*structure=*/"SSRSRS",
          /*tile_binds=*/std::nullopt,
          /*max_innermost_factor=*/Integer(32),
          /*vector_load_lens=*/std::nullopt,
          /*reuse_read=*/std::nullopt,
          /*reuse_write=*/
          ffi::Map<ffi::String, ffi::Any>{{"req", ffi::String("may")},
                                          {"levels", ffi::Array<Integer>{1, 2}},
                                          {"scope", ffi::String("global")}}),
      ScheduleRule::MultiLevelTilingWithIntrin(
          /*intrin_name=*/ffi::String("dot_4x4_u8u8u32_udot"),
          /*structure=*/"SSRSRS",
          /*tile_binds=*/std::nullopt,
          /*max_innermost_factor=*/Integer(32),
          /*vector_load_lens=*/std::nullopt,
          /*reuse_read=*/std::nullopt,
          /*reuse_write=*/
          ffi::Map<ffi::String, ffi::Any>{{"req", ffi::String("may")},
                                          {"levels", ffi::Array<Integer>{1, 2}},
                                          {"scope", ffi::String("global")}}),
      ScheduleRule::MultiLevelTilingWithIntrin(
          /*intrin_name=*/ffi::String("dot_4x4_u8u8i32_hdot"),
          /*structure=*/"SSRSRS",
          /*tile_binds=*/std::nullopt,
          /*max_innermost_factor=*/Integer(32),
          /*vector_load_lens=*/std::nullopt,
          /*reuse_read=*/std::nullopt,
          /*reuse_write=*/
          ffi::Map<ffi::String, ffi::Any>{{"req", ffi::String("may")},
                                          {"levels", ffi::Array<Integer>{1, 2}},
                                          {"scope", ffi::String("global")}}),
  };
}

ffi::Array<ScheduleRule> ScheduleRule::DefaultARM(const ffi::String& type) {
  return ffi::Array<ScheduleRule>::Agregate(
      ScheduleRule::ApplyCustomRule(), ScheduleRule::InlineConstantScalars(),
      ScheduleRule::AutoInline(
          /*into_producer=*/false,
          /*into_consumer=*/true,
          /*inline_const_tensor=*/true,
          /*disallow_if_then_else=*/true,
          /*require_injective=*/true,
          /*require_ordered=*/true,
          /*disallow_op=*/ffi::Array<ffi::String>{"tir.exp"}),
      ScheduleRule::AddRFactor(
          /*max_jobs_per_core=*/8,
          /*max_innermost_factor=*/Integer(32)),
      "neon" == type ? GetARMNeonSpecificRules() : ffi::Array<ScheduleRule>{},
      "dotprod" == type ? GetARMDotprodSpecificRules() : ffi::Array<ScheduleRule>{},
      ScheduleRule::MultiLevelTiling(
          /*structure=*/"SSRSRS",
          /*tile_binds=*/std::nullopt,
          /*max_innermost_factor=*/Integer(32),
          /*vector_load_lens=*/std::nullopt,
          /*reuse_read=*/std::nullopt,
          /*reuse_write=*/
          ffi::Map<ffi::String, ffi::Any>{{"req", ffi::String("may")},
                                          {"levels", ffi::Array<Integer>{1, 2}},
                                          {"scope", ffi::String("global")}}),
      ScheduleRule::ParallelizeVectorizeUnroll(
          /*max_jobs_per_core=*/8,
          /*max_vectorize_extent=*/32,
          /*unroll_max_steps=*/ffi::Array<Integer>{0, 8, 32, 256},
          /*unroll_explicit=*/true),
      ScheduleRule::RandomComputeLocation());
}

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<PyScheduleRuleNode>([](const ObjectRef& n, ReprPrinter* p) {
      const auto* self = n.as<PyScheduleRuleNode>();
      ICHECK(self);
      PyScheduleRuleNode::FAsString f_as_string = (*self).f_as_string;
      ICHECK(f_as_string != nullptr) << "PyScheduleRule's AsString method not implemented!";
      p->stream << f_as_string();
    });

TVM_FFI_STATIC_INIT_BLOCK() {
  ScheduleRuleNode::RegisterReflection();
  PyScheduleRuleNode::RegisterReflection();
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def_method("meta_schedule.ScheduleRuleInitializeWithTuneContext",
                  &ScheduleRuleNode::InitializeWithTuneContext)
      .def_method("meta_schedule.ScheduleRuleApply", &ScheduleRuleNode::Apply)
      .def_method("meta_schedule.ScheduleRuleClone", &ScheduleRuleNode::Clone)
      .def("meta_schedule.ScheduleRulePyScheduleRule", ScheduleRule::PyScheduleRule)
      .def("meta_schedule.ScheduleRuleDefaultLLVM", ScheduleRule::DefaultLLVM)
      .def("meta_schedule.ScheduleRuleDefaultCUDA", ScheduleRule::DefaultCUDA)
      .def("meta_schedule.ScheduleRuleDefaultCUDATensorCore", ScheduleRule::DefaultCUDATensorCore)
      .def("meta_schedule.ScheduleRuleDefaultHexagon", ScheduleRule::DefaultHexagon)
      .def("meta_schedule.ScheduleRuleDefaultARM", ScheduleRule::DefaultARM);
}

}  // namespace meta_schedule
}  // namespace tvm
