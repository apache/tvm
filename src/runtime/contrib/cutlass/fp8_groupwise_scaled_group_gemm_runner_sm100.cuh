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

#include <fstream>
#include <iostream>
#include <sstream>
#include <variant>
#include <vector>

#include "../../cuda/cuda_common.h"

// clang-format off
#include "cutlass/cutlass.h"

#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
// clang-format on

#define CUTLASS_CHECK(status)                                      \
  {                                                                \
    cutlass::Status error = status;                                \
    CHECK(error == cutlass::Status::kSuccess)                      \
        << "Got cutlass error: " << cutlassGetStatusString(error); \
  }

using namespace cute;
using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;  // <M,N,K> per group

inline size_t aligned(size_t value, size_t alignment = 16) {
  return (value + alignment - 1) / alignment * alignment;
}

template <typename TileShape, typename ClusterShape, typename ElementA, typename ElementB,
          typename ElementC, typename ElementBlockScale>
struct CutlassFP8ScaledGroupwiseGroupGemmRunnerSM100 {
  using LayoutA = cutlass::layout::RowMajor;
  static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;

  using LayoutB = cutlass::layout::ColumnMajor;
  static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;

  using LayoutC = cutlass::layout::RowMajor;
  static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;

  using ElementAccumulator = float;
  using ElementCompute = float;

  static constexpr int ScaleGranularityM = 1;
  static constexpr int ScaleGranularityN = 128;
  static constexpr int ScaleGranularityK = 128;
  using ScaleConfig =
      cutlass::detail::Sm100BlockwiseScaleConfig<ScaleGranularityM, ScaleGranularityN,
                                                 ScaleGranularityK, UMMA::Major::K, UMMA::Major::K>;

  using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
  using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());

  using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecialized2Sm;
  using KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedBlockwise2SmSm100;
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp, TileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator, ElementCompute, ElementC,
      LayoutC*, AlignmentC, ElementC, LayoutC*, AlignmentC, EpilogueSchedule>::CollectiveOp;
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp, ElementA,
      cute::tuple<LayoutA*, LayoutSFA*>, AlignmentA, ElementB, cute::tuple<LayoutB*, LayoutSFB*>,
      AlignmentB, ElementAccumulator, TileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
          sizeof(typename CollectiveEpilogue::SharedStorage))>,
      KernelSchedule>::CollectiveOp;
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<ProblemShape, CollectiveMainloop,
                                                          CollectiveEpilogue, void>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using StrideA = typename Gemm::GemmKernel::InternalStrideA;
  using StrideB = typename Gemm::GemmKernel::InternalStrideB;
  using StrideC = typename Gemm::GemmKernel::InternalStrideC;
  using StrideD = typename Gemm::GemmKernel::InternalStrideD;

  void run_group_gemm(const ElementA** ptr_A, const ElementB** ptr_B,
                      const ElementBlockScale** ptr_scales_a,
                      const ElementBlockScale** ptr_scales_b, const ElementC** ptr_C,
                      ElementC** ptr_D,
                      typename ProblemShape::UnderlyingProblemShape* problem_sizes,
                      typename ProblemShape::UnderlyingProblemShape* problem_sizes_host,
                      StrideA* stride_A, StrideB* stride_B, LayoutSFA* layout_scales_a,
                      LayoutSFB* layout_scales_b, StrideC* stride_C, StrideD* stride_D,
                      uint8_t* workspace, int64_t workspace_size, int num_groups,
                      cudaStream_t stream) {
    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = 0;
    hw_info.sm_count =
        cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);
    typename Gemm::Arguments arguments{cutlass::gemm::GemmUniversalMode::kGrouped,
                                       {num_groups, problem_sizes, problem_sizes_host},
                                       {ptr_A, stride_A, ptr_B, stride_B, ptr_scales_a,
                                        layout_scales_a, ptr_scales_b, layout_scales_b},
                                       {{}, ptr_C, stride_C, ptr_D, stride_D},
                                       hw_info};
    auto& fusion_args = arguments.epilogue.thread;
    fusion_args.alpha = 1.0f;
    fusion_args.beta = 0.0f;

    Gemm gemm_op;
    CUTLASS_CHECK(gemm_op.can_implement(arguments));
    CHECK_GE(workspace_size, gemm_op.get_workspace_size(arguments));
    CUTLASS_CHECK(gemm_op.initialize(arguments, workspace, stream));
    CUTLASS_CHECK(gemm_op.run(stream));
  }
};

template <typename ScaleConfig, typename ElementA, typename ElementB, typename ElementC,
          typename ElementBlockScale, typename StrideA, typename StrideB, typename StrideC,
          typename LayoutSFA, typename LayoutSFB>
__global__ void prepare_group_gemm_arguments(
    const ElementA** ptr_A, const ElementB** ptr_B, const ElementBlockScale** ptr_scales_a,
    const ElementBlockScale** ptr_scales_b, ElementC** ptr_D,
    typename ProblemShape::UnderlyingProblemShape* problem_sizes, StrideA* stride_A,
    StrideB* stride_B, LayoutSFA* layout_scales_a, LayoutSFB* layout_scales_b, StrideC* stride_D,
    const ElementA* a, const ElementB* b, const ElementBlockScale* scales_a,
    const ElementBlockScale* scales_b, ElementC* out, int64_t* indptr, int64_t n, int64_t k,
    int num_groups) {
  int group_id = threadIdx.x;
  if (group_id >= num_groups) return;
  int prev_rows = group_id == 0 ? 0 : indptr[group_id - 1];
  ptr_A[group_id] = a + prev_rows * k;
  ptr_B[group_id] = b + group_id * k * n;
  ptr_D[group_id] = out + prev_rows * n;
  ptr_scales_a[group_id] = scales_a + prev_rows * ((k + 127) / 128);
  ptr_scales_b[group_id] = scales_b + group_id * ((k + 127) / 128) * ((n + 127) / 128);
  int64_t m = indptr[group_id] - prev_rows;
  problem_sizes[group_id] = {static_cast<int>(m), static_cast<int>(n), static_cast<int>(k)};
  stride_A[group_id] = cute::make_stride(k, Int<1>{}, Int<0>{});
  stride_B[group_id] = cute::make_stride(k, Int<1>{}, Int<0>{});
  stride_D[group_id] = cute::make_stride(n, Int<1>{}, Int<0>{});
  layout_scales_a[group_id] = ScaleConfig::tile_atom_to_shape_SFA(
      make_shape(static_cast<int>(m), static_cast<int>(n), static_cast<int>(k), 1));
  layout_scales_b[group_id] = ScaleConfig::tile_atom_to_shape_SFB(
      make_shape(static_cast<int>(m), static_cast<int>(n), static_cast<int>(k), 1));
}

template <typename ElementA, typename ElementB, typename ElementC, typename ElementBlockScale>
void cutlass_fp8_groupwise_scaled_group_gemm_sm100(
    ElementA* a, ElementB* b, const ElementBlockScale* scales_a, const ElementBlockScale* scales_b,
    int64_t* indptr, uint8_t* workspace, int64_t workspace_size, int64_t n, int64_t k,
    int64_t num_groups, ElementC* out, cudaStream_t stream) {
  using TileShape = Shape<_256, _128, _128>;
  using ClusterShape = Shape<_2, _1, _1>;
  using Runner =
      CutlassFP8ScaledGroupwiseGroupGemmRunnerSM100<TileShape, ClusterShape, ElementA, ElementB,
                                                    ElementC, ElementBlockScale>;
  using ScaleConfig = typename Runner::ScaleConfig;
  using StrideA = typename Runner::StrideA;
  using StrideB = typename Runner::StrideB;
  using StrideC = typename Runner::StrideC;
  using LayoutSFA = typename Runner::LayoutSFA;
  using LayoutSFB = typename Runner::LayoutSFB;

  Runner runner;
  std::ptrdiff_t offset = 0;
  const ElementA** ptr_A = reinterpret_cast<const ElementA**>(workspace + offset);
  offset += aligned(sizeof(ElementA*) * num_groups);
  const ElementB** ptr_B = reinterpret_cast<const ElementB**>(workspace + offset);
  offset += aligned(sizeof(ElementB*) * num_groups);
  const ElementBlockScale** ptr_scales_a =
      reinterpret_cast<const ElementBlockScale**>(workspace + offset);
  offset += aligned(sizeof(ElementBlockScale*) * num_groups);
  const ElementBlockScale** ptr_scales_b =
      reinterpret_cast<const ElementBlockScale**>(workspace + offset);
  offset += aligned(sizeof(ElementBlockScale*) * num_groups);
  ElementC** ptr_D = reinterpret_cast<ElementC**>(workspace + offset);
  offset += aligned(sizeof(ElementC*) * num_groups);
  typename ProblemShape::UnderlyingProblemShape* problem_sizes =
      reinterpret_cast<typename ProblemShape::UnderlyingProblemShape*>(workspace + offset);
  offset += aligned(sizeof(typename ProblemShape::UnderlyingProblemShape) * num_groups);
  StrideA* stride_A = reinterpret_cast<StrideA*>(workspace + offset);
  offset += aligned(sizeof(StrideA) * num_groups);
  StrideB* stride_B = reinterpret_cast<StrideB*>(workspace + offset);
  offset += aligned(sizeof(StrideB) * num_groups);
  StrideC* stride_D = reinterpret_cast<StrideC*>(workspace + offset);
  offset += aligned(sizeof(StrideC) * num_groups);
  LayoutSFA* layout_scales_a = reinterpret_cast<LayoutSFA*>(workspace + offset);
  offset += aligned(sizeof(LayoutSFA) * num_groups);
  LayoutSFB* layout_scales_b = reinterpret_cast<LayoutSFB*>(workspace + offset);
  offset += aligned(sizeof(LayoutSFB) * num_groups);
  prepare_group_gemm_arguments<ScaleConfig, ElementA, ElementB, ElementC, ElementBlockScale,
                               StrideA, StrideB, StrideC, LayoutSFA, LayoutSFB>
      <<<1, num_groups, 0, stream>>>(ptr_A, ptr_B, ptr_scales_a, ptr_scales_b, ptr_D, problem_sizes,
                                     stride_A, stride_B, layout_scales_a, layout_scales_b, stride_D,
                                     a, b, scales_a, scales_b, out, indptr, n, k, num_groups);
  offset = aligned(offset, 256);
  runner.run_group_gemm(ptr_A, ptr_B, ptr_scales_a, ptr_scales_b,
                        const_cast<const ElementC**>(ptr_D), ptr_D, problem_sizes, nullptr,
                        stride_A, stride_B, layout_scales_a, layout_scales_b, stride_D, stride_D,
                        workspace + offset, workspace_size - offset, num_groups, stream);
}
