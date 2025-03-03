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
#include "cutlass/gemm/gemm.h"
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
using ProblemShape = Shape<int, int, int>;  // <M, N, K>

template <typename KernelTraits, typename ElementA, typename ElementB, typename ElementC,
          typename LayoutA = cutlass::layout::RowMajor,
          typename LayoutB = cutlass::layout::ColumnMajor,
          typename LayoutC = cutlass::layout::RowMajor>
struct CutlassGemmRunner {
  static constexpr int AlignmentA =
      128 / cutlass::sizeof_bits<ElementA>::value;  // Alignment of A matrix in units of elements
                                                    // (up to 16 bytes)

  static constexpr int AlignmentB =
      128 / cutlass::sizeof_bits<ElementB>::value;  // Alignment of B matrix in units of elements
                                                    // (up to 16 bytes)

  static constexpr int AlignmentC =
      128 / cutlass::sizeof_bits<ElementC>::value;  // Alignment of C matrix in units of elements
                                                    // (up to 16 bytes)

  // Core kernel configurations
  using ElementAccumulator = float;  // Element type for internal accumulation
  using ScaleType = std::variant<ElementAccumulator, const ElementAccumulator*>;
  using ArchTag =
      cutlass::arch::Sm90;  // Tag indicating the minimum SM that supports the intended feature
  using OperatorClass = cutlass::arch::OpClassTensorOp;  // Operator class tag
  using TileShape = typename KernelTraits::TileShape;
  using ClusterShape = typename KernelTraits::ClusterShape;
  using StageCountType =
      cutlass::gemm::collective::StageCountAuto;  // Stage count maximized based on the tile size
  using KernelSchedule = typename KernelTraits::KernelSchedule;    // Kernel to launch
  using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized;  // Epilogue to launch

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag, OperatorClass, TileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator, ElementAccumulator,
      ElementC, LayoutC, AlignmentC, ElementC, LayoutC, AlignmentC, EpilogueSchedule>::CollectiveOp;
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag, OperatorClass, ElementA, LayoutA, AlignmentA, ElementB, LayoutB, AlignmentB,
      ElementAccumulator, TileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
          sizeof(typename CollectiveEpilogue::SharedStorage))>,
      KernelSchedule>::CollectiveOp;

  using GemmKernel =
      cutlass::gemm::kernel::GemmUniversal<ProblemShape, CollectiveMainloop, CollectiveEpilogue>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;

  void run_gemm(const ElementA* ptr_A, const ElementB* ptr_B, const ElementC* ptr_C,
                ElementC* ptr_D, ProblemShape* problem_size, StrideA* stride_A, StrideB* stride_B,
                StrideC* stride_C, StrideD* stride_D, uint8_t* workspace, int64_t workspace_size,
                ScaleType alpha, ScaleType beta, cudaStream_t stream) {
    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = 0;
    hw_info.sm_count =
        cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);
    typename Gemm::Arguments arguments{cutlass::gemm::GemmUniversalMode::kGemm,
                                       *problem_size,
                                       {ptr_A, *stride_A, ptr_B, *stride_B},
                                       {{}, ptr_C, *stride_C, ptr_D, *stride_D},
                                       //  {epilogue_params, ptr_C, *stride_C, ptr_D, *stride_D},
                                       hw_info};

    ICHECK(alpha.index() == beta.index()) << "alpha and beta must have the same type";
    if (std::holds_alternative<ElementAccumulator>(alpha)) {
      arguments.epilogue.thread.alpha = std::get<ElementAccumulator>(alpha);
      arguments.epilogue.thread.beta = std::get<ElementAccumulator>(beta);
    } else if (std::holds_alternative<const ElementAccumulator*>(alpha)) {
      arguments.epilogue.thread.alpha_ptr = std::get<const ElementAccumulator*>(alpha);
      arguments.epilogue.thread.beta_ptr = std::get<const ElementAccumulator*>(beta);
    } else {
      LOG(FATAL) << "Unsupported alpha and beta type";
      throw;
    }

    Gemm gemm_op;
    CUTLASS_CHECK(gemm_op.can_implement(arguments));
    CHECK_GE(workspace_size, gemm_op.get_workspace_size(arguments));
    CUTLASS_CHECK(gemm_op.initialize(arguments, workspace, stream));
    CUTLASS_CHECK(gemm_op.run(stream));
  }
};

template <typename KernelTraits, typename ElementA, typename ElementB, typename ElementC>
void cutlass_gemm(ElementA* x, ElementB* weight, uint8_t* workspace, int64_t workspace_size,
                  int64_t m, int64_t n, int64_t k, std::variant<float, const float*> alpha,
                  std::variant<float, const float*> beta, ElementC* out, cudaStream_t stream) {
  using Runner = CutlassGemmRunner<KernelTraits, ElementA, ElementB, ElementC>;
  using StrideA = typename Runner::StrideA;
  using StrideB = typename Runner::StrideB;
  using StrideC = typename Runner::StrideC;

  Runner runner;
  StrideA stride_A = cute::make_stride(k, Int<1>{}, int64_t{0});
  StrideB stride_B = cute::make_stride(k, Int<1>{}, int64_t{0});
  StrideC stride_D = cute::make_stride(n, Int<1>{}, int64_t{0});
  ProblemShape problem_size{static_cast<int>(m), static_cast<int>(n), static_cast<int>(k)};
  runner.run_gemm(x, weight, out, out, &problem_size, &stride_A, &stride_B, &stride_D, &stride_D,
                  workspace, workspace_size, alpha, beta, stream);
}
