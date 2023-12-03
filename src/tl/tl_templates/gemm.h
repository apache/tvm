#pragma once

#include "common.h"

#include <cutlass/cutlass.h>
#include <cutlass/gemm/warp/mma_tensor_op.h>

using cutlass::gemm::GemmShape;

template<typename A_type, typename B_type, typename C_type>
class DispatchInstruction;

template<>
class DispatchInstruction<half_t, half_t, half_t> {
 public:
  using Shape = GemmShape<16, 8, 16>;
};

template<>
class DispatchInstruction<half_t, half_t, float> {
 public:
  using Shape = GemmShape<16, 8, 16>;
};

template<>
class DispatchInstruction<bfloat16_t, bfloat16_t, float> {
 public:
  using Shape = GemmShape<16, 8, 16>;
};

template<>
class DispatchInstruction<tfloat32_t, tfloat32_t, float> {
 public:
  using Shape = GemmShape<16, 8, 8>;
};

template <
  typename Shape,
  bool trans_A,
  bool trans_B,
  typename A_type_raw,
  typename B_type_raw,
  typename C_type
>
class GemmTensorOp {
public:
  using A_type = typename std::conditional<
    std::is_same<A_type_raw, float>::value, tfloat32_t, A_type_raw>::type;
  using B_type = typename std::conditional<
    std::is_same<B_type_raw, float>::value, tfloat32_t, A_type_raw>::type;

  using InstructionShape = typename DispatchInstruction<A_type, B_type, C_type>::Shape;

  using SMemLayoutA = typename std::conditional<
    trans_A,
    cutlass::layout::ColumnMajorTensorOpMultiplicandCongruous<8 * sizeof(A_type), 128 / sizeof(A_type)>,
    cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<8 * sizeof(A_type), 64 / sizeof(A_type)>
  >::type;

  using SMemLayoutB = typename std::conditional<
    trans_B,
    cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<8 * sizeof(B_type), 64 / sizeof(B_type)>,
    cutlass::layout::RowMajorTensorOpMultiplicandCongruous<8 * sizeof(B_type), 128 / sizeof(B_type)>
  >::type;

  using Policy = cutlass::gemm::warp::MmaTensorOpPolicy<
    cutlass::arch::Mma<
      InstructionShape,
      32,
      A_type,
      cutlass::layout::RowMajor,
      B_type,
      cutlass::layout::ColumnMajor,
      C_type,
      cutlass::layout::RowMajor,
      cutlass::arch::OpMultiplyAdd
    >,
    cutlass::MatrixShape<1, 1>
  >;

  using MmaWarp = typename cutlass::gemm::warp::MmaTensorOp<
    GemmShape<Shape::kM, Shape::kN, InstructionShape::kK>,
    A_type,
    SMemLayoutA,
    B_type,
    SMemLayoutB,
    C_type,
    cutlass::layout::RowMajor,
    Policy
  >;

  using TensorRefA = typename MmaWarp::IteratorA::TensorRef;
  using TensorRefB = typename MmaWarp::IteratorB::TensorRef;
  using FragmentA = typename MmaWarp::FragmentA;
  using FragmentB = typename MmaWarp::FragmentB;
  using FragmentC = typename MmaWarp::FragmentC;
  using IteratorA = typename MmaWarp::IteratorA;
  using IteratorB = typename MmaWarp::IteratorB;

  static_assert(Shape::kK % InstructionShape::kK == 0);
  static int constexpr kKgroups = Shape::kK / InstructionShape::kK;

  static CUTLASS_DEVICE
  void body(const TensorRefA &ref_A, const TensorRefB &ref_B, FragmentC &accum,
            const int warp_idx_m, const int warp_idx_n, const int lane_id) {
    MmaWarp mma_op;
    FragmentA frag_A;
    FragmentB frag_B;
    IteratorA iter_A(ref_A, lane_id);
    IteratorB iter_B(ref_B, lane_id);
    iter_A.add_tile_offset({warp_idx_m, 0});
    iter_B.add_tile_offset({0, warp_idx_n});
    #pragma unroll
    for (int k = 0; k < kKgroups; ++k) {
      iter_A.load(frag_A);
      iter_B.load(frag_B);
      ++iter_A;
      ++iter_B;
      mma_op(accum, frag_A, frag_B, accum);
    }
  }

  static CUTLASS_DEVICE
  void body_rs(const FragmentA *frag_A, const TensorRefB &ref_B, FragmentC &accum,
               const int warp_idx_n, const int lane_id) {
    MmaWarp mma_op;
    FragmentB frag_B;
    IteratorB iter_B(ref_B, lane_id);
    iter_B.add_tile_offset({0, warp_idx_n});
    #pragma unroll
    for (int k = 0; k < kKgroups; ++k) {
      iter_B.load(frag_B);
      ++iter_B;
      mma_op(accum, frag_A[k], frag_B, accum);
    }
  }
};

namespace tl {

template<int M, int N, int K, int warp_size_M, int warp_size_N, bool trans_A, bool trans_B, typename A_type, typename B_type, typename C_type>
CUTLASS_DEVICE void gemm_ss(A_type* pA, B_type* pB, C_type* accum) {
  using MMA = GemmTensorOp<GemmShape<M / warp_size_M, N / warp_size_N, K>, trans_A, trans_B, A_type, B_type, C_type>;
  using TensorRefA = typename MMA::TensorRefA;
  using TensorRefB = typename MMA::TensorRefB;
  using FragmentC = typename MMA::FragmentC;
  TensorRefA refA{(typename TensorRefA::Element*)pA, trans_A ? M : K};
  TensorRefB refB{(typename TensorRefB::Element*)pB, trans_B ? K : N};
  int warp_id = threadIdx.x / 32;
  int lane_id = threadIdx.x % 32;
  MMA::body(refA, refB, *(FragmentC*)(accum), warp_id / warp_size_N, warp_id % warp_size_N, lane_id);
}

template<int M, int N, int K, int warp_size_M, int warp_size_N, bool trans_A, bool trans_B, typename A_type, typename B_type, typename C_type>
CUTLASS_DEVICE void gemm_rs(A_type* pA, B_type* pB, C_type* accum) {
  using MMA = GemmTensorOp<GemmShape<M / warp_size_M, N / warp_size_N, K>, trans_A, trans_B, A_type, B_type, C_type>;
  using TensorRefB = typename MMA::TensorRefB;
  using FragmentA = typename MMA::FragmentA;
  using FragmentC = typename MMA::FragmentC;
  TensorRefB refB{(typename TensorRefB::Element*)pB, trans_B ? K : N};
  int warp_id = threadIdx.x / 32;
  int lane_id = threadIdx.x % 32;
  MMA::body_rs((const FragmentA*)(pA), refB, *(FragmentC*)(accum), warp_id % warp_size_N, lane_id);
}

} // namespace tl
