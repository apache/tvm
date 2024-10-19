#pragma once

#include <cutlass/cutlass.h>
#include <cutlass/gemm/warp/mma_tensor_op_sm70.h>

#include "common.h"

using cutlass::gemm::GemmShape;

// Primary template
// Add 128 bits padding when the last dim is a multiple of 256 bits
template <typename T, bool transpose, int M, int K, typename Enable = void>
struct DispatchSharedMemoryLayoutA {
  using Layout = typename std::conditional<transpose, cutlass::layout::ColumnMajor,
                                           cutlass::layout::RowMajor>::type;
  static int constexpr Dim = transpose ? M : K;
  static int constexpr Stride = (Dim * sizeof(T) % 32 == 0) ? Dim + 16 / sizeof(T) : Dim;
};
template <typename T, bool transpose, int N, int K, typename Enable = void>
struct DispatchSharedMemoryLayoutB {
  using Layout = typename std::conditional<transpose, cutlass::layout::ColumnMajor,
                                           cutlass::layout::RowMajor>::type;
  static int constexpr Dim = transpose ? K : N;
  static int constexpr Stride = (Dim * sizeof(T) % 32 == 0) ? Dim + 16 / sizeof(T) : Dim;
};

// Partial specialization for half_t
template <int M, int K>
struct DispatchSharedMemoryLayoutA<half_t, true, M, K, typename std::enable_if<M % 64 == 0>::type> {
  using Layout = cutlass::layout::ColumnMajorVoltaTensorOpMultiplicandCongruous<16>;
  static int constexpr Stride = M;
};

template <int M, int K>
struct DispatchSharedMemoryLayoutA<half_t, false, M, K> {
  using Layout = cutlass::layout::RowMajorVoltaTensorOpMultiplicandCrosswise<16, K>;
  static int constexpr Stride = M;
};

template <int N, int K>
struct DispatchSharedMemoryLayoutB<half_t, true, N, K> {
  using Layout = cutlass::layout::ColumnMajorVoltaTensorOpMultiplicandCrosswise<16, K>;
  static int constexpr Stride = N;
};

template <int N, int K>
struct DispatchSharedMemoryLayoutB<half_t, false, N, K,
                                   typename std::enable_if<N % 64 == 0>::type> {
  using Layout = cutlass::layout::RowMajorVoltaTensorOpMultiplicandBCongruous<16>;
  static int constexpr Stride = N;
};

template <typename Shape, int num_warp_m, int num_warp_n, bool trans_A, bool trans_B,
          typename A_type_raw, typename B_type_raw, typename C_type_raw>
class GemmTensorOp {
 public:
  using A_type = A_type_raw;
  using B_type = B_type_raw;
  using C_type = C_type_raw;
  using InstructionShape = GemmShape<16, 16, 4>;
  using SMemLayoutA =
      typename DispatchSharedMemoryLayoutA<A_type, trans_A, Shape::kM, Shape::kK>::Layout;
  using SMemLayoutB =
      typename DispatchSharedMemoryLayoutB<B_type, trans_B, Shape::kN, Shape::kK>::Layout;
  static constexpr int stride_A =
      DispatchSharedMemoryLayoutA<A_type, trans_A, Shape::kM, Shape::kK>::Stride;
  static constexpr int stride_B =
      DispatchSharedMemoryLayoutB<B_type, trans_B, Shape::kN, Shape::kK>::Stride;

  using Policy = cutlass::gemm::warp::MmaTensorOpPolicy<
      cutlass::arch::Mma<InstructionShape, 32, A_type,
                         typename std::conditional<trans_A, cutlass::layout::ColumnMajor,
                                                   cutlass::layout::RowMajor>::type,
                         B_type,
                         typename std::conditional<trans_B, cutlass::layout::ColumnMajor,
                                                   cutlass::layout::RowMajor>::type,
                         C_type, cutlass::layout::RowMajor, cutlass::arch::OpMultiplyAdd>,
      cutlass::MatrixShape<1, 1> >;

  static_assert(Shape::kM % num_warp_m == 0);
  static_assert(Shape::kN % num_warp_n == 0);

  using MmaWarp = typename cutlass::gemm::warp::MmaVoltaTensorOp<
      GemmShape<Shape::kM / num_warp_m, Shape::kN / num_warp_n, InstructionShape::kK>, A_type,
      SMemLayoutA, B_type, SMemLayoutB, C_type, cutlass::layout::RowMajor, Policy>;

  using TensorRefA = typename MmaWarp::IteratorA::TensorRef;
  using TensorRefB = typename MmaWarp::IteratorB::TensorRef;
  using FragmentA = typename MmaWarp::FragmentA;
  using FragmentB = typename MmaWarp::FragmentB;
  using FragmentC = typename MmaWarp::FragmentC;
  using IteratorA = typename MmaWarp::IteratorA;
  using IteratorB = typename MmaWarp::IteratorB;

  static_assert(Shape::kK % InstructionShape::kK == 0);
  static int constexpr kKgroups = Shape::kK / InstructionShape::kK;

  static CUTLASS_DEVICE void body(A_type_raw* pA, B_type_raw* pB, FragmentC& accum,
                                  const int warp_idx_m, const int warp_idx_n, const int lane_id) {
    MmaWarp mma_op;
    FragmentA frag_A;
    FragmentB frag_B;
    const TensorRefA ref_A((A_type*)pA, stride_A);
    const TensorRefB ref_B((B_type*)pB, stride_B);
    IteratorA iter_A(ref_A, lane_id);
    IteratorB iter_B(ref_B, lane_id);
    iter_A.add_tile_offset({warp_idx_m, 0});
    iter_B.add_tile_offset({0, warp_idx_n});
    CUTLASS_PRAGMA_UNROLL
    for (int k = 0; k < kKgroups; ++k) {
      iter_A.load(frag_A);
      iter_B.load(frag_B);
      ++iter_A;
      ++iter_B;
      mma_op(accum, frag_A, frag_B, accum);
    }
  }

  static CUTLASS_DEVICE void body_rs(const FragmentA* frag_A, B_type_raw* pB, FragmentC& accum,
                                     const int warp_idx_n, const int lane_id) {
    MmaWarp mma_op;
    FragmentB frag_B;
    const TensorRefB ref_B((B_type*)pB, stride_B);
    IteratorB iter_B(ref_B, lane_id);
    iter_B.add_tile_offset({0, warp_idx_n});
    CUTLASS_PRAGMA_UNROLL
    for (int k = 0; k < kKgroups; ++k) {
      iter_B.load(frag_B);
      ++iter_B;
      mma_op(accum, frag_A[k], frag_B, accum);
    }
  }
};

namespace tl {

template <int M, int N, int K, int num_warp_m, int num_warp_n, bool trans_A, bool trans_B,
          typename A_type, typename B_type, typename C_type>
CUTLASS_DEVICE void gemm_ss(A_type* pA, B_type* pB, C_type* accum) {
  using MMA = GemmTensorOp<GemmShape<M, N, K>, num_warp_m, num_warp_n, trans_A, trans_B, A_type,
                           B_type, C_type>;
  using FragmentC = typename MMA::FragmentC;
  int warp_id = threadIdx.x / 32;
  int lane_id = threadIdx.x % 32;
  MMA::body(pA, pB, *(FragmentC*)(accum), warp_id / num_warp_n, warp_id % num_warp_n, lane_id);
}

template <int M, int N, int K, int num_warp_m, int num_warp_n, bool trans_A, bool trans_B,
          typename A_type, typename B_type, typename C_type>
CUTLASS_DEVICE void gemm_rs(A_type* pA, B_type* pB, C_type* accum) {
  using MMA = GemmTensorOp<GemmShape<M, N, K>, num_warp_m, num_warp_n, trans_A, trans_B, A_type,
                           B_type, C_type>;
  using FragmentA = typename MMA::FragmentA;
  using FragmentC = typename MMA::FragmentC;
  int warp_id = threadIdx.x / 32;
  int lane_id = threadIdx.x % 32;
  MMA::body_rs((const FragmentA*)(pA), pB, *(FragmentC*)(accum), warp_id % num_warp_n, lane_id);
}

};  // namespace tl
