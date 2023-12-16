#pragma once

#include <cutlass/cutlass.h>
#include <cutlass/gemm/warp/mma_tensor_op.h>

#include "common.h"

using cutlass::gemm::GemmShape;

template <typename A_type, typename B_type, typename C_type>
struct DispatchInstruction;

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
template <>
struct DispatchInstruction<half_t, half_t, half_t> {
  using Shape = GemmShape<16, 8, 16>;
};
template <>
struct DispatchInstruction<half_t, half_t, float> {
  using Shape = GemmShape<16, 8, 16>;
};
template <>
struct DispatchInstruction<bfloat16_t, bfloat16_t, float> {
  using Shape = GemmShape<16, 8, 16>;
};
template <>
struct DispatchInstruction<tfloat32_t, tfloat32_t, float> {
  using Shape = GemmShape<16, 8, 8>;
};
template <>
struct DispatchInstruction<double, double, double> {
  using Shape = GemmShape<8, 8, 4>;
};
template <>
struct DispatchInstruction<int8_t, int8_t, int> {
  using Shape = GemmShape<16, 8, 32>;
};
#elif (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 750))
template <>
struct DispatchInstruction<half_t, half_t, half_t> {
  using Shape = GemmShape<16, 8, 8>;
};
template <>
struct DispatchInstruction<half_t, half_t, float> {
  using Shape = GemmShape<16, 8, 8>;
};
template <>
struct DispatchInstruction<int8_t, int8_t, int> {
  using Shape = GemmShape<8, 8, 16>;
};
#endif

template <typename T, bool transpose, int M, int K>
struct DispatchSharedMemoryLayoutA;
template <typename T, bool transpose, int N, int K>
struct DispatchSharedMemoryLayoutB;
template <int M, int K>
struct DispatchSharedMemoryLayoutA<double, true, M, K> {
  using Layout = cutlass::layout::ColumnMajorTensorOpMultiplicandCongruous64b;
  static int constexpr Stride = M;
};
template <int M, int K>
struct DispatchSharedMemoryLayoutA<double, false, M, K> {
  using Layout = cutlass::layout::RowMajorTensorOpMultiplicand64bCrosswise;
  static int constexpr Stride = M;  // special case
};
template <int N, int K>
struct DispatchSharedMemoryLayoutB<double, true, N, K> {
  using Layout = cutlass::layout::ColumnMajorTensorOpMultiplicand64bCrosswise;
  static int constexpr Stride = N;  // special case
};
template <int N, int K>
struct DispatchSharedMemoryLayoutB<double, false, N, K> {
  using Layout = cutlass::layout::RowMajorTensorOpMultiplicandCongruous64b;
  static int constexpr Stride = N;
};
template <int M, int K>
struct DispatchSharedMemoryLayoutA<int8_t, true, M, K> {
  using Layout = cutlass::layout::ColumnMajor;
  static int constexpr Stride = M + 16;  // padded
};
template <int N, int K>
struct DispatchSharedMemoryLayoutB<int8_t, false, N, K> {
  using Layout = cutlass::layout::RowMajor;
  static int constexpr Stride = N + 16;  // padded
};
template <typename T, int M, int K>
struct DispatchSharedMemoryLayoutA<T, true, M, K> {
  using Layout =
      cutlass::layout::ColumnMajorTensorOpMultiplicandCongruous<8 * sizeof(T), 128 / sizeof(T)>;
  static int constexpr Stride = M;
};
template <typename T, int M, int K>
struct DispatchSharedMemoryLayoutA<T, false, M, K> {
  using Layout =
      cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<8 * sizeof(T), 64 / sizeof(T)>;
  static int constexpr Stride = K;
};
template <typename T, int N, int K>
struct DispatchSharedMemoryLayoutB<T, true, N, K> {
  using Layout =
      cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<8 * sizeof(T), 64 / sizeof(T)>;
  static int constexpr Stride = K;
};
template <typename T, int N, int K>
struct DispatchSharedMemoryLayoutB<T, false, N, K> {
  using Layout =
      cutlass::layout::RowMajorTensorOpMultiplicandCongruous<8 * sizeof(T), 128 / sizeof(T)>;
  static int constexpr Stride = N;
};

template <typename Shape, int num_warp_m, int num_warp_n, bool trans_A, bool trans_B,
          typename A_type_raw, typename B_type_raw, typename C_type_raw>
class GemmTensorOp {
 public:
  using A_type = typename std::conditional<std::is_same<A_type_raw, float>::value, tfloat32_t,
                                           A_type_raw>::type;
  using B_type = typename std::conditional<std::is_same<B_type_raw, float>::value, tfloat32_t,
                                           A_type_raw>::type;
  using C_type = C_type_raw;
  using InstructionShape = typename DispatchInstruction<A_type, B_type, C_type>::Shape;
  using SMemLayoutA =
      typename DispatchSharedMemoryLayoutA<A_type, trans_A, Shape::kM, Shape::kK>::Layout;
  using SMemLayoutB =
      typename DispatchSharedMemoryLayoutB<B_type, trans_B, Shape::kN, Shape::kK>::Layout;
  static constexpr int stride_A =
      DispatchSharedMemoryLayoutA<A_type, trans_A, Shape::kM, Shape::kK>::Stride;
  static constexpr int stride_B =
      DispatchSharedMemoryLayoutB<B_type, trans_B, Shape::kN, Shape::kK>::Stride;

  using Policy = cutlass::gemm::warp::MmaTensorOpPolicy<
      cutlass::arch::Mma<InstructionShape, 32, A_type, cutlass::layout::RowMajor, B_type,
                         cutlass::layout::ColumnMajor, C_type, cutlass::layout::RowMajor,
                         cutlass::arch::OpMultiplyAdd>,
      cutlass::MatrixShape<1, 1> >;

  static_assert(Shape::kM % num_warp_m == 0);
  static_assert(Shape::kN % num_warp_n == 0);

  using MmaWarp = typename cutlass::gemm::warp::MmaTensorOp<
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

}  // namespace tl
