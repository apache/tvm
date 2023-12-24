#pragma once

#include <cute/algorithm/copy.hpp>

#include "common.h"

using namespace cute;

template <class... Args>
__device__ __inline__ auto remove_swizzle(Layout<Args...> const& layout) {
  return layout;
}

template <class... Args>
__device__ __inline__ auto remove_swizzle(ComposedLayout<Args...> const& layout) {
  return layout.layout_b();
}

template <typename A_type, typename B_type, typename C_type>
struct DispatchInstruction;

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
template <>
struct DispatchInstruction<half_t, half_t, half_t> {
  using MMA = MMA_Atom<SM80_16x8x16_F16F16F16F16_TN>;
  using MMA_Group = Layout<Shape<_1, _2, _1>>;
};
template <>
struct DispatchInstruction<half_t, half_t, float> {
  using MMA = MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>;
  using MMA_Group = Layout<Shape<_1, _2, _1>>;
};
template <>
struct DispatchInstruction<bfloat16_t, bfloat16_t, float> {
  using MMA = MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>;
  using MMA_Group = Layout<Shape<_1, _2, _1>>;
};
template <>
struct DispatchInstruction<tfloat32_t, tfloat32_t, float> {
  using MMA = MMA_Atom<SM80_16x8x8_F32TF32TF32F32_TN>;
  using MMA_Group = Layout<Shape<_1, _2, _1>>;
};
template <>
struct DispatchInstruction<int8_t, int8_t, int> {
  using MMA = MMA_Atom<SM80_16x8x32_S32S8S8S32_TN>;
  using MMA_Group = Layout<Shape<_1, _2, _1>>;
};
#elif (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 750))
template <>
struct DispatchInstruction<half_t, half_t, float> {
  using MMA = MMA_Atom<SM75_16x8x8_F32F16F16F32_TN>;
  using MMA_Group = Layout<Shape<_1, _2, _2>>;
};
#endif

template <int Bits, int N, int K, bool K_inner, typename Enable = void>
struct DispatchSharedMemoryLayout;

template <int N, int K>
struct DispatchSharedMemoryLayout<16, N, K, true,
                                  typename std::enable_if<K % 32 == 0 && K % 64 != 0>::type> {
  using LayoutAtom =
      decltype(composition(Swizzle<2, 3, 3>{}, Layout<Shape<_8, _32>, Stride<_32, _1>>{}));
  using Layout = decltype(tile_to_shape(LayoutAtom{}, Shape<Int<N>, Int<K>>{}));
  using Copy = SM75_U32x4_LDSM_N;
};

template <int N, int K>
struct DispatchSharedMemoryLayout<16, N, K, true, typename std::enable_if<K % 64 == 0>::type> {
  using LayoutAtom =
      decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<_8, _64>, Stride<_64, _1>>{}));
  using Layout = decltype(tile_to_shape(LayoutAtom{}, Shape<Int<N>, Int<K>>{}));
  using Copy = SM75_U32x4_LDSM_N;
};

template <int N, int K>
struct DispatchSharedMemoryLayout<16, N, K, false,
                                  typename std::enable_if<N % 32 == 0 && N % 64 != 0>::type> {
  using LayoutAtom =
      decltype(composition(Swizzle<2, 3, 3>{}, Layout<Shape<_32, _8>, Stride<_1, _32>>{}));
  using Layout = decltype(tile_to_shape(LayoutAtom{}, Shape<Int<N>, Int<K>>{}, Step<_2, _1>{}));
  using Copy = SM75_U16x8_LDSM_T;
};

template <int N, int K>
struct DispatchSharedMemoryLayout<16, N, K, false, typename std::enable_if<N % 64 == 0>::type> {
  using LayoutAtom =
      decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<_64, _8>, Stride<_1, _64>>{}));
  using Layout = decltype(tile_to_shape(LayoutAtom{}, Shape<Int<N>, Int<K>>{}, Step<_2, _1>{}));
  using Copy = SM75_U16x8_LDSM_T;
};

template <int N, int K>
struct DispatchSharedMemoryLayout<32, N, K, true, typename std::enable_if<K % 32 == 0>::type> {
  using LayoutAtom =
      decltype(composition(Swizzle<3, 2, 3>{}, Layout<Shape<_8, _32>, Stride<_32, _1>>{}));
  using Layout = decltype(tile_to_shape(LayoutAtom{}, Shape<Int<N>, Int<K>>{}));
  using Copy = SM75_U32x4_LDSM_N;
};

template <int N, int K>
struct DispatchSharedMemoryLayout<32, N, K, false, typename std::enable_if<N % 32 == 0>::type> {
  using LayoutAtom =
      decltype(composition(Swizzle<3, 2, 3>{}, Layout<Shape<_32, _8>, Stride<_1, _32>>{}));
  using Layout = decltype(tile_to_shape(LayoutAtom{}, Shape<Int<N>, Int<K>>{}, Step<_2, _1>{}));
  using Copy = SM75_U16x8_LDSM_T;
};

template <int M, int N, int K, int num_warp_m, int num_warp_n, bool trans_A, bool trans_B,
          typename A_type_raw, typename B_type_raw, typename C_type_raw>
class GemmTensorOp {
 public:
  using A_type = typename std::conditional<std::is_same<A_type_raw, float>::value, tfloat32_t,
                                           A_type_raw>::type;
  using B_type = typename std::conditional<std::is_same<B_type_raw, float>::value, tfloat32_t,
                                           A_type_raw>::type;
  using C_type = C_type_raw;
  using Instruction = DispatchInstruction<A_type, B_type, C_type>;

  using OperandATraits = DispatchSharedMemoryLayout<sizeof_bits<A_type>::value, M, K, !trans_A>;
  using OperandBTraits = DispatchSharedMemoryLayout<sizeof_bits<B_type>::value, N, K, trans_B>;
  using SmemLayoutA = typename OperandATraits::Layout;
  using SmemLayoutB = typename OperandBTraits::Layout;
  using SmemCopyA = Copy_Atom<typename OperandATraits::Copy, A_type>;
  using SmemCopyB = Copy_Atom<typename OperandBTraits::Copy, B_type>;

  using TileMma =
      TiledMMA<typename Instruction::MMA, Layout<Shape<Int<num_warp_m>, Int<num_warp_n>, _1>>,
               typename Instruction::MMA_Group>;

  static CUTE_DEVICE void body(A_type_raw* pA, B_type_raw* pB, C_type_raw* pC) {
    const int tid = threadIdx.x;
    Tensor sA = make_tensor(make_smem_ptr(reinterpret_cast<A_type*>(pA)), SmemLayoutA{});
    Tensor sB = make_tensor(make_smem_ptr(reinterpret_cast<B_type*>(pB)), SmemLayoutB{});
    TileMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tid);
    auto tiled_copy_A = make_tiled_copy_A(SmemCopyA{}, tiled_mma);
    auto tiled_copy_B = make_tiled_copy_B(SmemCopyB{}, tiled_mma);
    auto thr_copy_A = tiled_copy_A.get_thread_slice(tid);
    auto thr_copy_B = tiled_copy_B.get_thread_slice(tid);

    Tensor tCrA = thr_mma.partition_fragment_A(sA);
    Tensor tCrB = thr_mma.partition_fragment_B(sB);
    Tensor tCsA = thr_copy_A.partition_S(sA);
    Tensor tCsB = thr_copy_B.partition_S(sB);

    Tensor tCrA_copy_view = thr_copy_A.retile_D(tCrA);
    Tensor tCrB_copy_view = thr_copy_B.retile_D(tCrB);

    Tensor acc = make_tensor(make_rmem_ptr(reinterpret_cast<C_type*>(pC)),
                             partition_shape_C(tiled_mma, Shape<Int<M>, Int<N>>{}));

    // when layout is KxN and n_warp is 1, there seem to be a bug, use this as a workaround
    auto tCrA_view = make_tensor(tCrA.data(), remove_swizzle(tCrA.layout()));
    auto tCrB_view = make_tensor(tCrB.data(), remove_swizzle(tCrB.layout()));
    CUTE_UNROLL
    for (int k = 0; k < size<2>(tCrA); ++k) {
      copy(tiled_copy_A, tCsA(_, _, k), tCrA_copy_view(_, _, k));
      copy(tiled_copy_B, tCsB(_, _, k), tCrB_copy_view(_, _, k));
      gemm(tiled_mma, tCrA_view(_, _, k), tCrB_view(_, _, k), acc);
    }
  }

  static CUTE_DEVICE void body_rs(A_type_raw* pA, B_type_raw* pB, C_type_raw* pC) {
    const int tid = threadIdx.x;
    Tensor sB = make_tensor(make_smem_ptr(reinterpret_cast<B_type*>(pB)), SmemLayoutB{});
    TileMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tid);
    auto tiled_copy_B = make_tiled_copy_B(SmemCopyB{}, tiled_mma);
    auto thr_copy_B = tiled_copy_B.get_thread_slice(tid);

    Tensor tCrB = thr_mma.partition_fragment_B(sB);
    Tensor tCsB = thr_copy_B.partition_S(sB);

    Tensor tCrB_copy_view = thr_copy_B.retile_D(tCrB);

    Tensor acc = make_tensor(make_rmem_ptr(reinterpret_cast<C_type*>(pC)),
                             partition_shape_C(tiled_mma, Shape<Int<M>, Int<N>>{}));
    Tensor tCrA = make_tensor(make_rmem_ptr(reinterpret_cast<A_type*>(pA)),
                              partition_shape_A(tiled_mma, Shape<Int<M>, Int<K>>{}));

    auto tCrB_view = make_tensor(tCrB.data(), remove_swizzle(tCrB.layout()));
    copy(tiled_copy_B, tCsB(_, _, 0), tCrB_copy_view(_, _, 0));
    CUTE_UNROLL
    for (int k = 0; k < size<2>(tCrA); ++k) {
      if (k < size<2>(tCrA) - 1) {
        copy(tiled_copy_B, tCsB(_, _, k + 1), tCrB_copy_view(_, _, k + 1));
      }
      gemm(tiled_mma, tCrA(_, _, k), tCrB_view(_, _, k), acc);
    }
  }
};

namespace tl {

template <int M, int N, int K, int num_warp_m, int num_warp_n, bool trans_A, bool trans_B,
          typename A_type, typename B_type, typename C_type>
CUTLASS_DEVICE void gemm_ss(A_type* pA, B_type* pB, C_type* accum) {
  using MMA =
      GemmTensorOp<M, N, K, num_warp_m, num_warp_n, trans_A, trans_B, A_type, B_type, C_type>;
  MMA::body(pA, pB, accum);
}

template <int M, int N, int K, int num_warp_m, int num_warp_n, bool trans_A, bool trans_B,
          typename A_type, typename B_type, typename C_type>
CUTLASS_DEVICE void gemm_rs(A_type* pA, B_type* pB, C_type* accum) {
  using MMA =
      GemmTensorOp<M, N, K, num_warp_m, num_warp_n, trans_A, trans_B, A_type, B_type, C_type>;
  MMA::body_rs(pA, pB, accum);
}

}  // namespace tl
