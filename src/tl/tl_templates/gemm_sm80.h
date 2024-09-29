#pragma once

#include <cute/algorithm/copy.hpp>

#include "common.h"

namespace cute {

template <typename A_type, typename B_type, typename C_type>
struct DispatchInstruction;

#if (defined(__CUDA_ARCH_LIST__) && (__CUDA_ARCH_LIST__ >= 800))
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
template <>
struct DispatchInstruction<double, double, double> {
  using MMA = MMA_Atom<SM80_8x8x4_F64F64F64F64_TN>;
  using MMA_Group = Layout<Shape<_2, _2, _1>>;
};
#elif (defined(__CUDA_ARCH_LIST__) && (__CUDA_ARCH_LIST__ >= 750))
template <>
struct DispatchInstruction<half_t, half_t, float> {
  using MMA = MMA_Atom<SM75_16x8x8_F32F16F16F32_TN>;
  using MMA_Group = Layout<Shape<_1, _2, _2>>;
};
#endif

template <int Bits, int N, int K, bool K_inner, typename Enable = void>
struct OperandTraits {
  // Primary template, use padded layout and default copy
  static constexpr int stride = K_inner ? K : N;
  static constexpr int padded = stride % (256 / Bits) == 0 ? stride + 128 / Bits : stride;
  using Layout =
      typename std::conditional<K_inner, Layout<Shape<Int<N>, Int<K>>, Shape<Int<padded>, _1>>,
                                Layout<Shape<Int<N>, Int<K>>, Shape<_1, Int<padded>>>>::type;
  using Copy = DefaultCopy;
};

template <int N, int K>
struct OperandTraits<16, N, K, true, typename std::enable_if<K % 64 == 32>::type> {
  using LayoutAtom =
      decltype(composition(Swizzle<2, 3, 3>{}, Layout<Shape<_8, _32>, Stride<_32, _1>>{}));
  using Layout = decltype(tile_to_shape(LayoutAtom{}, Shape<Int<N>, Int<K>>{}));
  using Copy = SM75_U32x4_LDSM_N;
};

template <int N, int K>
struct OperandTraits<16, N, K, true, typename std::enable_if<K % 64 == 0>::type> {
  using LayoutAtom =
      decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<_8, _64>, Stride<_64, _1>>{}));
  using Layout = decltype(tile_to_shape(LayoutAtom{}, Shape<Int<N>, Int<K>>{}));
  using Copy = SM75_U32x4_LDSM_N;
};

template <int N, int K>
struct OperandTraits<16, N, K, false, typename std::enable_if<N % 64 == 32>::type> {
  using LayoutAtom =
      decltype(composition(Swizzle<2, 3, 3>{}, Layout<Shape<_32, _8>, Stride<_1, _32>>{}));
  using Layout = decltype(tile_to_shape(LayoutAtom{}, Shape<Int<N>, Int<K>>{}, Step<_2, _1>{}));
  using Copy = SM75_U16x8_LDSM_T;
};

template <int N, int K>
struct OperandTraits<16, N, K, false, typename std::enable_if<N % 64 == 0>::type> {
  using LayoutAtom =
      decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<_64, _8>, Stride<_1, _64>>{}));
  using Layout = decltype(tile_to_shape(LayoutAtom{}, Shape<Int<N>, Int<K>>{}, Step<_2, _1>{}));
  using Copy = SM75_U16x8_LDSM_T;
};

template <int N, int K>
struct OperandTraits<32, N, K, true, typename std::enable_if<K % 32 == 0>::type> {
  using LayoutAtom =
      decltype(composition(Swizzle<3, 2, 3>{}, Layout<Shape<_8, _32>, Stride<_32, _1>>{}));
  using Layout = decltype(tile_to_shape(LayoutAtom{}, Shape<Int<N>, Int<K>>{}));
  using Copy = SM75_U32x4_LDSM_N;
};

template <int N, int K>
struct OperandTraits<32, N, K, true, typename std::enable_if<K % 32 == 16>::type> {
  using LayoutAtom =
      decltype(composition(Swizzle<2, 2, 3>{}, Layout<Shape<_8, _16>, Stride<_16, _1>>{}));
  using Layout = decltype(tile_to_shape(LayoutAtom{}, Shape<Int<N>, Int<K>>{}));
  using Copy = SM75_U32x4_LDSM_N;
};

template <int N, int K>
struct OperandTraits<32, N, K, false, typename std::enable_if<N % 32 == 0>::type> {
  using LayoutAtom =
      decltype(composition(Swizzle<3, 2, 3>{}, Layout<Shape<_32, _8>, Stride<_1, _32>>{}));
  using Layout = decltype(tile_to_shape(LayoutAtom{}, Shape<Int<N>, Int<K>>{}, Step<_2, _1>{}));
  using Copy = UniversalCopy<tfloat32_t>;
};

template <int N, int K>
struct OperandTraits<32, N, K, false, typename std::enable_if<N % 32 == 16>::type> {
  using LayoutAtom =
      decltype(composition(Swizzle<2, 2, 3>{}, Layout<Shape<_16, _8>, Stride<_1, _16>>{}));
  using Layout = decltype(tile_to_shape(LayoutAtom{}, Shape<Int<N>, Int<K>>{}, Step<_2, _1>{}));
  using Copy = UniversalCopy<tfloat32_t>;
};

template <int N, int K>
struct OperandTraits<8, N, K, true, typename std::enable_if<K % 128 == 64>::type> {
  using LayoutAtom =
      decltype(composition(Swizzle<2, 4, 3>{}, Layout<Shape<_8, _64>, Stride<_64, _1>>{}));
  using Layout = decltype(tile_to_shape(LayoutAtom{}, Shape<Int<N>, Int<K>>{}));
  using Copy = SM75_U32x4_LDSM_N;
};

template <int N, int K>
struct OperandTraits<8, N, K, true, typename std::enable_if<K % 128 == 0>::type> {
  using LayoutAtom =
      decltype(composition(Swizzle<3, 4, 3>{}, Layout<Shape<_8, _128>, Stride<_128, _1>>{}));
  using Layout = decltype(tile_to_shape(LayoutAtom{}, Shape<Int<N>, Int<K>>{}));
  using Copy = SM75_U32x4_LDSM_N;
};

template <int N, int K>
struct OperandTraits<64, N, K, true, typename std::enable_if<K % 16 == 0>::type> {
  using LayoutAtom =
      decltype(composition(Swizzle<2, 0, 4>{}, Layout<Shape<_4, _16>, Stride<_16, _1>>{}));
  using Layout = decltype(tile_to_shape(LayoutAtom{}, Shape<Int<N>, Int<K>>{}));
  using Copy = DefaultCopy;
};

template <int N, int K>
struct OperandTraits<64, N, K, false, typename std::enable_if<N % 16 == 0>::type> {
  using LayoutAtom =
      decltype(composition(Swizzle<2, 2, 2>{}, Layout<Shape<_16, _4>, Stride<_1, _16>>{}));
  using Layout = decltype(tile_to_shape(LayoutAtom{}, Shape<Int<N>, Int<K>>{}, Step<_2, _1>{}));
  using Copy = DefaultCopy;
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

  using OperandATraits = OperandTraits<sizeof_bits<A_type>::value, M, K, !trans_A>;
  using OperandBTraits = OperandTraits<sizeof_bits<B_type>::value, N, K, trans_B>;
  using SmemLayoutA = typename OperandATraits::Layout;
  using SmemLayoutB = typename OperandBTraits::Layout;
  using SmemCopyA = Copy_Atom<typename OperandATraits::Copy, A_type>;
  using SmemCopyB = Copy_Atom<typename OperandBTraits::Copy, B_type>;

  using TileMma =
      TiledMMA<typename Instruction::MMA, Layout<Shape<Int<num_warp_m>, Int<num_warp_n>, _1>>,
               typename Instruction::MMA_Group>;

  template <class... Args>
  static CUTE_DEVICE auto remove_swizzle(Layout<Args...> const& layout) {
    return layout;
  }
  // In fp16, when layout is KxN and n_warp is 1 and N % 64 == 0
  // the original layout fail to compile, currently using this as a workaround
  template <class... Args>
  static CUTE_DEVICE auto remove_swizzle(ComposedLayout<Args...> const& layout) {
    if constexpr (sizeof(A_type) == 2)
      return layout.layout_b();
    else
      return layout;
  }

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

  static CUTE_DEVICE void body_sr(A_type_raw* pA, B_type_raw* pB, C_type_raw* pC) {
    const int tid = threadIdx.x;
    Tensor sA = make_tensor(make_smem_ptr(reinterpret_cast<A_type*>(pA)), SmemLayoutA{});
    TileMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tid);
    auto tiled_copy_A = make_tiled_copy_A(SmemCopyA{}, tiled_mma);
    auto thr_copy_A = tiled_copy_A.get_thread_slice(tid);

    Tensor tCrA = thr_mma.partition_fragment_A(sA);
    Tensor tCsA = thr_copy_A.partition_S(sA);

    Tensor tCrA_copy_view = thr_copy_A.retile_D(tCrA);

    Tensor acc = make_tensor(make_rmem_ptr(reinterpret_cast<C_type*>(pC)),
                             partition_shape_C(tiled_mma, Shape<Int<M>, Int<N>>{}));
    Tensor tCrB = make_tensor(make_rmem_ptr(reinterpret_cast<B_type*>(pB)),
                              partition_shape_B(tiled_mma, Shape<Int<N>, Int<K>>{}));

    auto tCrA_view = make_tensor(tCrA.data(), remove_swizzle(tCrA.layout()));
    copy(tiled_copy_A, tCsA(_, _, 0), tCrA_copy_view(_, _, 0));
    CUTE_UNROLL
    for (int k = 0; k < size<2>(tCrA); ++k) {
      if (k < size<2>(tCrA) - 1) {
        copy(tiled_copy_A, tCsA(_, _, k + 1), tCrA_copy_view(_, _, k + 1));
      }
      gemm(tiled_mma, tCrA_view(_, _, k), tCrB(_, _, k), acc);
    }
  }
};

}  // namespace cute

namespace tl {

template <int M, int N, int K, int num_warp_m, int num_warp_n, bool trans_A, bool trans_B,
          typename A_type, typename B_type, typename C_type>
CUTLASS_DEVICE void gemm_ss(A_type* pA, B_type* pB, C_type* accum) {
  using MMA =
      cute::GemmTensorOp<M, N, K, num_warp_m, num_warp_n, trans_A, trans_B, A_type, B_type, C_type>;
  MMA::body(pA, pB, accum);
}

template <int M, int N, int K, int num_warp_m, int num_warp_n, bool trans_A, bool trans_B,
          typename A_type, typename B_type, typename C_type>
CUTLASS_DEVICE void gemm_rs(A_type* pA, B_type* pB, C_type* accum) {
  using MMA =
      cute::GemmTensorOp<M, N, K, num_warp_m, num_warp_n, trans_A, trans_B, A_type, B_type, C_type>;
  MMA::body_rs(pA, pB, accum);
}

template <int M, int N, int K, int num_warp_m, int num_warp_n, bool trans_A, bool trans_B,
          typename A_type, typename B_type, typename C_type>
CUTLASS_DEVICE void gemm_sr(A_type* pA, B_type* pB, C_type* accum) {
  using MMA =
      cute::GemmTensorOp<M, N, K, num_warp_m, num_warp_n, trans_A, trans_B, A_type, B_type, C_type>;
  MMA::body_sr(pA, pB, accum);
}

}  // namespace tl
