#pragma once

#include <cutlass/cutlass.h>
#include <cutlass/arch/barrier.h>
#include <cute/algorithm/copy.hpp>

#include "common.h"

namespace cute {

template <GMMA::Major major, class ElementType, class BLK_MN, class BLK_K>
CUTE_HOST_DEVICE constexpr auto ss_smem_selector() {
  auto BLK_MN0 = size<0>(BLK_MN{});
  auto BLK_K0 = size<0>(BLK_K{});

  static_assert(BLK_MN0 % 8 == 0, "BLK_MN0 must be a multiple of 8.");
  static_assert(BLK_K0 % 8 == 0, "BLK_K0 must be a multiple of 8.");

  if constexpr (major == GMMA::Major::MN) {
    if constexpr (BLK_MN0 % size<0>(GMMA::Layout_MN_SW128_Atom<ElementType>{}) == 0) {
      return GMMA::Layout_MN_SW128_Atom<ElementType>{};
    } else if constexpr (BLK_MN0 % size<0>(GMMA::Layout_MN_SW64_Atom<ElementType>{}) == 0) {
      return GMMA::Layout_MN_SW64_Atom<ElementType>{};
    } else if constexpr (BLK_MN0 % size<0>(GMMA::Layout_MN_SW32_Atom<ElementType>{}) == 0) {
      return GMMA::Layout_MN_SW32_Atom<ElementType>{};
    } else if constexpr (BLK_MN0 % size<0>(GMMA::Layout_MN_INTER_Atom<ElementType>{}) == 0) {
      return GMMA::Layout_MN_INTER_Atom<ElementType>{};
    } else {
      static_assert(
          BLK_MN0 % size<0>(GMMA::Layout_MN_INTER_Atom<ElementType>{}) == 0,
          "BLK_MN0 must be a multiple of size<0>(GMMA::Layout_MN_INTER_Atom<ElementType>{})");
    }
  } else if constexpr (major == GMMA::Major::K) {
    if constexpr (BLK_K0 % size<1>(GMMA::Layout_K_SW128_Atom<ElementType>{}) == 0) {
      return GMMA::Layout_K_SW128_Atom<ElementType>{};
    } else if constexpr (BLK_K0 % size<1>(GMMA::Layout_K_SW64_Atom<ElementType>{}) == 0) {
      return GMMA::Layout_K_SW64_Atom<ElementType>{};
    } else if constexpr (BLK_K0 % size<1>(GMMA::Layout_K_SW32_Atom<ElementType>{}) == 0) {
      return GMMA::Layout_K_SW32_Atom<ElementType>{};
    } else if constexpr (BLK_K0 % size<1>(GMMA::Layout_K_INTER_Atom<ElementType>{}) == 0) {
      return GMMA::Layout_K_INTER_Atom<ElementType>{};
    } else {
      static_assert(
          BLK_K0 % size<1>(GMMA::Layout_K_INTER_Atom<ElementType>{}) == 0,
          "BLK_K0 must be a multiple of size<1>(GMMA::Layout_K_INTER_Atom<ElementType>{})");
    }
  }
}

template <int M, int N, int K, int num_warp_m, int num_warp_n, bool trans_A, bool trans_B,
          typename A_type_raw, typename B_type_raw, typename C_type_raw>
class GemmTensorOp {
 public:
  using A_type = conditional_t<std::is_same<A_type_raw, float>::value, tfloat32_t, A_type_raw>;
  using B_type = conditional_t<std::is_same<B_type_raw, float>::value, tfloat32_t, B_type_raw>;
  using C_type = C_type_raw;

  static constexpr GMMA::Major GmmaMajorA = trans_A ? GMMA::Major::MN : GMMA::Major::K;
  static constexpr GMMA::Major GmmaMajorB = trans_B ? GMMA::Major::K : GMMA::Major::MN;

  using SmemLayoutAtomA = decltype(ss_smem_selector<GmmaMajorA, A_type, Int<M>, Int<K>>());
  using SmemLayoutAtomB = decltype(ss_smem_selector<GmmaMajorB, B_type, Int<N>, Int<K>>());

  using SmemLayoutA = decltype(tile_to_shape(SmemLayoutAtomA{}, Shape<Int<M>, Int<K>>{},
                                             conditional_t<trans_A, Step<_2, _1>, Step<_1, _2>>{}));
  using SmemLayoutB = decltype(tile_to_shape(SmemLayoutAtomB{}, Shape<Int<N>, Int<K>>{},
                                             conditional_t<trans_B, Step<_1, _2>, Step<_2, _1>>{}));

  // static_assert(num_warp_n == 1);
  static_assert(num_warp_m % 4 == 0);

  template <int wg_wait=0>
  static CUTE_DEVICE void body(A_type_raw* pA, B_type_raw* pB, C_type_raw* pC) {
    const int tid = threadIdx.x;
    Tensor sA = make_tensor(make_smem_ptr(reinterpret_cast<A_type*>(pA)), SmemLayoutA{});
    Tensor sB = make_tensor(make_smem_ptr(reinterpret_cast<B_type*>(pB)), SmemLayoutB{});
    auto tiled_mma =
        make_tiled_mma(GMMA::ss_op_selector<A_type, B_type, C_type, Shape<Int<M>, Int<N / num_warp_n>, Int<K>>,
                                            GmmaMajorA, GmmaMajorB>(),
                       Layout<Shape<Int<num_warp_m / 4>, Int<num_warp_n>, _1>>{});
    auto thr_mma = tiled_mma.get_thread_slice(tid);

    // Allocate registers for pipelining
    Tensor tCsA = thr_mma.partition_A(sA);  // (MMA,MMA_M,MMA_K,PIPE)
    Tensor tCsB = thr_mma.partition_B(sB);  // (MMA,MMA_N,MMA_K,PIPE)

    Tensor tCrA = thr_mma.make_fragment_A(tCsA);  // (MMA,MMA_N,MMA_K,PIPE)
    Tensor tCrB = thr_mma.make_fragment_B(tCsB);  // (MMA,MMA_M,MMA_N,PIPE)

    Tensor acc = make_tensor(make_rmem_ptr(reinterpret_cast<C_type*>(pC)),
                             partition_shape_C(tiled_mma, Shape<Int<M>, Int<N>>{}));

    warpgroup_fence_operand(acc);
    warpgroup_arrive();
    CUTLASS_PRAGMA_UNROLL
    for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
      // warpgroup_arrive();
      // (V,M) x (V,N) => (V,M,N)
      gemm(tiled_mma, tCrA(_, _, k_block), tCrB(_, _, k_block), acc);
      tiled_mma.accumulate_ = GMMA::ScaleOut::One;
    }

    warpgroup_commit_batch();
    if constexpr (wg_wait >= 0) { warpgroup_wait<wg_wait>(); }
    warpgroup_fence_operand(acc);
    // warpgroup_fence_operand(acc);
    // warpgroup_arrive();

    // gemm(tiled_mma, tCrA(_, _, _), tCrB(_, _, _), acc);

    // warpgroup_commit_batch();
    // if constexpr (wg_wait >= 0) { warpgroup_wait<wg_wait>(); }
    // warpgroup_fence_operand(acc);
  }

  template <int wg_wait=0>
  static CUTE_DEVICE void body_rs(A_type_raw* pA, B_type_raw* pB, C_type_raw* pC) {
    // TODO: Move bar.sync out of body_rs
    // asm volatile("bar.sync %0, %1;" : : "r"(1), "r"(num_warp_m * num_warp_n * 32));
    const int tid = threadIdx.x;
    Tensor sB = make_tensor(make_smem_ptr(reinterpret_cast<B_type*>(pB)), SmemLayoutB{});
    auto tiled_mma =
        make_tiled_mma(GMMA::rs_op_selector<A_type, B_type, C_type, Shape<Int<M>, Int<N / num_warp_n>, Int<K>>,
                                            GmmaMajorA, GmmaMajorB>(),
                       Layout<Shape<Int<num_warp_m / 4>, Int<num_warp_n>, _1>>{});
    auto thr_mma = tiled_mma.get_thread_slice(tid);

    // Allocate registers for pipelining
    Tensor tCsB = thr_mma.partition_B(sB);        // (MMA,MMA_N,MMA_K,PIPE)
    Tensor tCrB = thr_mma.make_fragment_B(tCsB);  // (MMA,MMA_M,MMA_N,PIPE)
    Tensor tCrA = make_tensor(make_rmem_ptr(reinterpret_cast<A_type*>(pA)),
                              partition_shape_A(tiled_mma, Shape<Int<M>, Int<K>>{}));
    Tensor acc = make_tensor(make_rmem_ptr(reinterpret_cast<C_type*>(pC)),
                             partition_shape_C(tiled_mma, Shape<Int<M>, Int<N>>{}));

    warpgroup_fence_operand(tCrA);
    warpgroup_fence_operand(acc);
    warpgroup_arrive();
    CUTLASS_PRAGMA_UNROLL
    for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
      // warpgroup_arrive();
      // (V,M) x (V,N) => (V,M,N)
      gemm(tiled_mma, tCrA(_, _, k_block), tCrB(_, _, k_block), acc);
      tiled_mma.accumulate_ = GMMA::ScaleOut::One;
    }
    warpgroup_commit_batch();
    if constexpr (wg_wait >= 0) { warpgroup_wait<wg_wait>(); }
    warpgroup_fence_operand(acc);
    warpgroup_fence_operand(tCrA);

    // warpgroup_fence_operand(acc);
    // warpgroup_arrive();

    // gemm(tiled_mma, tCrA(_, _, _), tCrB(_, _, _), acc);

    // warpgroup_commit_batch();
    
    // if constexpr (wg_wait >= 0) { warpgroup_wait<wg_wait>(); }
    // warpgroup_fence_operand(acc);
  }
};

}  // namespace cute

namespace tl {

template <int M, int N, int K, int num_warp_m, int num_warp_n, bool trans_A, bool trans_B, int wg_wait=0,
          typename A_type, typename B_type, typename C_type>
TL_DEVICE void gemm_ss(A_type* pA, B_type* pB, C_type* accum) {
  using MMA =
      cute::GemmTensorOp<M, N, K, num_warp_m, num_warp_n, trans_A, trans_B, A_type, B_type, C_type>;
  MMA::body<wg_wait>(pA, pB, accum);
}

template <int M, int N, int K, int num_warp_m, int num_warp_n, bool trans_A, bool trans_B, int wg_wait=0,
          typename A_type, typename B_type, typename C_type>
TL_DEVICE void gemm_rs(A_type* pA, B_type* pB, C_type* accum) {
  using MMA =
      cute::GemmTensorOp<M, N, K, num_warp_m, num_warp_n, trans_A, trans_B, A_type, B_type, C_type>;
  MMA::body_rs<wg_wait>(pA, pB, accum);
}

template <int num_mma>
TL_DEVICE void wait_wgmma() {
  warpgroup_wait<num_mma>();
}

template <int NumMmaThreads>
TL_DEVICE void warp_scheduler_barrier_sync() {
  cutlass::arch::NamedBarrier::sync(
    NumMmaThreads, 
    cutlass::canonical_warp_group_idx() /*id*/);
}

template <int NumMmaThreads>
TL_DEVICE void warp_scheduler_barrier_arrive() {
  static_assert(NumMmaThreads == 256 || NumMmaThreads == 384);
  if constexpr (NumMmaThreads == 256) {
    cutlass::arch::NamedBarrier::arrive(NumMmaThreads, (1 - cutlass::canonical_warp_group_idx()) /*id*/);
  } else {
    cutlass::arch::NamedBarrier::arrive(NumMmaThreads, (cutlass::canonical_warp_group_idx() <= 1 ? cutlass::canonical_warp_group_idx() + 1 : cutlass::canonical_warp_group_idx() + 1 - 3)  /*id*/);
    cutlass::arch::NamedBarrier::arrive(NumMmaThreads, (cutlass::canonical_warp_group_idx() <= 0 ? cutlass::canonical_warp_group_idx() + 2 : cutlass::canonical_warp_group_idx() + 2 - 3)  /*id*/);
  }
}

template <int NumMmaThreads>
TL_DEVICE void mma_init() {
  static_assert(NumMmaThreads == 256 || NumMmaThreads == 384);
  if (cutlass::canonical_warp_group_idx() > 0) {
    cutlass::arch::NamedBarrier::arrive(NumMmaThreads, 0);
  }
  if constexpr (NumMmaThreads == 384) {
    if (cutlass::canonical_warp_group_idx() > 1) {
      cutlass::arch::NamedBarrier::arrive(NumMmaThreads, 1 /*id*/);
    }
  }
}
}  // namespace tl
