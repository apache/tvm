#pragma once

#include "common.h"

#include <cute/algorithm/copy.hpp>

using namespace cute;

#include <cutlass/gemm/collective/builders/sm90_common.inl>

template <int M, int N, int K, int num_warp_m, int num_warp_n, bool trans_A, bool trans_B,
          typename A_type_raw, typename B_type_raw, typename C_type_raw>
class GemmTensorOp {
 public:
  using A_type = conditional_t<std::is_same<A_type_raw, float>::value, tfloat32_t,
                                           A_type_raw>;
  using B_type = conditional_t<std::is_same<B_type_raw, float>::value, tfloat32_t,
                                           A_type_raw>;
  using C_type = C_type_raw;

  static constexpr GMMA::Major GmmaMajorA = trans_A ? GMMA::Major::MN : GMMA::Major::K;
  static constexpr GMMA::Major GmmaMajorB = trans_B ? GMMA::Major::K : GMMA::Major::MN;

  using SmemLayoutAtomA = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GmmaMajorA, A_type, Int<M>, Int<K>>());
  using SmemLayoutAtomB = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GmmaMajorB, B_type, Int<N>, Int<K>>());

  using SmemLayoutA = decltype(tile_to_shape(SmemLayoutAtomA{}, Shape<Int<M>, Int<K>>{}, conditional_t<trans_A, Step<_2, _1>, Step<_1, _2>>{}));
  using SmemLayoutB = decltype(tile_to_shape(SmemLayoutAtomB{}, Shape<Int<N>, Int<K>>{}, conditional_t<trans_B, Step<_1, _2>, Step<_2, _1>>{}));

  static_assert(num_warp_n == 1);
  static_assert(num_warp_m % 4 == 0);

  using TileMma = decltype(make_tiled_mma(GMMA::ss_op_selector<
      A_type, B_type, C_type, Shape<Int<M>, Int<N>, Int<K>>, GmmaMajorA, GmmaMajorB>(), Layout<Shape<Int<num_warp_m/4>,_1,_1>>{}));

  static CUTE_DEVICE void body(A_type_raw* pA, B_type_raw* pB, C_type_raw* pC) {
    const int tid = threadIdx.x;
    Tensor sA = make_tensor(make_smem_ptr(reinterpret_cast<A_type*>(pA)), SmemLayoutA{});
    Tensor sB = make_tensor(make_smem_ptr(reinterpret_cast<B_type*>(pB)), SmemLayoutB{});
    TileMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tid);

    // Allocate registers for pipelining
    Tensor tCsA = thr_mma.partition_A(sA);                                     // (MMA,MMA_M,MMA_K,PIPE)
    Tensor tCsB = thr_mma.partition_B(sB);                                     // (MMA,MMA_N,MMA_K,PIPE)

    Tensor tCrA = thr_mma.make_fragment_A(tCsA);                               // (MMA,MMA_N,MMA_K,PIPE)
    Tensor tCrB = thr_mma.make_fragment_B(tCsB);                               // (MMA,MMA_M,MMA_N,PIPE)

    Tensor acc = make_tensor(make_rmem_ptr(reinterpret_cast<C_type*>(pC)),
                             partition_shape_C(tiled_mma, Shape<Int<M>, Int<N>>{}));

    warpgroup_fence_operand(acc);
    warpgroup_arrive();

    gemm(tiled_mma, tCrA(_, _, _), tCrB(_, _, _), acc);
    
    warpgroup_commit_batch();
    warpgroup_wait<0>();
    warpgroup_fence_operand(acc);

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

}  // namespace tl
