#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "common.h"

namespace ck_tile {

template <int M, int N, int K, int num_warp_m, int num_warp_n, bool TransposeA, bool TransposeB,
          typename A_type, typename B_type, typename C_type, typename AccDataType = float>
class GemmTensorOp {
 public:
  static constexpr int micro_size_x = 16;
  static constexpr int micro_size_y = 16;
  static constexpr int micro_size_k = 16;

  // This part comes from the Codegen
  static constexpr int M_Tile = M;
  static constexpr int N_Tile = N;
  static constexpr int K_Tile = K;

  static constexpr int block_row_warps = num_warp_m;
  static constexpr int block_col_warps = num_warp_n;

  static constexpr int inner_k = K_Tile / micro_size_k;
  static constexpr int warp_rows = M_Tile / (block_row_warps * micro_size_x);
  static constexpr int warp_cols = N_Tile / (block_col_warps * micro_size_y);

  // The kPadA, kPadB, kPadC & kBlockPerCu should also come from the Codegen part.
  static constexpr bool kPadA = true;
  static constexpr bool kPadB = true;
  static constexpr bool kPadC = true;

  static constexpr int warp_size = 64;

  CK_TILE_DEVICE static constexpr auto reverse_index_map(int thread_id, int local_id) {
    return std::make_pair(thread_id % 16, (thread_id / 16) * 4 + local_id);
  }

  CK_TILE_DEVICE static constexpr auto reverse_index_map_transposed(int thread_id, int local_id) {
    return std::make_pair((thread_id / 16) * 4 + local_id, thread_id % 16);
  }

  static CK_TILE_DEVICE void body(A_type* A_shared, B_type* B_shared, C_type* C_local) {
    auto tid = threadIdx.x;
    auto warp_id = tid / warp_size;
    auto warp_m = warp_id / block_col_warps;
    auto warp_n = warp_id % block_col_warps;
    auto warp_row_tiles = warp_rows * micro_size_x;
    auto warp_col_tiles = warp_cols * micro_size_y;

    auto lane_id = tid % warp_size;
    auto tz = warp_m;
    auto ty = warp_n;
    auto tx = lane_id;

    constexpr auto local_size_a = (micro_size_x * micro_size_k) / warp_size;
    constexpr auto local_size_b = (micro_size_y * micro_size_k) / warp_size;
    constexpr auto local_size_c = (micro_size_x * micro_size_y) / warp_size;

    A_type A_local[warp_rows * local_size_a];
    B_type B_local[warp_cols * local_size_b];

    // if (tid == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
    //     printf("allocating %d for A_local\n", warp_rows * local_size_a);
    //     printf("allocating %d for B_local\n", warp_cols * local_size_b);
    //     printf("warp_rows: %d, warp_cols: %d\n", warp_rows, warp_cols);
    //     printf("local_size_a: %d, local_size_b: %d\n", local_size_a, local_size_b);
    // }

    for (int ki = 0; ki < inner_k; ki++) {
      // Fetch A into register
      for (int i = 0; i < warp_rows; i++) {
        for (int local_id = 0; local_id < local_size_a; local_id++) {
          const auto l = tz * warp_row_tiles + i * micro_size_x;
          const auto r = ki * micro_size_k;
          auto [row, col] = reverse_index_map(lane_id, local_id);
          A_local[i * local_size_a + local_id] = A_shared[(l + row) * K_Tile + r + col];
        }
      }

      // Fetch B into register
      for (int j = 0; j < warp_cols; j++) {
        for (int local_id = 0; local_id < local_size_b; local_id++) {
          const auto l = ty * warp_col_tiles + j * micro_size_y;
          const auto r = ki * micro_size_k;
          auto [row, col] = reverse_index_map(lane_id, local_id);
          B_local[j * local_size_b + local_id] = B_shared[(l + row) * K_Tile + r + col];
        }
      }

      // Compute
      for (int i = 0; i < warp_rows; ++i) {
        for (int j = 0; j < warp_cols; ++j) {
          *(((float32x4*)C_local) + ((i * warp_cols) + j)) = __builtin_amdgcn_mfma_f32_16x16x16f16(
              *(((float16x4*)A_local) + i), *(((float16x4*)B_local) + j),
              *(((float32x4*)C_local) + ((i * warp_cols) + j)), 0, 0, 0);
        }
      }
    }
  }
};

}  // namespace ck_tile

namespace tl {

template <int M, int N, int K, int num_warp_m, int num_warp_n, bool trans_A, bool trans_B,
          typename A_type, typename B_type, typename C_type>
CK_TILE_DEVICE void gemm_ss(A_type* pA, B_type* pB, C_type* accum) {
  using Compute = ck_tile::GemmTensorOp<M, N, K, num_warp_m, num_warp_n, trans_A, trans_B, A_type,
                                        B_type, C_type>;
  Compute::body(pA, pB, accum);
}

}  // namespace tl