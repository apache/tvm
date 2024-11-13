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

  static constexpr int BANK_SIZE_BYTES = 128;

  static constexpr int warp_size = 64;

  TL_DEVICE static constexpr auto reverse_index_map(int thread_id, int local_id) {
    return std::make_pair(thread_id % 16, (thread_id / 16) * 4 + local_id);
  }

  TL_DEVICE static constexpr auto reverse_index_map_transposed(int thread_id, int local_id) {
    return std::make_pair((thread_id / 16) * 4 + local_id, thread_id % 16);
  }

  /*
   * Detailed Implementation please
   * checkout bitblas/tl/utils.py:get_swizzle_layout
   */
  template <int continuous = 32, int element_size = 2>
  TL_DEVICE static auto make_full_bank_swizzle_layout(const int row, const int col) {
    const auto dtype_bits = element_size * 8;
    const auto bank_elems = BANK_SIZE_BYTES / dtype_bits;

    const auto col_idx_outer = col / bank_elems;
    const auto col_idx_inner = col % bank_elems;
    const auto row_idx_sub = row % bank_elems;
    const auto new_col_idx_outer = col_idx_outer ^ row_idx_sub;

    return std::make_pair(row, new_col_idx_outer * bank_elems + col_idx_inner);
  }

  template <int continuous = 32, int element_size = 2>
  TL_DEVICE static auto make_half_bank_swizzle_layout(const int row, const int col) {
    const auto dtype_bits = element_size * 8;
    const auto bank_elems = BANK_SIZE_BYTES / dtype_bits;

    const auto col_idx_outer = col / bank_elems;
    const auto col_idx_inner = col % bank_elems;

    const auto row_idx_sub = row % bank_elems;
    const auto interleave_elems = 32 / dtype_bits;
    const auto new_col_idx_outer = col_idx_outer ^ (row_idx_sub / interleave_elems);

    return std::make_pair(row, new_col_idx_outer * bank_elems + col_idx_inner);
  }

  template <int continuous = 32, int element_size = 2>
  TL_DEVICE static constexpr auto make_layout_padded(const int row, const int col) {
    return std::make_pair(row, col);
  }

  template <int continuous = 32, int element_size = 2>
  TL_DEVICE static constexpr auto make_swizzle_layout(const int row, const int col) {
    constexpr auto vector_size = BANK_SIZE_BYTES / (element_size * 8);

    if (continuous % (vector_size * 8) == 0) {
      auto [n_row, n_col] = make_full_bank_swizzle_layout<continuous, element_size>(row, col);
      return n_row * continuous + n_col;
    } else if (continuous % (vector_size * 4) == 0) {
      auto [n_row, n_col] = make_half_bank_swizzle_layout<continuous, element_size>(row, col);
      return n_row * continuous + n_col;
    }
    else {
      auto [n_row, n_col] = make_layout_padded(row, col);
      return n_row * (continuous + continuous / 4) + n_col;
    }
  }

  static TL_DEVICE void body(A_type* A_shared, B_type* B_shared, C_type* C_local) {
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

    constexpr auto last_dim_a = TransposeA ? M_Tile : K_Tile;
    constexpr auto last_dim_b = TransposeB ? K_Tile : N_Tile;

    A_type A_local[warp_rows * local_size_a];
    B_type B_local[warp_cols * local_size_b];

    for (int ki = 0; ki < inner_k; ki++) {
      // Fetch A into register
      for (int i = 0; i < warp_rows; i++) {
        const auto l = tz * warp_row_tiles + i * micro_size_x;
        const auto r = ki * micro_size_k;
        for (int local_id = 0; local_id < local_size_a; local_id++) {
          auto [row, col] = reverse_index_map(lane_id, local_id);
          A_local[i * local_size_a + local_id] = A_shared[make_swizzle_layout<last_dim_a, sizeof(A_type)>(l + row, r + col)];
        }
      }

      // Fetch B into register
      for (int j = 0; j < warp_cols; j++) {
        const auto l = ty * warp_col_tiles + j * micro_size_y;
        const auto r = ki * micro_size_k;
        for (int local_id = 0; local_id < local_size_b; local_id++) {
          auto [row, col] = reverse_index_map(lane_id, local_id);
          B_local[j * local_size_b + local_id] = B_shared[make_swizzle_layout<last_dim_b, sizeof(B_type)>(l + row, r + col)];
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
TL_DEVICE void gemm_ss(A_type* pA, B_type* pB, C_type* accum) {
  using Compute = ck_tile::GemmTensorOp<M, N, K, num_warp_m, num_warp_n, trans_A, trans_B, A_type,
                                        B_type, C_type>;
  Compute::body(pA, pB, accum);
}

}  // namespace tl
