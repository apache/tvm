#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "common.h"

namespace ck_tile {

template <int M, int N, int K, int num_warp_m, int num_warp_n, bool TransposeA, bool TransposeB, typename A_type_raw,
          typename B_type_raw, typename C_type_raw>
class GemmTensorOp {
 public:
  using A_type = A_type_raw;
  using B_type = B_type_raw;
  using C_type = C_type_raw;
  using AccDataType = float;

  // The Matrix Multiplication goes with Matrix A (M, K), Matrix B (N, K) = Matrix C (M, N).
  using matrix_a_layout = ck_tile::tensor_layout::gemm::RowMajor;
  using matrix_b_layout = ck_tile::tensor_layout::gemm::ColumnMajor;
  using matrix_c_layout = ck_tile::tensor_layout::gemm::RowMajor;
  // This part comes from the Codegen
  static constexpr ck_tile::index_t M_Tile = M;
  static constexpr ck_tile::index_t N_Tile = N;
  static constexpr ck_tile::index_t K_Tile = K;

  static constexpr ck_tile::index_t M_Warp = num_warp_m;
  static constexpr ck_tile::index_t N_Warp = num_warp_n;
  static constexpr ck_tile::index_t K_Warp = 1;

  static constexpr ck_tile::index_t M_Warp_Tile = 16;
  static constexpr ck_tile::index_t N_Warp_Tile = 16;
  static constexpr ck_tile::index_t K_Warp_Tile = 16;

  // The kPadA, kPadB, kPadC & kBlockPerCu should also come from the Codegen part.
  static constexpr bool kPadA = true;
  static constexpr bool kPadB = true;
  static constexpr bool kPadC = true;

  using CodegenGemmShape =
      ck_tile::TileGemmShape<ck_tile::sequence<M_Tile, N_Tile, K_Tile>,
                             ck_tile::sequence<M_Warp, N_Warp, K_Warp>,
                             ck_tile::sequence<M_Warp_Tile, N_Warp_Tile, K_Warp_Tile>>;

  using CodegenGemmTraits = ck_tile::TileGemmTraits<kPadA, kPadB, kPadC, matrix_a_layout,
                                                    matrix_b_layout, matrix_c_layout>;

  using Problem = ck_tile::GemmPipelineProblem<A_type, B_type, AccDataType, CodegenGemmShape,
                                               CodegenGemmTraits>;

  static CK_TILE_DEVICE void body(A_type* pA, B_type* pB, C_type* pC) {
    static constexpr auto I0 = ck_tile::number<0>{};
    static constexpr auto I1 = ck_tile::number<1>{};
    static constexpr auto I2 = ck_tile::number<2>{};

    // Load A and B from Shared memory
    using AccDataType = float;
    using BlockWarps = typename Problem::BlockGemmShape::BlockWarps;
    using WarpTile = typename Problem::BlockGemmShape::WarpTile;
    using WarpGemm =
        ck_tile::WarpGemmMfmaDispatcher<typename Problem::ADataType, typename Problem::BDataType,
                                        AccDataType, WarpTile::at(I0), WarpTile::at(I1),
                                        WarpTile::at(I2), false>;
    using BlockGemmPolicy = ck_tile::BlockGemmASmemBSmemCRegV1CustomPolicy<
        typename Problem::ADataType, typename Problem::BDataType, typename Problem::CDataType,
        BlockWarps, WarpGemm>;
    auto block_gemm = ck_tile::BlockGemmASmemBSmemCRegV1<Problem, BlockGemmPolicy>{};
    // Then do GEMM and store the result to C
    // Create Tile Window

    constexpr index_t kMPerBlock = Problem::BlockGemmShape::kM;
    constexpr index_t kNPerBlock = Problem::BlockGemmShape::kN;
    constexpr index_t kKPerBlock = Problem::BlockGemmShape::kK;

    constexpr auto a_lds_block_desc =
        make_naive_tensor_descriptor_packed(make_tuple(kMPerBlock, kKPerBlock));

    auto a_lds_block = make_tensor_view<address_space_enum::lds>(pA, a_lds_block_desc);

    constexpr auto b_lds_block_desc =
        make_naive_tensor_descriptor_packed(make_tuple(kNPerBlock, kKPerBlock));

    auto b_lds_block = make_tensor_view<address_space_enum::lds>(pB, b_lds_block_desc);

    // A LDS tile for block SMEM
    auto a_lds_gemm_window = make_tile_window(
        a_lds_block, make_tuple(number<kMPerBlock>{}, number<kKPerBlock>{}), {0, 0});

    // B LDS tile for block SMEM
    auto b_lds_gemm_window = make_tile_window(
        b_lds_block, make_tuple(number<kNPerBlock>{}, number<kKPerBlock>{}), {0, 0});

    // we should use MakeCBlockTile() to link the C block tensor with ptrC
    // Acc register tile
    auto c_block_tile = decltype(block_gemm(a_lds_gemm_window, b_lds_gemm_window)){};
    using CVec = ext_vector_t<typename Problem::CDataType, c_block_tile.get_thread_buffer_size()>;
    auto c_vec = *c_style_pointer_cast<const CVec*>(pC);
    c_block_tile.get_thread_buffer().template set_as(
                number<0>{}, c_vec);
    // if(threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
    //     printf("============c_vec========\n");
    //     printf("pC[0]: %f\n", pC[0]);
    //     pC[0] = 1.0;
    //     printf("pC[0]: %f\n", pC[0]);
    //     printf("c_block_tile[0]: %f\n", (float)c_block_tile.get_thread_buffer()[0]);
    //     printf("c_block_tile.get_thread_buffer_size(): %d\n", c_block_tile.get_thread_buffer_size());
    //     c_block_tile.get_thread_buffer().template set_as(
    //             number<0>{}, c_vec);
    //     printf("c_vec[0]: %f\n", (float)c_vec[0]);
    //     printf("c_block_tile[0]: %f\n", (float)c_block_tile.get_thread_buffer()[0]);
    //     c_block_tile.get_thread_buffer()[0] = 2.0;
    //     printf("c_block_tile[0]: %f\n", (float)c_block_tile.get_thread_buffer()[0]);
    //     printf("c_vec[0]: %f\n", (float)c_vec[0]);
    //     // c_block_tile.get_thread_buffer().data_ptr = pC;
    //     printf("c_block_tile[0]: %f\n", (float)c_block_tile.get_thread_buffer()[0]);
    //     printf("c_block_tile[0].data: %f\n", (float)c_block_tile.get_thread_buffer().data[0]);
    //     // printf("c_block_tile[0].data.data: %d\n", &(c_block_tile.get_thread_buffer().data[0]));
    // }
    // // c_block_tile.get_thread_buffer().

    block_gemm(c_block_tile, a_lds_gemm_window, b_lds_gemm_window);
   
    c_vec = c_block_tile.get_thread_buffer().template get_as<CVec>(number<0>{});
   
    for(int j = 0; j < c_block_tile.get_thread_buffer_size(); j++) {
        pC[j] = c_block_tile.get_thread_buffer()[j];
    }
    
    // print c
    if(threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        printf("============pC========\n");
        for(int j = 0; j < c_block_tile.get_thread_buffer_size(); j++) {
            printf("%d: %f \n", j, pC[j]);
        }
    }
  }
};

}  // namespace ck_tile

namespace tl {

template <int M, int N, int K, int num_warp_m, int num_warp_n, bool trans_A, bool trans_B,
          typename A_type, typename B_type, typename C_type>
CK_TILE_DEVICE void gemm_ss(A_type* pA, B_type* pB, C_type* accum) {
  using Compute = ck_tile::GemmTensorOp<M, N, K, num_warp_m, num_warp_n, trans_A, trans_B, A_type, B_type, C_type>;
  Compute::body(pA, pB, accum);
}

}  // namespace tl