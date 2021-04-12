#include <mma.h>

#ifdef _WIN32
  using uint = unsigned int;
  using uchar = unsigned char;
  using ushort = unsigned short;
  using int64_t = long long;
  using uint64_t = unsigned long long;
#else
  #define uint unsigned int
  #define uchar unsigned char
  #define ushort unsigned short
  #define int64_t long long
  #define uint64_t unsigned long long
#endif
extern "C" __global__ void default_function_kernel0(int* __restrict__ placeholder, int* __restrict__ packed_kernel, int* __restrict__ Conv) {
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 8, 8, 32, int> Conv_wmma_accumulator[16];
  __shared__ int pad_data_shared[384];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 8, 8, 32, nvcuda::wmma::experimental::precision::s4, nvcuda::wmma::row_major> pad_data_shared_wmma_matrix_a[4];
  __shared__ int packed_kernel_shared[512];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 8, 8, 32, nvcuda::wmma::experimental::precision::s4, nvcuda::wmma::col_major> packed_kernel_shared_wmma_matrix_b[16];
  __shared__ int Conv_wmma_accumulator_shared[1024];
  for (int h_inner_w_fused_inner = 0; h_inner_w_fused_inner < 32; ++h_inner_w_fused_inner) {
    #pragma unroll
    for (int n_c_init = 0; n_c_init < 2; ++n_c_init) {
      #pragma unroll
      for (int o_c_init = 0; o_c_init < 8; ++o_c_init) {
        (void)nvcuda::wmma::fill_fragment(Conv_wmma_accumulator[((n_c_init * 8) + o_c_init)], 0.000000e+00f);
      }
    }
    for (int kh = 0; kh < 3; ++kh) {
      for (int ic_outer = 0; ic_outer < 4; ++ic_outer) {
        #pragma unroll
        for (int ax1 = 0; ax1 < 3; ++ax1) {
          #pragma unroll
          for (int ax2_inner_inner = 0; ax2_inner_inner < 2; ++ax2_inner_inner) {
            #pragma unroll
            for (int ax3 = 0; ax3 < 2; ++ax3) {
              if ((((((((int)blockIdx.z) * 32) + h_inner_w_fused_inner) / 7) * 2) + kh) < 16) {
                ((int*)(pad_data_shared + (((((ax1 * 1024) + (ax2_inner_inner * 512)) + (ax3 * 256)) + (((int)threadIdx.x) * 8))) / 8))[0] = ((((1 <= (((((((int)blockIdx.z) * 32) + h_inner_w_fused_inner) / 7) * 2) + kh)) && ((((((((int)blockIdx.z) * 32) + h_inner_w_fused_inner) / 7) * 2) + kh) < 15)) && (1 <= (((((((int)blockIdx.z) * 32) + h_inner_w_fused_inner) % 7) * 2) + ax1))) ? ((int*)(placeholder + (((((((((((((((((int)blockIdx.z) * 32) + h_inner_w_fused_inner) / 7) * 229376) + (kh * 114688)) + ((((((int)blockIdx.z) * 32) + h_inner_w_fused_inner) % 7) * 16384)) + (ax1 * 8192)) + (((int)blockIdx.x) * 4096)) + (ax2_inner_inner * 2048)) + ((((int)threadIdx.x) >> 2) * 256)) + (ic_outer * 64)) + (ax3 * 32)) + ((((int)threadIdx.x) & 3) * 8)) - 122880)) / 8))[0] : make_int((int)0, (int)0, (int)0, (int)0, (int)0, (int)0, (int)0, (int)0));
              }
            }
          }
        }
        __syncthreads();
        #pragma unroll
        for (int kw = 0; kw < 3; ++kw) {
          #pragma unroll
          for (int ax2 = 0; ax2 < 2; ++ax2) {
            #pragma unroll
            for (int ax31 = 0; ax31 < 2; ++ax31) {
              if ((((((((int)blockIdx.z) * 32) + h_inner_w_fused_inner) / 7) * 2) + kh) < 16) {
                (void)nvcuda::wmma::load_matrix_sync(pad_data_shared_wmma_matrix_a[((ax2 * 2) + ax31)], ((int *)pad_data_shared + ((((kw * 1024) + (ax2 * 512)) + (ax31 * 256)) / 8)), 32);
              }
            }
          }
          __syncthreads();
          #pragma unroll
          for (int ax21 = 0; ax21 < 8; ++ax21) {
            #pragma unroll
            for (int ax3_inner_inner = 0; ax3_inner_inner < 2; ++ax3_inner_inner) {
              #pragma unroll
              for (int ax4_ax5_fused_inner_inner = 0; ax4_ax5_fused_inner_inner < 8; ++ax4_ax5_fused_inner_inner) {
                packed_kernel_shared[(((((ax21 * 512) + (ax3_inner_inner * 256)) + (((int)threadIdx.x) * 8)) + ax4_ax5_fused_inner_inner)) / 8] = packed_kernel[(((((((((kh * 393216) + (kw * 131072)) + (((int)blockIdx.y) * 16384)) + (ax21 * 2048)) + (ic_outer * 512)) + (ax3_inner_inner * 256)) + (((int)threadIdx.x) * 8)) + ax4_ax5_fused_inner_inner)) / 8];
              }
            }
          }
          __syncthreads();
          #pragma unroll
          for (int ax22 = 0; ax22 < 8; ++ax22) {
            #pragma unroll
            for (int ax32 = 0; ax32 < 2; ++ax32) {
              (void)nvcuda::wmma::load_matrix_sync(packed_kernel_shared_wmma_matrix_b[((ax22 * 2) + ax32)], ((int *)packed_kernel_shared + (((ax22 * 512) + (ax32 * 256)) / 8)), 32);
            }
          }
          #pragma unroll
          for (int ic_inner = 0; ic_inner < 2; ++ic_inner) {
            #pragma unroll
            for (int n_c = 0; n_c < 2; ++n_c) {
              #pragma unroll
              for (int o_c = 0; o_c < 8; ++o_c) {
                if (((((int)blockIdx.z) * 32) + h_inner_w_fused_inner) < 49) {
                  (void)nvcuda::wmma::mma_sync(Conv_wmma_accumulator[((n_c * 8) + o_c)], pad_data_shared_wmma_matrix_a[((n_c * 2) + ic_inner)], packed_kernel_shared_wmma_matrix_b[((o_c * 2) + ic_inner)], Conv_wmma_accumulator[((n_c * 8) + o_c)]);
                }
              }
            }
          }
        }
      }
    }
    #pragma unroll
    for (int ax2_inner = 0; ax2_inner < 2; ++ax2_inner) {
      #pragma unroll
      for (int ax3_inner = 0; ax3_inner < 8; ++ax3_inner) {
        if (((((int)blockIdx.z) * 32) + h_inner_w_fused_inner) < 49) {
          (void)nvcuda::wmma::store_matrix_sync(((int *)Conv_wmma_accumulator_shared + (((ax2_inner * 512) + (ax3_inner * 64)))), Conv_wmma_accumulator[((ax2_inner * 8) + ax3_inner)], 8, nvcuda::wmma::mem_row_major);
        }
      }
    }
    __syncthreads();
    #pragma unroll
    for (int n_inner = 0; n_inner < 2; ++n_inner) {
      #pragma unroll
      for (int o_inner = 0; o_inner < 8; ++o_inner) {
        #pragma unroll
        for (int nn_oo_fused_outer = 0; nn_oo_fused_outer < 2; ++nn_oo_fused_outer) {
          if (((((int)blockIdx.z) * 32) + h_inner_w_fused_inner) < 49) {
            Conv[(((((((((((int)blockIdx.z) * 524288) + (h_inner_w_fused_inner * 16384)) + (((int)blockIdx.x) * 8192)) + (n_inner * 4096)) + (((int)blockIdx.y) * 512)) + (o_inner * 64)) + (nn_oo_fused_outer * 32)) + ((int)threadIdx.x)))] = Conv_wmma_accumulator_shared[(((((n_inner * 512) + (o_inner * 64)) + (nn_oo_fused_outer * 32)) + ((int)threadIdx.x)))];
          }
        }
      }
    }
  }
}

