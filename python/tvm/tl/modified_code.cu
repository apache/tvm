#include <tl_templates/gemm.h>
#include <tl_templates/copy.h>
#include <tl_templates/reduce.h>
#include <tl_templates/ldsm.h>
#include <tl_templates/threadblock_swizzle.h>

extern "C" __global__ void __launch_bounds__(384) main_kernel(__grid_constant__ const CUtensorMap K_desc, __grid_constant__ const CUtensorMap Output_desc, __grid_constant__ const CUtensorMap Q_desc, __grid_constant__ const CUtensorMap V_desc, __grid_constant__ const CUtensorMap mask_desc) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  float r[2];
  float r_new[2];
  float r_wo_clamp[2];
  float acc_o[112];
  float acc_s[16];
  half_t mask_local[16];
  float acc_s_1[32];
  float abs_sum[2];
  half_t acc_s_cast[32];
  __shared__ uint64_t _mbarrier[11];
  if (((int)threadIdx.x) == 0) {
    tl::prefetch_tma_descriptor(Q_desc);
    tl::prefetch_tma_descriptor(K_desc);
    tl::prefetch_tma_descriptor(mask_desc);
    tl::prefetch_tma_descriptor(V_desc);
    tl::prefetch_tma_descriptor(Output_desc);
    tl::mbarrier_init(_mbarrier[0], 128);
    tl::mbarrier_init(_mbarrier[1], 128);
    tl::mbarrier_init(_mbarrier[2], 128);
    tl::mbarrier_init(_mbarrier[3], 256);
    tl::mbarrier_init(_mbarrier[4], 256);
    tl::mbarrier_init(_mbarrier[5], 256);
    tl::mbarrier_init(_mbarrier[6], 128);
    tl::mbarrier_init(_mbarrier[7], 256);
    tl::mbarrier_init(_mbarrier[8], 256);
    tl::mbarrier_init(_mbarrier[9], 256);
    tl::mbarrier_init(_mbarrier[10], 256);
  }
  __syncthreads();
  if (256 <= ((int)threadIdx.x)) {
    tl::warpgroup_reg_dealloc<24>();
    // if (((int)threadIdx.x) == 256) {
    //   tl::mbarrier_expect_tx(_mbarrier[6], 32768);
    // }
    // if (((int)threadIdx.x) == 256) {
    //   tl::tma_load(Q_desc, _mbarrier[6], (&(((half_t*)buf_dyn_shmem)[12288])), 0, ((int)blockIdx.y), (((int)blockIdx.x) * 64), 0);
    //   tl::tma_load(Q_desc, _mbarrier[6], (&(((half_t*)buf_dyn_shmem)[16384])), 64, ((int)blockIdx.y), (((int)blockIdx.x) * 64), 0);
    //   tl::tma_load(Q_desc, _mbarrier[6], (&(((half_t*)buf_dyn_shmem)[20480])), 128, ((int)blockIdx.y), (((int)blockIdx.x) * 64), 0);
    //   tl::tma_load(Q_desc, _mbarrier[6], (&(((half_t*)buf_dyn_shmem)[24576])), 192, ((int)blockIdx.y), (((int)blockIdx.x) * 64), 0);
    // }
    // tl::mbarrier_arrive(_mbarrier[6]);
    // for (int k = 0; k < 128; ++k) {
    //   tl::mbarrier_wait(_mbarrier[3], ((k & 1) ^ 1));
    //   if (((int)threadIdx.x) == 256) {
    //     tl::mbarrier_expect_tx(_mbarrier[0], 32768);
    //   }
    //   if (((int)threadIdx.x) == 256) {
    //     tl::tma_load(K_desc, _mbarrier[0], (&(((half_t*)buf_dyn_shmem)[28672])), 0, ((int)blockIdx.y), (k * 64), 0);
    //     tl::tma_load(K_desc, _mbarrier[0], (&(((half_t*)buf_dyn_shmem)[32768])), 64, ((int)blockIdx.y), (k * 64), 0);
    //     tl::tma_load(K_desc, _mbarrier[0], (&(((half_t*)buf_dyn_shmem)[36864])), 128, ((int)blockIdx.y), (k * 64), 0);
    //     tl::tma_load(K_desc, _mbarrier[0], (&(((half_t*)buf_dyn_shmem)[40960])), 192, ((int)blockIdx.y), (k * 64), 0);
    //   }
    //   tl::mbarrier_arrive(_mbarrier[0]);
    //   tl::mbarrier_wait(_mbarrier[4], ((k & 1) ^ 1));
    //   if (((int)threadIdx.x) == 256) {
    //     tl::mbarrier_expect_tx(_mbarrier[1], 8192);
    //   }
    //   if (((int)threadIdx.x) == 256) {
    //     tl::tma_load(mask_desc, _mbarrier[1], (&(((half_t*)buf_dyn_shmem)[0])), (k * 64), (((int)blockIdx.x) * 64), ((int)blockIdx.y));
    //   }
    //   tl::mbarrier_arrive(_mbarrier[1]);
    //   tl::mbarrier_wait(_mbarrier[5], ((k & 1) ^ 1));
    //   if (((int)threadIdx.x) == 256) {
    //     tl::mbarrier_expect_tx(_mbarrier[2], 57344);
    //   }
    //   if (((int)threadIdx.x) == 256) {
    //     tl::tma_load(V_desc, _mbarrier[2], (&(((half_t*)buf_dyn_shmem)[45056])), 0, ((int)blockIdx.y), (k * 64), 0);
    //     tl::tma_load(V_desc, _mbarrier[2], (&(((half_t*)buf_dyn_shmem)[49152])), 64, ((int)blockIdx.y), (k * 64), 0);
    //     tl::tma_load(V_desc, _mbarrier[2], (&(((half_t*)buf_dyn_shmem)[53248])), 128, ((int)blockIdx.y), (k * 64), 0);
    //     tl::tma_load(V_desc, _mbarrier[2], (&(((half_t*)buf_dyn_shmem)[57344])), 192, ((int)blockIdx.y), (k * 64), 0);
    //     tl::tma_load(V_desc, _mbarrier[2], (&(((half_t*)buf_dyn_shmem)[61440])), 256, ((int)blockIdx.y), (k * 64), 0);
    //     tl::tma_load(V_desc, _mbarrier[2], (&(((half_t*)buf_dyn_shmem)[65536])), 320, ((int)blockIdx.y), (k * 64), 0);
    //     tl::tma_load(V_desc, _mbarrier[2], (&(((half_t*)buf_dyn_shmem)[69632])), 384, ((int)blockIdx.y), (k * 64), 0);
    //   }
    //   tl::mbarrier_arrive(_mbarrier[2]);
    // }
  } else {
    tl::warpgroup_reg_alloc<240>();
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
      r[i] = 0.000000e+00f;
    }
    #pragma unroll
    for (int i_1 = 0; i_1 < 2; ++i_1) {
      r_new[i_1] = 0.000000e+00f;
    }
    #pragma unroll
    for (int i_2 = 0; i_2 < 2; ++i_2) {
      r_wo_clamp[i_2] = 0.000000e+00f;
    }
    #pragma unroll
    for (int i_3 = 0; i_3 < 112; ++i_3) {
      acc_o[i_3] = 0.000000e+00f;
    }
    tl::fence_proxy_async();
    // tl::mbarrier_wait(_mbarrier[6], 0);
    #pragma unroll 1
    for (int k_1 = 0; k_1 < 128; ++k_1) {
      #pragma unroll
      for (int i_4 = 0; i_4 < 16; ++i_4) {
        acc_s[i_4] = 0.000000e+00f;
      }
      tl::fence_proxy_async();
      // tl::mbarrier_wait(_mbarrier[0], (k_1 & 1));
      tl::gemm_ss<64, 64, 256, 4, 2, 0, 1>((&(((half_t*)buf_dyn_shmem)[12288])), (&(((half_t*)buf_dyn_shmem)[28672])), (&(acc_s[0])));
      // tl::mbarrier_arrive(_mbarrier[3]);
      // tl::mbarrier_wait(_mbarrier[1], (k_1 & 1));
      #pragma unroll
      for (int i_5 = 0; i_5 < 2; ++i_5) {
        tl::ptx_ldmatrix_x4((&(((half_t*)buf_dyn_shmem)[(((((((((int)threadIdx.x) & 127) >> 5) * 1024) + ((((int)threadIdx.x) & 15) * 64)) + ((((int)threadIdx.x) >> 7) * 32)) + (i_5 * 16)) + (((((int)threadIdx.x) & 31) >> 4) * 8))])), (&(mask_local[(i_5 * 8)])));
      }
      tl::fence_proxy_async();
      // tl::mbarrier_arrive(_mbarrier[4]);
      #pragma unroll
      for (int i_6 = 0; i_6 < 16; ++i_6) {
        acc_s[i_6] = (acc_s[i_6] * ((float)mask_local[i_6]));
      }
      tl::syncthreads_partial(_mbarrier[7]);
      #pragma unroll
      for (int i_7 = 0; i_7 < 8; ++i_7) {
        ((float2*)buf_dyn_shmem)[(((((((((((int)threadIdx.x) & 127) >> 5) * 512) + ((i_7 & 1) * 256)) + (((((int)threadIdx.x) & 31) >> 2) * 32)) + ((((int)threadIdx.x) >> 7) * 16)) + ((i_7 >> 1) * 4)) + (((int)threadIdx.x) & 3)) + 1024)] = *(float2*)(acc_s + (i_7 * 2));
      }
      tl::syncthreads_partial(_mbarrier[8]);
      #pragma unroll
      for (int i_8 = 0; i_8 < 16; ++i_8) {
        *(float2*)(acc_s_1 + (i_8 * 2)) = ((float2*)buf_dyn_shmem)[((((((((((int)threadIdx.x) & 127) >> 5) * 512) + ((i_8 & 1) * 256)) + (((((int)threadIdx.x) & 31) >> 2) * 32)) + ((i_8 >> 1) * 4)) + (((int)threadIdx.x) & 3)) + 1024)];
      }
      #pragma unroll
      for (int i_9 = 0; i_9 < 2; ++i_9) {
        abs_sum[i_9] = 0.000000e+00f;
        #pragma unroll
        for (int rv = 0; rv < 16; ++rv) {
          abs_sum[i_9] = (abs_sum[i_9] + max(acc_s_1[((((rv & 7) * 4) + (i_9 * 2)) + (rv >> 3))], (0.000000e+00f - acc_s_1[((((rv & 7) * 4) + (i_9 * 2)) + (rv >> 3))])));
        }
        abs_sum[i_9] = tl::AllReduce<tl::SumOp, 4, 1>::run(abs_sum[i_9]);
      }
      #pragma unroll
      for (int i_10 = 0; i_10 < 2; ++i_10) {
        r_wo_clamp[i_10] = (r_wo_clamp[i_10] + abs_sum[i_10]);
      }
      #pragma unroll
      for (int i_11 = 0; i_11 < 2; ++i_11) {
        r_new[i_11] = max(r_wo_clamp[i_11], 1.000000e+00f);
      }
      #pragma unroll
      for (int i_12 = 0; i_12 < 112; ++i_12) {
        acc_o[i_12] = ((0 < k_1) ? ((acc_o[i_12] * r[((i_12 & 3) >> 1)]) / r_new[((i_12 & 3) >> 1)]) : acc_o[i_12]);
      }
      #pragma unroll
      for (int i_13 = 0; i_13 < 2; ++i_13) {
        r[i_13] = r_new[i_13];
      }
      #pragma unroll
      for (int i_14 = 0; i_14 < 32; ++i_14) {
        acc_s_1[i_14] = (acc_s_1[i_14] / r_new[((i_14 & 3) >> 1)]);
      }
      #pragma unroll
      for (int i_15 = 0; i_15 < 32; ++i_15) {
        acc_s_cast[i_15] = ((half_t)acc_s_1[i_15]);
      }
      tl::fence_proxy_async();
      // tl::mbarrier_wait(_mbarrier[2], (k_1 & 1));
      tl::gemm_rs<64, 448, 64, 4, 2, 0, 0>((&(acc_s_cast[0])), (&(((half_t*)buf_dyn_shmem)[45056])), (&(acc_o[0])));
      // tl::mbarrier_arrive(_mbarrier[5]);
    }
    tl::syncthreads_partial(_mbarrier[9]);
    #pragma unroll
    for (int i_16 = 0; i_16 < 14; ++i_16) {
      tl::ptx_stmatrix_x4((&(((half_t*)buf_dyn_shmem)[(((((((((((((int)threadIdx.x) >> 7) * 7) + (i_16 >> 1)) >> 1) * 4096) + (((((int)threadIdx.x) & 127) >> 5) * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (((((int)threadIdx.x) >> 7) + (i_16 >> 1)) & 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_16 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 45056)])), __pack_half2(((half_t)acc_o[(i_16 * 8)]), ((half_t)acc_o[((i_16 * 8) + 1)])), __pack_half2(((half_t)acc_o[((i_16 * 8) + 2)]), ((half_t)acc_o[((i_16 * 8) + 3)])), __pack_half2(((half_t)acc_o[((i_16 * 8) + 4)]), ((half_t)acc_o[((i_16 * 8) + 5)])), __pack_half2(((half_t)acc_o[((i_16 * 8) + 6)]), ((half_t)acc_o[((i_16 * 8) + 7)])));
    }
    tl::fence_proxy_async();
    tl::syncthreads_partial(_mbarrier[10]);
    if (((int)threadIdx.x) == 0) {
      tl::tma_store(Output_desc, (&(((half_t*)buf_dyn_shmem)[45056])), 0, ((int)blockIdx.y), (((int)blockIdx.x) * 64), 0);
      tl::tma_store(Output_desc, (&(((half_t*)buf_dyn_shmem)[49152])), 64, ((int)blockIdx.y), (((int)blockIdx.x) * 64), 0);
      tl::tma_store(Output_desc, (&(((half_t*)buf_dyn_shmem)[53248])), 128, ((int)blockIdx.y), (((int)blockIdx.x) * 64), 0);
      tl::tma_store(Output_desc, (&(((half_t*)buf_dyn_shmem)[57344])), 192, ((int)blockIdx.y), (((int)blockIdx.x) * 64), 0);
      tl::tma_store(Output_desc, (&(((half_t*)buf_dyn_shmem)[61440])), 256, ((int)blockIdx.y), (((int)blockIdx.x) * 64), 0);
      tl::tma_store(Output_desc, (&(((half_t*)buf_dyn_shmem)[65536])), 320, ((int)blockIdx.y), (((int)blockIdx.x) * 64), 0);
      tl::tma_store(Output_desc, (&(((half_t*)buf_dyn_shmem)[69632])), 384, ((int)blockIdx.y), (((int)blockIdx.x) * 64), 0);
    }
  }
}