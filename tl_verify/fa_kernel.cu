#include <tl_templates/gemm_sm90.h>
#include <tl_templates/copy.h>
#include <tl_templates/copy_sm90.h>
#include <tl_templates/reduce.h>
#include <tl_templates/ldsm.h>
#include <tl_templates/threadblock_swizzle.h>
#include "fa_kernel.hpp"

extern "C" __global__ void __launch_bounds__(512) main_kernel(__grid_constant__ const CUtensorMap K_desc, __grid_constant__ const CUtensorMap Output_desc, __grid_constant__ const CUtensorMap Q_desc, __grid_constant__ const CUtensorMap V_desc) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  float acc_o[32];
  float logsum[2];
  float scores_max[2];
  float acc_s[64];
  float scores_max_prev[2];
  float scores_scale[2];
  float scores_sum[2];
  half_t acc_s_cast[64];
  __shared__ uint64_t _mbarrier[11];
  if (((int)threadIdx.x) == 0) {
    tl::prefetch_tma_descriptor(Q_desc);
    tl::prefetch_tma_descriptor(K_desc);
    tl::prefetch_tma_descriptor(V_desc);
    tl::prefetch_tma_descriptor(Output_desc);
    tl::mbarrier_init(_mbarrier[0], 128);
    tl::mbarrier_init(_mbarrier[1], 128);
    tl::mbarrier_init(_mbarrier[2], 128);
    tl::mbarrier_init(_mbarrier[3], 128);
    tl::mbarrier_init(_mbarrier[4], 384);
    tl::mbarrier_init(_mbarrier[5], 384);
    tl::mbarrier_init(_mbarrier[6], 384);
    tl::mbarrier_init(_mbarrier[7], 384);
    tl::mbarrier_init(_mbarrier[8], 128);
    tl::mbarrier_init(_mbarrier[9], 384);
    tl::mbarrier_init(_mbarrier[10], 384);
  }
  __syncthreads();
  if (384 <= ((int)threadIdx.x)) {
    tl::warpgroup_reg_dealloc<32>();
    if (((int)threadIdx.x) == 384) {
      tl::mbarrier_expect_tx(_mbarrier[8], 24576);
    }
    if (((int)threadIdx.x) == 384) {
      tl::tma_load(Q_desc, _mbarrier[8], (&(((half_t*)buf_dyn_shmem)[0])), 0, ((int)blockIdx.y), (((int)blockIdx.x) * 192), ((int)blockIdx.z));
    }
    tl::mbarrier_arrive(_mbarrier[8]);
    for (int k = 0; k < 4; ++k) {
      tl::mbarrier_wait(_mbarrier[((k & 1) + 4)], ((k >> 1) ^ 1));
      if (((int)threadIdx.x) == 384) {
        tl::mbarrier_expect_tx(_mbarrier[(k & 1)], 16384);
      }
      if (((int)threadIdx.x) == 384) {
        tl::tma_load(K_desc, _mbarrier[(k & 1)], (&(((half_t*)buf_dyn_shmem)[(((k & 1) * 8192) + 12288)])), 0, ((int)blockIdx.y), (k * 128), ((int)blockIdx.z));
      }
      tl::mbarrier_arrive(_mbarrier[(k & 1)]);
      tl::mbarrier_wait(_mbarrier[((k & 1) + 6)], ((k >> 1) ^ 1));
      if (((int)threadIdx.x) == 384) {
        tl::mbarrier_expect_tx(_mbarrier[((k & 1) + 2)], 16384);
      }
      if (((int)threadIdx.x) == 384) {
        tl::tma_load(V_desc, _mbarrier[((k & 1) + 2)], (&(((half_t*)buf_dyn_shmem)[(((k & 1) * 8192) + 28672)])), 0, ((int)blockIdx.y), (k * 128), ((int)blockIdx.z));
      }
      tl::mbarrier_arrive(_mbarrier[((k & 1) + 2)]);
    }
  } else {
    tl::warpgroup_reg_alloc<160>();
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
      acc_o[i] = 0.000000e+00f;
    }
    #pragma unroll
    for (int i_1 = 0; i_1 < 2; ++i_1) {
      logsum[i_1] = 0.000000e+00f;
    }
    #pragma unroll
    for (int i_2 = 0; i_2 < 2; ++i_2) {
      scores_max[i_2] = -CUDART_INF_F;
    }
    tl::fence_proxy_async();
    tl::mbarrier_wait(_mbarrier[8], 0);
    #pragma unroll
    for (int i_3 = 0; i_3 < 64; ++i_3) {
      acc_s[i_3] = 0.000000e+00f;
    }
    tl::fence_proxy_async();
    tl::mbarrier_wait(_mbarrier[0], 0);
    tl::gemm_ss<192, 128, 64, 12, 1, 0, 1>((&(((half_t*)buf_dyn_shmem)[0])), (&(((half_t*)buf_dyn_shmem)[12288])), (&(acc_s[0])));
    tl::mbarrier_arrive(_mbarrier[4]);
    #pragma unroll
    for (int i_4 = 0; i_4 < 2; ++i_4) {
      scores_max_prev[i_4] = scores_max[i_4];
    }
    #pragma unroll
    for (int i_5 = 0; i_5 < 2; ++i_5) {
      scores_max[i_5] = -CUDART_INF_F;
    }
    #pragma unroll
    for (int i_6 = 0; i_6 < 2; ++i_6) {
      #pragma unroll
      for (int rv = 0; rv < 32; ++rv) {
        scores_max[i_6] = max(scores_max[i_6], acc_s[((((rv & 15) * 4) + (i_6 * 2)) + (rv >> 4))]);
      }
      scores_max[i_6] = tl::AllReduce<tl::MaxOp, 4, 1>::run(scores_max[i_6]);
    }
    #pragma unroll
    for (int i_7 = 0; i_7 < 2; ++i_7) {
      scores_scale[i_7] = exp2f(((scores_max_prev[i_7] * 1.803369e-01f) - (scores_max[i_7] * 1.803369e-01f)));
    }
    #pragma unroll
    for (int i_8 = 0; i_8 < 64; ++i_8) {
      acc_s[i_8] = exp2f(((acc_s[i_8] * 1.803369e-01f) - (scores_max[((i_8 & 3) >> 1)] * 1.803369e-01f)));
    }
    #pragma unroll
    for (int i_9 = 0; i_9 < 2; ++i_9) {
      scores_sum[i_9] = 0.000000e+00f;
      #pragma unroll
      for (int rv_1 = 0; rv_1 < 32; ++rv_1) {
        scores_sum[i_9] = (scores_sum[i_9] + acc_s[((((rv_1 & 15) * 4) + (i_9 * 2)) + (rv_1 >> 4))]);
      }
      scores_sum[i_9] = tl::AllReduce<tl::SumOp, 4, 1>::run(scores_sum[i_9]);
    }
    #pragma unroll
    for (int i_10 = 0; i_10 < 2; ++i_10) {
      logsum[i_10] = ((logsum[i_10] * scores_scale[i_10]) + scores_sum[i_10]);
    }
    #pragma unroll
    for (int i_11 = 0; i_11 < 32; ++i_11) {
      acc_o[i_11] = (acc_o[i_11] * scores_scale[((i_11 & 3) >> 1)]);
    }
    #pragma unroll
    for (int i_12 = 0; i_12 < 64; ++i_12) {
      acc_s_cast[i_12] = ((half_t)acc_s[i_12]);
    }
    #pragma unroll 1
for (int k_1 = 0; k_1 < 3; ++k_1) {
      #pragma unroll
      for (int i_13 = 0; i_13 < 64; ++i_13) {
        acc_s[i_13] = 0.000000e+00f;
      }
      tl::fence_proxy_async();
      tl::mbarrier_wait(_mbarrier[((k_1 + 1) & 1)], ((k_1 + 1) >> 1));
      tl::gemm_ss<192, 128, 64, 12, 1, 0, 1,-1>((&(((half_t*)buf_dyn_shmem)[0])), (&(((half_t*)buf_dyn_shmem)[((((k_1 + 1) & 1) * 8192) + 12288)])), (&(acc_s[0])));
      
      tl::mbarrier_wait(_mbarrier[((k_1 & 1) + 2)], (k_1 >> 1));
      tl::gemm_rs<192, 64, 128, 12, 1, 0, 0,-1>((&(acc_s_cast[0])), (&(((half_t*)buf_dyn_shmem)[(((k_1 & 1) * 8192) + 28672)])), (&(acc_o[0])));
      
  cute::warpgroup_wait<1>();
  tl::mbarrier_arrive(_mbarrier[(((k_1 + 1) & 1) + 4)]);
      #pragma unroll
      for (int i_14 = 0; i_14 < 2; ++i_14) {
        scores_max_prev[i_14] = scores_max[i_14];
      }
      #pragma unroll
      for (int i_15 = 0; i_15 < 2; ++i_15) {
        scores_max[i_15] = -CUDART_INF_F;
      }
      #pragma unroll
      for (int i_16 = 0; i_16 < 2; ++i_16) {
        #pragma unroll
        for (int rv_2 = 0; rv_2 < 32; ++rv_2) {
          scores_max[i_16] = max(scores_max[i_16], acc_s[((((rv_2 & 15) * 4) + (i_16 * 2)) + (rv_2 >> 4))]);
        }
        scores_max[i_16] = tl::AllReduce<tl::MaxOp, 4, 1>::run(scores_max[i_16]);
      }
      #pragma unroll
      for (int i_17 = 0; i_17 < 2; ++i_17) {
        scores_scale[i_17] = exp2f(((scores_max_prev[i_17] * 1.803369e-01f) - (scores_max[i_17] * 1.803369e-01f)));
      }
      #pragma unroll
      for (int i_18 = 0; i_18 < 64; ++i_18) {
        acc_s[i_18] = exp2f(((acc_s[i_18] * 1.803369e-01f) - (scores_max[((i_18 & 3) >> 1)] * 1.803369e-01f)));
      }
      #pragma unroll
      for (int i_19 = 0; i_19 < 2; ++i_19) {
        scores_sum[i_19] = 0.000000e+00f;
        #pragma unroll
        for (int rv_3 = 0; rv_3 < 32; ++rv_3) {
          scores_sum[i_19] = (scores_sum[i_19] + acc_s[((((rv_3 & 15) * 4) + (i_19 * 2)) + (rv_3 >> 4))]);
        }
        scores_sum[i_19] = tl::AllReduce<tl::SumOp, 4, 1>::run(scores_sum[i_19]);
      }
      #pragma unroll
      for (int i_20 = 0; i_20 < 2; ++i_20) {
        logsum[i_20] = ((logsum[i_20] * scores_scale[i_20]) + scores_sum[i_20]);
      }
  cute::warpgroup_wait<0>();
  tl::mbarrier_arrive(_mbarrier[((k_1 & 1) + 6)]);
      #pragma unroll
      for (int i_21 = 0; i_21 < 32; ++i_21) {
        acc_o[i_21] = (acc_o[i_21] * scores_scale[((i_21 & 3) >> 1)]);
      }
      #pragma unroll
      for (int i_22 = 0; i_22 < 64; ++i_22) {
        acc_s_cast[i_22] = ((half_t)acc_s[i_22]);
      }
    }
    tl::mbarrier_wait(_mbarrier[3], 1);
    tl::gemm_rs<192, 64, 128, 12, 1, 0, 0>((&(acc_s_cast[0])), (&(((half_t*)buf_dyn_shmem)[36864])), (&(acc_o[0])));
    tl::mbarrier_arrive(_mbarrier[7]);
    #pragma unroll
    for (int i_23 = 0; i_23 < 32; ++i_23) {
      acc_o[i_23] = (acc_o[i_23] / logsum[((i_23 & 3) >> 1)]);
    }
    tl::syncthreads_partial(_mbarrier[9]);
    #pragma unroll
    for (int i_24 = 0; i_24 < 4; ++i_24) {
      tl::ptx_stmatrix_x4((&(((half_t*)buf_dyn_shmem)[(((((((int)threadIdx.x) >> 5) * 1024) + ((((int)threadIdx.x) & 15) * 64)) + (i_24 * 16)) + (((((int)threadIdx.x) & 31) >> 4) * 8))])), __pack_half2(((half_t)acc_o[(i_24 * 8)]), ((half_t)acc_o[((i_24 * 8) + 1)])), __pack_half2(((half_t)acc_o[((i_24 * 8) + 2)]), ((half_t)acc_o[((i_24 * 8) + 3)])), __pack_half2(((half_t)acc_o[((i_24 * 8) + 4)]), ((half_t)acc_o[((i_24 * 8) + 5)])), __pack_half2(((half_t)acc_o[((i_24 * 8) + 6)]), ((half_t)acc_o[((i_24 * 8) + 7)])));
    }
    tl::fence_proxy_async();
    tl::syncthreads_partial(_mbarrier[10]);
    if (((int)threadIdx.x) == 0) {
      tl::tma_store(Output_desc, (&(((half_t*)buf_dyn_shmem)[0])), 0, ((int)blockIdx.y), (((int)blockIdx.x) * 192), ((int)blockIdx.z));
    }
  }
}

template <typename T>
static std::string ArrayToStr(const T* ptr, size_t n) {
  std::stringstream ss;
  ss << "[";
  for (size_t i = 0; i < n; i++) {
    if (i > 0) ss << ", ";
    ss << ptr[i];
  }
  ss << "]";
  return ss.str();
}

struct TensorMapArgs {
  CUtensorMap* map;
  CUtensorMapDataType type;
  cuuint32_t tensorRank;
  void* globalAddress;
  cuuint64_t globalDim[5], globalStride[5];
  cuuint32_t boxDim[5], elementStrides[5];
  CUtensorMapInterleave interleave;
  CUtensorMapSwizzle swizzle;
  CUtensorMapL2promotion l2Promotion;
  CUtensorMapFloatOOBfill oobFill;

  std::string ToDebugString() {
    std::stringstream ss;
    ss << "TMA Desc Addr:   " << map << std::endl
       << "format         " << type << std::endl
       << "dim            " << tensorRank << std::endl
       << "gmem_address   " << globalAddress << std::endl
       << "globalDim      " << ArrayToStr(globalDim, tensorRank) << std::endl
       << "globalStrides  " << ArrayToStr(globalStride, tensorRank) << std::endl
       << "boxDim         " << ArrayToStr(boxDim, tensorRank) << std::endl
       << "elementStrides " << ArrayToStr(elementStrides, tensorRank) << std::endl
       << "interleave     " << interleave << std::endl
       << "swizzle        " << swizzle << std::endl
       << "l2Promotion    " << l2Promotion << std::endl
       << "oobFill        " << oobFill << std::endl;
    return ss.str();
  }
};

void host_function(Flash_fwd_params params) {
  int num_m_blocks = (params.seq_len + params.block_M - 1) / params.block_M;
  dim3 grid(num_m_blocks, params.head, params.batch);
  dim3 block(params.threads);
  size_t sharedMemSize = (params.block_M + 4 * params.block_N) * params.dim * sizeof(half_t);

  CUtensorMap Q_desc = {0};
  CUtensorMap K_desc = {0};
  CUtensorMap V_desc = {0};
  CUtensorMap O_desc = {0};
  TensorMapArgs Q_arg;
  TensorMapArgs K_arg;
  TensorMapArgs V_arg;
  TensorMapArgs O_arg;

  Q_arg.map = &Q_desc;
  Q_arg.type = CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
  Q_arg.tensorRank = 4;
  Q_arg.globalAddress = params.q_ptr;
  Q_arg.globalDim[0] = static_cast<cuuint64_t>(params.dim);
  Q_arg.globalDim[1] = static_cast<cuuint64_t>(params.head);
  Q_arg.globalDim[2] = static_cast<cuuint64_t>(params.seq_len);
  Q_arg.globalDim[3] = static_cast<cuuint64_t>(params.batch);
  Q_arg.globalStride[0] = static_cast<cuuint64_t>(2);
  Q_arg.globalStride[1] = static_cast<cuuint64_t>(2 * params.dim);
  Q_arg.globalStride[2] = static_cast<cuuint64_t>(2 * params.dim * params.head);
  Q_arg.globalStride[3] = static_cast<cuuint64_t>(2 * params.dim * params.head * params.seq_len);
  Q_arg.boxDim[0] = static_cast<cuuint64_t>(64);
  Q_arg.boxDim[1] = static_cast<cuuint64_t>(1);
  Q_arg.boxDim[2] = static_cast<cuuint64_t>(params.block_M);
  Q_arg.boxDim[3] = static_cast<cuuint64_t>(1);
  Q_arg.elementStrides[0] = static_cast<cuuint64_t>(1);
  Q_arg.elementStrides[1] = static_cast<cuuint64_t>(1);
  Q_arg.elementStrides[2] = static_cast<cuuint64_t>(1);
  Q_arg.elementStrides[3] = static_cast<cuuint64_t>(1);
  Q_arg.interleave = CU_TENSOR_MAP_INTERLEAVE_NONE;
  Q_arg.swizzle = CU_TENSOR_MAP_SWIZZLE_128B;
  Q_arg.l2Promotion = CU_TENSOR_MAP_L2_PROMOTION_L2_128B;
  Q_arg.oobFill = CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;

  K_arg.map = &K_desc;
  K_arg.type = CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
  K_arg.tensorRank = 4;
  K_arg.globalAddress = params.k_ptr;
  K_arg.globalDim[0] = static_cast<cuuint64_t>(params.dim);
  K_arg.globalDim[1] = static_cast<cuuint64_t>(params.head);
  K_arg.globalDim[2] = static_cast<cuuint64_t>(params.seq_len);
  K_arg.globalDim[3] = static_cast<cuuint64_t>(params.batch);
  K_arg.globalStride[0] = static_cast<cuuint64_t>(2);
  K_arg.globalStride[1] = static_cast<cuuint64_t>(2 * params.dim);
  K_arg.globalStride[2] = static_cast<cuuint64_t>(2 * params.dim * params.head);
  K_arg.globalStride[3] = static_cast<cuuint64_t>(2 * params.dim * params.head * params.seq_len);
  K_arg.boxDim[0] = static_cast<cuuint64_t>(64);
  K_arg.boxDim[1] = static_cast<cuuint64_t>(1);
  K_arg.boxDim[2] = static_cast<cuuint64_t>(params.block_N);
  K_arg.boxDim[3] = static_cast<cuuint64_t>(1);
  K_arg.elementStrides[0] = static_cast<cuuint64_t>(1);
  K_arg.elementStrides[1] = static_cast<cuuint64_t>(1);
  K_arg.elementStrides[2] = static_cast<cuuint64_t>(1);
  K_arg.elementStrides[3] = static_cast<cuuint64_t>(1);
  K_arg.interleave = CU_TENSOR_MAP_INTERLEAVE_NONE;
  K_arg.swizzle = CU_TENSOR_MAP_SWIZZLE_128B;
  K_arg.l2Promotion = CU_TENSOR_MAP_L2_PROMOTION_L2_128B;
  K_arg.oobFill = CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;

  V_arg.map = &V_desc;
  V_arg.type = CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
  V_arg.tensorRank = 4;
  V_arg.globalAddress = params.v_ptr;
  V_arg.globalDim[0] = static_cast<cuuint64_t>(params.dim);
  V_arg.globalDim[1] = static_cast<cuuint64_t>(params.head);
  V_arg.globalDim[2] = static_cast<cuuint64_t>(params.seq_len);
  V_arg.globalDim[3] = static_cast<cuuint64_t>(params.batch);
  V_arg.globalStride[0] = static_cast<cuuint64_t>(2);
  V_arg.globalStride[1] = static_cast<cuuint64_t>(2 * params.dim);
  V_arg.globalStride[2] = static_cast<cuuint64_t>(2 * params.dim * params.head);
  V_arg.globalStride[3] = static_cast<cuuint64_t>(2 * params.dim * params.head * params.seq_len);
  V_arg.boxDim[0] = static_cast<cuuint64_t>(64);
  V_arg.boxDim[1] = static_cast<cuuint64_t>(1);
  V_arg.boxDim[2] = static_cast<cuuint64_t>(params.block_N);
  V_arg.boxDim[3] = static_cast<cuuint64_t>(1);
  V_arg.elementStrides[0] = static_cast<cuuint64_t>(1);
  V_arg.elementStrides[1] = static_cast<cuuint64_t>(1);
  V_arg.elementStrides[2] = static_cast<cuuint64_t>(1);
  V_arg.elementStrides[3] = static_cast<cuuint64_t>(1);
  V_arg.interleave = CU_TENSOR_MAP_INTERLEAVE_NONE;
  V_arg.swizzle = CU_TENSOR_MAP_SWIZZLE_128B;
  V_arg.l2Promotion = CU_TENSOR_MAP_L2_PROMOTION_L2_128B;
  V_arg.oobFill = CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;

  O_arg.map = &O_desc;
  O_arg.type = CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
  O_arg.tensorRank = 4;
  O_arg.globalAddress = params.output_ptr;
  O_arg.globalDim[0] = static_cast<cuuint64_t>(params.dim);
  O_arg.globalDim[1] = static_cast<cuuint64_t>(params.head);
  O_arg.globalDim[2] = static_cast<cuuint64_t>(params.seq_len);
  O_arg.globalDim[3] = static_cast<cuuint64_t>(params.batch);
  O_arg.globalStride[0] = static_cast<cuuint64_t>(2);
  O_arg.globalStride[1] = static_cast<cuuint64_t>(2 * params.dim);
  O_arg.globalStride[2] = static_cast<cuuint64_t>(2 * params.dim * params.head);
  O_arg.globalStride[3] = static_cast<cuuint64_t>(2 * params.dim * params.head * params.seq_len);
  O_arg.boxDim[0] = static_cast<cuuint64_t>(64);
  O_arg.boxDim[1] = static_cast<cuuint64_t>(1);
  O_arg.boxDim[2] = static_cast<cuuint64_t>(params.block_M);
  O_arg.boxDim[3] = static_cast<cuuint64_t>(1);
  O_arg.elementStrides[0] = static_cast<cuuint64_t>(1);
  O_arg.elementStrides[1] = static_cast<cuuint64_t>(1);
  O_arg.elementStrides[2] = static_cast<cuuint64_t>(1);
  O_arg.elementStrides[3] = static_cast<cuuint64_t>(1);
  O_arg.interleave = CU_TENSOR_MAP_INTERLEAVE_NONE;
  O_arg.swizzle = CU_TENSOR_MAP_SWIZZLE_NONE;
  O_arg.l2Promotion = CU_TENSOR_MAP_L2_PROMOTION_L2_128B;
  O_arg.oobFill = CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;

  CUresult result;
  result = cuTensorMapEncodeTiled(
      Q_arg.map, Q_arg.type, Q_arg.tensorRank, Q_arg.globalAddress, Q_arg.globalDim, Q_arg.globalStride + 1, Q_arg.boxDim,
      Q_arg.elementStrides, Q_arg.interleave, Q_arg.swizzle, Q_arg.l2Promotion, Q_arg.oobFill);
  if (result != CUDA_SUCCESS) {
    std::cout << "Failed to initialize the TMA descriptor " << result << std::endl
              << Q_arg.ToDebugString();
  }

  result = cuTensorMapEncodeTiled(
      K_arg.map, K_arg.type, K_arg.tensorRank, K_arg.globalAddress, K_arg.globalDim, K_arg.globalStride + 1, K_arg.boxDim,
      K_arg.elementStrides, K_arg.interleave, K_arg.swizzle, K_arg.l2Promotion, K_arg.oobFill);
  if (result != CUDA_SUCCESS) {
    std::cout << "Failed to initialize the TMA descriptor " << result << std::endl
              << K_arg.ToDebugString();
  }

  result = cuTensorMapEncodeTiled(
      V_arg.map, V_arg.type, V_arg.tensorRank, V_arg.globalAddress, V_arg.globalDim, V_arg.globalStride + 1, V_arg.boxDim,
      V_arg.elementStrides, V_arg.interleave, V_arg.swizzle, V_arg.l2Promotion, V_arg.oobFill);
  if (result != CUDA_SUCCESS) {
    std::cout << "Failed to initialize the TMA descriptor " << result << std::endl
              << V_arg.ToDebugString();
  }

  result = cuTensorMapEncodeTiled(
      O_arg.map, O_arg.type, O_arg.tensorRank, O_arg.globalAddress, O_arg.globalDim, O_arg.globalStride + 1, O_arg.boxDim,
      O_arg.elementStrides, O_arg.interleave, O_arg.swizzle, O_arg.l2Promotion, O_arg.oobFill);
  if (result != CUDA_SUCCESS) {
    std::cout << "Failed to initialize the TMA descriptor " << result << std::endl
              << O_arg.ToDebugString();
  }

  const int MAXBYTES  = 1024 * 226;
  cudaFuncSetAttribute(main_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, MAXBYTES);

  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
      std::cerr << "CUDA device synchronization failed: " << cudaGetErrorString(err) << std::endl;
      return;
  }

  main_kernel<<<grid, block, sharedMemSize>>>(K_desc, O_desc, Q_desc, V_desc);

  err = cudaGetLastError();
  if (err != cudaSuccess) {
      std::cerr << "CUDA kernel launch failed: " << cudaGetErrorString(err) << std::endl;
      return;
  }

  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
      std::cerr << "CUDA device synchronization failed: " << cudaGetErrorString(err) << std::endl;
      return;
  }
}