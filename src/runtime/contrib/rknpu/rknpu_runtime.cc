/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file src/runtime/contrib/rknpu/rknpu_runtime.cc
 * \brief RKNPU runtime for the RK3588 NPU.
 *
 * Extends JSONRuntimeBase with:
 * - Regcmd blob deserialization (v1, v2, v3, v4 header formats) and DMA address patching
 * - DMA buffer allocation for input/weight/output/bias/regcmd/task
 * - Multi-op support via op_type dispatch:
 *   - Matmul: [M,K] scatter, [K,N] weight pack, [M,N] gather
 *   - Conv2D: [1,C,H,W] spatial scatter, [N,C,kH,kW] im2col weight pack,
 *             [1,N,H_out,W_out] spatial gather
 *   - Elementwise: same as matmul scatter/gather (reuses matmul path)
 *   - MaxPool: [1,C,H,W] spatial scatter, no weights, [1,N,H_out,W_out] spatial gather
 * - Bias FP16→FP32 conversion for DPU_RDMA
 * - NPU task submission via ioctl
 */

#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/tensor.h>

#include <chrono>
#include <limits>
#include <cstdarg>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef __aarch64__
#include <arm_neon.h>
#endif

#include "../../../support/bytes_io.h"
#include "../json/json_node.h"
#include "../json/json_runtime.h"

#ifdef TVM_RKNPU_RUNTIME
#include "rknpu_device.h"
#endif

namespace tvm {
namespace runtime {
namespace contrib {

using namespace tvm::runtime::json;

// ---------------------------------------------------------------------------
// Regcmd binary data format
// ---------------------------------------------------------------------------
// Header v1 (40 bytes) — matmul only (backward compatibility):
//   [4] magic = "RKNP"
//   [4] version = 1
//   [4] M, [4] K, [4] N
//   [4] num_regcmds
//   [4] enable_mask, [4] int_mask, [4] regcfg_amount
//   [4] num_relocations
//
// Header v2 (64 bytes) — all op types:
//   [4] magic = "RKNP"
//   [4] version = 2
//   [4] op_type (0=matmul, 1=conv2d, 2=elementwise, 3=maxpool, ..., 6=lut)
//   [4] M, [4] K, [4] N
//   [4] num_regcmds
//   [4] enable_mask, [4] int_mask, [4] regcfg_amount
//   [4] num_relocations
//   [4] C, [4] H, [4] W            — input spatial dims (conv2d/maxpool)
//   [4] H_out, [4] W_out           — output spatial dims (conv2d/maxpool)
//
// Body (same for v1 and v2):
//   [num_regcmds * 8] regcmd blob (uint64 LE)
//   [num_relocations * 8] relocation entries:
//     [4] regcmd_index, [4] type (0=input, 1=weight, 2=output, 3=bias)
//
// Header v3 (40 bytes) — M-tiled matmul:
//   [4] magic = "RKNP"
//   [4] version = 3
//   [4] M_full, [4] K, [4] N_full
//   [4] num_tiles
//   [4] enable_mask, [4] int_mask, [4] regcfg_amount
//   [4] num_relocations
//
// Body v3:
//   [num_tiles * 12] tile metadata:
//     [4] m_offset, [4] M_tile, [4] num_regcmds
//   Per tile: [num_regcmds * 8] regcmd blob (uint64 LE)
//   [num_relocations * 8] shared relocation entries
//
// Header v4 (40 bytes) — M+N tiled matmul (same layout as V3):
//   [4] magic = "RKNP"
//   [4] version = 4
//   [4] M_full, [4] K, [4] N_full
//   [4] num_tiles
//   [4] enable_mask, [4] int_mask, [4] regcfg_amount
//   [4] num_relocations
//
// Body v4:
//   [num_tiles * 20] tile metadata:
//     [4] m_offset, [4] M_tile, [4] n_offset, [4] N_tile, [4] num_regcmds
//   Per tile: [num_regcmds * 8] regcmd blob (uint64 LE)
//   [num_relocations * 8] shared relocation entries
//
// Header v5 (64 bytes) — N-tiled conv2d (V2-sized header + V4-style tile body):
//   [4] magic = "RKNP"
//   [4] version = 5
//   [4] op_type
//   [4] M_full, [4] K, [4] N_full
//   [4] num_tiles
//   [4] enable_mask, [4] int_mask, [4] regcfg_amount
//   [4] num_relocations
//   [4] C, [4] H, [4] W
//   [4] H_out, [4] W_out
//
// Body v5 (same as v4):
//   [num_tiles * 20] tile metadata:
//     [4] m_offset, [4] M_tile, [4] n_offset, [4] N_tile, [4] num_regcmds
//   Per tile: [num_regcmds * 8] regcmd blob (uint64 LE)
//   [num_relocations * 8] shared relocation entries
//
static constexpr uint32_t kRegcmdMagic = 0x504E4B52;  // "RKNP" in little-endian
static constexpr uint32_t kRegcmdVersionV1 = 1;
static constexpr uint32_t kRegcmdVersionV2 = 2;
static constexpr uint32_t kRegcmdVersionV3 = 3;
static constexpr uint32_t kRegcmdVersionV4 = 4;
static constexpr uint32_t kRegcmdVersionV5 = 5;
static constexpr uint32_t kRegcmdVersionV6 = 6;
static constexpr uint32_t kRegcmdVersionV7 = 7;
static constexpr uint32_t kRegcmdVersionV8 = 8;

// Operation type codes.
static constexpr uint32_t kOpMatmul = 0;
static constexpr uint32_t kOpConv2D = 1;
static constexpr uint32_t kOpElementwise = 2;
static constexpr uint32_t kOpMaxPool = 3;
static constexpr uint32_t kOpDepthwiseConv2D = 4;
static constexpr uint32_t kOpAvgPool = 5;
static constexpr uint32_t kOpLut = 6;
static constexpr uint32_t kOpGelu = 7;
static constexpr uint32_t kOpLayerNorm = 8;

// Relocation type codes (used in PatchRegcmds under TVM_RKNPU_RUNTIME).
#ifdef TVM_RKNPU_RUNTIME
static constexpr uint32_t kRelocInput = 0;
static constexpr uint32_t kRelocWeight = 1;
static constexpr uint32_t kRelocOutput = 2;
static constexpr uint32_t kRelocBias = 3;
static constexpr uint32_t kRelocIntermediate = 5;  // Base: type = 5 + buffer_index
static constexpr uint32_t kRelocConstant = 16;     // Base: type = 16 + constant_index

// PC-chaining register constants.
static constexpr uint16_t kPcBaseAddress = 0x0010;
static constexpr uint16_t kPcRegisterAmounts = 0x0014;
static constexpr uint16_t kOpRegPc = 0x0101;
#endif

// V1 header: 40 bytes (matmul only, backward compatibility).
struct RegcmdHeaderV1 {
  uint32_t magic;
  uint32_t version;
  uint32_t M;
  uint32_t K;
  uint32_t N;
  uint32_t num_regcmds;
  uint32_t enable_mask;
  uint32_t int_mask;
  uint32_t regcfg_amount;
  uint32_t num_relocations;
};

// V2 header: 64 bytes (all op types).
struct RegcmdHeaderV2 {
  uint32_t magic;
  uint32_t version;
  uint32_t op_type;       // 0=matmul, 1=conv2d, 2=ew, 3=maxpool, 4=dw_conv2d, 5=avgpool, 6=lut
  uint32_t M;
  uint32_t K;
  uint32_t N;
  uint32_t num_regcmds;
  uint32_t enable_mask;
  uint32_t int_mask;
  uint32_t regcfg_amount;
  uint32_t num_relocations;
  uint32_t C;             // Input channels (conv2d/maxpool, 0 otherwise)
  uint32_t H;             // Input height (conv2d/maxpool, 0 otherwise)
  uint32_t W;             // Input width (conv2d/maxpool, 0 otherwise)
  uint32_t H_out;         // Output height (conv2d/maxpool, 0 otherwise)
  uint32_t W_out;         // Output width (conv2d/maxpool, 0 otherwise)
};

// V3 header: 40 bytes (M-tiled matmul — same size as V1, num_regcmds replaced by num_tiles).
struct RegcmdHeaderV3 {
  uint32_t magic;
  uint32_t version;
  uint32_t M_full;         // Full M dimension (all tiles share this for strides)
  uint32_t K;
  uint32_t N_full;         // Full N dimension
  uint32_t num_tiles;      // Number of M-tiles
  uint32_t enable_mask;
  uint32_t int_mask;
  uint32_t regcfg_amount;  // Shared across tiles (same template)
  uint32_t num_relocations;
};

// Per-tile metadata (12 bytes each, follows V3 header).
struct TileInfo {
  uint32_t m_offset;       // Offset in M dimension (in elements, not bytes)
  uint32_t M_tile;         // M dimension for this tile
  uint32_t num_regcmds;    // Number of regcmds in this tile's blob
};

// V4 header: same 40-byte layout as V3 (used for M+N tiled matmul).
struct RegcmdHeaderV4 {
  uint32_t magic;
  uint32_t version;
  uint32_t M_full;
  uint32_t K;
  uint32_t N_full;
  uint32_t num_tiles;
  uint32_t enable_mask;
  uint32_t int_mask;
  uint32_t regcfg_amount;
  uint32_t num_relocations;
};

// V4 per-tile metadata (20 bytes each, adds n_offset and N_tile).
struct TileInfoV4 {
  uint32_t m_offset;       // Offset in M dimension (in elements)
  uint32_t M_tile;         // M dimension for this tile
  uint32_t n_offset;       // Offset in N dimension (in elements)
  uint32_t N_tile;         // N dimension for this tile
  uint32_t num_regcmds;    // Number of regcmds in this tile's blob
};

// V5 header: 64 bytes (N-tiled conv2d — V2-sized header with tile metadata body like V4).
// Field 6 is num_tiles (same position as num_regcmds in V2).
struct RegcmdHeaderV5 {
  uint32_t magic;
  uint32_t version;
  uint32_t op_type;
  uint32_t M_full;
  uint32_t K;
  uint32_t N_full;
  uint32_t num_tiles;
  uint32_t enable_mask;
  uint32_t int_mask;
  uint32_t regcfg_amount;
  uint32_t num_relocations;
  uint32_t C;
  uint32_t H;
  uint32_t W;
  uint32_t H_out;
  uint32_t W_out;
};

struct RelocationEntry {
  uint32_t regcmd_index;
  uint32_t type;
};

// V8 graph-level format structs.
struct V8RelocationEntry {
  uint32_t regcmd_index;
  uint32_t buf_idx;
  uint32_t byte_offset;
};

struct V8ExtInput {
  uint32_t param_idx;
  uint32_t buf_idx;
  uint32_t scatter_type;  // 0=feature, 1=weight, 2=bias_fp32
  uint32_t dim0;
  uint32_t dim1;
};

struct V8ExtOutput {
  uint32_t output_idx;
  uint32_t buf_idx;
  uint32_t gather_type;  // 0=feature
  uint32_t dim0;
  uint32_t dim1;
};

struct V8Constant {
  uint32_t buf_idx;
  uint32_t scatter_type;
  uint32_t dim0;
  uint32_t dim1;
  std::vector<char> data;
};

struct V8NpuTask {
  uint32_t num_regcmds;
  uint32_t enable_mask;
  uint32_t int_mask;
  uint32_t regcfg_amount;
  std::vector<uint64_t> regcmds;
  std::vector<V8RelocationEntry> relocations;
};

struct V8CpuTask {
  uint32_t op_type;  // 0=max_reduce, 1=reciprocal
  uint32_t in_buf;
  uint32_t out_buf;
  uint32_t M;
  uint32_t K;
};

struct V8Segment {
  uint32_t type;  // 0=NPU, 1=CPU
  std::vector<V8NpuTask> npu_tasks;
  std::vector<V8CpuTask> cpu_tasks;
};

// V8 scatter type codes (must match Python codegen).
static constexpr uint32_t kV8ScatterFeature = 0;
static constexpr uint32_t kV8ScatterWeight = 1;
static constexpr uint32_t kV8ScatterBiasFP32 = 2;
static constexpr uint32_t kV8GatherFeature = 0;
static constexpr uint32_t kV8SegNPU = 0;
static constexpr uint32_t kV8SegCPU = 1;
static constexpr uint32_t kV8CpuMaxReduce = 0;
static constexpr uint32_t kV8CpuReciprocal = 1;

// ---------------------------------------------------------------------------
// NPU layout utilities (shared between stub and runtime builds)
// ---------------------------------------------------------------------------

static inline int AlignUp(int val, int align) { return ((val + align - 1) / align) * align; }

static inline int PadM(int M) {
  if (M <= 1) return M;
  return ((M + 3) / 4) * 4;
}

static inline bool BridgeDisableNeonTransforms() {
  static int disabled = []() -> int {
    const char* env = std::getenv("TVM_RKNPU_BRIDGE_DISABLE_NEON_TRANSFORMS");
    if (!env) return 0;
    if (std::strcmp(env, "1") == 0 || std::strcmp(env, "true") == 0 ||
        std::strcmp(env, "on") == 0 || std::strcmp(env, "yes") == 0) {
      return 1;
    }
    return 0;
  }();
  return disabled != 0;
}

#ifdef TVM_RKNPU_RUNTIME

/*! \brief Compute flat index in NPU feature layout [C/C2, H, W, C2]. */
static inline int FeatureIndex(int /*C_aligned*/, int H, int W, int C2, int c, int h, int w) {
  int plane = c / C2;
  int offset = c % C2;
  return plane * H * W * C2 + C2 * (h * W + w) + offset;
}

/*! \brief Compute flat index in NPU FP16 weight layout (16 kernels x 32 channels). */
static inline int WeightIndexFP16(int C, int k, int c) {
  // k, c are 1-based
  int kpg = (k - 1) / 16;
  int cpg = (c - 1) / 32;
  int base = (cpg * 32) * 16 + kpg * 16 * C;
  return base + (c - 1) % 32 + ((k - 1) % 16) * 32;
}

/*! \brief Scatter row-major [M, K] FP16 input to NPU feature layout [G, M_pad, 8].
 *
 * NEON-accelerated: G-outer loop writes each group's slab sequentially,
 * 8 elements per NEON load/store. ~7-9x faster than scalar version.
 */
static void ScatterInputFP16(uint16_t* dst, const uint16_t* src, int M, int K, int M_pad,
                              int K_aligned) {
  const int G = K_aligned / 8;
  const int Mp8 = M_pad * 8;
  std::memset(dst, 0, static_cast<size_t>(G) * Mp8 * sizeof(uint16_t));
#ifdef __aarch64__
  if (!BridgeDisableNeonTransforms()) {
    for (int m = 0; m < M; m++) {
      const uint16_t* row = src + m * K;
      int k = 0;
      for (int g = 0; g < G; g++) {
        uint16_t* out = dst + g * Mp8 + m * 8;
        int remaining = K - k;
        if (remaining >= 8) {
          vst1q_u16(out, vld1q_u16(row + k));
        } else if (remaining > 0) {
          uint16_t tmp[8] = {0};
          std::memcpy(tmp, row + k, remaining * sizeof(uint16_t));
          vst1q_u16(out, vld1q_u16(tmp));
        }
        k += 8;
      }
    }
    return;
  }
#endif
  const int C2 = 8;
  for (int m = 0; m < M; m++) {
    for (int k = 0; k < K; k++) {
      int idx = FeatureIndex(K_aligned, M_pad, 1, C2, k, m, 0);
      dst[idx] = src[m * K + k];
    }
  }
}

/*! \brief Gather NPU feature layout [G, M_pad, 8] output to row-major [M, N] FP16.
 *
 * NEON-accelerated: loads 8 elements per NEON op, slices M_pad→M and G*8→N.
 */
static void GatherOutputFP16(uint16_t* dst, const uint16_t* src, int M, int N, int M_pad,
                              int N_aligned) {
  const int G = N_aligned / 8;
  const int Mp8 = M_pad * 8;
#ifdef __aarch64__
  if (!BridgeDisableNeonTransforms()) {
    for (int m = 0; m < M; m++) {
      uint16_t* out = dst + m * N;
      int n_remaining = N;
      for (int g = 0; g < G && n_remaining > 0; g++) {
        const uint16_t* in = src + g * Mp8 + m * 8;
        if (n_remaining >= 8) {
          vst1q_u16(out + g * 8, vld1q_u16(in));
          n_remaining -= 8;
        } else {
          uint16_t tmp[8];
          vst1q_u16(tmp, vld1q_u16(in));
          std::memcpy(out + g * 8, tmp, n_remaining * sizeof(uint16_t));
          n_remaining = 0;
        }
      }
    }
    return;
  }
#endif
  const int C2 = 8;
  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
      int idx = FeatureIndex(N_aligned, M_pad, 1, C2, n, m, 0);
      dst[m * N + n] = src[idx];
    }
  }
}

/*!
 * \brief Scatter spatial [1, C, H, W] FP16 input to NPU feature layout [G, H, W, 8].
 *
 * NEON-accelerated: G-outer loop for sequential writes, 8 elements per store.
 * Source is [C, H, W] (NCHW without batch); we scatter interleaved channel groups.
 *
 * Note: source layout is [C, H, W] so channels are NOT contiguous per spatial position.
 * We must gather 8 channels from C strided positions for each (h, w).
 */
static void ScatterSpatialFP16(uint16_t* dst, const uint16_t* src, int C, int H, int W,
                                int C_aligned) {
  const int G = C_aligned / 8;
  const int HW8 = H * W * 8;
  std::memset(dst, 0, static_cast<size_t>(C_aligned) * H * W * sizeof(uint16_t));
  // Source is [C, H, W] (channel-first), not [H, W, C], so we can't do
  // contiguous 8-wide loads along the channel axis. Use a temp buffer to
  // gather 8 channels per spatial position, then NEON-store.
  for (int g = 0; g < G; g++) {
    int c_base = g * 8;
    for (int h = 0; h < H; h++) {
      for (int w = 0; w < W; w++) {
        uint16_t* out = dst + g * HW8 + (h * W + w) * 8;
        uint16_t tmp[8] = {0};
        int c_avail = C - c_base;
        int cnt = (c_avail >= 8) ? 8 : (c_avail > 0 ? c_avail : 0);
        for (int i = 0; i < cnt; i++) {
          tmp[i] = src[(c_base + i) * H * W + h * W + w];
        }
#ifdef __aarch64__
        vst1q_u16(out, vld1q_u16(tmp));
#else
        std::memcpy(out, tmp, 8 * sizeof(uint16_t));
#endif
      }
    }
  }
}

/*!
 * \brief Gather NPU feature output [G, M_pad, 8] to spatial [1, N, H_out, W_out] FP16.
 *
 * The NPU output is in feature layout [G, M_pad, 8] where M_pad = padded(H_out * W_out).
 * This function reshapes back to [1, N, H_out, W_out] row-major (NCHW).
 *
 * Since output is NCHW (channel-first), we scatter 8 channels from each group
 * into separate channel planes.
 */
static void GatherSpatialOutputFP16(uint16_t* dst, const uint16_t* src, int N, int H_out,
                                     int W_out, int M_pad, int N_aligned) {
  const int G = N_aligned / 8;
  const int Mp8 = M_pad * 8;
  const int spatial = H_out * W_out;
  for (int g = 0; g < G; g++) {
    int n_base = g * 8;
    for (int h = 0; h < H_out; h++) {
      for (int w = 0; w < W_out; w++) {
        int m = h * W_out + w;
        const uint16_t* in = src + g * Mp8 + m * 8;
        uint16_t tmp[8];
#ifdef __aarch64__
        vst1q_u16(tmp, vld1q_u16(in));
#else
        std::memcpy(tmp, in, 8 * sizeof(uint16_t));
#endif
        int n_avail = N - n_base;
        int cnt = (n_avail >= 8) ? 8 : (n_avail > 0 ? n_avail : 0);
        for (int i = 0; i < cnt; i++) {
          dst[(n_base + i) * spatial + h * W_out + w] = tmp[i];
        }
      }
    }
  }
}

/*! \brief Convert FP16 bias to FP32 for DPU_RDMA (which reads FP32 bias). */
static void ConvertBiasFP16ToFP32(float* dst, const uint16_t* src, int N, int N_aligned) {
  std::memset(dst, 0, static_cast<size_t>(N_aligned) * sizeof(float));
  for (int n = 0; n < N; n++) {
    // Convert FP16 to FP32 via bit manipulation.
    uint16_t h = src[n];
    uint32_t sign = (static_cast<uint32_t>(h) & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t f;
    if (exp == 0) {
      if (mant == 0) {
        f = sign;
      } else {
        // Subnormal FP16 → normal FP32.
        exp = 1;
        while ((mant & 0x400) == 0) {
          mant <<= 1;
          exp--;
        }
        mant &= 0x3FF;
        f = sign | ((exp + 127 - 15) << 23) | (mant << 13);
      }
    } else if (exp == 31) {
      f = sign | 0x7F800000 | (mant << 13);  // Inf/NaN
    } else {
      f = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    std::memcpy(&dst[n], &f, sizeof(float));
  }
}

/*! \brief Pack row-major [K, N] FP16 weights to NPU tiled layout [N_al/16, K_al/32, 16, 32].
 *
 * NEON-accelerated: transposes each 32x16 source sub-tile to 16x32 output tile
 * using NEON 4x4 sub-block transposes. ~3-5x faster than scalar version.
 */
static void PackWeightsFP16(uint16_t* dst, const uint16_t* src, int K, int N) {
  int K_aligned = AlignUp(K, 32);
  int N_aligned = AlignUp(N, 16);
  std::memset(dst, 0, static_cast<size_t>(K_aligned) * N_aligned * sizeof(uint16_t));

#ifdef __aarch64__
  if (!BridgeDisableNeonTransforms()) {
    const int KB = K_aligned / 32;
    const int NB = N_aligned / 16;

    for (int nb = 0; nb < NB; nb++) {
      for (int kb = 0; kb < KB; kb++) {
        // Load 32x16 sub-tile from source into local buffer
        uint16_t buf[32 * 16];
        std::memset(buf, 0, sizeof(buf));

        int k_start = kb * 32;
        int k_lim = k_start + 32;
        if (k_lim > K) k_lim = K;
        int n_start = nb * 16;
        int n_lim = n_start + 16;
        if (n_lim > N) n_lim = N;
        int n_cnt = n_lim - n_start;

        for (int k = k_start; k < k_lim; k++) {
          std::memcpy(&buf[(k - k_start) * 16],
                      src + static_cast<size_t>(k) * N + n_start,
                      n_cnt * sizeof(uint16_t));
        }

        // Transpose 32x16 -> 16x32 using NEON 4x4 sub-block transposes
        uint16_t* tile = dst + (static_cast<size_t>(nb) * KB + kb) * 512;

        for (int bj = 0; bj < 4; bj++) {       // 16/4 column blocks
          for (int bi = 0; bi < 8; bi++) {      // 32/4 row blocks
            uint16x4_t r0 = vld1_u16(&buf[(bi * 4 + 0) * 16 + bj * 4]);
            uint16x4_t r1 = vld1_u16(&buf[(bi * 4 + 1) * 16 + bj * 4]);
            uint16x4_t r2 = vld1_u16(&buf[(bi * 4 + 2) * 16 + bj * 4]);
            uint16x4_t r3 = vld1_u16(&buf[(bi * 4 + 3) * 16 + bj * 4]);

            uint16x4x2_t t01 = vtrn_u16(r0, r1);
            uint16x4x2_t t23 = vtrn_u16(r2, r3);

            uint32x2x2_t u02 = vtrn_u32(
                vreinterpret_u32_u16(t01.val[0]),
                vreinterpret_u32_u16(t23.val[0]));
            uint32x2x2_t u13 = vtrn_u32(
                vreinterpret_u32_u16(t01.val[1]),
                vreinterpret_u32_u16(t23.val[1]));

            vst1_u16(&tile[(bj * 4 + 0) * 32 + bi * 4],
                     vreinterpret_u16_u32(u02.val[0]));
            vst1_u16(&tile[(bj * 4 + 1) * 32 + bi * 4],
                     vreinterpret_u16_u32(u13.val[0]));
            vst1_u16(&tile[(bj * 4 + 2) * 32 + bi * 4],
                     vreinterpret_u16_u32(u02.val[1]));
            vst1_u16(&tile[(bj * 4 + 3) * 32 + bi * 4],
                     vreinterpret_u16_u32(u13.val[1]));
          }
        }
      }
    }
    return;
  }
#endif
  for (int n = 0; n < N; n++) {
    for (int k = 0; k < K; k++) {
      int dst_idx = WeightIndexFP16(K_aligned, n + 1, k + 1);
      int src_idx = k * N + n;
      dst[dst_idx] = src[src_idx];
    }
  }
}

/*!
 * \brief Pack conv2d weights [N, C, kH, kW] FP16 to NPU tiled layout via im2col.
 *
 * The kernel is unrolled so that the effective K dimension is C * kH * kW.
 * Each output filter n has C * kH * kW elements packed using WeightIndexFP16
 * with k = c * kH * kW + kh * kW + kw.
 *
 * \param dst      Destination buffer (K_aligned * N_aligned elements, pre-zeroed).
 * \param src      Source weights in [N, C, kH, kW] row-major order.
 * \param N        Number of output filters.
 * \param C        Number of input channels.
 * \param kH       Kernel height.
 * \param kW       Kernel width.
 * \param K_aligned  Aligned K dimension (AlignUp(C * kH * kW, 32)).
 * \param N_aligned  Aligned N dimension (AlignUp(N, 16)).
 */
static void PackWeightsConv2DFP16(uint16_t* dst, const uint16_t* src, int N, int C, int kH,
                                   int kW, int K_aligned, int N_aligned) {
  // CNA im2col order is [kH, kW, C_al32]: the effective-K index is
  //   k = kh * kW * C_al32 + kw * C_al32 + c
  // where C_al32 = AlignUp(C, 32). Source weights are [N, C, kH, kW] row-major.
  int C_al32 = AlignUp(C, 32);
  std::memset(dst, 0, static_cast<size_t>(K_aligned) * N_aligned * sizeof(uint16_t));
  for (int n = 0; n < N; n++) {
    for (int kh = 0; kh < kH; kh++) {
      for (int kw = 0; kw < kW; kw++) {
        for (int c = 0; c < C; c++) {
          int k = kh * kW * C_al32 + kw * C_al32 + c;
          int src_idx = n * C * kH * kW + c * kH * kW + kh * kW + kw;
          int dst_idx = WeightIndexFP16(K_aligned, n + 1, k + 1);
          dst[dst_idx] = src[src_idx];
        }
      }
    }
  }
}

/*!
 * \brief Pack depthwise conv2d weights [C, 1, kH, kW] FP16 to NPU flat layout.
 *
 * Group-of-32 position-major flat layout (no WeightIndexFP16 tiling):
 *   flat_idx = g * G * kH * kW + pos * G + c_local
 *   where G=32, g=c/32, c_local=c%32, pos=kh*kW+kw
 *
 * \param dst       Destination buffer (weight_elems elements, pre-zeroed).
 * \param src       Source weights in [C, 1, kH, kW] row-major order.
 * \param C         Number of channels (= groups for depthwise).
 * \param kH        Kernel height.
 * \param kW        Kernel width.
 * \param weight_elems  Total number of elements in destination buffer.
 */
static void PackWeightsDepthwiseFP16(uint16_t* dst, const uint16_t* src, int C, int kH, int kW,
                                      int weight_elems) {
  const int G = 32;
  int positions = kH * kW;
  std::memset(dst, 0, static_cast<size_t>(weight_elems) * sizeof(uint16_t));
  for (int c = 0; c < C; c++) {
    int g = c / G;
    int c_local = c % G;
    for (int kh = 0; kh < kH; kh++) {
      for (int kw = 0; kw < kW; kw++) {
        int pos = kh * kW + kw;
        int flat_idx = g * G * positions + pos * G + c_local;
        // src layout: [C, 1, kH, kW] → src_idx = c * kH * kW + kh * kW + kw
        int src_idx = c * kH * kW + kh * kW + kw;
        if (flat_idx < weight_elems) {
          dst[flat_idx] = src[src_idx];
        }
      }
    }
  }
}

/*!
 * \brief Convert a float to FP16 bit representation.
 */
static inline uint16_t FloatToHalf(float f) {
  uint32_t bits;
  std::memcpy(&bits, &f, sizeof(float));
  uint32_t sign = (bits >> 16) & 0x8000;
  int32_t exp = ((bits >> 23) & 0xFF) - 127;
  uint32_t mant = bits & 0x7FFFFF;
  if (exp > 15) {
    return static_cast<uint16_t>(sign | 0x7C00);  // Inf
  } else if (exp < -14) {
    return static_cast<uint16_t>(sign);  // Zero (flush subnormals)
  }
  return static_cast<uint16_t>(sign | ((exp + 15) << 10) | (mant >> 13));
}

/*!
 * \brief Generate and pack constant avgpool weights for depthwise conv2d emulation.
 *
 * Creates a uniform [C, 1, kH, kW] weight tensor where each element is
 * FP16(1/(kH*kW)), then packs it in the depthwise group-of-32 layout.
 */
static void GenerateAvgPoolWeightsFP16(uint16_t* dst, int C, int kH, int kW,
                                        int weight_elems) {
  // Create uniform source weights: all = 1/(kH*kW)
  float recip = 1.0f / static_cast<float>(kH * kW);
  uint16_t recip_fp16 = FloatToHalf(recip);

  int src_size = C * kH * kW;
  std::vector<uint16_t> src(src_size, recip_fp16);

  // Pack using the standard depthwise layout
  PackWeightsDepthwiseFP16(dst, src.data(), C, kH, kW, weight_elems);
}

#endif  // TVM_RKNPU_RUNTIME

// ---------------------------------------------------------------------------
// RKNPURuntime
// ---------------------------------------------------------------------------

class RKNPURuntime : public JSONRuntimeBase {
 public:
  RKNPURuntime(const std::string& symbol_name, const std::string& graph_json,
               const ffi::Array<ffi::String>& const_names, const std::string& regcmd_data)
      : JSONRuntimeBase(symbol_name, graph_json, const_names), regcmd_data_(regcmd_data) {}

  const char* kind() const final { return "rknpu"; }

  // Custom SaveToBytes that includes regcmd data.
  ffi::Bytes SaveToBytes() const override {
    std::string result;
    support::BytesOutStream stream(&result);
    stream.Write(symbol_name_);
    stream.Write(graph_json_);
    std::vector<std::string> consts;
    for (const auto& it : const_names_) {
      consts.push_back(it);
    }
    stream.Write(consts);
    stream.Write(regcmd_data_);
    return ffi::Bytes(std::move(result));
  }

  static ffi::Module LoadFromBytesRKNPU(const ffi::Bytes& bytes) {
    support::BytesInStream stream(bytes);
    std::string symbol;
    std::string graph_json;
    std::vector<std::string> consts;
    std::string regcmd_data;

    TVM_FFI_ICHECK(stream.Read(&symbol)) << "Loading symbol name failed";
    TVM_FFI_ICHECK(stream.Read(&graph_json)) << "Loading graph json failed";
    TVM_FFI_ICHECK(stream.Read(&consts)) << "Loading const name list failed";
    // regcmd_data may be empty for old modules.
    stream.Read(&regcmd_data);

    ffi::Array<ffi::String> const_names;
    for (const auto& it : consts) {
      const_names.push_back(it);
    }
    auto n = ffi::make_object<RKNPURuntime>(symbol, graph_json, const_names, regcmd_data);
    return ffi::Module(n);
  }

#ifdef TVM_RKNPU_RUNTIME
  // =========================================================================
  // Real NPU execution (TVM_RKNPU_RUNTIME defined)
  // =========================================================================

  void Init(const ffi::Array<Tensor>& consts) final {
    TVM_FFI_ICHECK_EQ(consts.size(), const_idx_.size())
        << "The number of input constants must match the number of required constants.";
    SetupConstants(consts);

    TVM_FFI_ICHECK(!regcmd_data_.empty()) << "RKNPURuntime requires regcmd data";

    ParseRegcmdData();

    if (is_v8_) {
      InitV8();
      return;
    }

    OpenDeviceAndAllocate();

    LOG(INFO) << "RKNPURuntime::Init completed for " << symbol_name_ << " (op="
              << OpTypeName(op_type_) << " M=" << M_ << " K=" << K_ << " N=" << N_;
    if (op_type_ == kOpConv2D || op_type_ == kOpDepthwiseConv2D || op_type_ == kOpMaxPool || op_type_ == kOpAvgPool) {
      LOG(INFO) << "  spatial: C=" << C_ << " H=" << H_ << " W=" << W_ << " H_out=" << H_out_
                << " W_out=" << W_out_;
    }
    if (num_sub_tasks_ > 0) {
      LOG(INFO) << "  " << num_sub_tasks_ << " sub-tasks";
    } else if (num_tiles_ > 1) {
      const char* tile_label = (version_ == kRegcmdVersionV4) ? " V4-tiles"
                             : (version_ == kRegcmdVersionV5) ? " V5-tiles"
                             : " M-tiles";
      LOG(INFO) << "  " << num_tiles_ << tile_label;
    }
    LOG(INFO) << "  " << num_regcmds_ << " regcmds)";
  }

  void Run() final {
    if (is_v8_) {
      RunV8();
      return;
    }

    const DLTensor* input_tensor = data_entry_[input_var_eid_[0]];
    const DLTensor* output_tensor = data_entry_[EntryID(outputs_[0])];
    auto maybe_dump_regcmds = [&](const char* tag, const uint64_t* regs, size_t n) {
      const char* env = std::getenv("TVM_RKNPU_DUMP_REGCMDS");
      if (!env || regs == nullptr || n == 0) return;
      if (!(std::strcmp(env, "1") == 0 || std::strcmp(env, "true") == 0 ||
            std::strcmp(env, "on") == 0 || std::strcmp(env, "yes") == 0)) {
        return;
      }
      std::fprintf(stderr, "%s n_regcmds=%zu first=0x%016llx second=0x%016llx\n", tag, n,
                   static_cast<unsigned long long>(regs[0]),
                   static_cast<unsigned long long>(n > 1 ? regs[1] : 0ULL));
      size_t limit = n < 20 ? n : 20;
      std::ostringstream os;
      os << tag << " n_regcmds=" << n;
      for (size_t i = 0; i < limit; ++i) {
        uint64_t u = regs[i];
        uint16_t reg = static_cast<uint16_t>(u & 0xFFFF);
        uint32_t addr = static_cast<uint32_t>((u >> 16) & 0xFFFFFFFFULL);
        os << " [" << i << ":reg=0x" << std::hex << reg << ",addr=0x" << addr
           << ",u64=0x" << u << std::dec << "]";
      }
      LOG(INFO) << os.str();
    };

    // --- Scatter input ---
    switch (op_type_) {
      case kOpMatmul:
      case kOpElementwise:
      case kOpLut:
      case kOpGelu:
        // Matmul, elementwise, LUT, and GELU use the same [M, K] -> feature layout scatter.
        ScatterInputFP16(input_buf_.As<uint16_t>(),
                         static_cast<const uint16_t*>(input_tensor->data), M_, K_, M_pad_,
                         K_aligned_);
        break;
      case kOpConv2D:
      case kOpDepthwiseConv2D:
      case kOpMaxPool:
      case kOpAvgPool:
        // Conv2d, depthwise conv2d, and pool use spatial [1, C, H, W] -> feature layout scatter.
        ScatterSpatialFP16(input_buf_.As<uint16_t>(),
                           static_cast<const uint16_t*>(input_tensor->data), C_, H_, W_,
                           C_aligned_);
        break;
      default:
        LOG(FATAL) << "Unsupported op_type in Run scatter: " << op_type_;
    }

    // --- Pack weights (not applicable for maxpool; avgpool generates weights internally) ---
    if (op_type_ == kOpAvgPool) {
      // Avg pool uses depthwise conv2d with constant 1/(kH*kW) weights.
      int weight_elems = static_cast<int>(weight_buf_.size / sizeof(uint16_t));
      GenerateAvgPoolWeightsFP16(weight_buf_.As<uint16_t>(), C_, dw_kH_, dw_kW_, weight_elems);
      device_->SyncToDevice(weight_buf_);
    } else if (op_type_ != kOpMaxPool && op_type_ != kOpLut && op_type_ != kOpGelu) {
      if (op_type_ == kOpElementwise && input_var_eid_.size() == 1) {
        // Self-op (e.g., multiply(x, x)): scatter same input to both buffers.
        ScatterInputFP16(weight_buf_.As<uint16_t>(),
                         static_cast<const uint16_t*>(input_tensor->data), M_, K_, M_pad_,
                         K_aligned_);
      } else {
      TVM_FFI_ICHECK_GE(input_var_eid_.size(), 2u) << "Expected at least 2 inputs (data + weight)";
      const DLTensor* weight_tensor = data_entry_[input_var_eid_[1]];

      switch (op_type_) {
        case kOpMatmul:
          // Matmul: [K, N] row-major weights -> NPU tiled layout.
          PackWeightsFP16(weight_buf_.As<uint16_t>(),
                          static_cast<const uint16_t*>(weight_tensor->data), K_, N_);
          break;
        case kOpElementwise:
          // Elementwise: second input uses same feature layout as first input.
          ScatterInputFP16(weight_buf_.As<uint16_t>(),
                           static_cast<const uint16_t*>(weight_tensor->data), M_, K_, M_pad_,
                           K_aligned_);
          break;
        case kOpConv2D: {
          // Conv2d: [N, C, kH, kW] weights via im2col packing.
          // Derive kH, kW from K = C_aligned * kH * kW (K_eff uses 32-aligned C).
          int kHkW = (C_aligned_ > 0) ? static_cast<int>(K_ / C_aligned_) : 1;
          int kH = 1, kW = kHkW;
          // Try to find integer square root for common square kernels.
          for (int s = 1; s * s <= kHkW; s++) {
            if (kHkW % s == 0) {
              kH = s;
              kW = kHkW / s;
            }
          }
          PackWeightsConv2DFP16(weight_buf_.As<uint16_t>(),
                                static_cast<const uint16_t*>(weight_tensor->data), N_, C_, kH, kW,
                                K_aligned_, N_aligned_);
          break;
        }
        case kOpDepthwiseConv2D: {
          // Depthwise conv2d: [C, 1, kH, kW] weights via flat group-of-32 packing.
          int weight_elems = static_cast<int>(weight_buf_.size / sizeof(uint16_t));
          PackWeightsDepthwiseFP16(weight_buf_.As<uint16_t>(),
                                   static_cast<const uint16_t*>(weight_tensor->data), C_,
                                   dw_kH_, dw_kW_, weight_elems);
          break;
        }
        default:
          break;
      }
      }  // end else (multi-input)
      device_->SyncToDevice(weight_buf_);
    }

    // --- Convert and upload bias if present ---
    if (has_bias_) {
      // Bias is the last input before output. For ops with weight: index 2. For maxpool: index 1.
      size_t bias_idx = (op_type_ == kOpMaxPool || op_type_ == kOpAvgPool) ? 1 : 2;
      if (input_var_eid_.size() > bias_idx) {
        const DLTensor* bias_tensor = data_entry_[input_var_eid_[bias_idx]];
        ConvertBiasFP16ToFP32(bias_buf_.As<float>(),
                              static_cast<const uint16_t*>(bias_tensor->data), N_, N_aligned_);
        device_->SyncToDevice(bias_buf_);
      }
    }

    // Sync input buffer to device.
    device_->SyncToDevice(input_buf_);
    // Flush output buffer to device so CPU cache lines are clean before NPU writes.
    device_->SyncToDevice(output_buf_);

    // Submit the task(s).
    if (version_ == kRegcmdVersionV7 && num_sub_tasks_ > 0) {
      // V7 PC-chaining: concatenate all sub-task regcmds into regcmd_buf_,
      // patch PC tail of each sub-task to chain to the next, submit once.
      auto* dst = static_cast<uint8_t*>(regcmd_buf_.Data());
      size_t offset = 0;
      for (uint32_t t = 0; t < num_sub_tasks_; t++) {
        const auto& st = sub_tasks_[t];
        size_t blob_bytes = static_cast<size_t>(st.num_regcmds) * 8;
        std::memcpy(dst + offset, st.regcmds.data(), blob_bytes);
        offset += blob_bytes;
      }

      // Patch PC tails: for each sub-task except the last, set the PC tail
      // to point to the next sub-task's DMA address.
      auto* regcmd_ptr = reinterpret_cast<uint64_t*>(regcmd_buf_.Data());
      uint32_t cmd_offset = 0;
      for (uint32_t t = 0; t + 1 < num_sub_tasks_; t++) {
        const auto& st = sub_tasks_[t];
        // PC tail is the last 4 regcmds of this sub-task.
        // ops[n-4] = placeholder (already set)
        // ops[n-3] = PC_REGISTER_AMOUNTS (patch with next task's cmd count)
        // ops[n-2] = marker (already set)
        // ops[n-1] = PC_OPERATION_ENABLE (already set)
        uint32_t next_cmd_offset = cmd_offset + st.num_regcmds;
        uint32_t next_dma_addr = regcmd_buf_.dma_addr + next_cmd_offset * 8;
        uint32_t next_n_cmds = sub_tasks_[t + 1].num_regcmds;

        // Patch ops[n-4]: PC base address -> DMA addr of next sub-task
        uint32_t pc_tail_base = cmd_offset + st.num_regcmds - 4;
        regcmd_ptr[pc_tail_base] =
            (static_cast<uint64_t>(kOpRegPc) << 48) |
            (static_cast<uint64_t>(next_dma_addr) << 16) |
            kPcBaseAddress;
        // Patch ops[n-3]: PC register amounts -> next task's cmd count
        regcmd_ptr[pc_tail_base + 1] =
            (static_cast<uint64_t>(kOpRegPc) << 48) |
            (static_cast<uint64_t>(next_n_cmds / 2 - 1) << 16) |
            kPcRegisterAmounts;

        cmd_offset += st.num_regcmds;
      }

      // Use first sub-task's masks for the task struct.
      enable_mask_ = sub_tasks_[0].enable_mask;
      int_mask_ = sub_tasks_[num_sub_tasks_ - 1].int_mask;  // Last task's interrupt
      regcfg_amount_ = sub_tasks_[0].regcfg_amount;
      num_regcmds_ = total_regcmds_;

      device_->SyncToDevice(regcmd_buf_);
      BuildTaskStruct();
      device_->SyncToDevice(task_buf_);
      device_->Submit(task_buf_, /*core_mask=*/1);
    } else if (num_sub_tasks_ > 0) {
      // V6 multi-task sequential: submit each sub-task in order.
      for (uint32_t t = 0; t < num_sub_tasks_; t++) {
        const auto& st = sub_tasks_[t];
        size_t st_blob_size = static_cast<size_t>(st.num_regcmds) * 8;
        std::memcpy(regcmd_buf_.Data(), st.regcmds.data(), st_blob_size);
        // Set task struct fields from sub-task.
        num_regcmds_ = st.num_regcmds;
        enable_mask_ = st.enable_mask;
        int_mask_ = st.int_mask;
        regcfg_amount_ = st.regcfg_amount;
        device_->SyncToDevice(regcmd_buf_);
        BuildTaskStruct();
        device_->SyncToDevice(task_buf_);
        device_->Submit(task_buf_, /*core_mask=*/1);
      }
    } else if (num_tiles_ > 1 && (version_ == kRegcmdVersionV4 || version_ == kRegcmdVersionV5)) {
      // V4/V5 multi-tile: 2D tiling (M + N offsets).
      for (uint32_t t = 0; t < num_tiles_; t++) {
        std::memcpy(regcmds_.data(), tile_regcmds_[t].data(), tiles_v4_[t].num_regcmds * 8);
        num_regcmds_ = tiles_v4_[t].num_regcmds;
        PatchRegcmdsTiledV4(tiles_v4_[t].m_offset, tiles_v4_[t].n_offset);
        std::memcpy(regcmd_buf_.Data(), regcmds_.data(), num_regcmds_ * 8);
        regcfg_amount_ = num_regcmds_ - 4;
        device_->SyncToDevice(regcmd_buf_);
        BuildTaskStruct();
        device_->SyncToDevice(task_buf_);
        device_->Submit(task_buf_, /*core_mask=*/1);
      }
    } else if (num_tiles_ > 1) {
      // V3 multi-tile M-tiled execution: patch and submit each tile sequentially.
      for (uint32_t t = 0; t < num_tiles_; t++) {
        std::memcpy(regcmds_.data(), tile_regcmds_[t].data(), tiles_[t].num_regcmds * 8);
        num_regcmds_ = tiles_[t].num_regcmds;
        PatchRegcmdsTiled(tiles_[t].m_offset);
        std::memcpy(regcmd_buf_.Data(), regcmds_.data(), num_regcmds_ * 8);
        regcfg_amount_ = num_regcmds_ - 4;
        device_->SyncToDevice(regcmd_buf_);
        BuildTaskStruct();
        device_->SyncToDevice(task_buf_);
        device_->Submit(task_buf_, /*core_mask=*/1);
      }
    } else {
      // Single-tile: regcmds already patched in Init.
      maybe_dump_regcmds("RKNPU BYOC submit regcmd sample", regcmds_.data(), num_regcmds_);
      device_->SyncToDevice(regcmd_buf_);
      device_->SyncToDevice(task_buf_);
      device_->Submit(task_buf_, /*core_mask=*/1);
    }

    // Sync output from device.
    device_->SyncFromDevice(output_buf_);

    // --- Gather output ---
    switch (op_type_) {
      case kOpMatmul:
        // Matmul: gather to [M, N] row-major.
        GatherOutputFP16(static_cast<uint16_t*>(output_tensor->data),
                         output_buf_.As<const uint16_t>(), M_, N_, M_pad_, N_aligned_);
        break;
      case kOpElementwise:
      case kOpLut:
      case kOpGelu:
        // Elementwise/LUT/GELU: gather to [M, K] row-major (same layout as input).
        GatherOutputFP16(static_cast<uint16_t*>(output_tensor->data),
                         output_buf_.As<const uint16_t>(), M_, K_, M_pad_, K_aligned_);
        break;
      case kOpConv2D:
      case kOpMaxPool:
        // Conv2d/maxpool: gather to [1, N, H_out, W_out] row-major.
        GatherSpatialOutputFP16(static_cast<uint16_t*>(output_tensor->data),
                                output_buf_.As<const uint16_t>(), N_, H_out_, W_out_, M_pad_,
                                N_aligned_);
        break;
      case kOpDepthwiseConv2D:
      case kOpAvgPool:
        // Depthwise conv2d: output uses C_aligned (32-aligned) not N_aligned (16-aligned).
        GatherSpatialOutputFP16(static_cast<uint16_t*>(output_tensor->data),
                                output_buf_.As<const uint16_t>(), N_, H_out_, W_out_, M_pad_,
                                C_aligned_);
        break;
      default:
        LOG(FATAL) << "Unsupported op_type in Run gather: " << op_type_;
    }
  }

  ~RKNPURuntime() { Cleanup(); }

#else   // !TVM_RKNPU_RUNTIME
  // =========================================================================
  // Stub mode (no hardware)
  // =========================================================================

  void Init(const ffi::Array<Tensor>& consts) final {
    TVM_FFI_ICHECK_EQ(consts.size(), const_idx_.size())
        << "The number of input constants must match the number of required constants.";
    SetupConstants(consts);

    // Parse regcmd header if present, so stub mode validates and logs all op types.
    if (!regcmd_data_.empty()) {
      ParseRegcmdData();
      if (is_v8_) {
        LOG(INFO) << "RKNPURuntime::Init (stub, no hardware) for " << symbol_name_
                  << " V8 graph: " << v8_buf_sizes_.size() << " buffers, "
                  << v8_segments_.size() << " segments, "
                  << v8_ext_inputs_.size() << " ext_inputs, "
                  << v8_ext_outputs_.size() << " ext_outputs, "
                  << total_regcmds_ << " total regcmds";
      } else {
        LOG(INFO) << "RKNPURuntime::Init (stub, no hardware) for " << symbol_name_ << " (op="
                  << OpTypeName(op_type_) << " M=" << M_ << " K=" << K_ << " N=" << N_ << ", "
                  << num_regcmds_ << " regcmds"
                  << (num_sub_tasks_ > 0
                          ? ", " + std::to_string(num_sub_tasks_) + " sub-tasks"
                      : num_tiles_ > 1
                          ? ", " + std::to_string(num_tiles_) +
                                (version_ == kRegcmdVersionV4 ? " V4-tiles"
                                 : version_ == kRegcmdVersionV5 ? " V5-tiles" : " tiles")
                          : "")
                  << ")";
      }
    } else {
      LOG(INFO) << "RKNPURuntime::Init (stub, no hardware) for " << symbol_name_
                << " (no regcmd data)";
    }
  }

  void Run() final {
    LOG(INFO) << "RKNPURuntime::Run (stub, no hardware) for " << symbol_name_ << " (op="
              << OpTypeName(op_type_) << ")";
  }
#endif  // TVM_RKNPU_RUNTIME

 private:
  std::string regcmd_data_;

  // Parsed header fields (common to all op types).
  uint32_t op_type_ = kOpMatmul;
  uint32_t M_ = 0;
  uint32_t K_ = 0;
  uint32_t N_ = 0;
  uint32_t num_regcmds_ = 0;
  uint32_t enable_mask_ = 0;
  uint32_t int_mask_ = 0;
  uint32_t regcfg_amount_ = 0;
  uint32_t num_relocations_ = 0;

  // Extended fields for conv2d/maxpool spatial ops.
  uint32_t C_ = 0;      // Input channels
  uint32_t H_ = 0;      // Input height
  uint32_t W_ = 0;      // Input width
  uint32_t H_out_ = 0;  // Output height
  uint32_t W_out_ = 0;  // Output width
  int dw_kH_ = 0;       // Depthwise kernel height
  int dw_kW_ = 0;       // Depthwise kernel width

  // Derived shape values.
  int K_aligned_ = 0;
  int N_aligned_ = 0;
  int M_pad_ = 0;
  int C_aligned_ = 0;  // AlignUp(C, 32) for conv2d spatial scatter

  // Parsed regcmd/relocation data (host copies).
  std::vector<uint64_t> regcmds_;
  std::vector<RelocationEntry> relocations_;

  // Multi-tile support (V3 format, M-tiling).
  uint32_t num_tiles_ = 1;
  uint32_t version_ = 0;
  std::vector<TileInfo> tiles_;
  std::vector<TileInfoV4> tiles_v4_;
  std::vector<std::vector<uint64_t>> tile_regcmds_;

  // V6 multi-task sequential support (e.g. GELU = LUT).
  struct SubTaskInfo {
    uint32_t num_regcmds;
    uint32_t enable_mask;
    uint32_t int_mask;
    uint32_t regcfg_amount;
    std::vector<uint64_t> regcmds;
    std::vector<RelocationEntry> relocations;
  };
  uint32_t num_sub_tasks_ = 0;
  std::vector<SubTaskInfo> sub_tasks_;

  // V7 multi-intermediate + constants support.
  uint32_t num_intermediates_ = 0;
  std::vector<uint32_t> intermediate_sizes_;     // Per-buffer sizes in bytes
  uint32_t num_constants_ = 0;
  std::vector<std::vector<uint8_t>> constant_data_;  // Embedded constant blobs
  uint32_t total_regcmds_ = 0;  // Sum of all sub-task regcmds (for PC-chain buffer)

#ifdef TVM_RKNPU_RUNTIME
  bool has_bias_ = false;
  std::unique_ptr<rknpu::RKNPUDevice> device_;
  rknpu::DMABuffer input_buf_{};
  rknpu::DMABuffer weight_buf_{};
  rknpu::DMABuffer output_buf_{};
  rknpu::DMABuffer bias_buf_{};
  rknpu::DMABuffer intermediate_buf_{};       // V6 single intermediate
  std::vector<rknpu::DMABuffer> intermediate_bufs_;  // V7 multi-intermediate
  std::vector<rknpu::DMABuffer> constant_bufs_;      // V7 embedded constants
  rknpu::DMABuffer regcmd_buf_{};
  rknpu::DMABuffer task_buf_{};

  // V8 graph-level state.
  bool is_v8_ = false;
  std::vector<uint32_t> v8_buf_sizes_;
  std::vector<V8ExtInput> v8_ext_inputs_;
  std::vector<V8ExtOutput> v8_ext_outputs_;
  std::vector<V8Constant> v8_constants_;
  std::vector<V8Segment> v8_segments_;
  std::vector<rknpu::DMABuffer> v8_bufs_;  // All DMA buffers for V8

  void Cleanup() {
    if (device_) {
      if (is_v8_) {
        // V8: free all graph buffers.
        for (auto& buf : v8_bufs_) {
          if (buf.Valid()) device_->Free(buf);
        }
      } else {
        device_->Free(input_buf_);
        if (weight_buf_.Valid()) device_->Free(weight_buf_);
        device_->Free(output_buf_);
        if (bias_buf_.Valid()) device_->Free(bias_buf_);
        if (intermediate_buf_.Valid()) device_->Free(intermediate_buf_);
        for (auto& buf : intermediate_bufs_) {
          if (buf.Valid()) device_->Free(buf);
        }
        for (auto& buf : constant_bufs_) {
          if (buf.Valid()) device_->Free(buf);
        }
      }
      if (regcmd_buf_.Valid()) device_->Free(regcmd_buf_);
      if (task_buf_.Valid()) device_->Free(task_buf_);
      device_->Close();
      device_.reset();
    }
  }
#endif  // TVM_RKNPU_RUNTIME

  /*! \brief Return human-readable name for an op_type code. */
  static const char* OpTypeName(uint32_t op_type) {
    switch (op_type) {
      case kOpMatmul:
        return "matmul";
      case kOpConv2D:
        return "conv2d";
      case kOpElementwise:
        return "elementwise";
      case kOpMaxPool:
        return "maxpool";
      case kOpDepthwiseConv2D:
        return "depthwise_conv2d";
      case kOpAvgPool:
        return "avgpool";
      case kOpLut:
        return "lut";
      case kOpGelu:
        return "gelu";
      case kOpLayerNorm:
        return "layer_norm";
      default:
        return "unknown";
    }
  }

  void ParseRegcmdData() {
    // Need at least 8 bytes to read magic + version.
    TVM_FFI_ICHECK_GE(regcmd_data_.size(), 8u) << "Regcmd data too small for header";

    uint32_t magic = 0, version = 0;
    std::memcpy(&magic, regcmd_data_.data(), 4);
    std::memcpy(&version, regcmd_data_.data() + 4, 4);

    TVM_FFI_ICHECK_EQ(magic, kRegcmdMagic) << "Invalid regcmd magic";
    TVM_FFI_ICHECK(version == kRegcmdVersionV1 || version == kRegcmdVersionV2 ||
                    version == kRegcmdVersionV3 || version == kRegcmdVersionV4 ||
                    version == kRegcmdVersionV5 || version == kRegcmdVersionV6 ||
                    version == kRegcmdVersionV7 || version == kRegcmdVersionV8)
        << "Unsupported regcmd version: " << version;

    version_ = version;

    size_t header_size = 0;

    // V8: Graph-level multi-composite compilation.
    if (version == kRegcmdVersionV8) {
      // V8 header: 32 bytes (8 uint32):
      //   magic, version, num_buffers, num_segments, num_ext_inputs, num_ext_outputs,
      //   num_constants, total_npu_regcmds
      TVM_FFI_ICHECK_GE(regcmd_data_.size(), 32u) << "Regcmd data too small for v8 header";

      // Debug: dump first 120 bytes of regcmd_data_
      {
        std::string hex;
        size_t n = std::min(regcmd_data_.size(), static_cast<size_t>(120));
        for (size_t j = 0; j < n; j++) {
          char buf[4];
          snprintf(buf, sizeof(buf), "%02x", static_cast<uint8_t>(regcmd_data_[j]));
          hex += buf;
        }
      }

      const char* ptr = regcmd_data_.data() + 8;  // Skip magic + version
      uint32_t num_buffers = 0, num_segments = 0, num_ext_inputs = 0;
      uint32_t num_ext_outputs = 0, num_constants = 0;
      std::memcpy(&num_buffers, ptr, 4); ptr += 4;
      std::memcpy(&num_segments, ptr, 4); ptr += 4;
      std::memcpy(&num_ext_inputs, ptr, 4); ptr += 4;
      std::memcpy(&num_ext_outputs, ptr, 4); ptr += 4;
      std::memcpy(&num_constants, ptr, 4); ptr += 4;
      std::memcpy(&total_regcmds_, ptr, 4); ptr += 4;

      // Buffer table
      v8_buf_sizes_.resize(num_buffers);
      for (uint32_t i = 0; i < num_buffers; i++) {
        std::memcpy(&v8_buf_sizes_[i], ptr, 4); ptr += 4;
      }

      // External input table
      v8_ext_inputs_.resize(num_ext_inputs);
      for (uint32_t i = 0; i < num_ext_inputs; i++) {
        std::memcpy(&v8_ext_inputs_[i].param_idx, ptr, 4); ptr += 4;
        std::memcpy(&v8_ext_inputs_[i].buf_idx, ptr, 4); ptr += 4;
        std::memcpy(&v8_ext_inputs_[i].scatter_type, ptr, 4); ptr += 4;
        std::memcpy(&v8_ext_inputs_[i].dim0, ptr, 4); ptr += 4;
        std::memcpy(&v8_ext_inputs_[i].dim1, ptr, 4); ptr += 4;
      }

      // External output table
      v8_ext_outputs_.resize(num_ext_outputs);
      for (uint32_t i = 0; i < num_ext_outputs; i++) {
        std::memcpy(&v8_ext_outputs_[i].output_idx, ptr, 4); ptr += 4;
        std::memcpy(&v8_ext_outputs_[i].buf_idx, ptr, 4); ptr += 4;
        std::memcpy(&v8_ext_outputs_[i].gather_type, ptr, 4); ptr += 4;
        std::memcpy(&v8_ext_outputs_[i].dim0, ptr, 4); ptr += 4;
        std::memcpy(&v8_ext_outputs_[i].dim1, ptr, 4); ptr += 4;
      }

      // Constants
      v8_constants_.resize(num_constants);
      for (uint32_t c = 0; c < num_constants; c++) {
        std::memcpy(&v8_constants_[c].buf_idx, ptr, 4); ptr += 4;
        std::memcpy(&v8_constants_[c].scatter_type, ptr, 4); ptr += 4;
        std::memcpy(&v8_constants_[c].dim0, ptr, 4); ptr += 4;
        std::memcpy(&v8_constants_[c].dim1, ptr, 4); ptr += 4;
        uint32_t data_size = 0;
        std::memcpy(&data_size, ptr, 4); ptr += 4;
        v8_constants_[c].data.resize(data_size);
        std::memcpy(v8_constants_[c].data.data(), ptr, data_size);
        ptr += data_size;
      }

      // Segments
      v8_segments_.resize(num_segments);
      for (uint32_t s = 0; s < num_segments; s++) {
        uint32_t seg_type = 0, num_tasks = 0;
        std::memcpy(&seg_type, ptr, 4); ptr += 4;
        std::memcpy(&num_tasks, ptr, 4); ptr += 4;
        v8_segments_[s].type = seg_type;

        if (seg_type == kV8SegNPU) {
          v8_segments_[s].npu_tasks.resize(num_tasks);
          for (uint32_t t = 0; t < num_tasks; t++) {
            auto& task = v8_segments_[s].npu_tasks[t];
            uint32_t nr = 0, em = 0, im = 0, ra = 0, nreloc = 0;
            std::memcpy(&nr, ptr, 4); ptr += 4;
            std::memcpy(&em, ptr, 4); ptr += 4;
            std::memcpy(&im, ptr, 4); ptr += 4;
            std::memcpy(&ra, ptr, 4); ptr += 4;
            std::memcpy(&nreloc, ptr, 4); ptr += 4;

            task.num_regcmds = nr;
            task.enable_mask = em;
            task.int_mask = im;
            task.regcfg_amount = ra;

            task.regcmds.resize(nr);
            std::memcpy(task.regcmds.data(), ptr, nr * 8);
            ptr += nr * 8;

            task.relocations.resize(nreloc);
            for (uint32_t r = 0; r < nreloc; r++) {
              std::memcpy(&task.relocations[r].regcmd_index, ptr, 4); ptr += 4;
              std::memcpy(&task.relocations[r].buf_idx, ptr, 4); ptr += 4;
              std::memcpy(&task.relocations[r].byte_offset, ptr, 4); ptr += 4;
            }
          }
        } else {
          // CPU segment
          v8_segments_[s].cpu_tasks.resize(num_tasks);
          for (uint32_t t = 0; t < num_tasks; t++) {
            std::memcpy(&v8_segments_[s].cpu_tasks[t].op_type, ptr, 4); ptr += 4;
            std::memcpy(&v8_segments_[s].cpu_tasks[t].in_buf, ptr, 4); ptr += 4;
            std::memcpy(&v8_segments_[s].cpu_tasks[t].out_buf, ptr, 4); ptr += 4;
            std::memcpy(&v8_segments_[s].cpu_tasks[t].M, ptr, 4); ptr += 4;
            std::memcpy(&v8_segments_[s].cpu_tasks[t].K, ptr, 4); ptr += 4;
          }
        }
      }

      // Set dimensions from first external output for compat logging
      if (!v8_ext_outputs_.empty()) {
        M_ = v8_ext_outputs_[0].dim0;
        K_ = v8_ext_outputs_[0].dim1;
        N_ = v8_ext_outputs_[0].dim1;
      }
      num_regcmds_ = total_regcmds_;
      is_v8_ = true;

      return;
    }

    // V7: Multi-task with intermediate buffers, constants, and PC-chaining.
    if (version == kRegcmdVersionV7) {
      // V7 header: 32 bytes (8 uint32):
      //   magic, version, op_type, M, K, N, num_sub_tasks, num_intermediates
      TVM_FFI_ICHECK_GE(regcmd_data_.size(), 32u) << "Regcmd data too small for v7 header";
      const char* ptr = regcmd_data_.data() + 8;  // Skip magic + version
      uint32_t v7_op_type = 0, v7_M = 0, v7_K = 0, v7_N = 0;
      uint32_t v7_num_tasks = 0, v7_num_intermediates = 0;
      std::memcpy(&v7_op_type, ptr, 4); ptr += 4;
      std::memcpy(&v7_M, ptr, 4); ptr += 4;
      std::memcpy(&v7_K, ptr, 4); ptr += 4;
      std::memcpy(&v7_N, ptr, 4); ptr += 4;
      std::memcpy(&v7_num_tasks, ptr, 4); ptr += 4;
      std::memcpy(&v7_num_intermediates, ptr, 4); ptr += 4;

      op_type_ = v7_op_type;
      M_ = v7_M;
      K_ = v7_K;
      N_ = v7_N;
      num_sub_tasks_ = v7_num_tasks;
      num_intermediates_ = v7_num_intermediates;

      // Compute derived alignment values.
      K_aligned_ = AlignUp(K_, 32);
      N_aligned_ = AlignUp(N_, 16);
      M_pad_ = PadM(M_);

      // Read intermediate buffer sizes.
      intermediate_sizes_.resize(num_intermediates_);
      for (uint32_t i = 0; i < num_intermediates_; i++) {
        uint32_t buf_size = 0;
        std::memcpy(&buf_size, ptr, 4); ptr += 4;
        intermediate_sizes_[i] = buf_size;
      }

      // Read constant data section.
      uint32_t num_constants = 0;
      std::memcpy(&num_constants, ptr, 4); ptr += 4;
      num_constants_ = num_constants;
      constant_data_.resize(num_constants);
      for (uint32_t c = 0; c < num_constants; c++) {
        uint32_t const_size = 0;
        std::memcpy(&const_size, ptr, 4); ptr += 4;
        constant_data_[c].resize(const_size);
        std::memcpy(constant_data_[c].data(), ptr, const_size);
        ptr += const_size;
      }

      // Parse sub-tasks (same format as V6).
      sub_tasks_.resize(num_sub_tasks_);
      uint32_t max_regcmds = 0;
      uint32_t total_regcmds = 0;
      for (uint32_t t = 0; t < num_sub_tasks_; t++) {
        uint32_t nr = 0, em = 0, im = 0, ra = 0, nreloc = 0;
        std::memcpy(&nr, ptr, 4); ptr += 4;
        std::memcpy(&em, ptr, 4); ptr += 4;
        std::memcpy(&im, ptr, 4); ptr += 4;
        std::memcpy(&ra, ptr, 4); ptr += 4;
        std::memcpy(&nreloc, ptr, 4); ptr += 4;

        sub_tasks_[t].num_regcmds = nr;
        sub_tasks_[t].enable_mask = em;
        sub_tasks_[t].int_mask = im;
        sub_tasks_[t].regcfg_amount = ra;

        size_t blob_size = static_cast<size_t>(nr) * 8;
        sub_tasks_[t].regcmds.resize(nr);
        std::memcpy(sub_tasks_[t].regcmds.data(), ptr, blob_size);
        ptr += blob_size;

        size_t reloc_size = static_cast<size_t>(nreloc) * sizeof(RelocationEntry);
        sub_tasks_[t].relocations.resize(nreloc);
        std::memcpy(sub_tasks_[t].relocations.data(), ptr, reloc_size);
        ptr += reloc_size;

        if (nr > max_regcmds) max_regcmds = nr;
        total_regcmds += nr;
      }

      num_regcmds_ = max_regcmds;
      total_regcmds_ = total_regcmds;
      if (num_sub_tasks_ > 0) {
        enable_mask_ = sub_tasks_[0].enable_mask;
        int_mask_ = sub_tasks_[0].int_mask;
        regcfg_amount_ = sub_tasks_[0].regcfg_amount;
      }

      return;
    }

    // V6: Multi-task sequential (e.g. GELU = LUT).
    if (version == kRegcmdVersionV6) {
      // V6 header: 28 bytes (7 uint32):
      //   magic, version, op_type, M, K, N, num_sub_tasks
      TVM_FFI_ICHECK_GE(regcmd_data_.size(), 28u) << "Regcmd data too small for v6 header";
      const char* ptr = regcmd_data_.data() + 8;  // Skip magic + version
      uint32_t v6_op_type = 0, v6_M = 0, v6_K = 0, v6_N = 0, v6_num_tasks = 0;
      std::memcpy(&v6_op_type, ptr, 4); ptr += 4;
      std::memcpy(&v6_M, ptr, 4); ptr += 4;
      std::memcpy(&v6_K, ptr, 4); ptr += 4;
      std::memcpy(&v6_N, ptr, 4); ptr += 4;
      std::memcpy(&v6_num_tasks, ptr, 4); ptr += 4;

      op_type_ = v6_op_type;
      M_ = v6_M;
      K_ = v6_K;
      N_ = v6_N;
      num_sub_tasks_ = v6_num_tasks;

      // Compute derived alignment values.
      K_aligned_ = AlignUp(K_, 32);
      N_aligned_ = AlignUp(N_, 16);
      M_pad_ = PadM(M_);

      // Parse sub-tasks.
      sub_tasks_.resize(num_sub_tasks_);
      uint32_t max_regcmds = 0;
      for (uint32_t t = 0; t < num_sub_tasks_; t++) {
        // Per sub-task header: 20 bytes (5 uint32):
        //   num_regcmds, enable_mask, int_mask, regcfg_amount, num_relocations
        TVM_FFI_ICHECK_GE(regcmd_data_.size(),
                          static_cast<size_t>(ptr - regcmd_data_.data()) + 20u)
            << "V6 sub-task header overflow";
        uint32_t nr = 0, em = 0, im = 0, ra = 0, nreloc = 0;
        std::memcpy(&nr, ptr, 4); ptr += 4;
        std::memcpy(&em, ptr, 4); ptr += 4;
        std::memcpy(&im, ptr, 4); ptr += 4;
        std::memcpy(&ra, ptr, 4); ptr += 4;
        std::memcpy(&nreloc, ptr, 4); ptr += 4;

        sub_tasks_[t].num_regcmds = nr;
        sub_tasks_[t].enable_mask = em;
        sub_tasks_[t].int_mask = im;
        sub_tasks_[t].regcfg_amount = ra;

        // Parse regcmd blob.
        size_t blob_size = static_cast<size_t>(nr) * 8;
        TVM_FFI_ICHECK_GE(regcmd_data_.size(),
                          static_cast<size_t>(ptr - regcmd_data_.data()) + blob_size)
            << "V6 sub-task regcmd blob overflow";
        sub_tasks_[t].regcmds.resize(nr);
        std::memcpy(sub_tasks_[t].regcmds.data(), ptr, blob_size);
        ptr += blob_size;

        // Parse relocation table.
        size_t reloc_size = static_cast<size_t>(nreloc) * sizeof(RelocationEntry);
        TVM_FFI_ICHECK_GE(regcmd_data_.size(),
                          static_cast<size_t>(ptr - regcmd_data_.data()) + reloc_size)
            << "V6 sub-task relocation overflow";
        sub_tasks_[t].relocations.resize(nreloc);
        std::memcpy(sub_tasks_[t].relocations.data(), ptr, reloc_size);
        ptr += reloc_size;

        if (nr > max_regcmds) max_regcmds = nr;
      }

      // Use the largest sub-task's regcmd count for buffer sizing.
      num_regcmds_ = max_regcmds;
      // Use first sub-task's masks as defaults (overridden per-submit in Run).
      if (num_sub_tasks_ > 0) {
        enable_mask_ = sub_tasks_[0].enable_mask;
        int_mask_ = sub_tasks_[0].int_mask;
        regcfg_amount_ = sub_tasks_[0].regcfg_amount;
      }

      return;  // V6 has different body layout
    }

    // V3: M-tiled matmul — different body layout (tile metadata + per-tile blobs).
    if (version == kRegcmdVersionV3) {
      TVM_FFI_ICHECK_GE(regcmd_data_.size(), sizeof(RegcmdHeaderV3))
          << "Regcmd data too small for v3 header";
      RegcmdHeaderV3 v3{};
      std::memcpy(&v3, regcmd_data_.data(), sizeof(v3));

      op_type_ = kOpMatmul;
      M_ = v3.M_full;
      K_ = v3.K;
      N_ = v3.N_full;
      num_tiles_ = v3.num_tiles;
      enable_mask_ = v3.enable_mask;
      int_mask_ = v3.int_mask;
      regcfg_amount_ = v3.regcfg_amount;
      num_relocations_ = v3.num_relocations;

      // Compute derived alignment values.
      K_aligned_ = AlignUp(K_, 32);
      N_aligned_ = AlignUp(N_, 16);
      M_pad_ = PadM(M_);

      // Read tile metadata.
      size_t offset = sizeof(RegcmdHeaderV3);
      size_t tile_meta_size = static_cast<size_t>(num_tiles_) * sizeof(TileInfo);
      TVM_FFI_ICHECK_GE(regcmd_data_.size(), offset + tile_meta_size)
          << "Regcmd data too small for tile metadata";
      tiles_.resize(num_tiles_);
      std::memcpy(tiles_.data(), regcmd_data_.data() + offset, tile_meta_size);
      offset += tile_meta_size;

      // Read per-tile regcmd blobs.
      tile_regcmds_.resize(num_tiles_);
      uint32_t max_regcmds = 0;
      for (uint32_t t = 0; t < num_tiles_; t++) {
        size_t blob_size = static_cast<size_t>(tiles_[t].num_regcmds) * 8;
        TVM_FFI_ICHECK_GE(regcmd_data_.size(), offset + blob_size)
            << "Regcmd data too small for tile " << t << " regcmds";
        tile_regcmds_[t].resize(tiles_[t].num_regcmds);
        std::memcpy(tile_regcmds_[t].data(), regcmd_data_.data() + offset, blob_size);
        offset += blob_size;
        if (tiles_[t].num_regcmds > max_regcmds) max_regcmds = tiles_[t].num_regcmds;
      }
      num_regcmds_ = max_regcmds;  // For regcmd buffer sizing

      // Allocate working regcmds buffer for the largest tile.
      regcmds_.resize(max_regcmds);

      // Read shared relocation table.
      size_t reloc_size = static_cast<size_t>(num_relocations_) * sizeof(RelocationEntry);
      TVM_FFI_ICHECK_GE(regcmd_data_.size(), offset + reloc_size)
          << "Regcmd data too small for relocations";
      relocations_.resize(num_relocations_);
      std::memcpy(relocations_.data(), regcmd_data_.data() + offset, reloc_size);

      return;  // V3 has different body layout, skip common V1/V2 parsing below
    }

    // V4: M+N tiled matmul — 20-byte tile metadata with both m_offset and n_offset.
    if (version == kRegcmdVersionV4) {
      TVM_FFI_ICHECK_GE(regcmd_data_.size(), sizeof(RegcmdHeaderV4))
          << "Regcmd data too small for v4 header";
      RegcmdHeaderV4 v4{};
      std::memcpy(&v4, regcmd_data_.data(), sizeof(v4));

      op_type_ = kOpMatmul;
      M_ = v4.M_full;
      K_ = v4.K;
      N_ = v4.N_full;
      num_tiles_ = v4.num_tiles;
      enable_mask_ = v4.enable_mask;
      int_mask_ = v4.int_mask;
      regcfg_amount_ = v4.regcfg_amount;
      num_relocations_ = v4.num_relocations;

      K_aligned_ = AlignUp(K_, 32);
      N_aligned_ = AlignUp(N_, 16);
      M_pad_ = PadM(M_);

      // Read V4 tile metadata (20 bytes each).
      size_t offset = sizeof(RegcmdHeaderV4);
      size_t tile_meta_size = static_cast<size_t>(num_tiles_) * sizeof(TileInfoV4);
      TVM_FFI_ICHECK_GE(regcmd_data_.size(), offset + tile_meta_size)
          << "Regcmd data too small for v4 tile metadata";
      tiles_v4_.resize(num_tiles_);
      std::memcpy(tiles_v4_.data(), regcmd_data_.data() + offset, tile_meta_size);
      offset += tile_meta_size;

      // Read per-tile regcmd blobs.
      tile_regcmds_.resize(num_tiles_);
      uint32_t max_regcmds = 0;
      for (uint32_t t = 0; t < num_tiles_; t++) {
        size_t blob_size = static_cast<size_t>(tiles_v4_[t].num_regcmds) * 8;
        TVM_FFI_ICHECK_GE(regcmd_data_.size(), offset + blob_size)
            << "Regcmd data too small for v4 tile " << t << " regcmds";
        tile_regcmds_[t].resize(tiles_v4_[t].num_regcmds);
        std::memcpy(tile_regcmds_[t].data(), regcmd_data_.data() + offset, blob_size);
        offset += blob_size;
        if (tiles_v4_[t].num_regcmds > max_regcmds) max_regcmds = tiles_v4_[t].num_regcmds;
      }
      num_regcmds_ = max_regcmds;
      regcmds_.resize(max_regcmds);

      // Read shared relocation table.
      size_t reloc_size = static_cast<size_t>(num_relocations_) * sizeof(RelocationEntry);
      TVM_FFI_ICHECK_GE(regcmd_data_.size(), offset + reloc_size)
          << "Regcmd data too small for v4 relocations";
      relocations_.resize(num_relocations_);
      std::memcpy(relocations_.data(), regcmd_data_.data() + offset, reloc_size);

      return;  // V4 has different body layout
    }

    // V5: N-tiled conv2d — 64-byte V2-style header + V4-style tile metadata body.
    if (version == kRegcmdVersionV5) {
      TVM_FFI_ICHECK_GE(regcmd_data_.size(), sizeof(RegcmdHeaderV5))
          << "Regcmd data too small for v5 header";
      RegcmdHeaderV5 v5{};
      std::memcpy(&v5, regcmd_data_.data(), sizeof(v5));

      op_type_ = v5.op_type;
      M_ = v5.M_full;
      K_ = v5.K;
      N_ = v5.N_full;
      num_tiles_ = v5.num_tiles;
      enable_mask_ = v5.enable_mask;
      int_mask_ = v5.int_mask;
      regcfg_amount_ = v5.regcfg_amount;
      num_relocations_ = v5.num_relocations;
      C_ = v5.C;
      H_ = v5.H;
      W_ = v5.W;
      H_out_ = v5.H_out;
      W_out_ = v5.W_out;

      K_aligned_ = AlignUp(K_, 32);
      N_aligned_ = AlignUp(N_, 16);
      M_pad_ = PadM(M_);
      C_aligned_ = (C_ > 0) ? AlignUp(C_, 32) : 0;

      // Read V5 tile metadata (20 bytes each, same as V4 TileInfoV4).
      size_t offset = sizeof(RegcmdHeaderV5);
      size_t tile_meta_size = static_cast<size_t>(num_tiles_) * sizeof(TileInfoV4);
      TVM_FFI_ICHECK_GE(regcmd_data_.size(), offset + tile_meta_size)
          << "Regcmd data too small for v5 tile metadata";
      tiles_v4_.resize(num_tiles_);
      std::memcpy(tiles_v4_.data(), regcmd_data_.data() + offset, tile_meta_size);
      offset += tile_meta_size;

      // Read per-tile regcmd blobs.
      tile_regcmds_.resize(num_tiles_);
      uint32_t max_regcmds = 0;
      for (uint32_t t = 0; t < num_tiles_; t++) {
        size_t blob_size = static_cast<size_t>(tiles_v4_[t].num_regcmds) * 8;
        TVM_FFI_ICHECK_GE(regcmd_data_.size(), offset + blob_size)
            << "Regcmd data too small for v5 tile " << t << " regcmds";
        tile_regcmds_[t].resize(tiles_v4_[t].num_regcmds);
        std::memcpy(tile_regcmds_[t].data(), regcmd_data_.data() + offset, blob_size);
        offset += blob_size;
        if (tiles_v4_[t].num_regcmds > max_regcmds) max_regcmds = tiles_v4_[t].num_regcmds;
      }
      num_regcmds_ = max_regcmds;
      regcmds_.resize(max_regcmds);

      // Read shared relocation table.
      size_t reloc_size = static_cast<size_t>(num_relocations_) * sizeof(RelocationEntry);
      TVM_FFI_ICHECK_GE(regcmd_data_.size(), offset + reloc_size)
          << "Regcmd data too small for v5 relocations";
      relocations_.resize(num_relocations_);
      std::memcpy(relocations_.data(), regcmd_data_.data() + offset, reloc_size);

      return;  // V5 has different body layout
    }

    if (version == kRegcmdVersionV1) {
      // V1 header: 40 bytes. Op type defaults to matmul.
      TVM_FFI_ICHECK_GE(regcmd_data_.size(), sizeof(RegcmdHeaderV1))
          << "Regcmd data too small for v1 header";
      RegcmdHeaderV1 v1{};
      std::memcpy(&v1, regcmd_data_.data(), sizeof(v1));

      op_type_ = kOpMatmul;
      M_ = v1.M;
      K_ = v1.K;
      N_ = v1.N;
      num_regcmds_ = v1.num_regcmds;
      enable_mask_ = v1.enable_mask;
      int_mask_ = v1.int_mask;
      regcfg_amount_ = v1.regcfg_amount;
      num_relocations_ = v1.num_relocations;
      C_ = 0;
      H_ = 0;
      W_ = 0;
      H_out_ = 0;
      W_out_ = 0;
      header_size = sizeof(RegcmdHeaderV1);
    } else {
      // V2 header: 64 bytes. Includes op_type and spatial fields.
      TVM_FFI_ICHECK_GE(regcmd_data_.size(), sizeof(RegcmdHeaderV2))
          << "Regcmd data too small for v2 header";
      RegcmdHeaderV2 v2{};
      std::memcpy(&v2, regcmd_data_.data(), sizeof(v2));

      op_type_ = v2.op_type;
      TVM_FFI_ICHECK_LE(op_type_, kOpGelu)
          << "Unknown op_type in regcmd header: " << op_type_;
      M_ = v2.M;
      K_ = v2.K;
      N_ = v2.N;
      num_regcmds_ = v2.num_regcmds;
      enable_mask_ = v2.enable_mask;
      int_mask_ = v2.int_mask;
      regcfg_amount_ = v2.regcfg_amount;
      num_relocations_ = v2.num_relocations;
      C_ = v2.C;
      H_ = v2.H;
      W_ = v2.W;
      H_out_ = v2.H_out;
      W_out_ = v2.W_out;
      header_size = sizeof(RegcmdHeaderV2);
    }

    // Compute derived alignment values.
    K_aligned_ = AlignUp(K_, 32);
    C_aligned_ = (C_ > 0) ? AlignUp(C_, 32) : 0;

    if (op_type_ == kOpDepthwiseConv2D || op_type_ == kOpAvgPool) {
      // Depthwise / avgpool (emulated as depthwise): output uses C_aligned (32-aligned) channels.
      N_aligned_ = AlignUp(C_, 32);
      // Derive kH, kW from K = C_aligned * kH * kW.
      int kHkW = (C_aligned_ > 0) ? static_cast<int>(K_ / C_aligned_) : 1;
      dw_kH_ = 1;
      dw_kW_ = kHkW;
      for (int s = 1; s * s <= kHkW; s++) {
        if (kHkW % s == 0) {
          dw_kH_ = s;
          dw_kW_ = kHkW / s;
        }
      }
    } else {
      N_aligned_ = AlignUp(N_, 16);
    }

    // PPU writes output with stride H_out * W_out (no padding), so the gather
    // must also use an unpadded M dimension for maxpool.
    // Note: avgpool is emulated via depthwise conv2d (DPU), so it uses PadM like conv2d.
    if (op_type_ == kOpMaxPool) {
      M_pad_ = M_;
    } else {
      M_pad_ = PadM(M_);
    }

    // Parse regcmd blob.
    size_t blob_offset = header_size;
    size_t blob_size = static_cast<size_t>(num_regcmds_) * 8;
    TVM_FFI_ICHECK_GE(regcmd_data_.size(), blob_offset + blob_size)
        << "Regcmd data too small for blob";

    regcmds_.resize(num_regcmds_);
    std::memcpy(regcmds_.data(), regcmd_data_.data() + blob_offset, blob_size);

    // Parse relocation table.
    size_t reloc_offset = blob_offset + blob_size;
    size_t reloc_size = static_cast<size_t>(num_relocations_) * sizeof(RelocationEntry);
    TVM_FFI_ICHECK_GE(regcmd_data_.size(), reloc_offset + reloc_size)
        << "Regcmd data too small for relocations";

    relocations_.resize(num_relocations_);
    std::memcpy(relocations_.data(), regcmd_data_.data() + reloc_offset, reloc_size);
  }

#ifdef TVM_RKNPU_RUNTIME
  void OpenDeviceAndAllocate() {
    device_ = std::make_unique<rknpu::RKNPUDevice>();

    // Detect bias from relocation table.
    has_bias_ = false;
    for (const auto& reloc : relocations_) {
      if (reloc.type == kRelocBias) {
        has_bias_ = true;
        break;
      }
    }

    // Detect weight from relocation table.
    bool has_weight = false;
    for (const auto& reloc : relocations_) {
      if (reloc.type == kRelocWeight) {
        has_weight = true;
        break;
      }
    }

    // Compute buffer sizes based on op type.
    size_t input_size = 0;
    size_t weight_size = 0;
    // Output buffer: the DPU WDMA may write up to AlignUp(N, 32) channels even
    // when the register configuration specifies fewer (AlignUp(N, 16)).  Allocate
    // the output buffer with 32-channel alignment to prevent DMA overflow.
    int N_output_aligned = (op_type_ == kOpDepthwiseConv2D || op_type_ == kOpAvgPool)
                               ? N_aligned_   // Already 32-aligned (uses C_aligned)
                               : AlignUp(N_, 32);
    size_t output_size = (op_type_ == kOpElementwise)
                             ? static_cast<size_t>(K_aligned_) * M_pad_ * 2
                             : static_cast<size_t>(N_output_aligned) * M_pad_ * 2;

    switch (op_type_) {
      case kOpMatmul:
        // Input: [K_aligned, M_pad] in feature layout (FP16).
        input_size = static_cast<size_t>(K_aligned_) * M_pad_ * 2;
        // Weight: [K_aligned, N_aligned] in tiled layout (FP16).
        weight_size = static_cast<size_t>(K_aligned_) * N_aligned_ * 2;
        break;
      case kOpElementwise:
        // Both inputs and output use same feature layout [K/8, M_pad, 1, 8].
        input_size = static_cast<size_t>(K_aligned_) * M_pad_ * 2;
        // Second input (stored in weight_buf_) uses same feature layout, not weight tiles.
        weight_size = input_size;
        break;
      case kOpConv2D:
        // Input: [C_aligned, H, W] in spatial feature layout (FP16).
        input_size = static_cast<size_t>(C_aligned_) * H_ * W_ * 2;
        // Weight: [K_aligned, N_aligned] in tiled layout (FP16), K = C * kH * kW.
        weight_size = static_cast<size_t>(K_aligned_) * N_aligned_ * 2;
        break;
      case kOpDepthwiseConv2D:
        // Input: [C_aligned, H, W] in spatial feature layout (FP16).
        input_size = static_cast<size_t>(C_aligned_) * H_ * W_ * 2;
        // Depthwise weight: flat layout, K_aligned elements.
        weight_size = static_cast<size_t>(K_aligned_) * 2;
        // Output: [C_aligned, M_pad] (depthwise output uses 32-aligned C).
        break;
      case kOpMaxPool:
        // Input: [C_aligned, H, W] in spatial feature layout (FP16).
        input_size = static_cast<size_t>(C_aligned_) * H_ * W_ * 2;
        // Max pool (PPU) has no weight buffer.
        weight_size = 0;
        break;
      case kOpAvgPool:
        // Avg pool emulated as depthwise conv2d: same layout as depthwise.
        input_size = static_cast<size_t>(C_aligned_) * H_ * W_ * 2;
        // Depthwise weight: flat layout, K_aligned elements.
        weight_size = static_cast<size_t>(K_aligned_) * 2;
        break;
      case kOpLut:
        // LUT: unary op, same feature layout as elementwise [K/8, M_pad, 1, 8].
        input_size = static_cast<size_t>(K_aligned_) * M_pad_ * 2;
        // No weight buffer needed (LUT tables are baked into upload regcmds).
        weight_size = 0;
        break;
      case kOpGelu:
        // GELU: unary op (two sequential tasks with intermediate buffer).
        input_size = static_cast<size_t>(K_aligned_) * M_pad_ * 2;
        weight_size = 0;
        break;
      default:
        LOG(FATAL) << "Unsupported op_type in OpenDeviceAndAllocate: " << op_type_;
    }

    size_t regcmd_size = static_cast<size_t>(num_regcmds_) * 8;
    size_t task_size = sizeof(rknpu::RknpuTask);

    // Allocate DMA buffers.
    input_buf_ = device_->Alloc(input_size);
    TVM_FFI_ICHECK(input_buf_.Valid()) << "Failed to allocate input DMA buffer";

    if (has_weight && weight_size > 0) {
      weight_buf_ = device_->Alloc(weight_size);
      TVM_FFI_ICHECK(weight_buf_.Valid()) << "Failed to allocate weight DMA buffer";
    }

    output_buf_ = device_->Alloc(output_size);
    TVM_FFI_ICHECK(output_buf_.Valid()) << "Failed to allocate output DMA buffer";

    if (has_bias_) {
      // Bias: FP32 per-channel, N_aligned values.
      size_t bias_size = static_cast<size_t>(N_aligned_) * 4;
      bias_buf_ = device_->Alloc(bias_size);
      TVM_FFI_ICHECK(bias_buf_.Valid()) << "Failed to allocate bias DMA buffer";
    }

    // GELU (V6) needs a single intermediate buffer for sigmoid output.
    if (op_type_ == kOpGelu && version_ == kRegcmdVersionV6) {
      size_t intermediate_size = static_cast<size_t>(K_aligned_) * M_pad_ * 2;
      intermediate_buf_ = device_->Alloc(intermediate_size);
      TVM_FFI_ICHECK(intermediate_buf_.Valid()) << "Failed to allocate intermediate DMA buffer";
    }

    // V7: allocate multiple intermediate buffers.
    if (version_ == kRegcmdVersionV7 && num_intermediates_ > 0) {
      intermediate_bufs_.resize(num_intermediates_);
      for (uint32_t i = 0; i < num_intermediates_; i++) {
        intermediate_bufs_[i] = device_->Alloc(intermediate_sizes_[i]);
        TVM_FFI_ICHECK(intermediate_bufs_[i].Valid())
            << "Failed to allocate intermediate buffer " << i;
      }
    }

    // V7: allocate and upload constant data buffers.
    if (version_ == kRegcmdVersionV7 && num_constants_ > 0) {
      constant_bufs_.resize(num_constants_);
      for (uint32_t c = 0; c < num_constants_; c++) {
        constant_bufs_[c] = device_->Alloc(constant_data_[c].size());
        TVM_FFI_ICHECK(constant_bufs_[c].Valid())
            << "Failed to allocate constant buffer " << c;
        // Upload constant data at init time.
        std::memcpy(constant_bufs_[c].Data(), constant_data_[c].data(), constant_data_[c].size());
        device_->SyncToDevice(constant_bufs_[c]);
      }
    }

    // V7 PC-chaining: regcmd buffer must hold ALL sub-tasks concatenated.
    if (version_ == kRegcmdVersionV7 && total_regcmds_ > 0) {
      regcmd_size = static_cast<size_t>(total_regcmds_) * 8;
    }

    regcmd_buf_ = device_->Alloc(regcmd_size);
    TVM_FFI_ICHECK(regcmd_buf_.Valid()) << "Failed to allocate regcmd DMA buffer";

    task_buf_ = device_->Alloc(task_size, rknpu::kMemTask);
    TVM_FFI_ICHECK(task_buf_.Valid()) << "Failed to allocate task DMA buffer";

    if (num_sub_tasks_ > 0) {
      // V6 multi-task: patch sub-tasks at init time (addresses don't change between runs).
      for (auto& st : sub_tasks_) {
        PatchSubTaskRegcmds(st);
      }
    } else if (num_tiles_ <= 1) {
      // Single-tile: patch and copy once at init time.
      PatchRegcmds();
      std::memcpy(regcmd_buf_.Data(), regcmds_.data(), regcmd_size);
      BuildTaskStruct();
    }
    // Multi-tile: patching and submission happen per-tile in Run().
  }

  /*!
   * \brief Patch placeholder DMA addresses in regcmds with real DMA addresses.
   *
   * RegCmd format: [op:16][value:32][reg_offset:16] = 64 bits.
   * The value field (bits 47-16) contains the DMA address.
   */
  void PatchRegcmds() {
    for (const auto& reloc : relocations_) {
      TVM_FFI_ICHECK_LT(reloc.regcmd_index, num_regcmds_) << "Relocation index out of bounds";

      uint32_t new_addr = 0;
      switch (reloc.type) {
        case kRelocInput:
          new_addr = input_buf_.dma_addr;
          break;
        case kRelocWeight:
          new_addr = weight_buf_.dma_addr;
          break;
        case kRelocOutput:
          new_addr = output_buf_.dma_addr;
          break;
        case kRelocBias:
          TVM_FFI_ICHECK(bias_buf_.Valid()) << "Bias relocation but no bias buffer";
          new_addr = bias_buf_.dma_addr;
          break;
        default:
          LOG(FATAL) << "Unknown relocation type: " << reloc.type;
      }

      uint64_t cmd = regcmds_[reloc.regcmd_index];
      // Clear value field (bits 47-16), insert new address.
      cmd = (cmd & 0xFFFF00000000FFFFULL) | (static_cast<uint64_t>(new_addr) << 16);
      regcmds_[reloc.regcmd_index] = cmd;
    }
  }

  /*!
   * \brief Patch placeholder DMA addresses in a V6 sub-task's regcmds.
   */
  void PatchSubTaskRegcmds(SubTaskInfo& st) {
    for (const auto& reloc : st.relocations) {
      TVM_FFI_ICHECK_LT(reloc.regcmd_index, st.num_regcmds)
          << "Sub-task relocation index out of bounds";

      uint32_t new_addr = 0;
      if (reloc.type == kRelocInput) {
        new_addr = input_buf_.dma_addr;
      } else if (reloc.type == kRelocWeight) {
        new_addr = weight_buf_.dma_addr;
      } else if (reloc.type == kRelocOutput) {
        new_addr = output_buf_.dma_addr;
      } else if (reloc.type == kRelocBias) {
        TVM_FFI_ICHECK(bias_buf_.Valid()) << "Bias relocation but no bias buffer";
        new_addr = bias_buf_.dma_addr;
      } else if (reloc.type == kRelocIntermediate && version_ == kRegcmdVersionV6) {
        // V6: single intermediate buffer
        TVM_FFI_ICHECK(intermediate_buf_.Valid()) << "Intermediate relocation but no buffer";
        new_addr = intermediate_buf_.dma_addr;
      } else if (reloc.type >= kRelocIntermediate && reloc.type < kRelocConstant) {
        // V7: multi-intermediate buffer (type = 5 + buffer_index)
        uint32_t buf_idx = reloc.type - kRelocIntermediate;
        TVM_FFI_ICHECK_LT(buf_idx, static_cast<uint32_t>(intermediate_bufs_.size()))
            << "Intermediate buffer index " << buf_idx << " out of range";
        new_addr = intermediate_bufs_[buf_idx].dma_addr;
      } else if (reloc.type >= kRelocConstant) {
        // V7: constant buffer (type = 16 + constant_index)
        uint32_t const_idx = reloc.type - kRelocConstant;
        TVM_FFI_ICHECK_LT(const_idx, static_cast<uint32_t>(constant_bufs_.size()))
            << "Constant buffer index " << const_idx << " out of range";
        new_addr = constant_bufs_[const_idx].dma_addr;
      } else {
        LOG(FATAL) << "Unknown relocation type in sub-task: " << reloc.type;
      }

      uint64_t cmd = st.regcmds[reloc.regcmd_index];
      cmd = (cmd & 0xFFFF00000000FFFFULL) | (static_cast<uint64_t>(new_addr) << 16);
      st.regcmds[reloc.regcmd_index] = cmd;
    }
  }

  /*!
   * \brief Patch regcmds for a single M-tile, adding m_offset to input/output DMA addresses.
   *
   * Each tile's regcmds use placeholder addresses. This method replaces them with
   * real DMA addresses, offsetting input and output by m_offset rows in the NPU
   * feature layout (each row = C2 * sizeof(FP16) = 16 bytes).
   */
  void PatchRegcmdsTiled(uint32_t m_offset) {
    const uint32_t dma_row_bytes = 8 * 2;  // C2 * sizeof(float16) = 16 bytes per M row
    for (const auto& reloc : relocations_) {
      TVM_FFI_ICHECK_LT(reloc.regcmd_index, num_regcmds_) << "Relocation index out of bounds";
      uint32_t new_addr = 0;
      switch (reloc.type) {
        case kRelocInput:
          new_addr = input_buf_.dma_addr + m_offset * dma_row_bytes;
          break;
        case kRelocWeight:
          new_addr = weight_buf_.dma_addr;  // Same for all M-tiles
          break;
        case kRelocOutput:
          new_addr = output_buf_.dma_addr + m_offset * dma_row_bytes;
          break;
        case kRelocBias:
          TVM_FFI_ICHECK(bias_buf_.Valid()) << "Bias relocation but no bias buffer";
          new_addr = bias_buf_.dma_addr;  // Same for all tiles
          break;
        default:
          LOG(FATAL) << "Unknown relocation type: " << reloc.type;
      }
      uint64_t cmd = regcmds_[reloc.regcmd_index];
      cmd = (cmd & 0xFFFF00000000FFFFULL) | (static_cast<uint64_t>(new_addr) << 16);
      regcmds_[reloc.regcmd_index] = cmd;
    }
  }

  /*!
   * \brief Patch regcmds for a V4 tile with both M and N offsets.
   *
   * Input is offset by m_offset rows. Weight is offset by n_offset kernel groups.
   * Output is offset by both m_offset rows and n_offset surfaces.
   * Bias is offset by n_offset channels.
   */
  void PatchRegcmdsTiledV4(uint32_t m_offset, uint32_t n_offset) {
    const uint32_t dma_row_bytes = 8 * 2;  // C2 * sizeof(float16) = 16 bytes per M row
    for (const auto& reloc : relocations_) {
      TVM_FFI_ICHECK_LT(reloc.regcmd_index, num_regcmds_) << "Relocation index out of bounds";
      uint32_t new_addr = 0;
      switch (reloc.type) {
        case kRelocInput:
          new_addr = input_buf_.dma_addr + m_offset * dma_row_bytes;
          break;
        case kRelocWeight:
          new_addr = weight_buf_.dma_addr + n_offset * K_aligned_ * 2;
          break;
        case kRelocOutput:
          new_addr = output_buf_.dma_addr
                   + m_offset * dma_row_bytes
                   + n_offset * M_pad_ * 2;
          break;
        case kRelocBias:
          TVM_FFI_ICHECK(bias_buf_.Valid()) << "Bias relocation but no bias buffer";
          new_addr = bias_buf_.dma_addr + n_offset * 4;
          break;
        default:
          LOG(FATAL) << "Unknown relocation type: " << reloc.type;
      }
      uint64_t cmd = regcmds_[reloc.regcmd_index];
      cmd = (cmd & 0xFFFF00000000FFFFULL) | (static_cast<uint64_t>(new_addr) << 16);
      regcmds_[reloc.regcmd_index] = cmd;
    }
  }

  // =========================================================================
  // V8 graph-level execution
  // =========================================================================

  void InitV8() {
    device_ = std::make_unique<rknpu::RKNPUDevice>();

    // Allocate ALL DMA buffers from buffer table.
    v8_bufs_.resize(v8_buf_sizes_.size());
    for (size_t i = 0; i < v8_buf_sizes_.size(); i++) {
      v8_bufs_[i] = device_->Alloc(v8_buf_sizes_[i]);
      TVM_FFI_ICHECK(v8_bufs_[i].Valid())
          << "V8: failed to allocate buffer " << i << " (" << v8_buf_sizes_[i] << " bytes)";
    }

    // Upload constants (scatter/pack as appropriate).
    for (const auto& c : v8_constants_) {
      auto& buf = v8_bufs_[c.buf_idx];
      int dim0 = static_cast<int>(c.dim0);
      int dim1 = static_cast<int>(c.dim1);
      if (c.scatter_type == kV8ScatterFeature) {
        // Feature scatter: [M, K] → feature layout
        ScatterInputFP16(buf.As<uint16_t>(),
                         reinterpret_cast<const uint16_t*>(c.data.data()),
                         dim0, dim1, PadM(dim0), AlignUp(dim1, 32));
      } else if (c.scatter_type == kV8ScatterWeight) {
        // Weight pack: [K, N] → weight layout
        PackWeightsFP16(buf.As<uint16_t>(),
                        reinterpret_cast<const uint16_t*>(c.data.data()),
                        dim0, dim1);
      } else if (c.scatter_type == kV8ScatterBiasFP32) {
        // Bias: FP16 → FP32
        ConvertBiasFP16ToFP32(buf.As<float>(),
                               reinterpret_cast<const uint16_t*>(c.data.data()),
                               dim0, AlignUp(dim0, 16));
      }
      device_->SyncToDevice(buf);
    }

    // Patch all NPU task regcmds with DMA addresses (one-time at init).
    for (auto& seg : v8_segments_) {
      if (seg.type == kV8SegNPU) {
        for (auto& task : seg.npu_tasks) {
          for (const auto& reloc : task.relocations) {
            TVM_FFI_ICHECK_LT(reloc.regcmd_index, task.num_regcmds)
                << "V8 relocation index out of bounds";
            TVM_FFI_ICHECK_LT(reloc.buf_idx, static_cast<uint32_t>(v8_bufs_.size()))
                << "V8 buffer index out of bounds: " << reloc.buf_idx;
            uint32_t new_addr = v8_bufs_[reloc.buf_idx].dma_addr + reloc.byte_offset;
            uint64_t cmd = task.regcmds[reloc.regcmd_index];
            cmd = (cmd & 0xFFFF00000000FFFFULL) | (static_cast<uint64_t>(new_addr) << 16);
            task.regcmds[reloc.regcmd_index] = cmd;
          }
        }
      }
    }

    // Allocate regcmd buffer large enough for the largest NPU segment.
    // Also track the max number of tasks in any segment for the task buffer.
    size_t max_segment_regcmds = 0;
    size_t max_segment_tasks = 1;
    for (const auto& seg : v8_segments_) {
      if (seg.type == kV8SegNPU) {
        size_t seg_total = 0;
        for (const auto& task : seg.npu_tasks) {
          seg_total += task.num_regcmds;
        }
        if (seg_total > max_segment_regcmds) {
          max_segment_regcmds = seg_total;
        }
        if (seg.npu_tasks.size() > max_segment_tasks) {
          max_segment_tasks = seg.npu_tasks.size();
        }
      }
    }
    if (max_segment_regcmds > 0) {
      regcmd_buf_ = device_->Alloc(max_segment_regcmds * 8);
      TVM_FFI_ICHECK(regcmd_buf_.Valid()) << "V8: failed to allocate regcmd buffer";
    }

    // Allocate task buffer for N task structs (one per task in the segment).
    task_buf_ = device_->Alloc(max_segment_tasks * sizeof(rknpu::RknpuTask), rknpu::kMemTask);
    TVM_FFI_ICHECK(task_buf_.Valid()) << "V8: failed to allocate task buffer";

    LOG(INFO) << "RKNPURuntime::InitV8 completed for " << symbol_name_
              << " (" << v8_bufs_.size() << " buffers, " << v8_segments_.size()
              << " segments, " << total_regcmds_ << " total regcmds)";
  }

  void RunV8() {

    // 1. Scatter external inputs to their assigned buffers.
    for (size_t i = 0; i < v8_ext_inputs_.size(); i++) {
      const auto& ei = v8_ext_inputs_[i];
      const DLTensor* tensor = data_entry_[input_var_eid_[ei.param_idx]];
      auto& buf = v8_bufs_[ei.buf_idx];
      int dim0 = static_cast<int>(ei.dim0);
      int dim1 = static_cast<int>(ei.dim1);

      if (ei.scatter_type == kV8ScatterFeature) {
        ScatterInputFP16(buf.As<uint16_t>(),
                         static_cast<const uint16_t*>(tensor->data),
                         dim0, dim1, PadM(dim0), AlignUp(dim1, 32));
      } else if (ei.scatter_type == kV8ScatterWeight) {
        PackWeightsFP16(buf.As<uint16_t>(),
                        static_cast<const uint16_t*>(tensor->data),
                        dim0, dim1);
      } else if (ei.scatter_type == kV8ScatterBiasFP32) {
        ConvertBiasFP16ToFP32(buf.As<float>(),
                               static_cast<const uint16_t*>(tensor->data),
                               dim0, AlignUp(dim0, 16));
      }
      device_->SyncToDevice(buf);
    }

    // Flush ALL V8 buffers to device so CPU cache lines are clean before NPU DMA.
    for (auto& buf : v8_bufs_) {
      device_->SyncToDevice(buf);
    }

    // 2. Execute segments.
    for (size_t si = 0; si < v8_segments_.size(); si++) {
      const auto& seg = v8_segments_[si];
      if (seg.type == kV8SegNPU) {
        // Submit NPU tasks in PC-chained batches.
        // Hardware limits observed: max 6 tasks per PC chain on RK3588.
        const uint32_t kMaxBatchSize = 1;
        uint32_t total_tasks = static_cast<uint32_t>(seg.npu_tasks.size());

        for (uint32_t batch_start = 0; batch_start < total_tasks;
             batch_start += kMaxBatchSize) {
          uint32_t batch_size =
              std::min(kMaxBatchSize, total_tasks - batch_start);

          // Copy this batch's regcmds to regcmd_buf.
          auto* dst = static_cast<uint8_t*>(regcmd_buf_.Data());
          size_t byte_offset = 0;
          for (uint32_t t = batch_start; t < batch_start + batch_size; t++) {
            const auto& task = seg.npu_tasks[t];
            size_t blob_bytes = static_cast<size_t>(task.num_regcmds) * 8;
            std::memcpy(dst + byte_offset, task.regcmds.data(), blob_bytes);
            byte_offset += blob_bytes;
          }

          // Patch PC tails to chain consecutive tasks within this batch.
          auto* regcmd_ptr = reinterpret_cast<uint64_t*>(regcmd_buf_.Data());
          uint32_t cmd_offset = 0;
          for (uint32_t t = 0; t + 1 < batch_size; t++) {
            const auto& task = seg.npu_tasks[batch_start + t];
            uint32_t next_cmd_offset = cmd_offset + task.num_regcmds;
            uint32_t next_dma_addr =
                regcmd_buf_.dma_addr + next_cmd_offset * 8;
            uint32_t next_n_cmds =
                seg.npu_tasks[batch_start + t + 1].num_regcmds;

            uint32_t pc_tail_base = cmd_offset + task.num_regcmds - 4;
            regcmd_ptr[pc_tail_base] =
                (static_cast<uint64_t>(kOpRegPc) << 48) |
                (static_cast<uint64_t>(next_dma_addr) << 16) |
                kPcBaseAddress;
            regcmd_ptr[pc_tail_base + 1] =
                (static_cast<uint64_t>(kOpRegPc) << 48) |
                (static_cast<uint64_t>(next_n_cmds / 2 - 1) << 16) |
                kPcRegisterAmounts;

            cmd_offset += task.num_regcmds;
          }

          // Build task structs for this batch.
          auto* task_arr = task_buf_.As<rknpu::RknpuTask>();
          uint32_t regcmd_off = 0;
          for (uint32_t t = 0; t < batch_size; t++) {
            const auto& npu_task = seg.npu_tasks[batch_start + t];
            auto* ts = &task_arr[t];
            std::memset(ts, 0, sizeof(rknpu::RknpuTask));
            ts->flags = 0;
            ts->op_idx = 0;
            ts->enable_mask = npu_task.enable_mask;
            ts->int_mask = npu_task.int_mask;
            ts->int_clear = 0x1FFFF;
            ts->int_status = 0;
            ts->regcfg_amount = npu_task.regcfg_amount;
            ts->regcfg_offset = 0;
            ts->regcmd_addr = regcmd_buf_.dma_addr + regcmd_off * 8;
            regcmd_off += npu_task.num_regcmds;
          }

          // Sync and submit this batch.
          device_->SyncToDevice(regcmd_buf_);
          device_->SyncToDevice(task_buf_);
          device_->Submit(task_buf_, /*core_mask=*/1,
                          /*num_tasks=*/batch_size);
        }

      } else {
        // CPU segment: sync required buffers, compute, sync back.
        for (const auto& ct : seg.cpu_tasks) {
          auto& in_buf = v8_bufs_[ct.in_buf];
          auto& out_buf = v8_bufs_[ct.out_buf];
          int M = static_cast<int>(ct.M);
          int K = static_cast<int>(ct.K);

          // Sync input buffer from device.
          device_->SyncFromDevice(in_buf);

          if (ct.op_type == kV8CpuMaxReduce) {
            // Row-wise max of [M, K] in feature layout → [M, 1] in feature layout.
            // The data is in NPU feature layout [G, M_pad, 8] where G = K_al/8.
            // We gather to temp, compute max, negate, scatter to output.
            int K_al = AlignUp(K, 32);
            int M_pad = PadM(M);
            std::vector<uint16_t> temp(M * K);
            GatherOutputFP16(temp.data(), in_buf.As<const uint16_t>(), M, K, M_pad, K_al);

            // Compute row-wise max and negate (for subtract via add).
            std::vector<uint16_t> neg_max(M);
            for (int m = 0; m < M; m++) {
              float max_val = -65504.0f;  // FP16 min
              for (int k = 0; k < K; k++) {
                uint16_t h = temp[m * K + k];
                // Quick FP16→float
                uint32_t sign = (static_cast<uint32_t>(h) & 0x8000) << 16;
                uint32_t exp = (h >> 10) & 0x1F;
                uint32_t mant = h & 0x3FF;
                uint32_t f;
                if (exp == 0) f = sign;
                else if (exp == 31) f = sign | 0x7F800000 | (mant << 13);
                else f = sign | ((exp + 127 - 15) << 23) | (mant << 13);
                float val;
                std::memcpy(&val, &f, 4);
                if (val > max_val) max_val = val;
              }
              // Negate the max (so adding it = subtracting)
              float neg = -max_val;
              // Float→FP16
              uint32_t fu;
              std::memcpy(&fu, &neg, 4);
              uint32_t fs = (fu >> 16) & 0x8000;
              int32_t fe = ((fu >> 23) & 0xFF) - 127 + 15;
              uint32_t fm = (fu >> 13) & 0x3FF;
              if (fe >= 31) { neg_max[m] = static_cast<uint16_t>(fs | 0x7C00); }
              else if (fe <= 0) { neg_max[m] = static_cast<uint16_t>(fs); }
              else { neg_max[m] = static_cast<uint16_t>(fs | (fe << 10) | fm); }
            }

            // Scatter [M, 1] to feature layout in output buffer.
            int K_out = 1;
            int K_out_al = AlignUp(K_out, 32);
            int M_pad_out = PadM(M);
            std::memset(out_buf.Data(), 0, out_buf.size);
            ScatterInputFP16(out_buf.As<uint16_t>(), neg_max.data(),
                             M, K_out, M_pad_out, K_out_al);
            device_->SyncToDevice(out_buf);

          } else if (ct.op_type == kV8CpuReciprocal) {
            // Element-wise 1/x on [M, K] in feature layout.
            int K_al = AlignUp(K, 32);
            int M_pad = PadM(M);
            std::vector<uint16_t> temp(M * K);
            GatherOutputFP16(temp.data(), in_buf.As<const uint16_t>(), M, K, M_pad, K_al);

            for (int i = 0; i < M * K; i++) {
              uint16_t h = temp[i];
              // FP16→float
              uint32_t sign = (static_cast<uint32_t>(h) & 0x8000) << 16;
              uint32_t exp = (h >> 10) & 0x1F;
              uint32_t mant = h & 0x3FF;
              uint32_t f;
              if (exp == 0) f = sign;
              else if (exp == 31) f = sign | 0x7F800000 | (mant << 13);
              else f = sign | ((exp + 127 - 15) << 23) | (mant << 13);
              float val;
              std::memcpy(&val, &f, 4);
              // Reciprocal
              float inv = (val != 0.0f) ? 1.0f / val : 0.0f;
              // Float→FP16
              uint32_t fu;
              std::memcpy(&fu, &inv, 4);
              uint32_t fs = (fu >> 16) & 0x8000;
              int32_t fe = ((fu >> 23) & 0xFF) - 127 + 15;
              uint32_t fm = (fu >> 13) & 0x3FF;
              if (fe >= 31) temp[i] = static_cast<uint16_t>(fs | 0x7C00);
              else if (fe <= 0) temp[i] = static_cast<uint16_t>(fs);
              else temp[i] = static_cast<uint16_t>(fs | (fe << 10) | fm);
            }

            ScatterInputFP16(out_buf.As<uint16_t>(), temp.data(),
                             M, K, PadM(M), AlignUp(K, 32));
            device_->SyncToDevice(out_buf);
          }
        }
      }
    }

    // 3. Gather external outputs.
    for (size_t oi = 0; oi < v8_ext_outputs_.size(); oi++) {
      const auto& eo = v8_ext_outputs_[oi];
      const DLTensor* output_tensor = data_entry_[EntryID(outputs_[eo.output_idx])];
      auto& buf = v8_bufs_[eo.buf_idx];
      int dim0 = static_cast<int>(eo.dim0);
      int dim1 = static_cast<int>(eo.dim1);

      device_->SyncFromDevice(buf);

      if (eo.gather_type == kV8GatherFeature) {
        GatherOutputFP16(static_cast<uint16_t*>(output_tensor->data),
                         buf.As<const uint16_t>(), dim0, dim1,
                         PadM(dim0), AlignUp(dim1, 32));
      }
    }
  }

  /*! \brief Build the RknpuTask struct in the task DMA buffer. */
  void BuildTaskStruct() {
    auto* task = task_buf_.As<rknpu::RknpuTask>();
    std::memset(task, 0, sizeof(rknpu::RknpuTask));
    task->flags = 0;
    task->op_idx = 0;
    task->enable_mask = enable_mask_;
    task->int_mask = int_mask_;
    task->int_clear = 0x1FFFF;
    task->int_status = 0;
    task->regcfg_amount = regcfg_amount_;
    task->regcfg_offset = 0;
    task->regcmd_addr = regcmd_buf_.dma_addr;
  }
#endif  // TVM_RKNPU_RUNTIME
};

// ---------------------------------------------------------------------------
// Experimental TIR runtime bridge (extern symbols for call_extern)
// ---------------------------------------------------------------------------

namespace {

static constexpr int kBridgeStageMatmul = 1;
static constexpr int kBridgeStageAdd = 2;
static constexpr int kBridgeStageRelu = 3;
static constexpr int kBridgeStageRelu4D = 4;
static constexpr int kBridgeStageConv2D = 5;
static constexpr int kBridgeStageMatmulBiasRelu = 6;
static constexpr int kBridgeStageAddRelu = 7;
static constexpr int kBridgeStageConv2DRelu = 8;
static constexpr int kBridgeStageMul = 9;
static constexpr int kBridgeStageExp = 10;
static constexpr int kBridgeStageMatmulBias = 11;
static constexpr int kBridgeStageReciprocal = 12;
static constexpr int kBridgeStageGelu = 13;

static inline bool BridgeTouchEnabled() {
  static int enabled = []() -> int {
    const char* env = std::getenv("TVM_RKNPU_BRIDGE_TOUCH");
    if (!env) return 0;
    if (std::strcmp(env, "1") == 0 || std::strcmp(env, "true") == 0 || std::strcmp(env, "on") == 0 ||
        std::strcmp(env, "yes") == 0) {
      return 1;
    }
    return 0;
  }();
  return enabled != 0;
}

static inline bool BridgeRealSubmitEnabled() {
  static int enabled = []() -> int {
    const char* env = std::getenv("TVM_RKNPU_BRIDGE_REAL_SUBMIT");
    if (!env) return 0;
    if (std::strcmp(env, "1") == 0 || std::strcmp(env, "true") == 0 || std::strcmp(env, "on") == 0 ||
        std::strcmp(env, "yes") == 0) {
      return 1;
    }
    return 0;
  }();
  return enabled != 0;
}

static inline bool BridgeLogEnabled() {
  static int enabled = []() -> int {
    const char* env = std::getenv("TVM_RKNPU_BRIDGE_LOG");
    if (!env) return 0;
    if (std::strcmp(env, "1") == 0 || std::strcmp(env, "true") == 0 || std::strcmp(env, "on") == 0 ||
        std::strcmp(env, "yes") == 0) {
      return 1;
    }
    return 0;
  }();
  return enabled != 0;
}

static inline bool BridgeUseRelocSubmit() {
  static int enabled = []() -> int {
    const char* env = std::getenv("TVM_RKNPU_BRIDGE_USE_RELOCS");
    if (!env) return 0;
    if (std::strcmp(env, "1") == 0 || std::strcmp(env, "true") == 0 || std::strcmp(env, "on") == 0 ||
        std::strcmp(env, "yes") == 0) {
      return 1;
    }
    return 0;
  }();
  return enabled != 0;
}

static inline bool BridgeValidateRelocSemanticsEnabled() {
  static int enabled = []() -> int {
    const char* env = std::getenv("TVM_RKNPU_BRIDGE_VALIDATE_RELOCS");
    if (!env) return 0;
    if (std::strcmp(env, "1") == 0 || std::strcmp(env, "true") == 0 || std::strcmp(env, "on") == 0 ||
        std::strcmp(env, "yes") == 0) {
      return 1;
    }
    return 0;
  }();
  return enabled != 0;
}

static inline bool BridgeOutputChecksEnabled() {
  static int enabled = []() -> int {
    const char* env = std::getenv("TVM_RKNPU_BRIDGE_CHECK_OUTPUTS");
    if (!env) return 0;
    if (std::strcmp(env, "1") == 0 || std::strcmp(env, "true") == 0 || std::strcmp(env, "on") == 0 ||
        std::strcmp(env, "yes") == 0) {
      return 1;
    }
    return 0;
  }();
  return enabled != 0;
}

static inline bool BridgeRunCpuAfterSubmitEnabled() {
  static int enabled = []() -> int {
    const char* env = std::getenv("TVM_RKNPU_BRIDGE_RUN_CPU_AFTER_SUBMIT");
    if (!env) return 0;  // default to strict NPU-only behavior unless explicitly enabled
    if (std::strcmp(env, "0") == 0 || std::strcmp(env, "false") == 0 || std::strcmp(env, "off") == 0 ||
        std::strcmp(env, "no") == 0) {
      return 0;
    }
    return 1;
  }();
  return enabled != 0;
}

static inline bool BridgeFailOnFallbackEnabled() {
  static int enabled = []() -> int {
    const char* env = std::getenv("TVM_RKNPU_BRIDGE_FAIL_ON_FALLBACK");
    if (!env) return 0;
    if (std::strcmp(env, "1") == 0 || std::strcmp(env, "true") == 0 || std::strcmp(env, "on") == 0 ||
        std::strcmp(env, "yes") == 0) {
      return 1;
    }
    return 0;
  }();
  return enabled != 0;
}

static inline bool BridgeDumpRegcmdsEnabled() {
  static int enabled = []() -> int {
    const char* env = std::getenv("TVM_RKNPU_BRIDGE_DUMP_REGCMDS");
    if (!env) return 0;
    if (std::strcmp(env, "1") == 0 || std::strcmp(env, "true") == 0 || std::strcmp(env, "on") == 0 ||
        std::strcmp(env, "yes") == 0) {
      return 1;
    }
    return 0;
  }();
  return enabled != 0;
}

static inline bool BridgeCompareNpuCpuOutputEnabled() {
  static int enabled = []() -> int {
    const char* env = std::getenv("TVM_RKNPU_BRIDGE_COMPARE_NPU_CPU_OUTPUT");
    if (!env) return 0;
    if (std::strcmp(env, "1") == 0 || std::strcmp(env, "true") == 0 || std::strcmp(env, "on") == 0 ||
        std::strcmp(env, "yes") == 0) {
      return 1;
    }
    return 0;
  }();
  return enabled != 0;
}

static inline bool BridgePackMatmulWeightsEnabled() {
  static int enabled = []() -> int {
    const char* env = std::getenv("TVM_RKNPU_BRIDGE_PACK_MATMUL_WEIGHTS");
    if (!env) return 1;
    if (std::strcmp(env, "0") == 0 || std::strcmp(env, "false") == 0 || std::strcmp(env, "off") == 0 ||
        std::strcmp(env, "no") == 0) {
      return 0;
    }
    return 1;
  }();
  return enabled != 0;
}

static inline bool BridgeCacheTransformsEnabled() {
  static int enabled = []() -> int {
    const char* env = std::getenv("TVM_RKNPU_BRIDGE_CACHE_TRANSFORMS");
    // Default on: persistent transform cache keys include source-content checksums
    // to avoid stale pointer-reuse hits across invocations.
    if (!env) return 1;
    if (std::strcmp(env, "0") == 0 || std::strcmp(env, "false") == 0 || std::strcmp(env, "off") == 0 ||
        std::strcmp(env, "no") == 0) {
      return 0;
    }
    return 1;
  }();
  return enabled != 0;
}

static inline bool BridgePersistentDmaCacheEnabled() {
  static int enabled = []() -> int {
    const char* env = std::getenv("TVM_RKNPU_BRIDGE_CACHE_DMA");
    if (!env) return 1;
    if (std::strcmp(env, "0") == 0 || std::strcmp(env, "false") == 0 || std::strcmp(env, "off") == 0 ||
        std::strcmp(env, "no") == 0) {
      return 0;
    }
    return 1;
  }();
  return enabled != 0;
}

static inline size_t BridgePersistentDmaCacheMaxEntries() {
  static size_t max_entries = []() -> size_t {
    const char* env = std::getenv("TVM_RKNPU_BRIDGE_CACHE_DMA_MAX_ENTRIES");
    if (!env || *env == '\0') return 512;
    char* end = nullptr;
    unsigned long long v = std::strtoull(env, &end, 10);
    if (end == env || v == 0) return 512;
    if (v > 8192ull) v = 8192ull;
    return static_cast<size_t>(v);
  }();
  return max_entries;
}

static inline bool BridgeAssumeInputsImmutableEnabled() {
  static int enabled = []() -> int {
    const char* env = std::getenv("TVM_RKNPU_BRIDGE_ASSUME_INPUTS_IMMUTABLE");
    if (!env) return 0;
    if (std::strcmp(env, "1") == 0 || std::strcmp(env, "true") == 0 || std::strcmp(env, "on") == 0 ||
        std::strcmp(env, "yes") == 0) {
      return 1;
    }
    return 0;
  }();
  return enabled != 0;
}

static inline bool BridgeDebugDmaEnabled() {
  static int enabled = []() -> int {
    const char* env = std::getenv("TVM_RKNPU_BRIDGE_DEBUG_DMA");
    if (!env) return 0;
    if (std::strcmp(env, "1") == 0 || std::strcmp(env, "true") == 0 || std::strcmp(env, "on") == 0 ||
        std::strcmp(env, "yes") == 0) {
      return 1;
    }
    return 0;
  }();
  return enabled != 0;
}

struct BridgeTransformCache {
  std::mutex mu;
  std::unordered_map<uint64_t, std::vector<uint8_t>> blobs;
};

static BridgeTransformCache& GetBridgeTransformCache() {
  static BridgeTransformCache cache;
  return cache;
}

struct BridgePersistentDmaCache {
  std::unordered_map<void*, rknpu::DMABuffer> by_host_ptr;
  std::unordered_map<void*, bool> uploaded;
};

static BridgePersistentDmaCache& GetBridgePersistentDmaCache() {
  static thread_local BridgePersistentDmaCache cache;
  return cache;
}

static inline size_t BridgeScratchBytes() {
  static size_t bytes = []() -> size_t {
    constexpr size_t kDefault = 32u * 1024u * 1024u;
    const char* env = std::getenv("TVM_RKNPU_BRIDGE_SCRATCH_BYTES");
    if (!env || *env == '\0') {
      return kDefault;
    }
    char* end = nullptr;
    unsigned long long v = std::strtoull(env, &end, 10);
    if (end == env || v == 0) {
      return kDefault;
    }
    // Keep a conservative minimum so common medium workloads still work.
    if (v < 4ull * 1024ull * 1024ull) {
      v = 4ull * 1024ull * 1024ull;
    }
    return static_cast<size_t>(v);
  }();
  return bytes;
}

static inline float HalfToFloat(uint16_t h) {
  uint32_t sign = (static_cast<uint32_t>(h) & 0x8000) << 16;
  uint32_t exp = (h >> 10) & 0x1F;
  uint32_t mant = h & 0x3FF;
  uint32_t f;
  if (exp == 0) {
    if (mant == 0) {
      f = sign;
    } else {
      exp = 1;
      while ((mant & 0x400) == 0) {
        mant <<= 1;
        exp--;
      }
      mant &= 0x3FF;
      f = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
  } else if (exp == 31) {
    f = sign | 0x7F800000 | (mant << 13);
  } else {
    f = sign | ((exp + 127 - 15) << 23) | (mant << 13);
  }
  float out;
  std::memcpy(&out, &f, sizeof(float));
  return out;
}

static inline void TouchBridgeDevice(size_t bytes) {
#ifdef TVM_RKNPU_RUNTIME
  if (!BridgeTouchEnabled()) {
    return;
  }
  static thread_local std::unique_ptr<rknpu::RKNPUDevice> dev;
  if (!dev) {
    dev = std::make_unique<rknpu::RKNPUDevice>();
  }
  // Small per-stage scratch allocation so traces show alloc/sync/free activity.
  rknpu::DMABuffer tmp = dev->Alloc(bytes < 256 ? 256 : bytes);
  std::memset(tmp.As<void>(), 0, tmp.size);
  dev->SyncToDevice(tmp);
  dev->Reset();
  dev->Free(tmp);
#else
  (void)bytes;
#endif
}

static inline void DumpRegcmdSample(const char* tag, const uint64_t* regs, size_t n) {
  if (!BridgeDumpRegcmdsEnabled() || regs == nullptr || n == 0) return;
  std::fprintf(stderr, "%s n_regcmds=%zu first=0x%016llx second=0x%016llx\n", tag, n,
               static_cast<unsigned long long>(regs[0]),
               static_cast<unsigned long long>(n > 1 ? regs[1] : 0ULL));
  size_t limit = n < 20 ? n : 20;
  std::ostringstream os;
  os << tag << " n_regcmds=" << n;
  for (size_t i = 0; i < limit; ++i) {
    uint64_t u = regs[i];
    uint16_t reg = static_cast<uint16_t>(u & 0xFFFF);
    uint32_t addr = static_cast<uint32_t>((u >> 16) & 0xFFFFFFFFULL);
    os << " [" << i << ":reg=0x" << std::hex << reg << ",addr=0x" << addr
       << ",u64=0x" << u << std::dec << "]";
  }
  LOG(INFO) << os.str();
}

static inline int BridgeComputeMatmul(const uint16_t* A, const uint16_t* B, uint16_t* C, int m, int k,
                                      int n) {
  if (!A || !B || !C || m <= 0 || k <= 0 || n <= 0) return -1;
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      float acc = 0.0f;
      for (int p = 0; p < k; ++p) {
        acc += HalfToFloat(A[i * k + p]) * HalfToFloat(B[p * n + j]);
      }
      C[i * n + j] = FloatToHalf(acc);
    }
  }
  return 0;
}

static inline int BridgeComputeAdd(const uint16_t* A, const uint16_t* B, uint16_t* C, int m, int n,
                                   int b_mode) {
  if (!A || !B || !C || m <= 0 || n <= 0) return -1;
  if (b_mode == 1) {
    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < n; ++j) {
        float v = HalfToFloat(A[i * n + j]) + HalfToFloat(B[j]);
        C[i * n + j] = FloatToHalf(v);
      }
    }
  } else if (b_mode == 2) {
    for (int i = 0; i < m; ++i) {
      float bv = HalfToFloat(B[i]);
      for (int j = 0; j < n; ++j) {
        float v = HalfToFloat(A[i * n + j]) + bv;
        C[i * n + j] = FloatToHalf(v);
      }
    }
  } else {
    for (int i = 0; i < m * n; ++i) {
      float v = HalfToFloat(A[i]) + HalfToFloat(B[i]);
      C[i] = FloatToHalf(v);
    }
  }
  return 0;
}

static inline int BridgeComputeMul(const uint16_t* A, const uint16_t* B, uint16_t* C, int m, int n,
                                   int b_mode) {
  if (A == nullptr || B == nullptr || C == nullptr || m <= 0 || n <= 0) return -1;
  for (int i = 0; i < m; ++i) {
    const uint16_t* a_row = A + static_cast<size_t>(i) * n;
    const uint16_t* b_row =
        (b_mode == 1) ? B : (b_mode == 2) ? (B + static_cast<size_t>(i)) : (B + static_cast<size_t>(i) * n);
    uint16_t* c_row = C + static_cast<size_t>(i) * n;
    for (int j = 0; j < n; ++j) {
      float av = HalfToFloat(a_row[j]);
      float bv = HalfToFloat((b_mode == 2) ? b_row[0] : b_row[j]);
      c_row[j] = FloatToHalf(av * bv);
    }
  }
  return 0;
}

static inline int BridgeComputeRelu(const uint16_t* A, uint16_t* C, int m, int n) {
  if (!A || !C || m <= 0 || n <= 0) return -1;
  for (int i = 0; i < m * n; ++i) {
    float v = HalfToFloat(A[i]);
    if (v < 0.0f) v = 0.0f;
    C[i] = FloatToHalf(v);
  }
  return 0;
}

static inline int BridgeComputeExp(const uint16_t* A, uint16_t* C, int m, int n) {
  if (!A || !C || m <= 0 || n <= 0) return -1;
  for (int i = 0; i < m * n; ++i) {
    float v = std::exp(HalfToFloat(A[i]));
    C[i] = FloatToHalf(v);
  }
  return 0;
}

static inline int BridgeComputeReciprocal(const uint16_t* A, uint16_t* C, int m, int n) {
  if (!A || !C || m <= 0 || n <= 0) return -1;
  for (int i = 0; i < m * n; ++i) {
    float av = HalfToFloat(A[i]);
    float v = (av == 0.0f) ? std::numeric_limits<float>::infinity() : (1.0f / av);
    C[i] = FloatToHalf(v);
  }
  return 0;
}

static inline int BridgeComputeGelu(const uint16_t* A, uint16_t* C, int m, int n) {
  if (!A || !C || m <= 0 || n <= 0) return -1;
  for (int i = 0; i < m * n; ++i) {
    float av = HalfToFloat(A[i]);
    float v = 0.5f * av * (1.0f + std::erff(av * 0.7071067811865475f));
    C[i] = FloatToHalf(v);
  }
  return 0;
}

static inline int BridgeComputeRelu4D(const uint16_t* A, uint16_t* C, int n, int ch, int h, int w) {
  if (!A || !C || n <= 0 || ch <= 0 || h <= 0 || w <= 0) return -1;
  int size = n * ch * h * w;
  for (int i = 0; i < size; ++i) {
    float v = HalfToFloat(A[i]);
    if (v < 0.0f) v = 0.0f;
    C[i] = FloatToHalf(v);
  }
  return 0;
}

static inline int BridgeComputeConv2D(const uint16_t* X, const uint16_t* W, uint16_t* Y, int n, int c,
                                      int h, int w, int oc, int kh, int kw, int oh, int ow, int sh,
                                      int sw, int pt, int pl) {
  if (!X || !W || !Y) return -1;
  if (n <= 0 || c <= 0 || h <= 0 || w <= 0 || oc <= 0 || kh <= 0 || kw <= 0 || oh <= 0 || ow <= 0 ||
      sh <= 0 || sw <= 0) {
    return -1;
  }
  for (int bn = 0; bn < n; ++bn) {
    for (int ocv = 0; ocv < oc; ++ocv) {
      for (int oy = 0; oy < oh; ++oy) {
        for (int ox = 0; ox < ow; ++ox) {
          float acc = 0.0f;
          for (int ic = 0; ic < c; ++ic) {
            for (int ky = 0; ky < kh; ++ky) {
              for (int kx = 0; kx < kw; ++kx) {
                int iy = oy * sh + ky - pt;
                int ix = ox * sw + kx - pl;
                if (iy < 0 || iy >= h || ix < 0 || ix >= w) continue;
                int x_idx = ((bn * c + ic) * h + iy) * w + ix;
                int w_idx = ((ocv * c + ic) * kh + ky) * kw + kx;
                acc += HalfToFloat(X[x_idx]) * HalfToFloat(W[w_idx]);
              }
            }
          }
          int y_idx = ((bn * oc + ocv) * oh + oy) * ow + ox;
          Y[y_idx] = FloatToHalf(acc);
        }
      }
    }
  }
  return 0;
}

struct BridgeStats {
  struct HostDmaDebugEntry {
    uint32_t submit_slot{0};
    uintptr_t host_ptr{0};
    uint32_t dma_addr{0};
    int64_t bytes{0};
    bool write_back{false};
    bool persistent_cached{false};
    bool persistent_cache_hit{false};
    bool upload_skipped{false};
    bool sync_to_device_requested{false};
    bool chain_reused{false};
  };
  struct SubmitTimingBucket {
    uint32_t submit_slot{0};
    uint32_t stage_count{0};
    uint32_t task_count{0};
    int64_t calls{0};
    int64_t ok_calls{0};
    int64_t fail_calls{0};
    int64_t total_ns{0};
    int64_t prep_ns{0};
    int64_t submit_ns{0};
    int64_t post_ns{0};
    int64_t sync_to_device_ns{0};
    int64_t sync_from_device_ns{0};
    int64_t hw_elapsed_ns{0};
    int64_t min_total_ns{0};
    int64_t max_total_ns{0};
    int64_t min_hw_elapsed_ns{0};
    int64_t max_hw_elapsed_ns{0};
    int64_t pc_tail_patch_writes{0};
    int64_t reloc_writeback_entries{0};
    int64_t host_dma_buffers{0};
    int64_t writeback_dma_buffers{0};
    int64_t chain_reuse_hits{0};
    int64_t chain_reuse_bytes{0};
    int64_t data_sync_to_device_bytes{0};
    int64_t data_sync_from_device_bytes{0};
    int64_t meta_sync_to_device_bytes{0};
  };
  int64_t chain_calls{0};
  int64_t real_submit_ok{0};
  int64_t real_submit_fail{0};
  int64_t touch_fallback{0};
  int64_t submitted_tasks{0};
  int64_t submitted_regcmd_qwords{0};
  int64_t submitted_regcmd_bytes{0};
  int64_t patch_addr_writes{0};
  int64_t patch_pc_tail_writes{0};
  int64_t reloc_submit_calls{0};
  int64_t reloc_submit_fallbacks{0};
  int64_t reloc_entries_patched{0};
  int64_t reloc_writeback_entries{0};
  int64_t reloc_semantic_mismatch{0};
  int64_t reloc_range_mismatch{0};
  int64_t output_checksum_checks{0};
  int64_t output_checksum_mismatch{0};
  int64_t output_checksum_last{0};
  int64_t host_dma_buffers{0};
  int64_t writeback_dma_buffers{0};
  int64_t chain_reuse_hits{0};
  int64_t chain_reuse_bytes{0};
  int64_t data_sync_to_device_bytes{0};
  int64_t data_sync_from_device_bytes{0};
  int64_t meta_sync_to_device_bytes{0};
  std::vector<HostDmaDebugEntry> host_dma_debug;
  std::vector<SubmitTimingBucket> submit_timing_buckets;
};

struct BridgeStageInvocation {
  int stage{0};
  std::vector<void*> ptrs;
  std::vector<size_t> ptr_sizes;
  std::vector<int> ints;
};

static inline BridgeStats& GetBridgeStats() {
  static thread_local BridgeStats s;
  return s;
}

static inline std::string BridgeStatsJSON() {
  const BridgeStats& s = GetBridgeStats();
  auto hex_u64 = [](uint64_t v) -> std::string {
    std::ostringstream os_hex;
    os_hex << "0x" << std::hex << v;
    return os_hex.str();
  };
  std::ostringstream os;
  os << "{"
     << "\"chain_calls\":" << s.chain_calls << ","
     << "\"real_submit_ok\":" << s.real_submit_ok << ","
     << "\"real_submit_fail\":" << s.real_submit_fail << ","
     << "\"touch_fallback\":" << s.touch_fallback << ","
     << "\"submitted_tasks\":" << s.submitted_tasks << ","
     << "\"submitted_regcmd_qwords\":" << s.submitted_regcmd_qwords << ","
     << "\"submitted_regcmd_bytes\":" << s.submitted_regcmd_bytes << ","
     << "\"patch_addr_writes\":" << s.patch_addr_writes << ","
     << "\"patch_pc_tail_writes\":" << s.patch_pc_tail_writes << ","
     << "\"reloc_submit_calls\":" << s.reloc_submit_calls << ","
     << "\"reloc_submit_fallbacks\":" << s.reloc_submit_fallbacks << ","
     << "\"reloc_entries_patched\":" << s.reloc_entries_patched << ","
     << "\"reloc_writeback_entries\":" << s.reloc_writeback_entries << ","
     << "\"reloc_semantic_mismatch\":" << s.reloc_semantic_mismatch << ","
     << "\"reloc_range_mismatch\":" << s.reloc_range_mismatch << ","
     << "\"output_checksum_checks\":" << s.output_checksum_checks << ","
     << "\"output_checksum_mismatch\":" << s.output_checksum_mismatch << ","
     << "\"output_checksum_last\":" << s.output_checksum_last << ","
     << "\"host_dma_buffers\":" << s.host_dma_buffers << ","
     << "\"writeback_dma_buffers\":" << s.writeback_dma_buffers << ","
     << "\"chain_reuse_hits\":" << s.chain_reuse_hits << ","
     << "\"chain_reuse_bytes\":" << s.chain_reuse_bytes << ","
     << "\"data_sync_to_device_bytes\":" << s.data_sync_to_device_bytes << ","
     << "\"data_sync_from_device_bytes\":" << s.data_sync_from_device_bytes << ","
     << "\"meta_sync_to_device_bytes\":" << s.meta_sync_to_device_bytes << ","
     << "\"host_dma_debug\":[";
  for (size_t i = 0; i < s.host_dma_debug.size(); ++i) {
    const auto& d = s.host_dma_debug[i];
    if (i != 0) os << ",";
    os << "{"
       << "\"submit_slot\":" << d.submit_slot << ","
       << "\"host_ptr\":\"" << hex_u64(static_cast<uint64_t>(d.host_ptr)) << "\","
       << "\"dma_addr\":\"" << hex_u64(static_cast<uint64_t>(d.dma_addr)) << "\","
       << "\"bytes\":" << d.bytes << ","
       << "\"write_back\":" << (d.write_back ? "true" : "false") << ","
       << "\"persistent_cached\":" << (d.persistent_cached ? "true" : "false") << ","
       << "\"persistent_cache_hit\":" << (d.persistent_cache_hit ? "true" : "false") << ","
       << "\"upload_skipped\":" << (d.upload_skipped ? "true" : "false") << ","
       << "\"sync_to_device_requested\":" << (d.sync_to_device_requested ? "true" : "false") << ","
       << "\"chain_reused\":" << (d.chain_reused ? "true" : "false")
       << "}";
  }
  os << "],"
     << "\"submit_timing_buckets\":[";
  for (size_t i = 0; i < s.submit_timing_buckets.size(); ++i) {
    const auto& b = s.submit_timing_buckets[i];
    if (i != 0) os << ",";
    os << "{"
       << "\"submit_slot\":" << b.submit_slot << ","
       << "\"stage_count\":" << b.stage_count << ","
       << "\"task_count\":" << b.task_count << ","
       << "\"calls\":" << b.calls << ","
       << "\"ok_calls\":" << b.ok_calls << ","
       << "\"fail_calls\":" << b.fail_calls << ","
       << "\"total_ns\":" << b.total_ns << ","
       << "\"prep_ns\":" << b.prep_ns << ","
       << "\"submit_ns\":" << b.submit_ns << ","
       << "\"post_ns\":" << b.post_ns << ","
       << "\"sync_to_device_ns\":" << b.sync_to_device_ns << ","
       << "\"sync_from_device_ns\":" << b.sync_from_device_ns << ","
       << "\"hw_elapsed_ns\":" << b.hw_elapsed_ns << ","
       << "\"min_total_ns\":" << b.min_total_ns << ","
       << "\"max_total_ns\":" << b.max_total_ns << ","
       << "\"min_hw_elapsed_ns\":" << b.min_hw_elapsed_ns << ","
       << "\"max_hw_elapsed_ns\":" << b.max_hw_elapsed_ns << ","
       << "\"pc_tail_patch_writes\":" << b.pc_tail_patch_writes << ","
       << "\"reloc_writeback_entries\":" << b.reloc_writeback_entries << ","
       << "\"host_dma_buffers\":" << b.host_dma_buffers << ","
       << "\"writeback_dma_buffers\":" << b.writeback_dma_buffers << ","
       << "\"chain_reuse_hits\":" << b.chain_reuse_hits << ","
       << "\"chain_reuse_bytes\":" << b.chain_reuse_bytes << ","
       << "\"data_sync_to_device_bytes\":" << b.data_sync_to_device_bytes << ","
       << "\"data_sync_from_device_bytes\":" << b.data_sync_from_device_bytes << ","
       << "\"meta_sync_to_device_bytes\":" << b.meta_sync_to_device_bytes
       << "}";
  }
  os << "]"
     << "}";
  return os.str();
}

static inline void BridgeResetStats() { GetBridgeStats() = BridgeStats(); }

static inline void RecordBridgeSubmitTiming(uint32_t submit_slot, uint32_t stage_count,
                                            uint32_t task_count, bool ok, int64_t total_ns,
                                            int64_t prep_ns, int64_t submit_ns, int64_t post_ns,
                                            int64_t sync_to_device_ns, int64_t sync_from_device_ns,
                                            int64_t hw_elapsed_ns,
                                            int64_t pc_tail_patch_writes,
                                            int64_t reloc_writeback_entries,
                                            int64_t host_dma_buffers,
                                            int64_t writeback_dma_buffers,
                                            int64_t chain_reuse_hits,
                                            int64_t chain_reuse_bytes,
                                            int64_t data_sync_to_device_bytes,
                                            int64_t data_sync_from_device_bytes,
                                            int64_t meta_sync_to_device_bytes) {
  BridgeStats& stats = GetBridgeStats();
  for (auto& bucket : stats.submit_timing_buckets) {
    if (bucket.submit_slot == submit_slot && bucket.stage_count == stage_count &&
        bucket.task_count == task_count) {
      bucket.calls += 1;
      bucket.ok_calls += ok ? 1 : 0;
      bucket.fail_calls += ok ? 0 : 1;
      bucket.total_ns += total_ns;
      bucket.prep_ns += prep_ns;
      bucket.submit_ns += submit_ns;
      bucket.post_ns += post_ns;
      bucket.sync_to_device_ns += sync_to_device_ns;
      bucket.sync_from_device_ns += sync_from_device_ns;
      bucket.hw_elapsed_ns += hw_elapsed_ns;
      bucket.pc_tail_patch_writes += pc_tail_patch_writes;
      bucket.reloc_writeback_entries += reloc_writeback_entries;
      bucket.host_dma_buffers += host_dma_buffers;
      bucket.writeback_dma_buffers += writeback_dma_buffers;
      bucket.chain_reuse_hits += chain_reuse_hits;
      bucket.chain_reuse_bytes += chain_reuse_bytes;
      bucket.data_sync_to_device_bytes += data_sync_to_device_bytes;
      bucket.data_sync_from_device_bytes += data_sync_from_device_bytes;
      bucket.meta_sync_to_device_bytes += meta_sync_to_device_bytes;
      if (bucket.min_total_ns == 0 || total_ns < bucket.min_total_ns) {
        bucket.min_total_ns = total_ns;
      }
      if (total_ns > bucket.max_total_ns) {
        bucket.max_total_ns = total_ns;
      }
      if (bucket.min_hw_elapsed_ns == 0 || hw_elapsed_ns < bucket.min_hw_elapsed_ns) {
        bucket.min_hw_elapsed_ns = hw_elapsed_ns;
      }
      if (hw_elapsed_ns > bucket.max_hw_elapsed_ns) {
        bucket.max_hw_elapsed_ns = hw_elapsed_ns;
      }
      return;
    }
  }
  BridgeStats::SubmitTimingBucket bucket;
  bucket.submit_slot = submit_slot;
  bucket.stage_count = stage_count;
  bucket.task_count = task_count;
  bucket.calls = 1;
  bucket.ok_calls = ok ? 1 : 0;
  bucket.fail_calls = ok ? 0 : 1;
  bucket.total_ns = total_ns;
  bucket.prep_ns = prep_ns;
  bucket.submit_ns = submit_ns;
  bucket.post_ns = post_ns;
  bucket.sync_to_device_ns = sync_to_device_ns;
  bucket.sync_from_device_ns = sync_from_device_ns;
  bucket.hw_elapsed_ns = hw_elapsed_ns;
  bucket.min_total_ns = total_ns;
  bucket.max_total_ns = total_ns;
  bucket.min_hw_elapsed_ns = hw_elapsed_ns;
  bucket.max_hw_elapsed_ns = hw_elapsed_ns;
  bucket.pc_tail_patch_writes = pc_tail_patch_writes;
  bucket.reloc_writeback_entries = reloc_writeback_entries;
  bucket.host_dma_buffers = host_dma_buffers;
  bucket.writeback_dma_buffers = writeback_dma_buffers;
  bucket.chain_reuse_hits = chain_reuse_hits;
  bucket.chain_reuse_bytes = chain_reuse_bytes;
  bucket.data_sync_to_device_bytes = data_sync_to_device_bytes;
  bucket.data_sync_from_device_bytes = data_sync_from_device_bytes;
  bucket.meta_sync_to_device_bytes = meta_sync_to_device_bytes;
  stats.submit_timing_buckets.push_back(std::move(bucket));
}

static constexpr uint32_t kBridgeChainBlobV2Magic = 0x32424352;  // "RCB2"
static constexpr uint32_t kBridgeChainBlobV3Magic = 0x33424352;  // "RCB3"
static constexpr uint32_t kBridgeChainBlobV4Magic = 0x34424352;  // "RCB4"

struct BridgeChainConfig {
  struct Task {
    struct Reloc {
      uint32_t cmd_index{0};
      uint16_t arg_index{0};
      uint16_t flags{0};
    };
    std::vector<uint64_t> regcmds;
    std::vector<Reloc> relocs;
    uint32_t stage_index{0};
    uint32_t enable_mask{0};
    uint32_t int_mask{0};
    uint32_t int_clear{0};
    uint32_t regcfg_amount{0};
  };
  struct Submit {
    uint32_t stage_count{0};
    std::vector<Task> tasks;
  };
  std::vector<Submit> submits;
  uint64_t next_submit{0};
  bool configured{false};
  std::mutex mu;
};

static BridgeChainConfig& GetBridgeChainConfig() {
  static BridgeChainConfig cfg;
  return cfg;
}

int SetBridgeSyntheticTask(const ffi::Bytes& regcmd_blob, int enable_mask, int int_mask,
                           int int_clear, int regcfg_amount) {
  TVM_FFI_ICHECK_EQ(regcmd_blob.size() % 8, 0) << "regcmd blob size must be multiple of 8";
  BridgeChainConfig& cfg = GetBridgeChainConfig();
  std::lock_guard<std::mutex> lock(cfg.mu);
  size_t n = regcmd_blob.size() / 8;
  cfg.submits.clear();
  cfg.submits.resize(1);
  cfg.submits[0].stage_count = 1;
  cfg.submits[0].tasks.resize(1);
  cfg.submits[0].tasks[0].regcmds.resize(n);
  if (regcmd_blob.size() > 0) {
    std::memcpy(cfg.submits[0].tasks[0].regcmds.data(), regcmd_blob.data(), regcmd_blob.size());
  }
  cfg.submits[0].tasks[0].enable_mask = static_cast<uint32_t>(enable_mask);
  cfg.submits[0].tasks[0].int_mask = static_cast<uint32_t>(int_mask);
  cfg.submits[0].tasks[0].int_clear = static_cast<uint32_t>(int_clear);
  cfg.submits[0].tasks[0].regcfg_amount = static_cast<uint32_t>(regcfg_amount);
  cfg.next_submit = 0;
  cfg.configured = true;
  if (BridgeLogEnabled()) {
    LOG(INFO) << "RKNPU bridge synthetic task configured: n_regcmds=" << n
              << " enable_mask=0x" << std::hex << cfg.submits[0].tasks[0].enable_mask
              << " int_mask=0x" << cfg.submits[0].tasks[0].int_mask
              << " int_clear=0x" << cfg.submits[0].tasks[0].int_clear << std::dec
              << " regcfg_amount=" << cfg.submits[0].tasks[0].regcfg_amount;
  }
  return 0;
}

int SetBridgeChainBlob(const ffi::Bytes& blob) {
  static constexpr uint16_t kRelocWriteBack = 1;
  // v3 format:
  // [u32 magic='RCB3', u32 num_submits]
  // repeat num_submits:
  //   [u32 num_stages, u32 num_tasks]
  //   repeat num_tasks:
  //     [u32 n_regcmds, u32 enable_mask, u32 int_mask, u32 int_clear, u32 regcfg_amount]
  //     [n_regcmds * u64 regcmds]
  //
  // Backward compatibility (v2):
  // [u32 magic='RCB2', u32 num_submits]
  // repeat num_submits:
  //   [u32 num_tasks]
  //   repeat num_tasks:
  //     [u32 n_regcmds, u32 enable_mask, u32 int_mask, u32 int_clear, u32 regcfg_amount]
  //     [n_regcmds * u64 regcmds]
  //
  // Backward compatibility (v1):
  // [u32 num_tasks] + task payload, interpreted as one submit.
  const char* ptr = blob.data();
  size_t rem = blob.size();
  TVM_FFI_ICHECK_GE(rem, 4u) << "chain blob too small";

  auto parse_task = [&](BridgeChainConfig::Task* task) {
    TVM_FFI_ICHECK_GE(rem, 20u) << "chain task header truncated";
    uint32_t n_regcmds = 0;
    std::memcpy(&n_regcmds, ptr, 4);
    std::memcpy(&task->enable_mask, ptr + 4, 4);
    std::memcpy(&task->int_mask, ptr + 8, 4);
    std::memcpy(&task->int_clear, ptr + 12, 4);
    std::memcpy(&task->regcfg_amount, ptr + 16, 4);
    ptr += 20;
    rem -= 20;
    size_t bytes = static_cast<size_t>(n_regcmds) * 8;
    TVM_FFI_ICHECK_GE(rem, bytes) << "chain regcmd payload truncated";
    task->regcmds.resize(n_regcmds);
    if (bytes > 0) {
      std::memcpy(task->regcmds.data(), ptr, bytes);
      ptr += bytes;
      rem -= bytes;
    }
  };

  auto parse_task_v4 = [&](BridgeChainConfig::Task* task) {
    TVM_FFI_ICHECK_GE(rem, 28u) << "chain v4 task header truncated";
    uint32_t n_regcmds = 0;
    uint32_t n_relocs = 0;
    std::memcpy(&task->stage_index, ptr, 4);
    std::memcpy(&n_regcmds, ptr + 4, 4);
    std::memcpy(&task->enable_mask, ptr + 8, 4);
    std::memcpy(&task->int_mask, ptr + 12, 4);
    std::memcpy(&task->int_clear, ptr + 16, 4);
    std::memcpy(&task->regcfg_amount, ptr + 20, 4);
    std::memcpy(&n_relocs, ptr + 24, 4);
    ptr += 28;
    rem -= 28;
    TVM_FFI_ICHECK_GE(rem, static_cast<size_t>(n_relocs) * 8u) << "chain v4 relocs truncated";
    task->relocs.resize(n_relocs);
    for (uint32_t r = 0; r < n_relocs; ++r) {
      std::memcpy(&task->relocs[r].cmd_index, ptr, 4);
      std::memcpy(&task->relocs[r].arg_index, ptr + 4, 2);
      std::memcpy(&task->relocs[r].flags, ptr + 6, 2);
      ptr += 8;
      rem -= 8;
    }
    size_t bytes = static_cast<size_t>(n_regcmds) * 8;
    TVM_FFI_ICHECK_GE(rem, bytes) << "chain v4 regcmd payload truncated";
    task->regcmds.resize(n_regcmds);
    if (bytes > 0) {
      std::memcpy(task->regcmds.data(), ptr, bytes);
      ptr += bytes;
      rem -= bytes;
    }
    (void)kRelocWriteBack;
  };

  BridgeChainConfig parsed;
  uint32_t tag = 0;
  std::memcpy(&tag, ptr, 4);
  if (tag == kBridgeChainBlobV4Magic) {
    TVM_FFI_ICHECK_GE(rem, 8u) << "chain blob v4 header truncated";
    uint32_t n_submits = 0;
    std::memcpy(&n_submits, ptr + 4, 4);
    ptr += 8;
    rem -= 8;
    TVM_FFI_ICHECK_GT(n_submits, 0u) << "chain blob must contain >=1 submit";
    parsed.submits.resize(n_submits);
    for (uint32_t s = 0; s < n_submits; ++s) {
      TVM_FFI_ICHECK_GE(rem, 8u) << "submit header truncated";
      uint32_t n_stages = 0;
      uint32_t n_tasks = 0;
      std::memcpy(&n_stages, ptr, 4);
      std::memcpy(&n_tasks, ptr + 4, 4);
      ptr += 8;
      rem -= 8;
      TVM_FFI_ICHECK_GT(n_stages, 0u) << "submit must contain >=1 stage";
      TVM_FFI_ICHECK_GT(n_tasks, 0u) << "submit must contain >=1 task";
      parsed.submits[s].stage_count = n_stages;
      parsed.submits[s].tasks.resize(n_tasks);
      for (uint32_t t = 0; t < n_tasks; ++t) {
        parse_task_v4(&parsed.submits[s].tasks[t]);
      }
    }
  } else if (tag == kBridgeChainBlobV3Magic) {
    TVM_FFI_ICHECK_GE(rem, 8u) << "chain blob v3 header truncated";
    uint32_t n_submits = 0;
    std::memcpy(&n_submits, ptr + 4, 4);
    ptr += 8;
    rem -= 8;
    TVM_FFI_ICHECK_GT(n_submits, 0u) << "chain blob must contain >=1 submit";
    parsed.submits.resize(n_submits);
    for (uint32_t s = 0; s < n_submits; ++s) {
      TVM_FFI_ICHECK_GE(rem, 8u) << "submit header truncated";
      uint32_t n_stages = 0;
      uint32_t n_tasks = 0;
      std::memcpy(&n_stages, ptr, 4);
      std::memcpy(&n_tasks, ptr + 4, 4);
      ptr += 8;
      rem -= 8;
      TVM_FFI_ICHECK_GT(n_stages, 0u) << "submit must contain >=1 stage";
      TVM_FFI_ICHECK_GT(n_tasks, 0u) << "submit must contain >=1 task";
      parsed.submits[s].stage_count = n_stages;
      parsed.submits[s].tasks.resize(n_tasks);
      for (uint32_t t = 0; t < n_tasks; ++t) {
        parse_task(&parsed.submits[s].tasks[t]);
        parsed.submits[s].tasks[t].stage_index = t;
      }
    }
  } else if (tag == kBridgeChainBlobV2Magic) {
    TVM_FFI_ICHECK_GE(rem, 8u) << "chain blob v2 header truncated";
    uint32_t n_submits = 0;
    std::memcpy(&n_submits, ptr + 4, 4);
    ptr += 8;
    rem -= 8;
    TVM_FFI_ICHECK_GT(n_submits, 0u) << "chain blob must contain >=1 submit";
    parsed.submits.resize(n_submits);
    for (uint32_t s = 0; s < n_submits; ++s) {
      TVM_FFI_ICHECK_GE(rem, 4u) << "submit header truncated";
      uint32_t n_tasks = 0;
      std::memcpy(&n_tasks, ptr, 4);
      ptr += 4;
      rem -= 4;
      TVM_FFI_ICHECK_GT(n_tasks, 0u) << "submit must contain >=1 task";
      parsed.submits[s].stage_count = n_tasks;
      parsed.submits[s].tasks.resize(n_tasks);
      for (uint32_t t = 0; t < n_tasks; ++t) {
        parse_task(&parsed.submits[s].tasks[t]);
        parsed.submits[s].tasks[t].stage_index = t;
      }
    }
  } else {
    uint32_t n_tasks = 0;
    std::memcpy(&n_tasks, ptr, 4);
    ptr += 4;
    rem -= 4;
    TVM_FFI_ICHECK_GT(n_tasks, 0u) << "v1 chain blob must contain >=1 task";
    parsed.submits.resize(1);
    parsed.submits[0].stage_count = n_tasks;
    parsed.submits[0].tasks.resize(n_tasks);
    for (uint32_t t = 0; t < n_tasks; ++t) {
      parse_task(&parsed.submits[0].tasks[t]);
      parsed.submits[0].tasks[t].stage_index = t;
    }
  }

  TVM_FFI_ICHECK_EQ(rem, 0u) << "trailing bytes in chain blob";
  parsed.next_submit = 0;
  parsed.configured = true;

  BridgeChainConfig& cfg = GetBridgeChainConfig();
  {
    std::lock_guard<std::mutex> lock(cfg.mu);
    cfg.submits = std::move(parsed.submits);
    cfg.next_submit = 0;
    cfg.configured = true;
  }
  if (BridgeLogEnabled()) {
    LOG(INFO) << "RKNPU bridge chain blob configured: n_submits=" << cfg.submits.size()
              << " submit0_stages=" << (cfg.submits.empty() ? 0 : cfg.submits[0].stage_count)
              << " submit0_tasks=" << (cfg.submits.empty() ? 0 : cfg.submits[0].tasks.size());
  }
  return 0;
}

int SetBridgeSyntheticChain(const ffi::Bytes& blob) { return SetBridgeChainBlob(blob); }

static inline bool BridgeSubmitConfiguredChain(
    int requested_tasks, const std::vector<BridgeStageInvocation>& stages) {
#ifdef TVM_RKNPU_RUNTIME
  using BridgeClock = std::chrono::steady_clock;
  using BridgeNs = std::chrono::nanoseconds;
  static constexpr uint16_t kRelocWriteBack = 1;
  static constexpr uint32_t kPlaceholderInput = 0xAAAA0000u;
  static constexpr uint32_t kPlaceholderWeight = 0xBBBB0000u;
  static constexpr uint32_t kPlaceholderOutput = 0xCCCC0000u;
  static constexpr uint32_t kPlaceholderBias = 0xDDDD0000u;
  static constexpr uint32_t kPlaceholderRelocMaxDelta = 0x08000000u;  // 128 MiB
  static constexpr uint8_t kRelocInput = 0;
  static constexpr uint8_t kRelocWeight = 1;
  static constexpr uint8_t kRelocOutput = 2;
  static constexpr uint8_t kRelocBias = 3;
  static thread_local std::unique_ptr<rknpu::RKNPUDevice> dev;
  static thread_local rknpu::DMABuffer regcmd_buf;
  static thread_local rknpu::DMABuffer task_buf;
  static thread_local rknpu::DMABuffer scratch_buf;
  auto submit_begin = BridgeClock::now();
  int64_t prep_ns = 0;
  int64_t submit_ns = 0;
  int64_t sync_to_device_ns = 0;
  int64_t sync_from_device_ns = 0;
  uint32_t submit_slot = 0;
  uint32_t submit_stage_count = 0;
  if (!dev) {
    dev = std::make_unique<rknpu::RKNPUDevice>();
    regcmd_buf = dev->Alloc(4 * sizeof(uint64_t));
    task_buf = dev->Alloc(sizeof(rknpu::RknpuTask), rknpu::kMemTask);
    scratch_buf = dev->Alloc(BridgeScratchBytes());
  } else if (!scratch_buf.Valid() || scratch_buf.size < BridgeScratchBytes()) {
    if (scratch_buf.Valid()) dev->Free(scratch_buf);
    scratch_buf = dev->Alloc(BridgeScratchBytes());
  }
  BridgeStats& stats = GetBridgeStats();
  bool use_reloc_submit = BridgeUseRelocSubmit();
  bool use_reloc_for_submit = use_reloc_submit;
  auto infer_reloc_kind = [&](uint64_t regcmd_u64, uint8_t* out_kind, uint32_t* out_delta) -> bool {
    uint32_t addr = static_cast<uint32_t>((regcmd_u64 >> 16) & 0xFFFFFFFFULL);
    bool found = false;
    uint32_t best_delta = 0;
    uint8_t best_kind = 0;
    auto consider = [&](uint32_t base, uint8_t kind) {
      if (addr < base) return;
      uint32_t delta = addr - base;
      if (delta >= kPlaceholderRelocMaxDelta) return;
      if (!found || delta < best_delta) {
        best_delta = delta;
        best_kind = kind;
        found = true;
      }
    };
    consider(kPlaceholderInput, kRelocInput);
    consider(kPlaceholderWeight, kRelocWeight);
    consider(kPlaceholderOutput, kRelocOutput);
    consider(kPlaceholderBias, kRelocBias);
    if (!found) return false;
    *out_kind = best_kind;
    *out_delta = best_delta;
    return true;
  };
  auto expected_reloc_kind_mask = [&](int stage, uint16_t arg_index) -> uint32_t {
    switch (stage) {
      case kBridgeStageMatmul:
        return (arg_index == 0) ? (1u << kRelocInput)
                                : (arg_index == 1) ? (1u << kRelocWeight)
                                                   : (arg_index == 2) ? (1u << kRelocOutput) : 0u;
      case kBridgeStageAdd:
      case kBridgeStageMul:
      case kBridgeStageAddRelu:
        return (arg_index == 0)
                   ? (1u << kRelocInput)
                   : (arg_index == 1) ? ((1u << kRelocWeight) | (1u << kRelocBias))
                                      : (arg_index == 2) ? (1u << kRelocOutput) : 0u;
      case kBridgeStageExp:
      case kBridgeStageReciprocal:
      case kBridgeStageGelu:
        return (arg_index == 0) ? (1u << kRelocInput)
                                : (arg_index == 1) ? (1u << kRelocOutput) : 0u;
      case kBridgeStageRelu:
      case kBridgeStageRelu4D:
        return (arg_index == 0) ? ((1u << kRelocInput) | (1u << kRelocWeight) | (1u << kRelocBias))
                                : (arg_index == 1) ? (1u << kRelocOutput) : 0u;
      case kBridgeStageConv2D:
      case kBridgeStageConv2DRelu:
        return (arg_index == 0) ? (1u << kRelocInput)
                                : (arg_index == 1) ? (1u << kRelocWeight)
                                                   : (arg_index == 2) ? (1u << kRelocOutput) : 0u;
      case kBridgeStageMatmulBiasRelu:
        return (arg_index == 0) ? (1u << kRelocInput)
                                : (arg_index == 1) ? (1u << kRelocWeight)
                                                   : (arg_index == 2)
                                                         ? (1u << kRelocBias)
                                                         : (arg_index == 3) ? (1u << kRelocOutput)
                                                                            : 0u;
      case kBridgeStageMatmulBias:
        return (arg_index == 0) ? (1u << kRelocInput)
                                : (arg_index == 1) ? (1u << kRelocWeight)
                                                   : (arg_index == 2)
                                                         ? (1u << kRelocBias)
                                                         : (arg_index == 3) ? (1u << kRelocOutput)
                                                                            : 0u;
      default:
        return 0u;
    }
  };
  auto expected_reloc_writeback = [&](int stage, uint16_t arg_index) -> bool {
    switch (stage) {
      case kBridgeStageMatmul:
      case kBridgeStageAdd:
      case kBridgeStageMul:
      case kBridgeStageAddRelu:
        return arg_index == 2;
      case kBridgeStageRelu:
      case kBridgeStageRelu4D:
      case kBridgeStageExp:
      case kBridgeStageReciprocal:
      case kBridgeStageGelu:
        return arg_index == 1;
      case kBridgeStageConv2D:
      case kBridgeStageConv2DRelu:
        return arg_index == 2;
      case kBridgeStageMatmulBiasRelu:
        return arg_index == 3;
      case kBridgeStageMatmulBias:
        return arg_index == 3;
      default:
        return false;
    }
  };
  auto checksum_bytes = [&](const void* data, size_t nbytes) -> uint64_t {
    const uint8_t* p = static_cast<const uint8_t*>(data);
    uint64_t h = 1469598103934665603ULL;  // FNV-1a
    for (size_t i = 0; i < nbytes; ++i) {
      h ^= static_cast<uint64_t>(p[i]);
      h *= 1099511628211ULL;
    }
    return h;
  };
  std::vector<BridgeChainConfig::Task> tasks_to_submit;
  int64_t host_dma_buffers = 0;
  int64_t writeback_dma_buffers = 0;
  int64_t chain_reuse_hits = 0;
  int64_t chain_reuse_bytes = 0;
  int64_t data_sync_to_device_bytes = 0;
  int64_t data_sync_from_device_bytes = 0;
  int64_t meta_sync_to_device_bytes = 0;
  int64_t hw = 0;
  auto record_submit_timing = [&](bool ok, int64_t pc_tail_patch_writes,
                                  int64_t reloc_writeback_entries) {
    uint32_t task_count = static_cast<uint32_t>(tasks_to_submit.size());
    int64_t total_ns =
        std::chrono::duration_cast<BridgeNs>(BridgeClock::now() - submit_begin).count();
    int64_t post_ns = total_ns - prep_ns - submit_ns;
    if (post_ns < 0) post_ns = 0;
    RecordBridgeSubmitTiming(submit_slot, submit_stage_count, task_count, ok, total_ns, prep_ns,
                             submit_ns, post_ns, sync_to_device_ns, sync_from_device_ns, hw,
                             pc_tail_patch_writes, reloc_writeback_entries, host_dma_buffers,
                             writeback_dma_buffers, chain_reuse_hits, chain_reuse_bytes,
                             data_sync_to_device_bytes, data_sync_from_device_bytes,
                             meta_sync_to_device_bytes);
  };
  {
    BridgeChainConfig& cfg = GetBridgeChainConfig();
    std::lock_guard<std::mutex> lock(cfg.mu);
    if (cfg.configured && !cfg.submits.empty()) {
      size_t submit_idx = static_cast<size_t>(cfg.next_submit % cfg.submits.size());
      cfg.next_submit += 1;
      const auto& submit = cfg.submits[submit_idx];
      submit_slot = static_cast<uint32_t>(submit_idx);
      submit_stage_count = submit.stage_count;
      if (BridgeLogEnabled()) {
        LOG(INFO) << "RKNPU bridge using configured chain submit_idx=" << submit_idx
                  << " submit_stages=" << submit.stage_count
                  << " submit_tasks=" << submit.tasks.size() << " requested=" << requested_tasks;
      }
      if (use_reloc_submit && requested_tasks > 0 &&
          static_cast<size_t>(requested_tasks) != submit.stage_count) {
        use_reloc_for_submit = false;
        if (BridgeLogEnabled()) {
          LOG(INFO) << "RKNPU bridge reloc disabled for this submit due stage mismatch: requested="
                    << requested_tasks << " submit_stages=" << submit.stage_count;
        }
      }
      tasks_to_submit = submit.tasks;
    } else {
      if (BridgeLogEnabled()) {
        LOG(INFO) << "RKNPU bridge chain blob not configured";
      }
      record_submit_timing(false, 0, 0);
      return false;
    }
  }
  size_t total_cmds = 0;
  for (const auto& t : tasks_to_submit) total_cmds += t.regcmds.size();
  size_t regcmd_bytes = total_cmds * sizeof(uint64_t);
  size_t task_bytes = tasks_to_submit.size() * sizeof(rknpu::RknpuTask);
  if (!regcmd_buf.Valid() || regcmd_buf.size < regcmd_bytes) {
    if (regcmd_buf.Valid()) dev->Free(regcmd_buf);
    regcmd_buf = dev->Alloc(regcmd_bytes);
  }
  if (!task_buf.Valid() || task_buf.size < task_bytes) {
    if (task_buf.Valid()) dev->Free(task_buf);
    task_buf = dev->Alloc(task_bytes, rknpu::kMemTask);
  }
  auto* regs = reinterpret_cast<uint64_t*>(regcmd_buf.Data());
  auto* tasks = task_buf.As<rknpu::RknpuTask>();
  std::memset(tasks, 0, task_bytes);

  struct HostDMA {
    rknpu::DMABuffer buf;
    size_t bytes{0};
    bool write_back{false};
    bool persistent_cached{false};
    bool persistent_cache_hit{false};
    bool upload_skipped{false};
    bool sync_to_device{false};
    bool chain_reused{false};
  };
  std::unordered_map<void*, size_t> host_index;
  std::vector<void*> host_ptrs;
  std::vector<HostDMA> host_dma;
  enum class DeviceLayoutClass : uint8_t { kUnknown = 0, kMatrixFeature = 1, kSpatialFeature = 2 };
  std::unordered_map<void*, DeviceLayoutClass> host_device_layout;
  std::unordered_map<void*, size_t> bias_fp32_index;
  std::vector<std::vector<uint8_t>> bias_fp32_storage;
  struct WriteBackGatherMN {
    int m{0};
    int n{0};
    int n_aligned{0};
  };
  std::unordered_map<void*, WriteBackGatherMN> writeback_gather_mn;
  struct WriteBackGatherSpatial {
    int channels{0};
    int h_out{0};
    int w_out{0};
    int n_aligned{0};
    int m_padded{0};
  };
  std::unordered_map<void*, WriteBackGatherSpatial> writeback_gather_spatial;
  std::unordered_map<uint64_t, size_t> transformed_cache;
  std::vector<std::vector<uint8_t>> transformed_storage;
  std::vector<std::vector<uint8_t>> synthetic_zero_storage;
  std::unordered_map<size_t, size_t> synthetic_zero_by_size;
  bool use_persistent_dma_cache = BridgePersistentDmaCacheEnabled();
  bool assume_inputs_immutable = BridgeAssumeInputsImmutableEnabled();
  BridgePersistentDmaCache* persistent_dma_cache =
      use_persistent_dma_cache ? &GetBridgePersistentDmaCache() : nullptr;
  auto free_host_dma_entries = [&](std::vector<HostDMA>* entries) {
    if (entries == nullptr) return;
    for (auto& e : *entries) {
      if (e.buf.Valid() && !e.persistent_cached) {
        dev->Free(e.buf);
      }
    }
  };
  auto maybe_trim_persistent_dma_cache = [&]() {
    if (!use_persistent_dma_cache || persistent_dma_cache == nullptr) return;
    size_t max_entries = BridgePersistentDmaCacheMaxEntries();
    if (persistent_dma_cache->by_host_ptr.size() <= max_entries) return;
    for (auto& kv : persistent_dma_cache->by_host_ptr) {
      if (kv.second.Valid()) dev->Free(kv.second);
    }
    persistent_dma_cache->by_host_ptr.clear();
    persistent_dma_cache->uploaded.clear();
  };
  auto patch_addr_legacy = [&](uint64_t u) -> uint64_t {
    uint16_t reg = static_cast<uint16_t>(u & 0xFFFF);
    uint32_t addr = 0;
    if (reg == 0x1070) {
      addr = scratch_buf.dma_addr + 0x00000;
    } else if (reg >= 0x1110 && reg <= 0x113C && ((reg - 0x1110) % 4 == 0)) {
      addr = scratch_buf.dma_addr + 0x10000;
    } else if (reg == 0x4020) {
      addr = scratch_buf.dma_addr + 0x20000;
    } else if (reg == 0x5018) {
      addr = scratch_buf.dma_addr + 0x30000;
    } else if (reg == 0x5020) {
      addr = scratch_buf.dma_addr + 0x34000;
    } else if (reg == 0x5038) {
      addr = scratch_buf.dma_addr + 0x40000;
    } else if (reg == 0x6070) {
      addr = scratch_buf.dma_addr + 0x50000;
    } else if (reg == 0x701C) {
      addr = scratch_buf.dma_addr + 0x60000;
    } else {
      return u;
    }
    return (u & 0xFFFF00000000FFFFULL) | (static_cast<uint64_t>(addr) << 16);
  };

  auto ensure_dma_for = [&](void* host_ptr, size_t nbytes, bool write_back) -> uint32_t {
    auto it = host_index.find(host_ptr);
    if (it != host_index.end()) {
      HostDMA& entry = host_dma[it->second];
      if (entry.buf.size < nbytes) {
        if (entry.persistent_cached && use_persistent_dma_cache && persistent_dma_cache != nullptr) {
          auto pit = persistent_dma_cache->by_host_ptr.find(host_ptr);
          if (pit != persistent_dma_cache->by_host_ptr.end() && pit->second.Valid()) {
            dev->Free(pit->second);
          }
          rknpu::DMABuffer resized = dev->Alloc(nbytes > 0 ? nbytes : 1);
          persistent_dma_cache->by_host_ptr[host_ptr] = resized;
          entry.buf = resized;
        } else {
          dev->Free(entry.buf);
          entry.buf = dev->Alloc(nbytes > 0 ? nbytes : 1);
        }
      }
      entry.bytes = std::max(entry.bytes, nbytes);
      entry.write_back = entry.write_back || write_back;
      // If this host pointer is also a writeback target in the same submit,
      // preserve device-produced data for chained consumers.
      if (!write_back && !entry.write_back && nbytes > 0) {
        std::memcpy(entry.buf.As<void>(), host_ptr, nbytes);
        entry.sync_to_device = true;
      }
      return entry.buf.dma_addr;
    }
    HostDMA entry;
    bool skip_upload = false;
    if (use_persistent_dma_cache && persistent_dma_cache != nullptr) {
      maybe_trim_persistent_dma_cache();
      auto pit = persistent_dma_cache->by_host_ptr.find(host_ptr);
      if (pit == persistent_dma_cache->by_host_ptr.end() || !pit->second.Valid()) {
        rknpu::DMABuffer b = dev->Alloc(nbytes > 0 ? nbytes : 1);
        persistent_dma_cache->by_host_ptr[host_ptr] = b;
        persistent_dma_cache->uploaded[host_ptr] = false;
        entry.buf = b;
      } else {
        entry.persistent_cache_hit = true;
        if (pit->second.size < nbytes) {
          dev->Free(pit->second);
          pit->second = dev->Alloc(nbytes > 0 ? nbytes : 1);
          persistent_dma_cache->uploaded[host_ptr] = false;
          entry.persistent_cache_hit = false;
        }
        entry.buf = pit->second;
        if (!write_back && assume_inputs_immutable) {
          auto uit = persistent_dma_cache->uploaded.find(host_ptr);
          if (uit != persistent_dma_cache->uploaded.end() && uit->second) {
            skip_upload = true;
          }
        }
      }
      entry.persistent_cached = true;
    } else {
      entry.buf = dev->Alloc(nbytes > 0 ? nbytes : 1);
      entry.persistent_cached = false;
    }
    entry.bytes = nbytes;
    entry.write_back = write_back;
    entry.upload_skipped = skip_upload;
    if (!write_back && nbytes > 0 && !skip_upload) {
      std::memcpy(entry.buf.As<void>(), host_ptr, nbytes);
      entry.sync_to_device = true;
      if (entry.persistent_cached && persistent_dma_cache != nullptr) {
        persistent_dma_cache->uploaded[host_ptr] = true;
      }
    }
    host_index[host_ptr] = host_dma.size();
    host_ptrs.push_back(host_ptr);
    host_dma.push_back(std::move(entry));
    return host_dma.back().buf.dma_addr;
  };
  auto transformed_key = [&](void* p, int stage, int arg, int m, int k, int n, int tag) -> uint64_t {
    uint64_t h = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(p));
    h ^= static_cast<uint64_t>(static_cast<uint32_t>(stage)) << 32;
    h ^= static_cast<uint64_t>(static_cast<uint32_t>(arg)) << 40;
    h ^= static_cast<uint64_t>(static_cast<uint32_t>(m * 1315423911u)) << 1;
    h ^= static_cast<uint64_t>(static_cast<uint32_t>(k * 2654435761u)) << 3;
    h ^= static_cast<uint64_t>(static_cast<uint32_t>(n * 2246822519u)) << 5;
    h ^= static_cast<uint64_t>(static_cast<uint32_t>(tag)) << 9;
    return h;
  };
  auto persistent_transformed_key = [&](void* p, size_t src_nbytes, int stage, int arg, int m, int k,
                                        int n, int tag) -> uint64_t {
    uint64_t h = transformed_key(p, stage, arg, m, k, n, tag);
    h ^= checksum_bytes(p, src_nbytes);
    h ^= static_cast<uint64_t>(src_nbytes * 11400714819323198485ull);
    return h;
  };
  auto expected_read_layout = [&](int stage, uint16_t arg_index) -> DeviceLayoutClass {
    switch (stage) {
      case kBridgeStageMatmul:
      case kBridgeStageMatmulBiasRelu:
      case kBridgeStageMatmulBias:
      case kBridgeStageAdd:
      case kBridgeStageMul:
      case kBridgeStageAddRelu:
      case kBridgeStageRelu:
      case kBridgeStageRelu4D:
      case kBridgeStageExp:
      case kBridgeStageReciprocal:
      case kBridgeStageGelu:
        return (arg_index == 0) ? DeviceLayoutClass::kMatrixFeature : DeviceLayoutClass::kUnknown;
      case kBridgeStageConv2D:
      case kBridgeStageConv2DRelu:
        return (arg_index == 0) ? DeviceLayoutClass::kSpatialFeature : DeviceLayoutClass::kUnknown;
      default:
        return DeviceLayoutClass::kUnknown;
    }
  };
  auto produced_writeback_layout = [&](int stage, uint16_t arg_index) -> DeviceLayoutClass {
    switch (stage) {
      case kBridgeStageMatmul:
        return (arg_index == 2) ? DeviceLayoutClass::kMatrixFeature : DeviceLayoutClass::kUnknown;
      case kBridgeStageMatmulBiasRelu:
        return (arg_index == 3) ? DeviceLayoutClass::kMatrixFeature : DeviceLayoutClass::kUnknown;
      case kBridgeStageMatmulBias:
        return (arg_index == 3) ? DeviceLayoutClass::kMatrixFeature : DeviceLayoutClass::kUnknown;
      case kBridgeStageAdd:
      case kBridgeStageMul:
      case kBridgeStageAddRelu:
        return (arg_index == 2) ? DeviceLayoutClass::kMatrixFeature : DeviceLayoutClass::kUnknown;
      case kBridgeStageRelu:
        return (arg_index == 1) ? DeviceLayoutClass::kMatrixFeature : DeviceLayoutClass::kUnknown;
      case kBridgeStageRelu4D:
        return (arg_index == 1) ? DeviceLayoutClass::kMatrixFeature : DeviceLayoutClass::kUnknown;
      case kBridgeStageExp:
      case kBridgeStageReciprocal:
      case kBridgeStageGelu:
        return (arg_index == 1) ? DeviceLayoutClass::kMatrixFeature : DeviceLayoutClass::kUnknown;
      case kBridgeStageConv2D:
      case kBridgeStageConv2DRelu:
        return (arg_index == 2) ? DeviceLayoutClass::kSpatialFeature : DeviceLayoutClass::kUnknown;
      default:
        return DeviceLayoutClass::kUnknown;
    }
  };

  size_t off = 0;
  int64_t addr_patch_writes = 0;
  int64_t reloc_entries = 0;
  int64_t reloc_writebacks = 0;
  for (size_t ti = 0; ti < tasks_to_submit.size(); ++ti) {
    const auto& t = tasks_to_submit[ti];
    if (t.regcmds.empty()) {
      if (BridgeLogEnabled()) {
        LOG(INFO) << "RKNPU bridge invalid task: empty regcmds at task " << ti;
      }
      return false;
    }
    if (use_reloc_for_submit && t.stage_index >= stages.size()) {
      if (BridgeLogEnabled()) {
        LOG(INFO) << "RKNPU bridge invalid task: stage_index=" << t.stage_index
                  << " out of range for stage count=" << stages.size();
      }
      return false;
    }
    for (size_t i = 0; i < t.regcmds.size(); ++i) {
      // Always patch fixed internal scratch pointers (CNA/DPU temp buffers).
      // Reloc patching below will then override IO/bias/output placeholders.
      regs[off + i] = patch_addr_legacy(t.regcmds[i]);
      addr_patch_writes += 1;
    }
    if (use_reloc_for_submit) for (const auto& reloc : t.relocs) {
      const auto& st = stages[t.stage_index];
      if (reloc.cmd_index >= t.regcmds.size()) {
        if (BridgeLogEnabled()) {
          LOG(INFO) << "RKNPU bridge invalid reloc: cmd_index=" << reloc.cmd_index
                    << " >= n_regcmds=" << t.regcmds.size() << " task=" << ti;
        }
        return false;
      }
      uint64_t u_raw = t.regcmds[reloc.cmd_index];
      uint64_t u_cur = regs[off + reloc.cmd_index];
      uint8_t reloc_kind = 0;
      uint32_t reloc_delta = 0;
      bool have_reloc_kind = infer_reloc_kind(u_raw, &reloc_kind, &reloc_delta);
      if (reloc.arg_index >= st.ptrs.size() || reloc.arg_index >= st.ptr_sizes.size()) {
        if (BridgeLogEnabled()) {
          LOG(INFO) << "RKNPU bridge invalid reloc: arg_index=" << reloc.arg_index
                    << " stage_ptrs=" << st.ptrs.size() << " task=" << ti;
        }
        return false;
      }
      void* host_ptr = st.ptrs[reloc.arg_index];
      size_t nbytes = st.ptr_sizes[reloc.arg_index];
      if (!host_ptr || nbytes == 0) {
        if (BridgeLogEnabled()) {
          LOG(INFO) << "RKNPU bridge invalid reloc target: null/empty host ptr task=" << ti
                    << " arg_index=" << reloc.arg_index;
        }
        return false;
      }
      bool write_back = (reloc.flags & kRelocWriteBack) != 0;
      void* dma_src_ptr = host_ptr;
      size_t dma_nbytes = nbytes;
      bool synthetic_zero_reloc = false;
      bool reuse_chained_device_buffer = false;
      if (!write_back) {
        auto hit = host_index.find(host_ptr);
        if (hit != host_index.end()) {
          const HostDMA& existing = host_dma[hit->second];
          if (existing.write_back) {
            DeviceLayoutClass have_layout = DeviceLayoutClass::kUnknown;
            auto lit = host_device_layout.find(host_ptr);
            if (lit != host_device_layout.end()) {
              have_layout = lit->second;
            }
            DeviceLayoutClass need_layout = expected_read_layout(st.stage, reloc.arg_index);
            if (have_layout != DeviceLayoutClass::kUnknown && have_layout == need_layout) {
              reuse_chained_device_buffer = true;
              dma_nbytes = existing.bytes;
              host_dma[hit->second].chain_reused = true;
              chain_reuse_hits += 1;
              chain_reuse_bytes += static_cast<int64_t>(existing.bytes);
            }
          }
        }
      }
      int m = st.ints.size() >= 1 ? st.ints[0] : 0;
      int k = st.ints.size() >= 2 ? st.ints[1] : 0;
      int n = st.ints.size() >= 3 ? st.ints[2] : 0;
      int b_mode = st.ints.size() >= 3 ? st.ints[2] : 0;
      int conv_n = 0;
      int conv_c = 0;
      int conv_h = 0;
      int conv_w = 0;
      int conv_oc = 0;
      int conv_kh = 0;
      int conv_kw = 0;
      int conv_oh = 0;
      int conv_ow = 0;
      if ((st.stage == kBridgeStageAdd || st.stage == kBridgeStageMul ||
           st.stage == kBridgeStageAddRelu) &&
          st.ints.size() >= 2) {
        m = st.ints[0];
        n = st.ints[1];
        b_mode = st.ints.size() >= 3 ? st.ints[2] : 0;
      } else if (st.stage == kBridgeStageRelu && st.ints.size() >= 2) {
        m = st.ints[0];
        n = st.ints[1];
      } else if ((st.stage == kBridgeStageExp || st.stage == kBridgeStageReciprocal ||
                  st.stage == kBridgeStageGelu) &&
                 st.ints.size() >= 2) {
        m = st.ints[0];
        n = st.ints[1];
      } else if (st.stage == kBridgeStageRelu4D && st.ints.size() >= 4) {
        m = st.ints[0] * st.ints[1] * st.ints[2] * st.ints[3];
        n = 1;
      } else if ((st.stage == kBridgeStageConv2D || st.stage == kBridgeStageConv2DRelu) &&
                 st.ints.size() >= 9) {
        conv_n = st.ints[0];
        conv_c = st.ints[1];
        conv_h = st.ints[2];
        conv_w = st.ints[3];
        conv_oc = st.ints[4];
        conv_kh = st.ints[5];
        conv_kw = st.ints[6];
        conv_oh = st.ints[7];
        conv_ow = st.ints[8];
      }
      // ReLU stages are encoded via the elementwise engine and carry synthetic
      // source-B relocations. Bind those synthetic reads to explicit zero DMA.
      if (!write_back && (st.stage == kBridgeStageRelu || st.stage == kBridgeStageRelu4D) &&
          have_reloc_kind && (reloc_kind == kRelocWeight || reloc_kind == kRelocBias)) {
        size_t zero_nbytes = nbytes;
        if (m > 0 && n > 0) {
          int m_pad = PadM(m);
          // Feature tensors are channel-aligned to 32 on RK3588 matrix path.
          int n_aligned = AlignUp(n, 32);
          size_t layout_nbytes =
              static_cast<size_t>(m_pad) * n_aligned * sizeof(uint16_t);
          if (layout_nbytes > zero_nbytes) {
            zero_nbytes = layout_nbytes;
          }
        }
        if (static_cast<size_t>(reloc_delta) >= zero_nbytes) {
          zero_nbytes = static_cast<size_t>(reloc_delta) + sizeof(uint16_t);
        }
        auto z = synthetic_zero_by_size.find(zero_nbytes);
        if (z == synthetic_zero_by_size.end()) {
          synthetic_zero_storage.push_back(std::vector<uint8_t>(zero_nbytes, 0));
          z = synthetic_zero_by_size.emplace(zero_nbytes, synthetic_zero_storage.size() - 1).first;
        }
        dma_src_ptr = synthetic_zero_storage[z->second].data();
        dma_nbytes = synthetic_zero_storage[z->second].size();
        synthetic_zero_reloc = true;
      }

      // Map host row-major tensors to NPU layouts on upload, and configure
      // gather metadata for writeback outputs.
      if (!reuse_chained_device_buffer &&
          (st.stage == kBridgeStageMatmul || st.stage == kBridgeStageMatmulBiasRelu ||
           st.stage == kBridgeStageMatmulBias) && m > 0 &&
          k > 0 && n > 0) {
        if (reloc.arg_index == 0 && !write_back) {
          uint64_t key = transformed_key(host_ptr, st.stage, reloc.arg_index, m, k, n, 1);
          auto it = transformed_cache.find(key);
          if (it == transformed_cache.end()) {
            int m_pad = PadM(m);
            int k_aligned = AlignUp(k, 32);
            std::vector<uint8_t> buf(static_cast<size_t>(m_pad) * k_aligned * sizeof(uint16_t));
            ScatterInputFP16(reinterpret_cast<uint16_t*>(buf.data()),
                             static_cast<const uint16_t*>(host_ptr), m, k, m_pad, k_aligned);
            transformed_storage.push_back(std::move(buf));
            it = transformed_cache.emplace(key, transformed_storage.size() - 1).first;
          }
          dma_src_ptr = transformed_storage[it->second].data();
          dma_nbytes = transformed_storage[it->second].size();
        } else if (reloc.arg_index == 1 && !write_back &&
                   (st.stage == kBridgeStageMatmulBiasRelu || st.stage == kBridgeStageMatmulBias ||
                    BridgePackMatmulWeightsEnabled())) {
          uint64_t key = transformed_key(host_ptr, st.stage, reloc.arg_index, m, k, n, 2);
          uint64_t persistent_key =
              persistent_transformed_key(host_ptr, nbytes, st.stage, reloc.arg_index, m, k, n, 2);
          bool persistent_hit = false;
          if (BridgeCacheTransformsEnabled()) {
            auto& cache = GetBridgeTransformCache();
            std::lock_guard<std::mutex> lock(cache.mu);
            auto git = cache.blobs.find(persistent_key);
            if (git != cache.blobs.end()) {
              dma_src_ptr = git->second.data();
              dma_nbytes = git->second.size();
              persistent_hit = true;
            }
          }
          if (!persistent_hit) {
            auto it = transformed_cache.find(key);
            if (it == transformed_cache.end()) {
              int k_aligned = AlignUp(k, 32);
              int n_aligned = AlignUp(n, 16);
              std::vector<uint8_t> buf(static_cast<size_t>(k_aligned) * n_aligned *
                                       sizeof(uint16_t));
              PackWeightsFP16(reinterpret_cast<uint16_t*>(buf.data()),
                              static_cast<const uint16_t*>(host_ptr), k, n);
              if (BridgeCacheTransformsEnabled()) {
                auto& cache = GetBridgeTransformCache();
                std::lock_guard<std::mutex> lock(cache.mu);
                cache.blobs[persistent_key] = std::move(buf);
                auto git = cache.blobs.find(persistent_key);
                dma_src_ptr = git->second.data();
                dma_nbytes = git->second.size();
              } else {
                transformed_storage.push_back(std::move(buf));
                it = transformed_cache.emplace(key, transformed_storage.size() - 1).first;
                dma_src_ptr = transformed_storage[it->second].data();
                dma_nbytes = transformed_storage[it->second].size();
              }
            } else {
              dma_src_ptr = transformed_storage[it->second].data();
              dma_nbytes = transformed_storage[it->second].size();
            }
          }
        } else if (((st.stage == kBridgeStageMatmul && reloc.arg_index == 2) ||
                    ((st.stage == kBridgeStageMatmulBiasRelu ||
                      st.stage == kBridgeStageMatmulBias) &&
                     reloc.arg_index == 3)) &&
                   write_back) {
          int m_pad = PadM(m);
          // DPU writeback may touch up to AlignUp(N, 32) channels.
          int n_aligned = AlignUp(n, 32);
          dma_nbytes = static_cast<size_t>(m_pad) * n_aligned * sizeof(uint16_t);
          writeback_gather_mn[host_ptr] = WriteBackGatherMN{m, n, n_aligned};
        }
      }
      if (!reuse_chained_device_buffer &&
          (st.stage == kBridgeStageAdd || st.stage == kBridgeStageMul ||
           st.stage == kBridgeStageAddRelu ||
           st.stage == kBridgeStageRelu ||
           st.stage == kBridgeStageExp ||
           st.stage == kBridgeStageReciprocal ||
           st.stage == kBridgeStageGelu) &&
          m > 0 && n > 0) {
        int m_pad = PadM(m);
        // Elementwise matrix-feature tensors also use 32-channel alignment.
        int n_aligned = AlignUp(n, 32);
        if (reloc.arg_index == 0 && !write_back && !synthetic_zero_reloc) {
          uint64_t key = transformed_key(host_ptr, st.stage, reloc.arg_index, m, 0, n, 3);
          auto it = transformed_cache.find(key);
          if (it == transformed_cache.end()) {
            std::vector<uint8_t> buf(static_cast<size_t>(m_pad) * n_aligned * sizeof(uint16_t));
            ScatterInputFP16(reinterpret_cast<uint16_t*>(buf.data()),
                             static_cast<const uint16_t*>(host_ptr), m, n, m_pad, n_aligned);
            transformed_storage.push_back(std::move(buf));
            it = transformed_cache.emplace(key, transformed_storage.size() - 1).first;
          }
          dma_src_ptr = transformed_storage[it->second].data();
          dma_nbytes = transformed_storage[it->second].size();
        } else if ((st.stage == kBridgeStageAdd || st.stage == kBridgeStageMul ||
                    st.stage == kBridgeStageAddRelu) &&
                   reloc.arg_index == 1 && !write_back) {
          uint64_t key = transformed_key(host_ptr, st.stage, reloc.arg_index, m, 0, n,
                                         b_mode == 1 ? 4 : b_mode == 2 ? 8 : 5);
          uint64_t persistent_key = persistent_transformed_key(
              host_ptr, nbytes, st.stage, reloc.arg_index, m, 0, n, b_mode == 1 ? 4 : b_mode == 2 ? 8 : 5);
          bool persistent_hit = false;
          if (BridgeCacheTransformsEnabled()) {
            auto& cache = GetBridgeTransformCache();
            std::lock_guard<std::mutex> lock(cache.mu);
            auto git = cache.blobs.find(persistent_key);
            if (git != cache.blobs.end()) {
              dma_src_ptr = git->second.data();
              dma_nbytes = git->second.size();
              persistent_hit = true;
            }
          }
          if (!persistent_hit) {
            auto it = transformed_cache.find(key);
            if (it == transformed_cache.end()) {
              std::vector<uint8_t> buf;
              if (b_mode == 1) {
                // Bias-1D path: materialize [m, n] row-major then scatter exactly
                // like the 2D add path so EW source-B layout is consistent.
                std::vector<uint16_t> bias2d(static_cast<size_t>(m) * n, 0);
                const uint16_t* bias1d = static_cast<const uint16_t*>(host_ptr);
                for (int row = 0; row < m; ++row) {
                  std::memcpy(bias2d.data() + static_cast<size_t>(row) * n, bias1d,
                              static_cast<size_t>(n) * sizeof(uint16_t));
                }
                buf.resize(static_cast<size_t>(m_pad) * n_aligned * sizeof(uint16_t));
                ScatterInputFP16(reinterpret_cast<uint16_t*>(buf.data()), bias2d.data(), m, n, m_pad,
                                 n_aligned);
              } else if (b_mode == 2) {
                std::vector<uint16_t> col2d(static_cast<size_t>(m) * n, 0);
                const uint16_t* col1d = static_cast<const uint16_t*>(host_ptr);
                for (int row = 0; row < m; ++row) {
                  std::fill_n(col2d.data() + static_cast<size_t>(row) * n, n, col1d[row]);
                }
                buf.resize(static_cast<size_t>(m_pad) * n_aligned * sizeof(uint16_t));
                ScatterInputFP16(reinterpret_cast<uint16_t*>(buf.data()), col2d.data(), m, n, m_pad,
                                 n_aligned);
              } else {
                buf.resize(static_cast<size_t>(m_pad) * n_aligned * sizeof(uint16_t));
                ScatterInputFP16(reinterpret_cast<uint16_t*>(buf.data()),
                                 static_cast<const uint16_t*>(host_ptr), m, n, m_pad, n_aligned);
              }
              if (BridgeCacheTransformsEnabled()) {
                auto& cache = GetBridgeTransformCache();
                std::lock_guard<std::mutex> lock(cache.mu);
                cache.blobs[persistent_key] = std::move(buf);
                auto git = cache.blobs.find(persistent_key);
                dma_src_ptr = git->second.data();
                dma_nbytes = git->second.size();
              } else {
                transformed_storage.push_back(std::move(buf));
                it = transformed_cache.emplace(key, transformed_storage.size() - 1).first;
                dma_src_ptr = transformed_storage[it->second].data();
                dma_nbytes = transformed_storage[it->second].size();
              }
            } else {
              dma_src_ptr = transformed_storage[it->second].data();
              dma_nbytes = transformed_storage[it->second].size();
            }
          }
        } else if (((st.stage == kBridgeStageAdd || st.stage == kBridgeStageMul ||
                     st.stage == kBridgeStageAddRelu) &&
                    reloc.arg_index == 2 && write_back) ||
                   ((st.stage == kBridgeStageRelu || st.stage == kBridgeStageExp ||
                     st.stage == kBridgeStageReciprocal || st.stage == kBridgeStageGelu) &&
                    reloc.arg_index == 1 && write_back)) {
          dma_nbytes = static_cast<size_t>(m_pad) * n_aligned * sizeof(uint16_t);
          writeback_gather_mn[host_ptr] = WriteBackGatherMN{m, n, n_aligned};
        }
      }
      if (!reuse_chained_device_buffer &&
          (st.stage == kBridgeStageConv2D || st.stage == kBridgeStageConv2DRelu) && conv_n == 1 &&
          conv_c > 0 && conv_h > 0 && conv_w > 0 && conv_oc > 0 && conv_kh > 0 && conv_kw > 0 &&
          conv_oh > 0 && conv_ow > 0) {
        int c_aligned = AlignUp(conv_c, 32);
        int k_eff = AlignUp(conv_c, 32) * conv_kh * conv_kw;
        int k_eff_aligned = AlignUp(k_eff, 32);
        int n_aligned = AlignUp(conv_oc, 16);
        int m_padded = PadM(conv_oh * conv_ow);
        if (reloc.arg_index == 0 && !write_back) {
          uint64_t key = transformed_key(host_ptr, st.stage, reloc.arg_index, conv_c, conv_h, conv_w, 6);
          auto it = transformed_cache.find(key);
          if (it == transformed_cache.end()) {
            std::vector<uint8_t> buf(static_cast<size_t>(c_aligned) * conv_h * conv_w * sizeof(uint16_t));
            ScatterSpatialFP16(reinterpret_cast<uint16_t*>(buf.data()),
                              static_cast<const uint16_t*>(host_ptr), conv_c, conv_h, conv_w, c_aligned);
            transformed_storage.push_back(std::move(buf));
            it = transformed_cache.emplace(key, transformed_storage.size() - 1).first;
          }
          dma_src_ptr = transformed_storage[it->second].data();
          dma_nbytes = transformed_storage[it->second].size();
        } else if (reloc.arg_index == 1 && !write_back) {
          uint64_t key =
              transformed_key(host_ptr, st.stage, reloc.arg_index, conv_oc, conv_c, k_eff, 7);
          uint64_t persistent_key = persistent_transformed_key(
              host_ptr, nbytes, st.stage, reloc.arg_index, conv_oc, conv_c, k_eff, 7);
          bool persistent_hit = false;
          if (BridgeCacheTransformsEnabled()) {
            auto& cache = GetBridgeTransformCache();
            std::lock_guard<std::mutex> lock(cache.mu);
            auto git = cache.blobs.find(persistent_key);
            if (git != cache.blobs.end()) {
              dma_src_ptr = git->second.data();
              dma_nbytes = git->second.size();
              persistent_hit = true;
            }
          }
          if (!persistent_hit) {
            auto it = transformed_cache.find(key);
            if (it == transformed_cache.end()) {
              std::vector<uint8_t> buf(static_cast<size_t>(k_eff_aligned) * n_aligned *
                                       sizeof(uint16_t));
              PackWeightsConv2DFP16(reinterpret_cast<uint16_t*>(buf.data()),
                                    static_cast<const uint16_t*>(host_ptr), conv_oc, conv_c, conv_kh, conv_kw,
                                    k_eff_aligned, n_aligned);
              if (BridgeCacheTransformsEnabled()) {
                auto& cache = GetBridgeTransformCache();
                std::lock_guard<std::mutex> lock(cache.mu);
                cache.blobs[persistent_key] = std::move(buf);
                auto git = cache.blobs.find(persistent_key);
                dma_src_ptr = git->second.data();
                dma_nbytes = git->second.size();
              } else {
                transformed_storage.push_back(std::move(buf));
                it = transformed_cache.emplace(key, transformed_storage.size() - 1).first;
                dma_src_ptr = transformed_storage[it->second].data();
                dma_nbytes = transformed_storage[it->second].size();
              }
            } else {
              dma_src_ptr = transformed_storage[it->second].data();
              dma_nbytes = transformed_storage[it->second].size();
            }
          }
        } else if (reloc.arg_index == 2 && write_back) {
          dma_nbytes = static_cast<size_t>(n_aligned) * m_padded * sizeof(uint16_t);
          writeback_gather_spatial[host_ptr] =
              WriteBackGatherSpatial{conv_oc, conv_oh, conv_ow, n_aligned, m_padded};
        }
      }
      if ((st.stage == kBridgeStageMatmulBiasRelu || st.stage == kBridgeStageMatmulBias) &&
          reloc.arg_index == 2 && !write_back) {
        auto it = bias_fp32_index.find(host_ptr);
        size_t bias_idx;
        if (it == bias_fp32_index.end()) {
          bias_idx = bias_fp32_storage.size();
          bias_fp32_index[host_ptr] = bias_idx;
          // Bias participates in matrix-feature channels; keep 32-channel alignment.
          int n_aligned = AlignUp(n > 0 ? n : static_cast<int>(nbytes / sizeof(uint16_t)), 32);
          size_t elems = static_cast<size_t>(n_aligned);
          std::vector<uint8_t> buf(elems * sizeof(float), 0);
          const uint16_t* in = static_cast<const uint16_t*>(host_ptr);
          float* out = reinterpret_cast<float*>(buf.data());
          size_t src_elems = nbytes / sizeof(uint16_t);
          for (size_t i = 0; i < src_elems; ++i) {
            out[i] = HalfToFloat(in[i]);
          }
          bias_fp32_storage.push_back(std::move(buf));
        } else {
          bias_idx = it->second;
        }
        dma_src_ptr = bias_fp32_storage[bias_idx].data();
        dma_nbytes = bias_fp32_storage[bias_idx].size();
      }
      if (BridgeValidateRelocSemanticsEnabled()) {
        uint32_t kind_mask = expected_reloc_kind_mask(st.stage, reloc.arg_index);
        bool kind_ok =
            have_reloc_kind && kind_mask != 0u && ((kind_mask & (1u << reloc_kind)) != 0u);
        bool write_back_ok = write_back == expected_reloc_writeback(st.stage, reloc.arg_index);
        if (!kind_ok || !write_back_ok) {
          stats.reloc_semantic_mismatch += 1;
          if (BridgeLogEnabled()) {
            LOG(INFO) << "RKNPU bridge reloc semantic mismatch: stage=" << st.stage
                      << " arg_index=" << reloc.arg_index << " flags=" << reloc.flags
                      << " task=" << ti << " have_kind=" << static_cast<int>(have_reloc_kind)
                      << " kind=" << static_cast<int>(reloc_kind)
                      << " expected_mask=" << kind_mask;
          }
          return false;
        }
        if (static_cast<size_t>(reloc_delta) >= dma_nbytes) {
          stats.reloc_range_mismatch += 1;
          if (BridgeLogEnabled()) {
            LOG(INFO) << "RKNPU bridge reloc range mismatch: stage=" << st.stage
                      << " arg_index=" << reloc.arg_index << " delta=" << reloc_delta
                      << " nbytes=" << dma_nbytes << " task=" << ti;
          }
          return false;
        }
      }
      uint32_t dma = ensure_dma_for(dma_src_ptr, dma_nbytes, write_back);
      host_dma_buffers = static_cast<int64_t>(host_dma.size());
      int64_t writeback_count = 0;
      for (const auto& entry : host_dma) {
        if (entry.write_back) writeback_count += 1;
      }
      writeback_dma_buffers = writeback_count;
      if (write_back) {
        DeviceLayoutClass produced = produced_writeback_layout(st.stage, reloc.arg_index);
        if (produced != DeviceLayoutClass::kUnknown) {
          host_device_layout[host_ptr] = produced;
        }
      }
      if (write_back && (st.stage == kBridgeStageRelu || st.stage == kBridgeStageRelu4D)) {
        auto hit = host_index.find(host_ptr);
        if (hit != host_index.end()) {
          HostDMA& entry = host_dma[hit->second];
          if (entry.buf.Valid() && entry.bytes > 0) {
            std::memset(entry.buf.As<void>(), 0, entry.bytes);
          }
        }
      }
      uint64_t patched_addr = static_cast<uint64_t>(dma) + static_cast<uint64_t>(reloc_delta);
      if (patched_addr > 0xFFFFFFFFULL) {
        stats.reloc_range_mismatch += 1;
        if (BridgeLogEnabled()) {
          LOG(INFO) << "RKNPU bridge reloc patched addr overflow: dma=" << dma
                    << " delta=" << reloc_delta << " task=" << ti;
        }
        return false;
      }
      regs[off + reloc.cmd_index] =
          (u_cur & 0xFFFF00000000FFFFULL) | ((patched_addr & 0xFFFFFFFFULL) << 16);
      addr_patch_writes += 1;
      reloc_entries += 1;
      if (write_back) reloc_writebacks += 1;
    }
    // Match RKNPURuntime::BuildTaskStruct semantics: zero task struct first,
    // then fill required fields explicitly.
    std::memset(&tasks[ti], 0, sizeof(rknpu::RknpuTask));
    tasks[ti].flags = 0;
    tasks[ti].op_idx = 0;
    tasks[ti].enable_mask = t.enable_mask;
    tasks[ti].int_mask = t.int_mask;
    tasks[ti].int_clear = t.int_clear;
    tasks[ti].int_status = 0;
    tasks[ti].regcfg_amount = t.regcfg_amount;
    tasks[ti].regcfg_offset = 0;
    tasks[ti].regcmd_addr = regcmd_buf.dma_addr + static_cast<uint32_t>(off * 8);
    off += t.regcmds.size();
  }
  // Stitch intermediate tasks into a single PC chain by patching each
  // task's PC_BASE_ADDRESS / PC_REGISTER_AMOUNTS in the 4-entry tail.
  size_t cmd_off = 0;
  int64_t pc_tail_patch_writes = 0;
  for (size_t ti = 0; ti + 1 < tasks_to_submit.size(); ++ti) {
    const auto& cur = tasks_to_submit[ti];
    const auto& next = tasks_to_submit[ti + 1];
    if (cur.regcmds.size() >= 4) {
      size_t tail = cmd_off + cur.regcmds.size() - 4;
      uint64_t next_dma = static_cast<uint64_t>(regcmd_buf.dma_addr) +
                          static_cast<uint64_t>(cmd_off + cur.regcmds.size()) * 8ULL;
      uint64_t next_amt = static_cast<uint64_t>(next.regcmds.size() / 2 - 1);
      regs[tail] = (static_cast<uint64_t>(kOpRegPc) << 48) |
                   ((next_dma & 0xFFFFFFFFULL) << 16) | kPcBaseAddress;
      regs[tail + 1] = (static_cast<uint64_t>(kOpRegPc) << 48) |
                       ((next_amt & 0xFFFFFFFFULL) << 16) | kPcRegisterAmounts;
      pc_tail_patch_writes += 2;
      }
      cmd_off += cur.regcmds.size();
  }
  if (BridgeDebugDmaEnabled()) {
    BridgeStats& dbg_stats = GetBridgeStats();
    for (size_t i = 0; i < host_dma.size(); ++i) {
      BridgeStats::HostDmaDebugEntry dbg;
      dbg.submit_slot = submit_slot;
      dbg.host_ptr = reinterpret_cast<uintptr_t>(host_ptrs[i]);
      dbg.dma_addr = host_dma[i].buf.dma_addr;
      dbg.bytes = static_cast<int64_t>(host_dma[i].bytes);
      dbg.write_back = host_dma[i].write_back;
      dbg.persistent_cached = host_dma[i].persistent_cached;
      dbg.persistent_cache_hit = host_dma[i].persistent_cache_hit;
      dbg.upload_skipped = host_dma[i].upload_skipped;
      dbg.sync_to_device_requested = host_dma[i].sync_to_device;
      dbg.chain_reused = host_dma[i].chain_reused;
      dbg_stats.host_dma_debug.push_back(std::move(dbg));
    }
  }
  DumpRegcmdSample("RKNPU chain submit regcmd sample", regs, total_cmds);
  if (use_reloc_for_submit) {
    auto sync_begin = BridgeClock::now();
    for (auto& entry : host_dma) {
      if (entry.bytes == 0) continue;
      if (entry.write_back) {
        // Match RKNPURuntime::Run behavior: clean output buffer cache lines
        // before NPU writes, otherwise stale CPU cache may corrupt readback.
        dev->SyncToDevice(entry.buf);
        data_sync_to_device_bytes += static_cast<int64_t>(entry.bytes);
      } else if (entry.sync_to_device) {
        dev->SyncToDevice(entry.buf);
        entry.sync_to_device = false;
        data_sync_to_device_bytes += static_cast<int64_t>(entry.bytes);
      }
    }
    sync_to_device_ns +=
        std::chrono::duration_cast<BridgeNs>(BridgeClock::now() - sync_begin).count();
  }
  {
    auto sync_begin = BridgeClock::now();
    dev->SyncToDevice(regcmd_buf);
    dev->SyncToDevice(task_buf);
    meta_sync_to_device_bytes += static_cast<int64_t>(regcmd_buf.size);
    meta_sync_to_device_bytes += static_cast<int64_t>(task_buf.size);
    sync_to_device_ns +=
        std::chrono::duration_cast<BridgeNs>(BridgeClock::now() - sync_begin).count();
  }
  prep_ns = std::chrono::duration_cast<BridgeNs>(BridgeClock::now() - submit_begin).count();
  int rc = 0;
  {
    auto submit_begin_tp = BridgeClock::now();
    rc = dev->TrySubmit(task_buf, /*core_mask=*/1, /*num_tasks=*/tasks_to_submit.size(),
                        /*task_start=*/0, /*flags=*/-1, /*timeout=*/1000, &hw);
    submit_ns +=
        std::chrono::duration_cast<BridgeNs>(BridgeClock::now() - submit_begin_tp).count();
  }
  bool used_reloc_submit = rc >= 0;
  if (use_reloc_for_submit) {
    stats.reloc_submit_calls += 1;
  }
  if (use_reloc_for_submit && rc < 0) {
    stats.reloc_submit_fallbacks += 1;
    dev->Reset();
    free_host_dma_entries(&host_dma);
    host_dma.clear();
    host_ptrs.clear();
    host_index.clear();
    size_t tmp_off = 0;
    for (const auto& t : tasks_to_submit) {
      for (size_t i = 0; i < t.regcmds.size(); ++i) {
        regs[tmp_off + i] = patch_addr_legacy(t.regcmds[i]);
      }
      tmp_off += t.regcmds.size();
    }
    dev->SyncToDevice(regcmd_buf);
    int64_t hw2 = 0;
    int rc2 = 0;
    {
      auto submit_begin_tp = BridgeClock::now();
      rc2 = dev->TrySubmit(task_buf, /*core_mask=*/1, /*num_tasks=*/tasks_to_submit.size(),
                           /*task_start=*/0, /*flags=*/-1, /*timeout=*/1000, &hw2);
      submit_ns +=
          std::chrono::duration_cast<BridgeNs>(BridgeClock::now() - submit_begin_tp).count();
    }
    if (rc2 >= 0) {
      if (BridgeLogEnabled()) {
        LOG(INFO) << "RKNPU bridge submit recovered via legacy patch fallback";
      }
      rc = rc2;
      hw = hw2;
      used_reloc_submit = false;
    } else if (BridgeLogEnabled()) {
      LOG(INFO) << "RKNPU bridge legacy patch fallback failed rc=" << rc2
                << " errno_str=" << strerror(-rc2);
    }
  }
  if (rc < 0) {
    if (BridgeLogEnabled()) {
      LOG(INFO) << "RKNPU bridge synthetic submit failed rc=" << rc
                << " errno_str=" << strerror(-rc);
    }
    free_host_dma_entries(&host_dma);
    dev->Reset();
    record_submit_timing(false, pc_tail_patch_writes, reloc_writebacks);
    return false;
  }
  stats.submitted_tasks += static_cast<int64_t>(tasks_to_submit.size());
  stats.submitted_regcmd_qwords += static_cast<int64_t>(total_cmds);
  stats.submitted_regcmd_bytes += static_cast<int64_t>(regcmd_bytes);
  stats.patch_addr_writes += addr_patch_writes;
  stats.patch_pc_tail_writes += pc_tail_patch_writes;
  stats.reloc_entries_patched += reloc_entries;
  stats.reloc_writeback_entries += reloc_writebacks;
  stats.host_dma_buffers += host_dma_buffers;
  stats.writeback_dma_buffers += writeback_dma_buffers;
  stats.chain_reuse_hits += chain_reuse_hits;
  stats.chain_reuse_bytes += chain_reuse_bytes;
  stats.data_sync_to_device_bytes += data_sync_to_device_bytes;
  stats.meta_sync_to_device_bytes += meta_sync_to_device_bytes;
  bool output_check_ok = true;
  if (use_reloc_for_submit && used_reloc_submit) {
    for (size_t i = 0; i < host_dma.size(); ++i) {
      if (host_dma[i].write_back && host_dma[i].bytes > 0 && host_ptrs[i] != nullptr) {
        auto sync_begin = BridgeClock::now();
        dev->SyncFromDevice(host_dma[i].buf);
        data_sync_from_device_bytes += static_cast<int64_t>(host_dma[i].bytes);
        sync_from_device_ns +=
            std::chrono::duration_cast<BridgeNs>(BridgeClock::now() - sync_begin).count();
        auto wit = writeback_gather_mn.find(host_ptrs[i]);
        if (wit != writeback_gather_mn.end()) {
          const WriteBackGatherMN& g = wit->second;
          GatherOutputFP16(static_cast<uint16_t*>(host_ptrs[i]),
                           host_dma[i].buf.As<const uint16_t>(), g.m, g.n, PadM(g.m),
                           g.n_aligned);
          if (BridgeOutputChecksEnabled()) {
            uint64_t host_ck = checksum_bytes(host_ptrs[i], static_cast<size_t>(g.m) * g.n * 2);
            stats.output_checksum_checks += 1;
            stats.output_checksum_last = static_cast<int64_t>(host_ck);
          }
        } else {
          auto sit = writeback_gather_spatial.find(host_ptrs[i]);
          if (sit != writeback_gather_spatial.end()) {
            const WriteBackGatherSpatial& g = sit->second;
            GatherSpatialOutputFP16(static_cast<uint16_t*>(host_ptrs[i]),
                                    host_dma[i].buf.As<const uint16_t>(), g.channels, g.h_out, g.w_out,
                                    g.m_padded, g.n_aligned);
            if (BridgeOutputChecksEnabled()) {
              size_t host_bytes = static_cast<size_t>(g.channels) * g.h_out * g.w_out * 2;
              uint64_t host_ck = checksum_bytes(host_ptrs[i], host_bytes);
              stats.output_checksum_checks += 1;
              stats.output_checksum_last = static_cast<int64_t>(host_ck);
            }
          } else {
          if (BridgeOutputChecksEnabled()) {
            uint64_t dma_ck = checksum_bytes(host_dma[i].buf.As<void>(), host_dma[i].bytes);
            std::memcpy(host_ptrs[i], host_dma[i].buf.As<void>(), host_dma[i].bytes);
            uint64_t host_ck = checksum_bytes(host_ptrs[i], host_dma[i].bytes);
            stats.output_checksum_checks += 1;
            stats.output_checksum_last = static_cast<int64_t>(host_ck);
            if (dma_ck != host_ck) {
              stats.output_checksum_mismatch += 1;
              output_check_ok = false;
              if (BridgeLogEnabled()) {
                LOG(INFO) << "RKNPU bridge output checksum mismatch: task_index=" << i
                          << " bytes=" << host_dma[i].bytes;
              }
            }
          } else {
            std::memcpy(host_ptrs[i], host_dma[i].buf.As<void>(), host_dma[i].bytes);
          }
          }
        }
      }
      if (!host_dma[i].persistent_cached) {
        dev->Free(host_dma[i].buf);
      }
    }
  }
  if (!output_check_ok) {
    dev->Reset();
    record_submit_timing(false, pc_tail_patch_writes, reloc_writebacks);
    return false;
  }
  stats.data_sync_from_device_bytes += data_sync_from_device_bytes;
  (void)hw;
  record_submit_timing(true, pc_tail_patch_writes, reloc_writebacks);
  return true;
#else
  return false;
#endif
}

}  // namespace

extern "C" int rknpu_submit_matmul_stage(void* a, void* b, void* c, int m, int k, int n) {
  TouchBridgeDevice(static_cast<size_t>(m) * n * sizeof(uint16_t));
  return BridgeComputeMatmul(static_cast<const uint16_t*>(a), static_cast<const uint16_t*>(b),
                             static_cast<uint16_t*>(c), m, k, n);
}

extern "C" int rknpu_submit_add_stage(void* a, void* b, void* c, int m, int n, int bias_1d) {
  TouchBridgeDevice(static_cast<size_t>(m) * n * sizeof(uint16_t));
  return BridgeComputeAdd(static_cast<const uint16_t*>(a), static_cast<const uint16_t*>(b),
                          static_cast<uint16_t*>(c), m, n, bias_1d);
}

extern "C" int rknpu_submit_mul_stage(void* a, void* b, void* c, int m, int n, int bias_1d) {
  TouchBridgeDevice(static_cast<size_t>(m) * n * sizeof(uint16_t));
  return BridgeComputeMul(static_cast<const uint16_t*>(a), static_cast<const uint16_t*>(b),
                          static_cast<uint16_t*>(c), m, n, bias_1d);
}

extern "C" int rknpu_submit_exp_stage(void* a, void* c, int m, int n) {
  TouchBridgeDevice(static_cast<size_t>(m) * n * sizeof(uint16_t));
  return BridgeComputeExp(static_cast<const uint16_t*>(a), static_cast<uint16_t*>(c), m, n);
}

extern "C" int rknpu_submit_reciprocal_stage(void* a, void* c, int m, int n) {
  TouchBridgeDevice(static_cast<size_t>(m) * n * sizeof(uint16_t));
  return BridgeComputeReciprocal(static_cast<const uint16_t*>(a), static_cast<uint16_t*>(c), m, n);
}

extern "C" int rknpu_submit_gelu_stage(void* a, void* c, int m, int n) {
  TouchBridgeDevice(static_cast<size_t>(m) * n * sizeof(uint16_t));
  return BridgeComputeGelu(static_cast<const uint16_t*>(a), static_cast<uint16_t*>(c), m, n);
}

extern "C" int rknpu_submit_relu_stage(void* a, void* c, int m, int n) {
  TouchBridgeDevice(static_cast<size_t>(m) * n * sizeof(uint16_t));
  return BridgeComputeRelu(static_cast<const uint16_t*>(a), static_cast<uint16_t*>(c), m, n);
}

extern "C" int rknpu_submit_relu_stage_4d(void* a, void* c, int n, int ch, int h, int w) {
  TouchBridgeDevice(static_cast<size_t>(n) * ch * h * w * sizeof(uint16_t));
  return BridgeComputeRelu4D(static_cast<const uint16_t*>(a), static_cast<uint16_t*>(c), n, ch, h,
                             w);
}

extern "C" int rknpu_submit_conv2d_stage(void* data, void* weight, void* out, int n, int c, int h,
                                         int w, int oc, int kh, int kw, int oh, int ow, int sh,
                                         int sw, int pt, int pl, int /*pb*/, int /*pr*/) {
  TouchBridgeDevice(static_cast<size_t>(n) * oc * oh * ow * sizeof(uint16_t));
  return BridgeComputeConv2D(static_cast<const uint16_t*>(data), static_cast<const uint16_t*>(weight),
                             static_cast<uint16_t*>(out), n, c, h, w, oc, kh, kw, oh, ow, sh, sw,
                             pt, pl);
}

extern "C" int rknpu_submit_chain_stage(int num_tasks) {
  TouchBridgeDevice(static_cast<size_t>(num_tasks > 0 ? num_tasks : 1) * 256);
  return 0;
}

extern "C" int rknpu_submit_chain_stage_v2(int num_tasks, ...) {
  if (num_tasks <= 0) return -1;
  va_list ap;
  va_start(ap, num_tasks);
  int rc = 0;
  std::vector<BridgeStageInvocation> stages;
  stages.reserve(num_tasks);
  struct StageExec {
    int stage{0};
    std::vector<void*> ptrs;
    std::vector<int> ints;
  };
  std::vector<StageExec> exec;
  exec.reserve(num_tasks);
  for (int i = 0; i < num_tasks; ++i) {
    int stage = va_arg(ap, int);
    StageExec st;
    st.stage = stage;
    BridgeStageInvocation inv;
    inv.stage = stage;
    if (stage == kBridgeStageMatmul) {
      void* a = va_arg(ap, void*);
      void* b = va_arg(ap, void*);
      void* c = va_arg(ap, void*);
      int m = va_arg(ap, int);
      int k = va_arg(ap, int);
      int n = va_arg(ap, int);
      st.ptrs = {a, b, c};
      st.ints = {m, k, n};
      inv.ptrs = st.ptrs;
      inv.ptr_sizes = {static_cast<size_t>(m) * k * 2, static_cast<size_t>(k) * n * 2,
                       static_cast<size_t>(m) * n * 2};
    } else if (stage == kBridgeStageAdd || stage == kBridgeStageMul) {
      void* a = va_arg(ap, void*);
      void* b = va_arg(ap, void*);
      void* c = va_arg(ap, void*);
      int m = va_arg(ap, int);
      int n = va_arg(ap, int);
      int b_mode = va_arg(ap, int);
      st.ptrs = {a, b, c};
      st.ints = {m, n, b_mode};
      inv.ptrs = st.ptrs;
      inv.ptr_sizes = {static_cast<size_t>(m) * n * 2,
                       static_cast<size_t>(b_mode == 1 ? n : b_mode == 2 ? m : m * n) * 2,
                       static_cast<size_t>(m) * n * 2};
    } else if (stage == kBridgeStageRelu) {
      void* a = va_arg(ap, void*);
      void* c = va_arg(ap, void*);
      int m = va_arg(ap, int);
      int n = va_arg(ap, int);
      st.ptrs = {a, c};
      st.ints = {m, n};
      inv.ptrs = st.ptrs;
      inv.ptr_sizes = {static_cast<size_t>(m) * n * 2, static_cast<size_t>(m) * n * 2};
    } else if (stage == kBridgeStageExp || stage == kBridgeStageReciprocal ||
               stage == kBridgeStageGelu) {
      void* a = va_arg(ap, void*);
      void* c = va_arg(ap, void*);
      int m = va_arg(ap, int);
      int n = va_arg(ap, int);
      st.ptrs = {a, c};
      st.ints = {m, n};
      inv.ptrs = st.ptrs;
      inv.ptr_sizes = {static_cast<size_t>(m) * n * 2, static_cast<size_t>(m) * n * 2};
    } else if (stage == kBridgeStageRelu4D) {
      void* a = va_arg(ap, void*);
      void* c = va_arg(ap, void*);
      int n = va_arg(ap, int);
      int ch = va_arg(ap, int);
      int h = va_arg(ap, int);
      int w = va_arg(ap, int);
      st.ptrs = {a, c};
      st.ints = {n, ch, h, w};
      inv.ptrs = st.ptrs;
      inv.ptr_sizes = {static_cast<size_t>(n) * ch * h * w * 2,
                       static_cast<size_t>(n) * ch * h * w * 2};
    } else if (stage == kBridgeStageConv2D || stage == kBridgeStageConv2DRelu) {
      void* data = va_arg(ap, void*);
      void* weight = va_arg(ap, void*);
      void* out = va_arg(ap, void*);
      int n = va_arg(ap, int);
      int c = va_arg(ap, int);
      int h = va_arg(ap, int);
      int w = va_arg(ap, int);
      int oc = va_arg(ap, int);
      int kh = va_arg(ap, int);
      int kw = va_arg(ap, int);
      int oh = va_arg(ap, int);
      int ow = va_arg(ap, int);
      int sh = va_arg(ap, int);
      int sw = va_arg(ap, int);
      int pt = va_arg(ap, int);
      int pl = va_arg(ap, int);
      (void)va_arg(ap, int);  // pb
      (void)va_arg(ap, int);  // pr
      st.ptrs = {data, weight, out};
      st.ints = {n, c, h, w, oc, kh, kw, oh, ow, sh, sw, pt, pl};
      inv.ptrs = st.ptrs;
      inv.ptr_sizes = {
          static_cast<size_t>(n) * c * h * w * 2, static_cast<size_t>(oc) * c * kh * kw * 2,
          static_cast<size_t>(n) * oc * oh * ow * 2};
    } else if (stage == kBridgeStageMatmulBiasRelu) {
      void* a = va_arg(ap, void*);
      void* b = va_arg(ap, void*);
      void* bias = va_arg(ap, void*);
      void* c = va_arg(ap, void*);
      int m = va_arg(ap, int);
      int k = va_arg(ap, int);
      int n = va_arg(ap, int);
      st.ptrs = {a, b, bias, c};
      st.ints = {m, k, n};
      inv.ptrs = st.ptrs;
      inv.ptr_sizes = {static_cast<size_t>(m) * k * 2, static_cast<size_t>(k) * n * 2,
                       static_cast<size_t>(n) * 2, static_cast<size_t>(m) * n * 2};
    } else if (stage == kBridgeStageMatmulBias) {
      void* a = va_arg(ap, void*);
      void* b = va_arg(ap, void*);
      void* bias = va_arg(ap, void*);
      void* c = va_arg(ap, void*);
      int m = va_arg(ap, int);
      int k = va_arg(ap, int);
      int n = va_arg(ap, int);
      st.ptrs = {a, b, bias, c};
      st.ints = {m, k, n};
      inv.ptrs = st.ptrs;
      inv.ptr_sizes = {static_cast<size_t>(m) * k * 2, static_cast<size_t>(k) * n * 2,
                       static_cast<size_t>(n) * 2, static_cast<size_t>(m) * n * 2};
    } else if (stage == kBridgeStageAddRelu) {
      void* a = va_arg(ap, void*);
      void* b = va_arg(ap, void*);
      void* c = va_arg(ap, void*);
      int m = va_arg(ap, int);
      int n = va_arg(ap, int);
      int b_mode = va_arg(ap, int);
      st.ptrs = {a, b, c};
      st.ints = {m, n, b_mode};
      inv.ptrs = st.ptrs;
      inv.ptr_sizes = {static_cast<size_t>(m) * n * 2,
                       static_cast<size_t>(b_mode == 1 ? n : b_mode == 2 ? m : m * n) * 2,
                       static_cast<size_t>(m) * n * 2};
    } else {
      rc = -2;
      break;
    }
    inv.ints = st.ints;
    exec.push_back(std::move(st));
    stages.push_back(std::move(inv));
  }
  va_end(ap);
  if (rc != 0) return rc;
  if (BridgeLogEnabled() && BridgeCompareNpuCpuOutputEnabled()) {
    for (size_t i = 0; i < exec.size(); ++i) {
      const auto& st = exec[i];
      bool alias01 = st.ptrs.size() >= 2 && st.ptrs[0] == st.ptrs[1];
      bool alias02 = st.ptrs.size() >= 3 && st.ptrs[0] == st.ptrs[2];
      bool alias12 = st.ptrs.size() >= 3 && st.ptrs[1] == st.ptrs[2];
      bool alias03 = st.ptrs.size() >= 4 && st.ptrs[0] == st.ptrs[3];
      LOG(INFO) << "RKNPU bridge stage args: idx=" << i << " stage=" << st.stage
                << " ptrs=" << st.ptrs.size()
                << " ints=" << st.ints.size()
                << " i0=" << (st.ints.size() > 0 ? st.ints[0] : 0)
                << " i1=" << (st.ints.size() > 1 ? st.ints[1] : 0)
                << " i2=" << (st.ints.size() > 2 ? st.ints[2] : 0)
                << " p0=" << (st.ptrs.size() > 0 ? st.ptrs[0] : nullptr)
                << " p1=" << (st.ptrs.size() > 1 ? st.ptrs[1] : nullptr)
                << " p2=" << (st.ptrs.size() > 2 ? st.ptrs[2] : nullptr)
                << " p3=" << (st.ptrs.size() > 3 ? st.ptrs[3] : nullptr)
                << " alias01=" << alias01 << " alias02=" << alias02
                << " alias12=" << alias12 << " alias03=" << alias03;
    }
  }

  BridgeStats& stats = GetBridgeStats();
  stats.chain_calls += 1;
  int64_t reloc_submit_fallbacks_before = stats.reloc_submit_fallbacks;
  bool submitted = false;
  if (BridgeRealSubmitEnabled()) {
    submitted = BridgeSubmitConfiguredChain(num_tasks, stages);
    if (submitted) {
      stats.real_submit_ok += 1;
    } else {
      stats.real_submit_fail += 1;
    }
  }
  if (!submitted) {
    stats.touch_fallback += 1;
    if (BridgeFailOnFallbackEnabled()) {
      if (BridgeLogEnabled()) {
        LOG(INFO) << "RKNPU bridge strict fallback gate hit: synthetic touch fallback";
      }
      return -3;
    }
    TouchBridgeDevice(static_cast<size_t>(num_tasks) * 256);
  }
  if (BridgeFailOnFallbackEnabled() && stats.reloc_submit_fallbacks > reloc_submit_fallbacks_before) {
    if (BridgeLogEnabled()) {
      LOG(INFO) << "RKNPU bridge strict fallback gate hit: reloc submit fallback was used";
    }
    return -4;
  }
  if (submitted && BridgeFailOnFallbackEnabled() && BridgeRunCpuAfterSubmitEnabled()) {
    if (BridgeLogEnabled()) {
      LOG(INFO) << "RKNPU bridge strict fallback gate hit: CPU-after-submit is enabled";
    }
    return -5;
  }
  if (BridgeLogEnabled()) {
    LOG(INFO) << "RKNPU bridge stats: chain_calls=" << stats.chain_calls
              << " real_submit_ok=" << stats.real_submit_ok
              << " real_submit_fail=" << stats.real_submit_fail
              << " touch_fallback=" << stats.touch_fallback
              << " submitted_tasks=" << stats.submitted_tasks
              << " submitted_regcmd_qwords=" << stats.submitted_regcmd_qwords
              << " submitted_regcmd_bytes=" << stats.submitted_regcmd_bytes
              << " patch_addr_writes=" << stats.patch_addr_writes
              << " patch_pc_tail_writes=" << stats.patch_pc_tail_writes
              << " reloc_submit_calls=" << stats.reloc_submit_calls
              << " reloc_submit_fallbacks=" << stats.reloc_submit_fallbacks
              << " reloc_entries_patched=" << stats.reloc_entries_patched
              << " reloc_writeback_entries=" << stats.reloc_writeback_entries
              << " reloc_semantic_mismatch=" << stats.reloc_semantic_mismatch
              << " reloc_range_mismatch=" << stats.reloc_range_mismatch
              << " output_checksum_checks=" << stats.output_checksum_checks
              << " output_checksum_mismatch=" << stats.output_checksum_mismatch
              << " output_checksum_last=" << stats.output_checksum_last;
  }

  // Real submit already materializes outputs via writeback DMA.
  if (submitted && !BridgeRunCpuAfterSubmitEnabled()) {
    return 0;
  }

  struct StageOutputSnapshot {
    int stage{0};
    int stage_index{0};
    void* ptr{nullptr};
    size_t bytes{0};
    std::vector<uint8_t> npu;
  };
  std::vector<StageOutputSnapshot> stage_output_snapshots;
  auto output_arg_index = [&](int stage) -> int {
    switch (stage) {
      case kBridgeStageMatmul:
      case kBridgeStageAdd:
      case kBridgeStageMul:
      case kBridgeStageConv2D:
      case kBridgeStageConv2DRelu:
      case kBridgeStageAddRelu:
        return 2;
      case kBridgeStageRelu:
      case kBridgeStageRelu4D:
      case kBridgeStageExp:
      case kBridgeStageReciprocal:
      case kBridgeStageGelu:
        return 1;
      case kBridgeStageMatmulBiasRelu:
      case kBridgeStageMatmulBias:
        return 3;
      default:
        return -1;
    }
  };
  if (submitted && BridgeCompareNpuCpuOutputEnabled() && !exec.empty() &&
      stages.size() == exec.size()) {
    stage_output_snapshots.reserve(exec.size());
    for (size_t idx = 0; idx < exec.size(); ++idx) {
      int out_idx = output_arg_index(exec[idx].stage);
      if (out_idx < 0 || static_cast<size_t>(out_idx) >= exec[idx].ptrs.size() ||
          static_cast<size_t>(out_idx) >= stages[idx].ptr_sizes.size()) {
        continue;
      }
      void* out_ptr = exec[idx].ptrs[out_idx];
      size_t out_bytes = stages[idx].ptr_sizes[out_idx];
      if (out_ptr == nullptr || out_bytes == 0) {
        continue;
      }
      StageOutputSnapshot snap;
      snap.stage = exec[idx].stage;
      snap.stage_index = static_cast<int>(idx);
      snap.ptr = out_ptr;
      snap.bytes = out_bytes;
      snap.npu.resize(out_bytes);
      std::memcpy(snap.npu.data(), out_ptr, out_bytes);
      stage_output_snapshots.push_back(std::move(snap));
    }
  }

  auto count_non_finite_fp16 = [&](const uint8_t* data, size_t nbytes) -> size_t {
    size_t elems = nbytes / sizeof(uint16_t);
    size_t non_finite = 0;
    for (size_t i = 0; i < elems; ++i) {
      uint16_t h;
      std::memcpy(&h, data + i * sizeof(uint16_t), sizeof(uint16_t));
      uint16_t exp = static_cast<uint16_t>((h >> 10) & 0x1F);
      if (exp == 0x1F) non_finite += 1;
    }
    return non_finite;
  };

  auto compare_snapshot = [&](const StageOutputSnapshot& snap, const char* tag) {
    if (snap.ptr == nullptr || snap.bytes == 0 || snap.npu.empty()) return;
    const uint8_t* npu = snap.npu.data();
    const uint8_t* cpu = static_cast<const uint8_t*>(snap.ptr);
    size_t mismatch_bytes = 0;
    for (size_t i = 0; i < snap.bytes; ++i) {
      if (npu[i] != cpu[i]) mismatch_bytes += 1;
    }
    size_t elems = snap.bytes / sizeof(uint16_t);
    float max_abs = 0.0f;
    for (size_t i = 0; i < elems; ++i) {
      uint16_t hn;
      uint16_t hc;
      std::memcpy(&hn, npu + i * sizeof(uint16_t), sizeof(uint16_t));
      std::memcpy(&hc, cpu + i * sizeof(uint16_t), sizeof(uint16_t));
      float d = HalfToFloat(hn) - HalfToFloat(hc);
      if (d < 0.0f) d = -d;
      if (d > max_abs) max_abs = d;
    }
    size_t npu_nf = count_non_finite_fp16(npu, snap.bytes);
    size_t cpu_nf = count_non_finite_fp16(cpu, snap.bytes);
    LOG(INFO) << "RKNPU bridge NPU-vs-CPU output compare (" << tag << "): stage="
              << snap.stage << " stage_index=" << snap.stage_index << " bytes=" << snap.bytes
              << " mismatch_bytes=" << mismatch_bytes << " mismatch_ratio="
              << (snap.bytes > 0 ? static_cast<double>(mismatch_bytes) / static_cast<double>(snap.bytes)
                                 : 0.0)
              << " npu_non_finite=" << npu_nf << " cpu_non_finite=" << cpu_nf
              << " max_abs=" << max_abs;
  };

  auto snapshot_for_stage_index = [&](int idx) -> const StageOutputSnapshot* {
    for (const auto& s : stage_output_snapshots) {
      if (s.stage_index == idx) return &s;
    }
    return nullptr;
  };

  if (submitted && BridgeCompareNpuCpuOutputEnabled()) {
    for (const auto& snap : stage_output_snapshots) {
      if (snap.stage_index == static_cast<int>(exec.size()) - 1) {
        continue;
      }
      compare_snapshot(snap, "pre_cpu_fallback");
    }
  }

  for (size_t i = 0; i < exec.size(); ++i) {
    const auto& st = exec[i];
    if (st.stage == kBridgeStageMatmul) {
      rc = BridgeComputeMatmul(static_cast<const uint16_t*>(st.ptrs[0]),
                               static_cast<const uint16_t*>(st.ptrs[1]),
                               static_cast<uint16_t*>(st.ptrs[2]), st.ints[0], st.ints[1], st.ints[2]);
    } else if (st.stage == kBridgeStageAdd) {
      rc = BridgeComputeAdd(static_cast<const uint16_t*>(st.ptrs[0]),
                            static_cast<const uint16_t*>(st.ptrs[1]),
                            static_cast<uint16_t*>(st.ptrs[2]), st.ints[0], st.ints[1], st.ints[2]);
    } else if (st.stage == kBridgeStageMul) {
      rc = BridgeComputeMul(static_cast<const uint16_t*>(st.ptrs[0]),
                            static_cast<const uint16_t*>(st.ptrs[1]),
                            static_cast<uint16_t*>(st.ptrs[2]), st.ints[0], st.ints[1], st.ints[2]);
    } else if (st.stage == kBridgeStageRelu) {
      rc = BridgeComputeRelu(static_cast<const uint16_t*>(st.ptrs[0]),
                             static_cast<uint16_t*>(st.ptrs[1]), st.ints[0], st.ints[1]);
    } else if (st.stage == kBridgeStageExp) {
      rc = BridgeComputeExp(static_cast<const uint16_t*>(st.ptrs[0]),
                            static_cast<uint16_t*>(st.ptrs[1]), st.ints[0], st.ints[1]);
    } else if (st.stage == kBridgeStageReciprocal) {
      rc = BridgeComputeReciprocal(static_cast<const uint16_t*>(st.ptrs[0]),
                                   static_cast<uint16_t*>(st.ptrs[1]), st.ints[0], st.ints[1]);
    } else if (st.stage == kBridgeStageGelu) {
      rc = BridgeComputeGelu(static_cast<const uint16_t*>(st.ptrs[0]),
                                    static_cast<uint16_t*>(st.ptrs[1]), st.ints[0], st.ints[1]);
    } else if (st.stage == kBridgeStageRelu4D) {
      rc = BridgeComputeRelu4D(static_cast<const uint16_t*>(st.ptrs[0]),
                               static_cast<uint16_t*>(st.ptrs[1]), st.ints[0], st.ints[1], st.ints[2],
                               st.ints[3]);
    } else if (st.stage == kBridgeStageConv2D) {
      rc = BridgeComputeConv2D(static_cast<const uint16_t*>(st.ptrs[0]),
                               static_cast<const uint16_t*>(st.ptrs[1]),
                               static_cast<uint16_t*>(st.ptrs[2]), st.ints[0], st.ints[1], st.ints[2],
                               st.ints[3], st.ints[4], st.ints[5], st.ints[6], st.ints[7], st.ints[8],
                               st.ints[9], st.ints[10], st.ints[11], st.ints[12]);
    } else if (st.stage == kBridgeStageConv2DRelu) {
      rc = BridgeComputeConv2D(static_cast<const uint16_t*>(st.ptrs[0]),
                               static_cast<const uint16_t*>(st.ptrs[1]),
                               static_cast<uint16_t*>(st.ptrs[2]), st.ints[0], st.ints[1], st.ints[2],
                               st.ints[3], st.ints[4], st.ints[5], st.ints[6], st.ints[7], st.ints[8],
                               st.ints[9], st.ints[10], st.ints[11], st.ints[12]);
      if (rc == 0) {
        rc = BridgeComputeRelu4D(static_cast<const uint16_t*>(st.ptrs[2]),
                                 static_cast<uint16_t*>(st.ptrs[2]), st.ints[0], st.ints[4], st.ints[7],
                                 st.ints[8]);
      }
    } else if (st.stage == kBridgeStageMatmulBiasRelu) {
      rc = BridgeComputeMatmul(static_cast<const uint16_t*>(st.ptrs[0]),
                               static_cast<const uint16_t*>(st.ptrs[1]),
                               static_cast<uint16_t*>(st.ptrs[3]), st.ints[0], st.ints[1], st.ints[2]);
      if (rc == 0) {
        rc = BridgeComputeAdd(static_cast<const uint16_t*>(st.ptrs[3]),
                              static_cast<const uint16_t*>(st.ptrs[2]),
                              static_cast<uint16_t*>(st.ptrs[3]), st.ints[0], st.ints[2], 1);
      }
      if (rc == 0) {
        rc = BridgeComputeRelu(static_cast<const uint16_t*>(st.ptrs[3]),
                               static_cast<uint16_t*>(st.ptrs[3]), st.ints[0], st.ints[2]);
      }
    } else if (st.stage == kBridgeStageMatmulBias) {
      rc = BridgeComputeMatmul(static_cast<const uint16_t*>(st.ptrs[0]),
                               static_cast<const uint16_t*>(st.ptrs[1]),
                               static_cast<uint16_t*>(st.ptrs[3]), st.ints[0], st.ints[1], st.ints[2]);
      if (rc == 0) {
        rc = BridgeComputeAdd(static_cast<const uint16_t*>(st.ptrs[3]),
                              static_cast<const uint16_t*>(st.ptrs[2]),
                              static_cast<uint16_t*>(st.ptrs[3]), st.ints[0], st.ints[2], 1);
      }
    } else if (st.stage == kBridgeStageAddRelu) {
      rc = BridgeComputeAdd(static_cast<const uint16_t*>(st.ptrs[0]),
                            static_cast<const uint16_t*>(st.ptrs[1]),
                            static_cast<uint16_t*>(st.ptrs[2]), st.ints[0], st.ints[1], st.ints[2]);
      if (rc == 0) {
        rc = BridgeComputeRelu(static_cast<const uint16_t*>(st.ptrs[2]),
                               static_cast<uint16_t*>(st.ptrs[2]), st.ints[0], st.ints[1]);
      }
    }
    if (rc == 0 && submitted && BridgeCompareNpuCpuOutputEnabled()) {
      const StageOutputSnapshot* snap = snapshot_for_stage_index(static_cast<int>(i));
      if (snap != nullptr) compare_snapshot(*snap, "post_cpu_stage");
    }
    if (rc != 0) break;
  }
  return rc;
}

// ---------------------------------------------------------------------------
// Factory functions
// ---------------------------------------------------------------------------

class RKNPUBridgeMetadataModule final : public ffi::ModuleObj {
 public:
  explicit RKNPUBridgeMetadataModule(std::string chain_blob, std::string schedule_report_json)
      : chain_blob_(std::move(chain_blob)),
        schedule_report_json_(std::move(schedule_report_json)) {}

  const char* kind() const final { return "rknpu_bridge_metadata"; }

  int GetPropertyMask() const final { return ffi::Module::kBinarySerializable; }

  ffi::Optional<ffi::Function> GetFunction(const ffi::String& name) final {
    if (name == "rknpu_bridge_get_chain_blob") {
      return ffi::Function([this](ffi::PackedArgs args, ffi::Any* rv) {
        *rv = ffi::Bytes(chain_blob_);
      });
    }
    if (name == "rknpu_bridge_apply_chain_blob") {
      return ffi::Function([this](ffi::PackedArgs args, ffi::Any* rv) {
        if (!chain_blob_.empty()) {
          SetBridgeChainBlob(ffi::Bytes(chain_blob_));
        }
      });
    }
    if (name == "rknpu_schedule_report_json") {
      return ffi::Function([this](ffi::PackedArgs args, ffi::Any* rv) {
        *rv = ffi::String(schedule_report_json_);
      });
    }
    return std::nullopt;
  }

  ffi::Bytes SaveToBytes() const final {
    std::string result;
    support::BytesOutStream stream(&result);
    stream.Write(chain_blob_);
    stream.Write(schedule_report_json_);
    return ffi::Bytes(std::move(result));
  }

  static ffi::Module LoadFromBytes(const ffi::Bytes& bytes) {
    support::BytesInStream stream(bytes);
    std::string chain_blob;
    std::string schedule_report_json;
    TVM_FFI_ICHECK(stream.Read(&chain_blob))
        << "Failed to load RKNPU bridge metadata chain blob";
    TVM_FFI_ICHECK(stream.Read(&schedule_report_json))
        << "Failed to load RKNPU bridge metadata schedule report";
    return ffi::Module(
        ffi::make_object<RKNPUBridgeMetadataModule>(chain_blob, schedule_report_json));
  }

 private:
  std::string chain_blob_;
  std::string schedule_report_json_;
};

ffi::Module RKNPUBridgeMetadataModuleCreate(const ffi::Bytes& chain_blob,
                                            const ffi::String& schedule_report_json) {
  return ffi::Module(ffi::make_object<RKNPUBridgeMetadataModule>(
      std::string(chain_blob.data(), chain_blob.size()), std::string(schedule_report_json)));
}

ffi::Module RKNPURuntimeCreate(const ffi::String& symbol_name, const ffi::String& graph_json,
                               const ffi::Array<ffi::String>& const_names,
                               const ffi::Bytes& regcmd_data) {
  auto n = ffi::make_object<RKNPURuntime>(
      symbol_name, graph_json, const_names,
      std::string(regcmd_data.data(), regcmd_data.size()));
  return ffi::Module(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("runtime.rknpu_runtime_create", RKNPURuntimeCreate)
      .def("runtime.rknpu_bridge_metadata_module_create", RKNPUBridgeMetadataModuleCreate)
      .def("runtime.rknpu_bridge_set_synthetic_task", SetBridgeSyntheticTask)
      .def("runtime.rknpu_bridge_set_chain_blob", SetBridgeChainBlob)
      .def("runtime.rknpu_bridge_set_synthetic_chain", SetBridgeSyntheticChain)
      .def("runtime.rknpu_bridge_get_stats_json", []() { return BridgeStatsJSON(); })
      .def("runtime.rknpu_bridge_reset_stats", []() { BridgeResetStats(); })
      .def("ffi.Module.load_from_bytes.rknpu_bridge_metadata",
           RKNPUBridgeMetadataModule::LoadFromBytes)
      .def("ffi.Module.load_from_bytes.rknpu", RKNPURuntime::LoadFromBytesRKNPU);
}

}  // namespace contrib
}  // namespace runtime
}  // namespace tvm
