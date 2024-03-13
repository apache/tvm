# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Tests for MMA m16n8k8 Auto Tensorization"""

import tempfile
import numpy as np

import tvm
from tvm import te
from tvm import meta_schedule as ms
from tvm._ffi import register_func
from tvm.meta_schedule.testing.space_generation import (
    check_sketches,
    generate_design_space,
)
from tvm.meta_schedule.builder import LocalBuilder
from tvm.script import ir as I
from tvm.script import tir as T
from tvm.target import Target
from tvm.tir import Schedule
from tvm.tir.schedule import Trace

# get tensor intrin
from tvm.tir.tensor_intrin import cuda  # pylint: disable=unused-import

import tvm.testing


@I.ir_module
class MmaModule:
    @T.prim_func
    def main(
        X: T.Buffer((4096, 4096), "float16"),
        Y: T.Buffer((4096, 4096), "float16"),
        C: T.Buffer((4096, 4096), "float16"),
    ):
        T.func_attr({"global_symbol": "main", "tir.noalias": T.bool(True)})
        # with T.block("root"):
        C_reindex_m16n8k8_matrixC = T.alloc_buffer((4096, 4096), "float16", scope="m16n8k8.matrixC")
        X_reindex_shared_dyn = T.alloc_buffer((4096, 4096), "float16", scope="shared.dyn")
        Y_reindex_shared_dyn = T.alloc_buffer((4096, 4096), "float16", scope="shared.dyn")
        X_reindex_shared_dyn_m16n8k8_matrixA = T.alloc_buffer(
            (4096, 4096), "float16", scope="m16n8k8.matrixA"
        )
        Y_reindex_shared_dyn_m16n8k8_matrixB = T.alloc_buffer(
            (4096, 4096), "float16", scope="m16n8k8.matrixB"
        )
        for ax0_0_0_ax1_0_0_fused in T.thread_binding(4, thread="blockIdx.x"):
            for ax0_0_1_ax1_0_1_fused in T.thread_binding(256, thread="blockIdx.y"):
                for ax0_0_2_ax1_0_2_fused in T.thread_binding(4, thread="threadIdx.y"):
                    for ax2_0_0 in T.serial(
                        128,
                        annotations={
                            "software_pipeline_async_stages": [0],
                            "software_pipeline_order": [0, 1, 3, 2, 4],
                            "software_pipeline_stage": [0, 0, 1, 2, 2],
                        },
                    ):
                        with T.block("X_reindex_shared.dyn"):
                            v0, v1 = T.axis.remap("SS", [ax0_0_1_ax1_0_1_fused, ax2_0_0])
                            T.reads(X[v0 // 8 * 128 : v0 // 8 * 128 + 128, v1 * 32 : v1 * 32 + 32])
                            T.writes(
                                X_reindex_shared_dyn[
                                    v0 // 8 * 128 : v0 // 8 * 128 + 128, v1 * 32 : v1 * 32 + 32
                                ]
                            )
                            T.block_attr(
                                {
                                    "auto_copy": 1,
                                    "buffer_dim_align": [[0, 0, 32, 8]],
                                    "permuted_layout": "g2s_A",
                                    "vector_bytes": 16,
                                }
                            )
                            for ax0, ax1 in T.grid(128, 32):
                                X_reindex_shared_dyn[v0 // 8 * 128 + ax0, v1 * 32 + ax1] = X[
                                    v0 // 8 * 128 + ax0, v1 * 32 + ax1
                                ]
                        with T.block("Y_reindex_shared.dyn"):
                            v0, v1, v2 = T.axis.remap(
                                "SSS", [ax2_0_0, ax0_0_0_ax1_0_0_fused, ax0_0_1_ax1_0_1_fused]
                            )
                            T.reads(
                                Y[
                                    v0 * 32 : v0 * 32 + 32,
                                    v1 * 1024 + v2 % 8 * 128 : v1 * 1024 + v2 % 8 * 128 + 128,
                                ]
                            )
                            T.writes(
                                Y_reindex_shared_dyn[
                                    v0 * 32 : v0 * 32 + 32,
                                    v1 * 1024 + v2 % 8 * 128 : v1 * 1024 + v2 % 8 * 128 + 128,
                                ]
                            )
                            T.block_attr(
                                {
                                    "auto_copy": 1,
                                    "buffer_dim_align": [[0, 0, 32, 8]],
                                    "permuted_layout": "g2s_B",
                                    "vector_bytes": 16,
                                }
                            )
                            for ax0, ax1 in T.grid(32, 128):
                                Y_reindex_shared_dyn[
                                    v0 * 32 + ax0, v1 * 1024 + v2 % 8 * 128 + ax1
                                ] = Y[v0 * 32 + ax0, v1 * 1024 + v2 % 8 * 128 + ax1]
                        for ax2_0_1 in T.serial(
                            4,
                            annotations={
                                "software_pipeline_order": [0, 1, 2],
                                "software_pipeline_stage": [0, 0, 1],
                            },
                        ):
                            for ax0_0, ax1_0 in T.grid(2, 1):
                                with T.block("X_reindex_shared.dyn_m16n8k8.matrixA_o"):
                                    v0_o = T.axis.spatial(
                                        128,
                                        ax0_0_1_ax1_0_1_fused // 8 * 4
                                        + ax0_0_2_ax1_0_2_fused // 2 * 2
                                        + ax0_0,
                                    )
                                    v1_o = T.axis.spatial(512, ax2_0_0 * 4 + ax2_0_1 + ax1_0)
                                    T.reads(
                                        X_reindex_shared_dyn[
                                            v0_o * 32 : v0_o * 32 + 32, v1_o * 8 : v1_o * 8 + 8
                                        ]
                                    )
                                    T.writes(
                                        X_reindex_shared_dyn_m16n8k8_matrixA[
                                            v0_o * 32 : v0_o * 32 + 32, v1_o * 8 : v1_o * 8 + 8
                                        ]
                                    )
                                    T.block_attr(
                                        {
                                            "meta_schedule.auto_tensorize": "mma_load_m16n8k8_f16_A_shared_dyn",
                                            "permuted_layout": "s2l_A",
                                        }
                                    )
                                    for ax0_1, ax1_1 in T.grid(32, 8):
                                        with T.block("X_reindex_shared.dyn_m16n8k8.matrixA"):
                                            v0_i, v1_i = T.axis.remap("SS", [ax0_1, ax1_1])
                                            T.reads(
                                                X_reindex_shared_dyn[
                                                    v0_o * 32 + v0_i, v1_o * 8 + v1_i
                                                ]
                                            )
                                            T.writes(
                                                X_reindex_shared_dyn_m16n8k8_matrixA[
                                                    v0_o * 32 + v0_i, v1_o * 8 + v1_i
                                                ]
                                            )
                                            X_reindex_shared_dyn_m16n8k8_matrixA[
                                                v0_o * 32 + v0_i, v1_o * 8 + v1_i
                                            ] = X_reindex_shared_dyn[
                                                v0_o * 32 + v0_i, v1_o * 8 + v1_i
                                            ]
                            for ax0_0, ax1_0 in T.grid(1, 2):
                                with T.block("Y_reindex_shared.dyn_m16n8k8.matrixB_o"):
                                    v0_o = T.axis.spatial(512, ax2_0_0 * 4 + ax2_0_1 + ax0_0)
                                    v1_o = T.axis.spatial(
                                        128,
                                        ax0_0_0_ax1_0_0_fused * 32
                                        + ax0_0_1_ax1_0_1_fused % 8 * 4
                                        + ax0_0_2_ax1_0_2_fused % 2 * 2
                                        + ax1_0,
                                    )
                                    T.reads(
                                        Y_reindex_shared_dyn[
                                            v0_o * 8 : v0_o * 8 + 8, v1_o * 32 : v1_o * 32 + 32
                                        ]
                                    )
                                    T.writes(
                                        Y_reindex_shared_dyn_m16n8k8_matrixB[
                                            v0_o * 8 : v0_o * 8 + 8, v1_o * 32 : v1_o * 32 + 32
                                        ]
                                    )
                                    T.block_attr(
                                        {
                                            "meta_schedule.auto_tensorize": "mma_load_m16n8k8_f16_B_shared_dyn",
                                            "permuted_layout": "s2l_B",
                                        }
                                    )
                                    for ax0_1, ax1_1 in T.grid(8, 32):
                                        with T.block("Y_reindex_shared.dyn_m16n8k8.matrixB"):
                                            v0_i, v1_i = T.axis.remap("SS", [ax0_1, ax1_1])
                                            T.reads(
                                                Y_reindex_shared_dyn[
                                                    v0_o * 8 + v0_i, v1_o * 32 + v1_i
                                                ]
                                            )
                                            T.writes(
                                                Y_reindex_shared_dyn_m16n8k8_matrixB[
                                                    v0_o * 8 + v0_i, v1_o * 32 + v1_i
                                                ]
                                            )
                                            Y_reindex_shared_dyn_m16n8k8_matrixB[
                                                v0_o * 8 + v0_i, v1_o * 32 + v1_i
                                            ] = Y_reindex_shared_dyn[
                                                v0_o * 8 + v0_i, v1_o * 32 + v1_i
                                            ]
                            for ax0_0_3, ax1_0_3, ax2_0_2, ax0_0_4, ax1_0_4 in T.grid(
                                1, 1, 1, 4, 8
                            ):
                                with T.block("C_o"):
                                    v0_o = T.axis.spatial(
                                        256,
                                        ax0_0_1_ax1_0_1_fused // 8 * 8
                                        + ax0_0_2_ax1_0_2_fused // 2 * 4
                                        + ax0_0_3 * 4
                                        + ax0_0_4,
                                    )
                                    v1_o = T.axis.spatial(
                                        512,
                                        ax0_0_0_ax1_0_0_fused * 128
                                        + ax0_0_1_ax1_0_1_fused % 8 * 16
                                        + ax0_0_2_ax1_0_2_fused % 2 * 8
                                        + ax1_0_3 * 8
                                        + ax1_0_4,
                                    )
                                    v2_o = T.axis.reduce(512, ax2_0_0 * 4 + ax2_0_1 + ax2_0_2)
                                    T.reads(
                                        X_reindex_shared_dyn_m16n8k8_matrixA[
                                            v0_o * 16 : v0_o * 16 + 16, v2_o * 8 : v2_o * 8 + 8
                                        ],
                                        Y_reindex_shared_dyn_m16n8k8_matrixB[
                                            v2_o * 8 : v2_o * 8 + 8, v1_o * 8 : v1_o * 8 + 8
                                        ],
                                    )
                                    T.writes(
                                        C_reindex_m16n8k8_matrixC[
                                            v0_o * 16 : v0_o * 16 + 16, v1_o * 8 : v1_o * 8 + 8
                                        ]
                                    )
                                    T.block_attr(
                                        {
                                            "meta_schedule.auto_tensorize": "mma_sync_m16n8k8_f16f16f16",
                                            "meta_schedule.auto_tensorize_init": "mma_init_m16n8k8_f16",
                                            "meta_schedule.thread_extent_high_inclusive": 1024,
                                            "meta_schedule.thread_extent_low_inclusive": 32,
                                            "warp_execution": 1,
                                        }
                                    )
                                    with T.init():
                                        for ax0_1, ax1_1 in T.grid(16, 8):
                                            with T.block("C_init"):
                                                v0_i_init, v1_i_init = T.axis.remap(
                                                    "SS", [ax0_1, ax1_1]
                                                )
                                                T.reads()
                                                T.writes(
                                                    C_reindex_m16n8k8_matrixC[
                                                        v0_o * 16 + v0_i_init, v1_o * 8 + v1_i_init
                                                    ]
                                                )
                                                C_reindex_m16n8k8_matrixC[
                                                    v0_o * 16 + v0_i_init, v1_o * 8 + v1_i_init
                                                ] = T.float16(0)
                                    for ax0_1, ax1_1, ax2_1 in T.grid(16, 8, 8):
                                        with T.block("C"):
                                            v0_i, v1_i, v2_i = T.axis.remap(
                                                "SSR", [ax0_1, ax1_1, ax2_1]
                                            )
                                            T.reads(
                                                C_reindex_m16n8k8_matrixC[
                                                    v0_o * 16 + v0_i, v1_o * 8 + v1_i
                                                ],
                                                X_reindex_shared_dyn_m16n8k8_matrixA[
                                                    v0_o * 16 + v0_i, v2_o * 8 + v2_i
                                                ],
                                                Y_reindex_shared_dyn_m16n8k8_matrixB[
                                                    v2_o * 8 + v2_i, v1_o * 8 + v1_i
                                                ],
                                            )
                                            T.writes(
                                                C_reindex_m16n8k8_matrixC[
                                                    v0_o * 16 + v0_i, v1_o * 8 + v1_i
                                                ]
                                            )
                                            T.block_attr(
                                                {"meta_schedule.tiling_structure": "SSSRRSRS"}
                                            )
                                            C_reindex_m16n8k8_matrixC[
                                                v0_o * 16 + v0_i, v1_o * 8 + v1_i
                                            ] = (
                                                C_reindex_m16n8k8_matrixC[
                                                    v0_o * 16 + v0_i, v1_o * 8 + v1_i
                                                ]
                                                + X_reindex_shared_dyn_m16n8k8_matrixA[
                                                    v0_o * 16 + v0_i, v2_o * 8 + v2_i
                                                ]
                                                * Y_reindex_shared_dyn_m16n8k8_matrixB[
                                                    v2_o * 8 + v2_i, v1_o * 8 + v1_i
                                                ]
                                            )
                    with T.block("C_reindex_m16n8k8.matrixC"):
                        v0, v1, v2 = T.axis.remap(
                            "SSS",
                            [ax0_0_1_ax1_0_1_fused, ax0_0_2_ax1_0_2_fused, ax0_0_0_ax1_0_0_fused],
                        )
                        T.reads(
                            C_reindex_m16n8k8_matrixC[
                                v0 // 8 * 128 + v1 // 2 * 64 : v0 // 8 * 128 + v1 // 2 * 64 + 64,
                                v2 * 1024
                                + v0 % 8 * 128
                                + v1 % 2 * 64 : v2 * 1024
                                + v0 % 8 * 128
                                + v1 % 2 * 64
                                + 64,
                            ]
                        )
                        T.writes(
                            C[
                                v0 // 8 * 128 + v1 // 2 * 64 : v0 // 8 * 128 + v1 // 2 * 64 + 64,
                                v2 * 1024
                                + v0 % 8 * 128
                                + v1 % 2 * 64 : v2 * 1024
                                + v0 % 8 * 128
                                + v1 % 2 * 64
                                + 64,
                            ]
                        )
                        T.block_attr({"auto_copy": 1})
                        for ax0, ax1 in T.grid(64, 64):
                            C[
                                v0 // 8 * 128 + v1 // 2 * 64 + ax0,
                                v2 * 1024 + v0 % 8 * 128 + v1 % 2 * 64 + ax1,
                            ] = C_reindex_m16n8k8_matrixC[
                                v0 // 8 * 128 + v1 // 2 * 64 + ax0,
                                v2 * 1024 + v0 % 8 * 128 + v1 % 2 * 64 + ax1,
                            ]


def matmul_fp16(N: int, M: int, K: int, out_dtype: str):
    x = te.placeholder((N, K), name="X", dtype="float16")
    y = te.placeholder((K, M), name="Y", dtype="float16")
    k = te.reduce_axis((0, K), name="k")
    c = te.compute(  # pylint: disable=invalid-name
        (N, M),
        lambda i, j: te.sum(x[i][k].astype(out_dtype) * y[k][j].astype(out_dtype), axis=[k]),
        name="C",
    )
    return (x, y, c)


def multi_level_tiling_mma(out_dtype):
    simplify_dict = {"float32": "f32", "float16": "f16"}
    out_dtype = simplify_dict[out_dtype]
    return ms.schedule_rule.MultiLevelTilingTensorCore(
        intrin_groups=[
            {
                "init": f"mma_init_m16n8k8_{out_dtype}",
                "load_a": "mma_load_m16n8k8_f16_A_shared_dyn",
                "load_b": "mma_load_m16n8k8_f16_B_shared_dyn",
                "compute": f"mma_sync_m16n8k8_f16f16{out_dtype}",
                "store": f"mma_store_m16n8k8_{out_dtype}_global",
            },
        ],
        structure="SSSRRSRS",
        tile_binds=["blockIdx.x", "blockIdx.y", "threadIdx.y"],
        max_innermost_factor=4,  # 64 // tensor intrin size
        vector_load_lens=[1, 2, 3, 4, 8, 16],
        reuse_read=ms.schedule_rule.ReuseType(
            req="must",
            levels=[4],
            scope="shared.dyn",
        ),
        reuse_write=ms.schedule_rule.ReuseType(
            req="no",
            levels=[2],
            scope="shared.dyn",
        ),
        use_software_pipeline=True,
    )


def _design_space(mod, out_dtype):
    return generate_design_space(
        kind="cuda-tensorcore",
        mod=mod,
        target=Target("nvidia/geforce-rtx-3080"),
        types=None,
        sch_rules=[multi_level_tiling_mma(out_dtype)],
    )


gemm_decision = [
    ("SamplePartitionedTile", [1, 32, 2, 1, 4]),
    ("SamplePartitionedTile", [4, 8, 2, 1, 8]),
    ("SamplePerfectTile", [128, 4, 1]),
]


def test_mma_auto_tensorization():
    mod = te.create_prim_func(matmul_fp16(M=4096, N=4096, K=4096, out_dtype="float16"))
    actual = _design_space(mod, "float16")
    check_sketches(
        mod,
        sketches=actual,
        expected_mods=[MmaModule],
        expected_decisions=[gemm_decision],
    )


expected_cuda_script = r"""#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
#include <cuda_fp16.h>
__device__ half max(half a, half b)
{
  return __hgt(__half(a), __half(b)) ? a : b;
}
__device__ half min(half a, half b)
{
  return __hlt(__half(a), __half(b)) ? a : b;
}
#else

typedef unsigned short uint16_t;
typedef unsigned char uint8_t;
typedef signed char int8_t;
typedef int int32_t;
typedef unsigned long long uint64_t;
typedef unsigned int uint32_t;

#define TVM_FORCE_INLINE inline __attribute__((always_inline))
#define TVM_XINLINE TVM_FORCE_INLINE __device__ __host__
#define TVM_ALIGNED(x) __attribute__ ((aligned(x)))
#define TVM_HALF_OPERATOR(RTYPE, OP)                              \
  TVM_XINLINE RTYPE operator OP (half a, half b) {                \
    return RTYPE(float(a) OP float(b));                           \
  }                                                               \
  template<typename T>                                            \
  TVM_XINLINE RTYPE operator OP (half a, T b) {                   \
    return RTYPE(float(a) OP float(b));                           \
  }                                                               \
  template<typename T>                                            \
  TVM_XINLINE RTYPE operator OP (T a, half b) {                   \
    return RTYPE(float(a) OP float(b));                           \
  }

#define TVM_HALF_ASSIGNOP(AOP, OP)                                \
  template<typename T>                                            \
  TVM_XINLINE half operator AOP (const T& a) {                    \
    return *this = half(float(*this) OP float(a));                \
  }                                                               \
  template<typename T>                                            \
  TVM_XINLINE half operator AOP (const volatile T& a) volatile {  \
    return *this = half(float(*this) OP float(a));                \
  }

class TVM_ALIGNED(2) half {
 public:
  uint16_t half_;

  static TVM_XINLINE half Binary(uint16_t value) {
    half res;
    res.half_ = value;
    return res;
  }

  TVM_XINLINE half() {}

  TVM_XINLINE half(const float& value) { constructor(value); }
  TVM_XINLINE explicit half(const double& value) { constructor(value); }
  TVM_XINLINE explicit half(const int8_t& value) { constructor(value); }
  TVM_XINLINE explicit half(const uint8_t& value) { constructor(value); }
  TVM_XINLINE explicit half(const int32_t& value) { constructor(value); }
  TVM_XINLINE explicit half(const uint32_t& value) { constructor(value); }
  TVM_XINLINE explicit half(const long long& value) { constructor(value); }
  TVM_XINLINE explicit half(const uint64_t& value) { constructor(value); }

  TVM_XINLINE operator float() const {                          \
    return float(half2float(half_));                            \
  }                                                             \
  TVM_XINLINE operator float() const volatile {                 \
    return float(half2float(half_));                            \
  }


  TVM_HALF_ASSIGNOP(+=, +)
  TVM_HALF_ASSIGNOP(-=, -)
  TVM_HALF_ASSIGNOP(*=, *)
  TVM_HALF_ASSIGNOP(/=, /)

  TVM_XINLINE half operator+() {
    return *this;
  }

  TVM_XINLINE half operator-() {
    return half(-float(*this));
  }

  TVM_XINLINE half operator=(const half& a) {
    half_ = a.half_;
    return a;
  }

  template<typename T>
  TVM_XINLINE half operator=(const T& a) {
    return *this = half(a);
  }

  TVM_XINLINE half operator=(const half& a) volatile {
    half_ = a.half_;
    return a;
  }

  template<typename T>
  TVM_XINLINE half operator=(const T& a) volatile {
    return *this = half(a);
  }

 private:
  union Bits {
    float f;
    int32_t si;
    uint32_t ui;
  };

  static int const fp16FractionBits = 10;
  static int const fp32FractionBits = 23;
  static int32_t const fp32FractionMask = ~(~0u << fp32FractionBits);   // == 0x7fffff
  static int32_t const fp32HiddenBit = 1 << fp32FractionBits;   // == 0x800000
  static int const shift = fp32FractionBits - fp16FractionBits;   // == 13
  static int const shiftSign = 16;
  static int32_t const expAdjust = 127 - 15;   // exp32-127 = exp16-15, so exp16 = exp32 - (127-15)

  static int32_t const infN = 0x7F800000;   // flt32 infinity
  static int32_t const maxN = 0x477FFFFF;   // max flt32 that's a flt16 normal after >> by shift
  static int32_t const minN = 0x38800000;   // min flt16 normal as a flt32
  static int32_t const maxZ = 0x33000000;   // max fp32 number that's still rounded to zero in fp16
  static int32_t const signN = 0x80000000;  // flt32 sign bit

  static int32_t const infC = infN >> shift;
  static int32_t const nanN = (infC + 1) << shift;   // minimum flt16 nan as a flt32
  static int32_t const maxC = maxN >> shift;
  static int32_t const minC = minN >> shift;
  static int32_t const signC = signN >> shiftSign;  // flt16 sign bit

  static int32_t const mulN = 0x52000000;  // (1 << 23) / minN
  static int32_t const mulC = 0x33800000;  // minN / (1 << (23 - shift))

  static int32_t const subC = 0x003FF;  // max flt32 subnormal down shifted
  static int32_t const norC = 0x00400;  // min flt32 normal down shifted

  static int32_t const maxD = infC - maxC - 1;
  static int32_t const minD = minC - subC - 1;

  TVM_XINLINE uint16_t float2half(const float& value) const {
    Bits v;
    v.f = value;
    uint32_t sign = v.si & signN;    // grab sign bit
    v.si ^= sign;                    // clear sign bit from v
    sign >>= shiftSign;              // logical shift sign to fp16 position

    if (v.si <= maxZ) {
      // Handle eventual zeros here to ensure
      // vshift will not exceed 32 below.
      v.ui = 0;
    } else if (v.si < minN) {
      // Handle denorms
      uint32_t exp32 = v.ui >> fp32FractionBits;
      int32_t exp16 = exp32 - expAdjust;
      // If exp16 == 0 (just into the denorm range), then significant should be shifted right 1.
      // Smaller (so negative) exp16 values should result in greater right shifts.
      uint32_t vshift = 1 - exp16;
      uint32_t significand = fp32HiddenBit | (v.ui & fp32FractionMask);
      v.ui = significand >> vshift;
      v.ui += (v.ui & 0x3fff) != 0x1000 || (significand & 0x7ff) ? 0x1000 : 0;
    } else if (v.si <= maxN) {
      // Handle norms
      v.ui += (v.ui & 0x3fff) != 0x1000 ? 0x1000 : 0;
      v.ui -= expAdjust << fp32FractionBits;
    } else if (v.si <= infN) {
      v.si = infN;
    } else if (v.si < nanN) {
      v.si = nanN;
    }

    v.ui >>= shift;
    return sign | (v.ui & 0x7fff);
  }

  // Same as above routine, except for addition of volatile keyword
  TVM_XINLINE uint16_t float2half(
    const volatile float& value) const volatile {
    Bits v;
    v.f = value;
    uint32_t sign = v.si & signN;    // grab sign bit
    v.si ^= sign;                    // clear sign bit from v
    sign >>= shiftSign;              // logical shift sign to fp16 position

    if (v.si <= maxZ) {
      // Handle eventual zeros here to ensure
      // vshift will not exceed 32 below.
      v.ui = 0;
    } else if (v.si < minN) {
      // Handle denorms
      uint32_t exp32 = v.ui >> fp32FractionBits;
      int32_t exp16 = exp32 - expAdjust;
      // If exp16 == 0 (just into the denorm range), then significant should be shifted right 1.
      // Smaller (so negative) exp16 values should result in greater right shifts.
      uint32_t vshift = 1 - exp16;
      uint32_t significand = fp32HiddenBit | (v.ui & fp32FractionMask);
      v.ui = significand >> vshift;
      v.ui += (v.ui & 0x3fff) != 0x1000 || (significand & 0x7ff) ? 0x1000 : 0;
    } else if (v.si <= maxN) {
      // Handle norms
      v.ui += (v.ui & 0x3fff) != 0x1000 ? 0x1000 : 0;
      v.ui -= expAdjust << fp32FractionBits;
    } else if (v.si <= infN) {
      v.si = infN;
    } else if (v.si < nanN) {
      v.si = nanN;
    }

    v.ui >>= shift;
    return sign | (v.ui & 0x7fff);
  }

  TVM_XINLINE float half2float(const uint16_t& value) const {
    Bits v;
    v.ui = value;
    int32_t sign = v.si & signC;
    v.si ^= sign;
    sign <<= shiftSign;
    v.si ^= ((v.si + minD) ^ v.si) & -(v.si > subC);
    v.si ^= ((v.si + maxD) ^ v.si) & -(v.si > maxC);
    Bits s;
    s.si = mulC;
    s.f *= v.si;
    int32_t mask = -(norC > v.si);
    v.si <<= shift;
    v.si ^= (s.si ^ v.si) & mask;
    v.si |= sign;
    return v.f;
  }

  TVM_XINLINE float half2float(
    const volatile uint16_t& value) const volatile {
    Bits v;
    v.ui = value;
    int32_t sign = v.si & signC;
    v.si ^= sign;
    sign <<= shiftSign;
    v.si ^= ((v.si + minD) ^ v.si) & -(v.si > subC);
    v.si ^= ((v.si + maxD) ^ v.si) & -(v.si > maxC);
    Bits s;
    s.si = mulC;
    s.f *= v.si;
    int32_t mask = -(norC > v.si);
    v.si <<= shift;
    v.si ^= (s.si ^ v.si) & mask;
    v.si |= sign;
    return v.f;
  }

  template<typename T>
  TVM_XINLINE void constructor(const T& value) {
    half_ = float2half(float(value));
  }
};

TVM_HALF_OPERATOR(half, +)
TVM_HALF_OPERATOR(half, -)
TVM_HALF_OPERATOR(half, *)
TVM_HALF_OPERATOR(half, /)
TVM_HALF_OPERATOR(bool, >)
TVM_HALF_OPERATOR(bool, <)
TVM_HALF_OPERATOR(bool, >=)
TVM_HALF_OPERATOR(bool, <=)

TVM_XINLINE half __float2half_rn(const float a) {
  return half(a);
}
#endif


// Pack two half values.
static inline __device__ __host__ unsigned
__pack_half2(const half x, const half y) {
  unsigned v0 = *((unsigned short *)&x);
  unsigned v1 = *((unsigned short *)&y);
  return (v1 << 16) | v0;
}

#define CUDA_UNSUPPORTED_HALF_MATH_BINARY(HALF_MATH_NAME, FP32_MATH_NAME) \
static inline __device__ __host__ half HALF_MATH_NAME(half x, half y) {   \
  float tmp_x = __half2float(x);                                          \
  float tmp_y = __half2float(y);                                          \
  float result = FP32_MATH_NAME(tmp_x, tmp_y);                            \
  return __float2half(result);                                            \
}

#define CUDA_UNSUPPORTED_HALF_MATH_UNARY(HALF_MATH_NAME, FP32_MATH_NAME) \
static inline __device__ __host__ half HALF_MATH_NAME(half x) {          \
  float tmp_x = __half2float(x);                                         \
  float result = FP32_MATH_NAME(tmp_x);                                  \
  return __float2half(result);                                           \
}

// Some fp16 math functions are not supported in cuda_fp16.h,
// so we define them here to make sure the generated CUDA code
// is valid.
#if defined(__CUDA_ARCH__)
#if (__CUDA_ARCH__ >= 530)
CUDA_UNSUPPORTED_HALF_MATH_BINARY(hpow, powf)
CUDA_UNSUPPORTED_HALF_MATH_UNARY(htanh, tanhf)
CUDA_UNSUPPORTED_HALF_MATH_UNARY(htan, tanf)
CUDA_UNSUPPORTED_HALF_MATH_UNARY(hatan, atanf)
CUDA_UNSUPPORTED_HALF_MATH_UNARY(herf, erf)
#else
CUDA_UNSUPPORTED_HALF_MATH_UNARY(hexp, exp)
#endif
#endif

#undef CUDA_UNSUPPORTED_HALF_MATH_BINARY
#undef CUDA_UNSUPPORTED_HALF_MATH_UNARY
__forceinline__ __device__ unsigned int
cast_smem_ptr_to_int(const void* const smem_ptr)
{
  unsigned int smem_int;
  asm volatile ("{ .reg .u64 smem_int; cvta.to.shared.u64 smem_int, %1; cvt.u32.u64 %0, smem_int; }"
    : "=r"(smem_int) : "l"(smem_ptr));
  return smem_int;
}

#if (((__CUDACC_VER_MAJOR__ == 11) && (__CUDACC_VER_MINOR__ >= 4)) || \
     (__CUDACC_VER_MAJOR__ > 11))
#define TVM_ENABLE_L2_PREFETCH 1
#else
#define TVM_ENABLE_L2_PREFETCH 0
#endif

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
extern "C" __global__ void __launch_bounds__(128) main_kernel(half* __restrict__ C, half* __restrict__ X, half* __restrict__ Y);
extern "C" __global__ void __launch_bounds__(128) main_kernel(half* __restrict__ C, half* __restrict__ X, half* __restrict__ Y) {
  extern __shared__ uchar buf_dyn_shmem[];
  uint1 C_reindex_m16n8k8_matrixC[64];
  half X_reindex_shared_dyn_m16n8k8_matrixA[32];
  half Y_reindex_shared_dyn_m16n8k8_matrixB[32];
  for (int ax0_0_4_init = 0; ax0_0_4_init < 4; ++ax0_0_4_init) {
    for (int ax1_0_4_init = 0; ax1_0_4_init < 8; ++ax1_0_4_init) {
      for (int b = 0; b < 2; ++b) {
        C_reindex_m16n8k8_matrixC[(((ax0_0_4_init * 16) + (ax1_0_4_init * 2)) + b)] = make_uint1(__pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)));
      }
    }
  }
  for (int ax0_ax1_fused_0 = 0; ax0_ax1_fused_0 < 4; ++ax0_ax1_fused_0) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + ((((ax0_ax1_fused_0 * 2048) + (((int)threadIdx.y) * 512)) + ((((int)threadIdx.x) >> 2) * 64)) + (((((int)threadIdx.x) & 3) ^ (((int)threadIdx.x) >> 3)) * 16)));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(X + ((((((((int)blockIdx.y) >> 3) * 524288) + (ax0_ax1_fused_0 * 131072)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)))), "n"(16)
    );
  }
  }
  for (int ax0_ax1_fused_0_1 = 0; ax0_ax1_fused_0_1 < 4; ++ax0_ax1_fused_0_1) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_1 * 2048) + (((int)threadIdx.y) * 512)) + ((((int)threadIdx.x) >> 4) * 256)) + (((((int)threadIdx.x) & 15) ^ ((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 4))) * 16)) + 24576));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(Y + ((((((ax0_ax1_fused_0_1 * 32768) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + (((int)blockIdx.x) * 1024)) + ((((int)blockIdx.y) & 7) * 128)) + ((((int)threadIdx.x) & 15) * 8)))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

  for (int ax0_ax1_fused_0_2 = 0; ax0_ax1_fused_0_2 < 4; ++ax0_ax1_fused_0_2) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_2 * 2048) + (((int)threadIdx.y) * 512)) + ((((int)threadIdx.x) >> 2) * 64)) + (((((int)threadIdx.x) & 3) ^ (((int)threadIdx.x) >> 3)) * 16)) + 8192));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(X + (((((((((int)blockIdx.y) >> 3) * 524288) + (ax0_ax1_fused_0_2 * 131072)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 32))), "n"(16)
    );
  }
  }
  for (int ax0_ax1_fused_0_3 = 0; ax0_ax1_fused_0_3 < 4; ++ax0_ax1_fused_0_3) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((ax0_ax1_fused_0_3 * 2048) + (((int)threadIdx.y) * 512)) + ((((int)threadIdx.x) >> 4) * 256)) + (((((int)threadIdx.x) & 15) ^ ((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 4))) * 16)) + 32768));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(Y + (((((((ax0_ax1_fused_0_3 * 32768) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + (((int)blockIdx.x) * 1024)) + ((((int)blockIdx.y) & 7) * 128)) + ((((int)threadIdx.x) & 15) * 8)) + 131072))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 1;");

  __syncthreads();
  for (int ax0_0 = 0; ax0_0 < 2; ++ax0_0) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[(((((((int)threadIdx.y) >> 1) * 2048) + (ax0_0 * 1024)) + (((int)threadIdx.x) * 32)) + ((0 ^ ((((int)threadIdx.x) & 7) >> 1)) * 8))])) + 0);
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(X_reindex_shared_dyn_m16n8k8_matrixA + (ax0_0 * 8)))[0]), "=r"(((unsigned *)(X_reindex_shared_dyn_m16n8k8_matrixA + (ax0_0 * 8)))[1]), "=r"(((unsigned *)(X_reindex_shared_dyn_m16n8k8_matrixA + (ax0_0 * 8)))[2]), "=r"(((unsigned *)(X_reindex_shared_dyn_m16n8k8_matrixA + (ax0_0 * 8)))[3])
      : "r"(addr)
    );
  }
  }
  for (int ax1_0 = 0; ax1_0 < 2; ++ax1_0) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.x) & 7) * 128) + ((((((((int)threadIdx.y) & 1) * 8) + (ax1_0 * 4)) + (((int)threadIdx.x) >> 3)) ^ (((int)threadIdx.x) & 7)) * 8)) + 12288)])) + 0);
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(Y_reindex_shared_dyn_m16n8k8_matrixB + (ax1_0 * 8)))[0]), "=r"(((unsigned *)(Y_reindex_shared_dyn_m16n8k8_matrixB + (ax1_0 * 8)))[1]), "=r"(((unsigned *)(Y_reindex_shared_dyn_m16n8k8_matrixB + (ax1_0 * 8)))[2]), "=r"(((unsigned *)(Y_reindex_shared_dyn_m16n8k8_matrixB + (ax1_0 * 8)))[3])
      : "r"(addr)
    );
  }
  }
  for (int ax2_0_0 = 0; ax2_0_0 < 126; ++ax2_0_0) {
    __syncthreads();
    for (int ax0_ax1_fused_0_4 = 0; ax0_ax1_fused_0_4 < 4; ++ax0_ax1_fused_0_4) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + (((((((ax2_0_0 + 2) % 3) * 8192) + (ax0_ax1_fused_0_4 * 2048)) + (((int)threadIdx.y) * 512)) + ((((int)threadIdx.x) >> 2) * 64)) + (((((int)threadIdx.x) & 3) ^ (((int)threadIdx.x) >> 3)) * 16)));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(X + ((((((((((int)blockIdx.y) >> 3) * 524288) + (ax0_ax1_fused_0_4 * 131072)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + (ax2_0_0 * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 64))), "n"(16)
    );
  }
    }
    for (int ax0_ax1_fused_0_5 = 0; ax0_ax1_fused_0_5 < 4; ++ax0_ax1_fused_0_5) {

  {
    unsigned int addr = cast_smem_ptr_to_int(buf_dyn_shmem + ((((((((ax2_0_0 + 2) % 3) * 8192) + (ax0_ax1_fused_0_5 * 2048)) + (((int)threadIdx.y) * 512)) + ((((int)threadIdx.x) >> 4) * 256)) + (((((int)threadIdx.x) & 15) ^ ((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 4))) * 16)) + 24576));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(Y + ((((((((ax2_0_0 * 131072) + (ax0_ax1_fused_0_5 * 32768)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + (((int)blockIdx.x) * 1024)) + ((((int)blockIdx.y) & 7) * 128)) + ((((int)threadIdx.x) & 15) * 8)) + 262144))), "n"(16)
    );
  }
    }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 1;");

    __syncthreads();
    for (int ax2_0_1 = 0; ax2_0_1 < 3; ++ax2_0_1) {
      for (int ax0_0_1 = 0; ax0_0_1 < 2; ++ax0_0_1) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((ax2_0_0 % 3) * 4096) + ((((int)threadIdx.y) >> 1) * 2048)) + (ax0_0_1 * 1024)) + (((int)threadIdx.x) * 32)) + (((ax2_0_1 + 1) ^ ((((int)threadIdx.x) & 7) >> 1)) * 8))])) + 0);
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(X_reindex_shared_dyn_m16n8k8_matrixA + ((((ax2_0_1 + 1) & 1) * 16) + (ax0_0_1 * 8))))[0]), "=r"(((unsigned *)(X_reindex_shared_dyn_m16n8k8_matrixA + ((((ax2_0_1 + 1) & 1) * 16) + (ax0_0_1 * 8))))[1]), "=r"(((unsigned *)(X_reindex_shared_dyn_m16n8k8_matrixA + ((((ax2_0_1 + 1) & 1) * 16) + (ax0_0_1 * 8))))[2]), "=r"(((unsigned *)(X_reindex_shared_dyn_m16n8k8_matrixA + ((((ax2_0_1 + 1) & 1) * 16) + (ax0_0_1 * 8))))[3])
      : "r"(addr)
    );
  }
      }
      for (int ax1_0_1 = 0; ax1_0_1 < 2; ++ax1_0_1) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((ax2_0_0 % 3) * 4096) + (ax2_0_1 * 1024)) + ((((int)threadIdx.x) & 7) * 128)) + ((((((((int)threadIdx.y) & 1) * 8) + (ax1_0_1 * 4)) + (((int)threadIdx.x) >> 3)) ^ (((int)threadIdx.x) & 7)) * 8)) + 13312)])) + 0);
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(Y_reindex_shared_dyn_m16n8k8_matrixB + ((((ax2_0_1 + 1) & 1) * 16) + (ax1_0_1 * 8))))[0]), "=r"(((unsigned *)(Y_reindex_shared_dyn_m16n8k8_matrixB + ((((ax2_0_1 + 1) & 1) * 16) + (ax1_0_1 * 8))))[1]), "=r"(((unsigned *)(Y_reindex_shared_dyn_m16n8k8_matrixB + ((((ax2_0_1 + 1) & 1) * 16) + (ax1_0_1 * 8))))[2]), "=r"(((unsigned *)(Y_reindex_shared_dyn_m16n8k8_matrixB + ((((ax2_0_1 + 1) & 1) * 16) + (ax1_0_1 * 8))))[3])
      : "r"(addr)
    );
  }
      }
      for (int ax0_0_4 = 0; ax0_0_4 < 4; ++ax0_0_4) {
        for (int ax1_0_4 = 0; ax1_0_4 < 8; ++ax1_0_4) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16"
      "{%0, %1}, {%2, %3}, {%4}, {%5, %6};\n"
      :  "=r"(((unsigned *)(C_reindex_m16n8k8_matrixC + ((ax0_0_4 * 16) + (ax1_0_4 * 2))))[0]), "=r"(((unsigned *)(C_reindex_m16n8k8_matrixC + ((ax0_0_4 * 16) + (ax1_0_4 * 2))))[1])
      : "r"(((unsigned *)(X_reindex_shared_dyn_m16n8k8_matrixA + (((ax2_0_1 & 1) * 16) + (ax0_0_4 * 4))))[0]), "r"(((unsigned *)(X_reindex_shared_dyn_m16n8k8_matrixA + (((ax2_0_1 & 1) * 16) + (ax0_0_4 * 4))))[1]), "r"(((unsigned *)(Y_reindex_shared_dyn_m16n8k8_matrixB + (((ax2_0_1 & 1) * 16) + (ax1_0_4 * 2))))[0]), "r"(((unsigned *)(C_reindex_m16n8k8_matrixC + ((ax0_0_4 * 16) + (ax1_0_4 * 2))))[0]), "r"(((unsigned *)(C_reindex_m16n8k8_matrixC + ((ax0_0_4 * 16) + (ax1_0_4 * 2))))[1]));
  }
        }
      }
    }
    for (int ax0_0_2 = 0; ax0_0_2 < 2; ++ax0_0_2) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[(((((((ax2_0_0 + 1) % 3) * 4096) + ((((int)threadIdx.y) >> 1) * 2048)) + (ax0_0_2 * 1024)) + (((int)threadIdx.x) * 32)) + ((0 ^ ((((int)threadIdx.x) & 7) >> 1)) * 8))])) + 0);
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(X_reindex_shared_dyn_m16n8k8_matrixA + (ax0_0_2 * 8)))[0]), "=r"(((unsigned *)(X_reindex_shared_dyn_m16n8k8_matrixA + (ax0_0_2 * 8)))[1]), "=r"(((unsigned *)(X_reindex_shared_dyn_m16n8k8_matrixA + (ax0_0_2 * 8)))[2]), "=r"(((unsigned *)(X_reindex_shared_dyn_m16n8k8_matrixA + (ax0_0_2 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax1_0_2 = 0; ax1_0_2 < 2; ++ax1_0_2) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((ax2_0_0 + 1) % 3) * 4096) + ((((int)threadIdx.x) & 7) * 128)) + ((((((((int)threadIdx.y) & 1) * 8) + (ax1_0_2 * 4)) + (((int)threadIdx.x) >> 3)) ^ (((int)threadIdx.x) & 7)) * 8)) + 12288)])) + 0);
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(Y_reindex_shared_dyn_m16n8k8_matrixB + (ax1_0_2 * 8)))[0]), "=r"(((unsigned *)(Y_reindex_shared_dyn_m16n8k8_matrixB + (ax1_0_2 * 8)))[1]), "=r"(((unsigned *)(Y_reindex_shared_dyn_m16n8k8_matrixB + (ax1_0_2 * 8)))[2]), "=r"(((unsigned *)(Y_reindex_shared_dyn_m16n8k8_matrixB + (ax1_0_2 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax0_0_4_1 = 0; ax0_0_4_1 < 4; ++ax0_0_4_1) {
      for (int ax1_0_4_1 = 0; ax1_0_4_1 < 8; ++ax1_0_4_1) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16"
      "{%0, %1}, {%2, %3}, {%4}, {%5, %6};\n"
      :  "=r"(((unsigned *)(C_reindex_m16n8k8_matrixC + ((ax0_0_4_1 * 16) + (ax1_0_4_1 * 2))))[0]), "=r"(((unsigned *)(C_reindex_m16n8k8_matrixC + ((ax0_0_4_1 * 16) + (ax1_0_4_1 * 2))))[1])
      : "r"(((unsigned *)(X_reindex_shared_dyn_m16n8k8_matrixA + ((ax0_0_4_1 * 4) + 16)))[0]), "r"(((unsigned *)(X_reindex_shared_dyn_m16n8k8_matrixA + ((ax0_0_4_1 * 4) + 16)))[1]), "r"(((unsigned *)(Y_reindex_shared_dyn_m16n8k8_matrixB + ((ax1_0_4_1 * 2) + 16)))[0]), "r"(((unsigned *)(C_reindex_m16n8k8_matrixC + ((ax0_0_4_1 * 16) + (ax1_0_4_1 * 2))))[0]), "r"(((unsigned *)(C_reindex_m16n8k8_matrixC + ((ax0_0_4_1 * 16) + (ax1_0_4_1 * 2))))[1]));
  }
      }
    }
  }
__asm__ __volatile__("cp.async.wait_group 0;");

  __syncthreads();
  for (int ax2_0_1_1 = 0; ax2_0_1_1 < 3; ++ax2_0_1_1) {
    for (int ax0_0_3 = 0; ax0_0_3 < 2; ++ax0_0_3) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[(((((((int)threadIdx.y) >> 1) * 2048) + (ax0_0_3 * 1024)) + (((int)threadIdx.x) * 32)) + (((ax2_0_1_1 + 1) ^ ((((int)threadIdx.x) & 7) >> 1)) * 8))])) + 0);
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(X_reindex_shared_dyn_m16n8k8_matrixA + ((((ax2_0_1_1 + 1) & 1) * 16) + (ax0_0_3 * 8))))[0]), "=r"(((unsigned *)(X_reindex_shared_dyn_m16n8k8_matrixA + ((((ax2_0_1_1 + 1) & 1) * 16) + (ax0_0_3 * 8))))[1]), "=r"(((unsigned *)(X_reindex_shared_dyn_m16n8k8_matrixA + ((((ax2_0_1_1 + 1) & 1) * 16) + (ax0_0_3 * 8))))[2]), "=r"(((unsigned *)(X_reindex_shared_dyn_m16n8k8_matrixA + ((((ax2_0_1_1 + 1) & 1) * 16) + (ax0_0_3 * 8))))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax1_0_3 = 0; ax1_0_3 < 2; ++ax1_0_3) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((ax2_0_1_1 * 1024) + ((((int)threadIdx.x) & 7) * 128)) + ((((((((int)threadIdx.y) & 1) * 8) + (ax1_0_3 * 4)) + (((int)threadIdx.x) >> 3)) ^ (((int)threadIdx.x) & 7)) * 8)) + 13312)])) + 0);
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(Y_reindex_shared_dyn_m16n8k8_matrixB + ((((ax2_0_1_1 + 1) & 1) * 16) + (ax1_0_3 * 8))))[0]), "=r"(((unsigned *)(Y_reindex_shared_dyn_m16n8k8_matrixB + ((((ax2_0_1_1 + 1) & 1) * 16) + (ax1_0_3 * 8))))[1]), "=r"(((unsigned *)(Y_reindex_shared_dyn_m16n8k8_matrixB + ((((ax2_0_1_1 + 1) & 1) * 16) + (ax1_0_3 * 8))))[2]), "=r"(((unsigned *)(Y_reindex_shared_dyn_m16n8k8_matrixB + ((((ax2_0_1_1 + 1) & 1) * 16) + (ax1_0_3 * 8))))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax0_0_4_2 = 0; ax0_0_4_2 < 4; ++ax0_0_4_2) {
      for (int ax1_0_4_2 = 0; ax1_0_4_2 < 8; ++ax1_0_4_2) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16"
      "{%0, %1}, {%2, %3}, {%4}, {%5, %6};\n"
      :  "=r"(((unsigned *)(C_reindex_m16n8k8_matrixC + ((ax0_0_4_2 * 16) + (ax1_0_4_2 * 2))))[0]), "=r"(((unsigned *)(C_reindex_m16n8k8_matrixC + ((ax0_0_4_2 * 16) + (ax1_0_4_2 * 2))))[1])
      : "r"(((unsigned *)(X_reindex_shared_dyn_m16n8k8_matrixA + (((ax2_0_1_1 & 1) * 16) + (ax0_0_4_2 * 4))))[0]), "r"(((unsigned *)(X_reindex_shared_dyn_m16n8k8_matrixA + (((ax2_0_1_1 & 1) * 16) + (ax0_0_4_2 * 4))))[1]), "r"(((unsigned *)(Y_reindex_shared_dyn_m16n8k8_matrixB + (((ax2_0_1_1 & 1) * 16) + (ax1_0_4_2 * 2))))[0]), "r"(((unsigned *)(C_reindex_m16n8k8_matrixC + ((ax0_0_4_2 * 16) + (ax1_0_4_2 * 2))))[0]), "r"(((unsigned *)(C_reindex_m16n8k8_matrixC + ((ax0_0_4_2 * 16) + (ax1_0_4_2 * 2))))[1]));
  }
      }
    }
  }
  for (int ax0_0_5 = 0; ax0_0_5 < 2; ++ax0_0_5) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((((int)threadIdx.y) >> 1) * 2048) + (ax0_0_5 * 1024)) + (((int)threadIdx.x) * 32)) + ((0 ^ ((((int)threadIdx.x) & 7) >> 1)) * 8)) + 4096)])) + 0);
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(X_reindex_shared_dyn_m16n8k8_matrixA + (ax0_0_5 * 8)))[0]), "=r"(((unsigned *)(X_reindex_shared_dyn_m16n8k8_matrixA + (ax0_0_5 * 8)))[1]), "=r"(((unsigned *)(X_reindex_shared_dyn_m16n8k8_matrixA + (ax0_0_5 * 8)))[2]), "=r"(((unsigned *)(X_reindex_shared_dyn_m16n8k8_matrixA + (ax0_0_5 * 8)))[3])
      : "r"(addr)
    );
  }
  }
  for (int ax1_0_5 = 0; ax1_0_5 < 2; ++ax1_0_5) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.x) & 7) * 128) + ((((((((int)threadIdx.y) & 1) * 8) + (ax1_0_5 * 4)) + (((int)threadIdx.x) >> 3)) ^ (((int)threadIdx.x) & 7)) * 8)) + 16384)])) + 0);
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(Y_reindex_shared_dyn_m16n8k8_matrixB + (ax1_0_5 * 8)))[0]), "=r"(((unsigned *)(Y_reindex_shared_dyn_m16n8k8_matrixB + (ax1_0_5 * 8)))[1]), "=r"(((unsigned *)(Y_reindex_shared_dyn_m16n8k8_matrixB + (ax1_0_5 * 8)))[2]), "=r"(((unsigned *)(Y_reindex_shared_dyn_m16n8k8_matrixB + (ax1_0_5 * 8)))[3])
      : "r"(addr)
    );
  }
  }
  for (int ax0_0_4_3 = 0; ax0_0_4_3 < 4; ++ax0_0_4_3) {
    for (int ax1_0_4_3 = 0; ax1_0_4_3 < 8; ++ax1_0_4_3) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16"
      "{%0, %1}, {%2, %3}, {%4}, {%5, %6};\n"
      :  "=r"(((unsigned *)(C_reindex_m16n8k8_matrixC + ((ax0_0_4_3 * 16) + (ax1_0_4_3 * 2))))[0]), "=r"(((unsigned *)(C_reindex_m16n8k8_matrixC + ((ax0_0_4_3 * 16) + (ax1_0_4_3 * 2))))[1])
      : "r"(((unsigned *)(X_reindex_shared_dyn_m16n8k8_matrixA + ((ax0_0_4_3 * 4) + 16)))[0]), "r"(((unsigned *)(X_reindex_shared_dyn_m16n8k8_matrixA + ((ax0_0_4_3 * 4) + 16)))[1]), "r"(((unsigned *)(Y_reindex_shared_dyn_m16n8k8_matrixB + ((ax1_0_4_3 * 2) + 16)))[0]), "r"(((unsigned *)(C_reindex_m16n8k8_matrixC + ((ax0_0_4_3 * 16) + (ax1_0_4_3 * 2))))[0]), "r"(((unsigned *)(C_reindex_m16n8k8_matrixC + ((ax0_0_4_3 * 16) + (ax1_0_4_3 * 2))))[1]));
  }
    }
  }
  for (int ax2_0_1_2 = 0; ax2_0_1_2 < 3; ++ax2_0_1_2) {
    for (int ax0_0_6 = 0; ax0_0_6 < 2; ++ax0_0_6) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((((((int)threadIdx.y) >> 1) * 2048) + (ax0_0_6 * 1024)) + (((int)threadIdx.x) * 32)) + (((ax2_0_1_2 + 1) ^ ((((int)threadIdx.x) & 7) >> 1)) * 8)) + 4096)])) + 0);
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(X_reindex_shared_dyn_m16n8k8_matrixA + ((((ax2_0_1_2 + 1) & 1) * 16) + (ax0_0_6 * 8))))[0]), "=r"(((unsigned *)(X_reindex_shared_dyn_m16n8k8_matrixA + ((((ax2_0_1_2 + 1) & 1) * 16) + (ax0_0_6 * 8))))[1]), "=r"(((unsigned *)(X_reindex_shared_dyn_m16n8k8_matrixA + ((((ax2_0_1_2 + 1) & 1) * 16) + (ax0_0_6 * 8))))[2]), "=r"(((unsigned *)(X_reindex_shared_dyn_m16n8k8_matrixA + ((((ax2_0_1_2 + 1) & 1) * 16) + (ax0_0_6 * 8))))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax1_0_6 = 0; ax1_0_6 < 2; ++ax1_0_6) {

  {
    unsigned int addr = cast_smem_ptr_to_int((&(((half*)buf_dyn_shmem)[((((ax2_0_1_2 * 1024) + ((((int)threadIdx.x) & 7) * 128)) + ((((((((int)threadIdx.y) & 1) * 8) + (ax1_0_6 * 4)) + (((int)threadIdx.x) >> 3)) ^ (((int)threadIdx.x) & 7)) * 8)) + 17408)])) + 0);
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(Y_reindex_shared_dyn_m16n8k8_matrixB + ((((ax2_0_1_2 + 1) & 1) * 16) + (ax1_0_6 * 8))))[0]), "=r"(((unsigned *)(Y_reindex_shared_dyn_m16n8k8_matrixB + ((((ax2_0_1_2 + 1) & 1) * 16) + (ax1_0_6 * 8))))[1]), "=r"(((unsigned *)(Y_reindex_shared_dyn_m16n8k8_matrixB + ((((ax2_0_1_2 + 1) & 1) * 16) + (ax1_0_6 * 8))))[2]), "=r"(((unsigned *)(Y_reindex_shared_dyn_m16n8k8_matrixB + ((((ax2_0_1_2 + 1) & 1) * 16) + (ax1_0_6 * 8))))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax0_0_4_4 = 0; ax0_0_4_4 < 4; ++ax0_0_4_4) {
      for (int ax1_0_4_4 = 0; ax1_0_4_4 < 8; ++ax1_0_4_4) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16"
      "{%0, %1}, {%2, %3}, {%4}, {%5, %6};\n"
      :  "=r"(((unsigned *)(C_reindex_m16n8k8_matrixC + ((ax0_0_4_4 * 16) + (ax1_0_4_4 * 2))))[0]), "=r"(((unsigned *)(C_reindex_m16n8k8_matrixC + ((ax0_0_4_4 * 16) + (ax1_0_4_4 * 2))))[1])
      : "r"(((unsigned *)(X_reindex_shared_dyn_m16n8k8_matrixA + (((ax2_0_1_2 & 1) * 16) + (ax0_0_4_4 * 4))))[0]), "r"(((unsigned *)(X_reindex_shared_dyn_m16n8k8_matrixA + (((ax2_0_1_2 & 1) * 16) + (ax0_0_4_4 * 4))))[1]), "r"(((unsigned *)(Y_reindex_shared_dyn_m16n8k8_matrixB + (((ax2_0_1_2 & 1) * 16) + (ax1_0_4_4 * 2))))[0]), "r"(((unsigned *)(C_reindex_m16n8k8_matrixC + ((ax0_0_4_4 * 16) + (ax1_0_4_4 * 2))))[0]), "r"(((unsigned *)(C_reindex_m16n8k8_matrixC + ((ax0_0_4_4 * 16) + (ax1_0_4_4 * 2))))[1]));
  }
      }
    }
  }
  for (int ax0_0_4_5 = 0; ax0_0_4_5 < 4; ++ax0_0_4_5) {
    for (int ax1_0_4_5 = 0; ax1_0_4_5 < 8; ++ax1_0_4_5) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16"
      "{%0, %1}, {%2, %3}, {%4}, {%5, %6};\n"
      :  "=r"(((unsigned *)(C_reindex_m16n8k8_matrixC + ((ax0_0_4_5 * 16) + (ax1_0_4_5 * 2))))[0]), "=r"(((unsigned *)(C_reindex_m16n8k8_matrixC + ((ax0_0_4_5 * 16) + (ax1_0_4_5 * 2))))[1])
      : "r"(((unsigned *)(X_reindex_shared_dyn_m16n8k8_matrixA + ((ax0_0_4_5 * 4) + 16)))[0]), "r"(((unsigned *)(X_reindex_shared_dyn_m16n8k8_matrixA + ((ax0_0_4_5 * 4) + 16)))[1]), "r"(((unsigned *)(Y_reindex_shared_dyn_m16n8k8_matrixB + ((ax1_0_4_5 * 2) + 16)))[0]), "r"(((unsigned *)(C_reindex_m16n8k8_matrixC + ((ax0_0_4_5 * 16) + (ax1_0_4_5 * 2))))[0]), "r"(((unsigned *)(C_reindex_m16n8k8_matrixC + ((ax0_0_4_5 * 16) + (ax1_0_4_5 * 2))))[1]));
  }
    }
  }
  for (int ax0_0_7 = 0; ax0_0_7 < 8; ++ax0_0_7) {
    __syncthreads();
    for (int ax1_0_7 = 0; ax1_0_7 < 8; ++ax1_0_7) {
      *(uint1*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.x) * 2050) + (((int)threadIdx.y) * 512)) + (ax1_0_7 * 64)) + 12288)) = C_reindex_m16n8k8_matrixC[((((ax0_0_7 >> 1) * 16) + (ax1_0_7 * 2)) + (ax0_0_7 & 1))];
    }
    __syncthreads();
    for (int threadIdx_x_cache_threadIdx_y_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_0 = 0; threadIdx_x_cache_threadIdx_y_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_0 < 512; ++threadIdx_x_cache_threadIdx_y_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_0) {
      C[(((((((((((((int)blockIdx.y) >> 3) * 524288) + (((threadIdx_x_cache_threadIdx_y_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_0 & 15) >> 3) * 262144)) + (ax0_0_7 * 32768)) + ((((int)threadIdx.y) & 1) * 16384)) + ((((int)threadIdx.x) >> 3) * 4096)) + (((int)blockIdx.x) * 1024)) + ((((int)blockIdx.y) & 7) * 128)) + ((threadIdx_x_cache_threadIdx_y_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_0 & 7) * 16)) + ((((int)threadIdx.y) >> 1) * 8)) + (((int)threadIdx.x) & 7))] = ((half*)buf_dyn_shmem)[((((threadIdx_x_cache_threadIdx_y_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_0 * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 12288)];
    }
  }
}

"""


@tvm.testing.requires_tensorcore
def test_mma_script_after_build():
    arch = tvm.contrib.nvcc.get_target_compute_version()
    major, _ = tvm.contrib.nvcc.parse_compute_version(arch)
    if major < 8:
        # At least sm80 is required
        return

    mod = te.create_prim_func(matmul_fp16(M=4096, N=4096, K=4096, out_dtype="float16"))
    actual = _design_space(mod, "float16")
    assert len(actual) == 1
    sketch = actual[0]

    i = 0
    new_decisions = {}
    for inst in sketch.trace.insts:
        if not inst.kind.name.startswith("Sample"):
            continue
        assert i < len(gemm_decision)
        if inst.kind.name == gemm_decision[i][0]:
            new_decisions[inst] = gemm_decision[i][1]
            i += 1
    assert len(new_decisions) == len(gemm_decision)
    sch = Schedule(mod)
    Trace(
        insts=sketch.trace.insts,
        decisions=new_decisions,
    ).apply_to_schedule(sch, remove_postproc=True)

    sch.enter_postproc()
    # DefaultCUDATensorCore
    ms.postproc.DisallowDynamicLoop().apply(sch)
    ms.postproc.RewriteCooperativeFetch().apply(sch)
    # Disable RewriteUnboundBlock here since max_threads_per_block_ is not set
    # ms.postproc.RewriteUnboundBlock(256).apply(sch)
    ms.postproc.RewriteParallelVectorizeUnroll().apply(sch)
    ms.postproc.RewriteReductionBlock().apply(sch)
    ms.postproc.VerifyGPUCode().apply(sch)
    ms.postproc.RewriteTensorize(False).apply(sch)

    with tvm.transform.PassContext(config={"tir.use_async_copy": 1}):
        rt_mod = tvm.build(sch.mod, target="cuda")
    print(rt_mod.imported_modules[0].get_source())
    assert rt_mod.imported_modules[0].get_source() == expected_cuda_script


def initializer():
    @register_func("meta_schedule.builder.async_build")
    def async_build(mod, target, _params):  # pylint: disable=unused-variable, unused-argument
        # pylint: disable=import-outside-toplevel
        from tvm.driver import build as tvm_build
        from tvm.tir.transform import RemoveWeightLayoutRewriteBlock

        # re-import here for local builder to register index_map_m16n8k8_matrixC
        # pylint: disable=import-outside-toplevel, unused-import
        from tvm.tir.tensor_intrin import cuda

        mod = RemoveWeightLayoutRewriteBlock(skip_ndarray_rewrite=True)(mod)
        with tvm.transform.PassContext(config={"tir.use_async_copy": 1}):
            rt_mod = tvm_build(mod, target=target)
        return rt_mod


@tvm.testing.requires_tensorcore
@tvm.testing.requires_cublas
def test_mma_tune():
    arch = tvm.contrib.nvcc.get_target_compute_version()
    major, _ = tvm.contrib.nvcc.parse_compute_version(arch)
    if major < 8:
        # At least sm80 is required
        return

    # pylint: disable=import-outside-toplevel
    from tvm.contrib import cublas

    def tune(out_dtype):
        M, N, K = 1024, 1024, 1024
        target = Target("nvidia/geforce-rtx-3080")
        func = te.create_prim_func(matmul_fp16(N=N, M=M, K=K, out_dtype=out_dtype)).with_attr(
            {"global_symbol": "main"}
        )
        mod = tvm.IRModule({"main": func})

        with tempfile.TemporaryDirectory() as work_dir:
            db = ms.tir_integration.tune_tir(
                mod=mod,
                target=target,
                work_dir=work_dir,
                max_trials_global=8,
                builder=LocalBuilder(
                    f_build="meta_schedule.builder.async_build", initializer=initializer
                ),
                space=ms.space_generator.PostOrderApply(
                    sch_rules=[multi_level_tiling_mma(out_dtype=out_dtype)],
                ),
            )
            sch = db.query_schedule(mod, target=target, workload_name="main")
            with tvm.transform.PassContext(config={"tir.use_async_copy": 1}):
                rt_mod = tvm.build(sch.mod, target=target)

        a_np = np.random.uniform(0, 1, size=(M, K)).astype("float16")
        b_np = np.random.uniform(0, 1, size=(K, N)).astype("float16")
        A_cublas = te.placeholder((M, K), name="A", dtype="float16")
        B_cublas = te.placeholder((K, N), name="B", dtype="float16")
        C_cublas = cublas.matmul(A_cublas, B_cublas, dtype=out_dtype)
        s = te.create_schedule(C_cublas.op)
        dev = tvm.cuda(0)
        f_cublas = tvm.build(s, [A_cublas, B_cublas, C_cublas], target)
        a_cublas = tvm.nd.array(a_np.astype("float16"), dev)
        b_cublas = tvm.nd.array(b_np.astype("float16"), dev)
        c_cublas = tvm.nd.array(np.zeros((M, N), dtype=C_cublas.dtype), dev)
        f_cublas(a_cublas, b_cublas, c_cublas)
        a_tvm = tvm.nd.array(a_np, device=tvm.cuda(0))
        b_tvm = tvm.nd.array(b_np, device=tvm.cuda(0))
        c_tvm = tvm.nd.array(np.empty((M, N)).astype(out_dtype), device=tvm.cuda(0))
        rt_mod(a_tvm, b_tvm, c_tvm)
        assert np.allclose(c_tvm.numpy(), c_cublas.numpy(), rtol=1e-2)

    tune("float16")
    tune("float32")


if __name__ == "__main__":
    test_mma_auto_tensorization()
    test_mma_script_after_build()
    test_mma_tune()
