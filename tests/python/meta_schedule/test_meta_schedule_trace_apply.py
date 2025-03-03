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
import pytest
import tvm
import tvm.meta_schedule as ms
import tvm.testing
from tvm.script import tir as T
from tvm.target import Target
from tvm.target.codegen import llvm_lookup_intrinsic_id
from tvm.tir import Schedule, floordiv, floormod
from tvm.tir.tensor_intrin.cuda import *
from tvm.tir.tensor_intrin.x86 import VNNI_DOT_16x4_INTRIN as VNNI_INTRIN


# fmt: off
@tvm.script.ir_module
class Dense:
    @T.prim_func
    def main(
        p0: T.Buffer((128, 128), "float32"),
        p1: T.Buffer((128, 128), "float32"),
        T_matmul_NT: T.Buffer((128, 128), "float32"),
    ) -> None:
        # function attr dict
        T.func_attr({"layout_free_buffers": [1], "tir.noalias": True, "global_symbol": "main"})
        # body
        # with T.block("root")
        for i0, i1, i2 in T.grid(128, 128, 128):
            with T.block("T_matmul_NT"):
                i, j, k = T.axis.remap("SSR", [i0, i1, i2])
                T.reads(p0[i, k], p1[j, k])
                T.writes(T_matmul_NT[i, j])
                T.block_attr({"layout_free_placeholders": []})
                with T.init():
                    T_matmul_NT[i, j] = T.float32(0)
                T_matmul_NT[i, j] = T_matmul_NT[i, j] + p0[i, k] * p1[j, k]


@tvm.script.ir_module
class DenseAdd:
    @T.prim_func
    def main(
        p0: T.Buffer((128, 128), "float32"),
        p1: T.Buffer((128, 128), "float32"),
        T_add: T.Buffer((128, 128), "float32"),
    ) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True, "layout_free_buffers": [1]})
        # body
        # with T.block("root")
        T_matmul_NT = T.alloc_buffer([128, 128], dtype="float32")
        compile_engine_const = T.alloc_buffer([], dtype="float32")
        for i0, i1, i2 in T.grid(128, 128, 128):
            with T.block("T_matmul_NT"):
                i, j, k = T.axis.remap("SSR", [i0, i1, i2])
                T.reads(p0[i, k], p1[j, k])
                T.writes(T_matmul_NT[i, j])
                T.block_attr({"layout_free_placeholders": []})
                with T.init():
                    T_matmul_NT[i, j] = T.float32(0)
                T_matmul_NT[i, j] = T_matmul_NT[i, j] + p0[i, k] * p1[j, k]
        with T.block("compile_engine_const"):
            vi = T.axis.spatial(1, 0)
            T.reads()
            T.writes(compile_engine_const[()])
            compile_engine_const[()] = T.float32(1)
        for i0, i1 in T.grid(128, 128):
            with T.block("T_add"):
                ax0, ax1 = T.axis.remap("SS", [i0, i1])
                T.reads(T_matmul_NT[ax0, ax1], compile_engine_const[()])
                T.writes(T_add[ax0, ax1])
                T_add[ax0, ax1] = T_matmul_NT[ax0, ax1] + compile_engine_const[()]


@tvm.script.ir_module
class DenseAdd_scheduled_cpu:
    @T.prim_func
    def main(
        p0: T.Buffer((128, 128), "float32"),
        p1: T.Buffer((128, 128), "float32"),
        T_add: T.Buffer((128, 128), "float32"),
    ) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True, "layout_free_buffers": [1]})
        # body
        # with T.block("root")
        T_matmul_NT_global = T.alloc_buffer([128, 128], dtype="float32")
        p1_global = T.alloc_buffer([2, 128, 64], dtype="float32")
        for ax0, ax1 in T.grid(128, 128):
            with T.block("p1_global"):
                v0, v1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(p1[v0, v1])
                T.writes(p1_global[v0 // 64, v1, v0 % 64])
                T.block_attr({"meta_schedule.layout_rewrite_preproc": 1})
                p1_global[v0 // 64, v1, v0 % 64] = p1[v0, v1]
        for i0_0_i1_0_fused_fused in T.parallel(4):
            for i0_1, i1_1 in T.grid(8, 1):
                for i0_2_init, i1_2_init, i0_3_init in T.grid(4, 1, 2):
                    for i1_3_fused_init in T.vectorized(64):
                        with T.block("T_matmul_NT_init"):
                            i = T.axis.spatial(
                                128,
                                i0_0_i1_0_fused_fused // 2 * 64
                                + i0_1 * 8
                                + i0_2_init * 2
                                + i0_3_init,
                            )
                            j = T.axis.spatial(
                                128,
                                i0_0_i1_0_fused_fused % 2 * 64
                                + i1_1 * 64
                                + i1_2_init * 64
                                + i1_3_fused_init,
                            )
                            T.reads()
                            T.writes(T_matmul_NT_global[i, j])
                            T.block_attr(
                                {
                                    "layout_free_placeholders": [],
                                    "meta_schedule.tiling_structure": "SSRSRS",
                                }
                            )
                            T_matmul_NT_global[i, j] = T.float32(0)
                for i2_0, i0_2, i1_2, i2_1, i0_3 in T.grid(128, 4, 1, 1, 2):
                    for i1_3_fused in T.vectorized(64):
                        with T.block("T_matmul_NT_update"):
                            i = T.axis.spatial(
                                128, i0_0_i1_0_fused_fused // 2 * 64 + i0_1 * 8 + i0_2 * 2 + i0_3
                            )
                            j = T.axis.spatial(
                                128,
                                i0_0_i1_0_fused_fused % 2 * 64 + i1_1 * 64 + i1_2 * 64 + i1_3_fused,
                            )
                            k = T.axis.reduce(128, i2_0 + i2_1)
                            T.reads(
                                T_matmul_NT_global[i, j], p0[i, k], p1_global[j // 64, k, j % 64]
                            )
                            T.writes(T_matmul_NT_global[i, j])
                            T.block_attr(
                                {
                                    "layout_free_placeholders": [],
                                    "meta_schedule.tiling_structure": "SSRSRS",
                                }
                            )
                            T_matmul_NT_global[i, j] = (
                                T_matmul_NT_global[i, j] + p0[i, k] * p1_global[j // 64, k, j % 64]
                            )
            for ax0 in T.serial(64):
                for ax1_fused in T.vectorized(64):
                    with T.block("T_matmul_NT_global"):
                        v0 = T.axis.spatial(128, i0_0_i1_0_fused_fused // 2 * 64 + ax0)
                        v1 = T.axis.spatial(128, i0_0_i1_0_fused_fused % 2 * 64 + ax1_fused)
                        T.reads(T_matmul_NT_global[v0, v1])
                        T.writes(T_add[v0, v1])
                        T_add[v0, v1] = T_matmul_NT_global[v0, v1] + T.float32(1)


@tvm.script.ir_module
class DenseAdd_cpu_no_write_cache:
    @T.prim_func
    def main(p0: T.Buffer((128, 128), "float32"), p1: T.Buffer((128, 128), "float32"), T_add: T.Buffer((128, 128), "float32")) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True, "layout_free_buffers": [1]})
        # body
        # with T.block("root")
        T_matmul_NT = T.alloc_buffer([128, 128], dtype="float32")
        p1_global = T.alloc_buffer([8, 4, 16, 32], dtype="float32")
        for ax0, ax1 in T.grid(128, 128):
            with T.block("p1_global"):
                v0, v1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(p1[v0, v1])
                T.writes(p1_global[v1 // 16, v0 // 32, v1 % 16, v0 % 32])
                T.block_attr({"meta_schedule.layout_rewrite_preproc":1})
                p1_global[v1 // 16, v0 // 32, v1 % 16, v0 % 32] = p1[v0, v1]
        for i0_0_i1_0_i0_1_i1_1_fused in T.parallel(16, annotations={"pragma_auto_unroll_max_step":16, "pragma_unroll_explicit":1}):
            for i0_2_init, i1_2_init, i0_3_init in T.grid(4, 4, 2):
                for i1_3_fused_init in T.vectorized(32):
                    with T.block("T_matmul_NT_init"):
                        i = T.axis.spatial(128, i0_0_i1_0_i0_1_i1_1_fused * 8 + i0_2_init * 2 + i0_3_init)
                        j = T.axis.spatial(128, i1_2_init * 32 + i1_3_fused_init)
                        T.reads()
                        T.writes(T_matmul_NT[i, j])
                        T.block_attr({"layout_free_placeholders":[], "meta_schedule.tiling_structure":"SSRSRS"})
                        T_matmul_NT[i, j] = T.float32(0)
            for i2_0, i0_2, i1_2, i2_1, i0_3 in T.grid(8, 4, 4, 16, 2):
                for i1_3_fused in T.vectorized(32):
                    with T.block("T_matmul_NT_update"):
                        i = T.axis.spatial(128, i0_0_i1_0_i0_1_i1_1_fused * 8 + i0_2 * 2 + i0_3)
                        j = T.axis.spatial(128, i1_2 * 32 + i1_3_fused)
                        k = T.axis.reduce(128, i2_0 * 16 + i2_1)
                        T.reads(T_matmul_NT[i, j], p0[i, k], p1_global[k // 16, j // 32, k % 16, j % 32])
                        T.writes(T_matmul_NT[i, j])
                        T.block_attr({"layout_free_placeholders":[], "meta_schedule.tiling_structure":"SSRSRS"})
                        T_matmul_NT[i, j] = T_matmul_NT[i, j] + p0[i, k] * p1_global[k // 16, j // 32, k % 16, j % 32]
        for i0_i1_fused in T.parallel(16384):
            with T.block("T_add"):
                ax0 = T.axis.spatial(128, i0_i1_fused // 128)
                ax1 = T.axis.spatial(128, i0_i1_fused % 128)
                T.reads(T_matmul_NT[ax0, ax1])
                T.writes(T_add[ax0, ax1])
                T_add[ax0, ax1] = T_matmul_NT[ax0, ax1] + T.float32(1)


@tvm.script.ir_module
class DenseAdd_scheduled_gpu:
    @T.prim_func
    def main(
        p0: T.Buffer((128, 128), "float32"),
        p1: T.Buffer((128, 128), "float32"),
        T_add: T.Buffer((128, 128), "float32"),
    ) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True, "layout_free_buffers": [1]})
        # body
        # with T.block("root")
        T_matmul_NT_local = T.alloc_buffer([128, 128], dtype="float32", scope="local")
        p0_shared = T.alloc_buffer([128, 128], dtype="float32", scope="shared")
        p1_shared = T.alloc_buffer([128, 128], dtype="float32", scope="shared")
        for i0_0_i1_0_fused in T.thread_binding(
            32,
            thread="blockIdx.x",
            annotations={"pragma_auto_unroll_max_step": 64, "pragma_unroll_explicit": 1},
        ):
            for i0_1_i1_1_fused in T.thread_binding(1, thread="vthread.x"):
                for i0_2_i1_2_fused in T.thread_binding(128, thread="threadIdx.x"):
                    for i0_3_init, i1_3_init, i0_4_init, i1_4_init in T.grid(1, 4, 1, 1):
                        with T.block("T_matmul_NT_init"):
                            i = T.axis.spatial(
                                128,
                                i0_0_i1_0_fused // 4 * 16
                                + i0_2_i1_2_fused // 8
                                + i0_3_init
                                + i0_4_init,
                            )
                            j = T.axis.spatial(
                                128,
                                i0_0_i1_0_fused % 4 * 32
                                + i0_2_i1_2_fused % 8 * 4
                                + i1_3_init
                                + i1_4_init,
                            )
                            T.reads()
                            T.writes(T_matmul_NT_local[i, j])
                            T.block_attr(
                                {
                                    "layout_free_placeholders": [],
                                    "meta_schedule.thread_extent_high_inclusive": 256,
                                    "meta_schedule.thread_extent_low_inclusive": 16,
                                    "meta_schedule.tiling_structure": "SSSRRSRS",
                                }
                            )
                            T_matmul_NT_local[i, j] = T.float32(0)
                    for i2_0 in T.serial(32):
                        for ax0_ax1_fused_0 in T.serial(1):
                            for ax0_ax1_fused_1 in T.thread_binding(128, thread="threadIdx.x"):
                                for ax0_ax1_fused_2 in T.vectorized(2):
                                    with T.block("p0_shared"):
                                        T.where(
                                            (ax0_ax1_fused_0 * 128 + ax0_ax1_fused_1) * 2
                                            + ax0_ax1_fused_2
                                            < 64
                                        )
                                        v0 = T.axis.spatial(
                                            128,
                                            i0_0_i1_0_fused // 4 * 16
                                            + (
                                                ax0_ax1_fused_0 * 256
                                                + ax0_ax1_fused_1 * 2
                                                + ax0_ax1_fused_2
                                            )
                                            // 4,
                                        )
                                        v1 = T.axis.spatial(
                                            128,
                                            i2_0 * 4
                                            + (
                                                ax0_ax1_fused_0 * 256
                                                + ax0_ax1_fused_1 * 2
                                                + ax0_ax1_fused_2
                                            )
                                            % 4,
                                        )
                                        T.reads(p0[v0, v1])
                                        T.writes(p0_shared[v0, v1])
                                        p0_shared[v0, v1] = p0[v0, v1]
                        for ax0_ax1_fused_0 in T.serial(1):
                            for ax0_ax1_fused_1 in T.thread_binding(128, thread="threadIdx.x"):
                                for ax0_ax1_fused_2 in T.vectorized(4):
                                    with T.block("p1_shared"):
                                        T.where(
                                            (ax0_ax1_fused_0 * 128 + ax0_ax1_fused_1) * 4
                                            + ax0_ax1_fused_2
                                            < 128
                                        )
                                        v0 = T.axis.spatial(
                                            128,
                                            i0_0_i1_0_fused % 4 * 32
                                            + (
                                                ax0_ax1_fused_0 * 512
                                                + ax0_ax1_fused_1 * 4
                                                + ax0_ax1_fused_2
                                            )
                                            // 4,
                                        )
                                        v1 = T.axis.spatial(
                                            128,
                                            i2_0 * 4
                                            + (
                                                ax0_ax1_fused_0 * 512
                                                + ax0_ax1_fused_1 * 4
                                                + ax0_ax1_fused_2
                                            )
                                            % 4,
                                        )
                                        T.reads(p1[v0, v1])
                                        T.writes(p1_shared[v0, v1])
                                        p1_shared[v0, v1] = p1[v0, v1]
                        for i2_1, i0_3, i1_3, i2_2, i0_4, i1_4 in T.grid(1, 1, 4, 4, 1, 1):
                            with T.block("T_matmul_NT_update"):
                                i = T.axis.spatial(
                                    128,
                                    i0_0_i1_0_fused // 4 * 16 + i0_2_i1_2_fused // 8 + i0_3 + i0_4,
                                )
                                j = T.axis.spatial(
                                    128,
                                    i0_0_i1_0_fused % 4 * 32
                                    + i0_2_i1_2_fused % 8 * 4
                                    + i1_3
                                    + i1_4,
                                )
                                k = T.axis.reduce(128, i2_0 * 4 + i2_1 * 4 + i2_2)
                                T.reads(T_matmul_NT_local[i, j], p0_shared[i, k], p1_shared[j, k])
                                T.writes(T_matmul_NT_local[i, j])
                                T.block_attr(
                                    {
                                        "layout_free_placeholders": [],
                                        "meta_schedule.thread_extent_high_inclusive": 256,
                                        "meta_schedule.thread_extent_low_inclusive": 16,
                                        "meta_schedule.tiling_structure": "SSSRRSRS",
                                    }
                                )
                                T_matmul_NT_local[i, j] = (
                                    T_matmul_NT_local[i, j] + p0_shared[i, k] * p1_shared[j, k]
                                )
                    for ax0, ax1 in T.grid(1, 4):
                        with T.block("T_matmul_NT_local"):
                            v0 = T.axis.spatial(
                                128, i0_0_i1_0_fused // 4 * 16 + i0_2_i1_2_fused // 8 + ax0
                            )
                            v1 = T.axis.spatial(
                                128, i0_0_i1_0_fused % 4 * 32 + i0_2_i1_2_fused % 8 * 4 + ax1
                            )
                            T.reads(T_matmul_NT_local[v0, v1])
                            T.writes(T_add[v0, v1])
                            T_add[v0, v1] = T_matmul_NT_local[v0, v1] + T.float32(1)


@tvm.script.ir_module
class Conv2dInt8:
    @T.prim_func
    def main(p0: T.Buffer((16, 56, 56, 64), "int8"), p1: T.Buffer((256, 1, 1, 64), "int8"), p2: T.Buffer((1, 1, 1, 256), "int32"), p3: T.Buffer((1, 1, 1, 256), "int32"), p4: T.Buffer((1, 1, 1, 256), "int64"), p5: T.Buffer((1, 1, 1, 256), "int64"), p6: T.Buffer((1, 1, 1, 256), "int64"), p7: T.Buffer((), "int32"), p8: T.Buffer(1, "int32"), compute: T.Buffer((16, 56, 56, 256), "int32")) -> None:
        # function attr dict
        T.func_attr({"tir.noalias": True, "global_symbol": "main"})
        # body
        # with T.block("root")
        pad_temp = T.alloc_buffer([16, 56, 56, 64], dtype="int8")
        conv2d_nhwc = T.alloc_buffer([16, 56, 56, 256], dtype="int32")
        T_subtract = T.alloc_buffer([16, 56, 56, 256], dtype="int32")
        T_add = T.alloc_buffer([16, 56, 56, 256], dtype="int32")
        T_cast = T.alloc_buffer([16, 56, 56, 256], dtype="int64")
        T_multiply = T.alloc_buffer([16, 56, 56, 256], dtype="int64")
        T_add_1 = T.alloc_buffer([16, 56, 56, 256], dtype="int64")
        T_right_shift = T.alloc_buffer([16, 56, 56, 256], dtype="int64")
        T_cast_1 = T.alloc_buffer([16, 56, 56, 256], dtype="int32")
        T_add_2 = T.alloc_buffer([16, 56, 56, 256], dtype="int32")
        compute_1 = T.alloc_buffer([16, 56, 56, 256], dtype="int32")
        T_cast_2 = T.alloc_buffer([16, 56, 56, 256], dtype="uint8")
        T_cast_3 = T.alloc_buffer([16, 56, 56, 256], dtype="int32")
        T_subtract_1 = T.alloc_buffer([16, 56, 56, 256], dtype="int32")
        for i0, i1, i2, i3 in T.grid(16, 56, 56, 64):
            with T.block("pad_temp"):
                i0_1, i1_1, i2_1, i3_1 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(p0[i0_1, i1_1, i2_1, i3_1])
                T.writes(pad_temp[i0_1, i1_1, i2_1, i3_1])
                pad_temp[i0_1, i1_1, i2_1, i3_1] = p0[i0_1, i1_1, i2_1, i3_1]
        for i0, i1, i2, i3, i4, i5, i6 in T.grid(16, 56, 56, 256, 1, 1, 64):
            with T.block("conv2d_nhwc"):
                nn, yy, xx, ff, ry, rx, rc = T.axis.remap("SSSSRRR", [i0, i1, i2, i3, i4, i5, i6])
                T.reads(pad_temp[nn, yy + ry, xx + rx, rc], p1[ff, ry, rx, rc])
                T.writes(conv2d_nhwc[nn, yy, xx, ff])
                with T.init():
                    conv2d_nhwc[nn, yy, xx, ff] = 0
                conv2d_nhwc[nn, yy, xx, ff] = conv2d_nhwc[nn, yy, xx, ff] + T.cast(pad_temp[nn, yy + ry, xx + rx, rc], "int32") * T.cast(p1[ff, ry, rx, rc], "int32")
        for i0, i1, i2, i3 in T.grid(16, 56, 56, 256):
            with T.block("T_subtract"):
                ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(conv2d_nhwc[ax0, ax1, ax2, ax3], p2[0, 0, 0, ax3])
                T.writes(T_subtract[ax0, ax1, ax2, ax3])
                T_subtract[ax0, ax1, ax2, ax3] = conv2d_nhwc[ax0, ax1, ax2, ax3] - p2[0, 0, 0, ax3]
        for i0, i1, i2, i3 in T.grid(16, 56, 56, 256):
            with T.block("T_add"):
                ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(T_subtract[ax0, ax1, ax2, ax3], p3[0, 0, 0, ax3])
                T.writes(T_add[ax0, ax1, ax2, ax3])
                T_add[ax0, ax1, ax2, ax3] = T_subtract[ax0, ax1, ax2, ax3] + p3[0, 0, 0, ax3]
        for i0, i1, i2, i3 in T.grid(16, 56, 56, 256):
            with T.block("T_cast"):
                ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(T_add[ax0, ax1, ax2, ax3])
                T.writes(T_cast[ax0, ax1, ax2, ax3])
                T_cast[ax0, ax1, ax2, ax3] = T.cast(T_add[ax0, ax1, ax2, ax3], "int64")
        for i0, i1, i2, i3 in T.grid(16, 56, 56, 256):
            with T.block("T_multiply"):
                ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(T_cast[ax0, ax1, ax2, ax3], p4[0, 0, 0, ax3])
                T.writes(T_multiply[ax0, ax1, ax2, ax3])
                T_multiply[ax0, ax1, ax2, ax3] = T_cast[ax0, ax1, ax2, ax3] * p4[0, 0, 0, ax3]
        for i0, i1, i2, i3 in T.grid(16, 56, 56, 256):
            with T.block("T_add_1"):
                ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(T_multiply[ax0, ax1, ax2, ax3], p5[0, 0, 0, ax3])
                T.writes(T_add_1[ax0, ax1, ax2, ax3])
                T_add_1[ax0, ax1, ax2, ax3] = T_multiply[ax0, ax1, ax2, ax3] + p5[0, 0, 0, ax3]
        for i0, i1, i2, i3 in T.grid(16, 56, 56, 256):
            with T.block("T_right_shift"):
                ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(T_add_1[ax0, ax1, ax2, ax3], p6[0, 0, 0, ax3])
                T.writes(T_right_shift[ax0, ax1, ax2, ax3])
                T_right_shift[ax0, ax1, ax2, ax3] = T.shift_right(T_add_1[ax0, ax1, ax2, ax3], p6[0, 0, 0, ax3], dtype="int64")
        for i0, i1, i2, i3 in T.grid(16, 56, 56, 256):
            with T.block("T_cast_1"):
                ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(T_right_shift[ax0, ax1, ax2, ax3])
                T.writes(T_cast_1[ax0, ax1, ax2, ax3])
                T_cast_1[ax0, ax1, ax2, ax3] = T.cast(T_right_shift[ax0, ax1, ax2, ax3], "int32")
        for i0, i1, i2, i3 in T.grid(16, 56, 56, 256):
            with T.block("T_add_2"):
                ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(p7[()], T_cast_1[ax0, ax1, ax2, ax3])
                T.writes(T_add_2[ax0, ax1, ax2, ax3])
                T_add_2[ax0, ax1, ax2, ax3] = p7[()] + T_cast_1[ax0, ax1, ax2, ax3]
        for i0, i1, i2, i3 in T.grid(16, 56, 56, 256):
            with T.block("compute"):
                i0_2, i1_2, i2_2, i3_2 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(T_add_2[i0_2, i1_2, i2_2, i3_2])
                T.writes(compute_1[i0_2, i1_2, i2_2, i3_2])
                compute_1[i0_2, i1_2, i2_2, i3_2] = T.max(T.min(T_add_2[i0_2, i1_2, i2_2, i3_2], 255), 0)
        for i0_3, i1_3, i2_3, i3_3 in T.grid(16, 56, 56, 256):
            with T.block("T_cast_2"):
                ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0_3, i1_3, i2_3, i3_3])
                T.reads(compute_1[ax0, ax1, ax2, ax3])
                T.writes(T_cast_2[ax0, ax1, ax2, ax3])
                T_cast_2[ax0, ax1, ax2, ax3] = T.cast(compute_1[ax0, ax1, ax2, ax3], "uint8")
        for i0_4, i1_4, i2_4, i3_4 in T.grid(16, 56, 56, 256):
            with T.block("T_cast_3"):
                ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0_4, i1_4, i2_4, i3_4])
                T.reads(T_cast_2[ax0, ax1, ax2, ax3])
                T.writes(T_cast_3[ax0, ax1, ax2, ax3])
                T_cast_3[ax0, ax1, ax2, ax3] = T.cast(T_cast_2[ax0, ax1, ax2, ax3], "int32")
        for i0_5, i1_5, i2_5, i3_5 in T.grid(16, 56, 56, 256):
            with T.block("T_subtract_1"):
                ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0_5, i1_5, i2_5, i3_5])
                T.reads(T_cast_3[ax0, ax1, ax2, ax3], p8[0])
                T.writes(T_subtract_1[ax0, ax1, ax2, ax3])
                T_subtract_1[ax0, ax1, ax2, ax3] = T_cast_3[ax0, ax1, ax2, ax3] - p8[0]
        for i0_6, i1_6, i2_6, i3_6 in T.grid(16, 56, 56, 256):
            with T.block("compute_1"):
                i0_7, i1_7, i2_7, i3_7 = T.axis.remap("SSSS", [i0_6, i1_6, i2_6, i3_6])
                T.reads(T_subtract_1[i0_7, i1_7, i2_7, i3_7])
                T.writes(compute[i0_7, i1_7, i2_7, i3_7])
                compute[i0_7, i1_7, i2_7, i3_7] = T.q_multiply_shift(T_subtract_1[i0_7, i1_7, i2_7, i3_7], 1963325822, 31, 1, dtype="int32")


@tvm.script.ir_module
class Conv2dInt8_target:
    @T.prim_func
    def main(p0: T.Buffer((16, 56, 56, 64), "int8"), p1: T.Buffer((256, 1, 1, 64), "int8"), p2: T.Buffer((1, 1, 1, 256), "int32"), p3: T.Buffer((1, 1, 1, 256), "int32"), p4: T.Buffer((1, 1, 1, 256), "int64"), p5: T.Buffer((1, 1, 1, 256), "int64"), p6: T.Buffer((1, 1, 1, 256), "int64"), p7: T.Buffer((), "int32"), p8: T.Buffer(1, "int32"), p9: T.Buffer((16, 56, 56, 256), "int32"), compute: T.Buffer((16, 56, 56, 256), "uint8")) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # body
        # with T.block("root")
        pad_temp = T.alloc_buffer([16, 56, 56, 64], dtype="int8")
        conv2d_nhwc = T.alloc_buffer([16, 56, 56, 256], dtype="int32")
        T_subtract = T.alloc_buffer([16, 56, 56, 256], dtype="int32")
        T_add = T.alloc_buffer([16, 56, 56, 256], dtype="int32")
        T_cast = T.alloc_buffer([16, 56, 56, 256], dtype="int64")
        T_multiply = T.alloc_buffer([16, 56, 56, 256], dtype="int64")
        T_add_1 = T.alloc_buffer([16, 56, 56, 256], dtype="int64")
        T_right_shift = T.alloc_buffer([16, 56, 56, 256], dtype="int64")
        T_cast_1 = T.alloc_buffer([16, 56, 56, 256], dtype="int32")
        T_add_2 = T.alloc_buffer([16, 56, 56, 256], dtype="int32")
        compute_1 = T.alloc_buffer([16, 56, 56, 256], dtype="int32")
        T_cast_2 = T.alloc_buffer([16, 56, 56, 256], dtype="uint8")
        T_cast_3 = T.alloc_buffer([16, 56, 56, 256], dtype="int32")
        T_subtract_1 = T.alloc_buffer([16, 56, 56, 256], dtype="int32")
        compute_2 = T.alloc_buffer([16, 56, 56, 256], dtype="int32")
        T_add_3 = T.alloc_buffer([16, 56, 56, 256], dtype="int32")
        compute_3 = T.alloc_buffer([16, 56, 56, 256], dtype="int32")
        T_cast_4 = T.alloc_buffer([16, 56, 56, 256], dtype="uint8")
        for i0, i1, i2, i3 in T.grid(16, 56, 56, 64):
            with T.block("pad_temp"):
                i0_1, i1_1, i2_1, i3_1 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(p0[i0_1, i1_1, i2_1, i3_1])
                T.writes(pad_temp[i0_1, i1_1, i2_1, i3_1])
                pad_temp[i0_1, i1_1, i2_1, i3_1] = p0[i0_1, i1_1, i2_1, i3_1]
        for i0, i1, i2, i3, i4, i5, i6 in T.grid(16, 56, 56, 256, 1, 1, 64):
            with T.block("conv2d_nhwc"):
                nn, yy, xx, ff, ry, rx, rc = T.axis.remap("SSSSRRR", [i0, i1, i2, i3, i4, i5, i6])
                T.reads(pad_temp[nn, yy + ry, xx + rx, rc], p1[ff, ry, rx, rc])
                T.writes(conv2d_nhwc[nn, yy, xx, ff])
                with T.init():
                    conv2d_nhwc[nn, yy, xx, ff] = 0
                conv2d_nhwc[nn, yy, xx, ff] = conv2d_nhwc[nn, yy, xx, ff] + T.cast(pad_temp[nn, yy + ry, xx + rx, rc], "int32") * T.cast(p1[ff, ry, rx, rc], "int32")
        for i0, i1, i2, i3 in T.grid(16, 56, 56, 256):
            with T.block("T_subtract"):
                ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(conv2d_nhwc[ax0, ax1, ax2, ax3], p2[0, 0, 0, ax3])
                T.writes(T_subtract[ax0, ax1, ax2, ax3])
                T_subtract[ax0, ax1, ax2, ax3] = conv2d_nhwc[ax0, ax1, ax2, ax3] - p2[0, 0, 0, ax3]
        for i0, i1, i2, i3 in T.grid(16, 56, 56, 256):
            with T.block("T_add"):
                ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(T_subtract[ax0, ax1, ax2, ax3], p3[0, 0, 0, ax3])
                T.writes(T_add[ax0, ax1, ax2, ax3])
                T_add[ax0, ax1, ax2, ax3] = T_subtract[ax0, ax1, ax2, ax3] + p3[0, 0, 0, ax3]
        for i0, i1, i2, i3 in T.grid(16, 56, 56, 256):
            with T.block("T_cast"):
                ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(T_add[ax0, ax1, ax2, ax3])
                T.writes(T_cast[ax0, ax1, ax2, ax3])
                T_cast[ax0, ax1, ax2, ax3] = T.cast(T_add[ax0, ax1, ax2, ax3], "int64")
        for i0, i1, i2, i3 in T.grid(16, 56, 56, 256):
            with T.block("T_multiply"):
                ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(T_cast[ax0, ax1, ax2, ax3], p4[0, 0, 0, ax3])
                T.writes(T_multiply[ax0, ax1, ax2, ax3])
                T_multiply[ax0, ax1, ax2, ax3] = T_cast[ax0, ax1, ax2, ax3] * p4[0, 0, 0, ax3]
        for i0, i1, i2, i3 in T.grid(16, 56, 56, 256):
            with T.block("T_add_1"):
                ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(T_multiply[ax0, ax1, ax2, ax3], p5[0, 0, 0, ax3])
                T.writes(T_add_1[ax0, ax1, ax2, ax3])
                T_add_1[ax0, ax1, ax2, ax3] = T_multiply[ax0, ax1, ax2, ax3] + p5[0, 0, 0, ax3]
        for i0, i1, i2, i3 in T.grid(16, 56, 56, 256):
            with T.block("T_right_shift"):
                ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(T_add_1[ax0, ax1, ax2, ax3], p6[0, 0, 0, ax3])
                T.writes(T_right_shift[ax0, ax1, ax2, ax3])
                T_right_shift[ax0, ax1, ax2, ax3] = T.shift_right(T_add_1[ax0, ax1, ax2, ax3], p6[0, 0, 0, ax3], dtype="int64")
        for i0, i1, i2, i3 in T.grid(16, 56, 56, 256):
            with T.block("T_cast_1"):
                ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(T_right_shift[ax0, ax1, ax2, ax3])
                T.writes(T_cast_1[ax0, ax1, ax2, ax3])
                T_cast_1[ax0, ax1, ax2, ax3] = T.cast(T_right_shift[ax0, ax1, ax2, ax3], "int32")
        for i0, i1, i2, i3 in T.grid(16, 56, 56, 256):
            with T.block("T_add_2"):
                ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(p7[()], T_cast_1[ax0, ax1, ax2, ax3])
                T.writes(T_add_2[ax0, ax1, ax2, ax3])
                T_add_2[ax0, ax1, ax2, ax3] = p7[()] + T_cast_1[ax0, ax1, ax2, ax3]
        for i0, i1, i2, i3 in T.grid(16, 56, 56, 256):
            with T.block("compute"):
                i0_2, i1_2, i2_2, i3_2 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(T_add_2[i0_2, i1_2, i2_2, i3_2])
                T.writes(compute_1[i0_2, i1_2, i2_2, i3_2])
                compute_1[i0_2, i1_2, i2_2, i3_2] = T.max(T.min(T_add_2[i0_2, i1_2, i2_2, i3_2], 255), 0)
        for i0_3, i1_3, i2_3, i3_3 in T.grid(16, 56, 56, 256):
            with T.block("T_cast_2"):
                ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0_3, i1_3, i2_3, i3_3])
                T.reads(compute_1[ax0, ax1, ax2, ax3])
                T.writes(T_cast_2[ax0, ax1, ax2, ax3])
                T_cast_2[ax0, ax1, ax2, ax3] = T.cast(compute_1[ax0, ax1, ax2, ax3], "uint8")
        for i0_4, i1_4, i2_4, i3_4 in T.grid(16, 56, 56, 256):
            with T.block("T_cast_3"):
                ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0_4, i1_4, i2_4, i3_4])
                T.reads(T_cast_2[ax0, ax1, ax2, ax3])
                T.writes(T_cast_3[ax0, ax1, ax2, ax3])
                T_cast_3[ax0, ax1, ax2, ax3] = T.cast(T_cast_2[ax0, ax1, ax2, ax3], "int32")
        for i0_5, i1_5, i2_5, i3_5 in T.grid(16, 56, 56, 256):
            with T.block("T_subtract_1"):
                ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0_5, i1_5, i2_5, i3_5])
                T.reads(T_cast_3[ax0, ax1, ax2, ax3], p8[0])
                T.writes(T_subtract_1[ax0, ax1, ax2, ax3])
                T_subtract_1[ax0, ax1, ax2, ax3] = T_cast_3[ax0, ax1, ax2, ax3] - p8[0]
        for i0_6, i1_6, i2_6, i3_6 in T.grid(16, 56, 56, 256):
            with T.block("compute_1"):
                i0_7, i1_7, i2_7, i3_7 = T.axis.remap("SSSS", [i0_6, i1_6, i2_6, i3_6])
                T.reads(T_subtract_1[i0_7, i1_7, i2_7, i3_7])
                T.writes(compute_2[i0_7, i1_7, i2_7, i3_7])
                compute_2[i0_7, i1_7, i2_7, i3_7] = T.q_multiply_shift(T_subtract_1[i0_7, i1_7, i2_7, i3_7], 1098990753, 31, 1, dtype="int32")
        for i0_8, i1_8, i2_8, i3_8 in T.grid(16, 56, 56, 256):
            with T.block("T_add_3"):
                ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0_8, i1_8, i2_8, i3_8])
                T.reads(compute_2[ax0, ax1, ax2, ax3], p9[ax0, ax1, ax2, ax3])
                T.writes(T_add_3[ax0, ax1, ax2, ax3])
                T_add_3[ax0, ax1, ax2, ax3] = compute_2[ax0, ax1, ax2, ax3] + p9[ax0, ax1, ax2, ax3]
        for i0_9, i1_9, i2_9, i3_9 in T.grid(16, 56, 56, 256):
            with T.block("compute_2"):
                i0_10, i1_10, i2_10, i3_10 = T.axis.remap("SSSS", [i0_9, i1_9, i2_9, i3_9])
                T.reads(T_add_3[i0_10, i1_10, i2_10, i3_10])
                T.writes(compute_3[i0_10, i1_10, i2_10, i3_10])
                compute_3[i0_10, i1_10, i2_10, i3_10] = T.max(T.min(T_add_3[i0_10, i1_10, i2_10, i3_10], 255), 0)
        for i0_11, i1_11, i2_11, i3_11 in T.grid(16, 56, 56, 256):
            with T.block("T_cast_4"):
                ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0_11, i1_11, i2_11, i3_11])
                T.reads(compute_3[ax0, ax1, ax2, ax3])
                T.writes(T_cast_4[ax0, ax1, ax2, ax3])
                T_cast_4[ax0, ax1, ax2, ax3] = T.cast(compute_3[ax0, ax1, ax2, ax3], "uint8")
        for i0_12, i1_12, i2_12, i3_12 in T.grid(16, 56, 56, 256):
            with T.block("compute_3"):
                i0_13, i1_13, i2_13, i3_13 = T.axis.remap("SSSS", [i0_12, i1_12, i2_12, i3_12])
                T.reads(T_cast_4[i0_13, i1_13, i2_13, i3_13])
                T.writes(compute[i0_13, i1_13, i2_13, i3_13])
                compute[i0_13, i1_13, i2_13, i3_13] = T.max(T.min(T_cast_4[i0_13, i1_13, i2_13, i3_13], T.uint8(255)), T.uint8(0))


@tvm.script.ir_module
class Conv2dInt8_tensorcore_scheduled:
    @T.prim_func
    def main(p0: T.Buffer((16, 56, 56, 64), "int8"), p1: T.Buffer((256, 1, 1, 64), "int8"), p2: T.Buffer((1, 1, 1, 256), "int32"), p3: T.Buffer((1, 1, 1, 256), "int32"), p4: T.Buffer((1, 1, 1, 256), "int64"), p5: T.Buffer((1, 1, 1, 256), "int64"), p6: T.Buffer((1, 1, 1, 256), "int64"), p7: T.Buffer((), "int32"), p8: T.Buffer((1,), "int32"), p9: T.Buffer((16, 56, 56, 256), "int32"), compute: T.Buffer((16, 56, 56, 256), "uint8")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        conv2d_nhwc_reindex_shared = T.alloc_buffer((50176, 256), "int32", scope="shared")
        conv2d_nhwc_reindex_shared_wmma_accumulator = T.alloc_buffer((50176, 256), "int32", scope="wmma.accumulator")
        pad_temp_reindex_shared = T.alloc_buffer((50176, 64), "int8", scope="shared")
        p1_reindex_shared = T.alloc_buffer((1, 1, 256, 64), "int8", scope="shared")
        pad_temp_reindex_shared_wmma_matrix_a = T.alloc_buffer((50176, 64), "int8", scope="wmma.matrix_a")
        p1_reindex_shared_wmma_matrix_b = T.alloc_buffer((1, 1, 256, 64), "int8", scope="wmma.matrix_b")
        for ax2_0_0_ax3_0_0_fused in T.thread_binding(3136, thread="blockIdx.x", annotations={"pragma_auto_unroll_max_step": 512, "pragma_unroll_explicit": 1}):
            for ax2_0_1_ax3_0_1_fused in T.thread_binding(1, thread="vthread.x"):
                for ax2_0_2_ax3_0_2_fused in T.thread_binding(16, thread="threadIdx.x"):
                    for ax0_0_init, ax1_0_init, ax0_1_init, ax1_1_init, ax2_0_3_init, ax3_0_3_init, ax0_2_init, ax1_2_init, ax2_0_4_init, ax3_0_4_init in T.grid(1, 1, 1, 1, 1, 1, 1, 1, 1, 1):
                        with T.block("conv2d_nhwc_o_init"):
                            v0_o = T.axis.spatial(1, ax0_0_init + ax0_1_init + ax0_2_init)
                            v1_o = T.axis.spatial(1, ax1_0_init + ax1_1_init + ax1_2_init)
                            v2_o = T.axis.spatial(3136, ax2_0_0_ax3_0_0_fused // 8 * 8 + ax2_0_2_ax3_0_2_fused // 2 + ax2_0_3_init + ax2_0_4_init)
                            v3_o = T.axis.spatial(16, ax2_0_0_ax3_0_0_fused % 8 * 2 + ax2_0_2_ax3_0_2_fused % 2 + ax3_0_3_init + ax3_0_4_init)
                            T.reads()
                            T.writes(conv2d_nhwc_reindex_shared_wmma_accumulator[v2_o * 16:v2_o * 16 + 16, v3_o * 16:v3_o * 16 + 16])
                            T.block_attr({"meta_schedule.thread_extent_high_inclusive": 1024, "meta_schedule.thread_extent_low_inclusive": 32, "warp_execution": 1})
                            C = T.match_buffer(conv2d_nhwc_reindex_shared_wmma_accumulator[v2_o * 16:v2_o * 16 + 16, v3_o * 16:v3_o * 16 + 16], (16, 16), "int32", strides=("C_s0", "C_s1"), scope="wmma.accumulator", offset_factor=16)
                            T.tvm_fill_fragment(C.data, 16, 16, 16, C.elem_offset // C.strides[0] // 16 * (C.strides[0] // 16) + C.elem_offset % C.strides[0] // 16, T.float32(0))
                    for ax0_0, ax1_0, ax4_0_0 in T.grid(1, 1, 2):
                        for ax0_ax1_fused_0 in range(16):
                            for ax0_ax1_fused_1 in T.thread_binding(16, thread="threadIdx.x"):
                                for ax0_ax1_fused_2 in T.vectorized(16):
                                    with T.block("pad_temp_reindex_shared"):
                                        v0 = T.axis.spatial(50176, ax2_0_0_ax3_0_0_fused // 8 * 128 + (ax0_ax1_fused_0 * 256 + ax0_ax1_fused_1 * 16 + ax0_ax1_fused_2) // 32)
                                        v1 = T.axis.spatial(64, ax4_0_0 * 32 + (ax0_ax1_fused_0 * 256 + ax0_ax1_fused_1 * 16 + ax0_ax1_fused_2) % 32)
                                        T.reads(p0[v0 // 3136, v0 % 3136 // 56, v0 % 56, v1])
                                        T.writes(pad_temp_reindex_shared[v0, v1])
                                        T.block_attr({"buffer_dim_align": [[0, 0, 32, 16]]})
                                        pad_temp_reindex_shared[v0, v1] = p0[v0 // 3136, v0 % 3136 // 56, v0 % 56, v1]
                        for ax0_ax1_ax2_ax3_fused_0 in range(8):
                            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(16, thread="threadIdx.x"):
                                for ax0_ax1_ax2_ax3_fused_2 in T.vectorized(8):
                                    with T.block("p1_reindex_shared"):
                                        v0 = T.axis.spatial(1, 0)
                                        v1 = T.axis.spatial(1, 0)
                                        v2 = T.axis.spatial(256, ax2_0_0_ax3_0_0_fused % 8 * 32 + (ax0_ax1_ax2_ax3_fused_0 * 128 + ax0_ax1_ax2_ax3_fused_1 * 8 + ax0_ax1_ax2_ax3_fused_2) // 32)
                                        v3 = T.axis.spatial(64, ax4_0_0 * 32 + (ax0_ax1_ax2_ax3_fused_0 * 128 + ax0_ax1_ax2_ax3_fused_1 * 8 + ax0_ax1_ax2_ax3_fused_2) % 32)
                                        T.reads(p1[v2, v0, v1, v3])
                                        T.writes(p1_reindex_shared[v0, v1, v2, v3])
                                        T.block_attr({"buffer_dim_align": [[0, 2, 32, 16]]})
                                        p1_reindex_shared[v0, v1, v2, v3] = p1[v2, v0, v1, v3]
                        for ax0_1, ax1_1, ax4_0_1 in T.grid(1, 1, 1):
                            for ax0_0_1, ax1_0_1 in T.grid(1, 2):
                                with T.block("pad_temp_reindex_shared_wmma.matrix_a_o"):
                                    v0_o = T.axis.spatial(3136, ax2_0_0_ax3_0_0_fused // 8 * 8 + ax2_0_2_ax3_0_2_fused // 2 + ax0_0_1)
                                    v1_o = T.axis.spatial(4, ax4_0_0 * 2 + ax1_0_1)
                                    T.reads(pad_temp_reindex_shared[v0_o * 16:v0_o * 16 + 16, v1_o * 16:v1_o * 16 + 16])
                                    T.writes(pad_temp_reindex_shared_wmma_matrix_a[v0_o * 16:v0_o * 16 + 16, v1_o * 16:v1_o * 16 + 16])
                                    A = T.match_buffer(pad_temp_reindex_shared[v0_o * 16:v0_o * 16 + 16, v1_o * 16:v1_o * 16 + 16], (16, 16), "int8", strides=("A_s0", "A_s1"), scope="shared", offset_factor=16)
                                    C = T.match_buffer(pad_temp_reindex_shared_wmma_matrix_a[v0_o * 16:v0_o * 16 + 16, v1_o * 16:v1_o * 16 + 16], (16, 16), "int8", strides=("C_s0", "C_s1"), scope="wmma.matrix_a", offset_factor=16)
                                    T.tvm_load_matrix_sync(C.data, 16, 16, 16, C.elem_offset // C.strides[0] // 16 * (C.strides[0] // 16) + C.elem_offset % C.strides[0] // 16, T.tvm_access_ptr(T.type_annotation("int8"), A.data, A.elem_offset, A.strides[0] * 16, 1), A.strides[0], "row_major")
                            for ax0, ax1, ax2_0, ax3_0 in T.grid(1, 1, 1, 2):
                                with T.block("p1_reindex_shared_wmma.matrix_b_o"):
                                    v0_o, v1_o = T.axis.remap("SS", [ax0, ax1])
                                    v2_o = T.axis.spatial(16, ax2_0_0_ax3_0_0_fused % 8 * 2 + ax2_0_2_ax3_0_2_fused % 2 + ax2_0)
                                    v3_o = T.axis.spatial(4, ax4_0_0 * 2 + ax3_0)
                                    T.reads(p1_reindex_shared[v0_o, v1_o, v2_o * 16:v2_o * 16 + 16, v3_o * 16:v3_o * 16 + 16])
                                    T.writes(p1_reindex_shared_wmma_matrix_b[v0_o, v1_o, v2_o * 16:v2_o * 16 + 16, v3_o * 16:v3_o * 16 + 16])
                                    A = T.match_buffer(p1_reindex_shared[v0_o, v1_o, v2_o * 16:v2_o * 16 + 16, v3_o * 16:v3_o * 16 + 16], (16, 16), "int8", strides=("A_s0", "A_s1"), scope="shared", offset_factor=16)
                                    C = T.match_buffer(p1_reindex_shared_wmma_matrix_b[v0_o, v1_o, v2_o * 16:v2_o * 16 + 16, v3_o * 16:v3_o * 16 + 16], (16, 16), "int8", strides=("C_s0", "C_s1"), scope="wmma.matrix_b", offset_factor=16)
                                    T.tvm_load_matrix_sync(C.data, 16, 16, 16, C.elem_offset // C.strides[0] // 16 * (C.strides[0] // 16) + C.elem_offset % C.strides[0] // 16, T.tvm_access_ptr(T.type_annotation("int8"), A.data, A.elem_offset, A.strides[0] * 16, 1), A.strides[0], "col_major")
                            for ax2_0_3, ax3_0_3, ax0_2, ax1_2, ax4_0_2, ax2_0_4, ax3_0_4 in T.grid(1, 1, 1, 1, 2, 1, 1):
                                with T.block("conv2d_nhwc_o_update"):
                                    v0_o = T.axis.spatial(1, ax0_0 + ax0_1 + ax0_2)
                                    v1_o = T.axis.spatial(1, ax1_0 + ax1_1 + ax1_2)
                                    v2_o = T.axis.spatial(3136, ax2_0_0_ax3_0_0_fused // 8 * 8 + ax2_0_2_ax3_0_2_fused // 2 + ax2_0_3 + ax2_0_4)
                                    v3_o = T.axis.spatial(16, ax2_0_0_ax3_0_0_fused % 8 * 2 + ax2_0_2_ax3_0_2_fused % 2 + ax3_0_3 + ax3_0_4)
                                    v4_o = T.axis.reduce(4, ax4_0_0 * 2 + ax4_0_1 * 2 + ax4_0_2)
                                    T.reads(conv2d_nhwc_reindex_shared_wmma_accumulator[v2_o * 16:v2_o * 16 + 16, v3_o * 16:v3_o * 16 + 16], pad_temp_reindex_shared_wmma_matrix_a[v2_o * 16:v2_o * 16 + 16, v4_o * 16:v4_o * 16 + 16], p1_reindex_shared_wmma_matrix_b[v0_o, v1_o, v3_o * 16:v3_o * 16 + 16, v4_o * 16:v4_o * 16 + 16])
                                    T.writes(conv2d_nhwc_reindex_shared_wmma_accumulator[v2_o * 16:v2_o * 16 + 16, v3_o * 16:v3_o * 16 + 16])
                                    T.block_attr({"meta_schedule.thread_extent_high_inclusive": 1024, "meta_schedule.thread_extent_low_inclusive": 32, "warp_execution": 1})
                                    A = T.match_buffer(pad_temp_reindex_shared_wmma_matrix_a[v2_o * 16:v2_o * 16 + 16, v4_o * 16:v4_o * 16 + 16], (16, 16), "int8", strides=("A_s0", "A_s1"), scope="wmma.matrix_a", offset_factor=16)
                                    B = T.match_buffer(p1_reindex_shared_wmma_matrix_b[v0_o, v1_o, v3_o * 16:v3_o * 16 + 16, v4_o * 16:v4_o * 16 + 16], (16, 16), "int8", strides=("B_s0", "B_s1"), scope="wmma.matrix_b", offset_factor=16)
                                    C = T.match_buffer(conv2d_nhwc_reindex_shared_wmma_accumulator[v2_o * 16:v2_o * 16 + 16, v3_o * 16:v3_o * 16 + 16], (16, 16), "int32", strides=("C_s0", "C_s1"), scope="wmma.accumulator", offset_factor=16)
                                    T.tvm_mma_sync(C.data, C.elem_offset // C.strides[0] // 16 * (C.strides[0] // 16) + C.elem_offset % C.strides[0] // 16, A.data, A.elem_offset // A.strides[0] // 16 * (A.strides[0] // 16) + A.elem_offset % A.strides[0] // 16, B.data, B.elem_offset // B.strides[0] // 16 * (B.strides[0] // 16) + B.elem_offset % B.strides[0] // 16, C.data, C.elem_offset // C.strides[0] // 16 * (C.strides[0] // 16) + C.elem_offset % C.strides[0] // 16)
                    for ax0_0, ax1_0 in T.grid(1, 1):
                        with T.block("conv2d_nhwc_reindex_shared_wmma.accumulator_o"):
                            v0_o = T.axis.spatial(3136, ax2_0_0_ax3_0_0_fused // 8 * 8 + ax2_0_2_ax3_0_2_fused // 2 + ax0_0)
                            v1_o = T.axis.spatial(16, ax2_0_0_ax3_0_0_fused % 8 * 2 + ax2_0_2_ax3_0_2_fused % 2 + ax1_0)
                            T.reads(conv2d_nhwc_reindex_shared_wmma_accumulator[v0_o * 16:v0_o * 16 + 16, v1_o * 16:v1_o * 16 + 16])
                            T.writes(conv2d_nhwc_reindex_shared[v0_o * 16:v0_o * 16 + 16, v1_o * 16:v1_o * 16 + 16])
                            A = T.match_buffer(conv2d_nhwc_reindex_shared_wmma_accumulator[v0_o * 16:v0_o * 16 + 16, v1_o * 16:v1_o * 16 + 16], (16, 16), "int32", strides=("A_s0", "A_s1"), scope="wmma.accumulator", offset_factor=16)
                            C = T.match_buffer(conv2d_nhwc_reindex_shared[v0_o * 16:v0_o * 16 + 16, v1_o * 16:v1_o * 16 + 16], (16, 16), "int32", strides=("C_s0", "C_s1"), scope="shared", offset_factor=16)
                            T.tvm_store_matrix_sync(A.data, 16, 16, 16, A.elem_offset // A.strides[0] // 16 * (A.strides[0] // 16) + A.elem_offset % A.strides[0] // 16, T.tvm_access_ptr(T.type_annotation("int32"), C.data, C.elem_offset, C.strides[0] * 16, 2), C.strides[0], "row_major")
                for ax0, ax1_0 in T.grid(128, 2):
                    for ax1_1 in T.thread_binding(16, thread="threadIdx.x"):
                        with T.block("conv2d_nhwc_reindex_shared"):
                            v0 = T.axis.spatial(50176, ax2_0_0_ax3_0_0_fused // 8 * 128 + ax0)
                            v1 = T.axis.spatial(256, ax2_0_0_ax3_0_0_fused % 8 * 32 + ax1_0 * 16 + ax1_1)
                            T.reads(p7[()], conv2d_nhwc_reindex_shared[v0, v1], p2[0, 0, 0, v1], p3[0, 0, 0, v1], p4[0, 0, 0, v1], p5[0, 0, 0, v1], p6[0, 0, 0, v1], p8[0], p9[v0 // 3136, v0 % 3136 // 56, v0 % 56, v1])
                            T.writes(compute[v0 // 3136, v0 % 3136 // 56, v0 % 56, v1])
                            compute[v0 // 3136, v0 % 3136 // 56, v0 % 56, v1] = T.max(T.min(T.Cast("uint8", T.max(T.min(T.q_multiply_shift(T.Cast("int32", T.Cast("uint8", T.max(T.min(p7[()] + T.Cast("int32", T.shift_right(T.Cast("int64", conv2d_nhwc_reindex_shared[v0, v1] - p2[0, 0, 0, v1] + p3[0, 0, 0, v1]) * p4[0, 0, 0, v1] + p5[0, 0, 0, v1], p6[0, 0, 0, v1])), 255), 0))) - p8[0], 1098990753, 31, 1) + p9[v0 // 3136, v0 % 3136 // 56, v0 % 56, v1], 255), 0)), T.uint8(255)), T.uint8(0))

@tvm.script.ir_module
class Conv2dInt8_NCHWc:
    @T.prim_func
    def main(p0: T.Buffer((1, 32, 7, 7, 16), "uint8"), p1: T.Buffer((128, 32, 1, 1, 4, 16, 4), "int8"), p2: T.Buffer((1, 128, 1, 1, 16), "int32"), p3: T.Buffer((1, 128, 1, 1, 16), "float32"), p4: T.Buffer(1, "float32"), p5: T.Buffer((1, 128, 7, 7, 16), "int32"), compute: T.Buffer((1, 128, 7, 7, 16), "uint8")) -> None:
        # function attr dict
        T.func_attr({"tir.noalias": True, "global_symbol": "main"})
        # body
        # with T.block("root")
        compile_engine_const = T.alloc_buffer([], dtype="float32")
        conv2d_NCHWc_int8 = T.alloc_buffer([1, 128, 7, 7, 16], dtype="int32")
        T_add = T.alloc_buffer([1, 128, 7, 7, 16], dtype="int32")
        T_cast = T.alloc_buffer([1, 128, 7, 7, 16], dtype="float32")
        T_multiply = T.alloc_buffer([1, 128, 7, 7, 16], dtype="float32")
        compile_engine_const_1 = T.alloc_buffer([], dtype="float32")
        T_add_1 = T.alloc_buffer([1, 128, 7, 7, 16], dtype="float32")
        T_floor = T.alloc_buffer([1, 128, 7, 7, 16], dtype="float32")
        T_cast_1 = T.alloc_buffer([1, 128, 7, 7, 16], dtype="int32")
        compute_1 = T.alloc_buffer([1, 128, 7, 7, 16], dtype="int32")
        T_cast_2 = T.alloc_buffer([1, 128, 7, 7, 16], dtype="uint8")
        T_cast_3 = T.alloc_buffer([1, 128, 7, 7, 16], dtype="float32")
        T_subtract = T.alloc_buffer([1, 128, 7, 7, 16], dtype="float32")
        T_multiply_1 = T.alloc_buffer([1, 128, 7, 7, 16], dtype="float32")
        compile_engine_const_2 = T.alloc_buffer([], dtype="float32")
        T_add_2 = T.alloc_buffer([1, 128, 7, 7, 16], dtype="float32")
        T_floor_1 = T.alloc_buffer([1, 128, 7, 7, 16], dtype="float32")
        T_cast_4 = T.alloc_buffer([1, 128, 7, 7, 16], dtype="int32")
        T_add_3 = T.alloc_buffer([1, 128, 7, 7, 16], dtype="int32")
        compute_2 = T.alloc_buffer([1, 128, 7, 7, 16], dtype="int32")
        T_cast_5 = T.alloc_buffer([1, 128, 7, 7, 16], dtype="uint8")
        with T.block("compile_engine_const"):
            vi = T.axis.spatial(1, 0)
            T.reads()
            T.writes(compile_engine_const[()])
            compile_engine_const[()] = T.float32(0.94537687301635742)
        for i0, i1, i2, i3, i4, i5, i6, i7, i8, i9 in T.grid(1, 128, 7, 7, 16, 1, 1, 32, 4, 4):
            with T.block("conv2d_NCHWc_int8"):
                n, oc_chunk, oh, ow, oc_block, kh, kw, ic_outer, ic_f_inner, ic_s_inner = T.axis.remap("SSSSSRRRRR", [i0, i1, i2, i3, i4, i5, i6, i7, i8, i9])
                T.reads(p0[n, ic_outer, oh + kh, ow + kw, ic_f_inner * 4 + ic_s_inner], p1[oc_chunk, ic_outer, kh, kw, ic_f_inner, oc_block, ic_s_inner])
                T.writes(conv2d_NCHWc_int8[n, oc_chunk, oh, ow, oc_block])
                T.block_attr({"schedule_rule":"meta_schedule.conv2d_NCHWc_int8", "workload":["conv2d_NCHWc_int8.x86", ["TENSOR", [1, 32, 7, 7, 16], "uint8"], ["TENSOR", [128, 32, 1, 1, 4, 16, 4], "int8"], [1, 1], [0, 0, 0, 0], [1, 1], "NCHW16c", "NCHW16c", "int32"]})
                with T.init():
                    conv2d_NCHWc_int8[n, oc_chunk, oh, ow, oc_block] = 0
                conv2d_NCHWc_int8[n, oc_chunk, oh, ow, oc_block] = conv2d_NCHWc_int8[n, oc_chunk, oh, ow, oc_block] + T.cast(p0[n, ic_outer, oh + kh, ow + kw, ic_f_inner * 4 + ic_s_inner], "int32") * T.cast(p1[oc_chunk, ic_outer, kh, kw, ic_f_inner, oc_block, ic_s_inner], "int32")
        for i0, i1, i2, i3, i4 in T.grid(1, 128, 7, 7, 16):
            with T.block("T_add"):
                ax0, ax1, ax2, ax3, ax4 = T.axis.remap("SSSSS", [i0, i1, i2, i3, i4])
                T.reads(conv2d_NCHWc_int8[ax0, ax1, ax2, ax3, ax4], p2[ax0, ax1, 0, 0, ax4])
                T.writes(T_add[ax0, ax1, ax2, ax3, ax4])
                T_add[ax0, ax1, ax2, ax3, ax4] = conv2d_NCHWc_int8[ax0, ax1, ax2, ax3, ax4] + p2[ax0, ax1, 0, 0, ax4]
        for i0, i1, i2, i3, i4 in T.grid(1, 128, 7, 7, 16):
            with T.block("T_cast"):
                ax0, ax1, ax2, ax3, ax4 = T.axis.remap("SSSSS", [i0, i1, i2, i3, i4])
                T.reads(T_add[ax0, ax1, ax2, ax3, ax4])
                T.writes(T_cast[ax0, ax1, ax2, ax3, ax4])
                T_cast[ax0, ax1, ax2, ax3, ax4] = T.cast(T_add[ax0, ax1, ax2, ax3, ax4], "float32")
        for i0, i1, i2, i3, i4 in T.grid(1, 128, 7, 7, 16):
            with T.block("T_multiply"):
                ax0, ax1, ax2, ax3, ax4 = T.axis.remap("SSSSS", [i0, i1, i2, i3, i4])
                T.reads(T_cast[ax0, ax1, ax2, ax3, ax4], p3[ax0, ax1, 0, 0, ax4])
                T.writes(T_multiply[ax0, ax1, ax2, ax3, ax4])
                T_multiply[ax0, ax1, ax2, ax3, ax4] = T_cast[ax0, ax1, ax2, ax3, ax4] * p3[ax0, ax1, 0, 0, ax4]
        with T.block("compile_engine_const_1"):
            vi = T.axis.spatial(1, 0)
            T.reads()
            T.writes(compile_engine_const_1[()])
            compile_engine_const_1[()] = T.float32(54.5)
        for i0, i1, i2, i3, i4 in T.grid(1, 128, 7, 7, 16):
            with T.block("T_add_1"):
                ax0, ax1, ax2, ax3, ax4 = T.axis.remap("SSSSS", [i0, i1, i2, i3, i4])
                T.reads(T_multiply[ax0, ax1, ax2, ax3, ax4], compile_engine_const_1[()])
                T.writes(T_add_1[ax0, ax1, ax2, ax3, ax4])
                T_add_1[ax0, ax1, ax2, ax3, ax4] = T_multiply[ax0, ax1, ax2, ax3, ax4] + compile_engine_const_1[()]
        for i0, i1, i2, i3, i4 in T.grid(1, 128, 7, 7, 16):
            with T.block("T_floor"):
                ax0, ax1, ax2, ax3, ax4 = T.axis.remap("SSSSS", [i0, i1, i2, i3, i4])
                T.reads(T_add_1[ax0, ax1, ax2, ax3, ax4])
                T.writes(T_floor[ax0, ax1, ax2, ax3, ax4])
                T_floor[ax0, ax1, ax2, ax3, ax4] = T.floor(T_add_1[ax0, ax1, ax2, ax3, ax4], dtype="float32")
        for i0, i1, i2, i3, i4 in T.grid(1, 128, 7, 7, 16):
            with T.block("T_cast_1"):
                ax0, ax1, ax2, ax3, ax4 = T.axis.remap("SSSSS", [i0, i1, i2, i3, i4])
                T.reads(T_floor[ax0, ax1, ax2, ax3, ax4])
                T.writes(T_cast_1[ax0, ax1, ax2, ax3, ax4])
                T_cast_1[ax0, ax1, ax2, ax3, ax4] = T.cast(T_floor[ax0, ax1, ax2, ax3, ax4], "int32")
        for i0, i1, i2, i3, i4 in T.grid(1, 128, 7, 7, 16):
            with T.block("compute"):
                i0_1, i1_1, i2_1, i3_1, i4_1 = T.axis.remap("SSSSS", [i0, i1, i2, i3, i4])
                T.reads(T_cast_1[i0_1, i1_1, i2_1, i3_1, i4_1])
                T.writes(compute_1[i0_1, i1_1, i2_1, i3_1, i4_1])
                compute_1[i0_1, i1_1, i2_1, i3_1, i4_1] = T.max(T.min(T_cast_1[i0_1, i1_1, i2_1, i3_1, i4_1], 255), 0)
        for i0, i1, i2, i3, i4 in T.grid(1, 128, 7, 7, 16):
            with T.block("T_cast_2"):
                ax0, ax1, ax2, ax3, ax4 = T.axis.remap("SSSSS", [i0, i1, i2, i3, i4])
                T.reads(compute_1[ax0, ax1, ax2, ax3, ax4])
                T.writes(T_cast_2[ax0, ax1, ax2, ax3, ax4])
                T_cast_2[ax0, ax1, ax2, ax3, ax4] = T.cast(compute_1[ax0, ax1, ax2, ax3, ax4], "uint8")
        for i0, i1, i2, i3, i4 in T.grid(1, 128, 7, 7, 16):
            with T.block("T_cast_3"):
                ax0, ax1, ax2, ax3, ax4 = T.axis.remap("SSSSS", [i0, i1, i2, i3, i4])
                T.reads(T_cast_2[ax0, ax1, ax2, ax3, ax4])
                T.writes(T_cast_3[ax0, ax1, ax2, ax3, ax4])
                T_cast_3[ax0, ax1, ax2, ax3, ax4] = T.cast(T_cast_2[ax0, ax1, ax2, ax3, ax4], "float32")
        for i0, i1, i2, i3, i4 in T.grid(1, 128, 7, 7, 16):
            with T.block("T_subtract"):
                ax0, ax1, ax2, ax3, ax4 = T.axis.remap("SSSSS", [i0, i1, i2, i3, i4])
                T.reads(T_cast_3[ax0, ax1, ax2, ax3, ax4], p4[0])
                T.writes(T_subtract[ax0, ax1, ax2, ax3, ax4])
                T_subtract[ax0, ax1, ax2, ax3, ax4] = T_cast_3[ax0, ax1, ax2, ax3, ax4] - p4[0]
        for i0, i1, i2, i3, i4 in T.grid(1, 128, 7, 7, 16):
            with T.block("T_multiply_1"):
                ax0, ax1, ax2, ax3, ax4 = T.axis.remap("SSSSS", [i0, i1, i2, i3, i4])
                T.reads(compile_engine_const[()], T_subtract[ax0, ax1, ax2, ax3, ax4])
                T.writes(T_multiply_1[ax0, ax1, ax2, ax3, ax4])
                T_multiply_1[ax0, ax1, ax2, ax3, ax4] = compile_engine_const[()] * T_subtract[ax0, ax1, ax2, ax3, ax4]
        with T.block("compile_engine_const_2"):
            vi = T.axis.spatial(1, 0)
            T.reads()
            T.writes(compile_engine_const_2[()])
            compile_engine_const_2[()] = T.float32(0.5)
        for i0, i1, i2, i3, i4 in T.grid(1, 128, 7, 7, 16):
            with T.block("T_add_2"):
                ax0, ax1, ax2, ax3, ax4 = T.axis.remap("SSSSS", [i0, i1, i2, i3, i4])
                T.reads(T_multiply_1[ax0, ax1, ax2, ax3, ax4], compile_engine_const_2[()])
                T.writes(T_add_2[ax0, ax1, ax2, ax3, ax4])
                T_add_2[ax0, ax1, ax2, ax3, ax4] = T_multiply_1[ax0, ax1, ax2, ax3, ax4] + compile_engine_const_2[()]
        for i0, i1, i2, i3, i4 in T.grid(1, 128, 7, 7, 16):
            with T.block("T_floor_1"):
                ax0, ax1, ax2, ax3, ax4 = T.axis.remap("SSSSS", [i0, i1, i2, i3, i4])
                T.reads(T_add_2[ax0, ax1, ax2, ax3, ax4])
                T.writes(T_floor_1[ax0, ax1, ax2, ax3, ax4])
                T_floor_1[ax0, ax1, ax2, ax3, ax4] = T.floor(T_add_2[ax0, ax1, ax2, ax3, ax4], dtype="float32")
        for i0, i1, i2, i3, i4 in T.grid(1, 128, 7, 7, 16):
            with T.block("T_cast_4"):
                ax0, ax1, ax2, ax3, ax4 = T.axis.remap("SSSSS", [i0, i1, i2, i3, i4])
                T.reads(T_floor_1[ax0, ax1, ax2, ax3, ax4])
                T.writes(T_cast_4[ax0, ax1, ax2, ax3, ax4])
                T_cast_4[ax0, ax1, ax2, ax3, ax4] = T.cast(T_floor_1[ax0, ax1, ax2, ax3, ax4], "int32")
        for i0, i1, i2, i3, i4 in T.grid(1, 128, 7, 7, 16):
            with T.block("T_add_3"):
                ax0, ax1, ax2, ax3, ax4 = T.axis.remap("SSSSS", [i0, i1, i2, i3, i4])
                T.reads(T_cast_4[ax0, ax1, ax2, ax3, ax4], p5[ax0, ax1, ax2, ax3, ax4])
                T.writes(T_add_3[ax0, ax1, ax2, ax3, ax4])
                T_add_3[ax0, ax1, ax2, ax3, ax4] = T_cast_4[ax0, ax1, ax2, ax3, ax4] + p5[ax0, ax1, ax2, ax3, ax4]
        for i0, i1, i2, i3, i4 in T.grid(1, 128, 7, 7, 16):
            with T.block("compute_1"):
                i0_2, i1_2, i2_2, i3_2, i4_2 = T.axis.remap("SSSSS", [i0, i1, i2, i3, i4])
                T.reads(T_add_3[i0_2, i1_2, i2_2, i3_2, i4_2])
                T.writes(compute_2[i0_2, i1_2, i2_2, i3_2, i4_2])
                compute_2[i0_2, i1_2, i2_2, i3_2, i4_2] = T.max(T.min(T_add_3[i0_2, i1_2, i2_2, i3_2, i4_2], 255), 0)
        for i0_3, i1_3, i2_3, i3_3, i4_3 in T.grid(1, 128, 7, 7, 16):
            with T.block("T_cast_5"):
                ax0, ax1, ax2, ax3, ax4 = T.axis.remap("SSSSS", [i0_3, i1_3, i2_3, i3_3, i4_3])
                T.reads(compute_2[ax0, ax1, ax2, ax3, ax4])
                T.writes(T_cast_5[ax0, ax1, ax2, ax3, ax4])
                T_cast_5[ax0, ax1, ax2, ax3, ax4] = T.cast(compute_2[ax0, ax1, ax2, ax3, ax4], "uint8")
        for i0_4, i1_4, i2_4, i3_4, i4_4 in T.grid(1, 128, 7, 7, 16):
            with T.block("compute_2"):
                i0_5, i1_5, i2_5, i3_5, i4_5 = T.axis.remap("SSSSS", [i0_4, i1_4, i2_4, i3_4, i4_4])
                T.reads(T_cast_5[i0_5, i1_5, i2_5, i3_5, i4_5])
                T.writes(compute[i0_5, i1_5, i2_5, i3_5, i4_5])
                compute[i0_5, i1_5, i2_5, i3_5, i4_5] = T.max(T.min(T_cast_5[i0_5, i1_5, i2_5, i3_5, i4_5], T.uint8(255)), T.uint8(0))


@tvm.script.ir_module
class Conv2dInt8_NCHWc_target:
    @T.prim_func
    def main(p0: T.Buffer((1, 32, 7, 7, 16), "uint8"), p1: T.Buffer((128, 32, 1, 1, 4, 16, 4), "int8"), p2: T.Buffer((1, 128, 1, 1, 16), "int32"), p3: T.Buffer((1, 128, 1, 1, 16), "float32"), p4: T.Buffer(1, "float32"), p5: T.Buffer((1, 128, 7, 7, 16), "uint8"), T_cast: T.Buffer((1, 128, 7, 7, 16), "int32")) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # body
        # with T.block("root")
        compile_engine_const = T.alloc_buffer([], dtype="float32")
        conv2d_NCHWc_int8 = T.alloc_buffer([1, 128, 7, 7, 16], dtype="int32")
        T_add = T.alloc_buffer([1, 128, 7, 7, 16], dtype="int32")
        T_cast_1 = T.alloc_buffer([1, 128, 7, 7, 16], dtype="float32")
        T_multiply = T.alloc_buffer([1, 128, 7, 7, 16], dtype="float32")
        compile_engine_const_1 = T.alloc_buffer([], dtype="float32")
        T_add_1 = T.alloc_buffer([1, 128, 7, 7, 16], dtype="float32")
        T_floor = T.alloc_buffer([1, 128, 7, 7, 16], dtype="float32")
        T_cast_2 = T.alloc_buffer([1, 128, 7, 7, 16], dtype="int32")
        compute = T.alloc_buffer([1, 128, 7, 7, 16], dtype="int32")
        T_cast_3 = T.alloc_buffer([1, 128, 7, 7, 16], dtype="uint8")
        T_cast_4 = T.alloc_buffer([1, 128, 7, 7, 16], dtype="float32")
        T_subtract = T.alloc_buffer([1, 128, 7, 7, 16], dtype="float32")
        T_multiply_1 = T.alloc_buffer([1, 128, 7, 7, 16], dtype="float32")
        compile_engine_const_2 = T.alloc_buffer([], dtype="float32")
        T_add_2 = T.alloc_buffer([1, 128, 7, 7, 16], dtype="float32")
        T_floor_1 = T.alloc_buffer([1, 128, 7, 7, 16], dtype="float32")
        T_cast_5 = T.alloc_buffer([1, 128, 7, 7, 16], dtype="int32")
        compile_engine_const_3 = T.alloc_buffer([], dtype="float32")
        T_cast_6 = T.alloc_buffer([1, 128, 7, 7, 16], dtype="float32")
        T_multiply_2 = T.alloc_buffer([1, 128, 7, 7, 16], dtype="float32")
        compile_engine_const_4 = T.alloc_buffer([], dtype="float32")
        T_add_3 = T.alloc_buffer([1, 128, 7, 7, 16], dtype="float32")
        T_floor_2 = T.alloc_buffer([1, 128, 7, 7, 16], dtype="float32")
        T_cast_7 = T.alloc_buffer([1, 128, 7, 7, 16], dtype="int32")
        T_add_4 = T.alloc_buffer([1, 128, 7, 7, 16], dtype="int32")
        compute_1 = T.alloc_buffer([1, 128, 7, 7, 16], dtype="int32")
        T_cast_8 = T.alloc_buffer([1, 128, 7, 7, 16], dtype="uint8")
        compute_2 = T.alloc_buffer([1, 128, 7, 7, 16], dtype="uint8")
        with T.block("compile_engine_const"):
            vi = T.axis.spatial(1, 0)
            T.reads()
            T.writes(compile_engine_const[()])
            compile_engine_const[()] = T.float32(0.95489668846130371)
        for i0, i1, i2, i3, i4, i5, i6, i7, i8, i9 in T.grid(1, 128, 7, 7, 16, 1, 1, 32, 4, 4):
            with T.block("conv2d_NCHWc_int8"):
                n, oc_chunk, oh, ow, oc_block, kh, kw, ic_outer, ic_f_inner, ic_s_inner = T.axis.remap("SSSSSRRRRR", [i0, i1, i2, i3, i4, i5, i6, i7, i8, i9])
                T.reads(p0[n, ic_outer, oh + kh, ow + kw, ic_f_inner * 4 + ic_s_inner], p1[oc_chunk, ic_outer, kh, kw, ic_f_inner, oc_block, ic_s_inner])
                T.writes(conv2d_NCHWc_int8[n, oc_chunk, oh, ow, oc_block])
                T.block_attr({"schedule_rule":"meta_schedule.conv2d_NCHWc_int8", "workload":["conv2d_NCHWc_int8.x86", ["TENSOR", [1, 32, 7, 7, 16], "uint8"], ["TENSOR", [128, 32, 1, 1, 4, 16, 4], "int8"], [1, 1], [0, 0, 0, 0], [1, 1], "NCHW16c", "NCHW16c", "int32"]})
                with T.init():
                    conv2d_NCHWc_int8[n, oc_chunk, oh, ow, oc_block] = 0
                conv2d_NCHWc_int8[n, oc_chunk, oh, ow, oc_block] = conv2d_NCHWc_int8[n, oc_chunk, oh, ow, oc_block] + T.cast(p0[n, ic_outer, oh + kh, ow + kw, ic_f_inner * 4 + ic_s_inner], "int32") * T.cast(p1[oc_chunk, ic_outer, kh, kw, ic_f_inner, oc_block, ic_s_inner], "int32")
        for i0, i1, i2, i3, i4 in T.grid(1, 128, 7, 7, 16):
            with T.block("T_add"):
                ax0, ax1, ax2, ax3, ax4 = T.axis.remap("SSSSS", [i0, i1, i2, i3, i4])
                T.reads(conv2d_NCHWc_int8[ax0, ax1, ax2, ax3, ax4], p2[ax0, ax1, 0, 0, ax4])
                T.writes(T_add[ax0, ax1, ax2, ax3, ax4])
                T_add[ax0, ax1, ax2, ax3, ax4] = conv2d_NCHWc_int8[ax0, ax1, ax2, ax3, ax4] + p2[ax0, ax1, 0, 0, ax4]
        for i0, i1, i2, i3, i4 in T.grid(1, 128, 7, 7, 16):
            with T.block("T_cast"):
                ax0, ax1, ax2, ax3, ax4 = T.axis.remap("SSSSS", [i0, i1, i2, i3, i4])
                T.reads(T_add[ax0, ax1, ax2, ax3, ax4])
                T.writes(T_cast_1[ax0, ax1, ax2, ax3, ax4])
                T_cast_1[ax0, ax1, ax2, ax3, ax4] = T.cast(T_add[ax0, ax1, ax2, ax3, ax4], "float32")
        for i0, i1, i2, i3, i4 in T.grid(1, 128, 7, 7, 16):
            with T.block("T_multiply"):
                ax0, ax1, ax2, ax3, ax4 = T.axis.remap("SSSSS", [i0, i1, i2, i3, i4])
                T.reads(T_cast_1[ax0, ax1, ax2, ax3, ax4], p3[ax0, ax1, 0, 0, ax4])
                T.writes(T_multiply[ax0, ax1, ax2, ax3, ax4])
                T_multiply[ax0, ax1, ax2, ax3, ax4] = T_cast_1[ax0, ax1, ax2, ax3, ax4] * p3[ax0, ax1, 0, 0, ax4]
        with T.block("compile_engine_const_1"):
            vi = T.axis.spatial(1, 0)
            T.reads()
            T.writes(compile_engine_const_1[()])
            compile_engine_const_1[()] = T.float32(65.5)
        for i0, i1, i2, i3, i4 in T.grid(1, 128, 7, 7, 16):
            with T.block("T_add_1"):
                ax0, ax1, ax2, ax3, ax4 = T.axis.remap("SSSSS", [i0, i1, i2, i3, i4])
                T.reads(T_multiply[ax0, ax1, ax2, ax3, ax4], compile_engine_const_1[()])
                T.writes(T_add_1[ax0, ax1, ax2, ax3, ax4])
                T_add_1[ax0, ax1, ax2, ax3, ax4] = T_multiply[ax0, ax1, ax2, ax3, ax4] + compile_engine_const_1[()]
        for i0, i1, i2, i3, i4 in T.grid(1, 128, 7, 7, 16):
            with T.block("T_floor"):
                ax0, ax1, ax2, ax3, ax4 = T.axis.remap("SSSSS", [i0, i1, i2, i3, i4])
                T.reads(T_add_1[ax0, ax1, ax2, ax3, ax4])
                T.writes(T_floor[ax0, ax1, ax2, ax3, ax4])
                T_floor[ax0, ax1, ax2, ax3, ax4] = T.floor(T_add_1[ax0, ax1, ax2, ax3, ax4], dtype="float32")
        for i0, i1, i2, i3, i4 in T.grid(1, 128, 7, 7, 16):
            with T.block("T_cast_1"):
                ax0, ax1, ax2, ax3, ax4 = T.axis.remap("SSSSS", [i0, i1, i2, i3, i4])
                T.reads(T_floor[ax0, ax1, ax2, ax3, ax4])
                T.writes(T_cast_2[ax0, ax1, ax2, ax3, ax4])
                T_cast_2[ax0, ax1, ax2, ax3, ax4] = T.cast(T_floor[ax0, ax1, ax2, ax3, ax4], "int32")
        for i0, i1, i2, i3, i4 in T.grid(1, 128, 7, 7, 16):
            with T.block("compute"):
                i0_1, i1_1, i2_1, i3_1, i4_1 = T.axis.remap("SSSSS", [i0, i1, i2, i3, i4])
                T.reads(T_cast_2[i0_1, i1_1, i2_1, i3_1, i4_1])
                T.writes(compute[i0_1, i1_1, i2_1, i3_1, i4_1])
                compute[i0_1, i1_1, i2_1, i3_1, i4_1] = T.max(T.min(T_cast_2[i0_1, i1_1, i2_1, i3_1, i4_1], 255), 0)
        for i0, i1, i2, i3, i4 in T.grid(1, 128, 7, 7, 16):
            with T.block("T_cast_2"):
                ax0, ax1, ax2, ax3, ax4 = T.axis.remap("SSSSS", [i0, i1, i2, i3, i4])
                T.reads(compute[ax0, ax1, ax2, ax3, ax4])
                T.writes(T_cast_3[ax0, ax1, ax2, ax3, ax4])
                T_cast_3[ax0, ax1, ax2, ax3, ax4] = T.cast(compute[ax0, ax1, ax2, ax3, ax4], "uint8")
        for i0, i1, i2, i3, i4 in T.grid(1, 128, 7, 7, 16):
            with T.block("T_cast_3"):
                ax0, ax1, ax2, ax3, ax4 = T.axis.remap("SSSSS", [i0, i1, i2, i3, i4])
                T.reads(T_cast_3[ax0, ax1, ax2, ax3, ax4])
                T.writes(T_cast_4[ax0, ax1, ax2, ax3, ax4])
                T_cast_4[ax0, ax1, ax2, ax3, ax4] = T.cast(T_cast_3[ax0, ax1, ax2, ax3, ax4], "float32")
        for i0, i1, i2, i3, i4 in T.grid(1, 128, 7, 7, 16):
            with T.block("T_subtract"):
                ax0, ax1, ax2, ax3, ax4 = T.axis.remap("SSSSS", [i0, i1, i2, i3, i4])
                T.reads(T_cast_4[ax0, ax1, ax2, ax3, ax4], p4[0])
                T.writes(T_subtract[ax0, ax1, ax2, ax3, ax4])
                T_subtract[ax0, ax1, ax2, ax3, ax4] = T_cast_4[ax0, ax1, ax2, ax3, ax4] - p4[0]
        for i0, i1, i2, i3, i4 in T.grid(1, 128, 7, 7, 16):
            with T.block("T_multiply_1"):
                ax0, ax1, ax2, ax3, ax4 = T.axis.remap("SSSSS", [i0, i1, i2, i3, i4])
                T.reads(compile_engine_const[()], T_subtract[ax0, ax1, ax2, ax3, ax4])
                T.writes(T_multiply_1[ax0, ax1, ax2, ax3, ax4])
                T_multiply_1[ax0, ax1, ax2, ax3, ax4] = compile_engine_const[()] * T_subtract[ax0, ax1, ax2, ax3, ax4]
        with T.block("compile_engine_const_2"):
            vi = T.axis.spatial(1, 0)
            T.reads()
            T.writes(compile_engine_const_2[()])
            compile_engine_const_2[()] = T.float32(0.5)
        for i0, i1, i2, i3, i4 in T.grid(1, 128, 7, 7, 16):
            with T.block("T_add_2"):
                ax0, ax1, ax2, ax3, ax4 = T.axis.remap("SSSSS", [i0, i1, i2, i3, i4])
                T.reads(T_multiply_1[ax0, ax1, ax2, ax3, ax4], compile_engine_const_2[()])
                T.writes(T_add_2[ax0, ax1, ax2, ax3, ax4])
                T_add_2[ax0, ax1, ax2, ax3, ax4] = T_multiply_1[ax0, ax1, ax2, ax3, ax4] + compile_engine_const_2[()]
        for i0, i1, i2, i3, i4 in T.grid(1, 128, 7, 7, 16):
            with T.block("T_floor_1"):
                ax0, ax1, ax2, ax3, ax4 = T.axis.remap("SSSSS", [i0, i1, i2, i3, i4])
                T.reads(T_add_2[ax0, ax1, ax2, ax3, ax4])
                T.writes(T_floor_1[ax0, ax1, ax2, ax3, ax4])
                T_floor_1[ax0, ax1, ax2, ax3, ax4] = T.floor(T_add_2[ax0, ax1, ax2, ax3, ax4], dtype="float32")
        for i0, i1, i2, i3, i4 in T.grid(1, 128, 7, 7, 16):
            with T.block("T_cast_4"):
                ax0, ax1, ax2, ax3, ax4 = T.axis.remap("SSSSS", [i0, i1, i2, i3, i4])
                T.reads(T_floor_1[ax0, ax1, ax2, ax3, ax4])
                T.writes(T_cast_5[ax0, ax1, ax2, ax3, ax4])
                T_cast_5[ax0, ax1, ax2, ax3, ax4] = T.cast(T_floor_1[ax0, ax1, ax2, ax3, ax4], "int32")
        with T.block("compile_engine_const_3"):
            vi = T.axis.spatial(1, 0)
            T.reads()
            T.writes(compile_engine_const_3[()])
            compile_engine_const_3[()] = T.float32(0.71245479583740234)
        for i0, i1, i2, i3, i4 in T.grid(1, 128, 7, 7, 16):
            with T.block("T_cast_5"):
                ax0, ax1, ax2, ax3, ax4 = T.axis.remap("SSSSS", [i0, i1, i2, i3, i4])
                T.reads(p5[ax0, ax1, ax2, ax3, ax4])
                T.writes(T_cast_6[ax0, ax1, ax2, ax3, ax4])
                T_cast_6[ax0, ax1, ax2, ax3, ax4] = T.cast(p5[ax0, ax1, ax2, ax3, ax4], "float32")
        for i0, i1, i2, i3, i4 in T.grid(1, 128, 7, 7, 16):
            with T.block("T_multiply_2"):
                ax0, ax1, ax2, ax3, ax4 = T.axis.remap("SSSSS", [i0, i1, i2, i3, i4])
                T.reads(compile_engine_const_3[()], T_cast_6[ax0, ax1, ax2, ax3, ax4])
                T.writes(T_multiply_2[ax0, ax1, ax2, ax3, ax4])
                T_multiply_2[ax0, ax1, ax2, ax3, ax4] = compile_engine_const_3[()] * T_cast_6[ax0, ax1, ax2, ax3, ax4]
        with T.block("compile_engine_const_4"):
            vi = T.axis.spatial(1, 0)
            T.reads()
            T.writes(compile_engine_const_4[()])
            compile_engine_const_4[()] = T.float32(0.5)
        for i0, i1, i2, i3, i4 in T.grid(1, 128, 7, 7, 16):
            with T.block("T_add_3"):
                ax0, ax1, ax2, ax3, ax4 = T.axis.remap("SSSSS", [i0, i1, i2, i3, i4])
                T.reads(T_multiply_2[ax0, ax1, ax2, ax3, ax4], compile_engine_const_4[()])
                T.writes(T_add_3[ax0, ax1, ax2, ax3, ax4])
                T_add_3[ax0, ax1, ax2, ax3, ax4] = T_multiply_2[ax0, ax1, ax2, ax3, ax4] + compile_engine_const_4[()]
        for i0, i1, i2, i3, i4 in T.grid(1, 128, 7, 7, 16):
            with T.block("T_floor_2"):
                ax0, ax1, ax2, ax3, ax4 = T.axis.remap("SSSSS", [i0, i1, i2, i3, i4])
                T.reads(T_add_3[ax0, ax1, ax2, ax3, ax4])
                T.writes(T_floor_2[ax0, ax1, ax2, ax3, ax4])
                T_floor_2[ax0, ax1, ax2, ax3, ax4] = T.floor(T_add_3[ax0, ax1, ax2, ax3, ax4], dtype="float32")
        for i0, i1, i2, i3, i4 in T.grid(1, 128, 7, 7, 16):
            with T.block("T_cast_6"):
                ax0, ax1, ax2, ax3, ax4 = T.axis.remap("SSSSS", [i0, i1, i2, i3, i4])
                T.reads(T_floor_2[ax0, ax1, ax2, ax3, ax4])
                T.writes(T_cast_7[ax0, ax1, ax2, ax3, ax4])
                T_cast_7[ax0, ax1, ax2, ax3, ax4] = T.cast(T_floor_2[ax0, ax1, ax2, ax3, ax4], "int32")
        for i0, i1, i2, i3, i4 in T.grid(1, 128, 7, 7, 16):
            with T.block("T_add_4"):
                ax0, ax1, ax2, ax3, ax4 = T.axis.remap("SSSSS", [i0, i1, i2, i3, i4])
                T.reads(T_cast_5[ax0, ax1, ax2, ax3, ax4], T_cast_7[ax0, ax1, ax2, ax3, ax4])
                T.writes(T_add_4[ax0, ax1, ax2, ax3, ax4])
                T_add_4[ax0, ax1, ax2, ax3, ax4] = T_cast_5[ax0, ax1, ax2, ax3, ax4] + T_cast_7[ax0, ax1, ax2, ax3, ax4]
        for i0, i1, i2, i3, i4 in T.grid(1, 128, 7, 7, 16):
            with T.block("compute_1"):
                i0_2, i1_2, i2_2, i3_2, i4_2 = T.axis.remap("SSSSS", [i0, i1, i2, i3, i4])
                T.reads(T_add_4[i0_2, i1_2, i2_2, i3_2, i4_2])
                T.writes(compute_1[i0_2, i1_2, i2_2, i3_2, i4_2])
                compute_1[i0_2, i1_2, i2_2, i3_2, i4_2] = T.max(T.min(T_add_4[i0_2, i1_2, i2_2, i3_2, i4_2], 255), 0)
        for i0_3, i1_3, i2_3, i3_3, i4_3 in T.grid(1, 128, 7, 7, 16):
            with T.block("T_cast_7"):
                ax0, ax1, ax2, ax3, ax4 = T.axis.remap("SSSSS", [i0_3, i1_3, i2_3, i3_3, i4_3])
                T.reads(compute_1[ax0, ax1, ax2, ax3, ax4])
                T.writes(T_cast_8[ax0, ax1, ax2, ax3, ax4])
                T_cast_8[ax0, ax1, ax2, ax3, ax4] = T.cast(compute_1[ax0, ax1, ax2, ax3, ax4], "uint8")
        for i0_4, i1_4, i2_4, i3_4, i4_4 in T.grid(1, 128, 7, 7, 16):
            with T.block("compute_2"):
                i0_5, i1_5, i2_5, i3_5, i4_5 = T.axis.remap("SSSSS", [i0_4, i1_4, i2_4, i3_4, i4_4])
                T.reads(T_cast_8[i0_5, i1_5, i2_5, i3_5, i4_5])
                T.writes(compute_2[i0_5, i1_5, i2_5, i3_5, i4_5])
                compute_2[i0_5, i1_5, i2_5, i3_5, i4_5] = T.max(T.min(T_cast_8[i0_5, i1_5, i2_5, i3_5, i4_5], T.uint8(255)), T.uint8(0))
        for i0_6, i1_6, i2_6, i3_6, i4_6 in T.grid(1, 128, 7, 7, 16):
            with T.block("T_cast_8"):
                ax0, ax1, ax2, ax3, ax4 = T.axis.remap("SSSSS", [i0_6, i1_6, i2_6, i3_6, i4_6])
                T.reads(compute_2[ax0, ax1, ax2, ax3, ax4])
                T.writes(T_cast[ax0, ax1, ax2, ax3, ax4])
                T_cast[ax0, ax1, ax2, ax3, ax4] = T.cast(compute_2[ax0, ax1, ax2, ax3, ax4], "int32")


def get_conv2d_vnni_mod(intrin_id):
    @tvm.script.ir_module
    class Conv2dInt8_NCHWc_scheduled:
        @T.prim_func
        def main(p0: T.Buffer((1, 32, 7, 7, 16), "uint8"), p1: T.Buffer((128, 32, 1, 1, 4, 16, 4), "int8"), p2: T.Buffer((1, 128, 1, 1, 16), "int32"), p3: T.Buffer((1, 128, 1, 1, 16), "float32"), p4: T.Buffer(1, "float32"), p5: T.Buffer((1, 128, 7, 7, 16), "uint8"), T_cast: T.Buffer((1, 128, 7, 7, 16), "int32")) -> None:
            # function attr dict
            T.func_attr({"global_symbol": "main", "tir.noalias": True})
            # body
            # with T.block("root")
            conv2d_NCHWc_int8 = T.alloc_buffer([1, 128, 7, 7, 16], dtype="int32")
            for i0_0_i1_0_i2_0_i3_0_i4_0_0_i0_1_i1_1_fused in T.parallel(128, annotations={"pragma_auto_unroll_max_step":64, "pragma_unroll_explicit":1}):
                for i2_1, i3_1, i4_0_1 in T.grid(7, 1, 1):
                    for i0_2_init, i1_2_init, i2_2_init, i3_2_init, i4_0_2_init, i0_3_init, i1_3_init, i2_3_init, i3_3_init, i4_0_3_init in T.grid(1, 1, 1, 1, 1, 1, 1, 1, 7, 1):
                        with T.block("conv2d_NCHWc_int8_o_init"):
                            n = T.axis.spatial(1, i0_2_init + i0_3_init)
                            oc_chunk = T.axis.spatial(128, i0_0_i1_0_i2_0_i3_0_i4_0_0_i0_1_i1_1_fused + i1_2_init + i1_3_init)
                            oh = T.axis.spatial(7, i2_1 + i2_2_init + i2_3_init)
                            ow = T.axis.spatial(7, i3_1 * 7 + i3_2_init * 7 + i3_3_init)
                            oc_block_o = T.axis.spatial(1, i4_0_1 + i4_0_2_init + i4_0_3_init)
                            T.reads()
                            T.writes(conv2d_NCHWc_int8[n, oc_chunk, oh, ow, 0 : 16])
                            for i4_1 in T.vectorized(16):
                                with T.block("conv2d_NCHWc_int8_init"):
                                    oc_block_i_init = T.axis.spatial(16, i4_1)
                                    T.reads()
                                    T.writes(conv2d_NCHWc_int8[n, oc_chunk, oh, ow, oc_block_i_init])
                                    conv2d_NCHWc_int8[n, oc_chunk, oh, ow, oc_block_i_init] = 0
                    for i5_0, i6_0, i7_0, i8_0, i9_0_0, i0_2, i1_2, i2_2, i3_2, i4_0_2, i5_1, i6_1, i7_1, i8_1, i9_0_1, i0_3, i1_3, i2_3, i3_3, i4_0_3 in T.grid(1, 1, 4, 4, 1, 1, 1, 1, 1, 1, 1, 1, 8, 1, 1, 1, 1, 1, 7, 1):
                        with T.block("conv2d_NCHWc_int8_o_update"):
                            n = T.axis.spatial(1, i0_2 + i0_3)
                            oc_chunk = T.axis.spatial(128,  i0_0_i1_0_i2_0_i3_0_i4_0_0_i0_1_i1_1_fused + i1_2 + i1_3)
                            oh = T.axis.spatial(7, i2_1 + i2_2 + i2_3)
                            ow = T.axis.spatial(7, i3_1 * 7 + i3_2 * 7 + i3_3)
                            oc_block_o = T.axis.spatial(1, i4_0_1 + i4_0_2 + i4_0_3)
                            kh = T.axis.reduce(1, i5_0 + i5_1)
                            kw = T.axis.reduce(1, i6_0 + i6_1)
                            ic_outer = T.axis.reduce(32, i7_0 * 8 + i7_1)
                            ic_f_inner = T.axis.reduce(4, i8_0 + i8_1)
                            ic_s_inner_o = T.axis.reduce(1, i9_0_0 + i9_0_1)
                            T.reads(conv2d_NCHWc_int8[n, oc_chunk, oh, ow, 0 : 16], p0[n, ic_outer, oh + kh, ow + kw, ic_f_inner * 4 : ic_f_inner * 4 + 4], p1[oc_chunk, ic_outer, kh, kw, ic_f_inner, 0 : 16, 0 : 4])
                            T.writes(conv2d_NCHWc_int8[n, oc_chunk, oh, ow, 0 : 16])
                            A = T.match_buffer(p0[n, ic_outer, oh + kh, ow + kw, ic_f_inner * 4 : ic_f_inner * 4 + 4], [4], dtype="uint8", offset_factor=1)
                            B = T.match_buffer(p1[oc_chunk, ic_outer, kh, kw, ic_f_inner, 0 : 16, 0 : 4], [16, 4], dtype="int8", offset_factor=1)
                            C = T.match_buffer(conv2d_NCHWc_int8[n, oc_chunk, oh, ow, 0 : 16], [16], dtype="int32", offset_factor=1)
                            A_u8x4: T.uint8x4 = A[0:4]
                            A_i32: T.int32 = T.reinterpret(A_u8x4, dtype="int32")
                            B_i8x64: T.int8x64 = B[0, 0:64]
                            B_i32x16: T.int32x16 = T.reinterpret(B_i8x64, dtype="int32x16")
                            C_i32x16: T.int32x16 = C[0:16]
                            C[0:16] = T.call_llvm_pure_intrin(T.uint32(intrin_id), T.uint32(3), C_i32x16, T.broadcast(A_i32, 16), B_i32x16, dtype="int32x16")
                    for ax0, ax1, ax2, ax3 in T.grid(1, 1, 1, 7):
                        for ax4_fused in T.vectorized(16):
                            with T.block("T_cast_8"):
                                ax0_1 = T.axis.spatial(1, ax0)
                                ax1_1 = T.axis.spatial(128, i0_0_i1_0_i2_0_i3_0_i4_0_0_i0_1_i1_1_fused + ax1)
                                ax2_1 = T.axis.spatial(7, i2_1 + ax2)
                                ax3_1, ax4 = T.axis.remap("SS", [ax3, ax4_fused])
                                T.reads(conv2d_NCHWc_int8[ax0_1, ax1_1, ax2_1, ax3_1, ax4], p2[ax0_1, ax1_1, 0, 0, ax4], p3[ax0_1, ax1_1, 0, 0, ax4], p4[0], p5[ax0_1, ax1_1, ax2_1, ax3_1, ax4])
                                T.writes(T_cast[ax0_1, ax1_1, ax2_1, ax3_1, ax4])
                                T_cast[ax0_1, ax1_1, ax2_1, ax3_1, ax4] = T.cast(T.max(T.min(T.cast(T.max(T.min(T.cast(T.floor(T.float32(0.95489668846130371) * (T.cast(T.cast(T.max(T.min(T.cast(T.floor(T.cast(conv2d_NCHWc_int8[ax0_1, ax1_1, ax2_1, ax3_1, ax4] + p2[ax0_1, ax1_1, 0, 0, ax4], "float32") * p3[ax0_1, ax1_1, 0, 0, ax4] + T.float32(65.5), dtype="float32"), "int32"), 255), 0), "uint8"), "float32") - p4[0]) + T.float32(0.5), dtype="float32"), "int32") + T.cast(T.floor(T.float32(0.71245479583740234) * T.cast(p5[ax0_1, ax1_1, ax2_1, ax3_1, ax4], "float32") + T.float32(0.5), dtype="float32"), "int32"), 255), 0), "uint8"), T.uint8(255)), T.uint8(0)), "int32")

    return Conv2dInt8_NCHWc_scheduled


@tvm.script.ir_module
class Conv2dWinogradAddRelu:
    @T.prim_func
    def main(p0: T.Buffer((1, 56, 56, 64), "float32"), p1: T.Buffer((6, 6, 64, 64), "float32"), p2: T.Buffer((1, 1, 1, 64), "float32"), T_relu: T.Buffer((1, 56, 56, 64), "float32")) -> None:
        # function attr dict
        T.func_attr({"layout_free_buffers": [1], "tir.noalias": True, "global_symbol": "main"})
        # body
        # with T.block("root")
        data_pad = T.alloc_buffer([1, 58, 58, 64], dtype="float32")
        input_tile = T.alloc_buffer([6, 6, 196, 64], dtype="float32")
        B = T.alloc_buffer([6, 6], dtype="float32")
        data_pack = T.alloc_buffer([6, 6, 196, 64], dtype="float32")
        bgemm = T.alloc_buffer([6, 6, 196, 64], dtype="float32")
        A = T.alloc_buffer([6, 4], dtype="float32")
        inverse = T.alloc_buffer([4, 4, 196, 64], dtype="float32")
        conv2d_winograd = T.alloc_buffer([1, 56, 56, 64], dtype="float32")
        T_add = T.alloc_buffer([1, 56, 56, 64], dtype="float32")
        for i0, i1, i2, i3 in T.grid(1, 58, 58, 64):
            with T.block("data_pad"):
                i0_1, i1_1, i2_1, i3_1 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(p0[i0_1, i1_1 - 1, i2_1 - 1, i3_1])
                T.writes(data_pad[i0_1, i1_1, i2_1, i3_1])
                T.block_attr({"schedule_rule":"None"})
                data_pad[i0_1, i1_1, i2_1, i3_1] = T.if_then_else(1 <= i1_1 and i1_1 < 57 and 1 <= i2_1 and i2_1 < 57, p0[i0_1, i1_1 - 1, i2_1 - 1, i3_1], T.float32(0), dtype="float32")
        for i0, i1, i2, i3 in T.grid(6, 6, 196, 64):
            with T.block("input_tile"):
                eps, nu, p, ci = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(data_pad[p // 196, p % 196 // 14 * 4 + eps, p % 14 * 4 + nu, ci])
                T.writes(input_tile[eps, nu, p, ci])
                T.block_attr({"schedule_rule":"None"})
                input_tile[eps, nu, p, ci] = data_pad[p // 196, p % 196 // 14 * 4 + eps, p % 14 * 4 + nu, ci]
        for i0, i1 in T.grid(6, 6):
            with T.block("B"):
                i, j = T.axis.remap("SS", [i0, i1])
                T.reads()
                T.writes(B[i, j])
                T.block_attr({"const_matrix":True, "schedule_rule":"meta_schedule.compute_inline"})
                B[i, j] = T.Select(i % 6 == 5 and j % 6 == 5, T.float32(1), T.Select(i % 6 == 5 and j % 6 == 4, T.float32(0), T.Select(i % 6 == 5 and j % 6 == 3, T.float32(0), T.Select(i % 6 == 5 and j % 6 == 2, T.float32(0), T.Select(i % 6 == 5 and j % 6 == 1, T.float32(0), T.Select(i % 6 == 5 and j % 6 == 0, T.float32(0), T.Select(i % 6 == 4 and j % 6 == 5, T.float32(1.5), T.Select(i % 6 == 4 and j % 6 == 4, T.float32(1), T.Select(i % 6 == 4 and j % 6 == 3, T.float32(1), T.Select(i % 6 == 4 and j % 6 == 2, T.float32(1), T.Select(i % 6 == 4 and j % 6 == 1, T.float32(1), T.Select(i % 6 == 4 and j % 6 == 0, T.float32(1), T.Select(i % 6 == 3 and j % 6 == 5, T.float32(-2), T.Select(i % 6 == 3 and j % 6 == 4, T.float32(-0.5), T.Select(i % 6 == 3 and j % 6 == 3, T.float32(2), T.Select(i % 6 == 3 and j % 6 == 2, T.float32(2.5), T.Select(i % 6 == 3 and j % 6 == 1, T.float32(0.5), T.Select(i % 6 == 3 and j % 6 == 0, T.float32(1.5), T.Select(i % 6 == 2 and j % 6 == 5, T.float32(-1.5), T.Select(i % 6 == 2 and j % 6 == 4, T.float32(-1), T.Select(i % 6 == 2 and j % 6 == 3, T.float32(-1), T.Select(i % 6 == 2 and j % 6 == 2, T.float32(0.5), T.Select(i % 6 == 2 and j % 6 == 1, T.float32(-2.5), T.Select(i % 6 == 2 and j % 6 == 0, T.float32(-2), T.Select(i % 6 == 1 and j % 6 == 5, T.float32(1), T.Select(i % 6 == 1 and j % 6 == 4, T.float32(0.5), T.Select(i % 6 == 1 and j % 6 == 3, T.float32(-2), T.Select(i % 6 == 1 and j % 6 == 2, T.float32(-1), T.Select(i % 6 == 1 and j % 6 == 1, T.float32(1), T.Select(i % 6 == 1 and j % 6 == 0, T.float32(-1.5), T.Select(i % 6 == 0 and j % 6 == 5, T.float32(0), T.Select(i % 6 == 0 and j % 6 == 4, T.float32(0), T.Select(i % 6 == 0 and j % 6 == 3, T.float32(0), T.Select(i % 6 == 0 and j % 6 == 2, T.float32(0), T.Select(i % 6 == 0 and j % 6 == 1, T.float32(0), T.Select(i % 6 == 0 and j % 6 == 0, T.float32(1), T.float32(0)))))))))))))))))))))))))))))))))))))
        for i0, i1, i2, i3, i4, i5 in T.grid(6, 6, 196, 64, 6, 6):
            with T.block("data_pack"):
                eps, nu, p, ci, r_a, r_b = T.axis.remap("SSSSRR", [i0, i1, i2, i3, i4, i5])
                T.reads(input_tile[r_a, r_b, p, ci], B[T.min(r_a, r_b) : T.max(r_a, r_b) + 1, T.min(eps, nu) : T.max(eps, nu) + 1])
                T.writes(data_pack[eps, nu, p, ci])
                T.block_attr({"auto_scheduler_simplify_const_tensor_indices":["eps", "nu", "r_a", "r_b"], "schedule_rule":"meta_schedule.winograd_data_pack.cuda"})
                with T.init():
                    data_pack[eps, nu, p, ci] = T.float32(0)
                data_pack[eps, nu, p, ci] = data_pack[eps, nu, p, ci] + input_tile[r_a, r_b, p, ci] * B[r_a, eps] * B[r_b, nu]
        for i0, i1, i2, i3, i4 in T.grid(6, 6, 196, 64, 64):
            with T.block("bgemm"):
                eps, nu, p, co, ci = T.axis.remap("SSSSR", [i0, i1, i2, i3, i4])
                T.reads(data_pack[eps, nu, p, ci], p1[eps, nu, co, ci])
                T.writes(bgemm[eps, nu, p, co])
                T.block_attr({"layout_free_placeholders":[]})
                with T.init():
                    bgemm[eps, nu, p, co] = T.float32(0)
                bgemm[eps, nu, p, co] = bgemm[eps, nu, p, co] + data_pack[eps, nu, p, ci] * p1[eps, nu, co, ci]
        for i0, i1 in T.grid(6, 4):
            with T.block("A"):
                i, j = T.axis.remap("SS", [i0, i1])
                T.reads()
                T.writes(A[i, j])
                T.block_attr({"const_matrix":True, "schedule_rule":"meta_schedule.compute_inline"})
                A[i, j] = T.Select(i % 6 == 5 and j % 4 == 3, T.float32(1), T.Select(i % 6 == 5 and j % 4 == 2, T.float32(0), T.Select(i % 6 == 5 and j % 4 == 1, T.float32(0), T.Select(i % 6 == 5 and j % 4 == 0, T.float32(0), T.Select(i % 6 == 4 and j % 4 == 3, T.float32(-8), T.Select(i % 6 == 4 and j % 4 == 2, T.float32(4), T.Select(i % 6 == 4 and j % 4 == 1, T.float32(-2), T.Select(i % 6 == 4 and j % 4 == 0, T.float32(1), T.Select(i % 6 == 3 and j % 4 == 3, T.float32(0.125), T.Select(i % 6 == 3 and j % 4 == 2, T.float32(0.25), T.Select(i % 6 == 3 and j % 4 == 1, T.float32(0.5), T.Select(i % 6 == 3 and j % 4 == 0, T.float32(1), T.Select(i % 6 == 2 and j % 4 == 3, T.float32(1), T.Select(i % 6 == 2 and j % 4 == 2, T.float32(1), T.Select(i % 6 == 2 and j % 4 == 1, T.float32(1), T.Select(i % 6 == 2 and j % 4 == 0, T.float32(1), T.Select(i % 6 == 1 and j % 4 == 3, T.float32(-1), T.Select(i % 6 == 1 and j % 4 == 2, T.float32(1), T.Select(i % 6 == 1 and j % 4 == 1, T.float32(-1), T.Select(i % 6 == 1 and j % 4 == 0, T.float32(1), T.Select(i % 6 == 0 and j % 4 == 3, T.float32(0), T.Select(i % 6 == 0 and j % 4 == 2, T.float32(0), T.Select(i % 6 == 0 and j % 4 == 1, T.float32(0), T.Select(i % 6 == 0 and j % 4 == 0, T.float32(1), T.float32(0)))))))))))))))))))))))))
        for i0, i1, i2, i3, i4, i5 in T.grid(4, 4, 196, 64, 6, 6):
            with T.block("inverse"):
                vh, vw, p, co, r_a, r_b = T.axis.remap("SSSSRR", [i0, i1, i2, i3, i4, i5])
                T.reads(bgemm[r_a, r_b, p, co], A[T.min(r_a, r_b) : T.max(r_a, r_b) + 1, T.min(vh, vw) : T.max(vh, vw) + 1])
                T.writes(inverse[vh, vw, p, co])
                T.block_attr({"auto_scheduler_simplify_const_tensor_indices":["vh", "vw", "r_a", "r_b"], "schedule_rule":"meta_schedule.winograd_inverse.cuda"})
                with T.init():
                    inverse[vh, vw, p, co] = T.float32(0)
                inverse[vh, vw, p, co] = inverse[vh, vw, p, co] + bgemm[r_a, r_b, p, co] * A[r_a, vh] * A[r_b, vw]
        for i0, i1, i2, i3 in T.grid(1, 56, 56, 64):
            with T.block("conv2d_winograd"):
                n, h, w, co = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(inverse[h % 4, w % 4, n * 196 + h // 4 * 14 + w // 4, co])
                T.writes(conv2d_winograd[n, h, w, co])
                conv2d_winograd[n, h, w, co] = inverse[h % 4, w % 4, n * 196 + h // 4 * 14 + w // 4, co]
        for i0, i1, i2, i3 in T.grid(1, 56, 56, 64):
            with T.block("T_add"):
                ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(conv2d_winograd[ax0, ax1, ax2, ax3], p2[ax0, 0, 0, ax3])
                T.writes(T_add[ax0, ax1, ax2, ax3])
                T_add[ax0, ax1, ax2, ax3] = conv2d_winograd[ax0, ax1, ax2, ax3] + p2[ax0, 0, 0, ax3]
        for i0, i1, i2, i3 in T.grid(1, 56, 56, 64):
            with T.block("T_relu"):
                ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(T_add[ax0, ax1, ax2, ax3])
                T.writes(T_relu[ax0, ax1, ax2, ax3])
                T_relu[ax0, ax1, ax2, ax3] = T.max(T_add[ax0, ax1, ax2, ax3], T.float32(0))


@tvm.script.ir_module
class Conv2dWinogradAddResidualRelu:
    @T.prim_func
    def main(p0: T.Buffer((1, 56, 56, 64), "float32"), p1: T.Buffer((6, 6, 64, 64), "float32"), p2: T.Buffer((1, 1, 1, 64), "float32"), p3: T.Buffer((1, 56, 56, 64), "float32"), T_relu: T.Buffer((1, 56, 56, 64), "float32")) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True, "layout_free_buffers": [1]})
        # body
        # with T.block("root")
        data_pad = T.alloc_buffer([1, 58, 58, 64], dtype="float32")
        input_tile = T.alloc_buffer([6, 6, 196, 64], dtype="float32")
        B = T.alloc_buffer([6, 6], dtype="float32")
        data_pack = T.alloc_buffer([6, 6, 196, 64], dtype="float32")
        bgemm = T.alloc_buffer([6, 6, 196, 64], dtype="float32")
        A = T.alloc_buffer([6, 4], dtype="float32")
        inverse = T.alloc_buffer([4, 4, 196, 64], dtype="float32")
        conv2d_winograd = T.alloc_buffer([1, 56, 56, 64], dtype="float32")
        T_add = T.alloc_buffer([1, 56, 56, 64], dtype="float32")
        T_add_1 = T.alloc_buffer([1, 56, 56, 64], dtype="float32")
        for i0, i1, i2, i3 in T.grid(1, 58, 58, 64):
            with T.block("data_pad"):
                i0_1, i1_1, i2_1, i3_1 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(p0[i0_1, i1_1 - 1, i2_1 - 1, i3_1])
                T.writes(data_pad[i0_1, i1_1, i2_1, i3_1])
                T.block_attr({"schedule_rule":"None"})
                data_pad[i0_1, i1_1, i2_1, i3_1] = T.if_then_else(1 <= i1_1 and i1_1 < 57 and 1 <= i2_1 and i2_1 < 57, p0[i0_1, i1_1 - 1, i2_1 - 1, i3_1], T.float32(0), dtype="float32")
        for i0, i1, i2, i3 in T.grid(6, 6, 196, 64):
            with T.block("input_tile"):
                eps, nu, p, ci = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(data_pad[p // 196, p % 196 // 14 * 4 + eps, p % 14 * 4 + nu, ci])
                T.writes(input_tile[eps, nu, p, ci])
                T.block_attr({"schedule_rule":"None"})
                input_tile[eps, nu, p, ci] = data_pad[p // 196, p % 196 // 14 * 4 + eps, p % 14 * 4 + nu, ci]
        for i0, i1 in T.grid(6, 6):
            with T.block("B"):
                i, j = T.axis.remap("SS", [i0, i1])
                T.reads()
                T.writes(B[i, j])
                T.block_attr({"const_matrix":True, "schedule_rule":"meta_schedule.compute_inline"})
                B[i, j] = T.Select(i % 6 == 5 and j % 6 == 5, T.float32(1), T.Select(i % 6 == 5 and j % 6 == 4, T.float32(0), T.Select(i % 6 == 5 and j % 6 == 3, T.float32(0), T.Select(i % 6 == 5 and j % 6 == 2, T.float32(0), T.Select(i % 6 == 5 and j % 6 == 1, T.float32(0), T.Select(i % 6 == 5 and j % 6 == 0, T.float32(0), T.Select(i % 6 == 4 and j % 6 == 5, T.float32(1.5), T.Select(i % 6 == 4 and j % 6 == 4, T.float32(1), T.Select(i % 6 == 4 and j % 6 == 3, T.float32(1), T.Select(i % 6 == 4 and j % 6 == 2, T.float32(1), T.Select(i % 6 == 4 and j % 6 == 1, T.float32(1), T.Select(i % 6 == 4 and j % 6 == 0, T.float32(1), T.Select(i % 6 == 3 and j % 6 == 5, T.float32(-2), T.Select(i % 6 == 3 and j % 6 == 4, T.float32(-0.5), T.Select(i % 6 == 3 and j % 6 == 3, T.float32(2), T.Select(i % 6 == 3 and j % 6 == 2, T.float32(2.5), T.Select(i % 6 == 3 and j % 6 == 1, T.float32(0.5), T.Select(i % 6 == 3 and j % 6 == 0, T.float32(1.5), T.Select(i % 6 == 2 and j % 6 == 5, T.float32(-1.5), T.Select(i % 6 == 2 and j % 6 == 4, T.float32(-1), T.Select(i % 6 == 2 and j % 6 == 3, T.float32(-1), T.Select(i % 6 == 2 and j % 6 == 2, T.float32(0.5), T.Select(i % 6 == 2 and j % 6 == 1, T.float32(-2.5), T.Select(i % 6 == 2 and j % 6 == 0, T.float32(-2), T.Select(i % 6 == 1 and j % 6 == 5, T.float32(1), T.Select(i % 6 == 1 and j % 6 == 4, T.float32(0.5), T.Select(i % 6 == 1 and j % 6 == 3, T.float32(-2), T.Select(i % 6 == 1 and j % 6 == 2, T.float32(-1), T.Select(i % 6 == 1 and j % 6 == 1, T.float32(1), T.Select(i % 6 == 1 and j % 6 == 0, T.float32(-1.5), T.Select(i % 6 == 0 and j % 6 == 5, T.float32(0), T.Select(i % 6 == 0 and j % 6 == 4, T.float32(0), T.Select(i % 6 == 0 and j % 6 == 3, T.float32(0), T.Select(i % 6 == 0 and j % 6 == 2, T.float32(0), T.Select(i % 6 == 0 and j % 6 == 1, T.float32(0), T.Select(i % 6 == 0 and j % 6 == 0, T.float32(1), T.float32(0)))))))))))))))))))))))))))))))))))))
        for i0, i1, i2, i3, i4, i5 in T.grid(6, 6, 196, 64, 6, 6):
            with T.block("data_pack"):
                eps, nu, p, ci, r_a, r_b = T.axis.remap("SSSSRR", [i0, i1, i2, i3, i4, i5])
                T.reads(input_tile[r_a, r_b, p, ci], B[T.min(r_a, r_b) : T.max(r_a, r_b) + 1, T.min(eps, nu) : T.max(eps, nu) + 1])
                T.writes(data_pack[eps, nu, p, ci])
                T.block_attr({"auto_scheduler_simplify_const_tensor_indices":["eps", "nu", "r_a", "r_b"], "schedule_rule":"meta_schedule.winograd_data_pack.cuda"})
                with T.init():
                    data_pack[eps, nu, p, ci] = T.float32(0)
                data_pack[eps, nu, p, ci] = data_pack[eps, nu, p, ci] + input_tile[r_a, r_b, p, ci] * B[r_a, eps] * B[r_b, nu]
        for i0, i1, i2, i3, i4 in T.grid(6, 6, 196, 64, 64):
            with T.block("bgemm"):
                eps, nu, p, co, ci = T.axis.remap("SSSSR", [i0, i1, i2, i3, i4])
                T.reads(data_pack[eps, nu, p, ci], p1[eps, nu, co, ci])
                T.writes(bgemm[eps, nu, p, co])
                T.block_attr({"layout_free_placeholders":[]})
                with T.init():
                    bgemm[eps, nu, p, co] = T.float32(0)
                bgemm[eps, nu, p, co] = bgemm[eps, nu, p, co] + data_pack[eps, nu, p, ci] * p1[eps, nu, co, ci]
        for i0, i1 in T.grid(6, 4):
            with T.block("A"):
                i, j = T.axis.remap("SS", [i0, i1])
                T.reads()
                T.writes(A[i, j])
                T.block_attr({"const_matrix":True, "schedule_rule":"meta_schedule.compute_inline"})
                A[i, j] = T.Select(i % 6 == 5 and j % 4 == 3, T.float32(1), T.Select(i % 6 == 5 and j % 4 == 2, T.float32(0), T.Select(i % 6 == 5 and j % 4 == 1, T.float32(0), T.Select(i % 6 == 5 and j % 4 == 0, T.float32(0), T.Select(i % 6 == 4 and j % 4 == 3, T.float32(-8), T.Select(i % 6 == 4 and j % 4 == 2, T.float32(4), T.Select(i % 6 == 4 and j % 4 == 1, T.float32(-2), T.Select(i % 6 == 4 and j % 4 == 0, T.float32(1), T.Select(i % 6 == 3 and j % 4 == 3, T.float32(0.125), T.Select(i % 6 == 3 and j % 4 == 2, T.float32(0.25), T.Select(i % 6 == 3 and j % 4 == 1, T.float32(0.5), T.Select(i % 6 == 3 and j % 4 == 0, T.float32(1), T.Select(i % 6 == 2 and j % 4 == 3, T.float32(1), T.Select(i % 6 == 2 and j % 4 == 2, T.float32(1), T.Select(i % 6 == 2 and j % 4 == 1, T.float32(1), T.Select(i % 6 == 2 and j % 4 == 0, T.float32(1), T.Select(i % 6 == 1 and j % 4 == 3, T.float32(-1), T.Select(i % 6 == 1 and j % 4 == 2, T.float32(1), T.Select(i % 6 == 1 and j % 4 == 1, T.float32(-1), T.Select(i % 6 == 1 and j % 4 == 0, T.float32(1), T.Select(i % 6 == 0 and j % 4 == 3, T.float32(0), T.Select(i % 6 == 0 and j % 4 == 2, T.float32(0), T.Select(i % 6 == 0 and j % 4 == 1, T.float32(0), T.Select(i % 6 == 0 and j % 4 == 0, T.float32(1), T.float32(0)))))))))))))))))))))))))
        for i0, i1, i2, i3, i4, i5 in T.grid(4, 4, 196, 64, 6, 6):
            with T.block("inverse"):
                vh, vw, p, co, r_a, r_b = T.axis.remap("SSSSRR", [i0, i1, i2, i3, i4, i5])
                T.reads(bgemm[r_a, r_b, p, co], A[T.min(r_a, r_b) : T.max(r_a, r_b) + 1, T.min(vh, vw) : T.max(vh, vw) + 1])
                T.writes(inverse[vh, vw, p, co])
                T.block_attr({"auto_scheduler_simplify_const_tensor_indices":["vh", "vw", "r_a", "r_b"], "schedule_rule":"meta_schedule.winograd_inverse.cuda"})
                with T.init():
                    inverse[vh, vw, p, co] = T.float32(0)
                inverse[vh, vw, p, co] = inverse[vh, vw, p, co] + bgemm[r_a, r_b, p, co] * A[r_a, vh] * A[r_b, vw]
        for i0, i1, i2, i3 in T.grid(1, 56, 56, 64):
            with T.block("conv2d_winograd"):
                n, h, w, co = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(inverse[h % 4, w % 4, n * 196 + h // 4 * 14 + w // 4, co])
                T.writes(conv2d_winograd[n, h, w, co])
                conv2d_winograd[n, h, w, co] = inverse[h % 4, w % 4, n * 196 + h // 4 * 14 + w // 4, co]
        for i0, i1, i2, i3 in T.grid(1, 56, 56, 64):
            with T.block("T_add"):
                ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(conv2d_winograd[ax0, ax1, ax2, ax3], p2[ax0, 0, 0, ax3])
                T.writes(T_add[ax0, ax1, ax2, ax3])
                T_add[ax0, ax1, ax2, ax3] = conv2d_winograd[ax0, ax1, ax2, ax3] + p2[ax0, 0, 0, ax3]
        for i0, i1, i2, i3 in T.grid(1, 56, 56, 64):
            with T.block("T_add_1"):
                ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(T_add[ax0, ax1, ax2, ax3], p3[ax0, ax1, ax2, ax3])
                T.writes(T_add_1[ax0, ax1, ax2, ax3])
                T_add_1[ax0, ax1, ax2, ax3] = T_add[ax0, ax1, ax2, ax3] + p3[ax0, ax1, ax2, ax3]
        for i0, i1, i2, i3 in T.grid(1, 56, 56, 64):
            with T.block("T_relu"):
                ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(T_add_1[ax0, ax1, ax2, ax3])
                T.writes(T_relu[ax0, ax1, ax2, ax3])
                T_relu[ax0, ax1, ax2, ax3] = T.max(T_add_1[ax0, ax1, ax2, ax3], T.float32(0))


@tvm.script.ir_module
class Conv2dWinogradAddResidualRelu_scheduled:
    @T.prim_func
    def main(p0: T.Buffer((1, 56, 56, 64), "float32"), p1: T.Buffer((6, 6, 64, 64), "float32"), p2: T.Buffer((1, 1, 1, 64), "float32"), p3: T.Buffer((1, 56, 56, 64), "float32"), T_relu: T.Buffer((1, 56, 56, 64), "float32")) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True, "layout_free_buffers": [1]})
        # body
        # with T.block("root")
        input_tile_local = T.alloc_buffer([6, 6, 196, 64], dtype="float32", scope="local")
        data_pack = T.alloc_buffer([6, 6, 196, 64], dtype="float32")
        bgemm = T.alloc_buffer([6, 6, 196, 64], dtype="float32")
        inverse = T.alloc_buffer([4, 4, 196, 64], dtype="float32")
        bgemm_local = T.alloc_buffer([6, 6, 196, 64], dtype="float32", scope="local")
        data_pack_shared = T.alloc_buffer([6, 6, 196, 64], dtype="float32", scope="shared")
        p1_shared = T.alloc_buffer([6, 6, 64, 64], dtype="float32", scope="shared")
        for i2_0_i3_0_i2_1_i3_1_fused_0 in T.thread_binding(98, thread="blockIdx.x", annotations={"pragma_auto_unroll_max_step":1024, "pragma_unroll_explicit":1}):
            for i2_0_i3_0_i2_1_i3_1_fused_1 in T.thread_binding(128, thread="threadIdx.x"):
                for ax0, ax1, ax2, ax3 in T.grid(6, 6, 1, 1):
                    with T.block("input_tile"):
                        eps, nu = T.axis.remap("SS", [ax0, ax1])
                        p = T.axis.spatial(196, (i2_0_i3_0_i2_1_i3_1_fused_0 * 128 + i2_0_i3_0_i2_1_i3_1_fused_1) // 896 * 14 + (i2_0_i3_0_i2_1_i3_1_fused_0 * 128 + i2_0_i3_0_i2_1_i3_1_fused_1) % 112 // 8 + ax2)
                        ci = T.axis.spatial(64, (i2_0_i3_0_i2_1_i3_1_fused_0 * 128 + i2_0_i3_0_i2_1_i3_1_fused_1) % 896 // 112 * 8 + (i2_0_i3_0_i2_1_i3_1_fused_0 * 128 + i2_0_i3_0_i2_1_i3_1_fused_1) % 8 + ax3)
                        T.reads(p0[p // 196, p % 196 // 14 * 4 + eps - 1, p % 14 * 4 + nu - 1, ci])
                        T.writes(input_tile_local[eps, nu, p, ci])
                        T.block_attr({"schedule_rule":"None"})
                        input_tile_local[eps, nu, p, ci] = T.if_then_else(1 <= p % 196 // 14 * 4 + eps and p % 196 // 14 * 4 + eps < 57 and 1 <= p % 14 * 4 + nu and p % 14 * 4 + nu < 57, p0[p // 196, p % 196 // 14 * 4 + eps - 1, p % 14 * 4 + nu - 1, ci], T.float32(0), dtype="float32")
                for i0 in T.unroll(6):
                    for i1 in T.unroll(6):
                        with T.block("data_pack_init"):
                            eps, nu = T.axis.remap("SS", [i0, i1])
                            p = T.axis.spatial(196, (i2_0_i3_0_i2_1_i3_1_fused_0 * 128 + i2_0_i3_0_i2_1_i3_1_fused_1) // 896 * 14 + (i2_0_i3_0_i2_1_i3_1_fused_0 * 128 + i2_0_i3_0_i2_1_i3_1_fused_1) % 112 // 8)
                            ci = T.axis.spatial(64, (i2_0_i3_0_i2_1_i3_1_fused_0 * 128 + i2_0_i3_0_i2_1_i3_1_fused_1) % 896 // 112 * 8 + (i2_0_i3_0_i2_1_i3_1_fused_0 * 128 + i2_0_i3_0_i2_1_i3_1_fused_1) % 8)
                            T.reads()
                            T.writes(data_pack[eps, nu, p, ci])
                            T.block_attr({"auto_scheduler_simplify_const_tensor_indices":["eps", "nu", "r_a", "r_b"], "schedule_rule":"meta_schedule.winograd_data_pack.cuda"})
                            data_pack[eps, nu, p, ci] = T.float32(0)
                        for i4 in T.unroll(6):
                            for i5 in T.unroll(6):
                                with T.block("data_pack_update"):
                                    eps, nu = T.axis.remap("SS", [i0, i1])
                                    p = T.axis.spatial(196, (i2_0_i3_0_i2_1_i3_1_fused_0 * 128 + i2_0_i3_0_i2_1_i3_1_fused_1) // 896 * 14 + (i2_0_i3_0_i2_1_i3_1_fused_0 * 128 + i2_0_i3_0_i2_1_i3_1_fused_1) % 112 // 8)
                                    ci = T.axis.spatial(64, (i2_0_i3_0_i2_1_i3_1_fused_0 * 128 + i2_0_i3_0_i2_1_i3_1_fused_1) % 896 // 112 * 8 + (i2_0_i3_0_i2_1_i3_1_fused_0 * 128 + i2_0_i3_0_i2_1_i3_1_fused_1) % 8)
                                    r_a, r_b = T.axis.remap("RR", [i4, i5])
                                    T.reads(data_pack[eps, nu, p, ci], input_tile_local[r_a, r_b, p, ci])
                                    T.writes(data_pack[eps, nu, p, ci])
                                    T.block_attr({"auto_scheduler_simplify_const_tensor_indices":["eps", "nu", "r_a", "r_b"], "schedule_rule":"meta_schedule.winograd_data_pack.cuda"})
                                    data_pack[eps, nu, p, ci] = data_pack[eps, nu, p, ci] + input_tile_local[r_a, r_b, p, ci] * T.Select(r_a % 6 == 5 and eps % 6 == 5, T.float32(1), T.Select(r_a % 6 == 5 and eps % 6 == 4, T.float32(0), T.Select(r_a % 6 == 5 and eps % 6 == 3, T.float32(0), T.Select(r_a % 6 == 5 and eps % 6 == 2, T.float32(0), T.Select(r_a % 6 == 5 and eps % 6 == 1, T.float32(0), T.Select(r_a % 6 == 5 and eps % 6 == 0, T.float32(0), T.Select(r_a % 6 == 4 and eps % 6 == 5, T.float32(1.5), T.Select(r_a % 6 == 4 and eps % 6 == 4, T.float32(1), T.Select(r_a % 6 == 4 and eps % 6 == 3, T.float32(1), T.Select(r_a % 6 == 4 and eps % 6 == 2, T.float32(1), T.Select(r_a % 6 == 4 and eps % 6 == 1, T.float32(1), T.Select(r_a % 6 == 4 and eps % 6 == 0, T.float32(1), T.Select(r_a % 6 == 3 and eps % 6 == 5, T.float32(-2), T.Select(r_a % 6 == 3 and eps % 6 == 4, T.float32(-0.5), T.Select(r_a % 6 == 3 and eps % 6 == 3, T.float32(2), T.Select(r_a % 6 == 3 and eps % 6 == 2, T.float32(2.5), T.Select(r_a % 6 == 3 and eps % 6 == 1, T.float32(0.5), T.Select(r_a % 6 == 3 and eps % 6 == 0, T.float32(1.5), T.Select(r_a % 6 == 2 and eps % 6 == 5, T.float32(-1.5), T.Select(r_a % 6 == 2 and eps % 6 == 4, T.float32(-1), T.Select(r_a % 6 == 2 and eps % 6 == 3, T.float32(-1), T.Select(r_a % 6 == 2 and eps % 6 == 2, T.float32(0.5), T.Select(r_a % 6 == 2 and eps % 6 == 1, T.float32(-2.5), T.Select(r_a % 6 == 2 and eps % 6 == 0, T.float32(-2), T.Select(r_a % 6 == 1 and eps % 6 == 5, T.float32(1), T.Select(r_a % 6 == 1 and eps % 6 == 4, T.float32(0.5), T.Select(r_a % 6 == 1 and eps % 6 == 3, T.float32(-2), T.Select(r_a % 6 == 1 and eps % 6 == 2, T.float32(-1), T.Select(r_a % 6 == 1 and eps % 6 == 1, T.float32(1), T.Select(r_a % 6 == 1 and eps % 6 == 0, T.float32(-1.5), T.Select(r_a % 6 == 0 and eps % 6 == 5, T.float32(0), T.Select(r_a % 6 == 0 and eps % 6 == 4, T.float32(0), T.Select(r_a % 6 == 0 and eps % 6 == 3, T.float32(0), T.Select(r_a % 6 == 0 and eps % 6 == 2, T.float32(0), T.Select(r_a % 6 == 0 and eps % 6 == 1, T.float32(0), T.Select(r_a % 6 == 0 and eps % 6 == 0, T.float32(1), T.float32(0))))))))))))))))))))))))))))))))))))) * T.Select(r_b % 6 == 5 and nu % 6 == 5, T.float32(1), T.Select(r_b % 6 == 5 and nu % 6 == 4, T.float32(0), T.Select(r_b % 6 == 5 and nu % 6 == 3, T.float32(0), T.Select(r_b % 6 == 5 and nu % 6 == 2, T.float32(0), T.Select(r_b % 6 == 5 and nu % 6 == 1, T.float32(0), T.Select(r_b % 6 == 5 and nu % 6 == 0, T.float32(0), T.Select(r_b % 6 == 4 and nu % 6 == 5, T.float32(1.5), T.Select(r_b % 6 == 4 and nu % 6 == 4, T.float32(1), T.Select(r_b % 6 == 4 and nu % 6 == 3, T.float32(1), T.Select(r_b % 6 == 4 and nu % 6 == 2, T.float32(1), T.Select(r_b % 6 == 4 and nu % 6 == 1, T.float32(1), T.Select(r_b % 6 == 4 and nu % 6 == 0, T.float32(1), T.Select(r_b % 6 == 3 and nu % 6 == 5, T.float32(-2), T.Select(r_b % 6 == 3 and nu % 6 == 4, T.float32(-0.5), T.Select(r_b % 6 == 3 and nu % 6 == 3, T.float32(2), T.Select(r_b % 6 == 3 and nu % 6 == 2, T.float32(2.5), T.Select(r_b % 6 == 3 and nu % 6 == 1, T.float32(0.5), T.Select(r_b % 6 == 3 and nu % 6 == 0, T.float32(1.5), T.Select(r_b % 6 == 2 and nu % 6 == 5, T.float32(-1.5), T.Select(r_b % 6 == 2 and nu % 6 == 4, T.float32(-1), T.Select(r_b % 6 == 2 and nu % 6 == 3, T.float32(-1), T.Select(r_b % 6 == 2 and nu % 6 == 2, T.float32(0.5), T.Select(r_b % 6 == 2 and nu % 6 == 1, T.float32(-2.5), T.Select(r_b % 6 == 2 and nu % 6 == 0, T.float32(-2), T.Select(r_b % 6 == 1 and nu % 6 == 5, T.float32(1), T.Select(r_b % 6 == 1 and nu % 6 == 4, T.float32(0.5), T.Select(r_b % 6 == 1 and nu % 6 == 3, T.float32(-2), T.Select(r_b % 6 == 1 and nu % 6 == 2, T.float32(-1), T.Select(r_b % 6 == 1 and nu % 6 == 1, T.float32(1), T.Select(r_b % 6 == 1 and nu % 6 == 0, T.float32(-1.5), T.Select(r_b % 6 == 0 and nu % 6 == 5, T.float32(0), T.Select(r_b % 6 == 0 and nu % 6 == 4, T.float32(0), T.Select(r_b % 6 == 0 and nu % 6 == 3, T.float32(0), T.Select(r_b % 6 == 0 and nu % 6 == 2, T.float32(0), T.Select(r_b % 6 == 0 and nu % 6 == 1, T.float32(0), T.Select(r_b % 6 == 0 and nu % 6 == 0, T.float32(1), T.float32(0)))))))))))))))))))))))))))))))))))))
        for i0_0_i1_0_i2_0_i3_0_fused in T.thread_binding(168, thread="blockIdx.x", annotations={"pragma_auto_unroll_max_step":1024, "pragma_unroll_explicit":1}):
            for i0_1_i1_1_i2_1_i3_1_fused in T.thread_binding(4, thread="vthread.x"):
                for i0_2_i1_2_i2_2_i3_2_fused in T.thread_binding(48, thread="threadIdx.x"):
                    for i0_3_init, i1_3_init, i2_3_init, i3_3_init, i0_4_init, i1_4_init, i2_4_init, i3_4_init in T.grid(1, 1, 14, 1, 1, 1, 1, 1):
                        with T.block("bgemm_init"):
                            eps = T.axis.spatial(6, i0_1_i1_1_i2_1_i3_1_fused // 2 * 3 + i0_2_i1_2_i2_2_i3_2_fused // 16 + i0_3_init + i0_4_init)
                            nu = T.axis.spatial(6, i0_0_i1_0_i2_0_i3_0_fused // 28 + i1_3_init + i1_4_init)
                            p = T.axis.spatial(196, i0_0_i1_0_i2_0_i3_0_fused % 28 // 4 * 28 + i0_1_i1_1_i2_1_i3_1_fused % 2 * 14 + i2_3_init + i2_4_init)
                            co = T.axis.spatial(64, i0_0_i1_0_i2_0_i3_0_fused % 4 * 16 + i0_2_i1_2_i2_2_i3_2_fused % 16 + i3_3_init + i3_4_init)
                            T.reads()
                            T.writes(bgemm_local[eps, nu, p, co])
                            T.block_attr({"layout_free_placeholders":[], "meta_schedule.thread_extent_high_inclusive":1024, "meta_schedule.thread_extent_low_inclusive":32, "meta_schedule.tiling_structure":"SSSRRSRS"})
                            bgemm_local[eps, nu, p, co] = T.float32(0)
                    for i4_0 in T.serial(2):
                        for ax0_ax1_ax2_ax3_fused_0 in T.serial(28):
                            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(48, thread="threadIdx.x"):
                                for ax0_ax1_ax2_ax3_fused_2 in T.vectorized(4):
                                    with T.block("data_pack_shared"):
                                        v0 = T.axis.spatial(6, (ax0_ax1_ax2_ax3_fused_0 * 192 + ax0_ax1_ax2_ax3_fused_1 * 4 + ax0_ax1_ax2_ax3_fused_2) // 896)
                                        v1 = T.axis.spatial(6, i0_0_i1_0_i2_0_i3_0_fused // 28)
                                        v2 = T.axis.spatial(196, i0_0_i1_0_i2_0_i3_0_fused % 28 // 4 * 28 + (ax0_ax1_ax2_ax3_fused_0 * 192 + ax0_ax1_ax2_ax3_fused_1 * 4 + ax0_ax1_ax2_ax3_fused_2) % 896 // 32)
                                        v3 = T.axis.spatial(64, i4_0 * 32 + (ax0_ax1_ax2_ax3_fused_0 * 192 + ax0_ax1_ax2_ax3_fused_1 * 4 + ax0_ax1_ax2_ax3_fused_2) % 32)
                                        T.reads(data_pack[v0, v1, v2, v3])
                                        T.writes(data_pack_shared[v0, v1, v2, v3])
                                        data_pack_shared[v0, v1, v2, v3] = data_pack[v0, v1, v2, v3]
                        for ax0_ax1_ax2_ax3_fused_0 in T.serial(16):
                            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(48, thread="threadIdx.x"):
                                for ax0_ax1_ax2_ax3_fused_2 in T.vectorized(4):
                                    with T.block("p1_shared"):
                                        v0 = T.axis.spatial(6, (ax0_ax1_ax2_ax3_fused_0 * 192 + ax0_ax1_ax2_ax3_fused_1 * 4 + ax0_ax1_ax2_ax3_fused_2) // 512)
                                        v1 = T.axis.spatial(6, i0_0_i1_0_i2_0_i3_0_fused // 28)
                                        v2 = T.axis.spatial(64, i0_0_i1_0_i2_0_i3_0_fused % 4 * 16 + (ax0_ax1_ax2_ax3_fused_0 * 192 + ax0_ax1_ax2_ax3_fused_1 * 4 + ax0_ax1_ax2_ax3_fused_2) % 512 // 32)
                                        v3 = T.axis.spatial(64, i4_0 * 32 + (ax0_ax1_ax2_ax3_fused_0 * 192 + ax0_ax1_ax2_ax3_fused_1 * 4 + ax0_ax1_ax2_ax3_fused_2) % 32)
                                        T.reads(p1[v0, v1, v2, v3])
                                        T.writes(p1_shared[v0, v1, v2, v3])
                                        p1_shared[v0, v1, v2, v3] = p1[v0, v1, v2, v3]
                        for i4_1, i0_3, i1_3, i2_3, i3_3, i4_2, i0_4, i1_4, i2_4, i3_4 in T.grid(2, 1, 1, 14, 1, 16, 1, 1, 1, 1):
                            with T.block("bgemm_update"):
                                eps = T.axis.spatial(6, i0_1_i1_1_i2_1_i3_1_fused // 2 * 3 + i0_2_i1_2_i2_2_i3_2_fused // 16 + i0_3 + i0_4)
                                nu = T.axis.spatial(6, i0_0_i1_0_i2_0_i3_0_fused // 28 + i1_3 + i1_4)
                                p = T.axis.spatial(196, i0_0_i1_0_i2_0_i3_0_fused % 28 // 4 * 28 + i0_1_i1_1_i2_1_i3_1_fused % 2 * 14 + i2_3 + i2_4)
                                co = T.axis.spatial(64, i0_0_i1_0_i2_0_i3_0_fused % 4 * 16 + i0_2_i1_2_i2_2_i3_2_fused % 16 + i3_3 + i3_4)
                                ci = T.axis.reduce(64, i4_0 * 32 + i4_1 * 16 + i4_2)
                                T.reads(bgemm_local[eps, nu, p, co], data_pack_shared[eps, nu, p, ci], p1_shared[eps, nu, co, ci])
                                T.writes(bgemm_local[eps, nu, p, co])
                                T.block_attr({"layout_free_placeholders":[], "meta_schedule.thread_extent_high_inclusive":1024, "meta_schedule.thread_extent_low_inclusive":32, "meta_schedule.tiling_structure":"SSSRRSRS"})
                                bgemm_local[eps, nu, p, co] = bgemm_local[eps, nu, p, co] + data_pack_shared[eps, nu, p, ci] * p1_shared[eps, nu, co, ci]
                    for ax0, ax1, ax2, ax3 in T.grid(1, 1, 14, 1):
                        with T.block("bgemm_local"):
                            v0 = T.axis.spatial(6, i0_1_i1_1_i2_1_i3_1_fused // 2 * 3 + i0_2_i1_2_i2_2_i3_2_fused // 16 + ax0)
                            v1 = T.axis.spatial(6, i0_0_i1_0_i2_0_i3_0_fused // 28 + ax1)
                            v2 = T.axis.spatial(196, i0_0_i1_0_i2_0_i3_0_fused % 28 // 4 * 28 + i0_1_i1_1_i2_1_i3_1_fused % 2 * 14 + ax2)
                            v3 = T.axis.spatial(64, i0_0_i1_0_i2_0_i3_0_fused % 4 * 16 + i0_2_i1_2_i2_2_i3_2_fused % 16 + ax3)
                            T.reads(bgemm_local[v0, v1, v2, v3])
                            T.writes(bgemm[v0, v1, v2, v3])
                            bgemm[v0, v1, v2, v3] = bgemm_local[v0, v1, v2, v3]
        for i2_0_i3_0_i2_1_i3_1_fused_0 in T.thread_binding(25, thread="blockIdx.x", annotations={"pragma_auto_unroll_max_step":1024, "pragma_unroll_explicit":1}):
            for i2_0_i3_0_i2_1_i3_1_fused_1 in T.thread_binding(512, thread="threadIdx.x"):
                for i0 in T.unroll(4):
                    for i1 in T.unroll(4):
                        with T.block("inverse_init"):
                            T.where(i2_0_i3_0_i2_1_i3_1_fused_0 * 512 + i2_0_i3_0_i2_1_i3_1_fused_1 < 12544)
                            vh, vw = T.axis.remap("SS", [i0, i1])
                            p = T.axis.spatial(196, (i2_0_i3_0_i2_1_i3_1_fused_0 * 512 + i2_0_i3_0_i2_1_i3_1_fused_1) // 448 * 7 + (i2_0_i3_0_i2_1_i3_1_fused_0 * 512 + i2_0_i3_0_i2_1_i3_1_fused_1) % 224 // 32)
                            co = T.axis.spatial(64, (i2_0_i3_0_i2_1_i3_1_fused_0 * 512 + i2_0_i3_0_i2_1_i3_1_fused_1) % 448 // 224 * 32 + (i2_0_i3_0_i2_1_i3_1_fused_0 * 512 + i2_0_i3_0_i2_1_i3_1_fused_1) % 32)
                            T.reads()
                            T.writes(inverse[vh, vw, p, co])
                            T.block_attr({"auto_scheduler_simplify_const_tensor_indices":["vh", "vw", "r_a", "r_b"], "schedule_rule":"meta_schedule.winograd_inverse.cuda"})
                            inverse[vh, vw, p, co] = T.float32(0)
                        for i4 in T.unroll(6):
                            for i5 in T.unroll(6):
                                with T.block("inverse_update"):
                                    T.where(i2_0_i3_0_i2_1_i3_1_fused_0 * 512 + i2_0_i3_0_i2_1_i3_1_fused_1 < 12544)
                                    vh, vw = T.axis.remap("SS", [i0, i1])
                                    p = T.axis.spatial(196, (i2_0_i3_0_i2_1_i3_1_fused_0 * 512 + i2_0_i3_0_i2_1_i3_1_fused_1) // 448 * 7 + (i2_0_i3_0_i2_1_i3_1_fused_0 * 512 + i2_0_i3_0_i2_1_i3_1_fused_1) % 224 // 32)
                                    co = T.axis.spatial(64, (i2_0_i3_0_i2_1_i3_1_fused_0 * 512 + i2_0_i3_0_i2_1_i3_1_fused_1) % 448 // 224 * 32 + (i2_0_i3_0_i2_1_i3_1_fused_0 * 512 + i2_0_i3_0_i2_1_i3_1_fused_1) % 32)
                                    r_a, r_b = T.axis.remap("RR", [i4, i5])
                                    T.reads(inverse[vh, vw, p, co], bgemm[r_a, r_b, p, co])
                                    T.writes(inverse[vh, vw, p, co])
                                    T.block_attr({"auto_scheduler_simplify_const_tensor_indices":["vh", "vw", "r_a", "r_b"], "schedule_rule":"meta_schedule.winograd_inverse.cuda"})
                                    inverse[vh, vw, p, co] = inverse[vh, vw, p, co] + bgemm[r_a, r_b, p, co] * T.Select(r_a % 6 == 5 and vh % 4 == 3, T.float32(1), T.Select(r_a % 6 == 5 and vh % 4 == 2, T.float32(0), T.Select(r_a % 6 == 5 and vh % 4 == 1, T.float32(0), T.Select(r_a % 6 == 5 and vh % 4 == 0, T.float32(0), T.Select(r_a % 6 == 4 and vh % 4 == 3, T.float32(-8), T.Select(r_a % 6 == 4 and vh % 4 == 2, T.float32(4), T.Select(r_a % 6 == 4 and vh % 4 == 1, T.float32(-2), T.Select(r_a % 6 == 4 and vh % 4 == 0, T.float32(1), T.Select(r_a % 6 == 3 and vh % 4 == 3, T.float32(0.125), T.Select(r_a % 6 == 3 and vh % 4 == 2, T.float32(0.25), T.Select(r_a % 6 == 3 and vh % 4 == 1, T.float32(0.5), T.Select(r_a % 6 == 3 and vh % 4 == 0, T.float32(1), T.Select(r_a % 6 == 2 and vh % 4 == 3, T.float32(1), T.Select(r_a % 6 == 2 and vh % 4 == 2, T.float32(1), T.Select(r_a % 6 == 2 and vh % 4 == 1, T.float32(1), T.Select(r_a % 6 == 2 and vh % 4 == 0, T.float32(1), T.Select(r_a % 6 == 1 and vh % 4 == 3, T.float32(-1), T.Select(r_a % 6 == 1 and vh % 4 == 2, T.float32(1), T.Select(r_a % 6 == 1 and vh % 4 == 1, T.float32(-1), T.Select(r_a % 6 == 1 and vh % 4 == 0, T.float32(1), T.Select(r_a % 6 == 0 and vh % 4 == 3, T.float32(0), T.Select(r_a % 6 == 0 and vh % 4 == 2, T.float32(0), T.Select(r_a % 6 == 0 and vh % 4 == 1, T.float32(0), T.Select(r_a % 6 == 0 and vh % 4 == 0, T.float32(1), T.float32(0))))))))))))))))))))))))) * T.Select(r_b % 6 == 5 and vw % 4 == 3, T.float32(1), T.Select(r_b % 6 == 5 and vw % 4 == 2, T.float32(0), T.Select(r_b % 6 == 5 and vw % 4 == 1, T.float32(0), T.Select(r_b % 6 == 5 and vw % 4 == 0, T.float32(0), T.Select(r_b % 6 == 4 and vw % 4 == 3, T.float32(-8), T.Select(r_b % 6 == 4 and vw % 4 == 2, T.float32(4), T.Select(r_b % 6 == 4 and vw % 4 == 1, T.float32(-2), T.Select(r_b % 6 == 4 and vw % 4 == 0, T.float32(1), T.Select(r_b % 6 == 3 and vw % 4 == 3, T.float32(0.125), T.Select(r_b % 6 == 3 and vw % 4 == 2, T.float32(0.25), T.Select(r_b % 6 == 3 and vw % 4 == 1, T.float32(0.5), T.Select(r_b % 6 == 3 and vw % 4 == 0, T.float32(1), T.Select(r_b % 6 == 2 and vw % 4 == 3, T.float32(1), T.Select(r_b % 6 == 2 and vw % 4 == 2, T.float32(1), T.Select(r_b % 6 == 2 and vw % 4 == 1, T.float32(1), T.Select(r_b % 6 == 2 and vw % 4 == 0, T.float32(1), T.Select(r_b % 6 == 1 and vw % 4 == 3, T.float32(-1), T.Select(r_b % 6 == 1 and vw % 4 == 2, T.float32(1), T.Select(r_b % 6 == 1 and vw % 4 == 1, T.float32(-1), T.Select(r_b % 6 == 1 and vw % 4 == 0, T.float32(1), T.Select(r_b % 6 == 0 and vw % 4 == 3, T.float32(0), T.Select(r_b % 6 == 0 and vw % 4 == 2, T.float32(0), T.Select(r_b % 6 == 0 and vw % 4 == 1, T.float32(0), T.Select(r_b % 6 == 0 and vw % 4 == 0, T.float32(1), T.float32(0)))))))))))))))))))))))))
        for i0_i1_i2_i3_fused_0 in T.thread_binding(1568, thread="blockIdx.x", annotations={"pragma_auto_unroll_max_step":1024, "pragma_unroll_explicit":1}):
            for i0_i1_i2_i3_fused_1 in T.thread_binding(128, thread="threadIdx.x"):
                with T.block("conv2d_winograd"):
                    n = T.axis.spatial(1, 0)
                    h = T.axis.spatial(56, (i0_i1_i2_i3_fused_0 * 128 + i0_i1_i2_i3_fused_1) // 3584)
                    w = T.axis.spatial(56, (i0_i1_i2_i3_fused_0 * 128 + i0_i1_i2_i3_fused_1) % 3584 // 64)
                    co = T.axis.spatial(64, (i0_i1_i2_i3_fused_0 * 128 + i0_i1_i2_i3_fused_1) % 64)
                    T.reads(inverse[h % 4, w % 4, n * 196 + h // 4 * 14 + w // 4, co], p2[n, 0, 0, co], p3[n, h, w, co])
                    T.writes(T_relu[n, h, w, co])
                    T_relu[n, h, w, co] = T.max(inverse[h % 4, w % 4, n * 196 + h // 4 * 14 + w // 4, co] + p2[n, 0, 0, co] + p3[n, h, w, co], T.float32(0))


@tvm.script.ir_module
class Conv2dInt8_with_predicate:
    @T.prim_func
    def main(p0: T.Buffer((16, 56, 56, 64), "int8"), p1: T.Buffer((256, 1, 1, 64), "int8"), p2: T.Buffer((1, 1, 1, 256), "int32"), p3: T.Buffer((1, 1, 1, 256), "int32"), p4: T.Buffer(256, "int32"), p5: T.Buffer(256, "int32"), p6: T.Buffer(256, "int32"), p7: T.Buffer((), "int32"), p8: T.Buffer(1, "int32"), compute: T.Buffer((16, 56, 56, 256), "int32")) -> None:
        # function attr dict
        T.func_attr({"tir.noalias": True, "global_symbol": "main"})
        # body
        # with T.block("root")
        pad_temp = T.alloc_buffer([16, 56, 56, 64], dtype="int8")
        conv2d_nhwc = T.alloc_buffer([16, 56, 56, 256], dtype="int32")
        T_subtract = T.alloc_buffer([16, 56, 56, 256], dtype="int32")
        T_add = T.alloc_buffer([16, 56, 56, 256], dtype="int32")
        compute_1 = T.alloc_buffer([16, 56, 56, 256], dtype="int32")
        T_add_1 = T.alloc_buffer([16, 56, 56, 256], dtype="int32")
        compute_2 = T.alloc_buffer([16, 56, 56, 256], dtype="int32")
        T_subtract_1 = T.alloc_buffer([16, 56, 56, 256], dtype="int32")
        for i0, i1, i2, i3 in T.grid(16, 56, 56, 64):
            with T.block("pad_temp"):
                i0_1, i1_1, i2_1, i3_1 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(p0[i0_1, i1_1, i2_1, i3_1])
                T.writes(pad_temp[i0_1, i1_1, i2_1, i3_1])
                pad_temp[i0_1, i1_1, i2_1, i3_1] = p0[i0_1, i1_1, i2_1, i3_1]
        for i0, i1, i2, i3, i4, i5, i6 in T.grid(16, 56, 56, 256, 1, 1, 64):
            with T.block("conv2d_nhwc"):
                nn, yy, xx, ff, ry, rx, rc = T.axis.remap("SSSSRRR", [i0, i1, i2, i3, i4, i5, i6])
                T.reads(pad_temp[nn, yy + ry, xx + rx, rc], p1[ff, ry, rx, rc])
                T.writes(conv2d_nhwc[nn, yy, xx, ff])
                with T.init():
                    conv2d_nhwc[nn, yy, xx, ff] = 0
                conv2d_nhwc[nn, yy, xx, ff] = conv2d_nhwc[nn, yy, xx, ff] + T.cast(pad_temp[nn, yy + ry, xx + rx, rc], "int32") * T.cast(p1[ff, ry, rx, rc], "int32")
        for i0, i1, i2, i3 in T.grid(16, 56, 56, 256):
            with T.block("T_subtract"):
                ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(conv2d_nhwc[ax0, ax1, ax2, ax3], p2[0, 0, 0, ax3])
                T.writes(T_subtract[ax0, ax1, ax2, ax3])
                T_subtract[ax0, ax1, ax2, ax3] = conv2d_nhwc[ax0, ax1, ax2, ax3] - p2[0, 0, 0, ax3]
        for i0, i1, i2, i3 in T.grid(16, 56, 56, 256):
            with T.block("T_add"):
                ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(T_subtract[ax0, ax1, ax2, ax3], p3[0, 0, 0, ax3])
                T.writes(T_add[ax0, ax1, ax2, ax3])
                T_add[ax0, ax1, ax2, ax3] = T_subtract[ax0, ax1, ax2, ax3] + p3[0, 0, 0, ax3]
        for i0, i1, i2, i3 in T.grid(16, 56, 56, 256):
            with T.block("compute"):
                i0_2, i1_2, i2_2, i3_2 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(T_add[i0_2, i1_2, i2_2, i3_2], p4[i3_2], p5[i3_2], p6[i3_2])
                T.writes(compute_1[i0_2, i1_2, i2_2, i3_2])
                compute_1[i0_2, i1_2, i2_2, i3_2] = T.q_multiply_shift_per_axis(T_add[i0_2, i1_2, i2_2, i3_2], p4[i3_2], p5[i3_2], p6[i3_2], 31, False, True, dtype="int32")
        for i0_3, i1_3, i2_3, i3_3 in T.grid(16, 56, 56, 256):
            with T.block("T_add_1"):
                ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0_3, i1_3, i2_3, i3_3])
                T.reads(p7[()], compute_1[ax0, ax1, ax2, ax3])
                T.writes(T_add_1[ax0, ax1, ax2, ax3])
                T_add_1[ax0, ax1, ax2, ax3] = p7[()] + compute_1[ax0, ax1, ax2, ax3]
        for i0_4, i1_4, i2_4, i3_4 in T.grid(16, 56, 56, 256):
            with T.block("compute_1"):
                i0_5, i1_5, i2_5, i3_5 = T.axis.remap("SSSS", [i0_4, i1_4, i2_4, i3_4])
                T.reads(T_add_1[i0_5, i1_5, i2_5, i3_5])
                T.writes(compute_2[i0_5, i1_5, i2_5, i3_5])
                compute_2[i0_5, i1_5, i2_5, i3_5] = T.max(T.min(T_add_1[i0_5, i1_5, i2_5, i3_5], 255), 0)
        for i0_6, i1_6, i2_6, i3_6 in T.grid(16, 56, 56, 256):
            with T.block("T_subtract_1"):
                ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0_6, i1_6, i2_6, i3_6])
                T.reads(compute_2[ax0, ax1, ax2, ax3], p8[0])
                T.writes(T_subtract_1[ax0, ax1, ax2, ax3])
                T_subtract_1[ax0, ax1, ax2, ax3] = compute_2[ax0, ax1, ax2, ax3] - p8[0]
        for i0_7, i1_7, i2_7, i3_7 in T.grid(16, 56, 56, 256):
            with T.block("compute_2"):
                i0_8, i1_8, i2_8, i3_8 = T.axis.remap("SSSS", [i0_7, i1_7, i2_7, i3_7])
                T.reads(T_subtract_1[i0_8, i1_8, i2_8, i3_8])
                T.writes(compute[i0_8, i1_8, i2_8, i3_8])
                compute[i0_8, i1_8, i2_8, i3_8] = T.q_multiply_shift(T_subtract_1[i0_8, i1_8, i2_8, i3_8], 1963325822, 31, 1, dtype="int32")


@tvm.script.ir_module
class Conv2dInt8_with_predicate_target:
    @T.prim_func
    def main(p0: T.Buffer((16, 56, 56, 64), "int8"), p1: T.Buffer((256, 1, 1, 64), "int8"), p2: T.Buffer((1, 1, 1, 256), "int32"), p3: T.Buffer((1, 1, 1, 256), "int32"), p4: T.Buffer(256, "int32"), p5: T.Buffer(256, "int32"), p6: T.Buffer(256, "int32"), p7: T.Buffer((), "int32"), p8: T.Buffer(1, "int32"), p9: T.Buffer((16, 56, 56, 256), "int32"), compute: T.Buffer((16, 56, 56, 256), "int32")) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # body
        # with T.block("root")
        pad_temp = T.alloc_buffer([16, 56, 56, 64], dtype="int8")
        conv2d_nhwc = T.alloc_buffer([16, 56, 56, 256], dtype="int32")
        T_subtract = T.alloc_buffer([16, 56, 56, 256], dtype="int32")
        T_add = T.alloc_buffer([16, 56, 56, 256], dtype="int32")
        compute_1 = T.alloc_buffer([16, 56, 56, 256], dtype="int32")
        T_add_1 = T.alloc_buffer([16, 56, 56, 256], dtype="int32")
        compute_2 = T.alloc_buffer([16, 56, 56, 256], dtype="int32")
        T_subtract_1 = T.alloc_buffer([16, 56, 56, 256], dtype="int32")
        compute_3 = T.alloc_buffer([16, 56, 56, 256], dtype="int32")
        compute_4 = T.alloc_buffer([16, 56, 56, 256], dtype="int32")
        T_add_2 = T.alloc_buffer([16, 56, 56, 256], dtype="int32")
        for i0, i1, i2, i3 in T.grid(16, 56, 56, 64):
            with T.block("pad_temp"):
                i0_1, i1_1, i2_1, i3_1 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(p0[i0_1, i1_1, i2_1, i3_1])
                T.writes(pad_temp[i0_1, i1_1, i2_1, i3_1])
                pad_temp[i0_1, i1_1, i2_1, i3_1] = p0[i0_1, i1_1, i2_1, i3_1]
        for i0, i1, i2, i3, i4, i5, i6 in T.grid(16, 56, 56, 256, 1, 1, 64):
            with T.block("conv2d_nhwc"):
                nn, yy, xx, ff, ry, rx, rc = T.axis.remap("SSSSRRR", [i0, i1, i2, i3, i4, i5, i6])
                T.reads(pad_temp[nn, yy + ry, xx + rx, rc], p1[ff, ry, rx, rc])
                T.writes(conv2d_nhwc[nn, yy, xx, ff])
                with T.init():
                    conv2d_nhwc[nn, yy, xx, ff] = 0
                conv2d_nhwc[nn, yy, xx, ff] = conv2d_nhwc[nn, yy, xx, ff] + T.cast(pad_temp[nn, yy + ry, xx + rx, rc], "int32") * T.cast(p1[ff, ry, rx, rc], "int32")
        for i0, i1, i2, i3 in T.grid(16, 56, 56, 256):
            with T.block("T_subtract"):
                ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(conv2d_nhwc[ax0, ax1, ax2, ax3], p2[0, 0, 0, ax3])
                T.writes(T_subtract[ax0, ax1, ax2, ax3])
                T_subtract[ax0, ax1, ax2, ax3] = conv2d_nhwc[ax0, ax1, ax2, ax3] - p2[0, 0, 0, ax3]
        for i0, i1, i2, i3 in T.grid(16, 56, 56, 256):
            with T.block("T_add"):
                ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(T_subtract[ax0, ax1, ax2, ax3], p3[0, 0, 0, ax3])
                T.writes(T_add[ax0, ax1, ax2, ax3])
                T_add[ax0, ax1, ax2, ax3] = T_subtract[ax0, ax1, ax2, ax3] + p3[0, 0, 0, ax3]
        for i0, i1, i2, i3 in T.grid(16, 56, 56, 256):
            with T.block("compute"):
                i0_2, i1_2, i2_2, i3_2 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(T_add[i0_2, i1_2, i2_2, i3_2], p4[i3_2], p5[i3_2], p6[i3_2])
                T.writes(compute_1[i0_2, i1_2, i2_2, i3_2])
                compute_1[i0_2, i1_2, i2_2, i3_2] = T.q_multiply_shift_per_axis(T_add[i0_2, i1_2, i2_2, i3_2], p4[i3_2], p5[i3_2], p6[i3_2], 31, False, True, dtype="int32")
        for i0_3, i1_3, i2_3, i3_3 in T.grid(16, 56, 56, 256):
            with T.block("T_add_1"):
                ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0_3, i1_3, i2_3, i3_3])
                T.reads(p7[()], compute_1[ax0, ax1, ax2, ax3])
                T.writes(T_add_1[ax0, ax1, ax2, ax3])
                T_add_1[ax0, ax1, ax2, ax3] = p7[()] + compute_1[ax0, ax1, ax2, ax3]
        for i0_4, i1_4, i2_4, i3_4 in T.grid(16, 56, 56, 256):
            with T.block("compute_1"):
                i0_5, i1_5, i2_5, i3_5 = T.axis.remap("SSSS", [i0_4, i1_4, i2_4, i3_4])
                T.reads(T_add_1[i0_5, i1_5, i2_5, i3_5])
                T.writes(compute_2[i0_5, i1_5, i2_5, i3_5])
                compute_2[i0_5, i1_5, i2_5, i3_5] = T.max(T.min(T_add_1[i0_5, i1_5, i2_5, i3_5], 255), 0)
        for i0_6, i1_6, i2_6, i3_6 in T.grid(16, 56, 56, 256):
            with T.block("T_subtract_1"):
                ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0_6, i1_6, i2_6, i3_6])
                T.reads(compute_2[ax0, ax1, ax2, ax3], p8[0])
                T.writes(T_subtract_1[ax0, ax1, ax2, ax3])
                T_subtract_1[ax0, ax1, ax2, ax3] = compute_2[ax0, ax1, ax2, ax3] - p8[0]
        for i0_7, i1_7, i2_7, i3_7 in T.grid(16, 56, 56, 256):
            with T.block("compute_2"):
                i0_8, i1_8, i2_8, i3_8 = T.axis.remap("SSSS", [i0_7, i1_7, i2_7, i3_7])
                T.reads(T_subtract_1[i0_8, i1_8, i2_8, i3_8])
                T.writes(compute_3[i0_8, i1_8, i2_8, i3_8])
                compute_3[i0_8, i1_8, i2_8, i3_8] = T.q_multiply_shift(T_subtract_1[i0_8, i1_8, i2_8, i3_8], 1457846997, 31, 0, dtype="int32")
        for i0_9, i1_9, i2_9, i3_9 in T.grid(16, 56, 56, 256):
            with T.block("compute_3"):
                i0_10, i1_10, i2_10, i3_10 = T.axis.remap("SSSS", [i0_9, i1_9, i2_9, i3_9])
                T.reads(p9[i0_10, i1_10, i2_10, i3_10])
                T.writes(compute_4[i0_10, i1_10, i2_10, i3_10])
                compute_4[i0_10, i1_10, i2_10, i3_10] = T.q_multiply_shift(p9[i0_10, i1_10, i2_10, i3_10], 2101000910, 31, 0, dtype="int32")
        for i0_11, i1_11, i2_11, i3_11 in T.grid(16, 56, 56, 256):
            with T.block("T_add_2"):
                ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0_11, i1_11, i2_11, i3_11])
                T.reads(compute_3[ax0, ax1, ax2, ax3], compute_4[ax0, ax1, ax2, ax3])
                T.writes(T_add_2[ax0, ax1, ax2, ax3])
                T_add_2[ax0, ax1, ax2, ax3] = compute_3[ax0, ax1, ax2, ax3] + compute_4[ax0, ax1, ax2, ax3]
        for i0_12, i1_12, i2_12, i3_12 in T.grid(16, 56, 56, 256):
            with T.block("compute_4"):
                i0_13, i1_13, i2_13, i3_13 = T.axis.remap("SSSS", [i0_12, i1_12, i2_12, i3_12])
                T.reads(T_add_2[i0_13, i1_13, i2_13, i3_13])
                T.writes(compute[i0_13, i1_13, i2_13, i3_13])
                compute[i0_13, i1_13, i2_13, i3_13] = T.max(T.min(T_add_2[i0_13, i1_13, i2_13, i3_13], 255), 0)


@tvm.script.ir_module
class Conv2dInt8_with_predicate_scheduled:
    @T.prim_func
    def main(p0: T.Buffer((16, 56, 56, 64), "int8"), p1: T.Buffer((256, 1, 1, 64), "int8"), p2: T.Buffer((1, 1, 1, 256), "int32"), p3: T.Buffer((1, 1, 1, 256), "int32"), p4: T.Buffer((256,), "int32"), p5: T.Buffer((256,), "int32"), p6: T.Buffer((256,), "int32"), p7: T.Buffer((), "int32"), p8: T.Buffer((1,), "int32"), p9: T.Buffer((16, 56, 56, 256), "int32"), compute: T.Buffer((16, 56, 56, 256), "int32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        with T.block("root"):
            T.reads()
            T.writes()
            T.block_attr({"meta_schedule.unroll_explicit": 1024})
            conv2d_nhwc_reindex_shared = T.alloc_buffer((50176, 256), "int32", scope="shared")
            conv2d_nhwc_reindex_shared_wmma_accumulator = T.alloc_buffer((50176, 256), "int32", scope="wmma.accumulator")
            pad_temp_reindex_shared = T.alloc_buffer((50176, 64), "int8", scope="shared")
            p1_reindex_shared = T.alloc_buffer((1, 1, 256, 64), "int8", scope="shared")
            pad_temp_reindex_shared_wmma_matrix_a = T.alloc_buffer((50176, 64), "int8", scope="wmma.matrix_a")
            p1_reindex_shared_wmma_matrix_b = T.alloc_buffer((1, 1, 256, 64), "int8", scope="wmma.matrix_b")
            for ax2_0_0_ax3_0_0_fused in T.thread_binding(32, thread="blockIdx.y"):
                for ax2_0_1_ax3_0_1_fused in T.thread_binding(196, thread="blockIdx.x"):
                    for ax2_0_2_ax3_0_2_fused in T.thread_binding(4, thread="threadIdx.y"):
                        for ax0_0, ax1_0, ax4_0_0 in T.grid(1, 1, 2):
                            for ax0_ax1_fused in range(1024):
                                with T.block("pad_temp_reindex_shared"):
                                    v0 = T.axis.spatial(50176, ax2_0_0_ax3_0_0_fused // 4 * 6272 + ax2_0_1_ax3_0_1_fused * 32 + ax0_ax1_fused // 32)
                                    v1 = T.axis.spatial(64, ax4_0_0 * 32 + ax0_ax1_fused % 32)
                                    T.reads(p0[v0 // 3136, v0 % 3136 // 56, v0 % 56, v1])
                                    T.writes(pad_temp_reindex_shared[v0, v1])
                                    T.block_attr({"buffer_dim_align": [[0, 0, 32, 16]], "meta_schedule.cooperative_fetch": 4})
                                    pad_temp_reindex_shared[v0, v1] = p0[v0 // 3136, v0 % 3136 // 56, v0 % 56, v1]
                            for ax0_ax1_ax2_ax3_fused in range(2048):
                                with T.block("p1_reindex_shared"):
                                    v0 = T.axis.spatial(1, 0)
                                    v1 = T.axis.spatial(1, 0)
                                    v2 = T.axis.spatial(256, ax2_0_0_ax3_0_0_fused % 4 * 64 + ax0_ax1_ax2_ax3_fused // 32)
                                    v3 = T.axis.spatial(64, ax4_0_0 * 32 + ax0_ax1_ax2_ax3_fused % 32)
                                    T.reads(p1[v2, v0, v1, v3])
                                    T.writes(p1_reindex_shared[v0, v1, v2, v3])
                                    T.block_attr({"buffer_dim_align": [[0, 2, 32, 16]], "meta_schedule.cooperative_fetch": 3})
                                    p1_reindex_shared[v0, v1, v2, v3] = p1[v2, v0, v1, v3]
                            for ax0_1, ax1_1, ax4_0_1 in T.grid(1, 1, 2):
                                for ax0_0_1, ax1_0_1 in T.grid(1, 1):
                                    with T.block("pad_temp_reindex_shared_wmma.matrix_a_o"):
                                        v0_o = T.axis.spatial(3136, ax2_0_0_ax3_0_0_fused // 4 * 392 + ax2_0_1_ax3_0_1_fused * 2 + ax2_0_2_ax3_0_2_fused // 2 + ax0_0_1)
                                        v1_o = T.axis.spatial(4, ax4_0_0 * 2 + ax4_0_1 + ax1_0_1)
                                        T.reads(pad_temp_reindex_shared[v0_o * 16:v0_o * 16 + 16, v1_o * 16:v1_o * 16 + 16])
                                        T.writes(pad_temp_reindex_shared_wmma_matrix_a[v0_o * 16:v0_o * 16 + 16, v1_o * 16:v1_o * 16 + 16])
                                        T.block_attr({"meta_schedule.auto_tensorize": "wmma_load_16x16x16_s8_a_shared"})
                                        for ax0_1_1, ax1_1_1 in T.grid(16, 16):
                                            with T.block("pad_temp_reindex_shared_wmma.matrix_a"):
                                                v0_i, v1_i = T.axis.remap("SS", [ax0_1_1, ax1_1_1])
                                                T.reads(pad_temp_reindex_shared[v0_o * 16 + v0_i, v1_o * 16 + v1_i])
                                                T.writes(pad_temp_reindex_shared_wmma_matrix_a[v0_o * 16 + v0_i, v1_o * 16 + v1_i])
                                                pad_temp_reindex_shared_wmma_matrix_a[v0_o * 16 + v0_i, v1_o * 16 + v1_i] = pad_temp_reindex_shared[v0_o * 16 + v0_i, v1_o * 16 + v1_i]
                                for ax0, ax1, ax2_0, ax3_0 in T.grid(1, 1, 2, 1):
                                    with T.block("p1_reindex_shared_wmma.matrix_b_o"):
                                        v0_o, v1_o = T.axis.remap("SS", [ax0, ax1])
                                        v2_o = T.axis.spatial(16, ax2_0_0_ax3_0_0_fused % 4 * 4 + ax2_0_2_ax3_0_2_fused % 2 * 2 + ax2_0)
                                        v3_o = T.axis.spatial(4, ax4_0_0 * 2 + ax4_0_1 + ax3_0)
                                        T.reads(p1_reindex_shared[v0_o, v1_o, v2_o * 16:v2_o * 16 + 16, v3_o * 16:v3_o * 16 + 16])
                                        T.writes(p1_reindex_shared_wmma_matrix_b[v0_o, v1_o, v2_o * 16:v2_o * 16 + 16, v3_o * 16:v3_o * 16 + 16])
                                        T.block_attr({"meta_schedule.auto_tensorize": "wmma_load_16x16x16_s8_b_trans_shared"})
                                        for ax2_1, ax3_1 in T.grid(16, 16):
                                            with T.block("p1_reindex_shared_wmma.matrix_b"):
                                                v2_i, v3_i = T.axis.remap("SS", [ax2_1, ax3_1])
                                                T.reads(p1_reindex_shared[v0_o, v1_o, v2_o * 16 + v2_i, v3_o * 16 + v3_i])
                                                T.writes(p1_reindex_shared_wmma_matrix_b[v0_o, v1_o, v2_o * 16 + v2_i, v3_o * 16 + v3_i])
                                                p1_reindex_shared_wmma_matrix_b[v0_o, v1_o, v2_o * 16 + v2_i, v3_o * 16 + v3_i] = p1_reindex_shared[v0_o, v1_o, v2_o * 16 + v2_i, v3_o * 16 + v3_i]
                                for ax2_0_3, ax3_0_3, ax0_2, ax1_2, ax4_0_2, ax2_0_4, ax3_0_4 in T.grid(1, 1, 1, 1, 1, 1, 2):
                                    with T.block("conv2d_nhwc_o"):
                                        v0_o = T.axis.spatial(1, ax0_0 + ax0_1 + ax0_2)
                                        v1_o = T.axis.spatial(1, ax1_0 + ax1_1 + ax1_2)
                                        v2_o = T.axis.spatial(3136, ax2_0_0_ax3_0_0_fused // 4 * 392 + ax2_0_1_ax3_0_1_fused * 2 + ax2_0_2_ax3_0_2_fused // 2 + ax2_0_3 + ax2_0_4)
                                        v3_o = T.axis.spatial(16, ax2_0_0_ax3_0_0_fused % 4 * 4 + ax2_0_2_ax3_0_2_fused % 2 * 2 + ax3_0_3 * 2 + ax3_0_4)
                                        v4_o = T.axis.reduce(4, ax4_0_0 * 2 + ax4_0_1 + ax4_0_2)
                                        T.reads(pad_temp_reindex_shared_wmma_matrix_a[v2_o * 16:v2_o * 16 + 16, v4_o * 16:v4_o * 16 + 16], p1_reindex_shared_wmma_matrix_b[v0_o, v1_o, v3_o * 16:v3_o * 16 + 16, v4_o * 16:v4_o * 16 + 16])
                                        T.writes(conv2d_nhwc_reindex_shared_wmma_accumulator[v2_o * 16:v2_o * 16 + 16, v3_o * 16:v3_o * 16 + 16])
                                        T.block_attr({"meta_schedule.auto_tensorize": "wmma_sync_16x16x16_s8s8s32_trans", "meta_schedule.auto_tensorize_init": "wmma_fill_16x16x16_s32", "meta_schedule.thread_extent_high_inclusive": 1024, "meta_schedule.thread_extent_low_inclusive": 32, "warp_execution": 1})
                                        with T.init():
                                            for ax2_1, ax3_1 in T.grid(16, 16):
                                                with T.block("conv2d_nhwc_init"):
                                                    v2_i_init, v3_i_init = T.axis.remap("SS", [ax2_1, ax3_1])
                                                    T.reads()
                                                    T.writes(conv2d_nhwc_reindex_shared_wmma_accumulator[v2_o * 16 + v2_i_init, v3_o * 16 + v3_i_init])
                                                    conv2d_nhwc_reindex_shared_wmma_accumulator[v2_o * 16 + v2_i_init, v3_o * 16 + v3_i_init] = 0
                                        for ax2_1, ax3_1, ax4_1 in T.grid(16, 16, 16):
                                            with T.block("conv2d_nhwc"):
                                                v2_i, v3_i, v4_i = T.axis.remap("SSR", [ax2_1, ax3_1, ax4_1])
                                                T.reads(conv2d_nhwc_reindex_shared_wmma_accumulator[v2_o * 16 + v2_i, v3_o * 16 + v3_i], pad_temp_reindex_shared_wmma_matrix_a[v2_o * 16 + v2_i, v4_o * 16 + v4_i], p1_reindex_shared_wmma_matrix_b[v0_o, v1_o, v3_o * 16 + v3_i, v4_o * 16 + v4_i])
                                                T.writes(conv2d_nhwc_reindex_shared_wmma_accumulator[v2_o * 16 + v2_i, v3_o * 16 + v3_i])
                                                T.block_attr({"meta_schedule.tiling_structure": "SSSRRSRS"})
                                                conv2d_nhwc_reindex_shared_wmma_accumulator[v2_o * 16 + v2_i, v3_o * 16 + v3_i] = conv2d_nhwc_reindex_shared_wmma_accumulator[v2_o * 16 + v2_i, v3_o * 16 + v3_i] + T.Cast("int32", pad_temp_reindex_shared_wmma_matrix_a[v2_o * 16 + v2_i, v4_o * 16 + v4_i]) * T.Cast("int32", p1_reindex_shared_wmma_matrix_b[v0_o, v1_o, v3_o * 16 + v3_i, v4_o * 16 + v4_i])
                        for ax0_0, ax1_0 in T.grid(1, 2):
                            with T.block("conv2d_nhwc_reindex_shared_wmma.accumulator_o"):
                                v0_o = T.axis.spatial(3136, ax2_0_0_ax3_0_0_fused // 4 * 392 + ax2_0_1_ax3_0_1_fused * 2 + ax2_0_2_ax3_0_2_fused // 2 + ax0_0)
                                v1_o = T.axis.spatial(16, ax2_0_0_ax3_0_0_fused % 4 * 4 + ax2_0_2_ax3_0_2_fused % 2 * 2 + ax1_0)
                                T.reads(conv2d_nhwc_reindex_shared_wmma_accumulator[v0_o * 16:v0_o * 16 + 16, v1_o * 16:v1_o * 16 + 16])
                                T.writes(conv2d_nhwc_reindex_shared[v0_o * 16:v0_o * 16 + 16, v1_o * 16:v1_o * 16 + 16])
                                T.block_attr({"meta_schedule.auto_tensorize": "wmma_store_16x16x16_s32_shared"})
                                for ax0_1, ax1_1 in T.grid(16, 16):
                                    with T.block("conv2d_nhwc_reindex_shared_wmma.accumulator"):
                                        v0_i, v1_i = T.axis.remap("SS", [ax0_1, ax1_1])
                                        T.reads(conv2d_nhwc_reindex_shared_wmma_accumulator[v0_o * 16 + v0_i, v1_o * 16 + v1_i])
                                        T.writes(conv2d_nhwc_reindex_shared[v0_o * 16 + v0_i, v1_o * 16 + v1_i])
                                        conv2d_nhwc_reindex_shared[v0_o * 16 + v0_i, v1_o * 16 + v1_i] = conv2d_nhwc_reindex_shared_wmma_accumulator[v0_o * 16 + v0_i, v1_o * 16 + v1_i]
                    for ax0, ax1_0, ax1_1, ax1_2, ax1_3 in T.grid(32, 1, 4, 32, 2):
                        with T.block("conv2d_nhwc_reindex_shared"):
                            v0 = T.axis.spatial(50176, ax2_0_0_ax3_0_0_fused // 4 * 6272 + ax2_0_1_ax3_0_1_fused * 32 + ax0)
                            v1 = T.axis.spatial(256, ax2_0_0_ax3_0_0_fused % 4 * 64 + (ax1_0 * 256 + ax1_1 * 64 + ax1_2 * 2 + ax1_3))
                            T.where(((ax1_0 * 4 + ax1_1) * 32 + ax1_2) * 2 + ax1_3 < 64)
                            T.reads(p7[()], conv2d_nhwc_reindex_shared[v0, v1], p2[0, 0, 0, v1], p3[0, 0, 0, v1], p4[v1], p5[v1], p6[v1], p8[0], p9[v0 // 3136, v0 % 3136 // 56, v0 % 56, v1])
                            T.writes(compute[v0 // 3136, v0 % 3136 // 56, v0 % 56, v1])
                            compute[v0 // 3136, v0 % 3136 // 56, v0 % 56, v1] = T.max(T.min(T.q_multiply_shift(T.max(T.min(p7[()] + T.q_multiply_shift_per_axis(conv2d_nhwc_reindex_shared[v0, v1] - p2[0, 0, 0, v1] + p3[0, 0, 0, v1], p4[v1], p5[v1], p6[v1], 31, T.bool(False), T.bool(True)), 255), 0) - p8[0], 1457846997, 31, 0) + T.q_multiply_shift(p9[v0 // 3136, v0 % 3136 // 56, v0 % 56, v1], 2101000910, 31, 0), 255), 0)

# fmt: on
def verify(anchor_mod, anchor_trace_fun, target_mod, target, ref):
    anchor_sch = Schedule(anchor_mod)
    anchor_trace_fun(anchor_sch)
    anchor_trace = anchor_sch.trace

    sch = Schedule(target_mod)

    ms.trace_apply.schedule_using_anchor_trace(sch, anchor_trace, Target(target))

    tvm.ir.assert_structural_equal(ref, sch.mod)


def test_dense_add_cpu():
    def apply_anchor_trace(sch: Schedule) -> None:
        b0 = sch.get_block(name="T_matmul_NT", func_name="main")
        b1 = sch.get_block(name="root", func_name="main")
        sch.annotate(block_or_loop=b0, ann_key="meta_schedule.tiling_structure", ann_val="SSRSRS")
        l2, l3, l4 = sch.get_loops(block=b0)
        v5, v6, v7, v8 = sch.sample_perfect_tile(
            loop=l2, n=4, max_innermost_factor=64, decision=[2, 8, 4, 2]
        )
        l9, l10, l11, l12 = sch.split(loop=l2, factors=[v5, v6, v7, v8], preserve_unit_iters=True)
        v13, v14, v15, v16 = sch.sample_perfect_tile(
            loop=l3, n=4, max_innermost_factor=64, decision=[2, 1, 1, 64]
        )
        l17, l18, l19, l20 = sch.split(
            loop=l3, factors=[v13, v14, v15, v16], preserve_unit_iters=True
        )
        v21, v22 = sch.sample_perfect_tile(loop=l4, n=2, max_innermost_factor=64, decision=[128, 1])
        l23, l24 = sch.split(loop=l4, factors=[v21, v22], preserve_unit_iters=True)
        sch.reorder(l9, l17, l10, l18, l23, l11, l19, l24, l12, l20)
        b25 = sch.cache_write(block=b0, write_buffer_index=0, storage_scope="global")
        sch.reverse_compute_at(block=b25, loop=l17, preserve_unit_loops=True, index=-1)
        sch.annotate(block_or_loop=b1, ann_key="meta_schedule.parallel", ann_val=160)
        sch.annotate(block_or_loop=b1, ann_key="meta_schedule.vectorize", ann_val=64)
        v26 = sch.sample_categorical(
            candidates=[0, 16, 64, 512], probs=[0.25, 0.25, 0.25, 0.25], decision=0
        )
        sch.annotate(block_or_loop=b1, ann_key="meta_schedule.unroll_explicit", ann_val=v26)
        sch.enter_postproc()
        b27 = sch.get_block(name="root", func_name="main")
        sch.unannotate(block_or_loop=b27, ann_key="meta_schedule.parallel")
        sch.unannotate(block_or_loop=b27, ann_key="meta_schedule.vectorize")
        sch.unannotate(block_or_loop=b27, ann_key="meta_schedule.unroll_explicit")
        b28, b29 = sch.get_child_blocks(b27)
        l30, l31, l32, l33, l34, l35, l36, l37, l38, l39 = sch.get_loops(block=b28)
        l40 = sch.fuse(l30, l31, preserve_unit_iters=True)
        sch.parallel(loop=l40)
        l41 = sch.fuse(l39, preserve_unit_iters=True)
        sch.vectorize(loop=l41)
        l42, l43, l44 = sch.get_loops(block=b29)
        l45 = sch.fuse(l42, preserve_unit_iters=True)
        sch.parallel(loop=l45)
        l46 = sch.fuse(l44, preserve_unit_iters=True)
        sch.vectorize(loop=l46)
        b47 = sch.get_block(name="T_matmul_NT", func_name="main")
        l48, l49, l50, l51, l52, l53, l54, l55, l56 = sch.get_loops(block=b47)
        b57 = sch.decompose_reduction(block=b47, loop=l51)
        b58 = sch.get_block(name="T_matmul_NT_update", func_name="main")
        b59 = sch.cache_read(block=b58, read_buffer_index=2, storage_scope="global")
        sch.transform_layout(
            block=b58,
            buffer=("read", 2),
            index_map=tvm.tir.IndexMap.from_func(
                lambda i0, i1: (
                    floordiv(i0, 64),
                    i1,
                    floormod(i0, 64),
                ),
                inverse_index_map=lambda i0, i1, i2: (
                    ((i0 * 64) + i2),
                    i1,
                ),
                index_dtype="int32",
            ),
            pad_value=None,
        )
        sch.annotate(block_or_loop=b59, ann_key="meta_schedule.layout_rewrite_preproc", ann_val=1)

    verify(Dense, apply_anchor_trace, DenseAdd, "llvm", DenseAdd_scheduled_cpu)


def test_dense_add_cpu_no_write_cache():
    def apply_trace(sch):
        b0 = sch.get_block(name="T_matmul_NT", func_name="main")
        b1 = sch.get_block(name="root", func_name="main")
        sch.annotate(block_or_loop=b0, ann_key="meta_schedule.tiling_structure", ann_val="SSRSRS")
        l2, l3, l4 = sch.get_loops(block=b0)
        v5, v6, v7, v8 = sch.sample_perfect_tile(
            loop=l2, n=4, max_innermost_factor=64, decision=[4, 4, 4, 2]
        )
        l9, l10, l11, l12 = sch.split(loop=l2, factors=[v5, v6, v7, v8], preserve_unit_iters=True)
        v13, v14, v15, v16 = sch.sample_perfect_tile(
            loop=l3, n=4, max_innermost_factor=64, decision=[1, 1, 4, 32]
        )
        l17, l18, l19, l20 = sch.split(
            loop=l3, factors=[v13, v14, v15, v16], preserve_unit_iters=True
        )
        v21, v22 = sch.sample_perfect_tile(loop=l4, n=2, max_innermost_factor=64, decision=[8, 16])
        l23, l24 = sch.split(loop=l4, factors=[v21, v22], preserve_unit_iters=True)
        sch.reorder(l9, l17, l10, l18, l23, l11, l19, l24, l12, l20)
        sch.annotate(block_or_loop=b1, ann_key="meta_schedule.parallel", ann_val=160)
        sch.annotate(block_or_loop=b1, ann_key="meta_schedule.vectorize", ann_val=64)
        v25 = sch.sample_categorical(
            candidates=[0, 16, 64, 512], probs=[0.25, 0.25, 0.25, 0.25], decision=1
        )
        sch.annotate(block_or_loop=b1, ann_key="meta_schedule.unroll_explicit", ann_val=v25)
        sch.enter_postproc()
        b26 = sch.get_block(name="root", func_name="main")
        sch.unannotate(block_or_loop=b26, ann_key="meta_schedule.parallel")
        sch.unannotate(block_or_loop=b26, ann_key="meta_schedule.vectorize")
        sch.unannotate(block_or_loop=b26, ann_key="meta_schedule.unroll_explicit")
        (b27,) = sch.get_child_blocks(b26)
        l28, l29, l30, l31, l32, l33, l34, l35, l36, l37 = sch.get_loops(block=b27)
        l38 = sch.fuse(l28, l29, l30, l31, preserve_unit_iters=True)
        sch.parallel(loop=l38)
        l39 = sch.fuse(l37, preserve_unit_iters=True)
        sch.vectorize(loop=l39)
        sch.annotate(block_or_loop=l38, ann_key="pragma_auto_unroll_max_step", ann_val=16)
        sch.annotate(block_or_loop=l38, ann_key="pragma_unroll_explicit", ann_val=1)
        b40 = sch.get_block(name="T_matmul_NT", func_name="main")
        l41, l42, l43, l44, l45, l46, l47 = sch.get_loops(block=b40)
        b48 = sch.decompose_reduction(block=b40, loop=l42)
        b49 = sch.get_block(name="T_matmul_NT_update", func_name="main")
        b50 = sch.cache_read(block=b49, read_buffer_index=2, storage_scope="global")
        sch.transform_layout(
            block=b49,
            buffer=("read", 2),
            index_map=tvm.tir.IndexMap.from_func(
                lambda i0, i1: (
                    floordiv(i1, 16),
                    floordiv(i0, 32),
                    floormod(i1, 16),
                    floormod(i0, 32),
                ),
                inverse_index_map=lambda i0, i1, i2, i3: (
                    ((i1 * 32) + i3),
                    ((i0 * 16) + i2),
                ),
                index_dtype="int32",
            ),
            pad_value=None,
        )
        sch.annotate(block_or_loop=b50, ann_key="meta_schedule.layout_rewrite_preproc", ann_val=1)

    verify(Dense, apply_trace, DenseAdd, "llvm", DenseAdd_cpu_no_write_cache)


def test_dense_add_gpu():
    def apply_anchor_trace(sch: Schedule) -> None:
        b0 = sch.get_block(name="T_matmul_NT", func_name="main")
        b1 = sch.get_block(name="root", func_name="main")
        sch.annotate(block_or_loop=b0, ann_key="meta_schedule.tiling_structure", ann_val="SSSRRSRS")
        l2, l3, l4 = sch.get_loops(block=b0)
        v5, v6, v7, v8, v9 = sch.sample_perfect_tile(
            loop=l2, n=5, max_innermost_factor=64, decision=[8, 1, 16, 1, 1]
        )
        l10, l11, l12, l13, l14 = sch.split(
            loop=l2, factors=[v5, v6, v7, v8, v9], preserve_unit_iters=True
        )
        v15, v16, v17, v18, v19 = sch.sample_perfect_tile(
            loop=l3, n=5, max_innermost_factor=64, decision=[4, 1, 8, 4, 1]
        )
        l20, l21, l22, l23, l24 = sch.split(
            loop=l3, factors=[v15, v16, v17, v18, v19], preserve_unit_iters=True
        )
        v25, v26, v27 = sch.sample_perfect_tile(
            loop=l4, n=3, max_innermost_factor=64, decision=[32, 1, 4]
        )
        l28, l29, l30 = sch.split(loop=l4, factors=[v25, v26, v27], preserve_unit_iters=True)
        sch.reorder(l10, l20, l11, l21, l12, l22, l28, l29, l13, l23, l30, l14, l24)
        l31 = sch.fuse(l10, l20, preserve_unit_iters=True)
        sch.bind(loop=l31, thread_axis="blockIdx.x")
        l32 = sch.fuse(l11, l21, preserve_unit_iters=True)
        sch.bind(loop=l32, thread_axis="vthread.x")
        l33 = sch.fuse(l12, l22, preserve_unit_iters=True)
        sch.bind(loop=l33, thread_axis="threadIdx.x")
        sch.annotate(
            block_or_loop=b0, ann_key="meta_schedule.thread_extent_low_inclusive", ann_val=16
        )
        sch.annotate(
            block_or_loop=b0, ann_key="meta_schedule.thread_extent_high_inclusive", ann_val=256
        )
        b34 = sch.cache_write(block=b0, write_buffer_index=0, storage_scope="local")
        sch.reverse_compute_at(block=b34, loop=l33, preserve_unit_loops=True, index=-1)
        b35 = sch.cache_read(
            block=b0, read_buffer_index=0, storage_scope="shared", consumer_blocks=[b0]
        )
        sch.compute_at(block=b35, loop=l28, preserve_unit_loops=True, index=-1)
        l36, l37, l38, l39, l40, l41 = sch.get_loops(block=b35)
        l42 = sch.fuse(l40, l41, preserve_unit_iters=True)
        v43 = sch.sample_categorical(
            candidates=[1, 2, 3, 4], probs=[0.25, 0.25, 0.25, 0.25], decision=1
        )
        sch.annotate(block_or_loop=b35, ann_key="meta_schedule.cooperative_fetch", ann_val=v43)
        b44 = sch.cache_read(
            block=b0, read_buffer_index=1, storage_scope="shared", consumer_blocks=[b0]
        )
        sch.compute_at(block=b44, loop=l28, preserve_unit_loops=True, index=-1)
        l45, l46, l47, l48, l49, l50 = sch.get_loops(block=b44)
        l51 = sch.fuse(l49, l50, preserve_unit_iters=True)
        v52 = sch.sample_categorical(
            candidates=[1, 2, 3, 4], probs=[0.25, 0.25, 0.25, 0.25], decision=3
        )
        sch.annotate(block_or_loop=b44, ann_key="meta_schedule.cooperative_fetch", ann_val=v52)
        v53 = sch.sample_categorical(
            candidates=[0, 16, 64, 512, 1024],
            probs=[
                0.20000000000000001,
                0.20000000000000001,
                0.20000000000000001,
                0.20000000000000001,
                0.20000000000000001,
            ],
            decision=2,
        )
        sch.annotate(block_or_loop=b1, ann_key="meta_schedule.unroll_explicit", ann_val=v53)
        sch.enter_postproc()
        sch.unannotate(block_or_loop=b35, ann_key="meta_schedule.cooperative_fetch")
        l54, l55, l56, l57, l58 = sch.get_loops(block=b35)
        l59, l60, l61 = sch.split(loop=l58, factors=[None, 128, 2], preserve_unit_iters=True)
        sch.vectorize(loop=l61)
        sch.bind(loop=l60, thread_axis="threadIdx.x")
        sch.unannotate(block_or_loop=b44, ann_key="meta_schedule.cooperative_fetch")
        l62, l63, l64, l65, l66 = sch.get_loops(block=b44)
        l67, l68, l69 = sch.split(loop=l66, factors=[None, 128, 4], preserve_unit_iters=True)
        sch.vectorize(loop=l69)
        sch.bind(loop=l68, thread_axis="threadIdx.x")
        b70 = sch.get_block(name="root", func_name="main")
        sch.unannotate(block_or_loop=b70, ann_key="meta_schedule.unroll_explicit")
        b71, b72, b73, b74 = sch.get_child_blocks(b70)
        l75, l76, l77, l78, l79, l80, l81 = sch.get_loops(block=b71)
        sch.annotate(block_or_loop=l75, ann_key="pragma_auto_unroll_max_step", ann_val=64)
        sch.annotate(block_or_loop=l75, ann_key="pragma_unroll_explicit", ann_val=1)
        l82, l83, l84, l85, l86, l87, l88 = sch.get_loops(block=b72)
        sch.annotate(block_or_loop=l82, ann_key="pragma_auto_unroll_max_step", ann_val=64)
        sch.annotate(block_or_loop=l82, ann_key="pragma_unroll_explicit", ann_val=1)
        l89, l90, l91, l92, l93, l94, l95, l96, l97, l98 = sch.get_loops(block=b73)
        sch.annotate(block_or_loop=l89, ann_key="pragma_auto_unroll_max_step", ann_val=64)
        sch.annotate(block_or_loop=l89, ann_key="pragma_unroll_explicit", ann_val=1)
        l99, l100, l101, l102, l103 = sch.get_loops(block=b74)
        sch.annotate(block_or_loop=l99, ann_key="pragma_auto_unroll_max_step", ann_val=64)
        sch.annotate(block_or_loop=l99, ann_key="pragma_unroll_explicit", ann_val=1)
        b104 = sch.get_block(name="T_matmul_NT", func_name="main")
        l105, l106, l107, l108, l109, l110, l111, l112, l113, l114 = sch.get_loops(block=b104)
        b115 = sch.decompose_reduction(block=b104, loop=l108)

    verify(Dense, apply_anchor_trace, DenseAdd, "cuda", DenseAdd_scheduled_gpu)


def test_conv2d_int8_tensorcore():
    def apply_trace(sch):
        b0 = sch.get_block(name="pad_temp", func_name="main")
        b1 = sch.get_block(name="conv2d_nhwc", func_name="main")
        b2 = sch.get_block(name="T_subtract", func_name="main")
        b3 = sch.get_block(name="T_add", func_name="main")
        b4 = sch.get_block(name="T_cast", func_name="main")
        b5 = sch.get_block(name="T_multiply", func_name="main")
        b6 = sch.get_block(name="T_add_1", func_name="main")
        b7 = sch.get_block(name="T_right_shift", func_name="main")
        b8 = sch.get_block(name="T_cast_1", func_name="main")
        b9 = sch.get_block(name="T_add_2", func_name="main")
        b10 = sch.get_block(name="compute", func_name="main")
        b11 = sch.get_block(name="T_cast_2", func_name="main")
        b12 = sch.get_block(name="T_cast_3", func_name="main")
        b13 = sch.get_block(name="T_subtract_1", func_name="main")
        b14 = sch.get_block(name="compute_1", func_name="main")
        b15 = sch.get_block(name="root", func_name="main")
        sch.annotate(block_or_loop=b1, ann_key="meta_schedule.tiling_structure", ann_val="SSSRRSRS")
        b16 = sch.reindex(block=b1, buffer=("write", 0))
        b17 = sch.reindex(block=b1, buffer=("read", 0))
        b18 = sch.reindex(block=b1, buffer=("read", 1))
        sch.transform_layout(
            block=b1,
            buffer=("read", 0),
            index_map=lambda nn, yy, xx, rc: (
                (((nn * 3136) + (yy * 56)) + xx),
                rc,
            ),
            pad_value=None,
        )
        sch.transform_layout(
            block=b1,
            buffer=("read", 1),
            index_map=lambda ff, ry, rx, rc: (
                ry,
                rx,
                ff,
                rc,
            ),
            pad_value=None,
        )
        sch.transform_layout(
            block=b1,
            buffer=("write", 0),
            index_map=lambda nn, yy, xx, ff: (
                (((nn * 3136) + (yy * 56)) + xx),
                ff,
            ),
            pad_value=None,
        )
        sch.transform_block_layout(
            block=b16,
            index_map=lambda nn, yy, xx, ff: (
                (((nn * 3136) + (yy * 56)) + xx),
                ff,
            ),
        )
        sch.transform_block_layout(
            block=b17,
            index_map=lambda nn, yy, xx, rc: (
                (((nn * 3136) + (yy * 56)) + xx),
                rc,
            ),
        )
        sch.transform_block_layout(
            block=b18,
            index_map=lambda ff, ry, rx, rc: (
                ry,
                rx,
                ff,
                rc,
            ),
        )
        sch.transform_block_layout(
            block=b1,
            index_map=lambda nn, yy, xx, ff, ry, rx, rc: (
                ry,
                rx,
                (((nn * 3136) + (yy * 56)) + xx),
                ff,
                rc,
            ),
        )
        l19, l20, l21, l22, l23 = sch.get_loops(block=b1)
        l24, l25 = sch.split(loop=l23, factors=[None, 16], preserve_unit_iters=True)
        l26, l27 = sch.split(loop=l22, factors=[None, 16], preserve_unit_iters=True)
        l28, l29 = sch.split(loop=l21, factors=[None, 16], preserve_unit_iters=True)
        l30, l31, l32, l33, l34, l35, l36, l37 = sch.get_loops(block=b1)
        sch.reorder(l34, l36, l29, l27, l25)
        b38 = sch.blockize(target=l29)
        sch.annotate(
            block_or_loop=b38,
            ann_key="meta_schedule.auto_tensorize",
            ann_val="wmma_sync_16x16x16_s8s8s32_trans",
        )
        sch.annotate(
            block_or_loop=b38,
            ann_key="meta_schedule.auto_tensorize_init",
            ann_val="wmma_fill_16x16x16_s32",
        )
        sch.annotate(block_or_loop=b38, ann_key="warp_execution", ann_val=1)
        l39, l40, l41, l42, l43 = sch.get_loops(block=b38)
        v44, v45, v46 = sch.sample_perfect_tile(
            loop=l39, n=3, max_innermost_factor=4, decision=[1, 1, 1]
        )
        l47, l48, l49 = sch.split(loop=l39, factors=[v44, v45, v46], preserve_unit_iters=True)
        v50, v51, v52 = sch.sample_perfect_tile(
            loop=l40, n=3, max_innermost_factor=4, decision=[1, 1, 1]
        )
        l53, l54, l55 = sch.split(loop=l40, factors=[v50, v51, v52], preserve_unit_iters=True)
        v56, v57, v58, v59, v60 = sch.sample_perfect_tile(
            loop=l41, n=5, max_innermost_factor=4, decision=[392, 1, 8, 1, 1]
        )
        l61, l62, l63, l64, l65 = sch.split(
            loop=l41, factors=[v56, v57, v58, v59, v60], preserve_unit_iters=True
        )
        v66, v67, v68, v69, v70 = sch.sample_perfect_tile(
            loop=l42, n=5, max_innermost_factor=4, decision=[8, 1, 2, 1, 1]
        )
        l71, l72, l73, l74, l75 = sch.split(
            loop=l42, factors=[v66, v67, v68, v69, v70], preserve_unit_iters=True
        )
        v76, v77, v78 = sch.sample_perfect_tile(
            loop=l43, n=3, max_innermost_factor=4, decision=[2, 1, 2]
        )
        l79, l80, l81 = sch.split(loop=l43, factors=[v76, v77, v78], preserve_unit_iters=True)
        sch.reorder(
            l61,
            l71,
            l62,
            l72,
            l63,
            l73,
            l47,
            l53,
            l79,
            l48,
            l54,
            l80,
            l64,
            l74,
            l49,
            l55,
            l81,
            l65,
            l75,
        )
        l82 = sch.fuse(l61, l71, preserve_unit_iters=True)
        sch.bind(loop=l82, thread_axis="blockIdx.x")
        l83 = sch.fuse(l62, l72, preserve_unit_iters=True)
        sch.bind(loop=l83, thread_axis="vthread.x")
        l84 = sch.fuse(l63, l73, preserve_unit_iters=True)
        sch.bind(loop=l84, thread_axis="threadIdx.x")
        sch.annotate(
            block_or_loop=b38, ann_key="meta_schedule.thread_extent_low_inclusive", ann_val=32
        )
        sch.annotate(
            block_or_loop=b38, ann_key="meta_schedule.thread_extent_high_inclusive", ann_val=1024
        )
        b85 = sch.cache_write(block=b38, write_buffer_index=0, storage_scope="shared")
        sch.reverse_compute_at(block=b85, loop=l83, preserve_unit_loops=True, index=-1)
        b86 = sch.cache_write(block=b38, write_buffer_index=0, storage_scope="wmma.accumulator")
        sch.reverse_compute_at(block=b86, loop=l84, preserve_unit_loops=True, index=-1)
        v87 = sch.sample_categorical(
            candidates=[1, 2, 3, 4, 8, 16],
            probs=[
                0.16666666666666666,
                0.16666666666666666,
                0.16666666666666666,
                0.16666666666666666,
                0.16666666666666666,
                0.16666666666666666,
            ],
            decision=0,
        )
        sch.annotate(block_or_loop=b85, ann_key="meta_schedule.cooperative_fetch", ann_val=v87)
        sch.reverse_compute_inline(block=b16)
        l88, l89, l90, l91, l92 = sch.get_loops(block=b86)
        l93, l94 = sch.split(loop=l92, factors=[None, 16], preserve_unit_iters=True)
        l95, l96 = sch.split(loop=l91, factors=[None, 16], preserve_unit_iters=True)
        l97, l98, l99, l100, l101, l102, l103 = sch.get_loops(block=b86)
        sch.reorder(l102, l96, l94)
        b104 = sch.blockize(target=l96)
        sch.annotate(
            block_or_loop=b104,
            ann_key="meta_schedule.auto_tensorize",
            ann_val="wmma_store_16x16x16_s32_shared",
        )
        b105 = sch.cache_read(
            block=b38, read_buffer_index=0, storage_scope="shared", consumer_blocks=[b38]
        )
        sch.compute_at(block=b105, loop=l79, preserve_unit_loops=True, index=-1)
        l106, l107, l108, l109, l110, l111, l112, l113 = sch.get_loops(block=b105)
        l114 = sch.fuse(l112, l113, preserve_unit_iters=True)
        v115 = sch.sample_categorical(
            candidates=[1, 2, 3, 4, 8, 16],
            probs=[
                0.16666666666666666,
                0.16666666666666666,
                0.16666666666666666,
                0.16666666666666666,
                0.16666666666666666,
                0.16666666666666666,
            ],
            decision=5,
        )
        sch.annotate(block_or_loop=b105, ann_key="meta_schedule.cooperative_fetch", ann_val=v115)
        b116 = sch.cache_read(
            block=b38, read_buffer_index=1, storage_scope="shared", consumer_blocks=[b38]
        )
        sch.compute_at(block=b116, loop=l79, preserve_unit_loops=True, index=-1)
        l117, l118, l119, l120, l121, l122, l123, l124, l125, l126 = sch.get_loops(block=b116)
        l127 = sch.fuse(l123, l124, l125, l126, preserve_unit_iters=True)
        v128 = sch.sample_categorical(
            candidates=[1, 2, 3, 4, 8, 16],
            probs=[
                0.16666666666666666,
                0.16666666666666666,
                0.16666666666666666,
                0.16666666666666666,
                0.16666666666666666,
                0.16666666666666666,
            ],
            decision=4,
        )
        sch.annotate(block_or_loop=b116, ann_key="meta_schedule.cooperative_fetch", ann_val=v128)
        b129 = sch.cache_read(block=b38, read_buffer_index=0, storage_scope="wmma.matrix_a")
        sch.compute_at(block=b129, loop=l80, preserve_unit_loops=True, index=-1)
        l130, l131, l132, l133, l134, l135, l136, l137, l138, l139, l140 = sch.get_loops(block=b129)
        l141, l142 = sch.split(loop=l140, factors=[None, 16], preserve_unit_iters=True)
        l143, l144 = sch.split(loop=l139, factors=[None, 16], preserve_unit_iters=True)
        (
            l145,
            l146,
            l147,
            l148,
            l149,
            l150,
            l151,
            l152,
            l153,
            l154,
            l155,
            l156,
            l157,
        ) = sch.get_loops(block=b129)
        sch.reorder(l156, l144, l142)
        b158 = sch.blockize(target=l144)
        sch.annotate(
            block_or_loop=b158,
            ann_key="meta_schedule.auto_tensorize",
            ann_val="wmma_load_16x16x16_s8_a_shared",
        )
        b159 = sch.cache_read(block=b38, read_buffer_index=1, storage_scope="wmma.matrix_b")
        sch.compute_at(block=b159, loop=l80, preserve_unit_loops=True, index=-1)
        (
            l160,
            l161,
            l162,
            l163,
            l164,
            l165,
            l166,
            l167,
            l168,
            l169,
            l170,
            l171,
            l172,
        ) = sch.get_loops(block=b159)
        l173, l174 = sch.split(loop=l172, factors=[None, 16], preserve_unit_iters=True)
        l175, l176 = sch.split(loop=l171, factors=[None, 16], preserve_unit_iters=True)
        (
            l177,
            l178,
            l179,
            l180,
            l181,
            l182,
            l183,
            l184,
            l185,
            l186,
            l187,
            l188,
            l189,
            l190,
            l191,
        ) = sch.get_loops(block=b159)
        sch.reorder(l190, l176, l174)
        b192 = sch.blockize(target=l176)
        sch.annotate(
            block_or_loop=b192,
            ann_key="meta_schedule.auto_tensorize",
            ann_val="wmma_load_16x16x16_s8_b_trans_shared",
        )
        sch.compute_inline(block=b17)
        sch.compute_inline(block=b18)
        sch.storage_align(block=b105, buffer_index=0, axis=-2, factor=32, offset=16)
        sch.storage_align(block=b116, buffer_index=0, axis=-2, factor=32, offset=16)
        sch.reverse_compute_inline(block=b14)
        sch.reverse_compute_inline(block=b13)
        sch.reverse_compute_inline(block=b12)
        sch.reverse_compute_inline(block=b11)
        sch.reverse_compute_inline(block=b10)
        sch.reverse_compute_inline(block=b9)
        sch.reverse_compute_inline(block=b8)
        sch.reverse_compute_inline(block=b7)
        sch.reverse_compute_inline(block=b6)
        sch.reverse_compute_inline(block=b5)
        sch.reverse_compute_inline(block=b4)
        sch.reverse_compute_inline(block=b3)
        sch.reverse_compute_inline(block=b2)
        sch.compute_inline(block=b0)
        v193 = sch.sample_categorical(
            candidates=[0, 16, 64, 512, 1024],
            probs=[
                0.20000000000000001,
                0.20000000000000001,
                0.20000000000000001,
                0.20000000000000001,
                0.20000000000000001,
            ],
            decision=3,
        )
        sch.annotate(block_or_loop=b15, ann_key="meta_schedule.unroll_explicit", ann_val=v193)
        sch.enter_postproc()
        sch.unannotate(block_or_loop=b85, ann_key="meta_schedule.cooperative_fetch")
        l194, l195, l196, l197 = sch.get_loops(block=b85)
        l198, l199 = sch.split(loop=l197, factors=[None, 16], preserve_unit_iters=True)
        sch.bind(loop=l199, thread_axis="threadIdx.x")
        sch.unannotate(block_or_loop=b105, ann_key="meta_schedule.cooperative_fetch")
        l200, l201, l202, l203, l204, l205, l206 = sch.get_loops(block=b105)
        l207, l208, l209 = sch.split(loop=l206, factors=[None, 16, 16], preserve_unit_iters=True)
        sch.vectorize(loop=l209)
        sch.bind(loop=l208, thread_axis="threadIdx.x")
        sch.unannotate(block_or_loop=b116, ann_key="meta_schedule.cooperative_fetch")
        l210, l211, l212, l213, l214, l215, l216 = sch.get_loops(block=b116)
        l217, l218, l219 = sch.split(loop=l216, factors=[None, 16, 8], preserve_unit_iters=True)
        sch.vectorize(loop=l219)
        sch.bind(loop=l218, thread_axis="threadIdx.x")
        b220 = sch.get_block(name="root", func_name="main")
        sch.unannotate(block_or_loop=b220, ann_key="meta_schedule.unroll_explicit")
        b221, b222, b223, b224, b225, b226, b227 = sch.get_child_blocks(b220)
        l228, l229, l230, l231, l232, l233, l234, l235, l236 = sch.get_loops(block=b221)
        sch.annotate(block_or_loop=l228, ann_key="pragma_auto_unroll_max_step", ann_val=512)
        sch.annotate(block_or_loop=l228, ann_key="pragma_unroll_explicit", ann_val=1)
        l237, l238, l239, l240, l241, l242, l243, l244, l245 = sch.get_loops(block=b222)
        sch.annotate(block_or_loop=l237, ann_key="pragma_auto_unroll_max_step", ann_val=512)
        sch.annotate(block_or_loop=l237, ann_key="pragma_unroll_explicit", ann_val=1)
        l246, l247, l248, l249, l250, l251, l252, l253, l254, l255, l256 = sch.get_loops(block=b223)
        sch.annotate(block_or_loop=l246, ann_key="pragma_auto_unroll_max_step", ann_val=512)
        sch.annotate(block_or_loop=l246, ann_key="pragma_unroll_explicit", ann_val=1)
        (
            l257,
            l258,
            l259,
            l260,
            l261,
            l262,
            l263,
            l264,
            l265,
            l266,
            l267,
            l268,
            l269,
        ) = sch.get_loops(block=b224)
        sch.annotate(block_or_loop=l257, ann_key="pragma_auto_unroll_max_step", ann_val=512)
        sch.annotate(block_or_loop=l257, ann_key="pragma_unroll_explicit", ann_val=1)
        (
            l270,
            l271,
            l272,
            l273,
            l274,
            l275,
            l276,
            l277,
            l278,
            l279,
            l280,
            l281,
            l282,
            l283,
            l284,
            l285,
        ) = sch.get_loops(block=b225)
        sch.annotate(block_or_loop=l270, ann_key="pragma_auto_unroll_max_step", ann_val=512)
        sch.annotate(block_or_loop=l270, ann_key="pragma_unroll_explicit", ann_val=1)
        l286, l287, l288, l289, l290 = sch.get_loops(block=b226)
        sch.annotate(block_or_loop=l286, ann_key="pragma_auto_unroll_max_step", ann_val=512)
        sch.annotate(block_or_loop=l286, ann_key="pragma_unroll_explicit", ann_val=1)
        l291, l292, l293, l294, l295 = sch.get_loops(block=b227)
        sch.annotate(block_or_loop=l291, ann_key="pragma_auto_unroll_max_step", ann_val=512)
        sch.annotate(block_or_loop=l291, ann_key="pragma_unroll_explicit", ann_val=1)
        b296 = sch.get_block(name="conv2d_nhwc_o", func_name="main")
        (
            l297,
            l298,
            l299,
            l300,
            l301,
            l302,
            l303,
            l304,
            l305,
            l306,
            l307,
            l308,
            l309,
            l310,
            l311,
            l312,
        ) = sch.get_loops(block=b296)
        b313 = sch.decompose_reduction(block=b296, loop=l300)
        sch.unannotate(block_or_loop=b313, ann_key="meta_schedule.auto_tensorize")
        sch.annotate(
            block_or_loop=b313,
            ann_key="meta_schedule.auto_tensorize",
            ann_val="wmma_fill_16x16x16_s32",
        )
        sch.unannotate(block_or_loop=b296, ann_key="meta_schedule.auto_tensorize_init")
        sch.unannotate(block_or_loop=b313, ann_key="meta_schedule.auto_tensorize_init")
        b314 = sch.get_block(name="conv2d_nhwc_o_init", func_name="main")
        sch.unannotate(block_or_loop=b314, ann_key="meta_schedule.auto_tensorize")
        sch.tensorize(block_or_loop=b314, tensor_intrin="wmma_fill_16x16x16_s32")
        b315 = sch.get_block(name="pad_temp_reindex_shared_wmma.matrix_a_o", func_name="main")
        sch.unannotate(block_or_loop=b315, ann_key="meta_schedule.auto_tensorize")
        sch.tensorize(block_or_loop=b315, tensor_intrin="wmma_load_16x16x16_s8_a_shared")
        b316 = sch.get_block(name="p1_reindex_shared_wmma.matrix_b_o", func_name="main")
        sch.unannotate(block_or_loop=b316, ann_key="meta_schedule.auto_tensorize")
        sch.tensorize(block_or_loop=b316, tensor_intrin="wmma_load_16x16x16_s8_b_trans_shared")
        b317 = sch.get_block(name="conv2d_nhwc_o_update", func_name="main")
        sch.unannotate(block_or_loop=b317, ann_key="meta_schedule.auto_tensorize")
        sch.tensorize(block_or_loop=b317, tensor_intrin="wmma_sync_16x16x16_s8s8s32_trans")
        b318 = sch.get_block(name="conv2d_nhwc_reindex_shared_wmma.accumulator_o", func_name="main")
        sch.unannotate(block_or_loop=b318, ann_key="meta_schedule.auto_tensorize")
        sch.tensorize(block_or_loop=b318, tensor_intrin="wmma_store_16x16x16_s32_shared")

    verify(Conv2dInt8, apply_trace, Conv2dInt8_target, "cuda", Conv2dInt8_tensorcore_scheduled)


def test_conv2d_int8_vnni():
    def apply_trace(sch):
        b0 = sch.get_block(name="compile_engine_const", func_name="main")
        b1 = sch.get_block(name="conv2d_NCHWc_int8", func_name="main")
        b2 = sch.get_block(name="T_add", func_name="main")
        b3 = sch.get_block(name="T_cast", func_name="main")
        b4 = sch.get_block(name="T_multiply", func_name="main")
        b5 = sch.get_block(name="compile_engine_const_1", func_name="main")
        b6 = sch.get_block(name="T_add_1", func_name="main")
        b7 = sch.get_block(name="T_floor", func_name="main")
        b8 = sch.get_block(name="T_cast_1", func_name="main")
        b9 = sch.get_block(name="compute", func_name="main")
        b10 = sch.get_block(name="T_cast_2", func_name="main")
        b11 = sch.get_block(name="T_cast_3", func_name="main")
        b12 = sch.get_block(name="T_subtract", func_name="main")
        b13 = sch.get_block(name="T_multiply_1", func_name="main")
        b14 = sch.get_block(name="compile_engine_const_2", func_name="main")
        b15 = sch.get_block(name="T_add_2", func_name="main")
        b16 = sch.get_block(name="T_floor_1", func_name="main")
        b17 = sch.get_block(name="T_cast_4", func_name="main")
        b18 = sch.get_block(name="T_add_3", func_name="main")
        b19 = sch.get_block(name="compute_1", func_name="main")
        b20 = sch.get_block(name="T_cast_5", func_name="main")
        b21 = sch.get_block(name="root", func_name="main")
        sch.compute_inline(block=b20)
        sch.compute_inline(block=b19)
        sch.compute_inline(block=b18)
        sch.compute_inline(block=b17)
        sch.compute_inline(block=b16)
        sch.compute_inline(block=b15)
        sch.compute_inline(block=b14)
        sch.compute_inline(block=b13)
        sch.compute_inline(block=b12)
        sch.compute_inline(block=b11)
        sch.compute_inline(block=b10)
        sch.compute_inline(block=b9)
        sch.compute_inline(block=b8)
        sch.compute_inline(block=b7)
        sch.compute_inline(block=b6)
        sch.compute_inline(block=b5)
        sch.compute_inline(block=b4)
        sch.compute_inline(block=b3)
        sch.compute_inline(block=b2)
        sch.compute_inline(block=b0)
        sch.annotate(block_or_loop=b1, ann_key="meta_schedule.tiling_structure", ann_val="SSRSRS")
        l22, l23, l24, l25, l26, l27, l28, l29, l30, l31 = sch.get_loops(block=b1)
        l32, l33 = sch.split(loop=l31, factors=[None, 4], preserve_unit_iters=True)
        l34, l35 = sch.split(loop=l26, factors=[None, 16], preserve_unit_iters=True)
        l36, l37, l38, l39, l40, l41, l42, l43, l44, l45, l46, l47 = sch.get_loops(block=b1)
        sch.reorder(l42, l43, l44, l45, l46, l35, l33)
        b48 = sch.blockize(target=l35)
        sch.annotate(block_or_loop=b48, ann_key="meta_schedule.auto_tensorize", ann_val=VNNI_INTRIN)
        l49, l50, l51, l52, l53, l54, l55, l56, l57, l58 = sch.get_loops(block=b48)
        v59, v60, v61, v62 = sch.sample_perfect_tile(
            loop=l49, n=4, max_innermost_factor=64, decision=[1, 1, 1, 1]
        )
        l63, l64, l65, l66 = sch.split(
            loop=l49, factors=[v59, v60, v61, v62], preserve_unit_iters=True
        )
        v67, v68, v69, v70 = sch.sample_perfect_tile(
            loop=l50, n=4, max_innermost_factor=64, decision=[4, 32, 1, 1]
        )
        l71, l72, l73, l74 = sch.split(
            loop=l50, factors=[v67, v68, v69, v70], preserve_unit_iters=True
        )
        v75, v76, v77, v78 = sch.sample_perfect_tile(
            loop=l51, n=4, max_innermost_factor=64, decision=[1, 7, 1, 1]
        )
        l79, l80, l81, l82 = sch.split(
            loop=l51, factors=[v75, v76, v77, v78], preserve_unit_iters=True
        )
        v83, v84, v85, v86 = sch.sample_perfect_tile(
            loop=l52, n=4, max_innermost_factor=64, decision=[1, 1, 1, 7]
        )
        l87, l88, l89, l90 = sch.split(
            loop=l52, factors=[v83, v84, v85, v86], preserve_unit_iters=True
        )
        v91, v92, v93, v94 = sch.sample_perfect_tile(
            loop=l53, n=4, max_innermost_factor=64, decision=[1, 1, 1, 1]
        )
        l95, l96, l97, l98 = sch.split(
            loop=l53, factors=[v91, v92, v93, v94], preserve_unit_iters=True
        )
        v99, v100 = sch.sample_perfect_tile(loop=l54, n=2, max_innermost_factor=64, decision=[1, 1])
        l101, l102 = sch.split(loop=l54, factors=[v99, v100], preserve_unit_iters=True)
        v103, v104 = sch.sample_perfect_tile(
            loop=l55, n=2, max_innermost_factor=64, decision=[1, 1]
        )
        l105, l106 = sch.split(loop=l55, factors=[v103, v104], preserve_unit_iters=True)
        v107, v108 = sch.sample_perfect_tile(
            loop=l56, n=2, max_innermost_factor=64, decision=[4, 8]
        )
        l109, l110 = sch.split(loop=l56, factors=[v107, v108], preserve_unit_iters=True)
        v111, v112 = sch.sample_perfect_tile(
            loop=l57, n=2, max_innermost_factor=64, decision=[4, 1]
        )
        l113, l114 = sch.split(loop=l57, factors=[v111, v112], preserve_unit_iters=True)
        v115, v116 = sch.sample_perfect_tile(
            loop=l58, n=2, max_innermost_factor=64, decision=[1, 1]
        )
        l117, l118 = sch.split(loop=l58, factors=[v115, v116], preserve_unit_iters=True)
        sch.reorder(
            l63,
            l71,
            l79,
            l87,
            l95,
            l64,
            l72,
            l80,
            l88,
            l96,
            l101,
            l105,
            l109,
            l113,
            l117,
            l65,
            l73,
            l81,
            l89,
            l97,
            l102,
            l106,
            l110,
            l114,
            l118,
            l66,
            l74,
            l82,
            l90,
            l98,
        )
        (b119,) = sch.get_consumers(block=b48)
        sch.reverse_compute_at(block=b119, loop=l96, preserve_unit_loops=True, index=-1)
        sch.annotate(block_or_loop=b21, ann_key="meta_schedule.parallel", ann_val=96)
        sch.annotate(block_or_loop=b21, ann_key="meta_schedule.vectorize", ann_val=64)
        v120 = sch.sample_categorical(
            candidates=[0, 16, 64, 512], probs=[0.25, 0.25, 0.25, 0.25], decision=2
        )
        sch.annotate(block_or_loop=b21, ann_key="meta_schedule.unroll_explicit", ann_val=v120)
        sch.enter_postproc()
        b121 = sch.get_block(name="root", func_name="main")
        sch.unannotate(block_or_loop=b121, ann_key="meta_schedule.parallel")
        sch.unannotate(block_or_loop=b121, ann_key="meta_schedule.vectorize")
        sch.unannotate(block_or_loop=b121, ann_key="meta_schedule.unroll_explicit")
        b122, b123 = sch.get_child_blocks(b121)
        (
            l124,
            l125,
            l126,
            l127,
            l128,
            l129,
            l130,
            l131,
            l132,
            l133,
            l134,
            l135,
            l136,
            l137,
            l138,
            l139,
            l140,
            l141,
            l142,
            l143,
            l144,
            l145,
            l146,
            l147,
            l148,
            l149,
            l150,
            l151,
            l152,
            l153,
        ) = sch.get_loops(block=b122)
        l154 = sch.fuse(l124, l125, l126, l127, l128, l129, l130, preserve_unit_iters=True)
        sch.parallel(loop=l154)
        sch.annotate(block_or_loop=l154, ann_key="pragma_auto_unroll_max_step", ann_val=64)
        sch.annotate(block_or_loop=l154, ann_key="pragma_unroll_explicit", ann_val=1)
        l155, l156, l157, l158, l159, l160, l161, l162, l163 = sch.get_loops(block=b123)
        l164 = sch.fuse(l163, preserve_unit_iters=True)
        sch.vectorize(loop=l164)
        sch.annotate(block_or_loop=l155, ann_key="pragma_auto_unroll_max_step", ann_val=64)
        sch.annotate(block_or_loop=l155, ann_key="pragma_unroll_explicit", ann_val=1)
        b165 = sch.get_block(name="conv2d_NCHWc_int8_o", func_name="main")
        (
            l166,
            l167,
            l168,
            l169,
            l170,
            l171,
            l172,
            l173,
            l174,
            l175,
            l176,
            l177,
            l178,
            l179,
            l180,
            l181,
            l182,
            l183,
            l184,
            l185,
            l186,
            l187,
            l188,
            l189,
        ) = sch.get_loops(block=b165)
        b190 = sch.decompose_reduction(block=b165, loop=l170)
        sch.unannotate(block_or_loop=b190, ann_key="meta_schedule.auto_tensorize")
        sch.annotate(block_or_loop=b190, ann_key="meta_schedule.auto_tensorize", ann_val="")
        b191 = sch.get_block(name="conv2d_NCHWc_int8_o_init", func_name="main")
        sch.unannotate(block_or_loop=b191, ann_key="meta_schedule.auto_tensorize")
        (b192,) = sch.get_child_blocks(b191)
        (l193,) = sch.get_loops(block=b192)
        sch.vectorize(loop=l193)
        b194 = sch.get_block(name="conv2d_NCHWc_int8_o_update", func_name="main")
        sch.unannotate(block_or_loop=b194, ann_key="meta_schedule.auto_tensorize")
        sch.tensorize(block_or_loop=b194, tensor_intrin=VNNI_INTRIN)

    vnni_id = llvm_lookup_intrinsic_id("llvm.x86.avx512.vpdpbusd.512")
    verify(
        Conv2dInt8_NCHWc,
        apply_trace,
        Conv2dInt8_NCHWc_target,
        "llvm -mcpu=cascadelake",
        get_conv2d_vnni_mod(vnni_id),
    )


def test_winograd_gpu():
    def apply_trace(sch):
        b0 = sch.get_block(name="B", func_name="main")
        b1 = sch.get_block(name="data_pack", func_name="main")
        b2 = sch.get_block(name="bgemm", func_name="main")
        b3 = sch.get_block(name="A", func_name="main")
        b4 = sch.get_block(name="inverse", func_name="main")
        b5 = sch.get_block(name="conv2d_winograd", func_name="main")
        b6 = sch.get_block(name="T_add", func_name="main")
        b7 = sch.get_block(name="T_relu", func_name="main")
        b8 = sch.get_block(name="root", func_name="main")
        sch.compute_inline(block=b0)
        (b9,) = sch.get_producers(block=b1)
        (b10,) = sch.get_producers(block=b9)
        l11, l12, l13, l14, l15, l16 = sch.get_loops(block=b1)
        v17, v18 = sch.sample_perfect_tile(
            loop=l13, n=2, max_innermost_factor=64, decision=[14, 14]
        )
        l19, l20 = sch.split(loop=l13, factors=[v17, v18], preserve_unit_iters=True)
        v21, v22 = sch.sample_perfect_tile(loop=l14, n=2, max_innermost_factor=64, decision=[8, 8])
        l23, l24 = sch.split(loop=l14, factors=[v21, v22], preserve_unit_iters=True)
        sch.unroll(loop=l11)
        sch.unroll(loop=l12)
        sch.unroll(loop=l15)
        sch.unroll(loop=l16)
        sch.reorder(l19, l23, l20, l24, l11, l12, l15, l16)
        sch.compute_at(block=b9, loop=l24, preserve_unit_loops=True, index=-1)
        sch.set_scope(block=b9, buffer_index=0, storage_scope="local")
        sch.compute_inline(block=b10)
        l25, l26, l27, l28, l29, l30, l31, l32 = sch.get_loops(block=b1)
        l33 = sch.fuse(l25, l26, l27, l28, preserve_unit_iters=True)
        v34 = sch.sample_categorical(
            candidates=[32, 64, 128, 256, 512, 1024],
            probs=[
                0.16666666666666666,
                0.16666666666666666,
                0.16666666666666666,
                0.16666666666666666,
                0.16666666666666666,
                0.16666666666666666,
            ],
            decision=2,
        )
        l35, l36 = sch.split(loop=l33, factors=[None, v34], preserve_unit_iters=True)
        sch.bind(loop=l35, thread_axis="blockIdx.x")
        sch.bind(loop=l36, thread_axis="threadIdx.x")
        sch.compute_inline(block=b3)
        l37, l38, l39, l40, l41, l42 = sch.get_loops(block=b4)
        v43, v44 = sch.sample_perfect_tile(loop=l39, n=2, max_innermost_factor=64, decision=[28, 7])
        l45, l46 = sch.split(loop=l39, factors=[v43, v44], preserve_unit_iters=True)
        v47, v48 = sch.sample_perfect_tile(loop=l40, n=2, max_innermost_factor=64, decision=[2, 32])
        l49, l50 = sch.split(loop=l40, factors=[v47, v48], preserve_unit_iters=True)
        sch.unroll(loop=l37)
        sch.unroll(loop=l38)
        sch.unroll(loop=l41)
        sch.unroll(loop=l42)
        sch.reorder(l45, l49, l46, l50, l37, l38, l41, l42)
        l51, l52, l53, l54, l55, l56, l57, l58 = sch.get_loops(block=b4)
        l59 = sch.fuse(l51, l52, l53, l54, preserve_unit_iters=True)
        v60 = sch.sample_categorical(
            candidates=[32, 64, 128, 256, 512, 1024],
            probs=[
                0.16666666666666666,
                0.16666666666666666,
                0.16666666666666666,
                0.16666666666666666,
                0.16666666666666666,
                0.16666666666666666,
            ],
            decision=4,
        )
        l61, l62 = sch.split(loop=l59, factors=[None, v60], preserve_unit_iters=True)
        sch.bind(loop=l61, thread_axis="blockIdx.x")
        sch.bind(loop=l62, thread_axis="threadIdx.x")
        sch.annotate(block_or_loop=b2, ann_key="meta_schedule.tiling_structure", ann_val="SSSRRSRS")
        l63, l64, l65, l66, l67 = sch.get_loops(block=b2)
        v68, v69, v70, v71, v72 = sch.sample_perfect_tile(
            loop=l63, n=5, max_innermost_factor=64, decision=[1, 2, 3, 1, 1]
        )
        l73, l74, l75, l76, l77 = sch.split(
            loop=l63, factors=[v68, v69, v70, v71, v72], preserve_unit_iters=True
        )
        v78, v79, v80, v81, v82 = sch.sample_perfect_tile(
            loop=l64, n=5, max_innermost_factor=64, decision=[6, 1, 1, 1, 1]
        )
        l83, l84, l85, l86, l87 = sch.split(
            loop=l64, factors=[v78, v79, v80, v81, v82], preserve_unit_iters=True
        )
        v88, v89, v90, v91, v92 = sch.sample_perfect_tile(
            loop=l65, n=5, max_innermost_factor=64, decision=[7, 2, 1, 14, 1]
        )
        l93, l94, l95, l96, l97 = sch.split(
            loop=l65, factors=[v88, v89, v90, v91, v92], preserve_unit_iters=True
        )
        v98, v99, v100, v101, v102 = sch.sample_perfect_tile(
            loop=l66, n=5, max_innermost_factor=64, decision=[4, 1, 16, 1, 1]
        )
        l103, l104, l105, l106, l107 = sch.split(
            loop=l66, factors=[v98, v99, v100, v101, v102], preserve_unit_iters=True
        )
        v108, v109, v110 = sch.sample_perfect_tile(
            loop=l67, n=3, max_innermost_factor=64, decision=[2, 2, 16]
        )
        l111, l112, l113 = sch.split(loop=l67, factors=[v108, v109, v110], preserve_unit_iters=True)
        sch.reorder(
            l73,
            l83,
            l93,
            l103,
            l74,
            l84,
            l94,
            l104,
            l75,
            l85,
            l95,
            l105,
            l111,
            l112,
            l76,
            l86,
            l96,
            l106,
            l113,
            l77,
            l87,
            l97,
            l107,
        )
        l114 = sch.fuse(l73, l83, l93, l103, preserve_unit_iters=True)
        sch.bind(loop=l114, thread_axis="blockIdx.x")
        l115 = sch.fuse(l74, l84, l94, l104, preserve_unit_iters=True)
        sch.bind(loop=l115, thread_axis="vthread.x")
        l116 = sch.fuse(l75, l85, l95, l105, preserve_unit_iters=True)
        sch.bind(loop=l116, thread_axis="threadIdx.x")
        sch.annotate(
            block_or_loop=b2, ann_key="meta_schedule.thread_extent_low_inclusive", ann_val=32
        )
        sch.annotate(
            block_or_loop=b2, ann_key="meta_schedule.thread_extent_high_inclusive", ann_val=1024
        )
        b117 = sch.cache_write(block=b2, write_buffer_index=0, storage_scope="local")
        sch.reverse_compute_at(block=b117, loop=l116, preserve_unit_loops=True, index=-1)
        b118 = sch.cache_read(
            block=b2, read_buffer_index=0, storage_scope="shared", consumer_blocks=[b2]
        )
        sch.compute_at(block=b118, loop=l111, preserve_unit_loops=True, index=-1)
        l119, l120, l121, l122, l123, l124, l125, l126 = sch.get_loops(block=b118)
        l127 = sch.fuse(l123, l124, l125, l126, preserve_unit_iters=True)
        v128 = sch.sample_categorical(
            candidates=[1, 2, 3, 4], probs=[0.25, 0.25, 0.25, 0.25], decision=3
        )
        sch.annotate(block_or_loop=b118, ann_key="meta_schedule.cooperative_fetch", ann_val=v128)
        b129 = sch.cache_read(
            block=b2, read_buffer_index=1, storage_scope="shared", consumer_blocks=[b2]
        )
        sch.compute_at(block=b129, loop=l111, preserve_unit_loops=True, index=-1)
        l130, l131, l132, l133, l134, l135, l136, l137 = sch.get_loops(block=b129)
        l138 = sch.fuse(l134, l135, l136, l137, preserve_unit_iters=True)
        v139 = sch.sample_categorical(
            candidates=[1, 2, 3, 4], probs=[0.25, 0.25, 0.25, 0.25], decision=3
        )
        sch.annotate(block_or_loop=b129, ann_key="meta_schedule.cooperative_fetch", ann_val=v139)
        sch.reverse_compute_inline(block=b7)
        sch.reverse_compute_inline(block=b6)
        v140 = sch.sample_categorical(
            candidates=[0, 16, 64, 512, 1024],
            probs=[
                0.20000000000000001,
                0.20000000000000001,
                0.20000000000000001,
                0.20000000000000001,
                0.20000000000000001,
            ],
            decision=4,
        )
        sch.annotate(block_or_loop=b8, ann_key="meta_schedule.unroll_explicit", ann_val=v140)
        l141, l142, l143, l144 = sch.get_loops(block=b5)
        l145 = sch.fuse(l141, l142, l143, l144, preserve_unit_iters=True)
        v146 = sch.sample_categorical(
            candidates=[32, 64, 128, 256, 512, 1024],
            probs=[
                0.16666666666666666,
                0.16666666666666666,
                0.16666666666666666,
                0.16666666666666666,
                0.16666666666666666,
                0.16666666666666666,
            ],
            decision=2,
        )
        l147, l148 = sch.split(loop=l145, factors=[None, v146], preserve_unit_iters=True)
        sch.bind(loop=l147, thread_axis="blockIdx.x")
        sch.bind(loop=l148, thread_axis="threadIdx.x")
        sch.enter_postproc()
        sch.unannotate(block_or_loop=b118, ann_key="meta_schedule.cooperative_fetch")
        l149, l150, l151, l152, l153 = sch.get_loops(block=b118)
        l154, l155, l156 = sch.split(loop=l153, factors=[None, 48, 4], preserve_unit_iters=True)
        sch.vectorize(loop=l156)
        sch.bind(loop=l155, thread_axis="threadIdx.x")
        sch.unannotate(block_or_loop=b129, ann_key="meta_schedule.cooperative_fetch")
        l157, l158, l159, l160, l161 = sch.get_loops(block=b129)
        l162, l163, l164 = sch.split(loop=l161, factors=[None, 48, 4], preserve_unit_iters=True)
        sch.vectorize(loop=l164)
        sch.bind(loop=l163, thread_axis="threadIdx.x")
        b165 = sch.get_block(name="root", func_name="main")
        sch.unannotate(block_or_loop=b165, ann_key="meta_schedule.unroll_explicit")
        b166, b167, b168, b169, b170, b171, b172, b173 = sch.get_child_blocks(b165)
        l174, l175, l176, l177, l178, l179 = sch.get_loops(block=b166)
        sch.annotate(block_or_loop=l174, ann_key="pragma_auto_unroll_max_step", ann_val=1024)
        sch.annotate(block_or_loop=l174, ann_key="pragma_unroll_explicit", ann_val=1)
        l180, l181, l182, l183, l184, l185 = sch.get_loops(block=b167)
        sch.annotate(block_or_loop=l180, ann_key="pragma_auto_unroll_max_step", ann_val=1024)
        sch.annotate(block_or_loop=l180, ann_key="pragma_unroll_explicit", ann_val=1)
        l186, l187, l188, l189, l190, l191, l192 = sch.get_loops(block=b168)
        sch.annotate(block_or_loop=l186, ann_key="pragma_auto_unroll_max_step", ann_val=1024)
        sch.annotate(block_or_loop=l186, ann_key="pragma_unroll_explicit", ann_val=1)
        l193, l194, l195, l196, l197, l198, l199 = sch.get_loops(block=b169)
        sch.annotate(block_or_loop=l193, ann_key="pragma_auto_unroll_max_step", ann_val=1024)
        sch.annotate(block_or_loop=l193, ann_key="pragma_unroll_explicit", ann_val=1)
        (
            l200,
            l201,
            l202,
            l203,
            l204,
            l205,
            l206,
            l207,
            l208,
            l209,
            l210,
            l211,
            l212,
            l213,
        ) = sch.get_loops(block=b170)
        sch.annotate(block_or_loop=l200, ann_key="pragma_auto_unroll_max_step", ann_val=1024)
        sch.annotate(block_or_loop=l200, ann_key="pragma_unroll_explicit", ann_val=1)
        l214, l215, l216, l217, l218, l219, l220 = sch.get_loops(block=b171)
        sch.annotate(block_or_loop=l214, ann_key="pragma_auto_unroll_max_step", ann_val=1024)
        sch.annotate(block_or_loop=l214, ann_key="pragma_unroll_explicit", ann_val=1)
        l221, l222, l223, l224, l225, l226 = sch.get_loops(block=b172)
        sch.annotate(block_or_loop=l221, ann_key="pragma_auto_unroll_max_step", ann_val=1024)
        sch.annotate(block_or_loop=l221, ann_key="pragma_unroll_explicit", ann_val=1)
        l227, l228 = sch.get_loops(block=b173)
        sch.annotate(block_or_loop=l227, ann_key="pragma_auto_unroll_max_step", ann_val=1024)
        sch.annotate(block_or_loop=l227, ann_key="pragma_unroll_explicit", ann_val=1)
        b229 = sch.get_block(name="data_pack", func_name="main")
        l230, l231, l232, l233, l234, l235 = sch.get_loops(block=b229)
        b236 = sch.decompose_reduction(block=b229, loop=l234)
        b237 = sch.get_block(name="bgemm", func_name="main")
        (
            l238,
            l239,
            l240,
            l241,
            l242,
            l243,
            l244,
            l245,
            l246,
            l247,
            l248,
            l249,
            l250,
            l251,
        ) = sch.get_loops(block=b237)
        b252 = sch.decompose_reduction(block=b237, loop=l241)
        b253 = sch.get_block(name="inverse", func_name="main")
        l254, l255, l256, l257, l258, l259 = sch.get_loops(block=b253)
        b260 = sch.decompose_reduction(block=b253, loop=l258)

    verify(
        Conv2dWinogradAddRelu,
        apply_trace,
        Conv2dWinogradAddResidualRelu,
        "cuda",
        Conv2dWinogradAddResidualRelu_scheduled,
    )


def test_inline_order():
    # In this test, the order of applying AutoInline is tested.
    # We need to make sure that the last block in Conv2dInt8_with_predicate_target,
    # "compute_4", is AutoInline-ed after all other blocks have been processed.
    #
    # Otherwise, if the order is "T_add_2" -> "compute_4" -> "compute_3", "compute_4" is neither
    # inlined (because this is the last block) nor reverse-inlined
    # (because it has multiple producers). This results in the "compute_4" block being
    # reverse-inlined at the very end of ScheduleUsingAnchorTrace, where its producer block
    # "conv2d_nhwc_reindex_shared" has the predicate
    # T.where(((ax1_0 * 4 + ax1_1) * 32 + ax1_2) * 2 + ax1_3 < 64) due to anchor-block scheduling
    # (see Conv2dInt8_with_predicate_scheduled). ReverseComputeInline cannot be applied in
    # such cases.

    def apply_trace(sch: Schedule) -> None:
        b0 = sch.get_block(name="pad_temp", func_name="main")
        b1 = sch.get_block(name="conv2d_nhwc", func_name="main")
        b2 = sch.get_block(name="T_subtract", func_name="main")
        b3 = sch.get_block(name="T_add", func_name="main")
        b4 = sch.get_block(name="compute", func_name="main")
        b5 = sch.get_block(name="T_add_1", func_name="main")
        b6 = sch.get_block(name="compute_1", func_name="main")
        b7 = sch.get_block(name="T_subtract_1", func_name="main")
        b8 = sch.get_block(name="compute_2", func_name="main")
        b9 = sch.get_block(name="root", func_name="main")
        sch.annotate(block_or_loop=b1, ann_key="meta_schedule.tiling_structure", ann_val="SSSRRSRS")
        b10 = sch.reindex(block=b1, buffer=("write", 0))
        b11 = sch.reindex(block=b1, buffer=("read", 0))
        b12 = sch.reindex(block=b1, buffer=("read", 1))
        sch.transform_layout(
            block=b1,
            buffer=("read", 0),
            index_map=lambda nn, yy, xx, rc: (
                (((nn * 3136) + (yy * 56)) + xx),
                rc,
            ),
            pad_value=None,
        )
        sch.transform_layout(
            block=b1,
            buffer=("read", 1),
            index_map=lambda ff, ry, rx, rc: (
                ry,
                rx,
                ff,
                rc,
            ),
            pad_value=None,
        )
        sch.transform_layout(
            block=b1,
            buffer=("write", 0),
            index_map=lambda nn, yy, xx, ff: (
                (((nn * 3136) + (yy * 56)) + xx),
                ff,
            ),
            pad_value=None,
        )
        sch.transform_block_layout(
            block=b10,
            index_map=lambda nn, yy, xx, ff: (
                (((nn * 3136) + (yy * 56)) + xx),
                ff,
            ),
        )
        sch.transform_block_layout(
            block=b11,
            index_map=lambda nn, yy, xx, rc: (
                (((nn * 3136) + (yy * 56)) + xx),
                rc,
            ),
        )
        sch.transform_block_layout(
            block=b12,
            index_map=lambda ff, ry, rx, rc: (
                ry,
                rx,
                ff,
                rc,
            ),
        )
        sch.transform_block_layout(
            block=b1,
            index_map=lambda nn, yy, xx, ff, ry, rx, rc: (
                ry,
                rx,
                (((nn * 3136) + (yy * 56)) + xx),
                ff,
                rc,
            ),
        )
        l13, l14, l15, l16, l17 = sch.get_loops(block=b1)
        l18, l19 = sch.split(loop=l17, factors=[None, 16], preserve_unit_iters=True)
        l20, l21 = sch.split(loop=l16, factors=[None, 16], preserve_unit_iters=True)
        l22, l23 = sch.split(loop=l15, factors=[None, 16], preserve_unit_iters=True)
        l24, l25, l26, l27, l28, l29, l30, l31 = sch.get_loops(block=b1)
        sch.reorder(l28, l30, l23, l21, l19)
        b32 = sch.blockize(target=l23)
        sch.annotate(
            block_or_loop=b32,
            ann_key="meta_schedule.auto_tensorize",
            ann_val="wmma_sync_16x16x16_s8s8s32_trans",
        )
        sch.annotate(
            block_or_loop=b32,
            ann_key="meta_schedule.auto_tensorize_init",
            ann_val="wmma_fill_16x16x16_s32",
        )
        sch.annotate(block_or_loop=b32, ann_key="warp_execution", ann_val=1)
        l33, l34, l35, l36, l37 = sch.get_loops(block=b32)
        v38, v39, v40 = sch.sample_perfect_tile(
            loop=l33, n=3, max_innermost_factor=4, decision=[1, 1, 1]
        )
        l41, l42, l43 = sch.split(loop=l33, factors=[v38, v39, v40], preserve_unit_iters=True)
        v44, v45, v46 = sch.sample_perfect_tile(
            loop=l34, n=3, max_innermost_factor=4, decision=[1, 1, 1]
        )
        l47, l48, l49 = sch.split(loop=l34, factors=[v44, v45, v46], preserve_unit_iters=True)
        v50, v51, v52, v53, v54 = sch.sample_perfect_tile(
            loop=l35, n=5, max_innermost_factor=4, decision=[8, 196, 2, 1, 1]
        )
        l55, l56, l57, l58, l59 = sch.split(
            loop=l35, factors=[v50, v51, v52, v53, v54], preserve_unit_iters=True
        )
        v60, v61, v62, v63, v64 = sch.sample_perfect_tile(
            loop=l36, n=5, max_innermost_factor=4, decision=[4, 1, 2, 1, 2]
        )
        l65, l66, l67, l68, l69 = sch.split(
            loop=l36, factors=[v60, v61, v62, v63, v64], preserve_unit_iters=True
        )
        v70, v71, v72 = sch.sample_perfect_tile(
            loop=l37, n=3, max_innermost_factor=4, decision=[2, 2, 1]
        )
        l73, l74, l75 = sch.split(loop=l37, factors=[v70, v71, v72], preserve_unit_iters=True)
        sch.reorder(
            l55,
            l65,
            l56,
            l66,
            l57,
            l67,
            l41,
            l47,
            l73,
            l42,
            l48,
            l74,
            l58,
            l68,
            l43,
            l49,
            l75,
            l59,
            l69,
        )
        l76 = sch.fuse(l55, l65, preserve_unit_iters=True)
        sch.bind(loop=l76, thread_axis="blockIdx.y")
        l77 = sch.fuse(l56, l66, preserve_unit_iters=True)
        sch.bind(loop=l77, thread_axis="blockIdx.x")
        l78 = sch.fuse(l57, l67, preserve_unit_iters=True)
        sch.bind(loop=l78, thread_axis="threadIdx.y")
        sch.annotate(
            block_or_loop=b32, ann_key="meta_schedule.thread_extent_low_inclusive", ann_val=32
        )
        sch.annotate(
            block_or_loop=b32, ann_key="meta_schedule.thread_extent_high_inclusive", ann_val=1024
        )
        b79 = sch.cache_write(block=b32, write_buffer_index=0, storage_scope="shared")
        sch.reverse_compute_at(block=b79, loop=l77, preserve_unit_loops=True, index=-1)
        b80 = sch.cache_write(block=b32, write_buffer_index=0, storage_scope="wmma.accumulator")
        sch.reverse_compute_at(block=b80, loop=l78, preserve_unit_loops=True, index=-1)
        v81 = sch.sample_categorical(
            candidates=[1, 2, 3, 4, 8, 16],
            probs=[
                0.16666666666666666,
                0.16666666666666666,
                0.16666666666666666,
                0.16666666666666666,
                0.16666666666666666,
                0.16666666666666666,
            ],
            decision=1,
        )
        sch.annotate(block_or_loop=b79, ann_key="meta_schedule.cooperative_fetch", ann_val=v81)
        sch.reverse_compute_inline(block=b10)
        l82, l83, l84, l85, l86 = sch.get_loops(block=b80)
        l87, l88 = sch.split(loop=l86, factors=[None, 16], preserve_unit_iters=True)
        l89, l90 = sch.split(loop=l85, factors=[None, 16], preserve_unit_iters=True)
        l91, l92, l93, l94, l95, l96, l97 = sch.get_loops(block=b80)
        sch.reorder(l96, l90, l88)
        b98 = sch.blockize(target=l90)
        sch.annotate(
            block_or_loop=b98,
            ann_key="meta_schedule.auto_tensorize",
            ann_val="wmma_store_16x16x16_s32_shared",
        )
        b99 = sch.cache_read(
            block=b32, read_buffer_index=0, storage_scope="shared", consumer_blocks=[b32]
        )
        sch.compute_at(block=b99, loop=l73, preserve_unit_loops=True, index=-1)
        l100, l101, l102, l103, l104, l105, l106, l107 = sch.get_loops(block=b99)
        l108 = sch.fuse(l106, l107, preserve_unit_iters=True)
        v109 = sch.sample_categorical(
            candidates=[1, 2, 3, 4, 8, 16],
            probs=[
                0.16666666666666666,
                0.16666666666666666,
                0.16666666666666666,
                0.16666666666666666,
                0.16666666666666666,
                0.16666666666666666,
            ],
            decision=3,
        )
        sch.annotate(block_or_loop=b99, ann_key="meta_schedule.cooperative_fetch", ann_val=v109)
        b110 = sch.cache_read(
            block=b32, read_buffer_index=1, storage_scope="shared", consumer_blocks=[b32]
        )
        sch.compute_at(block=b110, loop=l73, preserve_unit_loops=True, index=-1)
        l111, l112, l113, l114, l115, l116, l117, l118, l119, l120 = sch.get_loops(block=b110)
        l121 = sch.fuse(l117, l118, l119, l120, preserve_unit_iters=True)
        v122 = sch.sample_categorical(
            candidates=[1, 2, 3, 4, 8, 16],
            probs=[
                0.16666666666666666,
                0.16666666666666666,
                0.16666666666666666,
                0.16666666666666666,
                0.16666666666666666,
                0.16666666666666666,
            ],
            decision=2,
        )
        sch.annotate(block_or_loop=b110, ann_key="meta_schedule.cooperative_fetch", ann_val=v122)
        b123 = sch.cache_read(block=b32, read_buffer_index=0, storage_scope="wmma.matrix_a")
        sch.compute_at(block=b123, loop=l74, preserve_unit_loops=True, index=-1)
        l124, l125, l126, l127, l128, l129, l130, l131, l132, l133, l134 = sch.get_loops(block=b123)
        l135, l136 = sch.split(loop=l134, factors=[None, 16], preserve_unit_iters=True)
        l137, l138 = sch.split(loop=l133, factors=[None, 16], preserve_unit_iters=True)
        (
            l139,
            l140,
            l141,
            l142,
            l143,
            l144,
            l145,
            l146,
            l147,
            l148,
            l149,
            l150,
            l151,
        ) = sch.get_loops(block=b123)
        sch.reorder(l150, l138, l136)
        b152 = sch.blockize(target=l138)
        sch.annotate(
            block_or_loop=b152,
            ann_key="meta_schedule.auto_tensorize",
            ann_val="wmma_load_16x16x16_s8_a_shared",
        )
        b153 = sch.cache_read(block=b32, read_buffer_index=1, storage_scope="wmma.matrix_b")
        sch.compute_at(block=b153, loop=l74, preserve_unit_loops=True, index=-1)
        (
            l154,
            l155,
            l156,
            l157,
            l158,
            l159,
            l160,
            l161,
            l162,
            l163,
            l164,
            l165,
            l166,
        ) = sch.get_loops(block=b153)
        l167, l168 = sch.split(loop=l166, factors=[None, 16], preserve_unit_iters=True)
        l169, l170 = sch.split(loop=l165, factors=[None, 16], preserve_unit_iters=True)
        (
            l171,
            l172,
            l173,
            l174,
            l175,
            l176,
            l177,
            l178,
            l179,
            l180,
            l181,
            l182,
            l183,
            l184,
            l185,
        ) = sch.get_loops(block=b153)
        sch.reorder(l184, l170, l168)
        b186 = sch.blockize(target=l170)
        sch.annotate(
            block_or_loop=b186,
            ann_key="meta_schedule.auto_tensorize",
            ann_val="wmma_load_16x16x16_s8_b_trans_shared",
        )
        sch.compute_inline(block=b11)
        sch.compute_inline(block=b12)
        sch.storage_align(block=b99, buffer_index=0, axis=-2, factor=32, offset=16)
        sch.storage_align(block=b110, buffer_index=0, axis=-2, factor=32, offset=16)
        sch.reverse_compute_inline(block=b8)
        sch.reverse_compute_inline(block=b7)
        sch.reverse_compute_inline(block=b6)
        sch.reverse_compute_inline(block=b5)
        sch.reverse_compute_inline(block=b4)
        sch.reverse_compute_inline(block=b3)
        sch.reverse_compute_inline(block=b2)
        sch.compute_inline(block=b0)

        v187 = sch.sample_categorical(
            candidates=[0, 16, 64, 512, 1024],
            probs=[
                0.20000000000000001,
                0.20000000000000001,
                0.20000000000000001,
                0.20000000000000001,
                0.20000000000000001,
            ],
            decision=4,
        )
        sch.annotate(block_or_loop=b9, ann_key="meta_schedule.unroll_explicit", ann_val=v187)
        sch.enter_postproc()
        sch.unannotate(block_or_loop=b79, ann_key="meta_schedule.cooperative_fetch")
        l188, l189, l190, l191 = sch.get_loops(block=b79)

        l192, l193, l194, l195 = sch.split(
            loop=l191, factors=[None, 4, 32, 2], preserve_unit_iters=True
        )

    verify(
        Conv2dInt8_with_predicate,
        apply_trace,
        Conv2dInt8_with_predicate_target,
        "cuda",
        Conv2dInt8_with_predicate_scheduled,
    )


if __name__ == "__main__":
    tvm.testing.main()
