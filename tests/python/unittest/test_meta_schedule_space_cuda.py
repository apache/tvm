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
"""Tests for MetaSchedule search space on CUDA"""
from tvm import meta_schedule as ms
from tvm.meta_schedule.testing.space_generation import check_sketches, print_sketches
from tvm.meta_schedule.testing.te_workload import create_te_workload
from tvm.script import tir as T
from tvm.target import Target


def _target():
    return Target("nvidia/geforce-rtx-3070")


def test_cuda_c1d():
    # fmt: off
    @T.prim_func
    def c1d_0(inputs: T.Buffer[(1, 256, 64), "float32"], weight: T.Buffer[(3, 64, 128), "float32"], conv1d_nlc: T.Buffer[(1, 128, 128), "float32"]) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # body
        with T.block("root"):
            T.reads()
            T.writes()
            T.block_attr({"meta_schedule.unroll_explicit":16})
            conv1d_nlc_local = T.alloc_buffer([1, 128, 128], dtype="float32", scope="local")
            PadInput_shared = T.alloc_buffer([1, 258, 64], dtype="float32", scope="shared")
            weight_shared = T.alloc_buffer([3, 64, 128], dtype="float32", scope="shared")
            for i0_0_i1_0_i2_0_fused in T.thread_binding(4, thread="blockIdx.x"):
                for i0_1_i1_1_i2_1_fused in T.thread_binding(16, thread="vthread.x"):
                    for i0_2_i1_2_i2_2_fused in T.thread_binding(4, thread="threadIdx.x"):
                        for i3_0, i4_0 in T.grid(1, 16):
                            for ax0_ax1_ax2_fused in T.serial(260):
                                with T.block("PadInput_shared"):
                                    v0 = T.axis.spatial(1, 0)
                                    v1 = T.axis.spatial(258, i0_0_i1_0_i2_0_fused * 64 + ax0_ax1_ax2_fused // 4)
                                    v2 = T.axis.spatial(64, i4_0 * 4 + ax0_ax1_ax2_fused % 4)
                                    T.reads(inputs[v0, v1 - 1, v2])
                                    T.writes(PadInput_shared[v0, v1, v2])
                                    T.block_attr({"meta_schedule.cooperative_fetch":4})
                                    PadInput_shared[v0, v1, v2] = T.if_then_else(1 <= v1 and v1 < 257, inputs[v0, v1 - 1, v2], T.float32(0), dtype="float32")
                            for ax0_ax1_ax2_fused in T.serial(1536):
                                with T.block("weight_shared"):
                                    v0 = T.axis.spatial(3, ax0_ax1_ax2_fused // 512)
                                    v1 = T.axis.spatial(64, i4_0 * 4 + ax0_ax1_ax2_fused % 512 // 128)
                                    v2 = T.axis.spatial(128, ax0_ax1_ax2_fused % 128)
                                    T.reads(weight[v0, v1, v2])
                                    T.writes(weight_shared[v0, v1, v2])
                                    T.block_attr({"meta_schedule.cooperative_fetch":3})
                                    weight_shared[v0, v1, v2] = weight[v0, v1, v2]
                            for i3_1, i4_1, i0_3, i1_3, i2_3, i3_2, i4_2, i0_4, i1_4, i2_4 in T.grid(1, 2, 1, 1, 2, 3, 2, 1, 4, 8):
                                with T.block("conv1d_nlc"):
                                    n = T.axis.spatial(1, i0_4 + i0_3)
                                    l = T.axis.spatial(128, i0_0_i1_0_i2_0_fused * 32 + i0_1_i1_1_i2_1_fused // 2 * 4 + i1_3 * 4 + i1_4)
                                    co = T.axis.spatial(128, i0_1_i1_1_i2_1_fused % 2 * 64 + i0_2_i1_2_i2_2_fused * 16 + i2_3 * 8 + i2_4)
                                    rl = T.axis.reduce(3, i3_0 * 3 + i3_1 * 3 + i3_2)
                                    rc = T.axis.reduce(64, i4_0 * 4 + i4_1 * 2 + i4_2)
                                    T.reads(PadInput_shared[n, l * 2 + rl, co // 128 * 64 + rc], weight_shared[rl, rc, co])
                                    T.writes(conv1d_nlc_local[n, l, co])
                                    T.block_attr({"meta_schedule.thread_extent_high_inclusive":1024, "meta_schedule.thread_extent_low_inclusive":32, "meta_schedule.tiling_structure":"SSSRRSRS"})
                                    with T.init():
                                        conv1d_nlc_local[n, l, co] = T.float32(0)
                                    conv1d_nlc_local[n, l, co] = conv1d_nlc_local[n, l, co] + PadInput_shared[n, l * 2 + rl, co // 128 * 64 + rc] * weight_shared[rl, rc, co]
                        for ax0, ax1, ax2 in T.grid(1, 4, 16):
                            with T.block("conv1d_nlc_local"):
                                v0 = T.axis.spatial(1, ax0)
                                v1 = T.axis.spatial(128, i0_0_i1_0_i2_0_fused * 32 + i0_1_i1_1_i2_1_fused // 2 * 4 + ax1)
                                v2 = T.axis.spatial(128, i0_1_i1_1_i2_1_fused % 2 * 64 + i0_2_i1_2_i2_2_fused * 16 + ax2)
                                T.reads(conv1d_nlc_local[v0, v1, v2])
                                T.writes(conv1d_nlc[v0, v1, v2])
                                conv1d_nlc[v0, v1, v2] = conv1d_nlc_local[v0, v1, v2]
    # fmt: on

    decision_0 = [
        ("SamplePerfectTile", [1, 1, 1, 1, 1]),
        ("SamplePerfectTile", [4, 8, 1, 1, 4]),
        ("SamplePerfectTile", [1, 2, 4, 2, 8]),
        ("SamplePerfectTile", [1, 1, 3]),
        ("SamplePerfectTile", [16, 2, 2]),
        ("SampleCategorical", 3),
        ("SampleCategorical", 2),
        ("SampleCategorical", 1),
    ]

    mod = create_te_workload("C1D", 0)
    actual = ms.TuneContext(
        mod=mod,
        target=_target(),
        space_generator=ms.space_generator.PostOrderApply(),
        sch_rules="default",
    ).generate_design_space()
    check_sketches(
        mod,
        sketches=actual,
        expected_mods=[c1d_0],
        expected_decisions=[decision_0],
    )


def test_cuda_c2d():
    # fmt: off
    @T.prim_func
    def c2d_0(inputs: T.Buffer[(1, 224, 224, 3), "float32"], weight: T.Buffer[(7, 7, 3, 64), "float32"], conv2d_nhwc: T.Buffer[(1, 112, 112, 64), "float32"]) -> None:
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        with T.block("root"):
            T.reads()
            T.writes()
            T.block_attr({"meta_schedule.unroll_explicit":16})
            conv2d_nhwc_local = T.alloc_buffer([1, 112, 112, 64], dtype="float32", scope="local")
            PadInput_shared = T.alloc_buffer([1, 230, 230, 3], dtype="float32", scope="shared")
            weight_shared = T.alloc_buffer([7, 7, 3, 64], dtype="float32", scope="shared")
            for i0_0_i1_0_i2_0_i3_0_fused in T.thread_binding(16, thread="blockIdx.x"):
                for i0_1_i1_1_i2_1_i3_1_fused in T.thread_binding(56, thread="vthread.x"):
                    for i0_2_i1_2_i2_2_i3_2_fused in T.thread_binding(14, thread="threadIdx.x"):
                        for i4_0, i5_0, i6_0 in T.grid(1, 1, 1):
                            for ax0_ax1_ax2_ax3_fused in T.serial(80379):
                                with T.block("PadInput_shared"):
                                    v0 = T.axis.spatial(1, 0)
                                    v1 = T.axis.spatial(230, ax0_ax1_ax2_ax3_fused % 80379 // 351)
                                    v2 = T.axis.spatial(230, i0_0_i1_0_i2_0_i3_0_fused // 8 * 112 + ax0_ax1_ax2_ax3_fused % 351 // 3)
                                    v3 = T.axis.spatial(3, ax0_ax1_ax2_ax3_fused % 3)
                                    T.reads(inputs[v0, v1 - 3, v2 - 3, v3])
                                    T.writes(PadInput_shared[v0, v1, v2, v3])
                                    T.block_attr({"meta_schedule.cooperative_fetch":2})
                                    PadInput_shared[v0, v1, v2, v3] = T.if_then_else(3 <= v1 and v1 < 227 and 3 <= v2 and v2 < 227, inputs[v0, v1 - 3, v2 - 3, v3], T.float32(0), dtype="float32")
                            for ax0_ax1_ax2_ax3_fused in T.serial(1176):
                                with T.block("weight_shared"):
                                    v0 = T.axis.spatial(7, ax0_ax1_ax2_ax3_fused // 168)
                                    v1 = T.axis.spatial(7, ax0_ax1_ax2_ax3_fused % 168 // 24)
                                    v2 = T.axis.spatial(3, ax0_ax1_ax2_ax3_fused % 24 // 8)
                                    v3 = T.axis.spatial(64, i0_0_i1_0_i2_0_i3_0_fused % 8 * 8 + ax0_ax1_ax2_ax3_fused % 8)
                                    T.reads(weight[v0, v1, v2, v3])
                                    T.writes(weight_shared[v0, v1, v2, v3])
                                    T.block_attr({"meta_schedule.cooperative_fetch":4})
                                    weight_shared[v0, v1, v2, v3] = weight[v0, v1, v2, v3]
                            for i4_1, i5_1, i6_1, i0_3, i1_3, i2_3, i3_3, i4_2, i5_2, i6_2, i0_4, i1_4, i2_4, i3_4 in T.grid(1, 7, 1, 1, 8, 4, 1, 7, 1, 3, 1, 1, 1, 2):
                                with T.block("conv2d_nhwc"):
                                    n = T.axis.spatial(1, i0_4 + i0_3 + 0 + 0 + 0)
                                    h = T.axis.spatial(112, ((0 + 0) * 14 + i0_2_i1_2_i2_2_i3_2_fused % 14) * 8 + i1_3 + i1_4)
                                    w = T.axis.spatial(112, (i0_0_i1_0_i2_0_i3_0_fused % 16 // 8 * 14 + i0_1_i1_1_i2_1_i3_1_fused % 56 // 4 + 0) * 4 + i2_3 + i2_4)
                                    co = T.axis.spatial(64, (i0_0_i1_0_i2_0_i3_0_fused % 8 * 4 + i0_1_i1_1_i2_1_i3_1_fused % 4 + 0 + i3_3) * 2 + i3_4)
                                    rh = T.axis.reduce(7, (i4_0 + i4_1) * 7 + i4_2)
                                    rw = T.axis.reduce(7, i5_0 * 7 + i5_1 + i5_2)
                                    rc = T.axis.reduce(3, (i6_0 + i6_1) * 3 + i6_2)
                                    T.reads(PadInput_shared[n, h * 2 + rh, w * 2 + rw, co // 64 * 3 + rc], weight_shared[rh, rw, rc, co])
                                    T.writes(conv2d_nhwc_local[n, h, w, co])
                                    T.block_attr({"meta_schedule.thread_extent_high_inclusive":1024, "meta_schedule.thread_extent_low_inclusive":32, "meta_schedule.tiling_structure":"SSSRRSRS"})
                                    with T.init():
                                        conv2d_nhwc_local[n, h, w, co] = T.float32(0)
                                    conv2d_nhwc_local[n, h, w, co] = conv2d_nhwc_local[n, h, w, co] + PadInput_shared[n, h * 2 + rh, w * 2 + rw, co // 64 * 3 + rc] * weight_shared[rh, rw, rc, co]
                        for ax0, ax1, ax2, ax3 in T.grid(1, 8, 4, 2):
                            with T.block("conv2d_nhwc_local"):
                                v0 = T.axis.spatial(1, ax0)
                                v1 = T.axis.spatial(112, i0_2_i1_2_i2_2_i3_2_fused * 8 + ax1)
                                v2 = T.axis.spatial(112, i0_0_i1_0_i2_0_i3_0_fused // 8 * 56 + i0_1_i1_1_i2_1_i3_1_fused // 4 * 4 + ax2)
                                v3 = T.axis.spatial(64, i0_0_i1_0_i2_0_i3_0_fused % 8 * 8 + i0_1_i1_1_i2_1_i3_1_fused % 4 * 2 + ax3)
                                T.reads(conv2d_nhwc_local[v0, v1, v2, v3])
                                T.writes(conv2d_nhwc[v0, v1, v2, v3])
                                conv2d_nhwc[v0, v1, v2, v3] = conv2d_nhwc_local[v0, v1, v2, v3]
    # fmt: on
    decision_0 = [
        ("SamplePerfectTile", [1, 1, 1, 1, 1]),
        ("SamplePerfectTile", [1, 1, 14, 8, 1]),
        ("SamplePerfectTile", [2, 14, 1, 4, 1]),
        ("SamplePerfectTile", [8, 4, 1, 1, 2]),
        ("SamplePerfectTile", [1, 1, 7]),
        ("SamplePerfectTile", [1, 7, 1]),
        ("SamplePerfectTile", [1, 1, 3]),
        ("SampleCategorical", 1),
        ("SampleCategorical", 3),
        ("SampleCategorical", 1),
    ]

    mod = create_te_workload("C2D", 0)
    actual = ms.TuneContext(
        mod=mod,
        target=_target(),
        space_generator=ms.space_generator.PostOrderApply(),
        sch_rules="default",
    ).generate_design_space()
    check_sketches(
        mod,
        sketches=actual,
        expected_mods=[c2d_0],
        expected_decisions=[decision_0],
    )


def test_cuda_c3d():
    # fmt: off
    @T.prim_func
    def c3d_0(inputs: T.Buffer[(1, 16, 224, 224, 3), "float32"], weight: T.Buffer[(7, 7, 7, 3, 64), "float32"], conv3d_ndhwc: T.Buffer[(1, 8, 112, 112, 64), "float32"]) -> None:
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        with T.block("root"):
            T.reads()
            T.writes()
            T.block_attr({"meta_schedule.unroll_explicit":16})
            conv3d_ndhwc_local = T.alloc_buffer([1, 8, 112, 112, 64], dtype="float32", scope="local")
            PadInput_shared = T.alloc_buffer([1, 22, 230, 230, 3], dtype="float32", scope="shared")
            weight_shared = T.alloc_buffer([7, 7, 7, 3, 64], dtype="float32", scope="shared")
            for i0_0_i1_0_i2_0_i3_0_i4_0_fused in T.thread_binding(2, thread="blockIdx.x"):
                for i0_1_i1_1_i2_1_i3_1_i4_1_fused in T.thread_binding(8, thread="vthread.x"):
                    for i0_2_i1_2_i2_2_i3_2_i4_2_fused in T.thread_binding(392, thread="threadIdx.x"):
                        for i5_0, i6_0, i7_0, i8_0 in T.grid(1, 1, 1, 1):
                            for ax0_ax1_ax2_ax3_ax4_fused in T.serial(1687959):
                                with T.block("PadInput_shared"):
                                    v0 = T.axis.spatial(1, 0)
                                    v1 = T.axis.spatial(22, ax0_ax1_ax2_ax3_ax4_fused % 1687959 // 80379)
                                    v2 = T.axis.spatial(230, ax0_ax1_ax2_ax3_ax4_fused % 80379 // 351)
                                    v3 = T.axis.spatial(230, i0_0_i1_0_i2_0_i3_0_i4_0_fused * 112 + ax0_ax1_ax2_ax3_ax4_fused % 351 // 3)
                                    v4 = T.axis.spatial(3, ax0_ax1_ax2_ax3_ax4_fused % 3)
                                    T.reads(inputs[v0, v1 - 3, v2 - 3, v3 - 3, v4])
                                    T.writes(PadInput_shared[v0, v1, v2, v3, v4])
                                    T.block_attr({"meta_schedule.cooperative_fetch":4})
                                    PadInput_shared[v0, v1, v2, v3, v4] = T.if_then_else(3 <= v1 and v1 < 19 and 3 <= v2 and v2 < 227 and 3 <= v3 and v3 < 227, inputs[v0, v1 - 3, v2 - 3, v3 - 3, v4], T.float32(0), dtype="float32")
                            for ax0_ax1_ax2_ax3_ax4_fused in T.serial(65856):
                                with T.block("weight_shared"):
                                    v0 = T.axis.spatial(7, ax0_ax1_ax2_ax3_ax4_fused // 9408)
                                    v1 = T.axis.spatial(7, ax0_ax1_ax2_ax3_ax4_fused % 9408 // 1344)
                                    v2 = T.axis.spatial(7, ax0_ax1_ax2_ax3_ax4_fused % 1344 // 192)
                                    v3 = T.axis.spatial(3, ax0_ax1_ax2_ax3_ax4_fused % 192 // 64)
                                    v4 = T.axis.spatial(64, ax0_ax1_ax2_ax3_ax4_fused % 64)
                                    T.reads(weight[v0, v1, v2, v3, v4])
                                    T.writes(weight_shared[v0, v1, v2, v3, v4])
                                    T.block_attr({"meta_schedule.cooperative_fetch":3})
                                    weight_shared[v0, v1, v2, v3, v4] = weight[v0, v1, v2, v3, v4]
                            for i5_1, i6_1, i7_1, i8_1, i0_3, i1_3, i2_3, i3_3, i4_3, i5_2, i6_2, i7_2, i8_2, i0_4, i1_4, i2_4, i3_4, i4_4 in T.grid(7, 7, 1, 3, 1, 2, 2, 1, 32, 1, 1, 7, 1, 1, 1, 2, 4, 1):
                                with T.block("conv3d_ndhwc"):
                                    n = T.axis.spatial(1, i0_4 + i0_3 + 0 + 0 + 0)
                                    d = T.axis.spatial(8, ((0 + 0) * 4 + i0_2_i1_2_i2_2_i3_2_i4_2_fused % 392 // 98) * 2 + i1_3 + i1_4)
                                    h = T.axis.spatial(112, (((0 * 4 + i0_1_i1_1_i2_1_i3_1_i4_1_fused % 8 // 2) * 7 + i0_2_i1_2_i2_2_i3_2_i4_2_fused % 98 // 14) * 2 + i2_3) * 2 + i2_4)
                                    w = T.axis.spatial(112, ((i0_0_i1_0_i2_0_i3_0_i4_0_fused % 2 * 2 + i0_1_i1_1_i2_1_i3_1_i4_1_fused % 2) * 7 + i0_2_i1_2_i2_2_i3_2_i4_2_fused % 14 // 2 + i3_3) * 4 + i3_4)
                                    co = T.axis.spatial(64, ((0 + 0) * 2 + i0_2_i1_2_i2_2_i3_2_i4_2_fused % 2) * 32 + i4_3 + i4_4)
                                    rd = T.axis.reduce(7, i5_0 * 7 + i5_1 + i5_2)
                                    rh = T.axis.reduce(7, i6_0 * 7 + i6_1 + i6_2)
                                    rw = T.axis.reduce(7, (i7_0 + i7_1) * 7 + i7_2)
                                    rc = T.axis.reduce(3, i8_0 * 3 + i8_1 + i8_2)
                                    T.reads(PadInput_shared[n, d * 2 + rd, h * 2 + rh, w * 2 + rw, co // 64 * 3 + rc], weight_shared[rd, rh, rw, rc, co])
                                    T.writes(conv3d_ndhwc_local[n, d, h, w, co])
                                    T.block_attr({"meta_schedule.thread_extent_high_inclusive":1024, "meta_schedule.thread_extent_low_inclusive":32, "meta_schedule.tiling_structure":"SSSRRSRS"})
                                    with T.init():
                                        conv3d_ndhwc_local[n, d, h, w, co] = T.float32(0)
                                    conv3d_ndhwc_local[n, d, h, w, co] = conv3d_ndhwc_local[n, d, h, w, co] + PadInput_shared[n, d * 2 + rd, h * 2 + rh, w * 2 + rw, co // 64 * 3 + rc] * weight_shared[rd, rh, rw, rc, co]
                        for ax0, ax1, ax2, ax3, ax4 in T.grid(1, 2, 4, 4, 32):
                            with T.block("conv3d_ndhwc_local"):
                                v0 = T.axis.spatial(1, ax0)
                                v1 = T.axis.spatial(8, i0_2_i1_2_i2_2_i3_2_i4_2_fused // 98 * 2 + ax1)
                                v2 = T.axis.spatial(112, i0_1_i1_1_i2_1_i3_1_i4_1_fused // 2 * 28 + i0_2_i1_2_i2_2_i3_2_i4_2_fused % 98 // 14 * 4 + ax2)
                                v3 = T.axis.spatial(112, i0_0_i1_0_i2_0_i3_0_i4_0_fused * 56 + i0_1_i1_1_i2_1_i3_1_i4_1_fused % 2 * 28 + i0_2_i1_2_i2_2_i3_2_i4_2_fused % 14 // 2 * 4 + ax3)
                                v4 = T.axis.spatial(64, i0_2_i1_2_i2_2_i3_2_i4_2_fused % 2 * 32 + ax4)
                                T.reads(conv3d_ndhwc_local[v0, v1, v2, v3, v4])
                                T.writes(conv3d_ndhwc[v0, v1, v2, v3, v4])
                                conv3d_ndhwc[v0, v1, v2, v3, v4] = conv3d_ndhwc_local[v0, v1, v2, v3, v4]
    # fmt: on
    decision_0 = [
        ("SamplePerfectTile", [1, 1, 1, 1, 1]),
        ("SamplePerfectTile", [1, 1, 4, 2, 1]),
        ("SamplePerfectTile", [1, 4, 7, 2, 2]),
        ("SamplePerfectTile", [2, 2, 7, 1, 4]),
        ("SamplePerfectTile", [1, 1, 2, 32, 1]),
        ("SamplePerfectTile", [1, 7, 1]),
        ("SamplePerfectTile", [1, 7, 1]),
        ("SamplePerfectTile", [1, 1, 7]),
        ("SamplePerfectTile", [1, 3, 1]),
        ("SampleCategorical", 3),
        ("SampleCategorical", 2),
        ("SampleCategorical", 1),
    ]
    mod = create_te_workload("C3D", 0)
    actual = ms.TuneContext(
        mod=mod,
        target=_target(),
        space_generator=ms.space_generator.PostOrderApply(),
        sch_rules="default",
    ).generate_design_space()
    check_sketches(
        mod,
        sketches=actual,
        expected_mods=[c3d_0],
        expected_decisions=[decision_0],
    )


def test_cuda_cap():
    # fmt: off
    @T.prim_func
    def cap_0(inputs: T.Buffer[(1, 16, 16, 4, 4, 32), "float32"], weight: T.Buffer[(3, 3, 4, 4, 32, 32), "float32"], conv2d_capsule_nhwijc: T.Buffer[(1, 8, 8, 4, 4, 32), "float32"]) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # body
        with T.block("root"):
            T.reads()
            T.writes()
            T.block_attr({"meta_schedule.unroll_explicit":64})
            conv2d_capsule_nhwijc_local = T.alloc_buffer([1, 8, 8, 4, 4, 32], dtype="float32", scope="local")
            PadInput_shared = T.alloc_buffer([1, 18, 18, 4, 4, 32], dtype="float32", scope="shared")
            weight_shared = T.alloc_buffer([3, 3, 4, 4, 32, 32], dtype="float32", scope="shared")
            for i0_0_i1_0_i2_0_i3_0_i4_0_i5_0_fused in T.thread_binding(256, thread="blockIdx.x"):
                for i0_1_i1_1_i2_1_i3_1_i4_1_i5_1_fused in T.thread_binding(1, thread="vthread.x"):
                    for i0_2_i1_2_i2_2_i3_2_i4_2_i5_2_fused in T.thread_binding(4, thread="threadIdx.x"):
                        for i6_0, i7_0, i8_0, i9_0 in T.grid(3, 3, 2, 8):
                            for ax0_ax1_ax2_ax3_ax4_ax5_fused in T.serial(48):
                                with T.block("PadInput_shared"):
                                    v0 = T.axis.spatial(1, 0)
                                    v1 = T.axis.spatial(18, i0_0_i1_0_i2_0_i3_0_i4_0_i5_0_fused // 64 * 4 + i6_0 + ax0_ax1_ax2_ax3_ax4_ax5_fused % 48 // 16)
                                    v2 = T.axis.spatial(18, i0_0_i1_0_i2_0_i3_0_i4_0_i5_0_fused % 64 // 8 * 2 + i7_0 + 0)
                                    v3 = T.axis.spatial(4, i0_0_i1_0_i2_0_i3_0_i4_0_i5_0_fused % 8 // 4 * 2 + ax0_ax1_ax2_ax3_ax4_ax5_fused % 16 // 8)
                                    v4 = T.axis.spatial(4, i8_0 * 2 + ax0_ax1_ax2_ax3_ax4_ax5_fused % 8 // 4)
                                    v5 = T.axis.spatial(32, i9_0 * 4 + ax0_ax1_ax2_ax3_ax4_ax5_fused % 4)
                                    T.reads(inputs[v0, v1 - 1, v2 - 1, v3, v4, v5])
                                    T.writes(PadInput_shared[v0, v1, v2, v3, v4, v5])
                                    T.block_attr({"meta_schedule.cooperative_fetch":2})
                                    PadInput_shared[v0, v1, v2, v3, v4, v5] = T.if_then_else(1 <= v1 and v1 < 17 and 1 <= v2 and v2 < 17, inputs[v0, v1 - 1, v2 - 1, v3, v4, v5], T.float32(0), dtype="float32")
                            for ax0_ax1_ax2_ax3_ax4_ax5_fused in T.serial(256):
                                with T.block("weight_shared"):
                                    v0, v1 = T.axis.remap("SS", [i6_0, i7_0])
                                    v2 = T.axis.spatial(4, i8_0 * 2 + ax0_ax1_ax2_ax3_ax4_ax5_fused // 128)
                                    v3 = T.axis.spatial(4, ax0_ax1_ax2_ax3_ax4_ax5_fused % 128 // 32)
                                    v4 = T.axis.spatial(32, i9_0 * 4 + ax0_ax1_ax2_ax3_ax4_ax5_fused % 32 // 8)
                                    v5 = T.axis.spatial(32, i0_0_i1_0_i2_0_i3_0_i4_0_i5_0_fused % 4 * 8 + ax0_ax1_ax2_ax3_ax4_ax5_fused % 8)
                                    T.reads(weight[v0, v1, v2, v3, v4, v5])
                                    T.writes(weight_shared[v0, v1, v2, v3, v4, v5])
                                    T.block_attr({"meta_schedule.cooperative_fetch":4})
                                    weight_shared[v0, v1, v2, v3, v4, v5] = weight[v0, v1, v2, v3, v4, v5]
                            for i6_1, i7_1, i8_1, i9_1, i0_3, i1_3, i2_3, i3_3, i4_3, i5_3, i6_2, i7_2, i8_2, i9_2, i0_4, i1_4, i2_4, i3_4, i4_4, i5_4 in T.grid(1, 1, 1, 4, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 8):
                                with T.block("conv2d_capsule_nhwijc"):
                                    n = T.axis.spatial(1, i0_4 + i0_3 + 0 + 0 + 0)
                                    h = T.axis.spatial(8, (i0_0_i1_0_i2_0_i3_0_i4_0_i5_0_fused % 256 // 64 + 0 + 0) * 2 + i1_3 + i1_4)
                                    w = T.axis.spatial(8, i0_0_i1_0_i2_0_i3_0_i4_0_i5_0_fused % 64 // 8 + 0 + 0 + i2_3 + i2_4)
                                    cap_i = T.axis.spatial(4, (i0_0_i1_0_i2_0_i3_0_i4_0_i5_0_fused % 8 // 4 + 0) * 2 + i0_2_i1_2_i2_2_i3_2_i4_2_i5_2_fused % 4 // 2 + i3_3 + i3_4)
                                    cap_j = T.axis.spatial(4, ((0 + 0) * 2 + i0_2_i1_2_i2_2_i3_2_i4_2_i5_2_fused % 2 + i4_3) * 2 + i4_4)
                                    co = T.axis.spatial(32, (i0_0_i1_0_i2_0_i3_0_i4_0_i5_0_fused % 4 + 0 + 0 + i5_3) * 8 + i5_4)
                                    rh = T.axis.reduce(3, i6_0 + i6_1 + i6_2)
                                    rw = T.axis.reduce(3, i7_0 + i7_1 + i7_2)
                                    cap_k = T.axis.reduce(4, (i8_0 + i8_1) * 2 + i8_2)
                                    rc = T.axis.reduce(32, i9_0 * 4 + i9_1 + i9_2)
                                    T.reads(PadInput_shared[n, h * 2 + rh, w * 2 + rw, cap_i, cap_k, rc], weight_shared[rh, rw, cap_k, cap_j, rc, co])
                                    T.writes(conv2d_capsule_nhwijc_local[n, h, w, cap_i, cap_j, co])
                                    T.block_attr({"meta_schedule.thread_extent_high_inclusive":1024, "meta_schedule.thread_extent_low_inclusive":32, "meta_schedule.tiling_structure":"SSSRRSRS"})
                                    with T.init():
                                        conv2d_capsule_nhwijc_local[n, h, w, cap_i, cap_j, co] = T.float32(0)
                                    conv2d_capsule_nhwijc_local[n, h, w, cap_i, cap_j, co] = conv2d_capsule_nhwijc_local[n, h, w, cap_i, cap_j, co] + PadInput_shared[n, h * 2 + rh, w * 2 + rw, cap_i, cap_k, rc] * weight_shared[rh, rw, cap_k, cap_j, rc, co]
                        for ax0, ax1, ax2, ax3, ax4, ax5 in T.grid(1, 2, 1, 1, 2, 8):
                            with T.block("conv2d_capsule_nhwijc_local"):
                                v0 = T.axis.spatial(1, ax0)
                                v1 = T.axis.spatial(8, i0_0_i1_0_i2_0_i3_0_i4_0_i5_0_fused // 64 * 2 + ax1)
                                v2 = T.axis.spatial(8, i0_0_i1_0_i2_0_i3_0_i4_0_i5_0_fused % 64 // 8 + ax2)
                                v3 = T.axis.spatial(4, i0_0_i1_0_i2_0_i3_0_i4_0_i5_0_fused % 8 // 4 * 2 + i0_2_i1_2_i2_2_i3_2_i4_2_i5_2_fused // 2 + ax3)
                                v4 = T.axis.spatial(4, i0_2_i1_2_i2_2_i3_2_i4_2_i5_2_fused % 2 * 2 + ax4)
                                v5 = T.axis.spatial(32, i0_0_i1_0_i2_0_i3_0_i4_0_i5_0_fused % 4 * 8 + ax5)
                                T.reads(conv2d_capsule_nhwijc_local[v0, v1, v2, v3, v4, v5])
                                T.writes(conv2d_capsule_nhwijc[v0, v1, v2, v3, v4, v5])
                                conv2d_capsule_nhwijc[v0, v1, v2, v3, v4, v5] = conv2d_capsule_nhwijc_local[v0, v1, v2, v3, v4, v5]
    # fmt: on
    decision_0 = [
        ("SamplePerfectTile", [1, 1, 1, 1, 1]),
        ("SamplePerfectTile", [4, 1, 1, 2, 1]),
        ("SamplePerfectTile", [8, 1, 1, 1, 1]),
        ("SamplePerfectTile", [2, 1, 2, 1, 1]),
        ("SamplePerfectTile", [1, 1, 2, 1, 2]),
        ("SamplePerfectTile", [4, 1, 1, 1, 8]),
        ("SamplePerfectTile", [3, 1, 1]),
        ("SamplePerfectTile", [3, 1, 1]),
        ("SamplePerfectTile", [2, 1, 2]),
        ("SamplePerfectTile", [8, 4, 1]),
        ("SampleCategorical", 1),
        ("SampleCategorical", 3),
        ("SampleCategorical", 2),
    ]
    mod = create_te_workload("CAP", 0)
    actual = ms.TuneContext(
        mod=mod,
        target=_target(),
        space_generator=ms.space_generator.PostOrderApply(),
        sch_rules="default",
    ).generate_design_space()
    check_sketches(
        mod,
        sketches=actual,
        expected_mods=[cap_0],
        expected_decisions=[decision_0],
    )


def test_cuda_dep():
    # fmt: off
    @T.prim_func
    def dep_0(placeholder: T.Buffer[(1, 112, 112, 32), "float32"], placeholder_1: T.Buffer[(1, 3, 3, 32), "float32"], depth_conv2d_nhwc: T.Buffer[(1, 112, 112, 32), "float32"]) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # body
        with T.block("root"):
            T.reads()
            T.writes()
            T.block_attr({"meta_schedule.unroll_explicit":16})
            depth_conv2d_nhwc_local = T.alloc_buffer([1, 112, 112, 32], dtype="float32", scope="local")
            PadInput_shared = T.alloc_buffer([1, 114, 114, 32], dtype="float32", scope="shared")
            placeholder_shared = T.alloc_buffer([1, 3, 3, 32], dtype="float32", scope="shared")
            for i0_0_i1_0_i2_0_i3_0_fused in T.thread_binding(1, thread="blockIdx.x"):
                for i0_1_i1_1_i2_1_i3_1_fused in T.thread_binding(8, thread="vthread.x"):
                    for i0_2_i1_2_i2_2_i3_2_fused in T.thread_binding(14, thread="threadIdx.x"):
                        for i4_0, i5_0 in T.grid(1, 1):
                            for ax0_ax1_ax2_ax3_fused in T.serial(415872):
                                with T.block("PadInput_shared"):
                                    v0 = T.axis.spatial(1, 0)
                                    v1 = T.axis.spatial(114, ax0_ax1_ax2_ax3_fused // 3648)
                                    v2 = T.axis.spatial(114, ax0_ax1_ax2_ax3_fused % 3648 // 32)
                                    v3 = T.axis.spatial(32, ax0_ax1_ax2_ax3_fused % 32)
                                    T.reads(placeholder[v0, v1 - 1, v2 - 1, v3])
                                    T.writes(PadInput_shared[v0, v1, v2, v3])
                                    T.block_attr({"meta_schedule.cooperative_fetch":3})
                                    PadInput_shared[v0, v1, v2, v3] = T.if_then_else(1 <= v1 and v1 < 113 and 1 <= v2 and v2 < 113, placeholder[v0, v1 - 1, v2 - 1, v3], T.float32(0), dtype="float32")
                            for ax0_ax1_ax2_ax3_fused in T.serial(288):
                                with T.block("placeholder_shared"):
                                    v0 = T.axis.spatial(1, 0)
                                    v1 = T.axis.spatial(3, ax0_ax1_ax2_ax3_fused // 96)
                                    v2 = T.axis.spatial(3, ax0_ax1_ax2_ax3_fused % 96 // 32)
                                    v3 = T.axis.spatial(32, ax0_ax1_ax2_ax3_fused % 32)
                                    T.reads(placeholder_1[v0, v1, v2, v3])
                                    T.writes(placeholder_shared[v0, v1, v2, v3])
                                    T.block_attr({"meta_schedule.cooperative_fetch":3})
                                    placeholder_shared[v0, v1, v2, v3] = placeholder_1[v0, v1, v2, v3]
                            for i4_1, i5_1, i0_3, i1_3, i2_3, i3_3, i4_2, i5_2, i0_4, i1_4, i2_4, i3_4 in T.grid(3, 1, 1, 4, 16, 8, 1, 3, 1, 7, 1, 1):
                                with T.block("depth_conv2d_nhwc"):
                                    n = T.axis.spatial(1, i0_4 + i0_3 + 0 + 0 + 0)
                                    h = T.axis.spatial(112, ((0 * 4 + i0_1_i1_1_i2_1_i3_1_fused % 8 // 2 + 0) * 4 + i1_3) * 7 + i1_4)
                                    w = T.axis.spatial(112, ((0 + 0) * 7 + i0_2_i1_2_i2_2_i3_2_fused % 14 // 2) * 16 + i2_3 + i2_4)
                                    c = T.axis.spatial(32, ((0 * 2 + i0_1_i1_1_i2_1_i3_1_fused % 2) * 2 + i0_2_i1_2_i2_2_i3_2_fused % 2) * 8 + i3_3 + i3_4)
                                    rh = T.axis.reduce(3, i4_0 * 3 + i4_1 + i4_2)
                                    rw = T.axis.reduce(3, (i5_0 + i5_1) * 3 + i5_2)
                                    T.reads(PadInput_shared[n, h + rh, w + rw, c], placeholder_shared[0, rh, rw, c])
                                    T.writes(depth_conv2d_nhwc_local[n, h, w, c])
                                    T.block_attr({"meta_schedule.thread_extent_high_inclusive":1024, "meta_schedule.thread_extent_low_inclusive":32, "meta_schedule.tiling_structure":"SSSRRSRS"})
                                    with T.init():
                                        depth_conv2d_nhwc_local[n, h, w, c] = T.float32(0)
                                    depth_conv2d_nhwc_local[n, h, w, c] = depth_conv2d_nhwc_local[n, h, w, c] + PadInput_shared[n, h + rh, w + rw, c] * placeholder_shared[0, rh, rw, c]
                        for ax0, ax1, ax2, ax3 in T.grid(1, 28, 16, 8):
                            with T.block("depth_conv2d_nhwc_local"):
                                v0 = T.axis.spatial(1, ax0)
                                v1 = T.axis.spatial(112, i0_1_i1_1_i2_1_i3_1_fused // 2 * 28 + ax1)
                                v2 = T.axis.spatial(112, i0_2_i1_2_i2_2_i3_2_fused // 2 * 16 + ax2)
                                v3 = T.axis.spatial(32, i0_1_i1_1_i2_1_i3_1_fused % 2 * 16 + i0_2_i1_2_i2_2_i3_2_fused % 2 * 8 + ax3)
                                T.reads(depth_conv2d_nhwc_local[v0, v1, v2, v3])
                                T.writes(depth_conv2d_nhwc[v0, v1, v2, v3])
                                depth_conv2d_nhwc[v0, v1, v2, v3] = depth_conv2d_nhwc_local[v0, v1, v2, v3]
    # fmt: on
    decision_0 = [
        ("SamplePerfectTile", [1, 1, 1, 1, 1]),
        ("SamplePerfectTile", [1, 4, 1, 4, 7]),
        ("SamplePerfectTile", [1, 1, 7, 16, 1]),
        ("SamplePerfectTile", [1, 2, 2, 8, 1]),
        ("SamplePerfectTile", [1, 3, 1]),
        ("SamplePerfectTile", [1, 1, 3]),
        ("SampleCategorical", 2),
        ("SampleCategorical", 2),
        ("SampleCategorical", 1),
    ]
    mod = create_te_workload("DEP", 0)
    actual = ms.TuneContext(
        mod=mod,
        target=_target(),
        space_generator=ms.space_generator.PostOrderApply(),
        sch_rules="default",
    ).generate_design_space()
    check_sketches(
        mod,
        sketches=actual,
        expected_mods=[dep_0],
        expected_decisions=[decision_0],
    )


def test_cuda_dil():
    # fmt: off
    @T.prim_func
    def dil_0(inputs: T.Buffer[(1, 224, 224, 3), "float32"], weight: T.Buffer[(7, 7, 3, 64), "float32"], conv2d_nhwc: T.Buffer[(1, 109, 109, 64), "float32"]) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # body
        with T.block("root"):
            T.reads()
            T.writes()
            T.block_attr({"meta_schedule.unroll_explicit":512})
            conv2d_nhwc_local = T.alloc_buffer([1, 109, 109, 64], dtype="float32", scope="local")
            PadInput_shared = T.alloc_buffer([1, 230, 230, 3], dtype="float32", scope="shared")
            weight_shared = T.alloc_buffer([7, 7, 3, 64], dtype="float32", scope="shared")
            for i0_0_i1_0_i2_0_i3_0_fused in T.thread_binding(218, thread="blockIdx.x"):
                for i0_1_i1_1_i2_1_i3_1_fused in T.thread_binding(109, thread="vthread.x"):
                    for i0_2_i1_2_i2_2_i3_2_fused in T.thread_binding(1, thread="threadIdx.x"):
                        for i4_0, i5_0, i6_0 in T.grid(7, 7, 3):
                            for ax0_ax1_ax2_ax3_fused in T.serial(217):
                                with T.block("PadInput_shared"):
                                    v0 = T.axis.spatial(1, 0)
                                    v1 = T.axis.spatial(230, i0_0_i1_0_i2_0_i3_0_fused // 2 * 2 + i4_0 * 2 + 0)
                                    v2 = T.axis.spatial(230, i5_0 * 2 + ax0_ax1_ax2_ax3_fused % 217)
                                    v3 = T.axis.spatial(3, i6_0 + 0)
                                    T.reads(inputs[v0, v1 - 3, v2 - 3, v3])
                                    T.writes(PadInput_shared[v0, v1, v2, v3])
                                    T.block_attr({"meta_schedule.cooperative_fetch":2})
                                    PadInput_shared[v0, v1, v2, v3] = T.if_then_else(3 <= v1 and v1 < 227 and 3 <= v2 and v2 < 227, inputs[v0, v1 - 3, v2 - 3, v3], T.float32(0), dtype="float32")
                            for ax0_ax1_ax2_ax3_fused in T.serial(32):
                                with T.block("weight_shared"):
                                    v0, v1, v2 = T.axis.remap("SSS", [i4_0, i5_0, i6_0])
                                    v3 = T.axis.spatial(64, i0_0_i1_0_i2_0_i3_0_fused % 2 * 32 + ax0_ax1_ax2_ax3_fused)
                                    T.reads(weight[v0, v1, v2, v3])
                                    T.writes(weight_shared[v0, v1, v2, v3])
                                    T.block_attr({"meta_schedule.cooperative_fetch":4})
                                    weight_shared[v0, v1, v2, v3] = weight[v0, v1, v2, v3]
                            for i4_1, i5_1, i6_1, i0_3, i1_3, i2_3, i3_3, i4_2, i5_2, i6_2, i0_4, i1_4, i2_4, i3_4 in T.grid(1, 1, 1, 1, 1, 1, 8, 1, 1, 1, 1, 1, 1, 4):
                                with T.block("conv2d_nhwc"):
                                    n = T.axis.spatial(1, i0_4 + i0_3 + 0 + 0 + 0)
                                    h = T.axis.spatial(109, i0_0_i1_0_i2_0_i3_0_fused % 218 // 2 + 0 + 0 + i1_3 + i1_4)
                                    w = T.axis.spatial(109, 0 * 109 + i0_1_i1_1_i2_1_i3_1_fused % 109 + 0 + i2_3 + i2_4)
                                    co = T.axis.spatial(64, ((i0_0_i1_0_i2_0_i3_0_fused % 2 + 0 + 0) * 8 + i3_3) * 4 + i3_4)
                                    rh = T.axis.reduce(7, i4_0 + i4_1 + i4_2)
                                    rw = T.axis.reduce(7, i5_0 + i5_1 + i5_2)
                                    rc = T.axis.reduce(3, i6_0 + i6_1 + i6_2)
                                    T.reads(PadInput_shared[n, h * 2 + rh * 2, w * 2 + rw * 2, co // 64 * 3 + rc], weight_shared[rh, rw, rc, co])
                                    T.writes(conv2d_nhwc_local[n, h, w, co])
                                    T.block_attr({"meta_schedule.thread_extent_high_inclusive":1024, "meta_schedule.thread_extent_low_inclusive":32, "meta_schedule.tiling_structure":"SSSRRSRS"})
                                    with T.init():
                                        conv2d_nhwc_local[n, h, w, co] = T.float32(0)
                                    conv2d_nhwc_local[n, h, w, co] = conv2d_nhwc_local[n, h, w, co] + PadInput_shared[n, h * 2 + rh * 2, w * 2 + rw * 2, co // 64 * 3 + rc] * weight_shared[rh, rw, rc, co]
                        for ax0, ax1, ax2, ax3 in T.grid(1, 1, 1, 32):
                            with T.block("conv2d_nhwc_local"):
                                v0 = T.axis.spatial(1, ax0)
                                v1 = T.axis.spatial(109, i0_0_i1_0_i2_0_i3_0_fused // 2 + ax1)
                                v2 = T.axis.spatial(109, i0_1_i1_1_i2_1_i3_1_fused + ax2)
                                v3 = T.axis.spatial(64, i0_0_i1_0_i2_0_i3_0_fused % 2 * 32 + ax3)
                                T.reads(conv2d_nhwc_local[v0, v1, v2, v3])
                                T.writes(conv2d_nhwc[v0, v1, v2, v3])
                                conv2d_nhwc[v0, v1, v2, v3] = conv2d_nhwc_local[v0, v1, v2, v3]
    # fmt: on
    decision_0 = [
        ("SamplePerfectTile", [1, 1, 1, 1, 1]),
        ("SamplePerfectTile", [109, 1, 1, 1, 1]),
        ("SamplePerfectTile", [1, 109, 1, 1, 1]),
        ("SamplePerfectTile", [2, 1, 1, 8, 4]),
        ("SamplePerfectTile", [7, 1, 1]),
        ("SamplePerfectTile", [7, 1, 1]),
        ("SamplePerfectTile", [3, 1, 1]),
        ("SampleCategorical", 1),
        ("SampleCategorical", 3),
        ("SampleCategorical", 3),
    ]
    mod = create_te_workload("DIL", 0)
    actual = ms.TuneContext(
        mod=mod,
        target=_target(),
        space_generator=ms.space_generator.PostOrderApply(),
        sch_rules="default",
    ).generate_design_space()
    check_sketches(
        mod,
        sketches=actual,
        expected_mods=[dil_0],
        expected_decisions=[decision_0],
    )


if __name__ == "__main__":
    test_cuda_c1d()
    test_cuda_c2d()
    test_cuda_c3d()
    test_cuda_cap()
    test_cuda_dep()
    test_cuda_dil()
