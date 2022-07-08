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
                                    v1 = T.axis.spatial(258, i0_0_i1_0_i2_0_fused * 64 + ax0_ax1_ax2_fused % 260 // 4)
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
                                    n = T.axis.spatial(1, i0_4 + i0_3 + 0 + 0 + 0)
                                    l = T.axis.spatial(128, (i0_0_i1_0_i2_0_fused % 4 * 8 + i0_1_i1_1_i2_1_fused % 16 // 2 + 0 + i1_3) * 4 + i1_4)
                                    co = T.axis.spatial(128, (((0 * 2 + i0_1_i1_1_i2_1_fused % 2) * 4 + i0_2_i1_2_i2_2_fused % 4) * 2 + i2_3) * 8 + i2_4)
                                    rl = T.axis.reduce(3, (i3_0 + i3_1) * 3 + i3_2)
                                    rc = T.axis.reduce(64, (i4_0 * 2 + i4_1) * 2 + i4_2)
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


if __name__ == "__main__":
    test_cuda_c1d()
    test_cuda_c2d()
