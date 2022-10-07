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
"""Tests for MetaSchedule search space on CPU"""
from tvm import meta_schedule as ms
from tvm.meta_schedule.testing.space_generation import (
    check_sketches,
    print_sketches,
    generate_design_space,
)
from tvm.meta_schedule.testing.te_workload import create_te_workload
from tvm.script import tir as T
from tvm.target import Target


def _target():
    return Target("aws/cpu/c5.9xlarge")


def _design_space(mod):
    return generate_design_space(
        kind="llvm",
        mod=mod,
        target=_target(),
        types=ms.ScheduleRule,
    )


def test_cpu_c1d():
    # fmt: off
    @T.prim_func
    def c1d_0(inputs: T.Buffer[(1, 256, 64), "float32"], weight: T.Buffer[(3, 64, 128), "float32"], conv1d_nlc: T.Buffer[(1, 128, 128), "float32"]) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # body
        with T.block("root"):
            T.reads()
            T.writes()
            T.block_attr({"meta_schedule.parallel":288, "meta_schedule.unroll_explicit":512, "meta_schedule.vectorize":64})
            PadInput = T.alloc_buffer([1, 258, 64], dtype="float32")
            conv1d_nlc_global = T.alloc_buffer([1, 128, 128], dtype="float32")
            for i0, i1, i2 in T.grid(1, 258, 64):
                with T.block("PadInput"):
                    i0_1, i1_1, i2_1 = T.axis.remap("SSS", [i0, i1, i2])
                    T.reads(inputs[i0_1, i1_1 - 1, i2_1])
                    T.writes(PadInput[i0_1, i1_1, i2_1])
                    PadInput[i0_1, i1_1, i2_1] = T.if_then_else(1 <= i1_1 and i1_1 < 257, inputs[i0_1, i1_1 - 1, i2_1], T.float32(0), dtype="float32")
            for i0_0, i1_0, i2_0, i0_1_1, i1_1_1, i2_1_1 in T.grid(1, 1, 2, 1, 1, 8):
                for i3_0, i4_0, i0_2, i1_2, i2_2, i3_1, i4_1, i0_3, i1_3, i2_3 in T.grid(1, 64, 1, 64, 8, 3, 1, 1, 2, 1):
                    with T.block("conv1d_nlc"):
                        n = T.axis.spatial(1, i0_1_1 + i0_2 + i0_3 + i0_0)
                        l = T.axis.spatial(128, i1_0 * 128 + i1_1_1 * 128 + i1_2 * 2 + i1_3)
                        co = T.axis.spatial(128, i2_3 + i2_0 * 64 + i2_1_1 * 8 + i2_2)
                        rl = T.axis.reduce(3, i3_0 * 3 + i3_1)
                        rc = T.axis.reduce(64, i4_1 + i4_0)
                        T.reads(PadInput[n, l * 2 + rl, co // 128 * 64 + rc], weight[rl, rc, co])
                        T.writes(conv1d_nlc_global[n, l, co])
                        T.block_attr({"meta_schedule.tiling_structure":"SSRSRS"})
                        with T.init():
                            conv1d_nlc_global[n, l, co] = T.float32(0)
                        conv1d_nlc_global[n, l, co] = conv1d_nlc_global[n, l, co] + PadInput[n, l * 2 + rl, co // 128 * 64 + rc] * weight[rl, rc, co]
                for ax0, ax1, ax2 in T.grid(1, 128, 8):
                    with T.block("conv1d_nlc_global"):
                        v0, v1 = T.axis.remap("SS", [ax0, ax1])
                        v2 = T.axis.spatial(128, i2_0 * 64 + i2_1_1 * 8 + ax2)
                        T.reads(conv1d_nlc_global[v0, v1, v2])
                        T.writes(conv1d_nlc[v0, v1, v2])
                        conv1d_nlc[v0, v1, v2] = conv1d_nlc_global[v0, v1, v2]
    @T.prim_func
    def c1d_1(inputs: T.Buffer[(1, 256, 64), "float32"], weight: T.Buffer[(3, 64, 128), "float32"], conv1d_nlc: T.Buffer[(1, 128, 128), "float32"]) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # body
        with T.block("root"):
            T.reads()
            T.writes()
            T.block_attr({"meta_schedule.parallel":288, "meta_schedule.unroll_explicit":512, "meta_schedule.vectorize":64})
            PadInput = T.alloc_buffer([1, 258, 64], dtype="float32")
            conv1d_nlc_global = T.alloc_buffer([1, 128, 128], dtype="float32")
            for i0_0, i1_0, i2_0 in T.grid(1, 1, 2):
                for i0_1, i1_1, i2_1 in T.grid(1, 1, 8):
                    for ax0, ax1, ax2 in T.grid(1, 257, 64):
                        with T.block("PadInput"):
                            i0 = T.axis.spatial(1, ax0)
                            i1 = T.axis.spatial(258, ax1)
                            i2 = T.axis.spatial(64, ax2)
                            T.reads(inputs[i0, i1 - 1, i2])
                            T.writes(PadInput[i0, i1, i2])
                            PadInput[i0, i1, i2] = T.if_then_else(1 <= i1 and i1 < 257, inputs[i0, i1 - 1, i2], T.float32(0), dtype="float32")
                    for i3_0, i4_0, i0_2, i1_2, i2_2, i3_1, i4_1, i0_3, i1_3, i2_3 in T.grid(1, 64, 1, 64, 8, 3, 1, 1, 2, 1):
                        with T.block("conv1d_nlc"):
                            n = T.axis.spatial(1, i0_1 + i0_2 + i0_3 + i0_0)
                            l = T.axis.spatial(128, i1_0 * 128 + i1_1 * 128 + i1_2 * 2 + i1_3)
                            co = T.axis.spatial(128, i2_3 + i2_0 * 64 + i2_1 * 8 + i2_2)
                            rl = T.axis.reduce(3, i3_0 * 3 + i3_1)
                            rc = T.axis.reduce(64, i4_1 + i4_0)
                            T.reads(PadInput[n, l * 2 + rl, co // 128 * 64 + rc], weight[rl, rc, co])
                            T.writes(conv1d_nlc_global[n, l, co])
                            T.block_attr({"meta_schedule.tiling_structure":"SSRSRS"})
                            with T.init():
                                conv1d_nlc_global[n, l, co] = T.float32(0)
                            conv1d_nlc_global[n, l, co] = conv1d_nlc_global[n, l, co] + PadInput[n, l * 2 + rl, co // 128 * 64 + rc] * weight[rl, rc, co]
                for ax0, ax1, ax2 in T.grid(1, 128, 64):
                    with T.block("conv1d_nlc_global"):
                        v0, v1 = T.axis.remap("SS", [ax0, ax1])
                        v2 = T.axis.spatial(128, i2_0 * 64 + ax2)
                        T.reads(conv1d_nlc_global[v0, v1, v2])
                        T.writes(conv1d_nlc[v0, v1, v2])
                        conv1d_nlc[v0, v1, v2] = conv1d_nlc_global[v0, v1, v2]

    @T.prim_func
    def c1d_2(inputs: T.Buffer[(1, 256, 64), "float32"], weight: T.Buffer[(3, 64, 128), "float32"], conv1d_nlc: T.Buffer[(1, 128, 128), "float32"]) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # body
        with T.block("root"):
            T.reads()
            T.writes()
            T.block_attr({"meta_schedule.parallel":288, "meta_schedule.unroll_explicit":16, "meta_schedule.vectorize":64})
            for i0_0, i1_0, i2_0, i0_1, i1_1, i2_1, i3_0, i4_0, i0_2, i1_2, i2_2, i3_1, i4_1, i0_3, i1_3, i2_3 in T.grid(1, 1, 2, 1, 1, 8, 1, 64, 1, 64, 8, 3, 1, 1, 2, 1):
                with T.block("conv1d_nlc"):
                    n = T.axis.spatial(1, i0_1 + i0_2 + i0_3 + i0_0)
                    l = T.axis.spatial(128, i1_0 * 128 + i1_1 * 128 + i1_2 * 2 + i1_3)
                    co = T.axis.spatial(128, i2_3 + i2_0 * 64 + i2_1 * 8 + i2_2)
                    rl = T.axis.reduce(3, i3_0 * 3 + i3_1)
                    rc = T.axis.reduce(64, i4_1 + i4_0)
                    T.reads(inputs[n, l * 2 + rl - 1, co // 128 * 64 + rc], weight[rl, rc, co])
                    T.writes(conv1d_nlc[n, l, co])
                    T.block_attr({"meta_schedule.tiling_structure":"SSRSRS"})
                    with T.init():
                        conv1d_nlc[n, l, co] = T.float32(0)
                    conv1d_nlc[n, l, co] = conv1d_nlc[n, l, co] + T.if_then_else(1 <= l * 2 + rl and l * 2 + rl < 257, inputs[n, l * 2 + rl - 1, co // 128 * 64 + rc], T.float32(0), dtype="float32") * weight[rl, rc, co]
    # fmt: on

    decision_0 = [
        ("SamplePerfectTile", [1, 1, 1, 1]),
        ("SamplePerfectTile", [1, 1, 64, 2]),
        ("SamplePerfectTile", [2, 8, 8, 1]),
        ("SamplePerfectTile", [1, 3]),
        ("SamplePerfectTile", [64, 1]),
        ("SampleCategorical", 3),
        ("SampleComputeLocation", -1),
    ]
    decision_1 = [
        ("SamplePerfectTile", [1, 1, 1, 1]),
        ("SamplePerfectTile", [1, 1, 64, 2]),
        ("SamplePerfectTile", [2, 8, 8, 1]),
        ("SamplePerfectTile", [1, 3]),
        ("SamplePerfectTile", [64, 1]),
        ("SampleCategorical", 3),
        ("SampleComputeLocation", 5),
    ]
    decision_2 = [
        ("SamplePerfectTile", [1, 1, 1, 1]),
        ("SamplePerfectTile", [1, 1, 64, 2]),
        ("SamplePerfectTile", [2, 8, 8, 1]),
        ("SamplePerfectTile", [1, 3]),
        ("SamplePerfectTile", [64, 1]),
        ("SampleCategorical", 1),
        ("SampleComputeLocation", -2),
    ]

    mod = create_te_workload("C1D", 0)
    actual = _design_space(mod)
    check_sketches(
        mod,
        sketches=actual,
        expected_mods=[c1d_0, c1d_1, c1d_2],
        expected_decisions=[decision_0, decision_1, decision_2],
    )


def test_cpu_c2d():
    # fmt: off
    @T.prim_func
    def c2d_0(inputs: T.Buffer[(1, 224, 224, 3), "float32"], weight: T.Buffer[(7, 7, 3, 64), "float32"], conv2d_nhwc: T.Buffer[(1, 112, 112, 64), "float32"]) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # body
        with T.block("root"):
            T.reads()
            T.writes()
            T.block_attr({"meta_schedule.parallel":288, "meta_schedule.unroll_explicit":16, "meta_schedule.vectorize":64})
            PadInput = T.alloc_buffer([1, 230, 230, 3], dtype="float32")
            conv2d_nhwc_global = T.alloc_buffer([1, 112, 112, 64], dtype="float32")
            for i0_0, i1_0, i2_0, i3_0, i0_1, i1_1, i2_1 in T.grid(1, 7, 4, 2, 1, 1, 28):
                for ax0, ax1, ax2, ax3 in T.grid(1, 37, 7, 3):
                    with T.block("PadInput"):
                        i0 = T.axis.spatial(1, ax0)
                        i1 = T.axis.spatial(230, i1_0 * 32 + ax1)
                        i2 = T.axis.spatial(230, i2_0 * 56 + i2_1 * 2 + ax2)
                        i3 = T.axis.spatial(3, ax3)
                        T.reads(inputs[i0, i1 - 3, i2 - 3, i3])
                        T.writes(PadInput[i0, i1, i2, i3])
                        PadInput[i0, i1, i2, i3] = T.if_then_else(3 <= i1 and i1 < 227 and 3 <= i2 and i2 < 227, inputs[i0, i1 - 3, i2 - 3, i3], T.float32(0), dtype="float32")
                for i3_1 in T.serial(8):
                    for i4_0, i5_0, i6_0, i0_2, i1_2, i2_2, i3_2, i4_1, i5_1, i6_1, i0_3, i1_3, i2_3, i3_3 in T.grid(7, 7, 1, 1, 2, 1, 1, 1, 1, 3, 1, 8, 1, 4):
                        with T.block("conv2d_nhwc"):
                            n = T.axis.spatial(1, i0_3 + i0_0 + i0_1 + i0_2)
                            h = T.axis.spatial(112, i1_0 * 16 + i1_1 * 16 + i1_2 * 8 + i1_3)
                            w = T.axis.spatial(112, i2_3 + i2_0 * 28 + i2_1 + i2_2)
                            co = T.axis.spatial(64, i3_0 * 32 + i3_1 * 4 + i3_2 * 4 + i3_3)
                            rh = T.axis.reduce(7, i4_1 + i4_0)
                            rw = T.axis.reduce(7, i5_0 + i5_1)
                            rc = T.axis.reduce(3, i6_0 * 3 + i6_1)
                            T.reads(PadInput[n, h * 2 + rh, w * 2 + rw, co // 64 * 3 + rc], weight[rh, rw, rc, co])
                            T.writes(conv2d_nhwc_global[n, h, w, co])
                            T.block_attr({"meta_schedule.tiling_structure":"SSRSRS"})
                            with T.init():
                                conv2d_nhwc_global[n, h, w, co] = T.float32(0)
                            conv2d_nhwc_global[n, h, w, co] = conv2d_nhwc_global[n, h, w, co] + PadInput[n, h * 2 + rh, w * 2 + rw, co // 64 * 3 + rc] * weight[rh, rw, rc, co]
                    for ax0, ax1, ax2, ax3 in T.grid(1, 16, 1, 4):
                        with T.block("conv2d_nhwc_global"):
                            v0 = T.axis.spatial(1, ax0)
                            v1 = T.axis.spatial(112, i1_0 * 16 + ax1)
                            v2 = T.axis.spatial(112, i2_0 * 28 + i2_1 + ax2)
                            v3 = T.axis.spatial(64, i3_0 * 32 + i3_1 * 4 + ax3)
                            T.reads(conv2d_nhwc_global[v0, v1, v2, v3])
                            T.writes(conv2d_nhwc[v0, v1, v2, v3])
                            conv2d_nhwc[v0, v1, v2, v3] = conv2d_nhwc_global[v0, v1, v2, v3]
    @T.prim_func
    def c2d_1(inputs: T.Buffer[(1, 224, 224, 3), "float32"], weight: T.Buffer[(7, 7, 3, 64), "float32"], conv2d_nhwc: T.Buffer[(1, 112, 112, 64), "float32"]) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # body
        with T.block("root"):
            T.reads()
            T.writes()
            T.block_attr({"meta_schedule.parallel":288, "meta_schedule.unroll_explicit":512, "meta_schedule.vectorize":64})
            PadInput = T.alloc_buffer([1, 230, 230, 3], dtype="float32")
            conv2d_nhwc_global = T.alloc_buffer([1, 112, 112, 64], dtype="float32")
            for i0, i1, i2, i3 in T.grid(1, 230, 230, 3):
                with T.block("PadInput"):
                    i0_1, i1_1, i2_1, i3_1 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                    T.reads(inputs[i0_1, i1_1 - 3, i2_1 - 3, i3_1])
                    T.writes(PadInput[i0_1, i1_1, i2_1, i3_1])
                    PadInput[i0_1, i1_1, i2_1, i3_1] = T.if_then_else(3 <= i1_1 and i1_1 < 227 and 3 <= i2_1 and i2_1 < 227, inputs[i0_1, i1_1 - 3, i2_1 - 3, i3_1], T.float32(0), dtype="float32")
            for i0_0, i1_0, i2_0, i3_0 in T.grid(1, 7, 4, 2):
                for i0_1_1, i1_1_1, i2_1_1, i3_1_1, i4_0, i5_0, i6_0, i0_2, i1_2, i2_2, i3_2, i4_1, i5_1, i6_1, i0_3, i1_3, i2_3, i3_3 in T.grid(1, 1, 28, 8, 7, 7, 1, 1, 2, 1, 1, 1, 1, 3, 1, 8, 1, 4):
                    with T.block("conv2d_nhwc"):
                        n = T.axis.spatial(1, i0_3 + i0_0 + i0_1_1 + i0_2)
                        h = T.axis.spatial(112, i1_0 * 16 + i1_1_1 * 16 + i1_2 * 8 + i1_3)
                        w = T.axis.spatial(112, i2_3 + i2_0 * 28 + i2_1_1 + i2_2)
                        co = T.axis.spatial(64, i3_0 * 32 + i3_1_1 * 4 + i3_2 * 4 + i3_3)
                        rh = T.axis.reduce(7, i4_1 + i4_0)
                        rw = T.axis.reduce(7, i5_0 + i5_1)
                        rc = T.axis.reduce(3, i6_0 * 3 + i6_1)
                        T.reads(PadInput[n, h * 2 + rh, w * 2 + rw, co // 64 * 3 + rc], weight[rh, rw, rc, co])
                        T.writes(conv2d_nhwc_global[n, h, w, co])
                        T.block_attr({"meta_schedule.tiling_structure":"SSRSRS"})
                        with T.init():
                            conv2d_nhwc_global[n, h, w, co] = T.float32(0)
                        conv2d_nhwc_global[n, h, w, co] = conv2d_nhwc_global[n, h, w, co] + PadInput[n, h * 2 + rh, w * 2 + rw, co // 64 * 3 + rc] * weight[rh, rw, rc, co]
                for ax0, ax1, ax2, ax3 in T.grid(1, 16, 28, 32):
                    with T.block("conv2d_nhwc_global"):
                        v0 = T.axis.spatial(1, ax0)
                        v1 = T.axis.spatial(112, i1_0 * 16 + ax1)
                        v2 = T.axis.spatial(112, i2_0 * 28 + ax2)
                        v3 = T.axis.spatial(64, i3_0 * 32 + ax3)
                        T.reads(conv2d_nhwc_global[v0, v1, v2, v3])
                        T.writes(conv2d_nhwc[v0, v1, v2, v3])
                        conv2d_nhwc[v0, v1, v2, v3] = conv2d_nhwc_global[v0, v1, v2, v3]
    @T.prim_func
    def c2d_2(inputs: T.Buffer[(1, 224, 224, 3), "float32"], weight: T.Buffer[(7, 7, 3, 64), "float32"], conv2d_nhwc: T.Buffer[(1, 112, 112, 64), "float32"]) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # body
        with T.block("root"):
            T.reads()
            T.writes()
            T.block_attr({"meta_schedule.parallel":288, "meta_schedule.unroll_explicit":0, "meta_schedule.vectorize":64})
            PadInput = T.alloc_buffer([1, 230, 230, 3], dtype="float32")
            for i0_0, i1_0 in T.grid(1, 7):
                for ax0, ax1, ax2, ax3 in T.grid(1, 37, 229, 3):
                    with T.block("PadInput"):
                        i0 = T.axis.spatial(1, ax0)
                        i1 = T.axis.spatial(230, i1_0 * 32 + ax1)
                        i2 = T.axis.spatial(230, ax2)
                        i3 = T.axis.spatial(3, ax3)
                        T.reads(inputs[i0, i1 - 3, i2 - 3, i3])
                        T.writes(PadInput[i0, i1, i2, i3])
                        PadInput[i0, i1, i2, i3] = T.if_then_else(3 <= i1 and i1 < 227 and 3 <= i2 and i2 < 227, inputs[i0, i1 - 3, i2 - 3, i3], T.float32(0), dtype="float32")
                for i2_0, i3_0, i0_1, i1_1, i2_1, i3_1, i4_0, i5_0, i6_0, i0_2, i1_2, i2_2, i3_2, i4_1, i5_1, i6_1, i0_3, i1_3, i2_3, i3_3 in T.grid(4, 2, 1, 1, 28, 8, 7, 7, 1, 1, 2, 1, 1, 1, 1, 3, 1, 8, 1, 4):
                    with T.block("conv2d_nhwc"):
                        n = T.axis.spatial(1, i0_3 + i0_0 + i0_1 + i0_2)
                        h = T.axis.spatial(112, i1_0 * 16 + i1_1 * 16 + i1_2 * 8 + i1_3)
                        w = T.axis.spatial(112, i2_3 + i2_0 * 28 + i2_1 + i2_2)
                        co = T.axis.spatial(64, i3_0 * 32 + i3_1 * 4 + i3_2 * 4 + i3_3)
                        rh = T.axis.reduce(7, i4_1 + i4_0)
                        rw = T.axis.reduce(7, i5_0 + i5_1)
                        rc = T.axis.reduce(3, i6_0 * 3 + i6_1)
                        T.reads(PadInput[n, h * 2 + rh, w * 2 + rw, co // 64 * 3 + rc], weight[rh, rw, rc, co])
                        T.writes(conv2d_nhwc[n, h, w, co])
                        T.block_attr({"meta_schedule.tiling_structure":"SSRSRS"})
                        with T.init():
                            conv2d_nhwc[n, h, w, co] = T.float32(0)
                        conv2d_nhwc[n, h, w, co] = conv2d_nhwc[n, h, w, co] + PadInput[n, h * 2 + rh, w * 2 + rw, co // 64 * 3 + rc] * weight[rh, rw, rc, co]
    # fmt: on

    decision_0 = [
        ("SamplePerfectTile", [1, 1, 1, 1]),
        ("SamplePerfectTile", [7, 1, 2, 8]),
        ("SamplePerfectTile", [4, 28, 1, 1]),
        ("SamplePerfectTile", [2, 8, 1, 4]),
        ("SamplePerfectTile", [7, 1]),
        ("SamplePerfectTile", [7, 1]),
        ("SamplePerfectTile", [1, 3]),
        ("SampleCategorical", 1),
        ("SampleComputeLocation", 6),
    ]
    decision_1 = [
        ("SamplePerfectTile", [1, 1, 1, 1]),
        ("SamplePerfectTile", [7, 1, 2, 8]),
        ("SamplePerfectTile", [4, 28, 1, 1]),
        ("SamplePerfectTile", [2, 8, 1, 4]),
        ("SamplePerfectTile", [7, 1]),
        ("SamplePerfectTile", [7, 1]),
        ("SamplePerfectTile", [1, 3]),
        ("SampleCategorical", 3),
        ("SampleComputeLocation", -1),
    ]
    decision_2 = [
        ("SamplePerfectTile", [1, 1, 1, 1]),
        ("SamplePerfectTile", [7, 1, 2, 8]),
        ("SamplePerfectTile", [4, 28, 1, 1]),
        ("SamplePerfectTile", [2, 8, 1, 4]),
        ("SamplePerfectTile", [7, 1]),
        ("SamplePerfectTile", [7, 1]),
        ("SamplePerfectTile", [1, 3]),
        ("SampleCategorical", 0),
        ("SampleComputeLocation", 1),
    ]

    mod = create_te_workload("C2D", 0)
    actual = _design_space(mod)
    check_sketches(
        mod,
        sketches=actual,
        expected_mods=[c2d_0, c2d_1, c2d_2],
        expected_decisions=[decision_0, decision_1, decision_2],
    )


def test_cpu_c3d():
    # fmt: off
    @T.prim_func
    def c3d_0(inputs: T.Buffer[(1, 16, 224, 224, 3), "float32"], weight: T.Buffer[(7, 7, 7, 3, 64), "float32"], conv3d_ndhwc: T.Buffer[(1, 8, 112, 112, 64), "float32"]) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # body
        with T.block("root"):
            T.reads()
            T.writes()
            T.block_attr({"meta_schedule.parallel":288, "meta_schedule.unroll_explicit":512, "meta_schedule.vectorize":64})
            PadInput = T.alloc_buffer([1, 22, 230, 230, 3], dtype="float32")
            conv3d_ndhwc_global = T.alloc_buffer([1, 8, 112, 112, 64], dtype="float32")
            for i0_0, i1_0, i2_0, i3_0, i4_0 in T.grid(1, 2, 4, 1, 2):
                for ax0, ax1, ax2, ax3, ax4 in T.grid(1, 13, 61, 229, 3):
                    with T.block("PadInput"):
                        i0 = T.axis.spatial(1, ax0)
                        i1 = T.axis.spatial(22, i1_0 * 8 + ax1)
                        i2 = T.axis.spatial(230, i2_0 * 56 + ax2)
                        i3 = T.axis.spatial(230, ax3)
                        i4 = T.axis.spatial(3, ax4)
                        T.reads(inputs[i0, i1 - 3, i2 - 3, i3 - 3, i4])
                        T.writes(PadInput[i0, i1, i2, i3, i4])
                        PadInput[i0, i1, i2, i3, i4] = T.if_then_else(3 <= i1 and i1 < 19 and 3 <= i2 and i2 < 227 and 3 <= i3 and i3 < 227, inputs[i0, i1 - 3, i2 - 3, i3 - 3, i4], T.float32(0), dtype="float32")
                for i0_1, i1_1, i2_1, i3_1, i4_1 in T.grid(1, 4, 4, 14, 1):
                    for i5_0, i6_0, i7_0, i8_0, i0_2, i1_2, i2_2, i3_2, i4_2, i5_1, i6_1, i7_1, i8_1, i0_3, i1_3, i2_3, i3_3, i4_3 in T.grid(1, 7, 7, 3, 1, 1, 1, 1, 32, 7, 1, 1, 1, 1, 1, 7, 8, 1):
                        with T.block("conv3d_ndhwc"):
                            n = T.axis.spatial(1, i0_1 + i0_2 + i0_3 + i0_0)
                            d = T.axis.spatial(8, i1_3 + i1_0 * 4 + i1_1 + i1_2)
                            h = T.axis.spatial(112, i2_0 * 28 + i2_1 * 7 + i2_2 * 7 + i2_3)
                            w = T.axis.spatial(112, i3_0 * 112 + i3_1 * 8 + i3_2 * 8 + i3_3)
                            co = T.axis.spatial(64, i4_3 + i4_0 * 32 + i4_1 * 32 + i4_2)
                            rd = T.axis.reduce(7, i5_0 * 7 + i5_1)
                            rh = T.axis.reduce(7, i6_1 + i6_0)
                            rw = T.axis.reduce(7, i7_0 + i7_1)
                            rc = T.axis.reduce(3, i8_1 + i8_0)
                            T.reads(PadInput[n, d * 2 + rd, h * 2 + rh, w * 2 + rw, co // 64 * 3 + rc], weight[rd, rh, rw, rc, co])
                            T.writes(conv3d_ndhwc_global[n, d, h, w, co])
                            T.block_attr({"meta_schedule.tiling_structure":"SSRSRS"})
                            with T.init():
                                conv3d_ndhwc_global[n, d, h, w, co] = T.float32(0)
                            conv3d_ndhwc_global[n, d, h, w, co] = conv3d_ndhwc_global[n, d, h, w, co] + PadInput[n, d * 2 + rd, h * 2 + rh, w * 2 + rw, co // 64 * 3 + rc] * weight[rd, rh, rw, rc, co]
                    for ax0, ax1, ax2, ax3, ax4 in T.grid(1, 1, 7, 8, 32):
                        with T.block("conv3d_ndhwc_global"):
                            v0 = T.axis.spatial(1, ax0)
                            v1 = T.axis.spatial(8, i1_0 * 4 + i1_1 + ax1)
                            v2 = T.axis.spatial(112, i2_0 * 28 + i2_1 * 7 + ax2)
                            v3 = T.axis.spatial(112, i3_1 * 8 + ax3)
                            v4 = T.axis.spatial(64, i4_0 * 32 + ax4)
                            T.reads(conv3d_ndhwc_global[v0, v1, v2, v3, v4])
                            T.writes(conv3d_ndhwc[v0, v1, v2, v3, v4])
                            conv3d_ndhwc[v0, v1, v2, v3, v4] = conv3d_ndhwc_global[v0, v1, v2, v3, v4]
    @T.prim_func
    def c3d_1(inputs: T.Buffer[(1, 16, 224, 224, 3), "float32"], weight: T.Buffer[(7, 7, 7, 3, 64), "float32"], conv3d_ndhwc: T.Buffer[(1, 8, 112, 112, 64), "float32"]) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # body
        with T.block("root"):
            T.reads()
            T.writes()
            T.block_attr({"meta_schedule.parallel":288, "meta_schedule.unroll_explicit":64, "meta_schedule.vectorize":64})
            PadInput = T.alloc_buffer([1, 22, 230, 230, 3], dtype="float32")
            conv3d_ndhwc_global = T.alloc_buffer([1, 8, 112, 112, 64], dtype="float32")
            for i0_0, i1_0, i2_0, i3_0, i4_0 in T.grid(1, 2, 4, 1, 2):
                for i0_1, i1_1, i2_1, i3_1 in T.grid(1, 4, 4, 14):
                    for ax0, ax1, ax2, ax3, ax4 in T.grid(1, 7, 19, 21, 3):
                        with T.block("PadInput"):
                            i0 = T.axis.spatial(1, ax0)
                            i1 = T.axis.spatial(22, i1_0 * 8 + i1_1 * 2 + ax1)
                            i2 = T.axis.spatial(230, i2_0 * 56 + i2_1 * 14 + ax2)
                            i3 = T.axis.spatial(230, i3_1 * 16 + ax3)
                            i4 = T.axis.spatial(3, ax4)
                            T.reads(inputs[i0, i1 - 3, i2 - 3, i3 - 3, i4])
                            T.writes(PadInput[i0, i1, i2, i3, i4])
                            PadInput[i0, i1, i2, i3, i4] = T.if_then_else(3 <= i1 and i1 < 19 and 3 <= i2 and i2 < 227 and 3 <= i3 and i3 < 227, inputs[i0, i1 - 3, i2 - 3, i3 - 3, i4], T.float32(0), dtype="float32")
                    for i4_1, i5_0, i6_0, i7_0, i8_0, i0_2, i1_2, i2_2, i3_2, i4_2, i5_1, i6_1, i7_1, i8_1, i0_3, i1_3, i2_3, i3_3, i4_3 in T.grid(1, 1, 7, 7, 3, 1, 1, 1, 1, 32, 7, 1, 1, 1, 1, 1, 7, 8, 1):
                        with T.block("conv3d_ndhwc"):
                            n = T.axis.spatial(1, i0_1 + i0_2 + i0_3 + i0_0)
                            d = T.axis.spatial(8, i1_3 + i1_0 * 4 + i1_1 + i1_2)
                            h = T.axis.spatial(112, i2_0 * 28 + i2_1 * 7 + i2_2 * 7 + i2_3)
                            w = T.axis.spatial(112, i3_0 * 112 + i3_1 * 8 + i3_2 * 8 + i3_3)
                            co = T.axis.spatial(64, i4_3 + i4_0 * 32 + i4_1 * 32 + i4_2)
                            rd = T.axis.reduce(7, i5_0 * 7 + i5_1)
                            rh = T.axis.reduce(7, i6_1 + i6_0)
                            rw = T.axis.reduce(7, i7_0 + i7_1)
                            rc = T.axis.reduce(3, i8_1 + i8_0)
                            T.reads(PadInput[n, d * 2 + rd, h * 2 + rh, w * 2 + rw, co // 64 * 3 + rc], weight[rd, rh, rw, rc, co])
                            T.writes(conv3d_ndhwc_global[n, d, h, w, co])
                            T.block_attr({"meta_schedule.tiling_structure":"SSRSRS"})
                            with T.init():
                                conv3d_ndhwc_global[n, d, h, w, co] = T.float32(0)
                            conv3d_ndhwc_global[n, d, h, w, co] = conv3d_ndhwc_global[n, d, h, w, co] + PadInput[n, d * 2 + rd, h * 2 + rh, w * 2 + rw, co // 64 * 3 + rc] * weight[rd, rh, rw, rc, co]
                for ax0, ax1, ax2, ax3, ax4 in T.grid(1, 4, 28, 112, 32):
                    with T.block("conv3d_ndhwc_global"):
                        v0 = T.axis.spatial(1, ax0)
                        v1 = T.axis.spatial(8, i1_0 * 4 + ax1)
                        v2 = T.axis.spatial(112, i2_0 * 28 + ax2)
                        v3 = T.axis.spatial(112, ax3)
                        v4 = T.axis.spatial(64, i4_0 * 32 + ax4)
                        T.reads(conv3d_ndhwc_global[v0, v1, v2, v3, v4])
                        T.writes(conv3d_ndhwc[v0, v1, v2, v3, v4])
                        conv3d_ndhwc[v0, v1, v2, v3, v4] = conv3d_ndhwc_global[v0, v1, v2, v3, v4]
    @T.prim_func
    def c3d_2(inputs: T.Buffer[(1, 16, 224, 224, 3), "float32"], weight: T.Buffer[(7, 7, 7, 3, 64), "float32"], conv3d_ndhwc: T.Buffer[(1, 8, 112, 112, 64), "float32"]) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # body
        with T.block("root"):
            T.reads()
            T.writes()
            T.block_attr({"meta_schedule.parallel":288, "meta_schedule.unroll_explicit":16, "meta_schedule.vectorize":64})
            PadInput = T.alloc_buffer([1, 22, 230, 230, 3], dtype="float32")
            for i0_0, i1_0, i2_0, i3_0, i4_0, i0_1, i1_1, i2_1, i3_1 in T.grid(1, 2, 4, 1, 2, 1, 4, 4, 14):
                for ax0, ax1, ax2, ax3, ax4 in T.grid(1, 7, 19, 21, 3):
                    with T.block("PadInput"):
                        i0 = T.axis.spatial(1, ax0)
                        i1 = T.axis.spatial(22, i1_0 * 8 + i1_1 * 2 + ax1)
                        i2 = T.axis.spatial(230, i2_0 * 56 + i2_1 * 14 + ax2)
                        i3 = T.axis.spatial(230, i3_1 * 16 + ax3)
                        i4 = T.axis.spatial(3, ax4)
                        T.reads(inputs[i0, i1 - 3, i2 - 3, i3 - 3, i4])
                        T.writes(PadInput[i0, i1, i2, i3, i4])
                        PadInput[i0, i1, i2, i3, i4] = T.if_then_else(3 <= i1 and i1 < 19 and 3 <= i2 and i2 < 227 and 3 <= i3 and i3 < 227, inputs[i0, i1 - 3, i2 - 3, i3 - 3, i4], T.float32(0), dtype="float32")
                for i4_1, i5_0, i6_0, i7_0, i8_0, i0_2, i1_2, i2_2, i3_2, i4_2, i5_1, i6_1, i7_1, i8_1, i0_3, i1_3, i2_3, i3_3, i4_3 in T.grid(1, 1, 7, 7, 3, 1, 1, 1, 1, 32, 7, 1, 1, 1, 1, 1, 7, 8, 1):
                    with T.block("conv3d_ndhwc"):
                        n = T.axis.spatial(1, i0_1 + i0_2 + i0_3 + i0_0)
                        d = T.axis.spatial(8, i1_3 + i1_0 * 4 + i1_1 + i1_2)
                        h = T.axis.spatial(112, i2_0 * 28 + i2_1 * 7 + i2_2 * 7 + i2_3)
                        w = T.axis.spatial(112, i3_0 * 112 + i3_1 * 8 + i3_2 * 8 + i3_3)
                        co = T.axis.spatial(64, i4_3 + i4_0 * 32 + i4_1 * 32 + i4_2)
                        rd = T.axis.reduce(7, i5_0 * 7 + i5_1)
                        rh = T.axis.reduce(7, i6_1 + i6_0)
                        rw = T.axis.reduce(7, i7_0 + i7_1)
                        rc = T.axis.reduce(3, i8_1 + i8_0)
                        T.reads(PadInput[n, d * 2 + rd, h * 2 + rh, w * 2 + rw, co // 64 * 3 + rc], weight[rd, rh, rw, rc, co])
                        T.writes(conv3d_ndhwc[n, d, h, w, co])
                        T.block_attr({"meta_schedule.tiling_structure":"SSRSRS"})
                        with T.init():
                            conv3d_ndhwc[n, d, h, w, co] = T.float32(0)
                        conv3d_ndhwc[n, d, h, w, co] = conv3d_ndhwc[n, d, h, w, co] + PadInput[n, d * 2 + rd, h * 2 + rh, w * 2 + rw, co // 64 * 3 + rc] * weight[rd, rh, rw, rc, co]
    # fmt: on

    decision_0 = [
        ("SamplePerfectTile", [1, 1, 1, 1]),
        ("SamplePerfectTile", [2, 4, 1, 1]),
        ("SamplePerfectTile", [4, 4, 1, 7]),
        ("SamplePerfectTile", [1, 14, 1, 8]),
        ("SamplePerfectTile", [2, 1, 32, 1]),
        ("SamplePerfectTile", [1, 7]),
        ("SamplePerfectTile", [7, 1]),
        ("SamplePerfectTile", [7, 1]),
        ("SamplePerfectTile", [3, 1]),
        ("SampleCategorical", 3),
        ("SampleComputeLocation", 4),
    ]
    decision_1 = [
        ("SamplePerfectTile", [1, 1, 1, 1]),
        ("SamplePerfectTile", [2, 4, 1, 1]),
        ("SamplePerfectTile", [4, 4, 1, 7]),
        ("SamplePerfectTile", [1, 14, 1, 8]),
        ("SamplePerfectTile", [2, 1, 32, 1]),
        ("SamplePerfectTile", [1, 7]),
        ("SamplePerfectTile", [7, 1]),
        ("SamplePerfectTile", [7, 1]),
        ("SamplePerfectTile", [3, 1]),
        ("SampleCategorical", 2),
        ("SampleComputeLocation", 8),
    ]
    decision_2 = [
        ("SamplePerfectTile", [1, 1, 1, 1]),
        ("SamplePerfectTile", [2, 4, 1, 1]),
        ("SamplePerfectTile", [4, 4, 1, 7]),
        ("SamplePerfectTile", [1, 14, 1, 8]),
        ("SamplePerfectTile", [2, 1, 32, 1]),
        ("SamplePerfectTile", [1, 7]),
        ("SamplePerfectTile", [7, 1]),
        ("SamplePerfectTile", [7, 1]),
        ("SamplePerfectTile", [3, 1]),
        ("SampleCategorical", 1),
        ("SampleComputeLocation", 8),
    ]

    mod = create_te_workload("C3D", 0)
    actual = _design_space(mod)
    check_sketches(
        mod,
        sketches=actual,
        expected_mods=[c3d_0, c3d_1, c3d_2],
        expected_decisions=[decision_0, decision_1, decision_2],
    )


def test_cpu_cap():
    # fmt: off
    @T.prim_func
    def cap_0(inputs: T.Buffer[(1, 16, 16, 4, 4, 32), "float32"], weight: T.Buffer[(3, 3, 4, 4, 32, 32), "float32"], conv2d_capsule_nhwijc: T.Buffer[(1, 8, 8, 4, 4, 32), "float32"]) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # body
        with T.block("root"):
            T.reads()
            T.writes()
            T.block_attr({"meta_schedule.parallel":288, "meta_schedule.unroll_explicit":0, "meta_schedule.vectorize":64})
            PadInput = T.alloc_buffer([1, 18, 18, 4, 4, 32], dtype="float32")
            conv2d_capsule_nhwijc_global = T.alloc_buffer([1, 8, 8, 4, 4, 32], dtype="float32")
            for i0_0, i1_0, i2_0, i3_0, i4_0, i5_0, i0_1, i1_1 in T.grid(1, 2, 1, 1, 1, 1, 1, 4):
                for ax0, ax1, ax2, ax3, ax4, ax5 in T.grid(1, 3, 17, 4, 4, 32):
                    with T.block("PadInput"):
                        i0 = T.axis.spatial(1, ax0)
                        i1 = T.axis.spatial(18, i1_0 * 8 + i1_1 * 2 + ax1)
                        i2 = T.axis.spatial(18, ax2)
                        i3, i4, i5 = T.axis.remap("SSS", [ax3, ax4, ax5])
                        T.reads(inputs[i0, i1 - 1, i2 - 1, i3, i4, i5])
                        T.writes(PadInput[i0, i1, i2, i3, i4, i5])
                        PadInput[i0, i1, i2, i3, i4, i5] = T.if_then_else(1 <= i1 and i1 < 17 and 1 <= i2 and i2 < 17, inputs[i0, i1 - 1, i2 - 1, i3, i4, i5], T.float32(0), dtype="float32")
                for i2_1, i3_1, i4_1, i5_1 in T.grid(4, 1, 4, 2):
                    for i6_0, i7_0, i8_0, i9_0, i0_2, i1_2, i2_2, i3_2, i4_2, i5_2, i6_1, i7_1, i8_1, i9_1, i0_3, i1_3, i2_3, i3_3, i4_3, i5_3 in T.grid(1, 3, 4, 1, 1, 1, 2, 1, 1, 1, 3, 1, 1, 32, 1, 1, 1, 4, 1, 16):
                        with T.block("conv2d_capsule_nhwijc"):
                            n = T.axis.spatial(1, i0_2 + i0_3 + i0_0 + i0_1)
                            h = T.axis.spatial(8, i1_0 * 4 + i1_1 + i1_2 + i1_3)
                            w = T.axis.spatial(8, i2_0 * 8 + i2_1 * 2 + i2_2 + i2_3)
                            cap_i = T.axis.spatial(4, i3_0 * 4 + i3_1 * 4 + i3_2 * 4 + i3_3)
                            cap_j = T.axis.spatial(4, i4_0 * 4 + i4_1 + i4_2 + i4_3)
                            co = T.axis.spatial(32, i5_0 * 32 + i5_1 * 16 + i5_2 * 16 + i5_3)
                            rh = T.axis.reduce(3, i6_0 * 3 + i6_1)
                            rw = T.axis.reduce(3, i7_1 + i7_0)
                            cap_k = T.axis.reduce(4, i8_0 + i8_1)
                            rc = T.axis.reduce(32, i9_0 * 32 + i9_1)
                            T.reads(PadInput[n, h * 2 + rh, w * 2 + rw, cap_i, cap_k, rc], weight[rh, rw, cap_k, cap_j, rc, co])
                            T.writes(conv2d_capsule_nhwijc_global[n, h, w, cap_i, cap_j, co])
                            T.block_attr({"meta_schedule.tiling_structure":"SSRSRS"})
                            with T.init():
                                conv2d_capsule_nhwijc_global[n, h, w, cap_i, cap_j, co] = T.float32(0)
                            conv2d_capsule_nhwijc_global[n, h, w, cap_i, cap_j, co] = conv2d_capsule_nhwijc_global[n, h, w, cap_i, cap_j, co] + PadInput[n, h * 2 + rh, w * 2 + rw, cap_i, cap_k, rc] * weight[rh, rw, cap_k, cap_j, rc, co]
                    for ax0, ax1, ax2, ax3, ax4, ax5 in T.grid(1, 1, 2, 4, 1, 16):
                        with T.block("conv2d_capsule_nhwijc_global"):
                            v0 = T.axis.spatial(1, ax0)
                            v1 = T.axis.spatial(8, i1_0 * 4 + i1_1 + ax1)
                            v2 = T.axis.spatial(8, i2_1 * 2 + ax2)
                            v3 = T.axis.spatial(4, ax3)
                            v4 = T.axis.spatial(4, i4_1 + ax4)
                            v5 = T.axis.spatial(32, i5_1 * 16 + ax5)
                            T.reads(conv2d_capsule_nhwijc_global[v0, v1, v2, v3, v4, v5])
                            T.writes(conv2d_capsule_nhwijc[v0, v1, v2, v3, v4, v5])
                            conv2d_capsule_nhwijc[v0, v1, v2, v3, v4, v5] = conv2d_capsule_nhwijc_global[v0, v1, v2, v3, v4, v5]
    @T.prim_func
    def cap_1(inputs: T.Buffer[(1, 16, 16, 4, 4, 32), "float32"], weight: T.Buffer[(3, 3, 4, 4, 32, 32), "float32"], conv2d_capsule_nhwijc: T.Buffer[(1, 8, 8, 4, 4, 32), "float32"]) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # body
        with T.block("root"):
            T.reads()
            T.writes()
            T.block_attr({"meta_schedule.parallel":288, "meta_schedule.unroll_explicit":0, "meta_schedule.vectorize":64})
            PadInput = T.alloc_buffer([1, 18, 18, 4, 4, 32], dtype="float32")
            conv2d_capsule_nhwijc_global = T.alloc_buffer([1, 8, 8, 4, 4, 32], dtype="float32")
            for i0_0, i1_0, i2_0, i3_0, i4_0, i5_0 in T.grid(1, 2, 1, 1, 1, 1):
                for i0_1, i1_1, i2_1, i3_1, i4_1, i5_1 in T.grid(1, 4, 4, 1, 4, 2):
                    for ax0, ax1, ax2, ax3, ax4, ax5 in T.grid(1, 3, 5, 4, 4, 32):
                        with T.block("PadInput"):
                            i0 = T.axis.spatial(1, ax0)
                            i1 = T.axis.spatial(18, i1_0 * 8 + i1_1 * 2 + ax1)
                            i2 = T.axis.spatial(18, i2_1 * 4 + ax2)
                            i3, i4, i5 = T.axis.remap("SSS", [ax3, ax4, ax5])
                            T.reads(inputs[i0, i1 - 1, i2 - 1, i3, i4, i5])
                            T.writes(PadInput[i0, i1, i2, i3, i4, i5])
                            PadInput[i0, i1, i2, i3, i4, i5] = T.if_then_else(1 <= i1 and i1 < 17 and 1 <= i2 and i2 < 17, inputs[i0, i1 - 1, i2 - 1, i3, i4, i5], T.float32(0), dtype="float32")
                    for i6_0, i7_0, i8_0, i9_0, i0_2, i1_2, i2_2, i3_2, i4_2, i5_2, i6_1, i7_1, i8_1, i9_1, i0_3, i1_3, i2_3, i3_3, i4_3, i5_3 in T.grid(1, 3, 4, 1, 1, 1, 2, 1, 1, 1, 3, 1, 1, 32, 1, 1, 1, 4, 1, 16):
                        with T.block("conv2d_capsule_nhwijc"):
                            n = T.axis.spatial(1, i0_2 + i0_3 + i0_0 + i0_1)
                            h = T.axis.spatial(8, i1_0 * 4 + i1_1 + i1_2 + i1_3)
                            w = T.axis.spatial(8, i2_0 * 8 + i2_1 * 2 + i2_2 + i2_3)
                            cap_i = T.axis.spatial(4, i3_0 * 4 + i3_1 * 4 + i3_2 * 4 + i3_3)
                            cap_j = T.axis.spatial(4, i4_0 * 4 + i4_1 + i4_2 + i4_3)
                            co = T.axis.spatial(32, i5_0 * 32 + i5_1 * 16 + i5_2 * 16 + i5_3)
                            rh = T.axis.reduce(3, i6_0 * 3 + i6_1)
                            rw = T.axis.reduce(3, i7_1 + i7_0)
                            cap_k = T.axis.reduce(4, i8_0 + i8_1)
                            rc = T.axis.reduce(32, i9_0 * 32 + i9_1)
                            T.reads(PadInput[n, h * 2 + rh, w * 2 + rw, cap_i, cap_k, rc], weight[rh, rw, cap_k, cap_j, rc, co])
                            T.writes(conv2d_capsule_nhwijc_global[n, h, w, cap_i, cap_j, co])
                            T.block_attr({"meta_schedule.tiling_structure":"SSRSRS"})
                            with T.init():
                                conv2d_capsule_nhwijc_global[n, h, w, cap_i, cap_j, co] = T.float32(0)
                            conv2d_capsule_nhwijc_global[n, h, w, cap_i, cap_j, co] = conv2d_capsule_nhwijc_global[n, h, w, cap_i, cap_j, co] + PadInput[n, h * 2 + rh, w * 2 + rw, cap_i, cap_k, rc] * weight[rh, rw, cap_k, cap_j, rc, co]
                for ax0, ax1, ax2, ax3, ax4, ax5 in T.grid(1, 4, 8, 4, 4, 32):
                    with T.block("conv2d_capsule_nhwijc_global"):
                        v0 = T.axis.spatial(1, ax0)
                        v1 = T.axis.spatial(8, i1_0 * 4 + ax1)
                        v2, v3, v4, v5 = T.axis.remap("SSSS", [ax2, ax3, ax4, ax5])
                        T.reads(conv2d_capsule_nhwijc_global[v0, v1, v2, v3, v4, v5])
                        T.writes(conv2d_capsule_nhwijc[v0, v1, v2, v3, v4, v5])
                        conv2d_capsule_nhwijc[v0, v1, v2, v3, v4, v5] = conv2d_capsule_nhwijc_global[v0, v1, v2, v3, v4, v5]
    @T.prim_func
    def cap_2(inputs: T.Buffer[(1, 16, 16, 4, 4, 32), "float32"], weight: T.Buffer[(3, 3, 4, 4, 32, 32), "float32"], conv2d_capsule_nhwijc: T.Buffer[(1, 8, 8, 4, 4, 32), "float32"]) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # body
        with T.block("root"):
            T.reads()
            T.writes()
            T.block_attr({"meta_schedule.parallel":288, "meta_schedule.unroll_explicit":16, "meta_schedule.vectorize":64})
            PadInput = T.alloc_buffer([1, 18, 18, 4, 4, 32], dtype="float32")
            for i0, i1, i2, i3, i4, i5 in T.grid(1, 18, 18, 4, 4, 32):
                with T.block("PadInput"):
                    i0_1, i1_1, i2_1, i3_1, i4_1, i5_1 = T.axis.remap("SSSSSS", [i0, i1, i2, i3, i4, i5])
                    T.reads(inputs[i0_1, i1_1 - 1, i2_1 - 1, i3_1, i4_1, i5_1])
                    T.writes(PadInput[i0_1, i1_1, i2_1, i3_1, i4_1, i5_1])
                    PadInput[i0_1, i1_1, i2_1, i3_1, i4_1, i5_1] = T.if_then_else(1 <= i1_1 and i1_1 < 17 and 1 <= i2_1 and i2_1 < 17, inputs[i0_1, i1_1 - 1, i2_1 - 1, i3_1, i4_1, i5_1], T.float32(0), dtype="float32")
            for i0_0, i1_0, i2_0, i3_0, i4_0, i5_0, i0_1_1, i1_1_1, i2_1_1, i3_1_1, i4_1_1, i5_1_1, i6_0, i7_0, i8_0, i9_0, i0_2, i1_2, i2_2, i3_2, i4_2, i5_2, i6_1, i7_1, i8_1, i9_1, i0_3, i1_3, i2_3, i3_3, i4_3, i5_3 in T.grid(1, 2, 1, 1, 1, 1, 1, 4, 4, 1, 4, 2, 1, 3, 4, 1, 1, 1, 2, 1, 1, 1, 3, 1, 1, 32, 1, 1, 1, 4, 1, 16):
                with T.block("conv2d_capsule_nhwijc"):
                    n = T.axis.spatial(1, i0_2 + i0_3 + i0_0 + i0_1_1)
                    h = T.axis.spatial(8, i1_0 * 4 + i1_1_1 + i1_2 + i1_3)
                    w = T.axis.spatial(8, i2_0 * 8 + i2_1_1 * 2 + i2_2 + i2_3)
                    cap_i = T.axis.spatial(4, i3_0 * 4 + i3_1_1 * 4 + i3_2 * 4 + i3_3)
                    cap_j = T.axis.spatial(4, i4_0 * 4 + i4_1_1 + i4_2 + i4_3)
                    co = T.axis.spatial(32, i5_0 * 32 + i5_1_1 * 16 + i5_2 * 16 + i5_3)
                    rh = T.axis.reduce(3, i6_0 * 3 + i6_1)
                    rw = T.axis.reduce(3, i7_1 + i7_0)
                    cap_k = T.axis.reduce(4, i8_0 + i8_1)
                    rc = T.axis.reduce(32, i9_0 * 32 + i9_1)
                    T.reads(PadInput[n, h * 2 + rh, w * 2 + rw, cap_i, cap_k, rc], weight[rh, rw, cap_k, cap_j, rc, co])
                    T.writes(conv2d_capsule_nhwijc[n, h, w, cap_i, cap_j, co])
                    T.block_attr({"meta_schedule.tiling_structure":"SSRSRS"})
                    with T.init():
                        conv2d_capsule_nhwijc[n, h, w, cap_i, cap_j, co] = T.float32(0)
                    conv2d_capsule_nhwijc[n, h, w, cap_i, cap_j, co] = conv2d_capsule_nhwijc[n, h, w, cap_i, cap_j, co] + PadInput[n, h * 2 + rh, w * 2 + rw, cap_i, cap_k, rc] * weight[rh, rw, cap_k, cap_j, rc, co]
    # fmt: on
    decision_0 = [
        ("SamplePerfectTile", [1, 1, 1, 1]),
        ("SamplePerfectTile", [2, 4, 1, 1]),
        ("SamplePerfectTile", [1, 4, 2, 1]),
        ("SamplePerfectTile", [1, 1, 1, 4]),
        ("SamplePerfectTile", [1, 4, 1, 1]),
        ("SamplePerfectTile", [1, 2, 1, 16]),
        ("SamplePerfectTile", [1, 3]),
        ("SamplePerfectTile", [3, 1]),
        ("SamplePerfectTile", [4, 1]),
        ("SamplePerfectTile", [1, 32]),
        ("SampleCategorical", 0),
        ("SampleComputeLocation", 7),
    ]
    decision_1 = [
        ("SamplePerfectTile", [1, 1, 1, 1]),
        ("SamplePerfectTile", [2, 4, 1, 1]),
        ("SamplePerfectTile", [1, 4, 2, 1]),
        ("SamplePerfectTile", [1, 1, 1, 4]),
        ("SamplePerfectTile", [1, 4, 1, 1]),
        ("SamplePerfectTile", [1, 2, 1, 16]),
        ("SamplePerfectTile", [1, 3]),
        ("SamplePerfectTile", [3, 1]),
        ("SamplePerfectTile", [4, 1]),
        ("SamplePerfectTile", [1, 32]),
        ("SampleCategorical", 0),
        ("SampleComputeLocation", 11),
    ]
    decision_2 = [
        ("SamplePerfectTile", [1, 1, 1, 1]),
        ("SamplePerfectTile", [2, 4, 1, 1]),
        ("SamplePerfectTile", [1, 4, 2, 1]),
        ("SamplePerfectTile", [1, 1, 1, 4]),
        ("SamplePerfectTile", [1, 4, 1, 1]),
        ("SamplePerfectTile", [1, 2, 1, 16]),
        ("SamplePerfectTile", [1, 3]),
        ("SamplePerfectTile", [3, 1]),
        ("SamplePerfectTile", [4, 1]),
        ("SamplePerfectTile", [1, 32]),
        ("SampleCategorical", 1),
        ("SampleComputeLocation", -1),
    ]
    mod = create_te_workload("CAP", 0)
    actual = _design_space(mod)
    check_sketches(
        mod,
        sketches=actual,
        expected_mods=[cap_0, cap_1, cap_2],
        expected_decisions=[decision_0, decision_1, decision_2],
    )


def test_cpu_dep():
    # fmt: off
    @T.prim_func
    def dep_0(placeholder: T.Buffer[(1, 112, 112, 32), "float32"], placeholder_1: T.Buffer[(1, 3, 3, 32), "float32"], depth_conv2d_nhwc: T.Buffer[(1, 112, 112, 32), "float32"]) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # body
        with T.block("root"):
            T.reads()
            T.writes()
            T.block_attr({"meta_schedule.parallel":288, "meta_schedule.unroll_explicit":64, "meta_schedule.vectorize":64})
            PadInput = T.alloc_buffer([1, 114, 114, 32], dtype="float32")
            depth_conv2d_nhwc_global = T.alloc_buffer([1, 112, 112, 32], dtype="float32")
            for i0, i1, i2, i3 in T.grid(1, 114, 114, 32):
                with T.block("PadInput"):
                    i0_1, i1_1, i2_1, i3_1 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                    T.reads(placeholder[i0_1, i1_1 - 1, i2_1 - 1, i3_1])
                    T.writes(PadInput[i0_1, i1_1, i2_1, i3_1])
                    PadInput[i0_1, i1_1, i2_1, i3_1] = T.if_then_else(1 <= i1_1 and i1_1 < 113 and 1 <= i2_1 and i2_1 < 113, placeholder[i0_1, i1_1 - 1, i2_1 - 1, i3_1], T.float32(0), dtype="float32")
            for i0_0, i1_0, i2_0, i3_0, i0_1_1, i1_1_1, i2_1_1, i3_1_1 in T.grid(1, 1, 1, 1, 1, 4, 4, 8):
                for i4_0, i5_0, i0_2, i1_2, i2_2, i3_2, i4_1, i5_1, i0_3, i1_3, i2_3, i3_3 in T.grid(1, 1, 1, 2, 7, 2, 3, 3, 1, 14, 4, 2):
                    with T.block("depth_conv2d_nhwc"):
                        n = T.axis.spatial(1, i0_2 + i0_3 + i0_0 + i0_1_1)
                        h = T.axis.spatial(112, i1_0 * 112 + i1_1_1 * 28 + i1_2 * 14 + i1_3)
                        w = T.axis.spatial(112, i2_0 * 112 + i2_1_1 * 28 + i2_2 * 4 + i2_3)
                        c = T.axis.spatial(32, i3_0 * 32 + i3_1_1 * 4 + i3_2 * 2 + i3_3)
                        rh = T.axis.reduce(3, i4_0 * 3 + i4_1)
                        rw = T.axis.reduce(3, i5_0 * 3 + i5_1)
                        T.reads(PadInput[n, h + rh, w + rw, c], placeholder_1[0, rh, rw, c])
                        T.writes(depth_conv2d_nhwc_global[n, h, w, c])
                        T.block_attr({"meta_schedule.tiling_structure":"SSRSRS"})
                        with T.init():
                            depth_conv2d_nhwc_global[n, h, w, c] = T.float32(0)
                        depth_conv2d_nhwc_global[n, h, w, c] = depth_conv2d_nhwc_global[n, h, w, c] + PadInput[n, h + rh, w + rw, c] * placeholder_1[0, rh, rw, c]
                for ax0, ax1, ax2, ax3 in T.grid(1, 28, 28, 4):
                    with T.block("depth_conv2d_nhwc_global"):
                        v0 = T.axis.spatial(1, ax0)
                        v1 = T.axis.spatial(112, i1_1_1 * 28 + ax1)
                        v2 = T.axis.spatial(112, i2_1_1 * 28 + ax2)
                        v3 = T.axis.spatial(32, i3_1_1 * 4 + ax3)
                        T.reads(depth_conv2d_nhwc_global[v0, v1, v2, v3])
                        T.writes(depth_conv2d_nhwc[v0, v1, v2, v3])
                        depth_conv2d_nhwc[v0, v1, v2, v3] = depth_conv2d_nhwc_global[v0, v1, v2, v3]
    @T.prim_func
    def dep_1(placeholder: T.Buffer[(1, 112, 112, 32), "float32"], placeholder_1: T.Buffer[(1, 3, 3, 32), "float32"], depth_conv2d_nhwc: T.Buffer[(1, 112, 112, 32), "float32"]) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # body
        with T.block("root"):
            T.reads()
            T.writes()
            T.block_attr({"meta_schedule.parallel":288, "meta_schedule.unroll_explicit":16, "meta_schedule.vectorize":64})
            PadInput = T.alloc_buffer([1, 114, 114, 32], dtype="float32")
            depth_conv2d_nhwc_global = T.alloc_buffer([1, 112, 112, 32], dtype="float32")
            for i0, i1, i2, i3 in T.grid(1, 114, 114, 32):
                with T.block("PadInput"):
                    i0_1, i1_1, i2_1, i3_1 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                    T.reads(placeholder[i0_1, i1_1 - 1, i2_1 - 1, i3_1])
                    T.writes(PadInput[i0_1, i1_1, i2_1, i3_1])
                    PadInput[i0_1, i1_1, i2_1, i3_1] = T.if_then_else(1 <= i1_1 and i1_1 < 113 and 1 <= i2_1 and i2_1 < 113, placeholder[i0_1, i1_1 - 1, i2_1 - 1, i3_1], T.float32(0), dtype="float32")
            for i0_0, i1_0, i2_0, i3_0 in T.grid(1, 1, 1, 1):
                for i0_1_1, i1_1_1, i2_1_1, i3_1_1, i4_0, i5_0, i0_2, i1_2, i2_2, i3_2, i4_1, i5_1, i0_3, i1_3, i2_3, i3_3 in T.grid(1, 4, 4, 8, 1, 1, 1, 2, 7, 2, 3, 3, 1, 14, 4, 2):
                    with T.block("depth_conv2d_nhwc"):
                        n = T.axis.spatial(1, i0_2 + i0_3 + i0_0 + i0_1_1)
                        h = T.axis.spatial(112, i1_0 * 112 + i1_1_1 * 28 + i1_2 * 14 + i1_3)
                        w = T.axis.spatial(112, i2_0 * 112 + i2_1_1 * 28 + i2_2 * 4 + i2_3)
                        c = T.axis.spatial(32, i3_0 * 32 + i3_1_1 * 4 + i3_2 * 2 + i3_3)
                        rh = T.axis.reduce(3, i4_0 * 3 + i4_1)
                        rw = T.axis.reduce(3, i5_0 * 3 + i5_1)
                        T.reads(PadInput[n, h + rh, w + rw, c], placeholder_1[0, rh, rw, c])
                        T.writes(depth_conv2d_nhwc_global[n, h, w, c])
                        T.block_attr({"meta_schedule.tiling_structure":"SSRSRS"})
                        with T.init():
                            depth_conv2d_nhwc_global[n, h, w, c] = T.float32(0)
                        depth_conv2d_nhwc_global[n, h, w, c] = depth_conv2d_nhwc_global[n, h, w, c] + PadInput[n, h + rh, w + rw, c] * placeholder_1[0, rh, rw, c]
                for ax0, ax1, ax2, ax3 in T.grid(1, 112, 112, 32):
                    with T.block("depth_conv2d_nhwc_global"):
                        v0, v1, v2, v3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                        T.reads(depth_conv2d_nhwc_global[v0, v1, v2, v3])
                        T.writes(depth_conv2d_nhwc[v0, v1, v2, v3])
                        depth_conv2d_nhwc[v0, v1, v2, v3] = depth_conv2d_nhwc_global[v0, v1, v2, v3]
    @T.prim_func
    def dep_2(placeholder: T.Buffer[(1, 112, 112, 32), "float32"], placeholder_1: T.Buffer[(1, 3, 3, 32), "float32"], depth_conv2d_nhwc: T.Buffer[(1, 112, 112, 32), "float32"]) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # body
        with T.block("root"):
            T.reads()
            T.writes()
            T.block_attr({"meta_schedule.parallel":288, "meta_schedule.unroll_explicit":0, "meta_schedule.vectorize":64})
            PadInput = T.alloc_buffer([1, 114, 114, 32], dtype="float32")
            for i0_0, i1_0, i2_0, i3_0, i0_1, i1_1 in T.grid(1, 1, 1, 1, 1, 4):
                for ax0, ax1, ax2, ax3 in T.grid(1, 30, 114, 32):
                    with T.block("PadInput"):
                        i0 = T.axis.spatial(1, ax0)
                        i1 = T.axis.spatial(114, i1_1 * 28 + ax1)
                        i2, i3 = T.axis.remap("SS", [ax2, ax3])
                        T.reads(placeholder[i0, i1 - 1, i2 - 1, i3])
                        T.writes(PadInput[i0, i1, i2, i3])
                        PadInput[i0, i1, i2, i3] = T.if_then_else(1 <= i1 and i1 < 113 and 1 <= i2 and i2 < 113, placeholder[i0, i1 - 1, i2 - 1, i3], T.float32(0), dtype="float32")
                for i2_1, i3_1, i4_0, i5_0, i0_2, i1_2, i2_2, i3_2, i4_1, i5_1, i0_3, i1_3, i2_3, i3_3 in T.grid(4, 8, 1, 1, 1, 2, 7, 2, 3, 3, 1, 14, 4, 2):
                    with T.block("depth_conv2d_nhwc"):
                        n = T.axis.spatial(1, i0_2 + i0_3 + i0_0 + i0_1)
                        h = T.axis.spatial(112, i1_0 * 112 + i1_1 * 28 + i1_2 * 14 + i1_3)
                        w = T.axis.spatial(112, i2_0 * 112 + i2_1 * 28 + i2_2 * 4 + i2_3)
                        c = T.axis.spatial(32, i3_0 * 32 + i3_1 * 4 + i3_2 * 2 + i3_3)
                        rh = T.axis.reduce(3, i4_0 * 3 + i4_1)
                        rw = T.axis.reduce(3, i5_0 * 3 + i5_1)
                        T.reads(PadInput[n, h + rh, w + rw, c], placeholder_1[0, rh, rw, c])
                        T.writes(depth_conv2d_nhwc[n, h, w, c])
                        T.block_attr({"meta_schedule.tiling_structure":"SSRSRS"})
                        with T.init():
                            depth_conv2d_nhwc[n, h, w, c] = T.float32(0)
                        depth_conv2d_nhwc[n, h, w, c] = depth_conv2d_nhwc[n, h, w, c] + PadInput[n, h + rh, w + rw, c] * placeholder_1[0, rh, rw, c]
    # fmt: on
    decision_0 = [
        ("SamplePerfectTile", [1, 1, 1, 1]),
        ("SamplePerfectTile", [1, 4, 2, 14]),
        ("SamplePerfectTile", [1, 4, 7, 4]),
        ("SamplePerfectTile", [1, 8, 2, 2]),
        ("SamplePerfectTile", [1, 3]),
        ("SamplePerfectTile", [1, 3]),
        ("SampleCategorical", 2),
        ("SampleComputeLocation", -1),
    ]
    decision_1 = [
        ("SamplePerfectTile", [1, 1, 1, 1]),
        ("SamplePerfectTile", [1, 4, 2, 14]),
        ("SamplePerfectTile", [1, 4, 7, 4]),
        ("SamplePerfectTile", [1, 8, 2, 2]),
        ("SamplePerfectTile", [1, 3]),
        ("SamplePerfectTile", [1, 3]),
        ("SampleCategorical", 1),
        ("SampleComputeLocation", -1),
    ]
    decision_2 = [
        ("SamplePerfectTile", [1, 1, 1, 1]),
        ("SamplePerfectTile", [1, 4, 2, 14]),
        ("SamplePerfectTile", [1, 4, 7, 4]),
        ("SamplePerfectTile", [1, 8, 2, 2]),
        ("SamplePerfectTile", [1, 3]),
        ("SamplePerfectTile", [1, 3]),
        ("SampleCategorical", 0),
        ("SampleComputeLocation", 5),
    ]
    mod = create_te_workload("DEP", 0)
    actual = _design_space(mod)
    check_sketches(
        mod,
        sketches=actual,
        expected_mods=[dep_0, dep_1, dep_2],
        expected_decisions=[decision_0, decision_1, decision_2],
    )


def test_cpu_dil():
    # fmt: off
    @T.prim_func
    def dil_0(inputs: T.Buffer[(1, 224, 224, 3), "float32"], weight: T.Buffer[(7, 7, 3, 64), "float32"], conv2d_nhwc: T.Buffer[(1, 109, 109, 64), "float32"]) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # body
        with T.block("root"):
            T.reads()
            T.writes()
            T.block_attr({"meta_schedule.parallel":288, "meta_schedule.unroll_explicit":64, "meta_schedule.vectorize":64})
            PadInput = T.alloc_buffer([1, 230, 230, 3], dtype="float32")
            conv2d_nhwc_global = T.alloc_buffer([1, 109, 109, 64], dtype="float32")
            for i0_0, i1_0, i2_0, i3_0, i0_1, i1_1, i2_1, i3_1 in T.grid(1, 109, 1, 4, 1, 1, 1, 2):
                for ax0, ax1, ax2, ax3 in T.grid(1, 13, 229, 3):
                    with T.block("PadInput"):
                        i0 = T.axis.spatial(1, ax0)
                        i1 = T.axis.spatial(230, i1_0 * 2 + ax1)
                        i2 = T.axis.spatial(230, ax2)
                        i3 = T.axis.spatial(3, ax3)
                        T.reads(inputs[i0, i1 - 3, i2 - 3, i3])
                        T.writes(PadInput[i0, i1, i2, i3])
                        PadInput[i0, i1, i2, i3] = T.if_then_else(3 <= i1 and i1 < 227 and 3 <= i2 and i2 < 227, inputs[i0, i1 - 3, i2 - 3, i3], T.float32(0), dtype="float32")
                for i4_0, i5_0, i6_0, i0_2, i1_2, i2_2, i3_2, i4_1, i5_1, i6_1, i0_3, i1_3, i2_3, i3_3 in T.grid(7, 1, 1, 1, 1, 109, 8, 1, 7, 3, 1, 1, 1, 1):
                    with T.block("conv2d_nhwc"):
                        n = T.axis.spatial(1, i0_3 + i0_0 + i0_1 + i0_2)
                        h = T.axis.spatial(109, i1_2 + i1_3 + i1_0 + i1_1)
                        w = T.axis.spatial(109, i2_3 + i2_0 * 109 + i2_1 * 109 + i2_2)
                        co = T.axis.spatial(64, i3_0 * 16 + i3_1 * 8 + i3_2 + i3_3)
                        rh = T.axis.reduce(7, i4_1 + i4_0)
                        rw = T.axis.reduce(7, i5_0 * 7 + i5_1)
                        rc = T.axis.reduce(3, i6_0 * 3 + i6_1)
                        T.reads(PadInput[n, h * 2 + rh * 2, w * 2 + rw * 2, co // 64 * 3 + rc], weight[rh, rw, rc, co])
                        T.writes(conv2d_nhwc_global[n, h, w, co])
                        T.block_attr({"meta_schedule.tiling_structure":"SSRSRS"})
                        with T.init():
                            conv2d_nhwc_global[n, h, w, co] = T.float32(0)
                        conv2d_nhwc_global[n, h, w, co] = conv2d_nhwc_global[n, h, w, co] + PadInput[n, h * 2 + rh * 2, w * 2 + rw * 2, co // 64 * 3 + rc] * weight[rh, rw, rc, co]
                for ax0, ax1, ax2, ax3 in T.grid(1, 1, 109, 8):
                    with T.block("conv2d_nhwc_global"):
                        v0 = T.axis.spatial(1, ax0)
                        v1 = T.axis.spatial(109, i1_0 + ax1)
                        v2 = T.axis.spatial(109, ax2)
                        v3 = T.axis.spatial(64, i3_0 * 16 + i3_1 * 8 + ax3)
                        T.reads(conv2d_nhwc_global[v0, v1, v2, v3])
                        T.writes(conv2d_nhwc[v0, v1, v2, v3])
                        conv2d_nhwc[v0, v1, v2, v3] = conv2d_nhwc_global[v0, v1, v2, v3]
    @T.prim_func
    def dil_1(inputs: T.Buffer[(1, 224, 224, 3), "float32"], weight: T.Buffer[(7, 7, 3, 64), "float32"], conv2d_nhwc: T.Buffer[(1, 109, 109, 64), "float32"]) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # body
        with T.block("root"):
            T.reads()
            T.writes()
            T.block_attr({"meta_schedule.parallel":288, "meta_schedule.unroll_explicit":0, "meta_schedule.vectorize":64})
            PadInput = T.alloc_buffer([1, 230, 230, 3], dtype="float32")
            conv2d_nhwc_global = T.alloc_buffer([1, 109, 109, 64], dtype="float32")
            for i0_0, i1_0, i2_0, i3_0 in T.grid(1, 109, 1, 4):
                for i0_1, i1_1, i2_1, i3_1, i4_0 in T.grid(1, 1, 1, 2, 7):
                    for ax0, ax1, ax2, ax3 in T.grid(1, 1, 229, 3):
                        with T.block("PadInput"):
                            i0 = T.axis.spatial(1, ax0)
                            i1 = T.axis.spatial(230, i1_0 * 2 + i4_0 * 2 + ax1)
                            i2 = T.axis.spatial(230, ax2)
                            i3 = T.axis.spatial(3, ax3)
                            T.reads(inputs[i0, i1 - 3, i2 - 3, i3])
                            T.writes(PadInput[i0, i1, i2, i3])
                            PadInput[i0, i1, i2, i3] = T.if_then_else(3 <= i1 and i1 < 227 and 3 <= i2 and i2 < 227, inputs[i0, i1 - 3, i2 - 3, i3], T.float32(0), dtype="float32")
                    for i5_0, i6_0, i0_2, i1_2, i2_2, i3_2, i4_1, i5_1, i6_1, i0_3, i1_3, i2_3, i3_3 in T.grid(1, 1, 1, 1, 109, 8, 1, 7, 3, 1, 1, 1, 1):
                        with T.block("conv2d_nhwc"):
                            n = T.axis.spatial(1, i0_3 + i0_0 + i0_1 + i0_2)
                            h = T.axis.spatial(109, i1_2 + i1_3 + i1_0 + i1_1)
                            w = T.axis.spatial(109, i2_3 + i2_0 * 109 + i2_1 * 109 + i2_2)
                            co = T.axis.spatial(64, i3_0 * 16 + i3_1 * 8 + i3_2 + i3_3)
                            rh = T.axis.reduce(7, i4_1 + i4_0)
                            rw = T.axis.reduce(7, i5_0 * 7 + i5_1)
                            rc = T.axis.reduce(3, i6_0 * 3 + i6_1)
                            T.reads(PadInput[n, h * 2 + rh * 2, w * 2 + rw * 2, co // 64 * 3 + rc], weight[rh, rw, rc, co])
                            T.writes(conv2d_nhwc_global[n, h, w, co])
                            T.block_attr({"meta_schedule.tiling_structure":"SSRSRS"})
                            with T.init():
                                conv2d_nhwc_global[n, h, w, co] = T.float32(0)
                            conv2d_nhwc_global[n, h, w, co] = conv2d_nhwc_global[n, h, w, co] + PadInput[n, h * 2 + rh * 2, w * 2 + rw * 2, co // 64 * 3 + rc] * weight[rh, rw, rc, co]
                for ax0, ax1, ax2, ax3 in T.grid(1, 1, 109, 16):
                    with T.block("conv2d_nhwc_global"):
                        v0 = T.axis.spatial(1, ax0)
                        v1 = T.axis.spatial(109, i1_0 + ax1)
                        v2 = T.axis.spatial(109, ax2)
                        v3 = T.axis.spatial(64, i3_0 * 16 + ax3)
                        T.reads(conv2d_nhwc_global[v0, v1, v2, v3])
                        T.writes(conv2d_nhwc[v0, v1, v2, v3])
                        conv2d_nhwc[v0, v1, v2, v3] = conv2d_nhwc_global[v0, v1, v2, v3]
    @T.prim_func
    def dil_2(inputs: T.Buffer[(1, 224, 224, 3), "float32"], weight: T.Buffer[(7, 7, 3, 64), "float32"], conv2d_nhwc: T.Buffer[(1, 109, 109, 64), "float32"]) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # body
        with T.block("root"):
            T.reads()
            T.writes()
            T.block_attr({"meta_schedule.parallel":288, "meta_schedule.unroll_explicit":0, "meta_schedule.vectorize":64})
            PadInput = T.alloc_buffer([1, 230, 230, 3], dtype="float32")
            for i0_0, i1_0 in T.grid(1, 109):
                for ax0, ax1, ax2, ax3 in T.grid(1, 13, 229, 3):
                    with T.block("PadInput"):
                        i0 = T.axis.spatial(1, ax0)
                        i1 = T.axis.spatial(230, i1_0 * 2 + ax1)
                        i2 = T.axis.spatial(230, ax2)
                        i3 = T.axis.spatial(3, ax3)
                        T.reads(inputs[i0, i1 - 3, i2 - 3, i3])
                        T.writes(PadInput[i0, i1, i2, i3])
                        PadInput[i0, i1, i2, i3] = T.if_then_else(3 <= i1 and i1 < 227 and 3 <= i2 and i2 < 227, inputs[i0, i1 - 3, i2 - 3, i3], T.float32(0), dtype="float32")
                for i2_0, i3_0, i0_1, i1_1, i2_1, i3_1, i4_0, i5_0, i6_0, i0_2, i1_2, i2_2, i3_2, i4_1, i5_1, i6_1, i0_3, i1_3, i2_3, i3_3 in T.grid(1, 4, 1, 1, 1, 2, 7, 1, 1, 1, 1, 109, 8, 1, 7, 3, 1, 1, 1, 1):
                    with T.block("conv2d_nhwc"):
                        n = T.axis.spatial(1, i0_3 + i0_0 + i0_1 + i0_2)
                        h = T.axis.spatial(109, i1_2 + i1_3 + i1_0 + i1_1)
                        w = T.axis.spatial(109, i2_3 + i2_0 * 109 + i2_1 * 109 + i2_2)
                        co = T.axis.spatial(64, i3_0 * 16 + i3_1 * 8 + i3_2 + i3_3)
                        rh = T.axis.reduce(7, i4_1 + i4_0)
                        rw = T.axis.reduce(7, i5_0 * 7 + i5_1)
                        rc = T.axis.reduce(3, i6_0 * 3 + i6_1)
                        T.reads(PadInput[n, h * 2 + rh * 2, w * 2 + rw * 2, co // 64 * 3 + rc], weight[rh, rw, rc, co])
                        T.writes(conv2d_nhwc[n, h, w, co])
                        T.block_attr({"meta_schedule.tiling_structure":"SSRSRS"})
                        with T.init():
                            conv2d_nhwc[n, h, w, co] = T.float32(0)
                        conv2d_nhwc[n, h, w, co] = conv2d_nhwc[n, h, w, co] + PadInput[n, h * 2 + rh * 2, w * 2 + rw * 2, co // 64 * 3 + rc] * weight[rh, rw, rc, co]

    # fmt: on
    decision_0 = [
        ("SamplePerfectTile", [1, 1, 1, 1]),
        ("SamplePerfectTile", [109, 1, 1, 1]),
        ("SamplePerfectTile", [1, 1, 109, 1]),
        ("SamplePerfectTile", [4, 2, 8, 1]),
        ("SamplePerfectTile", [7, 1]),
        ("SamplePerfectTile", [1, 7]),
        ("SamplePerfectTile", [1, 3]),
        ("SampleCategorical", 2),
        ("SampleComputeLocation", 7),
    ]
    decision_1 = [
        ("SamplePerfectTile", [1, 1, 1, 1]),
        ("SamplePerfectTile", [109, 1, 1, 1]),
        ("SamplePerfectTile", [1, 1, 109, 1]),
        ("SamplePerfectTile", [4, 2, 8, 1]),
        ("SamplePerfectTile", [7, 1]),
        ("SamplePerfectTile", [1, 7]),
        ("SamplePerfectTile", [1, 3]),
        ("SampleCategorical", 0),
        ("SampleComputeLocation", 8),
    ]
    decision_2 = [
        ("SamplePerfectTile", [1, 1, 1, 1]),
        ("SamplePerfectTile", [109, 1, 1, 1]),
        ("SamplePerfectTile", [1, 1, 109, 1]),
        ("SamplePerfectTile", [4, 2, 8, 1]),
        ("SamplePerfectTile", [7, 1]),
        ("SamplePerfectTile", [1, 7]),
        ("SamplePerfectTile", [1, 3]),
        ("SampleCategorical", 0),
        ("SampleComputeLocation", 1),
    ]
    mod = create_te_workload("DIL", 0)
    actual = _design_space(mod)
    check_sketches(
        mod,
        sketches=actual,
        expected_mods=[dil_0, dil_1, dil_2],
        expected_decisions=[decision_0, decision_1, decision_2],
    )


def test_cpu_gmm():
    # fmt: off
    @T.prim_func
    def gmm_0(X: T.Buffer[(1, 128, 128), "float32"], Y: T.Buffer[(1, 128, 128), "float32"], Z: T.Buffer[(1, 128, 128), "float32"]) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # body
        with T.block("root"):
            T.reads()
            T.writes()
            T.block_attr({"meta_schedule.parallel":288, "meta_schedule.unroll_explicit":16, "meta_schedule.vectorize":64})
            Z_global = T.alloc_buffer([1, 128, 128], dtype="float32")
            for i0_0, i1_0, i2_0, i0_1, i1_1, i2_1 in T.grid(1, 4, 2, 1, 1, 8):
                for i3_0, i0_2, i1_2, i2_2, i3_1, i0_3, i1_3, i2_3 in T.grid(128, 1, 16, 1, 1, 1, 2, 8):
                    with T.block("Z"):
                        b = T.axis.spatial(1, i0_0 + i0_1 + i0_2 + i0_3)
                        i = T.axis.spatial(128, i1_0 * 32 + i1_1 * 32 + i1_2 * 2 + i1_3)
                        j = T.axis.spatial(128, i2_0 * 64 + i2_1 * 8 + i2_2 * 8 + i2_3)
                        k = T.axis.reduce(128, i3_1 + i3_0)
                        T.reads(X[b, i, k], Y[b, k, j])
                        T.writes(Z_global[b, i, j])
                        T.block_attr({"meta_schedule.tiling_structure":"SSRSRS"})
                        with T.init():
                            Z_global[b, i, j] = T.float32(0)
                        Z_global[b, i, j] = Z_global[b, i, j] + X[b, i, k] * Y[b, k, j]
                for ax0, ax1, ax2 in T.grid(1, 32, 8):
                    with T.block("Z_global"):
                        v0 = T.axis.spatial(1, ax0)
                        v1 = T.axis.spatial(128, i1_0 * 32 + ax1)
                        v2 = T.axis.spatial(128, i2_0 * 64 + i2_1 * 8 + ax2)
                        T.reads(Z_global[v0, v1, v2])
                        T.writes(Z[v0, v1, v2])
                        Z[v0, v1, v2] = Z_global[v0, v1, v2]
    @T.prim_func
    def gmm_1(X: T.Buffer[(1, 128, 128), "float32"], Y: T.Buffer[(1, 128, 128), "float32"], Z: T.Buffer[(1, 128, 128), "float32"]) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # body
        with T.block("root"):
            T.reads()
            T.writes()
            T.block_attr({"meta_schedule.parallel":288, "meta_schedule.unroll_explicit":16, "meta_schedule.vectorize":64})
            Z_global = T.alloc_buffer([1, 128, 128], dtype="float32")
            for i0_0, i1_0, i2_0 in T.grid(1, 4, 2):
                for i0_1, i1_1, i2_1, i3_0, i0_2, i1_2, i2_2, i3_1, i0_3, i1_3, i2_3 in T.grid(1, 1, 8, 128, 1, 16, 1, 1, 1, 2, 8):
                    with T.block("Z"):
                        b = T.axis.spatial(1, i0_0 + i0_1 + i0_2 + i0_3)
                        i = T.axis.spatial(128, i1_0 * 32 + i1_1 * 32 + i1_2 * 2 + i1_3)
                        j = T.axis.spatial(128, i2_0 * 64 + i2_1 * 8 + i2_2 * 8 + i2_3)
                        k = T.axis.reduce(128, i3_1 + i3_0)
                        T.reads(X[b, i, k], Y[b, k, j])
                        T.writes(Z_global[b, i, j])
                        T.block_attr({"meta_schedule.tiling_structure":"SSRSRS"})
                        with T.init():
                            Z_global[b, i, j] = T.float32(0)
                        Z_global[b, i, j] = Z_global[b, i, j] + X[b, i, k] * Y[b, k, j]
                for ax0, ax1, ax2 in T.grid(1, 32, 64):
                    with T.block("Z_global"):
                        v0 = T.axis.spatial(1, ax0)
                        v1 = T.axis.spatial(128, i1_0 * 32 + ax1)
                        v2 = T.axis.spatial(128, i2_0 * 64 + ax2)
                        T.reads(Z_global[v0, v1, v2])
                        T.writes(Z[v0, v1, v2])
                        Z[v0, v1, v2] = Z_global[v0, v1, v2]
    @T.prim_func
    def gmm_2(X: T.Buffer[(1, 128, 128), "float32"], Y: T.Buffer[(1, 128, 128), "float32"], Z: T.Buffer[(1, 128, 128), "float32"]) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # body
        with T.block("root"):
            T.reads()
            T.writes()
            T.block_attr({"meta_schedule.parallel":288, "meta_schedule.unroll_explicit":16, "meta_schedule.vectorize":64})
            for i0_0, i1_0, i2_0, i0_1, i1_1, i2_1, i3_0, i0_2, i1_2, i2_2, i3_1, i0_3, i1_3, i2_3 in T.grid(1, 4, 2, 1, 1, 8, 128, 1, 16, 1, 1, 1, 2, 8):
                with T.block("Z"):
                    b = T.axis.spatial(1, i0_0 + i0_1 + i0_2 + i0_3)
                    i = T.axis.spatial(128, i1_0 * 32 + i1_1 * 32 + i1_2 * 2 + i1_3)
                    j = T.axis.spatial(128, i2_0 * 64 + i2_1 * 8 + i2_2 * 8 + i2_3)
                    k = T.axis.reduce(128, i3_1 + i3_0)
                    T.reads(X[b, i, k], Y[b, k, j])
                    T.writes(Z[b, i, j])
                    T.block_attr({"meta_schedule.tiling_structure":"SSRSRS"})
                    with T.init():
                        Z[b, i, j] = T.float32(0)
                    Z[b, i, j] = Z[b, i, j] + X[b, i, k] * Y[b, k, j]
    # fmt: on
    decision_0 = [
        ("SamplePerfectTile", [1, 1, 1, 1]),
        ("SamplePerfectTile", [4, 1, 16, 2]),
        ("SamplePerfectTile", [2, 8, 1, 8]),
        ("SamplePerfectTile", [128, 1]),
        ("SampleCategorical", 1),
    ]
    decision_1 = [
        ("SamplePerfectTile", [1, 1, 1, 1]),
        ("SamplePerfectTile", [4, 1, 16, 2]),
        ("SamplePerfectTile", [2, 8, 1, 8]),
        ("SamplePerfectTile", [128, 1]),
        ("SampleCategorical", 1),
    ]
    decision_2 = [
        ("SamplePerfectTile", [1, 1, 1, 1]),
        ("SamplePerfectTile", [4, 1, 16, 2]),
        ("SamplePerfectTile", [2, 8, 1, 8]),
        ("SamplePerfectTile", [128, 1]),
        ("SampleCategorical", 1),
    ]
    mod = create_te_workload("GMM", 0)
    actual = _design_space(mod)
    check_sketches(
        mod,
        sketches=actual,
        expected_mods=[gmm_0, gmm_1, gmm_2],
        expected_decisions=[decision_0, decision_1, decision_2],
    )


def test_cpu_grp():
    # fmt: off
    @T.prim_func
    def grp_0(inputs: T.Buffer[(1, 56, 56, 64), "float32"], weight: T.Buffer[(3, 3, 16, 128), "float32"], conv2d_nhwc: T.Buffer[(1, 28, 28, 128), "float32"]) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # body
        with T.block("root"):
            T.reads()
            T.writes()
            T.block_attr({"meta_schedule.parallel":288, "meta_schedule.unroll_explicit":16, "meta_schedule.vectorize":64})
            PadInput = T.alloc_buffer([1, 58, 58, 64], dtype="float32")
            conv2d_nhwc_global = T.alloc_buffer([1, 28, 28, 128], dtype="float32")
            for i0_0, i1_0, i2_0, i3_0 in T.grid(1, 7, 1, 2):
                for ax0, ax1, ax2, ax3 in T.grid(1, 9, 57, 32):
                    with T.block("PadInput"):
                        i0 = T.axis.spatial(1, ax0)
                        i1 = T.axis.spatial(58, i1_0 * 8 + ax1)
                        i2 = T.axis.spatial(58, ax2)
                        i3 = T.axis.spatial(64, i3_0 * 32 + ax3)
                        T.reads(inputs[i0, i1 - 1, i2 - 1, i3])
                        T.writes(PadInput[i0, i1, i2, i3])
                        PadInput[i0, i1, i2, i3] = T.if_then_else(1 <= i1 and i1 < 57 and 1 <= i2 and i2 < 57, inputs[i0, i1 - 1, i2 - 1, i3], T.float32(0), dtype="float32")
                for i0_1, i1_1, i2_1, i3_1 in T.grid(1, 4, 1, 1):
                    for i4_0, i5_0, i6_0, i0_2, i1_2, i2_2, i3_2, i4_1, i5_1, i6_1, i0_3, i1_3, i2_3, i3_3 in T.grid(1, 3, 8, 1, 1, 4, 4, 3, 1, 2, 1, 1, 7, 16):
                        with T.block("conv2d_nhwc"):
                            n = T.axis.spatial(1, i0_3 + i0_0 + i0_1 + i0_2)
                            h = T.axis.spatial(28, i1_0 * 4 + i1_1 + i1_2 + i1_3)
                            w = T.axis.spatial(28, i2_0 * 28 + i2_1 * 28 + i2_2 * 7 + i2_3)
                            co = T.axis.spatial(128, i3_0 * 64 + i3_1 * 64 + i3_2 * 16 + i3_3)
                            rh = T.axis.reduce(3, i4_0 * 3 + i4_1)
                            rw = T.axis.reduce(3, i5_0 + i5_1)
                            rc = T.axis.reduce(16, i6_0 * 2 + i6_1)
                            T.reads(PadInput[n, h * 2 + rh, w * 2 + rw, co // 32 * 16 + rc], weight[rh, rw, rc, co])
                            T.writes(conv2d_nhwc_global[n, h, w, co])
                            T.block_attr({"meta_schedule.tiling_structure":"SSRSRS"})
                            with T.init():
                                conv2d_nhwc_global[n, h, w, co] = T.float32(0)
                            conv2d_nhwc_global[n, h, w, co] = conv2d_nhwc_global[n, h, w, co] + PadInput[n, h * 2 + rh, w * 2 + rw, co // 32 * 16 + rc] * weight[rh, rw, rc, co]
                    for ax0, ax1, ax2, ax3 in T.grid(1, 1, 28, 64):
                        with T.block("conv2d_nhwc_global"):
                            v0 = T.axis.spatial(1, ax0)
                            v1 = T.axis.spatial(28, i1_0 * 4 + i1_1 + ax1)
                            v2 = T.axis.spatial(28, ax2)
                            v3 = T.axis.spatial(128, i3_0 * 64 + ax3)
                            T.reads(conv2d_nhwc_global[v0, v1, v2, v3])
                            T.writes(conv2d_nhwc[v0, v1, v2, v3])
                            conv2d_nhwc[v0, v1, v2, v3] = conv2d_nhwc_global[v0, v1, v2, v3]
    @T.prim_func
    def grp_1(inputs: T.Buffer[(1, 56, 56, 64), "float32"], weight: T.Buffer[(3, 3, 16, 128), "float32"], conv2d_nhwc: T.Buffer[(1, 28, 28, 128), "float32"]) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # body
        with T.block("root"):
            T.reads()
            T.writes()
            T.block_attr({"meta_schedule.parallel":288, "meta_schedule.unroll_explicit":512, "meta_schedule.vectorize":64})
            PadInput = T.alloc_buffer([1, 58, 58, 64], dtype="float32")
            conv2d_nhwc_global = T.alloc_buffer([1, 28, 28, 128], dtype="float32")
            for i0, i1, i2, i3 in T.grid(1, 58, 58, 64):
                with T.block("PadInput"):
                    i0_1, i1_1, i2_1, i3_1 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                    T.reads(inputs[i0_1, i1_1 - 1, i2_1 - 1, i3_1])
                    T.writes(PadInput[i0_1, i1_1, i2_1, i3_1])
                    PadInput[i0_1, i1_1, i2_1, i3_1] = T.if_then_else(1 <= i1_1 and i1_1 < 57 and 1 <= i2_1 and i2_1 < 57, inputs[i0_1, i1_1 - 1, i2_1 - 1, i3_1], T.float32(0), dtype="float32")
            for i0_0, i1_0, i2_0, i3_0 in T.grid(1, 7, 1, 2):
                for i0_1_1, i1_1_1, i2_1_1, i3_1_1, i4_0, i5_0, i6_0, i0_2, i1_2, i2_2, i3_2, i4_1, i5_1, i6_1, i0_3, i1_3, i2_3, i3_3 in T.grid(1, 4, 1, 1, 1, 3, 8, 1, 1, 4, 4, 3, 1, 2, 1, 1, 7, 16):
                    with T.block("conv2d_nhwc"):
                        n = T.axis.spatial(1, i0_3 + i0_0 + i0_1_1 + i0_2)
                        h = T.axis.spatial(28, i1_0 * 4 + i1_1_1 + i1_2 + i1_3)
                        w = T.axis.spatial(28, i2_0 * 28 + i2_1_1 * 28 + i2_2 * 7 + i2_3)
                        co = T.axis.spatial(128, i3_0 * 64 + i3_1_1 * 64 + i3_2 * 16 + i3_3)
                        rh = T.axis.reduce(3, i4_0 * 3 + i4_1)
                        rw = T.axis.reduce(3, i5_0 + i5_1)
                        rc = T.axis.reduce(16, i6_0 * 2 + i6_1)
                        T.reads(PadInput[n, h * 2 + rh, w * 2 + rw, co // 32 * 16 + rc], weight[rh, rw, rc, co])
                        T.writes(conv2d_nhwc_global[n, h, w, co])
                        T.block_attr({"meta_schedule.tiling_structure":"SSRSRS"})
                        with T.init():
                            conv2d_nhwc_global[n, h, w, co] = T.float32(0)
                        conv2d_nhwc_global[n, h, w, co] = conv2d_nhwc_global[n, h, w, co] + PadInput[n, h * 2 + rh, w * 2 + rw, co // 32 * 16 + rc] * weight[rh, rw, rc, co]
                for ax0, ax1, ax2, ax3 in T.grid(1, 4, 28, 64):
                    with T.block("conv2d_nhwc_global"):
                        v0 = T.axis.spatial(1, ax0)
                        v1 = T.axis.spatial(28, i1_0 * 4 + ax1)
                        v2 = T.axis.spatial(28, ax2)
                        v3 = T.axis.spatial(128, i3_0 * 64 + ax3)
                        T.reads(conv2d_nhwc_global[v0, v1, v2, v3])
                        T.writes(conv2d_nhwc[v0, v1, v2, v3])
                        conv2d_nhwc[v0, v1, v2, v3] = conv2d_nhwc_global[v0, v1, v2, v3]
    @T.prim_func
    def grp_2(inputs: T.Buffer[(1, 56, 56, 64), "float32"], weight: T.Buffer[(3, 3, 16, 128), "float32"], conv2d_nhwc: T.Buffer[(1, 28, 28, 128), "float32"]) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # body
        with T.block("root"):
            T.reads()
            T.writes()
            T.block_attr({"meta_schedule.parallel":288, "meta_schedule.unroll_explicit":16, "meta_schedule.vectorize":64})
            PadInput = T.alloc_buffer([1, 58, 58, 64], dtype="float32")
            for i0_0, i1_0, i2_0, i3_0, i0_1, i1_1, i2_1, i3_1, i4_0, i5_0 in T.grid(1, 7, 1, 2, 1, 4, 1, 1, 1, 3):
                for ax0, ax1, ax2, ax3 in T.grid(1, 3, 55, 32):
                    with T.block("PadInput"):
                        i0 = T.axis.spatial(1, ax0)
                        i1 = T.axis.spatial(58, i1_0 * 8 + i1_1 * 2 + ax1)
                        i2 = T.axis.spatial(58, i5_0 + ax2)
                        i3 = T.axis.spatial(64, i3_0 * 32 + ax3)
                        T.reads(inputs[i0, i1 - 1, i2 - 1, i3])
                        T.writes(PadInput[i0, i1, i2, i3])
                        PadInput[i0, i1, i2, i3] = T.if_then_else(1 <= i1 and i1 < 57 and 1 <= i2 and i2 < 57, inputs[i0, i1 - 1, i2 - 1, i3], T.float32(0), dtype="float32")
                for i6_0, i0_2, i1_2, i2_2, i3_2, i4_1, i5_1, i6_1, i0_3, i1_3, i2_3, i3_3 in T.grid(8, 1, 1, 4, 4, 3, 1, 2, 1, 1, 7, 16):
                    with T.block("conv2d_nhwc"):
                        n = T.axis.spatial(1, i0_3 + i0_0 + i0_1 + i0_2)
                        h = T.axis.spatial(28, i1_0 * 4 + i1_1 + i1_2 + i1_3)
                        w = T.axis.spatial(28, i2_0 * 28 + i2_1 * 28 + i2_2 * 7 + i2_3)
                        co = T.axis.spatial(128, i3_0 * 64 + i3_1 * 64 + i3_2 * 16 + i3_3)
                        rh = T.axis.reduce(3, i4_0 * 3 + i4_1)
                        rw = T.axis.reduce(3, i5_0 + i5_1)
                        rc = T.axis.reduce(16, i6_0 * 2 + i6_1)
                        T.reads(PadInput[n, h * 2 + rh, w * 2 + rw, co // 32 * 16 + rc], weight[rh, rw, rc, co])
                        T.writes(conv2d_nhwc[n, h, w, co])
                        T.block_attr({"meta_schedule.tiling_structure":"SSRSRS"})
                        with T.init():
                            conv2d_nhwc[n, h, w, co] = T.float32(0)
                        conv2d_nhwc[n, h, w, co] = conv2d_nhwc[n, h, w, co] + PadInput[n, h * 2 + rh, w * 2 + rw, co // 32 * 16 + rc] * weight[rh, rw, rc, co]
    # fmt: on
    decision_0 = [
        ("SamplePerfectTile", [1, 1, 1, 1]),
        ("SamplePerfectTile", [7, 4, 1, 1]),
        ("SamplePerfectTile", [1, 1, 4, 7]),
        ("SamplePerfectTile", [2, 1, 4, 16]),
        ("SamplePerfectTile", [1, 3]),
        ("SamplePerfectTile", [3, 1]),
        ("SamplePerfectTile", [8, 2]),
        ("SampleCategorical", 1),
        ("SampleComputeLocation", 3),
    ]
    decision_1 = [
        ("SamplePerfectTile", [1, 1, 1, 1]),
        ("SamplePerfectTile", [7, 4, 1, 1]),
        ("SamplePerfectTile", [1, 1, 4, 7]),
        ("SamplePerfectTile", [2, 1, 4, 16]),
        ("SamplePerfectTile", [1, 3]),
        ("SamplePerfectTile", [3, 1]),
        ("SamplePerfectTile", [8, 2]),
        ("SampleCategorical", 3),
        ("SampleComputeLocation", -1),
    ]
    decision_2 = [
        ("SamplePerfectTile", [1, 1, 1, 1]),
        ("SamplePerfectTile", [7, 4, 1, 1]),
        ("SamplePerfectTile", [1, 1, 4, 7]),
        ("SamplePerfectTile", [2, 1, 4, 16]),
        ("SamplePerfectTile", [1, 3]),
        ("SamplePerfectTile", [3, 1]),
        ("SamplePerfectTile", [8, 2]),
        ("SampleCategorical", 1),
        ("SampleComputeLocation", 9),
    ]
    mod = create_te_workload("GRP", 0)
    actual = _design_space(mod)
    check_sketches(
        mod,
        sketches=actual,
        expected_mods=[grp_0, grp_1, grp_2],
        expected_decisions=[decision_0, decision_1, decision_2],
    )


def test_cpu_t2d():
    # fmt: off
    @T.prim_func
    def t2d_0(inputs: T.Buffer[(1, 4, 4, 512), "float32"], weight: T.Buffer[(4, 4, 512, 256), "float32"], conv2d_transpose_nhwc: T.Buffer[(1, 8, 8, 256), "float32"]) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # body
        with T.block("root"):
            T.reads()
            T.writes()
            T.block_attr({"meta_schedule.parallel":288, "meta_schedule.unroll_explicit":64, "meta_schedule.vectorize":64})
            PadInput = T.alloc_buffer([1, 6, 6, 512], dtype="float32")
            conv2d_transpose_nhwc_global = T.alloc_buffer([1, 8, 8, 256], dtype="float32")
            for i0, i1, i2, i3 in T.grid(1, 6, 6, 512):
                with T.block("PadInput"):
                    i0_1, i1_1, i2_1, i3_1 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                    T.reads(inputs[i0_1, i1_1 - 1, i2_1 - 1, i3_1])
                    T.writes(PadInput[i0_1, i1_1, i2_1, i3_1])
                    PadInput[i0_1, i1_1, i2_1, i3_1] = T.if_then_else(1 <= i1_1 and i1_1 < 5 and 1 <= i2_1 and i2_1 < 5, inputs[i0_1, i1_1 - 1, i2_1 - 1, i3_1], T.float32(0), dtype="float32")
            for i0_0, i1_0, i2_0, i3_0, i0_1_1, i1_1_1, i2_1_1, i3_1_1 in T.grid(1, 1, 2, 8, 1, 4, 1, 4):
                for i4_0, i5_0, i6_0, i0_2, i1_2, i2_2, i3_2, i4_1, i5_1, i6_1, i0_3, i1_3, i2_3, i3_3 in T.grid(2, 2, 64, 1, 1, 1, 1, 2, 2, 8, 1, 2, 4, 8):
                    with T.block("conv2d_transpose_nhwc"):
                        n = T.axis.spatial(1, i0_3 + i0_0 + i0_1_1 + i0_2)
                        h = T.axis.spatial(8, i1_0 * 8 + i1_1_1 * 2 + i1_2 * 2 + i1_3)
                        w = T.axis.spatial(8, i2_0 * 4 + i2_1_1 * 4 + i2_2 * 4 + i2_3)
                        co = T.axis.spatial(256, i3_0 * 32 + i3_1_1 * 8 + i3_2 * 8 + i3_3)
                        rh = T.axis.reduce(4, i4_0 * 2 + i4_1)
                        rw = T.axis.reduce(4, i5_0 * 2 + i5_1)
                        rc = T.axis.reduce(512, i6_0 * 8 + i6_1)
                        T.reads(PadInput[n, (h + rh) // 2, (w + rw) // 2, rc], weight[3 - rh, 3 - rw, rc, co])
                        T.writes(conv2d_transpose_nhwc_global[n, h, w, co])
                        T.block_attr({"meta_schedule.tiling_structure":"SSRSRS"})
                        with T.init():
                            conv2d_transpose_nhwc_global[n, h, w, co] = T.float32(0)
                        conv2d_transpose_nhwc_global[n, h, w, co] = conv2d_transpose_nhwc_global[n, h, w, co] + T.if_then_else((h + rh) % 2 == 0 and (w + rw) % 2 == 0, PadInput[n, (h + rh) // 2, (w + rw) // 2, rc], T.float32(0), dtype="float32") * weight[3 - rh, 3 - rw, rc, co]
                for ax0, ax1, ax2, ax3 in T.grid(1, 2, 4, 8):
                    with T.block("conv2d_transpose_nhwc_global"):
                        v0 = T.axis.spatial(1, ax0)
                        v1 = T.axis.spatial(8, i1_1_1 * 2 + ax1)
                        v2 = T.axis.spatial(8, i2_0 * 4 + ax2)
                        v3 = T.axis.spatial(256, i3_0 * 32 + i3_1_1 * 8 + ax3)
                        T.reads(conv2d_transpose_nhwc_global[v0, v1, v2, v3])
                        T.writes(conv2d_transpose_nhwc[v0, v1, v2, v3])
                        conv2d_transpose_nhwc[v0, v1, v2, v3] = conv2d_transpose_nhwc_global[v0, v1, v2, v3]
    @T.prim_func
    def t2d_1(inputs: T.Buffer[(1, 4, 4, 512), "float32"], weight: T.Buffer[(4, 4, 512, 256), "float32"], conv2d_transpose_nhwc: T.Buffer[(1, 8, 8, 256), "float32"]) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # body
        with T.block("root"):
            T.reads()
            T.writes()
            T.block_attr({"meta_schedule.parallel":288, "meta_schedule.unroll_explicit":64, "meta_schedule.vectorize":64})
            PadInput = T.alloc_buffer([1, 6, 6, 512], dtype="float32")
            conv2d_transpose_nhwc_global = T.alloc_buffer([1, 8, 8, 256], dtype="float32")
            for i0_0, i1_0, i2_0, i3_0 in T.grid(1, 1, 2, 8):
                for ax0, ax1, ax2, ax3 in T.grid(1, 6, 4, 512):
                    with T.block("PadInput"):
                        i0, i1 = T.axis.remap("SS", [ax0, ax1])
                        i2 = T.axis.spatial(6, i2_0 * 2 + ax2)
                        i3 = T.axis.spatial(512, ax3)
                        T.reads(inputs[i0, i1 - 1, i2 - 1, i3])
                        T.writes(PadInput[i0, i1, i2, i3])
                        PadInput[i0, i1, i2, i3] = T.if_then_else(1 <= i1 and i1 < 5 and 1 <= i2 and i2 < 5, inputs[i0, i1 - 1, i2 - 1, i3], T.float32(0), dtype="float32")
                for i0_1, i1_1, i2_1, i3_1, i4_0, i5_0, i6_0, i0_2, i1_2, i2_2, i3_2, i4_1, i5_1, i6_1, i0_3, i1_3, i2_3, i3_3 in T.grid(1, 4, 1, 4, 2, 2, 64, 1, 1, 1, 1, 2, 2, 8, 1, 2, 4, 8):
                    with T.block("conv2d_transpose_nhwc"):
                        n = T.axis.spatial(1, i0_3 + i0_0 + i0_1 + i0_2)
                        h = T.axis.spatial(8, i1_0 * 8 + i1_1 * 2 + i1_2 * 2 + i1_3)
                        w = T.axis.spatial(8, i2_0 * 4 + i2_1 * 4 + i2_2 * 4 + i2_3)
                        co = T.axis.spatial(256, i3_0 * 32 + i3_1 * 8 + i3_2 * 8 + i3_3)
                        rh = T.axis.reduce(4, i4_0 * 2 + i4_1)
                        rw = T.axis.reduce(4, i5_0 * 2 + i5_1)
                        rc = T.axis.reduce(512, i6_0 * 8 + i6_1)
                        T.reads(PadInput[n, (h + rh) // 2, (w + rw) // 2, rc], weight[3 - rh, 3 - rw, rc, co])
                        T.writes(conv2d_transpose_nhwc_global[n, h, w, co])
                        T.block_attr({"meta_schedule.tiling_structure":"SSRSRS"})
                        with T.init():
                            conv2d_transpose_nhwc_global[n, h, w, co] = T.float32(0)
                        conv2d_transpose_nhwc_global[n, h, w, co] = conv2d_transpose_nhwc_global[n, h, w, co] + T.if_then_else((h + rh) % 2 == 0 and (w + rw) % 2 == 0, PadInput[n, (h + rh) // 2, (w + rw) // 2, rc], T.float32(0), dtype="float32") * weight[3 - rh, 3 - rw, rc, co]
                for ax0, ax1, ax2, ax3 in T.grid(1, 8, 4, 32):
                    with T.block("conv2d_transpose_nhwc_global"):
                        v0, v1 = T.axis.remap("SS", [ax0, ax1])
                        v2 = T.axis.spatial(8, i2_0 * 4 + ax2)
                        v3 = T.axis.spatial(256, i3_0 * 32 + ax3)
                        T.reads(conv2d_transpose_nhwc_global[v0, v1, v2, v3])
                        T.writes(conv2d_transpose_nhwc[v0, v1, v2, v3])
                        conv2d_transpose_nhwc[v0, v1, v2, v3] = conv2d_transpose_nhwc_global[v0, v1, v2, v3]
    @T.prim_func
    def t2d_2(inputs: T.Buffer[(1, 4, 4, 512), "float32"], weight: T.Buffer[(4, 4, 512, 256), "float32"], conv2d_transpose_nhwc: T.Buffer[(1, 8, 8, 256), "float32"]) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # body
        with T.block("root"):
            T.reads()
            T.writes()
            T.block_attr({"meta_schedule.parallel":288, "meta_schedule.unroll_explicit":512, "meta_schedule.vectorize":64})
            for i0_0, i1_0, i2_0, i3_0, i0_1, i1_1, i2_1, i3_1, i4_0, i5_0, i6_0, i0_2, i1_2, i2_2, i3_2, i4_1, i5_1, i6_1, i0_3, i1_3, i2_3, i3_3 in T.grid(1, 1, 2, 8, 1, 4, 1, 4, 2, 2, 64, 1, 1, 1, 1, 2, 2, 8, 1, 2, 4, 8):
                with T.block("conv2d_transpose_nhwc"):
                    n = T.axis.spatial(1, i0_3 + i0_0 + i0_1 + i0_2)
                    h = T.axis.spatial(8, i1_0 * 8 + i1_1 * 2 + i1_2 * 2 + i1_3)
                    w = T.axis.spatial(8, i2_0 * 4 + i2_1 * 4 + i2_2 * 4 + i2_3)
                    co = T.axis.spatial(256, i3_0 * 32 + i3_1 * 8 + i3_2 * 8 + i3_3)
                    rh = T.axis.reduce(4, i4_0 * 2 + i4_1)
                    rw = T.axis.reduce(4, i5_0 * 2 + i5_1)
                    rc = T.axis.reduce(512, i6_0 * 8 + i6_1)
                    T.reads(inputs[n, (h + rh) // 2 - 1, (w + rw) // 2 - 1, rc], weight[3 - rh, 3 - rw, rc, co])
                    T.writes(conv2d_transpose_nhwc[n, h, w, co])
                    T.block_attr({"meta_schedule.tiling_structure":"SSRSRS"})
                    with T.init():
                        conv2d_transpose_nhwc[n, h, w, co] = T.float32(0)
                    conv2d_transpose_nhwc[n, h, w, co] = conv2d_transpose_nhwc[n, h, w, co] + T.if_then_else((h + rh) % 2 == 0 and (w + rw) % 2 == 0, T.if_then_else(1 <= (h + rh) // 2 and (h + rh) // 2 < 5 and 1 <= (w + rw) // 2 and (w + rw) // 2 < 5, inputs[n, (h + rh) // 2 - 1, (w + rw) // 2 - 1, rc], T.float32(0), dtype="float32"), T.float32(0), dtype="float32") * weight[3 - rh, 3 - rw, rc, co]
    # fmt: on
    decision_0 = [
        ("SamplePerfectTile", [1, 1, 1, 1]),
        ("SamplePerfectTile", [1, 4, 1, 2]),
        ("SamplePerfectTile", [2, 1, 1, 4]),
        ("SamplePerfectTile", [8, 4, 1, 8]),
        ("SamplePerfectTile", [2, 2]),
        ("SamplePerfectTile", [2, 2]),
        ("SamplePerfectTile", [64, 8]),
        ("SampleCategorical", 2),
        ("SampleComputeLocation", -1),
    ]
    decision_1 = [
        ("SamplePerfectTile", [1, 1, 1, 1]),
        ("SamplePerfectTile", [1, 4, 1, 2]),
        ("SamplePerfectTile", [2, 1, 1, 4]),
        ("SamplePerfectTile", [8, 4, 1, 8]),
        ("SamplePerfectTile", [2, 2]),
        ("SamplePerfectTile", [2, 2]),
        ("SamplePerfectTile", [64, 8]),
        ("SampleCategorical", 2),
        ("SampleComputeLocation", 3),
    ]
    decision_2 = [
        ("SamplePerfectTile", [1, 1, 1, 1]),
        ("SamplePerfectTile", [1, 4, 1, 2]),
        ("SamplePerfectTile", [2, 1, 1, 4]),
        ("SamplePerfectTile", [8, 4, 1, 8]),
        ("SamplePerfectTile", [2, 2]),
        ("SamplePerfectTile", [2, 2]),
        ("SamplePerfectTile", [64, 8]),
        ("SampleCategorical", 3),
        ("SampleComputeLocation", -2),
    ]
    mod = create_te_workload("T2D", 0)
    actual = _design_space(mod)
    check_sketches(
        mod,
        sketches=actual,
        expected_mods=[t2d_0, t2d_1, t2d_2],
        expected_decisions=[decision_0, decision_1, decision_2],
        debug_mask=0,
    )


def test_cpu_nrm():
    # fmt: off
    @T.prim_func
    def nrm_0(A: T.Buffer[(1, 256, 256), "float32"], D: T.Buffer[1, "float32"]) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # body
        with T.block("root"):
            T.reads()
            T.writes()
            T.block_attr({"meta_schedule.parallel":288, "meta_schedule.unroll_explicit":0, "meta_schedule.vectorize":64})
            C = T.alloc_buffer([1], dtype="float32")
            C_rf = T.alloc_buffer([1, 32768], dtype="float32")
            for i0, i1_i2_fused_0, i1_i2_fused_1 in T.grid(1, 32768, 2):
                with T.block("C_rf"):
                    vi1_i2_fused_0, b, vi1_i2_fused_1 = T.axis.remap("SSR", [i1_i2_fused_0, i0, i1_i2_fused_1])
                    T.reads(A[b, (vi1_i2_fused_0 * 2 + vi1_i2_fused_1) // 256, (vi1_i2_fused_0 * 2 + vi1_i2_fused_1) % 256])
                    T.writes(C_rf[b, vi1_i2_fused_0])
                    with T.init():
                        C_rf[b, vi1_i2_fused_0] = T.float32(0)
                    C_rf[b, vi1_i2_fused_0] = C_rf[b, vi1_i2_fused_0] + A[b, (vi1_i2_fused_0 * 2 + vi1_i2_fused_1) // 256, (vi1_i2_fused_0 * 2 + vi1_i2_fused_1) % 256] * A[b, (vi1_i2_fused_0 * 2 + vi1_i2_fused_1) // 256, (vi1_i2_fused_0 * 2 + vi1_i2_fused_1) % 256]
            for i0, i1_i2_fused_0 in T.grid(1, 32768):
                with T.block("C"):
                    vi1_i2_fused_0, b = T.axis.remap("RS", [i1_i2_fused_0, i0])
                    T.reads(C_rf[b, vi1_i2_fused_0])
                    T.writes(C[b])
                    with T.init():
                        C[b] = T.float32(0)
                    C[b] = C[b] + C_rf[b, vi1_i2_fused_0]
            for i0 in T.serial(1):
                with T.block("D"):
                    b = T.axis.spatial(1, i0)
                    T.reads(C[b])
                    T.writes(D[b])
                    D[b] = T.sqrt(C[b], dtype="float32")
    @T.prim_func
    def nrm_1(A: T.Buffer[(1, 256, 256), "float32"], D: T.Buffer[1, "float32"]) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # body
        with T.block("root"):
            T.reads()
            T.writes()
            T.block_attr({"meta_schedule.parallel":288, "meta_schedule.unroll_explicit":16, "meta_schedule.vectorize":64})
            C = T.alloc_buffer([1], dtype="float32")
            C_rf = T.alloc_buffer([1, 2], dtype="float32")
            for i0, i1_i2_fused_0, i1_i2_fused_1 in T.grid(1, 32768, 2):
                with T.block("C_rf"):
                    vi1_i2_fused_1, b, vi1_i2_fused_0 = T.axis.remap("SSR", [i1_i2_fused_1, i0, i1_i2_fused_0])
                    T.reads(A[b, (vi1_i2_fused_0 * 2 + vi1_i2_fused_1) // 256, (vi1_i2_fused_0 * 2 + vi1_i2_fused_1) % 256])
                    T.writes(C_rf[b, vi1_i2_fused_1])
                    with T.init():
                        C_rf[b, vi1_i2_fused_1] = T.float32(0)
                    C_rf[b, vi1_i2_fused_1] = C_rf[b, vi1_i2_fused_1] + A[b, (vi1_i2_fused_0 * 2 + vi1_i2_fused_1) // 256, (vi1_i2_fused_0 * 2 + vi1_i2_fused_1) % 256] * A[b, (vi1_i2_fused_0 * 2 + vi1_i2_fused_1) // 256, (vi1_i2_fused_0 * 2 + vi1_i2_fused_1) % 256]
            for i0, i1_i2_fused_1 in T.grid(1, 2):
                with T.block("C"):
                    vi1_i2_fused_1, b = T.axis.remap("RS", [i1_i2_fused_1, i0])
                    T.reads(C_rf[b, vi1_i2_fused_1])
                    T.writes(C[b])
                    with T.init():
                        C[b] = T.float32(0)
                    C[b] = C[b] + C_rf[b, vi1_i2_fused_1]
            for i0 in T.serial(1):
                with T.block("D"):
                    b = T.axis.spatial(1, i0)
                    T.reads(C[b])
                    T.writes(D[b])
                    D[b] = T.sqrt(C[b], dtype="float32")
    @T.prim_func
    def nrm_2(A: T.Buffer[(1, 256, 256), "float32"], D: T.Buffer[1, "float32"]) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # body
        with T.block("root"):
            T.reads()
            T.writes()
            T.block_attr({"meta_schedule.parallel":288, "meta_schedule.unroll_explicit":0, "meta_schedule.vectorize":64})
            C = T.alloc_buffer([1], dtype="float32")
            for i0, i1, i2 in T.grid(1, 256, 256):
                with T.block("C"):
                    b, i, j = T.axis.remap("SRR", [i0, i1, i2])
                    T.reads(A[b, i, j])
                    T.writes(C[b])
                    with T.init():
                        C[b] = T.float32(0)
                    C[b] = C[b] + A[b, i, j] * A[b, i, j]
            for i0 in T.serial(1):
                with T.block("D"):
                    b = T.axis.spatial(1, i0)
                    T.reads(C[b])
                    T.writes(D[b])
                    D[b] = T.sqrt(C[b], dtype="float32")
    # fmt: on
    decision_0 = [
        ("SamplePerfectTile", [32768, 2]),
        ("SampleCategorical", 0),
        ("SampleComputeLocation", -1),
        ("SampleComputeLocation", -1),
    ]
    decision_1 = [
        ("SamplePerfectTile", [32768, 2]),
        ("SampleCategorical", 1),
        ("SampleComputeLocation", -1),
        ("SampleComputeLocation", -1),
    ]
    decision_2 = [
        ("SampleCategorical", 0),
        ("SampleComputeLocation", -1),
    ]
    mod = create_te_workload("NRM", 0)
    actual = _design_space(mod)
    check_sketches(
        mod,
        sketches=actual,
        expected_mods=[nrm_0, nrm_1, nrm_2],
        expected_decisions=[decision_0, decision_1, decision_2],
    )


def test_cpu_sfm():
    # fmt: off
    @T.prim_func
    def sfm_0(A: T.Buffer[(256, 256), "float32"], T_softmax_norm: T.Buffer[(256, 256), "float32"]) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # body
        with T.block("root"):
            T.reads()
            T.writes()
            T.block_attr({"meta_schedule.parallel":288, "meta_schedule.unroll_explicit":0, "meta_schedule.vectorize":64})
            T_softmax_maxelem = T.alloc_buffer([256], dtype="float32")
            T_softmax_expsum = T.alloc_buffer([256], dtype="float32")
            T_softmax_expsum_rf = T.alloc_buffer([256, 16], dtype="float32")
            T_softmax_maxelem_rf = T.alloc_buffer([256, 4], dtype="float32")
            for i0, i1_0, i1_1 in T.grid(256, 4, 64):
                with T.block("T_softmax_maxelem_rf"):
                    vi1_0, i0_1, vi1_1 = T.axis.remap("SSR", [i1_0, i0, i1_1])
                    T.reads(A[i0_1, vi1_0 * 64 + vi1_1])
                    T.writes(T_softmax_maxelem_rf[i0_1, vi1_0])
                    with T.init():
                        T_softmax_maxelem_rf[i0_1, vi1_0] = T.float32(-3.4028234663852886e+38)
                    T_softmax_maxelem_rf[i0_1, vi1_0] = T.max(T_softmax_maxelem_rf[i0_1, vi1_0], A[i0_1, vi1_0 * 64 + vi1_1])
            for i0, i1_0 in T.grid(256, 4):
                with T.block("T_softmax_maxelem"):
                    vi1_0, i0_2 = T.axis.remap("RS", [i1_0, i0])
                    T.reads(T_softmax_maxelem_rf[i0_2, vi1_0])
                    T.writes(T_softmax_maxelem[i0_2])
                    with T.init():
                        T_softmax_maxelem[i0_2] = T.float32(-3.4028234663852886e+38)
                    T_softmax_maxelem[i0_2] = T.max(T_softmax_maxelem[i0_2], T_softmax_maxelem_rf[i0_2, vi1_0])
            for i0_3, i1_0, i1_1 in T.grid(256, 16, 16):
                with T.block("T_softmax_expsum_rf"):
                    vi1_0, i0_4, vi1_1 = T.axis.remap("SSR", [i1_0, i0_3, i1_1])
                    T.reads(A[i0_4, vi1_0 * 16 + vi1_1], T_softmax_maxelem[i0_4])
                    T.writes(T_softmax_expsum_rf[i0_4, vi1_0])
                    with T.init():
                        T_softmax_expsum_rf[i0_4, vi1_0] = T.float32(0)
                    T_softmax_expsum_rf[i0_4, vi1_0] = T_softmax_expsum_rf[i0_4, vi1_0] + T.exp(A[i0_4, vi1_0 * 16 + vi1_1] - T_softmax_maxelem[i0_4], dtype="float32")
            for i0_5, i1 in T.grid(256, 256):
                for ax0, ax1 in T.grid(16, 1):
                    with T.block("T_softmax_expsum"):
                        vi1_0 = T.axis.reduce(16, ax0)
                        i0_6 = T.axis.spatial(256, i0_5 + ax1)
                        T.reads(T_softmax_expsum_rf[i0_6, vi1_0])
                        T.writes(T_softmax_expsum[i0_6])
                        with T.init():
                            T_softmax_expsum[i0_6] = T.float32(0)
                        T_softmax_expsum[i0_6] = T_softmax_expsum[i0_6] + T_softmax_expsum_rf[i0_6, vi1_0]
                with T.block("T_softmax_norm"):
                    i0_7, i1_2 = T.axis.remap("SS", [i0_5, i1])
                    T.reads(A[i0_7, i1_2], T_softmax_maxelem[i0_7], T_softmax_expsum[i0_7])
                    T.writes(T_softmax_norm[i0_7, i1_2])
                    T.block_attr({"axis":1})
                    T_softmax_norm[i0_7, i1_2] = T.exp(A[i0_7, i1_2] - T_softmax_maxelem[i0_7], dtype="float32") / T_softmax_expsum[i0_7]
    @T.prim_func
    def sfm_1(A: T.Buffer[(256, 256), "float32"], T_softmax_norm: T.Buffer[(256, 256), "float32"]) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # body
        with T.block("root"):
            T.reads()
            T.writes()
            T.block_attr({"meta_schedule.parallel":288, "meta_schedule.unroll_explicit":16, "meta_schedule.vectorize":64})
            T_softmax_maxelem = T.alloc_buffer([256], dtype="float32")
            T_softmax_exp = T.alloc_buffer([256, 256], dtype="float32")
            T_softmax_expsum = T.alloc_buffer([256], dtype="float32")
            T_softmax_expsum_rf = T.alloc_buffer([256, 16], dtype="float32")
            T_softmax_maxelem_rf = T.alloc_buffer([256, 64], dtype="float32")
            for i0 in T.serial(256):
                for ax0, ax1, ax2 in T.grid(64, 1, 4):
                    with T.block("T_softmax_maxelem_rf"):
                        vi1_1 = T.axis.spatial(64, ax0)
                        i0_1 = T.axis.spatial(256, i0 + ax1)
                        vi1_0 = T.axis.reduce(4, ax2)
                        T.reads(A[i0_1, vi1_0 * 64 + vi1_1])
                        T.writes(T_softmax_maxelem_rf[i0_1, vi1_1])
                        with T.init():
                            T_softmax_maxelem_rf[i0_1, vi1_1] = T.float32(-3.4028234663852886e+38)
                        T_softmax_maxelem_rf[i0_1, vi1_1] = T.max(T_softmax_maxelem_rf[i0_1, vi1_1], A[i0_1, vi1_0 * 64 + vi1_1])
                for i1 in T.serial(256):
                    for ax0, ax1 in T.grid(64, 1):
                        with T.block("T_softmax_maxelem"):
                            vi1_1 = T.axis.reduce(64, ax0)
                            i0_2 = T.axis.spatial(256, i0 + ax1)
                            T.reads(T_softmax_maxelem_rf[i0_2, vi1_1])
                            T.writes(T_softmax_maxelem[i0_2])
                            with T.init():
                                T_softmax_maxelem[i0_2] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem[i0_2] = T.max(T_softmax_maxelem[i0_2], T_softmax_maxelem_rf[i0_2, vi1_1])
                    with T.block("T_softmax_exp"):
                        i0_3, i1_1 = T.axis.remap("SS", [i0, i1])
                        T.reads(A[i0_3, i1_1], T_softmax_maxelem[i0_3])
                        T.writes(T_softmax_exp[i0_3, i1_1])
                        T_softmax_exp[i0_3, i1_1] = T.exp(A[i0_3, i1_1] - T_softmax_maxelem[i0_3], dtype="float32")
            for i0_4, i1_0, i1_1_1 in T.grid(256, 16, 16):
                with T.block("T_softmax_expsum_rf"):
                    vi1_0, i0_5, vi1_1 = T.axis.remap("SSR", [i1_0, i0_4, i1_1_1])
                    T.reads(T_softmax_exp[i0_5, vi1_0 * 16 + vi1_1])
                    T.writes(T_softmax_expsum_rf[i0_5, vi1_0])
                    with T.init():
                        T_softmax_expsum_rf[i0_5, vi1_0] = T.float32(0)
                    T_softmax_expsum_rf[i0_5, vi1_0] = T_softmax_expsum_rf[i0_5, vi1_0] + T_softmax_exp[i0_5, vi1_0 * 16 + vi1_1]
            for i0_6, i1_0 in T.grid(256, 16):
                with T.block("T_softmax_expsum"):
                    vi1_0, i0_7 = T.axis.remap("RS", [i1_0, i0_6])
                    T.reads(T_softmax_expsum_rf[i0_7, vi1_0])
                    T.writes(T_softmax_expsum[i0_7])
                    with T.init():
                        T_softmax_expsum[i0_7] = T.float32(0)
                    T_softmax_expsum[i0_7] = T_softmax_expsum[i0_7] + T_softmax_expsum_rf[i0_7, vi1_0]
            for i0_8, i1 in T.grid(256, 256):
                with T.block("T_softmax_norm"):
                    i0_9, i1_2 = T.axis.remap("SS", [i0_8, i1])
                    T.reads(T_softmax_exp[i0_9, i1_2], T_softmax_expsum[i0_9])
                    T.writes(T_softmax_norm[i0_9, i1_2])
                    T.block_attr({"axis":1})
                    T_softmax_norm[i0_9, i1_2] = T_softmax_exp[i0_9, i1_2] / T_softmax_expsum[i0_9]
    @T.prim_func
    def sfm_2(A: T.Buffer[(256, 256), "float32"], T_softmax_norm: T.Buffer[(256, 256), "float32"]) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # body
        with T.block("root"):
            T.reads()
            T.writes()
            T.block_attr({"meta_schedule.parallel":288, "meta_schedule.unroll_explicit":512, "meta_schedule.vectorize":64})
            T_softmax_maxelem = T.alloc_buffer([256], dtype="float32")
            T_softmax_expsum = T.alloc_buffer([256], dtype="float32")
            T_softmax_expsum_rf = T.alloc_buffer([256, 16], dtype="float32")
            for i0, i1 in T.grid(256, 256):
                with T.block("T_softmax_maxelem"):
                    i0_1, k = T.axis.remap("SR", [i0, i1])
                    T.reads(A[i0_1, k])
                    T.writes(T_softmax_maxelem[i0_1])
                    with T.init():
                        T_softmax_maxelem[i0_1] = T.float32(-3.4028234663852886e+38)
                    T_softmax_maxelem[i0_1] = T.max(T_softmax_maxelem[i0_1], A[i0_1, k])
            for i0, i1_0, i1_1 in T.grid(256, 16, 16):
                with T.block("T_softmax_expsum_rf"):
                    vi1_0, i0_2, vi1_1 = T.axis.remap("SSR", [i1_0, i0, i1_1])
                    T.reads(A[i0_2, vi1_0 * 16 + vi1_1], T_softmax_maxelem[i0_2])
                    T.writes(T_softmax_expsum_rf[i0_2, vi1_0])
                    with T.init():
                        T_softmax_expsum_rf[i0_2, vi1_0] = T.float32(0)
                    T_softmax_expsum_rf[i0_2, vi1_0] = T_softmax_expsum_rf[i0_2, vi1_0] + T.exp(A[i0_2, vi1_0 * 16 + vi1_1] - T_softmax_maxelem[i0_2], dtype="float32")
            for i0_3, i1_0 in T.grid(256, 16):
                with T.block("T_softmax_expsum"):
                    vi1_0, i0_4 = T.axis.remap("RS", [i1_0, i0_3])
                    T.reads(T_softmax_expsum_rf[i0_4, vi1_0])
                    T.writes(T_softmax_expsum[i0_4])
                    with T.init():
                        T_softmax_expsum[i0_4] = T.float32(0)
                    T_softmax_expsum[i0_4] = T_softmax_expsum[i0_4] + T_softmax_expsum_rf[i0_4, vi1_0]
            for i0_5, i1 in T.grid(256, 256):
                with T.block("T_softmax_norm"):
                    i0_6, i1_2 = T.axis.remap("SS", [i0_5, i1])
                    T.reads(A[i0_6, i1_2], T_softmax_maxelem[i0_6], T_softmax_expsum[i0_6])
                    T.writes(T_softmax_norm[i0_6, i1_2])
                    T.block_attr({"axis":1})
                    T_softmax_norm[i0_6, i1_2] = T.exp(A[i0_6, i1_2] - T_softmax_maxelem[i0_6], dtype="float32") / T_softmax_expsum[i0_6]
    @T.prim_func
    def sfm_3(A: T.Buffer[(256, 256), "float32"], T_softmax_norm: T.Buffer[(256, 256), "float32"]) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # body
        with T.block("root"):
            T.reads()
            T.writes()
            T.block_attr({"meta_schedule.parallel":288, "meta_schedule.unroll_explicit":512, "meta_schedule.vectorize":64})
            T_softmax_maxelem = T.alloc_buffer([256], dtype="float32")
            T_softmax_exp = T.alloc_buffer([256, 256], dtype="float32")
            T_softmax_expsum = T.alloc_buffer([256], dtype="float32")
            T_softmax_expsum_rf = T.alloc_buffer([256, 16], dtype="float32")
            T_softmax_maxelem_rf = T.alloc_buffer([256, 256], dtype="float32")
            for i0, i1 in T.grid(256, 256):
                for ax0, ax1, ax2 in T.grid(256, 1, 1):
                    with T.block("T_softmax_maxelem_rf"):
                        vi1_0 = T.axis.spatial(256, ax0)
                        i0_1 = T.axis.spatial(256, i0 + ax1)
                        vi1_1 = T.axis.reduce(1, ax2)
                        T.reads(A[i0_1, vi1_1 + vi1_0])
                        T.writes(T_softmax_maxelem_rf[i0_1, vi1_0])
                        with T.init():
                            T_softmax_maxelem_rf[i0_1, vi1_0] = T.float32(-3.4028234663852886e+38)
                        T_softmax_maxelem_rf[i0_1, vi1_0] = T.max(T_softmax_maxelem_rf[i0_1, vi1_0], A[i0_1, vi1_1 + vi1_0])
                for ax0, ax1 in T.grid(256, 1):
                    with T.block("T_softmax_maxelem"):
                        vi1_0 = T.axis.reduce(256, ax0)
                        i0_2 = T.axis.spatial(256, i0 + ax1)
                        T.reads(T_softmax_maxelem_rf[i0_2, vi1_0])
                        T.writes(T_softmax_maxelem[i0_2])
                        with T.init():
                            T_softmax_maxelem[i0_2] = T.float32(-3.4028234663852886e+38)
                        T_softmax_maxelem[i0_2] = T.max(T_softmax_maxelem[i0_2], T_softmax_maxelem_rf[i0_2, vi1_0])
                for ax0, ax1 in T.grid(1, 256):
                    with T.block("T_softmax_exp"):
                        i0_3 = T.axis.spatial(256, i0 + ax0)
                        i1_1 = T.axis.spatial(256, ax1)
                        T.reads(A[i0_3, i1_1], T_softmax_maxelem[i0_3])
                        T.writes(T_softmax_exp[i0_3, i1_1])
                        T_softmax_exp[i0_3, i1_1] = T.exp(A[i0_3, i1_1] - T_softmax_maxelem[i0_3], dtype="float32")
                for ax0 in T.serial(16):
                    for ax0_1, ax1, ax2 in T.grid(1, 1, 16):
                        with T.block("T_softmax_expsum_rf"):
                            vi1_1 = T.axis.spatial(16, ax0 + ax0_1)
                            i0_4 = T.axis.spatial(256, i0 + ax1)
                            vi1_0 = T.axis.reduce(16, ax2)
                            T.reads(T_softmax_exp[i0_4, vi1_0 * 16 + vi1_1])
                            T.writes(T_softmax_expsum_rf[i0_4, vi1_1])
                            with T.init():
                                T_softmax_expsum_rf[i0_4, vi1_1] = T.float32(0)
                            T_softmax_expsum_rf[i0_4, vi1_1] = T_softmax_expsum_rf[i0_4, vi1_1] + T_softmax_exp[i0_4, vi1_0 * 16 + vi1_1]
                    for ax1 in T.serial(1):
                        with T.block("T_softmax_expsum"):
                            vi1_1 = T.axis.reduce(16, ax0)
                            i0_5 = T.axis.spatial(256, i0 + ax1)
                            T.reads(T_softmax_expsum_rf[i0_5, vi1_1])
                            T.writes(T_softmax_expsum[i0_5])
                            with T.init():
                                T_softmax_expsum[i0_5] = T.float32(0)
                            T_softmax_expsum[i0_5] = T_softmax_expsum[i0_5] + T_softmax_expsum_rf[i0_5, vi1_1]
                with T.block("T_softmax_norm"):
                    i0_6, i1_2 = T.axis.remap("SS", [i0, i1])
                    T.reads(T_softmax_exp[i0_6, i1_2], T_softmax_expsum[i0_6])
                    T.writes(T_softmax_norm[i0_6, i1_2])
                    T.block_attr({"axis":1})
                    T_softmax_norm[i0_6, i1_2] = T_softmax_exp[i0_6, i1_2] / T_softmax_expsum[i0_6]
    @T.prim_func
    def sfm_4(A: T.Buffer[(256, 256), "float32"], T_softmax_norm: T.Buffer[(256, 256), "float32"]) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # body
        with T.block("root"):
            T.reads()
            T.writes()
            T.block_attr({"meta_schedule.parallel":288, "meta_schedule.unroll_explicit":0, "meta_schedule.vectorize":64})
            T_softmax_maxelem = T.alloc_buffer([256], dtype="float32")
            T_softmax_exp = T.alloc_buffer([256, 256], dtype="float32")
            T_softmax_expsum = T.alloc_buffer([256], dtype="float32")
            T_softmax_expsum_rf = T.alloc_buffer([256, 16], dtype="float32")
            T_softmax_maxelem_rf = T.alloc_buffer([256, 1], dtype="float32")
            for i0 in T.serial(256):
                for ax0, ax1, ax2 in T.grid(1, 1, 256):
                    with T.block("T_softmax_maxelem_rf"):
                        vi1_1 = T.axis.spatial(1, ax0)
                        i0_1 = T.axis.spatial(256, i0 + ax1)
                        vi1_0 = T.axis.reduce(256, ax2)
                        T.reads(A[i0_1, vi1_1 + vi1_0])
                        T.writes(T_softmax_maxelem_rf[i0_1, vi1_1])
                        with T.init():
                            T_softmax_maxelem_rf[i0_1, vi1_1] = T.float32(-3.4028234663852886e+38)
                        T_softmax_maxelem_rf[i0_1, vi1_1] = T.max(T_softmax_maxelem_rf[i0_1, vi1_1], A[i0_1, vi1_1 + vi1_0])
                for i1_1 in T.serial(1):
                    with T.block("T_softmax_maxelem"):
                        vi1_1, i0_2 = T.axis.remap("RS", [i1_1, i0])
                        T.reads(T_softmax_maxelem_rf[i0_2, vi1_1])
                        T.writes(T_softmax_maxelem[i0_2])
                        with T.init():
                            T_softmax_maxelem[i0_2] = T.float32(-3.4028234663852886e+38)
                        T_softmax_maxelem[i0_2] = T.max(T_softmax_maxelem[i0_2], T_softmax_maxelem_rf[i0_2, vi1_1])
            for i0_3, i1 in T.grid(256, 256):
                with T.block("T_softmax_exp"):
                    i0_4, i1_2 = T.axis.remap("SS", [i0_3, i1])
                    T.reads(A[i0_4, i1_2], T_softmax_maxelem[i0_4])
                    T.writes(T_softmax_exp[i0_4, i1_2])
                    T_softmax_exp[i0_4, i1_2] = T.exp(A[i0_4, i1_2] - T_softmax_maxelem[i0_4], dtype="float32")
            for i0_5, i1_0, i1_1 in T.grid(256, 16, 16):
                with T.block("T_softmax_expsum_rf"):
                    vi1_1, i0_6, vi1_0 = T.axis.remap("SSR", [i1_1, i0_5, i1_0])
                    T.reads(T_softmax_exp[i0_6, vi1_0 * 16 + vi1_1])
                    T.writes(T_softmax_expsum_rf[i0_6, vi1_1])
                    with T.init():
                        T_softmax_expsum_rf[i0_6, vi1_1] = T.float32(0)
                    T_softmax_expsum_rf[i0_6, vi1_1] = T_softmax_expsum_rf[i0_6, vi1_1] + T_softmax_exp[i0_6, vi1_0 * 16 + vi1_1]
            for i0_7, i1_1 in T.grid(256, 16):
                with T.block("T_softmax_expsum"):
                    vi1_1, i0_8 = T.axis.remap("RS", [i1_1, i0_7])
                    T.reads(T_softmax_expsum_rf[i0_8, vi1_1])
                    T.writes(T_softmax_expsum[i0_8])
                    with T.init():
                        T_softmax_expsum[i0_8] = T.float32(0)
                    T_softmax_expsum[i0_8] = T_softmax_expsum[i0_8] + T_softmax_expsum_rf[i0_8, vi1_1]
            for i0_9, i1_3 in T.grid(256, 256):
                with T.block("T_softmax_norm"):
                    i0_10, i1_4 = T.axis.remap("SS", [i0_9, i1_3])
                    T.reads(T_softmax_exp[i0_10, i1_4], T_softmax_expsum[i0_10])
                    T.writes(T_softmax_norm[i0_10, i1_4])
                    T.block_attr({"axis":1})
                    T_softmax_norm[i0_10, i1_4] = T_softmax_exp[i0_10, i1_4] / T_softmax_expsum[i0_10]
    @T.prim_func
    def sfm_5(A: T.Buffer[(256, 256), "float32"], T_softmax_norm: T.Buffer[(256, 256), "float32"]) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # body
        with T.block("root"):
            T.reads()
            T.writes()
            T.block_attr({"meta_schedule.parallel":288, "meta_schedule.unroll_explicit":512, "meta_schedule.vectorize":64})
            T_softmax_maxelem = T.alloc_buffer([256], dtype="float32")
            T_softmax_exp = T.alloc_buffer([256, 256], dtype="float32")
            T_softmax_expsum = T.alloc_buffer([256], dtype="float32")
            T_softmax_expsum_rf = T.alloc_buffer([256, 16], dtype="float32")
            for i0 in T.serial(256):
                for ax0, ax1 in T.grid(1, 256):
                    with T.block("T_softmax_maxelem"):
                        i0_1 = T.axis.spatial(256, i0 + ax0)
                        k = T.axis.reduce(256, ax1)
                        T.reads(A[i0_1, k])
                        T.writes(T_softmax_maxelem[i0_1])
                        with T.init():
                            T_softmax_maxelem[i0_1] = T.float32(-3.4028234663852886e+38)
                        T_softmax_maxelem[i0_1] = T.max(T_softmax_maxelem[i0_1], A[i0_1, k])
                for ax0, ax1 in T.grid(1, 256):
                    with T.block("T_softmax_exp"):
                        i0_2 = T.axis.spatial(256, i0 + ax0)
                        i1 = T.axis.spatial(256, ax1)
                        T.reads(A[i0_2, i1], T_softmax_maxelem[i0_2])
                        T.writes(T_softmax_exp[i0_2, i1])
                        T_softmax_exp[i0_2, i1] = T.exp(A[i0_2, i1] - T_softmax_maxelem[i0_2], dtype="float32")
                for ax0 in T.serial(16):
                    for ax0_1, ax1, ax2 in T.grid(1, 1, 16):
                        with T.block("T_softmax_expsum_rf"):
                            vi1_1 = T.axis.spatial(16, ax0 + ax0_1)
                            i0_3 = T.axis.spatial(256, i0 + ax1)
                            vi1_0 = T.axis.reduce(16, ax2)
                            T.reads(T_softmax_exp[i0_3, vi1_0 * 16 + vi1_1])
                            T.writes(T_softmax_expsum_rf[i0_3, vi1_1])
                            with T.init():
                                T_softmax_expsum_rf[i0_3, vi1_1] = T.float32(0)
                            T_softmax_expsum_rf[i0_3, vi1_1] = T_softmax_expsum_rf[i0_3, vi1_1] + T_softmax_exp[i0_3, vi1_0 * 16 + vi1_1]
                    for ax1 in T.serial(1):
                        with T.block("T_softmax_expsum"):
                            vi1_1 = T.axis.reduce(16, ax0)
                            i0_4 = T.axis.spatial(256, i0 + ax1)
                            T.reads(T_softmax_expsum_rf[i0_4, vi1_1])
                            T.writes(T_softmax_expsum[i0_4])
                            with T.init():
                                T_softmax_expsum[i0_4] = T.float32(0)
                            T_softmax_expsum[i0_4] = T_softmax_expsum[i0_4] + T_softmax_expsum_rf[i0_4, vi1_1]
                for i1 in T.serial(256):
                    with T.block("T_softmax_norm"):
                        i0_5, i1_1 = T.axis.remap("SS", [i0, i1])
                        T.reads(T_softmax_exp[i0_5, i1_1], T_softmax_expsum[i0_5])
                        T.writes(T_softmax_norm[i0_5, i1_1])
                        T.block_attr({"axis":1})
                        T_softmax_norm[i0_5, i1_1] = T_softmax_exp[i0_5, i1_1] / T_softmax_expsum[i0_5]
    @T.prim_func
    def sfm_6(A: T.Buffer[(256, 256), "float32"], T_softmax_norm: T.Buffer[(256, 256), "float32"]) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # body
        with T.block("root"):
            T.reads()
            T.writes()
            T.block_attr({"meta_schedule.parallel":288, "meta_schedule.unroll_explicit":64, "meta_schedule.vectorize":64})
            T_softmax_maxelem = T.alloc_buffer([256], dtype="float32")
            T_softmax_expsum = T.alloc_buffer([256], dtype="float32")
            T_softmax_maxelem_rf = T.alloc_buffer([256, 64], dtype="float32")
            for i0 in T.serial(256):
                for ax0, ax1, ax2 in T.grid(64, 1, 4):
                    with T.block("T_softmax_maxelem_rf"):
                        vi1_0 = T.axis.spatial(64, ax0)
                        i0_1 = T.axis.spatial(256, i0 + ax1)
                        vi1_1 = T.axis.reduce(4, ax2)
                        T.reads(A[i0_1, vi1_0 * 4 + vi1_1])
                        T.writes(T_softmax_maxelem_rf[i0_1, vi1_0])
                        with T.init():
                            T_softmax_maxelem_rf[i0_1, vi1_0] = T.float32(-3.4028234663852886e+38)
                        T_softmax_maxelem_rf[i0_1, vi1_0] = T.max(T_softmax_maxelem_rf[i0_1, vi1_0], A[i0_1, vi1_0 * 4 + vi1_1])
                for i1_0 in T.serial(64):
                    with T.block("T_softmax_maxelem"):
                        vi1_0, i0_2 = T.axis.remap("RS", [i1_0, i0])
                        T.reads(T_softmax_maxelem_rf[i0_2, vi1_0])
                        T.writes(T_softmax_maxelem[i0_2])
                        with T.init():
                            T_softmax_maxelem[i0_2] = T.float32(-3.4028234663852886e+38)
                        T_softmax_maxelem[i0_2] = T.max(T_softmax_maxelem[i0_2], T_softmax_maxelem_rf[i0_2, vi1_0])
            for i0_3, i1 in T.grid(256, 256):
                with T.block("T_softmax_expsum"):
                    i0_4, k = T.axis.remap("SR", [i0_3, i1])
                    T.reads(A[i0_4, k], T_softmax_maxelem[i0_4])
                    T.writes(T_softmax_expsum[i0_4])
                    with T.init():
                        T_softmax_expsum[i0_4] = T.float32(0)
                    T_softmax_expsum[i0_4] = T_softmax_expsum[i0_4] + T.exp(A[i0_4, k] - T_softmax_maxelem[i0_4], dtype="float32")
            for i0_5, i1 in T.grid(256, 256):
                with T.block("T_softmax_norm"):
                    i0_6, i1_1 = T.axis.remap("SS", [i0_5, i1])
                    T.reads(A[i0_6, i1_1], T_softmax_maxelem[i0_6], T_softmax_expsum[i0_6])
                    T.writes(T_softmax_norm[i0_6, i1_1])
                    T.block_attr({"axis":1})
                    T_softmax_norm[i0_6, i1_1] = T.exp(A[i0_6, i1_1] - T_softmax_maxelem[i0_6], dtype="float32") / T_softmax_expsum[i0_6]
    @T.prim_func
    def sfm_7(A: T.Buffer[(256, 256), "float32"], T_softmax_norm: T.Buffer[(256, 256), "float32"]) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # body
        with T.block("root"):
            T.reads()
            T.writes()
            T.block_attr({"meta_schedule.parallel":288, "meta_schedule.unroll_explicit":64, "meta_schedule.vectorize":64})
            T_softmax_maxelem = T.alloc_buffer([256], dtype="float32")
            T_softmax_expsum = T.alloc_buffer([256], dtype="float32")
            T_softmax_maxelem_rf = T.alloc_buffer([256, 4], dtype="float32")
            for i0, i1_0, i1_1 in T.grid(256, 64, 4):
                with T.block("T_softmax_maxelem_rf"):
                    vi1_1, i0_1, vi1_0 = T.axis.remap("SSR", [i1_1, i0, i1_0])
                    T.reads(A[i0_1, vi1_0 * 4 + vi1_1])
                    T.writes(T_softmax_maxelem_rf[i0_1, vi1_1])
                    with T.init():
                        T_softmax_maxelem_rf[i0_1, vi1_1] = T.float32(-3.4028234663852886e+38)
                    T_softmax_maxelem_rf[i0_1, vi1_1] = T.max(T_softmax_maxelem_rf[i0_1, vi1_1], A[i0_1, vi1_0 * 4 + vi1_1])
            for i0, i1_1 in T.grid(256, 4):
                with T.block("T_softmax_maxelem"):
                    vi1_1, i0_2 = T.axis.remap("RS", [i1_1, i0])
                    T.reads(T_softmax_maxelem_rf[i0_2, vi1_1])
                    T.writes(T_softmax_maxelem[i0_2])
                    with T.init():
                        T_softmax_maxelem[i0_2] = T.float32(-3.4028234663852886e+38)
                    T_softmax_maxelem[i0_2] = T.max(T_softmax_maxelem[i0_2], T_softmax_maxelem_rf[i0_2, vi1_1])
            for i0_3, i1 in T.grid(256, 256):
                for ax0, ax1 in T.grid(1, 256):
                    with T.block("T_softmax_expsum"):
                        i0_4 = T.axis.spatial(256, i0_3 + ax0)
                        k = T.axis.reduce(256, ax1)
                        T.reads(A[i0_4, k], T_softmax_maxelem[i0_4])
                        T.writes(T_softmax_expsum[i0_4])
                        with T.init():
                            T_softmax_expsum[i0_4] = T.float32(0)
                        T_softmax_expsum[i0_4] = T_softmax_expsum[i0_4] + T.exp(A[i0_4, k] - T_softmax_maxelem[i0_4], dtype="float32")
                with T.block("T_softmax_norm"):
                    i0_5, i1_2 = T.axis.remap("SS", [i0_3, i1])
                    T.reads(A[i0_5, i1_2], T_softmax_maxelem[i0_5], T_softmax_expsum[i0_5])
                    T.writes(T_softmax_norm[i0_5, i1_2])
                    T.block_attr({"axis":1})
                    T_softmax_norm[i0_5, i1_2] = T.exp(A[i0_5, i1_2] - T_softmax_maxelem[i0_5], dtype="float32") / T_softmax_expsum[i0_5]
    @T.prim_func
    def sfm_8(A: T.Buffer[(256, 256), "float32"], T_softmax_norm: T.Buffer[(256, 256), "float32"]) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # body
        with T.block("root"):
            T.reads()
            T.writes()
            T.block_attr({"meta_schedule.parallel":288, "meta_schedule.unroll_explicit":512, "meta_schedule.vectorize":64})
            T_softmax_maxelem = T.alloc_buffer([256], dtype="float32")
            T_softmax_exp = T.alloc_buffer([256, 256], dtype="float32")
            T_softmax_expsum = T.alloc_buffer([256], dtype="float32")
            for i0 in T.serial(256):
                for ax0, ax1 in T.grid(1, 256):
                    with T.block("T_softmax_maxelem"):
                        i0_1 = T.axis.spatial(256, i0 + ax0)
                        k = T.axis.reduce(256, ax1)
                        T.reads(A[i0_1, k])
                        T.writes(T_softmax_maxelem[i0_1])
                        with T.init():
                            T_softmax_maxelem[i0_1] = T.float32(-3.4028234663852886e+38)
                        T_softmax_maxelem[i0_1] = T.max(T_softmax_maxelem[i0_1], A[i0_1, k])
                for i1 in T.serial(256):
                    with T.block("T_softmax_exp"):
                        i0_2, i1_1 = T.axis.remap("SS", [i0, i1])
                        T.reads(A[i0_2, i1_1], T_softmax_maxelem[i0_2])
                        T.writes(T_softmax_exp[i0_2, i1_1])
                        T_softmax_exp[i0_2, i1_1] = T.exp(A[i0_2, i1_1] - T_softmax_maxelem[i0_2], dtype="float32")
            for i0_3, i1 in T.grid(256, 256):
                with T.block("T_softmax_expsum"):
                    i0_4, k = T.axis.remap("SR", [i0_3, i1])
                    T.reads(T_softmax_exp[i0_4, k])
                    T.writes(T_softmax_expsum[i0_4])
                    with T.init():
                        T_softmax_expsum[i0_4] = T.float32(0)
                    T_softmax_expsum[i0_4] = T_softmax_expsum[i0_4] + T_softmax_exp[i0_4, k]
            for i0_5, i1 in T.grid(256, 256):
                with T.block("T_softmax_norm"):
                    i0_6, i1_2 = T.axis.remap("SS", [i0_5, i1])
                    T.reads(T_softmax_exp[i0_6, i1_2], T_softmax_expsum[i0_6])
                    T.writes(T_softmax_norm[i0_6, i1_2])
                    T.block_attr({"axis":1})
                    T_softmax_norm[i0_6, i1_2] = T_softmax_exp[i0_6, i1_2] / T_softmax_expsum[i0_6]
    # fmt: on
    decision_0 = [
        ("SamplePerfectTile", [16, 16]),
        ("SamplePerfectTile", [4, 64]),
        ("SampleCategorical", 0),
        ("SampleComputeLocation", 1),
        ("SampleComputeLocation", -1),
        ("SampleComputeLocation", -2),
        ("SampleComputeLocation", -1),
        ("SampleComputeLocation", -1),
    ]
    decision_1 = [
        ("SamplePerfectTile", [16, 16]),
        ("SamplePerfectTile", [4, 64]),
        ("SampleCategorical", 1),
        ("SampleComputeLocation", -1),
        ("SampleComputeLocation", -1),
        ("SampleComputeLocation", -1),
        ("SampleComputeLocation", 1),
        ("SampleComputeLocation", 0),
    ]
    decision_2 = [
        ("SamplePerfectTile", [16, 16]),
        ("SampleCategorical", 3),
        ("SampleComputeLocation", -1),
        ("SampleComputeLocation", -1),
        ("SampleComputeLocation", -2),
        ("SampleComputeLocation", -1),
    ]
    decision_3 = [
        ("SamplePerfectTile", [16, 16]),
        ("SamplePerfectTile", [256, 1]),
        ("SampleCategorical", 3),
        ("SampleComputeLocation", 1),
        ("SampleComputeLocation", 2),
        ("SampleComputeLocation", 1),
        ("SampleComputeLocation", 1),
        ("SampleComputeLocation", 1),
    ]
    decision_4 = [
        ("SamplePerfectTile", [16, 16]),
        ("SamplePerfectTile", [256, 1]),
        ("SampleCategorical", 0),
        ("SampleComputeLocation", -1),
        ("SampleComputeLocation", -1),
        ("SampleComputeLocation", -1),
        ("SampleComputeLocation", -1),
        ("SampleComputeLocation", 0),
    ]
    decision_5 = [
        ("SamplePerfectTile", [16, 16]),
        ("SampleCategorical", 3),
        ("SampleComputeLocation", 0),
        ("SampleComputeLocation", 1),
        ("SampleComputeLocation", 0),
        ("SampleComputeLocation", 0),
    ]
    decision_6 = [
        ("SamplePerfectTile", [64, 4]),
        ("SampleCategorical", 2),
        ("SampleComputeLocation", -1),
        ("SampleComputeLocation", -2),
        ("SampleComputeLocation", -1),
        ("SampleComputeLocation", 0),
    ]
    decision_7 = [
        ("SamplePerfectTile", [64, 4]),
        ("SampleCategorical", 2),
        ("SampleComputeLocation", 1),
        ("SampleComputeLocation", -2),
        ("SampleComputeLocation", -1),
        ("SampleComputeLocation", -1),
    ]
    decision_8 = [
        ("SampleCategorical", 3),
        ("SampleComputeLocation", -1),
        ("SampleComputeLocation", -1),
        ("SampleComputeLocation", 0),
    ]
    mod = create_te_workload("SFM", 0)
    actual = _design_space(mod)
    check_sketches(
        mod,
        sketches=actual,
        expected_mods=[sfm_0, sfm_1, sfm_2, sfm_3, sfm_4, sfm_5, sfm_6, sfm_7, sfm_8],
        expected_decisions=[
            decision_0,
            decision_1,
            decision_2,
            decision_3,
            decision_4,
            decision_5,
            decision_6,
            decision_7,
            decision_8,
        ],
    )


def test_cpu_cbr():
    # fmt: off
    @T.prim_func
    def cbr_0(data: T.Buffer[(1, 224, 224, 3), "float32"], kernel: T.Buffer[(7, 7, 3, 64), "float32"], bias: T.Buffer[64, "float32"], bn_offset: T.Buffer[64, "float32"], bn_scale: T.Buffer[64, "float32"], compute: T.Buffer[(1, 112, 112, 64), "float32"]) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # body
        with T.block("root"):
            T.reads()
            T.writes()
            T.block_attr({"meta_schedule.parallel":288, "meta_schedule.unroll_explicit":64, "meta_schedule.vectorize":64})
            Conv2dOutput = T.alloc_buffer([1, 112, 112, 64], dtype="float32")
            for i0_0, i1_0, i2_0, i3_0, i0_1, i1_1, i2_1, i3_1, i4_0, i5_0, i6_0, i0_2, i1_2, i2_2, i3_2, i4_1, i5_1, i6_1, i0_3, i1_3, i2_3, i3_3 in T.grid(1, 2, 7, 1, 1, 2, 2, 32, 7, 7, 1, 1, 1, 4, 1, 1, 1, 3, 1, 28, 2, 2):
                with T.block("Conv2dOutput"):
                    nn = T.axis.spatial(1, i0_3 + i0_0 + i0_1 + i0_2)
                    yy = T.axis.spatial(112, i1_0 * 56 + i1_1 * 28 + i1_2 * 28 + i1_3)
                    xx = T.axis.spatial(112, i2_0 * 16 + i2_1 * 8 + i2_2 * 2 + i2_3)
                    ff = T.axis.spatial(64, i3_0 * 64 + i3_1 * 2 + i3_2 * 2 + i3_3)
                    ry = T.axis.reduce(7, i4_1 + i4_0)
                    rx = T.axis.reduce(7, i5_0 + i5_1)
                    rc = T.axis.reduce(3, i6_0 * 3 + i6_1)
                    T.reads(data[nn, yy * 2 + ry - 3, xx * 2 + rx - 3, rc], kernel[ry, rx, rc, ff])
                    T.writes(Conv2dOutput[nn, yy, xx, ff])
                    T.block_attr({"meta_schedule.tiling_structure":"SSRSRS"})
                    with T.init():
                        Conv2dOutput[nn, yy, xx, ff] = T.float32(0)
                    Conv2dOutput[nn, yy, xx, ff] = Conv2dOutput[nn, yy, xx, ff] + T.if_then_else(3 <= yy * 2 + ry and yy * 2 + ry < 227 and 3 <= xx * 2 + rx and xx * 2 + rx < 227, data[nn, yy * 2 + ry - 3, xx * 2 + rx - 3, rc], T.float32(0), dtype="float32") * kernel[ry, rx, rc, ff]
            for i0, i1, i2, i3 in T.grid(1, 112, 112, 64):
                with T.block("compute"):
                    i0_4, i1_4, i2_4, i3_4 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                    T.reads(Conv2dOutput[i0_4, i1_4, i2_4, i3_4], bias[i3_4], bn_scale[i3_4], bn_offset[i3_4])
                    T.writes(compute[i0_4, i1_4, i2_4, i3_4])
                    compute[i0_4, i1_4, i2_4, i3_4] = T.max((Conv2dOutput[i0_4, i1_4, i2_4, i3_4] + bias[i3_4]) * bn_scale[i3_4] + bn_offset[i3_4], T.float32(0))
    @T.prim_func
    def cbr_1(data: T.Buffer[(1, 224, 224, 3), "float32"], kernel: T.Buffer[(7, 7, 3, 64), "float32"], bias: T.Buffer[64, "float32"], bn_offset: T.Buffer[64, "float32"], bn_scale: T.Buffer[64, "float32"], compute: T.Buffer[(1, 112, 112, 64), "float32"]) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # body
        with T.block("root"):
            T.reads()
            T.writes()
            T.block_attr({"meta_schedule.parallel":288, "meta_schedule.unroll_explicit":512, "meta_schedule.vectorize":64})
            PaddedInput = T.alloc_buffer([1, 230, 230, 3], dtype="float32")
            Conv2dOutput = T.alloc_buffer([1, 112, 112, 64], dtype="float32")
            for i0_0, i1_0 in T.grid(1, 2):
                for ax0, ax1, ax2, ax3 in T.grid(1, 117, 229, 3):
                    with T.block("PaddedInput"):
                        i0 = T.axis.spatial(1, ax0)
                        i1 = T.axis.spatial(230, i1_0 * 112 + ax1)
                        i2 = T.axis.spatial(230, ax2)
                        i3 = T.axis.spatial(3, ax3)
                        T.reads(data[i0, i1 - 3, i2 - 3, i3])
                        T.writes(PaddedInput[i0, i1, i2, i3])
                        PaddedInput[i0, i1, i2, i3] = T.if_then_else(3 <= i1 and i1 < 227 and 3 <= i2 and i2 < 227, data[i0, i1 - 3, i2 - 3, i3], T.float32(0), dtype="float32")
                for i2_0, i3_0, i0_1, i1_1, i2_1, i3_1 in T.grid(7, 1, 1, 2, 2, 32):
                    for i4_0, i5_0, i6_0, i0_2, i1_2, i2_2, i3_2, i4_1, i5_1, i6_1, i0_3, i1_3, i2_3, i3_3 in T.grid(7, 7, 1, 1, 1, 4, 1, 1, 1, 3, 1, 28, 2, 2):
                        with T.block("Conv2dOutput"):
                            nn = T.axis.spatial(1, i0_3 + i0_0 + i0_1 + i0_2)
                            yy = T.axis.spatial(112, i1_0 * 56 + i1_1 * 28 + i1_2 * 28 + i1_3)
                            xx = T.axis.spatial(112, i2_0 * 16 + i2_1 * 8 + i2_2 * 2 + i2_3)
                            ff = T.axis.spatial(64, i3_0 * 64 + i3_1 * 2 + i3_2 * 2 + i3_3)
                            ry = T.axis.reduce(7, i4_1 + i4_0)
                            rx = T.axis.reduce(7, i5_0 + i5_1)
                            rc = T.axis.reduce(3, i6_0 * 3 + i6_1)
                            T.reads(PaddedInput[nn, yy * 2 + ry, xx * 2 + rx, rc], kernel[ry, rx, rc, ff])
                            T.writes(Conv2dOutput[nn, yy, xx, ff])
                            T.block_attr({"meta_schedule.tiling_structure":"SSRSRS"})
                            with T.init():
                                Conv2dOutput[nn, yy, xx, ff] = T.float32(0)
                            Conv2dOutput[nn, yy, xx, ff] = Conv2dOutput[nn, yy, xx, ff] + PaddedInput[nn, yy * 2 + ry, xx * 2 + rx, rc] * kernel[ry, rx, rc, ff]
                    for ax0, ax1, ax2, ax3 in T.grid(1, 28, 8, 2):
                        with T.block("compute"):
                            i0 = T.axis.spatial(1, ax0)
                            i1 = T.axis.spatial(112, i1_0 * 56 + i1_1 * 28 + ax1)
                            i2 = T.axis.spatial(112, i2_0 * 16 + i2_1 * 8 + ax2)
                            i3 = T.axis.spatial(64, i3_1 * 2 + ax3)
                            T.reads(Conv2dOutput[i0, i1, i2, i3], bias[i3], bn_scale[i3], bn_offset[i3])
                            T.writes(compute[i0, i1, i2, i3])
                            compute[i0, i1, i2, i3] = T.max((Conv2dOutput[i0, i1, i2, i3] + bias[i3]) * bn_scale[i3] + bn_offset[i3], T.float32(0))
    @T.prim_func
    def cbr_2(data: T.Buffer[(1, 224, 224, 3), "float32"], kernel: T.Buffer[(7, 7, 3, 64), "float32"], bias: T.Buffer[64, "float32"], bn_offset: T.Buffer[64, "float32"], bn_scale: T.Buffer[64, "float32"], compute: T.Buffer[(1, 112, 112, 64), "float32"]) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # body
        with T.block("root"):
            T.reads()
            T.writes()
            T.block_attr({"meta_schedule.parallel":288, "meta_schedule.unroll_explicit":64, "meta_schedule.vectorize":64})
            PaddedInput = T.alloc_buffer([1, 230, 230, 3], dtype="float32")
            Conv2dOutput = T.alloc_buffer([1, 112, 112, 64], dtype="float32")
            for i0_0, i1_0 in T.grid(1, 2):
                for ax0, ax1, ax2, ax3 in T.grid(1, 117, 229, 3):
                    with T.block("PaddedInput"):
                        i0 = T.axis.spatial(1, ax0)
                        i1 = T.axis.spatial(230, i1_0 * 112 + ax1)
                        i2 = T.axis.spatial(230, ax2)
                        i3 = T.axis.spatial(3, ax3)
                        T.reads(data[i0, i1 - 3, i2 - 3, i3])
                        T.writes(PaddedInput[i0, i1, i2, i3])
                        PaddedInput[i0, i1, i2, i3] = T.if_then_else(3 <= i1 and i1 < 227 and 3 <= i2 and i2 < 227, data[i0, i1 - 3, i2 - 3, i3], T.float32(0), dtype="float32")
                for i2_0, i3_0 in T.grid(7, 1):
                    for i0_1, i1_1, i2_1, i3_1, i4_0, i5_0, i6_0, i0_2, i1_2, i2_2, i3_2, i4_1, i5_1, i6_1, i0_3, i1_3, i2_3, i3_3 in T.grid(1, 2, 2, 32, 7, 7, 1, 1, 1, 4, 1, 1, 1, 3, 1, 28, 2, 2):
                        with T.block("Conv2dOutput"):
                            nn = T.axis.spatial(1, i0_3 + i0_0 + i0_1 + i0_2)
                            yy = T.axis.spatial(112, i1_0 * 56 + i1_1 * 28 + i1_2 * 28 + i1_3)
                            xx = T.axis.spatial(112, i2_0 * 16 + i2_1 * 8 + i2_2 * 2 + i2_3)
                            ff = T.axis.spatial(64, i3_0 * 64 + i3_1 * 2 + i3_2 * 2 + i3_3)
                            ry = T.axis.reduce(7, i4_1 + i4_0)
                            rx = T.axis.reduce(7, i5_0 + i5_1)
                            rc = T.axis.reduce(3, i6_0 * 3 + i6_1)
                            T.reads(PaddedInput[nn, yy * 2 + ry, xx * 2 + rx, rc], kernel[ry, rx, rc, ff])
                            T.writes(Conv2dOutput[nn, yy, xx, ff])
                            T.block_attr({"meta_schedule.tiling_structure":"SSRSRS"})
                            with T.init():
                                Conv2dOutput[nn, yy, xx, ff] = T.float32(0)
                            Conv2dOutput[nn, yy, xx, ff] = Conv2dOutput[nn, yy, xx, ff] + PaddedInput[nn, yy * 2 + ry, xx * 2 + rx, rc] * kernel[ry, rx, rc, ff]
                    for ax0, ax1, ax2, ax3 in T.grid(1, 56, 16, 64):
                        with T.block("compute"):
                            i0 = T.axis.spatial(1, ax0)
                            i1 = T.axis.spatial(112, i1_0 * 56 + ax1)
                            i2 = T.axis.spatial(112, i2_0 * 16 + ax2)
                            i3 = T.axis.spatial(64, ax3)
                            T.reads(Conv2dOutput[i0, i1, i2, i3], bias[i3], bn_scale[i3], bn_offset[i3])
                            T.writes(compute[i0, i1, i2, i3])
                            compute[i0, i1, i2, i3] = T.max((Conv2dOutput[i0, i1, i2, i3] + bias[i3]) * bn_scale[i3] + bn_offset[i3], T.float32(0))
    # fmt: on
    decision_0 = [
        ("SamplePerfectTile", [1, 1, 1, 1]),
        ("SamplePerfectTile", [2, 2, 1, 28]),
        ("SamplePerfectTile", [7, 2, 4, 2]),
        ("SamplePerfectTile", [1, 32, 1, 2]),
        ("SamplePerfectTile", [7, 1]),
        ("SamplePerfectTile", [7, 1]),
        ("SamplePerfectTile", [1, 3]),
        ("SampleCategorical", 2),
        ("SampleComputeLocation", -2),
    ]
    decision_1 = [
        ("SamplePerfectTile", [1, 1, 1, 1]),
        ("SamplePerfectTile", [2, 2, 1, 28]),
        ("SamplePerfectTile", [7, 2, 4, 2]),
        ("SamplePerfectTile", [1, 32, 1, 2]),
        ("SamplePerfectTile", [7, 1]),
        ("SamplePerfectTile", [7, 1]),
        ("SamplePerfectTile", [1, 3]),
        ("SampleCategorical", 3),
        ("SampleComputeLocation", 1),
    ]
    decision_2 = [
        ("SamplePerfectTile", [1, 1, 1, 1]),
        ("SamplePerfectTile", [2, 2, 1, 28]),
        ("SamplePerfectTile", [7, 2, 4, 2]),
        ("SamplePerfectTile", [1, 32, 1, 2]),
        ("SamplePerfectTile", [7, 1]),
        ("SamplePerfectTile", [7, 1]),
        ("SamplePerfectTile", [1, 3]),
        ("SampleCategorical", 2),
        ("SampleComputeLocation", 1),
    ]
    mod = create_te_workload("CBR", 0)
    actual = _design_space(mod)
    check_sketches(
        mod,
        sketches=actual,
        expected_mods=[cbr_0, cbr_1, cbr_2],
        expected_decisions=[decision_0, decision_1, decision_2],
    )


def test_cpu_tbg():
    # fmt: off
    @T.prim_func
    def tbg_0(query: T.Buffer[(1, 128, 12, 64), "float32"], value: T.Buffer[(1, 128, 12, 64), "float32"], C: T.Buffer[(1, 12, 128, 128), "float32"]) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # body
        with T.block("root"):
            T.reads()
            T.writes()
            T.block_attr({"meta_schedule.parallel":288, "meta_schedule.unroll_explicit":64, "meta_schedule.vectorize":64})
            query_T = T.alloc_buffer([1, 12, 128, 64], dtype="float32")
            value_T = T.alloc_buffer([1, 12, 64, 128], dtype="float32")
            C_global = T.alloc_buffer([1, 12, 128, 128], dtype="float32")
            for i0_0, i1_0, i2_0, i3_0, i0_1, i1_1, i2_1 in T.grid(1, 1, 1, 2, 1, 6, 2):
                for ax0, ax1, ax2, ax3 in T.grid(1, 2, 64, 64):
                    with T.block("value_T"):
                        b = T.axis.spatial(1, ax0)
                        h = T.axis.spatial(12, i1_1 * 2 + ax1)
                        d = T.axis.spatial(64, ax2)
                        l = T.axis.spatial(128, i3_0 * 64 + ax3)
                        T.reads(value[b, l, h, d])
                        T.writes(value_T[b, h, d, l])
                        value_T[b, h, d, l] = value[b, l, h, d]
                for ax0, ax1, ax2, ax3 in T.grid(1, 2, 64, 64):
                    with T.block("query_T"):
                        b = T.axis.spatial(1, ax0)
                        h = T.axis.spatial(12, i1_1 * 2 + ax1)
                        l = T.axis.spatial(128, i2_1 * 64 + ax2)
                        d = T.axis.spatial(64, ax3)
                        T.reads(query[b, l, h, d])
                        T.writes(query_T[b, h, l, d])
                        query_T[b, h, l, d] = query[b, l, h, d]
                for i3_1 in T.serial(8):
                    for i4_0, i0_2, i1_2, i2_2, i3_2, i4_1, i0_3, i1_3, i2_3, i3_3 in T.grid(1, 1, 2, 2, 4, 64, 1, 1, 32, 2):
                        with T.block("C"):
                            b = T.axis.spatial(1, i0_1 + i0_2 + i0_3 + i0_0)
                            h = T.axis.spatial(12, i1_0 * 12 + i1_1 * 2 + i1_2 + i1_3)
                            i = T.axis.spatial(128, i2_0 * 128 + i2_1 * 64 + i2_2 * 32 + i2_3)
                            j = T.axis.spatial(128, i3_0 * 64 + i3_1 * 8 + i3_2 * 2 + i3_3)
                            k = T.axis.reduce(64, i4_0 * 64 + i4_1)
                            T.reads(query_T[b, h, i, k], value_T[b, h, k, j])
                            T.writes(C_global[b, h, i, j])
                            T.block_attr({"meta_schedule.tiling_structure":"SSRSRS"})
                            with T.init():
                                C_global[b, h, i, j] = T.float32(0)
                            C_global[b, h, i, j] = C_global[b, h, i, j] + query_T[b, h, i, k] * value_T[b, h, k, j]
                    for ax0, ax1, ax2, ax3 in T.grid(1, 2, 64, 8):
                        with T.block("C_global"):
                            v0 = T.axis.spatial(1, ax0)
                            v1 = T.axis.spatial(12, i1_1 * 2 + ax1)
                            v2 = T.axis.spatial(128, i2_1 * 64 + ax2)
                            v3 = T.axis.spatial(128, i3_0 * 64 + i3_1 * 8 + ax3)
                            T.reads(C_global[v0, v1, v2, v3])
                            T.writes(C[v0, v1, v2, v3])
                            C[v0, v1, v2, v3] = C_global[v0, v1, v2, v3]
    @T.prim_func
    def tbg_1(query: T.Buffer[(1, 128, 12, 64), "float32"], value: T.Buffer[(1, 128, 12, 64), "float32"], C: T.Buffer[(1, 12, 128, 128), "float32"]) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # body
        with T.block("root"):
            T.reads()
            T.writes()
            T.block_attr({"meta_schedule.parallel":288, "meta_schedule.unroll_explicit":64, "meta_schedule.vectorize":64})
            query_T = T.alloc_buffer([1, 12, 128, 64], dtype="float32")
            value_T = T.alloc_buffer([1, 12, 64, 128], dtype="float32")
            C_global = T.alloc_buffer([1, 12, 128, 128], dtype="float32")
            for i0, i1, i2, i3 in T.grid(1, 12, 128, 64):
                with T.block("query_T"):
                    b, h, l, d = T.axis.remap("SSSS", [i0, i1, i2, i3])
                    T.reads(query[b, l, h, d])
                    T.writes(query_T[b, h, l, d])
                    query_T[b, h, l, d] = query[b, l, h, d]
            for i0_0, i1_0, i2_0, i3_0 in T.grid(1, 1, 1, 2):
                for i0_1, i1_1, i2_1, i3_1, i4_0, i0_2, i1_2, i2_2, i3_2, i4_1 in T.grid(1, 6, 2, 8, 1, 1, 2, 2, 4, 64):
                    for ax0, ax1, ax2, ax3 in T.grid(1, 1, 1, 2):
                        with T.block("value_T"):
                            b = T.axis.spatial(1, ax0)
                            h = T.axis.spatial(12, i1_1 * 2 + i1_2 + ax1)
                            d = T.axis.spatial(64, i4_1 + ax2)
                            l = T.axis.spatial(128, i3_0 * 64 + i3_1 * 8 + i3_2 * 2 + ax3)
                            T.reads(value[b, l, h, d])
                            T.writes(value_T[b, h, d, l])
                            value_T[b, h, d, l] = value[b, l, h, d]
                    for i0_3, i1_3, i2_3, i3_3 in T.grid(1, 1, 32, 2):
                        with T.block("C"):
                            b = T.axis.spatial(1, i0_1 + i0_2 + i0_3 + i0_0)
                            h = T.axis.spatial(12, i1_0 * 12 + i1_1 * 2 + i1_2 + i1_3)
                            i = T.axis.spatial(128, i2_0 * 128 + i2_1 * 64 + i2_2 * 32 + i2_3)
                            j = T.axis.spatial(128, i3_0 * 64 + i3_1 * 8 + i3_2 * 2 + i3_3)
                            k = T.axis.reduce(64, i4_0 * 64 + i4_1)
                            T.reads(query_T[b, h, i, k], value_T[b, h, k, j])
                            T.writes(C_global[b, h, i, j])
                            T.block_attr({"meta_schedule.tiling_structure":"SSRSRS"})
                            with T.init():
                                C_global[b, h, i, j] = T.float32(0)
                            C_global[b, h, i, j] = C_global[b, h, i, j] + query_T[b, h, i, k] * value_T[b, h, k, j]
                for ax0, ax1, ax2, ax3 in T.grid(1, 12, 128, 64):
                    with T.block("C_global"):
                        v0, v1, v2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                        v3 = T.axis.spatial(128, i3_0 * 64 + ax3)
                        T.reads(C_global[v0, v1, v2, v3])
                        T.writes(C[v0, v1, v2, v3])
                        C[v0, v1, v2, v3] = C_global[v0, v1, v2, v3]
    @T.prim_func
    def tbg_2(query: T.Buffer[(1, 128, 12, 64), "float32"], value: T.Buffer[(1, 128, 12, 64), "float32"], C: T.Buffer[(1, 12, 128, 128), "float32"]) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # body
        with T.block("root"):
            T.reads()
            T.writes()
            T.block_attr({"meta_schedule.parallel":288, "meta_schedule.unroll_explicit":512, "meta_schedule.vectorize":64})
            value_T = T.alloc_buffer([1, 12, 64, 128], dtype="float32")
            for i0_0, i1_0, i2_0, i3_0, i0_1, i1_1, i2_1, i3_1 in T.grid(1, 1, 1, 2, 1, 6, 2, 8):
                for ax0, ax1, ax2, ax3 in T.grid(1, 2, 64, 8):
                    with T.block("value_T"):
                        b = T.axis.spatial(1, ax0)
                        h = T.axis.spatial(12, i1_1 * 2 + ax1)
                        d = T.axis.spatial(64, ax2)
                        l = T.axis.spatial(128, i3_0 * 64 + i3_1 * 8 + ax3)
                        T.reads(value[b, l, h, d])
                        T.writes(value_T[b, h, d, l])
                        value_T[b, h, d, l] = value[b, l, h, d]
                for i4_0, i0_2, i1_2, i2_2, i3_2, i4_1, i0_3, i1_3, i2_3, i3_3 in T.grid(1, 1, 2, 2, 4, 64, 1, 1, 32, 2):
                    with T.block("C"):
                        b = T.axis.spatial(1, i0_1 + i0_2 + i0_3 + i0_0)
                        h = T.axis.spatial(12, i1_0 * 12 + i1_1 * 2 + i1_2 + i1_3)
                        i = T.axis.spatial(128, i2_0 * 128 + i2_1 * 64 + i2_2 * 32 + i2_3)
                        j = T.axis.spatial(128, i3_0 * 64 + i3_1 * 8 + i3_2 * 2 + i3_3)
                        k = T.axis.reduce(64, i4_0 * 64 + i4_1)
                        T.reads(query[b, i, h, k], value_T[b, h, k, j])
                        T.writes(C[b, h, i, j])
                        T.block_attr({"meta_schedule.tiling_structure":"SSRSRS"})
                        with T.init():
                            C[b, h, i, j] = T.float32(0)
                        C[b, h, i, j] = C[b, h, i, j] + query[b, i, h, k] * value_T[b, h, k, j]
    # fmt: on
    decision_0 = [
        ("SamplePerfectTile", [1, 1, 1, 1]),
        ("SamplePerfectTile", [1, 6, 2, 1]),
        ("SamplePerfectTile", [1, 2, 2, 32]),
        ("SamplePerfectTile", [2, 8, 4, 2]),
        ("SamplePerfectTile", [1, 64]),
        ("SampleCategorical", 2),
        ("SampleComputeLocation", 6),
        ("SampleComputeLocation", 6),
    ]
    decision_1 = [
        ("SamplePerfectTile", [1, 1, 1, 1]),
        ("SamplePerfectTile", [1, 6, 2, 1]),
        ("SamplePerfectTile", [1, 2, 2, 32]),
        ("SamplePerfectTile", [2, 8, 4, 2]),
        ("SamplePerfectTile", [1, 64]),
        ("SampleCategorical", 2),
        ("SampleComputeLocation", 13),
        ("SampleComputeLocation", -1),
    ]
    decision_2 = [
        ("SamplePerfectTile", [1, 1, 1, 1]),
        ("SamplePerfectTile", [1, 6, 2, 1]),
        ("SamplePerfectTile", [1, 2, 2, 32]),
        ("SamplePerfectTile", [2, 8, 4, 2]),
        ("SamplePerfectTile", [1, 64]),
        ("SampleCategorical", 3),
        ("SampleComputeLocation", 7),
        ("SampleComputeLocation", -2),
    ]
    mod = create_te_workload("TBG", 0)
    actual = _design_space(mod)
    check_sketches(
        mod,
        sketches=actual,
        expected_mods=[tbg_0, tbg_1, tbg_2],
        expected_decisions=[decision_0, decision_1, decision_2],
    )


if __name__ == "__main__":
    test_cpu_c1d()
    test_cpu_c2d()
    test_cpu_c3d()
    test_cpu_cap()
    test_cpu_dep()
    test_cpu_dil()
    test_cpu_gmm()
    test_cpu_grp()
    test_cpu_t2d()
    test_cpu_nrm()
    test_cpu_sfm()
    test_cpu_cbr()
    test_cpu_tbg()
