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
from tvm.meta_schedule.testing.space_generation import check_sketches, print_sketches
from tvm.meta_schedule.testing.te_workload import create_te_workload
from tvm.script import tir as T
from tvm.target import Target


def _target():
    return Target("aws/cpu/c5.9xlarge")


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
                        n = T.axis.spatial(1, i0_0 + i0_1_1 + i0_2 + i0_3)
                        l = T.axis.spatial(128, i1_1_1 * 128 + i1_0 * 128 + i1_2 * 2 + i1_3)
                        co = T.axis.spatial(128, (i2_0 * 8 + i2_1_1) * 8 + i2_2 + i2_3)
                        rl = T.axis.reduce(3, i3_0 * 3 + i3_1)
                        rc = T.axis.reduce(64, i4_0 + i4_1)
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
                            n = T.axis.spatial(1, i0_0 + i0_1 + i0_2 + i0_3)
                            l = T.axis.spatial(128, i1_1 * 128 + i1_0 * 128 + i1_2 * 2 + i1_3)
                            co = T.axis.spatial(128, (i2_0 * 8 + i2_1) * 8 + i2_2 + i2_3)
                            rl = T.axis.reduce(3, i3_0 * 3 + i3_1)
                            rc = T.axis.reduce(64, i4_0 + i4_1)
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
                    n = T.axis.spatial(1, i0_0 + i0_1 + i0_2 + i0_3)
                    l = T.axis.spatial(128, i1_1 * 128 + i1_0 * 128 + i1_2 * 2 + i1_3)
                    co = T.axis.spatial(128, (i2_0 * 8 + i2_1) * 8 + i2_2 + i2_3)
                    rl = T.axis.reduce(3, i3_0 * 3 + i3_1)
                    rc = T.axis.reduce(64, i4_0 + i4_1)
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
    actual = ms.TuneContext(
        mod=mod,
        target=_target(),
        space_generator=ms.space_generator.PostOrderApply(),
        sch_rules="default",
    ).generate_design_space()
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
                            n = T.axis.spatial(1, i0_3 + i0_2 + i0_1 + i0_0)
                            h = T.axis.spatial(112, ((i1_0 + i1_1) * 2 + i1_2) * 8 + i1_3)
                            w = T.axis.spatial(112, i2_0 * 28 + i2_1 + i2_2 + i2_3)
                            co = T.axis.spatial(64, (i3_0 * 8 + i3_1 + i3_2) * 4 + i3_3)
                            rh = T.axis.reduce(7, i4_0 + i4_1)
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
                        n = T.axis.spatial(1, i0_3 + i0_2 + i0_1_1 + i0_0)
                        h = T.axis.spatial(112, ((i1_0 + i1_1_1) * 2 + i1_2) * 8 + i1_3)
                        w = T.axis.spatial(112, i2_0 * 28 + i2_1_1 + i2_2 + i2_3)
                        co = T.axis.spatial(64, (i3_0 * 8 + i3_1_1 + i3_2) * 4 + i3_3)
                        rh = T.axis.reduce(7, i4_0 + i4_1)
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
                        n = T.axis.spatial(1, i0_3 + i0_2 + i0_1 + i0_0)
                        h = T.axis.spatial(112, ((i1_0 + i1_1) * 2 + i1_2) * 8 + i1_3)
                        w = T.axis.spatial(112, i2_0 * 28 + i2_1 + i2_2 + i2_3)
                        co = T.axis.spatial(64, (i3_0 * 8 + i3_1 + i3_2) * 4 + i3_3)
                        rh = T.axis.reduce(7, i4_0 + i4_1)
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
    actual = ms.TuneContext(
        mod=mod,
        target=_target(),
        space_generator=ms.space_generator.PostOrderApply(),
        sch_rules="default",
    ).generate_design_space()
    check_sketches(
        mod,
        sketches=actual,
        expected_mods=[c2d_0, c2d_1, c2d_2],
        expected_decisions=[decision_0, decision_1, decision_2],
    )


if __name__ == "__main__":
    test_cpu_c1d()
    test_cpu_c2d()
