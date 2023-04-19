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
from tvm.meta_schedule.testing.space_generation import (
    check_sketches,
    generate_design_space,
    print_sketches,
)
from tvm.meta_schedule.testing.te_workload import create_te_workload
from tvm.script import tir as T
from tvm.target import Target


def _target():
    return Target("nvidia/geforce-rtx-2080")  # disable async trace using sm75


def _design_space(mod):
    return generate_design_space(
        kind="cuda",
        mod=mod,
        target=_target(),
        types=ms.ScheduleRule,
    )


def test_cuda_c1d():
    # fmt: off
    @T.prim_func
    def c1d_0(inputs: T.Buffer((1, 256, 64), "float32"), weight: T.Buffer((3, 64, 128), "float32"), conv1d_nlc: T.Buffer((1, 128, 128), "float32")) -> None:
        T.func_attr({"global_symbol": "main", "tir.noalias": T.bool(True)})
        with T.block("root"):
            T.reads()
            T.writes()
            T.block_attr({"meta_schedule.unroll_explicit": 16})
            conv1d_nlc_local = T.alloc_buffer((1, 128, 128), scope="local")
            PadInput_shared = T.alloc_buffer((1, 258, 64), scope="shared")
            weight_shared = T.alloc_buffer((3, 64, 128), scope="shared")
            for n_0_l_0_co_0_fused in T.thread_binding(4, thread="blockIdx.x"):
                for n_1_l_1_co_1_fused in T.thread_binding(16, thread="vthread.x"):
                    for n_2_l_2_co_2_fused in T.thread_binding(4, thread="threadIdx.x"):
                        for rl_0, rc_0 in T.grid(1, 16):
                            for ax0_ax1_ax2_fused in range(260):
                                with T.block("PadInput_shared"):
                                    v0 = T.axis.spatial(1, 0)
                                    v1 = T.axis.spatial(258, n_0_l_0_co_0_fused * 64 + ax0_ax1_ax2_fused // 4)
                                    v2 = T.axis.spatial(64, rc_0 * 4 + ax0_ax1_ax2_fused % 4)
                                    T.reads(inputs[v0, v1 - 1, v2])
                                    T.writes(PadInput_shared[v0, v1, v2])
                                    T.block_attr({"meta_schedule.cooperative_fetch": 4})
                                    PadInput_shared[v0, v1, v2] = T.if_then_else(1 <= v1 and v1 < 257, inputs[v0, v1 - 1, v2], T.float32(0))
                            for ax0_ax1_ax2_fused in range(1536):
                                with T.block("weight_shared"):
                                    v0 = T.axis.spatial(3, ax0_ax1_ax2_fused // 512)
                                    v1 = T.axis.spatial(64, rc_0 * 4 + ax0_ax1_ax2_fused % 512 // 128)
                                    v2 = T.axis.spatial(128, ax0_ax1_ax2_fused % 128)
                                    T.reads(weight[v0, v1, v2])
                                    T.writes(weight_shared[v0, v1, v2])
                                    T.block_attr({"meta_schedule.cooperative_fetch": 3})
                                    weight_shared[v0, v1, v2] = weight[v0, v1, v2]
                            for rl_1, rc_1, n_3, l_3, co_3, rl_2, rc_2, n_4, l_4, co_4 in T.grid(1, 2, 1, 1, 2, 3, 2, 1, 4, 8):
                                with T.block("conv1d_nlc"):
                                    v_n = T.axis.spatial(1, n_3 + n_4)
                                    v_l = T.axis.spatial(128, n_0_l_0_co_0_fused * 32 + n_1_l_1_co_1_fused // 2 * 4 + l_3 * 4 + l_4)
                                    v_co = T.axis.spatial(128, n_1_l_1_co_1_fused % 2 * 64 + n_2_l_2_co_2_fused * 16 + co_3 * 8 + co_4)
                                    v_rl = T.axis.reduce(3, rl_0 * 3 + rl_1 * 3 + rl_2)
                                    v_rc = T.axis.reduce(64, rc_0 * 4 + rc_1 * 2 + rc_2)
                                    T.reads(PadInput_shared[v_n, v_l * 2 + v_rl, v_co // 128 * 64 + v_rc], weight_shared[v_rl, v_rc, v_co])
                                    T.writes(conv1d_nlc_local[v_n, v_l, v_co])
                                    T.block_attr({"meta_schedule.thread_extent_high_inclusive": 1024, "meta_schedule.thread_extent_low_inclusive": 32, "meta_schedule.tiling_structure": "SSSRRSRS"})
                                    with T.init():
                                        conv1d_nlc_local[v_n, v_l, v_co] = T.float32(0)
                                    conv1d_nlc_local[v_n, v_l, v_co] = conv1d_nlc_local[v_n, v_l, v_co] + PadInput_shared[v_n, v_l * 2 + v_rl, v_co // 128 * 64 + v_rc] * weight_shared[v_rl, v_rc, v_co]
                        for ax0, ax1, ax2 in T.grid(1, 4, 16):
                            with T.block("conv1d_nlc_local"):
                                v0 = T.axis.spatial(1, ax0)
                                v1 = T.axis.spatial(128, n_0_l_0_co_0_fused * 32 + n_1_l_1_co_1_fused // 2 * 4 + ax1)
                                v2 = T.axis.spatial(128, n_1_l_1_co_1_fused % 2 * 64 + n_2_l_2_co_2_fused * 16 + ax2)
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
    actual = _design_space(mod)
    check_sketches(
        mod,
        sketches=actual,
        expected_mods=[c1d_0],
        expected_decisions=[decision_0],
    )


def test_cuda_c2d():
    # fmt: off
    @T.prim_func
    def c2d_0(inputs: T.Buffer((1, 224, 224, 3), "float32"), weight: T.Buffer((7, 7, 3, 64), "float32"), conv2d_nhwc: T.Buffer((1, 112, 112, 64), "float32")) -> None:
        T.func_attr({"global_symbol": "main", "tir.noalias": T.bool(True)})
        with T.block("root"):
            T.reads()
            T.writes()
            T.block_attr({"meta_schedule.unroll_explicit": 16})
            conv2d_nhwc_local = T.alloc_buffer((1, 112, 112, 64), scope="local")
            PadInput_shared = T.alloc_buffer((1, 230, 230, 3), scope="shared")
            weight_shared = T.alloc_buffer((7, 7, 3, 64), scope="shared")
            for n_0_h_0_w_0_co_0_fused in T.thread_binding(16, thread="blockIdx.x"):
                for n_1_h_1_w_1_co_1_fused in T.thread_binding(56, thread="vthread.x"):
                    for n_2_h_2_w_2_co_2_fused in T.thread_binding(14, thread="threadIdx.x"):
                        for rh_0, rw_0, rc_0 in T.grid(1, 1, 1):
                            for ax0_ax1_ax2_ax3_fused in range(80379):
                                with T.block("PadInput_shared"):
                                    v0 = T.axis.spatial(1, 0)
                                    v1 = T.axis.spatial(230, ax0_ax1_ax2_ax3_fused // 351)
                                    v2 = T.axis.spatial(230, n_0_h_0_w_0_co_0_fused // 8 * 112 + ax0_ax1_ax2_ax3_fused % 351 // 3)
                                    v3 = T.axis.spatial(3, ax0_ax1_ax2_ax3_fused % 3)
                                    T.reads(inputs[v0, v1 - 3, v2 - 3, v3])
                                    T.writes(PadInput_shared[v0, v1, v2, v3])
                                    T.block_attr({"meta_schedule.cooperative_fetch": 2})
                                    PadInput_shared[v0, v1, v2, v3] = T.if_then_else(3 <= v1 and v1 < 227 and 3 <= v2 and v2 < 227, inputs[v0, v1 - 3, v2 - 3, v3], T.float32(0))
                            for ax0_ax1_ax2_ax3_fused in range(1176):
                                with T.block("weight_shared"):
                                    v0 = T.axis.spatial(7, ax0_ax1_ax2_ax3_fused // 168)
                                    v1 = T.axis.spatial(7, ax0_ax1_ax2_ax3_fused % 168 // 24)
                                    v2 = T.axis.spatial(3, ax0_ax1_ax2_ax3_fused % 24 // 8)
                                    v3 = T.axis.spatial(64, n_0_h_0_w_0_co_0_fused % 8 * 8 + ax0_ax1_ax2_ax3_fused % 8)
                                    T.reads(weight[v0, v1, v2, v3])
                                    T.writes(weight_shared[v0, v1, v2, v3])
                                    T.block_attr({"meta_schedule.cooperative_fetch": 4})
                                    weight_shared[v0, v1, v2, v3] = weight[v0, v1, v2, v3]
                            for rh_1, rw_1, rc_1, n_3, h_3, w_3, co_3, rh_2, rw_2, rc_2, n_4, h_4, w_4, co_4 in T.grid(1, 7, 1, 1, 8, 4, 1, 7, 1, 3, 1, 1, 1, 2):
                                with T.block("conv2d_nhwc"):
                                    v_n = T.axis.spatial(1, n_3 + n_4)
                                    v_h = T.axis.spatial(112, n_2_h_2_w_2_co_2_fused * 8 + h_3 + h_4)
                                    v_w = T.axis.spatial(112, n_0_h_0_w_0_co_0_fused // 8 * 56 + n_1_h_1_w_1_co_1_fused // 4 * 4 + w_3 + w_4)
                                    v_co = T.axis.spatial(64, n_0_h_0_w_0_co_0_fused % 8 * 8 + n_1_h_1_w_1_co_1_fused % 4 * 2 + co_3 * 2 + co_4)
                                    v_rh = T.axis.reduce(7, rh_0 * 7 + rh_1 * 7 + rh_2)
                                    v_rw = T.axis.reduce(7, rw_0 * 7 + rw_1 + rw_2)
                                    v_rc = T.axis.reduce(3, rc_0 * 3 + rc_1 * 3 + rc_2)
                                    T.reads(PadInput_shared[v_n, v_h * 2 + v_rh, v_w * 2 + v_rw, v_co // 64 * 3 + v_rc], weight_shared[v_rh, v_rw, v_rc, v_co])
                                    T.writes(conv2d_nhwc_local[v_n, v_h, v_w, v_co])
                                    T.block_attr({"meta_schedule.thread_extent_high_inclusive": 1024, "meta_schedule.thread_extent_low_inclusive": 32, "meta_schedule.tiling_structure": "SSSRRSRS"})
                                    with T.init():
                                        conv2d_nhwc_local[v_n, v_h, v_w, v_co] = T.float32(0)
                                    conv2d_nhwc_local[v_n, v_h, v_w, v_co] = conv2d_nhwc_local[v_n, v_h, v_w, v_co] + PadInput_shared[v_n, v_h * 2 + v_rh, v_w * 2 + v_rw, v_co // 64 * 3 + v_rc] * weight_shared[v_rh, v_rw, v_rc, v_co]
                        for ax0, ax1, ax2, ax3 in T.grid(1, 8, 4, 2):
                            with T.block("conv2d_nhwc_local"):
                                v0 = T.axis.spatial(1, ax0)
                                v1 = T.axis.spatial(112, n_2_h_2_w_2_co_2_fused * 8 + ax1)
                                v2 = T.axis.spatial(112, n_0_h_0_w_0_co_0_fused // 8 * 56 + n_1_h_1_w_1_co_1_fused // 4 * 4 + ax2)
                                v3 = T.axis.spatial(64, n_0_h_0_w_0_co_0_fused % 8 * 8 + n_1_h_1_w_1_co_1_fused % 4 * 2 + ax3)
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
    actual = _design_space(mod)
    check_sketches(
        mod,
        sketches=actual,
        expected_mods=[c2d_0],
        expected_decisions=[decision_0],
    )


def test_cuda_c3d():
    # fmt: off
    @T.prim_func
    def c3d_0(inputs: T.Buffer((1, 16, 224, 224, 3), "float32"), weight: T.Buffer((7, 7, 7, 3, 64), "float32"), conv3d_ndhwc: T.Buffer((1, 8, 112, 112, 64), "float32")) -> None:
        T.func_attr({"global_symbol": "main", "tir.noalias": T.bool(True)})
        with T.block("root"):
            T.reads()
            T.writes()
            T.block_attr({"meta_schedule.unroll_explicit": 16})
            conv3d_ndhwc_local = T.alloc_buffer((1, 8, 112, 112, 64), scope="local")
            PadInput_shared = T.alloc_buffer((1, 22, 230, 230, 3), scope="shared")
            weight_shared = T.alloc_buffer((7, 7, 7, 3, 64), scope="shared")
            for n_0_d_0_h_0_w_0_co_0_fused in T.thread_binding(2, thread="blockIdx.x"):
                for n_1_d_1_h_1_w_1_co_1_fused in T.thread_binding(8, thread="vthread.x"):
                    for n_2_d_2_h_2_w_2_co_2_fused in T.thread_binding(392, thread="threadIdx.x"):
                        for rd_0, rh_0, rw_0, rc_0 in T.grid(1, 1, 1, 1):
                            for ax0_ax1_ax2_ax3_ax4_fused in range(1687959):
                                with T.block("PadInput_shared"):
                                    v0 = T.axis.spatial(1, 0)
                                    v1 = T.axis.spatial(22, ax0_ax1_ax2_ax3_ax4_fused // 80379)
                                    v2 = T.axis.spatial(230, ax0_ax1_ax2_ax3_ax4_fused % 80379 // 351)
                                    v3 = T.axis.spatial(230, n_0_d_0_h_0_w_0_co_0_fused * 112 + ax0_ax1_ax2_ax3_ax4_fused % 351 // 3)
                                    v4 = T.axis.spatial(3, ax0_ax1_ax2_ax3_ax4_fused % 3)
                                    T.reads(inputs[v0, v1 - 3, v2 - 3, v3 - 3, v4])
                                    T.writes(PadInput_shared[v0, v1, v2, v3, v4])
                                    T.block_attr({"meta_schedule.cooperative_fetch": 4})
                                    PadInput_shared[v0, v1, v2, v3, v4] = T.if_then_else(3 <= v1 and v1 < 19 and 3 <= v2 and v2 < 227 and 3 <= v3 and v3 < 227, inputs[v0, v1 - 3, v2 - 3, v3 - 3, v4], T.float32(0))
                            for ax0_ax1_ax2_ax3_ax4_fused in range(65856):
                                with T.block("weight_shared"):
                                    v0 = T.axis.spatial(7, ax0_ax1_ax2_ax3_ax4_fused // 9408)
                                    v1 = T.axis.spatial(7, ax0_ax1_ax2_ax3_ax4_fused % 9408 // 1344)
                                    v2 = T.axis.spatial(7, ax0_ax1_ax2_ax3_ax4_fused % 1344 // 192)
                                    v3 = T.axis.spatial(3, ax0_ax1_ax2_ax3_ax4_fused % 192 // 64)
                                    v4 = T.axis.spatial(64, ax0_ax1_ax2_ax3_ax4_fused % 64)
                                    T.reads(weight[v0, v1, v2, v3, v4])
                                    T.writes(weight_shared[v0, v1, v2, v3, v4])
                                    T.block_attr({"meta_schedule.cooperative_fetch": 3})
                                    weight_shared[v0, v1, v2, v3, v4] = weight[v0, v1, v2, v3, v4]
                            for rd_1, rh_1, rw_1, rc_1, n_3, d_3, h_3, w_3, co_3, rd_2, rh_2, rw_2, rc_2, n_4, d_4, h_4, w_4, co_4 in T.grid(7, 7, 1, 3, 1, 2, 2, 1, 32, 1, 1, 7, 1, 1, 1, 2, 4, 1):
                                with T.block("conv3d_ndhwc"):
                                    v_n = T.axis.spatial(1, n_3 + n_4)
                                    v_d = T.axis.spatial(8, n_2_d_2_h_2_w_2_co_2_fused // 98 * 2 + d_3 + d_4)
                                    v_h = T.axis.spatial(112, n_1_d_1_h_1_w_1_co_1_fused // 2 * 28 + n_2_d_2_h_2_w_2_co_2_fused % 98 // 14 * 4 + h_3 * 2 + h_4)
                                    v_w = T.axis.spatial(112, n_0_d_0_h_0_w_0_co_0_fused * 56 + n_1_d_1_h_1_w_1_co_1_fused % 2 * 28 + n_2_d_2_h_2_w_2_co_2_fused % 14 // 2 * 4 + w_3 * 4 + w_4)
                                    v_co = T.axis.spatial(64, n_2_d_2_h_2_w_2_co_2_fused % 2 * 32 + co_3 + co_4)
                                    v_rd = T.axis.reduce(7, rd_0 * 7 + rd_1 + rd_2)
                                    v_rh = T.axis.reduce(7, rh_0 * 7 + rh_1 + rh_2)
                                    v_rw = T.axis.reduce(7, rw_0 * 7 + rw_1 * 7 + rw_2)
                                    v_rc = T.axis.reduce(3, rc_0 * 3 + rc_1 + rc_2)
                                    T.reads(PadInput_shared[v_n, v_d * 2 + v_rd, v_h * 2 + v_rh, v_w * 2 + v_rw, v_co // 64 * 3 + v_rc], weight_shared[v_rd, v_rh, v_rw, v_rc, v_co])
                                    T.writes(conv3d_ndhwc_local[v_n, v_d, v_h, v_w, v_co])
                                    T.block_attr({"meta_schedule.thread_extent_high_inclusive": 1024, "meta_schedule.thread_extent_low_inclusive": 32, "meta_schedule.tiling_structure": "SSSRRSRS"})
                                    with T.init():
                                        conv3d_ndhwc_local[v_n, v_d, v_h, v_w, v_co] = T.float32(0)
                                    conv3d_ndhwc_local[v_n, v_d, v_h, v_w, v_co] = conv3d_ndhwc_local[v_n, v_d, v_h, v_w, v_co] + PadInput_shared[v_n, v_d * 2 + v_rd, v_h * 2 + v_rh, v_w * 2 + v_rw, v_co // 64 * 3 + v_rc] * weight_shared[v_rd, v_rh, v_rw, v_rc, v_co]
                        for ax0, ax1, ax2, ax3, ax4 in T.grid(1, 2, 4, 4, 32):
                            with T.block("conv3d_ndhwc_local"):
                                v0 = T.axis.spatial(1, ax0)
                                v1 = T.axis.spatial(8, n_2_d_2_h_2_w_2_co_2_fused // 98 * 2 + ax1)
                                v2 = T.axis.spatial(112, n_1_d_1_h_1_w_1_co_1_fused // 2 * 28 + n_2_d_2_h_2_w_2_co_2_fused % 98 // 14 * 4 + ax2)
                                v3 = T.axis.spatial(112, n_0_d_0_h_0_w_0_co_0_fused * 56 + n_1_d_1_h_1_w_1_co_1_fused % 2 * 28 + n_2_d_2_h_2_w_2_co_2_fused % 14 // 2 * 4 + ax3)
                                v4 = T.axis.spatial(64, n_2_d_2_h_2_w_2_co_2_fused % 2 * 32 + ax4)
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
    actual = _design_space(mod)
    check_sketches(
        mod,
        sketches=actual,
        expected_mods=[c3d_0],
        expected_decisions=[decision_0],
    )


def test_cuda_cap():
    # fmt: off
    @T.prim_func
    def cap_0(inputs: T.Buffer((1, 16, 16, 4, 4, 32), "float32"), weight: T.Buffer((3, 3, 4, 4, 32, 32), "float32"), conv2d_capsule_nhwijc: T.Buffer((1, 8, 8, 4, 4, 32), "float32")) -> None:
        T.func_attr({"global_symbol": "main", "tir.noalias": T.bool(True)})
        with T.block("root"):
            T.reads()
            T.writes()
            T.block_attr({"meta_schedule.unroll_explicit": 64})
            conv2d_capsule_nhwijc_local = T.alloc_buffer((1, 8, 8, 4, 4, 32), scope="local")
            PadInput_shared = T.alloc_buffer((1, 18, 18, 4, 4, 32), scope="shared")
            weight_shared = T.alloc_buffer((3, 3, 4, 4, 32, 32), scope="shared")
            for n_0_h_0_w_0_cap_i_0_cap_j_0_co_0_fused in T.thread_binding(256, thread="blockIdx.x"):
                for n_1_h_1_w_1_cap_i_1_cap_j_1_co_1_fused in T.thread_binding(1, thread="vthread.x"):
                    for n_2_h_2_w_2_cap_i_2_cap_j_2_co_2_fused in T.thread_binding(4, thread="threadIdx.x"):
                        for rh_0, rw_0, cap_k_0, rc_0 in T.grid(3, 3, 2, 8):
                            for ax0_ax1_ax2_ax3_ax4_ax5_fused in range(48):
                                with T.block("PadInput_shared"):
                                    v0 = T.axis.spatial(1, 0)
                                    v1 = T.axis.spatial(18, n_0_h_0_w_0_cap_i_0_cap_j_0_co_0_fused // 64 * 4 + rh_0 + ax0_ax1_ax2_ax3_ax4_ax5_fused % 48 // 16)
                                    v2 = T.axis.spatial(18, T.Add(n_0_h_0_w_0_cap_i_0_cap_j_0_co_0_fused % 64 // 8 * 2 + rw_0, 0))
                                    v3 = T.axis.spatial(4, n_0_h_0_w_0_cap_i_0_cap_j_0_co_0_fused % 8 // 4 * 2 + ax0_ax1_ax2_ax3_ax4_ax5_fused % 16 // 8)
                                    v4 = T.axis.spatial(4, cap_k_0 * 2 + ax0_ax1_ax2_ax3_ax4_ax5_fused % 8 // 4)
                                    v5 = T.axis.spatial(32, rc_0 * 4 + ax0_ax1_ax2_ax3_ax4_ax5_fused % 4)
                                    T.reads(inputs[v0, v1 - 1, v2 - 1, v3, v4, v5])
                                    T.writes(PadInput_shared[v0, v1, v2, v3, v4, v5])
                                    T.block_attr({"meta_schedule.cooperative_fetch": 2})
                                    PadInput_shared[v0, v1, v2, v3, v4, v5] = T.if_then_else(1 <= v1 and v1 < 17 and 1 <= v2 and v2 < 17, inputs[v0, v1 - 1, v2 - 1, v3, v4, v5], T.float32(0))
                            for ax0_ax1_ax2_ax3_ax4_ax5_fused in range(256):
                                with T.block("weight_shared"):
                                    v0, v1 = T.axis.remap("SS", [rh_0, rw_0])
                                    v2 = T.axis.spatial(4, cap_k_0 * 2 + ax0_ax1_ax2_ax3_ax4_ax5_fused // 128)
                                    v3 = T.axis.spatial(4, ax0_ax1_ax2_ax3_ax4_ax5_fused % 128 // 32)
                                    v4 = T.axis.spatial(32, rc_0 * 4 + ax0_ax1_ax2_ax3_ax4_ax5_fused % 32 // 8)
                                    v5 = T.axis.spatial(32, n_0_h_0_w_0_cap_i_0_cap_j_0_co_0_fused % 4 * 8 + ax0_ax1_ax2_ax3_ax4_ax5_fused % 8)
                                    T.reads(weight[v0, v1, v2, v3, v4, v5])
                                    T.writes(weight_shared[v0, v1, v2, v3, v4, v5])
                                    T.block_attr({"meta_schedule.cooperative_fetch": 4})
                                    weight_shared[v0, v1, v2, v3, v4, v5] = weight[v0, v1, v2, v3, v4, v5]
                            for rh_1, rw_1, cap_k_1, rc_1, n_3, h_3, w_3, cap_i_3, cap_j_3, co_3, rh_2, rw_2, cap_k_2, rc_2, n_4, h_4, w_4, cap_i_4, cap_j_4, co_4 in T.grid(1, 1, 1, 4, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 8):
                                with T.block("conv2d_capsule_nhwijc"):
                                    v_n = T.axis.spatial(1, n_3 + n_4)
                                    v_h = T.axis.spatial(8, n_0_h_0_w_0_cap_i_0_cap_j_0_co_0_fused // 64 * 2 + h_3 + h_4)
                                    v_w = T.axis.spatial(8, n_0_h_0_w_0_cap_i_0_cap_j_0_co_0_fused % 64 // 8 + w_3 + w_4)
                                    v_cap_i = T.axis.spatial(4, n_0_h_0_w_0_cap_i_0_cap_j_0_co_0_fused % 8 // 4 * 2 + n_2_h_2_w_2_cap_i_2_cap_j_2_co_2_fused // 2 + cap_i_3 + cap_i_4)
                                    v_cap_j = T.axis.spatial(4, n_2_h_2_w_2_cap_i_2_cap_j_2_co_2_fused % 2 * 2 + cap_j_3 * 2 + cap_j_4)
                                    v_co = T.axis.spatial(32, n_0_h_0_w_0_cap_i_0_cap_j_0_co_0_fused % 4 * 8 + co_3 * 8 + co_4)
                                    v_rh = T.axis.reduce(3, rh_0 + rh_1 + rh_2)
                                    v_rw = T.axis.reduce(3, rw_0 + rw_1 + rw_2)
                                    v_cap_k = T.axis.reduce(4, cap_k_0 * 2 + cap_k_1 * 2 + cap_k_2)
                                    v_rc = T.axis.reduce(32, rc_0 * 4 + rc_1 + rc_2)
                                    T.reads(PadInput_shared[v_n, v_h * 2 + v_rh, v_w * 2 + v_rw, v_cap_i, v_cap_k, v_rc], weight_shared[v_rh, v_rw, v_cap_k, v_cap_j, v_rc, v_co])
                                    T.writes(conv2d_capsule_nhwijc_local[v_n, v_h, v_w, v_cap_i, v_cap_j, v_co])
                                    T.block_attr({"meta_schedule.thread_extent_high_inclusive": 1024, "meta_schedule.thread_extent_low_inclusive": 32, "meta_schedule.tiling_structure": "SSSRRSRS"})
                                    with T.init():
                                        conv2d_capsule_nhwijc_local[v_n, v_h, v_w, v_cap_i, v_cap_j, v_co] = T.float32(0)
                                    conv2d_capsule_nhwijc_local[v_n, v_h, v_w, v_cap_i, v_cap_j, v_co] = conv2d_capsule_nhwijc_local[v_n, v_h, v_w, v_cap_i, v_cap_j, v_co] + PadInput_shared[v_n, v_h * 2 + v_rh, v_w * 2 + v_rw, v_cap_i, v_cap_k, v_rc] * weight_shared[v_rh, v_rw, v_cap_k, v_cap_j, v_rc, v_co]
                        for ax0, ax1, ax2, ax3, ax4, ax5 in T.grid(1, 2, 1, 1, 2, 8):
                            with T.block("conv2d_capsule_nhwijc_local"):
                                v0 = T.axis.spatial(1, ax0)
                                v1 = T.axis.spatial(8, n_0_h_0_w_0_cap_i_0_cap_j_0_co_0_fused // 64 * 2 + ax1)
                                v2 = T.axis.spatial(8, n_0_h_0_w_0_cap_i_0_cap_j_0_co_0_fused % 64 // 8 + ax2)
                                v3 = T.axis.spatial(4, n_0_h_0_w_0_cap_i_0_cap_j_0_co_0_fused % 8 // 4 * 2 + n_2_h_2_w_2_cap_i_2_cap_j_2_co_2_fused // 2 + ax3)
                                v4 = T.axis.spatial(4, n_2_h_2_w_2_cap_i_2_cap_j_2_co_2_fused % 2 * 2 + ax4)
                                v5 = T.axis.spatial(32, n_0_h_0_w_0_cap_i_0_cap_j_0_co_0_fused % 4 * 8 + ax5)
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
    actual = _design_space(mod)
    check_sketches(
        mod,
        sketches=actual,
        expected_mods=[cap_0],
        expected_decisions=[decision_0],
    )


def test_cuda_dep():
    # fmt: off
    @T.prim_func
    def dep_0(placeholder: T.Buffer((1, 112, 112, 32), "float32"), placeholder_1: T.Buffer((1, 3, 3, 32), "float32"), depth_conv2d_nhwc: T.Buffer((1, 112, 112, 32), "float32")) -> None:
        T.func_attr({"global_symbol": "main", "tir.noalias": T.bool(True)})
        with T.block("root"):
            T.reads()
            T.writes()
            T.block_attr({"meta_schedule.unroll_explicit": 16})
            depth_conv2d_nhwc_local = T.alloc_buffer((1, 112, 112, 32), scope="local")
            PadInput_shared = T.alloc_buffer((1, 114, 114, 32), scope="shared")
            placeholder_shared = T.alloc_buffer((1, 3, 3, 32), scope="shared")
            for n_0_h_0_w_0_c_0_fused in T.thread_binding(1, thread="blockIdx.x"):
                for n_1_h_1_w_1_c_1_fused in T.thread_binding(8, thread="vthread.x"):
                    for n_2_h_2_w_2_c_2_fused in T.thread_binding(14, thread="threadIdx.x"):
                        for rh_0, rw_0 in T.grid(1, 1):
                            for ax0_ax1_ax2_ax3_fused in range(415872):
                                with T.block("PadInput_shared"):
                                    v0 = T.axis.spatial(1, 0)
                                    v1 = T.axis.spatial(114, ax0_ax1_ax2_ax3_fused // 3648)
                                    v2 = T.axis.spatial(114, ax0_ax1_ax2_ax3_fused % 3648 // 32)
                                    v3 = T.axis.spatial(32, ax0_ax1_ax2_ax3_fused % 32)
                                    T.reads(placeholder[v0, v1 - 1, v2 - 1, v3])
                                    T.writes(PadInput_shared[v0, v1, v2, v3])
                                    T.block_attr({"meta_schedule.cooperative_fetch": 3})
                                    PadInput_shared[v0, v1, v2, v3] = T.if_then_else(1 <= v1 and v1 < 113 and 1 <= v2 and v2 < 113, placeholder[v0, v1 - 1, v2 - 1, v3], T.float32(0))
                            for ax0_ax1_ax2_ax3_fused in range(288):
                                with T.block("placeholder_shared"):
                                    v0 = T.axis.spatial(1, 0)
                                    v1 = T.axis.spatial(3, ax0_ax1_ax2_ax3_fused // 96)
                                    v2 = T.axis.spatial(3, ax0_ax1_ax2_ax3_fused % 96 // 32)
                                    v3 = T.axis.spatial(32, ax0_ax1_ax2_ax3_fused % 32)
                                    T.reads(placeholder_1[v0, v1, v2, v3])
                                    T.writes(placeholder_shared[v0, v1, v2, v3])
                                    T.block_attr({"meta_schedule.cooperative_fetch": 3})
                                    placeholder_shared[v0, v1, v2, v3] = placeholder_1[v0, v1, v2, v3]
                            for rh_1, rw_1, n_3, h_3, w_3, c_3, rh_2, rw_2, n_4, h_4, w_4, c_4 in T.grid(3, 1, 1, 4, 16, 8, 1, 3, 1, 7, 1, 1):
                                with T.block("depth_conv2d_nhwc"):
                                    v_n = T.axis.spatial(1, n_3 + n_4)
                                    v_h = T.axis.spatial(112, n_1_h_1_w_1_c_1_fused // 2 * 28 + h_3 * 7 + h_4)
                                    v_w = T.axis.spatial(112, n_2_h_2_w_2_c_2_fused // 2 * 16 + w_3 + w_4)
                                    v_c = T.axis.spatial(32, n_1_h_1_w_1_c_1_fused % 2 * 16 + n_2_h_2_w_2_c_2_fused % 2 * 8 + c_3 + c_4)
                                    v_rh = T.axis.reduce(3, rh_0 * 3 + rh_1 + rh_2)
                                    v_rw = T.axis.reduce(3, rw_0 * 3 + rw_1 * 3 + rw_2)
                                    T.reads(PadInput_shared[v_n, v_h + v_rh, v_w + v_rw, v_c], placeholder_shared[0, v_rh, v_rw, v_c])
                                    T.writes(depth_conv2d_nhwc_local[v_n, v_h, v_w, v_c])
                                    T.block_attr({"meta_schedule.thread_extent_high_inclusive": 1024, "meta_schedule.thread_extent_low_inclusive": 32, "meta_schedule.tiling_structure": "SSSRRSRS"})
                                    with T.init():
                                        depth_conv2d_nhwc_local[v_n, v_h, v_w, v_c] = T.float32(0)
                                    depth_conv2d_nhwc_local[v_n, v_h, v_w, v_c] = depth_conv2d_nhwc_local[v_n, v_h, v_w, v_c] + PadInput_shared[v_n, v_h + v_rh, v_w + v_rw, v_c] * placeholder_shared[0, v_rh, v_rw, v_c]
                        for ax0, ax1, ax2, ax3 in T.grid(1, 28, 16, 8):
                            with T.block("depth_conv2d_nhwc_local"):
                                v0 = T.axis.spatial(1, ax0)
                                v1 = T.axis.spatial(112, n_1_h_1_w_1_c_1_fused // 2 * 28 + ax1)
                                v2 = T.axis.spatial(112, n_2_h_2_w_2_c_2_fused // 2 * 16 + ax2)
                                v3 = T.axis.spatial(32, n_1_h_1_w_1_c_1_fused % 2 * 16 + n_2_h_2_w_2_c_2_fused % 2 * 8 + ax3)
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
    actual = _design_space(mod)
    check_sketches(
        mod,
        sketches=actual,
        expected_mods=[dep_0],
        expected_decisions=[decision_0],
    )


def test_cuda_dil():
    # fmt: off
    @T.prim_func
    def dil_0(inputs: T.Buffer((1, 224, 224, 3), "float32"), weight: T.Buffer((7, 7, 3, 64), "float32"), conv2d_nhwc: T.Buffer((1, 109, 109, 64), "float32")) -> None:
        T.func_attr({"global_symbol": "main", "tir.noalias": T.bool(True)})
        with T.block("root"):
            T.reads()
            T.writes()
            T.block_attr({"meta_schedule.unroll_explicit": 512})
            conv2d_nhwc_local = T.alloc_buffer((1, 109, 109, 64), scope="local")
            PadInput_shared = T.alloc_buffer((1, 230, 230, 3), scope="shared")
            weight_shared = T.alloc_buffer((7, 7, 3, 64), scope="shared")
            for n_0_h_0_w_0_co_0_fused in T.thread_binding(218, thread="blockIdx.x"):
                for n_1_h_1_w_1_co_1_fused in T.thread_binding(109, thread="vthread.x"):
                    for n_2_h_2_w_2_co_2_fused in T.thread_binding(1, thread="threadIdx.x"):
                        for rh_0, rw_0, rc_0 in T.grid(7, 7, 3):
                            for ax0_ax1_ax2_ax3_fused in range(217):
                                with T.block("PadInput_shared"):
                                    v0 = T.axis.spatial(1, 0)
                                    v1 = T.axis.spatial(230, T.Add(n_0_h_0_w_0_co_0_fused // 2 * 2 + rh_0 * 2, 0))
                                    v2 = T.axis.spatial(230, rw_0 * 2 + ax0_ax1_ax2_ax3_fused % 217)
                                    v3 = T.axis.spatial(3, T.Add(rc_0, 0))
                                    T.reads(inputs[v0, v1 - 3, v2 - 3, v3])
                                    T.writes(PadInput_shared[v0, v1, v2, v3])
                                    T.block_attr({"meta_schedule.cooperative_fetch": 2})
                                    PadInput_shared[v0, v1, v2, v3] = T.if_then_else(3 <= v1 and v1 < 227 and 3 <= v2 and v2 < 227, inputs[v0, v1 - 3, v2 - 3, v3], T.float32(0))
                            for ax0_ax1_ax2_ax3_fused in range(32):
                                with T.block("weight_shared"):
                                    v0, v1, v2 = T.axis.remap("SSS", [rh_0, rw_0, rc_0])
                                    v3 = T.axis.spatial(64, n_0_h_0_w_0_co_0_fused % 2 * 32 + ax0_ax1_ax2_ax3_fused)
                                    T.reads(weight[v0, v1, v2, v3])
                                    T.writes(weight_shared[v0, v1, v2, v3])
                                    T.block_attr({"meta_schedule.cooperative_fetch": 4})
                                    weight_shared[v0, v1, v2, v3] = weight[v0, v1, v2, v3]
                            for rh_1, rw_1, rc_1, n_3, h_3, w_3, co_3, rh_2, rw_2, rc_2, n_4, h_4, w_4, co_4 in T.grid(1, 1, 1, 1, 1, 1, 8, 1, 1, 1, 1, 1, 1, 4):
                                with T.block("conv2d_nhwc"):
                                    v_n = T.axis.spatial(1, n_3 + n_4)
                                    v_h = T.axis.spatial(109, n_0_h_0_w_0_co_0_fused // 2 + h_3 + h_4)
                                    v_w = T.axis.spatial(109, n_1_h_1_w_1_co_1_fused + w_3 + w_4)
                                    v_co = T.axis.spatial(64, n_0_h_0_w_0_co_0_fused % 2 * 32 + co_3 * 4 + co_4)
                                    v_rh = T.axis.reduce(7, rh_0 + rh_1 + rh_2)
                                    v_rw = T.axis.reduce(7, rw_0 + rw_1 + rw_2)
                                    v_rc = T.axis.reduce(3, rc_0 + rc_1 + rc_2)
                                    T.reads(PadInput_shared[v_n, v_h * 2 + v_rh * 2, v_w * 2 + v_rw * 2, v_co // 64 * 3 + v_rc], weight_shared[v_rh, v_rw, v_rc, v_co])
                                    T.writes(conv2d_nhwc_local[v_n, v_h, v_w, v_co])
                                    T.block_attr({"meta_schedule.thread_extent_high_inclusive": 1024, "meta_schedule.thread_extent_low_inclusive": 32, "meta_schedule.tiling_structure": "SSSRRSRS"})
                                    with T.init():
                                        conv2d_nhwc_local[v_n, v_h, v_w, v_co] = T.float32(0)
                                    conv2d_nhwc_local[v_n, v_h, v_w, v_co] = conv2d_nhwc_local[v_n, v_h, v_w, v_co] + PadInput_shared[v_n, v_h * 2 + v_rh * 2, v_w * 2 + v_rw * 2, v_co // 64 * 3 + v_rc] * weight_shared[v_rh, v_rw, v_rc, v_co]
                        for ax0, ax1, ax2, ax3 in T.grid(1, 1, 1, 32):
                            with T.block("conv2d_nhwc_local"):
                                v0 = T.axis.spatial(1, ax0)
                                v1 = T.axis.spatial(109, n_0_h_0_w_0_co_0_fused // 2 + ax1)
                                v2 = T.axis.spatial(109, n_1_h_1_w_1_co_1_fused + ax2)
                                v3 = T.axis.spatial(64, n_0_h_0_w_0_co_0_fused % 2 * 32 + ax3)
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
    actual = _design_space(mod)
    check_sketches(
        mod,
        sketches=actual,
        expected_mods=[dil_0],
        expected_decisions=[decision_0],
    )


def test_cuda_gmm():
    # fmt: off
    @T.prim_func
    def gmm_0(X: T.Buffer((1, 128, 128), "float32"), Y: T.Buffer((1, 128, 128), "float32"), Z: T.Buffer((1, 128, 128), "float32")) -> None:
        T.func_attr({"global_symbol": "main", "tir.noalias": T.bool(True)})
        with T.block("root"):
            T.reads()
            T.writes()
            T.block_attr({"meta_schedule.unroll_explicit": 1024})
            Z_local = T.alloc_buffer((1, 128, 128), scope="local")
            X_shared = T.alloc_buffer((1, 128, 128), scope="shared")
            Y_shared = T.alloc_buffer((1, 128, 128), scope="shared")
            for b_0_i_0_j_0_fused in T.thread_binding(1, thread="blockIdx.x"):
                for b_1_i_1_j_1_fused in T.thread_binding(32, thread="vthread.x"):
                    for b_2_i_2_j_2_fused in T.thread_binding(2, thread="threadIdx.x"):
                        for k_0 in range(1):
                            for ax0_ax1_ax2_fused in range(16384):
                                with T.block("X_shared"):
                                    v0 = T.axis.spatial(1, 0)
                                    v1 = T.axis.spatial(128, ax0_ax1_ax2_fused // 128)
                                    v2 = T.axis.spatial(128, ax0_ax1_ax2_fused % 128)
                                    T.reads(X[v0, v1, v2])
                                    T.writes(X_shared[v0, v1, v2])
                                    T.block_attr({"meta_schedule.cooperative_fetch": 2})
                                    X_shared[v0, v1, v2] = X[v0, v1, v2]
                            for ax0_ax1_ax2_fused in range(16384):
                                with T.block("Y_shared"):
                                    v0 = T.axis.spatial(1, 0)
                                    v1 = T.axis.spatial(128, ax0_ax1_ax2_fused // 128)
                                    v2 = T.axis.spatial(128, ax0_ax1_ax2_fused % 128)
                                    T.reads(Y[v0, v1, v2])
                                    T.writes(Y_shared[v0, v1, v2])
                                    T.block_attr({"meta_schedule.cooperative_fetch": 1})
                                    Y_shared[v0, v1, v2] = Y[v0, v1, v2]
                            for k_1, b_3, i_3, j_3, k_2, b_4, i_4, j_4 in T.grid(32, 1, 2, 64, 4, 1, 2, 1):
                                with T.block("Z"):
                                    v_b = T.axis.spatial(1, b_3 + b_4)
                                    v_i = T.axis.spatial(128, b_1_i_1_j_1_fused * 4 + i_3 * 2 + i_4)
                                    v_j = T.axis.spatial(128, b_2_i_2_j_2_fused * 64 + j_3 + j_4)
                                    v_k = T.axis.reduce(128, k_0 * 128 + k_1 * 4 + k_2)
                                    T.reads(X_shared[v_b, v_i, v_k], Y_shared[v_b, v_k, v_j])
                                    T.writes(Z_local[v_b, v_i, v_j])
                                    T.block_attr({"meta_schedule.thread_extent_high_inclusive": 1024, "meta_schedule.thread_extent_low_inclusive": 32, "meta_schedule.tiling_structure": "SSSRRSRS"})
                                    with T.init():
                                        Z_local[v_b, v_i, v_j] = T.float32(0)
                                    Z_local[v_b, v_i, v_j] = Z_local[v_b, v_i, v_j] + X_shared[v_b, v_i, v_k] * Y_shared[v_b, v_k, v_j]
                        for ax0, ax1, ax2 in T.grid(1, 4, 64):
                            with T.block("Z_local"):
                                v0 = T.axis.spatial(1, ax0)
                                v1 = T.axis.spatial(128, b_1_i_1_j_1_fused * 4 + ax1)
                                v2 = T.axis.spatial(128, b_2_i_2_j_2_fused * 64 + ax2)
                                T.reads(Z_local[v0, v1, v2])
                                T.writes(Z[v0, v1, v2])
                                Z[v0, v1, v2] = Z_local[v0, v1, v2]
    # fmt: on
    decision_0 = [
        ("SamplePerfectTile", [1, 1, 1, 1, 1]),
        ("SamplePerfectTile", [1, 32, 1, 2, 2]),
        ("SamplePerfectTile", [1, 1, 2, 64, 1]),
        ("SamplePerfectTile", [1, 32, 4]),
        ("SampleCategorical", 1),
        ("SampleCategorical", 0),
        ("SampleCategorical", 4),
    ]
    mod = create_te_workload("GMM", 0)
    actual = _design_space(mod)
    check_sketches(
        mod,
        sketches=actual,
        expected_mods=[gmm_0],
        expected_decisions=[decision_0],
    )


def test_cuda_grp():
    # fmt: off
    @T.prim_func
    def grp_0(inputs: T.Buffer((1, 56, 56, 64), "float32"), weight: T.Buffer((3, 3, 16, 128), "float32"), conv2d_nhwc: T.Buffer((1, 28, 28, 128), "float32")) -> None:
        T.func_attr({"global_symbol": "main", "tir.noalias": T.bool(True)})
        with T.block("root"):
            T.reads()
            T.writes()
            T.block_attr({"meta_schedule.unroll_explicit": 16})
            conv2d_nhwc_local = T.alloc_buffer((1, 28, 28, 128), scope="local")
            PadInput_shared = T.alloc_buffer((1, 58, 58, 64), scope="shared")
            weight_shared = T.alloc_buffer((3, 3, 16, 128), scope="shared")
            for n_0_h_0_w_0_co_0_fused in T.thread_binding(2, thread="blockIdx.x"):
                for n_1_h_1_w_1_co_1_fused in T.thread_binding(1, thread="vthread.x"):
                    for n_2_h_2_w_2_co_2_fused in T.thread_binding(112, thread="threadIdx.x"):
                        for rh_0, rw_0, rc_0 in T.grid(3, 3, 1):
                            for ax0_ax1_ax2_ax3_fused in range(95040):
                                with T.block("PadInput_shared"):
                                    v0 = T.axis.spatial(1, 0)
                                    v1 = T.axis.spatial(58, n_0_h_0_w_0_co_0_fused * 28 + rh_0 + ax0_ax1_ax2_ax3_fused % 95040 // 3520)
                                    v2 = T.axis.spatial(58, rw_0 + ax0_ax1_ax2_ax3_fused % 3520 // 64)
                                    v3 = T.axis.spatial(64, ax0_ax1_ax2_ax3_fused % 64)
                                    T.reads(inputs[v0, v1 - 1, v2 - 1, v3])
                                    T.writes(PadInput_shared[v0, v1, v2, v3])
                                    T.block_attr({"meta_schedule.cooperative_fetch": 2})
                                    PadInput_shared[v0, v1, v2, v3] = T.if_then_else(1 <= v1 and v1 < 57 and 1 <= v2 and v2 < 57, inputs[v0, v1 - 1, v2 - 1, v3], T.float32(0))
                            for ax0_ax1_ax2_ax3_fused in range(2048):
                                with T.block("weight_shared"):
                                    v0, v1 = T.axis.remap("SS", [rh_0, rw_0])
                                    v2 = T.axis.spatial(16, ax0_ax1_ax2_ax3_fused // 128)
                                    v3 = T.axis.spatial(128, ax0_ax1_ax2_ax3_fused % 128)
                                    T.reads(weight[v0, v1, v2, v3])
                                    T.writes(weight_shared[v0, v1, v2, v3])
                                    T.block_attr({"meta_schedule.cooperative_fetch": 1})
                                    weight_shared[v0, v1, v2, v3] = weight[v0, v1, v2, v3]
                            for rh_1, rw_1, rc_1, n_3, h_3, w_3, co_3, rh_2, rw_2, rc_2, n_4, h_4, w_4, co_4 in T.grid(1, 1, 2, 1, 2, 1, 2, 1, 1, 8, 1, 7, 4, 4):
                                with T.block("conv2d_nhwc"):
                                    v_n = T.axis.spatial(1, n_3 + n_4)
                                    v_h = T.axis.spatial(28, n_0_h_0_w_0_co_0_fused * 14 + h_3 * 7 + h_4)
                                    v_w = T.axis.spatial(28, n_2_h_2_w_2_co_2_fused // 16 * 4 + w_3 * 4 + w_4)
                                    v_co = T.axis.spatial(128, n_2_h_2_w_2_co_2_fused % 16 * 8 + co_3 * 4 + co_4)
                                    v_rh = T.axis.reduce(3, rh_0 + rh_1 + rh_2)
                                    v_rw = T.axis.reduce(3, rw_0 + rw_1 + rw_2)
                                    v_rc = T.axis.reduce(16, rc_0 * 16 + rc_1 * 8 + rc_2)
                                    T.reads(PadInput_shared[v_n, v_h * 2 + v_rh, v_w * 2 + v_rw, v_co // 32 * 16 + v_rc], weight_shared[v_rh, v_rw, v_rc, v_co])
                                    T.writes(conv2d_nhwc_local[v_n, v_h, v_w, v_co])
                                    T.block_attr({"meta_schedule.thread_extent_high_inclusive": 1024, "meta_schedule.thread_extent_low_inclusive": 32, "meta_schedule.tiling_structure": "SSSRRSRS"})
                                    with T.init():
                                        conv2d_nhwc_local[v_n, v_h, v_w, v_co] = T.float32(0)
                                    conv2d_nhwc_local[v_n, v_h, v_w, v_co] = conv2d_nhwc_local[v_n, v_h, v_w, v_co] + PadInput_shared[v_n, v_h * 2 + v_rh, v_w * 2 + v_rw, v_co // 32 * 16 + v_rc] * weight_shared[v_rh, v_rw, v_rc, v_co]
                        for ax0, ax1, ax2, ax3 in T.grid(1, 14, 4, 8):
                            with T.block("conv2d_nhwc_local"):
                                v0 = T.axis.spatial(1, ax0)
                                v1 = T.axis.spatial(28, n_0_h_0_w_0_co_0_fused * 14 + ax1)
                                v2 = T.axis.spatial(28, n_2_h_2_w_2_co_2_fused // 16 * 4 + ax2)
                                v3 = T.axis.spatial(128, n_2_h_2_w_2_co_2_fused % 16 * 8 + ax3)
                                T.reads(conv2d_nhwc_local[v0, v1, v2, v3])
                                T.writes(conv2d_nhwc[v0, v1, v2, v3])
                                conv2d_nhwc[v0, v1, v2, v3] = conv2d_nhwc_local[v0, v1, v2, v3]
    # fmt: on
    decision_0 = [
        ("SamplePerfectTile", [1, 1, 1, 1, 1]),
        ("SamplePerfectTile", [2, 1, 1, 2, 7]),
        ("SamplePerfectTile", [1, 1, 7, 1, 4]),
        ("SamplePerfectTile", [1, 1, 16, 2, 4]),
        ("SamplePerfectTile", [3, 1, 1]),
        ("SamplePerfectTile", [3, 1, 1]),
        ("SamplePerfectTile", [1, 2, 8]),
        ("SampleCategorical", 1),
        ("SampleCategorical", 0),
        ("SampleCategorical", 1),
    ]
    mod = create_te_workload("GRP", 0)
    actual = _design_space(mod)
    check_sketches(
        mod,
        sketches=actual,
        expected_mods=[grp_0],
        expected_decisions=[decision_0],
    )


def test_cuda_t2d():
    # fmt: off
    @T.prim_func
    def t2d_0(inputs: T.Buffer((1, 4, 4, 512), "float32"), weight: T.Buffer((4, 4, 512, 256), "float32"), conv2d_transpose_nhwc: T.Buffer((1, 8, 8, 256), "float32")) -> None:
        T.func_attr({"global_symbol": "main", "tir.noalias": T.bool(True)})
        with T.block("root"):
            T.reads()
            T.writes()
            T.block_attr({"meta_schedule.unroll_explicit": 64})
            conv2d_transpose_nhwc_local = T.alloc_buffer((1, 8, 8, 256), scope="local")
            PadInput_shared = T.alloc_buffer((1, 6, 6, 512), scope="shared")
            weight_shared = T.alloc_buffer((4, 4, 512, 256), scope="shared")
            for n_0_h_0_w_0_co_0_fused in T.thread_binding(256, thread="blockIdx.x"):
                for n_1_h_1_w_1_co_1_fused in T.thread_binding(2, thread="vthread.x"):
                    for n_2_h_2_w_2_co_2_fused in T.thread_binding(1, thread="threadIdx.x"):
                        for rh_0, rw_0, rc_0 in T.grid(4, 1, 16):
                            for ax0_ax1_ax2_ax3_fused in range(rh_0 % 2 * 96 + 96):
                                with T.block("PadInput_shared"):
                                    v0 = T.axis.spatial(1, 0)
                                    v1 = T.axis.spatial(6, n_0_h_0_w_0_co_0_fused // 64 + rh_0 // 2 + ax0_ax1_ax2_ax3_fused % (96 * (rh_0 % 2 + 1)) // 96)
                                    v2 = T.axis.spatial(6, n_0_h_0_w_0_co_0_fused % 64 // 16 + ax0_ax1_ax2_ax3_fused % 96 // 32)
                                    v3 = T.axis.spatial(512, rc_0 * 32 + ax0_ax1_ax2_ax3_fused % 32)
                                    T.reads(inputs[v0, v1 - 1, v2 - 1, v3])
                                    T.writes(PadInput_shared[v0, v1, v2, v3])
                                    T.block_attr({"meta_schedule.cooperative_fetch": 2})
                                    PadInput_shared[v0, v1, v2, v3] = T.if_then_else(1 <= v1 and v1 < 5 and 1 <= v2 and v2 < 5, inputs[v0, v1 - 1, v2 - 1, v3], T.float32(0))
                            for ax0_ax1_ax2_ax3_fused in range(2048):
                                with T.block("weight_shared"):
                                    v0 = T.axis.spatial(4, rh_0 * -1 + 3)
                                    v1 = T.axis.spatial(4, ax0_ax1_ax2_ax3_fused // 512)
                                    v2 = T.axis.spatial(512, rc_0 * 32 + ax0_ax1_ax2_ax3_fused % 512 // 16)
                                    v3 = T.axis.spatial(256, n_0_h_0_w_0_co_0_fused % 16 * 16 + ax0_ax1_ax2_ax3_fused % 16)
                                    T.reads(weight[v0, v1, v2, v3])
                                    T.writes(weight_shared[v0, v1, v2, v3])
                                    T.block_attr({"meta_schedule.cooperative_fetch": 4})
                                    weight_shared[v0, v1, v2, v3] = weight[v0, v1, v2, v3]
                            for rh_1, rw_1, rc_1, n_3, h_3, w_3, co_3, rh_2, rw_2, rc_2, n_4, h_4, w_4, co_4 in T.grid(1, 1, 4, 1, 2, 1, 8, 1, 4, 8, 1, 1, 2, 1):
                                with T.block("conv2d_transpose_nhwc"):
                                    v_n = T.axis.spatial(1, n_3 + n_4)
                                    v_h = T.axis.spatial(8, n_0_h_0_w_0_co_0_fused // 64 * 2 + h_3 + h_4)
                                    v_w = T.axis.spatial(8, n_0_h_0_w_0_co_0_fused % 64 // 16 * 2 + w_3 * 2 + w_4)
                                    v_co = T.axis.spatial(256, n_0_h_0_w_0_co_0_fused % 16 * 16 + n_1_h_1_w_1_co_1_fused * 8 + co_3 + co_4)
                                    v_rh = T.axis.reduce(4, rh_0 + rh_1 + rh_2)
                                    v_rw = T.axis.reduce(4, rw_0 * 4 + rw_1 * 4 + rw_2)
                                    v_rc = T.axis.reduce(512, rc_0 * 32 + rc_1 * 8 + rc_2)
                                    T.reads(PadInput_shared[v_n, (v_h + v_rh) // 2, (v_w + v_rw) // 2, v_rc], weight_shared[3 - v_rh, 3 - v_rw, v_rc, v_co])
                                    T.writes(conv2d_transpose_nhwc_local[v_n, v_h, v_w, v_co])
                                    T.block_attr({"meta_schedule.thread_extent_high_inclusive": 1024, "meta_schedule.thread_extent_low_inclusive": 32, "meta_schedule.tiling_structure": "SSSRRSRS"})
                                    with T.init():
                                        conv2d_transpose_nhwc_local[v_n, v_h, v_w, v_co] = T.float32(0)
                                    conv2d_transpose_nhwc_local[v_n, v_h, v_w, v_co] = conv2d_transpose_nhwc_local[v_n, v_h, v_w, v_co] + T.if_then_else((v_h + v_rh) % 2 == 0 and (v_w + v_rw) % 2 == 0, PadInput_shared[v_n, (v_h + v_rh) // 2, (v_w + v_rw) // 2, v_rc], T.float32(0)) * weight_shared[3 - v_rh, 3 - v_rw, v_rc, v_co]
                        for ax0, ax1, ax2, ax3 in T.grid(1, 2, 2, 8):
                            with T.block("conv2d_transpose_nhwc_local"):
                                v0 = T.axis.spatial(1, ax0)
                                v1 = T.axis.spatial(8, n_0_h_0_w_0_co_0_fused // 64 * 2 + ax1)
                                v2 = T.axis.spatial(8, n_0_h_0_w_0_co_0_fused % 64 // 16 * 2 + ax2)
                                v3 = T.axis.spatial(256, n_0_h_0_w_0_co_0_fused % 16 * 16 + n_1_h_1_w_1_co_1_fused * 8 + ax3)
                                T.reads(conv2d_transpose_nhwc_local[v0, v1, v2, v3])
                                T.writes(conv2d_transpose_nhwc[v0, v1, v2, v3])
                                conv2d_transpose_nhwc[v0, v1, v2, v3] = conv2d_transpose_nhwc_local[v0, v1, v2, v3]
    # fmt: on
    decision_0 = [
        ("SamplePerfectTile", [1, 1, 1, 1, 1]),
        ("SamplePerfectTile", [4, 1, 1, 2, 1]),
        ("SamplePerfectTile", [4, 1, 1, 1, 2]),
        ("SamplePerfectTile", [16, 2, 1, 8, 1]),
        ("SamplePerfectTile", [4, 1, 1]),
        ("SamplePerfectTile", [1, 1, 4]),
        ("SamplePerfectTile", [16, 4, 8]),
        ("SampleCategorical", 1),
        ("SampleCategorical", 3),
        ("SampleCategorical", 2),
    ]
    mod = create_te_workload("T2D", 0)
    actual = _design_space(mod)
    check_sketches(
        mod,
        sketches=actual,
        expected_mods=[t2d_0],
        expected_decisions=[decision_0],
        debug_mask=0,
    )


def test_cuda_nrm():
    # fmt: off
    @T.prim_func
    def nrm_0(A: T.Buffer((1, 256, 256), "float32"), D: T.Buffer(1, "float32")) -> None:
        T.func_attr({"global_symbol": "main", "tir.noalias": T.bool(True)})
        with T.block("root"):
            T.reads()
            T.writes()
            T.block_attr({"meta_schedule.unroll_explicit": 512})
            C = T.alloc_buffer((1,))
            for b_fused_0 in T.thread_binding(1, thread="blockIdx.x"):
                for b_fused_1 in T.thread_binding(1, thread="threadIdx.x"):
                    for i, j in T.grid(256, 256):
                        with T.block("C"):
                            v_b = T.axis.spatial(1, 0)
                            v_i, v_j = T.axis.remap("RR", [i, j])
                            T.reads(A[v_b, v_i, v_j])
                            T.writes(C[v_b])
                            with T.init():
                                C[v_b] = T.float32(0)
                            C[v_b] = C[v_b] + A[v_b, v_i, v_j] * A[v_b, v_i, v_j]
            for b_fused_0 in T.thread_binding(1, thread="blockIdx.x"):
                for b_fused_1 in T.thread_binding(1, thread="threadIdx.x"):
                    with T.block("D"):
                        v_b = T.axis.spatial(1, 0)
                        T.reads(C[v_b])
                        T.writes(D[v_b])
                        D[v_b] = T.sqrt(C[v_b])
    @T.prim_func
    def nrm_1(A: T.Buffer((1, 256, 256), "float32"), D: T.Buffer(1, "float32")) -> None:
        T.func_attr({"global_symbol": "main", "tir.noalias": T.bool(True)})
        with T.block("root"):
            T.reads()
            T.writes()
            T.block_attr({"meta_schedule.unroll_explicit": 1024})
            C_shared = T.alloc_buffer((1,), scope="shared")
            for b_0_fused in T.thread_binding(1, thread="blockIdx.x"):
                for ax0, ax1_ax2_fused_0 in T.grid(1, 512):
                    for ax1_ax2_fused_1 in T.thread_binding(128, thread="threadIdx.x"):
                        with T.block("C"):
                            v_b = T.axis.spatial(1, ax0)
                            v_i = T.axis.reduce(256, (ax1_ax2_fused_0 * 128 + ax1_ax2_fused_1) // 256)
                            v_j = T.axis.reduce(256, (ax1_ax2_fused_0 * 128 + ax1_ax2_fused_1) % 256)
                            T.reads(A[v_b, v_i, v_j])
                            T.writes(C_shared[v_b])
                            with T.init():
                                C_shared[v_b] = T.float32(0)
                            C_shared[v_b] = C_shared[v_b] + A[v_b, v_i, v_j] * A[v_b, v_i, v_j]
                for b_1 in T.thread_binding(128, thread="threadIdx.x"):
                    with T.block("D"):
                        v_b = T.axis.spatial(1, b_1)
                        T.where(T.Mul(0, 128) + b_1 < 1)
                        T.reads(C_shared[v_b])
                        T.writes(D[v_b])
                        D[v_b] = T.sqrt(C_shared[v_b])
    # fmt: on
    decision_0 = [
        ("SampleCategorical", 3),
    ]
    decision_1 = [
        ("SampleCategorical", 5),
        ("SampleCategorical", 4),
    ]
    mod = create_te_workload("NRM", 0)
    actual = _design_space(mod)
    check_sketches(
        mod,
        sketches=actual,
        expected_mods=[nrm_0, nrm_1],
        expected_decisions=[decision_0, decision_1],
    )


def test_cuda_sfm():
    # fmt: off
    @T.prim_func
    def sfm_0(A: T.Buffer((256, 256), "float32"), T_softmax_norm: T.Buffer((256, 256), "float32")) -> None:
        T.func_attr({"global_symbol": "main", "tir.noalias": T.bool(True)})
        with T.block("root"):
            T.reads()
            T.writes()
            T.block_attr({"meta_schedule.unroll_explicit": 0})
            T_softmax_maxelem = T.alloc_buffer((256,))
            T_softmax_expsum = T.alloc_buffer((256,))
            for i0_fused_0 in T.thread_binding(2, thread="blockIdx.x"):
                for i0_fused_1 in T.thread_binding(128, thread="threadIdx.x"):
                    for k in range(256):
                        with T.block("T_softmax_maxelem"):
                            v_i0 = T.axis.spatial(256, i0_fused_0 * 128 + i0_fused_1)
                            v_k = T.axis.reduce(256, k)
                            T.reads(A[v_i0, v_k])
                            T.writes(T_softmax_maxelem[v_i0])
                            with T.init():
                                T_softmax_maxelem[v_i0] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem[v_i0] = T.max(T_softmax_maxelem[v_i0], A[v_i0, v_k])
            for i0_fused_0 in T.thread_binding(1, thread="blockIdx.x"):
                for i0_fused_1 in T.thread_binding(256, thread="threadIdx.x"):
                    for k in range(256):
                        with T.block("T_softmax_expsum"):
                            v_i0 = T.axis.spatial(256, i0_fused_0 * 256 + i0_fused_1)
                            v_k = T.axis.reduce(256, k)
                            T.reads(A[v_i0, v_k], T_softmax_maxelem[v_i0])
                            T.writes(T_softmax_expsum[v_i0])
                            with T.init():
                                T_softmax_expsum[v_i0] = T.float32(0)
                            T_softmax_expsum[v_i0] = T_softmax_expsum[v_i0] + T.exp(A[v_i0, v_k] - T_softmax_maxelem[v_i0])
            for i0_i1_fused_0 in T.thread_binding(1024, thread="blockIdx.x"):
                for i0_i1_fused_1 in T.thread_binding(64, thread="threadIdx.x"):
                    with T.block("T_softmax_norm"):
                        v_i0 = T.axis.spatial(256, (i0_i1_fused_0 * 64 + i0_i1_fused_1) // 256)
                        v_i1 = T.axis.spatial(256, (i0_i1_fused_0 * 64 + i0_i1_fused_1) % 256)
                        T.reads(A[v_i0, v_i1], T_softmax_maxelem[v_i0], T_softmax_expsum[v_i0])
                        T.writes(T_softmax_norm[v_i0, v_i1])
                        T.block_attr({"axis": 1})
                        T_softmax_norm[v_i0, v_i1] = T.exp(A[v_i0, v_i1] - T_softmax_maxelem[v_i0]) / T_softmax_expsum[v_i0]
    @T.prim_func
    def sfm_1(A: T.Buffer((256, 256), "float32"), T_softmax_norm: T.Buffer((256, 256), "float32")) -> None:
        T.func_attr({"global_symbol": "main", "tir.noalias": T.bool(True)})
        with T.block("root"):
            T.reads()
            T.writes()
            T.block_attr({"meta_schedule.unroll_explicit": 16})
            T_softmax_maxelem = T.alloc_buffer((256,))
            T_softmax_expsum = T.alloc_buffer((256,))
            for i0_fused in T.thread_binding(256, thread="blockIdx.x"):
                for k_0 in range(64):
                    for k_1 in T.thread_binding(4, thread="threadIdx.x"):
                        with T.block("T_softmax_maxelem"):
                            v_i0 = T.axis.spatial(256, i0_fused)
                            v_k = T.axis.reduce(256, k_0 * 4 + k_1)
                            T.reads(A[v_i0, v_k])
                            T.writes(T_softmax_maxelem[v_i0])
                            with T.init():
                                T_softmax_maxelem[v_i0] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem[v_i0] = T.max(T_softmax_maxelem[v_i0], A[v_i0, v_k])
            for i0_fused_0 in T.thread_binding(4, thread="blockIdx.x"):
                for i0_fused_1 in T.thread_binding(64, thread="threadIdx.x"):
                    for k in range(256):
                        with T.block("T_softmax_expsum"):
                            v_i0 = T.axis.spatial(256, i0_fused_0 * 64 + i0_fused_1)
                            v_k = T.axis.reduce(256, k)
                            T.reads(A[v_i0, v_k], T_softmax_maxelem[v_i0])
                            T.writes(T_softmax_expsum[v_i0])
                            with T.init():
                                T_softmax_expsum[v_i0] = T.float32(0)
                            T_softmax_expsum[v_i0] = T_softmax_expsum[v_i0] + T.exp(A[v_i0, v_k] - T_softmax_maxelem[v_i0])
            for i0_i1_fused_0 in T.thread_binding(256, thread="blockIdx.x"):
                for i0_i1_fused_1 in T.thread_binding(256, thread="threadIdx.x"):
                    with T.block("T_softmax_norm"):
                        v_i0 = T.axis.spatial(256, (i0_i1_fused_0 * 256 + i0_i1_fused_1) // 256)
                        v_i1 = T.axis.spatial(256, (i0_i1_fused_0 * 256 + i0_i1_fused_1) % 256)
                        T.reads(A[v_i0, v_i1], T_softmax_maxelem[v_i0], T_softmax_expsum[v_i0])
                        T.writes(T_softmax_norm[v_i0, v_i1])
                        T.block_attr({"axis": 1})
                        T_softmax_norm[v_i0, v_i1] = T.exp(A[v_i0, v_i1] - T_softmax_maxelem[v_i0]) / T_softmax_expsum[v_i0]
    @T.prim_func
    def sfm_2(A: T.Buffer((256, 256), "float32"), T_softmax_norm: T.Buffer((256, 256), "float32")) -> None:
        T.func_attr({"global_symbol": "main", "tir.noalias": T.bool(True)})
        with T.block("root"):
            T.reads()
            T.writes()
            T.block_attr({"meta_schedule.unroll_explicit": 512})
            T_softmax_maxelem = T.alloc_buffer((256,))
            T_softmax_expsum_shared = T.alloc_buffer((256,), scope="shared")
            for i0_fused_0 in T.thread_binding(8, thread="blockIdx.x"):
                for i0_fused_1 in T.thread_binding(32, thread="threadIdx.x"):
                    for k in range(256):
                        with T.block("T_softmax_maxelem"):
                            v_i0 = T.axis.spatial(256, i0_fused_0 * 32 + i0_fused_1)
                            v_k = T.axis.reduce(256, k)
                            T.reads(A[v_i0, v_k])
                            T.writes(T_softmax_maxelem[v_i0])
                            with T.init():
                                T_softmax_maxelem[v_i0] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem[v_i0] = T.max(T_softmax_maxelem[v_i0], A[v_i0, v_k])
            for i0_fused in T.thread_binding(256, thread="blockIdx.x"):
                for ax0, ax1_0 in T.grid(1, 1):
                    for ax1_1 in T.thread_binding(512, thread="threadIdx.x"):
                        with T.block("T_softmax_expsum"):
                            v_i0 = T.axis.spatial(256, i0_fused + ax0)
                            v_k = T.axis.reduce(256, ax1_0 * 512 + ax1_1)
                            T.where(ax1_0 * 512 + ax1_1 < 256)
                            T.reads(A[v_i0, v_k], T_softmax_maxelem[v_i0])
                            T.writes(T_softmax_expsum_shared[v_i0])
                            with T.init():
                                T_softmax_expsum_shared[v_i0] = T.float32(0)
                            T_softmax_expsum_shared[v_i0] = T_softmax_expsum_shared[v_i0] + T.exp(A[v_i0, v_k] - T_softmax_maxelem[v_i0])
                for i1_0 in range(1):
                    for i1_1 in T.thread_binding(512, thread="threadIdx.x"):
                        with T.block("T_softmax_norm"):
                            v_i0 = T.axis.spatial(256, i0_fused)
                            v_i1 = T.axis.spatial(256, i1_0 * 512 + i1_1)
                            T.where(i1_0 * 512 + i1_1 < 256)
                            T.reads(A[v_i0, v_i1], T_softmax_maxelem[v_i0], T_softmax_expsum_shared[v_i0])
                            T.writes(T_softmax_norm[v_i0, v_i1])
                            T.block_attr({"axis": 1})
                            T_softmax_norm[v_i0, v_i1] = T.exp(A[v_i0, v_i1] - T_softmax_maxelem[v_i0]) / T_softmax_expsum_shared[v_i0]
    @T.prim_func
    def sfm_3(A: T.Buffer((256, 256), "float32"), T_softmax_norm: T.Buffer((256, 256), "float32")) -> None:
        T.func_attr({"global_symbol": "main", "tir.noalias": T.bool(True)})
        with T.block("root"):
            T.reads()
            T.writes()
            T.block_attr({"meta_schedule.unroll_explicit": 0})
            T_softmax_maxelem_shared = T.alloc_buffer((256,), scope="shared")
            T_softmax_expsum_shared = T.alloc_buffer((256,), scope="shared")
            for i0_fused in T.thread_binding(256, thread="blockIdx.x"):
                for ax0, ax1_0 in T.grid(1, 1):
                    for ax1_1 in T.thread_binding(512, thread="threadIdx.x"):
                        with T.block("T_softmax_maxelem"):
                            v_i0 = T.axis.spatial(256, i0_fused + ax0)
                            v_k = T.axis.reduce(256, ax1_0 * 512 + ax1_1)
                            T.where(ax1_0 * 512 + ax1_1 < 256)
                            T.reads(A[v_i0, v_k])
                            T.writes(T_softmax_maxelem_shared[v_i0])
                            with T.init():
                                T_softmax_maxelem_shared[v_i0] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[v_i0] = T.max(T_softmax_maxelem_shared[v_i0], A[v_i0, v_k])
                for ax0, ax1_0 in T.grid(1, 1):
                    for ax1_1 in T.thread_binding(512, thread="threadIdx.x"):
                        with T.block("T_softmax_expsum"):
                            v_i0 = T.axis.spatial(256, i0_fused + ax0)
                            v_k = T.axis.reduce(256, ax1_0 * 512 + ax1_1)
                            T.where(ax1_0 * 512 + ax1_1 < 256)
                            T.reads(A[v_i0, v_k], T_softmax_maxelem_shared[v_i0])
                            T.writes(T_softmax_expsum_shared[v_i0])
                            with T.init():
                                T_softmax_expsum_shared[v_i0] = T.float32(0)
                            T_softmax_expsum_shared[v_i0] = T_softmax_expsum_shared[v_i0] + T.exp(A[v_i0, v_k] - T_softmax_maxelem_shared[v_i0])
                for i1_0 in range(1):
                    for i1_1 in T.thread_binding(512, thread="threadIdx.x"):
                        with T.block("T_softmax_norm"):
                            v_i0 = T.axis.spatial(256, i0_fused)
                            v_i1 = T.axis.spatial(256, i1_0 * 512 + i1_1)
                            T.where(i1_0 * 512 + i1_1 < 256)
                            T.reads(A[v_i0, v_i1], T_softmax_maxelem_shared[v_i0], T_softmax_expsum_shared[v_i0])
                            T.writes(T_softmax_norm[v_i0, v_i1])
                            T.block_attr({"axis": 1})
                            T_softmax_norm[v_i0, v_i1] = T.exp(A[v_i0, v_i1] - T_softmax_maxelem_shared[v_i0]) / T_softmax_expsum_shared[v_i0]
    # fmt: on
    decision_0 = [
        ("SampleCategorical", 0),
        ("SampleCategorical", 1),
        ("SampleCategorical", 3),
        ("SampleCategorical", 2),
    ]
    decision_1 = [
        ("SampleCategorical", 0),
        ("SampleCategorical", 1),
        ("SampleCategorical", 3),
        ("SampleCategorical", 1),
    ]
    decision_2 = [
        ("SampleCategorical", 7),
        ("SampleCategorical", 3),
        ("SampleCategorical", 0),
    ]
    decision_3 = [
        ("SampleCategorical", 7),
        ("SampleCategorical", 0),
        ("SampleCategorical", 0),
    ]
    mod = create_te_workload("SFM", 0)
    actual = _design_space(mod)
    check_sketches(
        mod,
        sketches=actual,
        expected_mods=[sfm_0, sfm_1, sfm_2, sfm_3],
        expected_decisions=[decision_0, decision_1, decision_2, decision_3],
    )


def test_cuda_cbr():
    # fmt: off
    @T.prim_func
    def cbr_0(data: T.Buffer((1, 224, 224, 3), "float32"), kernel: T.Buffer((7, 7, 3, 64), "float32"), bias: T.Buffer(64, "float32"), bn_offset: T.Buffer(64, "float32"), bn_scale: T.Buffer(64, "float32"), compute: T.Buffer((1, 112, 112, 64), "float32")) -> None:
        T.func_attr({"global_symbol": "main", "tir.noalias": T.bool(True)})
        with T.block("root"):
            T.reads()
            T.writes()
            T.block_attr({"meta_schedule.unroll_explicit": 512})
            Conv2dOutput_local = T.alloc_buffer((1, 112, 112, 64), scope="local")
            PaddedInput_shared = T.alloc_buffer((1, 230, 230, 3), scope="shared")
            kernel_shared = T.alloc_buffer((7, 7, 3, 64), scope="shared")
            for nn_0_yy_0_xx_0_ff_0_fused in T.thread_binding(14, thread="blockIdx.x"):
                for nn_1_yy_1_xx_1_ff_1_fused in T.thread_binding(4, thread="vthread.x"):
                    for nn_2_yy_2_xx_2_ff_2_fused in T.thread_binding(128, thread="threadIdx.x"):
                        for ry_0, rx_0, rc_0 in T.grid(7, 1, 3):
                            for ax0_ax1_ax2_ax3_fused in range(8251):
                                with T.block("PaddedInput_shared"):
                                    v0 = T.axis.spatial(1, 0)
                                    v1 = T.axis.spatial(230, ry_0 + ax0_ax1_ax2_ax3_fused // 37)
                                    v2 = T.axis.spatial(230, nn_0_yy_0_xx_0_ff_0_fused // 2 * 32 + ax0_ax1_ax2_ax3_fused % 37)
                                    v3 = T.axis.spatial(3, rc_0)
                                    T.reads(data[v0, v1 - 3, v2 - 3, v3])
                                    T.writes(PaddedInput_shared[v0, v1, v2, v3])
                                    T.block_attr({"meta_schedule.cooperative_fetch": 1})
                                    PaddedInput_shared[v0, v1, v2, v3] = T.if_then_else(3 <= v1 and v1 < 227 and 3 <= v2 and v2 < 227, data[v0, v1 - 3, v2 - 3, v3], T.float32(0))
                            for ax0_ax1_ax2_ax3_fused in range(224):
                                with T.block("kernel_shared"):
                                    v0 = T.axis.spatial(7, ry_0)
                                    v1 = T.axis.spatial(7, ax0_ax1_ax2_ax3_fused // 32)
                                    v2 = T.axis.spatial(3, rc_0)
                                    v3 = T.axis.spatial(64, nn_0_yy_0_xx_0_ff_0_fused % 2 * 32 + ax0_ax1_ax2_ax3_fused % 32)
                                    T.reads(kernel[v0, v1, v2, v3])
                                    T.writes(kernel_shared[v0, v1, v2, v3])
                                    T.block_attr({"meta_schedule.cooperative_fetch": 1})
                                    kernel_shared[v0, v1, v2, v3] = kernel[v0, v1, v2, v3]
                            for ry_1, rx_1, rc_1, nn_3, yy_3, xx_3, ff_3, ry_2, rx_2, rc_2, nn_4, yy_4, xx_4, ff_4 in T.grid(1, 1, 1, 1, 1, 1, 2, 1, 7, 1, 1, 7, 1, 8):
                                with T.block("Conv2dOutput"):
                                    v_nn = T.axis.spatial(1, nn_3 + nn_4)
                                    v_yy = T.axis.spatial(112, nn_1_yy_1_xx_1_ff_1_fused // 2 * 56 + nn_2_yy_2_xx_2_ff_2_fused // 16 * 7 + yy_3 * 7 + yy_4)
                                    v_xx = T.axis.spatial(112, nn_0_yy_0_xx_0_ff_0_fused // 2 * 16 + nn_2_yy_2_xx_2_ff_2_fused % 16 + xx_3 + xx_4)
                                    v_ff = T.axis.spatial(64, nn_0_yy_0_xx_0_ff_0_fused % 2 * 32 + nn_1_yy_1_xx_1_ff_1_fused % 2 * 16 + ff_3 * 8 + ff_4)
                                    v_ry = T.axis.reduce(7, ry_0 + ry_1 + ry_2)
                                    v_rx = T.axis.reduce(7, rx_0 * 7 + rx_1 * 7 + rx_2)
                                    v_rc = T.axis.reduce(3, rc_0 + rc_1 + rc_2)
                                    T.reads(PaddedInput_shared[v_nn, v_yy * 2 + v_ry, v_xx * 2 + v_rx, v_rc], kernel_shared[v_ry, v_rx, v_rc, v_ff])
                                    T.writes(Conv2dOutput_local[v_nn, v_yy, v_xx, v_ff])
                                    T.block_attr({"meta_schedule.thread_extent_high_inclusive": 1024, "meta_schedule.thread_extent_low_inclusive": 32, "meta_schedule.tiling_structure": "SSSRRSRS"})
                                    with T.init():
                                        Conv2dOutput_local[v_nn, v_yy, v_xx, v_ff] = T.float32(0)
                                    Conv2dOutput_local[v_nn, v_yy, v_xx, v_ff] = Conv2dOutput_local[v_nn, v_yy, v_xx, v_ff] + PaddedInput_shared[v_nn, v_yy * 2 + v_ry, v_xx * 2 + v_rx, v_rc] * kernel_shared[v_ry, v_rx, v_rc, v_ff]
                        for ax0, ax1, ax2, ax3 in T.grid(1, 7, 1, 16):
                            with T.block("Conv2dOutput_local"):
                                v0 = T.axis.spatial(1, ax0)
                                v1 = T.axis.spatial(112, nn_1_yy_1_xx_1_ff_1_fused // 2 * 56 + nn_2_yy_2_xx_2_ff_2_fused // 16 * 7 + ax1)
                                v2 = T.axis.spatial(112, nn_0_yy_0_xx_0_ff_0_fused // 2 * 16 + nn_2_yy_2_xx_2_ff_2_fused % 16 + ax2)
                                v3 = T.axis.spatial(64, nn_0_yy_0_xx_0_ff_0_fused % 2 * 32 + nn_1_yy_1_xx_1_ff_1_fused % 2 * 16 + ax3)
                                T.reads(Conv2dOutput_local[v0, v1, v2, v3], bias[v3], bn_scale[v3], bn_offset[v3])
                                T.writes(compute[v0, v1, v2, v3])
                                compute[v0, v1, v2, v3] = T.max((Conv2dOutput_local[v0, v1, v2, v3] + bias[v3]) * bn_scale[v3] + bn_offset[v3], T.float32(0))
    # fmt: on
    decision_0 = [
        ("SamplePerfectTile", [1, 1, 1, 1, 1]),
        ("SamplePerfectTile", [1, 2, 8, 1, 7]),
        ("SamplePerfectTile", [7, 1, 16, 1, 1]),
        ("SamplePerfectTile", [2, 2, 1, 2, 8]),
        ("SamplePerfectTile", [7, 1, 1]),
        ("SamplePerfectTile", [1, 1, 7]),
        ("SamplePerfectTile", [3, 1, 1]),
        ("SampleCategorical", 0),
        ("SampleCategorical", 0),
        ("SampleCategorical", 3),
    ]
    mod = create_te_workload("CBR", 0)
    actual = _design_space(mod)
    check_sketches(
        mod,
        sketches=actual,
        expected_mods=[cbr_0],
        expected_decisions=[decision_0],
    )


def test_cuda_tbg():
    # fmt: off
    @T.prim_func
    def tbg_0(query: T.Buffer((1, 128, 12, 64), "float32"), value: T.Buffer((1, 128, 12, 64), "float32"), C: T.Buffer((1, 12, 128, 128), "float32")) -> None:
        T.func_attr({"global_symbol": "main", "tir.noalias": T.bool(True)})
        with T.block("root"):
            T.reads()
            T.writes()
            T.block_attr({"meta_schedule.unroll_explicit": 1024})
            C_local = T.alloc_buffer((1, 12, 128, 128), scope="local")
            query_T_shared = T.alloc_buffer((1, 12, 128, 64), scope="shared")
            value_T_shared = T.alloc_buffer((1, 12, 64, 128), scope="shared")
            for b_0_h_0_i_0_j_0_fused in T.thread_binding(4, thread="blockIdx.x"):
                for b_1_h_1_i_1_j_1_fused in T.thread_binding(192, thread="vthread.x"):
                    for b_2_h_2_i_2_j_2_fused in T.thread_binding(32, thread="threadIdx.x"):
                        for k_0 in range(8):
                            for ax0_ax1_ax2_ax3_fused in range(12288):
                                with T.block("query_T_shared"):
                                    v0 = T.axis.spatial(1, 0)
                                    v1 = T.axis.spatial(12, ax0_ax1_ax2_ax3_fused // 1024)
                                    v2 = T.axis.spatial(128, ax0_ax1_ax2_ax3_fused % 1024 // 8)
                                    v3 = T.axis.spatial(64, k_0 * 8 + ax0_ax1_ax2_ax3_fused % 8)
                                    T.reads(query[v0, v2, v1, v3])
                                    T.writes(query_T_shared[v0, v1, v2, v3])
                                    T.block_attr({"meta_schedule.cooperative_fetch": 3})
                                    query_T_shared[v0, v1, v2, v3] = query[v0, v2, v1, v3]
                            for ax0_ax1_ax2_ax3_fused in range(3072):
                                with T.block("value_T_shared"):
                                    v0 = T.axis.spatial(1, 0)
                                    v1 = T.axis.spatial(12, ax0_ax1_ax2_ax3_fused // 256)
                                    v2 = T.axis.spatial(64, k_0 * 8 + ax0_ax1_ax2_ax3_fused % 256 // 32)
                                    v3 = T.axis.spatial(128, b_0_h_0_i_0_j_0_fused * 32 + ax0_ax1_ax2_ax3_fused % 32)
                                    T.reads(value[v0, v3, v1, v2])
                                    T.writes(value_T_shared[v0, v1, v2, v3])
                                    T.block_attr({"meta_schedule.cooperative_fetch": 4})
                                    value_T_shared[v0, v1, v2, v3] = value[v0, v3, v1, v2]
                            for k_1, b_3, h_3, i_3, j_3, k_2, b_4, h_4, i_4, j_4 in T.grid(4, 1, 2, 1, 1, 2, 1, 1, 4, 1):
                                with T.block("C"):
                                    v_b = T.axis.spatial(1, b_3 + b_4)
                                    v_h = T.axis.spatial(12, b_1_h_1_i_1_j_1_fused // 32 * 2 + h_3 + h_4)
                                    v_i = T.axis.spatial(128, b_1_h_1_i_1_j_1_fused % 32 // 8 * 32 + b_2_h_2_i_2_j_2_fused // 4 * 4 + i_3 * 4 + i_4)
                                    v_j = T.axis.spatial(128, b_0_h_0_i_0_j_0_fused * 32 + b_1_h_1_i_1_j_1_fused % 8 * 4 + b_2_h_2_i_2_j_2_fused % 4 + j_3 + j_4)
                                    v_k = T.axis.reduce(64, k_0 * 8 + k_1 * 2 + k_2)
                                    T.reads(query_T_shared[v_b, v_h, v_i, v_k], value_T_shared[v_b, v_h, v_k, v_j])
                                    T.writes(C_local[v_b, v_h, v_i, v_j])
                                    T.block_attr({"meta_schedule.thread_extent_high_inclusive": 1024, "meta_schedule.thread_extent_low_inclusive": 32, "meta_schedule.tiling_structure": "SSSRRSRS"})
                                    with T.init():
                                        C_local[v_b, v_h, v_i, v_j] = T.float32(0)
                                    C_local[v_b, v_h, v_i, v_j] = C_local[v_b, v_h, v_i, v_j] + query_T_shared[v_b, v_h, v_i, v_k] * value_T_shared[v_b, v_h, v_k, v_j]
                        for ax0, ax1, ax2, ax3 in T.grid(1, 2, 4, 1):
                            with T.block("C_local"):
                                v0 = T.axis.spatial(1, ax0)
                                v1 = T.axis.spatial(12, b_1_h_1_i_1_j_1_fused // 32 * 2 + ax1)
                                v2 = T.axis.spatial(128, b_1_h_1_i_1_j_1_fused % 32 // 8 * 32 + b_2_h_2_i_2_j_2_fused // 4 * 4 + ax2)
                                v3 = T.axis.spatial(128, b_0_h_0_i_0_j_0_fused * 32 + b_1_h_1_i_1_j_1_fused % 8 * 4 + b_2_h_2_i_2_j_2_fused % 4 + ax3)
                                T.reads(C_local[v0, v1, v2, v3])
                                T.writes(C[v0, v1, v2, v3])
                                C[v0, v1, v2, v3] = C_local[v0, v1, v2, v3]
    # fmt: on
    decision_0 = [
        ("SamplePerfectTile", [1, 1, 1, 1, 1]),
        ("SamplePerfectTile", [1, 6, 1, 2, 1]),
        ("SamplePerfectTile", [1, 4, 8, 1, 4]),
        ("SamplePerfectTile", [4, 8, 4, 1, 1]),
        ("SamplePerfectTile", [8, 4, 2]),
        ("SampleCategorical", 2),
        ("SampleCategorical", 3),
        ("SampleCategorical", 4),
    ]
    mod = create_te_workload("TBG", 0)
    actual = _design_space(mod)
    check_sketches(
        mod,
        sketches=actual,
        expected_mods=[tbg_0],
        expected_decisions=[decision_0],
    )


if __name__ == "__main__":
    test_cuda_c1d()
    test_cuda_c2d()
    test_cuda_c3d()
    test_cuda_cap()
    test_cuda_dep()
    test_cuda_dil()
    test_cuda_gmm()
    test_cuda_grp()
    test_cuda_t2d()
    test_cuda_nrm()
    test_cuda_sfm()
    test_cuda_cbr()
    test_cuda_tbg()
