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


def test_cuda_nhwc():
    # fmt: off
    @T.prim_func
    def cuda_nhwc_0(data: T.Buffer((1, 14, 14, 128), "float32"), weight: T.Buffer((6, 6, 128, 128), "float32"), conv2d_winograd: T.Buffer((1, 12, 12, 128), "float32")) -> None:
        T.func_attr({"global_symbol": "main", "layout_free_buffers": [1], "tir.noalias": T.bool(True)})
        with T.block("root"):
            T.reads()
            T.writes()
            T.block_attr({"meta_schedule.unroll_explicit": 16})
            input_tile_local = T.alloc_buffer((6, 6, 9, 128), scope="local")
            data_pack = T.alloc_buffer((6, 6, 9, 128))
            bgemm = T.alloc_buffer((6, 6, 9, 128))
            inverse = T.alloc_buffer((4, 4, 9, 128))
            data_pack_local = T.alloc_buffer((6, 6, 9, 128), scope="local")
            bgemm_local = T.alloc_buffer((6, 6, 9, 128), scope="local")
            data_pack_shared = T.alloc_buffer((6, 6, 9, 128), scope="shared")
            weight_shared = T.alloc_buffer((6, 6, 128, 128), scope="shared")
            for p_0_ci_0_p_1_ci_1_fused_0 in T.thread_binding(2, thread="blockIdx.x"):
                for p_0_ci_0_p_1_ci_1_fused_1 in T.thread_binding(1024, thread="threadIdx.x"):
                    for ax0, ax1, ax2, ax3 in T.grid(6, 6, 1, 1):
                        with T.block("input_tile"):
                            v_eps, v_nu = T.axis.remap("SS", [ax0, ax1])
                            v_p = T.axis.spatial(9, (p_0_ci_0_p_1_ci_1_fused_0 * 1024 + p_0_ci_0_p_1_ci_1_fused_1) // 384 * 3 + (p_0_ci_0_p_1_ci_1_fused_0 * 1024 + p_0_ci_0_p_1_ci_1_fused_1) % 24 // 8 + ax2)
                            v_ci = T.axis.spatial(128, (p_0_ci_0_p_1_ci_1_fused_0 * 1024 + p_0_ci_0_p_1_ci_1_fused_1) % 384 // 24 * 8 + (p_0_ci_0_p_1_ci_1_fused_0 * 1024 + p_0_ci_0_p_1_ci_1_fused_1) % 8 + ax3)
                            T.where(p_0_ci_0_p_1_ci_1_fused_0 * 1024 + p_0_ci_0_p_1_ci_1_fused_1 < 1152)
                            T.reads(data[v_p // 9, v_p % 9 // 3 * 4 + v_eps, v_p % 3 * 4 + v_nu, v_ci])
                            T.writes(input_tile_local[v_eps, v_nu, v_p, v_ci])
                            T.block_attr({"schedule_rule": "None"})
                            input_tile_local[v_eps, v_nu, v_p, v_ci] = T.if_then_else(0 <= v_p % 9 // 3 * 4 + v_eps and v_p % 9 // 3 * 4 + v_eps < 14 and 0 <= v_p % 3 * 4 + v_nu and v_p % 3 * 4 + v_nu < 14, data[v_p // 9, v_p % 9 // 3 * 4 + v_eps, v_p % 3 * 4 + v_nu, v_ci], T.float32(0))
                    for eps in T.unroll(6):
                        for nu in T.unroll(6):
                            for r_a in T.unroll(6):
                                for r_b in T.unroll(6):
                                    with T.block("data_pack"):
                                        v_eps, v_nu = T.axis.remap("SS", [eps, nu])
                                        v_p = T.axis.spatial(9, (p_0_ci_0_p_1_ci_1_fused_0 * 1024 + p_0_ci_0_p_1_ci_1_fused_1) // 384 * 3 + (p_0_ci_0_p_1_ci_1_fused_0 * 1024 + p_0_ci_0_p_1_ci_1_fused_1) % 24 // 8)
                                        v_ci = T.axis.spatial(128, (p_0_ci_0_p_1_ci_1_fused_0 * 1024 + p_0_ci_0_p_1_ci_1_fused_1) % 384 // 24 * 8 + (p_0_ci_0_p_1_ci_1_fused_0 * 1024 + p_0_ci_0_p_1_ci_1_fused_1) % 8)
                                        v_r_a, v_r_b = T.axis.remap("RR", [r_a, r_b])
                                        T.where(p_0_ci_0_p_1_ci_1_fused_0 * 1024 + p_0_ci_0_p_1_ci_1_fused_1 < 1152)
                                        T.reads(input_tile_local[v_r_a, v_r_b, v_p, v_ci])
                                        T.writes(data_pack_local[v_eps, v_nu, v_p, v_ci])
                                        T.block_attr({"schedule_rule": "conv2d_nhwc_winograd_data_pack"})
                                        with T.init():
                                            data_pack_local[v_eps, v_nu, v_p, v_ci] = T.float32(0)
                                        data_pack_local[v_eps, v_nu, v_p, v_ci] = data_pack_local[v_eps, v_nu, v_p, v_ci] + input_tile_local[v_r_a, v_r_b, v_p, v_ci] * T.Select(v_r_a % 6 == 5 and v_eps % 6 == 5, T.float32(1), T.Select(v_r_a % 6 == 5 and v_eps % 6 == 4, T.float32(0), T.Select(v_r_a % 6 == 5 and v_eps % 6 == 3, T.float32(0), T.Select(v_r_a % 6 == 5 and v_eps % 6 == 2, T.float32(0), T.Select(v_r_a % 6 == 5 and v_eps % 6 == 1, T.float32(0), T.Select(v_r_a % 6 == 5 and v_eps % 6 == 0, T.float32(0), T.Select(v_r_a % 6 == 4 and v_eps % 6 == 5, T.float32(1.5), T.Select(v_r_a % 6 == 4 and v_eps % 6 == 4, T.float32(1), T.Select(v_r_a % 6 == 4 and v_eps % 6 == 3, T.float32(1), T.Select(v_r_a % 6 == 4 and v_eps % 6 == 2, T.float32(1), T.Select(v_r_a % 6 == 4 and v_eps % 6 == 1, T.float32(1), T.Select(v_r_a % 6 == 4 and v_eps % 6 == 0, T.float32(1), T.Select(v_r_a % 6 == 3 and v_eps % 6 == 5, T.float32(-2), T.Select(v_r_a % 6 == 3 and v_eps % 6 == 4, T.float32(-0.5), T.Select(v_r_a % 6 == 3 and v_eps % 6 == 3, T.float32(2), T.Select(v_r_a % 6 == 3 and v_eps % 6 == 2, T.float32(2.5), T.Select(v_r_a % 6 == 3 and v_eps % 6 == 1, T.float32(0.5), T.Select(v_r_a % 6 == 3 and v_eps % 6 == 0, T.float32(1.5), T.Select(v_r_a % 6 == 2 and v_eps % 6 == 5, T.float32(-1.5), T.Select(v_r_a % 6 == 2 and v_eps % 6 == 4, T.float32(-1), T.Select(v_r_a % 6 == 2 and v_eps % 6 == 3, T.float32(-1), T.Select(v_r_a % 6 == 2 and v_eps % 6 == 2, T.float32(0.5), T.Select(v_r_a % 6 == 2 and v_eps % 6 == 1, T.float32(-2.5), T.Select(v_r_a % 6 == 2 and v_eps % 6 == 0, T.float32(-2), T.Select(v_r_a % 6 == 1 and v_eps % 6 == 5, T.float32(1), T.Select(v_r_a % 6 == 1 and v_eps % 6 == 4, T.float32(0.5), T.Select(v_r_a % 6 == 1 and v_eps % 6 == 3, T.float32(-2), T.Select(v_r_a % 6 == 1 and v_eps % 6 == 2, T.float32(-1), T.Select(v_r_a % 6 == 1 and v_eps % 6 == 1, T.float32(1), T.Select(v_r_a % 6 == 1 and v_eps % 6 == 0, T.float32(-1.5), T.Select(v_r_a % 6 == 0 and v_eps % 6 == 5, T.float32(0), T.Select(v_r_a % 6 == 0 and v_eps % 6 == 4, T.float32(0), T.Select(v_r_a % 6 == 0 and v_eps % 6 == 3, T.float32(0), T.Select(v_r_a % 6 == 0 and v_eps % 6 == 2, T.float32(0), T.Select(v_r_a % 6 == 0 and v_eps % 6 == 1, T.float32(0), T.Select(v_r_a % 6 == 0 and v_eps % 6 == 0, T.float32(1), T.float32(0))))))))))))))))))))))))))))))))))))) * T.Select(v_r_b % 6 == 5 and v_nu % 6 == 5, T.float32(1), T.Select(v_r_b % 6 == 5 and v_nu % 6 == 4, T.float32(0), T.Select(v_r_b % 6 == 5 and v_nu % 6 == 3, T.float32(0), T.Select(v_r_b % 6 == 5 and v_nu % 6 == 2, T.float32(0), T.Select(v_r_b % 6 == 5 and v_nu % 6 == 1, T.float32(0), T.Select(v_r_b % 6 == 5 and v_nu % 6 == 0, T.float32(0), T.Select(v_r_b % 6 == 4 and v_nu % 6 == 5, T.float32(1.5), T.Select(v_r_b % 6 == 4 and v_nu % 6 == 4, T.float32(1), T.Select(v_r_b % 6 == 4 and v_nu % 6 == 3, T.float32(1), T.Select(v_r_b % 6 == 4 and v_nu % 6 == 2, T.float32(1), T.Select(v_r_b % 6 == 4 and v_nu % 6 == 1, T.float32(1), T.Select(v_r_b % 6 == 4 and v_nu % 6 == 0, T.float32(1), T.Select(v_r_b % 6 == 3 and v_nu % 6 == 5, T.float32(-2), T.Select(v_r_b % 6 == 3 and v_nu % 6 == 4, T.float32(-0.5), T.Select(v_r_b % 6 == 3 and v_nu % 6 == 3, T.float32(2), T.Select(v_r_b % 6 == 3 and v_nu % 6 == 2, T.float32(2.5), T.Select(v_r_b % 6 == 3 and v_nu % 6 == 1, T.float32(0.5), T.Select(v_r_b % 6 == 3 and v_nu % 6 == 0, T.float32(1.5), T.Select(v_r_b % 6 == 2 and v_nu % 6 == 5, T.float32(-1.5), T.Select(v_r_b % 6 == 2 and v_nu % 6 == 4, T.float32(-1), T.Select(v_r_b % 6 == 2 and v_nu % 6 == 3, T.float32(-1), T.Select(v_r_b % 6 == 2 and v_nu % 6 == 2, T.float32(0.5), T.Select(v_r_b % 6 == 2 and v_nu % 6 == 1, T.float32(-2.5), T.Select(v_r_b % 6 == 2 and v_nu % 6 == 0, T.float32(-2), T.Select(v_r_b % 6 == 1 and v_nu % 6 == 5, T.float32(1), T.Select(v_r_b % 6 == 1 and v_nu % 6 == 4, T.float32(0.5), T.Select(v_r_b % 6 == 1 and v_nu % 6 == 3, T.float32(-2), T.Select(v_r_b % 6 == 1 and v_nu % 6 == 2, T.float32(-1), T.Select(v_r_b % 6 == 1 and v_nu % 6 == 1, T.float32(1), T.Select(v_r_b % 6 == 1 and v_nu % 6 == 0, T.float32(-1.5), T.Select(v_r_b % 6 == 0 and v_nu % 6 == 5, T.float32(0), T.Select(v_r_b % 6 == 0 and v_nu % 6 == 4, T.float32(0), T.Select(v_r_b % 6 == 0 and v_nu % 6 == 3, T.float32(0), T.Select(v_r_b % 6 == 0 and v_nu % 6 == 2, T.float32(0), T.Select(v_r_b % 6 == 0 and v_nu % 6 == 1, T.float32(0), T.Select(v_r_b % 6 == 0 and v_nu % 6 == 0, T.float32(1), T.float32(0)))))))))))))))))))))))))))))))))))))
                    for ax0, ax1, ax2, ax3 in T.grid(6, 6, 1, 1):
                        with T.block("data_pack_local"):
                            v0, v1 = T.axis.remap("SS", [ax0, ax1])
                            v2 = T.axis.spatial(9, (p_0_ci_0_p_1_ci_1_fused_0 * 1024 + p_0_ci_0_p_1_ci_1_fused_1) // 384 * 3 + (p_0_ci_0_p_1_ci_1_fused_0 * 1024 + p_0_ci_0_p_1_ci_1_fused_1) % 24 // 8 + ax2)
                            v3 = T.axis.spatial(128, (p_0_ci_0_p_1_ci_1_fused_0 * 1024 + p_0_ci_0_p_1_ci_1_fused_1) % 384 // 24 * 8 + (p_0_ci_0_p_1_ci_1_fused_0 * 1024 + p_0_ci_0_p_1_ci_1_fused_1) % 8 + ax3)
                            T.where(p_0_ci_0_p_1_ci_1_fused_0 * 1024 + p_0_ci_0_p_1_ci_1_fused_1 < 1152)
                            T.reads(data_pack_local[v0, v1, v2, v3])
                            T.writes(data_pack[v0, v1, v2, v3])
                            data_pack[v0, v1, v2, v3] = data_pack_local[v0, v1, v2, v3]
            for eps_0_nu_0_p_0_co_0_fused in T.thread_binding(96, thread="blockIdx.x"):
                for eps_1_nu_1_p_1_co_1_fused in T.thread_binding(4, thread="vthread.x"):
                    for eps_2_nu_2_p_2_co_2_fused in T.thread_binding(27, thread="threadIdx.x"):
                        for ci_0 in range(8):
                            for ax0_ax1_ax2_ax3_fused in range(1728):
                                with T.block("data_pack_shared"):
                                    v0 = T.axis.spatial(6, eps_0_nu_0_p_0_co_0_fused // 32 * 2 + ax0_ax1_ax2_ax3_fused // 864)
                                    v1 = T.axis.spatial(6, ax0_ax1_ax2_ax3_fused % 864 // 144)
                                    v2 = T.axis.spatial(9, ax0_ax1_ax2_ax3_fused % 144 // 16)
                                    v3 = T.axis.spatial(128, ci_0 * 16 + ax0_ax1_ax2_ax3_fused % 16)
                                    T.reads(data_pack[v0, v1, v2, v3])
                                    T.writes(data_pack_shared[v0, v1, v2, v3])
                                    T.block_attr({"meta_schedule.cooperative_fetch": 1})
                                    data_pack_shared[v0, v1, v2, v3] = data_pack[v0, v1, v2, v3]
                            for ax0_ax1_ax2_ax3_fused in range(768):
                                with T.block("weight_shared"):
                                    v0 = T.axis.spatial(6, eps_0_nu_0_p_0_co_0_fused // 32 * 2 + ax0_ax1_ax2_ax3_fused // 384)
                                    v1 = T.axis.spatial(6, ax0_ax1_ax2_ax3_fused % 384 // 64)
                                    v2 = T.axis.spatial(128, eps_0_nu_0_p_0_co_0_fused % 32 * 4 + ax0_ax1_ax2_ax3_fused % 64 // 16)
                                    v3 = T.axis.spatial(128, ci_0 * 16 + ax0_ax1_ax2_ax3_fused % 16)
                                    T.reads(weight[v0, v1, v2, v3])
                                    T.writes(weight_shared[v0, v1, v2, v3])
                                    T.block_attr({"meta_schedule.cooperative_fetch": 3})
                                    weight_shared[v0, v1, v2, v3] = weight[v0, v1, v2, v3]
                            for ci_1, eps_3, nu_3, p_3, co_3, ci_2, eps_4, nu_4, p_4, co_4 in T.grid(1, 2, 1, 1, 2, 16, 1, 1, 1, 1):
                                with T.block("bgemm"):
                                    v_eps = T.axis.spatial(6, eps_0_nu_0_p_0_co_0_fused // 32 * 2 + eps_3 + eps_4)
                                    v_nu = T.axis.spatial(6, eps_1_nu_1_p_1_co_1_fused // 2 * 3 + eps_2_nu_2_p_2_co_2_fused // 9 + nu_3 + nu_4)
                                    v_p = T.axis.spatial(9, eps_2_nu_2_p_2_co_2_fused % 9 + p_3 + p_4)
                                    v_co = T.axis.spatial(128, eps_0_nu_0_p_0_co_0_fused % 32 * 4 + eps_1_nu_1_p_1_co_1_fused % 2 * 2 + co_3 + co_4)
                                    v_ci = T.axis.reduce(128, ci_0 * 16 + ci_1 * 16 + ci_2)
                                    T.reads(data_pack_shared[v_eps, v_nu, v_p, v_ci], weight_shared[v_eps, v_nu, v_co, v_ci])
                                    T.writes(bgemm_local[v_eps, v_nu, v_p, v_co])
                                    T.block_attr({"meta_schedule.thread_extent_high_inclusive": 1024, "meta_schedule.thread_extent_low_inclusive": 32, "meta_schedule.tiling_structure": "SSSRRSRS", "meta_schedule.write_cache_level": [3]})
                                    with T.init():
                                        bgemm_local[v_eps, v_nu, v_p, v_co] = T.float32(0)
                                    bgemm_local[v_eps, v_nu, v_p, v_co] = bgemm_local[v_eps, v_nu, v_p, v_co] + data_pack_shared[v_eps, v_nu, v_p, v_ci] * weight_shared[v_eps, v_nu, v_co, v_ci]
                        for ax0, ax1, ax2, ax3 in T.grid(2, 1, 1, 2):
                            with T.block("bgemm_local"):
                                v0 = T.axis.spatial(6, eps_0_nu_0_p_0_co_0_fused // 32 * 2 + ax0)
                                v1 = T.axis.spatial(6, eps_1_nu_1_p_1_co_1_fused // 2 * 3 + eps_2_nu_2_p_2_co_2_fused // 9 + ax1)
                                v2 = T.axis.spatial(9, eps_2_nu_2_p_2_co_2_fused % 9 + ax2)
                                v3 = T.axis.spatial(128, eps_0_nu_0_p_0_co_0_fused % 32 * 4 + eps_1_nu_1_p_1_co_1_fused % 2 * 2 + ax3)
                                T.reads(bgemm_local[v0, v1, v2, v3])
                                T.writes(bgemm[v0, v1, v2, v3])
                                bgemm[v0, v1, v2, v3] = bgemm_local[v0, v1, v2, v3]
            for p_0_co_0_p_1_co_1_fused_0 in T.thread_binding(18, thread="blockIdx.x"):
                for p_0_co_0_p_1_co_1_fused_1 in T.thread_binding(64, thread="threadIdx.x"):
                    for vh in T.unroll(4):
                        for vw in T.unroll(4):
                            for r_a in T.unroll(6):
                                for r_b in T.unroll(6):
                                    with T.block("inverse"):
                                        v_vh, v_vw = T.axis.remap("SS", [vh, vw])
                                        v_p = T.axis.spatial(9, (p_0_co_0_p_1_co_1_fused_0 * 64 + p_0_co_0_p_1_co_1_fused_1) // 384 * 3 + (p_0_co_0_p_1_co_1_fused_0 * 64 + p_0_co_0_p_1_co_1_fused_1) % 24 // 8)
                                        v_co = T.axis.spatial(128, (p_0_co_0_p_1_co_1_fused_0 * 64 + p_0_co_0_p_1_co_1_fused_1) % 384 // 24 * 8 + (p_0_co_0_p_1_co_1_fused_0 * 64 + p_0_co_0_p_1_co_1_fused_1) % 8)
                                        v_r_a, v_r_b = T.axis.remap("RR", [r_a, r_b])
                                        T.reads(bgemm[v_r_a, v_r_b, v_p, v_co])
                                        T.writes(inverse[v_vh, v_vw, v_p, v_co])
                                        T.block_attr({"schedule_rule": "conv2d_nhwc_winograd_inverse"})
                                        with T.init():
                                            inverse[v_vh, v_vw, v_p, v_co] = T.float32(0)
                                        inverse[v_vh, v_vw, v_p, v_co] = inverse[v_vh, v_vw, v_p, v_co] + bgemm[v_r_a, v_r_b, v_p, v_co] * T.Select(v_r_a % 6 == 5 and v_vh % 4 == 3, T.float32(1), T.Select(v_r_a % 6 == 5 and v_vh % 4 == 2, T.float32(0), T.Select(v_r_a % 6 == 5 and v_vh % 4 == 1, T.float32(0), T.Select(v_r_a % 6 == 5 and v_vh % 4 == 0, T.float32(0), T.Select(v_r_a % 6 == 4 and v_vh % 4 == 3, T.float32(-8), T.Select(v_r_a % 6 == 4 and v_vh % 4 == 2, T.float32(4), T.Select(v_r_a % 6 == 4 and v_vh % 4 == 1, T.float32(-2), T.Select(v_r_a % 6 == 4 and v_vh % 4 == 0, T.float32(1), T.Select(v_r_a % 6 == 3 and v_vh % 4 == 3, T.float32(0.125), T.Select(v_r_a % 6 == 3 and v_vh % 4 == 2, T.float32(0.25), T.Select(v_r_a % 6 == 3 and v_vh % 4 == 1, T.float32(0.5), T.Select(v_r_a % 6 == 3 and v_vh % 4 == 0, T.float32(1), T.Select(v_r_a % 6 == 2 and v_vh % 4 == 3, T.float32(1), T.Select(v_r_a % 6 == 2 and v_vh % 4 == 2, T.float32(1), T.Select(v_r_a % 6 == 2 and v_vh % 4 == 1, T.float32(1), T.Select(v_r_a % 6 == 2 and v_vh % 4 == 0, T.float32(1), T.Select(v_r_a % 6 == 1 and v_vh % 4 == 3, T.float32(-1), T.Select(v_r_a % 6 == 1 and v_vh % 4 == 2, T.float32(1), T.Select(v_r_a % 6 == 1 and v_vh % 4 == 1, T.float32(-1), T.Select(v_r_a % 6 == 1 and v_vh % 4 == 0, T.float32(1), T.Select(v_r_a % 6 == 0 and v_vh % 4 == 3, T.float32(0), T.Select(v_r_a % 6 == 0 and v_vh % 4 == 2, T.float32(0), T.Select(v_r_a % 6 == 0 and v_vh % 4 == 1, T.float32(0), T.Select(v_r_a % 6 == 0 and v_vh % 4 == 0, T.float32(1), T.float32(0))))))))))))))))))))))))) * T.Select(v_r_b % 6 == 5 and v_vw % 4 == 3, T.float32(1), T.Select(v_r_b % 6 == 5 and v_vw % 4 == 2, T.float32(0), T.Select(v_r_b % 6 == 5 and v_vw % 4 == 1, T.float32(0), T.Select(v_r_b % 6 == 5 and v_vw % 4 == 0, T.float32(0), T.Select(v_r_b % 6 == 4 and v_vw % 4 == 3, T.float32(-8), T.Select(v_r_b % 6 == 4 and v_vw % 4 == 2, T.float32(4), T.Select(v_r_b % 6 == 4 and v_vw % 4 == 1, T.float32(-2), T.Select(v_r_b % 6 == 4 and v_vw % 4 == 0, T.float32(1), T.Select(v_r_b % 6 == 3 and v_vw % 4 == 3, T.float32(0.125), T.Select(v_r_b % 6 == 3 and v_vw % 4 == 2, T.float32(0.25), T.Select(v_r_b % 6 == 3 and v_vw % 4 == 1, T.float32(0.5), T.Select(v_r_b % 6 == 3 and v_vw % 4 == 0, T.float32(1), T.Select(v_r_b % 6 == 2 and v_vw % 4 == 3, T.float32(1), T.Select(v_r_b % 6 == 2 and v_vw % 4 == 2, T.float32(1), T.Select(v_r_b % 6 == 2 and v_vw % 4 == 1, T.float32(1), T.Select(v_r_b % 6 == 2 and v_vw % 4 == 0, T.float32(1), T.Select(v_r_b % 6 == 1 and v_vw % 4 == 3, T.float32(-1), T.Select(v_r_b % 6 == 1 and v_vw % 4 == 2, T.float32(1), T.Select(v_r_b % 6 == 1 and v_vw % 4 == 1, T.float32(-1), T.Select(v_r_b % 6 == 1 and v_vw % 4 == 0, T.float32(1), T.Select(v_r_b % 6 == 0 and v_vw % 4 == 3, T.float32(0), T.Select(v_r_b % 6 == 0 and v_vw % 4 == 2, T.float32(0), T.Select(v_r_b % 6 == 0 and v_vw % 4 == 1, T.float32(0), T.Select(v_r_b % 6 == 0 and v_vw % 4 == 0, T.float32(1), T.float32(0)))))))))))))))))))))))))
            for n_h_w_co_fused_0 in T.thread_binding(144, thread="blockIdx.x"):
                for n_h_w_co_fused_1 in T.thread_binding(128, thread="threadIdx.x"):
                    with T.block("conv2d_winograd"):
                        v_n = T.axis.spatial(1, 0)
                        v_h = T.axis.spatial(12, (n_h_w_co_fused_0 * 128 + n_h_w_co_fused_1) // 1536)
                        v_w = T.axis.spatial(12, (n_h_w_co_fused_0 * 128 + n_h_w_co_fused_1) % 1536 // 128)
                        v_co = T.axis.spatial(128, (n_h_w_co_fused_0 * 128 + n_h_w_co_fused_1) % 128)
                        T.reads(inverse[v_h % 4, v_w % 4, v_n * 9 + v_h // 4 * 3 + v_w // 4, v_co])
                        T.writes(conv2d_winograd[v_n, v_h, v_w, v_co])
                        conv2d_winograd[v_n, v_h, v_w, v_co] = inverse[v_h % 4, v_w % 4, v_n * 9 + v_h // 4 * 3 + v_w // 4, v_co]
    # fmt: on
    decision_0 = [
        ("SamplePerfectTile", [3, 3]),
        ("SamplePerfectTile", [16, 8]),
        ("SampleCategorical", 1),
        ("SamplePerfectTile", [3, 3]),
        ("SamplePerfectTile", [16, 8]),
        ("SampleCategorical", 5),
        ("SamplePerfectTile", [3, 1, 1, 2, 1]),
        ("SamplePerfectTile", [1, 2, 3, 1, 1]),
        ("SamplePerfectTile", [1, 1, 9, 1, 1]),
        ("SamplePerfectTile", [32, 2, 1, 2, 1]),
        ("SamplePerfectTile", [8, 1, 16]),
        ("SampleCategorical", 0),
        ("SampleCategorical", 2),
        ("SampleCategorical", 1),
        ("SampleCategorical", 2),
    ]
    with _target():
        mod = create_te_workload("C2D_WIN_NHWC", 0)
    actual = _design_space(mod)
    check_sketches(
        mod,
        sketches=actual,
        expected_mods=[cuda_nhwc_0],
        expected_decisions=[decision_0],
    )


def test_cuda_nchw():
    # fmt: off
    @T.prim_func
    def cuda_nchw_0(data: T.Buffer((1, 64, 56, 56), "float32"), weight: T.Buffer((6, 6, 64, 64), "float32"), conv2d_winograd: T.Buffer((1, 64, 56, 56), "float32")) -> None:
        T.func_attr({"global_symbol": "main", "layout_free_buffers": [1], "tir.noalias": T.bool(True)})
        with T.block("root"):
            T.reads()
            T.writes()
            T.block_attr({"meta_schedule.unroll_explicit": 16})
            input_tile_local = T.alloc_buffer((64, 196, 6, 6), scope="local")
            data_pack = T.alloc_buffer((6, 6, 64, 196))
            bgemm = T.alloc_buffer((6, 6, 64, 196))
            inverse_local = T.alloc_buffer((64, 196, 4, 4), scope="local")
            data_pack_local = T.alloc_buffer((6, 6, 64, 196), scope="local")
            bgemm_local = T.alloc_buffer((6, 6, 64, 196), scope="local")
            data_pack_shared = T.alloc_buffer((6, 6, 64, 196), scope="shared")
            weight_shared = T.alloc_buffer((6, 6, 64, 64), scope="shared")
            for ci_p_fused_0 in T.thread_binding(25, thread="blockIdx.x"):
                for ci_p_fused_1 in T.thread_binding(512, thread="threadIdx.x"):
                    for ax0, ax1, ax2, ax3 in T.grid(1, 1, 6, 6):
                        with T.block("input_tile"):
                            v_ci = T.axis.spatial(64, (ci_p_fused_0 * 512 + ci_p_fused_1) // 196 + ax0)
                            v_p = T.axis.spatial(196, (ci_p_fused_0 * 120 + ci_p_fused_1) % 196 + ax1)
                            v_eps, v_nu = T.axis.remap("SS", [ax2, ax3])
                            T.where(ci_p_fused_0 * 512 + ci_p_fused_1 < 12544)
                            T.reads(data[v_p // 196, v_ci, v_p % 196 // 14 * 4 + v_eps - 1, v_p % 14 * 4 + v_nu - 1])
                            T.writes(input_tile_local[v_ci, v_p, v_eps, v_nu])
                            T.block_attr({"schedule_rule": "None"})
                            input_tile_local[v_ci, v_p, v_eps, v_nu] = T.if_then_else(1 <= v_p % 196 // 14 * 4 + v_eps and v_p % 196 // 14 * 4 + v_eps < 57 and 1 <= v_p % 14 * 4 + v_nu and v_p % 14 * 4 + v_nu < 57, data[v_p // 196, v_ci, v_p % 196 // 14 * 4 + v_eps - 1, v_p % 14 * 4 + v_nu - 1], T.float32(0))
                    for eps in T.unroll(6):
                        for nu in T.unroll(6):
                            for r_a in T.unroll(6):
                                for r_b in T.unroll(6):
                                    with T.block("data_pack"):
                                        v_eps, v_nu = T.axis.remap("SS", [eps, nu])
                                        v_ci = T.axis.spatial(64, (ci_p_fused_0 * 512 + ci_p_fused_1) // 196)
                                        v_p = T.axis.spatial(196, (ci_p_fused_0 * 512 + ci_p_fused_1) % 196)
                                        v_r_a, v_r_b = T.axis.remap("RR", [r_a, r_b])
                                        T.where(ci_p_fused_0 * 512 + ci_p_fused_1 < 12544)
                                        T.reads(input_tile_local[v_ci, v_p, v_r_a, v_r_b])
                                        T.writes(data_pack_local[v_eps, v_nu, v_ci, v_p])
                                        T.block_attr({"schedule_rule": "conv2d_nchw_winograd_data_pack"})
                                        with T.init():
                                            data_pack_local[v_eps, v_nu, v_ci, v_p] = T.float32(0)
                                        data_pack_local[v_eps, v_nu, v_ci, v_p] = data_pack_local[v_eps, v_nu, v_ci, v_p] + input_tile_local[v_ci, v_p, v_r_a, v_r_b] * T.Select(v_r_a % 6 == 5 and v_eps % 6 == 5, T.float32(1), T.Select(v_r_a % 6 == 5 and v_eps % 6 == 4, T.float32(0), T.Select(v_r_a % 6 == 5 and v_eps % 6 == 3, T.float32(0), T.Select(v_r_a % 6 == 5 and v_eps % 6 == 2, T.float32(0), T.Select(v_r_a % 6 == 5 and v_eps % 6 == 1, T.float32(0), T.Select(v_r_a % 6 == 5 and v_eps % 6 == 0, T.float32(0), T.Select(v_r_a % 6 == 4 and v_eps % 6 == 5, T.float32(1.5), T.Select(v_r_a % 6 == 4 and v_eps % 6 == 4, T.float32(1), T.Select(v_r_a % 6 == 4 and v_eps % 6 == 3, T.float32(1), T.Select(v_r_a % 6 == 4 and v_eps % 6 == 2, T.float32(1), T.Select(v_r_a % 6 == 4 and v_eps % 6 == 1, T.float32(1), T.Select(v_r_a % 6 == 4 and v_eps % 6 == 0, T.float32(1), T.Select(v_r_a % 6 == 3 and v_eps % 6 == 5, T.float32(-2), T.Select(v_r_a % 6 == 3 and v_eps % 6 == 4, T.float32(-0.5), T.Select(v_r_a % 6 == 3 and v_eps % 6 == 3, T.float32(2), T.Select(v_r_a % 6 == 3 and v_eps % 6 == 2, T.float32(2.5), T.Select(v_r_a % 6 == 3 and v_eps % 6 == 1, T.float32(0.5), T.Select(v_r_a % 6 == 3 and v_eps % 6 == 0, T.float32(1.5), T.Select(v_r_a % 6 == 2 and v_eps % 6 == 5, T.float32(-1.5), T.Select(v_r_a % 6 == 2 and v_eps % 6 == 4, T.float32(-1), T.Select(v_r_a % 6 == 2 and v_eps % 6 == 3, T.float32(-1), T.Select(v_r_a % 6 == 2 and v_eps % 6 == 2, T.float32(0.5), T.Select(v_r_a % 6 == 2 and v_eps % 6 == 1, T.float32(-2.5), T.Select(v_r_a % 6 == 2 and v_eps % 6 == 0, T.float32(-2), T.Select(v_r_a % 6 == 1 and v_eps % 6 == 5, T.float32(1), T.Select(v_r_a % 6 == 1 and v_eps % 6 == 4, T.float32(0.5), T.Select(v_r_a % 6 == 1 and v_eps % 6 == 3, T.float32(-2), T.Select(v_r_a % 6 == 1 and v_eps % 6 == 2, T.float32(-1), T.Select(v_r_a % 6 == 1 and v_eps % 6 == 1, T.float32(1), T.Select(v_r_a % 6 == 1 and v_eps % 6 == 0, T.float32(-1.5), T.Select(v_r_a % 6 == 0 and v_eps % 6 == 5, T.float32(0), T.Select(v_r_a % 6 == 0 and v_eps % 6 == 4, T.float32(0), T.Select(v_r_a % 6 == 0 and v_eps % 6 == 3, T.float32(0), T.Select(v_r_a % 6 == 0 and v_eps % 6 == 2, T.float32(0), T.Select(v_r_a % 6 == 0 and v_eps % 6 == 1, T.float32(0), T.Select(v_r_a % 6 == 0 and v_eps % 6 == 0, T.float32(1), T.float32(0))))))))))))))))))))))))))))))))))))) * T.Select(v_r_b % 6 == 5 and v_nu % 6 == 5, T.float32(1), T.Select(v_r_b % 6 == 5 and v_nu % 6 == 4, T.float32(0), T.Select(v_r_b % 6 == 5 and v_nu % 6 == 3, T.float32(0), T.Select(v_r_b % 6 == 5 and v_nu % 6 == 2, T.float32(0), T.Select(v_r_b % 6 == 5 and v_nu % 6 == 1, T.float32(0), T.Select(v_r_b % 6 == 5 and v_nu % 6 == 0, T.float32(0), T.Select(v_r_b % 6 == 4 and v_nu % 6 == 5, T.float32(1.5), T.Select(v_r_b % 6 == 4 and v_nu % 6 == 4, T.float32(1), T.Select(v_r_b % 6 == 4 and v_nu % 6 == 3, T.float32(1), T.Select(v_r_b % 6 == 4 and v_nu % 6 == 2, T.float32(1), T.Select(v_r_b % 6 == 4 and v_nu % 6 == 1, T.float32(1), T.Select(v_r_b % 6 == 4 and v_nu % 6 == 0, T.float32(1), T.Select(v_r_b % 6 == 3 and v_nu % 6 == 5, T.float32(-2), T.Select(v_r_b % 6 == 3 and v_nu % 6 == 4, T.float32(-0.5), T.Select(v_r_b % 6 == 3 and v_nu % 6 == 3, T.float32(2), T.Select(v_r_b % 6 == 3 and v_nu % 6 == 2, T.float32(2.5), T.Select(v_r_b % 6 == 3 and v_nu % 6 == 1, T.float32(0.5), T.Select(v_r_b % 6 == 3 and v_nu % 6 == 0, T.float32(1.5), T.Select(v_r_b % 6 == 2 and v_nu % 6 == 5, T.float32(-1.5), T.Select(v_r_b % 6 == 2 and v_nu % 6 == 4, T.float32(-1), T.Select(v_r_b % 6 == 2 and v_nu % 6 == 3, T.float32(-1), T.Select(v_r_b % 6 == 2 and v_nu % 6 == 2, T.float32(0.5), T.Select(v_r_b % 6 == 2 and v_nu % 6 == 1, T.float32(-2.5), T.Select(v_r_b % 6 == 2 and v_nu % 6 == 0, T.float32(-2), T.Select(v_r_b % 6 == 1 and v_nu % 6 == 5, T.float32(1), T.Select(v_r_b % 6 == 1 and v_nu % 6 == 4, T.float32(0.5), T.Select(v_r_b % 6 == 1 and v_nu % 6 == 3, T.float32(-2), T.Select(v_r_b % 6 == 1 and v_nu % 6 == 2, T.float32(-1), T.Select(v_r_b % 6 == 1 and v_nu % 6 == 1, T.float32(1), T.Select(v_r_b % 6 == 1 and v_nu % 6 == 0, T.float32(-1.5), T.Select(v_r_b % 6 == 0 and v_nu % 6 == 5, T.float32(0), T.Select(v_r_b % 6 == 0 and v_nu % 6 == 4, T.float32(0), T.Select(v_r_b % 6 == 0 and v_nu % 6 == 3, T.float32(0), T.Select(v_r_b % 6 == 0 and v_nu % 6 == 2, T.float32(0), T.Select(v_r_b % 6 == 0 and v_nu % 6 == 1, T.float32(0), T.Select(v_r_b % 6 == 0 and v_nu % 6 == 0, T.float32(1), T.float32(0)))))))))))))))))))))))))))))))))))))
                    for ax0, ax1, ax2, ax3 in T.grid(6, 6, 1, 1):
                        with T.block("data_pack_local"):
                            v0, v1 = T.axis.remap("SS", [ax0, ax1])
                            v2 = T.axis.spatial(64, (ci_p_fused_0 * 512 + ci_p_fused_1) // 196 + ax2)
                            v3 = T.axis.spatial(196, (ci_p_fused_0 * 120 + ci_p_fused_1) % 196 + ax3)
                            T.where(ci_p_fused_0 * 512 + ci_p_fused_1 < 12544)
                            T.reads(data_pack_local[v0, v1, v2, v3])
                            T.writes(data_pack[v0, v1, v2, v3])
                            data_pack[v0, v1, v2, v3] = data_pack_local[v0, v1, v2, v3]
            for eps_0_nu_0_co_0_p_0_fused in T.thread_binding(14, thread="blockIdx.x"):
                for eps_1_nu_1_co_1_p_1_fused in T.thread_binding(224, thread="vthread.x"):
                    for eps_2_nu_2_co_2_p_2_fused in T.thread_binding(2, thread="threadIdx.x"):
                        for ci_0 in range(2):
                            for ax0_ax1_ax2_ax3_fused in range(32256):
                                with T.block("data_pack_shared"):
                                    v0 = T.axis.spatial(6, ax0_ax1_ax2_ax3_fused // 5376)
                                    v1 = T.axis.spatial(6, ax0_ax1_ax2_ax3_fused % 5376 // 896)
                                    v2 = T.axis.spatial(64, ci_0 * 32 + ax0_ax1_ax2_ax3_fused % 896 // 28)
                                    v3 = T.axis.spatial(196, eps_0_nu_0_co_0_p_0_fused % 7 * 28 + ax0_ax1_ax2_ax3_fused % 28)
                                    T.reads(data_pack[v0, v1, v2, v3])
                                    T.writes(data_pack_shared[v0, v1, v2, v3])
                                    T.block_attr({"meta_schedule.cooperative_fetch": 4})
                                    data_pack_shared[v0, v1, v2, v3] = data_pack[v0, v1, v2, v3]
                            for ax0_ax1_ax2_ax3_fused in range(36864):
                                with T.block("weight_shared"):
                                    v0 = T.axis.spatial(6, ax0_ax1_ax2_ax3_fused // 6144)
                                    v1 = T.axis.spatial(6, ax0_ax1_ax2_ax3_fused % 6144 // 1024)
                                    v2 = T.axis.spatial(64, ci_0 * 32 + ax0_ax1_ax2_ax3_fused % 1024 // 32)
                                    v3 = T.axis.spatial(64, eps_0_nu_0_co_0_p_0_fused // 7 * 32 + ax0_ax1_ax2_ax3_fused % 32)
                                    T.reads(weight[v0, v1, v2, v3])
                                    T.writes(weight_shared[v0, v1, v2, v3])
                                    T.block_attr({"meta_schedule.cooperative_fetch": 3})
                                    weight_shared[v0, v1, v2, v3] = weight[v0, v1, v2, v3]
                            for ci_1, eps_3, nu_3, co_3, p_3, ci_2, eps_4, nu_4, co_4, p_4 in T.grid(16, 2, 3, 1, 4, 2, 3, 1, 1, 1):
                                with T.block("bgemm"):
                                    v_eps = T.axis.spatial(6, eps_3 * 3 + eps_4)
                                    v_nu = T.axis.spatial(6, eps_1_nu_1_co_1_p_1_fused // 112 * 3 + nu_3 + nu_4)
                                    v_co = T.axis.spatial(64, eps_0_nu_0_co_0_p_0_fused // 7 * 32 + eps_1_nu_1_co_1_p_1_fused % 112 // 7 * 2 + eps_2_nu_2_co_2_p_2_fused + co_3 + co_4)
                                    v_p = T.axis.spatial(196, eps_0_nu_0_co_0_p_0_fused % 7 * 28 + eps_1_nu_1_co_1_p_1_fused % 7 * 4 + p_3 + p_4)
                                    v_ci = T.axis.reduce(64, ci_0 * 32 + ci_1 * 2 + ci_2)
                                    T.reads(data_pack_shared[v_eps, v_nu, v_ci, v_p], weight_shared[v_eps, v_nu, v_ci, v_co])
                                    T.writes(bgemm_local[v_eps, v_nu, v_co, v_p])
                                    T.block_attr({"meta_schedule.thread_extent_high_inclusive": 1024, "meta_schedule.thread_extent_low_inclusive": 32, "meta_schedule.tiling_structure": "SSSRRSRS"})
                                    with T.init():
                                        bgemm_local[v_eps, v_nu, v_co, v_p] = T.float32(0)
                                    bgemm_local[v_eps, v_nu, v_co, v_p] = bgemm_local[v_eps, v_nu, v_co, v_p] + data_pack_shared[v_eps, v_nu, v_ci, v_p] * weight_shared[v_eps, v_nu, v_ci, v_co]
                        for ax0, ax1, ax2, ax3 in T.grid(6, 3, 1, 4):
                            with T.block("bgemm_local"):
                                v0 = T.axis.spatial(6, ax0)
                                v1 = T.axis.spatial(6, eps_1_nu_1_co_1_p_1_fused // 112 * 3 + ax1)
                                v2 = T.axis.spatial(64, eps_0_nu_0_co_0_p_0_fused // 7 * 32 + eps_1_nu_1_co_1_p_1_fused % 112 // 7 * 2 + eps_2_nu_2_co_2_p_2_fused + ax2)
                                v3 = T.axis.spatial(196, eps_0_nu_0_co_0_p_0_fused % 7 * 28 + eps_1_nu_1_co_1_p_1_fused % 7 * 4 + ax3)
                                T.reads(bgemm_local[v0, v1, v2, v3])
                                T.writes(bgemm[v0, v1, v2, v3])
                                bgemm[v0, v1, v2, v3] = bgemm_local[v0, v1, v2, v3]
            for n_co_h_0_w_0_fused_0 in T.thread_binding(196, thread="blockIdx.x"):
                for n_co_h_0_w_0_fused_1 in T.thread_binding(64, thread="threadIdx.x"):
                    for ax0, ax1 in T.grid(1, 1):
                        for ax2 in T.unroll(4):
                            for ax3 in T.unroll(4):
                                for ax4 in T.unroll(6):
                                    for ax5 in T.unroll(6):
                                        with T.block("inverse"):
                                            v_co = T.axis.spatial(64, (n_co_h_0_w_0_fused_0 * 64 + n_co_h_0_w_0_fused_1) // 196 + ax0)
                                            v_p = T.axis.spatial(196, (n_co_h_0_w_0_fused_0 * 64 + n_co_h_0_w_0_fused_1) % 196 + ax1)
                                            v_vh, v_vw, v_r_a, v_r_b = T.axis.remap("SSRR", [ax2, ax3, ax4, ax5])
                                            T.reads(bgemm[v_r_a, v_r_b, v_co, v_p])
                                            T.writes(inverse_local[v_co, v_p, v_vh, v_vw])
                                            T.block_attr({"schedule_rule": "conv2d_nchw_winograd_inverse"})
                                            with T.init():
                                                inverse_local[v_co, v_p, v_vh, v_vw] = T.float32(0)
                                            inverse_local[v_co, v_p, v_vh, v_vw] = inverse_local[v_co, v_p, v_vh, v_vw] + bgemm[v_r_a, v_r_b, v_co, v_p] * T.Select(v_r_a % 6 == 5 and v_vh % 4 == 3, T.float32(1), T.Select(v_r_a % 6 == 5 and v_vh % 4 == 2, T.float32(0), T.Select(v_r_a % 6 == 5 and v_vh % 4 == 1, T.float32(0), T.Select(v_r_a % 6 == 5 and v_vh % 4 == 0, T.float32(0), T.Select(v_r_a % 6 == 4 and v_vh % 4 == 3, T.float32(-8), T.Select(v_r_a % 6 == 4 and v_vh % 4 == 2, T.float32(4), T.Select(v_r_a % 6 == 4 and v_vh % 4 == 1, T.float32(-2), T.Select(v_r_a % 6 == 4 and v_vh % 4 == 0, T.float32(1), T.Select(v_r_a % 6 == 3 and v_vh % 4 == 3, T.float32(0.125), T.Select(v_r_a % 6 == 3 and v_vh % 4 == 2, T.float32(0.25), T.Select(v_r_a % 6 == 3 and v_vh % 4 == 1, T.float32(0.5), T.Select(v_r_a % 6 == 3 and v_vh % 4 == 0, T.float32(1), T.Select(v_r_a % 6 == 2 and v_vh % 4 == 3, T.float32(1), T.Select(v_r_a % 6 == 2 and v_vh % 4 == 2, T.float32(1), T.Select(v_r_a % 6 == 2 and v_vh % 4 == 1, T.float32(1), T.Select(v_r_a % 6 == 2 and v_vh % 4 == 0, T.float32(1), T.Select(v_r_a % 6 == 1 and v_vh % 4 == 3, T.float32(-1), T.Select(v_r_a % 6 == 1 and v_vh % 4 == 2, T.float32(1), T.Select(v_r_a % 6 == 1 and v_vh % 4 == 1, T.float32(-1), T.Select(v_r_a % 6 == 1 and v_vh % 4 == 0, T.float32(1), T.Select(v_r_a % 6 == 0 and v_vh % 4 == 3, T.float32(0), T.Select(v_r_a % 6 == 0 and v_vh % 4 == 2, T.float32(0), T.Select(v_r_a % 6 == 0 and v_vh % 4 == 1, T.float32(0), T.Select(v_r_a % 6 == 0 and v_vh % 4 == 0, T.float32(1), T.float32(0))))))))))))))))))))))))) * T.Select(v_r_b % 6 == 5 and v_vw % 4 == 3, T.float32(1), T.Select(v_r_b % 6 == 5 and v_vw % 4 == 2, T.float32(0), T.Select(v_r_b % 6 == 5 and v_vw % 4 == 1, T.float32(0), T.Select(v_r_b % 6 == 5 and v_vw % 4 == 0, T.float32(0), T.Select(v_r_b % 6 == 4 and v_vw % 4 == 3, T.float32(-8), T.Select(v_r_b % 6 == 4 and v_vw % 4 == 2, T.float32(4), T.Select(v_r_b % 6 == 4 and v_vw % 4 == 1, T.float32(-2), T.Select(v_r_b % 6 == 4 and v_vw % 4 == 0, T.float32(1), T.Select(v_r_b % 6 == 3 and v_vw % 4 == 3, T.float32(0.125), T.Select(v_r_b % 6 == 3 and v_vw % 4 == 2, T.float32(0.25), T.Select(v_r_b % 6 == 3 and v_vw % 4 == 1, T.float32(0.5), T.Select(v_r_b % 6 == 3 and v_vw % 4 == 0, T.float32(1), T.Select(v_r_b % 6 == 2 and v_vw % 4 == 3, T.float32(1), T.Select(v_r_b % 6 == 2 and v_vw % 4 == 2, T.float32(1), T.Select(v_r_b % 6 == 2 and v_vw % 4 == 1, T.float32(1), T.Select(v_r_b % 6 == 2 and v_vw % 4 == 0, T.float32(1), T.Select(v_r_b % 6 == 1 and v_vw % 4 == 3, T.float32(-1), T.Select(v_r_b % 6 == 1 and v_vw % 4 == 2, T.float32(1), T.Select(v_r_b % 6 == 1 and v_vw % 4 == 1, T.float32(-1), T.Select(v_r_b % 6 == 1 and v_vw % 4 == 0, T.float32(1), T.Select(v_r_b % 6 == 0 and v_vw % 4 == 3, T.float32(0), T.Select(v_r_b % 6 == 0 and v_vw % 4 == 2, T.float32(0), T.Select(v_r_b % 6 == 0 and v_vw % 4 == 1, T.float32(0), T.Select(v_r_b % 6 == 0 and v_vw % 4 == 0, T.float32(1), T.float32(0)))))))))))))))))))))))))
                    for h_1, w_1 in T.grid(4, 4):
                        with T.block("conv2d_winograd"):
                            v_n = T.axis.spatial(1, 0)
                            v_co = T.axis.spatial(64, (n_co_h_0_w_0_fused_0 * 64 + n_co_h_0_w_0_fused_1) // 196)
                            v_h = T.axis.spatial(56, (n_co_h_0_w_0_fused_0 * 64 + n_co_h_0_w_0_fused_1) % 196 // 14 * 4 + h_1)
                            v_w = T.axis.spatial(56, (n_co_h_0_w_0_fused_0 * 64 + n_co_h_0_w_0_fused_1) % 14 * 4 + w_1)
                            T.reads(inverse_local[v_co, v_n * 196 + v_h // 4 * 14 + v_w // 4, v_h % 4, v_w % 4])
                            T.writes(conv2d_winograd[v_n, v_co, v_h, v_w])
                            conv2d_winograd[v_n, v_co, v_h, v_w] = inverse_local[v_co, v_n * 196 + v_h // 4 * 14 + v_w // 4, v_h % 4, v_w % 4]
    # fmt: on
    decision_0 = [
        ("SampleCategorical", 4),
        ("SamplePerfectTile", [1, 1, 1, 2, 3]),
        ("SamplePerfectTile", [1, 2, 1, 3, 1]),
        ("SamplePerfectTile", [2, 16, 2, 1, 1]),
        ("SamplePerfectTile", [7, 7, 1, 4, 1]),
        ("SamplePerfectTile", [2, 16, 2]),
        ("SampleCategorical", 3),
        ("SampleCategorical", 2),
        ("SampleCategorical", 1),
        ("SampleCategorical", 1),
    ]
    with _target():
        mod = create_te_workload("C2D_WIN_NCHW", 0)
    actual = _design_space(mod)
    check_sketches(
        mod,
        sketches=actual,
        expected_mods=[cuda_nchw_0],
        expected_decisions=[decision_0],
        debug_mask=0,
    )


def test_cuda_nchw_add_relu():
    # fmt: off
    @T.prim_func
    def nchw_add_relu(p0: T.Buffer((2, 2048, 50, 75), "float32"), p1: T.Buffer((4, 4, 2048, 2048), "float32"), p2: T.Buffer((1, 2048, 1, 1), "float32"), T_relu: T.Buffer((2, 2048, 50, 75), "float32")):
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True, "layout_free_buffers": [1]})
        # body
        # with T.block("root")
        data_pad = T.alloc_buffer([2, 2048, 52, 77], dtype="float32")
        input_tile = T.alloc_buffer([2048, 1900, 4, 4], dtype="float32")
        B = T.alloc_buffer([4, 4], dtype="float32")
        data_pack = T.alloc_buffer([4, 4, 2048, 1900], dtype="float32")
        bgemm = T.alloc_buffer([4, 4, 2048, 1900], dtype="float32")
        A = T.alloc_buffer([4, 2], dtype="float32")
        inverse = T.alloc_buffer([2048, 1900, 2, 2], dtype="float32")
        conv2d_winograd = T.alloc_buffer([2, 2048, 50, 75], dtype="float32")
        T_add = T.alloc_buffer([2, 2048, 50, 75], dtype="float32")
        for i0, i1, i2, i3 in T.grid(2, 2048, 52, 77):
            with T.block("data_pad"):
                i0_1, i1_1, i2_1, i3_1 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(p0[i0_1, i1_1, i2_1 - 1, i3_1 - 1])
                T.writes(data_pad[i0_1, i1_1, i2_1, i3_1])
                data_pad[i0_1, i1_1, i2_1, i3_1] = T.if_then_else(1 <= i2_1 and i2_1 < 51 and 1 <= i3_1 and i3_1 < 76, p0[i0_1, i1_1, i2_1 - 1, i3_1 - 1], T.float32(0), dtype="float32")
        for i0, i1, i2, i3 in T.grid(2048, 1900, 4, 4):
            with T.block("input_tile"):
                ci, p, eps, nu = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(data_pad[p // 950, ci, p % 950 // 38 * 2 + eps, p % 38 * 2 + nu])
                T.writes(input_tile[ci, p, eps, nu])
                T.block_attr({"schedule_rule":"None"})
                input_tile[ci, p, eps, nu] = data_pad[p // 950, ci, p % 950 // 38 * 2 + eps, p % 38 * 2 + nu]
        for i0, i1 in T.grid(4, 4):
            with T.block("B"):
                i, j = T.axis.remap("SS", [i0, i1])
                T.reads()
                T.writes(B[i, j])
                T.block_attr({"schedule_rule":"None"})
                B[i, j] = T.Select(i % 4 == 3 and j % 4 == 3, T.float32(1), T.Select(i % 4 == 3 and j % 4 == 2, T.float32(0), T.Select(i % 4 == 3 and j % 4 == 1, T.float32(0), T.Select(i % 4 == 3 and j % 4 == 0, T.float32(0), T.Select(i % 4 == 2 and j % 4 == 3, T.float32(0), T.Select(i % 4 == 2 and j % 4 == 2, T.float32(1), T.Select(i % 4 == 2 and j % 4 == 1, T.float32(1), T.Select(i % 4 == 2 and j % 4 == 0, T.float32(-1), T.Select(i % 4 == 1 and j % 4 == 3, T.float32(-1), T.Select(i % 4 == 1 and j % 4 == 2, T.float32(1), T.Select(i % 4 == 1 and j % 4 == 1, T.float32(-1), T.Select(i % 4 == 1 and j % 4 == 0, T.float32(0), T.Select(i % 4 == 0 and j % 4 == 3, T.float32(0), T.Select(i % 4 == 0 and j % 4 == 2, T.float32(0), T.Select(i % 4 == 0 and j % 4 == 1, T.float32(0), T.Select(i % 4 == 0 and j % 4 == 0, T.float32(1), T.float32(0)))))))))))))))))
        for i0, i1, i2, i3, i4, i5 in T.grid(4, 4, 2048, 1900, 4, 4):
            with T.block("data_pack"):
                eps, nu, ci, p, r_a, r_b = T.axis.remap("SSSSRR", [i0, i1, i2, i3, i4, i5])
                T.reads(input_tile[ci, p, r_a, r_b], B[T.min(r_a, r_b) : T.max(r_a, r_b) + 1, T.min(eps, nu) : T.max(eps, nu) + 1])
                T.writes(data_pack[eps, nu, ci, p])
                T.block_attr({"schedule_rule":"conv2d_nchw_winograd_data_pack"})
                with T.init():
                    data_pack[eps, nu, ci, p] = T.float32(0)
                data_pack[eps, nu, ci, p] = data_pack[eps, nu, ci, p] + input_tile[ci, p, r_a, r_b] * B[r_a, eps] * B[r_b, nu]
        for i0, i1, i2, i3, i4 in T.grid(4, 4, 2048, 1900, 2048):
            with T.block("bgemm"):
                eps, nu, co, p, ci = T.axis.remap("SSSSR", [i0, i1, i2, i3, i4])
                T.reads(data_pack[eps, nu, ci, p], p1[eps, nu, ci, co])
                T.writes(bgemm[eps, nu, co, p])
                with T.init():
                    bgemm[eps, nu, co, p] = T.float32(0)
                bgemm[eps, nu, co, p] = bgemm[eps, nu, co, p] + data_pack[eps, nu, ci, p] * p1[eps, nu, ci, co]
        for i0, i1 in T.grid(4, 2):
            with T.block("A"):
                i, j = T.axis.remap("SS", [i0, i1])
                T.reads()
                T.writes(A[i, j])
                T.block_attr({"schedule_rule":"None"})
                A[i, j] = T.Select(i % 4 == 3 and j % 2 == 1, T.float32(1), T.Select(i % 4 == 3 and j % 2 == 0, T.float32(0), T.Select(i % 4 == 2 and j % 2 == 1, T.float32(1), T.Select(i % 4 == 2 and j % 2 == 0, T.float32(1), T.Select(i % 4 == 1 and j % 2 == 1, T.float32(-1), T.Select(i % 4 == 1 and j % 2 == 0, T.float32(1), T.Select(i % 4 == 0 and j % 2 == 1, T.float32(0), T.Select(i % 4 == 0 and j % 2 == 0, T.float32(1), T.float32(0)))))))))
        for i0, i1, i2, i3, i4, i5 in T.grid(2048, 1900, 2, 2, 4, 4):
            with T.block("inverse"):
                co, p, vh, vw, r_a, r_b = T.axis.remap("SSSSRR", [i0, i1, i2, i3, i4, i5])
                T.reads(bgemm[r_a, r_b, co, p], A[T.min(r_a, r_b) : T.max(r_a, r_b) + 1, T.min(vh, vw) : T.max(vh, vw) + 1])
                T.writes(inverse[co, p, vh, vw])
                T.block_attr({"schedule_rule":"conv2d_nchw_winograd_inverse"})
                with T.init():
                    inverse[co, p, vh, vw] = T.float32(0)
                inverse[co, p, vh, vw] = inverse[co, p, vh, vw] + bgemm[r_a, r_b, co, p] * A[r_a, vh] * A[r_b, vw]
        for i0, i1, i2, i3 in T.grid(2, 2048, 50, 75):
            with T.block("conv2d_winograd"):
                n, co, h, w = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(inverse[co, n * 950 + h // 2 * 38 + w // 2, h % 2, w % 2])
                T.writes(conv2d_winograd[n, co, h, w])
                conv2d_winograd[n, co, h, w] = inverse[co, n * 950 + h // 2 * 38 + w // 2, h % 2, w % 2]
        for i0, i1, i2, i3 in T.grid(2, 2048, 50, 75):
            with T.block("T_add"):
                ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(conv2d_winograd[ax0, ax1, ax2, ax3], p2[0, ax1, 0, 0])
                T.writes(T_add[ax0, ax1, ax2, ax3])
                T_add[ax0, ax1, ax2, ax3] = conv2d_winograd[ax0, ax1, ax2, ax3] + p2[0, ax1, 0, 0]
        for i0, i1, i2, i3 in T.grid(2, 2048, 50, 75):
            with T.block("T_relu"):
                ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(T_add[ax0, ax1, ax2, ax3])
                T.writes(T_relu[ax0, ax1, ax2, ax3])
                T_relu[ax0, ax1, ax2, ax3] = T.max(T_add[ax0, ax1, ax2, ax3], T.float32(0))

    @T.prim_func
    def nchw_add_relu_scheduled(p0: T.Buffer((2, 2048, 50, 75), "float32"), p1: T.Buffer((4, 4, 2048, 2048), "float32"), p2: T.Buffer((1, 2048, 1, 1), "float32"), T_relu: T.Buffer((2, 2048, 50, 75), "float32")):
        T.func_attr({"global_symbol": "main", "layout_free_buffers": [1], "tir.noalias": T.bool(True)})
        with T.block("root"):
            T.reads()
            T.writes()
            T.block_attr({"meta_schedule.unroll_explicit": 1024})
            input_tile_local = T.alloc_buffer((2048, 1900, 4, 4), scope="local")
            data_pack = T.alloc_buffer((4, 4, 2048, 1900))
            bgemm = T.alloc_buffer((4, 4, 2048, 1900))
            inverse_local = T.alloc_buffer((2048, 1900, 2, 2), scope="local")
            data_pack_local = T.alloc_buffer((4, 4, 2048, 1900), scope="local")
            bgemm_local = T.alloc_buffer((4, 4, 2048, 1900), scope="local")
            data_pack_shared = T.alloc_buffer((4, 4, 2048, 1900), scope="shared")
            p1_shared = T.alloc_buffer((4, 4, 2048, 2048), scope="shared")
            for i2_i3_fused_1 in T.thread_binding(256, thread="blockIdx.x"):
                for i2_i3_fused_2 in T.thread_binding(1024, thread="threadIdx.x"):
                    for i2_i3_fused_0 in range(15):
                        for ax0, ax1, ax2, ax3 in T.grid(1, 1, 4, 4):
                            with T.block("input_tile"):
                                ci = T.axis.spatial(2048, (i2_i3_fused_0 * 262144 + i2_i3_fused_1 * 1024 + i2_i3_fused_2) // 1900 + ax0)
                                p = T.axis.spatial(1900, (i2_i3_fused_0 * 262144 + i2_i3_fused_1 * 1024 + i2_i3_fused_2) % 1900 + ax1)
                                eps, nu = T.axis.remap("SS", [ax2, ax3])
                                T.where(i2_i3_fused_0 * 256 + i2_i3_fused_1 < 3800)
                                T.reads(p0[p // 950, ci, p % 950 // 38 * 2 + eps - 1, p % 38 * 2 + nu - 1])
                                T.writes(input_tile_local[ci, p, eps, nu])
                                T.block_attr({"schedule_rule": "None"})
                                input_tile_local[ci, p, eps, nu] = T.if_then_else(1 <= p % 950 // 38 * 2 + eps and p % 950 // 38 * 2 + eps < 51 and 1 <= p % 38 * 2 + nu and p % 38 * 2 + nu < 76, p0[p // 950, ci, p % 950 // 38 * 2 + eps - 1, p % 38 * 2 + nu - 1], T.float32(0))
                        for i0 in T.unroll(4):
                            for i1 in T.unroll(4):
                                for i4 in T.unroll(4):
                                    for i5 in T.unroll(4):
                                        with T.block("data_pack"):
                                            eps, nu = T.axis.remap("SS", [i0, i1])
                                            ci = T.axis.spatial(2048, (i2_i3_fused_0 * 262144 + i2_i3_fused_1 * 1024 + i2_i3_fused_2) // 1900)
                                            p = T.axis.spatial(1900, (i2_i3_fused_0 * 262144 + i2_i3_fused_1 * 1024 + i2_i3_fused_2) % 1900)
                                            r_a, r_b = T.axis.remap("RR", [i4, i5])
                                            T.where((i2_i3_fused_0 * 256 + i2_i3_fused_1) * 1024 + i2_i3_fused_2 < 3891200)
                                            T.reads(input_tile_local[ci, p, r_a, r_b])
                                            T.writes(data_pack_local[eps, nu, ci, p])
                                            T.block_attr({"schedule_rule": "conv2d_nchw_winograd_data_pack"})
                                            with T.init():
                                                data_pack_local[eps, nu, ci, p] = T.float32(0)
                                            data_pack_local[eps, nu, ci, p] = data_pack_local[eps, nu, ci, p] + input_tile_local[ci, p, r_a, r_b] * T.Select(r_a % 4 == 3 and eps % 4 == 3, T.float32(1), T.Select(r_a % 4 == 3 and eps % 4 == 2, T.float32(0), T.Select(r_a % 4 == 3 and eps % 4 == 1, T.float32(0), T.Select(r_a % 4 == 3 and eps % 4 == 0, T.float32(0), T.Select(r_a % 4 == 2 and eps % 4 == 3, T.float32(0), T.Select(r_a % 4 == 2 and eps % 4 == 2, T.float32(1), T.Select(r_a % 4 == 2 and eps % 4 == 1, T.float32(1), T.Select(r_a % 4 == 2 and eps % 4 == 0, T.float32(-1), T.Select(r_a % 4 == 1 and eps % 4 == 3, T.float32(-1), T.Select(r_a % 4 == 1 and eps % 4 == 2, T.float32(1), T.Select(r_a % 4 == 1 and eps % 4 == 1, T.float32(-1), T.Select(r_a % 4 == 1 and eps % 4 == 0, T.float32(0), T.Select(r_a % 4 == 0 and eps % 4 == 3, T.float32(0), T.Select(r_a % 4 == 0 and eps % 4 == 2, T.float32(0), T.Select(r_a % 4 == 0 and eps % 4 == 1, T.float32(0), T.Select(r_a % 4 == 0 and eps % 4 == 0, T.float32(1), T.float32(0))))))))))))))))) * T.Select(r_b % 4 == 3 and nu % 4 == 3, T.float32(1), T.Select(r_b % 4 == 3 and nu % 4 == 2, T.float32(0), T.Select(r_b % 4 == 3 and nu % 4 == 1, T.float32(0), T.Select(r_b % 4 == 3 and nu % 4 == 0, T.float32(0), T.Select(r_b % 4 == 2 and nu % 4 == 3, T.float32(0), T.Select(r_b % 4 == 2 and nu % 4 == 2, T.float32(1), T.Select(r_b % 4 == 2 and nu % 4 == 1, T.float32(1), T.Select(r_b % 4 == 2 and nu % 4 == 0, T.float32(-1), T.Select(r_b % 4 == 1 and nu % 4 == 3, T.float32(-1), T.Select(r_b % 4 == 1 and nu % 4 == 2, T.float32(1), T.Select(r_b % 4 == 1 and nu % 4 == 1, T.float32(-1), T.Select(r_b % 4 == 1 and nu % 4 == 0, T.float32(0), T.Select(r_b % 4 == 0 and nu % 4 == 3, T.float32(0), T.Select(r_b % 4 == 0 and nu % 4 == 2, T.float32(0), T.Select(r_b % 4 == 0 and nu % 4 == 1, T.float32(0), T.Select(r_b % 4 == 0 and nu % 4 == 0, T.float32(1), T.float32(0)))))))))))))))))
                        for ax0, ax1, ax2, ax3 in T.grid(4, 4, 1, 1):
                            with T.block("data_pack_local"):
                                v0, v1 = T.axis.remap("SS", [ax0, ax1])
                                v2 = T.axis.spatial(2048, (i2_i3_fused_0 * 262144 + i2_i3_fused_1 * 1024 + i2_i3_fused_2) // 1900 + ax2)
                                v3 = T.axis.spatial(1900, (i2_i3_fused_0 * 262144 + i2_i3_fused_1 * 1024 + i2_i3_fused_2) % 1900 + ax3)
                                T.where(i2_i3_fused_0 * 256 + i2_i3_fused_1 < 3800)
                                T.reads(data_pack_local[v0, v1, v2, v3])
                                T.writes(data_pack[v0, v1, v2, v3])
                                data_pack[v0, v1, v2, v3] = data_pack_local[v0, v1, v2, v3]
            for i0_0_i1_0_i2_0_i3_0_fused in T.thread_binding(24320, thread="blockIdx.x"):
                for i0_1_i1_1_i2_1_i3_1_fused in T.thread_binding(2, thread="vthread.x"):
                    for i0_2_i1_2_i2_2_i3_2_fused in T.thread_binding(64, thread="threadIdx.x"):
                        for i4_0 in range(256):
                            for ax0_ax1_ax2_ax3_fused in range(640):
                                with T.block("data_pack_shared"):
                                    v0 = T.axis.spatial(4, i0_0_i1_0_i2_0_i3_0_fused // 12160 * 2 + ax0_ax1_ax2_ax3_fused // 320)
                                    v1 = T.axis.spatial(4, i0_0_i1_0_i2_0_i3_0_fused % 12160 // 6080 * 2 + ax0_ax1_ax2_ax3_fused % 320 // 160)
                                    v2 = T.axis.spatial(2048, i4_0 * 8 + ax0_ax1_ax2_ax3_fused % 160 // 20)
                                    v3 = T.axis.spatial(1900, i0_0_i1_0_i2_0_i3_0_fused % 95 * 20 + ax0_ax1_ax2_ax3_fused % 20)
                                    T.reads(data_pack[v0, v1, v2, v3])
                                    T.writes(data_pack_shared[v0, v1, v2, v3])
                                    T.block_attr({"meta_schedule.cooperative_fetch": 1})
                                    data_pack_shared[v0, v1, v2, v3] = data_pack[v0, v1, v2, v3]
                            for ax0_ax1_ax2_ax3_fused in range(1024):
                                with T.block("p1_shared"):
                                    v0 = T.axis.spatial(4, i0_0_i1_0_i2_0_i3_0_fused // 12160 * 2 + ax0_ax1_ax2_ax3_fused // 512)
                                    v1 = T.axis.spatial(4, i0_0_i1_0_i2_0_i3_0_fused % 12160 // 6080 * 2 + ax0_ax1_ax2_ax3_fused % 512 // 256)
                                    v2 = T.axis.spatial(2048, i4_0 * 8 + ax0_ax1_ax2_ax3_fused % 256 // 32)
                                    v3 = T.axis.spatial(2048, i0_0_i1_0_i2_0_i3_0_fused % 6080 // 95 * 32 + ax0_ax1_ax2_ax3_fused % 32)
                                    T.reads(p1[v0, v1, v2, v3])
                                    T.writes(p1_shared[v0, v1, v2, v3])
                                    T.block_attr({"meta_schedule.cooperative_fetch": 4})
                                    p1_shared[v0, v1, v2, v3] = p1[v0, v1, v2, v3]
                            for i4_1, i0_3, i1_3, i2_3, i3_3, i4_2, i0_4, i1_4, i2_4, i3_4 in T.grid(1, 1, 2, 1, 1, 8, 1, 1, 2, 5):
                                with T.block("bgemm"):
                                    eps = T.axis.spatial(4, i0_0_i1_0_i2_0_i3_0_fused // 12160 * 2 + i0_2_i1_2_i2_2_i3_2_fused // 32 + i0_3 + i0_4)
                                    nu = T.axis.spatial(4, i0_0_i1_0_i2_0_i3_0_fused % 12160 // 6080 * 2 + i1_3 + i1_4)
                                    co = T.axis.spatial(2048, i0_0_i1_0_i2_0_i3_0_fused % 6080 // 95 * 32 + i0_1_i1_1_i2_1_i3_1_fused * 16 + i0_2_i1_2_i2_2_i3_2_fused % 32 // 4 * 2 + i2_3 * 2 + i2_4)
                                    p = T.axis.spatial(1900, i0_0_i1_0_i2_0_i3_0_fused % 95 * 20 + i0_2_i1_2_i2_2_i3_2_fused % 4 * 5 + i3_3 * 5 + i3_4)
                                    ci = T.axis.reduce(2048, i4_0 * 8 + i4_1 * 8 + i4_2)
                                    T.reads(data_pack_shared[eps, nu, ci, p], p1_shared[eps, nu, ci, co])
                                    T.writes(bgemm_local[eps, nu, co, p])
                                    T.block_attr({"meta_schedule.thread_extent_high_inclusive": 1024, "meta_schedule.thread_extent_low_inclusive": 32, "meta_schedule.tiling_structure": "SSSRRSRS"})
                                    with T.init():
                                        bgemm_local[eps, nu, co, p] = T.float32(0)
                                    bgemm_local[eps, nu, co, p] = bgemm_local[eps, nu, co, p] + data_pack_shared[eps, nu, ci, p] * p1_shared[eps, nu, ci, co]
                        for ax0, ax1, ax2, ax3 in T.grid(1, 2, 2, 5):
                            with T.block("bgemm_local"):
                                v0 = T.axis.spatial(4, i0_0_i1_0_i2_0_i3_0_fused // 12160 * 2 + i0_2_i1_2_i2_2_i3_2_fused // 32 + ax0)
                                v1 = T.axis.spatial(4, i0_0_i1_0_i2_0_i3_0_fused % 12160 // 6080 * 2 + ax1)
                                v2 = T.axis.spatial(2048, i0_0_i1_0_i2_0_i3_0_fused % 6080 // 95 * 32 + i0_1_i1_1_i2_1_i3_1_fused * 16 + i0_2_i1_2_i2_2_i3_2_fused % 32 // 4 * 2 + ax2)
                                v3 = T.axis.spatial(1900, i0_0_i1_0_i2_0_i3_0_fused % 95 * 20 + i0_2_i1_2_i2_2_i3_2_fused % 4 * 5 + ax3)
                                T.reads(bgemm_local[v0, v1, v2, v3])
                                T.writes(bgemm[v0, v1, v2, v3])
                                bgemm[v0, v1, v2, v3] = bgemm_local[v0, v1, v2, v3]
            for i0_i1_i2_0_i3_0_fused_1 in T.thread_binding(256, thread="blockIdx.x"):
                for i0_i1_i2_0_i3_0_fused_2 in T.thread_binding(1024, thread="threadIdx.x"):
                    for i0_i1_i2_0_i3_0_fused_0 in range(15):
                        for ax0, ax1 in T.grid(1, 1):
                            for ax2 in T.unroll(2):
                                for ax3 in T.unroll(2):
                                    for ax4 in T.unroll(4):
                                        for ax5 in T.unroll(4):
                                            with T.block("inverse"):
                                                co = T.axis.spatial(2048, (i0_i1_i2_0_i3_0_fused_0 * 262144 + i0_i1_i2_0_i3_0_fused_1 * 1024 + i0_i1_i2_0_i3_0_fused_2) % 1945600 // 950 + ax0)
                                                p = T.axis.spatial(1900, (i0_i1_i2_0_i3_0_fused_0 * 262144 + i0_i1_i2_0_i3_0_fused_1 * 1024 + i0_i1_i2_0_i3_0_fused_2) // 1945600 * 950 + (i0_i1_i2_0_i3_0_fused_0 * 262144 + i0_i1_i2_0_i3_0_fused_1 * 1024 + i0_i1_i2_0_i3_0_fused_2) % 950 + ax1)
                                                vh, vw, r_a, r_b = T.axis.remap("SSRR", [ax2, ax3, ax4, ax5])
                                                T.where((i0_i1_i2_0_i3_0_fused_0 * 256 + i0_i1_i2_0_i3_0_fused_1) * 1024 + i0_i1_i2_0_i3_0_fused_2 < 3891200)
                                                T.reads(bgemm[r_a, r_b, co, p])
                                                T.writes(inverse_local[co, p, vh, vw])
                                                T.block_attr({"schedule_rule": "conv2d_nchw_winograd_inverse"})
                                                with T.init():
                                                    inverse_local[co, p, vh, vw] = T.float32(0)
                                                inverse_local[co, p, vh, vw] = inverse_local[co, p, vh, vw] + bgemm[r_a, r_b, co, p] * T.Select(r_a % 4 == 3 and vh % 2 == 1, T.float32(1), T.Select(r_a % 4 == 3 and vh % 2 == 0, T.float32(0), T.Select(r_a % 4 == 2 and vh % 2 == 1, T.float32(1), T.Select(r_a % 4 == 2 and vh % 2 == 0, T.float32(1), T.Select(r_a % 4 == 1 and vh % 2 == 1, T.float32(-1), T.Select(r_a % 4 == 1 and vh % 2 == 0, T.float32(1), T.Select(r_a % 4 == 0 and vh % 2 == 1, T.float32(0), T.Select(r_a % 4 == 0 and vh % 2 == 0, T.float32(1), T.float32(0))))))))) * T.Select(r_b % 4 == 3 and vw % 2 == 1, T.float32(1), T.Select(r_b % 4 == 3 and vw % 2 == 0, T.float32(0), T.Select(r_b % 4 == 2 and vw % 2 == 1, T.float32(1), T.Select(r_b % 4 == 2 and vw % 2 == 0, T.float32(1), T.Select(r_b % 4 == 1 and vw % 2 == 1, T.float32(-1), T.Select(r_b % 4 == 1 and vw % 2 == 0, T.float32(1), T.Select(r_b % 4 == 0 and vw % 2 == 1, T.float32(0), T.Select(r_b % 4 == 0 and vw % 2 == 0, T.float32(1), T.float32(0)))))))))
                        for i2_1, i3_1 in T.grid(2, 2):
                            with T.block("conv2d_winograd"):
                                n = T.axis.spatial(2, (i0_i1_i2_0_i3_0_fused_0 * 262144 + i0_i1_i2_0_i3_0_fused_1 * 1024 + i0_i1_i2_0_i3_0_fused_2) // 1945600)
                                co = T.axis.spatial(2048, (i0_i1_i2_0_i3_0_fused_0 * 262144 + i0_i1_i2_0_i3_0_fused_1 * 1024 + i0_i1_i2_0_i3_0_fused_2) % 1945600 // 950)
                                h = T.axis.spatial(50, (i0_i1_i2_0_i3_0_fused_0 * 262144 + i0_i1_i2_0_i3_0_fused_1 * 1024 + i0_i1_i2_0_i3_0_fused_2) % 950 // 38 * 2 + i2_1)
                                w = T.axis.spatial(75, (i0_i1_i2_0_i3_0_fused_0 * 262144 + i0_i1_i2_0_i3_0_fused_1 * 1024 + i0_i1_i2_0_i3_0_fused_2) % 38 * 2 + i3_1)
                                T.where(((i0_i1_i2_0_i3_0_fused_0 * 256 + i0_i1_i2_0_i3_0_fused_1) * 1024 + i0_i1_i2_0_i3_0_fused_2) % 38 * 2 + i3_1 < 75 and (i0_i1_i2_0_i3_0_fused_0 * 256 + i0_i1_i2_0_i3_0_fused_1) * 1024 + i0_i1_i2_0_i3_0_fused_2 < 3891200)
                                T.reads(inverse_local[co, n * 950 + h // 2 * 38 + w // 2, h % 2, w % 2], p2[0, co, 0, 0])
                                T.writes(T_relu[n, co, h, w])
                                T_relu[n, co, h, w] = T.max(inverse_local[co, n * 950 + h // 2 * 38 + w // 2, h % 2, w % 2] + p2[0, co, 0, 0], T.float32(0))

    # fmt: on
    decision_0 = [
        ("SamplePerfectTile", [2, 1, 2, 1, 1]),
        ("SamplePerfectTile", [2, 1, 1, 2, 1]),
        ("SamplePerfectTile", [64, 2, 8, 1, 2]),
        ("SamplePerfectTile", [95, 1, 4, 1, 5]),
        ("SamplePerfectTile", [256, 1, 8]),
        ("SampleCategorical", 0),
        ("SampleCategorical", 3),
        ("SampleCategorical", 4),
    ]
    with _target():
        mod = nchw_add_relu
    actual = _design_space(mod)
    check_sketches(
        mod,
        sketches=actual,
        expected_mods=[nchw_add_relu_scheduled],
        expected_decisions=[decision_0],
        debug_mask=0,
    )


if __name__ == "__main__":
    test_cuda_nhwc()
    test_cuda_nchw()
    test_cuda_nchw_add_relu()
