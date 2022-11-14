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
    return Target("nvidia/geforce-rtx-3070")


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
    def cuda_nhwc_0(data: T.Buffer[(1, 14, 14, 128), "float32"], weight: T.Buffer[(6, 6, 128, 128), "float32"], conv2d_winograd: T.Buffer[(1, 12, 12, 128), "float32"]) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True, "layout_free_buffers": [1]})
        # body
        with T.block("root"):
            T.reads()
            T.writes()
            T.block_attr({"meta_schedule.unroll_explicit":16})
            input_tile_local = T.alloc_buffer([6, 6, 9, 128], dtype="float32", scope="local")
            data_pack = T.alloc_buffer([6, 6, 9, 128], dtype="float32")
            bgemm = T.alloc_buffer([6, 6, 9, 128], dtype="float32")
            inverse = T.alloc_buffer([4, 4, 9, 128], dtype="float32")
            data_pack_local = T.alloc_buffer([6, 6, 9, 128], dtype="float32", scope="local")
            bgemm_local = T.alloc_buffer([6, 6, 9, 128], dtype="float32", scope="local")
            data_pack_shared = T.alloc_buffer([6, 6, 9, 128], dtype="float32", scope="shared")
            weight_shared = T.alloc_buffer([6, 6, 128, 128], dtype="float32", scope="shared")
            for i2_0_i3_0_i2_1_i3_1_fused_0 in T.thread_binding(2, thread="blockIdx.x"):
                for i2_0_i3_0_i2_1_i3_1_fused_1 in T.thread_binding(1024, thread="threadIdx.x"):
                    for ax0, ax1, ax2, ax3 in T.grid(6, 6, 1, 1):
                        with T.block("input_tile"):
                            T.where(i2_0_i3_0_i2_1_i3_1_fused_0 * 1024 + i2_0_i3_0_i2_1_i3_1_fused_1 < 1152)
                            eps, nu = T.axis.remap("SS", [ax0, ax1])
                            p = T.axis.spatial(9, (i2_0_i3_0_i2_1_i3_1_fused_0 * 1024 + i2_0_i3_0_i2_1_i3_1_fused_1) // 384 * 3 + (i2_0_i3_0_i2_1_i3_1_fused_0 * 1024 + i2_0_i3_0_i2_1_i3_1_fused_1) % 24 // 8 + ax2)
                            ci = T.axis.spatial(128, (i2_0_i3_0_i2_1_i3_1_fused_0 * 1024 + i2_0_i3_0_i2_1_i3_1_fused_1) % 384 // 24 * 8 + (i2_0_i3_0_i2_1_i3_1_fused_0 * 1024 + i2_0_i3_0_i2_1_i3_1_fused_1) % 8 + ax3)
                            T.reads(data[p // 9, p % 9 // 3 * 4 + eps, p % 3 * 4 + nu, ci])
                            T.writes(input_tile_local[eps, nu, p, ci])
                            T.block_attr({"schedule_rule":"None"})
                            input_tile_local[eps, nu, p, ci] = T.if_then_else(0 <= p % 9 // 3 * 4 + eps and p % 9 // 3 * 4 + eps < 14 and 0 <= p % 3 * 4 + nu and p % 3 * 4 + nu < 14, data[p // 9, p % 9 // 3 * 4 + eps, p % 3 * 4 + nu, ci], T.float32(0), dtype="float32")
                    for i0 in T.unroll(6):
                        for i1 in T.unroll(6):
                            for i4 in T.unroll(6):
                                for i5 in T.unroll(6):
                                    with T.block("data_pack"):
                                        T.where(i2_0_i3_0_i2_1_i3_1_fused_0 * 1024 + i2_0_i3_0_i2_1_i3_1_fused_1 < 1152)
                                        eps, nu = T.axis.remap("SS", [i0, i1])
                                        p = T.axis.spatial(9, (i2_0_i3_0_i2_1_i3_1_fused_0 * 1024 + i2_0_i3_0_i2_1_i3_1_fused_1) // 384 * 3 + (i2_0_i3_0_i2_1_i3_1_fused_0 * 1024 + i2_0_i3_0_i2_1_i3_1_fused_1) % 24 // 8)
                                        ci = T.axis.spatial(128, (i2_0_i3_0_i2_1_i3_1_fused_0 * 1024 + i2_0_i3_0_i2_1_i3_1_fused_1) % 384 // 24 * 8 + (i2_0_i3_0_i2_1_i3_1_fused_0 * 1024 + i2_0_i3_0_i2_1_i3_1_fused_1) % 8)
                                        r_a, r_b = T.axis.remap("RR", [i4, i5])
                                        T.reads(input_tile_local[r_a, r_b, p, ci])
                                        T.writes(data_pack_local[eps, nu, p, ci])
                                        T.block_attr({"schedule_rule":"conv2d_nhwc_winograd_data_pack"})
                                        with T.init():
                                            data_pack_local[eps, nu, p, ci] = T.float32(0)
                                        data_pack_local[eps, nu, p, ci] = data_pack_local[eps, nu, p, ci] + input_tile_local[r_a, r_b, p, ci] * T.Select(r_a % 6 == 5 and eps % 6 == 5, T.float32(1), T.Select(r_a % 6 == 5 and eps % 6 == 4, T.float32(0), T.Select(r_a % 6 == 5 and eps % 6 == 3, T.float32(0), T.Select(r_a % 6 == 5 and eps % 6 == 2, T.float32(0), T.Select(r_a % 6 == 5 and eps % 6 == 1, T.float32(0), T.Select(r_a % 6 == 5 and eps % 6 == 0, T.float32(0), T.Select(r_a % 6 == 4 and eps % 6 == 5, T.float32(1.5), T.Select(r_a % 6 == 4 and eps % 6 == 4, T.float32(1), T.Select(r_a % 6 == 4 and eps % 6 == 3, T.float32(1), T.Select(r_a % 6 == 4 and eps % 6 == 2, T.float32(1), T.Select(r_a % 6 == 4 and eps % 6 == 1, T.float32(1), T.Select(r_a % 6 == 4 and eps % 6 == 0, T.float32(1), T.Select(r_a % 6 == 3 and eps % 6 == 5, T.float32(-2), T.Select(r_a % 6 == 3 and eps % 6 == 4, T.float32(-0.5), T.Select(r_a % 6 == 3 and eps % 6 == 3, T.float32(2), T.Select(r_a % 6 == 3 and eps % 6 == 2, T.float32(2.5), T.Select(r_a % 6 == 3 and eps % 6 == 1, T.float32(0.5), T.Select(r_a % 6 == 3 and eps % 6 == 0, T.float32(1.5), T.Select(r_a % 6 == 2 and eps % 6 == 5, T.float32(-1.5), T.Select(r_a % 6 == 2 and eps % 6 == 4, T.float32(-1), T.Select(r_a % 6 == 2 and eps % 6 == 3, T.float32(-1), T.Select(r_a % 6 == 2 and eps % 6 == 2, T.float32(0.5), T.Select(r_a % 6 == 2 and eps % 6 == 1, T.float32(-2.5), T.Select(r_a % 6 == 2 and eps % 6 == 0, T.float32(-2), T.Select(r_a % 6 == 1 and eps % 6 == 5, T.float32(1), T.Select(r_a % 6 == 1 and eps % 6 == 4, T.float32(0.5), T.Select(r_a % 6 == 1 and eps % 6 == 3, T.float32(-2), T.Select(r_a % 6 == 1 and eps % 6 == 2, T.float32(-1), T.Select(r_a % 6 == 1 and eps % 6 == 1, T.float32(1), T.Select(r_a % 6 == 1 and eps % 6 == 0, T.float32(-1.5), T.Select(r_a % 6 == 0 and eps % 6 == 5, T.float32(0), T.Select(r_a % 6 == 0 and eps % 6 == 4, T.float32(0), T.Select(r_a % 6 == 0 and eps % 6 == 3, T.float32(0), T.Select(r_a % 6 == 0 and eps % 6 == 2, T.float32(0), T.Select(r_a % 6 == 0 and eps % 6 == 1, T.float32(0), T.Select(r_a % 6 == 0 and eps % 6 == 0, T.float32(1), T.float32(0))))))))))))))))))))))))))))))))))))) * T.Select(r_b % 6 == 5 and nu % 6 == 5, T.float32(1), T.Select(r_b % 6 == 5 and nu % 6 == 4, T.float32(0), T.Select(r_b % 6 == 5 and nu % 6 == 3, T.float32(0), T.Select(r_b % 6 == 5 and nu % 6 == 2, T.float32(0), T.Select(r_b % 6 == 5 and nu % 6 == 1, T.float32(0), T.Select(r_b % 6 == 5 and nu % 6 == 0, T.float32(0), T.Select(r_b % 6 == 4 and nu % 6 == 5, T.float32(1.5), T.Select(r_b % 6 == 4 and nu % 6 == 4, T.float32(1), T.Select(r_b % 6 == 4 and nu % 6 == 3, T.float32(1), T.Select(r_b % 6 == 4 and nu % 6 == 2, T.float32(1), T.Select(r_b % 6 == 4 and nu % 6 == 1, T.float32(1), T.Select(r_b % 6 == 4 and nu % 6 == 0, T.float32(1), T.Select(r_b % 6 == 3 and nu % 6 == 5, T.float32(-2), T.Select(r_b % 6 == 3 and nu % 6 == 4, T.float32(-0.5), T.Select(r_b % 6 == 3 and nu % 6 == 3, T.float32(2), T.Select(r_b % 6 == 3 and nu % 6 == 2, T.float32(2.5), T.Select(r_b % 6 == 3 and nu % 6 == 1, T.float32(0.5), T.Select(r_b % 6 == 3 and nu % 6 == 0, T.float32(1.5), T.Select(r_b % 6 == 2 and nu % 6 == 5, T.float32(-1.5), T.Select(r_b % 6 == 2 and nu % 6 == 4, T.float32(-1), T.Select(r_b % 6 == 2 and nu % 6 == 3, T.float32(-1), T.Select(r_b % 6 == 2 and nu % 6 == 2, T.float32(0.5), T.Select(r_b % 6 == 2 and nu % 6 == 1, T.float32(-2.5), T.Select(r_b % 6 == 2 and nu % 6 == 0, T.float32(-2), T.Select(r_b % 6 == 1 and nu % 6 == 5, T.float32(1), T.Select(r_b % 6 == 1 and nu % 6 == 4, T.float32(0.5), T.Select(r_b % 6 == 1 and nu % 6 == 3, T.float32(-2), T.Select(r_b % 6 == 1 and nu % 6 == 2, T.float32(-1), T.Select(r_b % 6 == 1 and nu % 6 == 1, T.float32(1), T.Select(r_b % 6 == 1 and nu % 6 == 0, T.float32(-1.5), T.Select(r_b % 6 == 0 and nu % 6 == 5, T.float32(0), T.Select(r_b % 6 == 0 and nu % 6 == 4, T.float32(0), T.Select(r_b % 6 == 0 and nu % 6 == 3, T.float32(0), T.Select(r_b % 6 == 0 and nu % 6 == 2, T.float32(0), T.Select(r_b % 6 == 0 and nu % 6 == 1, T.float32(0), T.Select(r_b % 6 == 0 and nu % 6 == 0, T.float32(1), T.float32(0)))))))))))))))))))))))))))))))))))))
                    for ax0, ax1, ax2, ax3 in T.grid(6, 6, 1, 1):
                        with T.block("data_pack_local"):
                            T.where(i2_0_i3_0_i2_1_i3_1_fused_0 * 1024 + i2_0_i3_0_i2_1_i3_1_fused_1 < 1152)
                            v0, v1 = T.axis.remap("SS", [ax0, ax1])
                            v2 = T.axis.spatial(9, (i2_0_i3_0_i2_1_i3_1_fused_0 * 1024 + i2_0_i3_0_i2_1_i3_1_fused_1) // 384 * 3 + (i2_0_i3_0_i2_1_i3_1_fused_0 * 1024 + i2_0_i3_0_i2_1_i3_1_fused_1) % 24 // 8 + ax2)
                            v3 = T.axis.spatial(128, (i2_0_i3_0_i2_1_i3_1_fused_0 * 1024 + i2_0_i3_0_i2_1_i3_1_fused_1) % 384 // 24 * 8 + (i2_0_i3_0_i2_1_i3_1_fused_0 * 1024 + i2_0_i3_0_i2_1_i3_1_fused_1) % 8 + ax3)
                            T.reads(data_pack_local[v0, v1, v2, v3])
                            T.writes(data_pack[v0, v1, v2, v3])
                            data_pack[v0, v1, v2, v3] = data_pack_local[v0, v1, v2, v3]
            for i0_0_i1_0_i2_0_i3_0_fused in T.thread_binding(96, thread="blockIdx.x"):
                for i0_1_i1_1_i2_1_i3_1_fused in T.thread_binding(4, thread="vthread.x"):
                    for i0_2_i1_2_i2_2_i3_2_fused in T.thread_binding(27, thread="threadIdx.x"):
                        for i4_0 in T.serial(8):
                            for ax0_ax1_ax2_ax3_fused in T.serial(1728):
                                with T.block("data_pack_shared"):
                                    v0 = T.axis.spatial(6, i0_0_i1_0_i2_0_i3_0_fused // 32 * 2 + ax0_ax1_ax2_ax3_fused // 864)
                                    v1 = T.axis.spatial(6, ax0_ax1_ax2_ax3_fused % 864 // 144)
                                    v2 = T.axis.spatial(9, ax0_ax1_ax2_ax3_fused % 144 // 16)
                                    v3 = T.axis.spatial(128, i4_0 * 16 + ax0_ax1_ax2_ax3_fused % 16)
                                    T.reads(data_pack[v0, v1, v2, v3])
                                    T.writes(data_pack_shared[v0, v1, v2, v3])
                                    T.block_attr({"meta_schedule.cooperative_fetch":1})
                                    data_pack_shared[v0, v1, v2, v3] = data_pack[v0, v1, v2, v3]
                            for ax0_ax1_ax2_ax3_fused in T.serial(768):
                                with T.block("weight_shared"):
                                    v0 = T.axis.spatial(6, i0_0_i1_0_i2_0_i3_0_fused // 32 * 2 + ax0_ax1_ax2_ax3_fused // 384)
                                    v1 = T.axis.spatial(6, ax0_ax1_ax2_ax3_fused % 384 // 64)
                                    v2 = T.axis.spatial(128, i0_0_i1_0_i2_0_i3_0_fused % 32 * 4 + ax0_ax1_ax2_ax3_fused % 64 // 16)
                                    v3 = T.axis.spatial(128, i4_0 * 16 + ax0_ax1_ax2_ax3_fused % 16)
                                    T.reads(weight[v0, v1, v2, v3])
                                    T.writes(weight_shared[v0, v1, v2, v3])
                                    T.block_attr({"meta_schedule.cooperative_fetch":3})
                                    weight_shared[v0, v1, v2, v3] = weight[v0, v1, v2, v3]
                            for i4_1, i0_3, i1_3, i2_3, i3_3, i4_2, i0_4, i1_4, i2_4, i3_4 in T.grid(1, 2, 1, 1, 2, 16, 1, 1, 1, 1):
                                with T.block("bgemm"):
                                    eps = T.axis.spatial(6, i0_0_i1_0_i2_0_i3_0_fused // 32 * 2 + i0_3 + i0_4)
                                    nu = T.axis.spatial(6, i1_3 + i1_4 + i0_1_i1_1_i2_1_i3_1_fused // 2 * 3 + i0_2_i1_2_i2_2_i3_2_fused // 9)
                                    p = T.axis.spatial(9, i0_2_i1_2_i2_2_i3_2_fused % 9 + i2_3 + i2_4)
                                    co = T.axis.spatial(128, i3_4 + i0_0_i1_0_i2_0_i3_0_fused % 32 * 4 + i0_1_i1_1_i2_1_i3_1_fused % 2 * 2 + i3_3)
                                    ci = T.axis.reduce(128, i4_0 * 16 + i4_1 * 16 + i4_2)
                                    T.reads(data_pack_shared[eps, nu, p, ci], weight_shared[eps, nu, co, ci])
                                    T.writes(bgemm_local[eps, nu, p, co])
                                    T.block_attr({"meta_schedule.thread_extent_high_inclusive":1024, "meta_schedule.thread_extent_low_inclusive":32, "meta_schedule.tiling_structure":"SSSRRSRS", "meta_schedule.write_cache_level":[3]})
                                    with T.init():
                                        bgemm_local[eps, nu, p, co] = T.float32(0)
                                    bgemm_local[eps, nu, p, co] = bgemm_local[eps, nu, p, co] + data_pack_shared[eps, nu, p, ci] * weight_shared[eps, nu, co, ci]
                        for ax0, ax1, ax2, ax3 in T.grid(2, 1, 1, 2):
                            with T.block("bgemm_local"):
                                v0 = T.axis.spatial(6, i0_0_i1_0_i2_0_i3_0_fused // 32 * 2 + ax0)
                                v1 = T.axis.spatial(6, i0_1_i1_1_i2_1_i3_1_fused // 2 * 3 + i0_2_i1_2_i2_2_i3_2_fused // 9 + ax1)
                                v2 = T.axis.spatial(9, i0_2_i1_2_i2_2_i3_2_fused % 9 + ax2)
                                v3 = T.axis.spatial(128, i0_0_i1_0_i2_0_i3_0_fused % 32 * 4 + i0_1_i1_1_i2_1_i3_1_fused % 2 * 2 + ax3)
                                T.reads(bgemm_local[v0, v1, v2, v3])
                                T.writes(bgemm[v0, v1, v2, v3])
                                bgemm[v0, v1, v2, v3] = bgemm_local[v0, v1, v2, v3]
            for i2_0_i3_0_i2_1_i3_1_fused_0 in T.thread_binding(18, thread="blockIdx.x"):
                for i2_0_i3_0_i2_1_i3_1_fused_1 in T.thread_binding(64, thread="threadIdx.x"):
                    for i0 in T.unroll(4):
                        for i1 in T.unroll(4):
                            for i4 in T.unroll(6):
                                for i5 in T.unroll(6):
                                    with T.block("inverse"):
                                        vh, vw = T.axis.remap("SS", [i0, i1])
                                        p = T.axis.spatial(9, (i2_0_i3_0_i2_1_i3_1_fused_0 * 64 + i2_0_i3_0_i2_1_i3_1_fused_1) // 384 * 3 + (i2_0_i3_0_i2_1_i3_1_fused_0 * 64 + i2_0_i3_0_i2_1_i3_1_fused_1) % 24 // 8)
                                        co = T.axis.spatial(128, (i2_0_i3_0_i2_1_i3_1_fused_0 * 64 + i2_0_i3_0_i2_1_i3_1_fused_1) % 384 // 24 * 8 + (i2_0_i3_0_i2_1_i3_1_fused_0 * 64 + i2_0_i3_0_i2_1_i3_1_fused_1) % 8)
                                        r_a, r_b = T.axis.remap("RR", [i4, i5])
                                        T.reads(bgemm[r_a, r_b, p, co])
                                        T.writes(inverse[vh, vw, p, co])
                                        T.block_attr({"schedule_rule":"conv2d_nhwc_winograd_inverse"})
                                        with T.init():
                                            inverse[vh, vw, p, co] = T.float32(0)
                                        inverse[vh, vw, p, co] = inverse[vh, vw, p, co] + bgemm[r_a, r_b, p, co] * T.Select(r_a % 6 == 5 and vh % 4 == 3, T.float32(1), T.Select(r_a % 6 == 5 and vh % 4 == 2, T.float32(0), T.Select(r_a % 6 == 5 and vh % 4 == 1, T.float32(0), T.Select(r_a % 6 == 5 and vh % 4 == 0, T.float32(0), T.Select(r_a % 6 == 4 and vh % 4 == 3, T.float32(-8), T.Select(r_a % 6 == 4 and vh % 4 == 2, T.float32(4), T.Select(r_a % 6 == 4 and vh % 4 == 1, T.float32(-2), T.Select(r_a % 6 == 4 and vh % 4 == 0, T.float32(1), T.Select(r_a % 6 == 3 and vh % 4 == 3, T.float32(0.125), T.Select(r_a % 6 == 3 and vh % 4 == 2, T.float32(0.25), T.Select(r_a % 6 == 3 and vh % 4 == 1, T.float32(0.5), T.Select(r_a % 6 == 3 and vh % 4 == 0, T.float32(1), T.Select(r_a % 6 == 2 and vh % 4 == 3, T.float32(1), T.Select(r_a % 6 == 2 and vh % 4 == 2, T.float32(1), T.Select(r_a % 6 == 2 and vh % 4 == 1, T.float32(1), T.Select(r_a % 6 == 2 and vh % 4 == 0, T.float32(1), T.Select(r_a % 6 == 1 and vh % 4 == 3, T.float32(-1), T.Select(r_a % 6 == 1 and vh % 4 == 2, T.float32(1), T.Select(r_a % 6 == 1 and vh % 4 == 1, T.float32(-1), T.Select(r_a % 6 == 1 and vh % 4 == 0, T.float32(1), T.Select(r_a % 6 == 0 and vh % 4 == 3, T.float32(0), T.Select(r_a % 6 == 0 and vh % 4 == 2, T.float32(0), T.Select(r_a % 6 == 0 and vh % 4 == 1, T.float32(0), T.Select(r_a % 6 == 0 and vh % 4 == 0, T.float32(1), T.float32(0))))))))))))))))))))))))) * T.Select(r_b % 6 == 5 and vw % 4 == 3, T.float32(1), T.Select(r_b % 6 == 5 and vw % 4 == 2, T.float32(0), T.Select(r_b % 6 == 5 and vw % 4 == 1, T.float32(0), T.Select(r_b % 6 == 5 and vw % 4 == 0, T.float32(0), T.Select(r_b % 6 == 4 and vw % 4 == 3, T.float32(-8), T.Select(r_b % 6 == 4 and vw % 4 == 2, T.float32(4), T.Select(r_b % 6 == 4 and vw % 4 == 1, T.float32(-2), T.Select(r_b % 6 == 4 and vw % 4 == 0, T.float32(1), T.Select(r_b % 6 == 3 and vw % 4 == 3, T.float32(0.125), T.Select(r_b % 6 == 3 and vw % 4 == 2, T.float32(0.25), T.Select(r_b % 6 == 3 and vw % 4 == 1, T.float32(0.5), T.Select(r_b % 6 == 3 and vw % 4 == 0, T.float32(1), T.Select(r_b % 6 == 2 and vw % 4 == 3, T.float32(1), T.Select(r_b % 6 == 2 and vw % 4 == 2, T.float32(1), T.Select(r_b % 6 == 2 and vw % 4 == 1, T.float32(1), T.Select(r_b % 6 == 2 and vw % 4 == 0, T.float32(1), T.Select(r_b % 6 == 1 and vw % 4 == 3, T.float32(-1), T.Select(r_b % 6 == 1 and vw % 4 == 2, T.float32(1), T.Select(r_b % 6 == 1 and vw % 4 == 1, T.float32(-1), T.Select(r_b % 6 == 1 and vw % 4 == 0, T.float32(1), T.Select(r_b % 6 == 0 and vw % 4 == 3, T.float32(0), T.Select(r_b % 6 == 0 and vw % 4 == 2, T.float32(0), T.Select(r_b % 6 == 0 and vw % 4 == 1, T.float32(0), T.Select(r_b % 6 == 0 and vw % 4 == 0, T.float32(1), T.float32(0)))))))))))))))))))))))))
            for i0_i1_i2_i3_fused_0 in T.thread_binding(144, thread="blockIdx.x"):
                for i0_i1_i2_i3_fused_1 in T.thread_binding(128, thread="threadIdx.x"):
                    with T.block("conv2d_winograd"):
                        n = T.axis.spatial(1, 0)
                        h = T.axis.spatial(12, (i0_i1_i2_i3_fused_0 * 128 + i0_i1_i2_i3_fused_1) // 1536)
                        w = T.axis.spatial(12, (i0_i1_i2_i3_fused_0 * 128 + i0_i1_i2_i3_fused_1) % 1536 // 128)
                        co = T.axis.spatial(128, (i0_i1_i2_i3_fused_0 * 128 + i0_i1_i2_i3_fused_1) % 128)
                        T.reads(inverse[h % 4, w % 4, n * 9 + h // 4 * 3 + w // 4, co])
                        T.writes(conv2d_winograd[n, h, w, co])
                        conv2d_winograd[n, h, w, co] = inverse[h % 4, w % 4, n * 9 + h // 4 * 3 + w // 4, co]
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
    def cuda_nchw_0(data: T.Buffer[(1, 64, 56, 56), "float32"], weight: T.Buffer[(6, 6, 64, 64), "float32"], conv2d_winograd: T.Buffer[(1, 64, 56, 56), "float32"]) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True, "layout_free_buffers": [1]})
        # body
        with T.block("root"):
            T.reads()
            T.writes()
            T.block_attr({"meta_schedule.unroll_explicit":16})
            input_tile_local = T.alloc_buffer([64, 196, 6, 6], dtype="float32", scope="local")
            data_pack = T.alloc_buffer([6, 6, 64, 196], dtype="float32")
            bgemm = T.alloc_buffer([6, 6, 64, 196], dtype="float32")
            inverse_local = T.alloc_buffer([64, 196, 4, 4], dtype="float32", scope="local")
            data_pack_local = T.alloc_buffer([6, 6, 64, 196], dtype="float32", scope="local")
            bgemm_local = T.alloc_buffer([6, 6, 64, 196], dtype="float32", scope="local")
            data_pack_shared = T.alloc_buffer([6, 6, 64, 196], dtype="float32", scope="shared")
            weight_shared = T.alloc_buffer([6, 6, 64, 64], dtype="float32", scope="shared")
            for i2_i3_fused_0 in T.thread_binding(25, thread="blockIdx.x"):
                for i2_i3_fused_1 in T.thread_binding(512, thread="threadIdx.x"):
                    for ax0, ax1, ax2, ax3 in T.grid(1, 1, 6, 6):
                        with T.block("input_tile"):
                            T.where(i2_i3_fused_0 * 512 + i2_i3_fused_1 < 12544)
                            ci = T.axis.spatial(64, (i2_i3_fused_0 * 512 + i2_i3_fused_1) // 196 + ax0)
                            p = T.axis.spatial(196, (i2_i3_fused_0 * 120 + i2_i3_fused_1) % 196 + ax1)
                            eps, nu = T.axis.remap("SS", [ax2, ax3])
                            T.reads(data[p // 196, ci, p % 196 // 14 * 4 + eps - 1, p % 14 * 4 + nu - 1])
                            T.writes(input_tile_local[ci, p, eps, nu])
                            T.block_attr({"schedule_rule":"None"})
                            input_tile_local[ci, p, eps, nu] = T.if_then_else(1 <= p % 196 // 14 * 4 + eps and p % 196 // 14 * 4 + eps < 57 and 1 <= p % 14 * 4 + nu and p % 14 * 4 + nu < 57, data[p // 196, ci, p % 196 // 14 * 4 + eps - 1, p % 14 * 4 + nu - 1], T.float32(0), dtype="float32")
                    for i0 in T.unroll(6):
                        for i1 in T.unroll(6):
                            for i4 in T.unroll(6):
                                for i5 in T.unroll(6):
                                    with T.block("data_pack"):
                                        T.where(i2_i3_fused_0 * 512 + i2_i3_fused_1 < 12544)
                                        eps, nu = T.axis.remap("SS", [i0, i1])
                                        ci = T.axis.spatial(64, (i2_i3_fused_0 * 512 + i2_i3_fused_1) // 196)
                                        p = T.axis.spatial(196, (i2_i3_fused_0 * 512 + i2_i3_fused_1) % 196)
                                        r_a, r_b = T.axis.remap("RR", [i4, i5])
                                        T.reads(input_tile_local[ci, p, r_a, r_b])
                                        T.writes(data_pack_local[eps, nu, ci, p])
                                        T.block_attr({"schedule_rule":"conv2d_nchw_winograd_data_pack"})
                                        with T.init():
                                            data_pack_local[eps, nu, ci, p] = T.float32(0)
                                        data_pack_local[eps, nu, ci, p] = data_pack_local[eps, nu, ci, p] + input_tile_local[ci, p, r_a, r_b] * T.Select(r_a % 6 == 5 and eps % 6 == 5, T.float32(1), T.Select(r_a % 6 == 5 and eps % 6 == 4, T.float32(0), T.Select(r_a % 6 == 5 and eps % 6 == 3, T.float32(0), T.Select(r_a % 6 == 5 and eps % 6 == 2, T.float32(0), T.Select(r_a % 6 == 5 and eps % 6 == 1, T.float32(0), T.Select(r_a % 6 == 5 and eps % 6 == 0, T.float32(0), T.Select(r_a % 6 == 4 and eps % 6 == 5, T.float32(1.5), T.Select(r_a % 6 == 4 and eps % 6 == 4, T.float32(1), T.Select(r_a % 6 == 4 and eps % 6 == 3, T.float32(1), T.Select(r_a % 6 == 4 and eps % 6 == 2, T.float32(1), T.Select(r_a % 6 == 4 and eps % 6 == 1, T.float32(1), T.Select(r_a % 6 == 4 and eps % 6 == 0, T.float32(1), T.Select(r_a % 6 == 3 and eps % 6 == 5, T.float32(-2), T.Select(r_a % 6 == 3 and eps % 6 == 4, T.float32(-0.5), T.Select(r_a % 6 == 3 and eps % 6 == 3, T.float32(2), T.Select(r_a % 6 == 3 and eps % 6 == 2, T.float32(2.5), T.Select(r_a % 6 == 3 and eps % 6 == 1, T.float32(0.5), T.Select(r_a % 6 == 3 and eps % 6 == 0, T.float32(1.5), T.Select(r_a % 6 == 2 and eps % 6 == 5, T.float32(-1.5), T.Select(r_a % 6 == 2 and eps % 6 == 4, T.float32(-1), T.Select(r_a % 6 == 2 and eps % 6 == 3, T.float32(-1), T.Select(r_a % 6 == 2 and eps % 6 == 2, T.float32(0.5), T.Select(r_a % 6 == 2 and eps % 6 == 1, T.float32(-2.5), T.Select(r_a % 6 == 2 and eps % 6 == 0, T.float32(-2), T.Select(r_a % 6 == 1 and eps % 6 == 5, T.float32(1), T.Select(r_a % 6 == 1 and eps % 6 == 4, T.float32(0.5), T.Select(r_a % 6 == 1 and eps % 6 == 3, T.float32(-2), T.Select(r_a % 6 == 1 and eps % 6 == 2, T.float32(-1), T.Select(r_a % 6 == 1 and eps % 6 == 1, T.float32(1), T.Select(r_a % 6 == 1 and eps % 6 == 0, T.float32(-1.5), T.Select(r_a % 6 == 0 and eps % 6 == 5, T.float32(0), T.Select(r_a % 6 == 0 and eps % 6 == 4, T.float32(0), T.Select(r_a % 6 == 0 and eps % 6 == 3, T.float32(0), T.Select(r_a % 6 == 0 and eps % 6 == 2, T.float32(0), T.Select(r_a % 6 == 0 and eps % 6 == 1, T.float32(0), T.Select(r_a % 6 == 0 and eps % 6 == 0, T.float32(1), T.float32(0))))))))))))))))))))))))))))))))))))) * T.Select(r_b % 6 == 5 and nu % 6 == 5, T.float32(1), T.Select(r_b % 6 == 5 and nu % 6 == 4, T.float32(0), T.Select(r_b % 6 == 5 and nu % 6 == 3, T.float32(0), T.Select(r_b % 6 == 5 and nu % 6 == 2, T.float32(0), T.Select(r_b % 6 == 5 and nu % 6 == 1, T.float32(0), T.Select(r_b % 6 == 5 and nu % 6 == 0, T.float32(0), T.Select(r_b % 6 == 4 and nu % 6 == 5, T.float32(1.5), T.Select(r_b % 6 == 4 and nu % 6 == 4, T.float32(1), T.Select(r_b % 6 == 4 and nu % 6 == 3, T.float32(1), T.Select(r_b % 6 == 4 and nu % 6 == 2, T.float32(1), T.Select(r_b % 6 == 4 and nu % 6 == 1, T.float32(1), T.Select(r_b % 6 == 4 and nu % 6 == 0, T.float32(1), T.Select(r_b % 6 == 3 and nu % 6 == 5, T.float32(-2), T.Select(r_b % 6 == 3 and nu % 6 == 4, T.float32(-0.5), T.Select(r_b % 6 == 3 and nu % 6 == 3, T.float32(2), T.Select(r_b % 6 == 3 and nu % 6 == 2, T.float32(2.5), T.Select(r_b % 6 == 3 and nu % 6 == 1, T.float32(0.5), T.Select(r_b % 6 == 3 and nu % 6 == 0, T.float32(1.5), T.Select(r_b % 6 == 2 and nu % 6 == 5, T.float32(-1.5), T.Select(r_b % 6 == 2 and nu % 6 == 4, T.float32(-1), T.Select(r_b % 6 == 2 and nu % 6 == 3, T.float32(-1), T.Select(r_b % 6 == 2 and nu % 6 == 2, T.float32(0.5), T.Select(r_b % 6 == 2 and nu % 6 == 1, T.float32(-2.5), T.Select(r_b % 6 == 2 and nu % 6 == 0, T.float32(-2), T.Select(r_b % 6 == 1 and nu % 6 == 5, T.float32(1), T.Select(r_b % 6 == 1 and nu % 6 == 4, T.float32(0.5), T.Select(r_b % 6 == 1 and nu % 6 == 3, T.float32(-2), T.Select(r_b % 6 == 1 and nu % 6 == 2, T.float32(-1), T.Select(r_b % 6 == 1 and nu % 6 == 1, T.float32(1), T.Select(r_b % 6 == 1 and nu % 6 == 0, T.float32(-1.5), T.Select(r_b % 6 == 0 and nu % 6 == 5, T.float32(0), T.Select(r_b % 6 == 0 and nu % 6 == 4, T.float32(0), T.Select(r_b % 6 == 0 and nu % 6 == 3, T.float32(0), T.Select(r_b % 6 == 0 and nu % 6 == 2, T.float32(0), T.Select(r_b % 6 == 0 and nu % 6 == 1, T.float32(0), T.Select(r_b % 6 == 0 and nu % 6 == 0, T.float32(1), T.float32(0)))))))))))))))))))))))))))))))))))))
                    for ax0, ax1, ax2, ax3 in T.grid(6, 6, 1, 1):
                        with T.block("data_pack_local"):
                            T.where(i2_i3_fused_0 * 512 + i2_i3_fused_1 < 12544)
                            v0, v1 = T.axis.remap("SS", [ax0, ax1])
                            v2 = T.axis.spatial(64, (i2_i3_fused_0 * 512 + i2_i3_fused_1) // 196 + ax2)
                            v3 = T.axis.spatial(196, (i2_i3_fused_0 * 120 + i2_i3_fused_1) % 196 + ax3)
                            T.reads(data_pack_local[v0, v1, v2, v3])
                            T.writes(data_pack[v0, v1, v2, v3])
                            data_pack[v0, v1, v2, v3] = data_pack_local[v0, v1, v2, v3]
            for i0_0_i1_0_i2_0_i3_0_fused in T.thread_binding(14, thread="blockIdx.x"):
                for i0_1_i1_1_i2_1_i3_1_fused in T.thread_binding(224, thread="vthread.x"):
                    for i0_2_i1_2_i2_2_i3_2_fused in T.thread_binding(2, thread="threadIdx.x"):
                        for i4_0 in T.serial(2):
                            for ax0_ax1_ax2_ax3_fused in T.serial(32256):
                                with T.block("data_pack_shared"):
                                    v0 = T.axis.spatial(6, ax0_ax1_ax2_ax3_fused // 5376)
                                    v1 = T.axis.spatial(6, ax0_ax1_ax2_ax3_fused % 5376 // 896)
                                    v2 = T.axis.spatial(64, i4_0 * 32 + ax0_ax1_ax2_ax3_fused % 896 // 28)
                                    v3 = T.axis.spatial(196, i0_0_i1_0_i2_0_i3_0_fused % 7 * 28 + ax0_ax1_ax2_ax3_fused % 28)
                                    T.reads(data_pack[v0, v1, v2, v3])
                                    T.writes(data_pack_shared[v0, v1, v2, v3])
                                    T.block_attr({"meta_schedule.cooperative_fetch":4})
                                    data_pack_shared[v0, v1, v2, v3] = data_pack[v0, v1, v2, v3]
                            for ax0_ax1_ax2_ax3_fused in T.serial(36864):
                                with T.block("weight_shared"):
                                    v0 = T.axis.spatial(6, ax0_ax1_ax2_ax3_fused // 6144)
                                    v1 = T.axis.spatial(6, ax0_ax1_ax2_ax3_fused % 6144 // 1024)
                                    v2 = T.axis.spatial(64, i4_0 * 32 + ax0_ax1_ax2_ax3_fused % 1024 // 32)
                                    v3 = T.axis.spatial(64, i0_0_i1_0_i2_0_i3_0_fused // 7 * 32 + ax0_ax1_ax2_ax3_fused % 32)
                                    T.reads(weight[v0, v1, v2, v3])
                                    T.writes(weight_shared[v0, v1, v2, v3])
                                    T.block_attr({"meta_schedule.cooperative_fetch":3})
                                    weight_shared[v0, v1, v2, v3] = weight[v0, v1, v2, v3]
                            for i4_1, i0_3, i1_3, i2_3, i3_3, i4_2, i0_4, i1_4, i2_4, i3_4 in T.grid(16, 2, 3, 1, 4, 2, 3, 1, 1, 1):
                                with T.block("bgemm"):
                                    eps = T.axis.spatial(6, i0_3 * 3 + i0_4)
                                    nu = T.axis.spatial(6, i1_4 + i0_1_i1_1_i2_1_i3_1_fused // 112 * 3 + i1_3)
                                    co = T.axis.spatial(64, i0_0_i1_0_i2_0_i3_0_fused // 7 * 32 + i0_1_i1_1_i2_1_i3_1_fused % 112 // 7 * 2 + i0_2_i1_2_i2_2_i3_2_fused + i2_3 + i2_4)
                                    p = T.axis.spatial(196, i3_4 + i0_0_i1_0_i2_0_i3_0_fused % 7 * 28 + i0_1_i1_1_i2_1_i3_1_fused % 7 * 4 + i3_3)
                                    ci = T.axis.reduce(64, i4_0 * 32 + i4_1 * 2 + i4_2)
                                    T.reads(data_pack_shared[eps, nu, ci, p], weight_shared[eps, nu, ci, co])
                                    T.writes(bgemm_local[eps, nu, co, p])
                                    T.block_attr({"meta_schedule.thread_extent_high_inclusive":1024, "meta_schedule.thread_extent_low_inclusive":32, "meta_schedule.tiling_structure":"SSSRRSRS"})
                                    with T.init():
                                        bgemm_local[eps, nu, co, p] = T.float32(0)
                                    bgemm_local[eps, nu, co, p] = bgemm_local[eps, nu, co, p] + data_pack_shared[eps, nu, ci, p] * weight_shared[eps, nu, ci, co]
                        for ax0, ax1, ax2, ax3 in T.grid(6, 3, 1, 4):
                            with T.block("bgemm_local"):
                                v0 = T.axis.spatial(6, ax0)
                                v1 = T.axis.spatial(6, i0_1_i1_1_i2_1_i3_1_fused // 112 * 3 + ax1)
                                v2 = T.axis.spatial(64, i0_0_i1_0_i2_0_i3_0_fused // 7 * 32 + i0_1_i1_1_i2_1_i3_1_fused % 112 // 7 * 2 + i0_2_i1_2_i2_2_i3_2_fused + ax2)
                                v3 = T.axis.spatial(196, i0_0_i1_0_i2_0_i3_0_fused % 7 * 28 + i0_1_i1_1_i2_1_i3_1_fused % 7 * 4 + ax3)
                                T.reads(bgemm_local[v0, v1, v2, v3])
                                T.writes(bgemm[v0, v1, v2, v3])
                                bgemm[v0, v1, v2, v3] = bgemm_local[v0, v1, v2, v3]
            for i0_i1_i2_0_i3_0_fused_0 in T.thread_binding(196, thread="blockIdx.x"):
                for i0_i1_i2_0_i3_0_fused_1 in T.thread_binding(64, thread="threadIdx.x"):
                    for ax0, ax1 in T.grid(1, 1):
                        for ax2 in T.unroll(4):
                            for ax3 in T.unroll(4):
                                for ax4 in T.unroll(6):
                                    for ax5 in T.unroll(6):
                                        with T.block("inverse"):
                                            co = T.axis.spatial(64, (i0_i1_i2_0_i3_0_fused_0 * 64 + i0_i1_i2_0_i3_0_fused_1) // 196 + ax0)
                                            p = T.axis.spatial(196, (i0_i1_i2_0_i3_0_fused_0 * 64 + i0_i1_i2_0_i3_0_fused_1) % 196 // 14 * 14 + (i0_i1_i2_0_i3_0_fused_0 * 64 + i0_i1_i2_0_i3_0_fused_1) % 14 + ax1)
                                            vh, vw, r_a, r_b = T.axis.remap("SSRR", [ax2, ax3, ax4, ax5])
                                            T.reads(bgemm[r_a, r_b, co, p])
                                            T.writes(inverse_local[co, p, vh, vw])
                                            T.block_attr({"schedule_rule":"conv2d_nchw_winograd_inverse"})
                                            with T.init():
                                                inverse_local[co, p, vh, vw] = T.float32(0)
                                            inverse_local[co, p, vh, vw] = inverse_local[co, p, vh, vw] + bgemm[r_a, r_b, co, p] * T.Select(r_a % 6 == 5 and vh % 4 == 3, T.float32(1), T.Select(r_a % 6 == 5 and vh % 4 == 2, T.float32(0), T.Select(r_a % 6 == 5 and vh % 4 == 1, T.float32(0), T.Select(r_a % 6 == 5 and vh % 4 == 0, T.float32(0), T.Select(r_a % 6 == 4 and vh % 4 == 3, T.float32(-8), T.Select(r_a % 6 == 4 and vh % 4 == 2, T.float32(4), T.Select(r_a % 6 == 4 and vh % 4 == 1, T.float32(-2), T.Select(r_a % 6 == 4 and vh % 4 == 0, T.float32(1), T.Select(r_a % 6 == 3 and vh % 4 == 3, T.float32(0.125), T.Select(r_a % 6 == 3 and vh % 4 == 2, T.float32(0.25), T.Select(r_a % 6 == 3 and vh % 4 == 1, T.float32(0.5), T.Select(r_a % 6 == 3 and vh % 4 == 0, T.float32(1), T.Select(r_a % 6 == 2 and vh % 4 == 3, T.float32(1), T.Select(r_a % 6 == 2 and vh % 4 == 2, T.float32(1), T.Select(r_a % 6 == 2 and vh % 4 == 1, T.float32(1), T.Select(r_a % 6 == 2 and vh % 4 == 0, T.float32(1), T.Select(r_a % 6 == 1 and vh % 4 == 3, T.float32(-1), T.Select(r_a % 6 == 1 and vh % 4 == 2, T.float32(1), T.Select(r_a % 6 == 1 and vh % 4 == 1, T.float32(-1), T.Select(r_a % 6 == 1 and vh % 4 == 0, T.float32(1), T.Select(r_a % 6 == 0 and vh % 4 == 3, T.float32(0), T.Select(r_a % 6 == 0 and vh % 4 == 2, T.float32(0), T.Select(r_a % 6 == 0 and vh % 4 == 1, T.float32(0), T.Select(r_a % 6 == 0 and vh % 4 == 0, T.float32(1), T.float32(0))))))))))))))))))))))))) * T.Select(r_b % 6 == 5 and vw % 4 == 3, T.float32(1), T.Select(r_b % 6 == 5 and vw % 4 == 2, T.float32(0), T.Select(r_b % 6 == 5 and vw % 4 == 1, T.float32(0), T.Select(r_b % 6 == 5 and vw % 4 == 0, T.float32(0), T.Select(r_b % 6 == 4 and vw % 4 == 3, T.float32(-8), T.Select(r_b % 6 == 4 and vw % 4 == 2, T.float32(4), T.Select(r_b % 6 == 4 and vw % 4 == 1, T.float32(-2), T.Select(r_b % 6 == 4 and vw % 4 == 0, T.float32(1), T.Select(r_b % 6 == 3 and vw % 4 == 3, T.float32(0.125), T.Select(r_b % 6 == 3 and vw % 4 == 2, T.float32(0.25), T.Select(r_b % 6 == 3 and vw % 4 == 1, T.float32(0.5), T.Select(r_b % 6 == 3 and vw % 4 == 0, T.float32(1), T.Select(r_b % 6 == 2 and vw % 4 == 3, T.float32(1), T.Select(r_b % 6 == 2 and vw % 4 == 2, T.float32(1), T.Select(r_b % 6 == 2 and vw % 4 == 1, T.float32(1), T.Select(r_b % 6 == 2 and vw % 4 == 0, T.float32(1), T.Select(r_b % 6 == 1 and vw % 4 == 3, T.float32(-1), T.Select(r_b % 6 == 1 and vw % 4 == 2, T.float32(1), T.Select(r_b % 6 == 1 and vw % 4 == 1, T.float32(-1), T.Select(r_b % 6 == 1 and vw % 4 == 0, T.float32(1), T.Select(r_b % 6 == 0 and vw % 4 == 3, T.float32(0), T.Select(r_b % 6 == 0 and vw % 4 == 2, T.float32(0), T.Select(r_b % 6 == 0 and vw % 4 == 1, T.float32(0), T.Select(r_b % 6 == 0 and vw % 4 == 0, T.float32(1), T.float32(0)))))))))))))))))))))))))
                    for i2_1, i3_1 in T.grid(4, 4):
                        with T.block("conv2d_winograd"):
                            n = T.axis.spatial(1, 0)
                            co = T.axis.spatial(64, (i0_i1_i2_0_i3_0_fused_0 * 64 + i0_i1_i2_0_i3_0_fused_1) // 196)
                            h = T.axis.spatial(56, (i0_i1_i2_0_i3_0_fused_0 * 64 + i0_i1_i2_0_i3_0_fused_1) % 196 // 14 * 4 + i2_1)
                            w = T.axis.spatial(56, (i0_i1_i2_0_i3_0_fused_0 * 64 + i0_i1_i2_0_i3_0_fused_1) % 14 * 4 + i3_1)
                            T.reads(inverse_local[co, n * 196 + h // 4 * 14 + w // 4, h % 4, w % 4])
                            T.writes(conv2d_winograd[n, co, h, w])
                            conv2d_winograd[n, co, h, w] = inverse_local[co, n * 196 + h // 4 * 14 + w // 4, h % 4, w % 4]
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


if __name__ == "__main__":
    test_cuda_nhwc()
    test_cuda_nchw()
