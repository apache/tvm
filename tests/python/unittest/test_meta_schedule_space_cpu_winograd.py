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


def test_cpu_nhwc():
    # fmt: off
    @T.prim_func
    def cpu_nhwc_0(X: T.Buffer((1, 14, 14, 128), "float32"), W: T.Buffer((6, 6, 128, 128), "float32"), conv2d_winograd: T.Buffer((1, 12, 12, 128), "float32")) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True, "layout_free_buffers": [1]})
        # body
        with T.block("root"):
            T.reads()
            T.writes()
            T.block_attr({"meta_schedule.parallel":288, "meta_schedule.unroll_explicit":64, "meta_schedule.vectorize":64})
            data_pad = T.alloc_buffer([1, 16, 16, 128], dtype="float32")
            input_tile = T.alloc_buffer([6, 6, 9, 128], dtype="float32")
            data_pack = T.alloc_buffer([6, 6, 9, 128], dtype="float32")
            bgemm = T.alloc_buffer([6, 6, 9, 128], dtype="float32")
            inverse = T.alloc_buffer([4, 4, 9, 128], dtype="float32")
            bgemm_global = T.alloc_buffer([6, 6, 9, 128], dtype="float32")
            for i2_0 in T.serial(9):
                for ax0, ax1, ax2, ax3 in T.grid(1, 6, 6, 128):
                    with T.block("data_pad"):
                        i0 = T.axis.spatial(1, ax0)
                        i1 = T.axis.spatial(16, i2_0 // 3 * 4 + ax1)
                        i2 = T.axis.spatial(16, i2_0 % 3 * 4 + ax2)
                        i3 = T.axis.spatial(128, ax3)
                        T.reads(X[i0, i1, i2, i3])
                        T.writes(data_pad[i0, i1, i2, i3])
                        T.block_attr({"schedule_rule":"None"})
                        data_pad[i0, i1, i2, i3] = T.if_then_else(0 <= i1 and i1 < 14 and 0 <= i2 and i2 < 14, X[i0, i1, i2, i3], T.float32(0), dtype="float32")
                for i3_0 in T.serial(2):
                    for ax0, ax1, ax2, ax3 in T.grid(6, 6, 1, 64):
                        with T.block("input_tile"):
                            eps, nu = T.axis.remap("SS", [ax0, ax1])
                            p = T.axis.spatial(9, i2_0 + ax2)
                            ci = T.axis.spatial(128, i3_0 * 64 + ax3)
                            T.reads(data_pad[p // 9, p % 9 // 3 * 4 + eps, p % 3 * 4 + nu, ci])
                            T.writes(input_tile[eps, nu, p, ci])
                            T.block_attr({"schedule_rule":"None"})
                            input_tile[eps, nu, p, ci] = data_pad[p // 9, p % 9 // 3 * 4 + eps, p % 3 * 4 + nu, ci]
                    for i2_1, i3_1 in T.grid(1, 64):
                        for i0 in T.unroll(6):
                            for i1 in T.unroll(6):
                                for i4 in T.unroll(6):
                                    for i5 in T.unroll(6):
                                        with T.block("data_pack"):
                                            eps, nu = T.axis.remap("SS", [i0, i1])
                                            p = T.axis.spatial(9, i2_0 + i2_1)
                                            ci = T.axis.spatial(128, i3_0 * 64 + i3_1)
                                            r_a, r_b = T.axis.remap("RR", [i4, i5])
                                            T.reads(input_tile[r_a, r_b, p, ci])
                                            T.writes(data_pack[eps, nu, p, ci])
                                            T.block_attr({"schedule_rule":"conv2d_nhwc_winograd_data_pack"})
                                            with T.init():
                                                data_pack[eps, nu, p, ci] = T.float32(0)
                                            data_pack[eps, nu, p, ci] = data_pack[eps, nu, p, ci] + input_tile[r_a, r_b, p, ci] * T.Select(r_a % 6 == 5 and eps % 6 == 5, T.float32(1), T.Select(r_a % 6 == 5 and eps % 6 == 4, T.float32(0), T.Select(r_a % 6 == 5 and eps % 6 == 3, T.float32(0), T.Select(r_a % 6 == 5 and eps % 6 == 2, T.float32(0), T.Select(r_a % 6 == 5 and eps % 6 == 1, T.float32(0), T.Select(r_a % 6 == 5 and eps % 6 == 0, T.float32(0), T.Select(r_a % 6 == 4 and eps % 6 == 5, T.float32(1.5), T.Select(r_a % 6 == 4 and eps % 6 == 4, T.float32(1), T.Select(r_a % 6 == 4 and eps % 6 == 3, T.float32(1), T.Select(r_a % 6 == 4 and eps % 6 == 2, T.float32(1), T.Select(r_a % 6 == 4 and eps % 6 == 1, T.float32(1), T.Select(r_a % 6 == 4 and eps % 6 == 0, T.float32(1), T.Select(r_a % 6 == 3 and eps % 6 == 5, T.float32(-2), T.Select(r_a % 6 == 3 and eps % 6 == 4, T.float32(-0.5), T.Select(r_a % 6 == 3 and eps % 6 == 3, T.float32(2), T.Select(r_a % 6 == 3 and eps % 6 == 2, T.float32(2.5), T.Select(r_a % 6 == 3 and eps % 6 == 1, T.float32(0.5), T.Select(r_a % 6 == 3 and eps % 6 == 0, T.float32(1.5), T.Select(r_a % 6 == 2 and eps % 6 == 5, T.float32(-1.5), T.Select(r_a % 6 == 2 and eps % 6 == 4, T.float32(-1), T.Select(r_a % 6 == 2 and eps % 6 == 3, T.float32(-1), T.Select(r_a % 6 == 2 and eps % 6 == 2, T.float32(0.5), T.Select(r_a % 6 == 2 and eps % 6 == 1, T.float32(-2.5), T.Select(r_a % 6 == 2 and eps % 6 == 0, T.float32(-2), T.Select(r_a % 6 == 1 and eps % 6 == 5, T.float32(1), T.Select(r_a % 6 == 1 and eps % 6 == 4, T.float32(0.5), T.Select(r_a % 6 == 1 and eps % 6 == 3, T.float32(-2), T.Select(r_a % 6 == 1 and eps % 6 == 2, T.float32(-1), T.Select(r_a % 6 == 1 and eps % 6 == 1, T.float32(1), T.Select(r_a % 6 == 1 and eps % 6 == 0, T.float32(-1.5), T.Select(r_a % 6 == 0 and eps % 6 == 5, T.float32(0), T.Select(r_a % 6 == 0 and eps % 6 == 4, T.float32(0), T.Select(r_a % 6 == 0 and eps % 6 == 3, T.float32(0), T.Select(r_a % 6 == 0 and eps % 6 == 2, T.float32(0), T.Select(r_a % 6 == 0 and eps % 6 == 1, T.float32(0), T.Select(r_a % 6 == 0 and eps % 6 == 0, T.float32(1), T.float32(0))))))))))))))))))))))))))))))))))))) * T.Select(r_b % 6 == 5 and nu % 6 == 5, T.float32(1), T.Select(r_b % 6 == 5 and nu % 6 == 4, T.float32(0), T.Select(r_b % 6 == 5 and nu % 6 == 3, T.float32(0), T.Select(r_b % 6 == 5 and nu % 6 == 2, T.float32(0), T.Select(r_b % 6 == 5 and nu % 6 == 1, T.float32(0), T.Select(r_b % 6 == 5 and nu % 6 == 0, T.float32(0), T.Select(r_b % 6 == 4 and nu % 6 == 5, T.float32(1.5), T.Select(r_b % 6 == 4 and nu % 6 == 4, T.float32(1), T.Select(r_b % 6 == 4 and nu % 6 == 3, T.float32(1), T.Select(r_b % 6 == 4 and nu % 6 == 2, T.float32(1), T.Select(r_b % 6 == 4 and nu % 6 == 1, T.float32(1), T.Select(r_b % 6 == 4 and nu % 6 == 0, T.float32(1), T.Select(r_b % 6 == 3 and nu % 6 == 5, T.float32(-2), T.Select(r_b % 6 == 3 and nu % 6 == 4, T.float32(-0.5), T.Select(r_b % 6 == 3 and nu % 6 == 3, T.float32(2), T.Select(r_b % 6 == 3 and nu % 6 == 2, T.float32(2.5), T.Select(r_b % 6 == 3 and nu % 6 == 1, T.float32(0.5), T.Select(r_b % 6 == 3 and nu % 6 == 0, T.float32(1.5), T.Select(r_b % 6 == 2 and nu % 6 == 5, T.float32(-1.5), T.Select(r_b % 6 == 2 and nu % 6 == 4, T.float32(-1), T.Select(r_b % 6 == 2 and nu % 6 == 3, T.float32(-1), T.Select(r_b % 6 == 2 and nu % 6 == 2, T.float32(0.5), T.Select(r_b % 6 == 2 and nu % 6 == 1, T.float32(-2.5), T.Select(r_b % 6 == 2 and nu % 6 == 0, T.float32(-2), T.Select(r_b % 6 == 1 and nu % 6 == 5, T.float32(1), T.Select(r_b % 6 == 1 and nu % 6 == 4, T.float32(0.5), T.Select(r_b % 6 == 1 and nu % 6 == 3, T.float32(-2), T.Select(r_b % 6 == 1 and nu % 6 == 2, T.float32(-1), T.Select(r_b % 6 == 1 and nu % 6 == 1, T.float32(1), T.Select(r_b % 6 == 1 and nu % 6 == 0, T.float32(-1.5), T.Select(r_b % 6 == 0 and nu % 6 == 5, T.float32(0), T.Select(r_b % 6 == 0 and nu % 6 == 4, T.float32(0), T.Select(r_b % 6 == 0 and nu % 6 == 3, T.float32(0), T.Select(r_b % 6 == 0 and nu % 6 == 2, T.float32(0), T.Select(r_b % 6 == 0 and nu % 6 == 1, T.float32(0), T.Select(r_b % 6 == 0 and nu % 6 == 0, T.float32(1), T.float32(0)))))))))))))))))))))))))))))))))))))
            for i0_0, i1_0, i2_0, i3_0, i0_1, i1_1, i2_1, i3_1 in T.grid(3, 2, 3, 1, 1, 1, 1, 1):
                for i4_0, i0_2, i1_2, i2_2, i3_2, i4_1, i0_3, i1_3, i2_3, i3_3 in T.grid(32, 1, 1, 1, 2, 4, 2, 3, 3, 64):
                    with T.block("bgemm"):
                        eps = T.axis.spatial(6, i0_0 * 2 + i0_1 * 2 + i0_2 * 2 + i0_3)
                        nu = T.axis.spatial(6, i1_0 * 3 + i1_1 * 3 + i1_2 * 3 + i1_3)
                        p = T.axis.spatial(9, i2_0 * 3 + i2_1 * 3 + i2_2 * 3 + i2_3)
                        co = T.axis.spatial(128, i3_0 * 128 + i3_1 * 128 + i3_2 * 64 + i3_3)
                        ci = T.axis.reduce(128, i4_0 * 4 + i4_1)
                        T.reads(data_pack[eps, nu, p, ci], W[eps, nu, co, ci])
                        T.writes(bgemm_global[eps, nu, p, co])
                        T.block_attr({"meta_schedule.tiling_structure":"SSRSRS", "meta_schedule.write_cache_level":[2]})
                        with T.init():
                            bgemm_global[eps, nu, p, co] = T.float32(0)
                        bgemm_global[eps, nu, p, co] = bgemm_global[eps, nu, p, co] + data_pack[eps, nu, p, ci] * W[eps, nu, co, ci]
                for ax0, ax1, ax2, ax3 in T.grid(2, 3, 3, 128):
                    with T.block("bgemm_global"):
                        v0 = T.axis.spatial(6, i0_0 * 2 + ax0)
                        v1 = T.axis.spatial(6, i1_0 * 3 + ax1)
                        v2 = T.axis.spatial(9, i2_0 * 3 + ax2)
                        v3 = T.axis.spatial(128, ax3)
                        T.reads(bgemm_global[v0, v1, v2, v3])
                        T.writes(bgemm[v0, v1, v2, v3])
                        bgemm[v0, v1, v2, v3] = bgemm_global[v0, v1, v2, v3]
            for i2_0, i3_0, i2_1, i3_1 in T.grid(3, 8, 3, 16):
                for i0 in T.unroll(4):
                    for i1 in T.unroll(4):
                        for i4 in T.unroll(6):
                            for i5 in T.unroll(6):
                                with T.block("inverse"):
                                    vh, vw = T.axis.remap("SS", [i0, i1])
                                    p = T.axis.spatial(9, i2_0 * 3 + i2_1)
                                    co = T.axis.spatial(128, i3_0 * 16 + i3_1)
                                    r_a, r_b = T.axis.remap("RR", [i4, i5])
                                    T.reads(bgemm[r_a, r_b, p, co])
                                    T.writes(inverse[vh, vw, p, co])
                                    T.block_attr({"schedule_rule":"conv2d_nhwc_winograd_inverse"})
                                    with T.init():
                                        inverse[vh, vw, p, co] = T.float32(0)
                                    inverse[vh, vw, p, co] = inverse[vh, vw, p, co] + bgemm[r_a, r_b, p, co] * T.Select(r_a % 6 == 5 and vh % 4 == 3, T.float32(1), T.Select(r_a % 6 == 5 and vh % 4 == 2, T.float32(0), T.Select(r_a % 6 == 5 and vh % 4 == 1, T.float32(0), T.Select(r_a % 6 == 5 and vh % 4 == 0, T.float32(0), T.Select(r_a % 6 == 4 and vh % 4 == 3, T.float32(-8), T.Select(r_a % 6 == 4 and vh % 4 == 2, T.float32(4), T.Select(r_a % 6 == 4 and vh % 4 == 1, T.float32(-2), T.Select(r_a % 6 == 4 and vh % 4 == 0, T.float32(1), T.Select(r_a % 6 == 3 and vh % 4 == 3, T.float32(0.125), T.Select(r_a % 6 == 3 and vh % 4 == 2, T.float32(0.25), T.Select(r_a % 6 == 3 and vh % 4 == 1, T.float32(0.5), T.Select(r_a % 6 == 3 and vh % 4 == 0, T.float32(1), T.Select(r_a % 6 == 2 and vh % 4 == 3, T.float32(1), T.Select(r_a % 6 == 2 and vh % 4 == 2, T.float32(1), T.Select(r_a % 6 == 2 and vh % 4 == 1, T.float32(1), T.Select(r_a % 6 == 2 and vh % 4 == 0, T.float32(1), T.Select(r_a % 6 == 1 and vh % 4 == 3, T.float32(-1), T.Select(r_a % 6 == 1 and vh % 4 == 2, T.float32(1), T.Select(r_a % 6 == 1 and vh % 4 == 1, T.float32(-1), T.Select(r_a % 6 == 1 and vh % 4 == 0, T.float32(1), T.Select(r_a % 6 == 0 and vh % 4 == 3, T.float32(0), T.Select(r_a % 6 == 0 and vh % 4 == 2, T.float32(0), T.Select(r_a % 6 == 0 and vh % 4 == 1, T.float32(0), T.Select(r_a % 6 == 0 and vh % 4 == 0, T.float32(1), T.float32(0))))))))))))))))))))))))) * T.Select(r_b % 6 == 5 and vw % 4 == 3, T.float32(1), T.Select(r_b % 6 == 5 and vw % 4 == 2, T.float32(0), T.Select(r_b % 6 == 5 and vw % 4 == 1, T.float32(0), T.Select(r_b % 6 == 5 and vw % 4 == 0, T.float32(0), T.Select(r_b % 6 == 4 and vw % 4 == 3, T.float32(-8), T.Select(r_b % 6 == 4 and vw % 4 == 2, T.float32(4), T.Select(r_b % 6 == 4 and vw % 4 == 1, T.float32(-2), T.Select(r_b % 6 == 4 and vw % 4 == 0, T.float32(1), T.Select(r_b % 6 == 3 and vw % 4 == 3, T.float32(0.125), T.Select(r_b % 6 == 3 and vw % 4 == 2, T.float32(0.25), T.Select(r_b % 6 == 3 and vw % 4 == 1, T.float32(0.5), T.Select(r_b % 6 == 3 and vw % 4 == 0, T.float32(1), T.Select(r_b % 6 == 2 and vw % 4 == 3, T.float32(1), T.Select(r_b % 6 == 2 and vw % 4 == 2, T.float32(1), T.Select(r_b % 6 == 2 and vw % 4 == 1, T.float32(1), T.Select(r_b % 6 == 2 and vw % 4 == 0, T.float32(1), T.Select(r_b % 6 == 1 and vw % 4 == 3, T.float32(-1), T.Select(r_b % 6 == 1 and vw % 4 == 2, T.float32(1), T.Select(r_b % 6 == 1 and vw % 4 == 1, T.float32(-1), T.Select(r_b % 6 == 1 and vw % 4 == 0, T.float32(1), T.Select(r_b % 6 == 0 and vw % 4 == 3, T.float32(0), T.Select(r_b % 6 == 0 and vw % 4 == 2, T.float32(0), T.Select(r_b % 6 == 0 and vw % 4 == 1, T.float32(0), T.Select(r_b % 6 == 0 and vw % 4 == 0, T.float32(1), T.float32(0)))))))))))))))))))))))))
            for i0, i1, i2, i3 in T.grid(1, 12, 12, 128):
                with T.block("conv2d_winograd"):
                    n, h, w, co = T.axis.remap("SSSS", [i0, i1, i2, i3])
                    T.reads(inverse[h % 4, w % 4, n * 9 + h // 4 * 3 + w // 4, co])
                    T.writes(conv2d_winograd[n, h, w, co])
                    conv2d_winograd[n, h, w, co] = inverse[h % 4, w % 4, n * 9 + h // 4 * 3 + w // 4, co]
    # fmt: on
    decision_0 = [
        ("SamplePerfectTile", [3, 3]),
        ("SamplePerfectTile", [8, 16]),
        ("SamplePerfectTile", [9, 1]),
        ("SamplePerfectTile", [2, 64]),
        ("SampleComputeLocation", 1),
        ("SampleComputeLocation", 0),
        ("SamplePerfectTile", [3, 1, 1, 2]),
        ("SamplePerfectTile", [2, 1, 1, 3]),
        ("SamplePerfectTile", [3, 1, 1, 3]),
        ("SamplePerfectTile", [1, 1, 2, 64]),
        ("SamplePerfectTile", [32, 4]),
        ("SampleCategorical", 2),
    ]
    with _target():
        mod = create_te_workload("C2D_WIN_NHWC", 0)
    actual = _design_space(mod)
    check_sketches(
        mod,
        sketches=actual,
        expected_mods=[cpu_nhwc_0],
        expected_decisions=[decision_0],
    )


if __name__ == "__main__":
    test_cpu_nhwc()
