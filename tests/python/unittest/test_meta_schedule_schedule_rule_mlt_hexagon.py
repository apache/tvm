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
# pylint: disable=missing-module-docstring,missing-function-docstring,missing-class-docstring
from tests.python.contrib.test_hexagon.test_meta_schedule import dense_compute
import tvm
from tvm.meta_schedule import schedule_rule
import tvm.testing
from tvm import meta_schedule as ms
from tvm import te
from tvm.meta_schedule.testing import te_workload
from tvm.meta_schedule.testing.space_generation import (
    check_sketches,
    generate_design_space,
    get_rules,
)
from tvm.script import tir as T
from tvm.tir.tensor_intrin.cuda import get_wmma_intrin_group
from tvm.tir.tensor_intrin.hexagon import VRMPY_u8u8i32_INTRIN, VRMPY_u8u8i32_VTCM_INTRIN


def multi_level_tiling_hexagon(
    *,
    write_reuse_scope="global.vtcm",
    in_dtype="uint8",
    out_dtype="int32",
    use_software_pipeline=False,
) -> ms.schedule_rule.ScheduleRule:
    assert write_reuse_scope in ["global", "global.vtcm"]
    if not isinstance(in_dtype, list):
        in_dtype = [in_dtype]
    if not isinstance(out_dtype, list):
        out_dtype = [out_dtype]
    return ms.schedule_rule.MultiLevelTilingHexagon(
        intrin_groups=[
            {"compute": VRMPY_u8u8i32_VTCM_INTRIN},
        ],
        structure="SRSRS",
        tile_binds=None,
        max_innermost_factor=64,  # 64 // tensor intrin size
        vector_load_lens=None,
        reuse_read=ms.schedule_rule.ReuseType(
            req="must",
            levels=[2],
            scope="global.vtcm",
        ),
        reuse_write=ms.schedule_rule.ReuseType(
            req="must" if write_reuse_scope == "shared" else "no",
            levels=[1],
            scope=write_reuse_scope,
        ),
        use_software_pipeline=use_software_pipeline,
    )


def test_dense_base():
    @T.prim_func
    def main(
        X: T.Buffer[(128, 768), "uint8"],
        packed_width: T.Buffer[(24, 192, 32, 4), "uint8"],
        compute: T.Buffer[(128, 768), "int32"],
    ):
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # body
        # with T.block("root")
        X_global_vtcm = T.alloc_buffer([128, 768], dtype="uint8", scope="global.vtcm")
        packed_width_global_vtcm = T.alloc_buffer(
            [24, 192, 32, 4], dtype="uint8", scope="global.vtcm"
        )
        for i_0, j_0_0, k_0_0 in T.grid(128, 6, 48):
            for ax0_ax1_fused in T.serial(16):
                with T.block("X_global.vtcm"):
                    v0 = T.axis.spatial(128, i_0)
                    v1 = T.axis.spatial(768, k_0_0 * 16 + ax0_ax1_fused)
                    T.reads(X[v0, v1])
                    T.writes(X_global_vtcm[v0, v1])
                    X_global_vtcm[v0, v1] = X[v0, v1]
            for ax0_ax1_ax2_ax3_fused in T.serial(2048):
                with T.block("packed_width_global.vtcm"):
                    v0 = T.axis.spatial(24, j_0_0 * 4 + ax0_ax1_ax2_ax3_fused // 512)
                    v1 = T.axis.spatial(192, k_0_0 * 4 + ax0_ax1_ax2_ax3_fused % 512 // 128)
                    v2 = T.axis.spatial(32, ax0_ax1_ax2_ax3_fused % 128 // 4)
                    v3 = T.axis.spatial(4, ax0_ax1_ax2_ax3_fused % 4)
                    T.reads(packed_width[v0, v1, v2, v3])
                    T.writes(packed_width_global_vtcm[v0, v1, v2, v3])
                    packed_width_global_vtcm[v0, v1, v2, v3] = packed_width[v0, v1, v2, v3]
            for i_1, j_0_1, k_0_1, i_2, j_0_2 in T.grid(1, 2, 4, 1, 2):
                with T.block("compute_o"):
                    v_i = T.axis.spatial(128, i_0 + i_1 + i_2)
                    v_j_o = T.axis.spatial(24, j_0_0 * 4 + j_0_1 * 2 + j_0_2)
                    v_k_o = T.axis.reduce(192, k_0_0 * 4 + k_0_1)
                    T.reads(
                        X_global_vtcm[v_i, v_k_o * 4 : v_k_o * 4 + 4],
                        packed_width_global_vtcm[v_j_o, v_k_o, 0:32, 0:4],
                    )
                    T.writes(compute[v_i, v_j_o * 32 : v_j_o * 32 + 32])
                    T.block_attr({"meta_schedule.auto_tensorize": "dot_32x4_u8u8i32_vtcm_vrmpy"})
                    with T.init():
                        for j_1 in T.serial(32):
                            with T.block("compute_init"):
                                v_j_i_init = T.axis.spatial(32, j_1)
                                T.reads()
                                T.writes(compute[v_i, v_j_o * 32 + v_j_i_init])
                                compute[v_i, v_j_o * 32 + v_j_i_init] = 0
                    for j_1, k_1 in T.grid(32, 4):
                        with T.block("compute"):
                            v_j_i, v_k_i = T.axis.remap("SR", [j_1, k_1])
                            T.reads(
                                compute[v_i, v_j_o * 32 + v_j_i],
                                X_global_vtcm[v_i, v_k_o * 4 + v_k_i],
                                packed_width_global_vtcm[v_j_o, v_k_o, v_j_i, v_k_i],
                            )
                            T.writes(compute[v_i, v_j_o * 32 + v_j_i])
                            T.block_attr({"meta_schedule.tiling_structure": "SRSRS"})
                            compute[v_i, v_j_o * 32 + v_j_i] = compute[
                                v_i, v_j_o * 32 + v_j_i
                            ] + T.Cast("int32", X_global_vtcm[v_i, v_k_o * 4 + v_k_i]) * T.Cast(
                                "int32", packed_width_global_vtcm[v_j_o, v_k_o, v_j_i, v_k_i]
                            )

    # fmt: on
    decision_0 = [
        ("SamplePerfectTile", [4, 1, 1]),
        ("SamplePerfectTile", [2, 2, 2]),
        ("SamplePerfectTile", [1, 4]),
    ]

    mod = te.create_prim_func(
        dense_compute(
            m=128,
            n=768,
            k=768,
        )
    )

    actual_design_space = generate_design_space(
        kind="hexagon",
        mod=mod,
        target=tvm.target.Target("hexagon"),
        types=None,
        sch_rules=[
            multi_level_tiling_hexagon(),
        ]
        + get_rules(kind="hexagon", types=ms.schedule_rule.AutoInline),
    )
    check_sketches(
        mod,
        sketches=actual_design_space,
        expected_mods=[main],
        expected_decisions=[decision_0],
    )


def test_dense_with_fallback():

    # from tvm.script import tir as T
    @T.prim_func
    def main(
        X: T.Buffer[(128, 768), "uint8"],
        packed_width: T.Buffer[(24, 192, 32, 4), "uint8"],
        compute: T.Buffer[(128, 768), "int32"],
    ):
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # body
        # with T.block("root")
        X_global_vtcm = T.alloc_buffer([128, 768], dtype="uint8", scope="global.vtcm")
        packed_width_global_vtcm = T.alloc_buffer(
            [24, 192, 32, 4], dtype="uint8", scope="global.vtcm"
        )
        for i_0, j_0_0, k_0_0 in T.grid(128, 6, 192):
            for ax0_ax1_fused in T.serial(4):
                with T.block("X_global.vtcm"):
                    v0 = T.axis.spatial(128, i_0)
                    v1 = T.axis.spatial(768, k_0_0 * 4 + ax0_ax1_fused)
                    T.reads(X[v0, v1])
                    T.writes(X_global_vtcm[v0, v1])
                    X_global_vtcm[v0, v1] = X[v0, v1]
            for ax0_ax1_ax2_ax3_fused in T.serial(512):
                with T.block("packed_width_global.vtcm"):
                    v0 = T.axis.spatial(24, j_0_0 * 4 + ax0_ax1_ax2_ax3_fused // 128)
                    v1 = T.axis.spatial(192, k_0_0)
                    v2 = T.axis.spatial(32, ax0_ax1_ax2_ax3_fused % 128 // 4)
                    v3 = T.axis.spatial(4, ax0_ax1_ax2_ax3_fused % 4)
                    T.reads(packed_width[v0, v1, v2, v3])
                    T.writes(packed_width_global_vtcm[v0, v1, v2, v3])
                    packed_width_global_vtcm[v0, v1, v2, v3] = packed_width[v0, v1, v2, v3]
            for i_1, j_0_1, k_0_1, i_2, j_0_2 in T.grid(1, 2, 1, 1, 2):
                with T.block("compute_o"):
                    v_i = T.axis.spatial(128, i_0 + i_1 + i_2)
                    v_j_o = T.axis.spatial(24, j_0_0 * 4 + j_0_1 * 2 + j_0_2)
                    v_k_o = T.axis.reduce(192, k_0_1 + k_0_0)
                    T.reads(
                        X_global_vtcm[v_i, v_k_o * 4 : v_k_o * 4 + 4],
                        packed_width_global_vtcm[v_j_o, v_k_o, 0:32, 0:4],
                    )
                    T.writes(compute[v_i, v_j_o * 32 : v_j_o * 32 + 32])
                    T.block_attr({"meta_schedule.auto_tensorize": "dot_32x4_u8u8i32_vtcm_vrmpy"})
                    with T.init():
                        for j_1 in T.serial(32):
                            with T.block("compute_init"):
                                v_j_i_init = T.axis.spatial(32, j_1)
                                T.reads()
                                T.writes(compute[v_i, v_j_o * 32 + v_j_i_init])
                                compute[v_i, v_j_o * 32 + v_j_i_init] = 0
                    for j_1, k_1 in T.grid(32, 4):
                        with T.block("compute"):
                            v_j_i, v_k_i = T.axis.remap("SR", [j_1, k_1])
                            T.reads(
                                compute[v_i, v_j_o * 32 + v_j_i],
                                X_global_vtcm[v_i, v_k_o * 4 + v_k_i],
                                packed_width_global_vtcm[v_j_o, v_k_o, v_j_i, v_k_i],
                            )
                            T.writes(compute[v_i, v_j_o * 32 + v_j_i])
                            T.block_attr({"meta_schedule.tiling_structure": "SRSRS"})
                            compute[v_i, v_j_o * 32 + v_j_i] = compute[
                                v_i, v_j_o * 32 + v_j_i
                            ] + T.Cast("int32", X_global_vtcm[v_i, v_k_o * 4 + v_k_i]) * T.Cast(
                                "int32", packed_width_global_vtcm[v_j_o, v_k_o, v_j_i, v_k_i]
                            )

    # fmt: on
    decision_0 = [
        ("SamplePerfectTile", [4, 1, 1]),
        ("SamplePerfectTile", [2, 2, 2]),
        ("SamplePerfectTile", [2, 1]),
    ]

    mod = te.create_prim_func(
        dense_compute(
            m=128,
            n=768,
            k=768,
        )
    )

    actual_design_space = generate_design_space(
        kind="hexagon",
        mod=mod,
        target=tvm.target.Target("hexagon"),
        types=None,
        sch_rules=[
            multi_level_tiling_hexagon(),
        ]
        + get_rules(kind="hexagon", types=ms.schedule_rule.AutoInline),
    )

    check_sketches(
        mod,
        sketches=actual_design_space,
        expected_mods=[main],
        expected_decisions=[decision_0],
    )


def test_dense_with_pipeline():
    @T.prim_func
    def main(
        X: T.Buffer[(128, 768), "uint8"],
        packed_width: T.Buffer[(24, 192, 32, 4), "uint8"],
        compute: T.Buffer[(128, 768), "int32"],
    ):
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # body
        # with T.block("root")
        X_global_vtcm = T.alloc_buffer([128, 768], dtype="uint8", scope="global.vtcm")
        packed_width_global_vtcm = T.alloc_buffer(
            [24, 192, 32, 4], dtype="uint8", scope="global.vtcm"
        )
        for i_0, j_0_0 in T.grid(128, 6):
            for k_0_0_fused in T.serial(
                48,
                annotations={
                    "software_pipeline_async_stages": [0],
                    "software_pipeline_order": [0, 1, 2],
                    "software_pipeline_stage": [0, 0, 1],
                },
            ):
                for ax0_ax1_fused in T.serial(16):
                    with T.block("X_global.vtcm"):
                        v0 = T.axis.spatial(128, i_0)
                        v1 = T.axis.spatial(768, k_0_0_fused * 16 + ax0_ax1_fused)
                        T.reads(X[v0, v1])
                        T.writes(X_global_vtcm[v0, v1])
                        X_global_vtcm[v0, v1] = X[v0, v1]
                for ax0_ax1_ax2_ax3_fused in T.serial(2048):
                    with T.block("packed_width_global.vtcm"):
                        v0 = T.axis.spatial(24, j_0_0 * 4 + ax0_ax1_ax2_ax3_fused // 512)
                        v1 = T.axis.spatial(
                            192, k_0_0_fused * 4 + ax0_ax1_ax2_ax3_fused % 512 // 128
                        )
                        v2 = T.axis.spatial(32, ax0_ax1_ax2_ax3_fused % 128 // 4)
                        v3 = T.axis.spatial(4, ax0_ax1_ax2_ax3_fused % 4)
                        T.reads(packed_width[v0, v1, v2, v3])
                        T.writes(packed_width_global_vtcm[v0, v1, v2, v3])
                        packed_width_global_vtcm[v0, v1, v2, v3] = packed_width[v0, v1, v2, v3]
                for i_1, j_0_1, k_0_1, i_2, j_0_2 in T.grid(1, 2, 4, 1, 2):
                    with T.block("compute_o"):
                        v_i = T.axis.spatial(128, i_1 + i_2 + i_0)
                        v_j_o = T.axis.spatial(24, j_0_0 * 4 + j_0_1 * 2 + j_0_2)
                        v_k_o = T.axis.reduce(192, k_0_0_fused * 4 + k_0_1)
                        T.reads(
                            X_global_vtcm[v_i, v_k_o * 4 : v_k_o * 4 + 4],
                            packed_width_global_vtcm[v_j_o, v_k_o, 0:32, 0:4],
                        )
                        T.writes(compute[v_i, v_j_o * 32 : v_j_o * 32 + 32])
                        T.block_attr(
                            {"meta_schedule.auto_tensorize": "dot_32x4_u8u8i32_vtcm_vrmpy"}
                        )
                        with T.init():
                            for j_1 in T.serial(32):
                                with T.block("compute_init"):
                                    v_j_i_init = T.axis.spatial(32, j_1)
                                    T.reads()
                                    T.writes(compute[v_i, v_j_o * 32 + v_j_i_init])
                                    compute[v_i, v_j_o * 32 + v_j_i_init] = 0
                        for j_1, k_1 in T.grid(32, 4):
                            with T.block("compute"):
                                v_j_i, v_k_i = T.axis.remap("SR", [j_1, k_1])
                                T.reads(
                                    compute[v_i, v_j_o * 32 + v_j_i],
                                    X_global_vtcm[v_i, v_k_o * 4 + v_k_i],
                                    packed_width_global_vtcm[v_j_o, v_k_o, v_j_i, v_k_i],
                                )
                                T.writes(compute[v_i, v_j_o * 32 + v_j_i])
                                T.block_attr({"meta_schedule.tiling_structure": "SRSRS"})
                                compute[v_i, v_j_o * 32 + v_j_i] = compute[
                                    v_i, v_j_o * 32 + v_j_i
                                ] + T.Cast("int32", X_global_vtcm[v_i, v_k_o * 4 + v_k_i]) * T.Cast(
                                    "int32", packed_width_global_vtcm[v_j_o, v_k_o, v_j_i, v_k_i]
                                )

    # fmt: on
    decision_0 = [
        ("SamplePerfectTile", [4, 1, 1]),
        ("SamplePerfectTile", [2, 2, 2]),
        ("SamplePerfectTile", [1, 4]),
    ]

    mod = te.create_prim_func(
        dense_compute(
            m=128,
            n=768,
            k=768,
        )
    )

    actual_design_space = generate_design_space(
        kind="hexagon",
        mod=mod,
        target=tvm.target.Target("hexagon"),
        types=None,
        sch_rules=[
            multi_level_tiling_hexagon(use_software_pipeline=True),
        ]
        + get_rules(kind="hexagon", types=ms.schedule_rule.AutoInline),
    )

    check_sketches(
        mod,
        sketches=actual_design_space,
        expected_mods=[main],
        expected_decisions=[decision_0],
    )


def test_dense_global():

    # from tvm.script import tir as T
    @T.prim_func
    def main(
        X: T.Buffer[(128, 768), "uint8"],
        packed_width: T.Buffer[(24, 192, 32, 4), "uint8"],
        compute: T.Buffer[(128, 768), "int32"],
    ):
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # body
        # with T.block("root")
        X_global_vtcm = T.alloc_buffer([128, 768], dtype="uint8", scope="global.vtcm")
        packed_width_global_vtcm = T.alloc_buffer(
            [24, 192, 32, 4], dtype="uint8", scope="global.vtcm"
        )
        for i_0, j_0_0, k_0_0 in T.grid(128, 6, 192):
            for ax0_ax1_fused in T.serial(4):
                with T.block("X_global.vtcm"):
                    v0 = T.axis.spatial(128, i_0)
                    v1 = T.axis.spatial(768, k_0_0 * 4 + ax0_ax1_fused)
                    T.reads(X[v0, v1])
                    T.writes(X_global_vtcm[v0, v1])
                    X_global_vtcm[v0, v1] = X[v0, v1]
            for ax0_ax1_ax2_ax3_fused in T.serial(512):
                with T.block("packed_width_global.vtcm"):
                    v0 = T.axis.spatial(24, j_0_0 * 4 + ax0_ax1_ax2_ax3_fused // 128)
                    v1 = T.axis.spatial(192, k_0_0)
                    v2 = T.axis.spatial(32, ax0_ax1_ax2_ax3_fused % 128 // 4)
                    v3 = T.axis.spatial(4, ax0_ax1_ax2_ax3_fused % 4)
                    T.reads(packed_width[v0, v1, v2, v3])
                    T.writes(packed_width_global_vtcm[v0, v1, v2, v3])
                    packed_width_global_vtcm[v0, v1, v2, v3] = packed_width[v0, v1, v2, v3]
            for i_1, j_0_1, k_0_1, i_2, j_0_2 in T.grid(1, 2, 1, 1, 2):
                with T.block("compute_o"):
                    v_i = T.axis.spatial(128, i_0 + i_1 + i_2)
                    v_j_o = T.axis.spatial(24, j_0_0 * 4 + j_0_1 * 2 + j_0_2)
                    v_k_o = T.axis.reduce(192, k_0_1 + k_0_0)
                    T.reads(
                        X_global_vtcm[v_i, v_k_o * 4 : v_k_o * 4 + 4],
                        packed_width_global_vtcm[v_j_o, v_k_o, 0:32, 0:4],
                    )
                    T.writes(compute[v_i, v_j_o * 32 : v_j_o * 32 + 32])
                    T.block_attr({"meta_schedule.auto_tensorize": "dot_32x4_u8u8i32_vtcm_vrmpy"})
                    with T.init():
                        for j_1 in T.serial(32):
                            with T.block("compute_init"):
                                v_j_i_init = T.axis.spatial(32, j_1)
                                T.reads()
                                T.writes(compute[v_i, v_j_o * 32 + v_j_i_init])
                                compute[v_i, v_j_o * 32 + v_j_i_init] = 0
                    for j_1, k_1 in T.grid(32, 4):
                        with T.block("compute"):
                            v_j_i, v_k_i = T.axis.remap("SR", [j_1, k_1])
                            T.reads(
                                compute[v_i, v_j_o * 32 + v_j_i],
                                X_global_vtcm[v_i, v_k_o * 4 + v_k_i],
                                packed_width_global_vtcm[v_j_o, v_k_o, v_j_i, v_k_i],
                            )
                            T.writes(compute[v_i, v_j_o * 32 + v_j_i])
                            T.block_attr({"meta_schedule.tiling_structure": "SRSRS"})
                            compute[v_i, v_j_o * 32 + v_j_i] = compute[
                                v_i, v_j_o * 32 + v_j_i
                            ] + T.Cast("int32", X_global_vtcm[v_i, v_k_o * 4 + v_k_i]) * T.Cast(
                                "int32", packed_width_global_vtcm[v_j_o, v_k_o, v_j_i, v_k_i]
                            )

    # fmt: on
    decision_0 = [
        ("SamplePerfectTile", [4, 1, 1]),
        ("SamplePerfectTile", [2, 2, 2]),
        ("SamplePerfectTile", [2, 1]),
    ]

    mod = te.create_prim_func(
        dense_compute(
            m=128,
            n=768,
            k=768,
        )
    )

    actual_design_space = generate_design_space(
        kind="hexagon",
        mod=mod,
        target=tvm.target.Target("hexagon"),
        types=None,
        sch_rules=[
            multi_level_tiling_hexagon(write_reuse_scope="global"),
        ]
        + get_rules(kind="hexagon", types=ms.schedule_rule.AutoInline),
    )
    check_sketches(
        mod,
        sketches=actual_design_space,
        expected_mods=[main],
        expected_decisions=[decision_0],
    )


def test_padded_dense():
    @T.prim_func
    def main(
        X: T.Buffer[(128, 768), "uint8"],
        packed_width: T.Buffer[(24, 192, 32, 4), "uint8"],
        compute: T.Buffer[(128, 768), "int32"],
    ):
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # body
        # with T.block("root")
        X_global_vtcm = T.alloc_buffer([128, 768], dtype="uint8", scope="global.vtcm")
        packed_width_global_vtcm = T.alloc_buffer(
            [24, 192, 32, 4], dtype="uint8", scope="global.vtcm"
        )
        for i_0, j_0_0, k_0_0 in T.grid(128, 6, 48):
            for ax0_ax1_fused in T.serial(16):
                with T.block("X_global.vtcm"):
                    v0 = T.axis.spatial(128, i_0)
                    v1 = T.axis.spatial(768, k_0_0 * 16 + ax0_ax1_fused)
                    T.reads(X[v0, v1])
                    T.writes(X_global_vtcm[v0, v1])
                    X_global_vtcm[v0, v1] = X[v0, v1]
            for ax0_ax1_ax2_ax3_fused in T.serial(2048):
                with T.block("packed_width_global.vtcm"):
                    v0 = T.axis.spatial(24, j_0_0 * 4 + ax0_ax1_ax2_ax3_fused // 512)
                    v1 = T.axis.spatial(192, k_0_0 * 4 + ax0_ax1_ax2_ax3_fused % 512 // 128)
                    v2 = T.axis.spatial(32, ax0_ax1_ax2_ax3_fused % 128 // 4)
                    v3 = T.axis.spatial(4, ax0_ax1_ax2_ax3_fused % 4)
                    T.reads(packed_width[v0, v1, v2, v3])
                    T.writes(packed_width_global_vtcm[v0, v1, v2, v3])
                    packed_width_global_vtcm[v0, v1, v2, v3] = packed_width[v0, v1, v2, v3]
            for i_1, j_0_1, k_0_1, i_2, j_0_2 in T.grid(1, 2, 4, 1, 2):
                with T.block("compute_o"):
                    v_i = T.axis.spatial(128, i_0 + i_1 + i_2)
                    v_j_o = T.axis.spatial(24, j_0_0 * 4 + j_0_1 * 2 + j_0_2)
                    v_k_o = T.axis.reduce(192, k_0_0 * 4 + k_0_1)
                    T.reads(
                        X_global_vtcm[v_i, v_k_o * 4 : v_k_o * 4 + 4],
                        packed_width_global_vtcm[v_j_o, v_k_o, 0:32, 0:4],
                    )
                    T.writes(compute[v_i, v_j_o * 32 : v_j_o * 32 + 32])
                    T.block_attr({"meta_schedule.auto_tensorize": "dot_32x4_u8u8i32_vtcm_vrmpy"})
                    with T.init():
                        for j_1 in T.serial(32):
                            with T.block("compute_init"):
                                v_j_i_init = T.axis.spatial(32, j_1)
                                T.reads()
                                T.writes(compute[v_i, v_j_o * 32 + v_j_i_init])
                                compute[v_i, v_j_o * 32 + v_j_i_init] = 0
                    for j_1, k_1 in T.grid(32, 4):
                        with T.block("compute"):
                            v_j_i, v_k_i = T.axis.remap("SR", [j_1, k_1])
                            T.reads(
                                compute[v_i, v_j_o * 32 + v_j_i],
                                X_global_vtcm[v_i, v_k_o * 4 + v_k_i],
                                packed_width_global_vtcm[v_j_o, v_k_o, v_j_i, v_k_i],
                            )
                            T.writes(compute[v_i, v_j_o * 32 + v_j_i])
                            T.block_attr({"meta_schedule.tiling_structure": "SRSRS"})
                            compute[v_i, v_j_o * 32 + v_j_i] = compute[
                                v_i, v_j_o * 32 + v_j_i
                            ] + T.Cast("int32", X_global_vtcm[v_i, v_k_o * 4 + v_k_i]) * T.Cast(
                                "int32", packed_width_global_vtcm[v_j_o, v_k_o, v_j_i, v_k_i]
                            )

    # fmt: on
    decision_0 = [
        ("SamplePerfectTile", [4, 1, 1]),
        ("SamplePerfectTile", [2, 2, 2]),
        ("SamplePerfectTile", [1, 4]),
    ]

    mod = te.create_prim_func(
        dense_compute(
            m=128,
            n=768,
            k=768,
        )
    )

    actual_design_space = generate_design_space(
        kind="hexagon",
        mod=mod,
        target=tvm.target.Target("hexagon"),
        types=None,
        sch_rules=[
            multi_level_tiling_hexagon(write_reuse_scope="global"),
        ]
        + get_rules(kind="hexagon", types=ms.schedule_rule.AutoInline),
    )

    check_sketches(
        mod,
        sketches=actual_design_space,
        expected_mods=[main],
        expected_decisions=[decision_0],
    )


def test_conv2d():

    # from tvm.script import tir as T
    @T.prim_func
    def main(
        inputs: T.Buffer[(1, 16, 16, 32), "uint8"],
        weight: T.Buffer[(3, 3, 32, 32), "uint8"],
        conv2d_nhwc: T.Buffer[(1, 16, 16, 32), "int32"],
    ):
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # body
        # with T.block("root")
        PadInput = T.alloc_buffer([1, 18, 18, 32], dtype="uint8")
        PadInput_global_vtcm = T.alloc_buffer([1, 18, 18, 32], dtype="uint8", scope="global.vtcm")
        weight_global_vtcm = T.alloc_buffer([3, 3, 32, 32], dtype="uint8", scope="global.vtcm")
        for i0, i1, i2, i3 in T.grid(1, 18, 18, 32):
            with T.block("PadInput"):
                v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(inputs[v_i0, v_i1 - 1, v_i2 - 1, v_i3])
                T.writes(PadInput[v_i0, v_i1, v_i2, v_i3])
                PadInput[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(
                    1 <= v_i1 and v_i1 < 17 and 1 <= v_i2 and v_i2 < 17,
                    inputs[v_i0, v_i1 - 1, v_i2 - 1, v_i3],
                    T.uint8(0),
                    dtype="uint8",
                )
        for n_0, h_0, w_0, co_0_0, rh_0, rw_0, rc_0_0 in T.grid(1, 4, 4, 1, 1, 1, 2):
            for ax0_ax1_ax2_ax3_fused in T.serial(576):
                with T.block("PadInput_global.vtcm"):
                    v0 = T.axis.spatial(1, 0)
                    v1 = T.axis.spatial(18, h_0 * 4 + ax0_ax1_ax2_ax3_fused // 96)
                    v2 = T.axis.spatial(18, w_0 * 4 + ax0_ax1_ax2_ax3_fused % 96 // 16)
                    v3 = T.axis.spatial(32, rc_0_0 * 16 + ax0_ax1_ax2_ax3_fused % 16)
                    T.reads(PadInput[v0, v1, v2, v3])
                    T.writes(PadInput_global_vtcm[v0, v1, v2, v3])
                    PadInput_global_vtcm[v0, v1, v2, v3] = PadInput[v0, v1, v2, v3]
            for ax0_ax1_ax2_ax3_fused in T.serial(4608):
                with T.block("weight_global.vtcm"):
                    v0 = T.axis.spatial(3, ax0_ax1_ax2_ax3_fused // 1536)
                    v1 = T.axis.spatial(3, ax0_ax1_ax2_ax3_fused % 1536 // 512)
                    v2 = T.axis.spatial(32, rc_0_0 * 16 + ax0_ax1_ax2_ax3_fused % 512 // 32)
                    v3 = T.axis.spatial(32, ax0_ax1_ax2_ax3_fused % 32)
                    T.reads(weight[v0, v1, v2, v3])
                    T.writes(weight_global_vtcm[v0, v1, v2, v3])
                    weight_global_vtcm[v0, v1, v2, v3] = weight[v0, v1, v2, v3]
            for n_1, h_1, w_1, co_0_1, rh_1, rw_1, rc_0_1, n_2, h_2, w_2, co_0_2 in T.grid(
                1, 1, 4, 1, 3, 3, 4, 1, 4, 1, 1
            ):
                with T.block("conv2d_nhwc_o"):
                    v_n = T.axis.spatial(1, n_1 + n_2 + n_0)
                    v_h = T.axis.spatial(16, h_0 * 4 + h_1 * 4 + h_2)
                    v_w = T.axis.spatial(16, w_2 + w_0 * 4 + w_1)
                    v_co_o = T.axis.spatial(1, co_0_1 + co_0_2 + co_0_0)
                    v_rh = T.axis.reduce(3, rh_0 * 3 + rh_1)
                    v_rw = T.axis.reduce(3, rw_0 * 3 + rw_1)
                    v_rc_o = T.axis.reduce(8, rc_0_0 * 4 + rc_0_1)
                    T.reads(
                        PadInput_global_vtcm[
                            v_n, v_h + v_rh, v_w + v_rw, v_rc_o * 4 : v_rc_o * 4 + 4
                        ],
                        weight_global_vtcm[v_rh, v_rw, v_rc_o * 4 : v_rc_o * 4 + 4, 0:32],
                    )
                    T.writes(conv2d_nhwc[v_n, v_h, v_w, 0:32])
                    T.block_attr({"meta_schedule.auto_tensorize": "dot_32x4_u8u8i32_vtcm_vrmpy"})
                    with T.init():
                        for co_1 in T.serial(32):
                            with T.block("conv2d_nhwc_init"):
                                v_co_i_init = T.axis.spatial(32, co_1)
                                T.reads()
                                T.writes(conv2d_nhwc[v_n, v_h, v_w, v_co_i_init])
                                conv2d_nhwc[v_n, v_h, v_w, v_co_i_init] = 0
                    for co_1, rc_1 in T.grid(32, 4):
                        with T.block("conv2d_nhwc"):
                            v_co_i, v_rc_i = T.axis.remap("SR", [co_1, rc_1])
                            T.reads(
                                conv2d_nhwc[v_n, v_h, v_w, v_co_i],
                                PadInput_global_vtcm[
                                    v_n, v_h + v_rh, v_w + v_rw, v_rc_o * 4 + v_rc_i
                                ],
                                weight_global_vtcm[v_rh, v_rw, v_rc_o * 4 + v_rc_i, v_co_i],
                            )
                            T.writes(conv2d_nhwc[v_n, v_h, v_w, v_co_i])
                            T.block_attr({"meta_schedule.tiling_structure": "SRSRS"})
                            conv2d_nhwc[v_n, v_h, v_w, v_co_i] = conv2d_nhwc[
                                v_n, v_h, v_w, v_co_i
                            ] + T.Cast(
                                "int32",
                                PadInput_global_vtcm[
                                    v_n, v_h + v_rh, v_w + v_rw, v_rc_o * 4 + v_rc_i
                                ],
                            ) * T.Cast(
                                "int32", weight_global_vtcm[v_rh, v_rw, v_rc_o * 4 + v_rc_i, v_co_i]
                            )

    # fmt: on
    decision_0 = [
        ("SamplePerfectTile", [4, 1, 1]),
        ("SamplePerfectTile", [2, 1, 4]),
        ("SamplePerfectTile", [2, 4, 1]),
        ("SamplePerfectTile", [2, 2, 2]),
        ("SamplePerfectTile", [2, 4]),
        ("SamplePerfectTile", [2, 4]),
        ("SamplePerfectTile", [2, 4]),
    ]

    mod = te.create_prim_func(
        te_workload.conv2d_nhwc(
            N=1,
            H=16,
            W=16,
            CI=32,
            CO=32,
            kernel_size=3,
            stride=1,
            padding=1,
            in_dtype="uint8",
            out_dtype="int32",
        )
    )

    actual_design_space = generate_design_space(
        kind="hexagon",
        mod=mod,
        target=tvm.target.Target("hexagon"),
        types=None,
        sch_rules=[
            multi_level_tiling_hexagon(),
        ]
        + get_rules(kind="hexagon", types=ms.schedule_rule.AutoInline),
    )

    check_sketches(
        mod,
        sketches=actual_design_space,
        expected_mods=[main],
        expected_decisions=[decision_0],
    )


def conv2d_NCHWc_int8(I, O, H, W, kH, kW, stride, padding, dilation, out_dtype="int32", n_elems=32):
    from tvm.topi.utils import get_const_tuple
    from tvm.topi.nn.utils import get_pad_tuple
    from tvm.topi.nn.pad import pad

    ic_bn = 32
    oc_bn = 32
    n_elems = 4
    dtype = "uint8"

    data = te.placeholder((1, I // ic_bn, H, W, ic_bn), name="data", dtype=dtype)
    kernel = te.placeholder(
        (O // oc_bn, I // ic_bn, kH, kW, ic_bn // n_elems, oc_bn, n_elems), dtype=dtype
    )

    # layout and out_layout are not used here,
    # we keep them for debug convenience when dumping autotvm workload
    HSTR, WSTR = stride if isinstance(stride, (tuple, list)) else (stride, stride)
    dilation_h, dilation_w = (
        dilation if isinstance(dilation, (tuple, list)) else (dilation, dilation)
    )

    n, ic_chunk, ih, iw, ic_bn = get_const_tuple(data.shape)
    in_channel = ic_chunk * ic_bn
    oc_chunk, ic_chunk_group, kernel_height, kernel_width, _, oc_bn, _ = get_const_tuple(
        kernel.shape
    )

    dilated_kernel_h = (kernel_height - 1) * dilation_h + 1
    dilated_kernel_w = (kernel_width - 1) * dilation_w + 1

    pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
        padding, (dilated_kernel_h, dilated_kernel_w)
    )
    HPAD = pad_top + pad_down
    WPAD = pad_left + pad_right

    # output shape
    out_height = (ih + HPAD - dilated_kernel_h) // HSTR + 1
    out_width = (iw + WPAD - dilated_kernel_w) // WSTR + 1
    oshape = (n, oc_chunk, out_height, out_width, oc_bn)
    pad_before = (0, 0, pad_top, pad_left, 0)
    pad_after = (0, 0, pad_down, pad_right, 0)

    # DOPAD
    DOPAD = HPAD != 0 or WPAD != 0
    if DOPAD:
        data_pad = pad(data, pad_before, pad_after, name="data_pad")
    else:
        data_pad = data

    kh = te.reduce_axis((0, kernel_height), name="kh")
    kw = te.reduce_axis((0, kernel_width), name="kw")

    ic_outer = te.reduce_axis((0, in_channel // ic_bn), name="ic_outer")
    ic_f_inner = te.reduce_axis((0, ic_bn // n_elems), name="ic_f_inner")
    ic_s_inner = te.reduce_axis((0, n_elems), name="ic_s_inner")

    out = te.compute(
        oshape,
        lambda n, oc_chunk, oh, ow, oc_block: te.sum(
            data_pad[
                n,
                ic_outer,
                oh * HSTR + kh * dilation_h,
                ow * WSTR + kw * dilation_w,
                ic_f_inner * n_elems + ic_s_inner,
            ].astype(out_dtype)
            * kernel[oc_chunk, ic_outer, kh, kw, ic_f_inner, oc_block, ic_s_inner].astype(
                out_dtype
            ),
            axis=[kh, kw, ic_outer, ic_f_inner, ic_s_inner],
        ),
    )

    return [data, kernel, out]


def test_conv2d_with_pipeline():

    # from tvm.script import tir as T
    @T.prim_func
    def main(
        data: T.Buffer[(1, 2, 56, 56, 32), "uint8"],
        placeholder: T.Buffer[(2, 2, 3, 3, 8, 32, 4), "uint8"],
        compute: T.Buffer[(1, 2, 56, 56, 32), "int32"],
    ):
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # body
        with T.block("root"):
            T.reads()
            T.writes()
            T.block_attr(
                {
                    "meta_schedule.parallel": 160,
                    "meta_schedule.unroll_explicit": 0,
                    "meta_schedule.vectorize": 32,
                }
            )
            data_pad = T.alloc_buffer([1, 2, 58, 58, 32], dtype="uint8")
            data_pad_global_vtcm = T.alloc_buffer(
                [1, 2, 58, 58, 32], dtype="uint8", scope="global.vtcm"
            )
            placeholder_global_vtcm = T.alloc_buffer(
                [2, 2, 3, 3, 8, 32, 4], dtype="uint8", scope="global.vtcm"
            )
            for i0, i1, i2, i3, i4 in T.grid(1, 2, 58, 58, 32):
                with T.block("data_pad"):
                    v_i0, v_i1, v_i2, v_i3, v_i4 = T.axis.remap("SSSSS", [i0, i1, i2, i3, i4])
                    T.reads(data[v_i0, v_i1, v_i2 - 1, v_i3 - 1, v_i4])
                    T.writes(data_pad[v_i0, v_i1, v_i2, v_i3, v_i4])
                    data_pad[v_i0, v_i1, v_i2, v_i3, v_i4] = T.if_then_else(
                        1 <= v_i2 and v_i2 < 57 and 1 <= v_i3 and v_i3 < 57,
                        data[v_i0, v_i1, v_i2 - 1, v_i3 - 1, v_i4],
                        T.uint8(0),
                        dtype="uint8",
                    )
            for n_0, oc_chunk_0, oh_0, ow_0, oc_block_0_0 in T.grid(1, 1, 14, 14, 1):
                for kh_0_kw_0_ic_outer_0_ic_f_inner_0_ic_s_inner_0_0_fused in T.serial(
                    2,
                    annotations={
                        "software_pipeline_async_stages": [0],
                        "software_pipeline_order": [0, 1, 2],
                        "software_pipeline_stage": [0, 0, 1],
                    },
                ):
                    for ax0_ax1_ax2_ax3_ax4_fused in T.serial(1152):
                        with T.block("data_pad_global.vtcm"):
                            v0 = T.axis.spatial(1, 0)
                            v1 = T.axis.spatial(2, ax0_ax1_ax2_ax3_ax4_fused // 576)
                            v2 = T.axis.spatial(
                                58, oh_0 * 4 + ax0_ax1_ax2_ax3_ax4_fused % 576 // 96
                            )
                            v3 = T.axis.spatial(58, ow_0 * 4 + ax0_ax1_ax2_ax3_ax4_fused % 96 // 16)
                            v4 = T.axis.spatial(
                                32,
                                kh_0_kw_0_ic_outer_0_ic_f_inner_0_ic_s_inner_0_0_fused * 16
                                + ax0_ax1_ax2_ax3_ax4_fused % 16,
                            )
                            T.reads(data_pad[v0, v1, v2, v3, v4])
                            T.writes(data_pad_global_vtcm[v0, v1, v2, v3, v4])
                            data_pad_global_vtcm[v0, v1, v2, v3, v4] = data_pad[v0, v1, v2, v3, v4]
                    for ax0_ax1_ax2_ax3_ax4_ax5_ax6_fused in T.serial(18432):
                        with T.block("placeholder_global.vtcm"):
                            v0 = T.axis.spatial(2, ax0_ax1_ax2_ax3_ax4_ax5_ax6_fused // 9216)
                            v1 = T.axis.spatial(2, ax0_ax1_ax2_ax3_ax4_ax5_ax6_fused % 9216 // 4608)
                            v2 = T.axis.spatial(3, ax0_ax1_ax2_ax3_ax4_ax5_ax6_fused % 4608 // 1536)
                            v3 = T.axis.spatial(3, ax0_ax1_ax2_ax3_ax4_ax5_ax6_fused % 1536 // 512)
                            v4 = T.axis.spatial(
                                8,
                                kh_0_kw_0_ic_outer_0_ic_f_inner_0_ic_s_inner_0_0_fused * 4
                                + ax0_ax1_ax2_ax3_ax4_ax5_ax6_fused % 512 // 128,
                            )
                            v5 = T.axis.spatial(32, ax0_ax1_ax2_ax3_ax4_ax5_ax6_fused % 128 // 4)
                            v6 = T.axis.spatial(4, ax0_ax1_ax2_ax3_ax4_ax5_ax6_fused % 4)
                            T.reads(placeholder[v0, v1, v2, v3, v4, v5, v6])
                            T.writes(placeholder_global_vtcm[v0, v1, v2, v3, v4, v5, v6])
                            placeholder_global_vtcm[v0, v1, v2, v3, v4, v5, v6] = placeholder[
                                v0, v1, v2, v3, v4, v5, v6
                            ]
                    for (
                        n_1,
                        oc_chunk_1,
                        oh_1,
                        ow_1,
                        oc_block_0_1,
                        kh_1,
                        kw_1,
                        ic_outer_1,
                        ic_f_inner_1,
                        ic_s_inner_0_1,
                        n_2,
                        oc_chunk_2,
                        oh_2,
                        ow_2,
                        oc_block_0_2,
                    ) in T.grid(1, 1, 4, 2, 1, 3, 3, 2, 4, 1, 1, 2, 1, 2, 1):
                        with T.block("compute_o"):
                            v_n = T.axis.spatial(1, n_2 + n_0 + n_1)
                            v_oc_chunk = T.axis.spatial(
                                2, oc_chunk_0 * 2 + oc_chunk_1 * 2 + oc_chunk_2
                            )
                            v_oh = T.axis.spatial(56, oh_2 + oh_0 * 4 + oh_1)
                            v_ow = T.axis.spatial(56, ow_0 * 4 + ow_1 * 2 + ow_2)
                            v_oc_block_o = T.axis.spatial(
                                1, oc_block_0_1 + oc_block_0_2 + oc_block_0_0
                            )
                            v_kh, v_kw, v_ic_outer = T.axis.remap("RRR", [kh_1, kw_1, ic_outer_1])
                            v_ic_f_inner = T.axis.reduce(
                                8,
                                kh_0_kw_0_ic_outer_0_ic_f_inner_0_ic_s_inner_0_0_fused * 4
                                + ic_f_inner_1,
                            )
                            v_ic_s_inner_o = T.axis.reduce(1, ic_s_inner_0_1)
                            T.reads(
                                data_pad_global_vtcm[
                                    v_n,
                                    v_ic_outer,
                                    v_oh + v_kh,
                                    v_ow + v_kw,
                                    v_ic_f_inner * 4 : v_ic_f_inner * 4 + 4,
                                ],
                                placeholder_global_vtcm[
                                    v_oc_chunk, v_ic_outer, v_kh, v_kw, v_ic_f_inner, 0:32, 0:4
                                ],
                            )
                            T.writes(compute[v_n, v_oc_chunk, v_oh, v_ow, 0:32])
                            T.block_attr(
                                {"meta_schedule.auto_tensorize": "dot_32x4_u8u8i32_vtcm_vrmpy"}
                            )
                            with T.init():
                                for oc_block_1 in T.serial(32):
                                    with T.block("compute_init"):
                                        v_oc_block_i_init = T.axis.spatial(32, oc_block_1)
                                        T.reads()
                                        T.writes(
                                            compute[v_n, v_oc_chunk, v_oh, v_ow, v_oc_block_i_init]
                                        )
                                        compute[v_n, v_oc_chunk, v_oh, v_ow, v_oc_block_i_init] = 0
                            for oc_block_1, ic_s_inner_1 in T.grid(32, 4):
                                with T.block("compute"):
                                    v_oc_block_i, v_ic_s_inner_i = T.axis.remap(
                                        "SR", [oc_block_1, ic_s_inner_1]
                                    )
                                    T.reads(
                                        compute[v_n, v_oc_chunk, v_oh, v_ow, v_oc_block_i],
                                        data_pad_global_vtcm[
                                            v_n,
                                            v_ic_outer,
                                            v_oh + v_kh,
                                            v_ow + v_kw,
                                            v_ic_f_inner * 4 + v_ic_s_inner_i,
                                        ],
                                        placeholder_global_vtcm[
                                            v_oc_chunk,
                                            v_ic_outer,
                                            v_kh,
                                            v_kw,
                                            v_ic_f_inner,
                                            v_oc_block_i,
                                            v_ic_s_inner_i,
                                        ],
                                    )
                                    T.writes(compute[v_n, v_oc_chunk, v_oh, v_ow, v_oc_block_i])
                                    T.block_attr({"meta_schedule.tiling_structure": "SRSRS"})
                                    compute[v_n, v_oc_chunk, v_oh, v_ow, v_oc_block_i] = compute[
                                        v_n, v_oc_chunk, v_oh, v_ow, v_oc_block_i
                                    ] + T.Cast(
                                        "int32",
                                        data_pad_global_vtcm[
                                            v_n,
                                            v_ic_outer,
                                            v_oh + v_kh,
                                            v_ow + v_kw,
                                            v_ic_f_inner * 4 + v_ic_s_inner_i,
                                        ],
                                    ) * T.Cast(
                                        "int32",
                                        placeholder_global_vtcm[
                                            v_oc_chunk,
                                            v_ic_outer,
                                            v_kh,
                                            v_kw,
                                            v_ic_f_inner,
                                            v_oc_block_i,
                                            v_ic_s_inner_i,
                                        ],
                                    )

    # fmt: on
    decision_0 = [
        ("SamplePerfectTile", [4, 1, 1]),
        ("SamplePerfectTile", [2, 1, 4]),
        ("SamplePerfectTile", [2, 4, 1]),
        ("SamplePerfectTile", [2, 2, 2]),
        ("SamplePerfectTile", [2, 4, 1]),
        ("SamplePerfectTile", [2, 4]),
        ("SamplePerfectTile", [2, 4]),
        ("SamplePerfectTile", [2, 4]),
        ("SamplePerfectTile", [2, 4]),
        ("SamplePerfectTile", [2, 4]),
        ("SampleCategorical", 0),
    ]

    strides = (1, 1)
    padding = (1, 1)
    dilation = (1, 1)

    mod = te.create_prim_func(
        conv2d_NCHWc_int8(64, 64, 56, 56, 3, 3, strides, padding, dilation, out_dtype="int32")
    )

    actual_design_space = generate_design_space(
        kind="hexagon",
        mod=mod,
        target=tvm.target.Target("hexagon -num-cores=10"),
        types=None,
        sch_rules=[
            multi_level_tiling_hexagon(use_software_pipeline=True),
            ms.schedule_rule.ParallelizeVectorizeUnroll(
                max_jobs_per_core=16,
                max_vectorize_extent=32,
                unroll_max_steps=[0, 16, 64, 512],
                unroll_explicit=True,
            ),
        ]
        + get_rules(kind="hexagon", types=ms.schedule_rule.AutoInline),
    )

    check_sketches(
        mod,
        sketches=actual_design_space,
        expected_mods=[main],
        expected_decisions=[decision_0],
    )


def test_conv_1x1():
    # fmt: off
    @T.prim_func
    def main(inputs: T.Buffer[(1, 16, 16, 32), "uint8"], weight: T.Buffer[(3, 3, 32, 32), "uint8"], conv2d_nhwc: T.Buffer[(1, 16, 16, 32), "int32"]):
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # body
        # with T.block("root")
        PadInput = T.alloc_buffer([1, 18, 18, 32], dtype="uint8")
        PadInput_global_vtcm = T.alloc_buffer([1, 18, 18, 32], dtype="uint8", scope="global.vtcm")
        weight_global_vtcm = T.alloc_buffer([3, 3, 32, 32], dtype="uint8", scope="global.vtcm")
        for i0, i1, i2, i3 in T.grid(1, 18, 18, 32):
            with T.block("PadInput"):
                v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(inputs[v_i0, v_i1 - 1, v_i2 - 1, v_i3])
                T.writes(PadInput[v_i0, v_i1, v_i2, v_i3])
                PadInput[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(1 <= v_i1 and v_i1 < 17 and 1 <= v_i2 and v_i2 < 17, inputs[v_i0, v_i1 - 1, v_i2 - 1, v_i3], T.uint8(0), dtype="uint8")
        for n_0, h_0, w_0, co_0_0, rh_0, rw_0, rc_0_0 in T.grid(1, 4, 4, 1, 1, 1, 2):
            for ax0_ax1_ax2_ax3_fused in T.serial(576):
                with T.block("PadInput_global.vtcm"):
                    v0 = T.axis.spatial(1, 0)
                    v1 = T.axis.spatial(18, h_0 * 4 + ax0_ax1_ax2_ax3_fused // 96)
                    v2 = T.axis.spatial(18, w_0 * 4 + ax0_ax1_ax2_ax3_fused % 96 // 16)
                    v3 = T.axis.spatial(32, rc_0_0 * 16 + ax0_ax1_ax2_ax3_fused % 16)
                    T.reads(PadInput[v0, v1, v2, v3])
                    T.writes(PadInput_global_vtcm[v0, v1, v2, v3])
                    PadInput_global_vtcm[v0, v1, v2, v3] = PadInput[v0, v1, v2, v3]
            for ax0_ax1_ax2_ax3_fused in T.serial(4608):
                with T.block("weight_global.vtcm"):
                    v0 = T.axis.spatial(3, ax0_ax1_ax2_ax3_fused // 1536)
                    v1 = T.axis.spatial(3, ax0_ax1_ax2_ax3_fused % 1536 // 512)
                    v2 = T.axis.spatial(32, rc_0_0 * 16 + ax0_ax1_ax2_ax3_fused % 512 // 32)
                    v3 = T.axis.spatial(32, ax0_ax1_ax2_ax3_fused % 32)
                    T.reads(weight[v0, v1, v2, v3])
                    T.writes(weight_global_vtcm[v0, v1, v2, v3])
                    weight_global_vtcm[v0, v1, v2, v3] = weight[v0, v1, v2, v3]
            for n_1, h_1, w_1, co_0_1, rh_1, rw_1, rc_0_1, n_2, h_2, w_2, co_0_2 in T.grid(1, 1, 4, 1, 3, 3, 4, 1, 4, 1, 1):
                with T.block("conv2d_nhwc_o"):
                    v_n = T.axis.spatial(1, n_1 + n_2 + n_0)
                    v_h = T.axis.spatial(16, h_0 * 4 + h_1 * 4 + h_2)
                    v_w = T.axis.spatial(16, w_2 + w_0 * 4 + w_1)
                    v_co_o = T.axis.spatial(1, co_0_1 + co_0_2 + co_0_0)
                    v_rh = T.axis.reduce(3, rh_0 * 3 + rh_1)
                    v_rw = T.axis.reduce(3, rw_0 * 3 + rw_1)
                    v_rc_o = T.axis.reduce(8, rc_0_0 * 4 + rc_0_1)
                    T.reads(PadInput_global_vtcm[v_n, v_h + v_rh, v_w + v_rw, v_rc_o * 4 : v_rc_o * 4 + 4], weight_global_vtcm[v_rh, v_rw, v_rc_o * 4 : v_rc_o * 4 + 4, 0 : 32])
                    T.writes(conv2d_nhwc[v_n, v_h, v_w, 0 : 32])
                    T.block_attr({"meta_schedule.auto_tensorize":"dot_32x4_u8u8i32_vtcm_vrmpy"})
                    with T.init():
                        for co_1 in T.serial(32):
                            with T.block("conv2d_nhwc_init"):
                                v_co_i_init = T.axis.spatial(32, co_1)
                                T.reads()
                                T.writes(conv2d_nhwc[v_n, v_h, v_w, v_co_i_init])
                                conv2d_nhwc[v_n, v_h, v_w, v_co_i_init] = 0
                    for co_1, rc_1 in T.grid(32, 4):
                        with T.block("conv2d_nhwc"):
                            v_co_i, v_rc_i = T.axis.remap("SR", [co_1, rc_1])
                            T.reads(conv2d_nhwc[v_n, v_h, v_w, v_co_i], PadInput_global_vtcm[v_n, v_h + v_rh, v_w + v_rw, v_rc_o * 4 + v_rc_i], weight_global_vtcm[v_rh, v_rw, v_rc_o * 4 + v_rc_i, v_co_i])
                            T.writes(conv2d_nhwc[v_n, v_h, v_w, v_co_i])
                            T.block_attr({"meta_schedule.tiling_structure":"SRSRS"})
                            conv2d_nhwc[v_n, v_h, v_w, v_co_i] = conv2d_nhwc[v_n, v_h, v_w, v_co_i] + T.Cast("int32", PadInput_global_vtcm[v_n, v_h + v_rh, v_w + v_rw, v_rc_o * 4 + v_rc_i]) * T.Cast("int32", weight_global_vtcm[v_rh, v_rw, v_rc_o * 4 + v_rc_i, v_co_i])
    # fmt: on
    decision_0 = [
        ("SamplePerfectTile", [4, 1, 1]),
        ("SamplePerfectTile", [2, 1, 4]),
        ("SamplePerfectTile", [2, 4, 1]),
        ("SamplePerfectTile", [2, 2, 2]),
        ("SamplePerfectTile", [2, 4]),
        ("SamplePerfectTile", [2, 4]),
        ("SamplePerfectTile", [2, 4]),
    ]

    mod = te.create_prim_func(
        te_workload.conv2d_nhwc(
            N=1,
            H=16,
            W=16,
            CI=32,
            CO=32,
            kernel_size=3,
            stride=1,
            padding=1,
            in_dtype="uint8",
            out_dtype="int32",
        )
    )

    actual_design_space = generate_design_space(
        kind="hexagon",
        mod=mod,
        target=tvm.target.Target("hexagon"),
        types=None,
        sch_rules=[
            multi_level_tiling_hexagon(),
        ]
        + get_rules(kind="hexagon", types=ms.schedule_rule.AutoInline),
    )

    check_sketches(
        mod,
        sketches=actual_design_space,
        expected_mods=[main],
        expected_decisions=[decision_0],
    )


def test_matmul_relu_non_tensorizable():
    # expected to do nothing on non-tensorizable workloads
    mod = te.create_prim_func(
        te_workload.matmul_relu(  # dtype doesn't match tensor intrin
            n=128,
            m=128,
            k=128,
        )
    )
    (sch,) = generate_design_space(
        kind="hexagon",
        mod=mod,
        target=tvm.target.Target("hexagon"),
        types=None,
        sch_rules=[multi_level_tiling_hexagon(write_reuse_scope="global")]
        + get_rules("hexagon", ms.schedule_rule.AutoInline),
    )
    tvm.ir.assert_structural_equal(mod, sch.mod["main"])


if __name__ == "__main__":
    test_dense_base()
    test_dense_with_fallback()
    test_dense_global()
    test_dense_with_pipeline()
    test_padded_dense()
    test_conv2d()
    test_conv_1x1()
    test_conv2d_with_pipeline()
    test_matmul_relu_non_tensorizable()
