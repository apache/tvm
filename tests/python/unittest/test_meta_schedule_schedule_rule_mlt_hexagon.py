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
            levels=[1],
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
    def main(X: T.Buffer[(128, 768), "uint8"], packed_width: T.Buffer[(24, 192, 32, 4), "uint8"], compute: T.Buffer[(128, 768), "int32"]):
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # body
        # with T.block("root")
        compute_reindex = T.alloc_buffer([128, 768], dtype="int32")
        X_reindex = T.alloc_buffer([128, 768], dtype="uint8")
        packed_width_reindex = T.alloc_buffer([768, 768], dtype="uint8")
        X_reindex_global_vtcm = T.alloc_buffer([128, 768], dtype="uint8", scope="global.vtcm")
        packed_width_reindex_global_vtcm = T.alloc_buffer([768, 768], dtype="uint8", scope="global.vtcm")
        for ax0, ax1 in T.grid(128, 768):
            with T.block("X_reindex_reindex"):
                v0, v1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(X[v0, v1])
                T.writes(X_reindex[v0, v1])
                X_reindex[v0, v1] = X[v0, v1]
        for ax0, ax1 in T.grid(768, 768):
            with T.block("packed_width_reindex_reindex"):
                v0, v1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(packed_width[v0 // 32, v1 // 4, v0 % 32, v1 % 4])
                T.writes(packed_width_reindex[v0, v1])
                packed_width_reindex[v0, v1] = packed_width[v0 // 32, v1 // 4, v0 % 32, v1 % 4]
        for ax0_0, ax1_0_0 in T.grid(128, 6):
            for ax0_ax1_fused in T.serial(768):
                with T.block("X_reindex_global.vtcm"):
                    v0, v1 = T.axis.remap("SS", [ax0_0, ax0_ax1_fused])
                    T.reads(X_reindex[v0, v1])
                    T.writes(X_reindex_global_vtcm[v0, v1])
                    X_reindex_global_vtcm[v0, v1] = X_reindex[v0, v1]
            for ax0_ax1_fused in T.serial(98304):
                with T.block("packed_width_reindex_global.vtcm"):
                    v0 = T.axis.spatial(768, ax1_0_0 * 128 + ax0_ax1_fused // 768)
                    v1 = T.axis.spatial(768, ax0_ax1_fused % 768)
                    T.reads(packed_width_reindex[v0, v1])
                    T.writes(packed_width_reindex_global_vtcm[v0, v1])
                    packed_width_reindex_global_vtcm[v0, v1] = packed_width_reindex[v0, v1]
            for ax2_0_0, ax0_1, ax1_0_1, ax2_0_1, ax0_2, ax1_0_2 in T.grid(48, 1, 2, 4, 1, 2):
                with T.block("compute_o"):
                    v0 = T.axis.spatial(128, ax0_0 + ax0_1 + ax0_2)
                    v1_o = T.axis.spatial(24, ax1_0_0 * 4 + ax1_0_1 * 2 + ax1_0_2)
                    v2_o = T.axis.reduce(192, ax2_0_0 * 4 + ax2_0_1)
                    T.reads(X_reindex_global_vtcm[v0, v2_o * 4 : v2_o * 4 + 4], packed_width_reindex_global_vtcm[v1_o * 32 : v1_o * 32 + 32, v2_o * 4 : v2_o * 4 + 4])
                    T.writes(compute_reindex[v0, v1_o * 32 : v1_o * 32 + 32])
                    T.block_attr({"meta_schedule.auto_tensorize":"dot_32x4_u8u8i32_vtcm_vrmpy"})
                    with T.init():
                        for ax1_1 in T.serial(32):
                            with T.block("compute_init"):
                                v1_i_init = T.axis.spatial(32, ax1_1)
                                T.reads()
                                T.writes(compute_reindex[v0, v1_o * 32 + v1_i_init])
                                compute_reindex[v0, v1_o * 32 + v1_i_init] = 0
                    for ax1_1, ax2_1 in T.grid(32, 4):
                        with T.block("compute"):
                            v1_i, v2_i = T.axis.remap("SR", [ax1_1, ax2_1])
                            T.reads(compute_reindex[v0, v1_o * 32 + v1_i], X_reindex_global_vtcm[v0, v2_o * 4 + v2_i], packed_width_reindex_global_vtcm[v1_o * 32 + v1_i, v2_o * 4 + v2_i])
                            T.writes(compute_reindex[v0, v1_o * 32 + v1_i])
                            T.block_attr({"meta_schedule.tiling_structure":"SRSRS"})
                            compute_reindex[v0, v1_o * 32 + v1_i] = compute_reindex[v0, v1_o * 32 + v1_i] + T.Cast("int32", X_reindex_global_vtcm[v0, v2_o * 4 + v2_i]) * T.Cast("int32", packed_width_reindex_global_vtcm[v1_o * 32 + v1_i, v2_o * 4 + v2_i])
        for ax0, ax1 in T.grid(128, 768):
            with T.block("compute_reindex"):
                v0, v1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(compute_reindex[v0, v1])
                T.writes(compute[v0, v1])
                compute[v0, v1] = compute_reindex[v0, v1]
    
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
    def main(X: T.Buffer[(128, 768), "uint8"], packed_width: T.Buffer[(24, 192, 32, 4), "uint8"], compute: T.Buffer[(128, 768), "int32"]):
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # body
        # with T.block("root")
        compute_reindex = T.alloc_buffer([128, 768], dtype="int32")
        X_reindex = T.alloc_buffer([128, 768], dtype="uint8")
        packed_width_reindex = T.alloc_buffer([768, 768], dtype="uint8")
        X_reindex_global_vtcm = T.alloc_buffer([128, 768], dtype="uint8", scope="global.vtcm")
        packed_width_reindex_global_vtcm = T.alloc_buffer([768, 768], dtype="uint8", scope="global.vtcm")
        for ax0, ax1 in T.grid(128, 768):
            with T.block("X_reindex_reindex"):
                v0, v1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(X[v0, v1])
                T.writes(X_reindex[v0, v1])
                X_reindex[v0, v1] = X[v0, v1]
        for ax0, ax1 in T.grid(768, 768):
            with T.block("packed_width_reindex_reindex"):
                v0, v1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(packed_width[v0 // 32, v1 // 4, v0 % 32, v1 % 4])
                T.writes(packed_width_reindex[v0, v1])
                packed_width_reindex[v0, v1] = packed_width[v0 // 32, v1 // 4, v0 % 32, v1 % 4]
        for ax0_0, ax1_0_0 in T.grid(128, 6):
            for ax0_ax1_fused in T.serial(768):
                with T.block("X_reindex_global.vtcm"):
                    v0, v1 = T.axis.remap("SS", [ax0_0, ax0_ax1_fused])
                    T.reads(X_reindex[v0, v1])
                    T.writes(X_reindex_global_vtcm[v0, v1])
                    X_reindex_global_vtcm[v0, v1] = X_reindex[v0, v1]
            for ax0_ax1_fused in T.serial(98304):
                with T.block("packed_width_reindex_global.vtcm"):
                    v0 = T.axis.spatial(768, ax1_0_0 * 128 + ax0_ax1_fused // 768)
                    v1 = T.axis.spatial(768, ax0_ax1_fused % 768)
                    T.reads(packed_width_reindex[v0, v1])
                    T.writes(packed_width_reindex_global_vtcm[v0, v1])
                    packed_width_reindex_global_vtcm[v0, v1] = packed_width_reindex[v0, v1]
            for ax2_0_0, ax0_1, ax1_0_1, ax2_0_1, ax0_2, ax1_0_2 in T.grid(192, 1, 2, 1, 1, 2):
                with T.block("compute_o"):
                    v0 = T.axis.spatial(128, ax0_0 + ax0_1 + ax0_2)
                    v1_o = T.axis.spatial(24, ax1_0_0 * 4 + ax1_0_1 * 2 + ax1_0_2)
                    v2_o = T.axis.reduce(192, ax2_0_1 + ax2_0_0)
                    T.reads(X_reindex_global_vtcm[v0, v2_o * 4 : v2_o * 4 + 4], packed_width_reindex_global_vtcm[v1_o * 32 : v1_o * 32 + 32, v2_o * 4 : v2_o * 4 + 4])
                    T.writes(compute_reindex[v0, v1_o * 32 : v1_o * 32 + 32])
                    T.block_attr({"meta_schedule.auto_tensorize":"dot_32x4_u8u8i32_vtcm_vrmpy"})
                    with T.init():
                        for ax1_1 in T.serial(32):
                            with T.block("compute_init"):
                                v1_i_init = T.axis.spatial(32, ax1_1)
                                T.reads()
                                T.writes(compute_reindex[v0, v1_o * 32 + v1_i_init])
                                compute_reindex[v0, v1_o * 32 + v1_i_init] = 0
                    for ax1_1, ax2_1 in T.grid(32, 4):
                        with T.block("compute"):
                            v1_i, v2_i = T.axis.remap("SR", [ax1_1, ax2_1])
                            T.reads(compute_reindex[v0, v1_o * 32 + v1_i], X_reindex_global_vtcm[v0, v2_o * 4 + v2_i], packed_width_reindex_global_vtcm[v1_o * 32 + v1_i, v2_o * 4 + v2_i])
                            T.writes(compute_reindex[v0, v1_o * 32 + v1_i])
                            T.block_attr({"meta_schedule.tiling_structure":"SRSRS"})
                            compute_reindex[v0, v1_o * 32 + v1_i] = compute_reindex[v0, v1_o * 32 + v1_i] + T.Cast("int32", X_reindex_global_vtcm[v0, v2_o * 4 + v2_i]) * T.Cast("int32", packed_width_reindex_global_vtcm[v1_o * 32 + v1_i, v2_o * 4 + v2_i])
        for ax0, ax1 in T.grid(128, 768):
            with T.block("compute_reindex"):
                v0, v1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(compute_reindex[v0, v1])
                T.writes(compute[v0, v1])
                compute[v0, v1] = compute_reindex[v0, v1]
    
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
    def main(X: T.Buffer[(128, 768), "uint8"], packed_width: T.Buffer[(24, 192, 32, 4), "uint8"], compute: T.Buffer[(128, 768), "int32"]):
        # function attr dict
        T.func_attr({"tir.noalias": True, "global_symbol": "main"})
        # body
        # with T.block("root")
        compute_reindex = T.alloc_buffer([128, 768], dtype="int32")
        X_reindex = T.alloc_buffer([128, 768], dtype="uint8")
        packed_width_reindex = T.alloc_buffer([768, 768], dtype="uint8")
        X_reindex_global_vtcm = T.alloc_buffer([128, 768], dtype="uint8", scope="global.vtcm")
        packed_width_reindex_global_vtcm = T.alloc_buffer([768, 768], dtype="uint8", scope="global.vtcm")
        for ax0, ax1 in T.grid(128, 768):
            with T.block("X_reindex_reindex"):
                v0, v1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(X[v0, v1])
                T.writes(X_reindex[v0, v1])
                X_reindex[v0, v1] = X[v0, v1]
        for ax0, ax1 in T.grid(768, 768):
            with T.block("packed_width_reindex_reindex"):
                v0, v1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(packed_width[v0 // 32, v1 // 4, v0 % 32, v1 % 4])
                T.writes(packed_width_reindex[v0, v1])
                packed_width_reindex[v0, v1] = packed_width[v0 // 32, v1 // 4, v0 % 32, v1 % 4]
        for ax0_0 in T.serial(128):
            for ax1_0_0 in T.serial(6, annotations={"software_pipeline_async_stages":[0], "software_pipeline_order":[0, 1, 2], "software_pipeline_stage":[0, 0, 1]}):
                for ax0_ax1_fused in T.serial(768):
                    with T.block("X_reindex_global.vtcm"):
                        v0, v1 = T.axis.remap("SS", [ax0_0, ax0_ax1_fused])
                        T.reads(X_reindex[v0, v1])
                        T.writes(X_reindex_global_vtcm[v0, v1])
                        X_reindex_global_vtcm[v0, v1] = X_reindex[v0, v1]
                for ax0_ax1_fused in T.serial(98304):
                    with T.block("packed_width_reindex_global.vtcm"):
                        v0 = T.axis.spatial(768, ax1_0_0 * 128 + ax0_ax1_fused // 768)
                        v1 = T.axis.spatial(768, ax0_ax1_fused % 768)
                        T.reads(packed_width_reindex[v0, v1])
                        T.writes(packed_width_reindex_global_vtcm[v0, v1])
                        packed_width_reindex_global_vtcm[v0, v1] = packed_width_reindex[v0, v1]
                for ax2_0_0, ax0_1, ax1_0_1, ax2_0_1, ax0_2, ax1_0_2 in T.grid(48, 1, 2, 4, 1, 2):
                    with T.block("compute_o"):
                        v0 = T.axis.spatial(128, ax0_0 + ax0_1 + ax0_2)
                        v1_o = T.axis.spatial(24, ax1_0_0 * 4 + ax1_0_1 * 2 + ax1_0_2)
                        v2_o = T.axis.reduce(192, ax2_0_0 * 4 + ax2_0_1)
                        T.reads(X_reindex_global_vtcm[v0, v2_o * 4 : v2_o * 4 + 4], packed_width_reindex_global_vtcm[v1_o * 32 : v1_o * 32 + 32, v2_o * 4 : v2_o * 4 + 4])
                        T.writes(compute_reindex[v0, v1_o * 32 : v1_o * 32 + 32])
                        T.block_attr({"meta_schedule.auto_tensorize":"dot_32x4_u8u8i32_vtcm_vrmpy"})
                        with T.init():
                            for ax1_1 in T.serial(32):
                                with T.block("compute_init"):
                                    v1_i_init = T.axis.spatial(32, ax1_1)
                                    T.reads()
                                    T.writes(compute_reindex[v0, v1_o * 32 + v1_i_init])
                                    compute_reindex[v0, v1_o * 32 + v1_i_init] = 0
                        for ax1_1, ax2_1 in T.grid(32, 4):
                            with T.block("compute"):
                                v1_i, v2_i = T.axis.remap("SR", [ax1_1, ax2_1])
                                T.reads(compute_reindex[v0, v1_o * 32 + v1_i], X_reindex_global_vtcm[v0, v2_o * 4 + v2_i], packed_width_reindex_global_vtcm[v1_o * 32 + v1_i, v2_o * 4 + v2_i])
                                T.writes(compute_reindex[v0, v1_o * 32 + v1_i])
                                T.block_attr({"meta_schedule.tiling_structure":"SRSRS"})
                                compute_reindex[v0, v1_o * 32 + v1_i] = compute_reindex[v0, v1_o * 32 + v1_i] + T.Cast("int32", X_reindex_global_vtcm[v0, v2_o * 4 + v2_i]) * T.Cast("int32", packed_width_reindex_global_vtcm[v1_o * 32 + v1_i, v2_o * 4 + v2_i])
        for ax0, ax1 in T.grid(128, 768):
            with T.block("compute_reindex"):
                v0, v1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(compute_reindex[v0, v1])
                T.writes(compute[v0, v1])
                compute[v0, v1] = compute_reindex[v0, v1]
    
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
    def main(X: T.Buffer[(128, 768), "uint8"], packed_width: T.Buffer[(24, 192, 32, 4), "uint8"], compute: T.Buffer[(128, 768), "int32"]):
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # body
        # with T.block("root")
        compute_reindex = T.alloc_buffer([128, 768], dtype="int32")
        X_reindex = T.alloc_buffer([128, 768], dtype="uint8")
        packed_width_reindex = T.alloc_buffer([768, 768], dtype="uint8")
        X_reindex_global_vtcm = T.alloc_buffer([128, 768], dtype="uint8", scope="global.vtcm")
        packed_width_reindex_global_vtcm = T.alloc_buffer([768, 768], dtype="uint8", scope="global.vtcm")
        for ax0, ax1 in T.grid(128, 768):
            with T.block("X_reindex_reindex"):
                v0, v1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(X[v0, v1])
                T.writes(X_reindex[v0, v1])
                X_reindex[v0, v1] = X[v0, v1]
        for ax0, ax1 in T.grid(768, 768):
            with T.block("packed_width_reindex_reindex"):
                v0, v1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(packed_width[v0 // 32, v1 // 4, v0 % 32, v1 % 4])
                T.writes(packed_width_reindex[v0, v1])
                packed_width_reindex[v0, v1] = packed_width[v0 // 32, v1 // 4, v0 % 32, v1 % 4]
        for ax0_0, ax1_0_0 in T.grid(128, 6):
            for ax0_ax1_fused in T.serial(768):
                with T.block("X_reindex_global.vtcm"):
                    v0, v1 = T.axis.remap("SS", [ax0_0, ax0_ax1_fused])
                    T.reads(X_reindex[v0, v1])
                    T.writes(X_reindex_global_vtcm[v0, v1])
                    X_reindex_global_vtcm[v0, v1] = X_reindex[v0, v1]
            for ax0_ax1_fused in T.serial(98304):
                with T.block("packed_width_reindex_global.vtcm"):
                    v0 = T.axis.spatial(768, ax1_0_0 * 128 + ax0_ax1_fused // 768)
                    v1 = T.axis.spatial(768, ax0_ax1_fused % 768)
                    T.reads(packed_width_reindex[v0, v1])
                    T.writes(packed_width_reindex_global_vtcm[v0, v1])
                    packed_width_reindex_global_vtcm[v0, v1] = packed_width_reindex[v0, v1]
            for ax2_0_0, ax0_1, ax1_0_1, ax2_0_1, ax0_2, ax1_0_2 in T.grid(192, 1, 2, 1, 1, 2):
                with T.block("compute_o"):
                    v0 = T.axis.spatial(128, ax0_0 + ax0_1 + ax0_2)
                    v1_o = T.axis.spatial(24, ax1_0_0 * 4 + ax1_0_1 * 2 + ax1_0_2)
                    v2_o = T.axis.reduce(192, ax2_0_1 + ax2_0_0)
                    T.reads(X_reindex_global_vtcm[v0, v2_o * 4 : v2_o * 4 + 4], packed_width_reindex_global_vtcm[v1_o * 32 : v1_o * 32 + 32, v2_o * 4 : v2_o * 4 + 4])
                    T.writes(compute_reindex[v0, v1_o * 32 : v1_o * 32 + 32])
                    T.block_attr({"meta_schedule.auto_tensorize":"dot_32x4_u8u8i32_vtcm_vrmpy"})
                    with T.init():
                        for ax1_1 in T.serial(32):
                            with T.block("compute_init"):
                                v1_i_init = T.axis.spatial(32, ax1_1)
                                T.reads()
                                T.writes(compute_reindex[v0, v1_o * 32 + v1_i_init])
                                compute_reindex[v0, v1_o * 32 + v1_i_init] = 0
                    for ax1_1, ax2_1 in T.grid(32, 4):
                        with T.block("compute"):
                            v1_i, v2_i = T.axis.remap("SR", [ax1_1, ax2_1])
                            T.reads(compute_reindex[v0, v1_o * 32 + v1_i], X_reindex_global_vtcm[v0, v2_o * 4 + v2_i], packed_width_reindex_global_vtcm[v1_o * 32 + v1_i, v2_o * 4 + v2_i])
                            T.writes(compute_reindex[v0, v1_o * 32 + v1_i])
                            T.block_attr({"meta_schedule.tiling_structure":"SRSRS"})
                            compute_reindex[v0, v1_o * 32 + v1_i] = compute_reindex[v0, v1_o * 32 + v1_i] + T.Cast("int32", X_reindex_global_vtcm[v0, v2_o * 4 + v2_i]) * T.Cast("int32", packed_width_reindex_global_vtcm[v1_o * 32 + v1_i, v2_o * 4 + v2_i])
        for ax0, ax1 in T.grid(128, 768):
            with T.block("compute_reindex"):
                v0, v1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(compute_reindex[v0, v1])
                T.writes(compute[v0, v1])
                compute[v0, v1] = compute_reindex[v0, v1]
    
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
    def main(X: T.Buffer[(128, 768), "uint8"], packed_width: T.Buffer[(24, 192, 32, 4), "uint8"], compute: T.Buffer[(128, 768), "int32"]):
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # body
        # with T.block("root")
        compute_reindex = T.alloc_buffer([128, 768], dtype="int32")
        X_reindex = T.alloc_buffer([128, 768], dtype="uint8")
        packed_width_reindex = T.alloc_buffer([768, 768], dtype="uint8")
        X_reindex_global_vtcm = T.alloc_buffer([128, 768], dtype="uint8", scope="global.vtcm")
        packed_width_reindex_global_vtcm = T.alloc_buffer([768, 768], dtype="uint8", scope="global.vtcm")
        for ax0, ax1 in T.grid(128, 768):
            with T.block("X_reindex_reindex"):
                v0, v1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(X[v0, v1])
                T.writes(X_reindex[v0, v1])
                X_reindex[v0, v1] = X[v0, v1]
        for ax0, ax1 in T.grid(768, 768):
            with T.block("packed_width_reindex_reindex"):
                v0, v1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(packed_width[v0 // 32, v1 // 4, v0 % 32, v1 % 4])
                T.writes(packed_width_reindex[v0, v1])
                packed_width_reindex[v0, v1] = packed_width[v0 // 32, v1 // 4, v0 % 32, v1 % 4]
        for ax0_0, ax1_0_0 in T.grid(128, 6):
            for ax0_ax1_fused in T.serial(768):
                with T.block("X_reindex_global.vtcm"):
                    v0, v1 = T.axis.remap("SS", [ax0_0, ax0_ax1_fused])
                    T.reads(X_reindex[v0, v1])
                    T.writes(X_reindex_global_vtcm[v0, v1])
                    X_reindex_global_vtcm[v0, v1] = X_reindex[v0, v1]
            for ax0_ax1_fused in T.serial(98304):
                with T.block("packed_width_reindex_global.vtcm"):
                    v0 = T.axis.spatial(768, ax1_0_0 * 128 + ax0_ax1_fused // 768)
                    v1 = T.axis.spatial(768, ax0_ax1_fused % 768)
                    T.reads(packed_width_reindex[v0, v1])
                    T.writes(packed_width_reindex_global_vtcm[v0, v1])
                    packed_width_reindex_global_vtcm[v0, v1] = packed_width_reindex[v0, v1]
            for ax2_0_0, ax0_1, ax1_0_1, ax2_0_1, ax0_2, ax1_0_2 in T.grid(48, 1, 2, 4, 1, 2):
                with T.block("compute_o"):
                    v0 = T.axis.spatial(128, ax0_0 + ax0_1 + ax0_2)
                    v1_o = T.axis.spatial(24, ax1_0_0 * 4 + ax1_0_1 * 2 + ax1_0_2)
                    v2_o = T.axis.reduce(192, ax2_0_0 * 4 + ax2_0_1)
                    T.reads(X_reindex_global_vtcm[v0, v2_o * 4 : v2_o * 4 + 4], packed_width_reindex_global_vtcm[v1_o * 32 : v1_o * 32 + 32, v2_o * 4 : v2_o * 4 + 4])
                    T.writes(compute_reindex[v0, v1_o * 32 : v1_o * 32 + 32])
                    T.block_attr({"meta_schedule.auto_tensorize":"dot_32x4_u8u8i32_vtcm_vrmpy"})
                    with T.init():
                        for ax1_1 in T.serial(32):
                            with T.block("compute_init"):
                                v1_i_init = T.axis.spatial(32, ax1_1)
                                T.reads()
                                T.writes(compute_reindex[v0, v1_o * 32 + v1_i_init])
                                compute_reindex[v0, v1_o * 32 + v1_i_init] = 0
                    for ax1_1, ax2_1 in T.grid(32, 4):
                        with T.block("compute"):
                            v1_i, v2_i = T.axis.remap("SR", [ax1_1, ax2_1])
                            T.reads(compute_reindex[v0, v1_o * 32 + v1_i], X_reindex_global_vtcm[v0, v2_o * 4 + v2_i], packed_width_reindex_global_vtcm[v1_o * 32 + v1_i, v2_o * 4 + v2_i])
                            T.writes(compute_reindex[v0, v1_o * 32 + v1_i])
                            T.block_attr({"meta_schedule.tiling_structure":"SRSRS"})
                            compute_reindex[v0, v1_o * 32 + v1_i] = compute_reindex[v0, v1_o * 32 + v1_i] + T.Cast("int32", X_reindex_global_vtcm[v0, v2_o * 4 + v2_i]) * T.Cast("int32", packed_width_reindex_global_vtcm[v1_o * 32 + v1_i, v2_o * 4 + v2_i])
        for ax0, ax1 in T.grid(128, 768):
            with T.block("compute_reindex"):
                v0, v1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(compute_reindex[v0, v1])
                T.writes(compute[v0, v1])
                compute[v0, v1] = compute_reindex[v0, v1]
    
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
    def main(inputs: T.Buffer[(1, 16, 16, 32), "uint8"], weight: T.Buffer[(3, 3, 32, 32), "uint8"], conv2d_nhwc: T.Buffer[(1, 16, 16, 32), "int32"]):
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # body
        # with T.block("root")
        PadInput = T.alloc_buffer([1, 18, 18, 32], dtype="uint8")
        conv2d_nhwc_reindex = T.alloc_buffer([1, 16, 16, 32], dtype="int32")
        PadInput_reindex = T.alloc_buffer([1, 16, 16, 288], dtype="uint8")
        weight_reindex = T.alloc_buffer([32, 288], dtype="uint8")
        PadInput_reindex_global_vtcm = T.alloc_buffer([1, 16, 16, 288], dtype="uint8", scope="global.vtcm")
        weight_reindex_global_vtcm = T.alloc_buffer([32, 288], dtype="uint8", scope="global.vtcm")
        for i0, i1, i2, i3 in T.grid(1, 18, 18, 32):
            with T.block("PadInput"):
                i0_1, i1_1, i2_1, i3_1 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(inputs[i0_1, i1_1 - 1, i2_1 - 1, i3_1])
                T.writes(PadInput[i0_1, i1_1, i2_1, i3_1])
                PadInput[i0_1, i1_1, i2_1, i3_1] = T.if_then_else(1 <= i1_1 and i1_1 < 17 and 1 <= i2_1 and i2_1 < 17, inputs[i0_1, i1_1 - 1, i2_1 - 1, i3_1], T.uint8(0), dtype="uint8")
        for ax0, ax1, ax2, ax3 in T.grid(1, 16, 16, 288):
            with T.block("PadInput_reindex_reindex"):
                v0, v1, v2, v3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(PadInput[v0, v3 // 96 + v1, v3 % 96 // 32 + v2, v3 % 32])
                T.writes(PadInput_reindex[v0, v1, v2, v3])
                PadInput_reindex[v0, v1, v2, v3] = PadInput[v0, v3 // 96 + v1, v3 % 96 // 32 + v2, v3 % 32]
        for ax0, ax1 in T.grid(32, 288):
            with T.block("weight_reindex_reindex"):
                v0, v1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(weight[v1 // 96, v1 % 96 // 32, v1 % 32, v0])
                T.writes(weight_reindex[v0, v1])
                weight_reindex[v0, v1] = weight[v1 // 96, v1 % 96 // 32, v1 % 32, v0]
        for ax0_0, ax1_0, ax2_0, ax3_0_0 in T.grid(1, 4, 4, 1):
            for ax0_ax1_ax2_ax3_fused in T.serial(4608):
                with T.block("PadInput_reindex_global.vtcm"):
                    v0 = T.axis.spatial(1, 0)
                    v1 = T.axis.spatial(16, ax1_0 * 4 + ax0_ax1_ax2_ax3_fused // 1152)
                    v2 = T.axis.spatial(16, ax2_0 * 4 + ax0_ax1_ax2_ax3_fused % 1152 // 288)
                    v3 = T.axis.spatial(288, ax0_ax1_ax2_ax3_fused % 288)
                    T.reads(PadInput_reindex[v0, v1, v2, v3])
                    T.writes(PadInput_reindex_global_vtcm[v0, v1, v2, v3])
                    PadInput_reindex_global_vtcm[v0, v1, v2, v3] = PadInput_reindex[v0, v1, v2, v3]
            for ax0_ax1_fused in T.serial(9216):
                with T.block("weight_reindex_global.vtcm"):
                    v0 = T.axis.spatial(32, ax0_ax1_fused // 288)
                    v1 = T.axis.spatial(288, ax0_ax1_fused % 288)
                    T.reads(weight_reindex[v0, v1])
                    T.writes(weight_reindex_global_vtcm[v0, v1])
                    weight_reindex_global_vtcm[v0, v1] = weight_reindex[v0, v1]
            for ax4_0_0, ax0_1, ax1_1, ax2_1, ax3_0_1, ax4_0_1, ax0_2, ax1_2, ax2_2, ax3_0_2 in T.grid(18, 1, 1, 4, 1, 4, 1, 4, 1, 1):
                with T.block("conv2d_nhwc_o"):
                    v0 = T.axis.spatial(1, 0)
                    v1 = T.axis.spatial(16, ax1_0 * 4 + ax1_1 * 4 + ax1_2)
                    v2 = T.axis.spatial(16, ax2_2 + ax2_0 * 4 + ax2_1)
                    v3_o = T.axis.spatial(1, 0)
                    v4_o = T.axis.reduce(72, ax4_0_0 * 4 + ax4_0_1)
                    T.reads(PadInput_reindex_global_vtcm[v0, v1, v2, v4_o * 4 : v4_o * 4 + 4], weight_reindex_global_vtcm[0 : 32, v4_o * 4 : v4_o * 4 + 4])
                    T.writes(conv2d_nhwc_reindex[v0, v1, v2, 0 : 32])
                    T.block_attr({"meta_schedule.auto_tensorize":"dot_32x4_u8u8i32_vtcm_vrmpy"})
                    with T.init():
                        for ax3_1 in T.serial(32):
                            with T.block("conv2d_nhwc_init"):
                                v3_i_init = T.axis.spatial(32, ax3_1)
                                T.reads()
                                T.writes(conv2d_nhwc_reindex[v0, v1, v2, v3_i_init])
                                conv2d_nhwc_reindex[v0, v1, v2, v3_i_init] = 0
                    for ax3_1, ax4_1 in T.grid(32, 4):
                        with T.block("conv2d_nhwc"):
                            v3_i, v4_i = T.axis.remap("SR", [ax3_1, ax4_1])
                            T.reads(conv2d_nhwc_reindex[v0, v1, v2, v3_i], PadInput_reindex_global_vtcm[v0, v1, v2, v4_o * 4 + v4_i], weight_reindex_global_vtcm[v3_i, v4_o * 4 + v4_i])
                            T.writes(conv2d_nhwc_reindex[v0, v1, v2, v3_i])
                            T.block_attr({"meta_schedule.tiling_structure":"SRSRS"})
                            conv2d_nhwc_reindex[v0, v1, v2, v3_i] = conv2d_nhwc_reindex[v0, v1, v2, v3_i] + T.Cast("int32", PadInput_reindex_global_vtcm[v0, v1, v2, v4_o * 4 + v4_i]) * T.Cast("int32", weight_reindex_global_vtcm[v3_i, v4_o * 4 + v4_i])
        for ax0, ax1, ax2, ax3 in T.grid(1, 16, 16, 32):
            with T.block("conv2d_nhwc_reindex"):
                v0, v1, v2, v3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(conv2d_nhwc_reindex[v0, v1, v2, v3])
                T.writes(conv2d_nhwc[v0, v1, v2, v3])
                conv2d_nhwc[v0, v1, v2, v3] = conv2d_nhwc_reindex[v0, v1, v2, v3]
    
    # fmt: on
    decision_0 = [
        ("SamplePerfectTile", [4, 1, 1]),
        ("SamplePerfectTile", [2, 1, 4]),
        ("SamplePerfectTile", [2, 4, 1]),
        ("SamplePerfectTile", [2, 2, 2]),
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

def test_conv2d_with_pipeline():

    # from tvm.script import tir as T
    @T.prim_func
    def main(inputs: T.Buffer[(1, 16, 16, 32), "uint8"], weight: T.Buffer[(3, 3, 32, 32), "uint8"], conv2d_nhwc: T.Buffer[(1, 16, 16, 32), "int32"]):
        # function attr dict
        T.func_attr({"tir.noalias": True, "global_symbol": "main"})
        # body
        # with T.block("root")
        PadInput = T.alloc_buffer([1, 18, 18, 32], dtype="uint8")
        conv2d_nhwc_reindex = T.alloc_buffer([1, 16, 16, 32], dtype="int32")
        PadInput_reindex = T.alloc_buffer([1, 16, 16, 288], dtype="uint8")
        weight_reindex = T.alloc_buffer([32, 288], dtype="uint8")
        PadInput_reindex_global_vtcm = T.alloc_buffer([1, 16, 16, 288], dtype="uint8", scope="global.vtcm")
        weight_reindex_global_vtcm = T.alloc_buffer([32, 288], dtype="uint8", scope="global.vtcm")
        for i0, i1, i2, i3 in T.grid(1, 18, 18, 32):
            with T.block("PadInput"):
                i0_1, i1_1, i2_1, i3_1 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(inputs[i0_1, i1_1 - 1, i2_1 - 1, i3_1])
                T.writes(PadInput[i0_1, i1_1, i2_1, i3_1])
                PadInput[i0_1, i1_1, i2_1, i3_1] = T.if_then_else(1 <= i1_1 and i1_1 < 17 and 1 <= i2_1 and i2_1 < 17, inputs[i0_1, i1_1 - 1, i2_1 - 1, i3_1], T.uint8(0), dtype="uint8")
        for ax0, ax1, ax2, ax3 in T.grid(1, 16, 16, 288):
            with T.block("PadInput_reindex_reindex"):
                v0, v1, v2, v3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(PadInput[v0, v3 // 96 + v1, v3 % 96 // 32 + v2, v3 % 32])
                T.writes(PadInput_reindex[v0, v1, v2, v3])
                PadInput_reindex[v0, v1, v2, v3] = PadInput[v0, v3 // 96 + v1, v3 % 96 // 32 + v2, v3 % 32]
        for ax0, ax1 in T.grid(32, 288):
            with T.block("weight_reindex_reindex"):
                v0, v1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(weight[v1 // 96, v1 % 96 // 32, v1 % 32, v0])
                T.writes(weight_reindex[v0, v1])
                weight_reindex[v0, v1] = weight[v1 // 96, v1 % 96 // 32, v1 % 32, v0]
        for ax0_0, ax1_0, ax2_0 in T.grid(1, 4, 4):
            for ax3_0_0 in T.serial(1, annotations={"software_pipeline_async_stages":[0], "software_pipeline_order":[0, 1, 2], "software_pipeline_stage":[0, 0, 1]}):
                for ax0_ax1_ax2_ax3_fused in T.serial(4608):
                    with T.block("PadInput_reindex_global.vtcm"):
                        v0 = T.axis.spatial(1, 0)
                        v1 = T.axis.spatial(16, ax1_0 * 4 + ax0_ax1_ax2_ax3_fused // 1152)
                        v2 = T.axis.spatial(16, ax2_0 * 4 + ax0_ax1_ax2_ax3_fused % 1152 // 288)
                        v3 = T.axis.spatial(288, ax0_ax1_ax2_ax3_fused % 288)
                        T.reads(PadInput_reindex[v0, v1, v2, v3])
                        T.writes(PadInput_reindex_global_vtcm[v0, v1, v2, v3])
                        PadInput_reindex_global_vtcm[v0, v1, v2, v3] = PadInput_reindex[v0, v1, v2, v3]
                for ax0_ax1_fused in T.serial(9216):
                    with T.block("weight_reindex_global.vtcm"):
                        v0 = T.axis.spatial(32, ax0_ax1_fused // 288)
                        v1 = T.axis.spatial(288, ax0_ax1_fused % 288)
                        T.reads(weight_reindex[v0, v1])
                        T.writes(weight_reindex_global_vtcm[v0, v1])
                        weight_reindex_global_vtcm[v0, v1] = weight_reindex[v0, v1]
                for ax4_0_0, ax0_1, ax1_1, ax2_1, ax3_0_1, ax4_0_1, ax0_2, ax1_2, ax2_2, ax3_0_2 in T.grid(18, 1, 1, 4, 1, 4, 1, 4, 1, 1):
                    with T.block("conv2d_nhwc_o"):
                        v0 = T.axis.spatial(1, 0)
                        v1 = T.axis.spatial(16, ax1_0 * 4 + ax1_1 * 4 + ax1_2)
                        v2 = T.axis.spatial(16, ax2_2 + ax2_0 * 4 + ax2_1)
                        v3_o = T.axis.spatial(1, 0)
                        v4_o = T.axis.reduce(72, ax4_0_0 * 4 + ax4_0_1)
                        T.reads(PadInput_reindex_global_vtcm[v0, v1, v2, v4_o * 4 : v4_o * 4 + 4], weight_reindex_global_vtcm[0 : 32, v4_o * 4 : v4_o * 4 + 4])
                        T.writes(conv2d_nhwc_reindex[v0, v1, v2, 0 : 32])
                        T.block_attr({"meta_schedule.auto_tensorize":"dot_32x4_u8u8i32_vtcm_vrmpy"})
                        with T.init():
                            for ax3_1 in T.serial(32):
                                with T.block("conv2d_nhwc_init"):
                                    v3_i_init = T.axis.spatial(32, ax3_1)
                                    T.reads()
                                    T.writes(conv2d_nhwc_reindex[v0, v1, v2, v3_i_init])
                                    conv2d_nhwc_reindex[v0, v1, v2, v3_i_init] = 0
                        for ax3_1, ax4_1 in T.grid(32, 4):
                            with T.block("conv2d_nhwc"):
                                v3_i, v4_i = T.axis.remap("SR", [ax3_1, ax4_1])
                                T.reads(conv2d_nhwc_reindex[v0, v1, v2, v3_i], PadInput_reindex_global_vtcm[v0, v1, v2, v4_o * 4 + v4_i], weight_reindex_global_vtcm[v3_i, v4_o * 4 + v4_i])
                                T.writes(conv2d_nhwc_reindex[v0, v1, v2, v3_i])
                                T.block_attr({"meta_schedule.tiling_structure":"SRSRS"})
                                conv2d_nhwc_reindex[v0, v1, v2, v3_i] = conv2d_nhwc_reindex[v0, v1, v2, v3_i] + T.Cast("int32", PadInput_reindex_global_vtcm[v0, v1, v2, v4_o * 4 + v4_i]) * T.Cast("int32", weight_reindex_global_vtcm[v3_i, v4_o * 4 + v4_i])
        for ax0, ax1, ax2, ax3 in T.grid(1, 16, 16, 32):
            with T.block("conv2d_nhwc_reindex"):
                v0, v1, v2, v3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(conv2d_nhwc_reindex[v0, v1, v2, v3])
                T.writes(conv2d_nhwc[v0, v1, v2, v3])
                conv2d_nhwc[v0, v1, v2, v3] = conv2d_nhwc_reindex[v0, v1, v2, v3]
    
    # fmt: on
    decision_0 = [
        ("SamplePerfectTile", [4, 1, 1]),
        ("SamplePerfectTile", [2, 1, 4]),
        ("SamplePerfectTile", [2, 4, 1]),
        ("SamplePerfectTile", [2, 2, 2]),
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

def test_conv_1x1():
    # fmt: off
    @T.prim_func
    def main(inputs: T.Buffer[(1, 16, 16, 32), "uint8"], weight: T.Buffer[(3, 3, 32, 32), "uint8"], conv2d_nhwc: T.Buffer[(1, 16, 16, 32), "int32"]):
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # body
        # with T.block("root")
        PadInput = T.alloc_buffer([1, 18, 18, 32], dtype="uint8")
        conv2d_nhwc_reindex = T.alloc_buffer([1, 16, 16, 32], dtype="int32")
        PadInput_reindex = T.alloc_buffer([1, 16, 16, 288], dtype="uint8")
        weight_reindex = T.alloc_buffer([32, 288], dtype="uint8")
        PadInput_reindex_global_vtcm = T.alloc_buffer([1, 16, 16, 288], dtype="uint8", scope="global.vtcm")
        weight_reindex_global_vtcm = T.alloc_buffer([32, 288], dtype="uint8", scope="global.vtcm")
        for i0, i1, i2, i3 in T.grid(1, 18, 18, 32):
            with T.block("PadInput"):
                i0_1, i1_1, i2_1, i3_1 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(inputs[i0_1, i1_1 - 1, i2_1 - 1, i3_1])
                T.writes(PadInput[i0_1, i1_1, i2_1, i3_1])
                PadInput[i0_1, i1_1, i2_1, i3_1] = T.if_then_else(1 <= i1_1 and i1_1 < 17 and 1 <= i2_1 and i2_1 < 17, inputs[i0_1, i1_1 - 1, i2_1 - 1, i3_1], T.uint8(0), dtype="uint8")
        for ax0, ax1, ax2, ax3 in T.grid(1, 16, 16, 288):
            with T.block("PadInput_reindex_reindex"):
                v0, v1, v2, v3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(PadInput[v0, v3 // 96 + v1, v3 % 96 // 32 + v2, v3 % 32])
                T.writes(PadInput_reindex[v0, v1, v2, v3])
                PadInput_reindex[v0, v1, v2, v3] = PadInput[v0, v3 // 96 + v1, v3 % 96 // 32 + v2, v3 % 32]
        for ax0, ax1 in T.grid(32, 288):
            with T.block("weight_reindex_reindex"):
                v0, v1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(weight[v1 // 96, v1 % 96 // 32, v1 % 32, v0])
                T.writes(weight_reindex[v0, v1])
                weight_reindex[v0, v1] = weight[v1 // 96, v1 % 96 // 32, v1 % 32, v0]
        for ax0_0, ax1_0, ax2_0, ax3_0_0 in T.grid(1, 4, 4, 1):
            for ax0_ax1_ax2_ax3_fused in T.serial(4608):
                with T.block("PadInput_reindex_global.vtcm"):
                    v0 = T.axis.spatial(1, 0)
                    v1 = T.axis.spatial(16, ax1_0 * 4 + ax0_ax1_ax2_ax3_fused // 1152)
                    v2 = T.axis.spatial(16, ax2_0 * 4 + ax0_ax1_ax2_ax3_fused % 1152 // 288)
                    v3 = T.axis.spatial(288, ax0_ax1_ax2_ax3_fused % 288)
                    T.reads(PadInput_reindex[v0, v1, v2, v3])
                    T.writes(PadInput_reindex_global_vtcm[v0, v1, v2, v3])
                    PadInput_reindex_global_vtcm[v0, v1, v2, v3] = PadInput_reindex[v0, v1, v2, v3]
            for ax0_ax1_fused in T.serial(9216):
                with T.block("weight_reindex_global.vtcm"):
                    v0 = T.axis.spatial(32, ax0_ax1_fused // 288)
                    v1 = T.axis.spatial(288, ax0_ax1_fused % 288)
                    T.reads(weight_reindex[v0, v1])
                    T.writes(weight_reindex_global_vtcm[v0, v1])
                    weight_reindex_global_vtcm[v0, v1] = weight_reindex[v0, v1]
            for ax4_0_0, ax0_1, ax1_1, ax2_1, ax3_0_1, ax4_0_1, ax0_2, ax1_2, ax2_2, ax3_0_2 in T.grid(18, 1, 1, 4, 1, 4, 1, 4, 1, 1):
                with T.block("conv2d_nhwc_o"):
                    v0 = T.axis.spatial(1, 0)
                    v1 = T.axis.spatial(16, ax1_0 * 4 + ax1_1 * 4 + ax1_2)
                    v2 = T.axis.spatial(16, ax2_2 + ax2_0 * 4 + ax2_1)
                    v3_o = T.axis.spatial(1, 0)
                    v4_o = T.axis.reduce(72, ax4_0_0 * 4 + ax4_0_1)
                    T.reads(PadInput_reindex_global_vtcm[v0, v1, v2, v4_o * 4 : v4_o * 4 + 4], weight_reindex_global_vtcm[0 : 32, v4_o * 4 : v4_o * 4 + 4])
                    T.writes(conv2d_nhwc_reindex[v0, v1, v2, 0 : 32])
                    T.block_attr({"meta_schedule.auto_tensorize":"dot_32x4_u8u8i32_vtcm_vrmpy"})
                    with T.init():
                        for ax3_1 in T.serial(32):
                            with T.block("conv2d_nhwc_init"):
                                v3_i_init = T.axis.spatial(32, ax3_1)
                                T.reads()
                                T.writes(conv2d_nhwc_reindex[v0, v1, v2, v3_i_init])
                                conv2d_nhwc_reindex[v0, v1, v2, v3_i_init] = 0
                    for ax3_1, ax4_1 in T.grid(32, 4):
                        with T.block("conv2d_nhwc"):
                            v3_i, v4_i = T.axis.remap("SR", [ax3_1, ax4_1])
                            T.reads(conv2d_nhwc_reindex[v0, v1, v2, v3_i], PadInput_reindex_global_vtcm[v0, v1, v2, v4_o * 4 + v4_i], weight_reindex_global_vtcm[v3_i, v4_o * 4 + v4_i])
                            T.writes(conv2d_nhwc_reindex[v0, v1, v2, v3_i])
                            T.block_attr({"meta_schedule.tiling_structure":"SRSRS"})
                            conv2d_nhwc_reindex[v0, v1, v2, v3_i] = conv2d_nhwc_reindex[v0, v1, v2, v3_i] + T.Cast("int32", PadInput_reindex_global_vtcm[v0, v1, v2, v4_o * 4 + v4_i]) * T.Cast("int32", weight_reindex_global_vtcm[v3_i, v4_o * 4 + v4_i])
        for ax0, ax1, ax2, ax3 in T.grid(1, 16, 16, 32):
            with T.block("conv2d_nhwc_reindex"):
                v0, v1, v2, v3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(conv2d_nhwc_reindex[v0, v1, v2, v3])
                T.writes(conv2d_nhwc[v0, v1, v2, v3])
                conv2d_nhwc[v0, v1, v2, v3] = conv2d_nhwc_reindex[v0, v1, v2, v3]
    # fmt: on
    decision_0 = [
        ("SamplePerfectTile", [4, 1, 1]),
        ("SamplePerfectTile", [2, 1, 4]),
        ("SamplePerfectTile", [2, 4, 1]),
        ("SamplePerfectTile", [2, 2, 2]),
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
    tvm.testing.main()
