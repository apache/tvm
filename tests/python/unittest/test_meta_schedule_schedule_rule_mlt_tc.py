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
import tvm
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


def multi_level_tiling_tensor_core(
    *,
    write_reuse_scope="shared",
    in_dtype="float16",
    out_dtype="float32",
    trans_b=False,
    use_software_pipeline=False,
) -> ms.schedule_rule.ScheduleRule:
    assert write_reuse_scope in ["shared", "global"]
    if not isinstance(in_dtype, list):
        in_dtype = [in_dtype]
    if not isinstance(out_dtype, list):
        out_dtype = [out_dtype]
    if not isinstance(trans_b, list):
        trans_b = [trans_b]
    return ms.schedule_rule.MultiLevelTilingTensorCore(
        intrin_groups=[
            get_wmma_intrin_group(write_reuse_scope, _in_dtype, _out_dtype, _trans_b)
            for _in_dtype in in_dtype
            for _out_dtype in out_dtype
            for _trans_b in trans_b
        ],
        structure="SSSRRSRS",
        tile_binds=["blockIdx.y", "blockIdx.x", "threadIdx.y"],
        max_innermost_factor=4,  # 64 // tensor intrin size
        vector_load_lens=[1, 2, 3, 4, 8, 16],
        reuse_read=ms.schedule_rule.ReuseType(
            req="must",
            levels=[4],
            scope="shared",
        ),
        reuse_write=ms.schedule_rule.ReuseType(
            req="must" if write_reuse_scope == "shared" else "no",
            levels=[2],
            scope=write_reuse_scope,
        ),
        use_software_pipeline=use_software_pipeline,
    )


def test_matmul_relu():
    # fmt: off
    @T.prim_func
    def matmul_relu_0(A: T.Buffer[(128, 128), "float16"], B: T.Buffer[(128, 128), "float16"], compute: T.Buffer[(128, 128), "float32"]) -> None:
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        C_reindex_shared = T.alloc_buffer([128, 128], dtype="float32", scope="shared")
        C_reindex_shared_wmma_accumulator = T.alloc_buffer([128, 128], dtype="float32", scope="wmma.accumulator")
        A_reindex_shared = T.alloc_buffer([128, 128], dtype="float16", scope="shared")
        B_reindex_shared = T.alloc_buffer([128, 128], dtype="float16", scope="shared")
        A_reindex_shared_wmma_matrix_a = T.alloc_buffer([128, 128], dtype="float16", scope="wmma.matrix_a")
        B_reindex_shared_wmma_matrix_b = T.alloc_buffer([128, 128], dtype="float16", scope="wmma.matrix_b")
        for ax0_0_0_ax1_0_0_fused in T.thread_binding(8, thread="blockIdx.y"):
            for ax0_0_1_ax1_0_1_fused in T.thread_binding(2, thread="blockIdx.x"):
                for ax0_0_2_ax1_0_2_fused in T.thread_binding(2, thread="threadIdx.y"):
                    for ax2_0_0 in T.serial(1):
                        for ax0_ax1_fused in T.serial(4096):
                            with T.block("A_reindex_shared"):
                                v0 = T.axis.spatial(128, ax0_0_0_ax1_0_0_fused // 2 * 32 + ax0_ax1_fused // 128)
                                v1 = T.axis.spatial(128, ax0_ax1_fused % 128)
                                T.reads(A[v0, v1])
                                T.writes(A_reindex_shared[v0, v1])
                                T.block_attr({"buffer_dim_align":[[0, 0, 32, 8]], "meta_schedule.cooperative_fetch":8})
                                A_reindex_shared[v0, v1] = A[v0, v1]
                        for ax0_ax1_fused in T.serial(4096):
                            with T.block("B_reindex_shared"):
                                v0 = T.axis.spatial(128, ax0_ax1_fused // 32)
                                v1 = T.axis.spatial(128, ax0_0_0_ax1_0_0_fused % 2 * 64 + ax0_0_1_ax1_0_1_fused * 32 + ax0_ax1_fused % 32)
                                T.reads(B[v0, v1])
                                T.writes(B_reindex_shared[v0, v1])
                                T.block_attr({"buffer_dim_align":[[0, 0, 32, 8]], "meta_schedule.cooperative_fetch":1})
                                B_reindex_shared[v0, v1] = B[v0, v1]
                        for ax2_0_1 in T.serial(4):
                            for ax0_0, ax1_0 in T.grid(2, 2):
                                with T.block("A_reindex_shared_wmma.matrix_a_o"):
                                    v0_o = T.axis.spatial(8, ax0_0_0_ax1_0_0_fused // 2 * 2 + ax0_0)
                                    v1_o = T.axis.spatial(8, ax2_0_1 * 2 + ax1_0)
                                    T.reads(A_reindex_shared[v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16])
                                    T.writes(A_reindex_shared_wmma_matrix_a[v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16])
                                    T.block_attr({"meta_schedule.auto_tensorize":"wmma_load_16x16x16_f16_a"})
                                    for ax0_1, ax1_1 in T.grid(16, 16):
                                        with T.block("A_reindex_shared_wmma.matrix_a"):
                                            v0_i, v1_i = T.axis.remap("SS", [ax0_1, ax1_1])
                                            T.reads(A_reindex_shared[v0_o * 16 + v0_i, v1_o * 16 + v1_i])
                                            T.writes(A_reindex_shared_wmma_matrix_a[v0_o * 16 + v0_i, v1_o * 16 + v1_i])
                                            A_reindex_shared_wmma_matrix_a[v0_o * 16 + v0_i, v1_o * 16 + v1_i] = A_reindex_shared[v0_o * 16 + v0_i, v1_o * 16 + v1_i]
                            for ax0_0, ax1_0 in T.grid(2, 1):
                                with T.block("B_reindex_shared_wmma.matrix_b_o"):
                                    v0_o = T.axis.spatial(8, ax2_0_1 * 2 + ax0_0)
                                    v1_o = T.axis.spatial(8, ax0_0_0_ax1_0_0_fused % 2 * 4 + ax0_0_1_ax1_0_1_fused * 2 + ax0_0_2_ax1_0_2_fused)
                                    T.reads(B_reindex_shared[v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16])
                                    T.writes(B_reindex_shared_wmma_matrix_b[v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16])
                                    T.block_attr({"meta_schedule.auto_tensorize":"wmma_load_16x16x16_f16_b"})
                                    for ax0_1, ax1_1 in T.grid(16, 16):
                                        with T.block("B_reindex_shared_wmma.matrix_b"):
                                            v0_i, v1_i = T.axis.remap("SS", [ax0_1, ax1_1])
                                            T.reads(B_reindex_shared[v0_o * 16 + v0_i, v1_o * 16 + v1_i])
                                            T.writes(B_reindex_shared_wmma_matrix_b[v0_o * 16 + v0_i, v1_o * 16 + v1_i])
                                            B_reindex_shared_wmma_matrix_b[v0_o * 16 + v0_i, v1_o * 16 + v1_i] = B_reindex_shared[v0_o * 16 + v0_i, v1_o * 16 + v1_i]
                            for ax0_0_3, ax1_0_3, ax2_0_2, ax0_0_4, ax1_0_4 in T.grid(1, 1, 2, 2, 1):
                                with T.block("C_o"):
                                    v0_o = T.axis.spatial(8, ax0_0_0_ax1_0_0_fused // 2 * 2 + ax0_0_3 * 2 + ax0_0_4)
                                    v1_o = T.axis.spatial(8, ax1_0_4 + ax0_0_0_ax1_0_0_fused % 2 * 4 + ax0_0_1_ax1_0_1_fused * 2 + ax0_0_2_ax1_0_2_fused + ax1_0_3)
                                    v2_o = T.axis.reduce(8, ax2_0_0 * 8 + ax2_0_1 * 2 + ax2_0_2)
                                    T.reads(A_reindex_shared_wmma_matrix_a[v0_o * 16 : v0_o * 16 + 16, v2_o * 16 : v2_o * 16 + 16], B_reindex_shared_wmma_matrix_b[v2_o * 16 : v2_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16])
                                    T.writes(C_reindex_shared_wmma_accumulator[v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16])
                                    T.block_attr({"meta_schedule.auto_tensorize":"wmma_sync_16x16x16_f16f16f32", "meta_schedule.auto_tensorize_init":"wmma_fill_16x16x16_f32", "warp_execution":1})
                                    with T.init():
                                        for ax0_1, ax1_1 in T.grid(16, 16):
                                            with T.block("C_init"):
                                                v0_i_init, v1_i_init = T.axis.remap("SS", [ax0_1, ax1_1])
                                                T.reads()
                                                T.writes(C_reindex_shared_wmma_accumulator[v0_o * 16 + v0_i_init, v1_o * 16 + v1_i_init])
                                                C_reindex_shared_wmma_accumulator[v0_o * 16 + v0_i_init, v1_o * 16 + v1_i_init] = T.float32(0)
                                    for ax0_1, ax1_1, ax2_1 in T.grid(16, 16, 16):
                                        with T.block("C"):
                                            v0_i, v1_i, v2_i = T.axis.remap("SSR", [ax0_1, ax1_1, ax2_1])
                                            T.reads(C_reindex_shared_wmma_accumulator[v0_o * 16 + v0_i, v1_o * 16 + v1_i], A_reindex_shared_wmma_matrix_a[v0_o * 16 + v0_i, v2_o * 16 + v2_i], B_reindex_shared_wmma_matrix_b[v2_o * 16 + v2_i, v1_o * 16 + v1_i])
                                            T.writes(C_reindex_shared_wmma_accumulator[v0_o * 16 + v0_i, v1_o * 16 + v1_i])
                                            T.block_attr({"meta_schedule.tiling_structure":"SSSRRSRS"})
                                            C_reindex_shared_wmma_accumulator[v0_o * 16 + v0_i, v1_o * 16 + v1_i] = C_reindex_shared_wmma_accumulator[v0_o * 16 + v0_i, v1_o * 16 + v1_i] + T.cast(A_reindex_shared_wmma_matrix_a[v0_o * 16 + v0_i, v2_o * 16 + v2_i], "float32") * T.cast(B_reindex_shared_wmma_matrix_b[v2_o * 16 + v2_i, v1_o * 16 + v1_i], "float32")
                    for ax0_0, ax1_0 in T.grid(2, 1):
                        with T.block("C_reindex_shared_wmma.accumulator_o"):
                            v0_o = T.axis.spatial(8, ax0_0_0_ax1_0_0_fused // 2 * 2 + ax0_0)
                            v1_o = T.axis.spatial(8, ax0_0_0_ax1_0_0_fused % 2 * 4 + ax0_0_1_ax1_0_1_fused * 2 + ax0_0_2_ax1_0_2_fused)
                            T.reads(C_reindex_shared_wmma_accumulator[v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16])
                            T.writes(C_reindex_shared[v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16])
                            T.block_attr({"meta_schedule.auto_tensorize":"wmma_store_16x16x16_f32_shared"})
                            for ax0_1, ax1_1 in T.grid(16, 16):
                                with T.block("C_reindex_shared_wmma.accumulator"):
                                    v0_i, v1_i = T.axis.remap("SS", [ax0_1, ax1_1])
                                    T.reads(C_reindex_shared_wmma_accumulator[v0_o * 16 + v0_i, v1_o * 16 + v1_i])
                                    T.writes(C_reindex_shared[v0_o * 16 + v0_i, v1_o * 16 + v1_i])
                                    C_reindex_shared[v0_o * 16 + v0_i, v1_o * 16 + v1_i] = C_reindex_shared_wmma_accumulator[v0_o * 16 + v0_i, v1_o * 16 + v1_i]
                for ax0_ax1_fused in T.serial(1024):
                    with T.block("C_reindex_shared"):
                        v0 = T.axis.spatial(128, ax0_0_0_ax1_0_0_fused // 2 * 32 + ax0_ax1_fused // 32)
                        v1 = T.axis.spatial(128, ax0_0_0_ax1_0_0_fused % 2 * 64 + ax0_0_1_ax1_0_1_fused * 32 + ax0_ax1_fused % 32)
                        T.reads(C_reindex_shared[v0, v1])
                        T.writes(compute[v0, v1])
                        T.block_attr({"meta_schedule.cooperative_fetch":4})
                        compute[v0, v1] = T.max(C_reindex_shared[v0, v1], T.float32(0))

    # fmt: on
    decision_0 = [
        ("SamplePerfectTile", [4, 1, 1, 1, 2]),
        ("SamplePerfectTile", [2, 2, 2, 1, 1]),
        ("SamplePerfectTile", [1, 4, 2]),
        ("SampleCategorical", 3),
        ("SampleCategorical", 3),
        ("SampleCategorical", 0),
    ]

    mod = te.create_prim_func(
        te_workload.matmul_relu(
            n=128,
            m=128,
            k=128,
            in_dtype="float16",
            out_dtype="float32",
        )
    )
    actual = generate_design_space(
        kind="cuda",
        mod=mod,
        target=tvm.target.Target("cuda"),
        types=None,
        sch_rules=[
            multi_level_tiling_tensor_core(),
        ]
        + get_rules(kind="cuda", types=ms.schedule_rule.AutoInline),
    )
    check_sketches(
        mod,
        sketches=actual,
        expected_mods=[matmul_relu_0],
        expected_decisions=[decision_0],
    )


def test_matmul_relu_with_fallback():
    # fmt: off
    @T.prim_func
    def matmul_relu_fallback_0(A: T.Buffer[(128, 128), "float16"], B: T.Buffer[(128, 128), "float16"], compute: T.Buffer[(128, 128), "float32"]) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # body
        # with T.block("root")
        C_reindex_shared = T.alloc_buffer([128, 128], dtype="float32", scope="shared")
        C_reindex_shared_wmma_accumulator = T.alloc_buffer([128, 128], dtype="float32", scope="wmma.accumulator")
        A_reindex_shared = T.alloc_buffer([128, 128], dtype="float16", scope="shared")
        B_reindex_shared = T.alloc_buffer([128, 128], dtype="float16", scope="shared")
        A_reindex_shared_wmma_matrix_a = T.alloc_buffer([128, 128], dtype="float16", scope="wmma.matrix_a")
        B_reindex_shared_wmma_matrix_b = T.alloc_buffer([128, 128], dtype="float16", scope="wmma.matrix_b")
        for ax0_0_0_ax1_0_0_fused in T.thread_binding(2, thread="blockIdx.y"):
            for ax0_0_1_ax1_0_1_fused in T.thread_binding(2, thread="blockIdx.x"):
                for ax0_0_2_ax1_0_2_fused in T.thread_binding(2, thread="threadIdx.y"):
                    for ax2_0_0 in T.serial(2):
                        for ax0_ax1_fused in T.serial(2048):
                            with T.block("A_reindex_shared"):
                                v0 = T.axis.spatial(128, ax0_0_0_ax1_0_0_fused * 64 + ax0_0_1_ax1_0_1_fused * 32 + ax0_ax1_fused // 64)
                                v1 = T.axis.spatial(128, ax2_0_0 * 64 + ax0_ax1_fused % 64)
                                T.reads(A[v0, v1])
                                T.writes(A_reindex_shared[v0, v1])
                                T.block_attr({"buffer_dim_align":[[0, 0, 32, 8]], "meta_schedule.cooperative_fetch":4})
                                A_reindex_shared[v0, v1] = A[v0, v1]
                        for ax0_ax1_fused in T.serial(8192):
                            with T.block("B_reindex_shared"):
                                v0 = T.axis.spatial(128, ax2_0_0 * 64 + ax0_ax1_fused // 128)
                                v1 = T.axis.spatial(128, ax0_ax1_fused % 128)
                                T.reads(B[v0, v1])
                                T.writes(B_reindex_shared[v0, v1])
                                T.block_attr({"buffer_dim_align":[[0, 0, 32, 8]], "meta_schedule.cooperative_fetch":2})
                                B_reindex_shared[v0, v1] = B[v0, v1]
                        for ax2_0_1 in T.serial(1):
                            for ax0_0, ax1_0 in T.grid(2, 4):
                                with T.block("A_reindex_shared_wmma.matrix_a_o"):
                                    v0_o = T.axis.spatial(8, ax0_0_0_ax1_0_0_fused * 4 + ax0_0_1_ax1_0_1_fused * 2 + ax0_0)
                                    v1_o = T.axis.spatial(8, ax2_0_0 * 4 + ax1_0)
                                    T.reads(A_reindex_shared[v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16])
                                    T.writes(A_reindex_shared_wmma_matrix_a[v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16])
                                    T.block_attr({"meta_schedule.auto_tensorize":"wmma_load_16x16x16_f16_a"})
                                    for ax0_1, ax1_1 in T.grid(16, 16):
                                        with T.block("A_reindex_shared_wmma.matrix_a"):
                                            v0_i, v1_i = T.axis.remap("SS", [ax0_1, ax1_1])
                                            T.reads(A_reindex_shared[v0_o * 16 + v0_i, v1_o * 16 + v1_i])
                                            T.writes(A_reindex_shared_wmma_matrix_a[v0_o * 16 + v0_i, v1_o * 16 + v1_i])
                                            A_reindex_shared_wmma_matrix_a[v0_o * 16 + v0_i, v1_o * 16 + v1_i] = A_reindex_shared[v0_o * 16 + v0_i, v1_o * 16 + v1_i]
                            for ax0_0, ax1_0 in T.grid(4, 4):
                                with T.block("B_reindex_shared_wmma.matrix_b_o"):
                                    v0_o = T.axis.spatial(8, ax2_0_0 * 4 + ax0_0)
                                    v1_o = T.axis.spatial(8, ax0_0_2_ax1_0_2_fused * 4 + ax1_0)
                                    T.reads(B_reindex_shared[v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16])
                                    T.writes(B_reindex_shared_wmma_matrix_b[v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16])
                                    T.block_attr({"meta_schedule.auto_tensorize":"wmma_load_16x16x16_f16_b"})
                                    for ax0_1, ax1_1 in T.grid(16, 16):
                                        with T.block("B_reindex_shared_wmma.matrix_b"):
                                            v0_i, v1_i = T.axis.remap("SS", [ax0_1, ax1_1])
                                            T.reads(B_reindex_shared[v0_o * 16 + v0_i, v1_o * 16 + v1_i])
                                            T.writes(B_reindex_shared_wmma_matrix_b[v0_o * 16 + v0_i, v1_o * 16 + v1_i])
                                            B_reindex_shared_wmma_matrix_b[v0_o * 16 + v0_i, v1_o * 16 + v1_i] = B_reindex_shared[v0_o * 16 + v0_i, v1_o * 16 + v1_i]
                            for ax0_0_3, ax1_0_3, ax2_0_2, ax0_0_4, ax1_0_4 in T.grid(1, 1, 4, 2, 4):
                                with T.block("C_o"):
                                    v0_o = T.axis.spatial(8, ax0_0_0_ax1_0_0_fused * 4 + ax0_0_1_ax1_0_1_fused * 2 + ax0_0_3 * 2 + ax0_0_4)
                                    v1_o = T.axis.spatial(8, ax0_0_2_ax1_0_2_fused * 4 + ax1_0_3 * 4 + ax1_0_4)
                                    v2_o = T.axis.reduce(8, ax2_0_0 * 4 + ax2_0_1 * 4 + ax2_0_2)
                                    T.reads(A_reindex_shared_wmma_matrix_a[v0_o * 16 : v0_o * 16 + 16, v2_o * 16 : v2_o * 16 + 16], B_reindex_shared_wmma_matrix_b[v2_o * 16 : v2_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16])
                                    T.writes(C_reindex_shared_wmma_accumulator[v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16])
                                    T.block_attr({"meta_schedule.auto_tensorize":"wmma_sync_16x16x16_f16f16f32", "meta_schedule.auto_tensorize_init":"wmma_fill_16x16x16_f32", "warp_execution":1})
                                    with T.init():
                                        for ax0_1, ax1_1 in T.grid(16, 16):
                                            with T.block("C_init"):
                                                v0_i_init, v1_i_init = T.axis.remap("SS", [ax0_1, ax1_1])
                                                T.reads()
                                                T.writes(C_reindex_shared_wmma_accumulator[v0_o * 16 + v0_i_init, v1_o * 16 + v1_i_init])
                                                C_reindex_shared_wmma_accumulator[v0_o * 16 + v0_i_init, v1_o * 16 + v1_i_init] = T.float32(0)
                                    for ax0_1, ax1_1, ax2_1 in T.grid(16, 16, 16):
                                        with T.block("C"):
                                            v0_i, v1_i, v2_i = T.axis.remap("SSR", [ax0_1, ax1_1, ax2_1])
                                            T.reads(C_reindex_shared_wmma_accumulator[v0_o * 16 + v0_i, v1_o * 16 + v1_i], A_reindex_shared_wmma_matrix_a[v0_o * 16 + v0_i, v2_o * 16 + v2_i], B_reindex_shared_wmma_matrix_b[v2_o * 16 + v2_i, v1_o * 16 + v1_i])
                                            T.writes(C_reindex_shared_wmma_accumulator[v0_o * 16 + v0_i, v1_o * 16 + v1_i])
                                            T.block_attr({"meta_schedule.tiling_structure":"SSSRRSRS"})
                                            C_reindex_shared_wmma_accumulator[v0_o * 16 + v0_i, v1_o * 16 + v1_i] = C_reindex_shared_wmma_accumulator[v0_o * 16 + v0_i, v1_o * 16 + v1_i] + T.cast(A_reindex_shared_wmma_matrix_a[v0_o * 16 + v0_i, v2_o * 16 + v2_i], "float32") * T.cast(B_reindex_shared_wmma_matrix_b[v2_o * 16 + v2_i, v1_o * 16 + v1_i], "float32")
                    for ax0_0, ax1_0 in T.grid(2, 4):
                        with T.block("C_reindex_shared_wmma.accumulator_o"):
                            v0_o = T.axis.spatial(8, ax0_0_0_ax1_0_0_fused * 4 + ax0_0_1_ax1_0_1_fused * 2 + ax0_0)
                            v1_o = T.axis.spatial(8, ax0_0_2_ax1_0_2_fused * 4 + ax1_0)
                            T.reads(C_reindex_shared_wmma_accumulator[v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16])
                            T.writes(C_reindex_shared[v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16])
                            T.block_attr({"meta_schedule.auto_tensorize":"wmma_store_16x16x16_f32_shared"})
                            for ax0_1, ax1_1 in T.grid(16, 16):
                                with T.block("C_reindex_shared_wmma.accumulator"):
                                    v0_i, v1_i = T.axis.remap("SS", [ax0_1, ax1_1])
                                    T.reads(C_reindex_shared_wmma_accumulator[v0_o * 16 + v0_i, v1_o * 16 + v1_i])
                                    T.writes(C_reindex_shared[v0_o * 16 + v0_i, v1_o * 16 + v1_i])
                                    C_reindex_shared[v0_o * 16 + v0_i, v1_o * 16 + v1_i] = C_reindex_shared_wmma_accumulator[v0_o * 16 + v0_i, v1_o * 16 + v1_i]
                for ax0_ax1_fused in T.serial(4096):
                    with T.block("C_reindex_shared"):
                        v0 = T.axis.spatial(128, ax0_0_0_ax1_0_0_fused * 64 + ax0_0_1_ax1_0_1_fused * 32 + ax0_ax1_fused // 128)
                        v1 = T.axis.spatial(128, ax0_ax1_fused % 128)
                        T.reads(C_reindex_shared[v0, v1])
                        T.writes(compute[v0, v1])
                        T.block_attr({"meta_schedule.cooperative_fetch":4})
                        compute[v0, v1] = T.max(C_reindex_shared[v0, v1], T.float32(0))
    # fmt: on
    decision_0 = [
        ("SamplePerfectTile", [2, 2, 1, 1, 2]),
        ("SamplePerfectTile", [1, 1, 2, 1, 4]),
        ("SamplePerfectTile", [2, 1, 4]),
        ("SampleCategorical", 3),
        ("SampleCategorical", 2),
        ("SampleCategorical", 1),
    ]

    mod = te.create_prim_func(
        te_workload.matmul_relu(
            n=128,
            m=128,
            k=128,
            in_dtype="float16",
            out_dtype="float32",
        )
    )
    actual = generate_design_space(
        kind="cuda",
        mod=mod,
        target=tvm.target.Target("cuda"),
        types=None,
        sch_rules=[
            multi_level_tiling_tensor_core(),
        ]
        + get_rules(
            "cuda",
            (
                ms.schedule_rule.MultiLevelTiling,
                ms.schedule_rule.AutoInline,
            ),
        ),
    )
    check_sketches(
        mod,
        sketches=actual,
        expected_mods=[matmul_relu_fallback_0],
        expected_decisions=[decision_0],
    )


def test_conv2d():
    # fmt: off
    @T.prim_func
    def conv2d_0(inputs: T.Buffer[(1, 16, 16, 32), "float16"], weight: T.Buffer[(3, 3, 32, 32), "float16"], conv2d_nhwc: T.Buffer[(1, 16, 16, 32), "float32"]) -> None:
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        PadInput = T.alloc_buffer([1, 18, 18, 32], dtype="float16")
        conv2d_nhwc_reindex_shared = T.alloc_buffer([256, 32], dtype="float32", scope="shared")
        conv2d_nhwc_reindex_shared_wmma_accumulator = T.alloc_buffer([256, 32], dtype="float32", scope="wmma.accumulator")
        PadInput_reindex_shared = T.alloc_buffer([256, 288], dtype="float16", scope="shared")
        weight_reindex_shared = T.alloc_buffer([288, 32], dtype="float16", scope="shared")
        PadInput_reindex_shared_wmma_matrix_a = T.alloc_buffer([256, 288], dtype="float16", scope="wmma.matrix_a")
        weight_reindex_shared_wmma_matrix_b = T.alloc_buffer([288, 32], dtype="float16", scope="wmma.matrix_b")
        for i0, i1, i2, i3 in T.grid(1, 18, 18, 32):
            with T.block("PadInput"):
                i0_1, i1_1, i2_1, i3_1 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(inputs[i0_1, i1_1 - 1, i2_1 - 1, i3_1])
                T.writes(PadInput[i0_1, i1_1, i2_1, i3_1])
                PadInput[i0_1, i1_1, i2_1, i3_1] = T.if_then_else(1 <= i1_1 and i1_1 < 17 and 1 <= i2_1 and i2_1 < 17, inputs[i0_1, i1_1 - 1, i2_1 - 1, i3_1], T.float16(0), dtype="float16")
        for ax0_0_0_ax1_0_0_fused in T.thread_binding(2, thread="blockIdx.y"):
            for ax0_0_1_ax1_0_1_fused in T.thread_binding(16, thread="blockIdx.x"):
                for ax0_0_2_ax1_0_2_fused in T.thread_binding(1, thread="threadIdx.y"):
                    for ax2_0_0 in T.serial(1):
                        for ax0_ax1_fused in T.serial(4608):
                            with T.block("PadInput_reindex_shared"):
                                v0 = T.axis.spatial(256, ax0_0_1_ax1_0_1_fused * 16 + ax0_ax1_fused // 288)
                                v1 = T.axis.spatial(288, ax0_ax1_fused % 288)
                                T.reads(PadInput[v0 // 256, v1 // 96 + v0 // 16, v1 % 96 // 32 + v0 % 16, v1 % 32])
                                T.writes(PadInput_reindex_shared[v0, v1])
                                T.block_attr({"buffer_dim_align":[[0, 0, 32, 8]], "meta_schedule.cooperative_fetch":2})
                                PadInput_reindex_shared[v0, v1] = PadInput[v0 // 256, v1 // 96 + v0 // 16, v1 % 96 // 32 + v0 % 16, v1 % 32]
                        for ax0_ax1_fused in T.serial(4608):
                            with T.block("weight_reindex_shared"):
                                v0 = T.axis.spatial(288, ax0_ax1_fused // 16)
                                v1 = T.axis.spatial(32, ax0_0_0_ax1_0_0_fused * 16 + ax0_ax1_fused % 16)
                                T.reads(weight[v0 // 96, v0 % 96 // 32, v0 % 32, v1])
                                T.writes(weight_reindex_shared[v0, v1])
                                T.block_attr({"buffer_dim_align":[[0, 0, 32, 8]], "meta_schedule.cooperative_fetch":8})
                                weight_reindex_shared[v0, v1] = weight[v0 // 96, v0 % 96 // 32, v0 % 32, v1]
                        for ax2_0_1 in T.serial(18):
                            for ax0_0, ax1_0 in T.grid(1, 1):
                                with T.block("PadInput_reindex_shared_wmma.matrix_a_o"):
                                    v0_o, v1_o = T.axis.remap("SS", [ax0_0_1_ax1_0_1_fused, ax2_0_1])
                                    T.reads(PadInput_reindex_shared[v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16])
                                    T.writes(PadInput_reindex_shared_wmma_matrix_a[v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16])
                                    T.block_attr({"meta_schedule.auto_tensorize":"wmma_load_16x16x16_f16_a"})
                                    for ax0_1, ax1_1 in T.grid(16, 16):
                                        with T.block("PadInput_reindex_shared_wmma.matrix_a"):
                                            v0_i, v1_i = T.axis.remap("SS", [ax0_1, ax1_1])
                                            T.reads(PadInput_reindex_shared[v0_o * 16 + v0_i, v1_o * 16 + v1_i])
                                            T.writes(PadInput_reindex_shared_wmma_matrix_a[v0_o * 16 + v0_i, v1_o * 16 + v1_i])
                                            PadInput_reindex_shared_wmma_matrix_a[v0_o * 16 + v0_i, v1_o * 16 + v1_i] = PadInput_reindex_shared[v0_o * 16 + v0_i, v1_o * 16 + v1_i]
                            for ax0_0, ax1_0 in T.grid(1, 1):
                                with T.block("weight_reindex_shared_wmma.matrix_b_o"):
                                    v0_o, v1_o = T.axis.remap("SS", [ax2_0_1, ax0_0_0_ax1_0_0_fused])
                                    T.reads(weight_reindex_shared[v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16])
                                    T.writes(weight_reindex_shared_wmma_matrix_b[v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16])
                                    T.block_attr({"meta_schedule.auto_tensorize":"wmma_load_16x16x16_f16_b"})
                                    for ax0_1, ax1_1 in T.grid(16, 16):
                                        with T.block("weight_reindex_shared_wmma.matrix_b"):
                                            v0_i, v1_i = T.axis.remap("SS", [ax0_1, ax1_1])
                                            T.reads(weight_reindex_shared[v0_o * 16 + v0_i, v1_o * 16 + v1_i])
                                            T.writes(weight_reindex_shared_wmma_matrix_b[v0_o * 16 + v0_i, v1_o * 16 + v1_i])
                                            weight_reindex_shared_wmma_matrix_b[v0_o * 16 + v0_i, v1_o * 16 + v1_i] = weight_reindex_shared[v0_o * 16 + v0_i, v1_o * 16 + v1_i]
                            for ax0_0_3, ax1_0_3, ax2_0_2, ax0_0_4, ax1_0_4 in T.grid(1, 1, 1, 1, 1):
                                with T.block("conv2d_nhwc_o"):
                                    v0_o = T.axis.spatial(16, ax0_0_4 + ax0_0_1_ax1_0_1_fused + ax0_0_3)
                                    v1_o = T.axis.spatial(2, ax0_0_0_ax1_0_0_fused + ax1_0_3 + ax1_0_4)
                                    v2_o = T.axis.reduce(18, ax2_0_0 * 18 + ax2_0_1 + ax2_0_2)
                                    T.reads(PadInput_reindex_shared_wmma_matrix_a[v0_o * 16 : v0_o * 16 + 16, v2_o * 16 : v2_o * 16 + 16], weight_reindex_shared_wmma_matrix_b[v2_o * 16 : v2_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16])
                                    T.writes(conv2d_nhwc_reindex_shared_wmma_accumulator[v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16])
                                    T.block_attr({"meta_schedule.auto_tensorize":"wmma_sync_16x16x16_f16f16f32", "meta_schedule.auto_tensorize_init":"wmma_fill_16x16x16_f32", "warp_execution":1})
                                    with T.init():
                                        for ax0_1, ax1_1 in T.grid(16, 16):
                                            with T.block("conv2d_nhwc_init"):
                                                v0_i_init, v1_i_init = T.axis.remap("SS", [ax0_1, ax1_1])
                                                T.reads()
                                                T.writes(conv2d_nhwc_reindex_shared_wmma_accumulator[v0_o * 16 + v0_i_init, v1_o * 16 + v1_i_init])
                                                conv2d_nhwc_reindex_shared_wmma_accumulator[v0_o * 16 + v0_i_init, v1_o * 16 + v1_i_init] = T.float32(0)
                                    for ax0_1, ax1_1, ax2_1 in T.grid(16, 16, 16):
                                        with T.block("conv2d_nhwc"):
                                            v0_i, v1_i, v2_i = T.axis.remap("SSR", [ax0_1, ax1_1, ax2_1])
                                            T.reads(conv2d_nhwc_reindex_shared_wmma_accumulator[v0_o * 16 + v0_i, v1_o * 16 + v1_i], PadInput_reindex_shared_wmma_matrix_a[v0_o * 16 + v0_i, v2_o * 16 + v2_i], weight_reindex_shared_wmma_matrix_b[v2_o * 16 + v2_i, v1_o * 16 + v1_i])
                                            T.writes(conv2d_nhwc_reindex_shared_wmma_accumulator[v0_o * 16 + v0_i, v1_o * 16 + v1_i])
                                            T.block_attr({"meta_schedule.tiling_structure":"SSSRRSRS"})
                                            conv2d_nhwc_reindex_shared_wmma_accumulator[v0_o * 16 + v0_i, v1_o * 16 + v1_i] = conv2d_nhwc_reindex_shared_wmma_accumulator[v0_o * 16 + v0_i, v1_o * 16 + v1_i] + T.cast(PadInput_reindex_shared_wmma_matrix_a[v0_o * 16 + v0_i, v2_o * 16 + v2_i], "float32") * T.cast(weight_reindex_shared_wmma_matrix_b[v2_o * 16 + v2_i, v1_o * 16 + v1_i], "float32")
                    for ax0_0, ax1_0 in T.grid(1, 1):
                        with T.block("conv2d_nhwc_reindex_shared_wmma.accumulator_o"):
                            v0_o, v1_o = T.axis.remap("SS", [ax0_0_1_ax1_0_1_fused, ax0_0_0_ax1_0_0_fused])
                            T.reads(conv2d_nhwc_reindex_shared_wmma_accumulator[v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16])
                            T.writes(conv2d_nhwc_reindex_shared[v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16])
                            T.block_attr({"meta_schedule.auto_tensorize":"wmma_store_16x16x16_f32_shared"})
                            for ax0_1, ax1_1 in T.grid(16, 16):
                                with T.block("conv2d_nhwc_reindex_shared_wmma.accumulator"):
                                    v0_i, v1_i = T.axis.remap("SS", [ax0_1, ax1_1])
                                    T.reads(conv2d_nhwc_reindex_shared_wmma_accumulator[v0_o * 16 + v0_i, v1_o * 16 + v1_i])
                                    T.writes(conv2d_nhwc_reindex_shared[v0_o * 16 + v0_i, v1_o * 16 + v1_i])
                                    conv2d_nhwc_reindex_shared[v0_o * 16 + v0_i, v1_o * 16 + v1_i] = conv2d_nhwc_reindex_shared_wmma_accumulator[v0_o * 16 + v0_i, v1_o * 16 + v1_i]
                for ax0_ax1_fused in T.serial(256):
                    with T.block("conv2d_nhwc_reindex_shared"):
                        v0 = T.axis.spatial(256, ax0_0_1_ax1_0_1_fused * 16 + ax0_ax1_fused // 16)
                        v1 = T.axis.spatial(32, ax0_0_0_ax1_0_0_fused * 16 +  ax0_ax1_fused % 16)
                        T.reads(conv2d_nhwc_reindex_shared[v0, v1])
                        T.writes(conv2d_nhwc[v0 // 256, v0 // 16, v0 % 16, v1])
                        T.block_attr({"meta_schedule.cooperative_fetch":3})
                        conv2d_nhwc[v0 // 256, v0 // 16, v0 % 16, v1] = conv2d_nhwc_reindex_shared[v0, v1]
    # fmt: on
    decision_0 = [
        ("SamplePerfectTile", [1, 16, 1, 1, 1]),
        ("SamplePerfectTile", [2, 1, 1, 1, 1]),
        ("SamplePerfectTile", [1, 18, 1]),
        ("SampleCategorical", 2),
        ("SampleCategorical", 1),
        ("SampleCategorical", 3),
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
            in_dtype="float16",
            out_dtype="float32",
        )
    )
    actual = generate_design_space(
        kind="cuda",
        mod=mod,
        target=tvm.target.Target("cuda"),
        types=None,
        sch_rules=[
            multi_level_tiling_tensor_core(),
        ],
    )
    check_sketches(
        mod,
        sketches=actual,
        expected_mods=[conv2d_0],
        expected_decisions=[decision_0],
    )

    # Test adding inapplicable tensor intrinsics doesn't change the search space
    # This test case uses the same workload, decision and the expected sketch as above
    actual = generate_design_space(
        kind="cuda",
        mod=mod,
        target=tvm.target.Target("cuda"),
        types=None,
        sch_rules=[
            multi_level_tiling_tensor_core(
                in_dtype="float16",
                out_dtype=["float16", "float32"],
            ),
        ],
    )
    check_sketches(
        mod,
        sketches=actual,
        expected_mods=[conv2d_0],
        expected_decisions=[decision_0],
    )


def test_matmul_relu_pipeline():
    # fmt: off
    @T.prim_func
    def matmul_relu_pipeline_0(A: T.Buffer[(128, 128), "float16"], B: T.Buffer[(128, 128), "float16"], compute: T.Buffer[(128, 128), "float32"]) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # body
        # with T.block("root")
        C = T.alloc_buffer([128, 128], dtype="float32")
        C_reindex_shared = T.alloc_buffer([128, 128], dtype="float32", scope="shared")
        C_reindex_shared_wmma_accumulator = T.alloc_buffer([128, 128], dtype="float32", scope="wmma.accumulator")
        A_reindex_shared = T.alloc_buffer([128, 128], dtype="float16", scope="shared")
        B_reindex_shared = T.alloc_buffer([128, 128], dtype="float16", scope="shared")
        A_reindex_shared_wmma_matrix_a = T.alloc_buffer([128, 128], dtype="float16", scope="wmma.matrix_a")
        B_reindex_shared_wmma_matrix_b = T.alloc_buffer([128, 128], dtype="float16", scope="wmma.matrix_b")
        for ax0_0_0_ax1_0_0_fused in T.thread_binding(1, thread="blockIdx.y"):
            for ax0_0_1_ax1_0_1_fused in T.thread_binding(16, thread="blockIdx.x"):
                for ax0_0_2_ax1_0_2_fused in T.thread_binding(1, thread="threadIdx.y"):
                    for ax2_0_0 in T.serial(4, annotations={"software_pipeline_order":[0, 3, 1, 4, 5, 2, 6], "software_pipeline_stage":[0, 0, 0, 0, 0, 1, 1]}):
                        for ax0_ax1_fused in T.serial(1024):
                            with T.block("A_reindex_shared"):
                                v0 = T.axis.spatial(128, ax0_0_1_ax1_0_1_fused // 4 * 32 + ax0_ax1_fused // 32)
                                v1 = T.axis.spatial(128, ax2_0_0 * 32 + ax0_ax1_fused % 32)
                                T.reads(A[v0, v1])
                                T.writes(A_reindex_shared[v0, v1])
                                T.block_attr({"buffer_dim_align":[[0, 0, 32, 8]], "double_buffer_scope":0, "meta_schedule.cooperative_fetch":4, "tir.manifest_shared_memory_local_stage":1})
                                A_reindex_shared[v0, v1] = A[v0, v1]
                        for ax0_ax1_fused in T.serial(1024):
                            with T.block("B_reindex_shared"):
                                v0 = T.axis.spatial(128, ax2_0_0 * 32 + ax0_ax1_fused // 32)
                                v1 = T.axis.spatial(128, ax0_0_1_ax1_0_1_fused % 4 * 32 + ax0_ax1_fused % 32)
                                T.reads(B[v0, v1])
                                T.writes(B_reindex_shared[v0, v1])
                                T.block_attr({"buffer_dim_align":[[0, 0, 32, 8]], "double_buffer_scope":0, "meta_schedule.cooperative_fetch":2, "tir.manifest_shared_memory_local_stage":1})
                                B_reindex_shared[v0, v1] = B[v0, v1]
                        for ax2_0_1 in T.serial(2, annotations={"software_pipeline_order":[0, 1, 2], "software_pipeline_stage":[0, 0, 1]}):
                            for ax0_0, ax1_0 in T.grid(2, 1):
                                with T.block("A_reindex_shared_wmma.matrix_a_o"):
                                    v0_o = T.axis.spatial(8, ax0_0_1_ax1_0_1_fused // 4 * 2 + ax0_0)
                                    v1_o = T.axis.spatial(8, ax2_0_0 * 2 + ax2_0_1)
                                    T.reads(A_reindex_shared[v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16])
                                    T.writes(A_reindex_shared_wmma_matrix_a[v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16])
                                    T.block_attr({"meta_schedule.auto_tensorize":"wmma_load_16x16x16_f16_a"})
                                    for ax0_1, ax1_1 in T.grid(16, 16):
                                        with T.block("A_reindex_shared_wmma.matrix_a"):
                                            v0_i, v1_i = T.axis.remap("SS", [ax0_1, ax1_1])
                                            T.reads(A_reindex_shared[v0_o * 16 + v0_i, v1_o * 16 + v1_i])
                                            T.writes(A_reindex_shared_wmma_matrix_a[v0_o * 16 + v0_i, v1_o * 16 + v1_i])
                                            A_reindex_shared_wmma_matrix_a[v0_o * 16 + v0_i, v1_o * 16 + v1_i] = A_reindex_shared[v0_o * 16 + v0_i, v1_o * 16 + v1_i]
                            for ax0_0, ax1_0 in T.grid(1, 2):
                                with T.block("B_reindex_shared_wmma.matrix_b_o"):
                                    v0_o = T.axis.spatial(8, ax2_0_0 * 2 + ax2_0_1)
                                    v1_o = T.axis.spatial(8, ax0_0_1_ax1_0_1_fused % 4 * 2 + ax1_0)
                                    T.reads(B_reindex_shared[v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16])
                                    T.writes(B_reindex_shared_wmma_matrix_b[v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16])
                                    T.block_attr({"meta_schedule.auto_tensorize":"wmma_load_16x16x16_f16_b"})
                                    for ax0_1, ax1_1 in T.grid(16, 16):
                                        with T.block("B_reindex_shared_wmma.matrix_b"):
                                            v0_i, v1_i = T.axis.remap("SS", [ax0_1, ax1_1])
                                            T.reads(B_reindex_shared[v0_o * 16 + v0_i, v1_o * 16 + v1_i])
                                            T.writes(B_reindex_shared_wmma_matrix_b[v0_o * 16 + v0_i, v1_o * 16 + v1_i])
                                            B_reindex_shared_wmma_matrix_b[v0_o * 16 + v0_i, v1_o * 16 + v1_i] = B_reindex_shared[v0_o * 16 + v0_i, v1_o * 16 + v1_i]
                            for ax0_0_3, ax1_0_3, ax2_0_2, ax0_0_4, ax1_0_4 in T.grid(1, 1, 1, 2, 2):
                                with T.block("C_o"):
                                    v0_o = T.axis.spatial(8, ax0_0_1_ax1_0_1_fused // 4 * 2 + ax0_0_3 * 2 + ax0_0_4)
                                    v1_o = T.axis.spatial(8, ax0_0_1_ax1_0_1_fused % 4 * 2 + ax1_0_3 * 2 + ax1_0_4)
                                    v2_o = T.axis.reduce(8, ax2_0_0 * 2 + ax2_0_1 + ax2_0_2)
                                    T.reads(A_reindex_shared_wmma_matrix_a[v0_o * 16 : v0_o * 16 + 16, v2_o * 16 : v2_o * 16 + 16], B_reindex_shared_wmma_matrix_b[v2_o * 16 : v2_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16])
                                    T.writes(C_reindex_shared_wmma_accumulator[v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16])
                                    T.block_attr({"meta_schedule.auto_tensorize":"wmma_sync_16x16x16_f16f16f32", "meta_schedule.auto_tensorize_init":"wmma_fill_16x16x16_f32", "warp_execution":1})
                                    with T.init():
                                        for ax0_1, ax1_1 in T.grid(16, 16):
                                            with T.block("C_init"):
                                                v0_i_init, v1_i_init = T.axis.remap("SS", [ax0_1, ax1_1])
                                                T.reads()
                                                T.writes(C_reindex_shared_wmma_accumulator[v0_o * 16 + v0_i_init, v1_o * 16 + v1_i_init])
                                                C_reindex_shared_wmma_accumulator[v0_o * 16 + v0_i_init, v1_o * 16 + v1_i_init] = T.float32(0)
                                    for ax0_1, ax1_1, ax2_1 in T.grid(16, 16, 16):
                                        with T.block("C"):
                                            v0_i, v1_i, v2_i = T.axis.remap("SSR", [ax0_1, ax1_1, ax2_1])
                                            T.reads(C_reindex_shared_wmma_accumulator[v0_o * 16 + v0_i, v1_o * 16 + v1_i], A_reindex_shared_wmma_matrix_a[v0_o * 16 + v0_i, v2_o * 16 + v2_i], B_reindex_shared_wmma_matrix_b[v2_o * 16 + v2_i, v1_o * 16 + v1_i])
                                            T.writes(C_reindex_shared_wmma_accumulator[v0_o * 16 + v0_i, v1_o * 16 + v1_i])
                                            T.block_attr({"meta_schedule.tiling_structure":"SSSRRSRS"})
                                            C_reindex_shared_wmma_accumulator[v0_o * 16 + v0_i, v1_o * 16 + v1_i] = C_reindex_shared_wmma_accumulator[v0_o * 16 + v0_i, v1_o * 16 + v1_i] + T.cast(A_reindex_shared_wmma_matrix_a[v0_o * 16 + v0_i, v2_o * 16 + v2_i], "float32") * T.cast(B_reindex_shared_wmma_matrix_b[v2_o * 16 + v2_i, v1_o * 16 + v1_i], "float32")
                    for ax0_0, ax1_0 in T.grid(2, 2):
                        with T.block("C_reindex_shared_wmma.accumulator_o"):
                            v0_o = T.axis.spatial(8, ax0_0_1_ax1_0_1_fused // 4 * 2 + ax0_0)
                            v1_o = T.axis.spatial(8, ax0_0_1_ax1_0_1_fused % 4 * 2 + ax1_0)
                            T.reads(C_reindex_shared_wmma_accumulator[v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16])
                            T.writes(C_reindex_shared[v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16])
                            T.block_attr({"meta_schedule.auto_tensorize":"wmma_store_16x16x16_f32_shared"})
                            for ax0_1, ax1_1 in T.grid(16, 16):
                                with T.block("C_reindex_shared_wmma.accumulator"):
                                    v0_i, v1_i = T.axis.remap("SS", [ax0_1, ax1_1])
                                    T.reads(C_reindex_shared_wmma_accumulator[v0_o * 16 + v0_i, v1_o * 16 + v1_i])
                                    T.writes(C_reindex_shared[v0_o * 16 + v0_i, v1_o * 16 + v1_i])
                                    C_reindex_shared[v0_o * 16 + v0_i, v1_o * 16 + v1_i] = C_reindex_shared_wmma_accumulator[v0_o * 16 + v0_i, v1_o * 16 + v1_i]
                for ax0_ax1_fused in T.grid(1024):
                    with T.block("C_reindex_shared"):
                        v0 = T.axis.spatial(128, ax0_0_1_ax1_0_1_fused // 4 * 32 + ax0_ax1_fused // 32)
                        v1 = T.axis.spatial(128, ax0_0_1_ax1_0_1_fused % 4 * 32 + ax0_ax1_fused % 32)
                        T.reads(C_reindex_shared[v0, v1])
                        T.writes(C[v0, v1])
                        T.block_attr({"meta_schedule.cooperative_fetch":3})
                        C[v0, v1] = C_reindex_shared[v0, v1]
        for i0, i1 in T.grid(128, 128):
            with T.block("compute"):
                i0_1, i1_1 = T.axis.remap("SS", [i0, i1])
                T.reads(C[i0_1, i1_1])
                T.writes(compute[i0_1, i1_1])
                compute[i0_1, i1_1] = T.max(C[i0_1, i1_1], T.float32(0))
    # fmt: on
    decision_0 = [
        ("SamplePerfectTile", [1, 4, 1, 1, 2]),
        ("SamplePerfectTile", [1, 4, 1, 1, 2]),
        ("SamplePerfectTile", [4, 2, 1]),
        ("SampleCategorical", 2),
        ("SampleCategorical", 2),
        ("SampleCategorical", 1),
    ]
    mod = te.create_prim_func(
        te_workload.matmul_relu(
            n=128,
            m=128,
            k=128,
            in_dtype="float16",
            out_dtype="float32",
        )
    )
    actual = generate_design_space(
        kind="cuda",
        mod=mod,
        target=tvm.target.Target("cuda"),
        types=None,
        sch_rules=[
            multi_level_tiling_tensor_core(
                use_software_pipeline=True,
            ),
        ],
    )
    check_sketches(
        mod,
        sketches=actual,
        expected_mods=[matmul_relu_pipeline_0],
        expected_decisions=[decision_0],
    )


def test_matmul_relu_global():
    # fmt: off
    @T.prim_func
    def matmul_relu_global_0(A: T.Buffer[(128, 128), "float16"], B: T.Buffer[(128, 128), "float16"], compute: T.Buffer[(128, 128), "float32"]) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # body
        # with T.block("root")
        C = T.alloc_buffer([128, 128], dtype="float32")
        C_reindex_wmma_accumulator = T.alloc_buffer([128, 128], dtype="float32", scope="wmma.accumulator")
        A_reindex_shared = T.alloc_buffer([128, 128], dtype="float16", scope="shared")
        B_reindex_shared = T.alloc_buffer([128, 128], dtype="float16", scope="shared")
        A_reindex_shared_wmma_matrix_a = T.alloc_buffer([128, 128], dtype="float16", scope="wmma.matrix_a")
        B_reindex_shared_wmma_matrix_b = T.alloc_buffer([128, 128], dtype="float16", scope="wmma.matrix_b")
        for ax0_0_0_ax1_0_0_fused in T.thread_binding(1, thread="blockIdx.y"):
            for ax0_0_1_ax1_0_1_fused in T.thread_binding(1, thread="blockIdx.x"):
                for ax0_0_2_ax1_0_2_fused in T.thread_binding(16, thread="threadIdx.y"):
                    for ax2_0_0 in T.serial(2):
                        for ax0_ax1_fused in T.serial(8192):
                            with T.block("A_reindex_shared"):
                                v0 = T.axis.spatial(128, ax0_ax1_fused // 64)
                                v1 = T.axis.spatial(128, ax2_0_0 * 64 + ax0_ax1_fused % 64)
                                T.reads(A[v0, v1])
                                T.writes(A_reindex_shared[v0, v1])
                                T.block_attr({"buffer_dim_align":[[0, 0, 32, 8]], "meta_schedule.cooperative_fetch":1})
                                A_reindex_shared[v0, v1] = A[v0, v1]
                        for ax0_ax1_fused in T.serial(8192):
                            with T.block("B_reindex_shared"):
                                v0 = T.axis.spatial(128, ax2_0_0 * 64 + ax0_ax1_fused // 128)
                                v1 = T.axis.spatial(128, ax0_ax1_fused % 128)
                                T.reads(B[v0, v1])
                                T.writes(B_reindex_shared[v0, v1])
                                T.block_attr({"buffer_dim_align":[[0, 0, 32, 8]], "meta_schedule.cooperative_fetch":1})
                                B_reindex_shared[v0, v1] = B[v0, v1]
                        for ax2_0_1 in T.serial(2):
                            for ax0_0, ax1_0 in T.grid(1, 2):
                                with T.block("A_reindex_shared_wmma.matrix_a_o"):
                                    v0_o = T.axis.spatial(8, ax0_0_2_ax1_0_2_fused // 2)
                                    v1_o = T.axis.spatial(8, ax2_0_0 * 4 + ax2_0_1 * 2 + ax1_0)
                                    T.reads(A_reindex_shared[v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16])
                                    T.writes(A_reindex_shared_wmma_matrix_a[v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16])
                                    T.block_attr({"meta_schedule.auto_tensorize":"wmma_load_16x16x16_f16_a"})
                                    for ax0_1, ax1_1 in T.grid(16, 16):
                                        with T.block("A_reindex_shared_wmma.matrix_a"):
                                            v0_i, v1_i = T.axis.remap("SS", [ax0_1, ax1_1])
                                            T.reads(A_reindex_shared[v0_o * 16 + v0_i, v1_o * 16 + v1_i])
                                            T.writes(A_reindex_shared_wmma_matrix_a[v0_o * 16 + v0_i, v1_o * 16 + v1_i])
                                            A_reindex_shared_wmma_matrix_a[v0_o * 16 + v0_i, v1_o * 16 + v1_i] = A_reindex_shared[v0_o * 16 + v0_i, v1_o * 16 + v1_i]
                            for ax0_0, ax1_0 in T.grid(2, 4):
                                with T.block("B_reindex_shared_wmma.matrix_b_o"):
                                    v0_o = T.axis.spatial(8, ax2_0_0 * 4 + ax2_0_1 * 2 + ax0_0)
                                    v1_o = T.axis.spatial(8, ax0_0_2_ax1_0_2_fused % 2 * 4 + ax1_0)
                                    T.reads(B_reindex_shared[v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16])
                                    T.writes(B_reindex_shared_wmma_matrix_b[v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16])
                                    T.block_attr({"meta_schedule.auto_tensorize":"wmma_load_16x16x16_f16_b"})
                                    for ax0_1, ax1_1 in T.grid(16, 16):
                                        with T.block("B_reindex_shared_wmma.matrix_b"):
                                            v0_i, v1_i = T.axis.remap("SS", [ax0_1, ax1_1])
                                            T.reads(B_reindex_shared[v0_o * 16 + v0_i, v1_o * 16 + v1_i])
                                            T.writes(B_reindex_shared_wmma_matrix_b[v0_o * 16 + v0_i, v1_o * 16 + v1_i])
                                            B_reindex_shared_wmma_matrix_b[v0_o * 16 + v0_i, v1_o * 16 + v1_i] = B_reindex_shared[v0_o * 16 + v0_i, v1_o * 16 + v1_i]
                            for ax0_0_3, ax1_0_3, ax2_0_2, ax0_0_4, ax1_0_4 in T.grid(1, 4, 2, 1, 1):
                                with T.block("C_o"):
                                    v0_o = T.axis.spatial(8, ax0_0_2_ax1_0_2_fused // 2 + ax0_0_3 + ax0_0_4)
                                    v1_o = T.axis.spatial(8, ax1_0_4 + ax0_0_2_ax1_0_2_fused % 2 * 4 + ax1_0_3)
                                    v2_o = T.axis.reduce(8, ax2_0_0 * 4 + ax2_0_1 * 2 + ax2_0_2)
                                    T.reads(A_reindex_shared_wmma_matrix_a[v0_o * 16 : v0_o * 16 + 16, v2_o * 16 : v2_o * 16 + 16], B_reindex_shared_wmma_matrix_b[v2_o * 16 : v2_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16])
                                    T.writes(C_reindex_wmma_accumulator[v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16])
                                    T.block_attr({"meta_schedule.auto_tensorize":"wmma_sync_16x16x16_f16f16f32", "meta_schedule.auto_tensorize_init":"wmma_fill_16x16x16_f32", "warp_execution":1})
                                    with T.init():
                                        for ax0_1, ax1_1 in T.grid(16, 16):
                                            with T.block("C_init"):
                                                v0_i_init, v1_i_init = T.axis.remap("SS", [ax0_1, ax1_1])
                                                T.reads()
                                                T.writes(C_reindex_wmma_accumulator[v0_o * 16 + v0_i_init, v1_o * 16 + v1_i_init])
                                                C_reindex_wmma_accumulator[v0_o * 16 + v0_i_init, v1_o * 16 + v1_i_init] = T.float32(0)
                                    for ax0_1, ax1_1, ax2_1 in T.grid(16, 16, 16):
                                        with T.block("C"):
                                            v0_i, v1_i, v2_i = T.axis.remap("SSR", [ax0_1, ax1_1, ax2_1])
                                            T.reads(C_reindex_wmma_accumulator[v0_o * 16 + v0_i, v1_o * 16 + v1_i], A_reindex_shared_wmma_matrix_a[v0_o * 16 + v0_i, v2_o * 16 + v2_i], B_reindex_shared_wmma_matrix_b[v2_o * 16 + v2_i, v1_o * 16 + v1_i])
                                            T.writes(C_reindex_wmma_accumulator[v0_o * 16 + v0_i, v1_o * 16 + v1_i])
                                            T.block_attr({"meta_schedule.tiling_structure":"SSSRRSRS"})
                                            C_reindex_wmma_accumulator[v0_o * 16 + v0_i, v1_o * 16 + v1_i] = C_reindex_wmma_accumulator[v0_o * 16 + v0_i, v1_o * 16 + v1_i] + T.cast(A_reindex_shared_wmma_matrix_a[v0_o * 16 + v0_i, v2_o * 16 + v2_i], "float32") * T.cast(B_reindex_shared_wmma_matrix_b[v2_o * 16 + v2_i, v1_o * 16 + v1_i], "float32")
                    for ax0_0, ax1_0 in T.grid(1, 4):
                        with T.block("C_reindex_wmma.accumulator_o"):
                            v0_o = T.axis.spatial(8, ax0_0_2_ax1_0_2_fused // 2)
                            v1_o = T.axis.spatial(8, ax0_0_2_ax1_0_2_fused % 2 * 4 + ax1_0)
                            T.reads(C_reindex_wmma_accumulator[v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16])
                            T.writes(C[v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16])
                            T.block_attr({"meta_schedule.auto_tensorize":"wmma_store_16x16x16_f32_global"})
                            for ax0_1, ax1_1 in T.grid(16, 16):
                                with T.block("C_reindex_wmma.accumulator"):
                                    v0_i, v1_i = T.axis.remap("SS", [ax0_1, ax1_1])
                                    T.reads(C_reindex_wmma_accumulator[v0_o * 16 + v0_i, v1_o * 16 + v1_i])
                                    T.writes(C[v0_o * 16 + v0_i, v1_o * 16 + v1_i])
                                    C[v0_o * 16 + v0_i, v1_o * 16 + v1_i] = C_reindex_wmma_accumulator[v0_o * 16 + v0_i, v1_o * 16 + v1_i]
        for i0, i1 in T.grid(128, 128):
            with T.block("compute"):
                i0_1, i1_1 = T.axis.remap("SS", [i0, i1])
                T.reads(C[i0_1, i1_1])
                T.writes(compute[i0_1, i1_1])
                compute[i0_1, i1_1] = T.max(C[i0_1, i1_1], T.float32(0))
    # fmt: on
    decision_0 = [
        ("SamplePerfectTile", [1, 1, 8, 1, 1]),
        ("SamplePerfectTile", [1, 1, 2, 4, 1]),
        ("SamplePerfectTile", [2, 2, 2]),
        ("SampleCategorical", 0),
        ("SampleCategorical", 0),
    ]
    mod = te.create_prim_func(
        te_workload.matmul_relu(
            n=128,
            m=128,
            k=128,
            in_dtype="float16",
            out_dtype="float32",
        )
    )
    actual = generate_design_space(
        kind="cuda",
        mod=mod,
        target=tvm.target.Target("cuda"),
        types=None,
        sch_rules=[multi_level_tiling_tensor_core(write_reuse_scope="global")]
        + get_rules("cuda", ms.schedule_rule.AutoInline),
    )
    check_sketches(
        mod,
        sketches=actual,
        expected_mods=[matmul_relu_global_0],
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
        kind="cuda",
        mod=mod,
        target=tvm.target.Target("cuda"),
        types=None,
        sch_rules=[multi_level_tiling_tensor_core(write_reuse_scope="global")]
        + get_rules("cuda", ms.schedule_rule.AutoInline),
    )
    tvm.ir.assert_structural_equal(mod, sch.mod["main"])


def test_padded_matmul_relu():
    # fmt: off
    @T.prim_func
    def padded_matmul_relu_0(A: T.Buffer[(127, 127), "float16"], B: T.Buffer[(127, 127), "float16"], compute: T.Buffer[(127, 127), "float32"]) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # body
        # with T.block("root")
        C_reindex_shared = T.alloc_buffer([128, 128], dtype="float32", scope="shared")
        C_reindex_shared_wmma_accumulator = T.alloc_buffer([128, 128], dtype="float32", scope="wmma.accumulator")
        A_reindex_shared = T.alloc_buffer([128, 128], dtype="float16", scope="shared")
        B_reindex_shared = T.alloc_buffer([128, 128], dtype="float16", scope="shared")
        A_reindex_shared_wmma_matrix_a = T.alloc_buffer([128, 128], dtype="float16", scope="wmma.matrix_a")
        B_reindex_shared_wmma_matrix_b = T.alloc_buffer([128, 128], dtype="float16", scope="wmma.matrix_b")
        for ax0_0_0_ax1_0_0_fused in T.thread_binding(8, thread="blockIdx.y"):
            for ax0_0_1_ax1_0_1_fused in T.thread_binding(2, thread="blockIdx.x"):
                for ax0_0_2_ax1_0_2_fused in T.thread_binding(2, thread="threadIdx.y"):
                    for ax2_0_0 in T.serial(1):
                        for ax0_ax1_fused in T.serial(4096):
                            with T.block("A_reindex_shared"):
                                v0 = T.axis.spatial(128, ax0_0_0_ax1_0_0_fused // 2 * 32 + ax0_ax1_fused // 128)
                                v1 = T.axis.spatial(128, ax0_ax1_fused % 128)
                                T.reads(A[v0, v1])
                                T.writes(A_reindex_shared[v0, v1])
                                T.block_attr({"buffer_dim_align":[[0, 0, 32, 8]], "meta_schedule.cooperative_fetch":8})
                                A_reindex_shared[v0, v1] = T.if_then_else(v0 < 127 and v1 < 127, A[v0, v1], T.float16(0), dtype="float16")
                        for ax0_ax1_fused in T.serial(4096):
                            with T.block("B_reindex_shared"):
                                v0 = T.axis.spatial(128, ax0_ax1_fused // 32)
                                v1 = T.axis.spatial(128, ax0_0_0_ax1_0_0_fused % 2 * 64 + ax0_0_1_ax1_0_1_fused * 32 + ax0_ax1_fused % 32)
                                T.reads(B[v0, v1])
                                T.writes(B_reindex_shared[v0, v1])
                                T.block_attr({"buffer_dim_align":[[0, 0, 32, 8]], "meta_schedule.cooperative_fetch":1})
                                B_reindex_shared[v0, v1] = T.if_then_else(v0 < 127 and v1 < 127, B[v0, v1], T.float16(0), dtype="float16")
                        for ax2_0_1 in T.serial(4):
                            for ax0_0, ax1_0 in T.grid(2, 2):
                                with T.block("A_reindex_shared_wmma.matrix_a_o"):
                                    v0_o = T.axis.spatial(8, ax0_0_0_ax1_0_0_fused // 2 * 2 + ax0_0)
                                    v1_o = T.axis.spatial(8, ax2_0_1 * 2 + ax1_0)
                                    T.reads(A_reindex_shared[v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16])
                                    T.writes(A_reindex_shared_wmma_matrix_a[v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16])
                                    T.block_attr({"meta_schedule.auto_tensorize":"wmma_load_16x16x16_f16_a"})
                                    for ax0_1, ax1_1 in T.grid(16, 16):
                                        with T.block("A_reindex_shared_wmma.matrix_a"):
                                            v0_i, v1_i = T.axis.remap("SS", [ax0_1, ax1_1])
                                            T.reads(A_reindex_shared[v0_o * 16 + v0_i, v1_o * 16 + v1_i])
                                            T.writes(A_reindex_shared_wmma_matrix_a[v0_o * 16 + v0_i, v1_o * 16 + v1_i])
                                            A_reindex_shared_wmma_matrix_a[v0_o * 16 + v0_i, v1_o * 16 + v1_i] = A_reindex_shared[v0_o * 16 + v0_i, v1_o * 16 + v1_i]
                            for ax0_0, ax1_0 in T.grid(2, 1):
                                with T.block("B_reindex_shared_wmma.matrix_b_o"):
                                    v0_o = T.axis.spatial(8, ax2_0_1 * 2 + ax0_0)
                                    v1_o = T.axis.spatial(8, ax0_0_0_ax1_0_0_fused % 2 * 4 + ax0_0_1_ax1_0_1_fused * 2 + ax0_0_2_ax1_0_2_fused)
                                    T.reads(B_reindex_shared[v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16])
                                    T.writes(B_reindex_shared_wmma_matrix_b[v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16])
                                    T.block_attr({"meta_schedule.auto_tensorize":"wmma_load_16x16x16_f16_b"})
                                    for ax0_1, ax1_1 in T.grid(16, 16):
                                        with T.block("B_reindex_shared_wmma.matrix_b"):
                                            v0_i, v1_i = T.axis.remap("SS", [ax0_1, ax1_1])
                                            T.reads(B_reindex_shared[v0_o * 16 + v0_i, v1_o * 16 + v1_i])
                                            T.writes(B_reindex_shared_wmma_matrix_b[v0_o * 16 + v0_i, v1_o * 16 + v1_i])
                                            B_reindex_shared_wmma_matrix_b[v0_o * 16 + v0_i, v1_o * 16 + v1_i] = B_reindex_shared[v0_o * 16 + v0_i, v1_o * 16 + v1_i]
                            for ax0_0_3, ax1_0_3, ax2_0_2, ax0_0_4, ax1_0_4 in T.grid(1, 1, 2, 2, 1):
                                with T.block("C_o"):
                                    v0_o = T.axis.spatial(8, ax0_0_0_ax1_0_0_fused // 2 * 2 + ax0_0_3 * 2 + ax0_0_4)
                                    v1_o = T.axis.spatial(8, ax1_0_4 + ax0_0_0_ax1_0_0_fused % 2 * 4 + ax0_0_1_ax1_0_1_fused * 2 + ax0_0_2_ax1_0_2_fused + ax1_0_3)
                                    v2_o = T.axis.reduce(8, ax2_0_0 * 8 + ax2_0_1 * 2 + ax2_0_2)
                                    T.reads(A_reindex_shared_wmma_matrix_a[v0_o * 16 : v0_o * 16 + 16, v2_o * 16 : v2_o * 16 + 16], B_reindex_shared_wmma_matrix_b[v2_o * 16 : v2_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16])
                                    T.writes(C_reindex_shared_wmma_accumulator[v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16])
                                    T.block_attr({"meta_schedule.auto_tensorize":"wmma_sync_16x16x16_f16f16f32", "meta_schedule.auto_tensorize_init":"wmma_fill_16x16x16_f32", "warp_execution":1})
                                    with T.init():
                                        for ax0_1, ax1_1 in T.grid(16, 16):
                                            with T.block("C_init"):
                                                v0_i_init, v1_i_init = T.axis.remap("SS", [ax0_1, ax1_1])
                                                T.reads()
                                                T.writes(C_reindex_shared_wmma_accumulator[v0_o * 16 + v0_i_init, v1_o * 16 + v1_i_init])
                                                C_reindex_shared_wmma_accumulator[v0_o * 16 + v0_i_init, v1_o * 16 + v1_i_init] = T.float32(0)
                                    for ax0_1, ax1_1, ax2_1 in T.grid(16, 16, 16):
                                        with T.block("C"):
                                            v0_i, v1_i, v2_i = T.axis.remap("SSR", [ax0_1, ax1_1, ax2_1])
                                            T.reads(C_reindex_shared_wmma_accumulator[v0_o * 16 + v0_i, v1_o * 16 + v1_i], A_reindex_shared_wmma_matrix_a[v0_o * 16 + v0_i, v2_o * 16 + v2_i], B_reindex_shared_wmma_matrix_b[v2_o * 16 + v2_i, v1_o * 16 + v1_i])
                                            T.writes(C_reindex_shared_wmma_accumulator[v0_o * 16 + v0_i, v1_o * 16 + v1_i])
                                            T.block_attr({"meta_schedule.tiling_structure":"SSSRRSRS"})
                                            C_reindex_shared_wmma_accumulator[v0_o * 16 + v0_i, v1_o * 16 + v1_i] = C_reindex_shared_wmma_accumulator[v0_o * 16 + v0_i, v1_o * 16 + v1_i] + T.cast(A_reindex_shared_wmma_matrix_a[v0_o * 16 + v0_i, v2_o * 16 + v2_i], "float32") * T.cast(B_reindex_shared_wmma_matrix_b[v2_o * 16 + v2_i, v1_o * 16 + v1_i], "float32")
                    for ax0_0, ax1_0 in T.grid(2, 1):
                        with T.block("C_reindex_shared_wmma.accumulator_o"):
                            v0_o = T.axis.spatial(8, ax0_0_0_ax1_0_0_fused // 2 * 2 + ax0_0)
                            v1_o = T.axis.spatial(8, ax0_0_0_ax1_0_0_fused % 2 * 4 + ax0_0_1_ax1_0_1_fused * 2 + ax0_0_2_ax1_0_2_fused)
                            T.reads(C_reindex_shared_wmma_accumulator[v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16])
                            T.writes(C_reindex_shared[v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16])
                            T.block_attr({"meta_schedule.auto_tensorize":"wmma_store_16x16x16_f32_shared"})
                            for ax0_1, ax1_1 in T.grid(16, 16):
                                with T.block("C_reindex_shared_wmma.accumulator"):
                                    v0_i, v1_i = T.axis.remap("SS", [ax0_1, ax1_1])
                                    T.reads(C_reindex_shared_wmma_accumulator[v0_o * 16 + v0_i, v1_o * 16 + v1_i])
                                    T.writes(C_reindex_shared[v0_o * 16 + v0_i, v1_o * 16 + v1_i])
                                    C_reindex_shared[v0_o * 16 + v0_i, v1_o * 16 + v1_i] = C_reindex_shared_wmma_accumulator[v0_o * 16 + v0_i, v1_o * 16 + v1_i]
                for ax0_ax1_fused in T.serial(1024):
                    with T.block("C_reindex_shared"):
                        T.where(ax0_0_0_ax1_0_0_fused // 2 * 32 + ax0_ax1_fused // 32 < 127 and ax0_0_0_ax1_0_0_fused % 2 * 64 + ax0_0_1_ax1_0_1_fused * 32 + ax0_ax1_fused % 32 < 127)
                        v0 = T.axis.spatial(128, ax0_0_0_ax1_0_0_fused // 2 * 32 + ax0_ax1_fused // 32)
                        v1 = T.axis.spatial(128, ax0_0_0_ax1_0_0_fused % 2 * 64 + ax0_0_1_ax1_0_1_fused * 32 + ax0_ax1_fused % 32)
                        T.reads(C_reindex_shared[v0, v1])
                        T.writes(compute[v0, v1])
                        T.block_attr({"meta_schedule.cooperative_fetch":4})
                        compute[v0, v1] = T.max(C_reindex_shared[v0, v1], T.float32(0))
    # fmt: on

    decision_0 = [
        ("SamplePerfectTile", [4, 1, 1, 1, 2]),
        ("SamplePerfectTile", [2, 2, 2, 1, 1]),
        ("SamplePerfectTile", [1, 4, 2]),
        ("SampleCategorical", 3),
        ("SampleCategorical", 3),
        ("SampleCategorical", 0),
    ]

    mod = te.create_prim_func(
        te_workload.matmul_relu(
            n=127,
            m=127,
            k=127,
            in_dtype="float16",
            out_dtype="float32",
        )
    )
    actual = generate_design_space(
        kind="cuda",
        mod=mod,
        target=tvm.target.Target("cuda"),
        types=None,
        sch_rules=[multi_level_tiling_tensor_core(write_reuse_scope="shared")]
        + get_rules("cuda", ms.schedule_rule.AutoInline),
    )
    check_sketches(
        mod,
        sketches=actual,
        expected_mods=[padded_matmul_relu_0],
        expected_decisions=[decision_0],
    )


def test_conv_1x1():
    # fmt: off
    @T.prim_func
    def conv2d_1x1_0(inputs: T.Buffer[(1, 16, 16, 64), "float16"], weight: T.Buffer[(1, 1, 64, 64), "float16"], conv2d_nhwc: T.Buffer[(1, 16, 16, 64), "float32"]) -> None:
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        conv2d_nhwc_reindex_shared = T.alloc_buffer([256, 64], dtype="float32", scope="shared")
        conv2d_nhwc_reindex_shared_wmma_accumulator = T.alloc_buffer([256, 64], dtype="float32", scope="wmma.accumulator")
        PadInput_reindex_shared = T.alloc_buffer([256, 64], dtype="float16", scope="shared")
        weight_reindex_shared = T.alloc_buffer([1, 1, 64, 64], dtype="float16", scope="shared")
        PadInput_reindex_shared_wmma_matrix_a = T.alloc_buffer([256, 64], dtype="float16", scope="wmma.matrix_a")
        weight_reindex_shared_wmma_matrix_b = T.alloc_buffer([1, 1, 64, 64], dtype="float16", scope="wmma.matrix_b")
        for ax2_0_0_ax3_0_0_fused in T.thread_binding(16, thread="blockIdx.y"):
            for ax2_0_1_ax3_0_1_fused in T.thread_binding(2, thread="blockIdx.x"):
                for ax2_0_2_ax3_0_2_fused in T.thread_binding(2, thread="threadIdx.y"):
                    for ax0_0, ax1_0, ax4_0_0 in T.grid(1, 1, 1):
                        for ax0_ax1_fused in T.serial(1024):
                            with T.block("PadInput_reindex_shared"):
                                v0 = T.axis.spatial(256, ax2_0_0_ax3_0_0_fused // 2 * 32 + ax2_0_1_ax3_0_1_fused * 16 + ax0_ax1_fused // 64)
                                v1 = T.axis.spatial(64, ax0_ax1_fused % 64)
                                T.reads(inputs[v0 // 256, v0 // 16, v0 % 16, v1])
                                T.writes(PadInput_reindex_shared[v0, v1])
                                T.block_attr({"buffer_dim_align":[[0, 0, 32, 8]], "meta_schedule.cooperative_fetch":1})
                                PadInput_reindex_shared[v0, v1] = inputs[v0 // 256, v0 // 16, v0 % 16, v1]
                        for ax0_ax1_ax2_ax3_fused in T.serial(2048):
                            with T.block("weight_reindex_shared"):
                                v0 = T.axis.spatial(1, 0)
                                v1 = T.axis.spatial(1, 0)
                                v2 = T.axis.spatial(64, ax0_ax1_ax2_ax3_fused // 32)
                                v3 = T.axis.spatial(64, ax2_0_0_ax3_0_0_fused % 2 * 32 + ax0_ax1_ax2_ax3_fused % 32)
                                T.reads(weight[v0, v1, v2, v3])
                                T.writes(weight_reindex_shared[v0, v1, v2, v3])
                                T.block_attr({"buffer_dim_align":[[0, 2, 32, 8]], "meta_schedule.cooperative_fetch":4})
                                weight_reindex_shared[v0, v1, v2, v3] = weight[v0, v1, v2, v3]
                        for ax0_1, ax1_1, ax4_0_1 in T.grid(1, 1, 1):
                            for ax0_0_1, ax1_0_1 in T.grid(1, 4):
                                with T.block("PadInput_reindex_shared_wmma.matrix_a_o"):
                                    v0_o = T.axis.spatial(16, ax2_0_0_ax3_0_0_fused // 2 * 2 + ax2_0_1_ax3_0_1_fused)
                                    v1_o = T.axis.spatial(4, ax1_0_1)
                                    T.reads(PadInput_reindex_shared[v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16])
                                    T.writes(PadInput_reindex_shared_wmma_matrix_a[v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16])
                                    T.block_attr({"meta_schedule.auto_tensorize":"wmma_load_16x16x16_f16_a"})
                                    for ax0_1_1, ax1_1_1 in T.grid(16, 16):
                                        with T.block("PadInput_reindex_shared_wmma.matrix_a"):
                                            v0_i, v1_i = T.axis.remap("SS", [ax0_1_1, ax1_1_1])
                                            T.reads(PadInput_reindex_shared[v0_o * 16 + v0_i, v1_o * 16 + v1_i])
                                            T.writes(PadInput_reindex_shared_wmma_matrix_a[v0_o * 16 + v0_i, v1_o * 16 + v1_i])
                                            PadInput_reindex_shared_wmma_matrix_a[v0_o * 16 + v0_i, v1_o * 16 + v1_i] = PadInput_reindex_shared[v0_o * 16 + v0_i, v1_o * 16 + v1_i]
                            for ax0, ax1, ax2_0, ax3_0 in T.grid(1, 1, 4, 1):
                                with T.block("weight_reindex_shared_wmma.matrix_b_o"):
                                    v0 = T.axis.spatial(1, 0)
                                    v1 = T.axis.spatial(1, 0)
                                    v2_o = T.axis.spatial(4, ax2_0)
                                    v3_o = T.axis.spatial(4, ax2_0_0_ax3_0_0_fused % 2 * 2 + ax2_0_2_ax3_0_2_fused)
                                    T.reads(weight_reindex_shared[v0, v1, v2_o * 16 : v2_o * 16 + 16, v3_o * 16 : v3_o * 16 + 16])
                                    T.writes(weight_reindex_shared_wmma_matrix_b[v0, v1, v2_o * 16 : v2_o * 16 + 16, v3_o * 16 : v3_o * 16 + 16])
                                    T.block_attr({"meta_schedule.auto_tensorize":"wmma_load_16x16x16_f16_b"})
                                    for ax2_1, ax3_1 in T.grid(16, 16):
                                        with T.block("weight_reindex_shared_wmma.matrix_b"):
                                            v2_i, v3_i = T.axis.remap("SS", [ax2_1, ax3_1])
                                            T.reads(weight_reindex_shared[v0, v1, v2_o * 16 + v2_i, v3_o * 16 + v3_i])
                                            T.writes(weight_reindex_shared_wmma_matrix_b[v0, v1, v2_o * 16 + v2_i, v3_o * 16 + v3_i])
                                            weight_reindex_shared_wmma_matrix_b[v0, v1, v2_o * 16 + v2_i, v3_o * 16 + v3_i] = weight_reindex_shared[v0, v1, v2_o * 16 + v2_i, v3_o * 16 + v3_i]
                            for ax2_0_3, ax3_0_3, ax0_2, ax1_2, ax4_0_2, ax2_0_4, ax3_0_4 in T.grid(1, 1, 1, 1, 4, 1, 1):
                                with T.block("conv2d_nhwc_o"):
                                    v0 = T.axis.reduce(1, 0)
                                    v1 = T.axis.reduce(1, 0)
                                    v2_o = T.axis.spatial(16, ax2_0_4 + ax2_0_0_ax3_0_0_fused // 2 * 2 + ax2_0_1_ax3_0_1_fused + ax2_0_3)
                                    v3_o = T.axis.spatial(4, ax3_0_4 + ax2_0_0_ax3_0_0_fused % 2 * 2 + ax2_0_2_ax3_0_2_fused + ax3_0_3)
                                    v4_o = T.axis.reduce(4, ax4_0_0 * 4 + ax4_0_1 * 4 + ax4_0_2)
                                    T.reads(PadInput_reindex_shared_wmma_matrix_a[v2_o * 16 : v2_o * 16 + 16, v4_o * 16 : v4_o * 16 + 16], weight_reindex_shared_wmma_matrix_b[v0, v1, v4_o * 16 : v4_o * 16 + 16, v3_o * 16 : v3_o * 16 + 16])
                                    T.writes(conv2d_nhwc_reindex_shared_wmma_accumulator[v2_o * 16 : v2_o * 16 + 16, v3_o * 16 : v3_o * 16 + 16])
                                    T.block_attr({"meta_schedule.auto_tensorize":"wmma_sync_16x16x16_f16f16f32", "meta_schedule.auto_tensorize_init":"wmma_fill_16x16x16_f32", "warp_execution":1})
                                    with T.init():
                                        for ax2_1, ax3_1 in T.grid(16, 16):
                                            with T.block("conv2d_nhwc_init"):
                                                v2_i_init, v3_i_init = T.axis.remap("SS", [ax2_1, ax3_1])
                                                T.reads()
                                                T.writes(conv2d_nhwc_reindex_shared_wmma_accumulator[v2_o * 16 + v2_i_init, v3_o * 16 + v3_i_init])
                                                conv2d_nhwc_reindex_shared_wmma_accumulator[v2_o * 16 + v2_i_init, v3_o * 16 + v3_i_init] = T.float32(0)
                                    for ax2_1, ax3_1, ax4_1 in T.grid(16, 16, 16):
                                        with T.block("conv2d_nhwc"):
                                            v2_i, v3_i, v4_i = T.axis.remap("SSR", [ax2_1, ax3_1, ax4_1])
                                            T.reads(conv2d_nhwc_reindex_shared_wmma_accumulator[v2_o * 16 + v2_i, v3_o * 16 + v3_i], PadInput_reindex_shared_wmma_matrix_a[v2_o * 16 + v2_i, v4_o * 16 + v4_i], weight_reindex_shared_wmma_matrix_b[v0, v1, v4_o * 16 + v4_i, v3_o * 16 + v3_i])
                                            T.writes(conv2d_nhwc_reindex_shared_wmma_accumulator[v2_o * 16 + v2_i, v3_o * 16 + v3_i])
                                            T.block_attr({"meta_schedule.tiling_structure":"SSSRRSRS"})
                                            conv2d_nhwc_reindex_shared_wmma_accumulator[v2_o * 16 + v2_i, v3_o * 16 + v3_i] = conv2d_nhwc_reindex_shared_wmma_accumulator[v2_o * 16 + v2_i, v3_o * 16 + v3_i] + T.cast(PadInput_reindex_shared_wmma_matrix_a[v2_o * 16 + v2_i, v4_o * 16 + v4_i], "float32") * T.cast(weight_reindex_shared_wmma_matrix_b[v0, v1, v4_o * 16 + v4_i, v3_o * 16 + v3_i], "float32")
                    for ax0_0, ax1_0 in T.grid(1, 1):
                        with T.block("conv2d_nhwc_reindex_shared_wmma.accumulator_o"):
                            v0_o = T.axis.spatial(16, ax2_0_0_ax3_0_0_fused // 2 * 2 + ax2_0_1_ax3_0_1_fused)
                            v1_o = T.axis.spatial(4, ax2_0_0_ax3_0_0_fused % 2 * 2 + ax2_0_2_ax3_0_2_fused)
                            T.reads(conv2d_nhwc_reindex_shared_wmma_accumulator[v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16])
                            T.writes(conv2d_nhwc_reindex_shared[v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16])
                            T.block_attr({"meta_schedule.auto_tensorize":"wmma_store_16x16x16_f32_shared"})
                            for ax0_1, ax1_1 in T.grid(16, 16):
                                with T.block("conv2d_nhwc_reindex_shared_wmma.accumulator"):
                                    v0_i, v1_i = T.axis.remap("SS", [ax0_1, ax1_1])
                                    T.reads(conv2d_nhwc_reindex_shared_wmma_accumulator[v0_o * 16 + v0_i, v1_o * 16 + v1_i])
                                    T.writes(conv2d_nhwc_reindex_shared[v0_o * 16 + v0_i, v1_o * 16 + v1_i])
                                    conv2d_nhwc_reindex_shared[v0_o * 16 + v0_i, v1_o * 16 + v1_i] = conv2d_nhwc_reindex_shared_wmma_accumulator[v0_o * 16 + v0_i, v1_o * 16 + v1_i]
                for ax0_ax1_fused in T.serial(512):
                    with T.block("conv2d_nhwc_reindex_shared"):
                        v0 = T.axis.spatial(256, ax2_0_0_ax3_0_0_fused // 2 * 32 + ax2_0_1_ax3_0_1_fused * 16 + ax0_ax1_fused // 32)
                        v1 = T.axis.spatial(64, ax2_0_0_ax3_0_0_fused % 2 * 32 + ax0_ax1_fused % 32)
                        T.reads(conv2d_nhwc_reindex_shared[v0, v1])
                        T.writes(conv2d_nhwc[v0 // 256, v0 // 16, v0 % 16, v1])
                        T.block_attr({"meta_schedule.cooperative_fetch":2})
                        conv2d_nhwc[v0 // 256, v0 // 16, v0 % 16, v1] = conv2d_nhwc_reindex_shared[v0, v1]
    # fmt: on

    decision_0 = [
        ("SamplePerfectTile", [1, 1, 1]),
        ("SamplePerfectTile", [1, 1, 1]),
        ("SamplePerfectTile", [8, 2, 1, 1, 1]),
        ("SamplePerfectTile", [2, 1, 2, 1, 1]),
        ("SamplePerfectTile", [1, 1, 4]),
        ("SampleCategorical", 1),
        ("SampleCategorical", 0),
        ("SampleCategorical", 2),
    ]

    mod = te.create_prim_func(
        te_workload.conv2d_nhwc(
            1,
            16,
            16,
            64,
            64,
            1,
            1,
            0,
            in_dtype="float16",
            out_dtype="float32",
        )
    )
    actual = generate_design_space(
        kind="cuda",
        mod=mod,
        target=tvm.target.Target("cuda"),
        types=None,
        sch_rules=[multi_level_tiling_tensor_core(write_reuse_scope="shared")]
        + get_rules("cuda", ms.schedule_rule.AutoInline),
    )
    check_sketches(
        mod,
        sketches=actual,
        expected_mods=[conv2d_1x1_0],
        expected_decisions=[decision_0],
    )


if __name__ == "__main__":
    tvm.testing.main()
