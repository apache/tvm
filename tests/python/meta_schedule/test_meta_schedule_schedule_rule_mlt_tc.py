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
# pylint: disable=missing-module-docstring,missing-function-docstring,missing-class-docstring,line-too-long,invalid-name,too-many-locals,too-many-statements,too-many-nested-blocks,too-many-branches,too-many-lines,chained-comparison

import pytest

import tvm
import tvm.testing
from tvm import meta_schedule as ms
from tvm import te
from tvm.meta_schedule.testing import te_workload
from tvm.meta_schedule.testing.space_generation import (
    check_sketches,
    generate_design_space,
    get_rules,
    print_sketches,
)
from tvm.script import tir as T
from tvm.tir.tensor_intrin.cuda import get_wmma_intrin_group


def multi_level_tiling_tensor_core(
    *,
    read_reuse_scope="shared",
    write_reuse_scope="shared",
    in_dtype="float16",
    out_dtype="float32",
    trans_b=False,
    use_software_pipeline=False,
) -> ms.schedule_rule.ScheduleRule:
    assert read_reuse_scope in ["shared", "shared.dyn"]
    assert write_reuse_scope in ["shared", "shared.dyn", "global"]
    if not isinstance(in_dtype, list):
        in_dtype = [in_dtype]
    if not isinstance(out_dtype, list):
        out_dtype = [out_dtype]
    if not isinstance(trans_b, list):
        trans_b = [trans_b]
    return ms.schedule_rule.MultiLevelTilingTensorCore(
        intrin_groups=[
            get_wmma_intrin_group(
                read_reuse_scope, write_reuse_scope, _in_dtype, _out_dtype, _trans_b
            )
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
            scope=read_reuse_scope,
        ),
        reuse_write=ms.schedule_rule.ReuseType(
            req="must" if write_reuse_scope.startswith("shared") else "no",
            levels=[2],
            scope=write_reuse_scope,
        ),
        use_software_pipeline=use_software_pipeline,
    )


@pytest.mark.parametrize("shared_scope", ["shared", "shared.dyn"])
def test_matmul_relu(shared_scope):
    intrin_suffix = shared_scope.replace(".", "_")
    # fmt: off
    @T.prim_func
    def matmul_relu_0(A: T.Buffer((128, 128), "float16"), B: T.Buffer((128, 128), "float16"), compute: T.Buffer((128, 128), "float32")) -> None:
        T.func_attr({"global_symbol": "main", "tir.noalias": T.bool(True)})
        # with T.block("root"):
        C_reindex_shared = T.alloc_buffer((4, 8, 2, 1, 16, 16), scope=shared_scope)
        C_reindex_shared_wmma_accumulator = T.alloc_buffer((4, 8, 2, 1, 16, 16), scope="wmma.accumulator")
        A_reindex_shared = T.alloc_buffer((128, 128), "float16", scope=shared_scope)
        B_reindex_shared = T.alloc_buffer((128, 128), "float16", scope=shared_scope)
        A_reindex_shared_wmma_matrix_a = T.alloc_buffer((128, 128), "float16", scope="wmma.matrix_a")
        B_reindex_shared_wmma_matrix_b = T.alloc_buffer((128, 128), "float16", scope="wmma.matrix_b")
        for ax0_0_0_ax1_0_0_fused in T.thread_binding(8, thread="blockIdx.y"):
            for ax0_0_1_ax1_0_1_fused in T.thread_binding(2, thread="blockIdx.x"):
                for ax0_0_2_ax1_0_2_fused in T.thread_binding(2, thread="threadIdx.y"):
                    for ax2_0_0 in range(1):
                        for ax0_ax1_fused in range(4096):
                            with T.block("A_reindex_shared"):
                                v0 = T.axis.spatial(128, ax0_0_0_ax1_0_0_fused // 2 * 32 + ax0_ax1_fused // 128)
                                v1 = T.axis.spatial(128, ax0_ax1_fused % 128)
                                T.reads(A[v0, v1])
                                T.writes(A_reindex_shared[v0, v1])
                                T.block_attr({"buffer_dim_align": [[0, 0, 32, 8]], "meta_schedule.cooperative_fetch": 8})
                                A_reindex_shared[v0, v1] = A[v0, v1]
                        for ax0_ax1_fused in range(4096):
                            with T.block("B_reindex_shared"):
                                v0 = T.axis.spatial(128, ax0_ax1_fused // 32)
                                v1 = T.axis.spatial(128, ax0_0_0_ax1_0_0_fused % 2 * 64 + ax0_0_1_ax1_0_1_fused * 32 + ax0_ax1_fused % 32)
                                T.reads(B[v0, v1])
                                T.writes(B_reindex_shared[v0, v1])
                                T.block_attr({"buffer_dim_align": [[0, 0, 32, 8]], "meta_schedule.cooperative_fetch": 1})
                                B_reindex_shared[v0, v1] = B[v0, v1]
                        for ax2_0_1 in range(4):
                            for ax0_0, ax1_0 in T.grid(2, 2):
                                with T.block("A_reindex_shared_wmma.matrix_a_o"):
                                    v0_o = T.axis.spatial(8, ax0_0_0_ax1_0_0_fused // 2 * 2 + ax0_0)
                                    v1_o = T.axis.spatial(8, ax2_0_1 * 2 + ax1_0)
                                    T.reads(A_reindex_shared[v0_o * 16:v0_o * 16 + 16, v1_o * 16:v1_o * 16 + 16])
                                    T.writes(A_reindex_shared_wmma_matrix_a[v0_o * 16:v0_o * 16 + 16, v1_o * 16:v1_o * 16 + 16])
                                    T.block_attr({"meta_schedule.auto_tensorize": f"wmma_load_16x16x16_f16_a_{intrin_suffix}"})
                                    for ax0_1, ax1_1 in T.grid(16, 16):
                                        with T.block("A_reindex_shared_wmma.matrix_a"):
                                            v0_i, v1_i = T.axis.remap("SS", [ax0_1, ax1_1])
                                            T.reads(A_reindex_shared[v0_o * 16 + v0_i, v1_o * 16 + v1_i])
                                            T.writes(A_reindex_shared_wmma_matrix_a[v0_o * 16 + v0_i, v1_o * 16 + v1_i])
                                            A_reindex_shared_wmma_matrix_a[v0_o * 16 + v0_i, v1_o * 16 + v1_i] = A_reindex_shared[v0_o * 16 + v0_i, v1_o * 16 + v1_i]
                            for ax0_0, ax1_0 in T.grid(2, 1):
                                with T.block("B_reindex_shared_wmma.matrix_b_o"):
                                    v0_o = T.axis.spatial(8, ax2_0_1 * 2 + ax0_0)
                                    v1_o = T.axis.spatial(8, ax0_0_0_ax1_0_0_fused % 2 * 4 + ax0_0_1_ax1_0_1_fused * 2 + ax0_0_2_ax1_0_2_fused + ax1_0)
                                    T.reads(B_reindex_shared[v0_o * 16:v0_o * 16 + 16, v1_o * 16:v1_o * 16 + 16])
                                    T.writes(B_reindex_shared_wmma_matrix_b[v0_o * 16:v0_o * 16 + 16, v1_o * 16:v1_o * 16 + 16])
                                    T.block_attr({"meta_schedule.auto_tensorize": f"wmma_load_16x16x16_f16_b_{intrin_suffix}"})
                                    for ax0_1, ax1_1 in T.grid(16, 16):
                                        with T.block("B_reindex_shared_wmma.matrix_b"):
                                            v0_i, v1_i = T.axis.remap("SS", [ax0_1, ax1_1])
                                            T.reads(B_reindex_shared[v0_o * 16 + v0_i, v1_o * 16 + v1_i])
                                            T.writes(B_reindex_shared_wmma_matrix_b[v0_o * 16 + v0_i, v1_o * 16 + v1_i])
                                            B_reindex_shared_wmma_matrix_b[v0_o * 16 + v0_i, v1_o * 16 + v1_i] = B_reindex_shared[v0_o * 16 + v0_i, v1_o * 16 + v1_i]
                            for ax0_0_3, ax1_0_3, ax2_0_2, ax0_0_4, ax1_0_4 in T.grid(1, 1, 2, 2, 1):
                                with T.block("C_o"):
                                    v0_o = T.axis.spatial(8, ax0_0_0_ax1_0_0_fused // 2 * 2 + ax0_0_3 * 2 + ax0_0_4)
                                    v1_o = T.axis.spatial(8, ax0_0_0_ax1_0_0_fused % 2 * 4 + ax0_0_1_ax1_0_1_fused * 2 + ax0_0_2_ax1_0_2_fused + ax1_0_3 + ax1_0_4)
                                    v2_o = T.axis.reduce(8, ax2_0_0 * 8 + ax2_0_1 * 2 + ax2_0_2)
                                    T.reads(A_reindex_shared_wmma_matrix_a[v0_o * 16:v0_o * 16 + 16, v2_o * 16:v2_o * 16 + 16], B_reindex_shared_wmma_matrix_b[v2_o * 16:v2_o * 16 + 16, v1_o * 16:v1_o * 16 + 16])
                                    T.writes(C_reindex_shared_wmma_accumulator[v0_o // 2, v1_o, v0_o % 2, 0, 0:16, 0:16])
                                    T.block_attr({"meta_schedule.auto_tensorize": "wmma_sync_16x16x16_f16f16f32", "meta_schedule.auto_tensorize_init": "wmma_fill_16x16x16_f32", "warp_execution": 1})
                                    with T.init():
                                        for ax0_1, ax1_1 in T.grid(16, 16):
                                            with T.block("C_init"):
                                                v0_i_init, v1_i_init = T.axis.remap("SS", [ax0_1, ax1_1])
                                                T.reads()
                                                T.writes(C_reindex_shared_wmma_accumulator[v0_o // 2, v1_o, v0_o % 2, 0, v0_i_init, v1_i_init])
                                                C_reindex_shared_wmma_accumulator[v0_o // 2, v1_o, v0_o % 2, 0, v0_i_init, v1_i_init] = T.float32(0)
                                    for ax0_1, ax1_1, ax2_1 in T.grid(16, 16, 16):
                                        with T.block("C"):
                                            v0_i, v1_i, v2_i = T.axis.remap("SSR", [ax0_1, ax1_1, ax2_1])
                                            T.reads(C_reindex_shared_wmma_accumulator[v0_o // 2, v1_o, v0_o % 2, 0, v0_i, v1_i], A_reindex_shared_wmma_matrix_a[v0_o * 16 + v0_i, v2_o * 16 + v2_i], B_reindex_shared_wmma_matrix_b[v2_o * 16 + v2_i, v1_o * 16 + v1_i])
                                            T.writes(C_reindex_shared_wmma_accumulator[v0_o // 2, v1_o, v0_o % 2, 0, v0_i, v1_i])
                                            T.block_attr({"meta_schedule.tiling_structure": "SSSRRSRS"})
                                            C_reindex_shared_wmma_accumulator[v0_o // 2, v1_o, v0_o % 2, 0, v0_i, v1_i] = C_reindex_shared_wmma_accumulator[v0_o // 2, v1_o, v0_o % 2, 0, v0_i, v1_i] + T.Cast("float32", A_reindex_shared_wmma_matrix_a[v0_o * 16 + v0_i, v2_o * 16 + v2_i]) * T.Cast("float32", B_reindex_shared_wmma_matrix_b[v2_o * 16 + v2_i, v1_o * 16 + v1_i])
                for ax2 in range(2):
                    for ax0_ax1_fused in T.thread_binding(2, thread="threadIdx.y"):
                        for ax2_1, ax3 in T.grid(1, 1):
                            with T.block("C_reindex_shared_wmma.accumulator_o"):
                                v0 = T.axis.spatial(4, ax0_0_0_ax1_0_0_fused // 2)
                                v1 = T.axis.spatial(8, ax0_0_0_ax1_0_0_fused % 2 * 4 + ax0_0_1_ax1_0_1_fused * 2 + ax0_ax1_fused)
                                v2 = T.axis.spatial(2, ax2 + ax2_1)
                                v3 = T.axis.spatial(1, ax3)
                                v4_o = T.axis.spatial(1, 0)
                                v5_o = T.axis.spatial(1, 0)
                                T.reads(C_reindex_shared_wmma_accumulator[v0, v1, v2, v3, 0:16, 0:16])
                                T.writes(C_reindex_shared[v0, v1, v2, v3, 0:16, 0:16])
                                T.block_attr({"meta_schedule.auto_tensorize": f"wmma_store_16x16x16_f32_{intrin_suffix}"})
                                for ax4, ax5 in T.grid(16, 16):
                                    with T.block("C_reindex_shared_wmma.accumulator"):
                                        v4_i, v5_i = T.axis.remap("SS", [ax4, ax5])
                                        T.reads(C_reindex_shared_wmma_accumulator[v0, v1, v2, v3, v4_i, v5_i])
                                        T.writes(C_reindex_shared[v0, v1, v2, v3, v4_i, v5_i])
                                        C_reindex_shared[v0, v1, v2, v3, v4_i, v5_i] = C_reindex_shared_wmma_accumulator[v0, v1, v2, v3, v4_i, v5_i]
                    for ax0_ax1_ax3_ax4_ax5_fused in range(512):
                        with T.block("C_reindex_shared"):
                            v0 = T.axis.spatial(4, ax0_0_0_ax1_0_0_fused // 2)
                            v1 = T.axis.spatial(8, ax0_0_0_ax1_0_0_fused % 2 * 4 + ax0_0_1_ax1_0_1_fused * 2 + ax0_ax1_ax3_ax4_ax5_fused // 256)
                            v2 = T.axis.spatial(2, ax2)
                            v3 = T.axis.spatial(1, 0)
                            v4 = T.axis.spatial(16, ax0_ax1_ax3_ax4_ax5_fused % 256 // 16)
                            v5 = T.axis.spatial(16, ax0_ax1_ax3_ax4_ax5_fused % 16)
                            T.reads(C_reindex_shared[v0, v1, v2, v3, v4, v5])
                            T.writes(compute[v4 + v2 * 16 + v0 * 32, v5 + v1 * 16])
                            T.block_attr({"meta_schedule.cooperative_fetch": 4})
                            compute[v4 + v2 * 16 + v0 * 32, v5 + v1 * 16] = T.max(C_reindex_shared[v0, v1, v2, v3, v4, v5], T.float32(0))
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
        target=tvm.target.Target("cuda --arch=sm_70"),
        types=None,
        sch_rules=[
            multi_level_tiling_tensor_core(
                read_reuse_scope=shared_scope, write_reuse_scope=shared_scope
            ),
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
    def matmul_relu_fallback_0(A: T.Buffer((128, 128), "float16"), B: T.Buffer((128, 128), "float16"), compute: T.Buffer((128, 128), "float32")) -> None:
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # with T.block("root"):
        C_reindex_shared = T.alloc_buffer((4, 2, 2, 4, 16, 16), scope="shared")
        C_reindex_shared_wmma_accumulator = T.alloc_buffer((4, 2, 2, 4, 16, 16), scope="wmma.accumulator")
        A_reindex_shared = T.alloc_buffer((128, 128), "float16", scope="shared")
        B_reindex_shared = T.alloc_buffer((128, 128), "float16", scope="shared")
        A_reindex_shared_wmma_matrix_a = T.alloc_buffer((128, 128), "float16", scope="wmma.matrix_a")
        B_reindex_shared_wmma_matrix_b = T.alloc_buffer((128, 128), "float16", scope="wmma.matrix_b")
        for ax0_0_0_ax1_0_0_fused in T.thread_binding(2, thread="blockIdx.y"):
            for ax0_0_1_ax1_0_1_fused in T.thread_binding(2, thread="blockIdx.x"):
                for ax0_0_2_ax1_0_2_fused in T.thread_binding(2, thread="threadIdx.y"):
                    for ax2_0_0 in range(2):
                        for ax0_ax1_fused in range(2048):
                            with T.block("A_reindex_shared"):
                                v0 = T.axis.spatial(128, ax0_0_0_ax1_0_0_fused * 64 + ax0_0_1_ax1_0_1_fused * 32 + ax0_ax1_fused // 64)
                                v1 = T.axis.spatial(128, ax2_0_0 * 64 + ax0_ax1_fused % 64)
                                T.reads(A[v0, v1])
                                T.writes(A_reindex_shared[v0, v1])
                                T.block_attr({"buffer_dim_align": [[0, 0, 32, 8]], "meta_schedule.cooperative_fetch": 4})
                                A_reindex_shared[v0, v1] = A[v0, v1]
                        for ax0_ax1_fused in range(8192):
                            with T.block("B_reindex_shared"):
                                v0 = T.axis.spatial(128, ax2_0_0 * 64 + ax0_ax1_fused // 128)
                                v1 = T.axis.spatial(128, ax0_ax1_fused % 128)
                                T.reads(B[v0, v1])
                                T.writes(B_reindex_shared[v0, v1])
                                T.block_attr({"buffer_dim_align": [[0, 0, 32, 8]], "meta_schedule.cooperative_fetch": 2})
                                B_reindex_shared[v0, v1] = B[v0, v1]
                        for ax2_0_1 in range(1):
                            for ax0_0, ax1_0 in T.grid(2, 4):
                                with T.block("A_reindex_shared_wmma.matrix_a_o"):
                                    v0_o = T.axis.spatial(8, ax0_0_0_ax1_0_0_fused * 4 + ax0_0_1_ax1_0_1_fused * 2 + ax0_0)
                                    v1_o = T.axis.spatial(8, ax2_0_0 * 4 + ax1_0)
                                    T.reads(A_reindex_shared[v0_o * 16:v0_o * 16 + 16, v1_o * 16:v1_o * 16 + 16])
                                    T.writes(A_reindex_shared_wmma_matrix_a[v0_o * 16:v0_o * 16 + 16, v1_o * 16:v1_o * 16 + 16])
                                    T.block_attr({"meta_schedule.auto_tensorize": "wmma_load_16x16x16_f16_a_shared"})
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
                                    T.reads(B_reindex_shared[v0_o * 16:v0_o * 16 + 16, v1_o * 16:v1_o * 16 + 16])
                                    T.writes(B_reindex_shared_wmma_matrix_b[v0_o * 16:v0_o * 16 + 16, v1_o * 16:v1_o * 16 + 16])
                                    T.block_attr({"meta_schedule.auto_tensorize": "wmma_load_16x16x16_f16_b_shared"})
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
                                    T.reads(A_reindex_shared_wmma_matrix_a[v0_o * 16:v0_o * 16 + 16, v2_o * 16:v2_o * 16 + 16], B_reindex_shared_wmma_matrix_b[v2_o * 16:v2_o * 16 + 16, v1_o * 16:v1_o * 16 + 16])
                                    T.writes(C_reindex_shared_wmma_accumulator[v0_o // 2, v1_o // 4, v0_o % 2, v1_o % 4, 0:16, 0:16])
                                    T.block_attr({"meta_schedule.auto_tensorize": "wmma_sync_16x16x16_f16f16f32", "meta_schedule.auto_tensorize_init": "wmma_fill_16x16x16_f32", "warp_execution": 1})
                                    with T.init():
                                        for ax0_1, ax1_1 in T.grid(16, 16):
                                            with T.block("C_init"):
                                                v0_i_init, v1_i_init = T.axis.remap("SS", [ax0_1, ax1_1])
                                                T.reads()
                                                T.writes(C_reindex_shared_wmma_accumulator[v0_o // 2, v1_o // 4, v0_o % 2, v1_o % 4, v0_i_init, v1_i_init])
                                                C_reindex_shared_wmma_accumulator[v0_o // 2, v1_o // 4, v0_o % 2, v1_o % 4, v0_i_init, v1_i_init] = T.float32(0)
                                    for ax0_1, ax1_1, ax2_1 in T.grid(16, 16, 16):
                                        with T.block("C"):
                                            v0_i, v1_i, v2_i = T.axis.remap("SSR", [ax0_1, ax1_1, ax2_1])
                                            T.reads(C_reindex_shared_wmma_accumulator[v0_o // 2, v1_o // 4, v0_o % 2, v1_o % 4, v0_i, v1_i], A_reindex_shared_wmma_matrix_a[v0_o * 16 + v0_i, v2_o * 16 + v2_i], B_reindex_shared_wmma_matrix_b[v2_o * 16 + v2_i, v1_o * 16 + v1_i])
                                            T.writes(C_reindex_shared_wmma_accumulator[v0_o // 2, v1_o // 4, v0_o % 2, v1_o % 4, v0_i, v1_i])
                                            T.block_attr({"meta_schedule.tiling_structure": "SSSRRSRS"})
                                            C_reindex_shared_wmma_accumulator[v0_o // 2, v1_o // 4, v0_o % 2, v1_o % 4, v0_i, v1_i] = C_reindex_shared_wmma_accumulator[v0_o // 2, v1_o // 4, v0_o % 2, v1_o % 4, v0_i, v1_i] + T.Cast("float32", A_reindex_shared_wmma_matrix_a[v0_o * 16 + v0_i, v2_o * 16 + v2_i]) * T.Cast("float32", B_reindex_shared_wmma_matrix_b[v2_o * 16 + v2_i, v1_o * 16 + v1_i])
                for ax2 in range(2):
                    for ax0_ax1_fused in T.thread_binding(2, thread="threadIdx.y"):
                        for ax2_1, ax3 in T.grid(1, 4):
                            with T.block("C_reindex_shared_wmma.accumulator_o"):
                                v0 = T.axis.spatial(4, ax0_0_0_ax1_0_0_fused * 2 + ax0_0_1_ax1_0_1_fused)
                                v1 = T.axis.spatial(2, ax0_ax1_fused)
                                v2 = T.axis.spatial(2, ax2 + ax2_1)
                                v3 = T.axis.spatial(4, ax3)
                                v4_o = T.axis.spatial(1, 0)
                                v5_o = T.axis.spatial(1, 0)
                                T.reads(C_reindex_shared_wmma_accumulator[v0, v1, v2, v3, 0:16, 0:16])
                                T.writes(C_reindex_shared[v0, v1, v2, v3, 0:16, 0:16])
                                T.block_attr({"meta_schedule.auto_tensorize": "wmma_store_16x16x16_f32_shared"})
                                for ax4, ax5 in T.grid(16, 16):
                                    with T.block("C_reindex_shared_wmma.accumulator"):
                                        v4_i, v5_i = T.axis.remap("SS", [ax4, ax5])
                                        T.reads(C_reindex_shared_wmma_accumulator[v0, v1, v2, v3, v4_i, v5_i])
                                        T.writes(C_reindex_shared[v0, v1, v2, v3, v4_i, v5_i])
                                        C_reindex_shared[v0, v1, v2, v3, v4_i, v5_i] = C_reindex_shared_wmma_accumulator[v0, v1, v2, v3, v4_i, v5_i]
                    for ax0_ax1_ax3_ax4_ax5_fused in range(2048):
                        with T.block("C_reindex_shared"):
                            v0 = T.axis.spatial(4, ax0_0_0_ax1_0_0_fused * 2 + ax0_0_1_ax1_0_1_fused)
                            v1 = T.axis.spatial(2, ax0_ax1_ax3_ax4_ax5_fused // 1024)
                            v2 = T.axis.spatial(2, ax2)
                            v3 = T.axis.spatial(4, ax0_ax1_ax3_ax4_ax5_fused % 1024 // 256)
                            v4 = T.axis.spatial(16, ax0_ax1_ax3_ax4_ax5_fused % 256 // 16)
                            v5 = T.axis.spatial(16, ax0_ax1_ax3_ax4_ax5_fused % 16)
                            T.reads(C_reindex_shared[v0, v1, v2, v3, v4, v5])
                            T.writes(compute[v4 + v2 * 16 + v0 * 32, v5 + v3 * 16 + v1 * 64])
                            T.block_attr({"meta_schedule.cooperative_fetch": 4})
                            compute[v4 + v2 * 16 + v0 * 32, v5 + v3 * 16 + v1 * 64] = T.max(C_reindex_shared[v0, v1, v2, v3, v4, v5], T.float32(0))
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
        target=tvm.target.Target("cuda --arch=sm_70"),
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


@pytest.mark.parametrize("shared_scope", ["shared", "shared.dyn"])
def test_conv2d(shared_scope):
    intrin_suffix = shared_scope.replace(".", "_")
    # fmt: off
    @T.prim_func
    def conv2d_0(inputs: T.Buffer((1, 16, 16, 32), "float16"), weight: T.Buffer((3, 3, 32, 32), "float16"), conv2d_nhwc: T.Buffer((1, 16, 16, 32), "float32")):
        T.func_attr({"global_symbol": "main", "tir.noalias": T.bool(True)})
        # with T.block("root"):
        PadInput = T.alloc_buffer((1, 18, 18, 32), "float16")
        conv2d_nhwc_reindex_shared_dyn = T.alloc_buffer((16, 2, 1, 1, 16, 16), scope=shared_scope)
        conv2d_nhwc_reindex_shared_dyn_wmma_accumulator = T.alloc_buffer((16, 2, 1, 1, 16, 16), scope="wmma.accumulator")
        PadInput_reindex_shared_dyn = T.alloc_buffer((256, 288), "float16", scope=shared_scope)
        weight_reindex_shared_dyn = T.alloc_buffer((288, 32), "float16", scope=shared_scope)
        PadInput_reindex_shared_dyn_wmma_matrix_a = T.alloc_buffer((256, 288), "float16", scope="wmma.matrix_a")
        weight_reindex_shared_dyn_wmma_matrix_b = T.alloc_buffer((288, 32), "float16", scope="wmma.matrix_b")
        for i0, i1, i2, i3 in T.grid(1, 18, 18, 32):
            with T.block("PadInput"):
                v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(inputs[v_i0, v_i1 - 1, v_i2 - 1, v_i3])
                T.writes(PadInput[v_i0, v_i1, v_i2, v_i3])
                PadInput[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(1 <= v_i1 and v_i1 < 17 and 1 <= v_i2 and v_i2 < 17, inputs[v_i0, v_i1 - 1, v_i2 - 1, v_i3], T.float16(0))
        for ax0_0_0_ax1_0_0_fused in T.thread_binding(2, thread="blockIdx.y"):
            for ax0_0_1_ax1_0_1_fused in T.thread_binding(16, thread="blockIdx.x"):
                for ax0_0_2_ax1_0_2_fused in T.thread_binding(1, thread="threadIdx.y"):
                    for ax2_0_0 in range(1):
                        for ax0_ax1_fused in range(4608):
                            with T.block("PadInput_reindex_shared.dyn"):
                                v0 = T.axis.spatial(256, ax0_0_1_ax1_0_1_fused * 16 + ax0_ax1_fused // 288)
                                v1 = T.axis.spatial(288, ax0_ax1_fused % 288)
                                T.reads(PadInput[0, v0 // 16 + v1 // 96, v0 % 16 + v1 % 96 // 32, v1 % 32])
                                T.writes(PadInput_reindex_shared_dyn[v0, v1])
                                T.block_attr({"buffer_dim_align": [[0, 0, 32, 8]], "meta_schedule.cooperative_fetch": 2})
                                PadInput_reindex_shared_dyn[v0, v1] = PadInput[0, v0 // 16 + v1 // 96, v0 % 16 + v1 % 96 // 32, v1 % 32]
                        for ax0_ax1_fused in range(4608):
                            with T.block("weight_reindex_shared.dyn"):
                                v0 = T.axis.spatial(288, ax0_ax1_fused // 16)
                                v1 = T.axis.spatial(32, ax0_0_0_ax1_0_0_fused * 16 + ax0_ax1_fused % 16)
                                T.reads(weight[v0 // 96, v0 % 96 // 32, v0 % 32, v1])
                                T.writes(weight_reindex_shared_dyn[v0, v1])
                                T.block_attr({"buffer_dim_align": [[0, 0, 32, 8]], "meta_schedule.cooperative_fetch": 8})
                                weight_reindex_shared_dyn[v0, v1] = weight[v0 // 96, v0 % 96 // 32, v0 % 32, v1]
                        for ax2_0_1 in range(18):
                            for ax0_0, ax1_0 in T.grid(1, 1):
                                with T.block("PadInput_reindex_shared.dyn_wmma.matrix_a_o"):
                                    v0_o = T.axis.spatial(16, ax0_0_1_ax1_0_1_fused + ax0_0)
                                    v1_o = T.axis.spatial(18, ax2_0_1 + ax1_0)
                                    T.reads(PadInput_reindex_shared_dyn[v0_o * 16:v0_o * 16 + 16, v1_o * 16:v1_o * 16 + 16])
                                    T.writes(PadInput_reindex_shared_dyn_wmma_matrix_a[v0_o * 16:v0_o * 16 + 16, v1_o * 16:v1_o * 16 + 16])
                                    T.block_attr({"meta_schedule.auto_tensorize": f"wmma_load_16x16x16_f16_a_{intrin_suffix}"})
                                    for ax0_1, ax1_1 in T.grid(16, 16):
                                        with T.block("PadInput_reindex_shared.dyn_wmma.matrix_a"):
                                            v0_i, v1_i = T.axis.remap("SS", [ax0_1, ax1_1])
                                            T.reads(PadInput_reindex_shared_dyn[v0_o * 16 + v0_i, v1_o * 16 + v1_i])
                                            T.writes(PadInput_reindex_shared_dyn_wmma_matrix_a[v0_o * 16 + v0_i, v1_o * 16 + v1_i])
                                            PadInput_reindex_shared_dyn_wmma_matrix_a[v0_o * 16 + v0_i, v1_o * 16 + v1_i] = PadInput_reindex_shared_dyn[v0_o * 16 + v0_i, v1_o * 16 + v1_i]
                            for ax0_0, ax1_0 in T.grid(1, 1):
                                with T.block("weight_reindex_shared.dyn_wmma.matrix_b_o"):
                                    v0_o = T.axis.spatial(18, ax2_0_1 + ax0_0)
                                    v1_o = T.axis.spatial(2, ax0_0_0_ax1_0_0_fused + ax1_0)
                                    T.reads(weight_reindex_shared_dyn[v0_o * 16:v0_o * 16 + 16, v1_o * 16:v1_o * 16 + 16])
                                    T.writes(weight_reindex_shared_dyn_wmma_matrix_b[v0_o * 16:v0_o * 16 + 16, v1_o * 16:v1_o * 16 + 16])
                                    T.block_attr({"meta_schedule.auto_tensorize": f"wmma_load_16x16x16_f16_b_{intrin_suffix}"})
                                    for ax0_1, ax1_1 in T.grid(16, 16):
                                        with T.block("weight_reindex_shared.dyn_wmma.matrix_b"):
                                            v0_i, v1_i = T.axis.remap("SS", [ax0_1, ax1_1])
                                            T.reads(weight_reindex_shared_dyn[v0_o * 16 + v0_i, v1_o * 16 + v1_i])
                                            T.writes(weight_reindex_shared_dyn_wmma_matrix_b[v0_o * 16 + v0_i, v1_o * 16 + v1_i])
                                            weight_reindex_shared_dyn_wmma_matrix_b[v0_o * 16 + v0_i, v1_o * 16 + v1_i] = weight_reindex_shared_dyn[v0_o * 16 + v0_i, v1_o * 16 + v1_i]
                            for ax0_0_3, ax1_0_3, ax2_0_2, ax0_0_4, ax1_0_4 in T.grid(1, 1, 1, 1, 1):
                                with T.block("conv2d_nhwc_o"):
                                    v0_o = T.axis.spatial(16, ax0_0_1_ax1_0_1_fused + ax0_0_3 + ax0_0_4)
                                    v1_o = T.axis.spatial(2, ax0_0_0_ax1_0_0_fused + ax1_0_3 + ax1_0_4)
                                    v2_o = T.axis.reduce(18, ax2_0_0 * 18 + ax2_0_1 + ax2_0_2)
                                    T.reads(PadInput_reindex_shared_dyn_wmma_matrix_a[v0_o * 16:v0_o * 16 + 16, v2_o * 16:v2_o * 16 + 16], weight_reindex_shared_dyn_wmma_matrix_b[v2_o * 16:v2_o * 16 + 16, v1_o * 16:v1_o * 16 + 16])
                                    T.writes(conv2d_nhwc_reindex_shared_dyn_wmma_accumulator[v0_o, v1_o, 0, 0, 0:16, 0:16])
                                    T.block_attr({"meta_schedule.auto_tensorize": "wmma_sync_16x16x16_f16f16f32", "meta_schedule.auto_tensorize_init": "wmma_fill_16x16x16_f32", "warp_execution": 1})
                                    with T.init():
                                        for ax0_1, ax1_1 in T.grid(16, 16):
                                            with T.block("conv2d_nhwc_init"):
                                                v0_i_init, v1_i_init = T.axis.remap("SS", [ax0_1, ax1_1])
                                                T.reads()
                                                T.writes(conv2d_nhwc_reindex_shared_dyn_wmma_accumulator[v0_o, v1_o, 0, 0, v0_i_init, v1_i_init])
                                                conv2d_nhwc_reindex_shared_dyn_wmma_accumulator[v0_o, v1_o, 0, 0, v0_i_init, v1_i_init] = T.float32(0)
                                    for ax0_1, ax1_1, ax2_1 in T.grid(16, 16, 16):
                                        with T.block("conv2d_nhwc"):
                                            v0_i, v1_i, v2_i = T.axis.remap("SSR", [ax0_1, ax1_1, ax2_1])
                                            T.reads(conv2d_nhwc_reindex_shared_dyn_wmma_accumulator[v0_o, v1_o, 0, 0, v0_i, v1_i], PadInput_reindex_shared_dyn_wmma_matrix_a[v0_o * 16 + v0_i, v2_o * 16 + v2_i], weight_reindex_shared_dyn_wmma_matrix_b[v2_o * 16 + v2_i, v1_o * 16 + v1_i])
                                            T.writes(conv2d_nhwc_reindex_shared_dyn_wmma_accumulator[v0_o, v1_o, 0, 0, v0_i, v1_i])
                                            T.block_attr({"meta_schedule.tiling_structure": "SSSRRSRS"})
                                            conv2d_nhwc_reindex_shared_dyn_wmma_accumulator[v0_o, v1_o, 0, 0, v0_i, v1_i] = conv2d_nhwc_reindex_shared_dyn_wmma_accumulator[v0_o, v1_o, 0, 0, v0_i, v1_i] + T.Cast("float32", PadInput_reindex_shared_dyn_wmma_matrix_a[v0_o * 16 + v0_i, v2_o * 16 + v2_i]) * T.Cast("float32", weight_reindex_shared_dyn_wmma_matrix_b[v2_o * 16 + v2_i, v1_o * 16 + v1_i])
                for ax2 in range(1):
                    for ax0_ax1_fused in T.thread_binding(1, thread="threadIdx.y"):
                        for ax2_1, ax3 in T.grid(1, 1):
                            with T.block("conv2d_nhwc_reindex_shared.dyn_wmma.accumulator_o"):
                                v0, v1, v2, v3 = T.axis.remap("SSSS", [ax0_0_1_ax1_0_1_fused, ax0_0_0_ax1_0_0_fused, ax2_1, ax3])
                                v4_o = T.axis.spatial(1, 0)
                                v5_o = T.axis.spatial(1, 0)
                                T.reads(conv2d_nhwc_reindex_shared_dyn_wmma_accumulator[v0, v1, v2, v3, 0:16, 0:16])
                                T.writes(conv2d_nhwc_reindex_shared_dyn[v0, v1, v2, v3, 0:16, 0:16])
                                T.block_attr({"meta_schedule.auto_tensorize": f"wmma_store_16x16x16_f32_{intrin_suffix}"})
                                for ax4, ax5 in T.grid(16, 16):
                                    with T.block("conv2d_nhwc_reindex_shared.dyn_wmma.accumulator"):
                                        v4_i, v5_i = T.axis.remap("SS", [ax4, ax5])
                                        T.reads(conv2d_nhwc_reindex_shared_dyn_wmma_accumulator[v0, v1, v2, v3, v4_i, v5_i])
                                        T.writes(conv2d_nhwc_reindex_shared_dyn[v0, v1, v2, v3, v4_i, v5_i])
                                        conv2d_nhwc_reindex_shared_dyn[v0, v1, v2, v3, v4_i, v5_i] = conv2d_nhwc_reindex_shared_dyn_wmma_accumulator[v0, v1, v2, v3, v4_i, v5_i]
                    for ax0_ax1_ax3_ax4_ax5_fused in range(256):
                        with T.block("conv2d_nhwc_reindex_shared.dyn"):
                            v0, v1, v2 = T.axis.remap("SSS", [ax0_0_1_ax1_0_1_fused, ax0_0_0_ax1_0_0_fused, ax2])
                            v3 = T.axis.spatial(1, 0)
                            v4 = T.axis.spatial(16, ax0_ax1_ax3_ax4_ax5_fused // 16)
                            v5 = T.axis.spatial(16, ax0_ax1_ax3_ax4_ax5_fused % 16)
                            T.reads(conv2d_nhwc_reindex_shared_dyn[v0, v1, v2, v3, v4, v5])
                            T.writes(conv2d_nhwc[0, (v4 + v0 * 16) // 16, (v4 + v0 * 16) % 16, v5 + v1 * 16])
                            T.block_attr({"meta_schedule.cooperative_fetch": 3})
                            conv2d_nhwc[0, (v4 + v0 * 16) // 16, (v4 + v0 * 16) % 16, v5 + v1 * 16] = conv2d_nhwc_reindex_shared_dyn[v0, v1, v2, v3, v4, v5]
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
        target=tvm.target.Target("cuda --arch=sm_70"),
        types=None,
        sch_rules=[
            multi_level_tiling_tensor_core(
                read_reuse_scope=shared_scope, write_reuse_scope=shared_scope
            ),
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
        target=tvm.target.Target("cuda --arch=sm_70"),
        types=None,
        sch_rules=[
            multi_level_tiling_tensor_core(
                read_reuse_scope=shared_scope,
                write_reuse_scope=shared_scope,
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


@pytest.mark.parametrize("shared_scope", ["shared", "shared.dyn"])
def test_matmul_relu_pipeline(shared_scope):
    intrin_suffix = shared_scope.replace(".", "_")
    # fmt: off
    @T.prim_func
    def matmul_relu_pipeline_0(A: T.Buffer((128, 128), "float16"), B: T.Buffer((128, 128), "float16"), compute: T.Buffer((128, 128), "float32")) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # body
        # with T.block("root")
        C = T.alloc_buffer((128, 128))
        C_reindex_shared = T.alloc_buffer((4, 4, 2, 2, 16, 16), scope=shared_scope)
        C_reindex_shared_wmma_accumulator = T.alloc_buffer((4, 4, 2, 2, 16, 16), scope="wmma.accumulator")
        A_reindex_shared = T.alloc_buffer((128, 128), "float16", scope=shared_scope)
        B_reindex_shared = T.alloc_buffer((128, 128), "float16", scope=shared_scope)
        A_reindex_shared_wmma_matrix_a = T.alloc_buffer((128, 128), "float16", scope="wmma.matrix_a")
        B_reindex_shared_wmma_matrix_b = T.alloc_buffer((128, 128), "float16", scope="wmma.matrix_b")
        for ax0_0_0_ax1_0_0_fused in T.thread_binding(1, thread="blockIdx.y"):
            for ax0_0_1_ax1_0_1_fused in T.thread_binding(16, thread="blockIdx.x"):
                for ax0_0_2_ax1_0_2_fused in T.thread_binding(1, thread="threadIdx.y"):
                    for ax2_0_0 in T.serial(4, annotations={"software_pipeline_order": [0, 3, 1, 4, 5, 2, 6], "software_pipeline_stage": [0, 0, 0, 0, 0, 1, 1]}):
                        for ax0_ax1_fused in range(1024):
                            with T.block("A_reindex_shared"):
                                v0 = T.axis.spatial(128, ax0_0_1_ax1_0_1_fused // 4 * 32 + ax0_ax1_fused // 32)
                                v1 = T.axis.spatial(128, ax2_0_0 * 32 + ax0_ax1_fused % 32)
                                T.reads(A[v0, v1])
                                T.writes(A_reindex_shared[v0, v1])
                                T.block_attr({"buffer_dim_align": [[0, 0, 32, 8]], "double_buffer_scope": 0, "meta_schedule.cooperative_fetch": 4, "tir.manifest_shared_memory_local_stage": 1})
                                A_reindex_shared[v0, v1] = A[v0, v1]
                        for ax0_ax1_fused in range(1024):
                            with T.block("B_reindex_shared"):
                                v0 = T.axis.spatial(128, ax2_0_0 * 32 + ax0_ax1_fused // 32)
                                v1 = T.axis.spatial(128, ax0_0_1_ax1_0_1_fused % 4 * 32 + ax0_ax1_fused % 32)
                                T.reads(B[v0, v1])
                                T.writes(B_reindex_shared[v0, v1])
                                T.block_attr({"buffer_dim_align": [[0, 0, 32, 8]], "double_buffer_scope": 0, "meta_schedule.cooperative_fetch": 2, "tir.manifest_shared_memory_local_stage": 1})
                                B_reindex_shared[v0, v1] = B[v0, v1]
                        for ax2_0_1 in T.serial(2, annotations={"software_pipeline_order": [0, 1, 2], "software_pipeline_stage": [0, 0, 1]}):
                            for ax0_0, ax1_0 in T.grid(2, 1):
                                with T.block("A_reindex_shared_wmma.matrix_a_o"):
                                    v0_o = T.axis.spatial(8, ax0_0_1_ax1_0_1_fused // 4 * 2 + ax0_0)
                                    v1_o = T.axis.spatial(8, ax2_0_0 * 2 + ax2_0_1 + ax1_0)
                                    T.reads(A_reindex_shared[v0_o * 16:v0_o * 16 + 16, v1_o * 16:v1_o * 16 + 16])
                                    T.writes(A_reindex_shared_wmma_matrix_a[v0_o * 16:v0_o * 16 + 16, v1_o * 16:v1_o * 16 + 16])
                                    T.block_attr({"meta_schedule.auto_tensorize": f"wmma_load_16x16x16_f16_a_{intrin_suffix}"})
                                    for ax0_1, ax1_1 in T.grid(16, 16):
                                        with T.block("A_reindex_shared_wmma.matrix_a"):
                                            v0_i, v1_i = T.axis.remap("SS", [ax0_1, ax1_1])
                                            T.reads(A_reindex_shared[v0_o * 16 + v0_i, v1_o * 16 + v1_i])
                                            T.writes(A_reindex_shared_wmma_matrix_a[v0_o * 16 + v0_i, v1_o * 16 + v1_i])
                                            A_reindex_shared_wmma_matrix_a[v0_o * 16 + v0_i, v1_o * 16 + v1_i] = A_reindex_shared[v0_o * 16 + v0_i, v1_o * 16 + v1_i]
                            for ax0_0, ax1_0 in T.grid(1, 2):
                                with T.block("B_reindex_shared_wmma.matrix_b_o"):
                                    v0_o = T.axis.spatial(8, ax2_0_0 * 2 + ax2_0_1 + ax0_0)
                                    v1_o = T.axis.spatial(8, ax0_0_1_ax1_0_1_fused % 4 * 2 + ax1_0)
                                    T.reads(B_reindex_shared[v0_o * 16:v0_o * 16 + 16, v1_o * 16:v1_o * 16 + 16])
                                    T.writes(B_reindex_shared_wmma_matrix_b[v0_o * 16:v0_o * 16 + 16, v1_o * 16:v1_o * 16 + 16])
                                    T.block_attr({"meta_schedule.auto_tensorize": f"wmma_load_16x16x16_f16_b_{intrin_suffix}"})
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
                                    T.reads(A_reindex_shared_wmma_matrix_a[v0_o * 16:v0_o * 16 + 16, v2_o * 16:v2_o * 16 + 16], B_reindex_shared_wmma_matrix_b[v2_o * 16:v2_o * 16 + 16, v1_o * 16:v1_o * 16 + 16])
                                    T.writes(C_reindex_shared_wmma_accumulator[v0_o // 2, v1_o // 2, v0_o % 2, v1_o % 2, 0:16, 0:16])
                                    T.block_attr({"meta_schedule.auto_tensorize": "wmma_sync_16x16x16_f16f16f32", "meta_schedule.auto_tensorize_init": "wmma_fill_16x16x16_f32", "warp_execution": 1})
                                    with T.init():
                                        for ax0_1, ax1_1 in T.grid(16, 16):
                                            with T.block("C_init"):
                                                v0_i_init, v1_i_init = T.axis.remap("SS", [ax0_1, ax1_1])
                                                T.reads()
                                                T.writes(C_reindex_shared_wmma_accumulator[v0_o // 2, v1_o // 2, v0_o % 2, v1_o % 2, v0_i_init, v1_i_init])
                                                C_reindex_shared_wmma_accumulator[v0_o // 2, v1_o // 2, v0_o % 2, v1_o % 2, v0_i_init, v1_i_init] = T.float32(0)
                                    for ax0_1, ax1_1, ax2_1 in T.grid(16, 16, 16):
                                        with T.block("C"):
                                            v0_i, v1_i, v2_i = T.axis.remap("SSR", [ax0_1, ax1_1, ax2_1])
                                            T.reads(C_reindex_shared_wmma_accumulator[v0_o // 2, v1_o // 2, v0_o % 2, v1_o % 2, v0_i, v1_i], A_reindex_shared_wmma_matrix_a[v0_o * 16 + v0_i, v2_o * 16 + v2_i], B_reindex_shared_wmma_matrix_b[v2_o * 16 + v2_i, v1_o * 16 + v1_i])
                                            T.writes(C_reindex_shared_wmma_accumulator[v0_o // 2, v1_o // 2, v0_o % 2, v1_o % 2, v0_i, v1_i])
                                            T.block_attr({"meta_schedule.tiling_structure": "SSSRRSRS"})
                                            C_reindex_shared_wmma_accumulator[v0_o // 2, v1_o // 2, v0_o % 2, v1_o % 2, v0_i, v1_i] = C_reindex_shared_wmma_accumulator[v0_o // 2, v1_o // 2, v0_o % 2, v1_o % 2, v0_i, v1_i] + T.Cast("float32", A_reindex_shared_wmma_matrix_a[v0_o * 16 + v0_i, v2_o * 16 + v2_i]) * T.Cast("float32", B_reindex_shared_wmma_matrix_b[v2_o * 16 + v2_i, v1_o * 16 + v1_i])
                for ax2 in range(2):
                    for ax0_ax1_fused in T.thread_binding(1, thread="threadIdx.y"):
                        for ax2_1, ax3 in T.grid(1, 2):
                            with T.block("C_reindex_shared_wmma.accumulator_o"):
                                v0 = T.axis.spatial(4, ax0_0_1_ax1_0_1_fused // 4)
                                v1 = T.axis.spatial(4, ax0_0_1_ax1_0_1_fused % 4)
                                v2 = T.axis.spatial(2, ax2 + ax2_1)
                                v3 = T.axis.spatial(2, ax3)
                                v4_o = T.axis.spatial(1, 0)
                                v5_o = T.axis.spatial(1, 0)
                                T.reads(C_reindex_shared_wmma_accumulator[v0, v1, v2, v3, 0:16, 0:16])
                                T.writes(C_reindex_shared[v0, v1, v2, v3, 0:16, 0:16])
                                T.block_attr({"meta_schedule.auto_tensorize": f"wmma_store_16x16x16_f32_{intrin_suffix}"})
                                for ax4, ax5 in T.grid(16, 16):
                                    with T.block("C_reindex_shared_wmma.accumulator"):
                                        v4_i, v5_i = T.axis.remap("SS", [ax4, ax5])
                                        T.reads(C_reindex_shared_wmma_accumulator[v0, v1, v2, v3, v4_i, v5_i])
                                        T.writes(C_reindex_shared[v0, v1, v2, v3, v4_i, v5_i])
                                        C_reindex_shared[v0, v1, v2, v3, v4_i, v5_i] = C_reindex_shared_wmma_accumulator[v0, v1, v2, v3, v4_i, v5_i]
                    for ax0_ax1_ax3_ax4_ax5_fused in range(512):
                        with T.block("C_reindex_shared"):
                            v0 = T.axis.spatial(4, ax0_0_1_ax1_0_1_fused // 4)
                            v1 = T.axis.spatial(4, ax0_0_1_ax1_0_1_fused % 4)
                            v2 = T.axis.spatial(2, ax2)
                            v3 = T.axis.spatial(2, ax0_ax1_ax3_ax4_ax5_fused // 256)
                            v4 = T.axis.spatial(16, ax0_ax1_ax3_ax4_ax5_fused % 256 // 16)
                            v5 = T.axis.spatial(16, ax0_ax1_ax3_ax4_ax5_fused % 16)
                            T.reads(C_reindex_shared[v0, v1, v2, v3, v4, v5])
                            T.writes(C[v4 + v2 * 16 + v0 * 32, v5 + v3 * 16 + v1 * 32])
                            T.block_attr({"meta_schedule.cooperative_fetch": 3})
                            C[v4 + v2 * 16 + v0 * 32, v5 + v3 * 16 + v1 * 32] = C_reindex_shared[v0, v1, v2, v3, v4, v5]
        for i0, i1 in T.grid(128, 128):
            with T.block("compute"):
                v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                T.reads(C[v_i0, v_i1])
                T.writes(compute[v_i0, v_i1])
                compute[v_i0, v_i1] = T.max(C[v_i0, v_i1], T.float32(0))

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
        target=tvm.target.Target("cuda --arch=sm_70"),
        types=None,
        sch_rules=[
            multi_level_tiling_tensor_core(
                read_reuse_scope=shared_scope,
                write_reuse_scope=shared_scope,
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
        target=tvm.target.Target("cuda --arch=sm_70"),
        types=None,
        sch_rules=[multi_level_tiling_tensor_core(write_reuse_scope="shared")]
        + get_rules("cuda", ms.schedule_rule.AutoInline),
    )
    tvm.ir.assert_structural_equal(mod, sch.mod["main"])


def test_padded_matmul_relu():
    # fmt: off
    @T.prim_func
    def padded_matmul_relu_0(A: T.Buffer((127, 127), "float16"), B: T.Buffer((127, 127), "float16"), compute: T.Buffer((127, 127), "float32")) -> None:
        T.func_attr({"global_symbol": "main", "tir.noalias": T.bool(True)})
        C_reindex_shared = T.alloc_buffer((4, 8, 2, 1, 16, 16), scope="shared")
        C_reindex_shared_wmma_accumulator = T.alloc_buffer((4, 8, 2, 1, 16, 16), scope="wmma.accumulator")
        A_reindex_shared = T.alloc_buffer((128, 128), "float16", scope="shared")
        B_reindex_shared = T.alloc_buffer((128, 128), "float16", scope="shared")
        A_reindex_shared_wmma_matrix_a = T.alloc_buffer((128, 128), "float16", scope="wmma.matrix_a")
        B_reindex_shared_wmma_matrix_b = T.alloc_buffer((128, 128), "float16", scope="wmma.matrix_b")
        for ax0_0_0_ax1_0_0_fused in T.thread_binding(8, thread="blockIdx.y"):
            for ax0_0_1_ax1_0_1_fused in T.thread_binding(2, thread="blockIdx.x"):
                for ax0_0_2_ax1_0_2_fused in T.thread_binding(2, thread="threadIdx.y"):
                    for ax2_0_0 in range(1):
                        for ax0_ax1_fused in range(4096):
                            with T.block("A_reindex_shared"):
                                v0 = T.axis.spatial(128, ax0_0_0_ax1_0_0_fused // 2 * 32 + ax0_ax1_fused // 128)
                                v1 = T.axis.spatial(128, ax0_ax1_fused % 128)
                                T.reads(A[v0, v1])
                                T.writes(A_reindex_shared[v0, v1])
                                T.block_attr({"buffer_dim_align": [[0, 0, 32, 8]], "meta_schedule.cooperative_fetch": 8})
                                A_reindex_shared[v0, v1] = T.if_then_else(v0 < 127 and v1 < 127, A[v0, v1], T.float16(0))
                        for ax0_ax1_fused in range(4096):
                            with T.block("B_reindex_shared"):
                                v0 = T.axis.spatial(128, ax0_ax1_fused // 32)
                                v1 = T.axis.spatial(128, ax0_0_0_ax1_0_0_fused % 2 * 64 + ax0_0_1_ax1_0_1_fused * 32 + ax0_ax1_fused % 32)
                                T.reads(B[v0, v1])
                                T.writes(B_reindex_shared[v0, v1])
                                T.block_attr({"buffer_dim_align": [[0, 0, 32, 8]], "meta_schedule.cooperative_fetch": 1})
                                B_reindex_shared[v0, v1] = T.if_then_else(v0 < 127 and v1 < 127, B[v0, v1], T.float16(0))
                        for ax2_0_1 in range(4):
                            for ax0_0, ax1_0 in T.grid(2, 2):
                                with T.block("A_reindex_shared_wmma.matrix_a_o"):
                                    v0_o = T.axis.spatial(8, ax0_0_0_ax1_0_0_fused // 2 * 2 + ax0_0)
                                    v1_o = T.axis.spatial(8, ax2_0_1 * 2 + ax1_0)
                                    T.reads(A_reindex_shared[v0_o * 16:v0_o * 16 + 16, v1_o * 16:v1_o * 16 + 16])
                                    T.writes(A_reindex_shared_wmma_matrix_a[v0_o * 16:v0_o * 16 + 16, v1_o * 16:v1_o * 16 + 16])
                                    T.block_attr({"meta_schedule.auto_tensorize": "wmma_load_16x16x16_f16_a_shared"})
                                    for ax0_1, ax1_1 in T.grid(16, 16):
                                        with T.block("A_reindex_shared_wmma.matrix_a"):
                                            v0_i, v1_i = T.axis.remap("SS", [ax0_1, ax1_1])
                                            T.reads(A_reindex_shared[v0_o * 16 + v0_i, v1_o * 16 + v1_i])
                                            T.writes(A_reindex_shared_wmma_matrix_a[v0_o * 16 + v0_i, v1_o * 16 + v1_i])
                                            A_reindex_shared_wmma_matrix_a[v0_o * 16 + v0_i, v1_o * 16 + v1_i] = A_reindex_shared[v0_o * 16 + v0_i, v1_o * 16 + v1_i]
                            for ax0_0, ax1_0 in T.grid(2, 1):
                                with T.block("B_reindex_shared_wmma.matrix_b_o"):
                                    v0_o = T.axis.spatial(8, ax2_0_1 * 2 + ax0_0)
                                    v1_o = T.axis.spatial(8, ax0_0_0_ax1_0_0_fused % 2 * 4 + ax0_0_1_ax1_0_1_fused * 2 + ax0_0_2_ax1_0_2_fused + ax1_0)
                                    T.reads(B_reindex_shared[v0_o * 16:v0_o * 16 + 16, v1_o * 16:v1_o * 16 + 16])
                                    T.writes(B_reindex_shared_wmma_matrix_b[v0_o * 16:v0_o * 16 + 16, v1_o * 16:v1_o * 16 + 16])
                                    T.block_attr({"meta_schedule.auto_tensorize": "wmma_load_16x16x16_f16_b_shared"})
                                    for ax0_1, ax1_1 in T.grid(16, 16):
                                        with T.block("B_reindex_shared_wmma.matrix_b"):
                                            v0_i, v1_i = T.axis.remap("SS", [ax0_1, ax1_1])
                                            T.reads(B_reindex_shared[v0_o * 16 + v0_i, v1_o * 16 + v1_i])
                                            T.writes(B_reindex_shared_wmma_matrix_b[v0_o * 16 + v0_i, v1_o * 16 + v1_i])
                                            B_reindex_shared_wmma_matrix_b[v0_o * 16 + v0_i, v1_o * 16 + v1_i] = B_reindex_shared[v0_o * 16 + v0_i, v1_o * 16 + v1_i]
                            for ax0_0_3, ax1_0_3, ax2_0_2, ax0_0_4, ax1_0_4 in T.grid(1, 1, 2, 2, 1):
                                with T.block("C_o"):
                                    v0_o = T.axis.spatial(8, ax0_0_0_ax1_0_0_fused // 2 * 2 + ax0_0_3 * 2 + ax0_0_4)
                                    v1_o = T.axis.spatial(8, ax0_0_0_ax1_0_0_fused % 2 * 4 + ax0_0_1_ax1_0_1_fused * 2 + ax0_0_2_ax1_0_2_fused + ax1_0_3 + ax1_0_4)
                                    v2_o = T.axis.reduce(8, ax2_0_0 * 8 + ax2_0_1 * 2 + ax2_0_2)
                                    T.reads(A_reindex_shared_wmma_matrix_a[v0_o * 16:v0_o * 16 + 16, v2_o * 16:v2_o * 16 + 16], B_reindex_shared_wmma_matrix_b[v2_o * 16:v2_o * 16 + 16, v1_o * 16:v1_o * 16 + 16])
                                    T.writes(C_reindex_shared_wmma_accumulator[v0_o // 2, v1_o, v0_o % 2, 0, 0:16, 0:16])
                                    T.block_attr({"meta_schedule.auto_tensorize": "wmma_sync_16x16x16_f16f16f32", "meta_schedule.auto_tensorize_init": "wmma_fill_16x16x16_f32", "warp_execution": 1})
                                    with T.init():
                                        for ax0_1, ax1_1 in T.grid(16, 16):
                                            with T.block("C_init"):
                                                v0_i_init, v1_i_init = T.axis.remap("SS", [ax0_1, ax1_1])
                                                T.reads()
                                                T.writes(C_reindex_shared_wmma_accumulator[v0_o // 2, v1_o, v0_o % 2, 0, v0_i_init, v1_i_init])
                                                C_reindex_shared_wmma_accumulator[v0_o // 2, v1_o, v0_o % 2, 0, v0_i_init, v1_i_init] = T.float32(0)
                                    for ax0_1, ax1_1, ax2_1 in T.grid(16, 16, 16):
                                        with T.block("C"):
                                            v0_i, v1_i, v2_i = T.axis.remap("SSR", [ax0_1, ax1_1, ax2_1])
                                            T.reads(C_reindex_shared_wmma_accumulator[v0_o // 2, v1_o, v0_o % 2, 0, v0_i, v1_i], A_reindex_shared_wmma_matrix_a[v0_o * 16 + v0_i, v2_o * 16 + v2_i], B_reindex_shared_wmma_matrix_b[v2_o * 16 + v2_i, v1_o * 16 + v1_i])
                                            T.writes(C_reindex_shared_wmma_accumulator[v0_o // 2, v1_o, v0_o % 2, 0, v0_i, v1_i])
                                            T.block_attr({"meta_schedule.tiling_structure": "SSSRRSRS"})
                                            C_reindex_shared_wmma_accumulator[v0_o // 2, v1_o, v0_o % 2, 0, v0_i, v1_i] = C_reindex_shared_wmma_accumulator[v0_o // 2, v1_o, v0_o % 2, 0, v0_i, v1_i] + T.Cast("float32", A_reindex_shared_wmma_matrix_a[v0_o * 16 + v0_i, v2_o * 16 + v2_i]) * T.Cast("float32", B_reindex_shared_wmma_matrix_b[v2_o * 16 + v2_i, v1_o * 16 + v1_i])
                for ax2 in range(2):
                    for ax0_ax1_fused in T.thread_binding(2, thread="threadIdx.y"):
                        for ax2_1, ax3 in T.grid(1, 1):
                            with T.block("C_reindex_shared_wmma.accumulator_o"):
                                v0 = T.axis.spatial(4, ax0_0_0_ax1_0_0_fused // 2)
                                v1 = T.axis.spatial(8, ax0_0_0_ax1_0_0_fused % 2 * 4 + ax0_0_1_ax1_0_1_fused * 2 + ax0_ax1_fused)
                                v2 = T.axis.spatial(2, ax2 + ax2_1)
                                v3 = T.axis.spatial(1, ax3)
                                v4_o = T.axis.spatial(1, 0)
                                v5_o = T.axis.spatial(1, 0)
                                T.reads(C_reindex_shared_wmma_accumulator[v0, v1, v2, v3, 0:16, 0:16])
                                T.writes(C_reindex_shared[v0, v1, v2, v3, 0:16, 0:16])
                                T.block_attr({"meta_schedule.auto_tensorize": "wmma_store_16x16x16_f32_shared"})
                                for ax4, ax5 in T.grid(16, 16):
                                    with T.block("C_reindex_shared_wmma.accumulator"):
                                        v4_i, v5_i = T.axis.remap("SS", [ax4, ax5])
                                        T.reads(C_reindex_shared_wmma_accumulator[v0, v1, v2, v3, v4_i, v5_i])
                                        T.writes(C_reindex_shared[v0, v1, v2, v3, v4_i, v5_i])
                                        C_reindex_shared[v0, v1, v2, v3, v4_i, v5_i] = C_reindex_shared_wmma_accumulator[v0, v1, v2, v3, v4_i, v5_i]
                    for ax0_ax1_ax3_ax4_ax5_fused in range(512):
                        with T.block("C_reindex_shared"):
                            v0 = T.axis.spatial(4, ax0_0_0_ax1_0_0_fused // 2)
                            v1 = T.axis.spatial(8, ax0_0_0_ax1_0_0_fused % 2 * 4 + ax0_0_1_ax1_0_1_fused * 2 + ax0_ax1_ax3_ax4_ax5_fused // 256)
                            v2 = T.axis.spatial(2, ax2)
                            v3 = T.axis.spatial(1, 0)
                            v4 = T.axis.spatial(16, ax0_ax1_ax3_ax4_ax5_fused % 256 // 16)
                            v5 = T.axis.spatial(16, ax0_ax1_ax3_ax4_ax5_fused % 16)
                            T.where(ax0_0_0_ax1_0_0_fused // 2 * 32 + ax2 * 16 + ax0_ax1_ax3_ax4_ax5_fused % 256 // 16 < 127 and ax0_0_0_ax1_0_0_fused % 2 * 64 + ax0_0_1_ax1_0_1_fused * 32 + ax0_ax1_ax3_ax4_ax5_fused // 256 * 16 + ax0_ax1_ax3_ax4_ax5_fused % 16 < 127)
                            T.reads(C_reindex_shared[v0, v1, v2, v3, v4, v5])
                            T.writes(compute[v4 + v2 * 16 + v0 * 32, v5 + v1 * 16])
                            T.block_attr({"meta_schedule.cooperative_fetch": 4})
                            compute[v4 + v2 * 16 + v0 * 32, v5 + v1 * 16] = T.max(C_reindex_shared[v0, v1, v2, v3, v4, v5], T.float32(0))
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
        target=tvm.target.Target("cuda --arch=sm_70"),
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
    def conv2d_1x1_0(inputs: T.Buffer((1, 16, 16, 64), "float16"), weight: T.Buffer((1, 1, 64, 64), "float16"), conv2d_nhwc: T.Buffer((1, 16, 16, 64), "float32")):
        T.func_attr({"global_symbol": "main", "tir.noalias": T.bool(True)})
        # with T.block("root"):
        conv2d_nhwc_reindex_shared = T.alloc_buffer((2, 1, 8, 4, 16, 16), scope="shared")
        conv2d_nhwc_reindex_shared_wmma_accumulator = T.alloc_buffer((2, 1, 8, 4, 16, 16), scope="wmma.accumulator")
        PadInput_reindex_shared = T.alloc_buffer((256, 64), "float16", scope="shared")
        weight_reindex_shared = T.alloc_buffer((1, 1, 64, 64), "float16", scope="shared")
        PadInput_reindex_shared_wmma_matrix_a = T.alloc_buffer((256, 64), "float16", scope="wmma.matrix_a")
        weight_reindex_shared_wmma_matrix_b = T.alloc_buffer((1, 1, 64, 64), "float16", scope="wmma.matrix_b")
        for ax0_ax1_ax2_0_0_ax3_0_0_fused in T.thread_binding(1, thread="blockIdx.y"):
            for ax2_0_1_ax3_0_1_fused in T.thread_binding(1, thread="blockIdx.x"):
                for ax2_0_2_ax3_0_2_fused in T.thread_binding(2, thread="threadIdx.y"):
                    for ax4_0_0 in range(2):
                        for ax0_ax1_fused in range(8192):
                            with T.block("PadInput_reindex_shared"):
                                v0 = T.axis.spatial(256, ax0_ax1_fused // 32)
                                v1 = T.axis.spatial(64, ax4_0_0 * 32 + ax0_ax1_fused % 32)
                                T.reads(inputs[0, v0 // 16, v0 % 16, v1])
                                T.writes(PadInput_reindex_shared[v0, v1])
                                T.block_attr({"buffer_dim_align": [[0, 0, 32, 8]], "meta_schedule.cooperative_fetch": 8})
                                PadInput_reindex_shared[v0, v1] = inputs[0, v0 // 16, v0 % 16, v1]
                        for ax0_ax1_ax2_ax3_fused in range(2048):
                            with T.block("weight_reindex_shared"):
                                v0 = T.axis.spatial(1, 0)
                                v1 = T.axis.spatial(1, 0)
                                v2 = T.axis.spatial(64, ax4_0_0 * 32 + ax0_ax1_ax2_ax3_fused // 64)
                                v3 = T.axis.spatial(64, ax0_ax1_ax2_ax3_fused % 64)
                                T.reads(weight[v0, v1, v2, v3])
                                T.writes(weight_reindex_shared[v0, v1, v2, v3])
                                T.block_attr({"buffer_dim_align": [[0, 2, 32, 8]], "meta_schedule.cooperative_fetch": 4})
                                weight_reindex_shared[v0, v1, v2, v3] = weight[v0, v1, v2, v3]
                        for ax4_0_1 in range(1):
                            for ax0_0, ax1_0 in T.grid(8, 2):
                                with T.block("PadInput_reindex_shared_wmma.matrix_a_o"):
                                    v0_o = T.axis.spatial(16, ax2_0_2_ax3_0_2_fused * 8 + ax0_0)
                                    v1_o = T.axis.spatial(4, ax4_0_0 * 2 + ax1_0)
                                    T.reads(PadInput_reindex_shared[v0_o * 16:v0_o * 16 + 16, v1_o * 16:v1_o * 16 + 16])
                                    T.writes(PadInput_reindex_shared_wmma_matrix_a[v0_o * 16:v0_o * 16 + 16, v1_o * 16:v1_o * 16 + 16])
                                    T.block_attr({"meta_schedule.auto_tensorize": "wmma_load_16x16x16_f16_a_shared"})
                                    for ax0_1, ax1_1 in T.grid(16, 16):
                                        with T.block("PadInput_reindex_shared_wmma.matrix_a"):
                                            v0_i, v1_i = T.axis.remap("SS", [ax0_1, ax1_1])
                                            T.reads(PadInput_reindex_shared[v0_o * 16 + v0_i, v1_o * 16 + v1_i])
                                            T.writes(PadInput_reindex_shared_wmma_matrix_a[v0_o * 16 + v0_i, v1_o * 16 + v1_i])
                                            PadInput_reindex_shared_wmma_matrix_a[v0_o * 16 + v0_i, v1_o * 16 + v1_i] = PadInput_reindex_shared[v0_o * 16 + v0_i, v1_o * 16 + v1_i]
                            for ax0, ax1, ax2_0, ax3_0 in T.grid(1, 1, 2, 4):
                                with T.block("weight_reindex_shared_wmma.matrix_b_o"):
                                    v0_o, v1_o = T.axis.remap("SS", [ax0, ax1])
                                    v2_o = T.axis.spatial(4, ax4_0_0 * 2 + ax2_0)
                                    v3_o = T.axis.spatial(4, ax3_0)
                                    T.reads(weight_reindex_shared[v0_o, v1_o, v2_o * 16:v2_o * 16 + 16, v3_o * 16:v3_o * 16 + 16])
                                    T.writes(weight_reindex_shared_wmma_matrix_b[v0_o, v1_o, v2_o * 16:v2_o * 16 + 16, v3_o * 16:v3_o * 16 + 16])
                                    T.block_attr({"meta_schedule.auto_tensorize": "wmma_load_16x16x16_f16_b_shared"})
                                    for ax2_1, ax3_1 in T.grid(16, 16):
                                        with T.block("weight_reindex_shared_wmma.matrix_b"):
                                            v2_i, v3_i = T.axis.remap("SS", [ax2_1, ax3_1])
                                            T.reads(weight_reindex_shared[v0_o, v1_o, v2_o * 16 + v2_i, v3_o * 16 + v3_i])
                                            T.writes(weight_reindex_shared_wmma_matrix_b[v0_o, v1_o, v2_o * 16 + v2_i, v3_o * 16 + v3_i])
                                            weight_reindex_shared_wmma_matrix_b[v0_o, v1_o, v2_o * 16 + v2_i, v3_o * 16 + v3_i] = weight_reindex_shared[v0_o, v1_o, v2_o * 16 + v2_i, v3_o * 16 + v3_i]
                            for ax2_0_3, ax3_0_3, ax4_0_2, ax2_0_4, ax3_0_4 in T.grid(8, 1, 2, 1, 4):
                                with T.block("conv2d_nhwc_o"):
                                    v0_o = T.axis.spatial(1, 0)
                                    v1_o = T.axis.spatial(1, 0)
                                    v2_o = T.axis.spatial(16, ax2_0_2_ax3_0_2_fused * 8 + ax2_0_3 + ax2_0_4)
                                    v3_o = T.axis.spatial(4, ax3_0_3 * 4 + ax3_0_4)
                                    v4_o = T.axis.reduce(4, ax4_0_0 * 2 + ax4_0_1 * 2 + ax4_0_2)
                                    T.reads(PadInput_reindex_shared_wmma_matrix_a[v2_o * 16:v2_o * 16 + 16, v4_o * 16:v4_o * 16 + 16], weight_reindex_shared_wmma_matrix_b[v0_o, v1_o, v4_o * 16:v4_o * 16 + 16, v3_o * 16:v3_o * 16 + 16])
                                    T.writes(conv2d_nhwc_reindex_shared_wmma_accumulator[v2_o // 8, 0, v2_o % 8, v3_o, 0:16, 0:16])
                                    T.block_attr({"meta_schedule.auto_tensorize": "wmma_sync_16x16x16_f16f16f32", "meta_schedule.auto_tensorize_init": "wmma_fill_16x16x16_f32", "warp_execution": 1})
                                    with T.init():
                                        for ax2_1, ax3_1 in T.grid(16, 16):
                                            with T.block("conv2d_nhwc_init"):
                                                v2_i_init, v3_i_init = T.axis.remap("SS", [ax2_1, ax3_1])
                                                T.reads()
                                                T.writes(conv2d_nhwc_reindex_shared_wmma_accumulator[v2_o // 8, 0, v2_o % 8, v3_o, v2_i_init, v3_i_init])
                                                conv2d_nhwc_reindex_shared_wmma_accumulator[v2_o // 8, 0, v2_o % 8, v3_o, v2_i_init, v3_i_init] = T.float32(0)
                                    for ax2_1, ax3_1, ax4_1 in T.grid(16, 16, 16):
                                        with T.block("conv2d_nhwc"):
                                            v2_i, v3_i, v4_i = T.axis.remap("SSR", [ax2_1, ax3_1, ax4_1])
                                            T.reads(conv2d_nhwc_reindex_shared_wmma_accumulator[v2_o // 8, 0, v2_o % 8, v3_o, v2_i, v3_i], PadInput_reindex_shared_wmma_matrix_a[v2_o * 16 + v2_i, v4_o * 16 + v4_i], weight_reindex_shared_wmma_matrix_b[v0_o, v1_o, v4_o * 16 + v4_i, v3_o * 16 + v3_i])
                                            T.writes(conv2d_nhwc_reindex_shared_wmma_accumulator[v2_o // 8, 0, v2_o % 8, v3_o, v2_i, v3_i])
                                            T.block_attr({"meta_schedule.tiling_structure": "SSSRRSRS"})
                                            conv2d_nhwc_reindex_shared_wmma_accumulator[v2_o // 8, 0, v2_o % 8, v3_o, v2_i, v3_i] = conv2d_nhwc_reindex_shared_wmma_accumulator[v2_o // 8, 0, v2_o % 8, v3_o, v2_i, v3_i] + T.Cast("float32", PadInput_reindex_shared_wmma_matrix_a[v2_o * 16 + v2_i, v4_o * 16 + v4_i]) * T.Cast("float32", weight_reindex_shared_wmma_matrix_b[v0_o, v1_o, v4_o * 16 + v4_i, v3_o * 16 + v3_i])
                for ax2 in range(8):
                    for ax0_ax1_fused in T.thread_binding(2, thread="threadIdx.y"):
                        for ax2_1, ax3 in T.grid(1, 4):
                            with T.block("conv2d_nhwc_reindex_shared_wmma.accumulator_o"):
                                v0_o = T.axis.spatial(2, ax0_ax1_fused)
                                v1_o = T.axis.spatial(1, 0)
                                v2_o = T.axis.spatial(8, ax2 + ax2_1)
                                v3_o = T.axis.spatial(4, ax3)
                                v4_o = T.axis.spatial(1, 0)
                                v5_o = T.axis.spatial(1, 0)
                                T.reads(conv2d_nhwc_reindex_shared_wmma_accumulator[v0_o, v1_o, v2_o, v3_o, 0:16, 0:16])
                                T.writes(conv2d_nhwc_reindex_shared[v0_o, v1_o, v2_o, v3_o, 0:16, 0:16])
                                T.block_attr({"meta_schedule.auto_tensorize": "wmma_store_16x16x16_f32_shared"})
                                for ax4, ax5 in T.grid(16, 16):
                                    with T.block("conv2d_nhwc_reindex_shared_wmma.accumulator"):
                                        v4_i, v5_i = T.axis.remap("SS", [ax4, ax5])
                                        T.reads(conv2d_nhwc_reindex_shared_wmma_accumulator[v0_o, v1_o, v2_o, v3_o, v4_i, v5_i])
                                        T.writes(conv2d_nhwc_reindex_shared[v0_o, v1_o, v2_o, v3_o, v4_i, v5_i])
                                        conv2d_nhwc_reindex_shared[v0_o, v1_o, v2_o, v3_o, v4_i, v5_i] = conv2d_nhwc_reindex_shared_wmma_accumulator[v0_o, v1_o, v2_o, v3_o, v4_i, v5_i]
                    for ax0_ax1_ax3_ax4_ax5_fused in range(2048):
                        with T.block("conv2d_nhwc_reindex_shared"):
                            v0 = T.axis.spatial(2, ax0_ax1_ax3_ax4_ax5_fused // 1024)
                            v1 = T.axis.spatial(1, 0)
                            v2 = T.axis.spatial(8, ax2)
                            v3 = T.axis.spatial(4, ax0_ax1_ax3_ax4_ax5_fused % 1024 // 256)
                            v4 = T.axis.spatial(16, ax0_ax1_ax3_ax4_ax5_fused % 256 // 16)
                            v5 = T.axis.spatial(16, ax0_ax1_ax3_ax4_ax5_fused % 16)
                            T.reads(conv2d_nhwc_reindex_shared[v0, v1, v2, v3, v4, v5])
                            T.writes(conv2d_nhwc[0, (v4 + v2 * 16 + v0 * 128) // 16, (v4 + v2 * 16 + v0 * 128) % 16, v5 + v3 * 16])
                            T.block_attr({"meta_schedule.cooperative_fetch": 1})
                            conv2d_nhwc[0, (v4 + v2 * 16 + v0 * 128) // 16, (v4 + v2 * 16 + v0 * 128) % 16, v5 + v3 * 16] = conv2d_nhwc_reindex_shared[v0, v1, v2, v3, v4, v5]
    # fmt: on

    decision_0 = [
        ("SamplePerfectTile", [1, 1, 2, 8, 1]),
        ("SamplePerfectTile", [1, 1, 1, 1, 4]),
        ("SamplePerfectTile", [2, 1, 2]),
        ("SampleCategorical", 0),
        ("SampleCategorical", 3),
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
        target=tvm.target.Target("cuda --arch=sm_70"),
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


def test_padded_conv():
    # fmt: off
    @T.prim_func
    def padded_conv2d_0(inputs: T.Buffer((1, 224, 224, 3), "float16"), weight: T.Buffer((7, 7, 3, 64), "float16"), conv2d_nhwc: T.Buffer((1, 112, 112, 64), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        conv2d_nhwc_reindex_shared = T.alloc_buffer((56, 2, 14, 2, 16, 16), scope="shared")
        conv2d_nhwc_reindex_shared_wmma_accumulator = T.alloc_buffer((56, 2, 14, 2, 16, 16), scope="wmma.accumulator")
        PadInput_reindex_pad_shared = T.alloc_buffer((12544, 160), "float16", scope="shared")
        weight_reindex_pad_shared = T.alloc_buffer((160, 64), "float16", scope="shared")
        PadInput_reindex_pad_shared_wmma_matrix_a = T.alloc_buffer((12544, 160), "float16", scope="wmma.matrix_a")
        weight_reindex_pad_shared_wmma_matrix_b = T.alloc_buffer((160, 64), "float16", scope="wmma.matrix_b")
        for ax0_0_0_ax1_0_0_fused in T.thread_binding(14, thread="blockIdx.y"):
            for ax0_0_1_ax1_0_1_fused in T.thread_binding(1, thread="blockIdx.x"):
                for ax0_0_2_ax1_0_2_fused in T.thread_binding(8, thread="threadIdx.y"):
                    for ax2_0_0 in range(10):
                        for ax0_ax1_fused in range(28672):
                            with T.block("PadInput_reindex_pad_shared"):
                                v0 = T.axis.spatial(12544, ax0_0_0_ax1_0_0_fused // 2 * 1792 + ax0_ax1_fused // 16)
                                v1 = T.axis.spatial(160, ax2_0_0 * 16 + ax0_ax1_fused % 16)
                                T.reads(inputs[0, v0 // 112 * 2 + v1 // 21 - 3, v0 % 112 * 2 + v1 % 21 // 3 - 3, v1 % 3])
                                T.writes(PadInput_reindex_pad_shared[v0, v1])
                                T.block_attr({"buffer_dim_align": [[0, 0, 32, 8]], "meta_schedule.cooperative_fetch": 4})
                                PadInput_reindex_pad_shared[v0, v1] = T.if_then_else(v1 < 147, T.if_then_else(3 <= v0 // 112 * 2 + v1 // 21 and v0 // 112 * 2 + v1 // 21 < 227 and 3 <= v0 % 112 * 2 + v1 % 21 // 3 and v0 % 112 * 2 + v1 % 21 // 3 < 227, inputs[0, v0 // 112 * 2 + v1 // 21 - 3, v0 % 112 * 2 + v1 % 21 // 3 - 3, v1 % 3], T.float16(0)), T.float16(0))
                        for ax0_ax1_fused in range(512):
                            with T.block("weight_reindex_pad_shared"):
                                v0 = T.axis.spatial(160, ax2_0_0 * 16 + ax0_ax1_fused // 32)
                                v1 = T.axis.spatial(64, ax0_0_0_ax1_0_0_fused % 2 * 32 + ax0_ax1_fused % 32)
                                T.reads(weight[v0 // 21, v0 % 21 // 3, v0 % 3, v1])
                                T.writes(weight_reindex_pad_shared[v0, v1])
                                T.block_attr({"buffer_dim_align": [[0, 0, 32, 8]], "meta_schedule.cooperative_fetch": 2})
                                weight_reindex_pad_shared[v0, v1] = T.if_then_else(v0 < 147, weight[v0 // 21, v0 % 21 // 3, v0 % 3, v1], T.float16(0))
                        for ax2_0_1 in range(1):
                            for ax0_0, ax1_0 in T.grid(14, 1):
                                with T.block("PadInput_reindex_pad_shared_wmma.matrix_a_o"):
                                    v0_o = T.axis.spatial(784, ax0_0_0_ax1_0_0_fused // 2 * 112 + ax0_0_2_ax1_0_2_fused * 14 + ax0_0)
                                    v1_o = T.axis.spatial(10, ax2_0_0 + ax1_0)
                                    T.reads(PadInput_reindex_pad_shared[v0_o * 16:v0_o * 16 + 16, v1_o * 16:v1_o * 16 + 16])
                                    T.writes(PadInput_reindex_pad_shared_wmma_matrix_a[v0_o * 16:v0_o * 16 + 16, v1_o * 16:v1_o * 16 + 16])
                                    T.block_attr({"meta_schedule.auto_tensorize": "wmma_load_16x16x16_f16_a_shared"})
                                    for ax0_1, ax1_1 in T.grid(16, 16):
                                        with T.block("PadInput_reindex_pad_shared_wmma.matrix_a"):
                                            v0_i, v1_i = T.axis.remap("SS", [ax0_1, ax1_1])
                                            T.reads(PadInput_reindex_pad_shared[v0_o * 16 + v0_i, v1_o * 16 + v1_i])
                                            T.writes(PadInput_reindex_pad_shared_wmma_matrix_a[v0_o * 16 + v0_i, v1_o * 16 + v1_i])
                                            PadInput_reindex_pad_shared_wmma_matrix_a[v0_o * 16 + v0_i, v1_o * 16 + v1_i] = PadInput_reindex_pad_shared[v0_o * 16 + v0_i, v1_o * 16 + v1_i]
                            for ax0_0, ax1_0 in T.grid(1, 2):
                                with T.block("weight_reindex_pad_shared_wmma.matrix_b_o"):
                                    v0_o = T.axis.spatial(10, ax2_0_0 + ax0_0)
                                    v1_o = T.axis.spatial(4, ax0_0_0_ax1_0_0_fused % 2 * 2 + ax1_0)
                                    T.reads(weight_reindex_pad_shared[v0_o * 16:v0_o * 16 + 16, v1_o * 16:v1_o * 16 + 16])
                                    T.writes(weight_reindex_pad_shared_wmma_matrix_b[v0_o * 16:v0_o * 16 + 16, v1_o * 16:v1_o * 16 + 16])
                                    T.block_attr({"meta_schedule.auto_tensorize": "wmma_load_16x16x16_f16_b_shared"})
                                    for ax0_1, ax1_1 in T.grid(16, 16):
                                        with T.block("weight_reindex_pad_shared_wmma.matrix_b"):
                                            v0_i, v1_i = T.axis.remap("SS", [ax0_1, ax1_1])
                                            T.reads(weight_reindex_pad_shared[v0_o * 16 + v0_i, v1_o * 16 + v1_i])
                                            T.writes(weight_reindex_pad_shared_wmma_matrix_b[v0_o * 16 + v0_i, v1_o * 16 + v1_i])
                                            weight_reindex_pad_shared_wmma_matrix_b[v0_o * 16 + v0_i, v1_o * 16 + v1_i] = weight_reindex_pad_shared[v0_o * 16 + v0_i, v1_o * 16 + v1_i]
                            for ax0_0_3, ax1_0_3, ax2_0_2, ax0_0_4, ax1_0_4 in T.grid(7, 2, 1, 2, 1):
                                with T.block("conv2d_nhwc_o"):
                                    v0_o = T.axis.spatial(784, ax0_0_0_ax1_0_0_fused // 2 * 112 + ax0_0_2_ax1_0_2_fused * 14 + ax0_0_3 * 2 + ax0_0_4)
                                    v1_o = T.axis.spatial(4, ax0_0_0_ax1_0_0_fused % 2 * 2 + ax1_0_3 + ax1_0_4)
                                    v2_o = T.axis.reduce(10, ax2_0_0 + ax2_0_1 + ax2_0_2)
                                    T.reads(PadInput_reindex_pad_shared_wmma_matrix_a[v0_o * 16:v0_o * 16 + 16, v2_o * 16:v2_o * 16 + 16], weight_reindex_pad_shared_wmma_matrix_b[v2_o * 16:v2_o * 16 + 16, v1_o * 16:v1_o * 16 + 16])
                                    T.writes(conv2d_nhwc_reindex_shared_wmma_accumulator[v0_o // 14, v1_o // 2, v0_o % 14, v1_o % 2, 0:16, 0:16])
                                    T.block_attr({"meta_schedule.auto_tensorize": "wmma_sync_16x16x16_f16f16f32", "meta_schedule.auto_tensorize_init": "wmma_fill_16x16x16_f32", "warp_execution": 1})
                                    with T.init():
                                        for ax0_1, ax1_1 in T.grid(16, 16):
                                            with T.block("conv2d_nhwc_init"):
                                                v0_i_init, v1_i_init = T.axis.remap("SS", [ax0_1, ax1_1])
                                                T.reads()
                                                T.writes(conv2d_nhwc_reindex_shared_wmma_accumulator[v0_o // 14, v1_o // 2, v0_o % 14, v1_o % 2, v0_i_init, v1_i_init])
                                                conv2d_nhwc_reindex_shared_wmma_accumulator[v0_o // 14, v1_o // 2, v0_o % 14, v1_o % 2, v0_i_init, v1_i_init] = T.float32(0)
                                    for ax0_1, ax1_1, ax2_1 in T.grid(16, 16, 16):
                                        with T.block("conv2d_nhwc"):
                                            v0_i, v1_i, v2_i = T.axis.remap("SSR", [ax0_1, ax1_1, ax2_1])
                                            T.reads(conv2d_nhwc_reindex_shared_wmma_accumulator[v0_o // 14, v1_o // 2, v0_o % 14, v1_o % 2, v0_i, v1_i], PadInput_reindex_pad_shared_wmma_matrix_a[v0_o * 16 + v0_i, v2_o * 16 + v2_i], weight_reindex_pad_shared_wmma_matrix_b[v2_o * 16 + v2_i, v1_o * 16 + v1_i])
                                            T.writes(conv2d_nhwc_reindex_shared_wmma_accumulator[v0_o // 14, v1_o // 2, v0_o % 14, v1_o % 2, v0_i, v1_i])
                                            T.block_attr({"meta_schedule.tiling_structure": "SSSRRSRS"})
                                            conv2d_nhwc_reindex_shared_wmma_accumulator[v0_o // 14, v1_o // 2, v0_o % 14, v1_o % 2, v0_i, v1_i] = conv2d_nhwc_reindex_shared_wmma_accumulator[v0_o // 14, v1_o // 2, v0_o % 14, v1_o % 2, v0_i, v1_i] + T.Cast("float32", PadInput_reindex_pad_shared_wmma_matrix_a[v0_o * 16 + v0_i, v2_o * 16 + v2_i]) * T.Cast("float32", weight_reindex_pad_shared_wmma_matrix_b[v2_o * 16 + v2_i, v1_o * 16 + v1_i])
                for ax2 in range(14):
                    for ax0_ax1_fused in T.thread_binding(8, thread="threadIdx.y"):
                        for ax2_1, ax3 in T.grid(1, 2):
                            with T.block("conv2d_nhwc_reindex_shared_wmma.accumulator_o"):
                                v0_o = T.axis.spatial(56, ax0_0_0_ax1_0_0_fused // 2 * 8 + ax0_ax1_fused)
                                v1_o = T.axis.spatial(2, ax0_0_0_ax1_0_0_fused % 2)
                                v2_o = T.axis.spatial(14, ax2 + ax2_1)
                                v3_o = T.axis.spatial(2, ax3)
                                v4_o = T.axis.spatial(1, 0)
                                v5_o = T.axis.spatial(1, 0)
                                T.reads(conv2d_nhwc_reindex_shared_wmma_accumulator[v0_o, v1_o, v2_o, v3_o, 0:16, 0:16])
                                T.writes(conv2d_nhwc_reindex_shared[v0_o, v1_o, v2_o, v3_o, 0:16, 0:16])
                                T.block_attr({"meta_schedule.auto_tensorize": "wmma_store_16x16x16_f32_shared"})
                                for ax4, ax5 in T.grid(16, 16):
                                    with T.block("conv2d_nhwc_reindex_shared_wmma.accumulator"):
                                        v4_i, v5_i = T.axis.remap("SS", [ax4, ax5])
                                        T.reads(conv2d_nhwc_reindex_shared_wmma_accumulator[v0_o, v1_o, v2_o, v3_o, v4_i, v5_i])
                                        T.writes(conv2d_nhwc_reindex_shared[v0_o, v1_o, v2_o, v3_o, v4_i, v5_i])
                                        conv2d_nhwc_reindex_shared[v0_o, v1_o, v2_o, v3_o, v4_i, v5_i] = conv2d_nhwc_reindex_shared_wmma_accumulator[v0_o, v1_o, v2_o, v3_o, v4_i, v5_i]
                    for ax0_ax1_ax3_ax4_ax5_fused in range(4096):
                        with T.block("conv2d_nhwc_reindex_shared"):
                            v0 = T.axis.spatial(56, ax0_0_0_ax1_0_0_fused // 2 * 8 + ax0_ax1_ax3_ax4_ax5_fused // 512)
                            v1 = T.axis.spatial(2, ax0_0_0_ax1_0_0_fused % 2)
                            v2 = T.axis.spatial(14, ax2)
                            v3 = T.axis.spatial(2, ax0_ax1_ax3_ax4_ax5_fused % 512 // 256)
                            v4 = T.axis.spatial(16, ax0_ax1_ax3_ax4_ax5_fused % 256 // 16)
                            v5 = T.axis.spatial(16, ax0_ax1_ax3_ax4_ax5_fused % 16)
                            T.reads(conv2d_nhwc_reindex_shared[v0, v1, v2, v3, v4, v5])
                            T.writes(conv2d_nhwc[0, (v4 + v2 * 16 + v0 * 224) // 112, (v4 + v2 * 16 + v0 * 224) % 112, v5 + v3 * 16 + v1 * 32])
                            T.block_attr({"meta_schedule.cooperative_fetch": 3})
                            conv2d_nhwc[0, (v4 + v2 * 16 + v0 * 224) // 112, (v4 + v2 * 16 + v0 * 224) % 112, v5 + v3 * 16 + v1 * 32] = conv2d_nhwc_reindex_shared[v0, v1, v2, v3, v4, v5]
    # fmt: on

    decision_0 = [
        ("SamplePerfectTile", [7, 1, 8, 7, 2]),
        ("SamplePerfectTile", [2, 1, 1, 2, 1]),
        ("SamplePerfectTile", [10, 1, 1]),
        ("SampleCategorical", 2),
        ("SampleCategorical", 2),
        ("SampleCategorical", 1),
    ]
    mod = te.create_prim_func(
        te_workload.conv2d_nhwc(
            1,
            224,
            224,
            3,
            64,
            7,
            2,
            3,
            in_dtype="float16",
            out_dtype="float32",
        )
    )
    actual = generate_design_space(
        kind="cuda",
        mod=mod,
        target=tvm.target.Target("cuda --arch=sm_70"),
        types=None,
        sch_rules=[multi_level_tiling_tensor_core(write_reuse_scope="shared")]
        + get_rules("cuda", ms.schedule_rule.AutoInline),
    )
    check_sketches(
        mod,
        sketches=actual,
        expected_mods=[padded_conv2d_0],
        expected_decisions=[decision_0],
    )


def test_padded_matmul_single_padded_input():
    # fmt: off
    @T.prim_func
    def padded_matmul_single_padded_input_0(A: T.Buffer((1023, 4096), "float16"), B: T.Buffer((4096, 1024), "float16"), C: T.Buffer((1023, 1024), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        C_reindex_pad_shared = T.alloc_buffer((8, 32, 8, 2, 16, 16), scope="shared")
        C_reindex_pad_shared_wmma_accumulator = T.alloc_buffer((8, 32, 8, 2, 16, 16), scope="wmma.accumulator")
        A_reindex_pad_shared = T.alloc_buffer((1024, 4096), "float16", scope="shared")
        B_reindex_shared = T.alloc_buffer((4096, 1024), "float16", scope="shared")
        A_reindex_pad_shared_wmma_matrix_a = T.alloc_buffer((1024, 4096), "float16", scope="wmma.matrix_a")
        B_reindex_shared_wmma_matrix_b = T.alloc_buffer((4096, 1024), "float16", scope="wmma.matrix_b")
        for ax0_0_0_ax1_0_0_fused in T.thread_binding(1, thread="blockIdx.y"):
            for ax0_0_1_ax1_0_1_fused in T.thread_binding(32, thread="blockIdx.x"):
                for ax0_0_2_ax1_0_2_fused in T.thread_binding(8, thread="threadIdx.y"):
                    for ax2_0_0 in range(32):
                        for ax0_ax1_fused in range(65536):
                            with T.block("A_reindex_pad_shared"):
                                v0 = T.axis.spatial(1024, ax0_0_1_ax1_0_1_fused // 16 * 512 + ax0_ax1_fused // 128)
                                v1 = T.axis.spatial(4096, ax2_0_0 * 128 + ax0_ax1_fused % 128)
                                T.reads(A[v0, v1])
                                T.writes(A_reindex_pad_shared[v0, v1])
                                T.block_attr({"buffer_dim_align": [[0, 0, 32, 8]], "meta_schedule.cooperative_fetch": 2})
                                A_reindex_pad_shared[v0, v1] = T.if_then_else(v0 < 1023, A[v0, v1], T.float16(0.0))
                        for ax0_ax1_fused in range(8192):
                            with T.block("B_reindex_shared"):
                                v0 = T.axis.spatial(4096, ax2_0_0 * 128 + ax0_ax1_fused // 64)
                                v1 = T.axis.spatial(1024, ax0_0_1_ax1_0_1_fused % 16 * 64 + ax0_ax1_fused % 64)
                                T.reads(B[v0, v1])
                                T.writes(B_reindex_shared[v0, v1])
                                T.block_attr({"buffer_dim_align": [[0, 0, 32, 8]], "meta_schedule.cooperative_fetch": 1})
                                B_reindex_shared[v0, v1] = B[v0, v1]
                        for ax2_0_1 in range(8):
                            for ax0_0, ax1_0 in T.grid(8, 1):
                                with T.block("A_reindex_pad_shared_wmma.matrix_a_o"):
                                    v0_o = T.axis.spatial(64, ax0_0_1_ax1_0_1_fused // 16 * 32 + ax0_0_2_ax1_0_2_fused // 2 * 8 + ax0_0)
                                    v1_o = T.axis.spatial(256, ax2_0_0 * 8 + ax2_0_1 + ax1_0)
                                    T.reads(A_reindex_pad_shared[v0_o * 16:v0_o * 16 + 16, v1_o * 16:v1_o * 16 + 16])
                                    T.writes(A_reindex_pad_shared_wmma_matrix_a[v0_o * 16:v0_o * 16 + 16, v1_o * 16:v1_o * 16 + 16])
                                    T.block_attr({"meta_schedule.auto_tensorize": "wmma_load_16x16x16_f16_a_shared"})
                                    for ax0_1, ax1_1 in T.grid(16, 16):
                                        with T.block("A_reindex_pad_shared_wmma.matrix_a"):
                                            v0_i, v1_i = T.axis.remap("SS", [ax0_1, ax1_1])
                                            T.reads(A_reindex_pad_shared[v0_o * 16 + v0_i, v1_o * 16 + v1_i])
                                            T.writes(A_reindex_pad_shared_wmma_matrix_a[v0_o * 16 + v0_i, v1_o * 16 + v1_i])
                                            A_reindex_pad_shared_wmma_matrix_a[v0_o * 16 + v0_i, v1_o * 16 + v1_i] = A_reindex_pad_shared[v0_o * 16 + v0_i, v1_o * 16 + v1_i]
                            for ax0_0, ax1_0 in T.grid(1, 2):
                                with T.block("B_reindex_shared_wmma.matrix_b_o"):
                                    v0_o = T.axis.spatial(256, ax2_0_0 * 8 + ax2_0_1 + ax0_0)
                                    v1_o = T.axis.spatial(64, ax0_0_1_ax1_0_1_fused % 16 * 4 + ax0_0_2_ax1_0_2_fused % 2 * 2 + ax1_0)
                                    T.reads(B_reindex_shared[v0_o * 16:v0_o * 16 + 16, v1_o * 16:v1_o * 16 + 16])
                                    T.writes(B_reindex_shared_wmma_matrix_b[v0_o * 16:v0_o * 16 + 16, v1_o * 16:v1_o * 16 + 16])
                                    T.block_attr({"meta_schedule.auto_tensorize": "wmma_load_16x16x16_f16_b_shared"})
                                    for ax0_1, ax1_1 in T.grid(16, 16):
                                        with T.block("B_reindex_shared_wmma.matrix_b"):
                                            v0_i, v1_i = T.axis.remap("SS", [ax0_1, ax1_1])
                                            T.reads(B_reindex_shared[v0_o * 16 + v0_i, v1_o * 16 + v1_i])
                                            T.writes(B_reindex_shared_wmma_matrix_b[v0_o * 16 + v0_i, v1_o * 16 + v1_i])
                                            B_reindex_shared_wmma_matrix_b[v0_o * 16 + v0_i, v1_o * 16 + v1_i] = B_reindex_shared[v0_o * 16 + v0_i, v1_o * 16 + v1_i]
                            for ax0_0_3, ax1_0_3, ax2_0_2, ax0_0_4, ax1_0_4 in T.grid(2, 1, 1, 4, 2):
                                with T.block("C_o"):
                                    v0_o = T.axis.spatial(64, ax0_0_1_ax1_0_1_fused // 16 * 32 + ax0_0_2_ax1_0_2_fused // 2 * 8 + ax0_0_3 * 4 + ax0_0_4)
                                    v1_o = T.axis.spatial(64, ax0_0_1_ax1_0_1_fused % 16 * 4 + ax0_0_2_ax1_0_2_fused % 2 * 2 + ax1_0_3 * 2 + ax1_0_4)
                                    v2_o = T.axis.reduce(256, ax2_0_0 * 8 + ax2_0_1 + ax2_0_2)
                                    T.reads(A_reindex_pad_shared_wmma_matrix_a[v0_o * 16:v0_o * 16 + 16, v2_o * 16:v2_o * 16 + 16], B_reindex_shared_wmma_matrix_b[v2_o * 16:v2_o * 16 + 16, v1_o * 16:v1_o * 16 + 16])
                                    T.writes(C_reindex_pad_shared_wmma_accumulator[v0_o // 8, v1_o // 2, v0_o % 8, v1_o % 2, 0:16, 0:16])
                                    T.block_attr({"meta_schedule.auto_tensorize": "wmma_sync_16x16x16_f16f16f32", "meta_schedule.auto_tensorize_init": "wmma_fill_16x16x16_f32", "warp_execution": 1})
                                    with T.init():
                                        for ax0_1, ax1_1 in T.grid(16, 16):
                                            with T.block("C_init"):
                                                v0_i_init, v1_i_init = T.axis.remap("SS", [ax0_1, ax1_1])
                                                T.reads()
                                                T.writes(C_reindex_pad_shared_wmma_accumulator[v0_o // 8, v1_o // 2, v0_o % 8, v1_o % 2, v0_i_init, v1_i_init])
                                                C_reindex_pad_shared_wmma_accumulator[v0_o // 8, v1_o // 2, v0_o % 8, v1_o % 2, v0_i_init, v1_i_init] = T.float32(0.0)
                                    for ax0_1, ax1_1, ax2_1 in T.grid(16, 16, 16):
                                        with T.block("C"):
                                            v0_i, v1_i, v2_i = T.axis.remap("SSR", [ax0_1, ax1_1, ax2_1])
                                            T.reads(C_reindex_pad_shared_wmma_accumulator[v0_o // 8, v1_o // 2, v0_o % 8, v1_o % 2, v0_i, v1_i], A_reindex_pad_shared_wmma_matrix_a[v0_o * 16 + v0_i, v2_o * 16 + v2_i], B_reindex_shared_wmma_matrix_b[v2_o * 16 + v2_i, v1_o * 16 + v1_i])
                                            T.writes(C_reindex_pad_shared_wmma_accumulator[v0_o // 8, v1_o // 2, v0_o % 8, v1_o % 2, v0_i, v1_i])
                                            T.block_attr({"meta_schedule.tiling_structure": "SSSRRSRS"})
                                            C_reindex_pad_shared_wmma_accumulator[v0_o // 8, v1_o // 2, v0_o % 8, v1_o % 2, v0_i, v1_i] = C_reindex_pad_shared_wmma_accumulator[v0_o // 8, v1_o // 2, v0_o % 8, v1_o % 2, v0_i, v1_i] + T.Cast("float32", A_reindex_pad_shared_wmma_matrix_a[v0_o * 16 + v0_i, v2_o * 16 + v2_i]) * T.Cast("float32", B_reindex_shared_wmma_matrix_b[v2_o * 16 + v2_i, v1_o * 16 + v1_i])
                for ax2 in range(8):
                    for ax0_ax1_fused in T.thread_binding(8, thread="threadIdx.y"):
                        for ax2_1, ax3 in T.grid(1, 2):
                            with T.block("C_reindex_pad_shared_wmma.accumulator_o"):
                                v0_o = T.axis.spatial(8, ax0_0_1_ax1_0_1_fused // 16 * 4 + ax0_ax1_fused // 2)
                                v1_o = T.axis.spatial(32, ax0_0_1_ax1_0_1_fused % 16 * 2 + ax0_ax1_fused % 2)
                                v2_o = T.axis.spatial(8, ax2 + ax2_1)
                                v3_o = T.axis.spatial(2, ax3)
                                v4_o = T.axis.spatial(1, 0)
                                v5_o = T.axis.spatial(1, 0)
                                T.reads(C_reindex_pad_shared_wmma_accumulator[v0_o, v1_o, v2_o, v3_o, 0:16, 0:16])
                                T.writes(C_reindex_pad_shared[v0_o, v1_o, v2_o, v3_o, 0:16, 0:16])
                                T.block_attr({"meta_schedule.auto_tensorize": "wmma_store_16x16x16_f32_shared"})
                                for ax4, ax5 in T.grid(16, 16):
                                    with T.block("C_reindex_pad_shared_wmma.accumulator"):
                                        v4_i, v5_i = T.axis.remap("SS", [ax4, ax5])
                                        T.reads(C_reindex_pad_shared_wmma_accumulator[v0_o, v1_o, v2_o, v3_o, v4_i, v5_i])
                                        T.writes(C_reindex_pad_shared[v0_o, v1_o, v2_o, v3_o, v4_i, v5_i])
                                        C_reindex_pad_shared[v0_o, v1_o, v2_o, v3_o, v4_i, v5_i] = C_reindex_pad_shared_wmma_accumulator[v0_o, v1_o, v2_o, v3_o, v4_i, v5_i]
                    for ax0_ax1_ax3_ax4_ax5_fused in range(4096):
                        with T.block("C_reindex_pad_shared"):
                            v0 = T.axis.spatial(8, ax0_0_1_ax1_0_1_fused // 16 * 4 + ax0_ax1_ax3_ax4_ax5_fused // 1024)
                            v1 = T.axis.spatial(32, ax0_0_1_ax1_0_1_fused % 16 * 2 + ax0_ax1_ax3_ax4_ax5_fused % 1024 // 512)
                            v2 = T.axis.spatial(8, ax2)
                            v3 = T.axis.spatial(2, ax0_ax1_ax3_ax4_ax5_fused % 512 // 256)
                            v4 = T.axis.spatial(16, ax0_ax1_ax3_ax4_ax5_fused % 256 // 16)
                            v5 = T.axis.spatial(16, ax0_ax1_ax3_ax4_ax5_fused % 16)
                            T.where(ax0_0_1_ax1_0_1_fused // 16 * 512 + ax0_ax1_ax3_ax4_ax5_fused // 1024 * 128 + ax2 * 16 + ax0_ax1_ax3_ax4_ax5_fused % 256 // 16 < 1023)
                            T.reads(C_reindex_pad_shared[v0, v1, v2, v3, v4, v5])
                            T.writes(C[v4 + v2 * 16 + v0 * 128, v5 + v3 * 16 + v1 * 32])
                            T.block_attr({"meta_schedule.cooperative_fetch": 4})
                            C[v4 + v2 * 16 + v0 * 128, v5 + v3 * 16 + v1 * 32] = C_reindex_pad_shared[v0, v1, v2, v3, v4, v5]
    # fmt: on

    decision_0 = [
        ("SamplePerfectTile", [1, 2, 4, 2, 4]),
        ("SamplePerfectTile", [1, 16, 2, 1, 2]),
        ("SamplePerfectTile", [32, 8, 1]),
        ("SampleCategorical", 3),
        ("SampleCategorical", 1),
        ("SampleCategorical", 0),
    ]
    mod = te.create_prim_func(
        te_workload.matmul(
            n=1023,
            m=1024,
            k=4096,
            in_dtype="float16",
            out_dtype="float32",
        )
    )
    actual = generate_design_space(
        kind="cuda",
        mod=mod,
        target=tvm.target.Target("cuda --arch=sm_70"),
        types=None,
        sch_rules=[multi_level_tiling_tensor_core()]
        + get_rules("cuda", ms.schedule_rule.AutoInline),
    )
    check_sketches(
        mod,
        sketches=actual,
        expected_mods=[padded_matmul_single_padded_input_0],
        expected_decisions=[decision_0],
    )


def test_padded_matmul_no_padded_output():
    # fmt: off
    @T.prim_func
    def padded_matmul_no_padded_output_0(A: T.Buffer((1024, 4095), "float16"), B: T.Buffer((4095, 1024), "float16"), C: T.Buffer((1024, 1024), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        C_reindex_shared = T.alloc_buffer((32, 16, 2, 4, 16, 16), scope="shared")
        C_reindex_shared_wmma_accumulator = T.alloc_buffer((32, 16, 2, 4, 16, 16), scope="wmma.accumulator")
        A_reindex_pad_shared = T.alloc_buffer((1024, 4096), "float16", scope="shared")
        B_reindex_pad_shared = T.alloc_buffer((4096, 1024), "float16", scope="shared")
        A_reindex_pad_shared_wmma_matrix_a = T.alloc_buffer((1024, 4096), "float16", scope="wmma.matrix_a")
        B_reindex_pad_shared_wmma_matrix_b = T.alloc_buffer((4096, 1024), "float16", scope="wmma.matrix_b")
        for ax0_0_0_ax1_0_0_fused in T.thread_binding(64, thread="blockIdx.y"):
            for ax0_0_1_ax1_0_1_fused in T.thread_binding(2, thread="blockIdx.x"):
                for ax0_0_2_ax1_0_2_fused in T.thread_binding(4, thread="threadIdx.y"):
                    for ax2_0_0 in range(128):
                        for ax0_ax1_fused in range(4096):
                            with T.block("A_reindex_pad_shared"):
                                v0 = T.axis.spatial(1024, ax0_0_0_ax1_0_0_fused // 16 * 256 + ax0_0_1_ax1_0_1_fused * 128 + ax0_ax1_fused // 32)
                                v1 = T.axis.spatial(4096, ax2_0_0 * 32 + ax0_ax1_fused % 32)
                                T.reads(A[v0, v1])
                                T.writes(A_reindex_pad_shared[v0, v1])
                                T.block_attr({"buffer_dim_align": [[0, 0, 32, 8]], "meta_schedule.cooperative_fetch": 8})
                                A_reindex_pad_shared[v0, v1] = T.if_then_else(v1 < 4095, A[v0, v1], T.float16(0.0))
                        for ax0_ax1_fused in range(2048):
                            with T.block("B_reindex_pad_shared"):
                                v0 = T.axis.spatial(4096, ax2_0_0 * 32 + ax0_ax1_fused // 64)
                                v1 = T.axis.spatial(1024, ax0_0_0_ax1_0_0_fused % 16 * 64 + ax0_ax1_fused % 64)
                                T.reads(B[v0, v1])
                                T.writes(B_reindex_pad_shared[v0, v1])
                                T.block_attr({"buffer_dim_align": [[0, 0, 32, 8]], "meta_schedule.cooperative_fetch": 1})
                                B_reindex_pad_shared[v0, v1] = T.if_then_else(v0 < 4095, B[v0, v1], T.float16(0.0))
                        for ax2_0_1 in range(2):
                            for ax0_0, ax1_0 in T.grid(2, 1):
                                with T.block("A_reindex_pad_shared_wmma.matrix_a_o"):
                                    v0_o = T.axis.spatial(64, ax0_0_0_ax1_0_0_fused // 16 * 16 + ax0_0_1_ax1_0_1_fused * 8 + ax0_0_2_ax1_0_2_fused * 2 + ax0_0)
                                    v1_o = T.axis.spatial(256, ax2_0_0 * 2 + ax2_0_1 + ax1_0)
                                    T.reads(A_reindex_pad_shared[v0_o * 16:v0_o * 16 + 16, v1_o * 16:v1_o * 16 + 16])
                                    T.writes(A_reindex_pad_shared_wmma_matrix_a[v0_o * 16:v0_o * 16 + 16, v1_o * 16:v1_o * 16 + 16])
                                    T.block_attr({"meta_schedule.auto_tensorize": "wmma_load_16x16x16_f16_a_shared"})
                                    for ax0_1, ax1_1 in T.grid(16, 16):
                                        with T.block("A_reindex_pad_shared_wmma.matrix_a"):
                                            v0_i, v1_i = T.axis.remap("SS", [ax0_1, ax1_1])
                                            T.reads(A_reindex_pad_shared[v0_o * 16 + v0_i, v1_o * 16 + v1_i])
                                            T.writes(A_reindex_pad_shared_wmma_matrix_a[v0_o * 16 + v0_i, v1_o * 16 + v1_i])
                                            A_reindex_pad_shared_wmma_matrix_a[v0_o * 16 + v0_i, v1_o * 16 + v1_i] = A_reindex_pad_shared[v0_o * 16 + v0_i, v1_o * 16 + v1_i]
                            for ax0_0, ax1_0 in T.grid(1, 4):
                                with T.block("B_reindex_pad_shared_wmma.matrix_b_o"):
                                    v0_o = T.axis.spatial(256, ax2_0_0 * 2 + ax2_0_1 + ax0_0)
                                    v1_o = T.axis.spatial(64, ax0_0_0_ax1_0_0_fused % 16 * 4 + ax1_0)
                                    T.reads(B_reindex_pad_shared[v0_o * 16:v0_o * 16 + 16, v1_o * 16:v1_o * 16 + 16])
                                    T.writes(B_reindex_pad_shared_wmma_matrix_b[v0_o * 16:v0_o * 16 + 16, v1_o * 16:v1_o * 16 + 16])
                                    T.block_attr({"meta_schedule.auto_tensorize": "wmma_load_16x16x16_f16_b_shared"})
                                    for ax0_1, ax1_1 in T.grid(16, 16):
                                        with T.block("B_reindex_pad_shared_wmma.matrix_b"):
                                            v0_i, v1_i = T.axis.remap("SS", [ax0_1, ax1_1])
                                            T.reads(B_reindex_pad_shared[v0_o * 16 + v0_i, v1_o * 16 + v1_i])
                                            T.writes(B_reindex_pad_shared_wmma_matrix_b[v0_o * 16 + v0_i, v1_o * 16 + v1_i])
                                            B_reindex_pad_shared_wmma_matrix_b[v0_o * 16 + v0_i, v1_o * 16 + v1_i] = B_reindex_pad_shared[v0_o * 16 + v0_i, v1_o * 16 + v1_i]
                            for ax0_0_3, ax1_0_3, ax2_0_2, ax0_0_4, ax1_0_4 in T.grid(2, 1, 1, 1, 4):
                                with T.block("C_o"):
                                    v0_o = T.axis.spatial(64, ax0_0_0_ax1_0_0_fused // 16 * 16 + ax0_0_1_ax1_0_1_fused * 8 + ax0_0_2_ax1_0_2_fused * 2 + ax0_0_3 + ax0_0_4)
                                    v1_o = T.axis.spatial(64, ax0_0_0_ax1_0_0_fused % 16 * 4 + ax1_0_3 * 4 + ax1_0_4)
                                    v2_o = T.axis.reduce(256, ax2_0_0 * 2 + ax2_0_1 + ax2_0_2)
                                    T.reads(A_reindex_pad_shared_wmma_matrix_a[v0_o * 16:v0_o * 16 + 16, v2_o * 16:v2_o * 16 + 16], B_reindex_pad_shared_wmma_matrix_b[v2_o * 16:v2_o * 16 + 16, v1_o * 16:v1_o * 16 + 16])
                                    T.writes(C_reindex_shared_wmma_accumulator[v0_o // 2, v1_o // 4, v0_o % 2, v1_o % 4, 0:16, 0:16])
                                    T.block_attr({"meta_schedule.auto_tensorize": "wmma_sync_16x16x16_f16f16f32", "meta_schedule.auto_tensorize_init": "wmma_fill_16x16x16_f32", "warp_execution": 1})
                                    with T.init():
                                        for ax0_1, ax1_1 in T.grid(16, 16):
                                            with T.block("C_init"):
                                                v0_i_init, v1_i_init = T.axis.remap("SS", [ax0_1, ax1_1])
                                                T.reads()
                                                T.writes(C_reindex_shared_wmma_accumulator[v0_o // 2, v1_o // 4, v0_o % 2, v1_o % 4, v0_i_init, v1_i_init])
                                                C_reindex_shared_wmma_accumulator[v0_o // 2, v1_o // 4, v0_o % 2, v1_o % 4, v0_i_init, v1_i_init] = T.float32(0.0)
                                    for ax0_1, ax1_1, ax2_1 in T.grid(16, 16, 16):
                                        with T.block("C"):
                                            v0_i, v1_i, v2_i = T.axis.remap("SSR", [ax0_1, ax1_1, ax2_1])
                                            T.reads(C_reindex_shared_wmma_accumulator[v0_o // 2, v1_o // 4, v0_o % 2, v1_o % 4, v0_i, v1_i], A_reindex_pad_shared_wmma_matrix_a[v0_o * 16 + v0_i, v2_o * 16 + v2_i], B_reindex_pad_shared_wmma_matrix_b[v2_o * 16 + v2_i, v1_o * 16 + v1_i])
                                            T.writes(C_reindex_shared_wmma_accumulator[v0_o // 2, v1_o // 4, v0_o % 2, v1_o % 4, v0_i, v1_i])
                                            T.block_attr({"meta_schedule.tiling_structure": "SSSRRSRS"})
                                            C_reindex_shared_wmma_accumulator[v0_o // 2, v1_o // 4, v0_o % 2, v1_o % 4, v0_i, v1_i] = C_reindex_shared_wmma_accumulator[v0_o // 2, v1_o // 4, v0_o % 2, v1_o % 4, v0_i, v1_i] + T.Cast("float32", A_reindex_pad_shared_wmma_matrix_a[v0_o * 16 + v0_i, v2_o * 16 + v2_i]) * T.Cast("float32", B_reindex_pad_shared_wmma_matrix_b[v2_o * 16 + v2_i, v1_o * 16 + v1_i])
                for ax2 in range(2):
                    for ax0_ax1_fused in T.thread_binding(4, thread="threadIdx.y"):
                        for ax2_1, ax3 in T.grid(1, 4):
                            with T.block("C_reindex_shared_wmma.accumulator_o"):
                                v0_o = T.axis.spatial(32, ax0_0_0_ax1_0_0_fused // 16 * 8 + ax0_0_1_ax1_0_1_fused * 4 + ax0_ax1_fused)
                                v1_o = T.axis.spatial(16, ax0_0_0_ax1_0_0_fused % 16)
                                v2_o = T.axis.spatial(2, ax2 + ax2_1)
                                v3_o = T.axis.spatial(4, ax3)
                                v4_o = T.axis.spatial(1, 0)
                                v5_o = T.axis.spatial(1, 0)
                                T.reads(C_reindex_shared_wmma_accumulator[v0_o, v1_o, v2_o, v3_o, 0:16, 0:16])
                                T.writes(C_reindex_shared[v0_o, v1_o, v2_o, v3_o, 0:16, 0:16])
                                T.block_attr({"meta_schedule.auto_tensorize": "wmma_store_16x16x16_f32_shared"})
                                for ax4, ax5 in T.grid(16, 16):
                                    with T.block("C_reindex_shared_wmma.accumulator"):
                                        v4_i, v5_i = T.axis.remap("SS", [ax4, ax5])
                                        T.reads(C_reindex_shared_wmma_accumulator[v0_o, v1_o, v2_o, v3_o, v4_i, v5_i])
                                        T.writes(C_reindex_shared[v0_o, v1_o, v2_o, v3_o, v4_i, v5_i])
                                        C_reindex_shared[v0_o, v1_o, v2_o, v3_o, v4_i, v5_i] = C_reindex_shared_wmma_accumulator[v0_o, v1_o, v2_o, v3_o, v4_i, v5_i]
                    for ax0_ax1_ax3_ax4_ax5_fused in range(4096):
                        with T.block("C_reindex_shared"):
                            v0 = T.axis.spatial(32, ax0_0_0_ax1_0_0_fused // 16 * 8 + ax0_0_1_ax1_0_1_fused * 4 + ax0_ax1_ax3_ax4_ax5_fused // 1024)
                            v1 = T.axis.spatial(16, ax0_0_0_ax1_0_0_fused % 16)
                            v2 = T.axis.spatial(2, ax2)
                            v3 = T.axis.spatial(4, ax0_ax1_ax3_ax4_ax5_fused % 1024 // 256)
                            v4 = T.axis.spatial(16, ax0_ax1_ax3_ax4_ax5_fused % 256 // 16)
                            v5 = T.axis.spatial(16, ax0_ax1_ax3_ax4_ax5_fused % 16)
                            T.reads(C_reindex_shared[v0, v1, v2, v3, v4, v5])
                            T.writes(C[v4 + v2 * 16 + v0 * 32, v5 + v3 * 16 + v1 * 64])
                            T.block_attr({"meta_schedule.cooperative_fetch": 3})
                            C[v4 + v2 * 16 + v0 * 32, v5 + v3 * 16 + v1 * 64] = C_reindex_shared[v0, v1, v2, v3, v4, v5]
    # fmt: on

    decision_0 = [
        ("SamplePerfectTile", [4, 2, 4, 2, 1]),
        ("SamplePerfectTile", [16, 1, 1, 1, 4]),
        ("SamplePerfectTile", [128, 2, 1]),
        ("SampleCategorical", 2),
        ("SampleCategorical", 3),
        ("SampleCategorical", 0),
    ]
    mod = te.create_prim_func(
        te_workload.matmul(
            n=1024,
            m=1024,
            k=4095,
            in_dtype="float16",
            out_dtype="float32",
        )
    )
    actual = generate_design_space(
        kind="cuda",
        mod=mod,
        target=tvm.target.Target("cuda --arch=sm_70"),
        types=None,
        sch_rules=[multi_level_tiling_tensor_core()]
        + get_rules("cuda", ms.schedule_rule.AutoInline),
    )
    check_sketches(
        mod,
        sketches=actual,
        expected_mods=[padded_matmul_no_padded_output_0],
        expected_decisions=[decision_0],
    )


if __name__ == "__main__":
    tvm.testing.main()
