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
from tvm import meta_schedule as ms
from tvm import te
from tvm.meta_schedule.testing import te_workload
from tvm.meta_schedule.testing.schedule_rule import get_rules
from tvm.meta_schedule.testing.space_generation import check_sketches
from tvm.script import tir as T
from tvm.target import Target


def test_cpu_matmul():
    @T.prim_func
    def cpu_matmul_0(
        A: T.Buffer[(512, 512), "float32"],
        B: T.Buffer[(512, 512), "float32"],
        C: T.Buffer[(512, 512), "float32"],
    ) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # body
        # with T.block("root")
        C_global = T.alloc_buffer([512, 512], dtype="float32")
        for i0_0, i1_0, i0_1, i1_1 in T.grid(1, 8, 8, 1):
            for i2_0, i0_2, i1_2, i2_1, i0_3, i1_3 in T.grid(16, 2, 8, 32, 32, 8):
                with T.block("C"):
                    i = T.axis.spatial(512, i0_0 * 512 + i0_1 * 64 + i0_2 * 32 + i0_3)
                    j = T.axis.spatial(512, i1_0 * 64 + i1_1 * 64 + i1_2 * 8 + i1_3)
                    k = T.axis.reduce(512, i2_0 * 32 + i2_1)
                    T.reads(A[i, k], B[k, j])
                    T.writes(C_global[i, j])
                    T.block_attr({"meta_schedule.tiling_structure": "SSRSRS"})
                    with T.init():
                        C_global[i, j] = T.float32(0)
                    C_global[i, j] = C_global[i, j] + A[i, k] * B[k, j]
            for ax0, ax1 in T.grid(64, 64):
                with T.block("C_global"):
                    v0 = T.axis.spatial(512, i0_1 * 64 + ax0)
                    v1 = T.axis.spatial(512, i1_0 * 64 + ax1)
                    T.reads(C_global[v0, v1])
                    T.writes(C[v0, v1])
                    C[v0, v1] = C_global[v0, v1]

    @T.prim_func
    def cpu_matmul_1(
        A: T.Buffer[(512, 512), "float32"],
        B: T.Buffer[(512, 512), "float32"],
        C: T.Buffer[(512, 512), "float32"],
    ) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # body
        # with T.block("root")
        C_global = T.alloc_buffer([512, 512], dtype="float32")
        for i0_0, i1_0 in T.grid(1, 8):
            for i0_1, i1_1, i2_0, i0_2, i1_2, i2_1, i0_3, i1_3 in T.grid(8, 1, 16, 2, 8, 32, 32, 8):
                with T.block("C"):
                    i = T.axis.spatial(512, i0_0 * 512 + i0_1 * 64 + i0_2 * 32 + i0_3)
                    j = T.axis.spatial(512, i1_0 * 64 + i1_1 * 64 + i1_2 * 8 + i1_3)
                    k = T.axis.reduce(512, i2_0 * 32 + i2_1)
                    T.reads(A[i, k], B[k, j])
                    T.writes(C_global[i, j])
                    T.block_attr({"meta_schedule.tiling_structure": "SSRSRS"})
                    with T.init():
                        C_global[i, j] = T.float32(0)
                    C_global[i, j] = C_global[i, j] + A[i, k] * B[k, j]
            for ax0, ax1 in T.grid(512, 64):
                with T.block("C_global"):
                    v0 = T.axis.spatial(512, ax0)
                    v1 = T.axis.spatial(512, i1_0 * 64 + ax1)
                    T.reads(C_global[v0, v1])
                    T.writes(C[v0, v1])
                    C[v0, v1] = C_global[v0, v1]

    @T.prim_func
    def cpu_matmul_2(
        A: T.Buffer[(512, 512), "float32"],
        B: T.Buffer[(512, 512), "float32"],
        C: T.Buffer[(512, 512), "float32"],
    ) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # body
        # with T.block("root")
        for i0_0, i1_0, i0_1, i1_1, i2_0, i0_2, i1_2, i2_1, i0_3, i1_3 in T.grid(
            1, 8, 8, 1, 16, 2, 8, 32, 32, 8
        ):
            with T.block("C"):
                i = T.axis.spatial(512, i0_0 * 512 + i0_1 * 64 + i0_2 * 32 + i0_3)
                j = T.axis.spatial(512, i1_0 * 64 + i1_1 * 64 + i1_2 * 8 + i1_3)
                k = T.axis.reduce(512, i2_0 * 32 + i2_1)
                T.reads(A[i, k], B[k, j])
                T.writes(C[i, j])
                T.block_attr({"meta_schedule.tiling_structure": "SSRSRS"})
                with T.init():
                    C[i, j] = T.float32(0)
                C[i, j] = C[i, j] + A[i, k] * B[k, j]

    decision_0 = [
        ("SamplePerfectTile", [1, 8, 2, 32]),
        ("SamplePerfectTile", [8, 1, 8, 8]),
        ("SamplePerfectTile", [16, 32]),
    ]
    decision_1 = [
        ("SamplePerfectTile", [1, 8, 2, 32]),
        ("SamplePerfectTile", [8, 1, 8, 8]),
        ("SamplePerfectTile", [16, 32]),
    ]
    decision_2 = [
        ("SamplePerfectTile", [1, 8, 2, 32]),
        ("SamplePerfectTile", [8, 1, 8, 8]),
        ("SamplePerfectTile", [16, 32]),
    ]

    mod = te.create_prim_func(te_workload.matmul(512, 512, 512))
    actual = ms.TuneContext(
        mod=mod,
        target=Target("llvm"),
        space_generator=ms.space_generator.PostOrderApply(),
        sch_rules=get_rules("llvm", ms.schedule_rule.MultiLevelTiling),
        task_name="test",
    ).generate_design_space()
    check_sketches(
        mod,
        sketches=actual,
        expected_mods=[cpu_matmul_0, cpu_matmul_1, cpu_matmul_2],
        expected_decisions=[decision_0, decision_1, decision_2],
    )


def test_cpu_matmul_relu():
    @T.prim_func
    def cpu_matmul_relu_0(
        A: T.Buffer[(512, 512), "float32"],
        B: T.Buffer[(512, 512), "float32"],
        compute: T.Buffer[(512, 512), "float32"],
    ) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # body
        # with T.block("root")
        C = T.alloc_buffer([512, 512], dtype="float32")
        for i0_0, i1_0, i0_1, i1_1, i2_0, i0_2, i1_2, i2_1, i0_3, i1_3 in T.grid(
            256, 4, 1, 4, 64, 1, 32, 8, 2, 1
        ):
            with T.block("C"):
                i = T.axis.spatial(512, i0_0 * 2 + i0_1 * 2 + i0_2 * 2 + i0_3)
                j = T.axis.spatial(512, i1_0 * 128 + i1_1 * 32 + i1_2 + i1_3)
                k = T.axis.reduce(512, i2_0 * 8 + i2_1)
                T.reads(A[i, k], B[k, j])
                T.writes(C[i, j])
                T.block_attr({"meta_schedule.tiling_structure": "SSRSRS"})
                with T.init():
                    C[i, j] = T.float32(0)
                C[i, j] = C[i, j] + A[i, k] * B[k, j]
        for i0, i1 in T.grid(512, 512):
            with T.block("compute"):
                i0_4, i1_4 = T.axis.remap("SS", [i0, i1])
                T.reads(C[i0_4, i1_4])
                T.writes(compute[i0_4, i1_4])
                compute[i0_4, i1_4] = T.max(C[i0_4, i1_4], T.float32(0))

    @T.prim_func
    def cpu_matmul_relu_1(
        A: T.Buffer[(512, 512), "float32"],
        B: T.Buffer[(512, 512), "float32"],
        compute: T.Buffer[(512, 512), "float32"],
    ) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # body
        # with T.block("root")
        C = T.alloc_buffer([512, 512], dtype="float32")
        for i0_0, i1_0, i0_1, i1_1 in T.grid(256, 4, 1, 4):
            for i2_0, i0_2, i1_2, i2_1, i0_3, i1_3 in T.grid(64, 1, 32, 8, 2, 1):
                with T.block("C"):
                    i = T.axis.spatial(512, i0_0 * 2 + i0_1 * 2 + i0_2 * 2 + i0_3)
                    j = T.axis.spatial(512, i1_0 * 128 + i1_1 * 32 + i1_2 + i1_3)
                    k = T.axis.reduce(512, i2_0 * 8 + i2_1)
                    T.reads(A[i, k], B[k, j])
                    T.writes(C[i, j])
                    T.block_attr({"meta_schedule.tiling_structure": "SSRSRS"})
                    with T.init():
                        C[i, j] = T.float32(0)
                    C[i, j] = C[i, j] + A[i, k] * B[k, j]
            for ax0, ax1 in T.grid(2, 32):
                with T.block("compute"):
                    i0 = T.axis.spatial(512, i0_0 * 2 + ax0)
                    i1 = T.axis.spatial(512, i1_0 * 128 + i1_1 * 32 + ax1)
                    T.reads(C[i0, i1])
                    T.writes(compute[i0, i1])
                    compute[i0, i1] = T.max(C[i0, i1], T.float32(0))

    @T.prim_func
    def cpu_matmul_relu_2(
        A: T.Buffer[(512, 512), "float32"],
        B: T.Buffer[(512, 512), "float32"],
        compute: T.Buffer[(512, 512), "float32"],
    ) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # body
        # with T.block("root")
        C = T.alloc_buffer([512, 512], dtype="float32")
        for i0_0, i1_0 in T.grid(256, 4):
            for i0_1, i1_1, i2_0, i0_2, i1_2, i2_1, i0_3, i1_3 in T.grid(1, 4, 64, 1, 32, 8, 2, 1):
                with T.block("C"):
                    i = T.axis.spatial(512, i0_0 * 2 + i0_1 * 2 + i0_2 * 2 + i0_3)
                    j = T.axis.spatial(512, i1_0 * 128 + i1_1 * 32 + i1_2 + i1_3)
                    k = T.axis.reduce(512, i2_0 * 8 + i2_1)
                    T.reads(A[i, k], B[k, j])
                    T.writes(C[i, j])
                    T.block_attr({"meta_schedule.tiling_structure": "SSRSRS"})
                    with T.init():
                        C[i, j] = T.float32(0)
                    C[i, j] = C[i, j] + A[i, k] * B[k, j]
            for ax0, ax1 in T.grid(2, 128):
                with T.block("compute"):
                    i0 = T.axis.spatial(512, i0_0 * 2 + ax0)
                    i1 = T.axis.spatial(512, i1_0 * 128 + ax1)
                    T.reads(C[i0, i1])
                    T.writes(compute[i0, i1])
                    compute[i0, i1] = T.max(C[i0, i1], T.float32(0))

    decision_0 = [
        ("SamplePerfectTile", [256, 1, 1, 2]),
        ("SamplePerfectTile", [4, 4, 32, 1]),
        ("SamplePerfectTile", [64, 8]),
    ]
    decision_1 = [
        ("SamplePerfectTile", [256, 1, 1, 2]),
        ("SamplePerfectTile", [4, 4, 32, 1]),
        ("SamplePerfectTile", [64, 8]),
    ]
    decision_2 = [
        ("SamplePerfectTile", [256, 1, 1, 2]),
        ("SamplePerfectTile", [4, 4, 32, 1]),
        ("SamplePerfectTile", [64, 8]),
    ]
    mod = te.create_prim_func(te_workload.matmul_relu(512, 512, 512))
    actual = ms.TuneContext(
        mod=mod,
        target=Target("llvm"),
        space_generator=ms.space_generator.PostOrderApply(),
        sch_rules=get_rules("llvm", ms.schedule_rule.MultiLevelTiling),
        task_name="test",
    ).generate_design_space()
    check_sketches(
        mod,
        sketches=actual,
        expected_mods=[cpu_matmul_relu_0, cpu_matmul_relu_1, cpu_matmul_relu_2],
        expected_decisions=[decision_0, decision_1, decision_2],
    )


def test_cuda_matmul():
    @T.prim_func
    def cuda_matmul_0(
        A: T.Buffer[(512, 512), "float32"],
        B: T.Buffer[(512, 512), "float32"],
        C: T.Buffer[(512, 512), "float32"],
    ) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # body
        # with T.block("root")
        C_local = T.alloc_buffer([512, 512], dtype="float32", scope="local")
        A_shared = T.alloc_buffer([512, 512], dtype="float32", scope="shared")
        B_shared = T.alloc_buffer([512, 512], dtype="float32", scope="shared")
        for i0_0_i1_0_fused in T.thread_binding(128, thread="blockIdx.x"):
            for i0_1_i1_1_fused in T.thread_binding(8, thread="vthread.x"):
                for i0_2_i1_2_fused in T.thread_binding(4, thread="threadIdx.x"):
                    for i2_0 in T.serial(128):
                        for ax0_ax1_fused in T.serial(256):
                            with T.block("A_shared"):
                                v0 = T.axis.spatial(
                                    512, i0_0_i1_0_fused // 16 * 64 + ax0_ax1_fused // 4
                                )
                                v1 = T.axis.spatial(512, i2_0 * 4 + ax0_ax1_fused % 4)
                                T.reads(A[v0, v1])
                                T.writes(A_shared[v0, v1])
                                T.block_attr({"meta_schedule.cooperative_fetch": 2})
                                A_shared[v0, v1] = A[v0, v1]
                        for ax0_ax1_fused in T.serial(128):
                            with T.block("B_shared"):
                                v0 = T.axis.spatial(512, i2_0 * 4 + ax0_ax1_fused // 32)
                                v1 = T.axis.spatial(
                                    512, i0_0_i1_0_fused % 16 * 32 + ax0_ax1_fused % 32
                                )
                                T.reads(B[v0, v1])
                                T.writes(B_shared[v0, v1])
                                T.block_attr({"meta_schedule.cooperative_fetch": 1})
                                B_shared[v0, v1] = B[v0, v1]
                        for i2_1, i0_3, i1_3, i2_2, i0_4, i1_4 in T.grid(2, 1, 1, 2, 16, 4):
                            with T.block("C"):
                                i = T.axis.spatial(
                                    512,
                                    i0_0_i1_0_fused // 16 * 64
                                    + i0_1_i1_1_fused // 2 * 16
                                    + i0_3 * 16
                                    + i0_4,
                                )
                                j = T.axis.spatial(
                                    512,
                                    i0_0_i1_0_fused % 16 * 32
                                    + i0_1_i1_1_fused % 2 * 16
                                    + i0_2_i1_2_fused * 4
                                    + i1_3 * 4
                                    + i1_4,
                                )
                                k = T.axis.reduce(512, i2_0 * 4 + i2_1 * 2 + i2_2)
                                T.reads(A_shared[i, k], B_shared[k, j])
                                T.writes(C_local[i, j])
                                T.block_attr(
                                    {
                                        "meta_schedule.thread_extent_high_inclusive": 1024,
                                        "meta_schedule.thread_extent_low_inclusive": 32,
                                        "meta_schedule.tiling_structure": "SSSRRSRS",
                                    }
                                )
                                with T.init():
                                    C_local[i, j] = T.float32(0)
                                C_local[i, j] = C_local[i, j] + A_shared[i, k] * B_shared[k, j]
                    for ax0, ax1 in T.grid(16, 4):
                        with T.block("C_local"):
                            v0 = T.axis.spatial(
                                512, i0_0_i1_0_fused // 16 * 64 + i0_1_i1_1_fused // 2 * 16 + ax0
                            )
                            v1 = T.axis.spatial(
                                512,
                                i0_0_i1_0_fused % 16 * 32
                                + i0_1_i1_1_fused % 2 * 16
                                + i0_2_i1_2_fused * 4
                                + ax1,
                            )
                            T.reads(C_local[v0, v1])
                            T.writes(C[v0, v1])
                            C[v0, v1] = C_local[v0, v1]

    decision_0 = [
        ("SamplePerfectTile", [8, 4, 1, 1, 16]),
        ("SamplePerfectTile", [16, 2, 4, 1, 4]),
        ("SamplePerfectTile", [128, 2, 2]),
        ("SampleCategorical", 1),
        ("SampleCategorical", 0),
    ]
    mod = te.create_prim_func(te_workload.matmul(512, 512, 512))
    actual = ms.TuneContext(
        mod=mod,
        target=Target("nvidia/geforce-rtx-3080"),
        space_generator=ms.space_generator.PostOrderApply(),
        sch_rules=get_rules("cuda", ms.schedule_rule.MultiLevelTiling),
        task_name="test",
    ).generate_design_space()
    check_sketches(
        mod,
        sketches=actual,
        expected_mods=[cuda_matmul_0],
        expected_decisions=[decision_0],
    )


def test_cuda_matmul_relu():
    @T.prim_func
    def cuda_matmul_relu_0(
        A: T.Buffer[(512, 512), "float32"],
        B: T.Buffer[(512, 512), "float32"],
        compute: T.Buffer[(512, 512), "float32"],
    ) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # body
        # with T.block("root")
        C = T.alloc_buffer([512, 512], dtype="float32")
        C_local = T.alloc_buffer([512, 512], dtype="float32", scope="local")
        A_shared = T.alloc_buffer([512, 512], dtype="float32", scope="shared")
        B_shared = T.alloc_buffer([512, 512], dtype="float32", scope="shared")
        for i0_0_i1_0_fused in T.thread_binding(64, thread="blockIdx.x"):
            for i0_1_i1_1_fused in T.thread_binding(64, thread="vthread.x"):
                for i0_2_i1_2_fused in T.thread_binding(8, thread="threadIdx.x"):
                    for i2_0 in T.serial(8):
                        for ax0_ax1_fused in T.serial(4096):
                            with T.block("A_shared"):
                                v0 = T.axis.spatial(
                                    512, i0_0_i1_0_fused // 8 * 64 + ax0_ax1_fused // 64
                                )
                                v1 = T.axis.spatial(512, i2_0 * 64 + ax0_ax1_fused % 64)
                                T.reads(A[v0, v1])
                                T.writes(A_shared[v0, v1])
                                T.block_attr({"meta_schedule.cooperative_fetch": 2})
                                A_shared[v0, v1] = A[v0, v1]
                        for ax0_ax1_fused in T.serial(4096):
                            with T.block("B_shared"):
                                v0 = T.axis.spatial(512, i2_0 * 64 + ax0_ax1_fused // 64)
                                v1 = T.axis.spatial(
                                    512, i0_0_i1_0_fused % 8 * 64 + ax0_ax1_fused % 64
                                )
                                T.reads(B[v0, v1])
                                T.writes(B_shared[v0, v1])
                                T.block_attr({"meta_schedule.cooperative_fetch": 4})
                                B_shared[v0, v1] = B[v0, v1]
                        for i2_1, i0_3, i1_3, i2_2, i0_4, i1_4 in T.grid(8, 2, 1, 8, 2, 2):
                            with T.block("C"):
                                i = T.axis.spatial(
                                    512,
                                    i0_0_i1_0_fused // 8 * 64
                                    + i0_1_i1_1_fused // 8 * 8
                                    + i0_2_i1_2_fused // 4 * 4
                                    + i0_3 * 2
                                    + i0_4,
                                )
                                j = T.axis.spatial(
                                    512,
                                    i0_0_i1_0_fused % 8 * 64
                                    + i0_1_i1_1_fused % 8 * 8
                                    + i0_2_i1_2_fused % 4 * 2
                                    + i1_3 * 2
                                    + i1_4,
                                )
                                k = T.axis.reduce(512, i2_0 * 64 + i2_1 * 8 + i2_2)
                                T.reads(A_shared[i, k], B_shared[k, j])
                                T.writes(C_local[i, j])
                                T.block_attr(
                                    {
                                        "meta_schedule.thread_extent_high_inclusive": 1024,
                                        "meta_schedule.thread_extent_low_inclusive": 32,
                                        "meta_schedule.tiling_structure": "SSSRRSRS",
                                    }
                                )
                                with T.init():
                                    C_local[i, j] = T.float32(0)
                                C_local[i, j] = C_local[i, j] + A_shared[i, k] * B_shared[k, j]
                    for ax0, ax1 in T.grid(4, 2):
                        with T.block("C_local"):
                            v0 = T.axis.spatial(
                                512,
                                i0_0_i1_0_fused // 8 * 64
                                + i0_1_i1_1_fused // 8 * 8
                                + i0_2_i1_2_fused // 4 * 4
                                + ax0,
                            )
                            v1 = T.axis.spatial(
                                512,
                                i0_0_i1_0_fused % 8 * 64
                                + i0_1_i1_1_fused % 8 * 8
                                + i0_2_i1_2_fused % 4 * 2
                                + ax1,
                            )
                            T.reads(C_local[v0, v1])
                            T.writes(C[v0, v1])
                            C[v0, v1] = C_local[v0, v1]
        for i0, i1 in T.grid(512, 512):
            with T.block("compute"):
                i0_1, i1_1 = T.axis.remap("SS", [i0, i1])
                T.reads(C[i0_1, i1_1])
                T.writes(compute[i0_1, i1_1])
                compute[i0_1, i1_1] = T.max(C[i0_1, i1_1], T.float32(0))

    decision_0 = [
        ("SamplePerfectTile", [8, 8, 2, 2, 2]),
        ("SamplePerfectTile", [8, 8, 4, 1, 2]),
        ("SamplePerfectTile", [8, 8, 8]),
        ("SampleCategorical", 1),
        ("SampleCategorical", 3),
    ]
    mod = te.create_prim_func(te_workload.matmul_relu(512, 512, 512))
    actual = ms.TuneContext(
        mod=mod,
        target=Target("nvidia/geforce-rtx-3080"),
        space_generator=ms.space_generator.PostOrderApply(),
        sch_rules=get_rules("cuda", ms.schedule_rule.MultiLevelTiling),
        task_name="test",
    ).generate_design_space()
    check_sketches(
        mod,
        sketches=actual,
        expected_mods=[cuda_matmul_relu_0],
        expected_decisions=[decision_0],
    )


def test_cuda_sum_with_trivial_block_iter():
    @T.prim_func
    def sum_with_trivial_block_iter(
        A: T.Buffer[(1, 64, 768), "float32"],
        B: T.Buffer[(1, 64, 1), "float32"],
    ) -> None:
        for i0, i1, i2, i3 in T.grid(1, 64, 1, 768):
            with T.block("sum"):
                ax0, ax1, ax2, k2 = T.axis.remap("SSSR", [i0, i1, i2, i3])
                T.reads(A[ax0, ax1, k2])
                T.writes(B[ax0, ax1, ax2])
                with T.init():
                    B[ax0, ax1, ax2] = T.float32(0)
                B[ax0, ax1, ax2] = B[ax0, ax1, ax2] + A[ax0, ax1, k2]

    # Expect nothing to happen - the rule is not supposed to be applied in this case
    mod = sum_with_trivial_block_iter
    (sch,) = ms.TuneContext(
        mod=mod,
        target=Target("nvidia/geforce-rtx-3080"),
        space_generator=ms.space_generator.PostOrderApply(),
        sch_rules=get_rules("cuda", ms.schedule_rule.MultiLevelTiling),
        task_name="test",
    ).generate_design_space()
    assert not sch.trace.simplified(remove_postproc=True).insts


def test_multi_level_tiling_hexagon():
    target_hexagon = tvm.target.hexagon("v69", num_cores=4)
    target = tvm.target.Target(target_hexagon, host=target_hexagon)

    I = 64
    O = 64
    H = 56
    W = 56

    data, kernel, out = te_workload.conv2d_nhwc(
        1, H, W, I, O, 3, 1, 1, 1, in_dtype="float16", out_dtype="float16"
    )
    workload = te.create_prim_func([data, kernel, out])

    ctx = _create_context(
        workload,
        target=target,
        rule=schedule_rule.MultiLevelTilingWideVector(
            structure="SRSRS",
            vector_length_in_bits=1024,
            max_innermost_factor=64,
            reuse_read=None,
            reuse_write=None,
        ),
    )
    spaces = ctx.space_generator.generate_design_space(mod=ctx.mod)
    assert len(spaces) == 1

    expected = [
        """b0 = sch.get_block(name="conv2d_nhwc", func_name="main")
sch.annotate(block_or_loop=b0, ann_key="meta_schedule.tiling_structure", ann_val="SRSRS")
l1, l2, l3, l4, l5, l6, l7 = sch.get_loops(block=b0)
v8, v9, v10 = sch.sample_perfect_tile(loop=l1, n=3, max_innermost_factor=64)
l11, l12, l13 = sch.split(loop=l1, factors=[v8, v9, v10], preserve_unit_iters=True)
v14, v15, v16 = sch.sample_perfect_tile(loop=l2, n=3, max_innermost_factor=64)
l17, l18, l19 = sch.split(loop=l2, factors=[v14, v15, v16], preserve_unit_iters=True)
v20, v21, v22 = sch.sample_perfect_tile(loop=l3, n=3, max_innermost_factor=64)
l23, l24, l25 = sch.split(loop=l3, factors=[v20, v21, v22], preserve_unit_iters=True)
l26, l27, l28 = sch.split(loop=l4, factors=[1, 1, 64], preserve_unit_iters=True)
v29, v30 = sch.sample_perfect_tile(loop=l5, n=2, max_innermost_factor=64)
l31, l32 = sch.split(loop=l5, factors=[v29, v30], preserve_unit_iters=True)
v33, v34 = sch.sample_perfect_tile(loop=l6, n=2, max_innermost_factor=64)
l35, l36 = sch.split(loop=l6, factors=[v33, v34], preserve_unit_iters=True)
v37, v38 = sch.sample_perfect_tile(loop=l7, n=2, max_innermost_factor=64)
l39, l40 = sch.split(loop=l7, factors=[v37, v38], preserve_unit_iters=True)
sch.reorder(l11, l17, l23, l26, l31, l35, l39, l12, l18, l24, l27, l32, l36, l40, l13, l19, l25, l28)""".split(
            "\n"
        )
    ]

    check_trace(spaces, expected)


if __name__ == "__main__":
    test_cpu_matmul()
    test_cpu_matmul_relu()
    test_cuda_matmul()
    test_cuda_matmul_relu()
    test_cuda_sum_with_trivial_block_iter()
