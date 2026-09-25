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
# ruff: noqa: F401
import sys

import numpy as np
import pytest

import tvm
import tvm.s_tir.tensor_intrin.cuda
import tvm.testing
from tvm import te, tirx
from tvm.s_tir.meta_schedule.testing import te_workload
from tvm.s_tir.tensor_intrin.cuda import (
    LDMATRIX_f16_A_DYN_INTRIN,
    LDMATRIX_f16_B_DYN_INTRIN,
    MMA_f16f16f32_INTRIN,
    MMA_fill_16x16_f32_INTRIN,
    MMA_store_16x16_f32_global_INTRIN,
    shared_16x16_to_ldmatrix_32x8_layout,
)
from tvm.script import s_tir as Ts
from tvm.script import tirx as T
from tvm.testing import env
from tvm.testing.tir import mma_schedule


def _check(original, transformed):
    func = original
    mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    mod = tvm.s_tir.transform.InjectSoftwarePipeline()(mod)
    mod = tvm.s_tir.transform.StmtSimplify()(mod)
    tvm.ir.assert_structural_equal(
        mod["main"], transformed.with_attr("global_symbol", "main"), True
    )


def _check_error(func):
    mod = tvm.IRModule.from_expr(func)
    with pytest.raises(ValueError):
        tvm.s_tir.transform.InjectSoftwarePipeline()(mod)


@Ts.prim_func
def trivial_pipeline(A: T.Buffer((16, 1), "float32"), C: T.Buffer((16, 1), "float32")):
    for tx in T.thread_binding(0, 16, thread="threadIdx.x"):
        for i in T.serial(
            0, 1, annotations={"software_pipeline_stage": [0, 1], "software_pipeline_order": [0, 1]}
        ):
            with Ts.sblock():
                Ts.reads(A[tx, i])
                Ts.writes(C[tx, i])
                B = Ts.sblock_alloc_buffer((16, 1), dtype="float32", scope="shared")
                with Ts.sblock():
                    Ts.reads(A[tx, i])
                    Ts.writes(B[tx, 0])
                    B[tx, 0] = A[tx, i] * T.float32(2)
                with Ts.sblock():
                    Ts.reads(B[tx, 0])
                    Ts.writes(C[tx, i])
                    C[tx, i] = B[tx, 0] + T.float32(1)


@Ts.prim_func
def transformed_trivial_pipeline(
    A: T.Buffer((16, 1), "float32"), C: T.Buffer((16, 1), "float32")
) -> None:
    for tx in T.thread_binding(16, thread="threadIdx.x"):
        with Ts.sblock():
            Ts.reads(A[tx, 0])
            Ts.writes(C[tx, 0])
            B = Ts.sblock_alloc_buffer([2, 16, 1], dtype="float32", scope="shared")
            with Ts.sblock():
                Ts.reads(A[tx, 0])
                Ts.writes(B[0, tx, 0])
                B[0, tx, 0] = A[tx, 0] * T.float32(2)
            with Ts.sblock():
                Ts.reads()
                Ts.writes()
                T.evaluate(0)
            with Ts.sblock():
                Ts.reads(B[0, tx, 0])
                Ts.writes(C[tx, 0])
                C[tx, 0] = B[0, tx, 0] + T.float32(1)


def gen_simple_compute(num_stages):
    @Ts.prim_func
    def simple_compute(A: T.Buffer((16, 16), "float32"), C: T.Buffer((16, 16), "float32")):
        for tx in T.thread_binding(0, 16, thread="threadIdx.x"):
            for i in T.serial(
                0,
                16,
                annotations={
                    "software_pipeline_stage": [0, num_stages],
                    "software_pipeline_order": [0, 1],
                },
            ):
                with Ts.sblock("compute"):
                    Ts.reads(A[tx, i])
                    Ts.writes(C[tx, i])
                    B = Ts.sblock_alloc_buffer((16, 1), dtype="float32", scope="shared")
                    with Ts.sblock():
                        Ts.reads(A[tx, i])
                        Ts.writes(B[tx, 0])
                        B[tx, 0] = A[tx, i] * T.float32(2)
                    with Ts.sblock():
                        Ts.reads(B[tx, 0])
                        Ts.writes(C[tx, i])
                        C[tx, i] = B[tx, 0] + T.float32(1)

    return simple_compute


@Ts.prim_func
def transformed_simple_compute(
    A: T.Buffer((16, 16), "float32"), C: T.Buffer((16, 16), "float32")
) -> None:
    for tx in T.thread_binding(0, 16, thread="threadIdx.x"):
        with Ts.sblock():
            Ts.reads([A[tx, 0:16]])
            Ts.writes([C[tx, 0:16]])
            B = Ts.sblock_alloc_buffer([2, 16, 1], dtype="float32", scope="shared")
            with Ts.sblock():
                Ts.reads([A[tx, 0]])
                Ts.writes([B[0, tx, 0]])
                B[0, tx, 0] = A[tx, 0] * T.float32(2)
            with Ts.sblock():
                Ts.reads([A[tx, 1:16], B[0:2, tx, 0]])
                Ts.writes([B[0:2, tx, 0], C[tx, 0:15]])
                for i in T.serial(0, 15):
                    with Ts.sblock():
                        Ts.reads([A[tx, i + 1]])
                        Ts.writes([B[(i + 1) % 2, tx, 0]])
                        B[(i + 1) % 2, tx, 0] = A[tx, i + 1] * T.float32(2)
                    with Ts.sblock():
                        Ts.reads([B[i % 2, tx, 0]])
                        Ts.writes([C[tx, i]])
                        C[tx, i] = B[i % 2, tx, 0] + T.float32(1)
            with Ts.sblock():
                Ts.reads([B[1, tx, 0]])
                Ts.writes([C[tx, 15]])
                C[tx, 15] = B[1, tx, 0] + T.float32(1)


@Ts.prim_func
def dynamic_compute(a_handle: T.handle, c_handle: T.handle):
    k = T.int32()
    A = T.match_buffer(a_handle, (16, k), "float32")
    C = T.match_buffer(c_handle, (16, k), "float32")
    for tx in T.thread_binding(0, 16, thread="threadIdx.x"):
        for i in T.serial(
            0,
            k,
            annotations={
                "software_pipeline_stage": [0, 1],
                "software_pipeline_order": [0, 1],
            },
        ):
            with Ts.sblock("compute"):
                Ts.reads(A[tx, i])
                Ts.writes(C[tx, i])
                B = Ts.sblock_alloc_buffer((16, 1), dtype="float32", scope="shared")
                with Ts.sblock():
                    Ts.reads(A[tx, i])
                    Ts.writes(B[tx, 0])
                    B[tx, 0] = A[tx, i] * T.float32(2)
                with Ts.sblock():
                    Ts.reads(B[tx, 0])
                    Ts.writes(C[tx, i])
                    C[tx, i] = B[tx, 0] + T.float32(1)


@Ts.prim_func
def transformed_dynamic_compute(a_handle: T.handle, c_handle: T.handle):
    k = T.int32()
    A = T.match_buffer(a_handle, (16, k), "float32")
    C = T.match_buffer(c_handle, (16, k), "float32")
    for tx in T.thread_binding(0, 16, thread="threadIdx.x"):
        with Ts.sblock():
            Ts.reads(A[tx, 0 : T.max(1, k)])
            Ts.writes(C[tx, T.min(0, k - 1) : T.min(0, k - 1) + T.max(k, 1)])
            B = Ts.sblock_alloc_buffer([2, 16, 1], dtype="float32", scope="shared")
            with Ts.sblock(""):
                Ts.reads(A[tx, 0])
                Ts.writes(B[0, tx, 0])
                with Ts.sblock(""):
                    Ts.where(0 < k)
                    Ts.reads(A[tx, 0])
                    Ts.writes(B[0, tx, 0])
                    B[0, tx, 0] = A[tx, 0] * T.float32(2)
            with Ts.sblock(""):
                Ts.reads(A[tx, 1 : 1 + (k - 1)], B[0:2, tx, 0])
                Ts.writes(B[0:2, tx, 0], C[tx, 0 : k - 1])
                for i in range(k - 1):
                    with Ts.sblock(""):
                        Ts.reads(A[tx, i + 1])
                        Ts.writes(B[(i + 1) % 2, tx, 0])
                        B[(i + 1) % 2, tx, 0] = A[tx, i + 1] * T.float32(2)
                    with Ts.sblock(""):
                        Ts.reads(B[i % 2, tx, 0])
                        Ts.writes(C[tx, i])
                        C[tx, i] = B[i % 2, tx, 0] + T.float32(1)
            with Ts.sblock(""):
                Ts.reads(B[(k + 1) % 2, tx, 0])
                Ts.writes(C[tx, k - 1])
                with Ts.sblock(""):
                    Ts.where(1 <= k)
                    Ts.reads(B[(k + 1) % 2, tx, 0])
                    Ts.writes(C[tx, k - 1])
                    C[tx, k - 1] = B[(k + 1) % 2, tx, 0] + T.float32(1)


@Ts.prim_func
def simple_compute_with_other_annotation(
    A: T.Buffer((16, 16), "float32"), C: T.Buffer((16, 16), "float32")
):
    for tx in T.thread_binding(0, 16, thread="threadIdx.x"):
        for i in T.serial(
            0,
            16,
            annotations={
                "software_pipeline_stage": [0, 1],
                "software_pipeline_order": [0, 1],
                "pragma_loop_partition_hint": True,
            },
        ):
            with Ts.sblock("compute"):
                Ts.reads(A[tx, i])
                Ts.writes(C[tx, i])
                B = Ts.sblock_alloc_buffer((16, 1), dtype="float32", scope="shared")
                with Ts.sblock():
                    Ts.reads(A[tx, i])
                    Ts.writes(B[tx, 0])
                    B[tx, 0] = A[tx, i] * T.float32(2)
                with Ts.sblock():
                    Ts.reads(B[tx, 0])
                    Ts.writes(C[tx, i])
                    C[tx, i] = B[tx, 0] + T.float32(1)


@Ts.prim_func
def transformed_simple_compute_with_other_annotation(
    A: T.Buffer((16, 16), "float32"), C: T.Buffer((16, 16), "float32")
) -> None:
    for tx in T.thread_binding(0, 16, thread="threadIdx.x"):
        with Ts.sblock():
            Ts.reads([A[tx, 0:16]])
            Ts.writes([C[tx, 0:16]])
            B = Ts.sblock_alloc_buffer([2, 16, 1], dtype="float32", scope="shared")
            with Ts.sblock():
                Ts.reads([A[tx, 0]])
                Ts.writes([B[0, tx, 0]])
                B[0, tx, 0] = A[tx, 0] * T.float32(2)
            with Ts.sblock():
                Ts.reads([A[tx, 1:16], B[0:2, tx, 0]])
                Ts.writes([B[0:2, tx, 0], C[tx, 0:15]])
                for i in T.serial(
                    0,
                    15,
                    annotations={"pragma_loop_partition_hint": True},
                ):
                    with Ts.sblock():
                        Ts.reads([A[tx, i + 1]])
                        Ts.writes([B[(i + 1) % 2, tx, 0]])
                        B[(i + 1) % 2, tx, 0] = A[tx, i + 1] * T.float32(2)
                    with Ts.sblock():
                        Ts.reads([B[i % 2, tx, 0]])
                        Ts.writes([C[tx, i]])
                        C[tx, i] = B[i % 2, tx, 0] + T.float32(1)
            with Ts.sblock():
                Ts.reads([B[1, tx, 0]])
                Ts.writes([C[tx, 15]])
                C[tx, 15] = B[1, tx, 0] + T.float32(1)


@Ts.prim_func
def three_stage_compute(A: T.Buffer((16, 16), "float32"), D: T.Buffer((16, 16), "float32")):
    for tx in T.thread_binding(0, 16, thread="threadIdx.x"):
        for i in T.serial(
            0,
            16,
            annotations={
                "software_pipeline_stage": [0, 1, 2],
                "software_pipeline_order": [0, 1, 2],
            },
        ):
            with Ts.sblock("compute"):
                Ts.reads(A[tx, i])
                Ts.writes(D[tx, i])
                B = Ts.sblock_alloc_buffer((16, 1), dtype="float32", scope="shared")
                C = Ts.sblock_alloc_buffer((16, 1), dtype="float32", scope="shared")
                with Ts.sblock():
                    Ts.reads(A[tx, i])
                    Ts.writes(B[tx, 0])
                    B[tx, 0] = A[tx, i] * T.float32(2)
                with Ts.sblock():
                    Ts.reads(B[tx, 0])
                    Ts.writes(C[tx, 0])
                    C[tx, 0] = B[tx, 0] + T.float32(2)
                with Ts.sblock():
                    Ts.reads(C[tx, 0])
                    Ts.writes(D[tx, i])
                    D[tx, i] = C[tx, 0] + T.float32(1)


@Ts.prim_func
def transformed_three_stage_compute(
    A: T.Buffer((16, 16), "float32"), D: T.Buffer((16, 16), "float32")
) -> None:
    for tx in T.thread_binding(16, thread="threadIdx.x"):
        with Ts.sblock():
            Ts.reads(A[tx, 0:16])
            Ts.writes(D[tx, 0:16])
            B = Ts.sblock_alloc_buffer([2, 16, 1], dtype="float32", scope="shared")
            C = Ts.sblock_alloc_buffer([2, 16, 1], dtype="float32", scope="shared")
            with Ts.sblock():
                Ts.reads(A[tx, 0:2], B[0:2, tx, 0])
                Ts.writes(B[0:2, tx, 0], C[0:2, tx, 0])
                for i in T.unroll(2):
                    with Ts.sblock():
                        Ts.reads(A[tx, i])
                        Ts.writes(B[0:2, tx, 0])
                        B[i, tx, 0] = A[tx, i] * T.float32(2)
                    with Ts.sblock():
                        Ts.where(i == 1)
                        Ts.reads(B[0:2, tx, 0])
                        Ts.writes(C[0:2, tx, 0])
                        C[(i + 1) % 2, tx, 0] = B[(i + 1) % 2, tx, 0] + T.float32(2)
            with Ts.sblock():
                Ts.reads(A[tx, 2:16], B[0:2, tx, 0], C[0:2, tx, 0])
                Ts.writes(B[0:2, tx, 0], C[0:2, tx, 0], D[tx, 0:14])
                for i in T.serial(14):
                    with Ts.sblock():
                        Ts.reads(A[tx, i + 2])
                        Ts.writes(B[0:2, tx, 0])
                        B[i % 2, tx, 0] = A[tx, i + 2] * T.float32(2)
                    with Ts.sblock():
                        Ts.reads(B[0:2, tx, 0])
                        Ts.writes(C[0:2, tx, 0])
                        C[(i + 1) % 2, tx, 0] = B[(i + 1) % 2, tx, 0] + T.float32(2)
                    with Ts.sblock():
                        Ts.reads(C[0:2, tx, 0])
                        Ts.writes(D[tx, i])
                        D[tx, i] = C[i % 2, tx, 0] + T.float32(1)
            with Ts.sblock():
                Ts.reads(B[0:2, tx, 0], C[0:2, tx, 0])
                Ts.writes(C[0:2, tx, 0], D[tx, 14:16])
                for i in T.unroll(2):
                    with Ts.sblock():
                        Ts.where(i < 1)
                        Ts.reads(B[0:2, tx, 0])
                        Ts.writes(C[0:2, tx, 0])
                        C[(i + 1) % 2, tx, 0] = B[(i + 1) % 2, tx, 0] + T.float32(2)
                    with Ts.sblock():
                        Ts.reads(C[0:2, tx, 0])
                        Ts.writes(D[tx, i + 14])
                        D[tx, i + 14] = C[i, tx, 0] + T.float32(1)


@Ts.prim_func
def dag_interleaving(
    A: T.Buffer((16, 16), "float32"),
    B: T.Buffer((16, 16), "float32"),
    C: T.Buffer((16, 16), "float32"),
) -> None:
    for tx in T.thread_binding(0, 16, thread="threadIdx.x"):
        for i in T.serial(
            0,
            16,
            annotations={
                "software_pipeline_stage": [0, 0, 0, 0, 1],
                "software_pipeline_order": [0, 2, 1, 3, 4],
            },
        ):
            with Ts.sblock():
                Ts.reads(A[tx, i])
                Ts.writes(C[tx, i])
                AS = Ts.sblock_alloc_buffer((16, 1), dtype="float32", scope="shared")
                BS = Ts.sblock_alloc_buffer((16, 1), dtype="float32", scope="shared")
                AL = Ts.sblock_alloc_buffer((1, 1), dtype="float32", scope="local")
                BL = Ts.sblock_alloc_buffer((1, 1), dtype="float32", scope="local")
                with Ts.sblock():
                    Ts.reads(A[tx, i])
                    Ts.writes(AS[tx, 0])
                    AS[tx, 0] = A[tx, i] * T.float32(2)
                with Ts.sblock():
                    Ts.reads(AS[tx, 0])
                    Ts.writes(AL[0, 0])
                    AL[0, 0] = AS[tx, 0]
                with Ts.sblock():
                    Ts.reads(B[tx, i])
                    Ts.writes(BS[tx, 0])
                    BS[tx, 0] = B[tx, i] + T.float32(2)
                with Ts.sblock():
                    Ts.reads(BS[tx, 0])
                    Ts.writes(BL[0, 0])
                    BL[0, 0] = BS[tx, 0]
                with Ts.sblock():
                    Ts.reads(AL[0, 0], BL[0, 0])
                    Ts.writes(C[tx, i])
                    C[tx, i] = AL[0, 0] * BL[0, 0]


@Ts.prim_func
def transformed_dag_interleaving(
    A: T.Buffer((16, 16), "float32"),
    B: T.Buffer((16, 16), "float32"),
    C: T.Buffer((16, 16), "float32"),
) -> None:
    for tx in T.thread_binding(16, thread="threadIdx.x"):
        with Ts.sblock():
            Ts.reads(A[tx, 0:16], B[tx, 0:16])
            Ts.writes(C[tx, 0:16])
            AS = Ts.sblock_alloc_buffer([16, 1], dtype="float32", scope="shared")
            BS = Ts.sblock_alloc_buffer([16, 1], dtype="float32", scope="shared")
            AL = Ts.sblock_alloc_buffer([2, 1, 1], dtype="float32", scope="local")
            BL = Ts.sblock_alloc_buffer([2, 1, 1], dtype="float32", scope="local")
            with Ts.sblock():
                Ts.reads(A[tx, 0], B[tx, 0], AS[tx, 0], BS[tx, 0])
                Ts.writes(AS[tx, 0], BS[tx, 0], AL[0, 0, 0], BL[0, 0, 0])
                with Ts.sblock():
                    Ts.reads(A[tx, 0])
                    Ts.writes(AS[tx, 0])
                    AS[tx, 0] = A[tx, 0] * T.float32(2)
                with Ts.sblock():
                    Ts.reads(B[tx, 0])
                    Ts.writes(BS[tx, 0])
                    BS[tx, 0] = B[tx, 0] + T.float32(2)
                with Ts.sblock():
                    Ts.reads(AS[tx, 0])
                    Ts.writes(AL[0, 0, 0])
                    AL[0, 0, 0] = AS[tx, 0]
                with Ts.sblock():
                    Ts.reads(BS[tx, 0])
                    Ts.writes(BL[0, 0, 0])
                    BL[0, 0, 0] = BS[tx, 0]
            with Ts.sblock():
                Ts.reads(
                    A[tx, 1:16], B[tx, 1:16], AS[tx, 0], BS[tx, 0], AL[0:2, 0, 0], BL[0:2, 0, 0]
                )
                Ts.writes(AS[tx, 0], BS[tx, 0], AL[0:2, 0, 0], BL[0:2, 0, 0], C[tx, 0:15])
                for i in T.serial(15):
                    with Ts.sblock():
                        Ts.reads(A[tx, i + 1])
                        Ts.writes(AS[tx, 0])
                        AS[tx, 0] = A[tx, i + 1] * T.float32(2)
                    with Ts.sblock():
                        Ts.reads(B[tx, i + 1])
                        Ts.writes(BS[tx, 0])
                        BS[tx, 0] = B[tx, i + 1] + T.float32(2)
                    with Ts.sblock():
                        Ts.reads(AS[tx, 0])
                        Ts.writes(AL[(i + 1) % 2, 0, 0])
                        AL[(i + 1) % 2, 0, 0] = AS[tx, 0]
                    with Ts.sblock():
                        Ts.reads(BS[tx, 0])
                        Ts.writes(BL[(i + 1) % 2, 0, 0])
                        BL[(i + 1) % 2, 0, 0] = BS[tx, 0]
                    with Ts.sblock():
                        Ts.reads(AL[i % 2, 0, 0], BL[i % 2, 0, 0])
                        Ts.writes(C[tx, i])
                        C[tx, i] = AL[i % 2, 0, 0] * BL[i % 2, 0, 0]
            with Ts.sblock():
                Ts.reads(AL[1, 0, 0], BL[1, 0, 0])
                Ts.writes(C[tx, 15])
                C[tx, 15] = AL[1, 0, 0] * BL[1, 0, 0]


@Ts.prim_func
def nested_pipeline_simple(
    A: T.Buffer((16, 16, 16), "float32"), C: T.Buffer((16, 16, 16), "float32")
):
    for tx in T.thread_binding(0, 16, thread="threadIdx.x"):
        for i in T.serial(
            0,
            16,
            annotations={
                "software_pipeline_stage": [0, 1, 1, 1],
                "software_pipeline_order": [0, 1, 2, 3],
            },
        ):
            with Ts.sblock():
                Ts.reads(A[tx, i, 0:16])
                Ts.writes(C[tx, i, 0:16])
                A_shared = Ts.sblock_alloc_buffer((16, 1, 16), dtype="float32", scope="shared")
                for j in T.serial(0, 16):
                    with Ts.sblock():
                        Ts.reads(A[tx, i, j])
                        Ts.writes(A_shared[tx, 0, j])
                        A_shared[tx, 0, j] = A[tx, i, j]
                for j in T.serial(
                    0,
                    16,
                    annotations={
                        "software_pipeline_stage": [0, 1],
                        "software_pipeline_order": [0, 1],
                    },
                ):
                    with Ts.sblock():
                        Ts.reads(A_shared[tx, 0, j])
                        Ts.writes(C[tx, i, j])
                        B = Ts.sblock_alloc_buffer((16, 1, 1), dtype="float32", scope="shared")
                        with Ts.sblock():
                            Ts.reads(A_shared[tx, i, j])
                            Ts.writes(B[tx, i, 0])
                            B[tx, i, 0] = A_shared[tx, 0, j] * T.float32(2)
                        with Ts.sblock():
                            Ts.reads(B[tx, i, 0])
                            Ts.writes(C[tx, i, j])
                            C[tx, i, j] = B[tx, i, 0] + T.float32(1)


@Ts.prim_func
def transformed_nested_pipeline_simple(
    A: T.Buffer((16, 16, 16), "float32"), C: T.Buffer((16, 16, 16), "float32")
) -> None:
    for tx in T.thread_binding(0, 16, thread="threadIdx.x"):
        with Ts.sblock():
            Ts.reads([A[tx, 0:16, 0:16]])
            Ts.writes([C[tx, 0:16, 0:16]])
            A_shared = Ts.sblock_alloc_buffer([2, 16, 1, 16], dtype="float32", scope="shared")
            B = Ts.sblock_alloc_buffer([2, 16, 1, 1], dtype="float32", scope="shared")
            with Ts.sblock():
                Ts.reads([A[tx, 0, 0:16]])
                Ts.writes([A_shared[0, tx, 0, 0:16]])
                for j in T.serial(0, 16):
                    with Ts.sblock():
                        Ts.reads([A[tx, 0, j]])
                        Ts.writes([A_shared[0, tx, 0, j]])
                        A_shared[0, tx, 0, j] = A[tx, 0, j]
            with Ts.sblock():
                Ts.reads([A[tx, 1:16, 0:16], A_shared[0:2, tx, 0:15, 0:16], B[0:2, tx, 0:15, 0]])
                Ts.writes([A_shared[0:2, tx, 0, 0:16], B[0:2, tx, 0:15, 0], C[tx, 0:15, 0:16]])
                for i in T.serial(0, 15):
                    with Ts.sblock():
                        Ts.reads([A[tx, i + 1, 0:16]])
                        Ts.writes([A_shared[(i + 1) % 2, tx, 0, 0:16]])
                        for j in T.serial(0, 16):
                            with Ts.sblock():
                                Ts.reads([A[tx, i + 1, j]])
                                Ts.writes([A_shared[(i + 1) % 2, tx, 0, j]])
                                A_shared[(i + 1) % 2, tx, 0, j] = A[tx, i + 1, j]
                    with Ts.sblock():
                        Ts.reads([A_shared[i % 2, tx, i, 0]])
                        Ts.writes([B[0, tx, i, 0]])
                        B[0, tx, i, 0] = A_shared[i % 2, tx, 0, 0] * T.float32(2)
                    with Ts.sblock():
                        Ts.reads([A_shared[i % 2, tx, i, 1:16], B[0:2, tx, i, 0]])
                        Ts.writes([B[0:2, tx, i, 0], C[tx, i, 0:15]])
                        for j in T.serial(0, 15):
                            with Ts.sblock():
                                Ts.reads([A_shared[i % 2, tx, i, j + 1]])
                                Ts.writes([B[(j + 1) % 2, tx, i, 0]])
                                B[(j + 1) % 2, tx, i, 0] = A_shared[
                                    i % 2, tx, 0, j + 1
                                ] * T.float32(2)
                            with Ts.sblock():
                                Ts.reads([B[j % 2, tx, i, 0]])
                                Ts.writes([C[tx, i, j]])
                                C[tx, i, j] = B[j % 2, tx, i, 0] + T.float32(1)
                    with Ts.sblock():
                        Ts.reads([B[1, tx, i, 0]])
                        Ts.writes([C[tx, i, 15]])
                        C[tx, i, 15] = B[1, tx, i, 0] + T.float32(1)
            with Ts.sblock():
                Ts.reads([A_shared[1, tx, 15, 0:16], B[0:2, tx, 15, 0]])
                Ts.writes([B[0:2, tx, 15, 0], C[tx, 15, 0:16]])
                with Ts.sblock():
                    Ts.reads([A_shared[1, tx, 15, 0]])
                    Ts.writes([B[0, tx, 15, 0]])
                    B[0, tx, 15, 0] = A_shared[1, tx, 0, 0] * T.float32(2)
                with Ts.sblock():
                    Ts.reads([A_shared[1, tx, 15, 1:16], B[0:2, tx, 15, 0]])
                    Ts.writes([B[0:2, tx, 15, 0], C[tx, 15, 0:15]])
                    for j in T.serial(0, 15):
                        with Ts.sblock():
                            Ts.reads([A_shared[1, tx, 15, j + 1]])
                            Ts.writes([B[(j + 1) % 2, tx, 15, 0]])
                            B[(j + 1) % 2, tx, 15, 0] = A_shared[1, tx, 0, j + 1] * T.float32(2)
                        with Ts.sblock():
                            Ts.reads([B[j % 2, tx, 15, 0]])
                            Ts.writes([C[tx, 15, j]])
                            C[tx, 15, j] = B[j % 2, tx, 15, 0] + T.float32(1)
                with Ts.sblock():
                    Ts.reads([B[1, tx, 15, 0]])
                    Ts.writes([C[tx, 15, 15]])
                    C[tx, 15, 15] = B[1, tx, 15, 0] + T.float32(1)


@Ts.prim_func
def nested_pipeline_prefetch_inner(
    A: T.Buffer((16, 16, 16), "float32"), C: T.Buffer((16, 16, 16), "float32")
):
    for tx in T.thread_binding(0, 16, thread="threadIdx.x"):
        for i in T.serial(
            0,
            16,
            annotations={
                "software_pipeline_stage": [0, 0, 1, 1],
                "software_pipeline_order": [0, 2, 1, 3],
            },
        ):
            with Ts.sblock():
                Ts.reads(A[tx, i, 0:16])
                Ts.writes(C[tx, i, 0:16])
                A_shared = Ts.sblock_alloc_buffer((16, 1, 16), dtype="float32", scope="shared")
                for j in T.serial(0, 16):
                    with Ts.sblock():
                        Ts.reads(A[tx, i, j])
                        Ts.writes(A_shared[tx, 0, j])
                        A_shared[tx, 0, j] = A[tx, i, j]
                for j in T.serial(
                    0,
                    16,
                    annotations={
                        "software_pipeline_stage": [0, 1],
                        "software_pipeline_order": [0, 1],
                    },
                ):
                    with Ts.sblock():
                        Ts.reads(A_shared[tx, 0, j])
                        Ts.writes(C[tx, i, j])
                        B = Ts.sblock_alloc_buffer((16, 1, 1), dtype="float32", scope="shared")
                        with Ts.sblock():
                            Ts.reads(A_shared[tx, i, j])
                            Ts.writes(B[tx, i, 0])
                            B[tx, i, 0] = A_shared[tx, 0, j] * T.float32(2)
                        with Ts.sblock():
                            Ts.reads(B[tx, i, 0])
                            Ts.writes(C[tx, i, j])
                            C[tx, i, j] = B[tx, i, 0] + T.float32(1)


@Ts.prim_func
def transformed_nested_pipeline_prefetch_inner(
    A: T.Buffer((16, 16, 16), "float32"), C: T.Buffer((16, 16, 16), "float32")
) -> None:
    for tx in T.thread_binding(0, 16, thread="threadIdx.x"):
        with Ts.sblock():
            Ts.reads([A[tx, 0:16, 0:16]])
            Ts.writes([C[tx, 0:16, 0:16]])
            A_shared = Ts.sblock_alloc_buffer([2, 16, 1, 16], dtype="float32", scope="shared")
            B = Ts.sblock_alloc_buffer([2, 16, 1, 1], dtype="float32", scope="shared")
            with Ts.sblock():
                Ts.reads([A[tx, 0, 0:16], A_shared[0, tx, 0, 0]])
                Ts.writes([A_shared[0, tx, 0, 0:16], B[0, tx, 0, 0]])
                with Ts.sblock():
                    Ts.reads([A[tx, 0, 0:16]])
                    Ts.writes([A_shared[0, tx, 0, 0:16]])
                    for j in T.serial(0, 16):
                        with Ts.sblock():
                            Ts.reads([A[tx, 0, j]])
                            Ts.writes([A_shared[0, tx, 0, j]])
                            A_shared[0, tx, 0, j] = A[tx, 0, j]
                with Ts.sblock():
                    Ts.reads([A_shared[0, tx, 0, 0]])
                    Ts.writes([B[0, tx, 0, 0]])
                    B[0, tx, 0, 0] = A_shared[0, tx, 0, 0] * T.float32(2)
            with Ts.sblock():
                Ts.reads([A[tx, 1:16, 0:16], A_shared[0:2, tx, 0:16, 0:16], B[0:2, tx, 0:15, 0]])
                Ts.writes([A_shared[0:2, tx, 0, 0:16], B[0:2, tx, 0:16, 0], C[tx, 0:15, 0:16]])
                for i in T.serial(0, 15):
                    with Ts.sblock():
                        Ts.reads([A[tx, i + 1, 0:16]])
                        Ts.writes([A_shared[(i + 1) % 2, tx, 0, 0:16]])
                        for j in T.serial(0, 16):
                            with Ts.sblock():
                                Ts.reads([A[tx, i + 1, j]])
                                Ts.writes([A_shared[(i + 1) % 2, tx, 0, j]])
                                A_shared[(i + 1) % 2, tx, 0, j] = A[tx, i + 1, j]
                    with Ts.sblock():
                        Ts.reads([A_shared[i % 2, tx, i, 1:16], B[0:2, tx, i, 0]])
                        Ts.writes([B[0:2, tx, i, 0], C[tx, i, 0:15]])
                        for j in T.serial(0, 15):
                            with Ts.sblock():
                                Ts.reads([A_shared[i % 2, tx, i, j + 1]])
                                Ts.writes([B[(j + 1) % 2, tx, i, 0]])
                                B[(j + 1) % 2, tx, i, 0] = A_shared[
                                    i % 2, tx, 0, j + 1
                                ] * T.float32(2)
                            with Ts.sblock():
                                Ts.reads([B[j % 2, tx, i, 0]])
                                Ts.writes([C[tx, i, j]])
                                C[tx, i, j] = B[j % 2, tx, i, 0] + T.float32(1)
                    with Ts.sblock():
                        Ts.reads([A_shared[(i + 1) % 2, tx, i + 1, 0]])
                        Ts.writes([B[0, tx, i + 1, 0]])
                        B[0, tx, i + 1, 0] = A_shared[(i + 1) % 2, tx, 0, 0] * T.float32(2)
                    with Ts.sblock():
                        Ts.reads([B[1, tx, i, 0]])
                        Ts.writes([C[tx, i, 15]])
                        C[tx, i, 15] = B[1, tx, i, 0] + T.float32(1)
            with Ts.sblock():
                Ts.reads([A_shared[1, tx, 15, 1:16], B[0:2, tx, 15, 0]])
                Ts.writes([B[0:2, tx, 15, 0], C[tx, 15, 0:16]])
                with Ts.sblock():
                    Ts.reads([A_shared[1, tx, 15, 1:16], B[0:2, tx, 15, 0]])
                    Ts.writes([B[0:2, tx, 15, 0], C[tx, 15, 0:15]])
                    for j in T.serial(0, 15):
                        with Ts.sblock():
                            Ts.reads([A_shared[1, tx, 15, j + 1]])
                            Ts.writes([B[(j + 1) % 2, tx, 15, 0]])
                            B[(j + 1) % 2, tx, 15, 0] = A_shared[1, tx, 0, j + 1] * T.float32(2)
                        with Ts.sblock():
                            Ts.reads([B[j % 2, tx, 15, 0]])
                            Ts.writes([C[tx, 15, j]])
                            C[tx, 15, j] = B[j % 2, tx, 15, 0] + T.float32(1)
                with Ts.sblock():
                    Ts.reads([B[1, tx, 15, 0]])
                    Ts.writes([C[tx, 15, 15]])
                    C[tx, 15, 15] = B[1, tx, 15, 0] + T.float32(1)


@Ts.prim_func
def nested_pipeline_interleaving(
    A: T.Buffer((16, 16, 16), "float32"), C: T.Buffer((16, 16, 16), "float32")
):
    for tx in T.thread_binding(0, 16, thread="threadIdx.x"):
        for i in T.serial(
            0,
            16,
            annotations={
                "software_pipeline_stage": [0, 0, 0, 1, 1],
                "software_pipeline_order": [0, 2, 3, 1, 4],
            },
        ):
            with Ts.sblock():
                Ts.reads(A[tx, i, 0:16])
                Ts.writes(C[tx, i, 0:16])
                A_shared = Ts.sblock_alloc_buffer((16, 1, 16), dtype="float32", scope="shared")
                A_local = Ts.sblock_alloc_buffer((1, 1, 16), dtype="float32", scope="local")
                for j in T.serial(0, 16):
                    with Ts.sblock():
                        Ts.reads(A[tx, i, j])
                        Ts.writes(A_shared[tx, 0, j])
                        A_shared[tx, 0, j] = A[tx, i, j]
                for j in T.serial(0, 16):
                    with Ts.sblock():
                        Ts.reads(A_shared[tx, 0, j])
                        Ts.writes(A_local[0, 0, j])
                        A_local[0, 0, j] = A_shared[tx, i, j]
                for j in T.serial(
                    0,
                    16,
                    annotations={
                        "software_pipeline_stage": [0, 1],
                        "software_pipeline_order": [0, 1],
                    },
                ):
                    with Ts.sblock():
                        Ts.reads(A_local[0, 0, j])
                        Ts.writes(C[tx, i, j])
                        B = Ts.sblock_alloc_buffer((16, 1, 1), dtype="float32", scope="shared")
                        with Ts.sblock():
                            Ts.reads(A_local[tx, i, j])
                            Ts.writes(B[tx, i, 0])
                            B[tx, i, 0] = A_local[0, 0, j] * T.float32(2)
                        with Ts.sblock():
                            Ts.reads(B[tx, i, 0])
                            Ts.writes(C[tx, i, j])
                            C[tx, i, j] = B[tx, i, 0] + T.float32(1)


@Ts.prim_func
def transformed_nested_pipeline_interleaving(
    A: T.Buffer((16, 16, 16), "float32"), C: T.Buffer((16, 16, 16), "float32")
) -> None:
    for tx in T.thread_binding(0, 16, thread="threadIdx.x"):
        with Ts.sblock():
            Ts.reads([A[tx, 0:16, 0:16]])
            Ts.writes([C[tx, 0:16, 0:16]])
            A_shared = Ts.sblock_alloc_buffer([16, 1, 16], dtype="float32", scope="shared")
            A_local = Ts.sblock_alloc_buffer([1, 1, 16], dtype="float32", scope="local")
            B = Ts.sblock_alloc_buffer([2, 16, 1, 1], dtype="float32", scope="shared")
            with Ts.sblock():
                Ts.reads([A[tx, 0, 0:16], A_shared[tx, 0, 0:16], A_local[tx, 0, 0]])
                Ts.writes([A_shared[tx, 0, 0:16], A_local[0, 0, 0:16], B[0, tx, 0, 0]])
                with Ts.sblock():
                    Ts.reads([A[tx, 0, 0:16]])
                    Ts.writes([A_shared[tx, 0, 0:16]])
                    for j in T.serial(0, 16):
                        with Ts.sblock():
                            Ts.reads([A[tx, 0, j]])
                            Ts.writes([A_shared[tx, 0, j]])
                            A_shared[tx, 0, j] = A[tx, 0, j]
                with Ts.sblock():
                    Ts.reads([A_shared[tx, 0, 0:16]])
                    Ts.writes([A_local[0, 0, 0:16]])
                    for j in T.serial(0, 16):
                        with Ts.sblock():
                            Ts.reads([A_shared[tx, 0, j]])
                            Ts.writes([A_local[0, 0, j]])
                            A_local[0, 0, j] = A_shared[tx, 0, j]
                with Ts.sblock():
                    Ts.reads([A_local[tx, 0, 0]])
                    Ts.writes([B[0, tx, 0, 0]])
                    B[0, tx, 0, 0] = A_local[0, 0, 0] * T.float32(2)
            with Ts.sblock():
                Ts.reads(
                    [
                        A[tx, 1:16, 0:16],
                        A_local[tx, 0:16, 0:16],
                        B[0:2, tx, 0:15, 0],
                        A_shared[tx, 0, 0:16],
                    ]
                )
                Ts.writes(
                    [
                        A_shared[tx, 0, 0:16],
                        B[0:2, tx, 0:16, 0],
                        C[tx, 0:15, 0:16],
                        A_local[0, 0, 0:16],
                    ]
                )
                for i in T.serial(0, 15):
                    with Ts.sblock():
                        Ts.reads([A[tx, i + 1, 0:16]])
                        Ts.writes([A_shared[tx, 0, 0:16]])
                        for j in T.serial(0, 16):
                            with Ts.sblock():
                                Ts.reads([A[tx, i + 1, j]])
                                Ts.writes([A_shared[tx, 0, j]])
                                A_shared[tx, 0, j] = A[tx, i + 1, j]
                    with Ts.sblock():
                        Ts.reads([A_local[tx, i, 1:16], B[0:2, tx, i, 0]])
                        Ts.writes([B[0:2, tx, i, 0], C[tx, i, 0:15]])
                        for j in T.serial(0, 15):
                            with Ts.sblock():
                                Ts.reads([A_local[tx, i, j + 1]])
                                Ts.writes([B[(j + 1) % 2, tx, i, 0]])
                                B[(j + 1) % 2, tx, i, 0] = A_local[0, 0, j + 1] * T.float32(2)
                            with Ts.sblock():
                                Ts.reads([B[j % 2, tx, i, 0]])
                                Ts.writes([C[tx, i, j]])
                                C[tx, i, j] = B[j % 2, tx, i, 0] + T.float32(1)
                    with Ts.sblock():
                        Ts.reads([A_shared[tx, 0, 0:16]])
                        Ts.writes([A_local[0, 0, 0:16]])
                        for j in T.serial(0, 16):
                            with Ts.sblock():
                                Ts.reads([A_shared[tx, 0, j]])
                                Ts.writes([A_local[0, 0, j]])
                                A_local[0, 0, j] = A_shared[tx, i + 1, j]
                    with Ts.sblock():
                        Ts.reads([A_local[tx, i + 1, 0]])
                        Ts.writes([B[0, tx, i + 1, 0]])
                        B[0, tx, i + 1, 0] = A_local[0, 0, 0] * T.float32(2)
                    with Ts.sblock():
                        Ts.reads([B[1, tx, i, 0]])
                        Ts.writes([C[tx, i, 15]])
                        C[tx, i, 15] = B[1, tx, i, 0] + T.float32(1)
            with Ts.sblock():
                Ts.reads([A_local[tx, 15, 1:16], B[0:2, tx, 15, 0]])
                Ts.writes([B[0:2, tx, 15, 0], C[tx, 15, 0:16]])
                with Ts.sblock():
                    Ts.reads([A_local[tx, 15, 1:16], B[0:2, tx, 15, 0]])
                    Ts.writes([B[0:2, tx, 15, 0], C[tx, 15, 0:15]])
                    for j in T.serial(0, 15):
                        with Ts.sblock():
                            Ts.reads([A_local[tx, 15, j + 1]])
                            Ts.writes([B[(j + 1) % 2, tx, 15, 0]])
                            B[(j + 1) % 2, tx, 15, 0] = A_local[0, 0, j + 1] * T.float32(2)
                        with Ts.sblock():
                            Ts.reads([B[j % 2, tx, 15, 0]])
                            Ts.writes([C[tx, 15, j]])
                            C[tx, 15, j] = B[j % 2, tx, 15, 0] + T.float32(1)
                with Ts.sblock():
                    Ts.reads([B[1, tx, 15, 0]])
                    Ts.writes([C[tx, 15, 15]])
                    C[tx, 15, 15] = B[1, tx, 15, 0] + T.float32(1)


@Ts.prim_func
def nested_pipeline_double_buffer(
    A: T.Buffer((16, 16, 16), "float32"), C: T.Buffer((16, 16, 16), "float32")
):
    for tx in T.thread_binding(0, 16, thread="threadIdx.x"):
        for i in T.serial(
            0,
            16,
            annotations={
                "software_pipeline_stage": [0, 0, 0, 1, 1],
                "software_pipeline_order": [0, 2, 3, 1, 4],
            },
        ):
            with Ts.sblock():
                Ts.reads(A[tx, i, 0:16])
                Ts.writes(C[tx, i, 0:16])
                A_shared = Ts.sblock_alloc_buffer((16, 1, 16), dtype="float32", scope="shared")
                A_local = Ts.sblock_alloc_buffer((1, 1, 16), dtype="float32", scope="local")
                for j in T.serial(0, 16):
                    with Ts.sblock():
                        Ts.reads(A[tx, i, j])
                        Ts.writes(A_shared[tx, 0, j])
                        A_shared[tx, 0, j] = A[tx, i, j]
                for j in T.serial(0, 16):
                    with Ts.sblock():
                        Ts.sblock_attr({"double_buffer_scope": 0})
                        Ts.reads(A_shared[tx, 0, j])
                        Ts.writes(A_local[0, 0, j])
                        A_local[0, 0, j] = A_shared[tx, i, j]
                for j in T.serial(
                    0,
                    16,
                    annotations={
                        "software_pipeline_stage": [0, 1],
                        "software_pipeline_order": [0, 1],
                    },
                ):
                    with Ts.sblock():
                        Ts.reads(A_local[0, 0, j])
                        Ts.writes(C[tx, i, j])
                        B = Ts.sblock_alloc_buffer((16, 1, 1), dtype="float32", scope="shared")
                        with Ts.sblock():
                            Ts.reads(A_local[tx, i, j])
                            Ts.writes(B[tx, i, 0])
                            B[tx, i, 0] = A_local[0, 0, j] * T.float32(2)
                        with Ts.sblock():
                            Ts.reads(B[tx, i, 0])
                            Ts.writes(C[tx, i, j])
                            C[tx, i, j] = B[tx, i, 0] + T.float32(1)


@Ts.prim_func
def transformed_nested_pipeline_double_buffer(
    A: T.Buffer((16, 16, 16), "float32"), C: T.Buffer((16, 16, 16), "float32")
) -> None:
    for tx in T.thread_binding(0, 16, thread="threadIdx.x"):
        with Ts.sblock():
            Ts.reads([A[tx, 0:16, 0:16]])
            Ts.writes([C[tx, 0:16, 0:16]])
            A_shared = Ts.sblock_alloc_buffer([16, 1, 16], dtype="float32", scope="shared")
            A_local = Ts.sblock_alloc_buffer([2, 1, 1, 16], dtype="float32", scope="local")
            B = Ts.sblock_alloc_buffer([2, 16, 1, 1], dtype="float32", scope="shared")
            with Ts.sblock():
                Ts.reads([A[tx, 0, 0:16], A_shared[tx, 0, 0:16], A_local[0, tx, 0, 0]])
                Ts.writes([A_shared[tx, 0, 0:16], A_local[0, 0, 0, 0:16], B[0, tx, 0, 0]])
                with Ts.sblock():
                    Ts.reads([A[tx, 0, 0:16]])
                    Ts.writes([A_shared[tx, 0, 0:16]])
                    for j in T.serial(0, 16):
                        with Ts.sblock():
                            Ts.reads([A[tx, 0, j]])
                            Ts.writes([A_shared[tx, 0, j]])
                            A_shared[tx, 0, j] = A[tx, 0, j]
                with Ts.sblock():
                    Ts.reads([A_shared[tx, 0, 0:16]])
                    Ts.writes([A_local[0, 0, 0, 0:16]])
                    for j in T.serial(0, 16):
                        with Ts.sblock():
                            Ts.reads([A_shared[tx, 0, j]])
                            Ts.writes([A_local[0, 0, 0, j]])
                            Ts.sblock_attr({"double_buffer_scope": 0})
                            A_local[0, 0, 0, j] = A_shared[tx, 0, j]
                with Ts.sblock():
                    Ts.reads([A_local[0, tx, 0, 0]])
                    Ts.writes([B[0, tx, 0, 0]])
                    B[0, tx, 0, 0] = A_local[0, 0, 0, 0] * T.float32(2)
            with Ts.sblock():
                Ts.reads(
                    [
                        A[tx, 1:16, 0:16],
                        A_local[0:2, tx, 0:16, 0:16],
                        B[0:2, tx, 0:15, 0],
                        A_shared[tx, 0, 0:16],
                    ]
                )
                Ts.writes(
                    [
                        A_shared[tx, 0, 0:16],
                        B[0:2, tx, 0:16, 0],
                        C[tx, 0:15, 0:16],
                        A_local[0:2, 0, 0, 0:16],
                    ]
                )
                for i in T.serial(0, 15):
                    with Ts.sblock():
                        Ts.reads([A[tx, i + 1, 0:16]])
                        Ts.writes([A_shared[tx, 0, 0:16]])
                        for j in T.serial(0, 16):
                            with Ts.sblock():
                                Ts.reads([A[tx, i + 1, j]])
                                Ts.writes([A_shared[tx, 0, j]])
                                A_shared[tx, 0, j] = A[tx, i + 1, j]
                    with Ts.sblock():
                        Ts.reads([A_local[i % 2, tx, i, 1:16], B[0:2, tx, i, 0]])
                        Ts.writes([B[0:2, tx, i, 0], C[tx, i, 0:15]])
                        for j in T.serial(0, 15):
                            with Ts.sblock():
                                Ts.reads([A_local[i % 2, tx, i, j + 1]])
                                Ts.writes([B[(j + 1) % 2, tx, i, 0]])
                                B[(j + 1) % 2, tx, i, 0] = A_local[i % 2, 0, 0, j + 1] * T.float32(
                                    2
                                )
                            with Ts.sblock():
                                Ts.reads([B[j % 2, tx, i, 0]])
                                Ts.writes([C[tx, i, j]])
                                C[tx, i, j] = B[j % 2, tx, i, 0] + T.float32(1)
                    with Ts.sblock():
                        Ts.reads([A_shared[tx, 0, 0:16]])
                        Ts.writes([A_local[(i + 1) % 2, 0, 0, 0:16]])
                        for j in T.serial(0, 16):
                            with Ts.sblock():
                                Ts.reads([A_shared[tx, 0, j]])
                                Ts.writes([A_local[(i + 1) % 2, 0, 0, j]])
                                Ts.sblock_attr({"double_buffer_scope": 0})
                                A_local[(i + 1) % 2, 0, 0, j] = A_shared[tx, i + 1, j]
                    with Ts.sblock():
                        Ts.reads([A_local[(i + 1) % 2, tx, i + 1, 0]])
                        Ts.writes([B[0, tx, i + 1, 0]])
                        B[0, tx, i + 1, 0] = A_local[(i + 1) % 2, 0, 0, 0] * T.float32(2)
                    with Ts.sblock():
                        Ts.reads([B[1, tx, i, 0]])
                        Ts.writes([C[tx, i, 15]])
                        C[tx, i, 15] = B[1, tx, i, 0] + T.float32(1)
            with Ts.sblock():
                Ts.reads([A_local[1, tx, 15, 1:16], B[0:2, tx, 15, 0]])
                Ts.writes([B[0:2, tx, 15, 0], C[tx, 15, 0:16]])
                with Ts.sblock():
                    Ts.reads([A_local[1, tx, 15, 1:16], B[0:2, tx, 15, 0]])
                    Ts.writes([B[0:2, tx, 15, 0], C[tx, 15, 0:15]])
                    for j in T.serial(0, 15):
                        with Ts.sblock():
                            Ts.reads([A_local[1, tx, 15, j + 1]])
                            Ts.writes([B[(j + 1) % 2, tx, 15, 0]])
                            B[(j + 1) % 2, tx, 15, 0] = A_local[1, 0, 0, j + 1] * T.float32(2)
                        with Ts.sblock():
                            Ts.reads([B[j % 2, tx, 15, 0]])
                            Ts.writes([C[tx, 15, j]])
                            C[tx, 15, j] = B[j % 2, tx, 15, 0] + T.float32(1)
                with Ts.sblock():
                    Ts.reads([B[1, tx, 15, 0]])
                    Ts.writes([C[tx, 15, 15]])
                    C[tx, 15, 15] = B[1, tx, 15, 0] + T.float32(1)


@Ts.prim_func
def simple_compute_incorrect_reorder(
    A: T.Buffer((16, 16), "float32"), D: T.Buffer((16, 16), "float32")
):
    for tx in T.thread_binding(0, 16, thread="threadIdx.x"):
        for i in T.serial(
            0,
            16,
            annotations={
                "software_pipeline_stage": [0, 1, 1],
                "software_pipeline_order": [0, 2, 1],
            },
        ):
            with Ts.sblock():
                Ts.reads(A[tx, i])
                Ts.writes(D[tx, i])
                B = Ts.sblock_alloc_buffer((16, 1), dtype="float32", scope="shared")
                C = Ts.sblock_alloc_buffer((16, 1), dtype="float32", scope="shared")
                with Ts.sblock():
                    Ts.reads(A[tx, i])
                    Ts.writes(B[tx, 0])
                    B[tx, 0] = A[tx, i] * T.float32(2)
                with Ts.sblock():
                    Ts.reads(B[tx, 0])
                    Ts.writes(C[tx, 0])
                    C[tx, 0] = B[tx, 0] + T.float32(2)
                with Ts.sblock():
                    Ts.reads(C[tx, 0])
                    Ts.writes(D[tx, i])
                    D[tx, i] = C[tx, 0] + T.float32(1)


@Ts.prim_func
def simple_compute_conflicting_order(
    A: T.Buffer((16, 16), "float32"), D: T.Buffer((16, 16), "float32")
):
    for tx in T.thread_binding(0, 16, thread="threadIdx.x"):
        for i in T.serial(
            0,
            16,
            annotations={
                "software_pipeline_stage": [0, 1, 1],
                "software_pipeline_order": [0, 1, 1],
            },
        ):
            with Ts.sblock():
                Ts.reads(A[tx, i])
                Ts.writes(D[tx, i])
                B = Ts.sblock_alloc_buffer((16, 1), dtype="float32", scope="shared")
                C = Ts.sblock_alloc_buffer((16, 1), dtype="float32", scope="shared")
                with Ts.sblock():
                    Ts.reads(A[tx, i])
                    Ts.writes(B[tx, 0])
                    B[tx, 0] = A[tx, i] * T.float32(2)
                with Ts.sblock():
                    Ts.reads(B[tx, 0])
                    Ts.writes(C[tx, 0])
                    C[tx, 0] = B[tx, 0] + T.float32(2)
                with Ts.sblock():
                    Ts.reads(C[tx, 0])
                    Ts.writes(D[tx, i])
                    D[tx, i] = C[tx, 0] + T.float32(1)


@Ts.prim_func
def simple_compute_missing_annotation(
    A: T.Buffer((16, 16), "float32"), C: T.Buffer((16, 16), "float32")
):
    for tx in T.thread_binding(0, 16, thread="threadIdx.x"):
        for i in T.serial(0, 16, annotations={"software_pipeline_stage": [0, 1]}):
            with Ts.sblock():
                Ts.reads(A[tx, i])
                Ts.writes(C[tx, i])
                B = Ts.sblock_alloc_buffer((16, 1), dtype="float32", scope="shared")
                with Ts.sblock():
                    Ts.reads(A[tx, i])
                    Ts.writes(B[tx, 0])
                    B[tx, 0] = A[tx, i] * T.float32(2)
                with Ts.sblock():
                    Ts.reads(B[tx, 0])
                    Ts.writes(C[tx, i])
                    C[tx, i] = B[tx, 0] + T.float32(1)


def test_simple_compute():
    _check(gen_simple_compute(1), transformed_simple_compute)


def test_simple_compute_with_other_annotation():
    _check(simple_compute_with_other_annotation, transformed_simple_compute_with_other_annotation)


def test_dynamic_compute():
    _check(dynamic_compute, transformed_dynamic_compute)


def test_trivial_pipeline():
    _check(trivial_pipeline, transformed_trivial_pipeline)


def test_three_stage_compute():
    _check(three_stage_compute, transformed_three_stage_compute)


def test_dag_interleaving():
    _check(dag_interleaving, transformed_dag_interleaving)


def test_nest_pipeline_simple():
    _check(nested_pipeline_simple, transformed_nested_pipeline_simple)


def test_nest_pipeline_prefetch_inner():
    _check(nested_pipeline_prefetch_inner, transformed_nested_pipeline_prefetch_inner)


def test_nest_pipeline_interleaving():
    _check(nested_pipeline_interleaving, transformed_nested_pipeline_interleaving)


def test_nest_pipeline_double_buffer():
    _check(nested_pipeline_double_buffer, transformed_nested_pipeline_double_buffer)


def test_error_reorder():
    _check_error(simple_compute_incorrect_reorder)


def test_error_conflicting_order():
    _check_error(simple_compute_conflicting_order)


def test_error_missing_annotation():
    _check_error(simple_compute_missing_annotation)


def test_simple_compute_async():
    mod = tvm.IRModule.from_expr(gen_simple_compute(1).with_attr("global_symbol", "main"))
    sch = tvm.s_tir.Schedule(mod)

    _, loop = sch.get_loops(sch.get_sblock("compute"))
    sch.annotate(loop, ann_key="software_pipeline_async_stages", ann_val=[0])
    mod = tvm.s_tir.transform.InjectSoftwarePipeline()(sch.mod)

    @Ts.prim_func
    def ref(A: T.Buffer((16, 16), "float32"), C: T.Buffer((16, 16), "float32")):
        for tx in T.thread_binding(16, thread="threadIdx.x"):
            with Ts.sblock():
                Ts.reads(A[tx, 0:16])
                Ts.writes(C[tx, 0:16])
                B = Ts.sblock_alloc_buffer([2, 16, 1], dtype="float32", scope="shared")
                with Ts.sblock():
                    Ts.reads(A[tx, 0])
                    Ts.writes(B[T.FloorMod(0, 2), tx, 0])
                    with T.attr(0, "async_commit_queue_scope", 0):
                        with T.attr(0, "async_scope", 1):
                            B[T.FloorMod(0, 2), tx, 0] = A[tx, 0] * T.float32(2)
                with Ts.sblock():
                    Ts.reads(A[tx, 1:16], B[0:2, tx, 0])
                    Ts.writes(B[0:2, tx, 0], C[tx, 0:15])
                    for i in T.serial(15):
                        with Ts.sblock():
                            Ts.where(i + 1 < 16)
                            Ts.reads(A[tx, i + 1])
                            Ts.writes(B[(i + 1) % 2, tx, 0])
                            with T.attr(0, "async_commit_queue_scope", 0):
                                with T.attr(0, "async_scope", 1):
                                    B[(i + 1) % 2, tx, 0] = A[tx, i + 1] * T.float32(2)
                        with Ts.sblock():
                            Ts.where(i + 1 - 1 < 16)
                            Ts.reads(B[(i - 1 + 1) % 2, tx, 0])
                            Ts.writes(C[tx, i - 1 + 1])
                            with T.attr(0, "async_wait_queue_scope", 0):
                                with T.attr(0, "async_wait_inflight_count", 1):
                                    C[tx, i - 1 + 1] = B[(i - 1 + 1) % 2, tx, 0] + T.float32(1)
                with Ts.sblock():
                    Ts.reads(B[T.FloorMod(15, 2), tx, 0])
                    Ts.writes(C[tx, 15])
                    with T.attr(0, "async_wait_queue_scope", 0):
                        with T.attr(0, "async_wait_inflight_count", 0):
                            C[tx, 15] = B[T.FloorMod(15, 2), tx, 0] + T.float32(1)

    tvm.ir.assert_structural_equal(mod["main"], ref.with_attr("global_symbol", "main"), True)

    mod = tvm.IRModule.from_expr(gen_simple_compute(3).with_attr("global_symbol", "main"))
    sch = tvm.s_tir.Schedule(mod)

    _, loop = sch.get_loops(sch.get_sblock("compute"))
    sch.annotate(loop, ann_key="software_pipeline_async_stages", ann_val=[0])
    mod = tvm.s_tir.transform.InjectSoftwarePipeline()(sch.mod)

    @Ts.prim_func
    def ref(A: T.Buffer((16, 16), "float32"), C: T.Buffer((16, 16), "float32")) -> None:
        for tx in T.thread_binding(16, thread="threadIdx.x"):
            with Ts.sblock():
                Ts.reads(A[tx, 0:16])
                Ts.writes(C[tx, 0:16])
                B = Ts.sblock_alloc_buffer([4, 16, 1], dtype="float32", scope="shared")
                with Ts.sblock():
                    Ts.reads(A[tx, 0:3])
                    Ts.writes(B[0:3, tx, 0])
                    for i in T.unroll(3):
                        with Ts.sblock():
                            Ts.where(i < 16)
                            Ts.reads(A[tx, i])
                            Ts.writes(B[i % 4, tx, 0])
                            T.attr(0, "async_commit_queue_scope", 0)
                            T.attr(0, "async_scope", 1)
                            B[i % 4, tx, 0] = A[tx, i] * T.float32(2)
                with Ts.sblock():
                    Ts.reads(A[tx, 3:16], B[0:4, tx, 0])
                    Ts.writes(B[0:4, tx, 0], C[tx, 0:13])
                    for i in T.serial(13):
                        with Ts.sblock():
                            Ts.where(i + 3 < 16)
                            Ts.reads(A[tx, i + 3])
                            Ts.writes(B[(i + 3) % 4, tx, 0])
                            T.attr(0, "async_commit_queue_scope", 0)
                            T.attr(0, "async_scope", 1)
                            B[(i + 3) % 4, tx, 0] = A[tx, i + 3] * T.float32(2)
                        with Ts.sblock():
                            Ts.where(i + 3 - 3 < 16)
                            Ts.reads(B[0:4, tx, 0])
                            Ts.writes(C[tx, i - 3 + 3])
                            with T.attr(0, "async_wait_queue_scope", 0):
                                with T.attr(0, "async_wait_inflight_count", 3):
                                    C[tx, i - 3 + 3] = B[(i - 3 + 3) % 4, tx, 0] + T.float32(1)
                with Ts.sblock():
                    Ts.reads(B[0:4, tx, 0])
                    Ts.writes(C[tx, 13:16])
                    for i in T.unroll(3):
                        with Ts.sblock():
                            Ts.where(i + 16 - 3 < 16)
                            Ts.reads(B[0:4, tx, 0])
                            Ts.writes(C[tx, i - 3 + 16])
                            with T.attr(0, "async_wait_queue_scope", 0):
                                with T.attr(0, "async_wait_inflight_count", 2 - i):
                                    C[tx, i - 3 + 16] = B[(i - 3 + 16) % 4, tx, 0] + T.float32(1)

    tvm.ir.assert_structural_equal(mod["main"], ref.with_attr("global_symbol", "main"), True)


def test_async_producer_interleaving():
    @Ts.prim_func
    def simple_compute(
        A: T.Buffer((16, 16), "float32"),
        B: T.Buffer((16, 16), "float32"),
        C: T.Buffer((16, 16), "float32"),
    ):
        for tx in T.thread_binding(0, 16, thread="threadIdx.x"):
            for i in range(16):
                with Ts.sblock("compute"):
                    Ts.reads(A[tx, i])
                    Ts.writes(C[tx, i])
                    A_shared = Ts.sblock_alloc_buffer((16, 1), dtype="float32", scope="shared")
                    B_shared = Ts.sblock_alloc_buffer((16, 1), dtype="float32", scope="shared")
                    with Ts.sblock():
                        Ts.reads(A[tx, i])
                        Ts.writes(A_shared[tx, 0])
                        A_shared[tx, 0] = A[tx, i]
                    with Ts.sblock():
                        Ts.reads(B[tx, i])
                        Ts.writes(B_shared[tx, 0])
                        B_shared[tx, 0] = B[tx, i]
                    with Ts.sblock():
                        Ts.reads(A_shared[tx, 0], B_shared[tx, 0])
                        Ts.writes(C[tx, i])
                        C[tx, i] = A_shared[tx, 0] + B_shared[tx, 0]

    mod = tvm.IRModule.from_expr(simple_compute.with_attr("global_symbol", "main"))
    sch = tvm.s_tir.Schedule(mod)

    _, loop = sch.get_loops(sch.get_sblock("compute"))
    sch.annotate(loop, ann_key="software_pipeline_stage", ann_val=[0, 0, 3])
    sch.annotate(loop, ann_key="software_pipeline_order", ann_val=[0, 2, 1])
    sch.annotate(loop, ann_key="software_pipeline_async_stages", ann_val=[0])
    mod = tvm.s_tir.transform.InjectSoftwarePipeline()(sch.mod)

    @Ts.prim_func
    def ref(
        A: T.Buffer((16, 16), "float32"),
        B: T.Buffer((16, 16), "float32"),
        C: T.Buffer((16, 16), "float32"),
    ) -> None:
        for tx in T.thread_binding(16, thread="threadIdx.x"):
            with Ts.sblock():
                Ts.reads(A[tx, 0:16], B[tx, 0:16])
                Ts.writes(C[tx, 0:16])
                A_shared = Ts.sblock_alloc_buffer([4, 16, 1], dtype="float32", scope="shared")
                B_shared = Ts.sblock_alloc_buffer([4, 16, 1], dtype="float32", scope="shared")
                with Ts.sblock():
                    Ts.reads(A[tx, 0:3], B[tx, 0:3])
                    Ts.writes(A_shared[0:3, tx, 0], B_shared[0:3, tx, 0])
                    for i in T.unroll(3):
                        with Ts.sblock():
                            Ts.where(i < 16)
                            Ts.reads(A[tx, i], B[tx, i])
                            Ts.writes(A_shared[i % 4, tx, 0], B_shared[i % 4, tx, 0])
                            with T.attr(0, "async_commit_queue_scope", 0):
                                with T.attr(0, "async_scope", 1):
                                    A_shared[i % 4, tx, 0] = A[tx, i]
                                with T.attr(0, "async_scope", 1):
                                    B_shared[i % 4, tx, 0] = B[tx, i]
                with Ts.sblock():
                    Ts.reads(A[tx, 3:16], A_shared[0:4, tx, 0], B_shared[0:4, tx, 0], B[tx, 3:16])
                    Ts.writes(A_shared[0:4, tx, 0], C[tx, 0:13], B_shared[0:4, tx, 0])
                    for i in T.serial(13):
                        with Ts.sblock():
                            Ts.where(i + 3 < 16)
                            Ts.reads(A[tx, i + 3])
                            Ts.writes(A_shared[(i + 3) % 4, tx, 0])
                            with T.attr(0, "async_commit_queue_scope", 0):
                                with T.attr(0, "async_scope", 1):
                                    A_shared[(i + 3) % 4, tx, 0] = A[tx, i + 3]
                        with Ts.sblock():
                            Ts.where(i + 3 - 3 < 16)
                            Ts.reads(A_shared[0:4, tx, 0], B_shared[0:4, tx, 0])
                            Ts.writes(C[tx, i - 3 + 3])
                            with T.attr(0, "async_wait_queue_scope", 0):
                                with T.attr(0, "async_wait_inflight_count", 5):
                                    C[tx, i - 3 + 3] = (
                                        A_shared[(i - 3 + 3) % 4, tx, 0]
                                        + B_shared[(i - 3 + 3) % 4, tx, 0]
                                    )
                        with Ts.sblock():
                            Ts.where(i + 3 < 16)
                            Ts.reads(B[tx, i + 3])
                            Ts.writes(B_shared[(i + 3) % 4, tx, 0])
                            with T.attr(0, "async_commit_queue_scope", 0):
                                with T.attr(0, "async_scope", 1):
                                    B_shared[(i + 3) % 4, tx, 0] = B[tx, i + 3]
                with Ts.sblock():
                    Ts.reads(A_shared[0:4, tx, 0], B_shared[0:4, tx, 0])
                    Ts.writes(C[tx, 13:16])
                    for i in T.unroll(3):
                        with Ts.sblock():
                            Ts.where(i + 16 - 3 < 16)
                            Ts.reads(A_shared[0:4, tx, 0], B_shared[0:4, tx, 0])
                            Ts.writes(C[tx, i - 3 + 16])
                            with T.attr(0, "async_wait_queue_scope", 0):
                                with T.attr(0, "async_wait_inflight_count", 2 - i):
                                    C[tx, i - 3 + 16] = (
                                        A_shared[(i - 3 + 16) % 4, tx, 0]
                                        + B_shared[(i - 3 + 16) % 4, tx, 0]
                                    )

    tvm.ir.assert_structural_equal(mod["main"], ref.with_attr("global_symbol", "main"), True)


def test_three_stage_compute_two_stage_async():
    mod = tvm.IRModule.from_expr(three_stage_compute.with_attr("global_symbol", "main"))
    sch = tvm.s_tir.Schedule(mod)

    _, loop = sch.get_loops(sch.get_sblock("compute"))
    sch.annotate(loop, ann_key="software_pipeline_async_stages", ann_val=[0, 1])

    mod = tvm.s_tir.transform.InjectSoftwarePipeline()(sch.mod)

    @Ts.prim_func
    def ref(A: T.Buffer((16, 16), "float32"), D: T.Buffer((16, 16), "float32")) -> None:
        for tx in T.thread_binding(16, thread="threadIdx.x"):
            with Ts.sblock():
                Ts.reads(A[tx, 0:16])
                Ts.writes(D[tx, 0:16])
                B = Ts.sblock_alloc_buffer([2, 16, 1], dtype="float32", scope="shared")
                C = Ts.sblock_alloc_buffer([2, 16, 1], dtype="float32", scope="shared")
                with Ts.sblock():
                    Ts.reads(A[tx, 0:2], B[0:2, tx, 0])
                    Ts.writes(B[0:2, tx, 0], C[0:2, tx, 0])
                    for i in T.unroll(2):
                        with Ts.sblock():
                            Ts.where(i < 16)
                            Ts.reads(A[tx, i])
                            Ts.writes(B[i % 2, tx, 0])
                            with T.attr(0, "async_commit_queue_scope", 0):
                                with T.attr(0, "async_scope", 1):
                                    B[i % 2, tx, 0] = A[tx, i] * T.float32(2)
                        with Ts.sblock():
                            Ts.where(i == 1 and i - 1 < 16)
                            Ts.reads(B[(i - 1) % 2, tx, 0])
                            Ts.writes(C[(i - 1) % 2, tx, 0])
                            with T.attr(0, "async_commit_queue_scope", 1):
                                with T.attr(0, "async_wait_queue_scope", 0):
                                    with T.attr(0, "async_wait_inflight_count", 1):
                                        with T.attr(0, "async_scope", 1):
                                            C[(i - 1) % 2, tx, 0] = B[
                                                (i - 1) % 2, tx, 0
                                            ] + T.float32(2)
                with Ts.sblock():
                    Ts.reads(A[tx, 2:16], B[0:2, tx, 0], C[0:2, tx, 0])
                    Ts.writes(B[0:2, tx, 0], C[0:2, tx, 0], D[tx, 0:14])
                    for i in T.serial(14):
                        with Ts.sblock():
                            Ts.where(i + 2 < 16)
                            Ts.reads(A[tx, i + 2])
                            Ts.writes(B[(i + 2) % 2, tx, 0])
                            with T.attr(0, "async_commit_queue_scope", 0):
                                with T.attr(0, "async_scope", 1):
                                    B[(i + 2) % 2, tx, 0] = A[tx, i + 2] * T.float32(2)
                        with Ts.sblock():
                            Ts.where(i + 2 - 1 < 16)
                            Ts.reads(B[(i - 1 + 2) % 2, tx, 0])
                            Ts.writes(C[(i - 1 + 2) % 2, tx, 0])
                            with T.attr(0, "async_commit_queue_scope", 1):
                                with T.attr(0, "async_wait_queue_scope", 0):
                                    with T.attr(0, "async_wait_inflight_count", 1):
                                        with T.attr(0, "async_scope", 1):
                                            C[(i - 1 + 2) % 2, tx, 0] = B[
                                                (i - 1 + 2) % 2, tx, 0
                                            ] + T.float32(2)
                        with Ts.sblock():
                            Ts.where(i + 2 - 2 < 16)
                            Ts.reads(C[0:2, tx, 0])
                            Ts.writes(D[tx, i - 2 + 2])
                            with T.attr(0, "async_wait_queue_scope", 1):
                                with T.attr(0, "async_wait_inflight_count", 1):
                                    D[tx, i - 2 + 2] = C[(i - 2 + 2) % 2, tx, 0] + T.float32(1)
                with Ts.sblock():
                    Ts.reads(B[0:2, tx, 0], C[0:2, tx, 0])
                    Ts.writes(C[0:2, tx, 0], D[tx, 14:16])
                    for i in T.unroll(2):
                        with Ts.sblock():
                            Ts.where(i + 16 - 1 < 16)
                            Ts.reads(B[(i - 1 + 16) % 2, tx, 0])
                            Ts.writes(C[(i - 1 + 16) % 2, tx, 0])
                            with T.attr(0, "async_commit_queue_scope", 1):
                                with T.attr(0, "async_wait_queue_scope", 0):
                                    with T.attr(0, "async_wait_inflight_count", 0 - i):
                                        with T.attr(0, "async_scope", 1):
                                            C[(i - 1 + 16) % 2, tx, 0] = B[
                                                (i - 1 + 16) % 2, tx, 0
                                            ] + T.float32(2)
                        with Ts.sblock():
                            Ts.where(i + 16 - 2 < 16)
                            Ts.reads(C[0:2, tx, 0])
                            Ts.writes(D[tx, i - 2 + 16])
                            with T.attr(0, "async_wait_queue_scope", 1):
                                with T.attr(
                                    0,
                                    "async_wait_inflight_count",
                                    T.if_then_else(i + 16 - 1 < 16, 1, 0, dtype="int32"),
                                ):
                                    D[tx, i - 2 + 16] = C[(i - 2 + 16) % 2, tx, 0] + T.float32(1)

    tvm.ir.assert_structural_equal(mod["main"], ref.with_attr("global_symbol", "main"), True)


N = K = M = 4096


def get_mma_schedule():
    i_factors, j_factors, k_factors = [1, 32, 1, 4, 2], [16, 2, 4, 1, 2], [128, 2, 1]

    def index_map(i, j):
        return (
            i // 16,
            j // 16,
            *shared_16x16_to_ldmatrix_32x8_layout(i % 16, j % 16),
        )

    workload = te.create_prim_func(
        te_workload.matmul(N, M, K, in_dtype="float16", out_dtype="float32")
    )

    return mma_schedule(
        workload,
        16,
        "float16",
        False,
        i_factors,
        j_factors,
        k_factors,
        index_map,
        index_map,
        index_map,
        LDMATRIX_f16_A_DYN_INTRIN,
        LDMATRIX_f16_B_DYN_INTRIN,
        MMA_f16f16f32_INTRIN,
        MMA_fill_16x16_f32_INTRIN,
        MMA_store_16x16_f32_global_INTRIN,
        "shared.dyn",
    )


def build_and_run(sch):
    if tvm.testing.is_ampere_or_newer():
        with tvm.transform.PassContext(config={"tirx.use_async_copy": 1}):
            f = tvm.compile(sch.mod["main"], target="cuda")

        a_np = np.random.uniform(size=(N, K)).astype("float16")
        b_np = np.random.uniform(size=(K, M)).astype("float16")
        c_np = np.dot(a_np.astype("float32"), b_np.astype("float32"))

        def run_and_check():
            dev = tvm.cuda(0)
            a = tvm.runtime.tensor(a_np, dev)
            b = tvm.runtime.tensor(b_np, dev)
            c = tvm.runtime.tensor(np.zeros((N, M), dtype="float32"), dev)
            f(a, b, c)
            tvm.testing.assert_allclose(c.numpy(), c_np, rtol=1e-3)

        tvm.testing.run_with_gpu_lock(run_and_check)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
def test_async_pipelined_mma_gemm_simple():
    sch = get_mma_schedule()

    k0 = sch.get_loops(sch.get_sblock("C_o_update"))[3]

    sch.annotate(k0, ann_key="software_pipeline_stage", ann_val=[0, 0, 3])
    sch.annotate(k0, ann_key="software_pipeline_order", ann_val=[0, 1, 2])
    sch.annotate(k0, ann_key="software_pipeline_async_stages", ann_val=[0])

    seq = tvm.transform.Sequential(
        [
            tvm.s_tir.transform.PlanAndUpdateBufferAllocationLocation(),
            tvm.s_tir.transform.ConvertBlocksToOpaque(),
            tvm.s_tir.transform.UnifyThreadBinding(),
            tvm.s_tir.transform.LowerMatchBuffer(),
            tvm.s_tir.transform.InjectSoftwarePipeline(),
        ]
    )
    mod = seq(sch.mod)

    pipeline = mod["main"].body.block.body.body.body.body.body.block.body[1].block.body
    prologue, body, epilogue = pipeline

    commit_queue_scope = prologue.block.body.body.block.body
    assert len(commit_queue_scope.body) == 2
    assert commit_queue_scope.value == 0

    commit_queue_scope = body.block.body.body[0].block.body
    assert len(commit_queue_scope.body) == 2
    assert commit_queue_scope.value == 0

    assert body.block.body.body[1].block.body.body.attr_key == "async_wait_inflight_count"
    assert body.block.body.body[1].block.body.body.value == 3

    assert epilogue.block.body.body.block.body.body.attr_key == "async_wait_inflight_count"
    assert str(epilogue.block.body.body.block.body.body.value) == "2 - k_0_0"

    build_and_run(sch)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
def test_async_nested_pipeline_mma_gemm_ideal_annotation():
    sch = get_mma_schedule()

    k0 = sch.get_loops(sch.get_sblock("C_o_update"))[3]
    k1 = sch.get_loops(sch.get_sblock("C_o_update"))[4]

    sch.annotate(k0, ann_key="software_pipeline_stage", ann_val=[0, 0, 2, 3, 3])
    sch.annotate(k0, ann_key="software_pipeline_order", ann_val=[0, 1, 3, 2, 4])
    sch.annotate(k0, ann_key="software_pipeline_async_stages", ann_val=[0])

    sch.annotate(k1, ann_key="software_pipeline_stage", ann_val=[0, 0, 1])
    sch.annotate(k1, ann_key="software_pipeline_order", ann_val=[0, 1, 2])

    seq = tvm.transform.Sequential(
        [
            tvm.s_tir.transform.PlanAndUpdateBufferAllocationLocation(),
            tvm.s_tir.transform.ConvertBlocksToOpaque(),
            tvm.s_tir.transform.UnifyThreadBinding(),
            tvm.s_tir.transform.LowerMatchBuffer(),
            tvm.s_tir.transform.InjectSoftwarePipeline(),
        ]
    )
    mod = seq(sch.mod)

    pipeline = mod["main"].body.block.body.body.body.body.body.block.body[1].block.body
    prologue, body, epilogue = pipeline

    commit_queue_scope = prologue.block.body.body[0].block.body
    assert len(commit_queue_scope.body) == 2
    assert commit_queue_scope.value == 0

    assert prologue.block.body.body[1].block.body.body.attr_key == "async_wait_inflight_count"
    assert prologue.block.body.body[1].block.body.body.value == 2

    commit_queue_scope = body.block.body.body[0].block.body
    assert len(commit_queue_scope.body) == 2
    assert commit_queue_scope.value == 0

    assert body.block.body.body[1].block.body.body.attr_key == "async_wait_inflight_count"
    assert body.block.body.body[1].block.body.body.value == 2

    assert str(epilogue.block.body.body[0].block.body.body.value) == "1 - k_0_0"

    build_and_run(sch)


def test_less_loop_than_num_stage():
    @Ts.prim_func
    def before(A: T.Buffer((2,), "float32"), E: T.Buffer((2,), "float32")):
        for i in T.serial(
            0,
            2,
            annotations={
                "software_pipeline_stage": [0, 1, 2, 3],
                "software_pipeline_order": [0, 1, 2, 3],
            },
        ):
            with Ts.sblock("compute"):
                B = Ts.sblock_alloc_buffer((1), dtype="float32", scope="shared")
                C = Ts.sblock_alloc_buffer((1), dtype="float32", scope="shared")
                D = Ts.sblock_alloc_buffer((1), dtype="float32", scope="shared")
                with Ts.sblock():
                    B[0] = A[i] * T.float32(2)
                with Ts.sblock():
                    C[0] = B[0] + T.float32(3)
                with Ts.sblock():
                    D[0] = C[0] + T.float32(4)
                with Ts.sblock():
                    E[i] = D[0] + T.float32(5)

    @Ts.prim_func
    def after(A: T.Buffer((2,), "float32"), E: T.Buffer((2,), "float32")):
        with Ts.sblock("root"):
            Ts.reads()
            Ts.writes()
            with Ts.sblock(""):
                Ts.reads(A[0:3])
                Ts.writes(E[0:2])
                B = Ts.sblock_alloc_buffer((2, 1), scope="shared")
                C = Ts.sblock_alloc_buffer((2, 1), scope="shared")
                D = Ts.sblock_alloc_buffer((2, 1), scope="shared")
                with Ts.sblock(""):
                    Ts.reads(A[0:3], B[0:2, 0], C[0:2, 0])
                    Ts.writes(B[0:2, 0], C[0:2, 0], D[0:2, 0])
                    for i in T.unroll(3):
                        with Ts.sblock(""):
                            Ts.where(i < 2)
                            Ts.reads(A[i])
                            Ts.writes(B[0:2, 0])
                            B[i % 2, 0] = A[i] * T.float32(2.0)
                        with Ts.sblock(""):
                            Ts.where(1 <= i)
                            Ts.reads(B[0:2, 0])
                            Ts.writes(C[0:2, 0])
                            C[(i + 1) % 2, 0] = B[(i + 1) % 2, 0] + T.float32(3.0)
                        with Ts.sblock(""):
                            Ts.where(i == 2)
                            Ts.reads(C[0:2, 0])
                            Ts.writes(D[0:2, 0])
                            D[i % 2, 0] = C[i % 2, 0] + T.float32(4.0)
                with Ts.sblock(""):
                    Ts.reads()
                    Ts.writes()
                    T.evaluate(0)
                with Ts.sblock(""):
                    Ts.reads(C[0:2, 0], D[0:2, 0])
                    Ts.writes(D[0:2, 0], E[0:2])
                    for i in T.unroll(2):
                        with Ts.sblock(""):
                            Ts.where(i < 1)
                            Ts.reads(C[0:2, 0])
                            Ts.writes(D[0:2, 0])
                            D[(i + 1) % 2, 0] = C[(i + 1) % 2, 0] + T.float32(4.0)
                        with Ts.sblock(""):
                            Ts.reads(D[0:2, 0])
                            Ts.writes(E[i])
                            E[i] = D[i, 0] + T.float32(5.0)

    _check(before, after)


def test_less_loop_than_num_stage_dynamic():
    @Ts.prim_func
    def before(a: T.handle, b: T.handle):
        K = T.int32()
        A = T.match_buffer(a, [K], "float32")
        E = T.match_buffer(b, [K], "float32")
        for i in T.serial(
            0,
            K,
            annotations={
                "software_pipeline_stage": [0, 1, 2, 3],
                "software_pipeline_order": [0, 1, 2, 3],
            },
        ):
            with Ts.sblock("compute"):
                B = Ts.sblock_alloc_buffer((1), dtype="float32", scope="shared")
                C = Ts.sblock_alloc_buffer((1), dtype="float32", scope="shared")
                D = Ts.sblock_alloc_buffer((1), dtype="float32", scope="shared")
                with Ts.sblock():
                    B[0] = A[i] * T.float32(2)
                with Ts.sblock():
                    C[0] = B[0] + T.float32(3)
                with Ts.sblock():
                    D[0] = C[0] + T.float32(4)
                with Ts.sblock():
                    E[i] = D[0] + T.float32(5)

    @Ts.prim_func
    def after(a: T.handle, b: T.handle):
        K = T.int32()
        A = T.match_buffer(a, [K], "float32")
        E = T.match_buffer(b, [K], "float32")
        with Ts.sblock("root"):
            Ts.reads()
            Ts.writes()
            with Ts.sblock(""):
                Ts.reads(A[0 : T.max(3, K)])
                Ts.writes(E[T.min(0, K - 3) : T.min(0, K - 3) + T.max(K, 3)])
                B = Ts.sblock_alloc_buffer((2, 1), scope="shared")
                C = Ts.sblock_alloc_buffer((2, 1), scope="shared")
                D = Ts.sblock_alloc_buffer((2, 1), scope="shared")
                with Ts.sblock(""):
                    Ts.reads(A[0:3], B[0:2, 0], C[0:2, 0])
                    Ts.writes(B[0:2, 0], C[0:2, 0], D[0:2, 0])
                    for i in T.unroll(3):
                        with Ts.sblock(""):
                            Ts.where(i < K)
                            Ts.reads(A[i])
                            Ts.writes(B[0:2, 0])
                            B[i % 2, 0] = A[i] * T.float32(2.0)
                        with Ts.sblock(""):
                            Ts.where(1 <= i and i <= K)
                            Ts.reads(B[0:2, 0])
                            Ts.writes(C[0:2, 0])
                            C[(i + 1) % 2, 0] = B[(i + 1) % 2, 0] + T.float32(3.0)
                        with Ts.sblock(""):
                            Ts.where(i == 2 and i < K + 2)
                            Ts.reads(C[0:2, 0])
                            Ts.writes(D[0:2, 0])
                            D[i % 2, 0] = C[i % 2, 0] + T.float32(4.0)
                with Ts.sblock(""):
                    Ts.reads(A[3 : 3 + (K - 3)], B[0:2, 0], C[0:2, 0], D[0:2, 0])
                    Ts.writes(B[0:2, 0], C[0:2, 0], D[0:2, 0], E[0 : K - 3])
                    for i in range(K - 3):
                        with Ts.sblock(""):
                            Ts.reads(A[i + 3])
                            Ts.writes(B[0:2, 0])
                            B[(i + 1) % 2, 0] = A[i + 3] * T.float32(2.0)
                        with Ts.sblock(""):
                            Ts.reads(B[0:2, 0])
                            Ts.writes(C[0:2, 0])
                            C[i % 2, 0] = B[i % 2, 0] + T.float32(3.0)
                        with Ts.sblock(""):
                            Ts.reads(C[0:2, 0])
                            Ts.writes(D[0:2, 0])
                            D[(i + 1) % 2, 0] = C[(i + 1) % 2, 0] + T.float32(4.0)
                        with Ts.sblock(""):
                            Ts.reads(D[0:2, 0])
                            Ts.writes(E[i])
                            E[i] = D[i % 2, 0] + T.float32(5.0)
                with Ts.sblock(""):
                    Ts.reads(B[0:2, 0], C[0:2, 0], D[0:2, 0])
                    Ts.writes(C[0:2, 0], D[0:2, 0], E[K - 3 : K - 3 + 3])
                    for i in T.unroll(3):
                        with Ts.sblock(""):
                            Ts.where(1 <= i + K and i + K == K and 3 <= i + K)
                            Ts.reads(B[0:2, 0])
                            Ts.writes(C[0:2, 0])
                            C[(i + K + 1) % 2, 0] = B[(i + K + 1) % 2, 0] + T.float32(3.0)
                        with Ts.sblock(""):
                            Ts.where(2 <= i + K and i < 2 and 3 <= i + K)
                            Ts.reads(C[0:2, 0])
                            Ts.writes(D[0:2, 0])
                            D[(i + K) % 2, 0] = C[(i + K) % 2, 0] + T.float32(4.0)
                        with Ts.sblock(""):
                            Ts.where(3 <= i + K and 3 <= i + K)
                            Ts.reads(D[0:2, 0])
                            Ts.writes(E[i + K - 3])
                            E[i + K - 3] = D[(i + K + 1) % 2, 0] + T.float32(5.0)

    _check(before, after)


if __name__ == "__main__":
    tvm.testing.main()
