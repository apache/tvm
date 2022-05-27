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
import pytest
import sys
import numpy as np

import tvm
import tvm.testing
import tvm.tir.tensor_intrin.cuda
from tvm import tir, te, TVMError
from tvm.script import tir as T


def _check(original, transformed):
    func = original
    mod = tvm.IRModule.from_expr(func)
    mod = tvm.tir.transform.InjectSoftwarePipeline()(mod)
    mod = tvm.tir.transform.Simplify()(mod)
    tvm.ir.assert_structural_equal(mod["main"], transformed, True)


def _check_error(func):
    mod = tvm.IRModule.from_expr(func)
    with pytest.raises(ValueError):
        tvm.tir.transform.InjectSoftwarePipeline()(mod)


@T.prim_func
def trivial_pipeline(A: T.Buffer[(16, 1), "float32"], C: T.Buffer[(16, 1), "float32"]):
    for tx in T.thread_binding(0, 16, thread="threadIdx.x"):
        for i in T.serial(
            0, 1, annotations={"software_pipeline_stage": [0, 1], "software_pipeline_order": [0, 1]}
        ):
            with T.block():
                T.reads(A[tx, i])
                T.writes(C[tx, i])
                B = T.alloc_buffer((16, 1), dtype="float32", scope="shared")
                with T.block():
                    T.reads(A[tx, i])
                    T.writes(B[tx, 0])
                    B[tx, 0] = A[tx, i] * T.float32(2)
                with T.block():
                    T.reads(B[tx, 0])
                    T.writes(C[tx, i])
                    C[tx, i] = B[tx, 0] + T.float32(1)


@T.prim_func
def transformed_trivial_pipeline(
    A: T.Buffer[(16, 1), "float32"], C: T.Buffer[(16, 1), "float32"]
) -> None:
    for tx in T.thread_binding(16, thread="threadIdx.x"):
        with T.block():
            T.reads(A[tx, 0])
            T.writes(C[tx, 0])
            B = T.alloc_buffer([2, 16, 1], dtype="float32", scope="shared")
            with T.block():
                T.reads(A[tx, 0])
                T.writes(B[0, tx, 0])
                B[0, tx, 0] = A[tx, 0] * T.float32(2)
            with T.block():
                T.reads()
                T.writes()
                T.evaluate(0)
            with T.block():
                T.reads(B[0, tx, 0])
                T.writes(C[tx, 0])
                C[tx, 0] = B[0, tx, 0] + T.float32(1)


@T.prim_func
def simple_compute(A: T.Buffer[(16, 16), "float32"], C: T.Buffer[(16, 16), "float32"]):
    for tx in T.thread_binding(0, 16, thread="threadIdx.x"):
        for i in T.serial(
            0,
            16,
            annotations={"software_pipeline_stage": [0, 1], "software_pipeline_order": [0, 1]},
        ):
            with T.block():
                T.reads(A[tx, i])
                T.writes(C[tx, i])
                B = T.alloc_buffer((16, 1), dtype="float32", scope="shared")
                with T.block():
                    T.reads(A[tx, i])
                    T.writes(B[tx, 0])
                    B[tx, 0] = A[tx, i] * T.float32(2)
                with T.block():
                    T.reads(B[tx, 0])
                    T.writes(C[tx, i])
                    C[tx, i] = B[tx, 0] + T.float32(1)


@T.prim_func
def transformed_simple_compute(
    A: T.Buffer[(16, 16), "float32"], C: T.Buffer[(16, 16), "float32"]
) -> None:
    for tx in T.thread_binding(0, 16, thread="threadIdx.x"):
        with T.block():
            T.reads([A[tx, 0:16]])
            T.writes([C[tx, 0:16]])
            B = T.alloc_buffer([2, 16, 1], dtype="float32", scope="shared")
            with T.block():
                T.reads([A[tx, 0]])
                T.writes([B[0, tx, 0]])
                B[0, tx, 0] = A[tx, 0] * T.float32(2)
            with T.block():
                T.reads([A[tx, 1:16], B[0:2, tx, 0]])
                T.writes([B[0:2, tx, 0], C[tx, 0:15]])
                for i in T.serial(0, 15):
                    with T.block():
                        T.reads([A[tx, i + 1]])
                        T.writes([B[(i + 1) % 2, tx, 0]])
                        B[(i + 1) % 2, tx, 0] = A[tx, i + 1] * T.float32(2)
                    with T.block():
                        T.reads([B[i % 2, tx, 0]])
                        T.writes([C[tx, i]])
                        C[tx, i] = B[i % 2, tx, 0] + T.float32(1)
            with T.block():
                T.reads([B[1, tx, 0]])
                T.writes([C[tx, 15]])
                C[tx, 15] = B[1, tx, 0] + T.float32(1)


@T.prim_func
def three_stage_compute(A: T.Buffer[(16, 16), "float32"], D: T.Buffer[(16, 16), "float32"]):
    for tx in T.thread_binding(0, 16, thread="threadIdx.x"):
        for i in T.serial(
            0,
            16,
            annotations={
                "software_pipeline_stage": [0, 1, 2],
                "software_pipeline_order": [0, 1, 2],
            },
        ):
            with T.block():
                T.reads(A[tx, i])
                T.writes(D[tx, i])
                B = T.alloc_buffer((16, 1), dtype="float32", scope="shared")
                C = T.alloc_buffer((16, 1), dtype="float32", scope="shared")
                with T.block():
                    T.reads(A[tx, i])
                    T.writes(B[tx, 0])
                    B[tx, 0] = A[tx, i] * T.float32(2)
                with T.block():
                    T.reads(B[tx, 0])
                    T.writes(C[tx, 0])
                    C[tx, 0] = B[tx, 0] + T.float32(2)
                with T.block():
                    T.reads(C[tx, 0])
                    T.writes(D[tx, i])
                    D[tx, i] = C[tx, 0] + T.float32(1)


@T.prim_func
def transformed_three_stage_compute(
    A: T.Buffer[(16, 16), "float32"], D: T.Buffer[(16, 16), "float32"]
) -> None:
    for tx in T.thread_binding(16, thread="threadIdx.x"):
        with T.block():
            T.reads(A[tx, 0:16])
            T.writes(D[tx, 0:16])
            B = T.alloc_buffer([2, 16, 1], dtype="float32", scope="shared")
            C = T.alloc_buffer([2, 16, 1], dtype="float32", scope="shared")
            with T.block():
                T.reads(A[tx, 0:2], B[0:2, tx, 0])
                T.writes(B[0:2, tx, 0], C[0:2, tx, 0])
                for i in T.unroll(2):
                    with T.block():
                        T.reads(A[tx, i])
                        T.writes(B[0:2, tx, 0])
                        B[i, tx, 0] = A[tx, i] * T.float32(2)
                    with T.block():
                        T.where(1 <= i)
                        T.reads(B[0:2, tx, 0])
                        T.writes(C[0:2, tx, 0])
                        C[(i + 1) % 2, tx, 0] = B[(i + 1) % 2, tx, 0] + T.float32(2)
            with T.block():
                T.reads(A[tx, 2:16], B[0:2, tx, 0], C[0:2, tx, 0])
                T.writes(B[0:2, tx, 0], C[0:2, tx, 0], D[tx, 0:14])
                for i in T.serial(14):
                    with T.block():
                        T.reads(A[tx, i + 2])
                        T.writes(B[0:2, tx, 0])
                        B[i % 2, tx, 0] = A[tx, i + 2] * T.float32(2)
                    with T.block():
                        T.reads(B[0:2, tx, 0])
                        T.writes(C[0:2, tx, 0])
                        C[(i + 1) % 2, tx, 0] = B[(i + 1) % 2, tx, 0] + T.float32(2)
                    with T.block():
                        T.reads(C[0:2, tx, 0])
                        T.writes(D[tx, i])
                        D[tx, i] = C[i % 2, tx, 0] + T.float32(1)
            with T.block():
                T.reads(B[0:2, tx, 0], C[0:2, tx, 0])
                T.writes(C[0:2, tx, 0], D[tx, 14:16])
                for i in T.unroll(2):
                    with T.block():
                        T.where(i < 1)
                        T.reads(B[0:2, tx, 0])
                        T.writes(C[0:2, tx, 0])
                        C[(i + 1) % 2, tx, 0] = B[(i + 1) % 2, tx, 0] + T.float32(2)
                    with T.block():
                        T.reads(C[0:2, tx, 0])
                        T.writes(D[tx, i + 14])
                        D[tx, i + 14] = C[i, tx, 0] + T.float32(1)


@T.prim_func
def dag_interleaving(
    A: T.Buffer[(16, 16), "float32"],
    B: T.Buffer[(16, 16), "float32"],
    C: T.Buffer[(16, 16), "float32"],
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
            with T.block():
                T.reads(A[tx, i])
                T.writes(C[tx, i])
                AS = T.alloc_buffer((16, 1), dtype="float32", scope="shared")
                BS = T.alloc_buffer((16, 1), dtype="float32", scope="shared")
                AL = T.alloc_buffer((1, 1), dtype="float32", scope="local")
                BL = T.alloc_buffer((1, 1), dtype="float32", scope="local")
                with T.block():
                    T.reads(A[tx, i])
                    T.writes(AS[tx, 0])
                    AS[tx, 0] = A[tx, i] * T.float32(2)
                with T.block():
                    T.reads(AS[tx, 0])
                    T.writes(AL[0, 0])
                    AL[0, 0] = AS[tx, 0]
                with T.block():
                    T.reads(B[tx, i])
                    T.writes(BS[tx, 0])
                    BS[tx, 0] = B[tx, i] + T.float32(2)
                with T.block():
                    T.reads(BS[tx, 0])
                    T.writes(BL[0, 0])
                    BL[0, 0] = BS[tx, 0]
                with T.block():
                    T.reads(AL[0, 0], BL[0, 0])
                    T.writes(C[tx, i])
                    C[tx, i] = AL[0, 0] * BL[0, 0]


@T.prim_func
def transformed_dag_interleaving(
    A: T.Buffer[(16, 16), "float32"],
    B: T.Buffer[(16, 16), "float32"],
    C: T.Buffer[(16, 16), "float32"],
) -> None:
    for tx in T.thread_binding(16, thread="threadIdx.x"):
        with T.block():
            T.reads(A[tx, 0:16], B[tx, 0:16])
            T.writes(C[tx, 0:16])
            AS = T.alloc_buffer([16, 1], dtype="float32", scope="shared")
            BS = T.alloc_buffer([16, 1], dtype="float32", scope="shared")
            AL = T.alloc_buffer([2, 1, 1], dtype="float32", scope="local")
            BL = T.alloc_buffer([2, 1, 1], dtype="float32", scope="local")
            with T.block():
                T.reads(A[tx, 0], B[tx, 0], AS[tx, 0], BS[tx, 0])
                T.writes(AS[tx, 0], BS[tx, 0], AL[0, 0, 0], BL[0, 0, 0])
                with T.block():
                    T.reads(A[tx, 0])
                    T.writes(AS[tx, 0])
                    AS[tx, 0] = A[tx, 0] * T.float32(2)
                with T.block():
                    T.reads(B[tx, 0])
                    T.writes(BS[tx, 0])
                    BS[tx, 0] = B[tx, 0] + T.float32(2)
                with T.block():
                    T.reads(AS[tx, 0])
                    T.writes(AL[0, 0, 0])
                    AL[0, 0, 0] = AS[tx, 0]
                with T.block():
                    T.reads(BS[tx, 0])
                    T.writes(BL[0, 0, 0])
                    BL[0, 0, 0] = BS[tx, 0]
            with T.block():
                T.reads(
                    A[tx, 1:16], B[tx, 1:16], AS[tx, 0], BS[tx, 0], AL[0:2, 0, 0], BL[0:2, 0, 0]
                )
                T.writes(AS[tx, 0], BS[tx, 0], AL[0:2, 0, 0], BL[0:2, 0, 0], C[tx, 0:15])
                for i in T.serial(15):
                    with T.block():
                        T.reads(A[tx, i + 1])
                        T.writes(AS[tx, 0])
                        AS[tx, 0] = A[tx, i + 1] * T.float32(2)
                    with T.block():
                        T.reads(B[tx, i + 1])
                        T.writes(BS[tx, 0])
                        BS[tx, 0] = B[tx, i + 1] + T.float32(2)
                    with T.block():
                        T.reads(AS[tx, 0])
                        T.writes(AL[(i + 1) % 2, 0, 0])
                        AL[(i + 1) % 2, 0, 0] = AS[tx, 0]
                    with T.block():
                        T.reads(BS[tx, 0])
                        T.writes(BL[(i + 1) % 2, 0, 0])
                        BL[(i + 1) % 2, 0, 0] = BS[tx, 0]
                    with T.block():
                        T.reads(AL[i % 2, 0, 0], BL[i % 2, 0, 0])
                        T.writes(C[tx, i])
                        C[tx, i] = AL[i % 2, 0, 0] * BL[i % 2, 0, 0]
            with T.block():
                T.reads(AL[1, 0, 0], BL[1, 0, 0])
                T.writes(C[tx, 15])
                C[tx, 15] = AL[1, 0, 0] * BL[1, 0, 0]


@T.prim_func
def nested_pipeline_simple(
    A: T.Buffer[(16, 16, 16), "float32"], C: T.Buffer[(16, 16, 16), "float32"]
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
            with T.block():
                T.reads(A[tx, i, 0:16])
                T.writes(C[tx, i, 0:16])
                A_shared = T.alloc_buffer((16, 1, 16), dtype="float32", scope="shared")
                for j in T.serial(0, 16):
                    with T.block():
                        T.reads(A[tx, i, j])
                        T.writes(A_shared[tx, 0, j])
                        A_shared[tx, 0, j] = A[tx, i, j]
                for j in T.serial(
                    0,
                    16,
                    annotations={
                        "software_pipeline_stage": [0, 1],
                        "software_pipeline_order": [0, 1],
                    },
                ):
                    with T.block():
                        T.reads(A_shared[tx, 0, j])
                        T.writes(C[tx, i, j])
                        B = T.alloc_buffer((16, 1, 1), dtype="float32", scope="shared")
                        with T.block():
                            T.reads(A_shared[tx, i, j])
                            T.writes(B[tx, i, 0])
                            B[tx, i, 0] = A_shared[tx, 0, j] * T.float32(2)
                        with T.block():
                            T.reads(B[tx, i, 0])
                            T.writes(C[tx, i, j])
                            C[tx, i, j] = B[tx, i, 0] + T.float32(1)


@T.prim_func
def transformed_nested_pipeline_simple(
    A: T.Buffer[(16, 16, 16), "float32"], C: T.Buffer[(16, 16, 16), "float32"]
) -> None:
    for tx in T.thread_binding(0, 16, thread="threadIdx.x"):
        with T.block():
            T.reads([A[tx, 0:16, 0:16]])
            T.writes([C[tx, 0:16, 0:16]])
            A_shared = T.alloc_buffer([2, 16, 1, 16], dtype="float32", scope="shared")
            B = T.alloc_buffer([2, 16, 1, 1], dtype="float32", scope="shared")
            with T.block():
                T.reads([A[tx, 0, 0:16]])
                T.writes([A_shared[0, tx, 0, 0:16]])
                for j in T.serial(0, 16):
                    with T.block():
                        T.reads([A[tx, 0, j]])
                        T.writes([A_shared[0, tx, 0, j]])
                        A_shared[0, tx, 0, j] = A[tx, 0, j]
            with T.block():
                T.reads([A[tx, 1:16, 0:16], A_shared[0:2, tx, 0:15, 0:16], B[0:2, tx, 0:15, 0]])
                T.writes([A_shared[0:2, tx, 0, 0:16], B[0:2, tx, 0:15, 0], C[tx, 0:15, 0:16]])
                for i in T.serial(0, 15):
                    with T.block():
                        T.reads([A[tx, i + 1, 0:16]])
                        T.writes([A_shared[(i + 1) % 2, tx, 0, 0:16]])
                        for j in T.serial(0, 16):
                            with T.block():
                                T.reads([A[tx, i + 1, j]])
                                T.writes([A_shared[(i + 1) % 2, tx, 0, j]])
                                A_shared[(i + 1) % 2, tx, 0, j] = A[tx, i + 1, j]
                    with T.block():
                        T.reads([A_shared[i % 2, tx, i, 0]])
                        T.writes([B[0, tx, i, 0]])
                        B[0, tx, i, 0] = A_shared[i % 2, tx, 0, 0] * T.float32(2)
                    with T.block():
                        T.reads([A_shared[i % 2, tx, i, 1:16], B[0:2, tx, i, 0]])
                        T.writes([B[0:2, tx, i, 0], C[tx, i, 0:15]])
                        for j in T.serial(0, 15):
                            with T.block():
                                T.reads([A_shared[i % 2, tx, i, j + 1]])
                                T.writes([B[(j + 1) % 2, tx, i, 0]])
                                B[(j + 1) % 2, tx, i, 0] = A_shared[
                                    i % 2, tx, 0, j + 1
                                ] * T.float32(2)
                            with T.block():
                                T.reads([B[j % 2, tx, i, 0]])
                                T.writes([C[tx, i, j]])
                                C[tx, i, j] = B[j % 2, tx, i, 0] + T.float32(1)
                    with T.block():
                        T.reads([B[1, tx, i, 0]])
                        T.writes([C[tx, i, 15]])
                        C[tx, i, 15] = B[1, tx, i, 0] + T.float32(1)
            with T.block():
                T.reads([A_shared[1, tx, 15, 0:16], B[0:2, tx, 15, 0]])
                T.writes([B[0:2, tx, 15, 0], C[tx, 15, 0:16]])
                with T.block():
                    T.reads([A_shared[1, tx, 15, 0]])
                    T.writes([B[0, tx, 15, 0]])
                    B[0, tx, 15, 0] = A_shared[1, tx, 0, 0] * T.float32(2)
                with T.block():
                    T.reads([A_shared[1, tx, 15, 1:16], B[0:2, tx, 15, 0]])
                    T.writes([B[0:2, tx, 15, 0], C[tx, 15, 0:15]])
                    for j in T.serial(0, 15):
                        with T.block():
                            T.reads([A_shared[1, tx, 15, j + 1]])
                            T.writes([B[(j + 1) % 2, tx, 15, 0]])
                            B[(j + 1) % 2, tx, 15, 0] = A_shared[1, tx, 0, j + 1] * T.float32(2)
                        with T.block():
                            T.reads([B[j % 2, tx, 15, 0]])
                            T.writes([C[tx, 15, j]])
                            C[tx, 15, j] = B[j % 2, tx, 15, 0] + T.float32(1)
                with T.block():
                    T.reads([B[1, tx, 15, 0]])
                    T.writes([C[tx, 15, 15]])
                    C[tx, 15, 15] = B[1, tx, 15, 0] + T.float32(1)


@T.prim_func
def nested_pipeline_prefetch_inner(
    A: T.Buffer[(16, 16, 16), "float32"], C: T.Buffer[(16, 16, 16), "float32"]
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
            with T.block():
                T.reads(A[tx, i, 0:16])
                T.writes(C[tx, i, 0:16])
                A_shared = T.alloc_buffer((16, 1, 16), dtype="float32", scope="shared")
                for j in T.serial(0, 16):
                    with T.block():
                        T.reads(A[tx, i, j])
                        T.writes(A_shared[tx, 0, j])
                        A_shared[tx, 0, j] = A[tx, i, j]
                for j in T.serial(
                    0,
                    16,
                    annotations={
                        "software_pipeline_stage": [0, 1],
                        "software_pipeline_order": [0, 1],
                    },
                ):
                    with T.block():
                        T.reads(A_shared[tx, 0, j])
                        T.writes(C[tx, i, j])
                        B = T.alloc_buffer((16, 1, 1), dtype="float32", scope="shared")
                        with T.block():
                            T.reads(A_shared[tx, i, j])
                            T.writes(B[tx, i, 0])
                            B[tx, i, 0] = A_shared[tx, 0, j] * T.float32(2)
                        with T.block():
                            T.reads(B[tx, i, 0])
                            T.writes(C[tx, i, j])
                            C[tx, i, j] = B[tx, i, 0] + T.float32(1)


@T.prim_func
def transformed_nested_pipeline_prefetch_inner(
    A: T.Buffer[(16, 16, 16), "float32"], C: T.Buffer[(16, 16, 16), "float32"]
) -> None:
    for tx in T.thread_binding(0, 16, thread="threadIdx.x"):
        with T.block():
            T.reads([A[tx, 0:16, 0:16]])
            T.writes([C[tx, 0:16, 0:16]])
            A_shared = T.alloc_buffer([2, 16, 1, 16], dtype="float32", scope="shared")
            B = T.alloc_buffer([2, 16, 1, 1], dtype="float32", scope="shared")
            with T.block():
                T.reads([A[tx, 0, 0:16], A_shared[0, tx, 0, 0]])
                T.writes([A_shared[0, tx, 0, 0:16], B[0, tx, 0, 0]])
                with T.block():
                    T.reads([A[tx, 0, 0:16]])
                    T.writes([A_shared[0, tx, 0, 0:16]])
                    for j in T.serial(0, 16):
                        with T.block():
                            T.reads([A[tx, 0, j]])
                            T.writes([A_shared[0, tx, 0, j]])
                            A_shared[0, tx, 0, j] = A[tx, 0, j]
                with T.block():
                    T.reads([A_shared[0, tx, 0, 0]])
                    T.writes([B[0, tx, 0, 0]])
                    B[0, tx, 0, 0] = A_shared[0, tx, 0, 0] * T.float32(2)
            with T.block():
                T.reads([A[tx, 1:16, 0:16], A_shared[0:2, tx, 0:16, 0:16], B[0:2, tx, 0:15, 0]])
                T.writes([A_shared[0:2, tx, 0, 0:16], B[0:2, tx, 0:16, 0], C[tx, 0:15, 0:16]])
                for i in T.serial(0, 15):
                    with T.block():
                        T.reads([A[tx, i + 1, 0:16]])
                        T.writes([A_shared[(i + 1) % 2, tx, 0, 0:16]])
                        for j in T.serial(0, 16):
                            with T.block():
                                T.reads([A[tx, i + 1, j]])
                                T.writes([A_shared[(i + 1) % 2, tx, 0, j]])
                                A_shared[(i + 1) % 2, tx, 0, j] = A[tx, i + 1, j]
                    with T.block():
                        T.reads([A_shared[i % 2, tx, i, 1:16], B[0:2, tx, i, 0]])
                        T.writes([B[0:2, tx, i, 0], C[tx, i, 0:15]])
                        for j in T.serial(0, 15):
                            with T.block():
                                T.reads([A_shared[i % 2, tx, i, j + 1]])
                                T.writes([B[(j + 1) % 2, tx, i, 0]])
                                B[(j + 1) % 2, tx, i, 0] = A_shared[
                                    i % 2, tx, 0, j + 1
                                ] * T.float32(2)
                            with T.block():
                                T.reads([B[j % 2, tx, i, 0]])
                                T.writes([C[tx, i, j]])
                                C[tx, i, j] = B[j % 2, tx, i, 0] + T.float32(1)
                    with T.block():
                        T.reads([A_shared[(i + 1) % 2, tx, i + 1, 0]])
                        T.writes([B[0, tx, i + 1, 0]])
                        B[0, tx, i + 1, 0] = A_shared[(i + 1) % 2, tx, 0, 0] * T.float32(2)
                    with T.block():
                        T.reads([B[1, tx, i, 0]])
                        T.writes([C[tx, i, 15]])
                        C[tx, i, 15] = B[1, tx, i, 0] + T.float32(1)
            with T.block():
                T.reads([A_shared[1, tx, 15, 1:16], B[0:2, tx, 15, 0]])
                T.writes([B[0:2, tx, 15, 0], C[tx, 15, 0:16]])
                with T.block():
                    T.reads([A_shared[1, tx, 15, 1:16], B[0:2, tx, 15, 0]])
                    T.writes([B[0:2, tx, 15, 0], C[tx, 15, 0:15]])
                    for j in T.serial(0, 15):
                        with T.block():
                            T.reads([A_shared[1, tx, 15, j + 1]])
                            T.writes([B[(j + 1) % 2, tx, 15, 0]])
                            B[(j + 1) % 2, tx, 15, 0] = A_shared[1, tx, 0, j + 1] * T.float32(2)
                        with T.block():
                            T.reads([B[j % 2, tx, 15, 0]])
                            T.writes([C[tx, 15, j]])
                            C[tx, 15, j] = B[j % 2, tx, 15, 0] + T.float32(1)
                with T.block():
                    T.reads([B[1, tx, 15, 0]])
                    T.writes([C[tx, 15, 15]])
                    C[tx, 15, 15] = B[1, tx, 15, 0] + T.float32(1)


@T.prim_func
def nested_pipeline_interleaving(
    A: T.Buffer[(16, 16, 16), "float32"], C: T.Buffer[(16, 16, 16), "float32"]
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
            with T.block():
                T.reads(A[tx, i, 0:16])
                T.writes(C[tx, i, 0:16])
                A_shared = T.alloc_buffer((16, 1, 16), dtype="float32", scope="shared")
                A_local = T.alloc_buffer((1, 1, 16), dtype="float32", scope="local")
                for j in T.serial(0, 16):
                    with T.block():
                        T.reads(A[tx, i, j])
                        T.writes(A_shared[tx, 0, j])
                        A_shared[tx, 0, j] = A[tx, i, j]
                for j in T.serial(0, 16):
                    with T.block():
                        T.reads(A_shared[tx, 0, j])
                        T.writes(A_local[0, 0, j])
                        A_local[0, 0, j] = A_shared[tx, i, j]
                for j in T.serial(
                    0,
                    16,
                    annotations={
                        "software_pipeline_stage": [0, 1],
                        "software_pipeline_order": [0, 1],
                    },
                ):
                    with T.block():
                        T.reads(A_local[0, 0, j])
                        T.writes(C[tx, i, j])
                        B = T.alloc_buffer((16, 1, 1), dtype="float32", scope="shared")
                        with T.block():
                            T.reads(A_local[tx, i, j])
                            T.writes(B[tx, i, 0])
                            B[tx, i, 0] = A_local[0, 0, j] * T.float32(2)
                        with T.block():
                            T.reads(B[tx, i, 0])
                            T.writes(C[tx, i, j])
                            C[tx, i, j] = B[tx, i, 0] + T.float32(1)


@T.prim_func
def transformed_nested_pipeline_interleaving(
    A: T.Buffer[(16, 16, 16), "float32"], C: T.Buffer[(16, 16, 16), "float32"]
) -> None:
    for tx in T.thread_binding(0, 16, thread="threadIdx.x"):
        with T.block():
            T.reads([A[tx, 0:16, 0:16]])
            T.writes([C[tx, 0:16, 0:16]])
            A_shared = T.alloc_buffer([16, 1, 16], dtype="float32", scope="shared")
            A_local = T.alloc_buffer([1, 1, 16], dtype="float32", scope="local")
            B = T.alloc_buffer([2, 16, 1, 1], dtype="float32", scope="shared")
            with T.block():
                T.reads([A[tx, 0, 0:16], A_shared[tx, 0, 0:16], A_local[tx, 0, 0]])
                T.writes([A_shared[tx, 0, 0:16], A_local[0, 0, 0:16], B[0, tx, 0, 0]])
                with T.block():
                    T.reads([A[tx, 0, 0:16]])
                    T.writes([A_shared[tx, 0, 0:16]])
                    for j in T.serial(0, 16):
                        with T.block():
                            T.reads([A[tx, 0, j]])
                            T.writes([A_shared[tx, 0, j]])
                            A_shared[tx, 0, j] = A[tx, 0, j]
                with T.block():
                    T.reads([A_shared[tx, 0, 0:16]])
                    T.writes([A_local[0, 0, 0:16]])
                    for j in T.serial(0, 16):
                        with T.block():
                            T.reads([A_shared[tx, 0, j]])
                            T.writes([A_local[0, 0, j]])
                            A_local[0, 0, j] = A_shared[tx, 0, j]
                with T.block():
                    T.reads([A_local[tx, 0, 0]])
                    T.writes([B[0, tx, 0, 0]])
                    B[0, tx, 0, 0] = A_local[0, 0, 0] * T.float32(2)
            with T.block():
                T.reads(
                    [
                        A[tx, 1:16, 0:16],
                        A_local[tx, 0:16, 0:16],
                        B[0:2, tx, 0:15, 0],
                        A_shared[tx, 0, 0:16],
                    ]
                )
                T.writes(
                    [
                        A_shared[tx, 0, 0:16],
                        B[0:2, tx, 0:16, 0],
                        C[tx, 0:15, 0:16],
                        A_local[0, 0, 0:16],
                    ]
                )
                for i in T.serial(0, 15):
                    with T.block():
                        T.reads([A[tx, i + 1, 0:16]])
                        T.writes([A_shared[tx, 0, 0:16]])
                        for j in T.serial(0, 16):
                            with T.block():
                                T.reads([A[tx, i + 1, j]])
                                T.writes([A_shared[tx, 0, j]])
                                A_shared[tx, 0, j] = A[tx, i + 1, j]
                    with T.block():
                        T.reads([A_local[tx, i, 1:16], B[0:2, tx, i, 0]])
                        T.writes([B[0:2, tx, i, 0], C[tx, i, 0:15]])
                        for j in T.serial(0, 15):
                            with T.block():
                                T.reads([A_local[tx, i, j + 1]])
                                T.writes([B[(j + 1) % 2, tx, i, 0]])
                                B[(j + 1) % 2, tx, i, 0] = A_local[0, 0, j + 1] * T.float32(2)
                            with T.block():
                                T.reads([B[j % 2, tx, i, 0]])
                                T.writes([C[tx, i, j]])
                                C[tx, i, j] = B[j % 2, tx, i, 0] + T.float32(1)
                    with T.block():
                        T.reads([A_shared[tx, 0, 0:16]])
                        T.writes([A_local[0, 0, 0:16]])
                        for j in T.serial(0, 16):
                            with T.block():
                                T.reads([A_shared[tx, 0, j]])
                                T.writes([A_local[0, 0, j]])
                                A_local[0, 0, j] = A_shared[tx, i + 1, j]
                    with T.block():
                        T.reads([A_local[tx, i + 1, 0]])
                        T.writes([B[0, tx, i + 1, 0]])
                        B[0, tx, i + 1, 0] = A_local[0, 0, 0] * T.float32(2)
                    with T.block():
                        T.reads([B[1, tx, i, 0]])
                        T.writes([C[tx, i, 15]])
                        C[tx, i, 15] = B[1, tx, i, 0] + T.float32(1)
            with T.block():
                T.reads([A_local[tx, 15, 1:16], B[0:2, tx, 15, 0]])
                T.writes([B[0:2, tx, 15, 0], C[tx, 15, 0:16]])
                with T.block():
                    T.reads([A_local[tx, 15, 1:16], B[0:2, tx, 15, 0]])
                    T.writes([B[0:2, tx, 15, 0], C[tx, 15, 0:15]])
                    for j in T.serial(0, 15):
                        with T.block():
                            T.reads([A_local[tx, 15, j + 1]])
                            T.writes([B[(j + 1) % 2, tx, 15, 0]])
                            B[(j + 1) % 2, tx, 15, 0] = A_local[0, 0, j + 1] * T.float32(2)
                        with T.block():
                            T.reads([B[j % 2, tx, 15, 0]])
                            T.writes([C[tx, 15, j]])
                            C[tx, 15, j] = B[j % 2, tx, 15, 0] + T.float32(1)
                with T.block():
                    T.reads([B[1, tx, 15, 0]])
                    T.writes([C[tx, 15, 15]])
                    C[tx, 15, 15] = B[1, tx, 15, 0] + T.float32(1)


@T.prim_func
def nested_pipeline_double_buffer(
    A: T.Buffer[(16, 16, 16), "float32"], C: T.Buffer[(16, 16, 16), "float32"]
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
            with T.block():
                T.reads(A[tx, i, 0:16])
                T.writes(C[tx, i, 0:16])
                A_shared = T.alloc_buffer((16, 1, 16), dtype="float32", scope="shared")
                A_local = T.alloc_buffer((1, 1, 16), dtype="float32", scope="local")
                for j in T.serial(0, 16):
                    with T.block():
                        T.reads(A[tx, i, j])
                        T.writes(A_shared[tx, 0, j])
                        A_shared[tx, 0, j] = A[tx, i, j]
                for j in T.serial(0, 16):
                    with T.block():
                        T.block_attr({"double_buffer_scope": 0})
                        T.reads(A_shared[tx, 0, j])
                        T.writes(A_local[0, 0, j])
                        A_local[0, 0, j] = A_shared[tx, i, j]
                for j in T.serial(
                    0,
                    16,
                    annotations={
                        "software_pipeline_stage": [0, 1],
                        "software_pipeline_order": [0, 1],
                    },
                ):
                    with T.block():
                        T.reads(A_local[0, 0, j])
                        T.writes(C[tx, i, j])
                        B = T.alloc_buffer((16, 1, 1), dtype="float32", scope="shared")
                        with T.block():
                            T.reads(A_local[tx, i, j])
                            T.writes(B[tx, i, 0])
                            B[tx, i, 0] = A_local[0, 0, j] * T.float32(2)
                        with T.block():
                            T.reads(B[tx, i, 0])
                            T.writes(C[tx, i, j])
                            C[tx, i, j] = B[tx, i, 0] + T.float32(1)


@T.prim_func
def transformed_nested_pipeline_double_buffer(
    A: T.Buffer[(16, 16, 16), "float32"], C: T.Buffer[(16, 16, 16), "float32"]
) -> None:
    for tx in T.thread_binding(0, 16, thread="threadIdx.x"):
        with T.block():
            T.reads([A[tx, 0:16, 0:16]])
            T.writes([C[tx, 0:16, 0:16]])
            A_shared = T.alloc_buffer([16, 1, 16], dtype="float32", scope="shared")
            A_local = T.alloc_buffer([2, 1, 1, 16], dtype="float32", scope="local")
            B = T.alloc_buffer([2, 16, 1, 1], dtype="float32", scope="shared")
            with T.block():
                T.reads([A[tx, 0, 0:16], A_shared[tx, 0, 0:16], A_local[0, tx, 0, 0]])
                T.writes([A_shared[tx, 0, 0:16], A_local[0, 0, 0, 0:16], B[0, tx, 0, 0]])
                with T.block():
                    T.reads([A[tx, 0, 0:16]])
                    T.writes([A_shared[tx, 0, 0:16]])
                    for j in T.serial(0, 16):
                        with T.block():
                            T.reads([A[tx, 0, j]])
                            T.writes([A_shared[tx, 0, j]])
                            A_shared[tx, 0, j] = A[tx, 0, j]
                with T.block():
                    T.reads([A_shared[tx, 0, 0:16]])
                    T.writes([A_local[0, 0, 0, 0:16]])
                    for j in T.serial(0, 16):
                        with T.block():
                            T.reads([A_shared[tx, 0, j]])
                            T.writes([A_local[0, 0, 0, j]])
                            T.block_attr({"double_buffer_scope": 0})
                            A_local[0, 0, 0, j] = A_shared[tx, 0, j]
                with T.block():
                    T.reads([A_local[0, tx, 0, 0]])
                    T.writes([B[0, tx, 0, 0]])
                    B[0, tx, 0, 0] = A_local[0, 0, 0, 0] * T.float32(2)
            with T.block():
                T.reads(
                    [
                        A[tx, 1:16, 0:16],
                        A_local[0:2, tx, 0:16, 0:16],
                        B[0:2, tx, 0:15, 0],
                        A_shared[tx, 0, 0:16],
                    ]
                )
                T.writes(
                    [
                        A_shared[tx, 0, 0:16],
                        B[0:2, tx, 0:16, 0],
                        C[tx, 0:15, 0:16],
                        A_local[0:2, 0, 0, 0:16],
                    ]
                )
                for i in T.serial(0, 15):
                    with T.block():
                        T.reads([A[tx, i + 1, 0:16]])
                        T.writes([A_shared[tx, 0, 0:16]])
                        for j in T.serial(0, 16):
                            with T.block():
                                T.reads([A[tx, i + 1, j]])
                                T.writes([A_shared[tx, 0, j]])
                                A_shared[tx, 0, j] = A[tx, i + 1, j]
                    with T.block():
                        T.reads([A_local[i % 2, tx, i, 1:16], B[0:2, tx, i, 0]])
                        T.writes([B[0:2, tx, i, 0], C[tx, i, 0:15]])
                        for j in T.serial(0, 15):
                            with T.block():
                                T.reads([A_local[i % 2, tx, i, j + 1]])
                                T.writes([B[(j + 1) % 2, tx, i, 0]])
                                B[(j + 1) % 2, tx, i, 0] = A_local[i % 2, 0, 0, j + 1] * T.float32(
                                    2
                                )
                            with T.block():
                                T.reads([B[j % 2, tx, i, 0]])
                                T.writes([C[tx, i, j]])
                                C[tx, i, j] = B[j % 2, tx, i, 0] + T.float32(1)
                    with T.block():
                        T.reads([A_shared[tx, 0, 0:16]])
                        T.writes([A_local[(i + 1) % 2, 0, 0, 0:16]])
                        for j in T.serial(0, 16):
                            with T.block():
                                T.reads([A_shared[tx, 0, j]])
                                T.writes([A_local[(i + 1) % 2, 0, 0, j]])
                                T.block_attr({"double_buffer_scope": 0})
                                A_local[(i + 1) % 2, 0, 0, j] = A_shared[tx, i + 1, j]
                    with T.block():
                        T.reads([A_local[(i + 1) % 2, tx, i + 1, 0]])
                        T.writes([B[0, tx, i + 1, 0]])
                        B[0, tx, i + 1, 0] = A_local[(i + 1) % 2, 0, 0, 0] * T.float32(2)
                    with T.block():
                        T.reads([B[1, tx, i, 0]])
                        T.writes([C[tx, i, 15]])
                        C[tx, i, 15] = B[1, tx, i, 0] + T.float32(1)
            with T.block():
                T.reads([A_local[1, tx, 15, 1:16], B[0:2, tx, 15, 0]])
                T.writes([B[0:2, tx, 15, 0], C[tx, 15, 0:16]])
                with T.block():
                    T.reads([A_local[1, tx, 15, 1:16], B[0:2, tx, 15, 0]])
                    T.writes([B[0:2, tx, 15, 0], C[tx, 15, 0:15]])
                    for j in T.serial(0, 15):
                        with T.block():
                            T.reads([A_local[1, tx, 15, j + 1]])
                            T.writes([B[(j + 1) % 2, tx, 15, 0]])
                            B[(j + 1) % 2, tx, 15, 0] = A_local[1, 0, 0, j + 1] * T.float32(2)
                        with T.block():
                            T.reads([B[j % 2, tx, 15, 0]])
                            T.writes([C[tx, 15, j]])
                            C[tx, 15, j] = B[j % 2, tx, 15, 0] + T.float32(1)
                with T.block():
                    T.reads([B[1, tx, 15, 0]])
                    T.writes([C[tx, 15, 15]])
                    C[tx, 15, 15] = B[1, tx, 15, 0] + T.float32(1)


@T.prim_func
def simple_compute_incorrect_reorder(
    A: T.Buffer[(16, 16), "float32"], D: T.Buffer[(16, 16), "float32"]
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
            with T.block():
                T.reads(A[tx, i])
                T.writes(D[tx, i])
                B = T.alloc_buffer((16, 1), dtype="float32", scope="shared")
                C = T.alloc_buffer((16, 1), dtype="float32", scope="shared")
                with T.block():
                    T.reads(A[tx, i])
                    T.writes(B[tx, 0])
                    B[tx, 0] = A[tx, i] * T.float32(2)
                with T.block():
                    T.reads(B[tx, 0])
                    T.writes(C[tx, 0])
                    C[tx, 0] = B[tx, 0] + T.float32(2)
                with T.block():
                    T.reads(C[tx, 0])
                    T.writes(D[tx, i])
                    D[tx, i] = C[tx, 0] + T.float32(1)


@T.prim_func
def simple_compute_conflicting_order(
    A: T.Buffer[(16, 16), "float32"], D: T.Buffer[(16, 16), "float32"]
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
            with T.block():
                T.reads(A[tx, i])
                T.writes(D[tx, i])
                B = T.alloc_buffer((16, 1), dtype="float32", scope="shared")
                C = T.alloc_buffer((16, 1), dtype="float32", scope="shared")
                with T.block():
                    T.reads(A[tx, i])
                    T.writes(B[tx, 0])
                    B[tx, 0] = A[tx, i] * T.float32(2)
                with T.block():
                    T.reads(B[tx, 0])
                    T.writes(C[tx, 0])
                    C[tx, 0] = B[tx, 0] + T.float32(2)
                with T.block():
                    T.reads(C[tx, 0])
                    T.writes(D[tx, i])
                    D[tx, i] = C[tx, 0] + T.float32(1)


@T.prim_func
def simple_compute_missing_annotation(
    A: T.Buffer[(16, 16), "float32"], C: T.Buffer[(16, 16), "float32"]
):
    for tx in T.thread_binding(0, 16, thread="threadIdx.x"):
        for i in T.serial(0, 16, annotations={"software_pipeline_stage": [0, 1]}):
            with T.block():
                T.reads(A[tx, i])
                T.writes(C[tx, i])
                B = T.alloc_buffer((16, 1), dtype="float32", scope="shared")
                with T.block():
                    T.reads(A[tx, i])
                    T.writes(B[tx, 0])
                    B[tx, 0] = A[tx, i] * T.float32(2)
                with T.block():
                    T.reads(B[tx, 0])
                    T.writes(C[tx, i])
                    C[tx, i] = B[tx, 0] + T.float32(1)


def test_simple_compute():
    _check(simple_compute, transformed_simple_compute)


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


def test_three_stage_gemm():
    @tvm.script.ir_module
    class Module_pipelined:
        @T.prim_func
        def main(
            A: T.Buffer[(4096, 4096), "float16"],
            B: T.Buffer[(4096, 4096), "float16"],
            C: T.Buffer[(4096, 4096), "float32"],
        ) -> None:
            # function attr dict
            T.func_attr({"global_symbol": "main", "tir.noalias": True})
            # var definition
            tx = T.env_thread("threadIdx.x")
            s0 = T.var("int32")
            s0_1 = T.var("int32")
            s0_2 = T.var("int32")
            s1 = T.var("int32")
            s1_1 = T.var("int32")
            s1_2 = T.var("int32")
            # body
            # with T.block("root")
            A_shared = T.alloc_buffer([4096, 4096], dtype="float16", scope="shared.dyn")
            B_shared = T.alloc_buffer([4096, 4096], dtype="float16", scope="shared.dyn")
            A_shared_warp = T.alloc_buffer([256, 256, 32, 8], dtype="float16", scope="warp")
            B_shared_warp = T.alloc_buffer([256, 256, 32, 8], dtype="float16", scope="warp")
            C_warp = T.alloc_buffer([256, 256, 32, 8], dtype="float32", scope="warp")
            for i0_0_0_i1_0_0_fused in T.thread_binding(4, thread="blockIdx.x"):
                for i0_0_1_i1_0_1_fused in T.thread_binding(512, thread="blockIdx.y"):
                    for i1_0_2_i0_0_2_fused in T.thread_binding(4, thread="threadIdx.y"):
                        for i0_0_3_init, i1_0_4_init in T.grid(4, 2):
                            with T.block("C_o_init"):
                                i_o = T.axis.spatial(
                                    256,
                                    i0_0_0_i1_0_0_fused * 64
                                    + i0_0_1_i1_0_1_fused // 64 * 8
                                    + i1_0_2_i0_0_2_fused % 2 * 4
                                    + i0_0_3_init,
                                )
                                j_o = T.axis.spatial(
                                    256,
                                    i0_0_1_i1_0_1_fused % 64 * 4
                                    + i1_0_2_i0_0_2_fused // 2 * 2
                                    + i1_0_4_init,
                                )
                                T.reads()
                                T.writes(C_warp[i_o, j_o, 0:32, 0:8])
                                with T.block("C_init_o"):
                                    i_init_o = T.axis.spatial(1, 0)
                                    j_init_o = T.axis.spatial(1, 0)
                                    T.reads()
                                    T.writes(C_warp[i_o, j_o, 0:32, 0:8])
                                    C_warp_1 = T.match_buffer(
                                        C_warp[i_o, j_o, 0:32, 0:8],
                                        [32, 8],
                                        dtype="float32",
                                        scope="warp",
                                        offset_factor=1,
                                    )
                                    T.launch_thread(tx, 32)
                                    T.evaluate(
                                        T.mma_fill(
                                            8, C_warp_1.data, C_warp_1.elem_offset, dtype="float32"
                                        )
                                    )
                        for i2_0_0 in T.serial(
                            128,
                            annotations={
                                "software_pipeline_order": [0, 1, 2],
                                "software_pipeline_stage": [0, 0, 3],
                            },
                        ):
                            for ax0_ax1_fused_0 in T.serial(4):
                                for ax0_ax1_fused_1 in T.thread_binding(4, thread="threadIdx.y"):
                                    for ax0_ax1_fused_2 in T.thread_binding(
                                        32, thread="threadIdx.x"
                                    ):
                                        for ax0_ax1_fused_3 in T.vectorized(8):
                                            with T.block("A_shared"):
                                                v0 = T.axis.spatial(
                                                    4096,
                                                    i0_0_0_i1_0_0_fused * 1024
                                                    + i0_0_1_i1_0_1_fused // 64 * 128
                                                    + (
                                                        ax0_ax1_fused_0 * 1024
                                                        + ax0_ax1_fused_1 * 256
                                                        + ax0_ax1_fused_2 * 8
                                                        + ax0_ax1_fused_3
                                                    )
                                                    // 32,
                                                )
                                                v1 = T.axis.spatial(
                                                    4096,
                                                    i2_0_0 * 32
                                                    + (
                                                        ax0_ax1_fused_0 * 1024
                                                        + ax0_ax1_fused_1 * 256
                                                        + ax0_ax1_fused_2 * 8
                                                        + ax0_ax1_fused_3
                                                    )
                                                    % 32,
                                                )
                                                T.reads(A[v0, v1])
                                                T.writes(A_shared[v0, v1])
                                                T.block_attr({"buffer_dim_align": [[0, 0, 32, 8]]})
                                                A_shared[v0, v1] = A[v0, v1]
                            for ax0_ax1_fused_0 in T.serial(2):
                                for ax0_ax1_fused_1 in T.thread_binding(4, thread="threadIdx.y"):
                                    for ax0_ax1_fused_2 in T.thread_binding(
                                        32, thread="threadIdx.x"
                                    ):
                                        for ax0_ax1_fused_3 in T.vectorized(8):
                                            with T.block("B_shared"):
                                                v0 = T.axis.spatial(
                                                    4096,
                                                    i2_0_0 * 32
                                                    + (
                                                        ax0_ax1_fused_0 * 1024
                                                        + ax0_ax1_fused_1 * 256
                                                        + ax0_ax1_fused_2 * 8
                                                        + ax0_ax1_fused_3
                                                    )
                                                    // 64,
                                                )
                                                v1 = T.axis.spatial(
                                                    4096,
                                                    i0_0_1_i1_0_1_fused % 64 * 64
                                                    + (
                                                        ax0_ax1_fused_0 * 1024
                                                        + ax0_ax1_fused_1 * 256
                                                        + ax0_ax1_fused_2 * 8
                                                        + ax0_ax1_fused_3
                                                    )
                                                    % 64,
                                                )
                                                T.reads(B[v0, v1])
                                                T.writes(B_shared[v0, v1])
                                                T.block_attr({"buffer_dim_align": [[0, 0, 32, 8]]})
                                                B_shared[v0, v1] = B[v0, v1]
                            for i2_0_1 in T.serial(2):
                                for ax0_0, ax1_0 in T.grid(4, 1):
                                    with T.block("A_shared_warp_o"):
                                        v0_o = T.axis.spatial(
                                            256,
                                            i0_0_0_i1_0_0_fused * 64
                                            + i0_0_1_i1_0_1_fused // 64 * 8
                                            + i1_0_2_i0_0_2_fused % 2 * 4
                                            + ax0_0,
                                        )
                                        v1_o = T.axis.spatial(256, i2_0_0 * 2 + i2_0_1)
                                        T.reads(
                                            A_shared[
                                                v0_o * 16 : v0_o * 16 + 16,
                                                v1_o * 16 : v1_o * 16 + 16,
                                            ]
                                        )
                                        T.writes(A_shared_warp[v0_o, v1_o, 0:32, 0:8])
                                        warp = T.match_buffer(
                                            A_shared_warp[v0_o, v1_o, 0:32, 0:8],
                                            [32, 8],
                                            dtype="float16",
                                            scope="warp",
                                            offset_factor=16,
                                        )
                                        shared = T.match_buffer(
                                            A_shared[
                                                v0_o * 16 : v0_o * 16 + 16,
                                                v1_o * 16 : v1_o * 16 + 16,
                                            ],
                                            [16, 16],
                                            dtype="float16",
                                            strides=[s0, s1],
                                            scope="shared.dyn",
                                            offset_factor=16,
                                        )
                                        T.launch_thread(tx, 32)
                                        T.evaluate(
                                            T.ptx_ldmatrix(
                                                False,
                                                4,
                                                ".b16",
                                                warp.data,
                                                warp.elem_offset + 8 * tx,
                                                T.tvm_access_ptr(
                                                    T.type_annotation(dtype="float16"),
                                                    shared.data,
                                                    shared.elem_offset,
                                                    s0 * 16,
                                                    1,
                                                    dtype="handle",
                                                ),
                                                s0 * (tx % 16) + 8 * (tx // 16),
                                                dtype="float16",
                                            )
                                        )
                                for ax0_0, ax1_0 in T.grid(1, 2):
                                    with T.block("B_shared_warp_o"):
                                        v0_o = T.axis.spatial(256, i2_0_0 * 2 + i2_0_1)
                                        v1_o = T.axis.spatial(
                                            256,
                                            i0_0_1_i1_0_1_fused % 64 * 4
                                            + i1_0_2_i0_0_2_fused // 2 * 2
                                            + ax1_0,
                                        )
                                        T.reads(
                                            B_shared[
                                                v0_o * 16 : v0_o * 16 + 16,
                                                v1_o * 16 : v1_o * 16 + 16,
                                            ]
                                        )
                                        T.writes(B_shared_warp[v0_o, v1_o, 0:32, 0:8])
                                        warp_1 = T.match_buffer(
                                            B_shared_warp[v0_o, v1_o, 0:32, 0:8],
                                            [32, 8],
                                            dtype="float16",
                                            scope="warp",
                                            offset_factor=16,
                                        )
                                        shared_1 = T.match_buffer(
                                            B_shared[
                                                v0_o * 16 : v0_o * 16 + 16,
                                                v1_o * 16 : v1_o * 16 + 16,
                                            ],
                                            [16, 16],
                                            dtype="float16",
                                            strides=[s0_1, s1_1],
                                            scope="shared.dyn",
                                            offset_factor=16,
                                        )
                                        T.launch_thread(tx, 32)
                                        T.evaluate(
                                            T.ptx_ldmatrix(
                                                True,
                                                4,
                                                ".b16",
                                                warp_1.data,
                                                warp_1.elem_offset + 8 * tx,
                                                T.tvm_access_ptr(
                                                    T.type_annotation(dtype="float16"),
                                                    shared_1.data,
                                                    shared_1.elem_offset,
                                                    s0_1 * 16,
                                                    1,
                                                    dtype="handle",
                                                ),
                                                s0_1 * (tx % 16) + 8 * (tx // 16),
                                                dtype="float16",
                                            )
                                        )
                                for i0_0_3, i1_0_3, i2_0_2, i0_0_4, i1_0_4 in T.grid(4, 1, 1, 1, 2):
                                    with T.block("C_o_update"):
                                        i_o = T.axis.spatial(
                                            256,
                                            i0_0_0_i1_0_0_fused * 64
                                            + i0_0_1_i1_0_1_fused // 64 * 8
                                            + i1_0_2_i0_0_2_fused % 2 * 4
                                            + i0_0_3,
                                        )
                                        j_o = T.axis.spatial(
                                            256,
                                            i0_0_1_i1_0_1_fused % 64 * 4
                                            + i1_0_2_i0_0_2_fused // 2 * 2
                                            + i1_0_4,
                                        )
                                        k_o = T.axis.reduce(256, i2_0_0 * 2 + i2_0_1)
                                        T.reads(
                                            C_warp[i_o, j_o, 0:32, 0:8],
                                            A_shared_warp[i_o, k_o, 0:32, 0:8],
                                            B_shared_warp[k_o, j_o, 0:32, 0:8],
                                        )
                                        T.writes(C_warp[i_o, j_o, 0:32, 0:8])
                                        with T.block("C_o"):
                                            i_o_1 = T.axis.spatial(1, 0)
                                            j_o_1 = T.axis.spatial(1, 0)
                                            k_o_1 = T.axis.reduce(1, 0)
                                            T.reads(
                                                C_warp[i_o, j_o, 0:32, 0:8],
                                                A_shared_warp[i_o, k_o, 0:32, 0:8],
                                                B_shared_warp[k_o, j_o, 0:32, 0:8],
                                            )
                                            T.writes(C_warp[i_o, j_o, 0:32, 0:8])
                                            A_1 = T.match_buffer(
                                                A_shared_warp[i_o, k_o, 0:32, 0:8],
                                                [32, 8],
                                                dtype="float16",
                                                scope="warp",
                                                offset_factor=16,
                                            )
                                            B_1 = T.match_buffer(
                                                B_shared_warp[k_o, j_o, 0:32, 0:8],
                                                [32, 8],
                                                dtype="float16",
                                                scope="warp",
                                                offset_factor=16,
                                            )
                                            C_1 = T.match_buffer(
                                                C_warp[i_o, j_o, 0:32, 0:8],
                                                [32, 8],
                                                dtype="float32",
                                                scope="warp",
                                                offset_factor=16,
                                            )
                                            T.launch_thread(tx, 32)
                                            T.evaluate(
                                                T.ptx_mma(
                                                    "m16n8k16",
                                                    "row",
                                                    "col",
                                                    "fp16",
                                                    "fp16",
                                                    "fp32",
                                                    A_1.data,
                                                    A_1.elem_offset + tx * 8,
                                                    B_1.data,
                                                    B_1.elem_offset + tx * 8,
                                                    C_1.data,
                                                    C_1.elem_offset + tx * 8,
                                                    False,
                                                    dtype="float32",
                                                )
                                            )
                                            T.evaluate(
                                                T.ptx_mma(
                                                    "m16n8k16",
                                                    "row",
                                                    "col",
                                                    "fp16",
                                                    "fp16",
                                                    "fp32",
                                                    A_1.data,
                                                    A_1.elem_offset + tx * 8,
                                                    B_1.data,
                                                    B_1.elem_offset + tx * 8 + 8 // 2,
                                                    C_1.data,
                                                    C_1.elem_offset + tx * 8 + 8 // 2,
                                                    False,
                                                    dtype="float32",
                                                )
                                            )
                        for ax0_0, ax1_0 in T.grid(4, 2):
                            with T.block("C_warp_o"):
                                v0_o = T.axis.spatial(
                                    256,
                                    i0_0_0_i1_0_0_fused * 64
                                    + i0_0_1_i1_0_1_fused // 64 * 8
                                    + i1_0_2_i0_0_2_fused % 2 * 4
                                    + ax0_0,
                                )
                                v1_o = T.axis.spatial(
                                    256,
                                    i0_0_1_i1_0_1_fused % 64 * 4
                                    + i1_0_2_i0_0_2_fused // 2 * 2
                                    + ax1_0,
                                )
                                T.reads(C_warp[v0_o, v1_o, 0:32, 0:8])
                                T.writes(C[v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16])
                                C_warp_2 = T.match_buffer(
                                    C_warp[v0_o, v1_o, 0:32, 0:8],
                                    [32, 8],
                                    dtype="float32",
                                    scope="warp",
                                    offset_factor=1,
                                )
                                C_2 = T.match_buffer(
                                    C[v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16],
                                    [16, 16],
                                    dtype="float32",
                                    strides=[s0_2, s1_2],
                                    offset_factor=1,
                                )
                                T.launch_thread(tx, 32)
                                T.evaluate(
                                    T.mma_store(
                                        16,
                                        16,
                                        T.tvm_access_ptr(
                                            T.type_annotation(dtype="float32"),
                                            C_2.data,
                                            C_2.elem_offset,
                                            s0_2 * 16,
                                            2,
                                            dtype="handle",
                                        ),
                                        C_warp_2.data,
                                        C_warp_2.elem_offset,
                                        s0_2,
                                        dtype="float32",
                                    )
                                )

    f = tvm.build(Module_pipelined, target="cuda")

    N = K = M = 4096
    dev = tvm.device("cuda", 0)
    a_np = np.random.uniform(size=(N, K)).astype("float16")
    b_np = np.random.uniform(size=(K, M)).astype("float16")
    c_np = np.dot(a_np.astype("float32"), b_np.astype("float32"))
    a = tvm.nd.array(a_np, dev)
    b = tvm.nd.array(b_np, dev)
    c = tvm.nd.array(np.zeros((N, M), dtype="float32"), dev)
    f(a, b, c)
    # print(f.imported_modules[0].get_source())
    tvm.testing.assert_allclose(c.numpy(), c_np, rtol=1e-3)
    print("ok")


if __name__ == "__main__":
    # tvm.testing.main()
    test_three_stage_gemm()
