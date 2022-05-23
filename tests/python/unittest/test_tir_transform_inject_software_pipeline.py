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

import tvm
import tvm.testing
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
                    C[tx, 0] = A[tx, 0] + T.float32(2)
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
                        C[(i + 1) % 2, tx, 0] = A[tx, 0] + T.float32(2)
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
                        C[(i + 1) % 2, tx, 0] = A[tx, 0] + T.float32(2)
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
                        C[(i + 1) % 2, tx, 0] = A[tx, 0] + T.float32(2)
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


if __name__ == "__main__":
    tvm.testing.main()
