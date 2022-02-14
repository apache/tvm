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
# pylint: disable=missing-function-docstring,missing-module-docstring
import sys

import pytest
import tvm
from tvm import tir
from tvm.script import tir as T
from tvm.tir.schedule.testing import verify_trace_roundtrip

# pylint: disable=no-member,invalid-name,unused-variable

########## Function before schedule ##########


@T.prim_func
def elementwise(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (128, 128))
    B = T.alloc_buffer((128, 128))
    C = T.match_buffer(c, (128, 128))
    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] * 2.0
    for i, j in T.grid(128, 128):
        with T.block("C"):
            vi, vj = T.axis.remap("SS", [i, j])
            C[vi, vj] = B[vi, vj] + 1.0


@T.prim_func
def access_under_scope(b: T.handle, c: T.handle) -> None:
    A = T.alloc_buffer((128, 128))
    B = T.match_buffer(b, (128, 128))
    C = T.match_buffer(c, (128, 128))

    for i0, j0 in T.grid(8, 8):
        with T.block("scope"):
            i, j = T.axis.remap("SS", [i0, j0])
            for x, y in T.grid(16, 16):
                with T.block("A"):
                    vi = T.axis.S(128, i * 16 + x)
                    vj = T.axis.S(128, j * 16 + y)
                    A[vi, vj] = 1.0
            for x, y in T.grid(16, 16):
                with T.block("B"):
                    vi = T.axis.S(128, i * 16 + x)
                    vj = T.axis.S(128, j * 16 + y)
                    B[vi, vj] = A[vi, vj] + 1.0
    for i, j in T.grid(128, 128):
        with T.block("C"):
            vi, vj = T.axis.remap("SS", [i, j])
            C[vi, vj] = A[vi, vj] * 2.0


@T.prim_func
def opaque_access(a: T.handle, b: T.handle, c: T.handle, d: T.handle) -> None:
    A = T.match_buffer(a, (128, 128), dtype="float16")
    B = T.match_buffer(b, (128, 128), dtype="float16")
    C = T.match_buffer(c, (128, 128), dtype="float16")
    D = T.match_buffer(d, (128, 128), dtype="float16")

    for i, j in T.grid(128, 128):
        with T.block("load_store"):
            vi, vj = T.axis.remap("SS", [i, j])
            T.reads(A[vi, vj])
            T.writes(D[vi, vj])
            D.data[vi * 128 + vj] = T.load("float16", A.data, vi * 128 + vj)
    for i, j in T.grid(8, 8):
        with T.block("opaque"):
            vi, vj = T.axis.remap("SS", [i, j])
            T.reads(A[vi * 16 : vi * 16 + 16, vj * 16 : vj * 16 + 16])
            T.writes(B[vi * 16 : vi * 16 + 16, vj * 16 : vj * 16 + 16])
            T.evaluate(
                T.tvm_load_matrix_sync(
                    B.data,
                    16,
                    16,
                    16,
                    vi * 8 + vj,
                    T.tvm_access_ptr(
                        T.type_annotation(dtype="float16"),
                        A.data,
                        vi * 2048 + vj * 16,
                        128,
                        1,
                        dtype="handle",
                    ),
                    128,
                    "row_major",
                    dtype="handle",
                )
            )
    for i, j in T.grid(8, 8):
        with T.block("match_buffer"):
            vi, vj = T.axis.remap("SS", [i, j])
            T.reads(A[vi * 16 : vi * 16 + 16, vj * 16 : vj * 16 + 16])
            T.writes(C[vi * 16 : vi * 16 + 16, vj * 16 : vj * 16 + 16])
            A0 = T.match_buffer(
                A[
                    vi * 16 : vi * 16 + 16,
                    vj * 16 : vj * 16 + 16,
                ],
                (16, 16),
                "float16",
                strides=[128, 1],
                offset_factor=1,
            )
            C0 = T.match_buffer(
                C[
                    vi * 16 : vi * 16 + 16,
                    vj * 16 : vj * 16 + 16,
                ],
                (16, 16),
                "float16",
                strides=[128, 1],
                offset_factor=1,
            )
            T.evaluate(
                T.tvm_load_matrix_sync(
                    C0.data,
                    16,
                    16,
                    16,
                    vi * 8 + vj,
                    T.tvm_access_ptr(
                        T.type_annotation(dtype="float16"),
                        A0.data,
                        A0.elem_offset,
                        A0.strides[0],
                        1,
                        dtype="handle",
                    ),
                    128,
                    "row_major",
                    dtype="handle",
                )
            )


@T.prim_func
def func_multi_consumer() -> None:
    A = T.alloc_buffer((128))
    B = T.alloc_buffer((128))
    C = T.alloc_buffer((128))
    for i in T.grid(8):
        for j in T.grid(16):
            with T.block("A"):
                vi = T.axis.S(128, i * 16 + j)
                A[vi] = 1.0
        for j in T.grid(16):
            with T.block("B"):
                vi = T.axis.S(128, i * 16 + j)
                B[vi] = A[vi] + 1.0
    for i in T.grid(128):
        with T.block("C"):
            vi = T.axis.S(128, i)
            C[vi] = A[vi]


@T.prim_func
def func_multi_producer() -> None:
    A = T.alloc_buffer((128))
    B = T.alloc_buffer((128))
    for i in range(128):
        with T.block("A0"):
            vi = T.axis.S(128, i)
            A[vi] = 1.0
    for i in range(128):
        with T.block("A1"):
            vi = T.axis.S(128, i)
            A[vi] = 2.0
    for i in range(128):
        with T.block("B"):
            vi = T.axis.S(128, i)
            B[vi] = A[vi]


@T.prim_func
def func_with_block_predicate() -> None:
    A = T.alloc_buffer((120))
    B = T.alloc_buffer((120))
    for i, j in T.grid(16, 8):
        with T.block("producer"):
            T.where(i * 8 + j < 120)
            ax = T.axis.S(120, i * 8 + j)
            A[ax] = 0.0
    for i, j in T.grid(16, 8):
        with T.block("consumer"):
            T.where(i * 8 + j < 120)
            ax = T.axis.S(120, i * 8 + j)
            B[ax] = A[ax] + 1.0


########## Expected function after cache_read ##########


@T.prim_func
def cache_read_elementwise(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (128, 128))
    C = T.match_buffer(c, (128, 128))
    B = T.alloc_buffer((128, 128))
    A_global = T.alloc_buffer((128, 128))
    B_local = T.alloc_buffer((128, 128), scope="local")
    for i, j in T.grid(128, 128):
        with T.block("A_global"):
            vi, vj = T.axis.remap("SS", [i, j])
            A_global[vi, vj] = A[vi, vj]
    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi, vj] = A_global[vi, vj] * 2.0
    for i, j in T.grid(128, 128):
        with T.block("B_local"):
            vi, vj = T.axis.remap("SS", [i, j])
            B_local[vi, vj] = B[vi, vj]
    for i, j in T.grid(128, 128):
        with T.block("C"):
            vi, vj = T.axis.remap("SS", [i, j])
            C[vi, vj] = B_local[vi, vj] + 1.0


@T.prim_func
def cache_read_under_scope(b: T.handle, c: T.handle) -> None:
    A = T.alloc_buffer((128, 128))
    B = T.match_buffer(b, (128, 128))
    C = T.match_buffer(c, (128, 128))
    A_global = T.alloc_buffer((128, 128))

    for i0, j0 in T.grid(8, 8):
        with T.block("scope"):
            i, j = T.axis.remap("SS", [i0, j0])
            A_local = T.alloc_buffer((128, 128), scope="local")
            for x, y in T.grid(16, 16):
                with T.block("A"):
                    vi = T.axis.S(128, i * 16 + x)
                    vj = T.axis.S(128, j * 16 + y)
                    A[vi, vj] = 1.0
            for x, y in T.grid(16, 16):
                with T.block("A_local"):
                    vi = T.axis.S(128, i * 16 + x)
                    vj = T.axis.S(128, j * 16 + y)
                    A_local[vi, vj] = A[vi, vj]
            for x, y in T.grid(16, 16):
                with T.block("B"):
                    vi = T.axis.S(128, i * 16 + x)
                    vj = T.axis.S(128, j * 16 + y)
                    B[vi, vj] = A_local[vi, vj] + 1.0
    for i, j in T.grid(128, 128):
        with T.block("A_global"):
            vi, vj = T.axis.remap("SS", [i, j])
            A_global[vi, vj] = A[vi, vj]
    for i, j in T.grid(128, 128):
        with T.block("C"):
            vi, vj = T.axis.remap("SS", [i, j])
            C[vi, vj] = A_global[vi, vj] * 2.0


@T.prim_func
def cache_read_opaque_access(a: T.handle, b: T.handle, c: T.handle, d: T.handle) -> None:
    A = T.match_buffer(a, (128, 128), dtype="float16")
    B = T.match_buffer(b, (128, 128), dtype="float16")
    C = T.match_buffer(c, (128, 128), dtype="float16")
    D = T.match_buffer(d, (128, 128), dtype="float16")
    A_global = T.alloc_buffer((128, 128), dtype="float16")

    for i, j in T.grid(128, 128):
        with T.block("A_global"):
            vi, vj = T.axis.remap("SS", [i, j])
            A_global[vi, vj] = A[vi, vj]
    for i, j in T.grid(128, 128):
        with T.block("load_store"):
            vi, vj = T.axis.remap("SS", [i, j])
            T.reads(A_global[vi, vj])
            T.writes(D[vi, vj])
            D.data[vi * 128 + vj] = T.load("float16", A_global.data, vi * 128 + vj)
    for i, j in T.grid(8, 8):
        with T.block("opaque"):
            vi, vj = T.axis.remap("SS", [i, j])
            T.reads(A_global[vi * 16 : vi * 16 + 16, vj * 16 : vj * 16 + 16])
            T.writes(B[vi * 16 : vi * 16 + 16, vj * 16 : vj * 16 + 16])
            T.evaluate(
                T.tvm_load_matrix_sync(
                    B.data,
                    16,
                    16,
                    16,
                    vi * 8 + vj,
                    T.tvm_access_ptr(
                        T.type_annotation(dtype="float16"),
                        A_global.data,
                        vi * 2048 + vj * 16,
                        128,
                        1,
                        dtype="handle",
                    ),
                    128,
                    "row_major",
                    dtype="handle",
                )
            )
    for i, j in T.grid(8, 8):
        with T.block("match_buffer"):
            vi, vj = T.axis.remap("SS", [i, j])
            T.reads(A_global[vi * 16 : vi * 16 + 16, vj * 16 : vj * 16 + 16])
            T.writes(C[vi * 16 : vi * 16 + 16, vj * 16 : vj * 16 + 16])
            A0 = T.match_buffer(
                A_global[
                    vi * 16 : vi * 16 + 16,
                    vj * 16 : vj * 16 + 16,
                ],
                (16, 16),
                "float16",
                strides=[128, 1],
                offset_factor=1,
            )
            C0 = T.match_buffer(
                C[
                    vi * 16 : vi * 16 + 16,
                    vj * 16 : vj * 16 + 16,
                ],
                (16, 16),
                "float16",
                strides=[128, 1],
                offset_factor=1,
            )
            T.evaluate(
                T.tvm_load_matrix_sync(
                    C0.data,
                    16,
                    16,
                    16,
                    vi * 8 + vj,
                    T.tvm_access_ptr(
                        T.type_annotation(dtype="float16"),
                        A0.data,
                        A0.elem_offset,
                        A0.strides[0],
                        1,
                        dtype="handle",
                    ),
                    128,
                    "row_major",
                    dtype="handle",
                )
            )


@T.prim_func
def cache_read_multi_consumer() -> None:
    A = T.alloc_buffer((128))
    B = T.alloc_buffer((128))
    C = T.alloc_buffer((128))
    A_global = T.alloc_buffer((128))
    for i in T.grid(8):
        for j in T.grid(16):
            with T.block("A"):
                vi = T.axis.S(128, i * 16 + j)
                A[vi] = 1.0
        for j in T.grid(16):
            with T.block("A"):
                vi = T.axis.S(128, i * 16 + j)
                A_global[vi] = A[vi]
        for j in T.grid(16):
            with T.block("B"):
                vi = T.axis.S(128, i * 16 + j)
                B[vi] = A_global[vi] + 1.0

    for i in T.grid(128):
        with T.block("C"):
            vi = T.axis.S(128, i)
            C[vi] = A_global[vi]


@T.prim_func
def continuous_cache_read(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (128, 128))
    C = T.match_buffer(c, (128, 128))
    B = T.alloc_buffer((128, 128))
    B_shared = T.alloc_buffer((128, 128), scope="shared")
    B_local = T.alloc_buffer((128, 128), scope="local")
    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] * 2.0
    for i, j in T.grid(128, 128):
        with T.block("B_shared"):
            vi, vj = T.axis.remap("SS", [i, j])
            B_shared[vi, vj] = B[vi, vj]
    for i, j in T.grid(128, 128):
        with T.block("B_local"):
            vi, vj = T.axis.remap("SS", [i, j])
            B_local[vi, vj] = B_shared[vi, vj]
    for i, j in T.grid(128, 128):
        with T.block("C"):
            vi, vj = T.axis.remap("SS", [i, j])
            C[vi, vj] = B_local[vi, vj] + 1.0


@T.prim_func
def block_predicate_cache_read() -> None:
    A = T.alloc_buffer([120], dtype="float32")
    B = T.alloc_buffer([120], dtype="float32")
    A_shared = T.alloc_buffer([120], dtype="float32", scope="shared")
    for i, j in T.grid(16, 8):
        with T.block("producer"):
            ax = T.axis.spatial(120, i * 8 + j)
            T.where(i * 8 + j < 120)
            A[ax] = T.float32(0)
    for ax0 in T.serial(120):
        with T.block("A_shared"):
            v0 = T.axis.spatial(120, ax0)
            A_shared[v0] = A[v0]
    for i, j in T.grid(16, 8):
        with T.block("consumer"):
            ax = T.axis.spatial(120, i * 8 + j)
            T.where(i * 8 + j < 120)
            B[ax] = A_shared[ax] + T.float32(1)


########## Expected function after cache_write ##########


@T.prim_func
def cache_write_elementwise(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (128, 128))
    C = T.match_buffer(c, (128, 128))
    B = T.alloc_buffer((128, 128))
    B_global = T.alloc_buffer((128, 128), scope="local")
    C_local = T.alloc_buffer((128, 128))
    for i, j in T.grid(128, 128):
        with T.block("B_global"):
            vi, vj = T.axis.remap("SS", [i, j])
            B_global[vi, vj] = A[vi, vj] * 2.0
    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi, vj] = B_global[vi, vj]
    for i, j in T.grid(128, 128):
        with T.block("C_local"):
            vi, vj = T.axis.remap("SS", [i, j])
            C_local[vi, vj] = B[vi, vj] + 1.0
    for i, j in T.grid(128, 128):
        with T.block("C"):
            vi, vj = T.axis.remap("SS", [i, j])
            C[vi, vj] = C_local[vi, vj]


@T.prim_func
def cache_write_under_scope(b: T.handle, c: T.handle) -> None:
    A = T.alloc_buffer((128, 128))
    B = T.match_buffer(b, (128, 128))
    C = T.match_buffer(c, (128, 128))
    A_global = T.alloc_buffer((128, 128))

    for i0, j0 in T.grid(8, 8):
        with T.block("scope"):
            i, j = T.axis.remap("SS", [i0, j0])
            A_local = T.alloc_buffer((128, 128), scope="local")
            B_global = T.alloc_buffer((128, 128))
            for x, y in T.grid(16, 16):
                with T.block("A_local"):
                    vi = T.axis.S(128, i * 16 + x)
                    vj = T.axis.S(128, j * 16 + y)
                    A_local[vi, vj] = 1.0
            for x, y in T.grid(16, 16):
                with T.block("A"):
                    vi = T.axis.S(128, i * 16 + x)
                    vj = T.axis.S(128, j * 16 + y)
                    A_global[vi, vj] = A_local[vi, vj]
            for x, y in T.grid(16, 16):
                with T.block("B_global"):
                    vi = T.axis.S(128, i * 16 + x)
                    vj = T.axis.S(128, j * 16 + y)
                    B_global[vi, vj] = A_global[vi, vj] + 1.0
            for x, y in T.grid(16, 16):
                with T.block("B_global"):
                    vi = T.axis.S(128, i * 16 + x)
                    vj = T.axis.S(128, j * 16 + y)
                    B[vi, vj] = B_global[vi, vj]
    for i, j in T.grid(128, 128):
        with T.block("A_global"):
            vi, vj = T.axis.remap("SS", [i, j])
            A[vi, vj] = A_global[vi, vj]
    for i, j in T.grid(128, 128):
        with T.block("C"):
            vi, vj = T.axis.remap("SS", [i, j])
            C[vi, vj] = A[vi, vj] * 2.0


@T.prim_func
def cache_write_opaque_access(a: T.handle, b: T.handle, c: T.handle, d: T.handle) -> None:
    A = T.match_buffer(a, (128, 128), dtype="float16")
    B = T.match_buffer(b, (128, 128), dtype="float16")
    C = T.match_buffer(c, (128, 128), dtype="float16")
    D = T.match_buffer(d, (128, 128), dtype="float16")
    D_global = T.alloc_buffer((128, 128), dtype="float16")
    B_global = T.alloc_buffer((128, 128), dtype="float16")
    C_global = T.alloc_buffer((128, 128), dtype="float16")

    for i, j in T.grid(128, 128):
        with T.block("load_store"):
            vi, vj = T.axis.remap("SS", [i, j])
            T.reads(A[vi, vj])
            T.writes(D_global[vi, vj])
            D_global.data[vi * 128 + vj] = T.load("float16", A.data, vi * 128 + vj)
    for i, j in T.grid(8, 8):
        with T.block("opaque"):
            vi, vj = T.axis.remap("SS", [i, j])
            T.reads(A[vi * 16 : vi * 16 + 16, vj * 16 : vj * 16 + 16])
            T.writes(B_global[vi * 16 : vi * 16 + 16, vj * 16 : vj * 16 + 16])
            T.evaluate(
                T.tvm_load_matrix_sync(
                    B_global.data,
                    16,
                    16,
                    16,
                    vi * 8 + vj,
                    T.tvm_access_ptr(
                        T.type_annotation(dtype="float16"),
                        A.data,
                        vi * 2048 + vj * 16,
                        128,
                        1,
                        dtype="handle",
                    ),
                    128,
                    "row_major",
                    dtype="handle",
                )
            )
    for i, j in T.grid(8, 8):
        with T.block("match_buffer"):
            vi, vj = T.axis.remap("SS", [i, j])
            T.reads(A[vi * 16 : vi * 16 + 16, vj * 16 : vj * 16 + 16])
            T.writes(C_global[vi * 16 : vi * 16 + 16, vj * 16 : vj * 16 + 16])
            A0 = T.match_buffer(
                A[
                    vi * 16 : vi * 16 + 16,
                    vj * 16 : vj * 16 + 16,
                ],
                (16, 16),
                "float16",
                strides=[128, 1],
                offset_factor=1,
            )
            C0 = T.match_buffer(
                C_global[
                    vi * 16 : vi * 16 + 16,
                    vj * 16 : vj * 16 + 16,
                ],
                (16, 16),
                "float16",
                strides=[128, 1],
                offset_factor=1,
            )
            T.evaluate(
                T.tvm_load_matrix_sync(
                    C0.data,
                    16,
                    16,
                    16,
                    vi * 8 + vj,
                    T.tvm_access_ptr(
                        T.type_annotation(dtype="float16"),
                        A0.data,
                        A0.elem_offset,
                        A0.strides[0],
                        1,
                        dtype="handle",
                    ),
                    128,
                    "row_major",
                    dtype="handle",
                )
            )

    for i, j in T.grid(128, 128):
        with T.block("D"):
            vi, vj = T.axis.remap("SS", [i, j])
            D[vi, vj] = D_global[vi, vj]
    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi, vj] = B_global[vi, vj]
    for i, j in T.grid(128, 128):
        with T.block("C"):
            vi, vj = T.axis.remap("SS", [i, j])
            C[vi, vj] = C_global[vi, vj]


@T.prim_func
def cache_write_multi_consumer() -> None:
    A = T.alloc_buffer((128))
    B = T.alloc_buffer((128))
    C = T.alloc_buffer((128))
    A_global = T.alloc_buffer((128))
    for i in T.grid(8):
        for j in T.grid(16):
            with T.block("A_global"):
                vi = T.axis.S(128, i * 16 + j)
                A_global[vi] = 1.0
        for j in T.grid(16):
            with T.block("A"):
                vi = T.axis.S(128, i * 16 + j)
                A[vi] = A_global[vi]
        for j in T.grid(16):
            with T.block("B"):
                vi = T.axis.S(128, i * 16 + j)
                B[vi] = A[vi] + 1.0

    for i in T.grid(128):
        with T.block("C"):
            vi = T.axis.S(128, i)
            C[vi] = A[vi]


@T.prim_func
def continuous_cache_write(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (128, 128))
    B = T.alloc_buffer((128, 128))
    C = T.match_buffer(c, (128, 128))
    B_shared = T.alloc_buffer((128, 128), scope="shared")
    B_local = T.alloc_buffer((128, 128), scope="local")
    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            B_local[vi, vj] = A[vi, vj] * 2.0
    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            B_shared[vi, vj] = B_local[vi, vj]
    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi, vj] = B_shared[vi, vj]
    for i, j in T.grid(128, 128):
        with T.block("C"):
            vi, vj = T.axis.remap("SS", [i, j])
            C[vi, vj] = B[vi, vj] + 1.0


@T.prim_func
def block_predicate_cache_write_intermediate_buf() -> None:
    A = T.alloc_buffer([120], dtype="float32")
    B = T.alloc_buffer([120], dtype="float32")
    A_shared = T.alloc_buffer([120], dtype="float32", scope="shared")
    for i, j in T.grid(16, 8):
        with T.block("producer"):
            ax = T.axis.spatial(120, i * 8 + j)
            T.where(i * 8 + j < 120)
            A_shared[ax] = T.float32(0)
    for ax0 in T.serial(120):
        with T.block("A_shared"):
            v0 = T.axis.spatial(120, ax0)
            A[v0] = A_shared[v0]
    for i, j in T.grid(16, 8):
        with T.block("consumer"):
            ax = T.axis.spatial(120, i * 8 + j)
            T.where(i * 8 + j < 120)
            B[ax] = A[ax] + 1.0


@T.prim_func
def block_predicate_cache_write_output_buf() -> None:
    A = T.alloc_buffer([120], dtype="float32")
    B = T.alloc_buffer([120], dtype="float32")
    B_shared = T.alloc_buffer([120], dtype="float32", scope="shared")
    for i, j in T.grid(16, 8):
        with T.block("producer"):
            ax = T.axis.spatial(120, i * 8 + j)
            T.where(i * 8 + j < 120)
            A[ax] = T.float32(0)
    for i, j in T.grid(16, 8):
        with T.block("consumer"):
            ax = T.axis.spatial(120, i * 8 + j)
            T.where(i * 8 + j < 120)
            B_shared[ax] = A[ax] + T.float32(1)
    for ax0 in T.serial(120):
        with T.block("B_shared"):
            v0 = T.axis.spatial(120, ax0)
            B[v0] = B_shared[v0]


########## Testcases for cache_read ##########


def test_cache_read_elementwise():
    sch = tir.Schedule(elementwise, debug_mask="all")
    block_b = sch.get_block("B")
    block_c = sch.get_block("C")
    cached_a = sch.cache_read(block_b, 0, "global")
    cached_b = sch.cache_read(block_c, 0, "local")
    assert sch.get(cached_a) == sch.get(sch.get_block("A_global"))
    assert sch.get(cached_b) == sch.get(sch.get_block("B_local"))
    assert sch.get(block_b) == sch.get(sch.get_block("B"))
    assert sch.get(block_c) == sch.get(sch.get_block("C"))
    tvm.ir.assert_structural_equal(cache_read_elementwise, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=elementwise)


def test_cache_read_under_scope():
    sch = tir.Schedule(access_under_scope, debug_mask="all")
    block_b = sch.get_block("B")
    block_c = sch.get_block("C")
    sch.cache_read(block_b, 0, "local")
    sch.cache_read(block_c, 0, "global")
    tvm.ir.assert_structural_equal(cache_read_under_scope, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=access_under_scope)


def test_cache_read_opaque_access():
    sch = tir.Schedule(opaque_access, debug_mask="all")
    block = sch.get_block("load_store")
    sch.cache_read(block, 0, "global")
    tvm.ir.assert_structural_equal(cache_read_opaque_access, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=opaque_access)


def test_cache_read_location():
    sch = tir.Schedule(func_multi_consumer, debug_mask="all")
    block_b = sch.get_block("B")
    sch.cache_read(block_b, 0, "global")
    tvm.ir.assert_structural_equal(cache_read_multi_consumer, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=func_multi_consumer)


def test_continuous_cache_read():
    sch = tir.Schedule(elementwise, debug_mask="all")
    block_c = sch.get_block("C")
    sch.cache_read(block_c, 0, "shared")
    sch.cache_read(block_c, 0, "local")
    tvm.ir.assert_structural_equal(continuous_cache_read, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=elementwise)


def test_cache_read_with_block_predicate():
    sch = tir.Schedule(func_with_block_predicate, debug_mask="all")
    block = sch.get_block("consumer")
    sch.cache_read(block, 0, "shared")
    tvm.ir.assert_structural_equal(block_predicate_cache_read, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=func_with_block_predicate)


def test_cache_read_fail_multi_producer():
    sch = tir.Schedule(func_multi_producer, debug_mask="all")
    block_b = sch.get_block("B")
    with pytest.raises(tvm.tir.ScheduleError):
        sch.cache_read(block_b, 0, "global")


def test_cache_read_fail_index_out_of_bound():
    sch = tir.Schedule(elementwise, debug_mask="all")
    block_b = sch.get_block("B")
    with pytest.raises(tvm.tir.ScheduleError):
        sch.cache_read(block_b, 1, "global")


def test_cache_read_fail_invalid_storage_scope():
    sch = tir.Schedule(elementwise, debug_mask="all")
    block_b = sch.get_block("B")
    with pytest.raises(tvm.tir.ScheduleError):
        sch.cache_read(block_b, 0, "test_scope")


########## Testcases for cache_write ##########


def test_cache_write_elementwise():
    sch = tir.Schedule(elementwise, debug_mask="all")
    block_b = sch.get_block("B")
    block_c = sch.get_block("C")
    cached_b = sch.cache_write(block_b, 0, "local")
    cached_c = sch.cache_write(block_c, 0, "global")
    assert sch.get(cached_b) == sch.get(sch.get_block("B_local"))
    assert sch.get(cached_c) == sch.get(sch.get_block("C_global"))
    assert sch.get(block_b) == sch.get(sch.get_block("B"))
    assert sch.get(block_c) == sch.get(sch.get_block("C"))
    tvm.ir.assert_structural_equal(cache_write_elementwise, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=elementwise)


def test_cache_write_under_scope():
    sch = tir.Schedule(access_under_scope, debug_mask="all")
    block_a = sch.get_block("A")
    block_b = sch.get_block("B")
    block_scope = sch.get_block("scope")
    sch.cache_write(block_a, 0, "local")
    sch.cache_write(block_b, 0, "global")
    sch.cache_write(block_scope, 0, "global")
    tvm.ir.assert_structural_equal(cache_write_under_scope, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=access_under_scope)


def test_cache_write_opaque_access():
    sch = tir.Schedule(opaque_access, debug_mask="all")
    block_store = sch.get_block("load_store")
    block_opaque = sch.get_block("opaque")
    block_match_buffer = sch.get_block("match_buffer")
    sch.cache_write(block_store, 0, "global")
    sch.cache_write(block_opaque, 0, "global")
    sch.cache_write(block_match_buffer, 0, "global")
    tvm.ir.assert_structural_equal(cache_write_opaque_access, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=opaque_access)


def test_cache_write_location():
    sch = tir.Schedule(func_multi_consumer, debug_mask="all")
    block_a = sch.get_block("A")
    sch.cache_write(block_a, 0, "global")
    tvm.ir.assert_structural_equal(cache_write_multi_consumer, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=func_multi_consumer)


def test_continuous_cache_write():
    sch = tir.Schedule(elementwise, debug_mask="all")
    block_b = sch.get_block("B")
    sch.cache_write(block_b, 0, "shared")
    sch.cache_write(block_b, 0, "local")
    tvm.ir.assert_structural_equal(continuous_cache_write, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=elementwise)


def test_cache_write_with_block_predicate():
    # cache write for intermediate buffer
    sch = tir.Schedule(func_with_block_predicate, debug_mask="all")
    block = sch.get_block("producer")
    sch.cache_write(block, 0, "shared")
    tvm.ir.assert_structural_equal(block_predicate_cache_write_intermediate_buf, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=func_with_block_predicate)
    # cache write for external buffer
    sch = tir.Schedule(func_with_block_predicate, debug_mask="all")
    block = sch.get_block("consumer")
    sch.cache_write(block, 0, "shared")
    tvm.ir.assert_structural_equal(block_predicate_cache_write_output_buf, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=func_with_block_predicate)


def test_cache_write_fail_multi_producer():
    sch = tir.Schedule(func_multi_producer, debug_mask="all")
    block_a0 = sch.get_block("A0")
    block_a1 = sch.get_block("A1")
    with pytest.raises(tvm.tir.ScheduleError):
        sch.cache_write(block_a0, 0, "global")
    with pytest.raises(tvm.tir.ScheduleError):
        sch.cache_write(block_a1, 0, "global")


def test_cache_write_fail_index_out_of_bound():
    sch = tir.Schedule(elementwise, debug_mask="all")
    block_b = sch.get_block("B")
    with pytest.raises(tvm.tir.ScheduleError):
        sch.cache_write(block_b, 1, "global")


def test_cache_write_fail_invalid_storage_scope():
    sch = tir.Schedule(elementwise, debug_mask="all")
    block_b = sch.get_block("B")
    with pytest.raises(tvm.tir.ScheduleError):
        sch.cache_write(block_b, 0, "test_scope")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__] + sys.argv[1:]))
