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
import tvm.testing
from tvm import tir
from tvm.script import tir as T
from tvm.tir.schedule.testing import (
    verify_trace_roundtrip,
    assert_structural_equal_ignore_global_symbol,
)

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
def elementwise_shape_int64(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (T.int64(128), T.int64(128)))
    B = T.alloc_buffer((T.int64(128), T.int64(128)))
    C = T.match_buffer(c, (T.int64(128), T.int64(128)))
    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] * 2.0
    for i, j in T.grid(128, 128):
        with T.block("C"):
            vi, vj = T.axis.remap("SS", [i, j])
            C[vi, vj] = B[vi, vj] + 1.0


@T.prim_func
def elementwise_reindex_cache_read(
    A: T.Buffer((128, 128), "float32"), C: T.Buffer((128, 128), "float32")
):
    B = T.alloc_buffer((128, 128))
    B_shared = T.alloc_buffer((128, 64, 2), scope="shared")
    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            T.reads(A[vi, vj])
            T.writes(B[vi, vj])
            B[vi, vj] = A[vi, vj] * T.float32(2)
    for i, j in T.grid(128, 128):
        with T.block("B_shared"):
            vi, vj = T.axis.remap("SS", [i, j])
            T.reads(B[vi, vj])
            T.writes(B_shared[vj, vi // 2, vi % 2])
            B_shared[vj, vi // 2, vi % 2] = B[vi, vj]
    for i, j in T.grid(128, 128):
        with T.block("C"):
            vi, vj = T.axis.remap("SS", [i, j])
            T.reads(B_shared[vj, vi // 2, vi % 2])
            T.writes(C[vi, vj])
            C[vi, vj] = B_shared[vj, vi // 2, vi % 2] + T.float32(1)


@T.prim_func
def elementwise_reindex_cache_write(
    A: T.Buffer((128, 128), "float32"), C: T.Buffer((128, 128), "float32")
):
    B = T.alloc_buffer((128, 128))
    B_shared = T.alloc_buffer((128, 128), scope="shared")
    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            T.reads(A[vi, vj])
            T.writes(B_shared[vj, vi])
            B_shared[vj, vi] = A[vi, vj] * T.float32(2)
    for i, j in T.grid(128, 128):
        with T.block("B_shared"):
            vi, vj = T.axis.remap("SS", [i, j])
            T.reads(B_shared[vj, vi])
            T.writes(B[vi, vj])
            B[vi, vj] = B_shared[vj, vi]
    for i, j in T.grid(128, 128):
        with T.block("C"):
            vi, vj = T.axis.remap("SS", [i, j])
            T.reads(B[vi, vj])
            T.writes(C[vi, vj])
            C[vi, vj] = B[vi, vj] + T.float32(1)


@T.prim_func
def reduce(A: T.Buffer((128, 128, 128, 128), "float32"), C: T.Buffer((128, 128), "float32")):
    B = T.alloc_buffer((128, 128, 128), dtype="float32")
    for i, j, k in T.grid(128, 128, 128):
        for l in range(128):
            with T.block("B"):
                vi, vj, vk, vl = T.axis.remap("SSSR", [i, j, k, l])
                with T.init():
                    B[vi, vj, vk] = T.float32(0)
                B[vi, vj, vk] = B[vi, vj, vk] + A[vi, vj, vk, vl]
        with T.block("C"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            with T.init():
                C[vi, vj] = T.float32(0)
            C[vi, vj] = C[vi, vj] + B[vi, vj, vk]


@T.prim_func
def reduce_reindex_cache_write_0(
    A: T.Buffer((128, 128, 128, 128), "float32"), C: T.Buffer((128, 128), "float32")
):
    B = T.alloc_buffer((128, 128, 128))
    B_shared = T.alloc_buffer((128, 128, 128), scope="shared")
    for i, j, k in T.grid(128, 128, 128):
        for l in range(128):
            with T.block("B"):
                vi, vj, vk, vl = T.axis.remap("SSSR", [i, j, k, l])
                T.reads(A[vi, vj, vk, vl])
                T.writes(B_shared[vj, vi, vk])
                with T.init():
                    B_shared[vj, vi, vk] = T.float32(0)
                B_shared[vj, vi, vk] = B_shared[vj, vi, vk] + A[vi, vj, vk, vl]
        with T.block("B_shared"):
            vi, vj, vk = T.axis.remap("SSS", [i, j, k])
            T.reads(B_shared[vj, vi, vk])
            T.writes(B[vi, vj, vk])
            B[vi, vj, vk] = B_shared[vj, vi, vk]
        with T.block("C"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            T.reads(B[vi, vj, vk])
            T.writes(C[vi, vj])
            with T.init():
                C[vi, vj] = T.float32(0)
            C[vi, vj] = C[vi, vj] + B[vi, vj, vk]


@T.prim_func
def reduce_reindex_cache_write_1(
    A: T.Buffer((128, 128, 128, 128), "float32"), C: T.Buffer((128, 128), "float32")
):
    B = T.alloc_buffer((128, 128, 128))
    B_shared = T.alloc_buffer((128, 128, 128), scope="shared")
    C_shared = T.alloc_buffer((128, 128), scope="shared")
    for i, j, k in T.grid(128, 128, 128):
        for l in range(128):
            with T.block("B"):
                vi, vj, vk, vl = T.axis.remap("SSSR", [i, j, k, l])
                T.reads(A[vi, vj, vk, vl])
                T.writes(B_shared[vj, vi, vk])
                with T.init():
                    B_shared[vj, vi, vk] = T.float32(0)
                B_shared[vj, vi, vk] = B_shared[vj, vi, vk] + A[vi, vj, vk, vl]
        with T.block("B_shared"):
            vi, vj, vk = T.axis.remap("SSS", [i, j, k])
            T.reads(B_shared[vj, vi, vk])
            T.writes(B[vi, vj, vk])
            B[vi, vj, vk] = B_shared[vj, vi, vk]
        with T.block("C"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            T.reads(B[vi, vj, vk])
            T.writes(C_shared[vj, vi])
            with T.init():
                C_shared[vj, vi] = T.float32(0)
            C_shared[vj, vi] = C_shared[vj, vi] + B[vi, vj, vk]
    for i, j in T.grid(128, 128):
        with T.block("C_shared"):
            vi, vj = T.axis.remap("SS", [i, j])
            T.reads(C_shared[vj, vi])
            T.writes(C[vi, vj])
            C[vi, vj] = C_shared[vj, vi]


@T.prim_func
def func_nested_seq(b: T.handle, c: T.handle) -> None:
    A = T.alloc_buffer((128, 128))
    B = T.match_buffer(b, (128, 128))
    C = T.match_buffer(c, (128, 128))

    for i, j in T.grid(128, 128):
        with T.block("A"):
            vi, vj = T.axis.remap("SS", [i, j])
            A[vi, vj] = 2.0
    for i, j in T.grid(8, 8):
        for x, y in T.grid(16, 16):
            with T.block("B0"):
                vi = T.axis.S(128, i * 16 + x)
                vj = T.axis.S(128, j * 16 + y)
                B[vi, vj] = 1.0
        for x, y in T.grid(16, 16):
            with T.block("B1"):
                vi = T.axis.S(128, i * 16 + x)
                vj = T.axis.S(128, j * 16 + y)
                B[vi, vj] = A[vi, vj] + B[vi, vj]
    for i, j in T.grid(128, 128):
        with T.block("C"):
            vi, vj = T.axis.remap("SS", [i, j])
            C[vi, vj] = A[vi, vj] * 2.0


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
            D[vi, vj] = A[vi, vj]
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
def reindex_cache_read_multi_consumer() -> None:
    A = T.alloc_buffer((128,))
    B = T.alloc_buffer((128,))
    C = T.alloc_buffer((128,))
    A_shared = T.alloc_buffer((4, 32), scope="shared")
    for i in range(8):
        for j in range(16):
            with T.block("A"):
                vi = T.axis.spatial(128, i * 16 + j)
                T.reads()
                T.writes(A[vi])
                A[vi] = T.float32(1)
        for j in range(16):
            with T.block("A_shared"):
                vi = T.axis.spatial(128, i * 16 + j)
                T.reads(A[vi])
                T.writes(A_shared[vi // 32, vi % 32])
                A_shared[vi // 32, vi % 32] = A[vi]
        for j in range(16):
            with T.block("B"):
                vi = T.axis.spatial(128, i * 16 + j)
                T.reads(A_shared[vi // 32, vi % 32])
                T.writes(B[vi])
                B[vi] = A_shared[vi // 32, vi % 32] + T.float32(1)
    for i in range(128):
        with T.block("C"):
            vi = T.axis.spatial(128, i)
            T.reads(A[vi])
            T.writes(C[vi])
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


@T.prim_func
def inplace_func(data_io: T.Buffer((64), "int32")):
    data_1d = T.alloc_buffer([64], dtype="int32")
    for i0 in T.serial(64):
        with T.block("copy_in"):
            v0 = T.axis.remap("S", [i0])
            data_1d[v0] = data_io[v0]
    for i0 in T.serial(1):
        with T.block("ext_call"):
            T.reads(data_1d[:64])
            T.writes(data_1d[:64])
            T.evaluate(T.call_extern("call_impl", data_1d.data, dtype=""))
    for i0 in T.serial(64):
        with T.block("copy_out"):
            v0 = T.axis.remap("S", [i0])
            data_io[v0] = data_1d[v0]


@T.prim_func
def inplace_call(data_io: T.Buffer((64), "int32")):
    for i0 in T.serial(1):
        with T.block("ext_call"):
            T.reads(data_io[:64])
            T.writes(data_io[:64])
            T.evaluate(T.call_extern("call_impl", data_io.data, dtype=""))


@T.prim_func
def cache_read_nested_seq_target(
    B: T.Buffer((128, 128), "float32"), C: T.Buffer((128, 128), "float32")
) -> None:
    A = T.alloc_buffer([128, 128], dtype="float32")
    A_global = T.alloc_buffer([128, 128], dtype="float32")
    for i, j in T.grid(128, 128):
        with T.block("A"):
            vi, vj = T.axis.remap("SS", [i, j])
            T.reads()
            T.writes(A[vi, vj])
            A[vi, vj] = T.float32(2)
    for i, j in T.grid(8, 8):
        for x, y in T.grid(16, 16):
            with T.block("B0"):
                vi = T.axis.spatial(128, i * 16 + x)
                vj = T.axis.spatial(128, j * 16 + y)
                T.reads()
                T.writes(B[vi, vj])
                B[vi, vj] = T.float32(1)
        for x, y in T.grid(16, 16):
            with T.block("B1"):
                vi = T.axis.spatial(128, i * 16 + x)
                vj = T.axis.spatial(128, j * 16 + y)
                T.reads(A[vi, vj], B[vi, vj])
                T.writes(B[vi, vj])
                B[vi, vj] = A[vi, vj] + B[vi, vj]
    for ax0, ax1 in T.grid(128, 128):
        with T.block("A_global"):
            v0, v1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(A[v0, v1])
            T.writes(A_global[v0, v1])
            A_global[v0, v1] = A[v0, v1]
    for i, j in T.grid(128, 128):
        with T.block("C"):
            vi, vj = T.axis.remap("SS", [i, j])
            T.reads(A_global[vi, vj])
            T.writes(C[vi, vj])
            C[vi, vj] = A_global[vi, vj] * T.float32(2)


@T.prim_func
def nested_buffer_access(var_A: T.handle, var_B: T.handle, var_C: T.handle):
    A = T.match_buffer(var_A, (T.int64(7), T.int64(512)), dtype="float32")
    B = T.match_buffer(var_B, T.int64(1), dtype="int32")
    C = T.match_buffer(var_C, (T.int64(1), T.int64(512)), dtype="float32")
    for ax0, ax1 in T.grid(T.int64(1), T.int64(512)):
        with T.block("C"):
            v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(A[B[v_ax0], v_ax1], B[v_ax0])
            T.writes(C[v_ax0, v_ax1])
            C[v_ax0, v_ax1] = A[B[v_ax0], v_ax1]


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
            A_local = T.alloc_buffer((16, 16), scope="local")
            for x, y in T.grid(16, 16):
                with T.block("A"):
                    vi = T.axis.S(128, i * 16 + x)
                    vj = T.axis.S(128, j * 16 + y)
                    A[vi, vj] = 1.0
            for x, y in T.grid(16, 16):
                with T.block("A_local"):
                    vi = T.axis.S(16, x)
                    vj = T.axis.S(16, y)
                    A_local[vi, vj] = A[i * 16 + vi, j * 16 + vj]
            for x, y in T.grid(16, 16):
                with T.block("B"):
                    vi = T.axis.S(128, i * 16 + x)
                    vj = T.axis.S(128, j * 16 + y)
                    B[vi, vj] = A_local[vi - i * 16, vj - j * 16] + 1.0
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
            D[vi, vj] = A_global[vi, vj]
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
def cache_read_multi_consumer_target() -> None:
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
            with T.block("B"):
                vi = T.axis.S(128, i * 16 + j)
                B[vi] = A[vi] + 1.0

    for i in T.grid(128):
        with T.block("A"):
            vi = T.axis.S(128, i)
            A_global[vi] = A[vi]
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


@T.prim_func
def cache_read_shape_int64(var_A: T.handle, var_C: T.handle) -> None:
    A = T.match_buffer(var_A, (T.int64(128), T.int64(128)), dtype="float32")
    C = T.match_buffer(var_C, (T.int64(128), T.int64(128)), dtype="float32")
    B = T.alloc_buffer([T.int64(128), T.int64(128)], dtype="float32")
    A_global = T.alloc_buffer([T.int64(128), T.int64(128)], dtype="float32")
    for ax0, ax1 in T.grid(T.int64(128), T.int64(128)):
        with T.block("A_global"):
            v0, v1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(A[v0, v1])
            T.writes(A_global[v0, v1])
            A_global[v0, v1] = A[v0, v1]
    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            T.reads(A_global[vi, vj])
            T.writes(B[vi, vj])
            B[vi, vj] = A_global[vi, vj] * T.float32(2)
    for i, j in T.grid(128, 128):
        with T.block("C"):
            vi, vj = T.axis.remap("SS", [i, j])
            T.reads(B[vi, vj])
            T.writes(C[vi, vj])
            C[vi, vj] = B[vi, vj] + T.float32(1)


@T.prim_func
def cache_read_inplace(data_io: T.Buffer(64, "int32")) -> None:
    data_1d = T.alloc_buffer([64], dtype="int32")
    data_io_local = T.alloc_buffer([64], dtype="int32", scope="local")
    for ax0 in T.serial(64):
        with T.block("data_io_local"):
            v0 = T.axis.spatial(64, ax0)
            T.reads(data_io[v0])
            T.writes(data_io_local[v0])
            data_io_local[v0] = data_io[v0]
    for i0 in T.serial(64):
        with T.block("copy_in"):
            v0 = T.axis.spatial(64, i0)
            T.reads(data_io_local[v0])
            T.writes(data_1d[v0])
            data_1d[v0] = data_io_local[v0]
    for i0 in T.serial(1):
        with T.block("ext_call"):
            T.reads(data_1d[0:64])
            T.writes(data_1d[0:64])
            T.evaluate(T.call_extern("call_impl", data_1d.data, dtype=""))
    for i0 in T.serial(64):
        with T.block("copy_out"):
            v0 = T.axis.spatial(64, i0)
            T.reads(data_1d[v0])
            T.writes(data_io[v0])
            data_io[v0] = data_1d[v0]


@T.prim_func
def cache_inplace_buffer(data_io: T.Buffer(64, "int32")) -> None:
    data_io_local = T.alloc_buffer([64], dtype="int32", scope="local")
    data_io_global = T.alloc_buffer([64], dtype="int32")
    data_io_global_1 = T.alloc_buffer([64], dtype="int32")
    for ax0 in T.serial(64):
        with T.block("data_io_global"):
            v0 = T.axis.spatial(64, ax0)
            T.reads(data_io[v0])
            T.writes(data_io_global[v0])
            data_io_global[v0] = data_io[v0]
    for i0 in T.serial(1):
        for ax0 in T.serial(64):
            with T.block("data_io_local"):
                v0 = T.axis.spatial(64, ax0)
                T.reads(data_io_global[v0])
                T.writes(data_io_local[v0])
                data_io_local[v0] = data_io_global[v0]
        with T.block("ext_call"):
            T.reads(data_io_local[0:64])
            T.writes(data_io_local[0:64])
            T.evaluate(T.call_extern("call_impl", data_io_local.data, dtype=""))
        for ax0 in T.serial(64):
            with T.block("data_io_local"):
                v0 = T.axis.spatial(64, ax0)
                T.reads(data_io_local[v0])
                T.writes(data_io_global_1[v0])
                data_io_global_1[v0] = data_io_local[v0]
    for ax0 in T.serial(64):
        with T.block("data_io_global"):
            v0 = T.axis.spatial(64, ax0)
            T.reads(data_io_global_1[v0])
            T.writes(data_io[v0])
            data_io[v0] = data_io_global_1[v0]


@T.prim_func
def cache_read_nested_buffer_access(var_A: T.handle, var_B: T.handle, var_C: T.handle):
    A = T.match_buffer(var_A, (T.int64(7), T.int64(512)), dtype="float32")
    B = T.match_buffer(var_B, T.int64(1), dtype="int32")
    C = T.match_buffer(var_C, (T.int64(1), T.int64(512)), dtype="float32")
    B_global = T.alloc_buffer((T.int64(1),), "int32")
    for ax0 in range(T.int64(1)):
        with T.block("B_global"):
            v0 = T.axis.spatial(T.int64(1), ax0)
            T.reads(B[v0])
            T.writes(B_global[v0])
            B_global[v0] = B[v0]
    for ax0, ax1 in T.grid(T.int64(1), T.int64(512)):
        with T.block("C"):
            v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(A[B_global[v_ax0], v_ax1], B_global[v_ax0])
            T.writes(C[v_ax0, v_ax1])
            C[v_ax0, v_ax1] = A[B_global[v_ax0], v_ax1]


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
            A_local = T.alloc_buffer((16, 16), scope="local")
            B_global = T.alloc_buffer((16, 16))
            for x, y in T.grid(16, 16):
                with T.block("A_local"):
                    vi = T.axis.S(128, i * 16 + x)
                    vj = T.axis.S(128, j * 16 + y)
                    A_local[vi - i * 16, vj - j * 16] = 1.0
            for x, y in T.grid(16, 16):
                with T.block("A"):
                    vi = T.axis.S(16, x)
                    vj = T.axis.S(16, y)
                    A_global[i * 16 + vi, j * 16 + vj] = A_local[vi, vj]
            for x, y in T.grid(16, 16):
                with T.block("B"):
                    vi = T.axis.S(128, i * 16 + x)
                    vj = T.axis.S(128, j * 16 + y)
                    B_global[vi - i * 16, vj - j * 16] = A_global[vi, vj] + 1.0
            for x, y in T.grid(16, 16):
                with T.block("B_global"):
                    vi = T.axis.S(16, x)
                    vj = T.axis.S(16, y)
                    B[i * 16 + vi, j * 16 + vj] = B_global[vi, vj]
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
            D_global[vi, vj] = A[vi, vj]
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
def cache_write_multi_consumer_B_consume_cache():
    A = T.alloc_buffer([128], dtype="float32")
    B = T.alloc_buffer([128], dtype="float32")
    C = T.alloc_buffer([128], dtype="float32")
    A_global = T.alloc_buffer([128], dtype="float32")
    for i in T.serial(8):
        for j in T.serial(16):
            with T.block("A"):
                vi = T.axis.spatial(128, i * 16 + j)
                A_global[vi] = 1.0
        for j in T.serial(16):
            with T.block("B"):
                vi = T.axis.spatial(128, i * 16 + j)
                B[vi] = A_global[vi] + 1.0
    for ax0 in T.serial(128):
        with T.block("A_global"):
            v0 = T.axis.spatial(128, ax0)
            A[v0] = A_global[v0]
    for i in T.serial(128):
        with T.block("C"):
            vi = T.axis.spatial(128, i)
            C[vi] = A[vi]


@T.prim_func
def cache_write_multi_consumer_C_consume_cache():
    A = T.alloc_buffer([128], dtype="float32")
    B = T.alloc_buffer([128], dtype="float32")
    C = T.alloc_buffer([128], dtype="float32")
    A_global = T.alloc_buffer([128], dtype="float32")
    for i in T.serial(8):
        for j in T.serial(16):
            with T.block("A"):
                vi = T.axis.spatial(128, i * 16 + j)
                A_global[vi] = T.float32(1)
        for ax0 in T.serial(16):
            with T.block("A_global"):
                v0 = T.axis.spatial(128, i * 16 + ax0)
                A[v0] = A_global[v0]
        for j in T.serial(16):
            with T.block("B"):
                vi = T.axis.spatial(128, i * 16 + j)
                B[vi] = A[vi] + T.float32(1)
    for i in T.serial(128):
        with T.block("C"):
            vi = T.axis.spatial(128, i)
            C[vi] = A_global[vi]


@T.prim_func
def cache_write_multi_consumer_all_consume_cache():
    A = T.alloc_buffer([128], dtype="float32")
    B = T.alloc_buffer([128], dtype="float32")
    C = T.alloc_buffer([128], dtype="float32")
    A_global = T.alloc_buffer([128], dtype="float32")
    for i in T.serial(8):
        for j in T.serial(16):
            with T.block("A"):
                vi = T.axis.spatial(128, i * 16 + j)
                A_global[vi] = T.float32(1)
        for j in T.serial(16):
            with T.block("B"):
                vi = T.axis.spatial(128, i * 16 + j)
                B[vi] = A_global[vi] + T.float32(1)
    for i in T.serial(128):
        with T.block("C"):
            vi = T.axis.spatial(128, i)
            C[vi] = A_global[vi]
    for ax0 in T.serial(128):
        with T.block("A_global"):
            v0 = T.axis.spatial(128, ax0)
            A[v0] = A_global[v0]


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


@T.prim_func
def symbolic_matmul_blocked(var_A: T.handle, var_B: T.handle, var_C: T.handle, n: T.int32):
    A = T.match_buffer(var_A, ((n + 31) // 32 * 32, 4))
    B = T.match_buffer(var_B, (4, (n + 31) // 32 * 32))
    C = T.match_buffer(var_C, ((n + 31) // 32 * 32, (n + 31) // 32 * 32))
    for i0_0, i1_0 in T.grid((n + 31) // 32, (n + 31) // 32):
        with T.block("matmul_o"):
            v_i0_o, v_i1_o = T.axis.remap("SS", [i0_0, i1_0])
            T.reads(
                A[v_i0_o * 32 : v_i0_o * 32 + 32, 0:4],
                B[0:4, v_i1_o * 32 : v_i1_o * 32 + 32],
            )
            T.writes(C[v_i0_o * 32 : v_i0_o * 32 + 32, v_i1_o * 32 : v_i1_o * 32 + 32])
            for i0_1, i1_1, k in T.grid(32, 32, 4):
                with T.block("matmul"):
                    v_i0_i, v_i1_i, v_k_i = T.axis.remap("SSR", [i0_1, i1_1, k])
                    T.reads(A[v_i0_o * 32 + v_i0_i, v_k_i], B[v_k_i, v_i1_o * 32 + v_i1_i])
                    T.writes(C[v_i0_o * 32 + v_i0_i, v_i1_o * 32 + v_i1_i])
                    with T.init():
                        C[v_i0_o * 32 + v_i0_i, v_i1_o * 32 + v_i1_i] = T.float32(0)
                    C[v_i0_o * 32 + v_i0_i, v_i1_o * 32 + v_i1_i] = (
                        C[v_i0_o * 32 + v_i0_i, v_i1_o * 32 + v_i1_i]
                        + A[v_i0_o * 32 + v_i0_i, v_k_i] * B[v_k_i, v_i1_o * 32 + v_i1_i]
                    )


@T.prim_func
def symbolic_matmul_blocked_cache_read(
    var_A: T.handle, var_B: T.handle, var_C: T.handle, n: T.int32
):
    A = T.match_buffer(var_A, ((n + 31) // 32 * 32, 4))
    B = T.match_buffer(var_B, (4, (n + 31) // 32 * 32))
    C = T.match_buffer(var_C, ((n + 31) // 32 * 32, (n + 31) // 32 * 32))
    for i0_0, i1_0 in T.grid((n + 31) // 32, (n + 31) // 32):
        with T.block("matmul_o"):
            v_i0_o, v_i1_o = T.axis.remap("SS", [i0_0, i1_0])
            T.reads(
                A[v_i0_o * 32 : v_i0_o * 32 + 32, 0:4],
                B[0:4, v_i1_o * 32 : v_i1_o * 32 + 32],
            )
            T.writes(C[v_i0_o * 32 : v_i0_o * 32 + 32, v_i1_o * 32 : v_i1_o * 32 + 32])
            A_shared = T.alloc_buffer((32, 4), scope="shared")
            for ax0, ax1 in T.grid(32, 4):
                with T.block("A_shared"):
                    v0 = T.axis.spatial(32, ax0)
                    v1 = T.axis.spatial(4, ax1)
                    T.reads(A[v_i0_o * 32 + v0, v1])
                    T.writes(A_shared[v0, v1])
                    A_shared[v0, v1] = A[v_i0_o * 32 + v0, v1]
            for i0_1, i1_1, k in T.grid(32, 32, 4):
                with T.block("matmul"):
                    v_i0_i, v_i1_i, v_k_i = T.axis.remap("SSR", [i0_1, i1_1, k])
                    T.reads(A_shared[v_i0_i, v_k_i], B[v_k_i, v_i1_o * 32 + v_i1_i])
                    T.writes(C[v_i0_o * 32 + v_i0_i, v_i1_o * 32 + v_i1_i])
                    with T.init():
                        C[v_i0_o * 32 + v_i0_i, v_i1_o * 32 + v_i1_i] = T.float32(0)
                    C[v_i0_o * 32 + v_i0_i, v_i1_o * 32 + v_i1_i] = (
                        C[v_i0_o * 32 + v_i0_i, v_i1_o * 32 + v_i1_i]
                        + A_shared[v_i0_i, v_k_i] * B[v_k_i, v_i1_o * 32 + v_i1_i]
                    )


@T.prim_func
def symbolic_matmul_blocked_cache_write(
    var_A: T.handle, var_B: T.handle, var_C: T.handle, n: T.int32
):
    A = T.match_buffer(var_A, ((n + 31) // 32 * 32, 4))
    B = T.match_buffer(var_B, (4, (n + 31) // 32 * 32))
    C = T.match_buffer(var_C, ((n + 31) // 32 * 32, (n + 31) // 32 * 32))
    for i0_0, i1_0 in T.grid((n + 31) // 32, (n + 31) // 32):
        with T.block("matmul_o"):
            v_i0_o, v_i1_o = T.axis.remap("SS", [i0_0, i1_0])
            T.reads(
                A[v_i0_o * 32 : v_i0_o * 32 + 32, 0:4],
                B[0:4, v_i1_o * 32 : v_i1_o * 32 + 32],
            )
            T.writes(C[v_i0_o * 32 : v_i0_o * 32 + 32, v_i1_o * 32 : v_i1_o * 32 + 32])
            C_pad_local = T.alloc_buffer((32, 32), scope="local")
            for i0_1, i1_1, k in T.grid(32, 32, 4):
                with T.block("matmul"):
                    v_i0_i, v_i1_i, v_k_i = T.axis.remap("SSR", [i0_1, i1_1, k])
                    T.reads(A[v_i0_o * 32 + v_i0_i, v_k_i], B[v_k_i, v_i1_o * 32 + v_i1_i])
                    T.writes(C_pad_local[v_i0_i, v_i1_i])
                    with T.init():
                        C_pad_local[v_i0_i, v_i1_i] = T.float32(0)
                    C_pad_local[v_i0_i, v_i1_i] = (
                        C_pad_local[v_i0_i, v_i1_i]
                        + A[v_i0_o * 32 + v_i0_i, v_k_i] * B[v_k_i, v_i1_o * 32 + v_i1_i]
                    )
            for ax0, ax1 in T.grid(32, 32):
                with T.block("C_pad_local"):
                    v0 = T.axis.spatial(32, ax0)
                    v1 = T.axis.spatial(32, ax1)
                    T.reads(C_pad_local[v0, v1])
                    T.writes(C[v_i0_o * 32 + v0, v_i1_o * 32 + v1])
                    C[v_i0_o * 32 + v0, v_i1_o * 32 + v1] = C_pad_local[v0, v1]


########## Testcases for cache_read ##########

use_block_name = tvm.testing.parameter(by_dict={"block_obj": False, "block_name": True})


def test_cache_read_elementwise(use_block_name):
    sch = tir.Schedule(elementwise, debug_mask="all")
    block_b = sch.get_block("B")
    block_c = sch.get_block("C")
    if use_block_name:
        cached_a = sch.cache_read("B", "A", "global")
        cached_b = sch.cache_read("C", "B", "local")
    else:
        cached_a = sch.cache_read(block_b, 0, "global")
        cached_b = sch.cache_read(block_c, 0, "local")
    assert sch.get(cached_a) == sch.get(sch.get_block("A_global"))
    assert sch.get(cached_b) == sch.get(sch.get_block("B_local"))
    assert sch.get(block_b) == sch.get(sch.get_block("B"))
    assert sch.get(block_c) == sch.get(sch.get_block("C"))
    assert_structural_equal_ignore_global_symbol(cache_read_elementwise, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=elementwise)


def test_cache_read_under_scope(use_block_name):
    sch = tir.Schedule(access_under_scope, debug_mask="all")
    block_b = "B" if use_block_name else sch.get_block("B")
    block_c = "C" if use_block_name else sch.get_block("C")
    sch.cache_read(block_b, 0, "local")
    sch.cache_read(block_c, 0, "global")
    assert_structural_equal_ignore_global_symbol(cache_read_under_scope, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=access_under_scope)


def test_cache_read_opaque_access(use_block_name):
    sch = tir.Schedule(opaque_access, debug_mask="all")
    block = "load_store" if use_block_name else sch.get_block("load_store")
    sch.cache_read(block, 0, "global")
    assert_structural_equal_ignore_global_symbol(cache_read_opaque_access, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=opaque_access)


def test_cache_read_location(use_block_name):
    sch = tir.Schedule(func_multi_consumer, debug_mask="all")
    block_b = "B" if use_block_name else sch.get_block("B")
    sch.cache_read(block_b, 0, "global")
    assert_structural_equal_ignore_global_symbol(cache_read_multi_consumer, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=func_multi_consumer)

    # Test that specific consumer block targeting works.
    sch = tir.Schedule(func_multi_consumer, debug_mask="all")
    block_b = "B" if use_block_name else sch.get_block("B")
    block_c = "C" if use_block_name else sch.get_block("C")
    sch.cache_read(block_b, 0, "global", consumer_blocks=[block_c])
    assert_structural_equal_ignore_global_symbol(cache_read_multi_consumer_target, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=func_multi_consumer)

    # Also test setting multiple consumers yields same result as unspecified.
    sch = tir.Schedule(func_multi_consumer, debug_mask="all")
    block_b = "B" if use_block_name else sch.get_block("B")
    block_c = "C" if use_block_name else sch.get_block("C")
    sch.cache_read(block_b, 0, "global", consumer_blocks=[block_b, block_c])
    assert_structural_equal_ignore_global_symbol(cache_read_multi_consumer, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=func_multi_consumer)


def test_continuous_cache_read(use_block_name):
    sch = tir.Schedule(elementwise, debug_mask="all")
    block_c = "C" if use_block_name else sch.get_block("C")
    sch.cache_read(block_c, 0, "shared")
    sch.cache_read(block_c, 0, "local")
    assert_structural_equal_ignore_global_symbol(continuous_cache_read, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=elementwise)


def test_cache_read_with_block_predicate(use_block_name):
    sch = tir.Schedule(func_with_block_predicate, debug_mask="all")
    block = "consumer" if use_block_name else sch.get_block("consumer")
    sch.cache_read(block, 0, "shared")
    assert_structural_equal_ignore_global_symbol(block_predicate_cache_read, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=func_with_block_predicate)


def test_cache_read_non_int32_shape(use_block_name):
    sch = tir.Schedule(elementwise_shape_int64, debug_mask="all")
    block_b = "B" if use_block_name else sch.get_block("B")
    sch.cache_read(block_b, 0, "global")
    assert_structural_equal_ignore_global_symbol(cache_read_shape_int64, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=elementwise_shape_int64)


def test_cache_read_nested_buffer_access(use_block_name):
    sch = tir.Schedule(nested_buffer_access, debug_mask="all")
    block_c = "C" if use_block_name else sch.get_block("C")
    sch.cache_read(block_c, 1, "global")
    assert_structural_equal_ignore_global_symbol(cache_read_nested_buffer_access, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=nested_buffer_access)


def test_cache_read_fail_multi_producer(use_block_name):
    sch = tir.Schedule(func_multi_producer, debug_mask="all")
    block_b = "B" if use_block_name else sch.get_block("B")
    with pytest.raises(tvm.tir.ScheduleError):
        sch.cache_read(block_b, 0, "global")


def test_cache_read_fail_index_out_of_bound(use_block_name):
    sch = tir.Schedule(elementwise, debug_mask="all")
    block_b = "B" if use_block_name else sch.get_block("B")
    with pytest.raises(tvm.tir.ScheduleError):
        sch.cache_read(block_b, 1, "global")


def test_cache_read_fail_invalid_storage_scope(use_block_name):
    sch = tir.Schedule(elementwise, debug_mask="all")
    block_b = "B" if use_block_name else sch.get_block("B")
    with pytest.raises(tvm.tir.ScheduleError):
        sch.cache_read(block_b, 0, "test_scope")


def test_cache_read_allocate_const():
    @T.prim_func
    def before(A: T.Buffer((8), "float32"), C: T.Buffer((8), "float32")):
        B = T.allocate_const([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], "float32", [8])
        B_buf = T.decl_buffer((8), dtype="float32", data=B)
        for i in range(8):
            with T.block("C"):
                vi = T.axis.spatial(8, i)
                C[vi] = A[vi] + B_buf[vi]

    @T.prim_func
    def expected(A: T.Buffer((8), "float32"), C: T.Buffer((8), "float32")):
        B_buf_global = T.alloc_buffer((8), dtype="float32")
        A_global = T.alloc_buffer((8), dtype="float32")
        B = T.allocate_const([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], "float32", [8])
        B_buf = T.decl_buffer((8), data=B)
        for ax0 in range(8):
            with T.block("A_global"):
                v0 = T.axis.spatial(8, ax0)
                A_global[v0] = A[v0]
        for ax0 in range(8):
            with T.block("B_buf_global"):
                v0 = T.axis.spatial(8, ax0)
                B_buf_global[v0] = B_buf[v0]
        for i in range(8):
            with T.block("C"):
                vi = T.axis.spatial(8, i)
                C[vi] = A_global[vi] + B_buf_global[vi]

    sch = tir.Schedule(before)
    block_c = sch.get_block("C")
    sch.cache_read(block_c, 1, "global")
    sch.cache_read(block_c, 0, "global")

    after = sch.mod["main"]

    assert_structural_equal_ignore_global_symbol(expected, after)
    verify_trace_roundtrip(sch=sch, mod=before)


def test_inplace_cache_read():
    sch = tvm.tir.Schedule(inplace_func, debug_mask="all")
    block = sch.get_block("copy_in")
    sch.cache_read(block, 0, "local", [block])
    assert_structural_equal_ignore_global_symbol(cache_read_inplace, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=inplace_func)


def test_cache_inplace():
    # cache_inplace could introduce WAR, which is expected but stage pipeline property changes
    debug_mask = tvm.tir.schedule.state.ScheduleDebugMask.VERIFY_SREF_TREE
    sch = tvm.tir.Schedule(inplace_call, debug_mask=debug_mask)
    block = sch.get_block("ext_call")
    blocks = sch.cache_inplace(block, 0, "local")
    block = sch.cache_read(blocks[0], 0, "global", [blocks[0]])
    block = sch.cache_write(blocks[1], 0, "global")

    assert_structural_equal_ignore_global_symbol(cache_inplace_buffer, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=inplace_call, debug_mask=debug_mask)


def test_cache_read_nested_seq(use_block_name):
    sch = tir.Schedule(func_nested_seq, debug_mask="all")
    block_c = "C" if use_block_name else sch.get_block("C")
    sch.cache_read(block_c, 0, "global", consumer_blocks=[block_c])
    assert_structural_equal_ignore_global_symbol(cache_read_nested_seq_target, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=func_nested_seq)


########## Testcases for cache_write ##########


def test_cache_write_elementwise(use_block_name):
    sch = tir.Schedule(elementwise, debug_mask="all")
    block_b = sch.get_block("B")
    block_c = sch.get_block("C")
    cached_b = sch.cache_write("B" if use_block_name else block_b, 0, "local")
    cached_c = sch.cache_write("C" if use_block_name else block_c, 0, "global")
    assert sch.get(cached_b) == sch.get(sch.get_block("B_local"))
    assert sch.get(cached_c) == sch.get(sch.get_block("C_global"))
    assert sch.get(block_b) == sch.get(sch.get_block("B"))
    assert sch.get(block_c) == sch.get(sch.get_block("C"))
    assert_structural_equal_ignore_global_symbol(cache_write_elementwise, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=elementwise)


def test_cache_write_under_scope(use_block_name):
    sch = tir.Schedule(access_under_scope, debug_mask="all")
    block_a = "A" if use_block_name else sch.get_block("A")
    block_b = "B" if use_block_name else sch.get_block("B")
    block_scope = sch.get_block("scope")
    sch.cache_write(block_a, 0, "local")
    sch.cache_write(block_b, 0, "global")
    sch.cache_write(block_scope, 0, "global")
    assert_structural_equal_ignore_global_symbol(cache_write_under_scope, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=access_under_scope)


def test_cache_write_opaque_access(use_block_name):
    sch = tir.Schedule(opaque_access, debug_mask="all")
    block_store = "load_store" if use_block_name else sch.get_block("load_store")
    block_opaque = "opaque" if use_block_name else sch.get_block("opaque")
    block_match_buffer = "match_buffer" if use_block_name else sch.get_block("match_buffer")
    sch.cache_write(block_store, 0, "global")
    sch.cache_write(block_opaque, 0, "global")
    sch.cache_write(block_match_buffer, 0, "global")
    assert_structural_equal_ignore_global_symbol(cache_write_opaque_access, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=opaque_access)


def test_cache_write_location(use_block_name):
    sch = tir.Schedule(func_multi_consumer, debug_mask="all")
    block_a = "A" if use_block_name else sch.get_block("A")
    sch.cache_write(block_a, 0, "global")
    assert_structural_equal_ignore_global_symbol(cache_write_multi_consumer, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=func_multi_consumer)

    # Test that specific consumer block targeting works.
    # B read cache buffer and C read original output buffer
    sch = tir.Schedule(func_multi_consumer, debug_mask="all")
    block_a = "A" if use_block_name else sch.get_block("A")
    block_b = "B" if use_block_name else sch.get_block("B")
    sch.cache_write(block_a, 0, "global", consumer_blocks=[block_b])
    assert_structural_equal_ignore_global_symbol(
        cache_write_multi_consumer_B_consume_cache, sch.mod["main"]
    )
    verify_trace_roundtrip(sch=sch, mod=func_multi_consumer)

    # Test that specific consumer block targeting works.
    # B read original output buffer and C read cache buffer
    sch = tir.Schedule(func_multi_consumer, debug_mask="all")
    block_a = "A" if use_block_name else sch.get_block("A")
    block_c = "C" if use_block_name else sch.get_block("C")
    sch.cache_write(block_a, 0, "global", consumer_blocks=[block_c])
    assert_structural_equal_ignore_global_symbol(
        cache_write_multi_consumer_C_consume_cache, sch.mod["main"]
    )
    verify_trace_roundtrip(sch=sch, mod=func_multi_consumer)

    # Test that specific consumer block targeting works.
    # B and C read cache buffer
    sch = tir.Schedule(func_multi_consumer, debug_mask="all")
    block_a = "A" if use_block_name else sch.get_block("A")
    block_b = "B" if use_block_name else sch.get_block("B")
    block_c = "C" if use_block_name else sch.get_block("C")
    sch.cache_write(block_a, 0, "global", consumer_blocks=[block_b, block_c])
    assert_structural_equal_ignore_global_symbol(
        cache_write_multi_consumer_all_consume_cache, sch.mod["main"]
    )
    verify_trace_roundtrip(sch=sch, mod=func_multi_consumer)


def test_continuous_cache_write(use_block_name):
    sch = tir.Schedule(elementwise, debug_mask="all")
    block_b = "B" if use_block_name else sch.get_block("B")
    sch.cache_write(block_b, 0, "shared")
    sch.cache_write(block_b, 0, "local")
    assert_structural_equal_ignore_global_symbol(continuous_cache_write, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=elementwise)


def test_cache_write_with_block_predicate(use_block_name):
    # cache write for intermediate buffer
    sch = tir.Schedule(func_with_block_predicate, debug_mask="all")
    block = "producer" if use_block_name else sch.get_block("producer")
    sch.cache_write(block, 0, "shared")
    assert_structural_equal_ignore_global_symbol(
        block_predicate_cache_write_intermediate_buf, sch.mod["main"]
    )
    verify_trace_roundtrip(sch=sch, mod=func_with_block_predicate)
    # cache write for external buffer
    sch = tir.Schedule(func_with_block_predicate, debug_mask="all")
    block = "consumer" if use_block_name else sch.get_block("consumer")
    sch.cache_write(block, 0, "shared")
    assert_structural_equal_ignore_global_symbol(
        block_predicate_cache_write_output_buf, sch.mod["main"]
    )
    verify_trace_roundtrip(sch=sch, mod=func_with_block_predicate)


def test_cache_write_fail_multi_producer(use_block_name):
    sch = tir.Schedule(func_multi_producer, debug_mask="all")
    block_a0 = "A0" if use_block_name else sch.get_block("A0")
    block_a1 = "A1" if use_block_name else sch.get_block("A1")
    with pytest.raises(tvm.tir.ScheduleError):
        sch.cache_write(block_a0, 0, "global")
    with pytest.raises(tvm.tir.ScheduleError):
        sch.cache_write(block_a1, 0, "global")


def test_cache_write_fail_index_out_of_bound(use_block_name):
    sch = tir.Schedule(elementwise, debug_mask="all")
    block_b = "B" if use_block_name else sch.get_block("B")
    with pytest.raises(tvm.tir.ScheduleError):
        sch.cache_write(block_b, 1, "global")


def test_cache_write_fail_invalid_storage_scope(use_block_name):
    sch = tir.Schedule(elementwise, debug_mask="all")
    block_b = "B" if use_block_name else sch.get_block("B")
    with pytest.raises(tvm.tir.ScheduleError):
        sch.cache_write(block_b, 0, "test_scope")


@pytest.mark.parametrize("use_decl_buffer", [True, False])
def test_cache_write_allocate_const(use_decl_buffer):
    def apply_decl_buffer(*args, **kwargs):
        if use_decl_buffer:
            return T.decl_buffer(*args, **kwargs)
        else:
            return T.Buffer(*args, **kwargs)

    @T.prim_func
    def before(A: T.Buffer((128, 128), "float32"), C: T.Buffer((128, 128), "float16")):
        B = T.alloc_buffer([128, 128], dtype="float32")
        const1 = T.allocate_const([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], "float32", [8])
        const1_buf = apply_decl_buffer([8], dtype="float32", data=const1)
        const2 = T.allocate_const([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], "float32", [8])
        const2_buf = apply_decl_buffer([8], dtype="float32", data=const2)
        for i, j in T.grid(128, 128):
            for x in range(8):
                with T.block("B"):
                    vi, vj, vx = T.axis.remap("SSS", [i, j, x])
                    T.reads(A[vi, vj], const1_buf[vx], const2_buf[vx])
                    T.writes(B[vi, vj])
                    B[vi, vj] = A[vi, vj] * const1_buf[vx] + const2_buf[vx]
        for i, j in T.grid(128, 128):
            with T.block("C"):
                vi, vj = T.axis.remap("SS", [i, j])
                T.reads(B[vi, vj])
                T.writes(C[vi, vj])
                C[vi, vj] = B[vi, vj] + 1.0

    @T.prim_func
    def expected(A: T.Buffer((128, 128), "float32"), C: T.Buffer((128, 128), "float16")):
        B = T.alloc_buffer([128, 128], dtype="float32")
        A_global = T.alloc_buffer([128, 128], dtype="float32")
        C_global = T.alloc_buffer([128, 128], dtype="float16")
        const1 = T.allocate_const([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], "float32", [8])
        const1_buf = apply_decl_buffer([8], dtype="float32", data=const1)
        const2 = T.allocate_const([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], "float32", [8])
        const2_buf = apply_decl_buffer([8], dtype="float32", data=const2)
        for ax0, ax1 in T.grid(128, 128):
            with T.block("A_global"):
                v0, v1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(A[v0, v1])
                T.writes(A_global[v0, v1])
                A_global[v0, v1] = A[v0, v1]
        for i, j, x in T.grid(128, 128, 8):
            with T.block("B"):
                vi, vj, vx = T.axis.remap("SSS", [i, j, x])
                T.reads(A_global[vi, vj], const1_buf[vx], const2_buf[vx])
                T.writes(B[vi, vj])
                B[vi, vj] = A_global[vi, vj] * const1_buf[vx] + const2_buf[vx]
        for i, j in T.grid(128, 128):
            with T.block("C"):
                vi, vj = T.axis.remap("SS", [i, j])
                T.reads(B[vi, vj])
                T.writes(C_global[vi, vj])
                C_global[vi, vj] = B[vi, vj] + T.float32(1)
        for ax0, ax1 in T.grid(128, 128):
            with T.block("C_global"):
                v0, v1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(C_global[v0, v1])
                T.writes(C[v0, v1])
                C[v0, v1] = C_global[v0, v1]

    sch = tir.Schedule(before)
    block_b = sch.get_block("B")
    block_c = sch.get_block("C")
    sch.cache_read(block_b, 0, "global")
    sch.cache_write(block_c, 0, "global")

    after = sch.mod["main"]

    assert_structural_equal_ignore_global_symbol(expected, after)
    verify_trace_roundtrip(sch=sch, mod=before)


def test_reindex_cache_read():
    sch = tir.Schedule(elementwise, debug_mask="all")
    sch.reindex_cache_read("C", 0, "shared", lambda i, j: (j, i // 2, i % 2))
    assert_structural_equal_ignore_global_symbol(elementwise_reindex_cache_read, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=elementwise)


def test_reindex_cache_read_multi_consumer():
    sch = tir.Schedule(func_multi_consumer)
    sch.reindex_cache_read("B", 0, "shared", lambda i: (i // 32, i % 32))
    assert_structural_equal_ignore_global_symbol(reindex_cache_read_multi_consumer, sch.mod["main"])
    # NOTE(zihao): we do not verify trace roundtrip because of in set analysis issues.


def test_reindex_cache_read_fail_not_match():
    sch = tir.Schedule(elementwise, debug_mask="all")
    with pytest.raises(tvm.tir.ScheduleError):
        sch.reindex_cache_read(
            "C",
            0,
            "shared",
            lambda i, j: j * 2,
        )


def test_reindex_cache_read_failed_not_single_point():
    sch = tir.Schedule(access_under_scope, debug_mask="all")
    with pytest.raises(tvm.tir.ScheduleError):
        sch.reindex_cache_read("scope", 0, "shared", lambda i, j: (i, j))


def test_reindex_cache_write():
    sch = tir.Schedule(elementwise, debug_mask="all")
    sch.reindex_cache_write("B", 0, "shared", lambda i, j: (j, i))
    assert_structural_equal_ignore_global_symbol(elementwise_reindex_cache_write, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=elementwise)


def test_reindex_cache_write_reduce():
    sch = tir.Schedule(reduce, debug_mask="all")
    sch.reindex_cache_write("B", 0, "shared", lambda i, j, k, l: (j, i, k))
    assert_structural_equal_ignore_global_symbol(reduce_reindex_cache_write_0, sch.mod["main"])
    sch.reindex_cache_write("C", 0, "shared", lambda i, j, k: [j, i])
    assert_structural_equal_ignore_global_symbol(reduce_reindex_cache_write_1, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=reduce)


def test_reindex_cache_write_fail_not_match():
    sch = tir.Schedule(elementwise, debug_mask="all")
    with pytest.raises(tvm.tir.ScheduleError):
        sch.reindex_cache_write(
            "B",
            0,
            "shared",
            lambda i, j: i,
        )


def test_reindex_cache_write_fail_not_single_point():
    sch = tir.Schedule(access_under_scope, debug_mask="all")
    with pytest.raises(tvm.tir.ScheduleError):
        sch.reindex_cache_write("scope", 0, "shared", lambda i, j: (i, j))


def test_symbolic_matmul_blocked_cache_read(use_block_name):
    sch = tir.Schedule(symbolic_matmul_blocked, debug_mask="all")
    block = "matmul" if use_block_name else sch.get_block("matmul")
    sch.cache_read(block=block, read_buffer_index=0, storage_scope="shared")
    assert_structural_equal_ignore_global_symbol(
        sch.mod["main"], symbolic_matmul_blocked_cache_read
    )
    verify_trace_roundtrip(sch=sch, mod=symbolic_matmul_blocked)


def test_symbolic_matmul_blocked_cache_write(use_block_name):
    sch = tir.Schedule(symbolic_matmul_blocked, debug_mask="all")
    block = "matmul" if use_block_name else sch.get_block("matmul")
    sch.cache_write(block=block, write_buffer_index=0, storage_scope="local")
    assert_structural_equal_ignore_global_symbol(
        sch.mod["main"], symbolic_matmul_blocked_cache_write
    )
    verify_trace_roundtrip(sch=sch, mod=symbolic_matmul_blocked)


if __name__ == "__main__":
    tvm.testing.main()
