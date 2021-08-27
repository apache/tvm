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
from tvm.script import ty
from tvm.tir.schedule.testing import verify_trace_roundtrip

# pylint: disable=no-member,invalid-name,unused-variable

########## Function before schedule ##########


@tvm.script.tir
def elementwise(a: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, (128, 128))
    B = tir.alloc_buffer((128, 128))
    C = tir.match_buffer(c, (128, 128))
    with tir.block([128, 128], "B") as [vi, vj]:
        B[vi, vj] = A[vi, vj] * 2.0
    with tir.block([128, 128], "C") as [vi, vj]:
        C[vi, vj] = B[vi, vj] + 1.0


@tvm.script.tir
def access_under_scope(b: ty.handle, c: ty.handle) -> None:
    A = tir.alloc_buffer((128, 128))
    B = tir.match_buffer(b, (128, 128))
    C = tir.match_buffer(c, (128, 128))

    with tir.block([8, 8], "scope") as [i, j]:
        for x, y in tir.grid(16, 16):
            with tir.block([128, 128], "A") as [vi, vj]:
                tir.bind(vi, i * 16 + x)
                tir.bind(vj, j * 16 + y)
                A[vi, vj] = 1.0
        for x, y in tir.grid(16, 16):
            with tir.block([128, 128], "B") as [vi, vj]:
                tir.bind(vi, i * 16 + x)
                tir.bind(vj, j * 16 + y)
                B[vi, vj] = A[vi, vj] + 1.0

    with tir.block([128, 128], "C") as [vi, vj]:
        C[vi, vj] = A[vi, vj] * 2.0


@tvm.script.tir
def opaque_access(a: ty.handle, b: ty.handle, c: ty.handle, d: ty.handle) -> None:
    A = tir.match_buffer(a, (128, 128), dtype="float16")
    B = tir.match_buffer(b, (128, 128), dtype="float16")
    C = tir.match_buffer(c, (128, 128), dtype="float16")
    D = tir.match_buffer(d, (128, 128), dtype="float16")

    with tir.block([128, 128], "load_store") as [vi, vj]:
        tir.reads(A[vi, vj])
        tir.writes(D[vi, vj])
        D.data[vi * 128 + vj] = tir.load("float16", A.data, vi * 128 + vj)
    with tir.block([8, 8], "opaque") as [vi, vj]:
        tir.reads(A[vi * 16 : vi * 16 + 16, vj * 16 : vj * 16 + 16])
        tir.writes(B[vi * 16 : vi * 16 + 16, vj * 16 : vj * 16 + 16])
        tir.evaluate(
            tir.tvm_load_matrix_sync(
                B.data,
                16,
                16,
                16,
                vi * 8 + vj,
                tir.tvm_access_ptr(
                    tir.type_annotation(dtype="float16"),
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
    with tir.block([8, 8], "match_buffer") as [vi, vj]:
        tir.reads(A[vi * 16 : vi * 16 + 16, vj * 16 : vj * 16 + 16])
        tir.writes(C[vi * 16 : vi * 16 + 16, vj * 16 : vj * 16 + 16])
        A0 = tir.match_buffer(
            A[
                vi * 16 : vi * 16 + 16,
                vj * 16 : vj * 16 + 16,
            ],
            (16, 16),
            "float16",
            strides=[128, 1],
            offset_factor=1,
        )
        C0 = tir.match_buffer(
            C[
                vi * 16 : vi * 16 + 16,
                vj * 16 : vj * 16 + 16,
            ],
            (16, 16),
            "float16",
            strides=[128, 1],
            offset_factor=1,
        )
        tir.evaluate(
            tir.tvm_load_matrix_sync(
                C0.data,
                16,
                16,
                16,
                vi * 8 + vj,
                tir.tvm_access_ptr(
                    tir.type_annotation(dtype="float16"),
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


@tvm.script.tir
def func_multi_consumer() -> None:
    A = tir.alloc_buffer((128))
    B = tir.alloc_buffer((128))
    C = tir.alloc_buffer((128))
    for i in tir.grid(8):
        for j in tir.grid(16):
            with tir.block([128], "A") as [vi]:
                tir.bind(vi, i * 16 + j)
                A[vi] = 1.0
        for j in tir.grid(16):
            with tir.block([128], "B") as [vi]:
                tir.bind(vi, i * 16 + j)
                B[vi] = A[vi] + 1.0
    for i in tir.grid(128):
        with tir.block([128], "C") as [vi]:
            C[vi] = A[vi]


@tvm.script.tir
def func_multi_producer() -> None:
    A = tir.alloc_buffer((128))
    B = tir.alloc_buffer((128))
    with tir.block([128], "A0") as [vi]:
        A[vi] = 1.0
    with tir.block([128], "A1") as [vi]:
        A[vi] = 2.0
    with tir.block([128], "B") as [vi]:
        B[vi] = A[vi]


########## Expected function after cache_read ##########


@tvm.script.tir
def cache_read_elementwise(a: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, (128, 128))
    C = tir.match_buffer(c, (128, 128))
    B = tir.alloc_buffer((128, 128))
    A_global = tir.alloc_buffer((128, 128))
    B_local = tir.alloc_buffer((128, 128), scope="local")
    with tir.block([128, 128], "A_global") as [vi, vj]:
        A_global[vi, vj] = A[vi, vj]
    with tir.block([128, 128], "B") as [vi, vj]:
        B[vi, vj] = A_global[vi, vj] * 2.0
    with tir.block([128, 128], "B_local") as [vi, vj]:
        B_local[vi, vj] = B[vi, vj]
    with tir.block([128, 128], "C") as [vi, vj]:
        C[vi, vj] = B_local[vi, vj] + 1.0


@tvm.script.tir
def cache_read_under_scope(b: ty.handle, c: ty.handle) -> None:
    A = tir.alloc_buffer((128, 128))
    B = tir.match_buffer(b, (128, 128))
    C = tir.match_buffer(c, (128, 128))
    A_global = tir.alloc_buffer((128, 128))

    with tir.block([8, 8], "scope") as [i, j]:
        A_local = tir.alloc_buffer((128, 128), scope="local")
        for x, y in tir.grid(16, 16):
            with tir.block([128, 128], "A") as [vi, vj]:
                tir.bind(vi, i * 16 + x)
                tir.bind(vj, j * 16 + y)
                A[vi, vj] = 1.0
        for x, y in tir.grid(16, 16):
            with tir.block([128, 128], "A_local") as [vi, vj]:
                tir.bind(vi, i * 16 + x)
                tir.bind(vj, j * 16 + y)
                A_local[vi, vj] = A[vi, vj]
        for x, y in tir.grid(16, 16):
            with tir.block([128, 128], "B") as [vi, vj]:
                tir.bind(vi, i * 16 + x)
                tir.bind(vj, j * 16 + y)
                B[vi, vj] = A_local[vi, vj] + 1.0
    with tir.block([128, 128], "A_global") as [vi, vj]:
        A_global[vi, vj] = A[vi, vj]
    with tir.block([128, 128], "C") as [vi, vj]:
        C[vi, vj] = A_global[vi, vj] * 2.0


@tvm.script.tir
def cache_read_opaque_access(a: ty.handle, b: ty.handle, c: ty.handle, d: ty.handle) -> None:
    A = tir.match_buffer(a, (128, 128), dtype="float16")
    B = tir.match_buffer(b, (128, 128), dtype="float16")
    C = tir.match_buffer(c, (128, 128), dtype="float16")
    D = tir.match_buffer(d, (128, 128), dtype="float16")
    A_global = tir.alloc_buffer((128, 128), dtype="float16")

    with tir.block([128, 128], "A_global") as [vi, vj]:
        A_global[vi, vj] = A[vi, vj]
    with tir.block([128, 128], "load_store") as [vi, vj]:
        tir.reads(A_global[vi, vj])
        tir.writes(D[vi, vj])
        D.data[vi * 128 + vj] = tir.load("float16", A_global.data, vi * 128 + vj)
    with tir.block([8, 8], "opaque") as [vi, vj]:
        tir.reads(A_global[vi * 16 : vi * 16 + 16, vj * 16 : vj * 16 + 16])
        tir.writes(B[vi * 16 : vi * 16 + 16, vj * 16 : vj * 16 + 16])
        tir.evaluate(
            tir.tvm_load_matrix_sync(
                B.data,
                16,
                16,
                16,
                vi * 8 + vj,
                tir.tvm_access_ptr(
                    tir.type_annotation(dtype="float16"),
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
    with tir.block([8, 8], "match_buffer") as [vi, vj]:
        tir.reads(A_global[vi * 16 : vi * 16 + 16, vj * 16 : vj * 16 + 16])
        tir.writes(C[vi * 16 : vi * 16 + 16, vj * 16 : vj * 16 + 16])
        A0 = tir.match_buffer(
            A_global[
                vi * 16 : vi * 16 + 16,
                vj * 16 : vj * 16 + 16,
            ],
            (16, 16),
            "float16",
            strides=[128, 1],
            offset_factor=1,
        )
        C0 = tir.match_buffer(
            C[
                vi * 16 : vi * 16 + 16,
                vj * 16 : vj * 16 + 16,
            ],
            (16, 16),
            "float16",
            strides=[128, 1],
            offset_factor=1,
        )
        tir.evaluate(
            tir.tvm_load_matrix_sync(
                C0.data,
                16,
                16,
                16,
                vi * 8 + vj,
                tir.tvm_access_ptr(
                    tir.type_annotation(dtype="float16"),
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


@tvm.script.tir
def cache_read_multi_consumer() -> None:
    A = tir.alloc_buffer((128))
    B = tir.alloc_buffer((128))
    C = tir.alloc_buffer((128))
    A_global = tir.alloc_buffer((128))
    for i in tir.grid(8):
        for j in tir.grid(16):
            with tir.block([128], "A") as [vi]:
                tir.bind(vi, i * 16 + j)
                A[vi] = 1.0
        for j in tir.grid(16):
            with tir.block([128], "A") as [vi]:
                tir.bind(vi, i * 16 + j)
                A_global[vi] = A[vi]
        for j in tir.grid(16):
            with tir.block([128], "B") as [vi]:
                tir.bind(vi, i * 16 + j)
                B[vi] = A_global[vi] + 1.0

    for i in tir.grid(128):
        with tir.block([128], "C") as [vi]:
            C[vi] = A_global[vi]


########## Expected function after cache_write ##########


@tvm.script.tir
def cache_write_elementwise(a: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, (128, 128))
    C = tir.match_buffer(c, (128, 128))
    B = tir.alloc_buffer((128, 128))
    B_global = tir.alloc_buffer((128, 128))
    C_local = tir.alloc_buffer((128, 128), scope="local")
    with tir.block([128, 128], "B_global") as [vi, vj]:
        B_global[vi, vj] = A[vi, vj] * 2.0
    with tir.block([128, 128], "B") as [vi, vj]:
        B[vi, vj] = B_global[vi, vj]
    with tir.block([128, 128], "C_local") as [vi, vj]:
        C_local[vi, vj] = B[vi, vj] + 1.0
    with tir.block([128, 128], "C") as [vi, vj]:
        C[vi, vj] = C_local[vi, vj]


@tvm.script.tir
def cache_write_under_scope(b: ty.handle, c: ty.handle) -> None:
    A = tir.alloc_buffer((128, 128))
    B = tir.match_buffer(b, (128, 128))
    C = tir.match_buffer(c, (128, 128))
    A_global = tir.alloc_buffer((128, 128))

    with tir.block([8, 8], "scope") as [i, j]:
        A_local = tir.alloc_buffer((128, 128), scope="local")
        B_global = tir.alloc_buffer((128, 128))
        for x, y in tir.grid(16, 16):
            with tir.block([128, 128], "A_local") as [vi, vj]:
                tir.bind(vi, i * 16 + x)
                tir.bind(vj, j * 16 + y)
                A_local[vi, vj] = 1.0
        for x, y in tir.grid(16, 16):
            with tir.block([128, 128], "A") as [vi, vj]:
                tir.bind(vi, i * 16 + x)
                tir.bind(vj, j * 16 + y)
                A_global[vi, vj] = A_local[vi, vj]
        for x, y in tir.grid(16, 16):
            with tir.block([128, 128], "B_global") as [vi, vj]:
                tir.bind(vi, i * 16 + x)
                tir.bind(vj, j * 16 + y)
                B_global[vi, vj] = A_global[vi, vj] + 1.0
        for x, y in tir.grid(16, 16):
            with tir.block([128, 128], "B_global") as [vi, vj]:
                tir.bind(vi, i * 16 + x)
                tir.bind(vj, j * 16 + y)
                B[vi, vj] = B_global[vi, vj]
    with tir.block([128, 128], "A_global") as [vi, vj]:
        A[vi, vj] = A_global[vi, vj]
    with tir.block([128, 128], "C") as [vi, vj]:
        C[vi, vj] = A[vi, vj] * 2.0


@tvm.script.tir
def cache_write_opaque_access(a: ty.handle, b: ty.handle, c: ty.handle, d: ty.handle) -> None:
    A = tir.match_buffer(a, (128, 128), dtype="float16")
    B = tir.match_buffer(b, (128, 128), dtype="float16")
    C = tir.match_buffer(c, (128, 128), dtype="float16")
    D = tir.match_buffer(d, (128, 128), dtype="float16")
    D_global = tir.alloc_buffer((128, 128), dtype="float16")
    B_global = tir.alloc_buffer((128, 128), dtype="float16")
    C_global = tir.alloc_buffer((128, 128), dtype="float16")

    with tir.block([128, 128], "load_store") as [vi, vj]:
        tir.reads(A[vi, vj])
        tir.writes(D_global[vi, vj])
        D_global.data[vi * 128 + vj] = tir.load("float16", A.data, vi * 128 + vj)
    with tir.block([8, 8], "opaque") as [vi, vj]:
        tir.reads(A[vi * 16 : vi * 16 + 16, vj * 16 : vj * 16 + 16])
        tir.writes(B_global[vi * 16 : vi * 16 + 16, vj * 16 : vj * 16 + 16])
        tir.evaluate(
            tir.tvm_load_matrix_sync(
                B_global.data,
                16,
                16,
                16,
                vi * 8 + vj,
                tir.tvm_access_ptr(
                    tir.type_annotation(dtype="float16"),
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
    with tir.block([8, 8], "match_buffer") as [vi, vj]:
        tir.reads(A[vi * 16 : vi * 16 + 16, vj * 16 : vj * 16 + 16])
        tir.writes(C_global[vi * 16 : vi * 16 + 16, vj * 16 : vj * 16 + 16])
        A0 = tir.match_buffer(
            A[
                vi * 16 : vi * 16 + 16,
                vj * 16 : vj * 16 + 16,
            ],
            (16, 16),
            "float16",
            strides=[128, 1],
            offset_factor=1,
        )
        C0 = tir.match_buffer(
            C_global[
                vi * 16 : vi * 16 + 16,
                vj * 16 : vj * 16 + 16,
            ],
            (16, 16),
            "float16",
            strides=[128, 1],
            offset_factor=1,
        )
        tir.evaluate(
            tir.tvm_load_matrix_sync(
                C0.data,
                16,
                16,
                16,
                vi * 8 + vj,
                tir.tvm_access_ptr(
                    tir.type_annotation(dtype="float16"),
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

    with tir.block([128, 128], "D") as [vi, vj]:
        D[vi, vj] = D_global[vi, vj]
    with tir.block([128, 128], "B") as [vi, vj]:
        B[vi, vj] = B_global[vi, vj]
    with tir.block([128, 128], "C") as [vi, vj]:
        C[vi, vj] = C_global[vi, vj]


@tvm.script.tir
def cache_write_multi_consumer() -> None:
    A = tir.alloc_buffer((128))
    B = tir.alloc_buffer((128))
    C = tir.alloc_buffer((128))
    A_global = tir.alloc_buffer((128))
    for i in tir.grid(8):
        for j in tir.grid(16):
            with tir.block([128], "A_global") as [vi]:
                tir.bind(vi, i * 16 + j)
                A_global[vi] = 1.0
        for j in tir.grid(16):
            with tir.block([128], "A") as [vi]:
                tir.bind(vi, i * 16 + j)
                A[vi] = A_global[vi]
        for j in tir.grid(16):
            with tir.block([128], "B") as [vi]:
                tir.bind(vi, i * 16 + j)
                B[vi] = A[vi] + 1.0

    for i in tir.grid(128):
        with tir.block([128], "C") as [vi]:
            C[vi] = A[vi]


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
    sch.cache_read(block_b, 0, "global")
    sch.cache_read(block_c, 0, "local")
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


if __name__ == "__main__":
    sys.exit(pytest.main([__file__] + sys.argv[1:]))
