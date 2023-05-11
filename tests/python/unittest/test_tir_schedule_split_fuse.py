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
import pytest
import tvm
import tvm.testing
from tvm import te, tir
from tvm.script import tir as T
from tvm.tir.expr import IntImm
from tvm.tir.schedule.testing import verify_trace_roundtrip

# pylint: disable=no-member,invalid-name,unused-variable


@T.prim_func
def elementwise(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, (128, 128, 128))
    B = T.match_buffer(b, (128, 128, 128))
    for i, j, k in T.grid(128, 128, 128):
        with T.block("B"):
            vi, vj, vk = T.axis.remap("SSS", [i, j, k])
            B[vi, vj, vk] = A[vi, vj, vk] * 2.0


@T.prim_func
def elementwise_dependent_loops(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, (128, 128, 128))
    B = T.match_buffer(b, (128, 128, 128))
    for i in T.serial(0, 128):
        for j, k in T.grid(i, 128):
            with T.block("B"):
                vi = T.axis.S(128, i)
                vj = T.axis.S(i, j)
                vk = T.axis.S(128, k)
                B[vi, vj, vk] = A[vi, vj, vk] * 2.0


@T.prim_func
def elementwise_symbolic(a: T.handle, b: T.handle, n: T.int32) -> None:
    A = T.match_buffer(a, (128, 128, n))
    B = T.match_buffer(b, (128, 128, n))
    for i, j, k in T.grid(128, 128, n):
        with T.block("B"):
            vi, vj, vk = T.axis.remap("SSS", [i, j, k])
            B[vi, vj, vk] = A[vi, vj, vk] * 2.0


@T.prim_func
def elementwise_symbolic_fused(a: T.handle, b: T.handle, n: T.int32) -> None:
    A = T.match_buffer(a, (128, 128, n))
    B = T.match_buffer(b, (128, 128, n))
    for i_j_k_fused in T.serial(0, (n * 16384)):
        with T.block("B"):
            vi = T.axis.S(128, T.floordiv(i_j_k_fused, n * 128))
            vj = T.axis.S(128, T.floordiv(T.floormod(i_j_k_fused, n * 128), n))
            vk = T.axis.S(n, T.floormod(i_j_k_fused, n))
            T.reads([A[vi, vj, vk]])
            T.writes([B[vi, vj, vk]])
            B[vi, vj, vk] = A[vi, vj, vk] * 2.0


@T.prim_func
def elementwise_symbolic_split(a: T.handle, b: T.handle, n: T.int32) -> None:
    A = T.match_buffer(a, (128, 128, n))
    B = T.match_buffer(b, (128, 128, n))
    for i, j, k0, k1 in T.grid(128, 128, 10, T.floordiv((n + 9), 10)):
        with T.block("B"):
            T.where((((k0 * T.floordiv((n + 9), 10)) + k1) < n))
            vi, vj = T.axis.remap("SS", [i, j])
            vk = T.axis.S(n, k0 * T.floordiv(n + 9, 10) + k1)
            T.reads([A[vi, vj, vk]])
            T.writes([B[vi, vj, vk]])
            B[vi, vj, vk] = A[vi, vj, vk] * 2.0


@T.prim_func
def elementwise_with_seq(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, (128, 128, 128))
    B = T.match_buffer(b, (128, 128, 128))
    C = T.alloc_buffer((128, 128, 128))
    for i, j in T.grid(128, 128):
        for k in T.serial(0, 128):
            with T.block("C"):
                vi, vj, vk = T.axis.remap("SSS", [i, j, k])
                C[vi, vj, vk] = A[vi, vj, vk] * 2.0
        for k in T.serial(0, 128):
            with T.block("B"):
                vi, vj, vk = T.axis.remap("SSS", [i, j, k])
                B[vi, vj, vk] = C[vi, vj, vk] * 2.0


@T.prim_func
def elementwise_with_anno(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, (128, 128, 128))
    B = T.match_buffer(b, (128, 128, 128))
    for i, j in T.grid(128, 128):
        for k in T.serial(0, 128, annotations={"useless_annotation": True}):
            with T.block("B"):
                vi, vj, vk = T.axis.remap("SSS", [i, j, k])
                T.reads([A[vi, vj, vk]])
                T.writes([B[vi, vj, vk]])
                B[vi, vj, vk] = A[vi, vj, vk] * 2.0


@T.prim_func
def elementwise_with_thread_binding(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, (128, 128, 128))
    B = T.match_buffer(b, (128, 128, 128))
    for i, j in T.grid(128, 128):
        for k in T.thread_binding(0, 128, thread="threadIdx.x"):
            with T.block("B"):
                vi, vj, vk = T.axis.remap("SSS", [i, j, k])
                T.reads([A[vi, vj, vk]])
                T.writes([B[vi, vj, vk]])
                B[vi, vj, vk] = A[vi, vj, vk] * 2.0


@T.prim_func
def elementwise_with_starting_point(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, (128, 128, 128))
    B = T.match_buffer(b, (128, 128, 128))
    for i, j in T.grid(128, 128):
        for k in T.serial(10, 128):
            with T.block("B"):
                vi, vj, vk = T.axis.remap("SSS", [i, j, k])
                T.reads([A[vi, vj, vk]])
                T.writes([B[vi, vj, vk]])
                B[vi, vj, vk] = A[vi, vj, vk] * 2.0


@T.prim_func
def elementwise_with_opaque_block(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, (128, 128, 128))
    B = T.match_buffer(b, (128, 128, 128))
    for i, j, k in T.grid(128, 128, 128):
        with T.block("opaque"):
            T.reads([A[i, j, k]])
            T.writes([B[i, j, k]])
            with T.block("B"):
                vi, vj, vk = T.axis.remap("SSS", [i, j, k])
                T.reads([A[vi, vj, vk]])
                T.writes([B[vi, vj, vk]])
                B[vi, vj, vk] = A[vi, vj, vk] * 2.0


@T.prim_func
def elementwise_fused(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, (128, 128, 128))
    B = T.match_buffer(b, (128, 128, 128))
    for fused in T.serial(0, 2097152):
        with T.block("B"):
            vi = T.axis.S(128, T.floordiv(fused, 16384))
            vj = T.axis.S(128, T.floordiv(T.floormod(fused, 16384), 128))
            vk = T.axis.S(128, T.floormod(fused, 128))
            T.reads([A[vi, vj, vk]])
            T.writes([B[vi, vj, vk]])
            B[vi, vj, vk] = A[vi, vj, vk] * 2.0


@T.prim_func
def elementwise_split_case0(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, [128, 128, 128])
    B = T.match_buffer(b, [128, 128, 128])
    for i1, i2, i3, j1, j2, k1, k2 in T.grid(2, 1, 64, 4, 32, 16, 8):
        with T.block("B"):
            vi = T.axis.S(128, i1 * 64 + i2 * 64 + i3)
            vj = T.axis.S(128, j1 * 32 + j2)
            vk = T.axis.S(128, k1 * 8 + k2)
            T.reads([A[vi, vj, vk]])
            T.writes([B[vi, vj, vk]])
            B[vi, vj, vk] = A[vi, vj, vk] * 2.0


@T.prim_func
def elementwise_split_case1(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, [128, 128, 128])
    B = T.match_buffer(b, [128, 128, 128])
    for i1, i2, i3, j1, j2, j3, k1, k2, k3 in T.grid(2, 1, 64, 2, 1, 64, 2, 1, 64):
        with T.block("B"):
            vi = T.axis.S(128, i1 * 64 + i2 * 64 + i3)
            vj = T.axis.S(128, j1 * 64 + j2 * 64 + j3)
            vk = T.axis.S(128, k1 * 64 + k2 * 64 + k3)
            T.reads([A[vi, vj, vk]])
            T.writes([B[vi, vj, vk]])
            B[vi, vj, vk] = A[vi, vj, vk] * 2.0


@T.prim_func
def elementwise_split_with_predicate(a: T.handle, b: T.handle) -> None:
    B = T.match_buffer(b, [128, 128, 128])
    A = T.match_buffer(a, [128, 128, 128])
    for i0, i1, i2, j0, j1, k0, k1 in T.grid(1000, 2, 3, 1, 129, 3, 43):
        with T.block("B"):
            vi = T.axis.S(128, i0 * 6 + i1 * 3 + i2)
            vj = T.axis.S(128, j0 * 129 + j1)
            vk = T.axis.S(128, k0 * 43 + k1)
            T.where((i0 * 2 + i1) * 3 + i2 < 128 and j0 * 129 + j1 < 128 and k0 * 43 + k1 < 128)
            T.reads([A[vi, vj, vk]])
            T.writes([B[vi, vj, vk]])
            B[vi, vj, vk] = A[vi, vj, vk] * 2.0


@T.prim_func
def elementwise_fuse_with_opaque_block(a: T.handle, b: T.handle) -> None:
    B = T.match_buffer(b, [128, 128, 128])
    A = T.match_buffer(a, [128, 128, 128])
    for i_j_k_fused in T.serial(0, 2097152):
        with T.block("opaque"):
            T.reads(
                [
                    A[
                        T.floordiv(i_j_k_fused, 16384),
                        T.floordiv(T.floormod(i_j_k_fused, 16384), 128),
                        T.floormod(i_j_k_fused, 128),
                    ]
                ]
            )
            T.writes(
                [
                    B[
                        T.floordiv(i_j_k_fused, 16384),
                        T.floordiv(T.floormod(i_j_k_fused, 16384), 128),
                        T.floormod(i_j_k_fused, 128),
                    ]
                ]
            )
            with T.block("B"):
                vi = T.axis.S(128, T.floordiv(i_j_k_fused, 16384))
                vj = T.axis.S(128, T.floordiv(T.floormod(i_j_k_fused, 16384), 128))
                vk = T.axis.S(128, T.floormod(i_j_k_fused, 128))
                T.reads([A[vi, vj, vk]])
                T.writes([B[vi, vj, vk]])
                B[vi, vj, vk] = A[vi, vj, vk] * 2.0


@T.prim_func
def elementwise_split_with_opaque_block(a: T.handle, b: T.handle) -> None:
    B = T.match_buffer(b, [128, 128, 128])
    A = T.match_buffer(a, [128, 128, 128])

    for i0, i1, j, k in T.grid(8, 16, 128, 128):
        with T.block("opaque"):
            T.reads([A[i0 * 16 + i1, j, k]])
            T.writes([B[i0 * 16 + i1, j, k]])
            with T.block("B"):
                vi = T.axis.S(128, i0 * 16 + i1)
                vj, vk = T.axis.remap("SS", [j, k])
                T.reads([A[vi, vj, vk]])
                T.writes([B[vi, vj, vk]])
                B[vi, vj, vk] = A[vi, vj, vk] * 2.0


@T.prim_func
def opaque_access(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, [16, 16], "float32")
    B = T.match_buffer(b, [16, 16], "float32")
    for i, j in T.grid(16, 16):
        with T.block("A"):
            vi, vj = T.axis.remap("SS", [i, j])
            T.reads([])
            T.writes([A[0:16, 0:16]])
            A[vi, vj] = 1
    for i, j in T.grid(16, 16):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            T.reads([])
            T.writes([B[0:16, 0:16]])
            T.evaluate(T.tvm_fill_fragment(B.data, 16, 16, 16, 0, vi * 16 + vj, dtype="handle"))


@T.prim_func
def opaque_access_fused(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, [16, 16])
    B = T.match_buffer(b, [16, 16])
    for i_j_fused in T.serial(0, 256):
        with T.block("A"):
            vi = T.axis.S(16, T.floordiv(i_j_fused, 16))
            vj = T.axis.S(16, T.floormod(i_j_fused, 16))
            T.reads([])
            T.writes([A[0:16, 0:16]])
            A[vi, vj] = 1
    for i_j_fused in T.serial(0, 256):
        with T.block("B"):
            vi = T.axis.S(16, T.floordiv(i_j_fused, 16))
            vj = T.axis.S(16, T.floormod(i_j_fused, 16))
            T.reads([])
            T.writes([B[0:16, 0:16]])
            T.evaluate(T.tvm_fill_fragment(B.data, 16, 16, 16, 0, ((vi * 16) + vj), dtype="handle"))


@T.prim_func
def opaque_access_split(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, (16, 16))
    B = T.match_buffer(b, (16, 16))
    for i, j0, j1 in T.grid(16, 4, 4):
        with T.block("A"):
            vi = T.axis.S(16, i)
            vj = T.axis.S(16, j0 * 4 + j1)
            T.reads([])
            T.writes([A[0:16, 0:16]])
            A[vi, vj] = 1
    for i, j0, j1 in T.grid(16, 4, 4):
        with T.block("B"):
            vi = T.axis.S(16, i)
            vj = T.axis.S(16, j0 * 4 + j1)
            T.reads([])
            T.writes([B[0:16, 0:16]])
            T.evaluate(T.tvm_fill_fragment(B.data, 16, 16, 16, 0, ((vi * 16) + vj), dtype="handle"))


@T.prim_func
def elementwise_not_affine(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, (127, 128))
    B = T.match_buffer(b, (127, 128))
    for i in T.serial(0, 4):
        for j, k in T.grid(T.min(31, 126 - i * 32) + 1, 128):
            with T.block("B"):
                vi = T.axis.S(127, i * 32 + j)
                vj = T.axis.S(128, k)
                B[vi, vj] = A[vi, vj]


@T.prim_func
def elementwise_not_affine_fused(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, [127, 128])
    B = T.match_buffer(b, [127, 128])
    for i in T.grid(4):
        for j_k_fused in T.serial(0, T.min(31, 126 - i * 32) * 128 + 128):
            with T.block("B"):
                vi = T.axis.S(
                    127,
                    i * 32 + T.floordiv(j_k_fused, 128),
                )
                vj = T.axis.S(128, T.floormod(j_k_fused, 128))
                T.reads([A[vi, vj]])
                T.writes([B[vi, vj]])
                B[vi, vj] = A[vi, vj]


# pylint: enable=no-member,invalid-name,unused-variable


def test_fuse():
    sch = tir.Schedule(elementwise, debug_mask="all")
    block_b = sch.get_block("B")
    i, j, k = sch.get_loops(block_b)
    sch.fuse(i, j, k)
    tvm.ir.assert_structural_equal(elementwise_fused, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=elementwise)


def test_split():
    sch = tir.Schedule(elementwise, debug_mask="all")
    block_b = sch.get_block("B")
    i, j, k = sch.get_loops(block_b)
    sch.split(i, factors=[2, 1, 64])
    sch.split(j, factors=[4, 32])
    sch.split(k, factors=[16, 8])
    tvm.ir.assert_structural_equal(elementwise_split_case0, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=elementwise)


def test_split_with_inferred_factor():
    sch = tir.Schedule(elementwise, debug_mask="all")
    block_b = sch.get_block("B")
    i, j, k = sch.get_loops(block_b)
    sch.split(i, factors=[None, 1, 64])
    sch.split(j, factors=[2, None, 64])
    sch.split(k, factors=[2, 1, None])
    tvm.ir.assert_structural_equal(elementwise_split_case1, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=elementwise)


def test_split_with_predicate():
    sch = tir.Schedule(elementwise, debug_mask="all")
    block_b = sch.get_block("B")
    i, j, k = sch.get_loops(block_b)
    sch.split(i, factors=[1000, 2, 3])
    sch.split(j, factors=[None, 129])
    sch.split(k, factors=[3, None])
    tvm.ir.assert_structural_equal(elementwise_split_with_predicate, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=elementwise)


def test_fuse_fail_not_only_child():
    sch = tir.Schedule(elementwise_with_seq, debug_mask="all")
    block_b = sch.get_block("B")
    _, j, k = sch.get_loops(block_b)
    with pytest.raises(tvm.tir.ScheduleError):
        sch.fuse(j, k)


def test_fuse_split_fail_with_annotation():
    sch = tir.Schedule(elementwise_with_anno, debug_mask="all")
    block_b = sch.get_block("B")
    _, j, k = sch.get_loops(block_b)
    with pytest.raises(tvm.tir.ScheduleError):
        sch.fuse(j, k)
    with pytest.raises(tvm.tir.ScheduleError):
        sch.split(k, factors=[None, 10])


def test_fuse_split_fail_not_start_with_zero():
    sch = tir.Schedule(elementwise_with_anno, debug_mask="all")
    block_b = sch.get_block("B")
    _, j, k = sch.get_loops(block_b)
    with pytest.raises(tvm.tir.ScheduleError):
        sch.fuse(j, k)
    with pytest.raises(tvm.tir.ScheduleError):
        sch.split(k, factors=[None, 10])


def test_fuse_with_opaque_block():
    sch = tir.Schedule(elementwise_with_opaque_block, debug_mask="all")
    block_opaque = sch.get_block("opaque")
    i, j, k = sch.get_loops(block_opaque)
    sch.fuse(i, j, k)
    tvm.ir.assert_structural_equal(elementwise_fuse_with_opaque_block, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=elementwise_with_opaque_block)


def test_fuse_with_opaque_access():
    sch = tir.Schedule(opaque_access, debug_mask="all")
    block_a = sch.get_block("A")
    i, j = sch.get_loops(block_a)
    sch.fuse(i, j)
    block_b = sch.get_block("B")
    i, j = sch.get_loops(block_b)
    sch.fuse(i, j)
    tvm.ir.assert_structural_equal(opaque_access_fused, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=opaque_access)


def test_split_with_opaque_block():
    sch = tir.Schedule(elementwise_with_opaque_block, debug_mask="all")
    block_opaque = sch.get_block("opaque")
    i, _, _ = sch.get_loops(block_opaque)
    sch.split(i, factors=[None, 16])
    tvm.ir.assert_structural_equal(elementwise_split_with_opaque_block, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=elementwise_with_opaque_block)


def test_split_with_opaque_access():
    sch = tir.Schedule(opaque_access, debug_mask="all")
    block_a = sch.get_block("A")
    _, j = sch.get_loops(block_a)
    sch.split(j, factors=[None, 4])
    block_b = sch.get_block("B")
    _, j = sch.get_loops(block_b)
    sch.split(j, factors=[None, 4])
    tvm.ir.assert_structural_equal(opaque_access_split, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=opaque_access)


def test_split_with_non_positive_factors():
    sch = tir.Schedule(elementwise, debug_mask="all")
    block_b = sch.get_block("B")
    i, j, k = sch.get_loops(block_b)
    with pytest.raises(tvm.tir.ScheduleError):
        sch.split(i, factors=[-2, -64])
    with pytest.raises(tvm.tir.ScheduleError):
        sch.split(j, factors=[0, None])
    with pytest.raises(tvm.tir.ScheduleError):
        sch.split(k, factors=[None, -16])


def test_fuse_split_fail_with_thread_binding():
    sch = tir.Schedule(elementwise_with_thread_binding, debug_mask="all")
    block_b = sch.get_block("B")
    _, j, k = sch.get_loops(block_b)
    with pytest.raises(tvm.tir.ScheduleError):
        sch.fuse(j, k)
    with pytest.raises(tvm.tir.ScheduleError):
        sch.split(k, factors=[None, 10])


def test_fuse_symbolic():
    sch = tir.Schedule(elementwise_symbolic, debug_mask="all")
    block_b = sch.get_block("B")
    i, j, k = sch.get_loops(block_b)
    sch.fuse(i, j, k)
    tvm.ir.assert_structural_equal(elementwise_symbolic_fused, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=elementwise_symbolic)


def test_split_symbolic():
    sch = tir.Schedule(elementwise_symbolic, debug_mask="all")
    block_b = sch.get_block("B")
    _, _, k = sch.get_loops(block_b)
    sch.split(k, factors=[10, None])
    tvm.ir.assert_structural_equal(elementwise_symbolic_split, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=elementwise_symbolic)


def test_fuse_fail_with_dependent_loops():
    sch = tir.Schedule(elementwise_dependent_loops, debug_mask="all")
    block_b = sch.get_block("B")
    i, j, _ = sch.get_loops(block_b)
    with pytest.raises(tvm.tir.ScheduleError):
        sch.fuse(i, j)


def test_fuse_not_affine():
    sch = tir.Schedule(elementwise_not_affine, debug_mask="all")
    block_b = sch.get_block("B")
    _, j, k = sch.get_loops(block_b)
    sch.fuse(j, k)
    tvm.ir.assert_structural_equal(elementwise_not_affine_fused, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=elementwise_not_affine)


def test_add_unit_loop_above_block():
    @T.prim_func
    def zero_dim(
        A: T.Buffer((), "int32"),
        B: T.Buffer((), "int32"),
        C: T.Buffer((), "int32"),
    ) -> None:
        with T.block("C"):
            vi = T.axis.spatial(1, 0)
            C[()] = A[()] + B[()]

    @T.prim_func
    def zero_dim_added(
        A: T.Buffer((), "int32"),
        B: T.Buffer((), "int32"),
        C: T.Buffer((), "int32"),
    ) -> None:
        for u in range(1):
            with T.block("C"):
                vi = T.axis.spatial(1, 0)
                C[()] = A[()] + B[()]

    sch = tir.Schedule(zero_dim, debug_mask="all")
    block = sch.get_block("C")
    sch.add_unit_loop(block)
    tvm.ir.assert_structural_equal(zero_dim_added, sch.mod["main"])


def test_add_unit_loop_above_loop():
    @T.prim_func
    def zero_dim(
        A: T.Buffer((), "int32"),
        B: T.Buffer((), "int32"),
        C: T.Buffer((), "int32"),
    ) -> None:
        for u in range(1):
            with T.block("C"):
                vi = T.axis.spatial(1, 0)
                C[()] = A[()] + B[()]

    @T.prim_func
    def zero_dim_added(
        A: T.Buffer((), "int32"),
        B: T.Buffer((), "int32"),
        C: T.Buffer((), "int32"),
    ) -> None:
        for u1, u2 in T.grid(1, 1):
            with T.block("C"):
                vi = T.axis.spatial(1, 0)
                C[()] = A[()] + B[()]

    sch = tir.Schedule(zero_dim, debug_mask="all")
    block = sch.get_block("C")
    (loop,) = sch.get_loops(block)
    sch.add_unit_loop(loop)
    tvm.ir.assert_structural_equal(zero_dim_added, sch.mod["main"])


@pytest.mark.skip("Pending fix in affine analysis")
def test_fuse_int64():
    def _create_prim_func():
        n = te.const(16, "int32")
        m = te.const(32, "int64")
        A = te.placeholder((n, m), name="A", dtype="int32")
        B = te.compute((n, m), lambda i, j: A[i, j] + 1, name="B")
        return te.create_prim_func([A, B])

    mod = _create_prim_func()
    sch = tir.Schedule(mod, debug_mask="all")
    i, j = sch.get_loops(sch.get_block("B"))
    sch.fuse(i, j)
    verify_trace_roundtrip(sch=sch, mod=mod)


def test_split_int64_extent_with_mixed_factors():
    def _create_prim_func():
        m = te.const(384, "int64")
        A = te.placeholder((m,), name="A", dtype="float32")
        B = te.compute((m,), lambda i: A[i] + 1, name="B")
        return te.create_prim_func([A, B])

    mod = _create_prim_func()
    sch = tir.Schedule(mod, debug_mask="all")
    (i,) = sch.get_loops(sch.get_block("B"))
    sch.split(
        i,
        factors=[
            te.const(1, "int64"),
            te.const(512, "int32"),
        ],
    )


def test_split_int64_extent_with_int32_factors():
    def _create_prim_func():
        m = te.const(12, "int64")
        A = te.placeholder((m,), name="A", dtype="float32")
        B = te.compute((m,), lambda i: A[i] + 1, name="B")
        return te.create_prim_func([A, B])

    mod = _create_prim_func()
    sch = tir.Schedule(mod, debug_mask="all")
    (i,) = sch.get_loops(sch.get_block("B"))
    sch.split(
        i,
        factors=[
            te.const(1, "int32"),
            te.const(1, "int32"),
            te.const(3, "int32"),
            te.const(1, "int32"),
            te.const(4, "int32"),
        ],
    )


def test_split_int64_factors():
    sch = tir.Schedule(elementwise_symbolic, debug_mask="all")
    block_b = sch.get_block("B")
    _, _, k = sch.get_loops(block_b)
    sch.split(k, factors=[IntImm(dtype="int64", value=10), None])
    tvm.ir.assert_structural_equal(elementwise_symbolic_split, sch.mod["main"])


if __name__ == "__main__":
    tvm.testing.main()
