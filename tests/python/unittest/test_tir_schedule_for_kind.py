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
from tvm.tir.schedule.testing import verify_trace_roundtrip

# pylint: disable=no-member,invalid-name,unused-variable


@T.prim_func
def element_wise(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, (128, 128))
    B = T.match_buffer(b, (128, 128))

    with T.block([128, 128], "B") as [vi, vj]:
        B[vi, vj] = A[vi, vj] * 2.0


@T.prim_func
def element_wise_parallelized(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, (128, 128))
    B = T.match_buffer(b, (128, 128))
    for i0 in T.parallel(0, 128):
        for i1 in T.serial(0, 128):
            with T.block([128, 128], "B") as [vi, vj]:
                T.bind(vi, i0)
                T.bind(vj, i1)
                B[vi, vj] = A[vi, vj] * 2.0


@T.prim_func
def element_wise_i_bound(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, (128, 128))
    B = T.match_buffer(b, (128, 128))
    for i0 in T.thread_binding(0, 128, thread="threadIdx.x"):
        for i1 in T.serial(0, 128):
            with T.block([128, 128], "B") as [vi, vj]:
                T.bind(vi, i0)
                T.bind(vj, i1)
                B[vi, vj] = A[vi, vj] * 2.0


@T.prim_func
def element_wise_compute_at_split(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (128, 128))
    C = T.match_buffer(c, (128, 128))
    B = T.alloc_buffer((128, 128))
    for i in T.serial(0, 128):
        for j0 in T.serial(0, 128):
            with T.block([128, 128], "B") as [vi, vj]:
                T.bind(vi, i)
                T.bind(vj, j0)
                B[vi, vj] = A[vi, vj] * 2.0
        for j1o, j1i in T.grid(32, 4):
            with T.block([128, 128], "C") as [vi, vj]:
                T.bind(vi, i)
                T.bind(vj, j1o * 4 + j1i)
                C[vi, vj] = B[vi, vj] + 1.0


@T.prim_func
def element_wise_compute_at_split_vectorized(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (128, 128))
    C = T.match_buffer(c, (128, 128))
    B = T.alloc_buffer((128, 128))
    for i in T.serial(0, 128):
        for j0 in T.serial(0, 128):
            with T.block([128, 128], "B") as [vi, vj]:
                T.bind(vi, i)
                T.bind(vj, j0)
                B[vi, vj] = A[vi, vj] * 2.0
        for j1o in T.serial(0, 32):
            for j1i in T.vectorized(0, 4):
                with T.block([128, 128], "C") as [vi, vj]:
                    T.bind(vi, i)
                    T.bind(vj, j1o * 4 + j1i)
                    C[vi, vj] = B[vi, vj] + 1.0


@T.prim_func
def element_wise_split_predicate(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, [128, 128])
    B = T.match_buffer(b, [128, 128])
    for i, j_0, j_1 in T.grid(128, 13, 10):
        with T.block([128, 128], "B") as [vi, vj]:
            T.where(j_0 * 10 + j_1 < 128)
            T.bind(vi, i)
            T.bind(vj, j_0 * 10 + j_1)
            B[vi, vj] = A[vi, vj] * 2.0


@T.prim_func
def element_wise_split_predicate_parallelized(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, [128, 128])
    B = T.match_buffer(b, [128, 128])
    for i in T.serial(0, 128):
        for j_0 in T.parallel(0, 13):
            for j_1 in T.serial(0, 10):
                with T.block([128, 128], "B") as [vi, vj]:
                    T.where(j_0 * 10 + j_1 < 128)
                    T.bind(vi, i)
                    T.bind(vj, j_0 * 10 + j_1)
                    B[vi, vj] = A[vi, vj] * 2.0


@T.prim_func
def element_wise_split_predicate_vectorized(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, [128, 128])
    B = T.match_buffer(b, [128, 128])
    for i in T.vectorized(0, 128):
        for j_0, j_1 in T.grid(13, 10):
            with T.block([128, 128], "B") as [vi, vj]:
                T.where(j_0 * 10 + j_1 < 128)
                T.bind(vi, i)
                T.bind(vj, j_0 * 10 + j_1)
                B[vi, vj] = A[vi, vj] * 2.0


@T.prim_func
def element_wise_compute_at_split_j0_j1o_bound(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (128, 128))
    C = T.match_buffer(c, (128, 128))
    B = T.alloc_buffer((128, 128))
    for i in T.serial(0, 128):
        for j0 in T.thread_binding(0, 128, thread="threadIdx.x"):
            with T.block([128, 128], "B") as [vi, vj]:
                T.bind(vi, i)
                T.bind(vj, j0)
                B[vi, vj] = A[vi, vj] * 2.0
        for j1o in T.thread_binding(0, 32, thread="threadIdx.x"):
            for j1i in T.serial(0, 4):
                with T.block([128, 128], "C") as [vi, vj]:
                    T.bind(vi, i)
                    T.bind(vj, j1o * 4 + j1i)
                    C[vi, vj] = B[vi, vj] + 1.0


@T.prim_func
def matmul(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (128, 128))
    B = T.match_buffer(b, (128, 128))
    C = T.match_buffer(c, (128, 128))

    with T.block([128, 128, T.reduce_axis(0, 128)], "C") as [vi, vj, vk]:
        with T.init():
            C[vi, vj] = 0.0
        C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]


@T.prim_func
def rowsum(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, (128, 128))
    B = T.match_buffer(b, (128,))

    with T.block([128, T.reduce_axis(0, 128)], "B") as [vi, vk]:
        with T.init():
            B[vi] = 0.0
        B[vi] = B[vi] + A[vi, vk]


@T.prim_func
def rowsum_unrolled(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, (128, 128))
    B = T.match_buffer(b, (128,))
    for i0 in T.unroll(0, 128):
        for i1 in T.serial(0, 128):
            with T.block([128, T.reduce_axis(0, 128)], "B") as [vi, vk]:
                T.bind(vi, i0)
                T.bind(vk, i1)
                with T.init():
                    B[vi] = 0.0
                B[vi] = B[vi] + A[vi, vk]


@T.prim_func
def rowsum_not_quasi_affine(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, (128, 128))
    B = T.match_buffer(b, (128,))

    for i, k in T.grid(128, 16):
        with T.block([128, T.reduce_axis(0, 128)], "B") as [vi, vk]:
            T.bind(vi, i)
            T.bind(vk, T.floordiv(k * k, 2))
            with T.init():
                B[vi] = 0.0
            B[vi] = B[vi] + A[vi, vk]


@T.prim_func
def rowsum_not_compact_data_flow(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, (128, 128))
    B = T.match_buffer(b, (128,))

    with T.block([128, T.reduce_axis(0, 128)], "B") as [vi, vk]:
        with T.init():
            B[vk] = 0.0
        B[vk] = B[vk] + A[vi, vk]


@T.prim_func
def rowsum_cross_thread_reduction(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, (128, 128))
    B = T.match_buffer(b, (128,))
    for i0 in T.serial(0, 128):
        for i1 in T.thread_binding(0, 128, thread="threadIdx.x"):
            with T.block([128, T.reduce_axis(0, 128)], "B") as [vi, vk]:
                T.bind(vi, i0)
                T.bind(vk, i1)
                with T.init():
                    B[vi] = 0.0
                B[vi] = B[vi] + A[vi, vk]


@T.prim_func
def opaque_block(a: T.handle) -> None:
    A = T.match_buffer(a, (16,))
    for i in T.serial(0, 15):
        with T.block([], "opaque"):
            A[i + 1] = A[i + 1] + A[i]


# pylint: enable=no-member,invalid-name,unused-variable


def test_parallel():
    s = tir.Schedule(element_wise, debug_mask="all")
    i, _ = s.get_loops(s.get_block("B"))
    s.parallel(i)
    tvm.ir.assert_structural_equal(s.mod["main"], element_wise_parallelized)
    verify_trace_roundtrip(s, mod=element_wise)


def test_parallel_predicate():
    s = tir.Schedule(element_wise_split_predicate, debug_mask="all")
    _, j, _ = s.get_loops(s.get_block("B"))
    s.parallel(j)
    tvm.ir.assert_structural_equal(s.mod["main"], element_wise_split_predicate_parallelized)
    verify_trace_roundtrip(s, mod=element_wise_split_predicate)


def test_parallel_reduction_block_iter():
    s = tir.Schedule(matmul, debug_mask="all")
    _, _, k = s.get_loops(s.get_block("C"))
    with pytest.raises(tvm.tir.ScheduleError):
        s.parallel(k)


def test_parallel_not_quasi_affine():
    s = tir.Schedule(rowsum_not_quasi_affine, debug_mask="all")
    i, _ = s.get_loops(s.get_block("B"))
    with pytest.raises(tvm.tir.ScheduleError):
        s.parallel(i)


def test_parallel_not_compact_data_flow():
    s = tir.Schedule(rowsum_not_compact_data_flow, debug_mask="all")
    i, _ = s.get_loops(s.get_block("B"))
    with pytest.raises(tvm.tir.ScheduleError):
        s.parallel(i)


def test_vectorize():
    s = tir.Schedule(element_wise_compute_at_split, debug_mask="all")
    _, _, j1i = s.get_loops(s.get_block("C"))
    s.vectorize(j1i)
    tvm.ir.assert_structural_equal(s.mod["main"], element_wise_compute_at_split_vectorized)
    verify_trace_roundtrip(s, mod=element_wise_compute_at_split)


def test_vectorize_predicate():
    s = tir.Schedule(element_wise_split_predicate, debug_mask="all")
    i, _, _ = s.get_loops(s.get_block("B"))
    s.vectorize(i)
    tvm.ir.assert_structural_equal(s.mod["main"], element_wise_split_predicate_vectorized)
    verify_trace_roundtrip(s, mod=element_wise_split_predicate)


def test_vectorize_opaque_block():
    s = tir.Schedule(opaque_block, debug_mask="all")
    (i,) = s.get_loops(s.get_block("opaque"))
    with pytest.raises(tvm.tir.ScheduleError):
        s.vectorize(i)


def test_unroll():
    s = tir.Schedule(rowsum, debug_mask="all")
    i, _ = s.get_loops(s.get_block("B"))
    s.unroll(i)
    tvm.ir.assert_structural_equal(s.mod["main"], rowsum_unrolled)
    verify_trace_roundtrip(s, mod=rowsum)


def test_unroll_after_bind():
    s = tir.Schedule(rowsum, debug_mask="all")
    i, _ = s.get_loops(s.get_block("B"))
    s.bind(i, "blockIdx.x")
    s.unroll(i)
    tvm.ir.assert_structural_equal(s.mod["main"], rowsum_unrolled)
    verify_trace_roundtrip(s, mod=rowsum)


def test_bind1():
    s = tir.Schedule(element_wise, debug_mask="all")
    i, _ = s.get_loops(s.get_block("B"))
    s.bind(i, "threadIdx.x")
    tvm.ir.assert_structural_equal(s.mod["main"], element_wise_i_bound)
    verify_trace_roundtrip(s, mod=element_wise)


def test_bind2():
    s = tir.Schedule(element_wise_compute_at_split, debug_mask="all")
    _, j0 = s.get_loops(s.get_block("B"))
    _, j1o, _ = s.get_loops(s.get_block("C"))
    s.bind(j0, "threadIdx.x")
    s.bind(j1o, "threadIdx.x")
    tvm.ir.assert_structural_equal(s.mod["main"], element_wise_compute_at_split_j0_j1o_bound)
    verify_trace_roundtrip(s, mod=element_wise_compute_at_split)


def test_bind_cross_thread_reduction():
    s = tir.Schedule(rowsum, debug_mask="all")
    _, k = s.get_loops(s.get_block("B"))
    s.bind(k, "threadIdx.x")
    tvm.ir.assert_structural_equal(s.mod["main"], rowsum_cross_thread_reduction)
    verify_trace_roundtrip(s, mod=rowsum)


def test_bind_not_cross_thread_reduction():
    s = tir.Schedule(rowsum, debug_mask="all")
    _, k = s.get_loops(s.get_block("B"))
    with pytest.raises(tvm.tir.ScheduleError):
        s.bind(k, "blockIdx.x")


def test_bind_after_bind():
    s = tir.Schedule(element_wise, debug_mask="all")
    i, _ = s.get_loops(s.get_block("B"))
    s.bind(i, "blockIdx.x")
    s.bind(i, "threadIdx.x")
    tvm.ir.assert_structural_equal(s.mod["main"], element_wise_i_bound)
    verify_trace_roundtrip(s, mod=element_wise)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__] + sys.argv[1:]))
