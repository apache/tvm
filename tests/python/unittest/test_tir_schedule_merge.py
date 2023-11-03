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
from tvm import tir
from tvm.script import tir as T
from tvm.tir.schedule.testing import (
    verify_trace_roundtrip,
    assert_structural_equal_ignore_global_symbol,
)

# pylint: disable=no-member,invalid-name,unused-variable


@T.prim_func
def elementwise(a: T.handle, c: T.handle, d: T.handle) -> None:
    A = T.match_buffer(a, (128, 128))
    C = T.match_buffer(c, (128, 128))
    D = T.match_buffer(d, (64, 64))
    B = T.alloc_buffer((128, 128))
    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            T.reads(A[vi, vj])
            T.writes(B[vi, vj])
            B[vi, vj] = A[vi, vj] * T.float32(2)
    for i_0, j_0, i_1, j_1 in T.grid(8, 8, 16, 16):
        with T.block("C"):
            vi = T.axis.spatial(128, i_0 * 16 + i_1)
            vj = T.axis.spatial(128, j_0 * 16 + j_1)
            T.reads(B[vi, vj])
            T.writes(C[vi, vj])
            C[vi, vj] = B[vi, vj] + T.float32(1)
    for i_0, j_0, i_1, j_1 in T.grid(8, 8, 8, 8):
        with T.block("D"):
            vi = T.axis.spatial(64, i_0 * 8 + i_1)
            vj = T.axis.spatial(64, j_0 * 8 + j_1)
            T.reads(B[vi, vj])
            T.writes(D[vi, vj])
            D[vi, vj] = B[vi, vj] + T.float32(2)


@T.prim_func
def elementwise_merged(a: T.handle, c: T.handle, d: T.handle) -> None:
    A = T.match_buffer(a, (128, 128))
    C = T.match_buffer(c, (128, 128))
    D = T.match_buffer(d, (64, 64))
    B = T.alloc_buffer((128, 128))
    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            T.reads(A[vi, vj])
            T.writes(B[vi, vj])
            B[vi, vj] = A[vi, vj] * T.float32(2)
    for i_0_m in range(8):
        for j_0, i_1, j_1 in T.grid(8, 16, 16):
            with T.block("C"):
                vi = T.axis.spatial(128, i_0_m * 16 + i_1)
                vj = T.axis.spatial(128, j_0 * 16 + j_1)
                T.reads(B[vi, vj])
                T.writes(C[vi, vj])
                C[vi, vj] = B[vi, vj] + T.float32(1)
        for j_0, i_1, j_1 in T.grid(8, 8, 8):
            with T.block("D"):
                vi = T.axis.spatial(64, i_0_m * 8 + i_1)
                vj = T.axis.spatial(64, j_0 * 8 + j_1)
                T.reads(B[vi, vj])
                T.writes(D[vi, vj])
                D[vi, vj] = B[vi, vj] + T.float32(2)


@T.prim_func
def elementwise_merged2(a: T.handle, c: T.handle, d: T.handle) -> None:
    A = T.match_buffer(a, (128, 128))
    C = T.match_buffer(c, (128, 128))
    D = T.match_buffer(d, (64, 64))
    B = T.alloc_buffer((128, 128))
    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            T.reads(A[vi, vj])
            T.writes(B[vi, vj])
            B[vi, vj] = A[vi, vj] * T.float32(2)
    for i_0_m, j_0_m in T.grid(8, 8):
        for i_1, j_1 in T.grid(16, 16):
            with T.block("C"):
                vi = T.axis.spatial(128, i_0_m * 16 + i_1)
                vj = T.axis.spatial(128, j_0_m * 16 + j_1)
                T.reads(B[vi, vj])
                T.writes(C[vi, vj])
                C[vi, vj] = B[vi, vj] + T.float32(1)
        for i_1, j_1 in T.grid(8, 8):
            with T.block("D"):
                vi = T.axis.spatial(64, i_0_m * 8 + i_1)
                vj = T.axis.spatial(64, j_0_m * 8 + j_1)
                T.reads(B[vi, vj])
                T.writes(D[vi, vj])
                D[vi, vj] = B[vi, vj] + T.float32(2)


def test_merge():
    sch = tir.Schedule(elementwise, debug_mask="all")
    block_c = sch.get_block("C")
    block_d = sch.get_block("D")
    i = sch.get_loops(block_c)[0]
    j = sch.get_loops(block_d)[0]
    sch.merge(i, j)
    assert_structural_equal_ignore_global_symbol(elementwise_merged, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=elementwise)


def test_merge2():
    sch = tir.Schedule(elementwise, debug_mask="all")
    block_c = sch.get_block("C")
    block_d = sch.get_block("D")
    i = sch.get_loops(block_c)[1]
    j = sch.get_loops(block_d)[1]
    sch.merge(i, j)
    assert_structural_equal_ignore_global_symbol(elementwise_merged2, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=elementwise)


def test_merge_fail_not_only_child():
    @T.prim_func
    def elementwise_with_seq(a: T.handle, c: T.handle) -> None:
        A = T.match_buffer(a, (128, 128, 128))
        C = T.match_buffer(c, (128, 128, 128))
        B = T.alloc_buffer((128, 128, 128))
        D = T.alloc_buffer((128, 128, 128))
        for i, j in T.grid(128, 128):
            for k in T.serial(0, 128):
                with T.block("D"):
                    vi, vj, vk = T.axis.remap("SSS", [i, j, k])
                    D[vi, vj, vk] = A[vi, vj, vk] * 2.0
            for k in T.serial(0, 128):
                with T.block("B"):
                    vi, vj, vk = T.axis.remap("SSS", [i, j, k])
                    B[vi, vj, vk] = A[vi, vj, vk] * 2.0
        for i, j in T.grid(128, 128):
            for k in T.serial(0, 128):
                with T.block("C"):
                    vi, vj, vk = T.axis.remap("SSS", [i, j, k])
                    C[vi, vj, vk] = B[vi, vj, vk] * 2.0

    sch = tir.Schedule(elementwise_with_seq, debug_mask="all")
    block_b = sch.get_block("B")
    _, _, b = sch.get_loops(block_b)
    block_c = sch.get_block("C")
    _, _, c = sch.get_loops(block_c)
    with pytest.raises(tvm.tir.ScheduleError):
        sch.merge(b, c)


def test_merge_fail_not_start_with_zero():
    @T.prim_func
    def elementwise_loops_not_start_with_zero(a: T.handle, c: T.handle) -> None:
        A = T.match_buffer(a, (128, 128, 128))
        C = T.match_buffer(c, (128, 128, 128))
        B = T.alloc_buffer((128, 128, 128))
        for i, j in T.grid(128, 128):
            for k in T.serial(1, 128):
                with T.block("B"):
                    vi, vj, vk = T.axis.remap("SSS", [i, j, k])
                    B[vi, vj, vk] = A[vi, vj, vk] * 2.0
        for i, j in T.grid(128, 128):
            for k in T.serial(0, 128):
                with T.block("C"):
                    vi, vj, vk = T.axis.remap("SSS", [i, j, k])
                    C[vi, vj, vk] = A[vi, vj, vk] * 2.0

    sch = tir.Schedule(elementwise_loops_not_start_with_zero, debug_mask="all")
    block_b = sch.get_block("B")
    _, _, b = sch.get_loops(block_b)
    block_c = sch.get_block("C")
    _, _, c = sch.get_loops(block_c)
    with pytest.raises(tvm.tir.ScheduleError):
        sch.merge(b, c)


def test_merge_fail_not_same_extent():
    @T.prim_func
    def elementwise_loops_not_same_extent(a: T.handle, c: T.handle) -> None:
        A = T.match_buffer(a, (128, 128, 128))
        C = T.match_buffer(c, (128, 128, 128))
        B = T.alloc_buffer((64, 128, 128))
        for i, j in T.grid(64, 128):
            for k in T.serial(0, 128):
                with T.block("B"):
                    vi, vj, vk = T.axis.remap("SSS", [i, j, k])
                    B[vi, vj, vk] = A[vi, vj, vk] * 2.0
        for i, j in T.grid(128, 128):
            for k in T.serial(0, 128):
                with T.block("C"):
                    vi, vj, vk = T.axis.remap("SSS", [i, j, k])
                    C[vi, vj, vk] = A[vi, vj, vk] * 2.0

    sch = tir.Schedule(elementwise_loops_not_same_extent, debug_mask="all")
    block_b = sch.get_block("B")
    _, _, b = sch.get_loops(block_b)
    block_c = sch.get_block("C")
    _, _, c = sch.get_loops(block_c)
    with pytest.raises(tvm.tir.ScheduleError):
        sch.merge(b, c)


def test_merge_fail_not_same_level():
    @T.prim_func
    def elementwise_not_same_level(a: T.handle, c: T.handle) -> None:
        A = T.match_buffer(a, (128, 128, 128))
        C = T.match_buffer(c, (128, 128, 128))
        B = T.alloc_buffer((128, 128, 128))
        for i, j in T.grid(128, 128):
            for k in T.serial(0, 128):
                with T.block("B"):
                    vi, vj, vk = T.axis.remap("SSS", [i, j, k])
                    B[vi, vj, vk] = A[vi, vj, vk] * 2.0
        for i, j in T.grid(128, 128):
            for k in T.serial(0, 128):
                with T.block("C"):
                    vi, vj, vk = T.axis.remap("SSS", [i, j, k])
                    C[vi, vj, vk] = A[vi, vj, vk] * 2.0

    sch = tir.Schedule(elementwise_not_same_level, debug_mask="all")
    block_b = sch.get_block("B")
    _, b, _ = sch.get_loops(block_b)
    block_c = sch.get_block("C")
    _, _, c = sch.get_loops(block_c)
    with pytest.raises(tvm.tir.ScheduleError):
        sch.merge(b, c)


def test_merge_fail_with_different_scope():
    @T.prim_func
    def elementwise_with_different_scope(a: T.handle, c: T.handle) -> None:
        A = T.match_buffer(a, (128, 128, 128))
        C = T.match_buffer(c, (128, 128, 128))
        B = T.alloc_buffer((128, 128, 128))
        with T.block("A"):
            for i, j in T.grid(128, 128):
                for k in T.serial(0, 128):
                    with T.block("B"):
                        vi, vj, vk = T.axis.remap("SSS", [i, j, k])
                        B[vi, vj, vk] = A[vi, vj, vk] * 2.0
        for i, j in T.grid(128, 128):
            for k in T.serial(0, 128):
                with T.block("C"):
                    vi, vj, vk = T.axis.remap("SSS", [i, j, k])
                    C[vi, vj, vk] = A[vi, vj, vk] * 2.0

    sch = tir.Schedule(elementwise_with_different_scope, debug_mask="all")
    block_b = sch.get_block("B")
    _, _, b = sch.get_loops(block_b)
    block_c = sch.get_block("C")
    _, _, c = sch.get_loops(block_c)
    with pytest.raises(tvm.tir.ScheduleError):
        sch.merge(b, c)


if __name__ == "__main__":
    tvm.testing.main()
