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
    assert_structural_equal_ignore_global_symbol,
    verify_trace_roundtrip,
)

# pylint: disable=no-member,invalid-name,unused-variable


@T.prim_func
def elementwise(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, (128, 128, 128, 128))
    B = T.match_buffer(b, (128, 128, 128, 128))
    for i, j, k, l in T.grid(128, 128, 128, 128):
        with T.block("B"):
            vi, vj, vk, vl = T.axis.remap("SSSS", [i, j, k, l])
            B[vi, vj, vk, vl] = A[vi, vj, vk, vl] * 2.0


@T.prim_func
def elementwise_not_affine(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, (128, 128, 128, 128))
    B = T.match_buffer(b, (128, 128, 128, 128))
    for i, j, k, l in T.grid(128, 128, 128, 8):
        with T.block("B"):
            vi, vj, vk = T.axis.remap("SSS", [i, j, k])
            vl = T.axis.S(128, l * 16)
            B[vi, vj, vk, vl] = A[vi, vj, vk, vl] * 2.0


@T.prim_func
def elementwise_dependent_loop(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, (128, 128, 128, 128))
    B = T.match_buffer(b, (128, 128, 128, 128))
    for i in T.serial(0, 128):
        for j, k, l in T.grid(128, i, 128):
            with T.block("B"):
                vi, vj, vk, vl = T.axis.remap("SSSS", [i, j, k, l])
                B[vi, vj, vk, vl] = A[vi, vj, vk, vl] * 2.0


@T.prim_func
def elementwise_predicate(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, (128, 128, 128, 128))
    B = T.match_buffer(b, (128, 128, 128, 128))
    for i, j, k, l in T.grid(128, 128, 128, 128):
        with T.block("B"):
            T.where(i * 2097152 + j * 16384 + k * 128 + l < 100)
            vi, vj, vk, vl = T.axis.remap("SSSS", [i, j, k, l])
            B[vi, vj, vk, vl] = A[vi, vj, vk, vl] * 2.0


@T.prim_func
def elementwise_non_single_branch(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, (128, 128, 128))
    C = T.alloc_buffer((128, 128, 128))
    B = T.match_buffer(b, (128, 128, 128))
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
def elementwise_with_loops_not_same_scope(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, (128, 128, 128))
    B = T.match_buffer(b, (128, 128, 128))
    for i, j in T.grid(128, 128):
        with T.block("A"):
            vi, vj = T.axis.remap("SS", [i, j])
            for k in T.serial(0, 128):
                with T.block("B"):
                    vk = T.axis.S(128, k)
                    T.reads([A[vi, vj, vk]])
                    T.writes([B[vi, vj, vk]])
                    B[vi, vj, vk] = A[vi, vj, vk] * 2.0


@T.prim_func
def elementwise_with_wrong_block_var_type(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, (128, 128, 128))
    B = T.match_buffer(b, (128, 128, 128))
    for i, j, k in T.grid(128, 128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            vk = T.axis.scan(128, k)
            T.reads([A[vi, vj, vk]])
            T.writes([B[vi, vj, vk]])
            B[vi, vj, vk] = A[vi, vj, vk] * 2.0


@T.prim_func
def elementwise_reordered(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, (128, 128, 128, 128))
    B = T.match_buffer(b, (128, 128, 128, 128))
    for l, j, k, i in T.grid(128, 128, 128, 128):
        with T.block("B"):
            vi, vj, vk, vl = T.axis.remap("SSSS", [i, j, k, l])
            B[vi, vj, vk, vl] = A[vi, vj, vk, vl] * 2.0


@T.prim_func
def elementwise_reordered2(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, (128, 128, 128, 128))
    B = T.match_buffer(b, (128, 128, 128, 128))
    for k, j, i, l in T.grid(128, 128, 128, 128):
        with T.block("B"):
            vi, vj, vk, vl = T.axis.remap("SSSS", [i, j, k, l])
            B[vi, vj, vk, vl] = A[vi, vj, vk, vl] * 2.0


@T.prim_func
def elementwise_reordered_with_predicate(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, (128, 128, 128, 128))
    B = T.match_buffer(b, (128, 128, 128, 128))
    for l, j, k, i in T.grid(128, 128, 128, 128):
        with T.block("B"):
            T.where(i * 2097152 + j * 16384 + k * 128 + l < 100)
            vi, vj, vk, vl = T.axis.remap("SSSS", [i, j, k, l])
            B[vi, vj, vk, vl] = A[vi, vj, vk, vl] * 2.0


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
def opaque_access_reorder(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, [16, 16], "float32")
    B = T.match_buffer(b, [16, 16], "float32")
    for j, i in T.grid(16, 16):
        with T.block("A"):
            vi, vj = T.axis.remap("SS", [i, j])
            T.reads([])
            T.writes([A[0:16, 0:16]])
            A[vi, vj] = 1
    for j, i in T.grid(16, 16):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            T.reads([])
            T.writes([B[0:16, 0:16]])
            T.evaluate(T.tvm_fill_fragment(B.data, 16, 16, 16, 0, vi * 16 + vj, dtype="handle"))


# pylint: enable=no-member,invalid-name,unused-variable


def test_reorder():
    sch = tir.Schedule(elementwise, debug_mask="all")
    block_b = sch.get_block("B")
    i, j, k, l = sch.get_loops(block_b)
    sch.reorder(l, i)
    assert_structural_equal_ignore_global_symbol(elementwise_reordered, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=elementwise)


def test_reorder2():
    sch = tir.Schedule(elementwise, debug_mask="all")
    block_b = sch.get_block("B")
    i, j, k, l = sch.get_loops(block_b)
    sch.reorder(k, i, l)
    assert_structural_equal_ignore_global_symbol(elementwise_reordered2, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=elementwise)


def test_reorder_with_opaque_access():
    sch = tir.Schedule(opaque_access, debug_mask="all")
    block_a = sch.get_block("A")
    i, j = sch.get_loops(block_a)
    sch.reorder(j, i)
    block_b = sch.get_block("B")
    i, j = sch.get_loops(block_b)
    sch.reorder(j, i)
    assert_structural_equal_ignore_global_symbol(opaque_access_reorder, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=opaque_access)


def test_reorder_overlapped_access():
    @T.prim_func
    def overlapped_access(A: T.Buffer((14, 4), "float32"), B: T.Buffer((14, 4), "float32")):
        # example to write first axis multiple times
        for v0, v1, v2 in T.grid(6, 4, 4):
            with T.block("block"):
                i = T.axis.spatial(14, v0 * 2 + v1)
                j = T.axis.spatial(4, v2)
                B[i, j] = A[i, j] + 1.0

    @T.prim_func
    def overlapped_access_reorder(A: T.Buffer((14, 4), "float32"), B: T.Buffer((14, 4), "float32")):
        # example to write first axis multiple times
        for v0, v2, v1 in T.grid(6, 4, 4):
            with T.block("block"):
                i = T.axis.spatial(14, v0 * 2 + v1)
                j = T.axis.spatial(4, v2)
                B[i, j] = A[i, j] + 1.0

    sch = tir.Schedule(overlapped_access, debug_mask="all")
    v0, v1, v2 = sch.get_loops(sch.get_block("block"))
    sch.reorder(v0, v2, v1)
    assert_structural_equal_ignore_global_symbol(overlapped_access_reorder, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=overlapped_access)


def test_reorder_with_partial_affineness():
    @T.prim_func
    def non_affine_func(A: T.Buffer((14, 4), "float32"), B: T.Buffer((14, 4), "float32")):
        for v0, v1, v2 in T.grid(6, 4, 4):
            with T.block("block"):
                i = T.axis.spatial(14, v0 * v0 + v1)
                j = T.axis.spatial(4, v2)
                B[i, j] = A[i, j] + 1.0

    @T.prim_func
    def non_affine_func_reorder(A: T.Buffer((14, 4), "float32"), B: T.Buffer((14, 4), "float32")):
        for v0, v2, v1 in T.grid(6, 4, 4):
            with T.block("block"):
                i = T.axis.spatial(14, v0 * v0 + v1)
                j = T.axis.spatial(4, v2)
                B[i, j] = A[i, j] + 1.0

    sch = tir.Schedule(non_affine_func, debug_mask="all")
    v0, v1, v2 = sch.get_loops(sch.get_block("block"))
    with pytest.raises(tvm.tir.ScheduleError):
        sch.reorder(v0, v2, v1)

    sch.reorder(v2, v1)
    assert_structural_equal_ignore_global_symbol(non_affine_func_reorder, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=non_affine_func)


def test_reorder_with_cascade_tiled_ops():
    @T.prim_func
    def cascade_pool_ops(
        x: T.Buffer((1, 16, 112, 112), "float32"), y2: T.Buffer((1, 16, 108, 108), "float32")
    ) -> None:
        y1 = T.alloc_buffer([1, 16, 110, 110], dtype="float32")
        for n, c, h, w, kh, kw in T.grid(1, 16, 110, 110, 3, 3):
            with T.block("pool_0"):
                ax0, ax1, ax2, ax3, rv0, rv1 = T.axis.remap("SSSSRR", [n, c, h, w, kh, kw])
                with T.init():
                    y1[ax0, ax1, ax2, ax3] = 0.0
                y1[ax0, ax1, ax2, ax3] = y1[ax0, ax1, ax2, ax3] + x[ax0, ax1, ax2 + rv0, ax3 + rv1]
        for n, c, h, w, kh, kw in T.grid(1, 16, 108, 108, 3, 3):
            with T.block("pool_1"):
                ax0, ax1, ax2, ax3, rv0, rv1 = T.axis.remap("SSSSRR", [n, c, h, w, kh, kw])
                with T.init():
                    y2[ax0, ax1, ax2, ax3] = 0.0
                y2[ax0, ax1, ax2, ax3] = y2[ax0, ax1, ax2, ax3] + y1[ax0, ax1, ax2 + rv0, ax3 + rv1]

    @T.prim_func
    def cascade_pool_ops_tile_reordered(
        x: T.Buffer((1, 16, 112, 112), "float32"), y2: T.Buffer((1, 16, 108, 108), "float32")
    ) -> None:
        y1 = T.alloc_buffer([1, 16, 110, 110], dtype="float32")
        for n, c, h_o in T.grid(1, 16, 27):
            for w, h_i, kh, kw in T.grid(110, 6, 3, 3):
                with T.block("pool_0"):
                    ax0 = T.axis.spatial(1, 0)
                    ax1 = T.axis.spatial(16, c)
                    ax2 = T.axis.spatial(110, h_o * 4 + h_i)
                    ax3, rv0, rv1 = T.axis.remap("SRR", [w, kh, kw])
                    with T.init():
                        y1[ax0, ax1, ax2, ax3] = 0.0
                    y1[ax0, ax1, ax2, ax3] = (
                        y1[ax0, ax1, ax2, ax3] + x[ax0, ax1, ax2 + rv0, ax3 + rv1]
                    )
            for h_i, w, kh, kw in T.grid(4, 108, 3, 3):
                with T.block("pool_1"):
                    ax0 = T.axis.spatial(1, n)
                    ax1 = T.axis.spatial(16, c)
                    ax2 = T.axis.spatial(108, h_o * 4 + h_i)
                    ax3, rv0, rv1 = T.axis.remap("SRR", [w, kh, kw])
                    with T.init():
                        y2[ax0, ax1, ax2, ax3] = 0.0
                    y2[ax0, ax1, ax2, ax3] = (
                        y2[ax0, ax1, ax2, ax3] + y1[ax0, ax1, ax2 + rv0, ax3 + rv1]
                    )

    sch = tvm.tir.schedule.Schedule(cascade_pool_ops)
    pool_0 = sch.get_block("pool_0")
    pool_1 = sch.get_block("pool_1")
    _, _, h, w, _, _ = sch.get_loops(pool_1)
    ho, _ = sch.split(h, factors=[None, 4])
    sch.compute_at(pool_0, ho)
    _, _, _, h_i, w, _, _ = sch.get_loops(pool_0)
    sch.reorder(w, h_i)
    assert_structural_equal_ignore_global_symbol(
        cascade_pool_ops_tile_reordered, sch.mod["main"], True
    )
    verify_trace_roundtrip(sch=sch, mod=cascade_pool_ops)


def test_reorder_with_predicate():
    sch = tir.Schedule(elementwise_predicate, debug_mask="all")
    block_b = sch.get_block("B")
    i, j, k, l = sch.get_loops(block_b)
    with pytest.raises(tvm.tir.ScheduleError):
        sch.reorder(l, i)


def test_reorder_fail_with_multi_appearance_loops():
    sch = tir.Schedule(elementwise, debug_mask="all")
    block_b = sch.get_block("B")
    i, j, k, l = sch.get_loops(block_b)
    with pytest.raises(tvm.tir.ScheduleError):
        sch.reorder(k, i, i)


def test_reorder_fail_with_non_single_branch_loop():
    sch = tir.Schedule(elementwise_non_single_branch, debug_mask="all")
    block_b = sch.get_block("B")
    i, j, k = sch.get_loops(block_b)
    with pytest.raises(tvm.tir.ScheduleError):
        sch.reorder(k, i)
    sch = tir.Schedule(elementwise_non_single_branch, debug_mask="all")
    block_b = sch.get_block("B")
    block_c = sch.get_block("C")
    i, j, k1 = sch.get_loops(block_b)
    _, _, k2 = sch.get_loops(block_c)
    with pytest.raises(tvm.tir.ScheduleError):
        sch.reorder(k1, i, k2)


def test_reorder_fail_with_loops_not_under_same_scope():
    sch = tir.Schedule(elementwise_with_loops_not_same_scope, debug_mask="all")
    block_b = sch.get_block("B")
    block_a = sch.get_block("A")
    i, j = sch.get_loops(block_a)
    k = sch.get_loops(block_b)[0]
    with pytest.raises(tvm.tir.ScheduleError):
        sch.reorder(k, i)


def test_reorder_fail_with_wrong_block_var_type():
    sch = tir.Schedule(elementwise_with_wrong_block_var_type, debug_mask="all")
    block_b = sch.get_block("B")
    i, j, k = sch.get_loops(block_b)
    with pytest.raises(tvm.tir.ScheduleError):
        sch.reorder(k, i)


def test_reorder_fail_with_dependent_loops():
    sch = tir.Schedule(elementwise_dependent_loop, debug_mask="all")
    block_b = sch.get_block("B")
    i, j, k, l = sch.get_loops(block_b)
    with pytest.raises(tvm.tir.ScheduleError):
        sch.reorder(l, i)


def test_reorder_fail_not_affine_bindings():
    sch = tir.Schedule(elementwise_not_affine, debug_mask="all")
    block_b = sch.get_block("B")
    i, j, k, l = sch.get_loops(block_b)
    with pytest.raises(tvm.tir.ScheduleError):
        sch.reorder(l, i)


if __name__ == "__main__":
    tvm.testing.main()
