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


@tvm.script.tir
def elementwise(a: ty.handle, b: ty.handle) -> None:
    A = tir.match_buffer(a, (128, 128, 128, 128))
    B = tir.match_buffer(b, (128, 128, 128, 128))
    with tir.block([128, 128, 128, 128], "B") as [vi, vj, vk, vl]:
        B[vi, vj, vk, vl] = A[vi, vj, vk, vl] * 2.0


@tvm.script.tir
def elementwise_not_affine(a: ty.handle, b: ty.handle) -> None:
    A = tir.match_buffer(a, (128, 128, 128, 128))
    B = tir.match_buffer(b, (128, 128, 128, 128))
    for i, j, k, l in tir.grid(128, 128, 128, 8):
        with tir.block([128, 128, 128, 128], "B") as [vi, vj, vk, vl]:
            tir.bind(vi, i)
            tir.bind(vj, j)
            tir.bind(vk, k)
            tir.bind(vl, l * 16)
            B[vi, vj, vk, vl] = A[vi, vj, vk, vl] * 2.0


@tvm.script.tir
def elementwise_dependent_loop(a: ty.handle, b: ty.handle) -> None:
    A = tir.match_buffer(a, (128, 128, 128, 128))
    B = tir.match_buffer(b, (128, 128, 128, 128))
    for i in tir.serial(0, 128):
        for j, k, l in tir.grid(128, i, 128):
            with tir.block([128, 128, i, 128], "B") as [vi, vj, vk, vl]:
                B[vi, vj, vk, vl] = A[vi, vj, vk, vl] * 2.0


@tvm.script.tir
def elementwise_predicate(a: ty.handle, b: ty.handle) -> None:
    A = tir.match_buffer(a, (128, 128, 128, 128))
    B = tir.match_buffer(b, (128, 128, 128, 128))
    for i, j, k, l in tir.grid(128, 128, 128, 128):
        with tir.block([128, 128, 128, 128], "B") as [vi, vj, vk, vl]:
            tir.where(i * 2097152 + j * 16384 + k * 128 + l < 100)
            B[vi, vj, vk, vl] = A[vi, vj, vk, vl] * 2.0


@tvm.script.tir
def elementwise_non_single_branch(a: ty.handle, b: ty.handle) -> None:
    A = tir.match_buffer(a, (128, 128, 128))
    C = tir.alloc_buffer((128, 128, 128))
    B = tir.match_buffer(b, (128, 128, 128))
    for i, j in tir.grid(128, 128):
        for k in tir.serial(0, 128):
            with tir.block([128, 128, 128], "C") as [vi, vj, vk]:
                tir.bind(vi, i)
                tir.bind(vj, j)
                tir.bind(vk, k)
                C[vi, vj, vk] = A[vi, vj, vk] * 2.0
        for k in tir.serial(0, 128):
            with tir.block([128, 128, 128], "B") as [vi, vj, vk]:
                tir.bind(vi, i)
                tir.bind(vj, j)
                tir.bind(vk, k)
                B[vi, vj, vk] = C[vi, vj, vk] * 2.0


@tvm.script.tir
def elementwise_with_loops_not_same_scope(a: ty.handle, b: ty.handle) -> None:
    A = tir.match_buffer(a, (128, 128, 128))
    B = tir.match_buffer(b, (128, 128, 128))
    for i, j in tir.grid(128, 128):
        with tir.block([128, 128], "A") as [vi, vj]:
            tir.bind(vi, i)
            tir.bind(vj, j)
            for k in tir.serial(0, 128):
                with tir.block([128], "B") as [vk]:
                    tir.bind(vk, k)
                    tir.reads([A[vi, vj, vk]])
                    tir.writes([B[vi, vj, vk]])
                    B[vi, vj, vk] = A[vi, vj, vk] * 2.0


@tvm.script.tir
def elementwise_with_wrong_block_var_type(a: ty.handle, b: ty.handle) -> None:
    A = tir.match_buffer(a, (128, 128, 128))
    B = tir.match_buffer(b, (128, 128, 128))
    for i, j, k in tir.grid(128, 128, 128):
        with tir.block([128, 128, tir.scan_axis(0, 128)], "B") as [vi, vj, vk]:
            tir.bind(vi, i)
            tir.bind(vj, j)
            tir.bind(vk, k)
            tir.reads([A[vi, vj, vk]])
            tir.writes([B[vi, vj, vk]])
            B[vi, vj, vk] = A[vi, vj, vk] * 2.0


@tvm.script.tir
def elementwise_reordered(a: ty.handle, b: ty.handle) -> None:
    A = tir.match_buffer(a, (128, 128, 128, 128))
    B = tir.match_buffer(b, (128, 128, 128, 128))
    for l, j, k, i in tir.grid(128, 128, 128, 128):
        with tir.block([128, 128, 128, 128], "B") as [vi, vj, vk, vl]:
            tir.bind(vi, i)
            tir.bind(vj, j)
            tir.bind(vk, k)
            tir.bind(vl, l)
            B[vi, vj, vk, vl] = A[vi, vj, vk, vl] * 2.0


@tvm.script.tir
def elementwise_reordered2(a: ty.handle, b: ty.handle) -> None:
    A = tir.match_buffer(a, (128, 128, 128, 128))
    B = tir.match_buffer(b, (128, 128, 128, 128))
    for k, j, i, l in tir.grid(128, 128, 128, 128):
        with tir.block([128, 128, 128, 128], "B") as [vi, vj, vk, vl]:
            tir.bind(vi, i)
            tir.bind(vj, j)
            tir.bind(vk, k)
            tir.bind(vl, l)
            B[vi, vj, vk, vl] = A[vi, vj, vk, vl] * 2.0


@tvm.script.tir
def elementwise_reordered_with_predicate(a: ty.handle, b: ty.handle) -> None:
    A = tir.match_buffer(a, (128, 128, 128, 128))
    B = tir.match_buffer(b, (128, 128, 128, 128))
    for l, j, k, i in tir.grid(128, 128, 128, 128):
        with tir.block([128, 128, 128, 128], "B") as [vi, vj, vk, vl]:
            tir.where(i * 2097152 + j * 16384 + k * 128 + l < 100)
            tir.bind(vi, i)
            tir.bind(vj, j)
            tir.bind(vk, k)
            tir.bind(vl, l)
            B[vi, vj, vk, vl] = A[vi, vj, vk, vl] * 2.0


@tvm.script.tir
def opaque_access(a: ty.handle, b: ty.handle) -> None:
    A = tir.match_buffer(a, [16, 16], "float32")
    B = tir.match_buffer(b, [16, 16], "float32")
    with tir.block([16, 16], "A") as [vi, vj]:
        tir.reads([])
        tir.writes([A[0:16, 0:16]])
        tir.store(A.data, vi * 16 + vj, 1)
    with tir.block([16, 16], "B") as [vi, vj]:
        tir.reads([])
        tir.writes([B[0:16, 0:16]])
        tir.evaluate(tir.tvm_fill_fragment(B.data, 16, 16, 16, 0, vi * 16 + vj, dtype="handle"))


@tvm.script.tir
def opaque_access_reorder(a: ty.handle, b: ty.handle) -> None:
    A = tir.match_buffer(a, [16, 16], "float32")
    B = tir.match_buffer(b, [16, 16], "float32")
    for j, i in tir.grid(16, 16):
        with tir.block([16, 16], "A") as [vi, vj]:
            tir.bind(vi, i)
            tir.bind(vj, j)
            tir.reads([])
            tir.writes([A[0:16, 0:16]])
            tir.store(A.data, vi * 16 + vj, 1)
    for j, i in tir.grid(16, 16):
        with tir.block([16, 16], "B") as [vi, vj]:
            tir.bind(vi, i)
            tir.bind(vj, j)
            tir.reads([])
            tir.writes([B[0:16, 0:16]])
            tir.evaluate(tir.tvm_fill_fragment(B.data, 16, 16, 16, 0, vi * 16 + vj, dtype="handle"))


# pylint: enable=no-member,invalid-name,unused-variable


def test_reorder():
    sch = tir.Schedule(elementwise, debug_mask="all")
    block_b = sch.get_block("B")
    i, j, k, l = sch.get_loops(block_b)
    sch.reorder(l, i)
    tvm.ir.assert_structural_equal(elementwise_reordered, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=elementwise)


def test_reorder2():
    sch = tir.Schedule(elementwise, debug_mask="all")
    block_b = sch.get_block("B")
    i, j, k, l = sch.get_loops(block_b)
    sch.reorder(k, i, l)
    tvm.ir.assert_structural_equal(elementwise_reordered2, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=elementwise)


def test_reorder_with_opaque_access():
    sch = tir.Schedule(opaque_access, debug_mask="all")
    block_a = sch.get_block("A")
    i, j = sch.get_loops(block_a)
    sch.reorder(j, i)
    block_b = sch.get_block("B")
    i, j = sch.get_loops(block_b)
    sch.reorder(j, i)
    tvm.ir.assert_structural_equal(opaque_access_reorder, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=opaque_access)


def test_reorder_with_predicate():
    sch = tir.Schedule(elementwise_predicate, debug_mask="all")
    block_b = sch.get_block("B")
    i, j, k, l = sch.get_loops(block_b)
    sch.reorder(l, i)
    tvm.ir.assert_structural_equal(elementwise_reordered_with_predicate, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=elementwise_predicate)


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
    sys.exit(pytest.main([__file__] + sys.argv[1:]))
