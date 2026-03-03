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
"""Regression test for reverse_compute_at with AllocBuffer in root scope.

Before the fix, reverse_compute_at would raise a spurious ScheduleError:
  "Block ... is the only leaf in the scope ..., which cannot be removed"

Root cause: LeafBlockRemovalPlan in transform.cc only peeled DeclBuffer
nodes from a block body but not AllocBuffer nodes. When the root block
body starts with an AllocBuffer (from T.alloc_buffer at function level),
the SeqStmt was not visible, causing the error.
"""

import tvm
import tvm.s_tir
import tvm.testing
from tvm.script import tir as T


# ------------------------------------------------------------------
# T.alloc_buffer variant (AllocBuffer statement node in root scope)
# ------------------------------------------------------------------
@T.prim_func(private=True)
def alloc_buffer_before(
    A: T.Buffer((128, 128), "float32"),
    B: T.Buffer((128, 128), "float32"),
    C: T.Buffer((128, 128), "float32"),
):
    T.func_attr({"tir.noalias": True})
    Y = T.alloc_buffer((128, 128))
    for i, j, k in T.grid(128, 128, 128):
        with T.sblock("Y"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            with T.init():
                Y[vi, vj] = T.float32(0)
            Y[vi, vj] = Y[vi, vj] + A[vi, vk] * B[vk, vj]
    for i, j in T.grid(128, 128):
        with T.sblock("C"):
            vi, vj = T.axis.remap("SS", [i, j])
            C[vi, vj] = T.max(Y[vi, vj], T.float32(0))


@T.prim_func(private=True)
def alloc_buffer_expected(
    A: T.Buffer((128, 128), "float32"),
    B: T.Buffer((128, 128), "float32"),
    C: T.Buffer((128, 128), "float32"),
):
    T.func_attr({"tir.noalias": True})
    Y = T.alloc_buffer((128, 128))
    for i, j_0 in T.grid(128, 16):
        for k, j_1 in T.grid(128, 8):
            with T.sblock("Y"):
                vi = T.axis.spatial(128, i)
                vj = T.axis.spatial(128, j_0 * 8 + j_1)
                vk = T.axis.reduce(128, k)
                with T.init():
                    Y[vi, vj] = T.float32(0)
                Y[vi, vj] = Y[vi, vj] + A[vi, vk] * B[vk, vj]
        for ax0 in range(8):
            with T.sblock("C"):
                vi = T.axis.spatial(128, i)
                vj = T.axis.spatial(128, j_0 * 8 + ax0)
                C[vi, vj] = T.max(Y[vi, vj], T.float32(0))


# ------------------------------------------------------------------
# T.sblock_alloc_buffer variant (SBlock metadata)
# ------------------------------------------------------------------
@T.prim_func(private=True)
def sblock_alloc_before(
    A: T.Buffer((128, 128), "float32"),
    B: T.Buffer((128, 128), "float32"),
    C: T.Buffer((128, 128), "float32"),
):
    T.func_attr({"tir.noalias": True})
    with T.sblock("root"):
        T.reads()
        T.writes()
        Y = T.sblock_alloc_buffer((128, 128))
        for i, j, k in T.grid(128, 128, 128):
            with T.sblock("Y"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    Y[vi, vj] = T.float32(0)
                Y[vi, vj] = Y[vi, vj] + A[vi, vk] * B[vk, vj]
        for i, j in T.grid(128, 128):
            with T.sblock("C"):
                vi, vj = T.axis.remap("SS", [i, j])
                C[vi, vj] = T.max(Y[vi, vj], T.float32(0))


@T.prim_func(private=True)
def sblock_alloc_expected(
    A: T.Buffer((128, 128), "float32"),
    B: T.Buffer((128, 128), "float32"),
    C: T.Buffer((128, 128), "float32"),
):
    T.func_attr({"tir.noalias": True})
    with T.sblock("root"):
        T.reads()
        T.writes()
        Y = T.sblock_alloc_buffer((128, 128))
        for i, j_0 in T.grid(128, 16):
            for k, j_1 in T.grid(128, 8):
                with T.sblock("Y"):
                    vi = T.axis.spatial(128, i)
                    vj = T.axis.spatial(128, j_0 * 8 + j_1)
                    vk = T.axis.reduce(128, k)
                    with T.init():
                        Y[vi, vj] = T.float32(0)
                    Y[vi, vj] = Y[vi, vj] + A[vi, vk] * B[vk, vj]
            for ax0 in range(8):
                with T.sblock("C"):
                    vi = T.axis.spatial(128, i)
                    vj = T.axis.spatial(128, j_0 * 8 + ax0)
                    C[vi, vj] = T.max(Y[vi, vj], T.float32(0))


def _apply_reverse_compute_at(func):
    """Apply split + reorder + reverse_compute_at schedule."""
    sch = tvm.s_tir.Schedule(func, debug_mask="all")
    Y = sch.get_sblock("Y")
    C_block = sch.get_sblock("C")
    i, j, k = sch.get_loops(Y)
    j0, j1 = sch.split(j, factors=[16, 8])
    sch.reorder(i, j0, k, j1)
    sch.reverse_compute_at(C_block, j0)
    return sch.mod["main"]


def test_reverse_compute_at_alloc_buffer():
    """reverse_compute_at with T.alloc_buffer (AllocBuffer statement node)."""
    result = _apply_reverse_compute_at(alloc_buffer_before)
    tvm.ir.assert_structural_equal(result, alloc_buffer_expected)


def test_reverse_compute_at_sblock_alloc_buffer():
    """reverse_compute_at with T.sblock_alloc_buffer (SBlock metadata)."""
    result = _apply_reverse_compute_at(sblock_alloc_before)
    tvm.ir.assert_structural_equal(result, sblock_alloc_expected)


if __name__ == "__main__":
    tvm.testing.main()
