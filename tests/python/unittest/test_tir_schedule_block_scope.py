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
from tvm.tir.schedule import DepKind
from tvm.tir.stmt_functor import post_order_visit

# pylint: disable=no-member,invalid-name,unused-variable


@T.prim_func
def elementwise(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (128, 128), "float32")
    C = T.match_buffer(c, (128, 128), "float32")
    B = T.alloc_buffer((128, 128), "float32")
    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] * 2.0
    for i, j in T.grid(128, 128):
        with T.block("C"):
            vi, vj = T.axis.remap("SS", [i, j])
            C[vi, vj] = B[vi, vj] + 1.0


@T.prim_func
def matmul(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, [128, 128])
    B = T.match_buffer(b, [128, 128])
    C = T.match_buffer(c, [128, 128])
    for i, j in T.grid(128, 128):
        with T.block("init"):
            vi, vj = T.axis.remap("SS", [i, j])
            C[vi, vj] = T.float32(0)
        for k in range(0, 128):
            with T.block("update"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]


@T.prim_func
def war_dependency(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (128, 128))
    B = T.match_buffer(b, (128, 128))
    C = T.match_buffer(c, (128, 128))

    for i, j in T.grid(128, 128):
        with T.block("C"):
            vi, vj = T.axis.remap("SS", [i, j])
            C[vi, vj] = B[vi, vj] + 1.0
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] * 2.0


# pylint: enable=no-member,invalid-name,unused-variable

# pylint: disable=invalid-name


def _get_block(s: tir.ScheduleState, name_hint: str) -> tir.StmtSRef:
    result = None

    def f_visit(node):
        nonlocal result
        if isinstance(node, tvm.tir.Block) and node.name_hint == name_hint:
            result = node

    func = s.mod["main"]
    post_order_visit(func.body, f_visit)
    assert result is not None and isinstance(result, tvm.tir.Block)
    return s.get_sref(result)


def test_elementwise_dependency():
    s = tir.ScheduleState(elementwise, debug_mask="all")
    root = _get_block(s, "root")
    block_b = _get_block(s, "B")
    block_c = _get_block(s, "C")
    # Check get_deps_by_src
    (dep,) = s.get_block_scope(root).get_deps_by_src(block_b)
    assert dep.src.same_as(block_b)
    assert dep.dst.same_as(block_c)
    assert dep.kind == DepKind.RAW
    # Check get_deps_by_dst
    (dep,) = s.get_block_scope(root).get_deps_by_dst(block_c)
    assert dep.src.same_as(block_b)
    assert dep.dst.same_as(block_c)
    assert dep.kind == DepKind.RAW


def test_matmul_dependency():
    s = tir.ScheduleState(matmul, debug_mask="all")
    root = _get_block(s, "root")
    init = _get_block(s, "init")
    update = _get_block(s, "update")
    # Check get_deps_by_src
    p0, p1 = s.get_block_scope(root).get_deps_by_src(init)
    assert p0.src.same_as(init)
    assert p0.dst.same_as(update)
    assert p1.src.same_as(init)
    assert p1.dst.same_as(update)
    assert (p0.kind == DepKind.RAW and p1.kind == DepKind.WAW) or (
        p0.kind == DepKind.WAW and p1.kind == DepKind.RAW
    )
    # Check get_deps_by_dst
    p0, p1 = s.get_block_scope(root).get_deps_by_dst(update)
    assert p0.src.same_as(init)
    assert p0.dst.same_as(update)
    assert p1.src.same_as(init)
    assert p1.dst.same_as(update)
    assert (p0.kind == DepKind.RAW and p1.kind == DepKind.WAW) or (
        p0.kind == DepKind.WAW and p1.kind == DepKind.RAW
    )


def test_war_dependency():
    s = tir.ScheduleState(war_dependency, debug_mask="all")
    root = _get_block(s, "root")
    block_c = _get_block(s, "C")
    block_b = _get_block(s, "B")
    # Check get_deps_by_src
    (dep,) = s.get_block_scope(root).get_deps_by_src(block_c)
    assert dep.src.same_as(block_c)
    assert dep.dst.same_as(block_b)
    assert dep.kind == DepKind.WAR
    # Check get_deps_by_dst
    (dep,) = s.get_block_scope(root).get_deps_by_dst(block_b)
    assert dep.src.same_as(block_c)
    assert dep.dst.same_as(block_b)
    assert dep.kind == DepKind.WAR


if __name__ == "__main__":
    tvm.testing.main()
