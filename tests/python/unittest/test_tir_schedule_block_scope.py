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
from tvm.tir.schedule import DepKind
from tvm.tir.stmt_functor import post_order_visit

# pylint: disable=no-member,invalid-name,unused-variable


@tvm.script.tir
def elementwise(a: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, (128, 128), "float32")
    C = tir.match_buffer(c, (128, 128), "float32")
    B = tir.alloc_buffer((128, 128), "float32")
    with tir.block([128, 128], "B") as [vi, vj]:
        B[vi, vj] = A[vi, vj] * 2.0
    with tir.block([128, 128], "C") as [vi, vj]:
        C[vi, vj] = B[vi, vj] + 1.0


@tvm.script.tir
def matmul(a: ty.handle, b: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, [128, 128])
    B = tir.match_buffer(b, [128, 128])
    C = tir.match_buffer(c, [128, 128])
    for i, j in tir.grid(128, 128):
        with tir.block([128, 128], "init") as [vi, vj]:
            C[vi, vj] = tir.float32(0)
        for k in range(0, 128):
            with tir.block([128, 128, tir.reduce_axis(0, 128)], "update") as [vi, vj, vk]:
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]


@tvm.script.tir
def war_dependency(a: ty.handle, b: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, (128, 128))
    B = tir.match_buffer(b, (128, 128))
    C = tir.match_buffer(c, (128, 128))

    for i, j in tir.grid(128, 128):
        with tir.block([128, 128], "C") as [vi, vj]:
            C[vi, vj] = B[vi, vj] + 1.0
        with tir.block([128, 128], "B") as [vi, vj]:
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
    s = tir.ScheduleState(elementwise, debug_mode=True)
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
    s = tir.ScheduleState(matmul, debug_mode=True)
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
    s = tir.ScheduleState(war_dependency, debug_mode=True)
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
    sys.exit(pytest.main([__file__] + sys.argv[1:]))
