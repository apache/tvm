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
from tvm.ir import IRModule
from tvm.script import tir as T
from tvm.tir.schedule.testing import verify_trace_roundtrip

# pylint: disable=no-member,invalid-name,unused-variable


@T.prim_func
def matmul(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, [128, 128])
    B = T.match_buffer(b, [128, 128])
    C = T.match_buffer(c, [128, 128])
    for i, j in T.grid(128, 128):
        with T.block("init"):
            vi, vj = T.axis.remap("SS", [i, j])
            C[vi, vj] = 0.0
        for k in range(0, 128):
            with T.block("update"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]


@T.prim_func
def matmul_relu(a: T.handle, b: T.handle, d: T.handle) -> None:
    A = T.match_buffer(a, (1024, 1024))
    B = T.match_buffer(b, (1024, 1024))
    C = T.alloc_buffer((1024, 1024))
    D = T.match_buffer(d, (1024, 1024))
    for i, j, k in T.grid(1024, 1024, 1024):
        with T.block("matmul"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            with T.init():
                C[vi, vj] = 0.0
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]
    for i, j in T.grid(1024, 1024):
        with T.block("relu"):
            vi, vj = T.axis.remap("SS", [i, j])
            D[vi, vj] = T.max(C[vi, vj], 0.0)


# pylint: enable=no-member,invalid-name,unused-variable


def test_tir_schedule_creation():
    # Tests:
    # - Schedule.__init__ for PrimFunc and IRModule
    # - Schedule.mod
    # - Schedule.state
    sch_1 = tir.Schedule(matmul, debug_mask="all")
    sch_2 = tir.Schedule(IRModule({"main": matmul}), debug_mask="all")
    assert sch_1.mod["main"].same_as(sch_2.mod["main"])
    assert sch_1.state.mod["main"].same_as(sch_2.state.mod["main"])


def test_tir_schedule_get_block():
    # Tests:
    # - Schedule.get_block
    # - Schedule.get_sref
    # - Schedule.get
    sch = tir.Schedule(matmul, debug_mask="all")
    block_rv = sch.get_block(name="update")
    block_sref = sch.get_sref(block_rv)
    block = sch.get(block_rv)
    assert block.name_hint == "update"
    assert block_sref.stmt.same_as(block)
    assert sch.state.get_sref(block).same_as(block_sref)
    assert block.same_as(matmul.body.block.body.body.body[1].body.block)


def test_tir_schedule_get_loops():
    # Tests:
    # - Schedule.get_loops
    # - Schedule.get
    sch = tir.Schedule(matmul, debug_mask="all")
    block_rv = sch.get_block(name="update")
    i, j, k = sch.get_loops(block_rv)
    assert sch.get(i).loop_var.name == "i"
    assert sch.get(j).loop_var.name == "j"
    assert sch.get(k).loop_var.name == "k"


def test_tir_schedule_copy_1():
    # Tests:
    # - Schedule.copy
    sch_1 = tir.Schedule(matmul, debug_mask="all")
    block_rv = sch_1.get_block(name="update")
    i, j, k = sch_1.get_loops(block_rv)
    assert sch_1.get(i).loop_var.name == "i"
    assert sch_1.get(j).loop_var.name == "j"
    assert sch_1.get(k).loop_var.name == "k"

    sch_2 = sch_1.copy()
    assert sch_2.get(block_rv).name_hint == "update"
    assert sch_2.get(i).loop_var.name == "i"
    assert sch_2.get(j).loop_var.name == "j"
    assert sch_2.get(k).loop_var.name == "k"


def test_tir_schedule_copy_2():
    sch = tir.Schedule(mod=matmul, debug_mask="all")
    i, j, k = sch.get_loops(sch.get_block("update"))
    sch_copy = sch.copy()
    assert not sch.get_sref(i).same_as(sch_copy.get_sref(i))
    assert not sch.get_sref(j).same_as(sch_copy.get_sref(j))
    assert not sch.get_sref(k).same_as(sch_copy.get_sref(k))
    assert sch.get_sref(i).stmt.same_as(sch_copy.get_sref(i).stmt)
    assert sch.get_sref(j).stmt.same_as(sch_copy.get_sref(j).stmt)
    assert sch.get_sref(k).stmt.same_as(sch_copy.get_sref(k).stmt)
    i_0, i_1 = sch.split(i, factors=[None, 64])
    j_0, j_1 = sch_copy.split(j, factors=[None, 32])

    assert sch.get_sref(i_0).stmt.extent == 2
    assert sch.get_sref(i_1).stmt.extent == 64
    with pytest.raises(IndexError):
        sch_copy.get_sref(i_0)
    with pytest.raises(IndexError):
        sch_copy.get_sref(i_1)

    with pytest.raises(IndexError):
        sch.get_sref(j_0)
    with pytest.raises(IndexError):
        sch.get_sref(j_1)
    assert sch_copy.get_sref(j_0).stmt.extent == 4
    assert sch_copy.get_sref(j_1).stmt.extent == 32
    verify_trace_roundtrip(sch, mod=matmul)
    verify_trace_roundtrip(sch_copy, mod=matmul)


def test_tir_schedule_remove_rv():
    # Tests:
    # - Schedule.remove_rv
    sch = tir.Schedule(matmul, debug_mask="all")
    block_rv = sch.get_block(name="update")
    assert sch.get(block_rv).name_hint == "update"
    sch.remove_rv(block_rv)
    with pytest.raises(IndexError):
        sch.get(block_rv)


def test_get_child_blocks():
    s = tir.Schedule(matmul, debug_mask="all")
    init = s.get_block("init")
    update = s.get_block("update")
    # loop
    blocks = s.get_child_blocks(s.get_loops(init)[0])
    assert len(blocks) == 2
    assert s.get(init) == s.get(blocks[0])
    assert s.get(update) == s.get(blocks[1])
    # block
    root = s.get_block("root")
    blocks = s.get_child_blocks(root)
    assert len(blocks) == 2
    assert s.get(init) == s.get(blocks[0])
    assert s.get(update) == s.get(blocks[1])


def test_get_producers():
    sch = tir.Schedule(mod=matmul_relu, debug_mask="all")
    block = sch.get_block("relu")
    (producer,) = sch.get_producers(block)
    assert tvm.ir.structural_equal(
        sch.get_sref(producer).stmt,
        sch.get_sref(sch.get_block("matmul")).stmt,
    )
    verify_trace_roundtrip(sch, mod=matmul_relu)


def test_get_consumers():
    sch = tir.Schedule(mod=matmul_relu, debug_mask="all")
    block = sch.get_block("matmul")
    (consumer,) = sch.get_consumers(block)
    assert tvm.ir.structural_equal(
        sch.get_sref(consumer).stmt,
        sch.get_sref(sch.get_block("relu")).stmt,
    )
    verify_trace_roundtrip(sch, mod=matmul_relu)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__] + sys.argv[1:]))
