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
from tvm import tir
from tvm.ir import IRModule
from tvm.script import ty


# pylint: disable=no-member,invalid-name,unused-variable


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


# pylint: enable=no-member,invalid-name,unused-variable


def test_tir_schedule_creation():
    # Tests:
    # - Schedule.__init__ for PrimFunc and IRModule
    # - Schedule.mod
    # - Schedule.state
    sch_1 = tir.Schedule(matmul, debug_mode=True)
    sch_2 = tir.Schedule(IRModule({"main": matmul}), debug_mode=True)
    assert sch_1.mod["main"].same_as(sch_2.mod["main"])
    assert sch_1.state.mod["main"].same_as(sch_2.state.mod["main"])


def test_tir_schedule_get_block():
    # Tests:
    # - Schedule.get_block
    # - Schedule.get_sref
    # - Schedule.get
    sch = tir.Schedule(matmul, debug_mode=True)
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
    sch = tir.Schedule(matmul, debug_mode=True)
    block_rv = sch.get_block(name="update")
    i, j, k = sch.get_loops(block_rv)
    assert sch.get(i).loop_var.name == "i"
    assert sch.get(j).loop_var.name == "j"
    assert sch.get(k).loop_var.name == "k"


def test_tir_schedule_copy():
    # Tests:
    # - Schedule.copy
    sch_1 = tir.Schedule(matmul, debug_mode=True)
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


def test_tir_schedule_remove_rv():
    # Tests:
    # - Schedule.remove_rv
    sch = tir.Schedule(matmul, debug_mode=True)
    block_rv = sch.get_block(name="update")
    assert sch.get(block_rv).name_hint == "update"
    sch.remove_rv(block_rv)
    with pytest.raises(IndexError):
        sch.get(block_rv)


if __name__ == "__main__":
    test_tir_schedule_creation()
    test_tir_schedule_get_block()
    test_tir_schedule_get_loops()
    test_tir_schedule_copy()
    test_tir_schedule_remove_rv()
