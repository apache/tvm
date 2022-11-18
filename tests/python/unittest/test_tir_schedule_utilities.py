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


@T.prim_func
def matmul_relu_ann1(a: T.handle, b: T.handle, d: T.handle) -> None:
    A = T.match_buffer(a, (1024, 1024))
    B = T.match_buffer(b, (1024, 1024))
    C = T.alloc_buffer((1024, 1024))
    D = T.match_buffer(d, (1024, 1024))
    for i in T.serial(0, 1024, annotations={"test1": "aaa", "test4": {"arr": [0, 0], "key": 3}}):
        for j in T.serial(0, 1024, annotations={"test2": 612, "test3": ["aa", 1]}):
            for k in T.serial(0, 1024):
                with T.block("matmul"):
                    vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                    with T.init():
                        C[vi, vj] = 0.0
                    C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]
    for i, j in T.grid(1024, 1024):
        with T.block("relu"):
            vi, vj = T.axis.remap("SS", [i, j])
            D[vi, vj] = T.max(C[vi, vj], 0.0)


@T.prim_func
def matmul_relu_ann2(a: T.handle, b: T.handle, d: T.handle) -> None:
    A = T.match_buffer(a, (1024, 1024))
    B = T.match_buffer(b, (1024, 1024))
    C = T.alloc_buffer((1024, 1024))
    D = T.match_buffer(d, (1024, 1024))
    for i, j, k in T.grid(1024, 1024, 1024):
        with T.block("matmul"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            with T.init():
                C[vi, vj] = 0.0
            T.block_attr({"test1": "aaa", "test4": {"arr": [0, 0], "key": 3}})
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]
    for i, j in T.grid(1024, 1024):
        with T.block("relu"):
            vi, vj = T.axis.remap("SS", [i, j])
            T.block_attr({"test2": 0.22, "test3": ["aa", 1]})
            D[vi, vj] = T.max(C[vi, vj], 0.0)


@tvm.script.ir_module
class ModuleWithMultipleFuncs:
    @T.prim_func
    def vector_add(
        A: T.Buffer[128, "float32"],
        B: T.Buffer[128, "float32"],
    ) -> None:
        for i in range(128):
            with T.block("init"):
                vi = T.axis.remap("S", [i])
                B[vi] = A[vi]

    @T.prim_func
    def vector_add_2(
        A: T.Buffer[128, "float32"],
        B: T.Buffer[128, "float32"],
    ) -> None:
        for i in range(128):
            with T.block("init"):
                vi = T.axis.remap("S", [i])
                B[vi] = A[vi]


@T.prim_func
def tuple_reduction(data: T.Buffer[(4, 32), "float32"], T_add: T.Buffer[(4,), "float32"]) -> None:
    # function attr dict
    T.func_attr({"global_symbol": "main", "tir.noalias": True})
    # body
    with T.block("root"):
        T.reads()
        T.writes()
        data_red_temp_v0 = T.alloc_buffer([4], dtype="float32")
        data_red_temp_v1 = T.alloc_buffer([4], dtype="float32")
        for i0, i1 in T.grid(4, 32):
            with T.block("data_red_temp"):
                ax0, k1 = T.axis.remap("SR", [i0, i1])
                T.reads(data[ax0, k1])
                T.writes(data_red_temp_v0[ax0], data_red_temp_v1[ax0])
                with T.init():
                    data_red_temp_v0[ax0] = T.float32(0)
                    data_red_temp_v1[ax0] = T.float32(0)
                v_data_red_temp_v0: T.float32 = data_red_temp_v0[ax0] + data[ax0, k1]
                v_data_red_temp_v1: T.float32 = (
                    data_red_temp_v1[ax0] + data[ax0, k1] * data[ax0, k1]
                )
                data_red_temp_v0[ax0] = v_data_red_temp_v0
                data_red_temp_v1[ax0] = v_data_red_temp_v1
        for i0 in range(4):
            with T.block("T_add"):
                ax0 = T.axis.remap("S", [i0])
                T.reads(data_red_temp_v0[ax0], data_red_temp_v1[ax0])
                T.writes(T_add[ax0])
                T_add[ax0] = data_red_temp_v0[ax0] + data_red_temp_v1[ax0]


# pylint: enable=no-member,invalid-name,unused-variable

use_block_name = tvm.testing.parameter(by_dict={"block_obj": False, "block_name": True})


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


def test_tir_schedule_work_on():
    sch = tir.Schedule(ModuleWithMultipleFuncs, debug_mask="all")
    with pytest.raises(ValueError, match="does not know which function to be working on"):
        sch.get_block(name="init")
    sch.work_on(func_name="vector_add")
    sch.get_block(name="init")


def test_tir_schedule_get_loops(use_block_name):
    # Tests:
    # - Schedule.get_loops
    # - Schedule.get
    sch = tir.Schedule(matmul, debug_mask="all")
    block = "update" if use_block_name else sch.get_block(name="update")
    i, j, k = sch.get_loops(block)
    assert sch.get(i).loop_var.name == "i"
    assert sch.get(j).loop_var.name == "j"
    assert sch.get(k).loop_var.name == "k"


def test_tir_schedule_copy_1(use_block_name):
    # Tests:
    # - Schedule.copy
    sch_1 = tir.Schedule(matmul, debug_mask="all")
    block_rv = sch_1.get_block(name="update")
    i, j, k = sch_1.get_loops(block="update" if use_block_name else block_rv)
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


def test_get_producers(use_block_name):
    sch = tir.Schedule(mod=matmul_relu, debug_mask="all")
    block = "relu" if use_block_name else sch.get_block("relu")
    (producer,) = sch.get_producers(block)
    assert tvm.ir.structural_equal(
        sch.get_sref(producer).stmt,
        sch.get_sref(sch.get_block("matmul")).stmt,
    )
    verify_trace_roundtrip(sch, mod=matmul_relu)


def test_get_producers_multiple_buffer_depdencies(use_block_name):
    sch = tir.Schedule(mod=tuple_reduction, debug_mask="all")
    block = "T_add" if use_block_name else sch.get_block("T_add")
    (producer,) = sch.get_producers(block)
    assert tvm.ir.structural_equal(
        sch.get_sref(producer).stmt,
        sch.get_sref(sch.get_block("data_red_temp")).stmt,
    )


def test_get_consumers(use_block_name):
    sch = tir.Schedule(mod=matmul_relu, debug_mask="all")
    block = "matmul" if use_block_name else sch.get_block("matmul")
    (consumer,) = sch.get_consumers(block)
    assert tvm.ir.structural_equal(
        sch.get_sref(consumer).stmt,
        sch.get_sref(sch.get_block("relu")).stmt,
    )
    verify_trace_roundtrip(sch, mod=matmul_relu)


def test_get_consumers_multiple_buffer_depdencies(use_block_name):
    sch = tir.Schedule(mod=tuple_reduction, debug_mask="all")
    block = "data_red_temp" if use_block_name else sch.get_block("data_red_temp")
    (consumer,) = sch.get_consumers(block)
    assert tvm.ir.structural_equal(
        sch.get_sref(consumer).stmt,
        sch.get_sref(sch.get_block("T_add")).stmt,
    )


def test_annotate_unannotate_loop():
    sch = tir.Schedule(mod=matmul_relu, debug_mask="all")
    matmul = sch.get_block("matmul")
    relu = sch.get_block("relu")
    sch.annotate(sch.get_loops(matmul)[0], "test1", "aaa")
    sch.annotate(sch.get_loops(matmul)[1], "test2", 612)
    sch.annotate(sch.get_loops(matmul)[1], "test3", ["aa", 1])
    sch.annotate(sch.get_loops(matmul)[0], "test4", {"arr": [0, 0], "key": 3})
    tvm.ir.assert_structural_equal(sch.mod["main"], matmul_relu_ann1)
    verify_trace_roundtrip(sch=sch, mod=matmul_relu)
    sch.unannotate(sch.get_loops(matmul)[0], "test1")
    sch.unannotate(sch.get_loops(matmul)[1], "test2")
    sch.unannotate(sch.get_loops(matmul)[1], "test3")
    sch.unannotate(sch.get_loops(matmul)[0], "test4")
    verify_trace_roundtrip(sch=sch, mod=matmul_relu)


def test_annotate_unannotate_block():
    sch = tir.Schedule(mod=matmul_relu, debug_mask="all")
    matmul = sch.get_block("matmul")
    relu = sch.get_block("relu")
    sch.annotate(matmul, "test1", "aaa")
    sch.annotate(relu, "test2", 0.22)
    sch.annotate(relu, "test3", ["aa", 1])
    sch.annotate(matmul, "test4", {"arr": [0, 0], "key": 3})
    tvm.ir.assert_structural_equal(sch.mod["main"], matmul_relu_ann2)
    verify_trace_roundtrip(sch=sch, mod=matmul_relu)
    sch.unannotate(matmul, "test1")
    sch.unannotate(relu, "test2")
    sch.unannotate(relu, "test3")
    sch.unannotate(matmul, "test4")
    verify_trace_roundtrip(sch=sch, mod=matmul_relu)


if __name__ == "__main__":
    tvm.testing.main()
