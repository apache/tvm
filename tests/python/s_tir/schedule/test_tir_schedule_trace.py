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
# mypy: ignore-errors
import sys

import pytest

import tvm
import tvm.testing
from tvm import s_tir, tir
from tvm.s_tir.schedule import Instruction, InstructionKind, LoopRV, SBlockRV, Trace
from tvm.s_tir.schedule.testing import assert_structural_equal_ignore_global_symbol
from tvm.script import tir as T


@T.prim_func
def elementwise(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (128, 128))
    B = T.alloc_buffer((128, 128))
    C = T.match_buffer(c, (128, 128))
    for i, j in T.grid(128, 128):
        with T.sblock("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] * 2.0
    for i, j in T.grid(128, 128):
        with T.sblock("C"):
            vi, vj = T.axis.remap("SS", [i, j])
            C[vi, vj] = B[vi, vj] + 1.0


@T.prim_func
def elementwise_inlined(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (128, 128))
    C = T.match_buffer(c, (128, 128))
    for i, j in T.grid(128, 128):
        with T.sblock("C"):
            vi, vj = T.axis.remap("SS", [i, j])
            C[vi, vj] = A[vi, vj] * 2.0 + 1.0


def _make_get_sblock(name, output):
    return Instruction(
        kind=InstructionKind.get("GetSBlock"),
        inputs=[],
        attrs=[name, "main"],
        outputs=[output],
    )


def _make_get_loops(input, outputs):
    return Instruction(
        kind=InstructionKind.get("GetLoops"),
        inputs=[input],
        attrs=[],
        outputs=outputs,
    )


def _make_compute_inline(input):
    return Instruction(
        kind=InstructionKind.get("ComputeInline"),
        inputs=[input],
        attrs=[],
        outputs=[],
    )


def _make_split(inputs, outputs):
    return Instruction(
        kind=InstructionKind.get("Split"),
        inputs=inputs,
        attrs=[T.bool(True), T.bool(False)],
        outputs=outputs,
    )


def _make_enter_postproc():
    return Instruction(
        kind=InstructionKind.get("EnterPostproc"),
        inputs=[],
        attrs=[],
        outputs=[],
    )


def _make_annotate(block: SBlockRV, annotation: str):
    return Instruction(
        kind=InstructionKind.get("Annotate"),
        inputs=[block, annotation],
        attrs=["meta_schedule.auto_tensorize"],
        outputs=[],
    )


def _make_trace_1(b0, l1, l2):
    return Trace(
        insts=[
            _make_get_sblock(name="block", output=b0),
            _make_get_loops(input=b0, outputs=[l1, l2]),
        ],
        decisions={},
    )


def _make_trace_2(b0):
    return Trace(
        insts=[
            _make_get_sblock(name="B", output=b0),
            _make_compute_inline(input=b0),
        ],
        decisions={},
    )


def _make_trace_3(b0, b1, add_postproc):
    if add_postproc:
        insts = [
            _make_get_sblock(name="B", output=b0),
            _make_compute_inline(input=b0),
            _make_get_sblock(name="C", output=b1),
            _make_enter_postproc(),
            _make_compute_inline(input=b1),
        ]
    else:
        insts = [
            _make_get_sblock(name="B", output=b0),
            _make_compute_inline(input=b0),
            _make_get_sblock(name="C", output=b1),
        ]
    return Trace(insts=insts, decisions={})


def _make_trace_4(b0, l1, l2, l3):
    return Trace(
        insts=[
            _make_get_sblock(name="B", output=b0),
            _make_get_loops(input=b0, outputs=[l1]),
            _make_split([l1, None, T.int32(32)], [l2, l3]),
        ],
        decisions={},
    )


def test_trace_construct_1():
    trace = _make_trace_1(SBlockRV(), LoopRV(), LoopRV())
    assert str(trace) == "\n".join(
        (
            "# from tvm import s_tir",
            "def apply_trace(sch: s_tir.Schedule) -> None:",
            '  b0 = sch.get_sblock(name="block", func_name="main")',
            "  l1, l2 = sch.get_loops(block=b0)",
        )
    )
    assert len(trace.insts) == 2
    assert len(trace.decisions) == 0


def test_trace_construct_get_decision_1():
    trace = _make_trace_1(SBlockRV(), LoopRV(), LoopRV())
    assert trace.get_decision(trace.insts[0]) is None
    assert trace.get_decision(trace.insts[1]) is None


def test_trace_construct_append_1():
    trace = _make_trace_1(SBlockRV(), LoopRV(), LoopRV())
    trace.append(inst=_make_get_sblock("block2", SBlockRV()))
    assert str(trace) == "\n".join(
        (
            "# from tvm import s_tir",
            "def apply_trace(sch: s_tir.Schedule) -> None:",
            '  b0 = sch.get_sblock(name="block", func_name="main")',
            "  l1, l2 = sch.get_loops(block=b0)",
            '  b3 = sch.get_sblock(name="block2", func_name="main")',
        )
    )


def test_trace_construct_pop_1():
    trace = _make_trace_1(SBlockRV(), LoopRV(), LoopRV())
    last_inst = trace.insts[-1]
    assert trace.pop().same_as(last_inst)
    assert str(trace) == "\n".join(
        (
            "# from tvm import s_tir",
            "def apply_trace(sch: s_tir.Schedule) -> None:",
            '  b0 = sch.get_sblock(name="block", func_name="main")',
        )
    )


def test_trace_construct_pop_2():
    trace = Trace([], {})
    assert str(trace) == "\n".join(
        (
            "# from tvm import s_tir",
            "def apply_trace(sch: s_tir.Schedule) -> None:",
            "  pass",
        )
    )
    assert trace.pop() is None
    assert str(trace) == "\n".join(
        (
            "# from tvm import s_tir",
            "def apply_trace(sch: s_tir.Schedule) -> None:",
            "  pass",
        )
    )


def test_trace_apply_to_schedule():
    trace = _make_trace_2(SBlockRV())
    sch = tvm.s_tir.Schedule(elementwise, debug_mask="all")
    trace.apply_to_schedule(sch, remove_postproc=False, decision_provider=None)
    assert_structural_equal_ignore_global_symbol(elementwise_inlined, sch.mod["main"])


def test_trace_as_json_1():
    trace = _make_trace_1(SBlockRV(), LoopRV(), LoopRV())
    obj = trace.as_json()
    assert obj == [
        [
            ["GetSBlock", [], ["block", "main"], ["b0"]],
            ["GetLoops", ["b0"], [], ["l1", "l2"]],
        ],
        [],
    ]


def test_trace_simplified_1():
    trace = _make_trace_3(SBlockRV(), SBlockRV(), add_postproc=True)
    assert str(trace) == "\n".join(
        (
            "# from tvm import s_tir",
            "def apply_trace(sch: s_tir.Schedule) -> None:",
            '  b0 = sch.get_sblock(name="B", func_name="main")',
            "  sch.compute_inline(block=b0)",
            '  b1 = sch.get_sblock(name="C", func_name="main")',
            "  sch.enter_postproc()",
            "  sch.compute_inline(block=b1)",
        )
    )
    trace = trace.simplified(remove_postproc=True)
    assert str(trace) == "\n".join(
        (
            "# from tvm import s_tir",
            "def apply_trace(sch: s_tir.Schedule) -> None:",
            '  b0 = sch.get_sblock(name="B", func_name="main")',
            "  sch.compute_inline(block=b0)",
        )
    )


def test_trace_simplified_2():
    trace = _make_trace_3(SBlockRV(), SBlockRV(), add_postproc=True)
    assert str(trace) == "\n".join(
        (
            "# from tvm import s_tir",
            "def apply_trace(sch: s_tir.Schedule) -> None:",
            '  b0 = sch.get_sblock(name="B", func_name="main")',
            "  sch.compute_inline(block=b0)",
            '  b1 = sch.get_sblock(name="C", func_name="main")',
            "  sch.enter_postproc()",
            "  sch.compute_inline(block=b1)",
        )
    )
    trace = trace.simplified(remove_postproc=False)
    assert str(trace) == "\n".join(
        (
            "# from tvm import s_tir",
            "def apply_trace(sch: s_tir.Schedule) -> None:",
            '  b0 = sch.get_sblock(name="B", func_name="main")',
            "  sch.compute_inline(block=b0)",
            '  b1 = sch.get_sblock(name="C", func_name="main")',
            "  sch.enter_postproc()",
            "  sch.compute_inline(block=b1)",
        )
    )


def test_trace_simplified_3():
    trace = _make_trace_4(SBlockRV(), LoopRV(), LoopRV(), LoopRV()).simplified(
        remove_postproc=False
    )
    assert str(trace) == "\n".join(
        (
            "# from tvm import s_tir",
            "def apply_trace(sch: s_tir.Schedule) -> None:",
            '  b0 = sch.get_sblock(name="B", func_name="main")',
            "  l1, = sch.get_loops(block=b0)",
            "  l2, l3 = sch.split(loop=l1, factors=[None, 32], preserve_unit_iters=True, disable_predication=False)",
        )
    )


def test_apply_json_to_schedule_1():
    trace = _make_trace_2(SBlockRV())
    json_obj = trace.as_json()
    sch = tvm.s_tir.Schedule(elementwise, debug_mask="all")
    Trace.apply_json_to_schedule(json_obj, sch)
    assert_structural_equal_ignore_global_symbol(elementwise_inlined, sch.mod["main"])


def test_apply_json_to_schedule_sample_categorical():
    var = tir.Var("v", "int32")
    trace1 = Trace(
        insts=[
            Instruction(
                kind=InstructionKind.get("SampleCategorical"),
                inputs=[],
                attrs=[[tvm.tir.IntImm("int32", 3)], [tvm.tir.FloatImm("float32", 1.0)]],
                outputs=[var],
            )
        ],
        decisions={},
    )
    json = trace1.as_json()
    assert str(json) == "[[['SampleCategorical', [], [[3], [T.float32(1.0)]], ['v0']]], []]"

    sch = tvm.s_tir.Schedule(elementwise, debug_mask="all")
    # As long as the application does not fail, it is fine.
    Trace.apply_json_to_schedule(json, sch)
    python_str = sch.trace.as_python()
    assert len(python_str) == 1
    assert python_str[0] == "v0 = sch.sample_categorical(candidates=[3], probs=[1], decision=0)"


def _test_apply_annotation_trace_from_json(annotation: str):
    """Test applying an annotation works without crashing.

    Designed to handle some previously failing edge cases like the
    empty string.
    """
    b0 = SBlockRV()
    trace = Trace(
        insts=[
            _make_get_sblock(name="B", output=b0),
            _make_annotate(block=b0, annotation=annotation),
        ],
        decisions={},
    )
    json_obj = trace.as_json()
    sch = tvm.s_tir.Schedule(elementwise, debug_mask="all")
    Trace.apply_json_to_schedule(json_obj, sch)

    @T.prim_func
    def elementwise_expected(a: T.handle, c: T.handle) -> None:
        A = T.match_buffer(a, (128, 128))
        B = T.alloc_buffer((128, 128))
        C = T.match_buffer(c, (128, 128))
        for i, j in T.grid(128, 128):
            with T.sblock("B"):
                T.sblock_attr({"meta_schedule.auto_tensorize": annotation})
                vi, vj = T.axis.remap("SS", [i, j])
                B[vi, vj] = A[vi, vj] * 2.0
        for i, j in T.grid(128, 128):
            with T.sblock("C"):
                vi, vj = T.axis.remap("SS", [i, j])
                C[vi, vj] = B[vi, vj] + 1.0

    assert_structural_equal_ignore_global_symbol(elementwise_expected, sch.mod["main"])


def test_apply_annotation_from_json():
    # Something reasonable
    _test_apply_annotation_trace_from_json("SSRSSR")

    # The empty string
    _test_apply_annotation_trace_from_json("")

    # A string of two quotation marks
    _test_apply_annotation_trace_from_json('""')

    # A string of one quotation mark
    _test_apply_annotation_trace_from_json('"')


if __name__ == "__main__":
    tvm.testing.main()
