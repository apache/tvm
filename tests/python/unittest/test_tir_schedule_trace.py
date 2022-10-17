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
# mypy: ignore-errors
import sys

import pytest
import tvm
import tvm.testing
from tvm import tir
from tvm.script import tir as T
from tvm.tir.schedule import BlockRV, Instruction, InstructionKind, LoopRV, Trace

# pylint: disable=no-member,invalid-name,unused-variable


@T.prim_func
def elementwise(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (128, 128))
    B = T.alloc_buffer((128, 128))
    C = T.match_buffer(c, (128, 128))
    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] * 2.0
    for i, j in T.grid(128, 128):
        with T.block("C"):
            vi, vj = T.axis.remap("SS", [i, j])
            C[vi, vj] = B[vi, vj] + 1.0


@T.prim_func
def elementwise_inlined(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (128, 128))
    C = T.match_buffer(c, (128, 128))
    for i, j in T.grid(128, 128):
        with T.block("C"):
            vi, vj = T.axis.remap("SS", [i, j])
            C[vi, vj] = A[vi, vj] * 2.0 + 1.0


# pylint: enable=no-member,invalid-name,unused-variable


def _make_get_block(name, output):
    return Instruction(
        kind=InstructionKind.get("GetBlock"),
        inputs=[],
        attrs=[name, "main"],
        outputs=[output],
    )


def _make_get_loops(input, outputs):  # pylint: disable=redefined-builtin
    return Instruction(
        kind=InstructionKind.get("GetLoops"),
        inputs=[input],
        attrs=[],
        outputs=outputs,
    )


def _make_compute_inline(input):  # pylint: disable=redefined-builtin
    return Instruction(
        kind=InstructionKind.get("ComputeInline"),
        inputs=[input],
        attrs=[],
        outputs=[],
    )


def _make_split(inputs, outputs):  # pylint: disable=redefined-builtin
    return Instruction(
        kind=InstructionKind.get("Split"),
        inputs=inputs,
        attrs=[True],
        outputs=outputs,
    )


def _make_enter_postproc():
    return Instruction(
        kind=InstructionKind.get("EnterPostproc"),
        inputs=[],
        attrs=[],
        outputs=[],
    )


def _make_annotate(block: BlockRV, annotation: str):
    return Instruction(
        kind=InstructionKind.get("Annotate"),
        inputs=[block, annotation],
        attrs=["meta_schedule.auto_tensorize"],
        outputs=[],
    )


def _make_trace_1(b0, l1, l2):  # pylint: disable=invalid-name
    return Trace(
        insts=[
            _make_get_block(name="block", output=b0),
            _make_get_loops(input=b0, outputs=[l1, l2]),
        ],
        decisions={},
    )


def _make_trace_2(b0):  # pylint: disable=invalid-name
    return Trace(
        insts=[
            _make_get_block(name="B", output=b0),
            _make_compute_inline(input=b0),
        ],
        decisions={},
    )


def _make_trace_3(b0, b1, add_postproc):  # pylint: disable=invalid-name
    if add_postproc:
        insts = [
            _make_get_block(name="B", output=b0),
            _make_compute_inline(input=b0),
            _make_get_block(name="C", output=b1),
            _make_enter_postproc(),
            _make_compute_inline(input=b1),
        ]
    else:
        insts = [
            _make_get_block(name="B", output=b0),
            _make_compute_inline(input=b0),
            _make_get_block(name="C", output=b1),
        ]
    return Trace(insts=insts, decisions={})


def _make_trace_4(b0, l1, l2, l3):  # pylint: disable=invalid-name
    return Trace(
        insts=[
            _make_get_block(name="B", output=b0),
            _make_get_loops(input=b0, outputs=[l1]),
            _make_split([l1, None, 32], [l2, l3]),
        ],
        decisions={},
    )


def test_trace_construct_1():
    trace = _make_trace_1(BlockRV(), LoopRV(), LoopRV())
    assert str(trace) == "\n".join(
        (
            "# from tvm import tir",
            "def apply_trace(sch: tir.Schedule) -> None:",
            '  b0 = sch.get_block(name="block", func_name="main")',
            "  l1, l2 = sch.get_loops(block=b0)",
        )
    )
    assert len(trace.insts) == 2
    assert len(trace.decisions) == 0


def test_trace_construct_get_decision_1():
    trace = _make_trace_1(BlockRV(), LoopRV(), LoopRV())
    assert trace.get_decision(trace.insts[0]) is None
    assert trace.get_decision(trace.insts[1]) is None


def test_trace_construct_append_1():
    trace = _make_trace_1(BlockRV(), LoopRV(), LoopRV())
    trace.append(inst=_make_get_block("block2", BlockRV()))
    assert str(trace) == "\n".join(
        (
            "# from tvm import tir",
            "def apply_trace(sch: tir.Schedule) -> None:",
            '  b0 = sch.get_block(name="block", func_name="main")',
            "  l1, l2 = sch.get_loops(block=b0)",
            '  b3 = sch.get_block(name="block2", func_name="main")',
        )
    )


def test_trace_construct_pop_1():
    trace = _make_trace_1(BlockRV(), LoopRV(), LoopRV())
    last_inst = trace.insts[-1]
    assert trace.pop().same_as(last_inst)
    assert str(trace) == "\n".join(
        (
            "# from tvm import tir",
            "def apply_trace(sch: tir.Schedule) -> None:",
            '  b0 = sch.get_block(name="block", func_name="main")',
        )
    )


def test_trace_construct_pop_2():
    trace = Trace([], {})
    assert str(trace) == "\n".join(
        (
            "# from tvm import tir",
            "def apply_trace(sch: tir.Schedule) -> None:",
            "  pass",
        )
    )
    assert trace.pop() is None
    assert str(trace) == "\n".join(
        (
            "# from tvm import tir",
            "def apply_trace(sch: tir.Schedule) -> None:",
            "  pass",
        )
    )


def test_trace_apply_to_schedule():
    trace = _make_trace_2(BlockRV())
    sch = tir.Schedule(elementwise, debug_mask="all")
    trace.apply_to_schedule(sch, remove_postproc=False, decision_provider=None)
    tvm.ir.assert_structural_equal(elementwise_inlined, sch.mod["main"])


def test_trace_as_json_1():
    trace = _make_trace_1(BlockRV(), LoopRV(), LoopRV())
    obj = trace.as_json()
    assert obj == [
        [
            ["GetBlock", [], ["block", "main"], ["b0"]],
            ["GetLoops", ["b0"], [], ["l1", "l2"]],
        ],
        [],
    ]


def test_trace_simplified_1():
    trace = _make_trace_3(BlockRV(), BlockRV(), add_postproc=True)
    assert str(trace) == "\n".join(
        (
            "# from tvm import tir",
            "def apply_trace(sch: tir.Schedule) -> None:",
            '  b0 = sch.get_block(name="B", func_name="main")',
            "  sch.compute_inline(block=b0)",
            '  b1 = sch.get_block(name="C", func_name="main")',
            "  sch.enter_postproc()",
            "  sch.compute_inline(block=b1)",
        )
    )
    trace = trace.simplified(remove_postproc=True)
    assert str(trace) == "\n".join(
        (
            "# from tvm import tir",
            "def apply_trace(sch: tir.Schedule) -> None:",
            '  b0 = sch.get_block(name="B", func_name="main")',
            "  sch.compute_inline(block=b0)",
        )
    )


def test_trace_simplified_2():
    trace = _make_trace_3(BlockRV(), BlockRV(), add_postproc=True)
    assert str(trace) == "\n".join(
        (
            "# from tvm import tir",
            "def apply_trace(sch: tir.Schedule) -> None:",
            '  b0 = sch.get_block(name="B", func_name="main")',
            "  sch.compute_inline(block=b0)",
            '  b1 = sch.get_block(name="C", func_name="main")',
            "  sch.enter_postproc()",
            "  sch.compute_inline(block=b1)",
        )
    )
    trace = trace.simplified(remove_postproc=False)
    assert str(trace) == "\n".join(
        (
            "# from tvm import tir",
            "def apply_trace(sch: tir.Schedule) -> None:",
            '  b0 = sch.get_block(name="B", func_name="main")',
            "  sch.compute_inline(block=b0)",
            '  b1 = sch.get_block(name="C", func_name="main")',
            "  sch.enter_postproc()",
            "  sch.compute_inline(block=b1)",
        )
    )


def test_trace_simplified_3():
    trace = _make_trace_4(BlockRV(), LoopRV(), LoopRV(), LoopRV()).simplified(remove_postproc=False)
    assert str(trace) == "\n".join(
        (
            "# from tvm import tir",
            "def apply_trace(sch: tir.Schedule) -> None:",
            '  b0 = sch.get_block(name="B", func_name="main")',
            "  l1, = sch.get_loops(block=b0)",
            "  l2, l3 = sch.split(loop=l1, factors=[None, 32], preserve_unit_iters=True)",
        )
    )


def test_apply_json_to_schedule_1():
    trace = _make_trace_2(BlockRV())
    json_obj = trace.as_json()
    sch = tir.Schedule(elementwise, debug_mask="all")
    Trace.apply_json_to_schedule(json_obj, sch)
    tvm.ir.assert_structural_equal(elementwise_inlined, sch.mod["main"])


def _test_apply_annotation_trace_from_json(annotation: str):
    """Test applying an annotation works without crashing.

    Designed to handle some previously failing edge cases like the
    empty string.
    """
    b0 = BlockRV()
    trace = Trace(
        insts=[
            _make_get_block(name="B", output=b0),
            _make_annotate(block=b0, annotation=annotation),
        ],
        decisions={},
    )
    json_obj = trace.as_json()
    sch = tir.Schedule(elementwise, debug_mask="all")
    Trace.apply_json_to_schedule(json_obj, sch)

    @T.prim_func
    def elementwise_expected(a: T.handle, c: T.handle) -> None:
        A = T.match_buffer(a, (128, 128))
        B = T.alloc_buffer((128, 128))
        C = T.match_buffer(c, (128, 128))
        for i, j in T.grid(128, 128):
            with T.block("B"):
                T.block_attr({"meta_schedule.auto_tensorize": annotation})
                vi, vj = T.axis.remap("SS", [i, j])
                B[vi, vj] = A[vi, vj] * 2.0
        for i, j in T.grid(128, 128):
            with T.block("C"):
                vi, vj = T.axis.remap("SS", [i, j])
                C[vi, vj] = B[vi, vj] + 1.0

    tvm.ir.assert_structural_equal(elementwise_expected, sch.mod["main"])


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
    test_trace_simplified_2()
    # tvm.testing.main()
