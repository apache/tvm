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
"""Schedule mutations validate the semantic thread-binding annotation atomically."""

import pytest

import tvm
import tvm.testing
from tvm import tirx
from tvm.s_tir.schedule.analysis import get_auto_tensorize_mapping_info
from tvm.script import tirx as T


@T.prim_func(s_tir=True)
def copy(A: T.Buffer((16,), "float32"), B: T.Buffer((16,), "float32")):
    for i in range(16):
        with T.sblock("copy"):
            vi = T.axis.spatial(16, i)
            B[vi] = A[vi]


def make_schedule(kind="parallel"):
    sch = tvm.s_tir.Schedule(copy, debug_mask="all")
    (loop,) = sch.get_loops(sch.get_sblock("copy"))
    if kind != "serial":
        getattr(sch, kind)(loop)
    return sch, loop


def test_binding_changes_use_schedule_primitives():
    sch, loop = make_schedule()
    sch.annotate(loop, "hint", 7)
    sch.bind(loop, "threadIdx.x")
    node = sch.get(loop)
    assert node.kind == tirx.ForKind.PARALLEL
    assert node.thread_binding.thread_tag == "threadIdx.x"
    assert node.annotations["thread_binding"].same_as(node.thread_binding)
    assert node.annotations["hint"] == 7

    sch.parallel(loop)
    node = sch.get(loop)
    assert node.kind == tirx.ForKind.PARALLEL
    assert node.thread_binding is None
    assert "thread_binding" not in node.annotations
    assert node.annotations["hint"] == 7


@pytest.mark.parametrize("kind", ["parallel", "serial", "vectorize", "unroll"])
@pytest.mark.parametrize("value", ["threadIdx.x", 1])
def test_annotate_rejects_semantic_binding_key(kind, value):
    sch, loop = make_schedule(kind)
    before = sch.mod.script()
    trace_before = str(sch.trace)
    with pytest.raises(ValueError, match="use Schedule.bind"):
        sch.annotate(loop, "thread_binding", value)
    assert sch.mod.script() == before
    assert str(sch.trace) == trace_before
    assert sch.get(loop).thread_binding is None


def test_unannotate_rejects_semantic_binding_key():
    sch, loop = make_schedule()
    sch.bind(loop, "threadIdx.x")
    before = sch.mod.script()
    trace_before = str(sch.trace)
    with pytest.raises(ValueError, match="use Schedule.parallel"):
        sch.unannotate(loop, "thread_binding")
    assert sch.mod.script() == before
    assert str(sch.trace) == trace_before
    assert sch.get(loop).thread_binding.thread_tag == "threadIdx.x"


@T.prim_func(s_tir=True)
def reduce(A: T.Buffer((16,), "float32"), B: T.Buffer((1,), "float32")):
    for i in range(16):
        with T.sblock("sum"):
            vi = T.axis.reduce(16, i)
            with T.init():
                B[0] = T.float32(0)
            B[0] = B[0] + A[vi]


def test_bound_reduction_cannot_bypass_parallel_legality():
    sch = tvm.s_tir.Schedule(reduce, debug_mask="all")
    (loop,) = sch.get_loops(sch.get_sblock("sum"))
    sch.bind(loop, "threadIdx.x")
    before = sch.mod.script()
    with pytest.raises(ValueError, match="use Schedule.parallel"):
        sch.unannotate(loop, "thread_binding")
    with pytest.raises(tvm.s_tir.ScheduleError):
        sch.parallel(loop)
    assert sch.mod.script() == before
    assert sch.get(loop).thread_binding.thread_tag == "threadIdx.x"


def test_sblock_thread_binding_hint_is_not_reserved():
    sch, _ = make_schedule()
    block = sch.get_sblock("copy")
    sch.annotate(block, "thread_binding", "block_hint")
    assert sch.get(block).annotations["thread_binding"] == "block_hint"
    sch.unannotate(block, "thread_binding")
    assert "thread_binding" not in sch.get(block).annotations


def test_sblock_thread_binding_hint_in_tensorize_comparison():
    sch, _ = make_schedule("serial")
    block = sch.get_sblock("copy")
    sch.annotate(block, "thread_binding", "block_hint")
    desc = sch.mod["main"]
    assert get_auto_tensorize_mapping_info(sch, block, desc) is not None
    sch.unannotate(block, "thread_binding")
    sch.annotate(block, "thread_binding", "different_hint")
    assert get_auto_tensorize_mapping_info(sch, block, desc) is None


if __name__ == "__main__":
    tvm.testing.main()
