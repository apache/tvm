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
"""Range loop annotations survive deferred frame construction."""

import pytest

from tvm import ir, tirx
from tvm.script import parser
from tvm.script.ir_builder import IRBuilder
from tvm.script import tirx as T


def annotated_source(bounds, s_tir):
    return f"""
@T.prim_func(s_tir={s_tir})
def main(A: T.Buffer((16,), "float32")):
    for i in range(
        {bounds}, annotations={{"pragma_1": "str_value", "pragma_2": 1, "pragma_3": 0.0}}
    ):
        A[i] = 0.0
"""


@pytest.mark.parametrize("s_tir", [False, True])
@pytest.mark.parametrize("bounds", ["16", "0, 16", "0, 16, 2"])
def test_range_annotations_match_direct_builder(bounds, s_tir):
    source = annotated_source(bounds, s_tir)
    with IRBuilder() as builder:
        with T.function(s_tir=s_tir):
            T.func_name("main")
            buffer = T.arg("A", T.Buffer((16,), "float32"))
            with T.for_(
                T.range_(
                    *(int(value) for value in bounds.split(",")),
                    annotations={"pragma_1": "str_value", "pragma_2": 1, "pragma_3": 0.0},
                ),
                names="i",
            ) as index:
                T.buffer_store(buffer, 0.0, [index])
    result = parser.parse(source)
    ir.assert_structural_equal(builder.get(), result)
    loop = result.body
    assert isinstance(loop, tirx.For)
    assert set(loop.annotations) == {"pragma_1", "pragma_2", "pragma_3"}


def test_range_arguments_and_annotations_evaluate_once_in_order():
    seen = []

    def mark(name, value):
        seen.append(name)
        return value

    source = """
@T.prim_func
def main(A: T.Buffer((16,), "float32")):
    for i in range(
        mark("start", 0), mark("stop", 16), mark("step", 2),
        annotations=mark("annotations", {"pragma_test": mark("value", 1)}),
    ):
        A[i] = 0.0
"""
    captures = {"mark": mark}
    result = parser.parse(source, extra_vars=captures)
    assert seen == ["start", "stop", "step", "value", "annotations"]
    expected_source = """
@T.prim_func
def main(A: T.Buffer((16,), "float32")):
    for i in range(0, 16, 2, annotations={"pragma_test": 1}):
        A[i] = 0.0
"""
    ir.assert_structural_equal(parser.parse(expected_source), result)


def test_range_descriptor_keeps_annotations_until_frame_construction():
    annotations = {"pragma_test": 3}
    descriptor = T.range_(4, annotations=annotations)
    assert descriptor.annotations is annotations
    with IRBuilder() as builder:
        with T.for_(descriptor):
            T.evaluate(1)
    assert builder.get().annotations["pragma_test"] == 3


@pytest.mark.parametrize("args", [(), (0, 1, 2, 3), (0, 4, 0)])
def test_annotations_do_not_weaken_range_argument_validation(args):
    with pytest.raises((TypeError, ValueError), match="range"):
        T.range_(*args, annotations={"pragma_test": 1})
