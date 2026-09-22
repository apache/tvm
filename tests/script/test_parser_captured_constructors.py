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
"""Captured public annotation constructors retain their canonical identities."""

import pytest

import tvm
from tvm.relax.script import builder as B
from tvm.script import parser
from tvm.script import relax as R
from tvm.script.ir_builder import IRBuilder


@pytest.mark.parametrize("name", ["Any", "Tensor", "Callable", "Tuple", "Shape"])
def test_captured_default_annotation_constructor(name):
    constructor = getattr(R, name)
    source = """
@R.function
def main(x: R.Any):
    y = R.match_cast(x, captured_constructor)
    return y
"""
    captures = {"R": R, "captured_constructor": constructor}
    with IRBuilder() as builder:
        with B.function():
            B.func_name("main")
            x = B.arg("x", B.Any())
            y = B.emit_match_cast(x, constructor())
            B.func_ret_value(y)
    actual = parser.parse(source, extra_vars=captures)
    tvm.ir.assert_structural_equal(builder.get(), actual)
    assert captures["captured_constructor"] is constructor
    assert getattr(R, name) is getattr(B, name) is constructor


def test_custom_captured_annotation_keeps_identity():
    annotation = tvm.relax.TensorType((2, 3), "float32")
    captures = {"annotation": annotation}
    actual = parser.parse(
        "@R.function\ndef main(x: annotation):\n    return x\n", extra_vars=captures
    )
    assert captures["annotation"] is annotation
    assert actual.params[0].ty.same_as(annotation)
    assert actual.body.body.same_as(actual.params[0])
