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
"""Foundational IR constructors keep their identity in public script namespaces."""

import pytest

import tvm
from tvm.relax.script import builder as R
from tvm.script import parser
from tvm.script.ir_builder import IRBuilder
from tvm.script.ir_builder import ir as I


@pytest.mark.parametrize("name", ["Range", "StringType", "StringImm", "GenericConst"])
def test_shared_ir_constructor_identity(name):
    # Initialize the source namespace before inspecting its shared constructors.
    parse = parser.parse
    assert callable(parse)
    assert getattr(I, name) is getattr(tvm.ir, name)
    assert getattr(parser.I, name) is getattr(tvm.ir, name)


def test_distributed_module_range():
    source = """
@I.ir_module
class Module:
    I.module_global_infos({"mesh": [R.device_mesh((2,), I.Range(0, 2))]})

    @R.function
    def main(
        x: R.DTensor((16,), "float32", "mesh[0]", "R"),
    ) -> R.DTensor((16,), "float32", "mesh[0]", "R"):
        return x
"""
    with IRBuilder() as builder:
        with I.ir_module():
            mesh = R.device_mesh((2,), tvm.ir.Range(0, 2))
            I.module_global_infos({"mesh": [mesh]})
            with R.function():
                R.func_name("main")
                annotation = R.DTensor((16,), "float32", mesh, "R")
                x = R.arg("x", annotation)
                R.func_ret_type(annotation)
                R.func_ret_value(x)
    expected = builder.get()
    actual = parser.parse(source)
    tvm.ir.assert_structural_equal(expected, actual)


def test_module_string_constants():
    source = """
@I.ir_module
class Module:
    I.module_attrs({
        "tag": I.StringImm("label"),
        "type": I.StringType(),
    })

    @R.function
    def main(x: R.Tensor((16,), "float32")):
        return x
"""
    with IRBuilder() as builder:
        with I.ir_module():
            I.module_attrs({"tag": tvm.ir.StringImm("label"), "type": tvm.ir.StringType()})
            with R.function():
                R.func_name("main")
                x = R.arg("x", R.Tensor((16,), "float32"))
                R.func_ret_value(x)
    expected = builder.get()
    actual = parser.parse(source)
    tvm.ir.assert_structural_equal(expected, actual)
    assert actual.attrs["tag"].value == "label"
    assert isinstance(actual.attrs["type"], tvm.ir.StringType)
