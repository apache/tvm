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
"""Construction contracts for the shared parser support namespace."""

import pytest

from tvm import tirx
from tvm.script import parser
from tvm.script.ir_builder import IRBuilder
from tvm.script.ir_builder import ir as I
from tvm.script.ir_builder import parser_support as PS


def test_selection_helpers_are_not_part_of_shared_support():
    assert not any(hasattr(PS, name) for name in ("if_expr", "and_expr", "or_expr"))


def test_global_info_requires_module_and_keeps_objects():
    marker = object()
    assert I.parser_support is PS
    assert PS.lookup_global_info(marker) is marker
    with pytest.raises(ValueError, match="enclosing module"):
        PS.lookup_global_info("cuda:1")
    with IRBuilder():
        with PS.TypeVarFrame():
            with pytest.raises(ValueError, match="enclosing module"):
                PS.lookup_global_info("mesh[0]")


def test_global_info_selectors_use_module_map():
    with IRBuilder():
        with I.ir_module():
            devices = [I.vdevice("llvm"), I.vdevice("cuda"), I.vdevice("cuda", 1)]
            I.module_global_infos({"vdevice": devices, "other": [I.dummy_global_info()]})
            assert PS.lookup_global_info("cuda:1").__chandle__() == devices[2].__chandle__()
            assert PS.lookup_global_info("vdevice[1]").__chandle__() == devices[1].__chandle__()
            assert PS.lookup_global_info("cuda").__chandle__() == devices[1].__chandle__()
            assert PS.lookup_global_info("other[0]") is not None


def test_parser_skips_invalid_ramp_and_preserves_support_name():
    source = """
@T.prim_func
def main():
    T.evaluate(T.ramp(0, 1, _PS) if I.constexpr(_PS > 1) else 0)
"""
    function = parser.parse(source, extra_vars={"_PS": 1})
    assert isinstance(function.body, tirx.Evaluate)
    assert function.body.value.value == 0


def test_source_logical_operands_skip_at_construction():
    def fail():
        raise AssertionError("skipped source operand was evaluated")

    function = parser.parse(
        """
@T.prim_func
def main():
    T.evaluate(1 if I.constexpr(False and fail()) else 2)
    T.evaluate(3 if I.constexpr(True or fail()) else 4)
""",
        extra_vars={"fail": fail},
    )
    assert [statement.value.value for statement in function.body.seq] == [2, 3]


def test_generated_module_enters_global_info_scope_before_annotations():
    module = parser.parse("""
@I.ir_module
class Module:
    I.module_global_infos({
        "vdevice": [I.vdevice("llvm"), I.vdevice("cuda"), I.vdevice("cuda", 1)],
        "mesh": [R.device_mesh((2,), [0, 1])],
    })
    @R.function
    def tensor(x: R.Tensor((4,), "float32", vdevice="cuda:1")):
        return x
    @R.function
    def distributed(x: R.DTensor((4,), "float32", device_mesh="mesh[0]", placement="S[0]")):
        return x
""")
    assert (
        module["tensor"].params[0].ty.vdevice.__chandle__()
        == module.global_infos["vdevice"][2].__chandle__()
    )
    assert (
        module["distributed"].params[0].ty.device_mesh.__chandle__()
        == module.global_infos["mesh"][0].__chandle__()
    )
