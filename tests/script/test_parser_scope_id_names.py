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
"""Anonymous scope declarations take source names without adding bindings."""

import pytest

import tvm
from tvm.script import parser
from tvm.script.ir_builder import IRBuilder
from tvm.script.ir_builder.base import BypassBind
from tvm.script.ir_builder.type_var_frame import TypeVarFrame
from tvm.script import tirx as T


@pytest.mark.parametrize(
    "constructor,args",
    [
        ("scope_id", ([32], "cta", "thread")),
        ("cluster_id", ([2],)),
        ("cta_id", ([2],)),
        ("cta_id_in_cluster", ([2],)),
        ("cta_id_in_pair", ()),
        ("warpgroup_id", ([2],)),
        ("warp_id", ([4],)),
        ("warp_id_in_wg", ([4],)),
        ("lane_id", ([32],)),
        ("thread_id", ([32],)),
        ("thread_id_in_wg", ([128],)),
    ],
)
def test_anonymous_scope_name_preserves_native_declaration(constructor, args):
    with IRBuilder() as builder, TypeVarFrame():
        with T.function():
            T.device_entry()
            result = getattr(T, constructor)(*args)
            original = result.value
            assert original.name == ""
            bound = T.bind_(result, name="source_id")
            assert bound.same_as(original)
            assert bound.name == "source_id"
            # Reusing the declaration must not rename its existing identity.
            assert T.bind_(result, name="alias").same_as(bound)
            assert bound.name == "source_id"
            T.evaluate(bound)
        function = builder.get()
    declaration, use = function.body.body.seq
    assert getattr(declaration, "def").def_ids[0].same_as(use.value)
    assert use.value.name == "source_id"


@pytest.mark.parametrize("target,names", [("cta_id", ["cta_id"]), ("bx, by", ["bx", "by"])])
def test_source_assignment_supplies_scalar_and_unpacked_names(target, names):
    extents = [2] if len(names) == 1 else [2, 3]
    evaluations = "\n".join(f"    T.evaluate({name})" for name in names)
    function = parser.parse(f"""
@T.prim_func
def main():
    T.device_entry()
    {target} = T.cta_id({extents})
{evaluations}
""")
    declaration, *uses = function.body.body.seq
    variables = getattr(declaration, "def").def_ids
    assert [var.name for var in variables] == names
    assert len(uses) == len(names)
    assert all(var.same_as(use.value) for var, use in zip(variables, uses))


def test_explicit_native_scope_name_is_retained():
    with IRBuilder(), TypeVarFrame(), T.function():
        T.device_entry()
        result = T.thread_id([32])
        IRBuilder.name("native_name", result.value)
        assert T.bind_(result, name="source_name").same_as(result.value)
        assert result.value.name == "native_name"
        T.evaluate(result.value)


def test_generic_bypass_does_not_opt_into_source_naming():
    variable = tvm.ir.Var("", "int32")
    assert T.bind_(BypassBind(variable), name="source_name").same_as(variable)
    assert variable.name == ""
    value = object()
    assert T.bind_(BypassBind(value), name="source_name") is value


def test_scope_tuple_preserves_value_and_individual_name_opt_in():
    with IRBuilder(), TypeVarFrame(), T.function():
        T.device_entry()
        result = T.cta_id([2, 3])
        values = T.bind_(result, name="ids")
        assert values is result.value
        # A tuple target supplies no individual scalar assignment names.
        assert [value.name for value in values] == ["", ""]
        x, y = T.unpack(result)
        for wrapped, value, name in zip((x, y), values, ("bx", "by")):
            assert T.bind_(wrapped, name=name).same_as(value)
            assert value.name == name
            T.evaluate(value)
