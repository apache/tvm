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
"""Normal IR constructors retain their contracts in TVMScript."""

import inspect
import linecache

import pytest

import tvm_tirx
import tvm_tirx.testing
from tvm_tirx import ir, tirx
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T


def test_cast_script_positional_and_keyword_type():
    @T.function(private=True)
    def positional(x: T.int32):
        T.evaluate(T.cast(x, "float32"))

    @T.function(private=True)
    def keyword(x: T.int32):
        T.evaluate(T.cast(x, ty="float32"))

    ir.assert_structural_equal(positional, keyword)
    cast_expr = positional.body[0].value
    assert cast_expr.ty == ir.PrimType("float32")
    assert isinstance(cast_expr.loc, ir.SourceLoc)
    ir.assert_structural_equal(
        positional, tvm_tirx.script.from_source(positional.script(), extra_vars={"T": T})
    )
    value = tirx.Var("x", "int32")
    loc = ir.SourceLoc(ir.SourceName("cast"), 1, 1, 1, 8)
    assert T.cast(value, "float32", loc=loc).loc.same_as(loc)
    assert T.cast(value, ty="float32", loc=loc).loc.same_as(loc)
    with pytest.raises(TypeError):
        T.cast(value, "float32", loc)


def test_call_type_and_validation_contract():
    x = ir.Var("x", "float32")
    loc = ir.SourceLoc(ir.SourceName("constructor"), 1, 1, 1, 5)
    for constructor in (ir.Call, I.Call):
        call = constructor("prim.exp", [x], loc=loc)
        assert isinstance(call, I.Call)
        assert call.ty == ir.PrimType("float32")
        assert call.loc.same_as(loc)
        unlocated = constructor("prim.exp", [x], ty="float32")
        assert unlocated.loc.same_as(ir.UNKNOWN_LOC)
        assert inspect.signature(constructor).parameters["loc"].default is ir.UNKNOWN_LOC
        with pytest.raises(TypeError):
            constructor("prim.exp", [x], loc=None, ty="float32")
        ir.assert_structural_equal(
            unlocated,
            ir.Call(
                "prim.exp",
                [x],
                ty=ir.PrimType("float32"),
            ),
        )
        with pytest.raises(TypeError):
            constructor("prim.exp", [], ty="float32").validate()
        provisional = constructor("prim.exp", [], loc=loc, ty="float32")
        assert not provisional.args
        assert provisional.loc.same_as(loc)
        assert provisional.ty.dtype == "float32"


@pytest.mark.parametrize("ty", [ir.Type.missing(), "float32", "handle"])
def test_raw_call_preserves_provisional_fields(ty):
    # Invalid arguments and type arguments must survive the raw representation.
    call = ir.Call(
        "prim.exp",
        [],
        attrs={"tag": 7},
        ty_args=[ir.StringType()],
        ty=ty,
    )
    source = call.script()
    assert "T.Call(" in source
    restored = eval(source, {"I": I, "T": T})
    ir.assert_structural_equal(call, restored)


def test_raw_call_preserves_checked_fields():
    call = ir.Call(
        "prim.exp",
        [tvm_tirx.ir.FloatImm("float32", 0)],
        attrs={"tag": 7},
        ty="float32",
    )
    source = call.script()
    assert "T.Call(" in source
    ir.assert_structural_equal(call, eval(source, {"I": I, "T": T}))


def test_range_constructor_parameters():
    loc = ir.SourceLoc(ir.SourceName("constructor"), 1, 1, 1, 5)
    for args in [(5,), (2, 5)]:
        actual = T.Range(*args, loc=loc)
        ir.assert_structural_equal(actual, ir.Range(*args, loc=loc))
        assert actual.loc.same_as(loc)
    ir.assert_structural_equal(T.Range.from_min_extent(2, 3), ir.Range(2, 5))


def test_operation_type_keywords_match_normal_constructors():
    for name in ("min_value", "max_value", "infinity"):
        ir.assert_structural_equal(
            getattr(T, name)(ty="float32"), getattr(tirx, name)(ty="float32")
        )
    x = ir.Var("x", "float32")
    for operation in (T.exp, tirx.exp):
        with pytest.raises(TypeError):
            operation(x, dtype="float32")
    vector = ir.Var("v", "int8x4")
    ir.assert_structural_equal(T.gpu_dp4a(vector, vector), tirx.gpu_dp4a(vector, vector))


def test_normal_constructors_in_parsed_function():
    @T.function
    def function(x: ir.PrimType("float32")):
        T.evaluate(ir.Call("prim.exp", [x], ty="float32"))

    call = function.body[0].value
    ir.assert_structural_equal(
        call,
        ir.Call(
            "prim.exp",
            [function.params[0]],
            ty=ir.PrimType("float32"),
        ),
    )
    loc = call.loc
    assert isinstance(loc, ir.SourceLoc)
    assert loc.source_name.name == __file__
    assert loc.start_line == loc.end_line
    assert (
        linecache.getline(__file__, loc.start_line)[loc.start_column - 1 : loc.end_column - 1]
        == 'ir.Call("prim.exp", [x], ty="float32")'
    )


if __name__ == "__main__":
    tvm_tirx.testing.main()
