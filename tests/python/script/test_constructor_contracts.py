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

import pytest

import tvm
import tvm.testing
from tvm import ir, relax, tirx
from tvm.script import ir as I
from tvm.script import relax as R
from tvm.script import tirx as T


def test_call_type_and_validation_contract():
    x = ir.Var("x", "float32")
    span = ir.Span(ir.SourceName("constructor"), 1, 1, 1, 5)
    for constructor in (ir.Call, I.Call):
        call = constructor("tirx.exp", [x], span=span)
        assert isinstance(call, I.Call)
        assert call.ty == ir.PrimType("float32")
        assert call.span.same_as(span)
        ir.assert_structural_equal(
            constructor("tirx.exp", [x], ty="float32"),
            ir.Call("tirx.exp", [x], ty=ir.PrimType("float32")),
        )
        with pytest.raises(TypeError):
            constructor("tirx.exp", [], ty="float32").validate()
        provisional = constructor("tirx.exp", [], ty="float32", span=span)
        assert not provisional.args
        assert provisional.span.same_as(span)
        assert provisional.ty.dtype == "float32"


@pytest.mark.parametrize("ty", [ir.Type.missing(), "float32", "handle"])
def test_raw_call_preserves_provisional_fields(ty):
    # Invalid arguments and type arguments must survive the raw representation.
    call = ir.Call("tirx.exp", [], attrs={"tag": 7}, ty_args=[ir.StringType()], ty=ty)
    source = call.script()
    assert "I.Call(" in source
    restored = eval(source, {"I": I, "R": R, "T": T})
    ir.assert_structural_equal(call, restored)


def test_raw_call_preserves_checked_fields():
    call = ir.Call("tirx.exp", [tirx.FloatImm("float32", 0)], attrs={"tag": 7}, ty="float32")
    source = call.script()
    assert "I.Call(" in source
    ir.assert_structural_equal(call, eval(source, {"I": I, "R": R, "T": T}))


def test_range_and_value_constructor_parameters():
    span = ir.Span(ir.SourceName("constructor"), 1, 1, 1, 5)
    for args in [(5,), (2, 5)]:
        actual = T.Range(*args, span=span)
        ir.assert_structural_equal(actual, ir.Range(*args, span=span))
        assert actual.span.same_as(span)
    ir.assert_structural_equal(T.Range.from_min_extent(2, 3), ir.Range(2, 5))
    for actual, expected in [
        (R.shape([2, 3], span=span), relax.ShapeExpr([2, 3], span=span)),
        (R.str("value", span=span), ir.StringImm("value", span=span)),
    ]:
        ir.assert_structural_equal(actual, expected)
        assert actual.span.same_as(span)
    ir.assert_structural_equal(R.prim_value(3, dtype="int32"), relax.prim_value(3, dtype="int32"))


def test_operation_dtype_keywords_match_normal_constructors():
    for name in ("min_value", "max_value", "infinity"):
        ir.assert_structural_equal(
            getattr(T, name)(dtype="float32"), getattr(tirx, name)(dtype="float32")
        )
    x = ir.Var("x", "float32")
    for operation in (T.exp, tirx.exp):
        with pytest.raises(TypeError):
            operation(x, dtype="float32")
    vector = ir.Var("v", "int8x4")
    ir.assert_structural_equal(T.dp4a(vector, vector), tirx.dp4a(vector, vector))


def test_normal_constructors_in_parsed_function():
    @T.function
    def function(x: ir.PrimType("float32")):
        T.evaluate(ir.Call("tirx.exp", [x], ty="float32"))

    call = function.body.value
    ir.assert_structural_equal(
        call, ir.Call("tirx.exp", [function.params[0]], ty=ir.PrimType("float32"))
    )
    assert call.span is not None

    @R.function
    def identity(x: relax.TensorType([2], "float32")) -> relax.TensorType([2], "float32"):
        return x

    ir.assert_structural_equal(identity.params[0].ty, relax.TensorType([2], "float32"))


if __name__ == "__main__":
    tvm.testing.main()
