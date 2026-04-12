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
# ruff: noqa: E711

import pytest

import tvm
from tvm import te, topi


def test_expr_constructor():
    x = tvm.tirx.Var("xx", "float32")
    assert isinstance(x, tvm.tirx.Var)
    assert x.name == "xx"

    x = tvm.tirx.Reduce(None, [1], [tvm.tirx.IterVar((0, 1), "x", 2)], None, 0)
    assert isinstance(x, tvm.tirx.Reduce)
    assert x.combiner is None
    assert x.value_index == 0

    x = tvm.tirx.FloatImm("float32", 1.0)
    assert isinstance(x, tvm.tirx.FloatImm)
    assert x.value == 1.0
    assert x.dtype == "float32"

    x = tvm.tirx.IntImm("int64", 2)
    assert isinstance(x, tvm.tirx.IntImm)
    assert x.value == 2
    assert x.dtype == "int64"

    x = tvm.tirx.StringImm("xyza")
    assert isinstance(x, tvm.tirx.StringImm)
    assert x.value == "xyza"

    x = tvm.tirx.Cast("float32", tvm.tirx.IntImm("uint32", 1))
    assert isinstance(x, tvm.tirx.Cast)
    assert x.dtype == "float32"
    assert x.value.value == 1

    a = tvm.tirx.const(1.0, dtype="float32")
    b = tvm.tirx.Var("x", "float32")

    for cls in [
        tvm.tirx.Add,
        tvm.tirx.Sub,
        tvm.tirx.Mul,
        tvm.tirx.Div,
        tvm.tirx.Mod,
        tvm.tirx.Min,
        tvm.tirx.Max,
        tvm.tirx.LT,
        tvm.tirx.LE,
        tvm.tirx.GT,
        tvm.tirx.GE,
    ]:
        x = cls(a, b)
        assert isinstance(x, cls)
        assert x.a == a
        assert x.b.same_as(b)

    a = tvm.runtime.convert(tvm.tirx.Var("x", "int32") > 1)
    b = tvm.runtime.convert(tvm.tirx.Var("x", "int32") == 1)

    for cls in [tvm.tirx.And, tvm.tirx.Or]:
        x = cls(a, b)
        assert isinstance(x, cls)
        assert x.a == a
        assert x.b.same_as(b)

    x = tvm.tirx.Not(a)
    assert isinstance(x, tvm.tirx.Not)
    assert x.a == a

    x = tvm.tirx.Select(a, a, b)
    assert isinstance(x, tvm.tirx.Select)
    assert x.true_value == a
    assert x.false_value == b
    assert x.condition == a

    buffer_var = tvm.tirx.Var("buf", tvm.ir.PointerType(tvm.ir.PrimType("float32")))
    buffer = tvm.tirx.decl_buffer([16], "float32", data=buffer_var)
    x = tvm.tirx.BufferLoad(buffer, [1])
    assert isinstance(x, tvm.tirx.BufferLoad)
    assert x.dtype == "float32"
    assert x.buffer == buffer
    assert x.buffer.data == buffer_var
    assert list(x.indices) == [1]

    x = tvm.tirx.Ramp(1, 2, 10)
    assert isinstance(x, tvm.tirx.Ramp)
    assert x.base.value == 1
    assert x.stride.value == 2
    assert x.lanes == 10

    x = tvm.tirx.Broadcast(a, 10)
    assert isinstance(x, tvm.tirx.Broadcast)
    assert x.value == a
    assert x.lanes == 10

    x = tvm.tirx.Shuffle([a], [0])
    assert isinstance(x, tvm.tirx.Shuffle)
    assert x.vectors[0] == a
    assert x.indices[0].value == 0

    x = tvm.tirx.Call("float32", "tirx.call_extern", [tvm.tirx.StringImm("xyz"), a])
    assert isinstance(x, tvm.tirx.Call)
    assert x.dtype == "float32"
    assert x.op.name == "tirx.call_extern"
    assert x.args[1] == a

    v = tvm.tirx.Var("aa", "int32")
    x = tvm.tirx.Let(v, 1, v)
    assert x.var == v
    assert x.value.value == 1
    assert x.body == v


def test_stmt_constructor():
    v = tvm.tirx.Var("aa", "int32")
    nop = tvm.tirx.Evaluate(1)
    x = tvm.tirx.Bind(v, 1)
    assert isinstance(x, tvm.tirx.Bind)
    assert x.var == v
    assert x.value.value == 1

    x = tvm.tirx.AttrStmt(v == 1, "xx", 1, tvm.tirx.Evaluate(1))
    assert isinstance(x, tvm.tirx.AttrStmt)
    assert x.value.value == 1

    x = tvm.tirx.AssertStmt(
        tvm.tirx.const(1, "bool"),
        tvm.tirx.StringImm("RuntimeError"),
        [tvm.tirx.StringImm("hellow")],
    )
    assert isinstance(x, tvm.tirx.AssertStmt)
    assert x.error_kind.value == "RuntimeError"
    assert len(x.message_parts) == 1
    assert x.message_parts[0].value == "hellow"

    x = tvm.tirx.For(tvm.tirx.Var("x", "int32"), 0, 10, tvm.tirx.ForKind.SERIAL, nop)
    assert isinstance(x, tvm.tirx.For)
    assert x.min.value == 0
    assert x.extent.value == 10
    assert x.body == nop

    buffer_var = tvm.tirx.Var("buf", tvm.ir.PointerType(tvm.ir.PrimType("bool")))
    buffer = tvm.tirx.decl_buffer([16], "bool", data=buffer_var)
    x = tvm.tirx.BufferStore(buffer, tvm.tirx.IntImm("bool", 1), [10])
    assert isinstance(x, tvm.tirx.BufferStore)
    assert x.buffer == buffer
    assert x.buffer.data == buffer_var
    assert list(x.indices) == [10]
    assert x.value.value == 1

    buf = tvm.tirx.decl_buffer([10], "float32")
    x = tvm.tirx.AllocBuffer(buf)
    assert isinstance(x, tvm.tirx.AllocBuffer)
    assert x.buffer == buf

    x = tvm.tirx.AttrStmt(buffer_var, "xyz", 1, nop)
    assert isinstance(x, tvm.tirx.AttrStmt)
    assert x.node == buffer_var
    assert x.attr_key == "xyz"
    assert x.body == nop

    x = tvm.tirx.IfThenElse(tvm.tirx.const(1, "bool"), tvm.tirx.Evaluate(11), nop)
    assert isinstance(x, tvm.tirx.IfThenElse)
    assert x.then_case.value.value == 11
    assert x.else_case == nop


def test_float_constructor_requires_float_dtype():
    with pytest.raises(tvm.TVMError):
        tvm.tirx.FloatImm("int32", 1.0)


def test_math_unary_constructor_requires_float_dtype():
    x = tvm.tirx.Var("x", "int32")

    with pytest.raises(TypeError, match=r"tirx\.tan only supports floating-point inputs"):
        tvm.tirx.tan(x)

    with pytest.raises(TypeError, match=r"tirx\.sin only supports floating-point inputs"):
        tvm.tirx.sin(x)

    y = tvm.tirx.Var("y", "float32")
    assert tvm.tirx.tan(y).dtype == "float32"


def test_topi_tan_requires_float_dtype():
    x = te.placeholder((2, 2), dtype="int32", name="x")

    with pytest.raises(TypeError, match=r"tirx\.tan only supports floating-point inputs"):
        topi.tan(x)


def test_math_unary_constructor_preserves_bfloat16():
    x = tvm.tirx.Var("x", "bfloat16")
    y = tvm.tirx.exp(x)
    assert y.dtype == "bfloat16"


if __name__ == "__main__":
    tvm.testing.main()
