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
# ruff: noqa: F811
import numpy as np
import pytest
import tvm_ffi

import tvm
from tvm import relax as rx
from tvm import tirx
from tvm.relax.expr import make_shape
from tvm.script import relax as R


def _check_equal(x, y, map_free_vars=False):
    tvm.ir.assert_structural_equal(x, y, map_free_vars)
    tvm.ir.assert_structural_equal(y, x, map_free_vars)

    xhash = tvm_ffi.structural_hash(x, map_free_vars)
    yhash = tvm_ffi.structural_hash(y, map_free_vars)

    assert xhash == yhash


def _check_json_roundtrip(x):
    xret = tvm.ir.load_json(tvm.ir.save_json(x))
    _check_equal(x, xret, map_free_vars=True)
    return xret


def test_var() -> None:
    v0 = rx.Var("v0")
    assert v0.name_hint == "v0"
    assert v0.ty is None
    shape = [54, 96]
    v1 = rx.Var("v1", R.Tensor(shape, "float32"))
    assert v1.name_hint == "v1"
    for s0, s1 in zip(v1.ty.shape, shape):
        assert s0 == s1
    tvm.ir.assert_structural_equal(v1.ty, rx.TensorType(shape, "float32"))


def test_relax_expr_ty_running_example() -> None:
    m = tirx.Var("m", "int64")
    x = rx.Var("x", R.Tensor([m, 16], "float32"))

    assert isinstance(x.ty, tvm.ir.Type)
    assert x.ty.dtype == "float32"
    assert x.ty.ndim == 2

    call = rx.op.add(x, x)
    assert call.ty is None

    bb = rx.BlockBuilder()
    normalized = bb.normalize(call)

    assert isinstance(normalized.ty, tvm.ir.Type)
    tvm.ir.assert_structural_equal(normalized.ty, x.ty)


def test_dataflow_var() -> None:
    v0 = rx.DataflowVar("v0")
    assert v0.name_hint == "v0"
    assert v0.ty is None

    shape = [54, 96]
    v1 = rx.DataflowVar("v1", R.Tensor(shape, "float16"))
    assert v1.name_hint == "v1"

    assert isinstance(v1, rx.DataflowVar)
    tvm.ir.assert_structural_equal(v1.ty, rx.TensorType(shape, "float16"))


def test_tuple() -> None:
    v0 = rx.Var("v0")
    v1 = rx.Var("v1")
    t = rx.Tuple((v0, v1))

    assert t.fields[0] == v0
    assert t.fields[1] == v1
    assert t[0] == v0
    assert t[1] == v1
    assert t[-1] == v1
    assert t[-2] == v0

    with pytest.raises(IndexError, match="Tuple index out of range"):
        t[2]

    with pytest.raises(IndexError, match="Tuple index out of range"):
        t[-3]


def test_tuple_ty_inferred_on_construction():
    v0 = rx.Var("v0", rx.ObjectType())
    v1 = rx.Var("v1", rx.ObjectType())
    tup = rx.Tuple((v0, v1))

    assert tup.ty is not None
    tvm.ir.assert_structural_equal(tup.ty, rx.TupleType([rx.ObjectType(), rx.ObjectType()]))


def test_tuple_ty_requires_fields_with_known_ty():
    v0 = rx.Var("v0", rx.ObjectType())
    v1 = rx.Var("v1")
    tup = rx.Tuple((v0, v1))

    assert tup.ty is None


def test_match_cast() -> None:
    # match_cast([16, 8], [m, n])
    m = tirx.Var("m", dtype="int64")
    n = tirx.Var("n", dtype="int64")
    shape = rx.const([16, 8], "int32")
    var = rx.Var("v0", R.Shape())
    b0 = rx.MatchCast(var, shape, R.Tensor([m, n], "int32"))
    assert b0.value == shape
    assert b0.pattern[0] == m
    assert b0.pattern[1] == n
    assert b0.var is not None

    # var1: R.Tensor((m, n), "float32") =
    #   match_cast(var0: R.Tensor("float32", ndim=-1), R.Tensor((m, n), "float32"))
    value = rx.Var("value", R.Tensor("float32", ndim=-1))

    var = rx.Var("v1", R.Tensor([m, n], "float32"))
    b1 = rx.MatchCast(var, value, R.Tensor([m, n], "float32"))
    assert b1.value == value
    assert b1.pattern[0] == m
    assert b1.pattern[1] == n
    assert b1.var is not None


def test_match_cast() -> None:
    m = tirx.Var("m", dtype="int64")
    n = tirx.Var("n", dtype="int64")
    ivalue = rx.Var("input_value")
    ty = rx.TensorType([n, m], "float32")
    b0 = rx.MatchCast(rx.Var("v"), ivalue, ty)
    assert b0.value.same_as(ivalue)
    assert b0.ty == ty
    _check_json_roundtrip(b0)


def test_var_binding() -> None:
    v0 = rx.Var("v0")
    val = rx.const(np.random.rand(24, 56))
    b0 = rx.VarBinding(v0, val)
    assert b0.var.name_hint == "v0"
    assert b0.value == val


def test_binding_block() -> None:
    m = tirx.Var("m", dtype="int64")
    n = tirx.Var("n", dtype="int64")
    shape = rx.const([16, 8], "int32")
    b0 = rx.MatchCast(rx.Var("v0"), shape, R.Tensor([m, n], "int32"))

    v0 = rx.Var("v0")
    val = rx.const(np.random.rand(24, 56))
    b1 = rx.VarBinding(v0, val)

    block0 = rx.BindingBlock([b0, b1])
    assert block0.bindings[0] == b0
    assert block0.bindings[1] == b1


def test_dataflow_block() -> None:
    m = tirx.Var("m", dtype="int64")
    n = tirx.Var("n", dtype="int64")
    shape = rx.const([16, 8], "int32")
    b0 = rx.MatchCast(rx.Var("v0"), shape, R.Tensor([m, n], "int32"))

    v0 = rx.Var("v0")
    val = rx.const(np.random.rand(24, 56))
    b1 = rx.VarBinding(v0, val)

    block0 = rx.DataflowBlock([b0, b1])
    assert block0.bindings[0] == b0
    assert block0.bindings[1] == b1
    assert isinstance(block0, rx.DataflowBlock)


def test_seq_expr() -> None:
    x = rx.Var("foo")
    bindings = [rx.VarBinding(x, rx.const(1))]
    blocks = [rx.BindingBlock(bindings)]
    seqe = rx.SeqExpr(blocks, x)
    assert seqe.blocks[0] == blocks[0]
    assert seqe.body == x


def test_func():
    x = rx.Var("foo", R.Tensor(dtype="float32", ndim=2))
    bindings = [rx.VarBinding(x, rx.const(1))]
    blocks = [rx.BindingBlock(bindings)]

    seqe = rx.SeqExpr(blocks, x)
    ret_ty = R.Tensor(dtype="float32", ndim=-1)
    func = rx.Function([x], seqe, ret_ty)
    func = func.with_attr("global_symbol", "func")
    assert func.params[0] == x
    assert func.body == seqe
    assert func.ret_ty == ret_ty
    assert func.attrs["global_symbol"] == "func"


def test_shape_of():
    shape = [96, 54]
    v1 = rx.Var("v1", R.Tensor(shape))
    s1 = rx.get_shape_of(v1)
    for x, y in zip(shape, s1):
        assert x == y


def test_shape_expr():
    m = tirx.Var("m", dtype="int64")
    n = tirx.Var("n", dtype="int64")
    s = rx.ShapeExpr([m, n])
    assert s.values[0] == m
    assert s.values[1] == n
    assert s[0] == m
    assert s[1] == n
    assert s[-1] == n
    assert s[-2] == m
    assert isinstance(s.ty, rx.ShapeType)

    with pytest.raises(IndexError, match="ShapeExpr index out of range"):
        s[2]

    with pytest.raises(IndexError, match="ShapeExpr index out of range"):
        s[-3]

    shape_expr = rx.ShapeExpr([10, 20])
    assert shape_expr.values[0] == 10
    assert shape_expr.values[1] == 20
    tvm.ir.assert_structural_equal(shape_expr.ty, R.Shape((10, 20)))

    x = rx.Var("v0", R.Tensor((10, 20), "float32"))
    assert x.ty.shape[0] == 10
    assert x.ty.shape[1] == 20
    tvm.ir.assert_structural_equal(x.ty.shape.ty, R.Shape((10, 20)))

    m = tirx.Var("m", "int32")
    with pytest.raises(RuntimeError, match="the value in ShapeType can only have dtype of int64"):
        rx.ShapeExpr([m, 3])


def test_prim_value():
    pv = rx.PrimValue(tirx.IntImm("int64", 1))
    assert pv.value.value == 1
    _check_equal(pv, rx.PrimValue(tirx.IntImm("int64", 1)))
    _check_json_roundtrip(pv)


def test_prim_value_with_var():
    n = tirx.Var("n", "int64")
    pv = rx.PrimValue(n)
    assert pv.value.same_as(n)
    tvm.ir.assert_structural_equal(pv.ty, tvm.ir.PrimType("int64"))
    _check_equal(pv, rx.PrimValue(n))
    _check_json_roundtrip(pv)


def test_prim_value_with_expr():
    n = tirx.Var("n", "int64")
    pv = rx.PrimValue(n + 1)
    tvm.ir.assert_structural_equal(pv.ty, tvm.ir.PrimType("int64"))
    _check_equal(pv, rx.PrimValue(n + 1))
    _check_json_roundtrip(pv)


def test_string_imm():
    s0 = rx.StringImm("hello")
    s1 = rx.StringImm("hello")
    assert s0.value == "hello"
    _check_equal(s0, s1)
    _check_json_roundtrip(s0)


def test_datatype_imm():
    d0 = rx.DataTypeImm("int32")
    d1 = rx.DataTypeImm("int32")
    assert d0.value == "int32"
    _check_equal(d0, d1)
    _check_json_roundtrip(d0)


def test_call():
    dtype = tvm.ir.PrimType("int32")
    func = rx.Var("func", rx.FuncType([dtype], dtype))
    arg = rx.Var("arg", dtype)
    call = rx.Call(func, [arg])
    assert call.op.same_as(func)
    assert len(call.args) == 1
    assert call.args[0].same_as(arg)


def test_call_raises_error_for_invalid_function():
    """relax::Call requires the function to have FuncType"""
    dtype = tvm.ir.PrimType("int32")
    func = rx.Var("func", dtype)
    arg = rx.Var("arg", dtype)

    with pytest.raises(ValueError):
        rx.Call(func, [arg])


def test_call_raises_error_for_missing_operator():
    """relax::Call requires a defined operator."""
    with pytest.raises(ValueError, match="defined operator"):
        rx.Call(None, [])


if __name__ == "__main__":
    tvm.testing.main()


def test_make_shape_invalid_type():
    with pytest.raises(TypeError):
        make_shape(123)


def test_make_shape_valid_list():
    shape = make_shape([1, 2, 3])
    assert len(shape) == 3


def test_make_shape_valid_tuple():
    shape = make_shape((4, 5))
    assert len(shape) == 2


if __name__ == "__main__":
    tvm.testing.main()
