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
# ruff: noqa: F821
"""Test type nodes in the IR"""

import pytest

import tvm
from tvm import ir
from tvm.script import ir as I
from tvm.script import tirx as T


def check_json_roundtrip(node):
    json_str = tvm.ir.save_json(node)
    back = tvm.ir.load_json(json_str)
    tvm.ir.assert_structural_equal(back, node, map_free_vars=True)


def test_missing_type():
    missing = tvm.ir.Type.missing()

    assert isinstance(missing, tvm.ir.Type)
    assert isinstance(missing, tvm.ir.MissingType)


def test_prim_type():
    x = tvm.ir.PrimType("int32")
    assert isinstance(x, tvm.ir.PrimType)
    assert x.dtype == "int32"
    with pytest.raises(TypeError, match="PointerType::VoidPointerTy"):
        tvm.ir.PrimType("handle")


def test_func_type():
    arg_types = tvm.runtime.convert([])
    ret_type = tvm.ir.PrimType("float32")
    tf = tvm.ir.FuncType(arg_types, ret_type)
    assert tf.arg_types == arg_types
    assert tf.ret_type == ret_type
    assert tf.span is None
    # TODO make sure we can set span
    str(tf)
    check_json_roundtrip(tf)


def test_tuple_type():
    tf = tvm.ir.FuncType([], tvm.ir.TupleType([]))
    tt = tvm.ir.PrimType("float32")
    fields = tvm.runtime.convert([tf, tt])

    tup_ty = tvm.ir.TupleType(fields)
    assert tup_ty.fields == fields
    str(tup_ty)
    check_json_roundtrip(tup_ty)


def test_lambda_expr():
    calls = []

    def body(*values):
        calls.append(values)
        return (values[0] + values[1],)

    value = ir.LambdaExpr([T.f32, "float32"], body, ret_type=T.Tuple(T.f32))
    assert isinstance(value, ir.StagingExpr)
    assert len(calls) == 1
    ir.assert_structural_equal(
        value.ty,
        ir.FuncType([ir.PrimType("float32")] * 2, ir.TupleType([ir.PrimType("float32")])),
    )
    restored = eval(value.script(), {"I": I, "T": T})  # pylint: disable=eval-used
    ir.assert_structural_equal(value, restored)
    call = ir.Call("tirx.call_extern", [ir.StringImm("consume"), value], ty="int32")
    func = tvm.tirx.PrimFunc([], tvm.tirx.Evaluate(call))
    ir.assert_structural_equal(
        func, tvm.script.from_source(func.script(), extra_vars={"I": I, "T": T})
    )
    check_json_roundtrip(value)


def test_lambda_expr_apply():
    pair = ir.LambdaExpr(["int32", "int32"], lambda x, y: (x, y))
    swapped = pair.apply([pair.vars[1], pair.vars[0]])
    assert swapped[0].same_as(pair.vars[1]) and swapped[1].same_as(pair.vars[0])

    capture = ir.Var("capture", "int32")
    outer = ir.LambdaExpr(["int32"], lambda x: ir.LambdaExpr(["int32"], lambda y: (x, y, capture)))
    result = outer.apply([11]).apply([22])
    assert result[0].value == 11 and result[1].value == 22
    assert result[2].same_as(capture)
    identity = ir.LambdaExpr(["int32"], lambda x, unused=None: x)
    assert identity.apply([capture]).same_as(capture)


if __name__ == "__main__":
    test_tensor_type_bad_constructor()
    test_func_type()
    test_tuple_type()
