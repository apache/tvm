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
import tvm_ffi

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

    def body(x, y):
        calls.append((x, y))
        return (x + y,)

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
    with pytest.raises(TypeError, match="[Mm]issing|type"):
        var = ir.Var("untyped", ir.TupleType([ir.Type.missing()]))
        tvm_ffi.get_global_func("ir.LambdaExpr")([var], var)


def test_lambda_expr_apply():
    make = tvm_ffi.get_global_func("ir.LambdaExpr")
    x, y = ir.Var("x", "int32"), ir.Var("y", "int32")
    swapped = make([x, y], ir.Tuple([x, y])).apply([y, x])
    assert swapped[0].same_as(y) and swapped[1].same_as(x)

    inner = make([x], make([y], ir.Tuple([x, y]))).apply([y])
    assert not inner.vars[0].same_as(y)
    assert inner.body[0].same_as(y)
    assert inner.body[1].same_as(inner.vars[0])
    assert inner.apply([22])[1].value == 22
    shadowed = make([x], make([x], x)).apply([11])
    assert shadowed.apply([22]).value == 22
    scoped = make([x], make([y], ir.Tuple([make([y], y), y]))).apply([11])
    assert scoped.body[1].same_as(scoped.vars[0])
    assert scoped.body[0].body.same_as(scoped.body[0].vars[0])

    applied = make([x], ir.prim.Let(y, x, x + y)).apply([y])
    assert not applied.var.same_as(y)
    assert applied.value.same_as(y) and applied.body.a.same_as(y)
    assert applied.body.b.same_as(applied.var)


def test_lambda_expr_binding_scopes():
    make = tvm_ffi.get_global_func("ir.LambdaExpr")
    x = ir.Var("x", "int32")
    y = tvm.relax.DataflowVar("y", ir.PrimType("int32"))
    scope = tvm.relax.SeqExpr(
        [tvm.relax.DataflowBlock([tvm.relax.VarBinding(y, tvm.runtime.const(1))])],
        ir.Tuple([x, y]),
    )
    applied = make([x], ir.Tuple([scope, scope, y])).apply([y])
    first, second = [applied[i].blocks[0].bindings[0].var for i in range(2)]
    assert isinstance(first, tvm.relax.DataflowVar)
    assert not first.same_as(y) and not second.same_as(y) and not first.same_as(second)
    assert applied[0].body[0].same_as(y) and applied[1].body[0].same_as(y)
    assert applied[0].body[1].same_as(first) and applied[1].body[1].same_as(second)
    assert applied[2].same_as(y)


def test_lambda_expr_dependent_types():
    value = ir.LambdaExpr(
        ["int64"], lambda n: ir.LambdaExpr([tvm.relax.TensorType([n], "float32")], lambda x: x)
    )
    renamed = ir.LambdaExpr(
        ["int64"], lambda m: ir.LambdaExpr([tvm.relax.TensorType([m], "float32")], lambda y: y)
    )
    ir.assert_structural_equal(value, renamed)
    assert tvm_ffi.structural_hash(value) == tvm_ffi.structural_hash(renamed)
    decoded = ir.load_json(ir.save_json(value))
    ir.assert_structural_equal(value, decoded)
    assert decoded.body.vars[0].ty.shape[0].same_as(decoded.vars[0])
    assert decoded.body.body.same_as(decoded.body.vars[0])
    argument = ir.Var("arg", tvm.relax.TensorType([4], "float32"))
    applied = decoded.apply([tvm.runtime.const(4, "int64")])
    ir.assert_structural_equal(applied.ty, ir.FuncType([argument.ty], argument.ty))
    assert applied.apply([argument]).same_as(argument)
    make = tvm_ffi.get_global_func("ir.LambdaExpr")
    dependent = make([value.vars[0], value.body.vars[0]], value.body.body)
    assert dependent.apply([tvm.runtime.const(4, "int64"), argument]).same_as(argument)
    assert not tvm_ffi.structural_equal(value.body, renamed.body)


def test_lambda_expr_mutation():
    value = ir.LambdaExpr(["int32"], lambda x: x)
    new_type = ir.PrimType("int64")
    rewritten = tvm_ffi.structural_map(
        value._move(),
        with_def_region_kind=(
            ir.Var,
            lambda var, kind: ir.Var(var.name, new_type) if kind else var,
        ),
    )
    assert rewritten.body.same_as(rewritten.vars[0])
    ir.assert_structural_equal(rewritten.ty, ir.FuncType([new_type], new_type))


if __name__ == "__main__":
    test_tensor_type_bad_constructor()
    test_func_type()
    test_tuple_type()
