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

import pytest
import tvm_ffi

import tvm
import tvm_tirx
from tvm import te, topi
from tvm_tirx.ir import TensorRegion
from tvm_tirx.ir.prim import expr_deep_equal
from tvm_tirx.script import tirx as T


def _is_tensor_binding(node, *op_names):
    return (
        isinstance(node, tvm_tirx.ir.Bind)
        and isinstance(node.value, tvm_tirx.ir.Call)
        and isinstance(node.value.op, tvm_tirx.ir.Op)
        and node.value.op.name in op_names
    )


def test_expr_constructor():
    x = tvm_tirx.tirx.Var(name="xx", ty="float32")
    assert isinstance(x, tvm_tirx.tirx.Var)
    assert x.name == "xx"

    lhs, rhs = tvm_tirx.tirx.Var("lhs", "int32"), tvm_tirx.tirx.Var("rhs", "int32")
    combiner = tvm.te.CommReducer([lhs], [rhs], [lhs + rhs], [tvm_tirx.tirx.const(0, "int32")])
    with pytest.raises(TypeError, match="te.CommReducer"):
        tvm.te.Reduce(None, [1], [tvm.s_tir.IterVar((0, 1), "x", 2)], None, 0)
    x = tvm.te.Reduce(combiner, [1], [tvm.s_tir.IterVar((0, 1), "x", 2)], None, 0)
    assert isinstance(x, tvm.te.Reduce)
    assert x.combiner.same_as(combiner)
    assert x.value_index == 0

    x = tvm_tirx.ir.FloatImm("float32", 1.0)
    assert isinstance(x, tvm_tirx.ir.FloatImm)
    assert x.value == 1.0
    assert x.ty == tvm_tirx.ir.PrimType("float32")

    x = tvm_tirx.ir.IntImm("int64", 2)
    assert isinstance(x, tvm_tirx.ir.IntImm)
    assert x.value == 2
    assert x.ty == tvm_tirx.ir.PrimType("int64")

    x = tvm_tirx.ir.StringImm("xyza")
    assert isinstance(x, tvm_tirx.ir.StringImm)
    assert x.value == "xyza"

    x = tvm_tirx.tirx.Cast("float32", tvm_tirx.ir.IntImm("uint32", 1))
    assert isinstance(x, tvm_tirx.tirx.Cast)
    assert x.ty == tvm_tirx.ir.PrimType("float32")
    assert x.value.value == 1

    a = tvm_tirx.tirx.const(1.0, dtype="float32")
    b = tvm_tirx.tirx.Var("x", "float32")

    for cls in [
        tvm_tirx.tirx.Add,
        tvm_tirx.tirx.Sub,
        tvm_tirx.tirx.Mul,
        tvm_tirx.tirx.Div,
        tvm_tirx.tirx.Mod,
        tvm_tirx.tirx.Min,
        tvm_tirx.tirx.Max,
        tvm_tirx.tirx.LT,
        tvm_tirx.tirx.LE,
        tvm_tirx.tirx.GT,
        tvm_tirx.tirx.GE,
    ]:
        x = cls(a, b)
        assert isinstance(x, cls)
        assert x.a == a
        assert x.b.same_as(b)

    a = tvm_ffi.convert(tvm_tirx.tirx.Var("x", "int32") > 1)
    b = tvm_ffi.convert(tvm_tirx.tirx.Var("x", "int32") == 1)

    for cls in [tvm_tirx.tirx.And, tvm_tirx.tirx.Or]:
        x = cls(a, b)
        assert isinstance(x, cls)
        assert x.a == a
        assert x.b.same_as(b)

    x = tvm_tirx.tirx.Not(a)
    assert isinstance(x, tvm_tirx.tirx.Not)
    assert x.a == a

    x = tvm_tirx.tirx.Select(a, a, b)
    assert isinstance(x, tvm_tirx.tirx.Select)
    assert x.true_value == a
    assert x.false_value == b
    assert x.condition == a

    tensor_var = tvm_tirx.tirx.Var("buf", tvm_tirx.ir.PtrType(tvm_tirx.ir.PrimType("float32")))
    tensor = tvm_tirx.tirx.decl_tensor([16], "float32", data=tensor_var)
    x = tvm_tirx.ir.TensorLoad(tensor, [1])
    assert isinstance(x, tvm_tirx.ir.TensorLoad)
    assert x.ty == tvm_tirx.ir.PrimType("float32")
    assert x.source == tensor
    assert x.source.data.args[0].same_as(tensor)
    assert x.source.data.ty == tensor.data.ty
    assert list(x.indices) == [1]

    x = tvm_tirx.ir.prim.vector_expr.Ramp(1, 2, 10)
    assert isinstance(x, tvm_tirx.ir.prim.vector_expr.Ramp)
    assert x.base.value == 1
    assert x.stride.value == 2
    assert x.lanes == 10

    x = tvm_tirx.ir.prim.vector_expr.Broadcast(a, 10)
    assert isinstance(x, tvm_tirx.ir.prim.vector_expr.Broadcast)
    assert x.value == a
    assert x.lanes == 10

    x = tvm_tirx.ir.prim.vector_expr.Shuffle([a], [0])
    assert isinstance(x, tvm_tirx.ir.prim.vector_expr.Shuffle)
    assert x.vectors[0] == a
    assert x.indices[0].value == 0

    x = tvm_tirx.ir.Call(
        "tirx.call_extern",
        [tvm_tirx.ir.StringImm("xyz"), a],
        ty="float32",
    )
    assert isinstance(x, tvm_tirx.ir.Call)
    assert tvm_tirx.ir.is_prim_expr(x)
    assert x.ty == tvm_tirx.ir.PrimType("float32")
    assert x.op.name == "tirx.call_extern"
    assert x.args[1] == a
    assert x.attrs is None

    attr_arg = tvm_tirx.tirx.Var("attr_arg", "float32")
    x_with_attrs = tvm_tirx.ir.Call(
        "tirx.call_extern",
        [tvm_tirx.ir.StringImm("xyz"), attr_arg],
        attrs={"disable_tma": True},
        ty="float32",
    )
    assert x_with_attrs.attrs["disable_tma"] is True
    assert not tvm_ffi.structural_equal(x, x_with_attrs)
    script = tvm_tirx.ir.Evaluate(x_with_attrs).script()
    assert "attrs" in script
    assert "disable_tma" in script
    func = tvm_tirx.tirx.Function([attr_arg], tvm_tirx.ir.Evaluate(x_with_attrs))
    assert (
        tvm_tirx.script.from_source(
            func.script(), extra_vars={"I": tvm_tirx.script.ir, "T": tvm_tirx.script.tirx}
        ).script()
        == func.script()
    )

    y = tvm_tirx.tirx.Var("y", "float32")
    mutated = tvm_ffi.structural_map(
        x_with_attrs,
        (tvm_tirx.tirx.Var, lambda var: y if var.same_as(attr_arg) else var),
        order="post",
    )
    assert mutated.attrs["disable_tma"] is True
    assert mutated.args[1].same_as(y)

    x_from_intrin = tvm_tirx.tirx.call_intrin(
        "tirx.call_extern", tvm_tirx.ir.StringImm("xyz"), attrs={"disable_tma": True}, ty="float32"
    )
    assert x_from_intrin.attrs["disable_tma"] is True
    x_with_other_attrs = tvm_tirx.ir.Call(
        "tirx.call_extern",
        [tvm_tirx.ir.StringImm("xyz"), attr_arg],
        attrs={"disable_tma": False},
        ty="float32",
    )
    assert not expr_deep_equal(x_with_attrs, x_with_other_attrs)

    tuple_arg = tvm_tirx.ir.Tuple([attr_arg, tvm_tirx.ir.IntImm("int32", 1)])
    same_tuple_arg = tvm_tirx.ir.Tuple([attr_arg, tvm_tirx.ir.IntImm("int32", 1)])
    different_tuple_arg = tvm_tirx.ir.Tuple([attr_arg, tvm_tirx.ir.IntImm("int32", 2)])

    def call_with(arg):
        return tvm_tirx.ir.Call(
            "tirx.call_extern",
            [tvm_tirx.ir.StringImm("tuple_arg"), arg],
            ty="int32",
        )

    assert expr_deep_equal(call_with(tuple_arg), call_with(same_tuple_arg))
    assert not expr_deep_equal(call_with(tuple_arg), call_with(different_tuple_arg))
    assert not expr_deep_equal(
        call_with(tuple_arg),
        call_with(tvm_tirx.ir.TupleGetItem(same_tuple_arg, 0)),
    )

    cond0 = tvm_tirx.tirx.Var("cond0", "bool")
    cond1 = tvm_tirx.tirx.Var("cond1", "bool")
    inner_if = tvm_tirx.ir.Call(
        "prim.if_then_else",
        [cond1, tvm_tirx.ir.IntImm("int32", 1), tvm_tirx.ir.IntImm("int32", 0)],
        ty="int32",
    )
    outer_if = tvm_tirx.ir.Call(
        "prim.if_then_else",
        [cond0, inner_if, tvm_tirx.ir.IntImm("int32", 0)],
        attrs={"keep": True},
        ty="int32",
    )
    simplified = (
        tvm_tirx.tirx.transform.StmtSimplify()(
            tvm_tirx.IRModule({"main": tvm_tirx.tirx.Function([], tvm_tirx.ir.Evaluate(outer_if))})
        )["main"]
        .body[0]
        .value
    )
    assert simplified.attrs["keep"] is True

    v = tvm_tirx.tirx.Var("aa", "int32")
    x = tvm_tirx.tirx.Let(v, 1, v)
    assert x.var == v
    assert x.value.value == 1
    assert x.body == v


def test_tensor_region_call_wrappers_reject():
    tensor = tvm_tirx.tirx.decl_tensor([4], "int32")
    region = tensor[0:4]
    calls = [
        lambda: tvm_tirx.tirx.call_intrin("tirx.reinterpret", region, ty="int32"),
        lambda: tvm_tirx.tirx.call_extern("consume", region, ty="int32"),
        lambda: tvm_tirx.tirx.call_pure_extern("consume", region, ty="int32"),
        lambda: tvm_tirx.tirx.call_packed("consume", region),
        lambda: tvm_tirx.tirx.call_cpacked("consume", region, 0),
        lambda: tvm_tirx.tirx.call_packed_lowered("consume", region),
        lambda: tvm_tirx.tirx.call_cpacked_lowered("consume", region, 0),
        lambda: T.evaluate(region),
    ]
    for call in calls:
        with pytest.raises(TypeError, match="construct a TensorLoad with explicit indices"):
            call()

    assert not hasattr(region, "to_tensor_load")


def test_tensor_region_type_is_singleton():
    lhs = tvm_tirx.tirx.decl_tensor([1], "int32")[0:1]
    rhs = tvm_tirx.tirx.decl_tensor([2], "float32")[0:2]
    assert isinstance(lhs, TensorRegion)
    assert isinstance(rhs, TensorRegion)
    assert lhs.ty.same_as(rhs.ty)


def test_tensor_region_is_not_arithmetic_operand():
    int_region = tvm_tirx.tirx.decl_tensor([4], "int32")[0:4]
    with pytest.raises(TypeError, match="construct a TensorLoad explicitly"):
        tvm.s_tir.IterVar((0, 4), "i", tvm.s_tir.IterVar.DataPar) + int_region


def test_operator_base_categories_have_primitive_type():
    var = tvm_tirx.tirx.Var("x", "int32")
    tensor = tvm_tirx.tirx.decl_tensor([4], "float32")
    lhs, rhs = tvm_tirx.tirx.Var("lhs", "int32"), tvm_tirx.tirx.Var("rhs", "int32")
    combiner = tvm.te.CommReducer([lhs], [rhs], [lhs + rhs], [tvm_tirx.tirx.const(0, "int32")])
    expressions = [
        tvm_tirx.ir.IntImm("int32", 1),
        tvm_tirx.tirx.Add(var, 1),
        tvm_tirx.tirx.LT(var, 1),
        tvm_tirx.tirx.And(var < 1, var < 2),
        tvm.te.Reduce(
            combiner,
            [1],
            [tvm.s_tir.IterVar((0, 1), "i", tvm.s_tir.IterVar.CommReduce)],
            None,
            0,
        ),
        tvm_tirx.tirx.Cast("float32", var),
        tvm_tirx.tirx.Select(var < 1, var, 1),
        tvm_tirx.ir.TensorLoad(tensor, [0]),
        tvm_tirx.ir.prim.vector_expr.Ramp(0, 1, 4),
        tvm_tirx.ir.prim.vector_expr.Broadcast(var, 4),
        tvm_tirx.ir.prim.vector_expr.Shuffle([tvm_tirx.ir.prim.vector_expr.Broadcast(var, 2)], [0]),
        tvm_tirx.tirx.Let(var, 1, var),
    ]

    for expression in expressions:
        assert isinstance(expression, tvm_tirx.ir.ExprWithOp)
        assert isinstance(expression.ty, tvm_tirx.ir.PrimType)


def test_stmt_constructor():
    v = tvm_tirx.tirx.Var("aa", "int32")
    nop = tvm_tirx.ir.Evaluate(1)
    x = tvm_tirx.ir.Bind(v, 1)
    assert isinstance(x, tvm_tirx.ir.Bind)
    assert x.var == v
    assert x.value.value == 1

    x = tvm_tirx.ir.AssertStmt(
        tvm_tirx.tirx.const(1, "bool"),
        tvm_tirx.ir.StringImm("RuntimeError"),
        [tvm_tirx.ir.StringImm("hellow")],
    )
    assert isinstance(x, tvm_tirx.ir.AssertStmt)
    assert x.error_kind.value == "RuntimeError"
    assert len(x.message_parts) == 1
    assert x.message_parts[0].value == "hellow"

    x = tvm_tirx.ir.For(tvm_tirx.tirx.Var("x", "int32"), 0, 10, tvm_tirx.ir.ForKind.DEFAULT, nop)
    assert isinstance(x, tvm_tirx.ir.For)
    assert x.min.value == 0
    assert x.extent.value == 10
    assert len(x.body) == 1 and x.body[0] == nop

    tensor_var = tvm_tirx.tirx.Var("buf", tvm_tirx.ir.PtrType(tvm_tirx.ir.PrimType("bool")))
    tensor = tvm_tirx.tirx.decl_tensor([16], "bool", data=tensor_var)
    x = tvm_tirx.ir.TensorStore(tensor, [10], tvm_tirx.ir.IntImm("bool", 1))
    assert isinstance(x, tvm_tirx.ir.TensorStore)
    assert x.dest == tensor
    assert x.dest.data.args[0].same_as(tensor)
    assert x.dest.data.ty == tensor.data.ty
    assert list(x.indices) == [10]
    assert x.value.value == 1

    buf = tvm_tirx.tirx.decl_tensor([10], "float32")
    x = tvm_tirx.ir.Bind(
        buf,
        tvm_tirx.ir.Call(
            "tirx.alloc_tensor",
            [],
            attrs=tvm_tirx.ir.DictAttrs({}),
            ty_args=[buf.ty],
        ),
    )
    assert _is_tensor_binding(x, "tirx.alloc_tensor")
    assert x.var == buf

    x = tvm_tirx.ir.If(tvm_tirx.tirx.const(1, "bool"), tvm_tirx.ir.Evaluate(11), nop)
    assert isinstance(x, tvm_tirx.ir.If)
    assert x.then_case[0].value.value == 11
    assert x.else_case is not None and len(x.else_case) == 1 and x.else_case[0] == nop


def test_float_constructor_requires_float_dtype():
    # FloatImm dtype validation raises a builtin ValueError.
    with pytest.raises(ValueError):
        tvm_tirx.ir.FloatImm("int32", 1.0)


def test_math_unary_constructor_requires_float_dtype():
    x = tvm_tirx.tirx.Var("x", "int32")

    with pytest.raises(TypeError, match=r"prim\.tan only supports floating-point inputs"):
        tvm_tirx.tirx.tan(x)

    with pytest.raises(TypeError, match=r"prim\.sin only supports floating-point inputs"):
        tvm_tirx.tirx.sin(x)

    y = tvm_tirx.tirx.Var("y", "float32")
    assert tvm_tirx.tirx.tan(y).ty == tvm_tirx.ir.PrimType("float32")


def test_topi_tan_requires_float_dtype():
    x = te.placeholder((2, 2), dtype="int32", name="x")

    with pytest.raises(TypeError, match=r"prim\.tan only supports floating-point inputs"):
        topi.tan(x)


def test_math_unary_constructor_preserves_bfloat16():
    x = tvm_tirx.tirx.Var("x", "bfloat16")
    y = tvm_tirx.tirx.exp(x)
    assert y.ty == tvm_tirx.ir.PrimType("bfloat16")


def test_function_declaration_optional_body_roundtrip():
    declaration = tvm_tirx.tirx.Function([], None)
    assert declaration.body is None
    assert len(declaration.with_body(tvm_tirx.ir.Evaluate(0)).body) == 0
    restored = tvm_tirx.ir.load_json(tvm_tirx.ir.save_json(declaration))
    tvm_tirx.ir.assert_structural_equal(declaration, restored, map_free_vars=True)
    assert "def" in declaration.script()


if __name__ == "__main__":
    tvm_tirx.testing.main()
