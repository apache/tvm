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

import tvm
from tvm.ir import _overload_prim_expr, _tensor_expr_overload


class WrappedOperand(tvm.ir.ExprOperand):
    __slots__ = ("value",)

    def __init__(self, value):
        self.value = value

    def _operand(self):
        return self.value


def test_expr_operand_is_slot_only_and_hashable():
    operand = WrappedOperand(tvm.tirx.IntImm("int32", 1))

    assert tvm.ir.ExprOperand.__slots__ == ()
    assert not hasattr(operand, "__dict__")
    assert isinstance(hash(operand), int)


def test_expr_operand_realizes_binary_operands_at_late_dispatch(monkeypatch):
    lhs = tvm.tirx.IntImm("int32", 1)
    rhs = tvm.tirx.IntImm("int32", 2)
    result = object()
    dispatched = []

    def add(actual_lhs, actual_rhs):
        dispatched.append((actual_lhs, actual_rhs))
        return result

    monkeypatch.setattr(_overload_prim_expr, "__add__", add)

    assert WrappedOperand(lhs) + WrappedOperand(rhs) is result
    assert len(dispatched) == 1
    assert dispatched[0][0].same_as(lhs)
    assert dispatched[0][1].same_as(rhs)


def test_expr_operand_realizes_call_receiver_and_arguments(monkeypatch):
    value_ty = tvm.ir.PrimType("int32")
    func = tvm.ir.Var("func", tvm.ir.FuncType([value_ty], value_ty))
    arg = tvm.ir.Var("arg", value_ty)
    attrs = {"key": "value"}
    result = object()
    dispatched = []

    def call(actual_func, *actual_args, attrs=None):
        dispatched.append((actual_func, actual_args, attrs))
        return result

    monkeypatch.setattr(_tensor_expr_overload, "__call__", call)

    assert WrappedOperand(func)(WrappedOperand(arg), attrs=attrs) is result
    assert len(dispatched) == 1
    assert dispatched[0][0].same_as(func)
    assert dispatched[0][1][0].same_as(arg)
    assert dispatched[0][2] == attrs


def test_public_operator_base_is_the_only_dialect_operator_surface():
    from tvm import relax
    from tvm.ir import expr as ir_expr
    from tvm.relax import expr as relax_expr
    from tvm.tirx import expr as tirx_expr

    assert tvm.ir.ExprWithOp.__bases__ == (
        tvm.ir.ExprOperand,
        tvm.ir.Expr,
        tvm.runtime.Scriptable,
    )
    assert tvm.ir.BaseFunc.__bases__ == (tvm.ir.ExprWithOp,)

    tirx_direct_bases = (
        tirx_expr.ConstExpr,
        tirx_expr.BinaryOpExpr,
        tirx_expr.CmpExpr,
        tirx_expr.LogicalExpr,
        tirx_expr.Reduce,
        tirx_expr.Cast,
        tirx_expr.Select,
        tirx_expr.BufferLoad,
        tirx_expr.Ramp,
        tirx_expr.Broadcast,
        tirx_expr.Shuffle,
        tirx_expr.Let,
    )
    relax_direct_bases = (
        relax.If,
        relax.ShapeExpr,
        relax.Constant,
        relax.SeqExpr,
    )

    assert all(cls.__bases__ == (tvm.ir.ExprWithOp,) for cls in tirx_direct_bases)
    assert all(cls.__bases__ == (tvm.ir.ExprWithOp,) for cls in relax_direct_bases)
    assert relax.ExternFunc.__bases__ == (tvm.ir.BaseFunc,)
    assert relax.ExternFunc.__mro__.count(tvm.ir.ExprWithOp) == 1
    assert relax.Function.__mro__.count(tvm.ir.ExprWithOp) == 1
    assert tvm.tirx.PrimFunc.__mro__.count(tvm.ir.ExprWithOp) == 1
    assert not hasattr(ir_expr, "_ExprWithOp")
    assert not hasattr(tirx_expr, "ExprWithOp")
    assert not hasattr(relax_expr, "ExprWithOp")


def test_unsupported_expression_type_has_clear_operator_errors():
    value = tvm.ir.Var("value", tvm.ir.PointerType(tvm.ir.PrimType("float32")))
    message = "Operator overloading is not supported for expression type"

    for operation in (
        lambda: value + 1,
        lambda: -value,
        lambda: value << 1,
        lambda: value.equal(value),
        lambda: value.astype("int32"),
    ):
        with pytest.raises(TypeError, match=message):
            operation()

    with pytest.raises(TypeError, match="cannot be called"):
        value()


if __name__ == "__main__":
    tvm.testing.main()
