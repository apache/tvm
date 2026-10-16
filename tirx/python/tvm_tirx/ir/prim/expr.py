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
"""Primitive expression nodes shared by TVM IR dialects."""

import tvm_ffi

from tvm_tirx.ir.location import UNKNOWN_LOC, Location

from ..expr import Expr, ExprWithOp, Var
from ..expr import IntImm as _IntImm
from ..type import PrimType
from . import _ffi_api as _prim_ffi_api


class BinaryOpExpr(ExprWithOp):
    a: Expr
    b: Expr


class CmpExpr(ExprWithOp):
    a: Expr
    b: Expr


class LogicalExpr(ExprWithOp):
    pass


@tvm_ffi.register_object("prim.Cast")
class Cast(ExprWithOp):
    """Cast expression.

    Parameters
    ----------
    value : Expr
        The value to cast.

    dtype : str | PrimType
        The target data type.

    loc : Location, optional
        The location of this expression in the source code.
    """

    value: Expr

    def __init__(self, value, dtype: str | PrimType, loc: Location = UNKNOWN_LOC) -> None:
        if isinstance(dtype, PrimType):
            dtype = dtype.dtype
        self.__init_handle_by_constructor__(_prim_ffi_api.Cast, value, dtype, loc)  # type: ignore


@tvm_ffi.register_object("prim.LShift")
class LShift(BinaryOpExpr):
    """Left shift node.

    Parameters
    ----------
    a : Expr
        The left hand operand.

    b : Expr
        The right hand operand.

    loc : Location, optional
        The location of this expression in the source code.
    """

    def __init__(self, a: Expr, b: Expr, loc: Location = UNKNOWN_LOC) -> None:
        self.__init_handle_by_constructor__(_prim_ffi_api.LShift, a, b, loc)  # type: ignore


@tvm_ffi.register_object("prim.RShift")
class RShift(BinaryOpExpr):
    """Right shift node.

    Parameters
    ----------
    a : Expr
        The left hand operand.

    b : Expr
        The right hand operand.

    loc : Location, optional
        The location of this expression in the source code.
    """

    def __init__(self, a: Expr, b: Expr, loc: Location = UNKNOWN_LOC) -> None:
        self.__init_handle_by_constructor__(_prim_ffi_api.RShift, a, b, loc)  # type: ignore


@tvm_ffi.register_object("prim.BitwiseAnd")
class BitwiseAnd(BinaryOpExpr):
    """Bitwise and node.

    Parameters
    ----------
    a : Expr
        The left hand operand.

    b : Expr
        The right hand operand.

    loc : Location, optional
        The location of this expression in the source code.
    """

    def __init__(self, a: Expr, b: Expr, loc: Location = UNKNOWN_LOC) -> None:
        self.__init_handle_by_constructor__(_prim_ffi_api.BitwiseAnd, a, b, loc)  # type: ignore


@tvm_ffi.register_object("prim.BitwiseOr")
class BitwiseOr(BinaryOpExpr):
    """Bitwise or node.

    Parameters
    ----------
    a : Expr
        The left hand operand.

    b : Expr
        The right hand operand.

    loc : Location, optional
        The location of this expression in the source code.
    """

    def __init__(self, a: Expr, b: Expr, loc: Location = UNKNOWN_LOC) -> None:
        self.__init_handle_by_constructor__(_prim_ffi_api.BitwiseOr, a, b, loc)  # type: ignore


@tvm_ffi.register_object("prim.BitwiseXor")
class BitwiseXor(BinaryOpExpr):
    """Bitwise xor node.

    Parameters
    ----------
    a : Expr
        The left hand operand.

    b : Expr
        The right hand operand.

    loc : Location, optional
        The location of this expression in the source code.
    """

    def __init__(self, a: Expr, b: Expr, loc: Location = UNKNOWN_LOC) -> None:
        self.__init_handle_by_constructor__(_prim_ffi_api.BitwiseXor, a, b, loc)  # type: ignore


@tvm_ffi.register_object("prim.BitwiseNot")
class BitwiseNot(ExprWithOp):
    """Bitwise not node.

    Parameters
    ----------
    a : Expr
        The input value.

    loc : Location, optional
        The location of this expression in the source code.
    """

    a: Expr

    def __init__(self, a: Expr, loc: Location = UNKNOWN_LOC) -> None:
        self.__init_handle_by_constructor__(_prim_ffi_api.BitwiseNot, a, loc)  # type: ignore


@tvm_ffi.register_object("prim.Add")
class Add(BinaryOpExpr):
    """Add node.

    Parameters
    ----------
    a : Expr
        The left hand operand.

    b : Expr
        The right hand operand.

    loc : Location, optional
        The location of this expression in the source code.
    """

    def __init__(self, a: Expr, b: Expr, loc: Location = UNKNOWN_LOC) -> None:
        self.__init_handle_by_constructor__(_prim_ffi_api.Add, a, b, loc)  # type: ignore


@tvm_ffi.register_object("prim.Sub")
class Sub(BinaryOpExpr):
    """Sub node.

    Parameters
    ----------
    a : Expr
        The left hand operand.

    b : Expr
        The right hand operand.

    loc : Location, optional
        The location of this expression in the source code.
    """

    def __init__(self, a: Expr, b: Expr, loc: Location = UNKNOWN_LOC) -> None:
        self.__init_handle_by_constructor__(_prim_ffi_api.Sub, a, b, loc)  # type: ignore


@tvm_ffi.register_object("prim.Mul")
class Mul(BinaryOpExpr):
    """Mul node.

    Parameters
    ----------
    a : Expr
        The left hand operand.

    b : Expr
        The right hand operand.

    loc : Location, optional
        The location of this expression in the source code.
    """

    def __init__(self, a: Expr, b: Expr, loc: Location = UNKNOWN_LOC) -> None:
        self.__init_handle_by_constructor__(_prim_ffi_api.Mul, a, b, loc)  # type: ignore


@tvm_ffi.register_object("prim.Div")
class Div(BinaryOpExpr):
    """Div node.

    Parameters
    ----------
    a : Expr
        The left hand operand.

    b : Expr
        The right hand operand.

    loc : Location, optional
        The location of this expression in the source code.
    """

    def __init__(self, a: Expr, b: Expr, loc: Location = UNKNOWN_LOC) -> None:
        self.__init_handle_by_constructor__(_prim_ffi_api.Div, a, b, loc)  # type: ignore


@tvm_ffi.register_object("prim.Mod")
class Mod(BinaryOpExpr):
    """Mod node.

    Parameters
    ----------
    a : Expr
        The left hand operand.

    b : Expr
        The right hand operand.

    loc : Location, optional
        The location of this expression in the source code.
    """

    def __init__(self, a: Expr, b: Expr, loc: Location = UNKNOWN_LOC) -> None:
        self.__init_handle_by_constructor__(_prim_ffi_api.Mod, a, b, loc)  # type: ignore


@tvm_ffi.register_object("prim.FloorDiv")
class FloorDiv(BinaryOpExpr):
    """FloorDiv node.

    Parameters
    ----------
    a : Expr
        The left hand operand.

    b : Expr
        The right hand operand.

    loc : Location, optional
        The location of this expression in the source code.
    """

    def __init__(self, a: Expr, b: Expr, loc: Location = UNKNOWN_LOC) -> None:
        self.__init_handle_by_constructor__(_prim_ffi_api.FloorDiv, a, b, loc)  # type: ignore


@tvm_ffi.register_object("prim.FloorMod")
class FloorMod(BinaryOpExpr):
    """FloorMod node.

    Parameters
    ----------
    a : Expr
        The left hand operand.

    b : Expr
        The right hand operand.

    loc : Location, optional
        The location of this expression in the source code.
    """

    def __init__(self, a: Expr, b: Expr, loc: Location = UNKNOWN_LOC) -> None:
        self.__init_handle_by_constructor__(_prim_ffi_api.FloorMod, a, b, loc)  # type: ignore


@tvm_ffi.register_object("prim.Min")
class Min(BinaryOpExpr):
    """Min node.

    Parameters
    ----------
    a : Expr
        The left hand operand.

    b : Expr
        The right hand operand.

    loc : Location, optional
        The location of this expression in the source code.
    """

    def __init__(self, a: Expr, b: Expr, loc: Location = UNKNOWN_LOC) -> None:
        self.__init_handle_by_constructor__(_prim_ffi_api.Min, a, b, loc)  # type: ignore


@tvm_ffi.register_object("prim.Max")
class Max(BinaryOpExpr):
    """Max node.

    Parameters
    ----------
    a : Expr
        The left hand operand.

    b : Expr
        The right hand operand.

    loc : Location, optional
        The location of this expression in the source code.
    """

    def __init__(self, a: Expr, b: Expr, loc: Location = UNKNOWN_LOC) -> None:
        self.__init_handle_by_constructor__(_prim_ffi_api.Max, a, b, loc)  # type: ignore


@tvm_ffi.register_object("prim.EQ")
class EQ(CmpExpr):
    """EQ node.

    Parameters
    ----------
    a : Expr
        The left hand operand.

    b : Expr
        The right hand operand.

    loc : Location, optional
        The location of this expression in the source code.
    """

    def __init__(self, a: Expr, b: Expr, loc: Location = UNKNOWN_LOC) -> None:
        self.__init_handle_by_constructor__(_prim_ffi_api.EQ, a, b, loc)  # type: ignore


@tvm_ffi.register_object("prim.NE")
class NE(CmpExpr):
    """NE node.

    Parameters
    ----------
    a : Expr
        The left hand operand.

    b : Expr
        The right hand operand.

    loc : Location, optional
        The location of this expression in the source code.
    """

    def __init__(self, a: Expr, b: Expr, loc: Location = UNKNOWN_LOC) -> None:
        self.__init_handle_by_constructor__(_prim_ffi_api.NE, a, b, loc)  # type: ignore


@tvm_ffi.register_object("prim.LT")
class LT(CmpExpr):
    """LT node.

    Parameters
    ----------
    a : Expr
        The left hand operand.

    b : Expr
        The right hand operand.

    loc : Location, optional
        The location of this expression in the source code.
    """

    def __init__(self, a: Expr, b: Expr, loc: Location = UNKNOWN_LOC) -> None:
        self.__init_handle_by_constructor__(_prim_ffi_api.LT, a, b, loc)  # type: ignore


@tvm_ffi.register_object("prim.LE")
class LE(CmpExpr):
    """LE node.

    Parameters
    ----------
    a : Expr
        The left hand operand.

    b : Expr
        The right hand operand.

    loc : Location, optional
        The location of this expression in the source code.
    """

    def __init__(self, a: Expr, b: Expr, loc: Location = UNKNOWN_LOC) -> None:
        self.__init_handle_by_constructor__(_prim_ffi_api.LE, a, b, loc)  # type: ignore


@tvm_ffi.register_object("prim.GT")
class GT(CmpExpr):
    """GT node.

    Parameters
    ----------
    a : Expr
        The left hand operand.

    b : Expr
        The right hand operand.

    loc : Location, optional
        The location of this expression in the source code.
    """

    def __init__(self, a: Expr, b: Expr, loc: Location = UNKNOWN_LOC) -> None:
        self.__init_handle_by_constructor__(_prim_ffi_api.GT, a, b, loc)  # type: ignore


@tvm_ffi.register_object("prim.GE")
class GE(CmpExpr):
    """GE node.

    Parameters
    ----------
    a : Expr
        The left hand operand.

    b : Expr
        The right hand operand.

    loc : Location, optional
        The location of this expression in the source code.
    """

    def __init__(self, a: Expr, b: Expr, loc: Location = UNKNOWN_LOC) -> None:
        self.__init_handle_by_constructor__(_prim_ffi_api.GE, a, b, loc)  # type: ignore


@tvm_ffi.register_object("prim.And")
class And(LogicalExpr):
    """And node.

    Parameters
    ----------
    a : Expr
        The left hand operand.

    b : Expr
        The right hand operand.

    loc : Location, optional
        The location of this expression in the source code.
    """

    def __init__(self, a: Expr, b: Expr, loc: Location = UNKNOWN_LOC) -> None:
        self.__init_handle_by_constructor__(_prim_ffi_api.And, a, b, loc)  # type: ignore


@tvm_ffi.register_object("prim.Or")
class Or(LogicalExpr):
    """Or node.

    Parameters
    ----------
    a : Expr
        The left hand operand.

    b : Expr
        The right hand operand.

    loc : Location, optional
        The location of this expression in the source code.
    """

    a: Expr
    b: Expr

    def __init__(self, a: Expr, b: Expr, loc: Location = UNKNOWN_LOC) -> None:
        self.__init_handle_by_constructor__(_prim_ffi_api.Or, a, b, loc)  # type: ignore


@tvm_ffi.register_object("prim.Not")
class Not(LogicalExpr):
    """Not node.

    Parameters
    ----------
    a : Expr
        The input value

    loc : Location, optional
        The location of this expression in the source code.
    """

    a: Expr

    def __init__(self, a: Expr, loc: Location = UNKNOWN_LOC) -> None:
        self.__init_handle_by_constructor__(_prim_ffi_api.Not, a, loc)  # type: ignore


@tvm_ffi.register_object("prim.Select")
class Select(ExprWithOp):
    """Select node.

    Note
    ----
    Select may compute both true_value and false_value.
    Use :py:func:`tvm_tirx.ir.prim.if_then_else` instead if you want to
    get a conditional expression that only evaluates
    the correct branch.

    Parameters
    ----------
    condition : Expr
        The condition expression.

    true_value : Expr
        The value to take when condition is true.

    false_value : Expr
        The value to take when condition is false.

    loc : Location, optional
        The location of this expression in the source code.
    """

    condition: Expr
    true_value: Expr
    false_value: Expr

    def __init__(
        self,
        condition: Expr,
        true_value: Expr,
        false_value: Expr,
        loc: Location = UNKNOWN_LOC,
    ) -> None:
        if isinstance(condition, bool):
            condition = _IntImm("bool", condition)
        self.__init_handle_by_constructor__(
            _prim_ffi_api.Select,
            condition,
            true_value,
            false_value,
            loc,  # type: ignore
        )


@tvm_ffi.register_object("prim.Let")
class Let(ExprWithOp):
    """Let node.

    Parameters
    ----------
    var : Var
        The variable in the binding.

    value : Expr
        The value in to be bound.

    body : Expr
        The body expression.

    loc : Location, optional
        The location of this expression in the source code.
    """

    var: Var
    value: Expr
    body: Expr

    def __init__(self, var: Var, value: Expr, body: Expr, loc: Location = UNKNOWN_LOC) -> None:
        self.__init_handle_by_constructor__(_prim_ffi_api.Let, var, value, body, loc)  # type: ignore
