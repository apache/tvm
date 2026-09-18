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

from ..base import Span
from ..expr import Expr, ExprWithOp, Var
from ..type import PrimType
from . import _ffi_api as _prim_ffi_api


class ConstExpr(ExprWithOp):
    pass


class BinaryOpExpr(ExprWithOp):
    a: Expr
    b: Expr


class CmpExpr(ExprWithOp):
    a: Expr
    b: Expr


class LogicalExpr(ExprWithOp):
    pass


@tvm_ffi.register_object("prim.FloatImm")
class FloatImm(ConstExpr):
    """Float constant.

    Parameters
    ----------
    dtype : str
        The data type

    value : float
        The constant value.

    span : Optional[Span]
        The location of this expression in the source code.
    """

    value: float

    def __init__(self, dtype: str | PrimType, value: float, span: Span | None = None) -> None:
        if isinstance(dtype, PrimType):
            dtype = dtype.dtype
        self.__init_handle_by_constructor__(
            _prim_ffi_api.FloatImm,
            dtype,
            value,
            span,  # type: ignore
        )

    def __float__(self) -> float:
        return self.value


@tvm_ffi.register_object("prim.IntImm")
class IntImm(ConstExpr):
    """Int constant.

    Parameters
    ----------
    dtype : str
        The data type

    value : int
        The constant value.

    span : Optional[Span]
        The location of this expression in the source code.
    """

    value: int

    def __init__(self, dtype: str | PrimType, value: int, span: Span | None = None) -> None:
        if isinstance(dtype, PrimType):
            dtype = dtype.dtype
        self.__init_handle_by_constructor__(
            _prim_ffi_api.IntImm,
            dtype,
            value,
            span,  # type: ignore
        )

    def __hash__(self) -> int:
        return self.value

    def __int__(self) -> int:
        return self.value

    def __nonzero__(self) -> bool:
        return self.value != 0

    def __eq__(self, other: Expr) -> Expr:
        return _prim_ffi_api._OpEQ(self, other, None)  # type: ignore

    def __ne__(self, other: Expr) -> Expr:
        return _prim_ffi_api._OpNE(self, other, None)  # type: ignore

    def __bool__(self) -> bool:
        return self.__nonzero__()


@tvm_ffi.register_object("prim.StringImm")  # type: ignore
class StringImm(ConstExpr):
    """String constant.

    Parameters
    ----------
    value : str
        The value of the function.

    span : Optional[Span]
        The location of this expression in the source code.
    """

    value: str

    def __init__(self, value: str, span: Span | None = None) -> None:
        self.__init_handle_by_constructor__(_prim_ffi_api.StringImm, value, span)  # type: ignore

    def __eq__(self, other: Expr) -> bool:
        if isinstance(other, ConstExpr):
            return self.value == other.value
        return self.value == other

    def __ne__(self, other: Expr) -> bool:
        if isinstance(other, ConstExpr):
            return self.value != other.value
        return self.value != other

    def __hash__(self) -> int:
        return Expr.__hash__(self)


@tvm_ffi.register_object("prim.Cast")
class Cast(ExprWithOp):
    """Cast expression.

    Parameters
    ----------
    dtype : str
        The data type

    value : Expr
        The value of the function.

    span : Optional[Span]
        The location of this expression in the source code.
    """

    value: Expr

    def __init__(self, dtype: str | PrimType, value, span: Span | None = None) -> None:
        if isinstance(dtype, PrimType):
            dtype = dtype.dtype
        self.__init_handle_by_constructor__(_prim_ffi_api.Cast, dtype, value, span)  # type: ignore


@tvm_ffi.register_object("prim.Add")
class Add(BinaryOpExpr):
    """Add node.

    Parameters
    ----------
    a : Expr
        The left hand operand.

    b : Expr
        The right hand operand.

    span : Optional[Span]
        The location of this expression in the source code.
    """

    def __init__(self, a: Expr, b: Expr, span: Span | None = None) -> None:
        self.__init_handle_by_constructor__(_prim_ffi_api.Add, a, b, span)  # type: ignore


@tvm_ffi.register_object("prim.Sub")
class Sub(BinaryOpExpr):
    """Sub node.

    Parameters
    ----------
    a : Expr
        The left hand operand.

    b : Expr
        The right hand operand.

    span : Optional[Span]
        The location of this expression in the source code.
    """

    def __init__(self, a: Expr, b: Expr, span: Span | None = None) -> None:
        self.__init_handle_by_constructor__(_prim_ffi_api.Sub, a, b, span)  # type: ignore


@tvm_ffi.register_object("prim.Mul")
class Mul(BinaryOpExpr):
    """Mul node.

    Parameters
    ----------
    a : Expr
        The left hand operand.

    b : Expr
        The right hand operand.

    span : Optional[Span]
        The location of this expression in the source code.
    """

    def __init__(self, a: Expr, b: Expr, span: Span | None = None) -> None:
        self.__init_handle_by_constructor__(_prim_ffi_api.Mul, a, b, span)  # type: ignore


@tvm_ffi.register_object("prim.Div")
class Div(BinaryOpExpr):
    """Div node.

    Parameters
    ----------
    a : Expr
        The left hand operand.

    b : Expr
        The right hand operand.

    span : Optional[Span]
        The location of this expression in the source code.
    """

    def __init__(self, a: Expr, b: Expr, span: Span | None = None) -> None:
        self.__init_handle_by_constructor__(_prim_ffi_api.Div, a, b, span)  # type: ignore


@tvm_ffi.register_object("prim.Mod")
class Mod(BinaryOpExpr):
    """Mod node.

    Parameters
    ----------
    a : Expr
        The left hand operand.

    b : Expr
        The right hand operand.

    span : Optional[Span]
        The location of this expression in the source code.
    """

    def __init__(self, a: Expr, b: Expr, span: Span | None = None) -> None:
        self.__init_handle_by_constructor__(_prim_ffi_api.Mod, a, b, span)  # type: ignore


@tvm_ffi.register_object("prim.FloorDiv")
class FloorDiv(BinaryOpExpr):
    """FloorDiv node.

    Parameters
    ----------
    a : Expr
        The left hand operand.

    b : Expr
        The right hand operand.

    span : Optional[Span]
        The location of this expression in the source code.
    """

    def __init__(self, a: Expr, b: Expr, span: Span | None = None) -> None:
        self.__init_handle_by_constructor__(_prim_ffi_api.FloorDiv, a, b, span)  # type: ignore


@tvm_ffi.register_object("prim.FloorMod")
class FloorMod(BinaryOpExpr):
    """FloorMod node.

    Parameters
    ----------
    a : Expr
        The left hand operand.

    b : Expr
        The right hand operand.

    span : Optional[Span]
        The location of this expression in the source code.
    """

    def __init__(self, a: Expr, b: Expr, span: Span | None = None) -> None:
        self.__init_handle_by_constructor__(_prim_ffi_api.FloorMod, a, b, span)  # type: ignore


@tvm_ffi.register_object("prim.Min")
class Min(BinaryOpExpr):
    """Min node.

    Parameters
    ----------
    a : Expr
        The left hand operand.

    b : Expr
        The right hand operand.

    span : Optional[Span]
        The location of this expression in the source code.
    """

    def __init__(self, a: Expr, b: Expr, span: Span | None = None) -> None:
        self.__init_handle_by_constructor__(_prim_ffi_api.Min, a, b, span)  # type: ignore


@tvm_ffi.register_object("prim.Max")
class Max(BinaryOpExpr):
    """Max node.

    Parameters
    ----------
    a : Expr
        The left hand operand.

    b : Expr
        The right hand operand.

    span : Optional[Span]
        The location of this expression in the source code.
    """

    def __init__(self, a: Expr, b: Expr, span: Span | None = None) -> None:
        self.__init_handle_by_constructor__(_prim_ffi_api.Max, a, b, span)  # type: ignore


@tvm_ffi.register_object("prim.EQ")
class EQ(CmpExpr):
    """EQ node.

    Parameters
    ----------
    a : Expr
        The left hand operand.

    b : Expr
        The right hand operand.

    span : Optional[Span]
        The location of this expression in the source code.
    """

    def __init__(self, a: Expr, b: Expr, span: Span | None = None) -> None:
        self.__init_handle_by_constructor__(_prim_ffi_api.EQ, a, b, span)  # type: ignore


@tvm_ffi.register_object("prim.NE")
class NE(CmpExpr):
    """NE node.

    Parameters
    ----------
    a : Expr
        The left hand operand.

    b : Expr
        The right hand operand.

    span : Optional[Span]
        The location of this expression in the source code.
    """

    def __init__(self, a: Expr, b: Expr, span: Span | None = None) -> None:
        self.__init_handle_by_constructor__(_prim_ffi_api.NE, a, b, span)  # type: ignore


@tvm_ffi.register_object("prim.LT")
class LT(CmpExpr):
    """LT node.

    Parameters
    ----------
    a : Expr
        The left hand operand.

    b : Expr
        The right hand operand.

    span : Optional[Span]
        The location of this expression in the source code.
    """

    def __init__(self, a: Expr, b: Expr, span: Span | None = None) -> None:
        self.__init_handle_by_constructor__(_prim_ffi_api.LT, a, b, span)  # type: ignore


@tvm_ffi.register_object("prim.LE")
class LE(CmpExpr):
    """LE node.

    Parameters
    ----------
    a : Expr
        The left hand operand.

    b : Expr
        The right hand operand.

    span : Optional[Span]
        The location of this expression in the source code.
    """

    def __init__(self, a: Expr, b: Expr, span: Span | None = None) -> None:
        self.__init_handle_by_constructor__(_prim_ffi_api.LE, a, b, span)  # type: ignore


@tvm_ffi.register_object("prim.GT")
class GT(CmpExpr):
    """GT node.

    Parameters
    ----------
    a : Expr
        The left hand operand.

    b : Expr
        The right hand operand.

    span : Optional[Span]
        The location of this expression in the source code.
    """

    def __init__(self, a: Expr, b: Expr, span: Span | None = None) -> None:
        self.__init_handle_by_constructor__(_prim_ffi_api.GT, a, b, span)  # type: ignore


@tvm_ffi.register_object("prim.GE")
class GE(CmpExpr):
    """GE node.

    Parameters
    ----------
    a : Expr
        The left hand operand.

    b : Expr
        The right hand operand.

    span : Optional[Span]
        The location of this expression in the source code.
    """

    def __init__(self, a: Expr, b: Expr, span: Span | None = None) -> None:
        self.__init_handle_by_constructor__(_prim_ffi_api.GE, a, b, span)  # type: ignore


@tvm_ffi.register_object("prim.And")
class And(LogicalExpr):
    """And node.

    Parameters
    ----------
    a : Expr
        The left hand operand.

    b : Expr
        The right hand operand.

    span : Optional[Span]
        The location of this expression in the source code.
    """

    def __init__(self, a: Expr, b: Expr, span: Span | None = None) -> None:
        self.__init_handle_by_constructor__(_prim_ffi_api.And, a, b, span)  # type: ignore


@tvm_ffi.register_object("prim.Or")
class Or(LogicalExpr):
    """Or node.

    Parameters
    ----------
    a : Expr
        The left hand operand.

    b : Expr
        The right hand operand.

    span : Optional[Span]
        The location of this expression in the source code.
    """

    a: Expr
    b: Expr

    def __init__(self, a: Expr, b: Expr, span: Span | None = None) -> None:
        self.__init_handle_by_constructor__(_prim_ffi_api.Or, a, b, span)  # type: ignore


@tvm_ffi.register_object("prim.Not")
class Not(LogicalExpr):
    """Not node.

    Parameters
    ----------
    a : Expr
        The input value

    span : Optional[Span]
        The location of this expression in the source code.
    """

    a: Expr

    def __init__(self, a: Expr, span: Span | None = None) -> None:
        self.__init_handle_by_constructor__(_prim_ffi_api.Not, a, span)  # type: ignore


@tvm_ffi.register_object("prim.Select")
class Select(ExprWithOp):
    """Select node.

    Note
    ----
    Select may compute both true_value and false_value.
    Use :py:class:`tvm.tirx.if_then_else` instead if you want to
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

    span : Optional[Span]
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
        span: Span | None = None,
    ) -> None:
        if isinstance(condition, bool):
            condition = IntImm("bool", condition)
        self.__init_handle_by_constructor__(
            _prim_ffi_api.Select,
            condition,
            true_value,
            false_value,
            span,  # type: ignore
        )


@tvm_ffi.register_object("prim.Ramp")
class Ramp(ExprWithOp):
    """Ramp node.

    Parameters
    ----------
    base : Expr
        The base expression.

    stride : Expr
        The stride of the ramp.

    lanes : Expr
        The lanes of the expression.

    span : Optional[Span]
        The location of this expression in the source code.
    """

    base: Expr
    stride: Expr
    lanes: Expr

    def __init__(self, base: Expr, stride: Expr, lanes: Expr, span: Span | None = None) -> None:
        self.__init_handle_by_constructor__(
            _prim_ffi_api.Ramp,
            base,
            stride,
            lanes,
            span,  # type: ignore
        )


@tvm_ffi.register_object("prim.Broadcast")
class Broadcast(ExprWithOp):
    """Broadcast node.

    Parameters
    ----------
    value : Expr
        The value of the expression.

    lanes : Expr
        The lanes of the expression.

    span : Optional[Span]
        The location of this expression in the source code.
    """

    value: Expr
    lanes: Expr

    def __init__(self, value: Expr, lanes: Expr, span: Span | None = None) -> None:
        self.__init_handle_by_constructor__(_prim_ffi_api.Broadcast, value, lanes, span)  # type: ignore


@tvm_ffi.register_object("prim.Shuffle")
class Shuffle(ExprWithOp):
    """Shuffle node.

    Parameters
    ----------
    vectors : List[Expr]
        The vectors

    indices : List[Expr]
        The indices

    span : Optional[Span]
        The location of this expression in the source code.
    """

    vectors: list[Expr]
    indices: list[Expr]

    def __init__(self, vectors: list[Expr], indices: list[Expr], span: Span | None = None) -> None:
        self.__init_handle_by_constructor__(
            _prim_ffi_api.Shuffle,
            vectors,
            indices,
            span,  # type: ignore
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

    span : Optional[Span]
        The location of this expression in the source code.
    """

    var: Var
    value: Expr
    body: Expr

    def __init__(self, var: Var, value: Expr, body: Expr, span: Span | None = None) -> None:
        self.__init_handle_by_constructor__(_prim_ffi_api.Let, var, value, body, span)  # type: ignore
