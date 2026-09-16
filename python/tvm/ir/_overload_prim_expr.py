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
"""Primitive-expression overloads for shared IR expressions."""

from ..runtime import DataTypeCode, ObjectConvertible, const
from .base import Span
from .expr import Expr, is_prim_expr
from .prim import _ffi_api as _prim_ffi_api
from .type import PrimType


def div_ambiguity_error() -> RuntimeError:
    return RuntimeError(
        "TVM supports multiple types of integer divisions, "
        + "please call div, indexdiv/indexmod, floordiv/floormod "
        + " or truncdiv/truncmod directly to avoid ambiguity in the code."
    )


def _dtype_is_int(value):
    if isinstance(value, int):
        return True
    if isinstance(value, ExprOp):
        return value.expr_ty().matches_code(DataTypeCode.INT)
    if is_prim_expr(value):
        return value.ty.matches_code(DataTypeCode.INT)
    return False


def _dtype_is_float(value):
    if isinstance(value, float):
        return True
    if isinstance(value, ExprOp):
        return value.expr_ty().matches_code(DataTypeCode.FLOAT)
    if is_prim_expr(value):
        return value.ty.matches_code(DataTypeCode.FLOAT)
    return False


def _is_scalar_operand(value):
    type_info = getattr(type(value), "__tvm_ffi_type_info__", None)
    if type_info is not None and type_info.type_key == "tirx.BufferRegion":
        raise TypeError(
            "BufferRegion is not a primitive operand; construct a BufferLoad explicitly"
        )
    return isinstance(value, ExprOp | int | float) or is_prim_expr(value)


class ExprOp:
    """Operator overloading for Expr like expressions."""

    # TODO(tkonolige): use inspect to add source information to these objects

    def expr_ty(self) -> PrimType:
        """Return the compile-time primitive type for expression operators."""
        ty = getattr(self, "ty", None)
        if isinstance(ty, PrimType):
            return ty
        raise TypeError(f"Cannot determine PrimType for {type(self).__name__}")

    def __add__(self, other: Expr) -> Expr:
        if not _is_scalar_operand(other):
            return NotImplemented
        return _prim_ffi_api._OpAdd(self, other, None)  # type: ignore

    def __radd__(self, other: Expr) -> Expr:
        if not _is_scalar_operand(other):
            return NotImplemented
        return _prim_ffi_api._OpAdd(other, self, None)  # type: ignore

    def __sub__(self, other: Expr) -> Expr:
        if not _is_scalar_operand(other):
            return NotImplemented
        return _prim_ffi_api._OpSub(self, other, None)  # type: ignore

    def __rsub__(self, other: Expr) -> Expr:
        if not _is_scalar_operand(other):
            return NotImplemented
        return _prim_ffi_api._OpSub(other, self, None)  # type: ignore

    def __mul__(self, other: Expr) -> Expr:
        if not _is_scalar_operand(other):
            return NotImplemented
        return _prim_ffi_api._OpMul(self, other, None)  # type: ignore

    def __rmul__(self, other: Expr) -> Expr:
        if not _is_scalar_operand(other):
            return NotImplemented
        return _prim_ffi_api._OpMul(other, self, None)  # type: ignore

    def __div__(self, other: Expr) -> Expr:
        if not _is_scalar_operand(other):
            return NotImplemented
        if _dtype_is_int(self) and _dtype_is_int(other):
            raise div_ambiguity_error()
        return _prim_ffi_api._OpDiv(self, other, None)  # type: ignore

    def __rdiv__(self, other: Expr) -> Expr:
        if not _is_scalar_operand(other):
            return NotImplemented
        if _dtype_is_int(self) and _dtype_is_int(other):
            raise div_ambiguity_error()
        return _prim_ffi_api._OpDiv(other, self, None)  # type: ignore

    def __truediv__(self, other: Expr) -> Expr:
        if not _is_scalar_operand(other):
            return NotImplemented
        if _dtype_is_int(self) and _dtype_is_int(other):
            raise div_ambiguity_error()
        return _prim_ffi_api._OpDiv(self, other, None)  # type: ignore

    def __rtruediv__(self, other: Expr) -> Expr:
        if not _is_scalar_operand(other):
            return NotImplemented
        if _dtype_is_int(self) and _dtype_is_int(other):
            raise div_ambiguity_error()
        return _prim_ffi_api._OpDiv(other, self, None)  # type: ignore

    def __floordiv__(self, other: Expr) -> Expr:
        return _prim_ffi_api._OpFloorDiv(self, other, None)  # type: ignore

    def __rfloordiv__(self, other: Expr) -> Expr:
        return _prim_ffi_api._OpFloorDiv(other, self, None)  # type: ignore

    def __mod__(self, other: Expr) -> Expr:
        return _prim_ffi_api._OpFloorMod(self, other, None)  # type: ignore

    def __rmod__(self, other: Expr) -> Expr:
        return _prim_ffi_api._OpFloorMod(other, self, None)  # type: ignore

    def __neg__(self) -> Expr:
        neg_one = const(-1, self.expr_ty().dtype)
        return self.__mul__(neg_one)

    def __lshift__(self, other: Expr) -> Expr:
        return _prim_ffi_api.left_shift(self, other, None)  # type: ignore

    def __rlshift__(self, other: Expr) -> Expr:
        return _prim_ffi_api.left_shift(other, self, None)  # type: ignore

    def __rshift__(self, other: Expr) -> Expr:
        return _prim_ffi_api.right_shift(self, other, None)  # type: ignore

    def __rrshift__(self, other: Expr) -> Expr:
        return _prim_ffi_api.right_shift(other, self, None)  # type: ignore

    def __and__(self, other: Expr) -> Expr:
        return _prim_ffi_api.bitwise_and(self, other, None)  # type: ignore

    def __rand__(self, other: Expr) -> Expr:
        return _prim_ffi_api.bitwise_and(other, self, None)  # type: ignore

    def __or__(self, other: Expr) -> Expr:
        return _prim_ffi_api.bitwise_or(self, other, None)  # type: ignore

    def __ror__(self, other: Expr) -> Expr:
        return _prim_ffi_api.bitwise_or(other, self, None)  # type: ignore

    def __xor__(self, other: Expr) -> Expr:
        return _prim_ffi_api.bitwise_xor(self, other, None)  # type: ignore

    def __rxor__(self, other: Expr) -> Expr:
        return _prim_ffi_api.bitwise_xor(other, self, None)  # type: ignore

    def __invert__(self) -> Expr:
        if _dtype_is_float(self):
            raise RuntimeError("Cannot use ~ operator on float type Expr.")
        return _prim_ffi_api.bitwise_not(self, None)  # type: ignore

    def __lt__(self, other: Expr) -> Expr:
        return _prim_ffi_api._OpLT(self, other, None)  # type: ignore

    def __le__(self, other: Expr) -> Expr:
        return _prim_ffi_api._OpLE(self, other, None)  # type: ignore

    def __eq__(self, other: Expr) -> Expr:
        return EqualOp(self, other)

    def __ne__(self, other: Expr) -> Expr:
        return NotEqualOp(self, other)

    def __gt__(self, other: Expr) -> Expr:
        return _prim_ffi_api._OpGT(self, other, None)  # type: ignore

    def __ge__(self, other: Expr) -> Expr:
        return _prim_ffi_api._OpGE(self, other, None)  # type: ignore

    def __nonzero__(self):
        raise ValueError(
            "Cannot use and / or / not operator to Expr, hint: use tvm.tirx.all / "
            "tvm.tirx.any, if it is None checking, use node is not None"
        )

    def __bool__(self) -> bool:
        return self.__nonzero__()

    def equal(self, other: Expr, span: Span | None = None) -> bool:
        """Build an equal check expression with other expr.

        Parameters
        ----------
        other : Expr
            The other expression

        span : Optional[Span]
            The location of the cast in the source.

        Returns
        -------
        ret : Expr
            The equality expression.
        """
        return _prim_ffi_api._OpEQ(self, other, span)  # type: ignore

    def astype(self, dtype: str | PrimType, span: Span | None = None) -> Expr:
        """Cast the expression to other type.

        Parameters
        ----------
        dtype : str
            The type of new expression

        span : Optional[Span]
            The location of the cast in the source.

        Returns
        -------
        expr : Expr
            Expression with new type
        """
        return _prim_ffi_api._cast(dtype, self, span)  # type: ignore


class EqualOp(ObjectConvertible, ExprOp):
    """Deferred equal operator.

    This is used to support sugar that a == b can either
    mean Object.same_as or Object.equal.

    Parameters
    ----------
    a : Expr
        Left operand.

    b : Expr
        Right operand.

    span : Optional[Span]
        The location of the cast in the source.
    """

    # This class is not manipulated by C++. So use python's identity check function is sufficient
    same_as = object.__eq__

    def __init__(self, a: Expr, b: Expr, span: Span | None = None):
        self.a = a
        self.b = b
        self.span = span

    def __nonzero__(self) -> bool:
        return self.a.same_as(self.b)

    def __bool__(self) -> bool:
        return self.__nonzero__()

    def asobject(self) -> Expr:
        """Convert object."""
        return _prim_ffi_api._OpEQ(self.a, self.b, self.span)  # type: ignore

    def expr_ty(self) -> PrimType:
        """Compile-time type of the equality result."""
        return PrimType("bool")

    def __repr__(self) -> str:
        return f"EqualOp({self.a!r}, {self.b!r})"


class NotEqualOp(ObjectConvertible, ExprOp):
    """Deferred NE operator.

    This is used to support sugar that a != b can either
    mean not Object.same_as or make.NE.

    Parameters
    ----------
    a : Expr
        Left operand.

    b : Expr
        Right operand.

    span : Optional[Span]
        The location of the cast in the source.
    """

    # This class is not manipulated by C++. So use python's identity check function is sufficient
    same_as = object.__eq__

    def __init__(self, a: Expr, b: Expr, span: Span | None = None) -> None:
        self.a = a
        self.b = b
        self.span = span

    def __nonzero__(self) -> bool:
        return not self.a.same_as(self.b)

    def __bool__(self) -> bool:
        return self.__nonzero__()

    def asobject(self) -> Expr:
        """Convert object."""
        return _prim_ffi_api._OpNE(self.a, self.b, self.span)  # type: ignore

    def expr_ty(self) -> PrimType:
        """Compile-time type of the inequality result."""
        return PrimType("bool")

    def __repr__(self) -> str:
        return f"NotEqualOp({self.a!r}, {self.b!r})"


__add__ = ExprOp.__add__
__radd__ = ExprOp.__radd__
__sub__ = ExprOp.__sub__
__rsub__ = ExprOp.__rsub__
__mul__ = ExprOp.__mul__
__rmul__ = ExprOp.__rmul__
__div__ = ExprOp.__div__
__rdiv__ = ExprOp.__rdiv__
__truediv__ = ExprOp.__truediv__
__rtruediv__ = ExprOp.__rtruediv__
__floordiv__ = ExprOp.__floordiv__
__rfloordiv__ = ExprOp.__rfloordiv__
__mod__ = ExprOp.__mod__
__rmod__ = ExprOp.__rmod__
__neg__ = ExprOp.__neg__
__lshift__ = ExprOp.__lshift__
__rlshift__ = ExprOp.__rlshift__
__rshift__ = ExprOp.__rshift__
__rrshift__ = ExprOp.__rrshift__
__and__ = ExprOp.__and__
__rand__ = ExprOp.__rand__
__or__ = ExprOp.__or__
__ror__ = ExprOp.__ror__
__xor__ = ExprOp.__xor__
__rxor__ = ExprOp.__rxor__
__invert__ = ExprOp.__invert__
__lt__ = ExprOp.__lt__
__le__ = ExprOp.__le__
__eq__ = ExprOp.__eq__
__ne__ = ExprOp.__ne__
__gt__ = ExprOp.__gt__
__ge__ = ExprOp.__ge__
equal = ExprOp.equal
astype = ExprOp.astype
