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
# pylint: disable=redefined-builtin
"""TIR expression nodes.

Each expression node have subfields that can be visited from python side.
For example, you can use addexp.a to get the left operand of an Add node.

.. code-block:: python

  x = tvm.tirx.Var("n", "int32")
  y = x + 2
  assert(isinstance(y, tvm.tirx.Add))
  assert(y.a == x)
"""

import tvm_ffi

import tvm.ir._ffi_api
import tvm.ir._overload_prim_expr as _overload_prim_expr
from tvm import ir
from tvm.ir import Expr
from tvm.ir.base import Span
from tvm.runtime import DataTypeCode, Object, ObjectConvertible, Scriptable, const

from . import _ffi_api
from . import generic as _generic
from .buffer import Buffer, DataProducer


def convert(expr) -> Expr:
    return _ffi_api.convert(expr)


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
    return False


def _dtype_is_float(value):
    if isinstance(value, float):
        return True
    if isinstance(value, ExprOp):
        return value.expr_ty().matches_code(DataTypeCode.FLOAT)
    return False


class ExprOp:
    """Operator overloading for Expr like expressions."""

    # TODO(tkonolige): use inspect to add source information to these objects

    def expr_ty(self) -> ir.PrimType:
        """Return the compile-time primitive type for expression operators."""
        ty = getattr(self, "ty", None)
        if isinstance(ty, ir.PrimType):
            return ty
        raise TypeError(f"Cannot determine PrimType for {type(self).__name__}")

    def __add__(self, other: Expr) -> Expr:
        return _generic.add(self, other)

    def __radd__(self, other: Expr) -> Expr:
        return _generic.add(other, self)

    def __sub__(self, other: Expr) -> Expr:
        return _generic.subtract(self, other)

    def __rsub__(self, other: Expr) -> Expr:
        return _generic.subtract(other, self)

    def __mul__(self, other: Expr) -> Expr:
        return _generic.multiply(self, other)

    def __rmul__(self, other: Expr) -> Expr:
        return _generic.multiply(other, self)

    def __div__(self, other: Expr) -> Expr:
        if _dtype_is_int(self) and _dtype_is_int(other):
            raise div_ambiguity_error()
        return _generic.divide(self, other)

    def __rdiv__(self, other: Expr) -> Expr:
        if _dtype_is_int(self) and _dtype_is_int(other):
            raise div_ambiguity_error()
        return _generic.divide(other, self)

    def __truediv__(self, other: Expr) -> Expr:
        if _dtype_is_int(self) and _dtype_is_int(other):
            raise div_ambiguity_error()
        return _generic.divide(self, other)

    def __rtruediv__(self, other: Expr) -> Expr:
        if _dtype_is_int(self) and _dtype_is_int(other):
            raise div_ambiguity_error()
        return _generic.divide(other, self)

    def __floordiv__(self, other: Expr) -> Expr:
        return _generic.floordiv(self, other)

    def __rfloordiv__(self, other: Expr) -> Expr:
        return _generic.floordiv(other, self, None)

    def __mod__(self, other: Expr) -> Expr:
        return _ffi_api._OpFloorMod(self, other, None)  # type: ignore

    def __rmod__(self, other: Expr) -> Expr:
        return _ffi_api._OpFloorMod(other, self, None)  # type: ignore

    def __neg__(self) -> Expr:
        neg_one = const(-1, self.expr_ty().dtype)
        return self.__mul__(neg_one)

    def __lshift__(self, other: Expr) -> Expr:
        return _ffi_api.left_shift(self, other, None)  # type: ignore

    def __rlshift__(self, other: Expr) -> Expr:
        return _ffi_api.left_shift(other, self, None)  # type: ignore

    def __rshift__(self, other: Expr) -> Expr:
        return _ffi_api.right_shift(self, other, None)  # type: ignore

    def __rrshift__(self, other: Expr) -> Expr:
        return _ffi_api.right_shift(other, self, None)  # type: ignore

    def __and__(self, other: Expr) -> Expr:
        return _ffi_api.bitwise_and(self, other, None)  # type: ignore

    def __rand__(self, other: Expr) -> Expr:
        return _ffi_api.bitwise_and(other, self, None)  # type: ignore

    def __or__(self, other: Expr) -> Expr:
        return _ffi_api.bitwise_or(self, other, None)  # type: ignore

    def __ror__(self, other: Expr) -> Expr:
        return _ffi_api.bitwise_or(other, self, None)  # type: ignore

    def __xor__(self, other: Expr) -> Expr:
        return _ffi_api.bitwise_xor(self, other, None)  # type: ignore

    def __rxor__(self, other: Expr) -> Expr:
        return _ffi_api.bitwise_xor(other, self, None)  # type: ignore

    def __invert__(self) -> Expr:
        if _dtype_is_float(self):
            raise RuntimeError("Cannot use ~ operator on float type Expr.")
        return _ffi_api.bitwise_not(self, None)  # type: ignore

    def __lt__(self, other: Expr) -> Expr:
        return _ffi_api._OpLT(self, other, None)  # type: ignore

    def __le__(self, other: Expr) -> Expr:
        return _ffi_api._OpLE(self, other, None)  # type: ignore

    def __eq__(self, other: Expr) -> Expr:
        return EqualOp(self, other)

    def __ne__(self, other: Expr) -> Expr:
        return NotEqualOp(self, other)

    def __gt__(self, other: Expr) -> Expr:
        return _ffi_api._OpGT(self, other, None)  # type: ignore

    def __ge__(self, other: Expr) -> Expr:
        return _ffi_api._OpGE(self, other, None)  # type: ignore

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
        return _ffi_api._OpEQ(self, other, span)  # type: ignore

    def astype(self, dtype: str | ir.PrimType, span: Span | None = None) -> Expr:
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
        return _generic.cast(self, dtype, span)


_overload_prim_expr.__add__ = ExprOp.__add__
_overload_prim_expr.__radd__ = ExprOp.__radd__
_overload_prim_expr.__sub__ = ExprOp.__sub__
_overload_prim_expr.__rsub__ = ExprOp.__rsub__
_overload_prim_expr.__mul__ = ExprOp.__mul__
_overload_prim_expr.__rmul__ = ExprOp.__rmul__
_overload_prim_expr.__div__ = ExprOp.__div__
_overload_prim_expr.__rdiv__ = ExprOp.__rdiv__
_overload_prim_expr.__truediv__ = ExprOp.__truediv__
_overload_prim_expr.__rtruediv__ = ExprOp.__rtruediv__
_overload_prim_expr.__floordiv__ = ExprOp.__floordiv__
_overload_prim_expr.__rfloordiv__ = ExprOp.__rfloordiv__
_overload_prim_expr.__mod__ = ExprOp.__mod__
_overload_prim_expr.__rmod__ = ExprOp.__rmod__
_overload_prim_expr.__neg__ = ExprOp.__neg__
_overload_prim_expr.__lshift__ = ExprOp.__lshift__
_overload_prim_expr.__rlshift__ = ExprOp.__rlshift__
_overload_prim_expr.__rshift__ = ExprOp.__rshift__
_overload_prim_expr.__rrshift__ = ExprOp.__rrshift__
_overload_prim_expr.__and__ = ExprOp.__and__
_overload_prim_expr.__rand__ = ExprOp.__rand__
_overload_prim_expr.__or__ = ExprOp.__or__
_overload_prim_expr.__ror__ = ExprOp.__ror__
_overload_prim_expr.__xor__ = ExprOp.__xor__
_overload_prim_expr.__rxor__ = ExprOp.__rxor__
_overload_prim_expr.__invert__ = ExprOp.__invert__
_overload_prim_expr.__lt__ = ExprOp.__lt__
_overload_prim_expr.__le__ = ExprOp.__le__
_overload_prim_expr.__eq__ = ExprOp.__eq__
_overload_prim_expr.__ne__ = ExprOp.__ne__
_overload_prim_expr.__gt__ = ExprOp.__gt__
_overload_prim_expr.__ge__ = ExprOp.__ge__
_overload_prim_expr.equal = ExprOp.equal
_overload_prim_expr.astype = ExprOp.astype


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
        return _ffi_api._OpEQ(self.a, self.b, self.span)  # type: ignore

    def expr_ty(self) -> ir.PrimType:
        """Compile-time type of the equality result."""
        return ir.PrimType("bool")

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
        return _ffi_api._OpNE(self.a, self.b, self.span)  # type: ignore

    def expr_ty(self) -> ir.PrimType:
        """Compile-time type of the inequality result."""
        return ir.PrimType("bool")

    def __repr__(self) -> str:
        return f"NotEqualOp({self.a!r}, {self.b!r})"


class IntImmEnum(ObjectConvertible):
    """Lazily evaluate an IntImm in case
    the constructor is not available in runtime.

    Parameters
    ----------
    value : int
        The enum value

    span : Optional[Span]
        The location of the cast in the source.
    """

    def __init__(self, value: int, span: Span | None = None) -> None:
        self.value = value
        self.span = span

    def asobject(self) -> "IntImm":
        """Convert object."""
        return IntImm("int32", self.value, self.span)  # type: ignore


class ExprWithOp(ExprOp, Expr, Scriptable):
    """Helper base class to inherit from Expr."""

    # In Python3, We have to explicitly tell interpreter to retain __hash__ if we overide __eq__
    # https://docs.python.org/3.1/reference/datamodel.html#object.__hash__
    __hash__ = Expr.__hash__


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


@tvm_ffi.register_object("tirx.Var")
class Var(ExprWithOp):
    """Symbolic variable.

    Parameters
    ----------
    name : str
        The name

    dtype : Union[str, ir.Type]
        The data type

    span : Optional[Span]
        The location of this expression in the source code.
    """

    name_hint: str
    def __init__(self, name: str, dtype: str | ir.Type, span: Span | None = None) -> None:
        self.__init_handle_by_constructor__(_ffi_api.Var, name, dtype, span)  # type: ignore


@tvm_ffi.register_object("tirx.SizeVar")
class SizeVar(Var):
    """Symbolic variable to represent a tensor index size
       which is greater or equal to zero.

    Parameters
    ----------
    name : str
        The name

    dtype : Union[str, ir.Type]
        The data type

    span : Optional[Span]
        The location of this expression in the source code.
    """

    # pylint: disable=super-init-not-called
    def __init__(self, name: str, dtype: str | ir.Type, span: Span | None = None) -> None:
        self.__init_handle_by_constructor__(_ffi_api.SizeVar, name, dtype, span)  # type: ignore


@tvm_ffi.register_object("tirx.IterVar")
class IterVar(ExprOp, Object, Scriptable):
    """Represent iteration variable.

    IterVar represents axis iterations in the computation.

    Parameters
    ----------
    dom : Range
        The domain of the iteration.

    var : Union[Var, str]
        The internal variable that is used for iteration.

    iter_type : int
        The iteration type.

    thread_tag : str
        The thread type tag.

    span : Optional[Span]
        The location of this expression in the source code.

    See Also
    --------
    te.thread_axis: Create thread axis IterVar.
    te.reduce_axis: Create reduce axis IterVar.
    """

    DataPar = 0
    ThreadIndex = 1
    CommReduce = 2
    Ordered = 3
    Opaque = 4
    Unrolled = 5
    Vectorized = 6
    Parallelized = 7
    Tensorized = 8

    dom: ir.Range
    var: Var
    iter_type: int
    thread_tag: str

    def __init__(
        self,
        dom: ir.Range,
        var: Var | str,
        iter_type: int,
        thread_tag: str = "",
        span: Span | None = None,
    ) -> None:
        if dom is not None:
            if isinstance(dom, list | tuple):
                if len(dom) != 2:
                    raise TypeError("need to be list of ranges")
                dom = tvm.ir.Range(dom[0], dom[1])

            if not isinstance(dom, tvm.ir.Range):
                raise TypeError("dom need to be Range")

        name = var if var is not None else "iter"
        dtype = "int32" if dom is None else dom.extent.ty
        var = Var(name, dtype=dtype, span=span) if not isinstance(var, Var) else var
        if dom is not None:
            assert var.ty == dom.extent.ty, "IterVar's Var type must match its domain's extent type"
        self.__init_handle_by_constructor__(
            _ffi_api.IterVar,
            dom,
            var,
            iter_type,
            thread_tag,
            span,  # type: ignore
        )

    def expr_ty(self) -> ir.PrimType:
        """Compile-time type of the iteration variable."""
        return self.var.ty


@tvm_ffi.register_object("tirx.CommReducer")
class CommReducer(Object, Scriptable):
    """Commutative reduce operator

    Parameters
    ----------
    lhs : List[Var]
       The left arguments of the reducer.

    rhs : List[Var]
       The right arguments of the reducer.

    result : List[Expr]
       The reduction results.

    identity_element : List[Expr]
       The identity elements.

    span : Optional[Span]
        The location of this expression in the source code.
    """

    lhs: list[Var]
    rhs: list[Var]
    result: list[Expr]
    identity_element: list[Expr]

    def __init__(
        self,
        lhs: list[Var],
        rhs: list[Var],
        result: list[Expr],
        identity_element: list[Expr],
        span: Span | None = None,
    ) -> None:
        self.__init_handle_by_constructor__(
            _ffi_api.CommReducer,
            lhs,
            rhs,
            result,
            identity_element,
            span,  # type: ignore
        )


@tvm_ffi.register_object("tirx.Reduce")
class Reduce(ExprWithOp):
    """Reduce node.

    Parameters
    ----------
    combiner : CommReducer
        The combiner.

    src : list of Expr
        The source expression.

    rdom : list of IterVar
        The iteration domain

    condition : Expr
        The reduce condition.

    value_index : int
        The value index.

    init : list of Expr
        The initial value for output. This can be an int, float or ProducerLoad

    span : Optional[Span]
        The location of this expression in the source code.
    """

    combiner: CommReducer
    source: list[Expr]
    init: list[Expr]
    axis: list[IterVar]
    condition: Expr
    value_index: int

    def __init__(
        self,
        combiner: CommReducer,
        src: list[Expr],
        rdom: list[IterVar],
        condition: Expr,
        value_index: int,
        init: list[Expr] | None = None,
        span: Span | None = None,
    ) -> None:
        init = [] if init is None else init
        self.__init_handle_by_constructor__(
            _ffi_api.Reduce,
            combiner,
            src,
            rdom,
            condition,
            value_index,
            init,
            span,  # type: ignore
        )


@tvm_ffi.register_object("ir.FloatImm")
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

    def __init__(self, dtype: str | ir.PrimType, value: float, span: Span | None = None) -> None:
        if isinstance(dtype, ir.PrimType):
            dtype = dtype.dtype
        self.__init_handle_by_constructor__(
            tvm.ir._ffi_api.FloatImm,
            dtype,
            value,
            span,  # type: ignore
        )

    def __float__(self) -> float:
        return self.value


@tvm_ffi.register_object("ir.IntImm")
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

    def __init__(self, dtype: str | ir.PrimType, value: int, span: Span | None = None) -> None:
        if isinstance(dtype, ir.PrimType):
            dtype = dtype.dtype
        self.__init_handle_by_constructor__(
            tvm.ir._ffi_api.IntImm,
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
        return _ffi_api._OpEQ(self, other, None)  # type: ignore

    def __ne__(self, other: Expr) -> Expr:
        return _ffi_api._OpNE(self, other, None)  # type: ignore

    def __bool__(self) -> bool:
        return self.__nonzero__()


@tvm_ffi.register_object("tirx.StringImm")  # type: ignore
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
        self.__init_handle_by_constructor__(_ffi_api.StringImm, value, span)  # type: ignore

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


@tvm_ffi.register_object("tirx.Cast")
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

    def __init__(self, dtype: str | ir.PrimType, value, span: Span | None = None) -> None:
        if isinstance(dtype, ir.PrimType):
            dtype = dtype.dtype
        self.__init_handle_by_constructor__(_ffi_api.Cast, dtype, value, span)  # type: ignore


@tvm_ffi.register_object("tirx.Add")
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
        self.__init_handle_by_constructor__(_ffi_api.Add, a, b, span)  # type: ignore


@tvm_ffi.register_object("tirx.Sub")
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
        self.__init_handle_by_constructor__(_ffi_api.Sub, a, b, span)  # type: ignore


@tvm_ffi.register_object("tirx.Mul")
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
        self.__init_handle_by_constructor__(_ffi_api.Mul, a, b, span)  # type: ignore


@tvm_ffi.register_object("tirx.Div")
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
        self.__init_handle_by_constructor__(_ffi_api.Div, a, b, span)  # type: ignore


@tvm_ffi.register_object("tirx.Mod")
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
        self.__init_handle_by_constructor__(_ffi_api.Mod, a, b, span)  # type: ignore


@tvm_ffi.register_object("tirx.FloorDiv")
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
        self.__init_handle_by_constructor__(_ffi_api.FloorDiv, a, b, span)  # type: ignore


@tvm_ffi.register_object("tirx.FloorMod")
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
        self.__init_handle_by_constructor__(_ffi_api.FloorMod, a, b, span)  # type: ignore


@tvm_ffi.register_object("tirx.Min")
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
        self.__init_handle_by_constructor__(_ffi_api.Min, a, b, span)  # type: ignore


@tvm_ffi.register_object("tirx.Max")
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
        self.__init_handle_by_constructor__(_ffi_api.Max, a, b, span)  # type: ignore


@tvm_ffi.register_object("tirx.EQ")
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
        self.__init_handle_by_constructor__(_ffi_api.EQ, a, b, span)  # type: ignore


@tvm_ffi.register_object("tirx.NE")
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
        self.__init_handle_by_constructor__(_ffi_api.NE, a, b, span)  # type: ignore


@tvm_ffi.register_object("tirx.LT")
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
        self.__init_handle_by_constructor__(_ffi_api.LT, a, b, span)  # type: ignore


@tvm_ffi.register_object("tirx.LE")
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
        self.__init_handle_by_constructor__(_ffi_api.LE, a, b, span)  # type: ignore


@tvm_ffi.register_object("tirx.GT")
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
        self.__init_handle_by_constructor__(_ffi_api.GT, a, b, span)  # type: ignore


@tvm_ffi.register_object("tirx.GE")
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
        self.__init_handle_by_constructor__(_ffi_api.GE, a, b, span)  # type: ignore


@tvm_ffi.register_object("tirx.And")
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
        self.__init_handle_by_constructor__(_ffi_api.And, a, b, span)  # type: ignore


@tvm_ffi.register_object("tirx.Or")
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
        self.__init_handle_by_constructor__(_ffi_api.Or, a, b, span)  # type: ignore


@tvm_ffi.register_object("tirx.Not")
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
        self.__init_handle_by_constructor__(_ffi_api.Not, a, span)  # type: ignore


@tvm_ffi.register_object("tirx.Select")
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
            _ffi_api.Select,
            condition,
            true_value,
            false_value,
            span,  # type: ignore
        )


@tvm_ffi.register_object("tirx.BufferLoad")
class BufferLoad(ExprWithOp):
    """Buffer load node.

    Parameters
    ----------
    buffer : Buffer
        The buffer to be loaded.

    indices : List[Expr]
        The buffer indices to load values from.

    span : Optional[Span]
        The location of this expression in the source code.

    predicate : Optional[Expr]
        A vector mask of boolean values indicating which lanes of a vector are to be
        loaded. The number lanes of the mask must be equal to the number of lanes being loaded.
    """

    buffer: Buffer
    indices: list[Expr]

    def __init__(
        self,
        buffer: Buffer,
        indices: list[Expr],
        predicate: Expr | None = None,
        span: Span | None = None,
    ) -> None:
        self.__init_handle_by_constructor__(
            _ffi_api.BufferLoad,
            buffer,
            indices,
            predicate,
            span,  # type: ignore
        )


@tvm_ffi.register_object("tirx.ProducerLoad")
class ProducerLoad(ExprWithOp):
    """Producer load node.

    Parameters
    ----------
    producer : DataProducer
        The buffer to be loaded.

    indices : List[Expr]
        The buffer indices.

    span : Optional[Span]
        The location of this expression in the source code.
    """

    producer: DataProducer
    indices: list[Expr]

    def __init__(
        self, producer: DataProducer, indices: list[Expr], span: Span | None = None
    ) -> None:
        self.__init_handle_by_constructor__(
            _ffi_api.ProducerLoad,
            producer,
            indices,
            span,  # type: ignore
        )


@tvm_ffi.register_object("tirx.Ramp")
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
            _ffi_api.Ramp,
            base,
            stride,
            lanes,
            span,  # type: ignore
        )


@tvm_ffi.register_object("tirx.Broadcast")
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
        self.__init_handle_by_constructor__(_ffi_api.Broadcast, value, lanes, span)  # type: ignore


@tvm_ffi.register_object("tirx.Shuffle")
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
            _ffi_api.Shuffle,
            vectors,
            indices,
            span,  # type: ignore
        )


class CallEffectKind:
    """Possible kinds of Call effects."""

    # only expose up to opaque
    ExprAnnotation = IntImmEnum(0)
    Pure = IntImmEnum(1)
    ReadState = IntImmEnum(2)
    UpdateState = IntImmEnum(3)
    Opaque = UpdateState


@tvm_ffi.register_object("tirx.Let")
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
        self.__init_handle_by_constructor__(_ffi_api.Let, var, value, body, span)  # type: ignore
