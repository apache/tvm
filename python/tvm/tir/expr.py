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

  x = tvm.tir.Var("n", "int32")
  y = x + 2
  assert(isinstance(y, tvm.tir.Add))
  assert(y.a == x)
"""
from typing import Optional, Union
from tvm import ir
import tvm._ffi
from tvm.ir.base import Span

from tvm.runtime import Object, ObjectGeneric, DataType, DataTypeCode, const
from tvm.ir import PrimExpr, Op
import tvm.ir._ffi_api
from . import generic as _generic
from . import _ffi_api


def div_ambiguity_error():
    return RuntimeError(
        "TVM supports multiple types of integer divisions, "
        + "please call div, indexdiv/indexmod, floordiv/floormod "
        + " or truncdiv/truncmod directly to avoid ambiguity in the code."
    )


def _dtype_is_int(value):
    if isinstance(value, int):
        return True
    return (
        isinstance(value, ExprOp) and DataType(value.dtype).type_code == DataTypeCode.INT
    )  # type: ignore


def _dtype_is_float(value):
    if isinstance(value, float):
        return True
    return (
        isinstance(value, ExprOp) and DataType(value.dtype).type_code == DataTypeCode.FLOAT
    )  # type: ignore


class ExprOp(object):
    """Operator overloading for Expr like expressions."""

    # TODO(tkonolige): use inspect to add source information to these objects

    def __add__(self, other):
        return _generic.add(self, other)

    def __radd__(self, other):
        return _generic.add(other, self)

    def __sub__(self, other):
        return _generic.subtract(self, other)

    def __rsub__(self, other):
        return _generic.subtract(other, self)

    def __mul__(self, other):
        return _generic.multiply(self, other)

    def __rmul__(self, other):
        return _generic.multiply(other, self)

    def __div__(self, other):
        if _dtype_is_int(self) and _dtype_is_int(other):
            raise div_ambiguity_error()
        return _generic.divide(self, other)

    def __rdiv__(self, other):
        if _dtype_is_int(self) and _dtype_is_int(other):
            raise div_ambiguity_error()
        return _generic.divide(other, self)

    def __truediv__(self, other):
        if _dtype_is_int(self) and _dtype_is_int(other):
            raise div_ambiguity_error()
        return _generic.divide(self, other)

    def __rtruediv__(self, other):
        if _dtype_is_int(self) and _dtype_is_int(other):
            raise div_ambiguity_error()
        return _generic.divide(other, self)

    def __floordiv__(self, other):
        return _generic.floordiv(self, other)

    def __rfloordiv__(self, other):
        return _generic.floordiv(other, self, None)

    def __mod__(self, other):
        return _ffi_api._OpFloorMod(self, other, None)  # type: ignore

    def __rmod__(self, other):
        return _ffi_api._OpFloorMod(other, self, None)  # type: ignore

    def __neg__(self):
        neg_one = const(-1, self.dtype)  # type: ignore
        return self.__mul__(neg_one)

    def __lshift__(self, other):
        return _ffi_api.left_shift(self, other, None)  # type: ignore

    def __rlshift__(self, other):
        return _ffi_api.left_shift(other, self, None)  # type: ignore

    def __rshift__(self, other):
        return _ffi_api.right_shift(self, other, None)  # type: ignore

    def __rrshift__(self, other):
        return _ffi_api.right_shift(other, self, None)  # type: ignore

    def __and__(self, other):
        return _ffi_api.bitwise_and(self, other, None)  # type: ignore

    def __rand__(self, other):
        return _ffi_api.bitwise_and(other, self, None)  # type: ignore

    def __or__(self, other):
        return _ffi_api.bitwise_or(self, other, None)  # type: ignore

    def __ror__(self, other):
        return _ffi_api.bitwise_or(other, self, None)  # type: ignore

    def __xor__(self, other):
        return _ffi_api.bitwise_xor(self, other, None)  # type: ignore

    def __rxor__(self, other):
        return _ffi_api.bitwise_xor(other, self, None)  # type: ignore

    def __invert__(self):
        if _dtype_is_float(self):
            raise RuntimeError("Cannot use ~ operator on float type Expr.")
        return _ffi_api.bitwise_not(self, None)  # type: ignore

    def __lt__(self, other):
        return _ffi_api._OpLT(self, other, None)  # type: ignore

    def __le__(self, other):
        return _ffi_api._OpLE(self, other, None)  # type: ignore

    def __eq__(self, other):
        return EqualOp(self, other)

    def __ne__(self, other):
        return NotEqualOp(self, other)

    def __gt__(self, other):
        return _ffi_api._OpGT(self, other, None)  # type: ignore

    def __ge__(self, other):
        return _ffi_api._OpGE(self, other, None)  # type: ignore

    def __nonzero__(self):
        raise ValueError(
            "Cannot use and / or / not operator to Expr, hint: "
            + "use tvm.tir.all / tvm.tir.any instead"
        )

    def __bool__(self):
        return self.__nonzero__()

    def equal(self, other, span=None):
        """Build an equal check expression with other expr.

        Parameters
        ----------
        other : PrimExpr
            The other expression

        span : Optional[Span]
            The location of the cast in the source.

        Returns
        -------
        ret : PrimExpr
            The equality expression.
        """
        return _ffi_api._OpEQ(self, other, span)  # type: ignore

    def astype(self, dtype: str, span: Optional[Span] = None):
        """Cast the expression to other type.

        Parameters
        ----------
        dtype : str
            The type of new expression

        span : Optional[Span]
            The location of the cast in the source.

        Returns
        -------
        expr : PrimExpr
            Expression with new type
        """
        return _generic.cast(self, dtype, span)


class EqualOp(ObjectGeneric, ExprOp):
    """Deferred equal operator.

    This is used to support sugar that a == b can either
    mean Object.same_as or Object.equal.

    Parameters
    ----------
    a : PrimExpr
        Left operand.

    b : PrimExpr
        Right operand.

    span : Optional[Span]
        The location of the cast in the source.
    """

    # This class is not manipulated by C++. So use python's identity check function is sufficient
    same_as = object.__eq__

    def __init__(self, a, b, span=None):
        self.a = a
        self.b = b
        self.span = span

    def __nonzero__(self):
        return self.a.same_as(self.b)

    def __bool__(self):
        return self.__nonzero__()

    def asobject(self):
        """Convert object."""
        return _ffi_api._OpEQ(self.a, self.b, self.span)  # type: ignore


class NotEqualOp(ObjectGeneric, ExprOp):
    """Deferred NE operator.

    This is used to support sugar that a != b can either
    mean not Object.same_as or make.NE.

    Parameters
    ----------
    a : PrimExpr
        Left operand.

    b : PrimExpr
        Right operand.

    span : Optional[Span]
        The location of the cast in the source.
    """

    # This class is not manipulated by C++. So use python's identity check function is sufficient
    same_as = object.__eq__

    def __init__(self, a, b, span=None):
        self.a = a
        self.b = b
        self.span = span

    def __nonzero__(self):
        return not self.a.same_as(self.b)

    def __bool__(self):
        return self.__nonzero__()

    def asobject(self):
        """Convert object."""
        return _ffi_api._OpNE(self.a, self.b, self.span)  # type: ignore


class IntImmEnum(ObjectGeneric):
    """Lazily evaluate an IntImm in case
    the constructor is not available in runtime.

    Parameters
    ----------
    value : int
        The enum value

    span : Optional[Span]
        The location of the cast in the source.
    """

    def __init__(self, value, span=None):
        self.value = value
        self.span = span

    def asobject(self):
        """Convert object."""
        return IntImm("int32", self.value, self.span)  # type: ignore


class PrimExprWithOp(ExprOp, PrimExpr):
    """Helper base class to inherit from PrimExpr."""

    # In Python3, We have to explicitly tell interpreter to retain __hash__ if we overide __eq__
    # https://docs.python.org/3.1/reference/datamodel.html#object.__hash__
    __hash__ = PrimExpr.__hash__


class ConstExpr(PrimExprWithOp):
    pass


class BinaryOpExpr(PrimExprWithOp):
    pass


class CmpExpr(PrimExprWithOp):
    pass


class LogicalExpr(PrimExprWithOp):
    pass


@tvm._ffi.register_object("tir.Var")
class Var(PrimExprWithOp):
    """Symbolic variable.

    Parameters
    ----------
    name : str
        The name

    dtype : Union[str, tvm.irType]
        The data type

    span : Optional[Span]
        The location of this itervar in the source code.
    """

    def __init__(self, name: str, dtype: Union[str, ir.Type], span: Optional[Span] = None):
        self.__init_handle_by_constructor__(_ffi_api.Var, name, dtype, span)  # type: ignore


@tvm._ffi.register_object("tir.SizeVar")
class SizeVar(Var):
    """Symbolic variable to represent a tensor index size
       which is greater or equal to zero.

    Parameters
    ----------
    name : str
        The name

    dtype : int
        The data type

    span : Optional[Span]
        The location of this itervar in the source code.
    """

    # pylint: disable=super-init-not-called
    def __init__(self, name, dtype, span=None):
        self.__init_handle_by_constructor__(_ffi_api.SizeVar, name, dtype, span)  # type: ignore


@tvm._ffi.register_object("tir.IterVar")
class IterVar(Object, ExprOp):
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
        The location of this itervar in the source code.

    See Also
    --------
    te.thread_axis: Create thread axis IterVar.
    te.reduce_axis: Create reduce axis IterVar.
    """

    DataPar = 0
    ThreadIndex = 1
    CommReduce = 2
    Ordered = 3
    DimInfo = 4
    Unrolled = 5
    Vectorized = 6
    Parallelized = 7
    Tensorized = 8

    def __init__(self, dom, var, iter_type, thread_tag="", span=None):
        if dom is not None:
            if isinstance(dom, (list, tuple)):
                if len(dom) != 2:
                    raise TypeError("need to be list of ranges")
                dom = tvm.ir.Range(dom[0], dom[1])

            if not isinstance(dom, tvm.ir.Range):
                raise TypeError("dom need to be Range")

        name = var if var is not None else "iter"
        dtype = "int32" if dom is None else dom.extent.dtype
        var = Var(name, dtype=dtype, span=span) if not isinstance(var, Var) else var
        self.__init_handle_by_constructor__(
            _ffi_api.IterVar, dom, var, iter_type, thread_tag, span  # type: ignore
        )


@tvm._ffi.register_object("tir.CommReducer")
class CommReducer(Object):
    """Commutative reduce operator

    Parameters
    ----------
    lhs : List[Var]
       The left arguments of the reducer.

    rhs : List[Var]
       The right arguments of the reducer.

    result : List[PrimExpr]
       The reduction results.

    identity_element : List[PrimExpr]
       The identity elements.

    span : Optional[Span]
        The location of this itervar in the source code.
    """

    def __init__(self, lhs, rhs, result, identity_element, span=None):
        self.__init_handle_by_constructor__(
            _ffi_api.CommReducer, lhs, rhs, result, identity_element, span  # type: ignore
        )


@tvm._ffi.register_object("tir.Reduce")
class Reduce(PrimExprWithOp):
    """Reduce node.

    Parameters
    ----------
    combiner : CommReducer
        The combiner.

    src : list of Expr
        The source expression.

    rdom : list of IterVar
        The iteration domain

    condition : PrimExpr
        The reduce condition.

    value_index : int
        The value index.

    init : list of Expr
        The initial value for output. This can be an int, float or ProducerLoad

    span : Optional[Span]
        The location of this itervar in the source code.
    """

    def __init__(self, combiner, src, rdom, condition, value_index, init=None, span=None):
        self.__init_handle_by_constructor__(
            _ffi_api.Reduce, combiner, src, rdom, condition, value_index, init, span  # type: ignore
        )


@tvm._ffi.register_object
class FloatImm(ConstExpr):
    """Float constant.

    Parameters
    ----------
    dtype : str
        The data type

    value : float
        The constant value.

    span : Optional[Span]
        The location of this itervar in the source code.
    """

    def __init__(self, dtype, value, span=None):
        self.__init_handle_by_constructor__(
            tvm.ir._ffi_api.FloatImm, dtype, value, span  # type: ignore
        )

    def __float__(self):
        return self.value


@tvm._ffi.register_object
class IntImm(ConstExpr):
    """Int constant.

    Parameters
    ----------
    dtype : str
        The data type

    value : int
        The constant value.

    span : Optional[Span]
        The location of this itervar in the source code.
    """

    def __init__(self, dtype, value, span=None):
        self.__init_handle_by_constructor__(
            tvm.ir._ffi_api.IntImm, dtype, value, span  # type: ignore
        )

    def __hash__(self):
        return self.value

    def __int__(self):
        return self.value

    def __nonzero__(self):
        return self.value != 0

    def __eq__(self, other):
        return _ffi_api._OpEQ(self, other, None)  # type: ignore

    def __ne__(self, other):
        return _ffi_api._OpNE(self, other, None)  # type: ignore

    def __bool__(self):
        return self.__nonzero__()


@tvm._ffi.register_object("tir.StringImm")  # type: ignore
class StringImm(ConstExpr):
    """String constant.

    Parameters
    ----------
    value : str
        The value of the function.

    span : Optional[Span]
        The location of this itervar in the source code.
    """

    def __init__(self, value, span=None):
        self.__init_handle_by_constructor__(_ffi_api.StringImm, value, span)  # type: ignore

    def __eq__(self, other):
        if isinstance(other, ConstExpr):
            return self.value == other.value
        return self.value == other

    def __ne__(self, other):
        if isinstance(other, ConstExpr):
            return self.value != other.value
        return self.value != other

    def __hash__(self):
        return PrimExpr.__hash__(self)


@tvm._ffi.register_object("tir.Cast")
class Cast(PrimExprWithOp):
    """Cast expression.

    Parameters
    ----------
    dtype : str
        The data type

    value : PrimExpr
        The value of the function.

    span : Optional[Span]
        The location of this itervar in the source code.
    """

    def __init__(self, dtype, value, span=None):
        self.__init_handle_by_constructor__(_ffi_api.Cast, dtype, value, span)  # type: ignore


@tvm._ffi.register_object("tir.Add")
class Add(BinaryOpExpr):
    """Add node.

    Parameters
    ----------
    a : PrimExpr
        The left hand operand.

    b : PrimExpr
        The right hand operand.

    span : Optional[Span]
        The location of this itervar in the source code.
    """

    def __init__(self, a, b, span=None):
        self.__init_handle_by_constructor__(_ffi_api.Add, a, b, span)  # type: ignore


@tvm._ffi.register_object("tir.Sub")
class Sub(BinaryOpExpr):
    """Sub node.

    Parameters
    ----------
    a : PrimExpr
        The left hand operand.

    b : PrimExpr
        The right hand operand.

    span : Optional[Span]
        The location of this itervar in the source code.
    """

    def __init__(self, a, b, span=None):
        self.__init_handle_by_constructor__(_ffi_api.Sub, a, b, span)  # type: ignore


@tvm._ffi.register_object("tir.Mul")
class Mul(BinaryOpExpr):
    """Mul node.

    Parameters
    ----------
    a : PrimExpr
        The left hand operand.

    b : PrimExpr
        The right hand operand.

    span : Optional[Span]
        The location of this itervar in the source code.
    """

    def __init__(self, a, b, span=None):
        self.__init_handle_by_constructor__(_ffi_api.Mul, a, b, span)  # type: ignore


@tvm._ffi.register_object("tir.Div")
class Div(BinaryOpExpr):
    """Div node.

    Parameters
    ----------
    a : PrimExpr
        The left hand operand.

    b : PrimExpr
        The right hand operand.

    span : Optional[Span]
        The location of this itervar in the source code.
    """

    def __init__(self, a, b, span=None):
        self.__init_handle_by_constructor__(_ffi_api.Div, a, b, span)  # type: ignore


@tvm._ffi.register_object("tir.Mod")
class Mod(BinaryOpExpr):
    """Mod node.

    Parameters
    ----------
    a : PrimExpr
        The left hand operand.

    b : PrimExpr
        The right hand operand.

    span : Optional[Span]
        The location of this itervar in the source code.
    """

    def __init__(self, a, b, span=None):
        self.__init_handle_by_constructor__(_ffi_api.Mod, a, b, span)  # type: ignore


@tvm._ffi.register_object("tir.FloorDiv")
class FloorDiv(BinaryOpExpr):
    """FloorDiv node.

    Parameters
    ----------
    a : PrimExpr
        The left hand operand.

    b : PrimExpr
        The right hand operand.

    span : Optional[Span]
        The location of this itervar in the source code.
    """

    def __init__(self, a, b, span=None):
        self.__init_handle_by_constructor__(_ffi_api.FloorDiv, a, b, span)  # type: ignore


@tvm._ffi.register_object("tir.FloorMod")
class FloorMod(BinaryOpExpr):
    """FloorMod node.

    Parameters
    ----------
    a : PrimExpr
        The left hand operand.

    b : PrimExpr
        The right hand operand.

    span : Optional[Span]
        The location of this itervar in the source code.
    """

    def __init__(self, a, b, span=None):
        self.__init_handle_by_constructor__(_ffi_api.FloorMod, a, b, span)  # type: ignore


@tvm._ffi.register_object("tir.Min")
class Min(BinaryOpExpr):
    """Min node.

    Parameters
    ----------
    a : PrimExpr
        The left hand operand.

    b : PrimExpr
        The right hand operand.

    span : Optional[Span]
        The location of this itervar in the source code.
    """

    def __init__(self, a, b, span=None):
        self.__init_handle_by_constructor__(_ffi_api.Min, a, b, span)  # type: ignore


@tvm._ffi.register_object("tir.Max")
class Max(BinaryOpExpr):
    """Max node.

    Parameters
    ----------
    a : PrimExpr
        The left hand operand.

    b : PrimExpr
        The right hand operand.

    span : Optional[Span]
        The location of this itervar in the source code.
    """

    def __init__(self, a, b, span=None):
        self.__init_handle_by_constructor__(_ffi_api.Max, a, b, span)  # type: ignore


@tvm._ffi.register_object("tir.EQ")
class EQ(CmpExpr):
    """EQ node.

    Parameters
    ----------
    a : PrimExpr
        The left hand operand.

    b : PrimExpr
        The right hand operand.

    span : Optional[Span]
        The location of this itervar in the source code.
    """

    def __init__(self, a, b, span=None):
        self.__init_handle_by_constructor__(_ffi_api.EQ, a, b, span)  # type: ignore


@tvm._ffi.register_object("tir.NE")
class NE(CmpExpr):
    """NE node.

    Parameters
    ----------
    a : PrimExpr
        The left hand operand.

    b : PrimExpr
        The right hand operand.

    span : Optional[Span]
        The location of this itervar in the source code.
    """

    def __init__(self, a, b, span=None):
        self.__init_handle_by_constructor__(_ffi_api.NE, a, b, span)  # type: ignore


@tvm._ffi.register_object("tir.LT")
class LT(CmpExpr):
    """LT node.

    Parameters
    ----------
    a : PrimExpr
        The left hand operand.

    b : PrimExpr
        The right hand operand.

    span : Optional[Span]
        The location of this itervar in the source code.
    """

    def __init__(self, a, b, span=None):
        self.__init_handle_by_constructor__(_ffi_api.LT, a, b, span)  # type: ignore


@tvm._ffi.register_object("tir.LE")
class LE(CmpExpr):
    """LE node.

    Parameters
    ----------
    a : PrimExpr
        The left hand operand.

    b : PrimExpr
        The right hand operand.

    span : Optional[Span]
        The location of this itervar in the source code.
    """

    def __init__(self, a, b, span=None):
        self.__init_handle_by_constructor__(_ffi_api.LE, a, b, span)  # type: ignore


@tvm._ffi.register_object("tir.GT")
class GT(CmpExpr):
    """GT node.

    Parameters
    ----------
    a : PrimExpr
        The left hand operand.

    b : PrimExpr
        The right hand operand.

    span : Optional[Span]
        The location of this itervar in the source code.
    """

    def __init__(self, a, b, span=None):
        self.__init_handle_by_constructor__(_ffi_api.GT, a, b, span)  # type: ignore


@tvm._ffi.register_object("tir.GE")
class GE(CmpExpr):
    """GE node.

    Parameters
    ----------
    a : PrimExpr
        The left hand operand.

    b : PrimExpr
        The right hand operand.

    span : Optional[Span]
        The location of this itervar in the source code.
    """

    def __init__(self, a, b, span=None):
        self.__init_handle_by_constructor__(_ffi_api.GE, a, b, span)  # type: ignore


@tvm._ffi.register_object("tir.And")
class And(LogicalExpr):
    """And node.

    Parameters
    ----------
    a : PrimExpr
        The left hand operand.

    b : PrimExpr
        The right hand operand.

    span : Optional[Span]
        The location of this itervar in the source code.
    """

    def __init__(self, a, b, span=None):
        self.__init_handle_by_constructor__(_ffi_api.And, a, b, span)  # type: ignore


@tvm._ffi.register_object("tir.Or")
class Or(LogicalExpr):
    """Or node.

    Parameters
    ----------
    a : PrimExpr
        The left hand operand.

    b : PrimExpr
        The right hand operand.

    span : Optional[Span]
        The location of this itervar in the source code.
    """

    def __init__(self, a, b, span=None):
        self.__init_handle_by_constructor__(_ffi_api.Or, a, b, span)  # type: ignore


@tvm._ffi.register_object("tir.Not")
class Not(LogicalExpr):
    """Not node.

    Parameters
    ----------
    a : PrimExpr
        The input value

    span : Optional[Span]
        The location of this itervar in the source code.
    """

    def __init__(self, a, span=None):
        self.__init_handle_by_constructor__(_ffi_api.Not, a, span)  # type: ignore


@tvm._ffi.register_object("tir.Select")
class Select(PrimExprWithOp):
    """Select node.

    Note
    ----
    Select may compute both true_value and false_value.
    Use :py:class:`tvm.tir.if_then_else` instead if you want to
    get a conditional expression that only evaluates
    the correct branch.

    Parameters
    ----------
    condition : PrimExpr
        The condition expression.

    true_value : PrimExpr
        The value to take when condition is true.

    false_value : PrimExpr
        The value to take when condition is false.

    span : Optional[Span]
        The location of this itervar in the source code.
    """

    def __init__(self, condition, true_value, false_value, span=None):
        self.__init_handle_by_constructor__(
            _ffi_api.Select, condition, true_value, false_value, span  # type: ignore
        )


@tvm._ffi.register_object("tir.Load")
class Load(PrimExprWithOp):
    """Load node.

    Parameters
    ----------
    dtype : str
        The data type.

    buffer_var : Var
        The buffer variable in the load expression.

    index : PrimExpr
        The index in the load.

    predicate : PrimExpr
        The load predicate.

    span : Optional[Span]
        The location of this itervar in the source code.
    """

    def __init__(self, dtype, buffer_var, index, predicate=None, span=None):
        if predicate is None:
            predicate = _ffi_api.const_true(dtype, span)  # type: ignore
        self.__init_handle_by_constructor__(
            _ffi_api.Load, dtype, buffer_var, index, predicate, span  # type: ignore
        )


@tvm._ffi.register_object("tir.BufferLoad")
class BufferLoad(PrimExprWithOp):
    """Buffer load node.

    Parameters
    ----------
    buffer : Buffer
        The buffer to be loaded.

    indices : List[PrimExpr]
        The buffer indices.

    span : Optional[Span]
        The location of this itervar in the source code.
    """

    def __init__(self, buffer, indices, span=None):
        self.__init_handle_by_constructor__(
            _ffi_api.BufferLoad, buffer, indices, span  # type: ignore
        )


@tvm._ffi.register_object("tir.ProducerLoad")
class ProducerLoad(PrimExprWithOp):
    """Producer load node.

    Parameters
    ----------
    producer : DataProducer
        The buffer to be loaded.

    indices : List[PrimExpr]
        The buffer indices.

    span : Optional[Span]
        The location of this itervar in the source code.
    """

    def __init__(self, producer, indices, span=None):
        self.__init_handle_by_constructor__(
            _ffi_api.ProducerLoad, producer, indices, span  # type: ignore
        )


@tvm._ffi.register_object("tir.Ramp")
class Ramp(PrimExprWithOp):
    """Ramp node.

    Parameters
    ----------
    base : PrimExpr
        The base expression.

    stride : ramp stride
        The stride of the ramp.

    lanes : int
        The lanes of the expression.

    span : Optional[Span]
        The location of this itervar in the source code.
    """

    def __init__(self, base, stride, lanes, span=None):
        self.__init_handle_by_constructor__(
            _ffi_api.Ramp, base, stride, lanes, span  # type: ignore
        )


@tvm._ffi.register_object("tir.Broadcast")
class Broadcast(PrimExprWithOp):
    """Broadcast node.

    Parameters
    ----------
    value : PrimExpr
        The value of the expression.

    lanes : int
        The lanes of the expression.

    span : Optional[Span]
        The location of this itervar in the source code.
    """

    def __init__(self, value, lanes, span=None):
        self.__init_handle_by_constructor__(_ffi_api.Broadcast, value, lanes, span)  # type: ignore


@tvm._ffi.register_object("tir.Shuffle")
class Shuffle(PrimExprWithOp):
    """Shuffle node.

    Parameters
    ----------
    vectors : Array of Expr
        The vectors

    indices : Array of indices
        The indices

    span : Optional[Span]
        The location of this itervar in the source code.
    """

    def __init__(self, vectors, indices, span=None):
        self.__init_handle_by_constructor__(
            _ffi_api.Shuffle, vectors, indices, span  # type: ignore
        )


class CallEffectKind:
    """Possible kinds of Call effects."""

    # only expose up to opaque
    ExprAnnotation = IntImmEnum(0)
    Pure = IntImmEnum(1)
    ReadState = IntImmEnum(2)
    UpdateState = IntImmEnum(3)
    Opaque = UpdateState


@tvm._ffi.register_object("tir.Call")
class Call(PrimExprWithOp):
    """Call node.

    Parameters
    ----------
    dtype : str
        The return data type

    op : Union[RelayExpr, str]
        The function to be called, or the name
        to the global tvm.Op

    args : list of Expr
        The input arguments to the call

    span : Optional[Span]
        The location of this itervar in the source code.
    """

    def __init__(self, dtype, op, args, span=None):
        if isinstance(op, str):
            if not op.startswith("tir."):
                raise ValueError(
                    (
                        "Cannot handle str op argument %s. This function only handles str "
                        + "argument with the tir namespace. If you are "
                        + "certain about the intrinsic name, pass in Op.get(name) instead"
                    )
                    % op
                )
            op = Op.get(op)
        self.__init_handle_by_constructor__(_ffi_api.Call, dtype, op, args, span)  # type: ignore


@tvm._ffi.register_object("tir.Let")
class Let(PrimExprWithOp):
    """Let node.

    Parameters
    ----------
    var : Var
        The variable in the binding.

    value : PrimExpr
        The value in to be binded.

    body : PrimExpr
        The body expression.

    span : Optional[Span]
        The location of this itervar in the source code.
    """

    def __init__(self, var, value, body, span=None):
        self.__init_handle_by_constructor__(_ffi_api.Let, var, value, body, span)  # type: ignore


@tvm._ffi.register_object("tir.Any")
class Any(PrimExprWithOp):
    """Any node.

    span : Optional[Span]
        The location of this itervar in the source code.
    """

    def __init__(self, span=None):
        self.__init_handle_by_constructor__(_ffi_api.Any, span)  # type: ignore
