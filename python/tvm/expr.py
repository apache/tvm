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
"""Expression AST Node in TVM.

User do not need to deal with expression AST node directly.
But they can be helpful for developer to do quick proptyping.
While not displayed in the document and python file.
Each expression node have subfields that can be visited from python side.

For example, you can use addexp.a to get the left operand of an Add node.

.. code-block:: python

  x = tvm.var("n")
  y = x + 2
  assert(isinstance(y, tvm.expr.Add))
  assert(y.a == x)
"""
# pylint: disable=missing-docstring
from __future__ import absolute_import as _abs
from ._ffi.node import NodeBase, NodeGeneric, register_node
from . import make as _make
from . import generic as _generic
from . import _api_internal


class ExprOp(object):
    def __add__(self, other):
        return _generic.add(self, other)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return _generic.subtract(self, other)

    def __rsub__(self, other):
        return _generic.subtract(other, self)

    def __mul__(self, other):
        return _generic.multiply(self, other)

    def __rmul__(self, other):
        return _generic.multiply(other, self)

    def __div__(self, other):
        return _generic.divide(self, other)

    def __rdiv__(self, other):
        return _generic.divide(other, self)

    def __truediv__(self, other):
        return self.__div__(other)

    def __rtruediv__(self, other):
        return self.__rdiv__(other)

    def __floordiv__(self, other):
        return self.__div__(other)

    def __rfloordiv__(self, other):
        return self.__rdiv__(other)

    def __mod__(self, other):
        return _make._OpMod(self, other)

    def __neg__(self):
        neg_one = _api_internal._const(-1, self.dtype)
        return self.__mul__(neg_one)

    def __lshift__(self, other):
        return _make.left_shift(self, other)

    def __rshift__(self, other):
        return _make.right_shift(self, other)

    def __and__(self, other):
        return _make.bitwise_and(self, other)

    def __or__(self, other):
        return _make.bitwise_or(self, other)

    def __xor__(self, other):
        return _make.bitwise_xor(self, other)

    def __invert__(self):
        return _make.Call(self.dtype, "bitwise_not", [self], Call.PureIntrinsic, None, 0)

    def __lt__(self, other):
        return _make._OpLT(self, other)

    def __le__(self, other):
        return _make._OpLE(self, other)

    def __eq__(self, other):
        return EqualOp(self, other)

    def __ne__(self, other):
        return NotEqualOp(self, other)

    def __gt__(self, other):
        return _make._OpGT(self, other)

    def __ge__(self, other):
        return _make._OpGE(self, other)

    def __nonzero__(self):
        raise ValueError("Cannot use and / or / not operator to Expr, hint: " +
                         "use tvm.all / tvm.any instead")

    def __bool__(self):
        return self.__nonzero__()

    def equal(self, other):
        """Build an equal check expression with other expr.

        Parameters
        ----------
        other : Expr
            The other expression

        Returns
        -------
        ret : Expr
            The equality expression.
        """
        return _make._OpEQ(self, other)

    def astype(self, dtype):
        """Cast the expression to other type.

        Parameters
        ----------
        dtype : str
            The type of new expression

        Returns
        -------
        expr : Expr
            Expression with new type
        """
        return _generic.cast(self, dtype)


class EqualOp(NodeGeneric, ExprOp):
    """Deferred equal operator.

    This is used to support sugar that a == b can either
    mean NodeBase.same_as or NodeBase.equal.

    Parameters
    ----------
    a : Expr
        Left operand.

    b : Expr
        Right operand.
    """
    # This class is not manipulated by C++. So use python's identity check function is sufficient
    same_as = object.__eq__

    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __nonzero__(self):
        return self.a.same_as(self.b)

    def __bool__(self):
        return self.__nonzero__()

    def asnode(self):
        """Convert node."""
        return _make._OpEQ(self.a, self.b)


class NotEqualOp(NodeGeneric, ExprOp):
    """Deferred NE operator.

    This is used to support sugar that a != b can either
    mean not NodeBase.same_as or make.NE.

    Parameters
    ----------
    a : Expr
        Left operand.

    b : Expr
        Right operand.
    """
    # This class is not manipulated by C++. So use python's identity check function is sufficient
    same_as = object.__eq__

    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __nonzero__(self):
        return not self.a.same_as(self.b)

    def __bool__(self):
        return self.__nonzero__()

    def asnode(self):
        """Convert node."""
        return _make._OpNE(self.a, self.b)


class Expr(ExprOp, NodeBase):
    """Base class of all tvm Expressions"""
    # In Python3, We have to explicitly tell interpreter to retain __hash__ if we overide __eq__
    # https://docs.python.org/3.1/reference/datamodel.html#object.__hash__
    __hash__ = NodeBase.__hash__


class ConstExpr(Expr):
    pass

class BinaryOpExpr(Expr):
    pass

class CmpExpr(Expr):
    pass

class LogicalExpr(Expr):
    pass

@register_node("Variable")
class Var(Expr):
    """Symbolic variable.

    Parameters
    ----------
    name : str
        The name

    dtype : int
        The data type
    """
    def __init__(self, name, dtype):
        self.__init_handle_by_constructor__(
            _api_internal._Var, name, dtype)


@register_node
class Reduce(Expr):
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
    """
    def __init__(self, combiner, src, rdom, condition, value_index):
        self.__init_handle_by_constructor__(
            _make.Reduce, combiner, src, rdom,
            condition, value_index)


@register_node
class FloatImm(ConstExpr):
    """Float constant.

    Parameters
    ----------
    dtype : str
        The data type

    value : float
        The constant value.
    """
    def __init__(self, dtype, value):
        self.__init_handle_by_constructor__(
            _make.FloatImm, dtype, value)

@register_node
class IntImm(ConstExpr):
    """Int constant.

    Parameters
    ----------
    dtype : str
        The data type

    value : int
        The constant value.
    """
    def __init__(self, dtype, value):
        self.__init_handle_by_constructor__(
            _make.IntImm, dtype, value)

    def __int__(self):
        return self.value


@register_node
class UIntImm(ConstExpr):
    """UInt constant.

    Parameters
    ----------
    dtype : str
        The data type

    value : int
        The constant value.
    """
    def __init__(self, dtype, value):
        self.__init_handle_by_constructor__(
            _make.UIntImm, dtype, value)


@register_node
class StringImm(ConstExpr):
    """String constant.

    Parameters
    ----------
    value : str
        The value of the function.
    """
    def __init__(self, value):
        self.__init_handle_by_constructor__(
            _make.StringImm, value)

    def __eq__(self, other):
        if isinstance(other, ConstExpr):
            return self.value == other.value
        return self.value == other

    def __ne__(self, other):
        if isinstance(other, ConstExpr):
            return self.value != other.value
        return self.value != other


@register_node
class Cast(Expr):
    """Cast expression.

    Parameters
    ----------
    dtype : str
        The data type

    value : Expr
        The value of the function.
    """
    def __init__(self, dtype, value):
        self.__init_handle_by_constructor__(
            _make.Cast, dtype, value)


@register_node
class Add(BinaryOpExpr):
    """Add node.

    Parameters
    ----------
    a : Expr
        The left hand operand.

    b : Expr
        The right hand operand.
    """
    def __init__(self, a, b):
        self.__init_handle_by_constructor__(
            _make.Add, a, b)


@register_node
class Sub(BinaryOpExpr):
    """Sub node.

    Parameters
    ----------
    a : Expr
        The left hand operand.

    b : Expr
        The right hand operand.
    """
    def __init__(self, a, b):
        self.__init_handle_by_constructor__(
            _make.Sub, a, b)


@register_node
class Mul(BinaryOpExpr):
    """Mul node.

    Parameters
    ----------
    a : Expr
        The left hand operand.

    b : Expr
        The right hand operand.
    """
    def __init__(self, a, b):
        self.__init_handle_by_constructor__(
            _make.Mul, a, b)


@register_node
class Div(BinaryOpExpr):
    """Div node.

    Parameters
    ----------
    a : Expr
        The left hand operand.

    b : Expr
        The right hand operand.
    """
    def __init__(self, a, b):
        self.__init_handle_by_constructor__(
            _make.Div, a, b)


@register_node
class Mod(BinaryOpExpr):
    """Mod node.

    Parameters
    ----------
    a : Expr
        The left hand operand.

    b : Expr
        The right hand operand.
    """
    def __init__(self, a, b):
        self.__init_handle_by_constructor__(
            _make.Mod, a, b)


@register_node
class FloorDiv(BinaryOpExpr):
    """FloorDiv node.

    Parameters
    ----------
    a : Expr
        The left hand operand.

    b : Expr
        The right hand operand.
    """
    def __init__(self, a, b):
        self.__init_handle_by_constructor__(
            _make.FloorDiv, a, b)


@register_node
class FloorMod(BinaryOpExpr):
    """FloorMod node.

    Parameters
    ----------
    a : Expr
        The left hand operand.

    b : Expr
        The right hand operand.
    """
    def __init__(self, a, b):
        self.__init_handle_by_constructor__(
            _make.FloorMod, a, b)


@register_node
class Min(BinaryOpExpr):
    """Min node.

    Parameters
    ----------
    a : Expr
        The left hand operand.

    b : Expr
        The right hand operand.
    """
    def __init__(self, a, b):
        self.__init_handle_by_constructor__(
            _make.Min, a, b)


@register_node
class Max(BinaryOpExpr):
    """Max node.

    Parameters
    ----------
    a : Expr
        The left hand operand.

    b : Expr
        The right hand operand.
    """
    def __init__(self, a, b):
        self.__init_handle_by_constructor__(
            _make.Max, a, b)


@register_node
class EQ(CmpExpr):
    """EQ node.

    Parameters
    ----------
    a : Expr
        The left hand operand.

    b : Expr
        The right hand operand.
    """
    def __init__(self, a, b):
        self.__init_handle_by_constructor__(
            _make.EQ, a, b)


@register_node
class NE(CmpExpr):
    """NE node.

    Parameters
    ----------
    a : Expr
        The left hand operand.

    b : Expr
        The right hand operand.
    """
    def __init__(self, a, b):
        self.__init_handle_by_constructor__(
            _make.NE, a, b)


@register_node
class LT(CmpExpr):
    """LT node.

    Parameters
    ----------
    a : Expr
        The left hand operand.

    b : Expr
        The right hand operand.
    """
    def __init__(self, a, b):
        self.__init_handle_by_constructor__(
            _make.LT, a, b)


@register_node
class LE(CmpExpr):
    """LE node.

    Parameters
    ----------
    a : Expr
        The left hand operand.

    b : Expr
        The right hand operand.
    """
    def __init__(self, a, b):
        self.__init_handle_by_constructor__(
            _make.LE, a, b)


@register_node
class GT(CmpExpr):
    """GT node.

    Parameters
    ----------
    a : Expr
        The left hand operand.

    b : Expr
        The right hand operand.
    """
    def __init__(self, a, b):
        self.__init_handle_by_constructor__(
            _make.GT, a, b)


@register_node
class GE(CmpExpr):
    """GE node.

    Parameters
    ----------
    a : Expr
        The left hand operand.

    b : Expr
        The right hand operand.
    """
    def __init__(self, a, b):
        self.__init_handle_by_constructor__(
            _make.GE, a, b)


@register_node
class And(LogicalExpr):
    """And node.

    Parameters
    ----------
    a : Expr
        The left hand operand.

    b : Expr
        The right hand operand.
    """
    def __init__(self, a, b):
        self.__init_handle_by_constructor__(
            _make.And, a, b)


@register_node
class Or(LogicalExpr):
    """Or node.

    Parameters
    ----------
    a : Expr
        The left hand operand.

    b : Expr
        The right hand operand.
    """
    def __init__(self, a, b):
        self.__init_handle_by_constructor__(
            _make.Or, a, b)


@register_node
class Not(LogicalExpr):
    """Not node.

    Parameters
    ----------
    a : Expr
        The input value
    """
    def __init__(self, a):
        self.__init_handle_by_constructor__(
            _make.Not, a)


@register_node
class Select(Expr):
    """Select node.

    Note
    ----
    Select may compute both true_value and false_value.
    Use :any:`tvm.if_then_else` instead if you want to
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

    """
    def __init__(self, condition, true_value, false_value):
        self.__init_handle_by_constructor__(
            _make.Select, condition, true_value, false_value)


@register_node
class Load(Expr):
    """Load node.

    Parameters
    ----------
    dtype : str
        The data type.

    buffer_var : Var
        The buffer variable in the load expression.

    index : Expr
        The index in the load.

    predicate : Expr
        The load predicate.
    """
    def __init__(self, dtype, buffer_var, index, predicate):
        self.__init_handle_by_constructor__(
            _make.Load, dtype, buffer_var, index, predicate)


@register_node
class Ramp(Expr):
    """Ramp node.

    Parameters
    ----------
    base : Expr
        The base expression.

    stride : ramp stride
        The stride of the ramp.

    lanes : int
        The lanes of the expression.
    """
    def __init__(self, base, stride, lanes):
        self.__init_handle_by_constructor__(
            _make.Ramp, base, stride, lanes)


@register_node
class Broadcast(Expr):
    """Broadcast node.

    Parameters
    ----------
    value : Expr
        The value of the expression.

    lanes : int
        The lanes of the expression.
    """
    def __init__(self, value, lanes):
        self.__init_handle_by_constructor__(
            _make.Broadcast, value, lanes)


@register_node
class Shuffle(Expr):
    """Shuffle node.

    Parameters
    ----------
    vectors : Array of Expr
        The vectors

    indices : Array of indices
        The indices
    """
    def __init__(self, vectors, indices):
        self.__init_handle_by_constructor__(
            _make.Shuffle, vectors, indices)


@register_node
class Call(Expr):
    """Call node.

    Parameters
    ----------
    dtype : str
        The return data type

    name : str
        The name of the function

    args : list of Expr
        The input arguments to the call

    call_type : int
        The type of the call

    func : Operation, optional
        Operation if call_type is Halide

    value_index : int
        The output value index
    """
    Extern = 0
    ExternCPlusPlus = 1
    PureExtern = 2
    Halide = 3
    Intrinsic = 4
    PureIntrinsic = 5
    def __init__(self, dtype, name, args, call_type, func, value_index):
        self.__init_handle_by_constructor__(
            _make.Call, dtype, name, args, call_type, func, value_index)


@register_node
class Let(Expr):
    """Let node.

    Parameters
    ----------
    var : Var
        The variable in the binding.

    value : Expr
        The value in to be binded.

    body : Expr
        The body expression.
    """
    def __init__(self, var, value, body):
        self.__init_handle_by_constructor__(
            _make.Let, var, value, body)
