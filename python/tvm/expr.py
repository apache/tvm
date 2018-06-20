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
        return _make.Mod(self, other)

    def __neg__(self):
        neg_one = _api_internal._const(-1, self.dtype)
        return self.__mul__(neg_one)

    def __lshift__(self, other):
        return _make.Call(self.dtype, "shift_left", [self, other], Call.PureIntrinsic, None, 0)

    def __rshift__(self, other):
        return _make.Call(self.dtype, "shift_right", [self, other], Call.PureIntrinsic, None, 0)

    def __and__(self, other):
        return _make.Call(self.dtype, "bitwise_and", [self, other], Call.PureIntrinsic, None, 0)

    def __or__(self, other):
        return _make.Call(self.dtype, "bitwise_or", [self, other], Call.PureIntrinsic, None, 0)

    def __xor__(self, other):
        return _make.Call(self.dtype, "bitwise_xor", [self, other], Call.PureIntrinsic, None, 0)

    def __invert__(self):
        return _make.Call(self.dtype, "bitwise_not", [self], Call.PureIntrinsic, None, 0)

    def __lt__(self, other):
        return _make.LT(self, other)

    def __le__(self, other):
        return _make.LE(self, other)

    def __eq__(self, other):
        return EqualOp(self, other)

    def __ne__(self, other):
        return NotEqualOp(self, other)

    def __gt__(self, other):
        return _make.GT(self, other)

    def __ge__(self, other):
        return _make.GE(self, other)

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
        return _make.EQ(self, other)

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
        return _make.static_cast(dtype, self)


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
        return _make.EQ(self.a, self.b)


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
        return _make.NE(self.a, self.b)


class Expr(ExprOp, NodeBase):
    """Base class of all tvm Expressions"""
    # In Python3, We have to explicity tell interpreter to retain __hash__ if we overide __eq__
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
    """Symbolic variable."""
    pass

@register_node
class Reduce(Expr):
    pass

@register_node
class FloatImm(ConstExpr):
    pass

@register_node
class IntImm(ConstExpr):
    pass

@register_node
class UIntImm(ConstExpr):
    pass

@register_node
class StringImm(ConstExpr):
    pass

@register_node
class Cast(Expr):
    pass

@register_node
class Add(BinaryOpExpr):
    pass

@register_node
class Sub(BinaryOpExpr):
    pass

@register_node
class Mul(BinaryOpExpr):
    pass

@register_node
class Div(BinaryOpExpr):
    pass

@register_node
class Mod(BinaryOpExpr):
    pass

@register_node
class Min(BinaryOpExpr):
    pass

@register_node
class Max(BinaryOpExpr):
    pass

@register_node
class EQ(CmpExpr):
    pass

@register_node
class NE(CmpExpr):
    pass

@register_node
class LT(CmpExpr):
    pass

@register_node
class LE(CmpExpr):
    pass

@register_node
class GT(CmpExpr):
    pass

@register_node
class GE(CmpExpr):
    pass

@register_node
class And(LogicalExpr):
    pass

@register_node
class Or(LogicalExpr):
    pass

@register_node
class Not(LogicalExpr):
    pass

@register_node
class Select(Expr):
    pass

@register_node
class Load(Expr):
    pass

@register_node
class Ramp(Expr):
    pass

@register_node
class Broadcast(Expr):
    pass

@register_node
class Shuffle(Expr):
    pass

@register_node
class Call(Expr):
    Extern = 0
    ExternCPlusPlus = 1
    PureExtern = 2
    Halide = 3
    Intrinsic = 4
    PureIntrinsic = 5


@register_node
class Let(Expr):
    pass
