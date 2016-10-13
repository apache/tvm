"""Base class of symbolic expression"""
from __future__ import absolute_import as _abs
from numbers import Number as _Number
from . import op as _op
from . import var_name as _name

class Expr(object):
    """Base class of expression.

    Expression object should be in general immutable.
    """

    def children(self):
        """get children of this expression.

        Returns
        -------
        children : generator of children
        """
        return ()

    def __add__(self, other):
        return BinaryOpExpr(_op.add, self, other)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return BinaryOpExpr(_op.sub, self, other)

    def __rsub__(self, other):
        return BinaryOpExpr(_op.sub, other, self)

    def __mul__(self, other):
        return BinaryOpExpr(_op.mul, self, other)

    def __rmul__(self, other):
        return BinaryOpExpr(_op.mul, other, self)

    def __div__(self, other):
        return BinaryOpExpr(_op.div, self, other)

    def __rdiv__(self, other):
        return BinaryOpExpr(_op.div, other, self)

    def __truediv__(self, other):
        return self.__div__(other)

    def __rtruediv__(self, other):
        return self.__rdiv__(other)

    def __neg__(self):
        return self.__mul__(-1)


def _symbol(value):
    """Convert a value to expression."""
    if isinstance(value, Expr):
        return value
    elif isinstance(value, _Number):
        return ConstExpr(value)
    else:
        raise TypeError("type %s not supported" % str(type(other)))


class Var(Expr):
    """Variable, is a symbolic placeholder.

    Each variable is uniquely identified by its address
    Note that name alone is not able to uniquely identify the var.

    Parameters
    ----------
    name : str
        optional name to the var.
    """
    def __init__(self, name=None):
        self.name = name if name else _name.NameManager.current.get(name)


class ConstExpr(Expr):
    """Constant expression."""
    def __init__(self, value):
        assert isinstance(value, _Number)
        self.value = value


class BinaryOpExpr(Expr):
    """Binary operator expression."""
    def __init__(self, op, lhs, rhs):
        self.op = op
        self.lhs = _symbol(lhs)
        self.rhs = _symbol(rhs)

    def children(self):
        return (self.lhs, self.rhs)

_op.binary_op_cls = BinaryOpExpr

class UnaryOpExpr(Expr):
    """Unary operator expression."""
    def __init__(self, op, src):
        self.op = op
        self.src = _symbol(src)

    def children(self):
        return (self.src)


def const(value):
    """Return a constant value"""
    return ConstExpr(value)
