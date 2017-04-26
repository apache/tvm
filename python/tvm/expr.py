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
from ._ctypes._node import NodeBase, register_node
from . import _api_internal
from . import make as _make

class ExprOp(object):
    def __add__(self, other):
        return _make.Add(self, other)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return _make.Sub(self, other)

    def __rsub__(self, other):
        return _make.Sub(other, self)

    def __mul__(self, other):
        return _make.Mul(self, other)

    def __rmul__(self, other):
        return _make.Mul(other, self)

    def __div__(self, other):
        return _make.Div(self, other)

    def __rdiv__(self, other):
        return _make.Div(other, self)

    def __truediv__(self, other):
        return self.__div__(other)

    def __rtruediv__(self, other):
        return self.__rdiv__(other)

    def __mod__(self, other):
        return _make.Mod(self, other)

    def __neg__(self):
        return self.__mul__(-1)

    def __lt__(self, other):
        return _make.LT(self, other)

    def __le__(self, other):
        return _make.LE(self, other)

    def __eq__(self, other):
        return _make.EQ(self, other)

    def __ne__(self, other):
        return _make.NE(self, other)

    def __gt__(self, other):
        return _make.GT(self, other)

    def __ge__(self, other):
        return _make.GE(self, other)


class Expr(NodeBase, ExprOp):
    """Base class of all tvm Expressions"""
    pass

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

@register_node
class IterVar(NodeBase, ExprOp):
    """Represent iteration variable.

    IterVar is normally created by Operation, to represent
    axis iterations in the computation.
    It can also created by schedule primitives like :any:`tvm.schedule.Stage.split`.

    See Also
    --------
    tvm.thread_axis: Create thread axis IterVar.
    tvm.reduce_axis: Create reduce axis IterVar.
    """
    DataPar = 0
    ThreadIndex = 1
    CommReduce = 2
    Ordered = 3
    DimInfo = 4
    Unrolled = 5
    Vectorized = 6
    Parallelized = 7

@register_node
class CommReducerNode(NodeBase):
    """Represent a general communicative reduce node."""
    def combine(self, lhs, rhs):
        return _api_internal._CommReducerCombine(self, lhs, rhs)

    def _make_reduce(self, expr, axis, where=None):
        axis = axis if isinstance(axis, list) else [axis]
        return _make.Reduce(self, expr, axis, where)

    def _reduce_directly(self, *args):
        num = len(args)
        # process `where` is None
        if num == 3 and args[2] is None:
            num = 2
        res = args[0]
        for i in range(num-1):
          res = self.combine(res, args[i+1])
        return res

    def _is_axis(self, a):
        return isinstance(a, IterVar) or isinstance(a, list)

    def __call__(self, *args, **kwargs):
      if len(kwargs) == 0 and len(args) >= 2 and \
        not isinstance(args[1], (IterVar, list)):
          return self._reduce_directly(*args)
      else:
          return self._make_reduce(*args, **kwargs)
