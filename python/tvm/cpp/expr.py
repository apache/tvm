from __future__ import absolute_import as _abs
from ._ctypes._api import NodeBase, register_node
from .function import binary_op

class Expr(NodeBase):
    def __add__(self, other):
        return binary_op('+', self, other)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return binary_op('-', self, other)

    def __rsub__(self, other):
        return binary_op('-', other, self)

    def __mul__(self, other):
        return binary_op('*', self, other)

    def __rmul__(self, other):
        return binary_op('*', other, self)

    def __div__(self, other):
        return binary_op('/', self, other)

    def __rdiv__(self, other):
        return binary_op('/', other, self)

    def __truediv__(self, other):
        return self.__div__(other)

    def __rtruediv__(self, other):
        return self.__rdiv__(other)

    def __neg__(self):
        return self.__mul__(-1)


@register_node("VarNode")
class Var(Expr):
    pass

@register_node("IntNode")
class IntExpr(Expr):
    pass

@register_node("FloatNode")
class FloatExpr(Expr):
    pass

@register_node("UnaryOpNode")
class UnaryOpExpr(Expr):
    pass

@register_node("BinaryOpNode")
class BinaryOpExpr(Expr):
    pass

@register_node("ReduceNode")
class ReduceExpr(Expr):
    pass

@register_node("TensorReadNode")
class TensorReadExpr(Expr):
    pass
