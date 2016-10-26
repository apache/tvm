from __future__ import absolute_import as _abs
from ._ctypes._api import NodeBase, register_node
from . import function as _func

class Expr(NodeBase):
    def __repr__(self):
        return _func.format_str(self)

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


@register_node("IntImm")
class IntImm(Expr):
    pass

@register_node("UIntImm")
class UIntImm(Expr):
    pass

@register_node("FloatImm")
class FloatImm(Expr):
    pass
