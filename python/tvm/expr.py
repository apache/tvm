from __future__ import absolute_import as _abs
from ._ctypes._api import NodeBase, register_node
from . import make as _make

class ExprCompatible(NodeBase):
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

    def __neg__(self):
        return self.__mul__(-1)


class Expr(ExprCompatible):
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
class Variable(Expr):
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
    pass

@register_node
class Let(Expr):
    pass
