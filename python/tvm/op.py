from __future__ import absolute_import as _abs
from . import expr as _expr

constant_canonical_key = '__constant__'

def canonical_to_expr(c):
    elements = []
    for k, v in sorted(c.items()):
        if k == constant_canonical_key:
            elements.append(_expr.const(v))
        elif v == 0:
            continue
        elif v == 1:
            elements.append(k)
        else:
            elements.append(k * v)
    if elements:
        expr = elements[0]
        for i in range(1, len(elements)):
            expr = expr + elements[i]
        return expr
    else:
        return _expr.const(0)

class BinaryOp(object):
    """Base class of binary operator"""
    def __call__(self, lhs, rhs):
        return _expr.BinaryOpExpr(self, lhs, rhs)

class AddOp(BinaryOp):
    def format_str(self, lhs, rhs):
        return '(%s + %s)' % (lhs, rhs)

    def canonical(self, lhs, rhs):
        lhs = lhs.copy()
        for k, v in rhs.items():
            if k in lhs:
                lhs[k] += v
            else:
                lhs[k] = v
        return lhs

class SubOp(BinaryOp):
    def format_str(self, lhs, rhs):
        return '(%s - %s)' % (lhs, rhs)

    def canonical(self, lhs, rhs):
        lhs = lhs.copy()
        for k, v in rhs.items():
            if k in lhs:
                lhs[k] -= v
            else:
                lhs[k] = -v
        return lhs

class MulOp(BinaryOp):
    def format_str(self, lhs, rhs):
        return '(%s * %s)' % (lhs, rhs)

    def canonical(self, lhs, rhs):
        elhs = canonical_to_expr(lhs)
        erhs = canonical_to_expr(rhs)
        if isinstance(erhs, _expr.ConstExpr):
            lhs = lhs.copy()
            for k, v in lhs.items():
                lhs[k] *= erhs.value
            return lhs
        if isinstance(elhs, _expr.ConstExpr):
            rhs = rhs.copy()
            for k, v in rhs.items():
                rhs[k] *= elhs.value
            return rhs
        return {elhs * erhs: 1}

class DivOp(BinaryOp):
    def format_str(self, lhs, rhs):
        return '(%s / %s)' % (lhs, rhs)

    def canonical(self, lhs, rhs):
        erhs = canonical_to_expr(rhs)
        if isinstance(erhs, _expr.ConstExpr):
            lhs = lhs.copy()
            for k, v in lhs.items():
                lhs[k] /= erhs.value
            return lhs
        elhs = canonical_to_expr(lhs)
        return {elhs / erhs: 1}

class MaxOp(BinaryOp):
    def format_str(self, lhs, rhs):
        return 'max(%s, %s)' % (lhs, rhs)

    def canonical(self, lhs, rhs):
        diff = SubOp().canonical(lhs, rhs)
        ediff = canonical_to_expr(diff)
        if isinstance(ediff, _expr.ConstExpr):
            return lhs if ediff.value >= 0 else rhs
        return {MaxOp()(lhs, rhs): 1}

class MinOp(BinaryOp):
    def format_str(self, lhs, rhs):
        return 'min(%s, %s)' % (lhs, rhs)

    def canonical(self, lhs, rhs):
        diff = SubOp().canonical(lhs, rhs)
        ediff = canonical_to_expr(diff)
        if isinstance(ediff, _expr.ConstExpr):
            return rhs if ediff.value >= 0 else lhs
        return {MinOp()(lhs, rhs): 1}


add = AddOp()
sub = SubOp()
mul = MulOp()
div = DivOp()
max = MaxOp()
min = MinOp()

_expr.__addop__ = add
_expr.__subop__ = sub
_expr.__mulop__ = mul
_expr.__divop__ = div
