from __future__ import absolute_import as _abs

_binary_op_cls = None

class BinaryOp(object):
    """Base class of binary operator"""
    def __call__(self, lhs, rhs):
        return _binary_op_cls(self, lhs, rhs)

class AddOp(BinaryOp):
    def format_str(self, lhs, rhs):
        return '(%s + %s)' % (lhs, rhs)

class SubOp(BinaryOp):
    def format_str(self, lhs, rhs):
        return '(%s - %s)' % (lhs, rhs)

class MulOp(BinaryOp):
    def format_str(self, lhs, rhs):
        return '(%s * %s)' % (lhs, rhs)

class DivOp(BinaryOp):
    def format_str(self, lhs, rhs):
        return '(%s / %s)' % (lhs, rhs)


add = AddOp()
sub = SubOp()
mul = MulOp()
div = DivOp()
