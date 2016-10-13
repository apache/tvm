from __future__ import absolute_import as _abs
from .expr import Expr

class Var(Expr):
    """Variables"""
    def __init__(self, name, expr=None):
        self.name = name
        self.expr = expr

    def children(self):
        if self.expr is None:
            return ()
        return self.expr.children()
