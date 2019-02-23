"""Arithmetic data structure and utility"""
from __future__ import absolute_import as _abs

from ._ffi.node import NodeBase, register_node
from ._ffi.function import _init_api
from . import _api_internal

class IntSet(NodeBase):
    """Represent a set of integer in one dimension."""
    def is_nothing(self):
        """Whether the set represent nothing"""
        return _api_internal._IntSetIsNothing(self)

    def is_everything(self):
        """Whether the set represent everything"""
        return _api_internal._IntSetIsEverything(self)


@register_node
class IntervalSet(IntSet):
    """Represent set of continuous interval"""
    def min(self):
        """get the minimum value"""
        return _api_internal._IntervalSetGetMin(self)

    def max(self):
        """get the maximum value"""
        return _api_internal._IntervalSetGetMax(self)


@register_node
class StrideSet(IntSet):
    """Represent set of strided integers"""


@register_node
class ModularSet(IntSet):
    """Represent range of (coeff * x + base) for x in Z """



@register_node("arith.ConstIntBound")
class ConstIntBound(IntSet):
    """Represent constant integer bound

    Parameters
    ----------
    min_value : int
        The minimum value of the bound.

    max_value : int
        The maximum value of the bound.
    """
    POS_INF = (1 << 63) - 1
    NEG_INF = -POS_INF

    def __init__(self, min_value, max_value):
        self.__init_handle_by_constructor__(
            _make_ConstIntBound, min_value, max_value)


class Analyzer:
    """Integer arithmetic analyzer

    This is a stateful analyzer class that can
    be used to perform various symbolic integer analysis.
    """
    def __init__(self):
        _mod = _CreateAnalyzer()
        self._const_int_bound = _mod("const_int_bound")
        self._const_int_bound_update = _mod("const_int_bound_update")

    def const_int_bound(self, expr):
        """Find constant integer bound for expr.

        Parameters
        ----------
        expr : tvm.Expr
            The expression.

        Returns
        -------
        bound : ConstIntBound
            The result bound
        """
        return self._const_int_bound(expr)

    def update(self, var, info, override=False):
        """Update infomation about var

        Parameters
        ----------
        var : tvm.Var
            The variable.

        info : tvm.NodeBase
            Related information.

        override : bool
            Whether allow override.
        """
        if isinstance(info, ConstIntBound):
            self._const_int_bound_update(var, info, override)
        else:
            raise TypeError(
                "Do not know how to handle type {}".format(type(info)))


_init_api("tvm.arith")
