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
    pass

@register_node
class ModularSet(IntSet):
    """Represent range of (coeff * x + base) for x in Z """
    pass

_init_api("tvm.arith")
