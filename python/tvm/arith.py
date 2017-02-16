# pylint: disable=protected-access, no-member
"""Arithmetic data structure and utility"""
from __future__ import absolute_import as _abs
from ._ctypes._node import NodeBase, register_node
from . import _api_internal

@register_node
class IntSet(NodeBase):
    """Represent a set of integer in one dimension."""
    pass

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

