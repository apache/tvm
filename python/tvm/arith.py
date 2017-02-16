# pylint: disable=protected-access, no-member
"""Arithmetic data structure and utility"""
from __future__ import absolute_import as _abs
from ._ctypes._node import NodeBase, register_node
from . import _api_internal

@register_node
class IntSet(NodeBase):
    pass

@register_node
class IntervalSet(IntSet):
    def min(self):
        return _api_internal._IntervalSetGetMin(self)

    def max(self):
        return _api_internal._IntervalSetGetMax(self)

@register_node
class StrideSet(IntSet):
    pass

