# pylint: disable=protected-access, no-member
"""Arithmetic"""
from __future__ import absolute_import as _abs
from ._ctypes._node import NodeBase, register_node

@register_node
class IntSet(NodeBase):
    pass

@register_node
class IntervalSet(IntSet):
    pass

@register_node
class StrideSet(IntSet):
    pass

