"""Collection structure in the high level DSL."""
from __future__ import absolute_import as _abs
from ._ctypes._api import NodeBase, register_node
from . import _function_internal

@register_node
class DimSplit(NodeBase):
    pass

@register_node
class AttachSpec(NodeBase):
    pass

@register_node
class Schedule(NodeBase):
    pass
