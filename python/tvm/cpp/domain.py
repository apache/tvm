from __future__ import absolute_import as _abs
from ._ctypes._api import NodeBase, register_node
from . import _function_internal

@register_node("RangeNode")
class Range(NodeBase):
    pass


@register_node("ArrayNode")
class Array(NodeBase):
    def __getitem__(self, i):
        return _function_internal._ArrayGetItem(self, i)

    def __len__(self):
        return _function_internal._ArraySize(self)
