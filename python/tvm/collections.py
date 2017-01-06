"""Collection structure in the high level DSL."""
from __future__ import absolute_import as _abs
from ._ctypes._api import NodeBase, register_node
from . import _function_internal
from . import expr as _expr

@register_node
class Array(NodeBase):
    def __getitem__(self, i):
        if i >= len(self):
            raise IndexError("array index out ot range")
        return _function_internal._ArrayGetItem(self, i)

    def __len__(self):
        return _function_internal._ArraySize(self)

    def __repr__(self):
        return '[' + (','.join(str(x) for x in self)) + ']'

@register_node
class Map(NodeBase):
    def __getitem__(self, k):
        return _function_internal._MapGetItem(self, k)

    def __contains__(self, k):
        return _function_internal._MapCount(self, k) != 0

    def items(self):
        akvs = _function_internal._MapItems(self)
        return [(akvs[i], akvs[i+1]) for i in range(0, len(akvs), 2)]

    def __len__(self):
        return _function_internal._MapSize(self)

    def __repr__(self):
        return '{' + (", ".join(str(x[0]) + ": " +str(x[1]) for x in self.items())) + '}'


@register_node
class Range(NodeBase):
    pass


@register_node
class IterVar(NodeBase, _expr.ExprOp):
    pass
