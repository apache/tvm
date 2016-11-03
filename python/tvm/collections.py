"""Collection structure in the high level DSL."""
from __future__ import absolute_import as _abs
from ._ctypes._api import NodeBase, register_node
from . import _function_internal

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
class Range(NodeBase):
    def __repr__(self):
        return ('Range(min='+ str(self.min) +
                ', extent=' + str(self.extent) + ')')


@register_node
class RDomain(NodeBase):
    pass
