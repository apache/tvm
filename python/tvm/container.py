"""Container data structures used in TVM DSL."""
from __future__ import absolute_import as _abs
from ._ffi.node import NodeBase, register_node
from . import _api_internal

@register_node
class Array(NodeBase):
    """Array container of TVM.

    You do not need to create Array explicitly.
    Normally python list and tuple will be converted automatically
    to Array during tvm function call.
    You may get Array in return values of TVM function call.
    """
    def __getitem__(self, i):
        if isinstance(i, slice):
            start = i.start if i.start is not None else 0
            stop = i.stop if i.stop is not None else len(self)
            step = i.step if i.step is not None else 1
            return [self[idx] for idx in range(start, stop, step)]

        if i >= len(self):
            raise IndexError("array index out of range")
        return _api_internal._ArrayGetItem(self, i)

    def __len__(self):
        return _api_internal._ArraySize(self)


@register_node
class Map(NodeBase):
    """Map container of TVM.

    You do not need to create Map explicitly.
    Normally python dict will be converted automatically
    to Array during tvm function call.
    You may get Map in return values of TVM function call.
    """
    def __getitem__(self, k):
        return _api_internal._MapGetItem(self, k)

    def __contains__(self, k):
        return _api_internal._MapCount(self, k) != 0

    def items(self):
        """Get the items from the map"""
        akvs = _api_internal._MapItems(self)
        return [(akvs[i], akvs[i+1]) for i in range(0, len(akvs), 2)]

    def __len__(self):
        return _api_internal._MapSize(self)


@register_node
class Range(NodeBase):
    """Represent range in TVM.

    You do not need to create Range explicitly.
    Python list and tuple will be converted automatically to Range in api functions.
    """
    pass

@register_node
class LoweredFunc(NodeBase):
    """Represent a LoweredFunc in TVM."""
    MixedFunc = 0
    HostFunc = 1
    DeviceFunc = 2
