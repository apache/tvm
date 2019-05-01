# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
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
            if start < 0:
                start += len(self)
            if stop < 0:
                stop += len(self)
            return [self[idx] for idx in range(start, stop, step)]

        if i < -len(self) or i >= len(self):
            raise IndexError("Array index out of range. Array size: {}, got index {}"
                             .format(len(self), i))
        if i < 0:
            i += len(self)
        return _api_internal._ArrayGetItem(self, i)

    def __len__(self):
        return _api_internal._ArraySize(self)


@register_node
class EnvFunc(NodeBase):
    """Environment function.

    This is a global function object that can be serialized by its name.
    """
    def __call__(self, *args):
        return _api_internal._EnvFuncCall(self, *args)

    @property
    def func(self):
        return _api_internal._EnvFuncGetPackedFunc(self)


@register_node
class Map(NodeBase):
    """Map container of TVM.

    You do not need to create Map explicitly.
    Normally python dict will be converted automaticall to Map during tvm function call.
    You can use convert to create a dict[NodeBase-> NodeBase] into a Map
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
class StrMap(Map):
    """A special map container that has str as key.

    You can use convert to create a dict[str->NodeBase] into a Map.
    """
    def items(self):
        """Get the items from the map"""
        akvs = _api_internal._MapItems(self)
        return [(akvs[i].value, akvs[i+1]) for i in range(0, len(akvs), 2)]


@register_node
class Range(NodeBase):
    """Represent a range in TVM.

    You do not need to create a Range explicitly.
    Python lists and tuples will be converted automatically to a Range in API functions.
    """


@register_node
class LoweredFunc(NodeBase):
    """Represent a LoweredFunc in TVM."""
    MixedFunc = 0
    HostFunc = 1
    DeviceFunc = 2
