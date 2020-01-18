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
from tvm import ndarray as _nd
from . import _api_internal
from ._ffi.object import Object, register_object, getitem_helper
from ._ffi.function import _init_api

@register_object
class Array(Object):
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


@register_object
class EnvFunc(Object):
    """Environment function.

    This is a global function object that can be serialized by its name.
    """
    def __call__(self, *args):
        return _api_internal._EnvFuncCall(self, *args)

    @property
    def func(self):
        return _api_internal._EnvFuncGetPackedFunc(self)


@register_object
class Map(Object):
    """Map container of TVM.

    You do not need to create Map explicitly.
    Normally python dict will be converted automaticall to Map during tvm function call.
    You can use convert to create a dict[Object-> Object] into a Map
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


@register_object
class StrMap(Map):
    """A special map container that has str as key.

    You can use convert to create a dict[str->Object] into a Map.
    """
    def items(self):
        """Get the items from the map"""
        akvs = _api_internal._MapItems(self)
        return [(akvs[i].value, akvs[i+1]) for i in range(0, len(akvs), 2)]


@register_object
class Range(Object):
    """Represent a range in TVM.

    You do not need to create a Range explicitly.
    Python lists and tuples will be converted automatically to a Range in API functions.
    """


@register_object
class LoweredFunc(Object):
    """Represent a LoweredFunc in TVM."""
    MixedFunc = 0
    HostFunc = 1
    DeviceFunc = 2


@register_object("vm.ADT")
class ADT(Object):
    """Algebatic data type(ADT) object.

    Parameters
    ----------
    tag : int
        The tag of ADT.

    fields : list[Object] or tuple[Object]
        The source tuple.
    """
    def __init__(self, tag, fields):
        for f in fields:
            assert isinstance(f, (Object, _nd.NDArray)), "Expect object or " \
            "tvm NDArray type, but received : {0}".format(type(f))
        self.__init_handle_by_constructor__(_ADT, tag, *fields)

    @property
    def tag(self):
        return _GetADTTag(self)

    def __getitem__(self, idx):
        return getitem_helper(
            self, _GetADTFields, len(self), idx)

    def __len__(self):
        return _GetADTSize(self)


def tuple_object(fields=None):
    """Create a ADT object from source tuple.

    Parameters
    ----------
    fields : list[Object] or tuple[Object]
        The source tuple.

    Returns
    -------
    ret : ADT
        The created object.
    """
    fields = fields if fields else []
    for f in fields:
        assert isinstance(f, (Object, _nd.NDArray)), "Expect object or tvm " \
        "NDArray type, but received : {0}".format(type(f))
    return _Tuple(*fields)


_init_api("tvm.container")
