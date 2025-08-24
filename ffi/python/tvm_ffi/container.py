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
"""Container classes."""
import collections.abc

from typing import Any, Mapping, Sequence
from . import core
from . import _ffi_api
from .registry import register_object

__all__ = ["Array", "Map"]


def getitem_helper(obj, elem_getter, length, idx):
    """Helper function to implement a pythonic getitem function.

    Parameters
    ----------
    obj: object
        The original object

    elem_getter : function
        A simple function that takes index and return a single element.

    length : int
        The size of the array

    idx : int or slice
        The argument passed to getitem

    Returns
    -------
    result : object
        The result of getitem
    """
    if isinstance(idx, slice):
        start = idx.start if idx.start is not None else 0
        stop = idx.stop if idx.stop is not None else length
        step = idx.step if idx.step is not None else 1
        if start < 0:
            start += length
        if stop < 0:
            stop += length
        return [elem_getter(obj, i) for i in range(start, stop, step)]

    if idx < -length or idx >= length:
        raise IndexError(f"Index out of range. size: {length}, got index {idx}")
    if idx < 0:
        idx += length
    return elem_getter(obj, idx)


@register_object("ffi.Array")
class Array(core.Object, collections.abc.Sequence):
    """Array container"""

    def __init__(self, input_list: Sequence[Any]):
        self.__init_handle_by_constructor__(_ffi_api.Array, *input_list)

    def __getitem__(self, idx):
        return getitem_helper(self, _ffi_api.ArrayGetItem, len(self), idx)

    def __len__(self):
        return _ffi_api.ArraySize(self)

    def __repr__(self):
        # exception safety handling for chandle=None
        if self.__chandle__() == 0:
            return type(self).__name__ + "(chandle=None)"
        return "[" + ", ".join([x.__repr__() for x in self]) + "]"


class KeysView(collections.abc.KeysView):
    """Helper class to return keys view"""

    def __init__(self, backend_map):
        self._backend_map = backend_map

    def __len__(self):
        return len(self._backend_map)

    def __iter__(self):
        if self.__len__() == 0:
            return
        functor = _ffi_api.MapForwardIterFunctor(self._backend_map)
        while True:
            k = functor(0)
            yield k
            if not functor(2):
                break

    def __contains__(self, k):
        return self._backend_map.__contains__(k)


class ValuesView(collections.abc.ValuesView):
    """Helper class to return values view"""

    def __init__(self, backend_map):
        self._backend_map = backend_map

    def __len__(self):
        return len(self._backend_map)

    def __iter__(self):
        if self.__len__() == 0:
            return
        functor = _ffi_api.MapForwardIterFunctor(self._backend_map)
        while True:
            v = functor(1)
            yield v
            if not functor(2):
                break


class ItemsView(collections.abc.ItemsView):
    """Helper class to return items view"""

    def __init__(self, backend_map):
        self.backend_map = backend_map

    def __len__(self):
        return len(self.backend_map)

    def __iter__(self):
        if self.__len__() == 0:
            return
        functor = _ffi_api.MapForwardIterFunctor(self.backend_map)
        while True:
            k = functor(0)
            v = functor(1)
            yield (k, v)
            if not functor(2):
                break


@register_object("ffi.Map")
class Map(core.Object, collections.abc.Mapping):
    """Map container."""

    def __init__(self, input_dict: Mapping[Any, Any]):
        list_kvs = []
        for k, v in input_dict.items():
            list_kvs.append(k)
            list_kvs.append(v)
        self.__init_handle_by_constructor__(_ffi_api.Map, *list_kvs)

    def __getitem__(self, k):
        return _ffi_api.MapGetItem(self, k)

    def __contains__(self, k):
        return _ffi_api.MapCount(self, k) != 0

    def keys(self):
        return KeysView(self)

    def values(self):
        return ValuesView(self)

    def items(self):
        """Get the items from the map"""
        return ItemsView(self)

    def __len__(self):
        return _ffi_api.MapSize(self)

    def __iter__(self):
        return iter(self.keys())

    def get(self, key, default=None):
        """Get an element with a default value.

        Parameters
        ----------
        key : object
            The attribute key.

        default : object
            The default object.

        Returns
        -------
        value: object
            The result value.
        """
        return self[key] if key in self else default

    def __repr__(self):
        # exception safety handling for chandle=None
        if self.__chandle__() == 0:
            return type(self).__name__ + "(chandle=None)"
        return "{" + ", ".join([f"{k.__repr__()}: {v.__repr__()}" for k, v in self.items()]) + "}"
