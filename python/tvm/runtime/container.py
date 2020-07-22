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
"""Runtime container structures."""
import tvm._ffi
from .object import Object, PyNativeObject
from .object_generic import ObjectTypes
from . import _ffi_api


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
        raise IndexError("Index out of range. size: {}, got index {}"
                         .format(length, idx))
    if idx < 0:
        idx += length
    return elem_getter(obj, idx)


@tvm._ffi.register_object("runtime.ADT")
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
            assert isinstance(f, ObjectTypes), "Expect object or " \
            "tvm NDArray type, but received : {0}".format(type(f))
        self.__init_handle_by_constructor__(_ffi_api.ADT, tag,
                                            *fields)

    @property
    def tag(self):
        return _ffi_api.GetADTTag(self)

    def __getitem__(self, idx):
        return getitem_helper(
            self, _ffi_api.GetADTFields, len(self), idx)

    def __len__(self):
        return _ffi_api.GetADTSize(self)


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
        assert isinstance(f, ObjectTypes), "Expect object or tvm " \
        "NDArray type, but received : {0}".format(type(f))
    return _ffi_api.Tuple(*fields)


@tvm._ffi.register_object("runtime.String")
class String(str, PyNativeObject):
    """TVM runtime.String object, represented as a python str.

    Parameters
    ----------
    content : str
        The content string used to construct the object.
    """
    __slots__ = ["__tvm_object__"]

    def __new__(cls, content):
        """Construct from string content."""
        val = str.__new__(cls, content)
        val.__init_tvm_object_by_constructor__(_ffi_api.String, content)
        return val

    # pylint: disable=no-self-argument
    def __from_tvm_object__(cls, obj):
        """Construct from a given tvm object."""
        content = _ffi_api.GetFFIString(obj)
        val = str.__new__(cls, content)
        val.__tvm_object__ = obj
        return val
