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
from tvm._ffi.base import string_types
from tvm.runtime import Object, ObjectTypes
from tvm.runtime import _ffi_api

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


@tvm._ffi.register_object("vm.ADT")
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
class String(Object):
    """The string object.

    Parameters
    ----------
    string : str
        The string used to construct a runtime String object

    Returns
    -------
    ret : String
        The created object.
    """
    def __init__(self, string):
        self.__init_handle_by_constructor__(_ffi_api.String, string)

    def __str__(self):
        return _ffi_api.GetStdString(self)

    def __len__(self):
        return _ffi_api.GetStringSize(self)

    def __hash__(self):
        return _ffi_api.StringHash(self)

    def __eq__(self, other):
        if isinstance(other, string_types):
            return self.__str__() == other

        if not isinstance(other, String):
            return False

        return _ffi_api.CompareString(self, other) == 0

    def __ne__(self, other):
        return not self.__eq__(other)

    def __gt__(self, other):
        return _ffi_api.CompareString(self, other) > 0

    def __lt__(self, other):
        return _ffi_api.CompareString(self, other) < 0

    def __getitem__(self, key):
        return self.__str__()[key]

    def startswith(self, string):
        """Check if the runtime string starts with a given string

        Parameters
        ----------
        string : str
            The provided string

        Returns
        -------
        ret : boolean
            Return true if the runtime string starts with the given string,
        otherwise, false.
        """
        return self.__str__().startswith(string)
