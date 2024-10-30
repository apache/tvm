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
""" TVM Attribute module, which is mainly used for defining attributes of operators."""
import tvm._ffi

from tvm.runtime import Object
import tvm.runtime._ffi_node_api
from . import _ffi_api


@tvm._ffi.register_object
class Attrs(Object):
    """Attribute node, which is mainly use for defining attributes of relay operators.

    Used by function registered in python side, such as compute, schedule and alter_layout.
    Attrs is passed as the first argument to these functions.
    """

    def list_field_info(self):
        """Get fields information

        Returns
        -------
        infos: list of AttrFieldInfo
            List of field information
        """
        return _ffi_api.AttrsListFieldInfo(self)

    def keys(self):
        """Get list of names in the attribute.

        Returns
        -------
        keys : list of str
            List of keys
        """
        return [field.name for field in self.list_field_info()]

    def get_int_tuple(self, key):
        """Get a python int tuple of a key

        Parameters
        ----------
        key: str

        Returns
        -------
        value: Tuple of int
        """
        return tuple(x if isinstance(x, int) else x.value for x in self.__getattr__(key))

    def get_int(self, key):
        """Get a python int value of a key

        Parameters
        ----------
        key: str

        Returns
        -------
        value: int
        """
        return self.__getattr__(key)

    def get_str(self, key):
        """Get a python int value of a key

        Parameters
        ----------
        key: str

        Returns
        -------
        value: int
        """
        return self.__getattr__(key)

    def __getitem__(self, item):
        return self.__getattr__(item)


@tvm._ffi.register_object
class DictAttrs(Attrs):
    """Dictionary attributes."""

    def _dict(self):
        """Get internal dict"""
        return _ffi_api.DictAttrsGetDict(self)

    def keys(self):
        """Get list of names in the attribute.

        Returns
        -------
        keys : list of str
            List of keys
        """
        return [k for k, _ in self.items()]

    def __getitem__(self, k):
        return self._dict().__getitem__(k)

    def get(self, key, default=None):
        """Get an element with a default value."""
        return self._dict().get(key, default)

    def __contains__(self, k):
        return self._dict().__contains__(k)

    def items(self):
        """Get items from the map."""
        return self._dict().items()

    def __len__(self):
        return self._dict().__len__()


def make_node(type_key, **kwargs):
    """Make a new IR node by its type key and fields

    Parameters
    ----------
    type_key : str
        The type key of the node.

    **kwargs : dict
        The fields of the node.

    Returns
    -------
    node : Node
        The corresponding IR Node

    Note
    ----
    If the created node is instance of AttrsNode, then
    the creator function will also run bound checks and
    default value setup as supported by Attrs.

    Example
    -------
    The following code constructs a IntImm object

    .. code-block:: python

       x = tvm.ir.make_node("IntImm", dtype="int32", value=10, span=None)
       assert isinstance(x, tvm.tir.IntImm)
       assert x.value == 10
    """
    args = [type_key]
    for k, v in kwargs.items():
        args += [k, v]
    return tvm.runtime._ffi_node_api.MakeNode(*args)
