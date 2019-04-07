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
""" TVM Attribute module, which is mainly used for defining attributes of operators"""
from ._ffi.node import NodeBase, register_node as _register_tvm_node
from ._ffi.function import _init_api
from . import _api_internal


@_register_tvm_node
class Attrs(NodeBase):
    """Attribute node, which is mainly use for defining attributes of relay operators.

    Used by function registered in python side, such as compute, schedule and alter_layout.
    Attrs is passed as the first argument to these functions.
    """
    def list_field_info(self):
        """ Get fields information

        Returns
        -------
        infos: list of AttrFieldInfo
            List of field information
        """
        return _api_internal._AttrsListFieldInfo(self)

    def keys(self):
        """Get list of names in the attribute.

        Returns
        -------
        keys : list of str
            List of keys
        """
        fields = self.list_field_info()
        for field in fields:
            yield field.name

    def get_int_tuple(self, key):
        """Get a python int tuple of a key

        Parameters
        ----------
        key: str

        Returns
        -------
        value: Tuple of int
        """
        return tuple(x.value for x in self.__getattr__(key))

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


_init_api("tvm.attrs")
