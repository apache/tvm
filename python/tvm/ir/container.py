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
"""Additional container data structures used across IR variants."""
import tvm._ffi

from tvm.runtime import Object
from tvm.runtime.container import getitem_helper
from tvm.runtime import _ffi_node_api


@tvm._ffi.register_object("Array")
class Array(Object):
    """Array container of TVM.

    You do not need to create Array explicitly.
    Normally python list and tuple will be converted automatically
    to Array during tvm function call.
    You may get Array in return values of TVM function call.
    """

    def __getitem__(self, idx):
        return getitem_helper(self, _ffi_node_api.ArrayGetItem, len(self), idx)

    def __len__(self):
        return _ffi_node_api.ArraySize(self)


@tvm._ffi.register_object
class Map(Object):
    """Map container of TVM.

    You do not need to create Map explicitly.
    Normally python dict will be converted automaticall to Map during tvm function call.
    You can use convert to create a dict[Object-> Object] into a Map
    """

    def __getitem__(self, k):
        return _ffi_node_api.MapGetItem(self, k)

    def __contains__(self, k):
        return _ffi_node_api.MapCount(self, k) != 0

    def items(self):
        """Get the items from the map"""
        akvs = _ffi_node_api.MapItems(self)
        return [(akvs[i], akvs[i + 1]) for i in range(0, len(akvs), 2)]

    def __len__(self):
        return _ffi_node_api.MapSize(self)

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
