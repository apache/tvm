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
"""Python bindings for creating VirtualDevices."""

import tvm
from tvm.runtime import Object

from . import _ffi_api


@tvm._ffi.register_object
class VirtualDevice(Object):
    """A compile time representation for where data is to be stored at runtime,
    and how to compile code to compute it."""

    def __init__(self, device, target=None, memory_scope="") -> None:
        self.__init_handle_by_constructor__(
            _ffi_api.VirtualDevice_ForDeviceTargetAndMemoryScope, device, target, memory_scope
        )

    @property
    def device_type(self) -> int:
        return self.device_type_int
