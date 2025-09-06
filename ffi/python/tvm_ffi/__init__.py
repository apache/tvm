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
"""TVM FFI Python package."""
# base always go first to load the libtvm_ffi
from . import base
from . import libinfo

# package init part
from .registry import (
    register_object,
    register_global_func,
    get_global_func,
    remove_global_func,
    init_ffi_api,
)
from ._dtype import dtype
from .core import Object, ObjectConvertible, Function
from ._convert import convert
from .error import register_error
from ._tensor import Device, device, DLDeviceType
from ._tensor import from_dlpack, Tensor, Shape
from .container import Array, Map
from .module import Module, system_lib, load_module
from . import serialization
from . import access_path
from . import testing


__all__ = [
    "dtype",
    "Device",
    "Object",
    "register_object",
    "register_global_func",
    "get_global_func",
    "remove_global_func",
    "init_ffi_api",
    "Object",
    "ObjectConvertible",
    "Function",
    "convert",
    "register_error",
    "Device",
    "device",
    "DLDeviceType",
    "from_dlpack",
    "Tensor",
    "Shape",
    "Array",
    "Map",
    "testing",
    "access_path",
    "serialization",
    "Module",
    "system_lib",
    "load_module",
]
