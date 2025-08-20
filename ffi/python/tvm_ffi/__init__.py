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
from .registry import register_object, register_func, get_global_func, _init_api
from .dtype import dtype, DataTypeCode
from .core import String, Bytes
from .core import Object, ObjectGeneric, Function
from .convert import convert
from .error import register_error
from .ndarray import Device, device
from .ndarray import cpu, cuda, rocm, opencl, metal, vpi, vulkan, ext_dev, hexagon, webgpu
from .ndarray import from_dlpack, NDArray, Shape
from .container import Array, Map
from .module import Module, ModulePropertyMask, system_lib, load_module
from . import serialization
from . import access_path
from . import testing


__all__ = [
    "dtype",
    "DataTypeCode",
    "Device",
    "Object",
    "register_object",
    "register_func",
    "get_global_func",
    "_init_api",
    "Object",
    "ObjectGeneric",
    "Function",
    "convert",
    "String",
    "Bytes",
    "register_error",
    "Device",
    "device",
    "cpu",
    "cuda",
    "rocm",
    "opencl",
    "metal",
    "vpi",
    "vulkan",
    "ext_dev",
    "hexagon",
    "webgpu",
    "from_dlpack",
    "NDArray",
    "Shape",
    "Array",
    "Map",
    "testing",
    "access_path",
    "serialization",
    "Module",
    "ModulePropertyMask",
    "system_lib",
    "load_module",
]
