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
"""TVM runtime namespace."""

from tvm_ffi import convert
from tvm_ffi._dtype import dtype as DataType, DataTypeCode

# class exposures
from .packed_func import PackedFunc
from .object import Object
from .script_printer import Scriptable
from .object_generic import ObjectConvertible
from .device import Device
from ._tensor import Tensor, tensor, empty
from .module import Module
from .profiling import Report
from .executable import Executable

# function exposures
from ._tensor import device, cpu, cuda, opencl, vulkan, metal
from ._tensor import vpi, rocm, ext_dev, from_dlpack
from .module import load_module, enabled, system_lib, load_static_library, num_threads
from .container import String, ShapeTuple
from .object_generic import const
from .params import (
    save_param_dict,
    load_param_dict,
    save_param_dict_to_file,
    load_param_dict_from_file,
)

from . import disco

from .support import _regex_match
