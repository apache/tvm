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
from tvm_ffi._dtype import DataTypeCode
from tvm_ffi._dtype import dtype as DataType

from . import disco

# function exposures
from ._tensor import (
    Tensor,
    cpu,
    cuda,
    device,
    empty,
    ext_dev,
    from_dlpack,
    metal,
    opencl,
    rocm,
    tensor,
    vpi,
    vulkan,
)
from .container import ShapeTuple, String
from .device import Device
from .executable import Executable
from .module import Module, enabled, load_module, load_static_library, num_threads, system_lib
from .object import Object
from .object_generic import ObjectConvertible, const

# class exposures
from .packed_func import PackedFunc
from .params import (
    load_param_dict,
    load_param_dict_from_file,
    save_param_dict,
    save_param_dict_to_file,
)
from .profiling import Report
from .script_printer import Scriptable
from .support import _regex_match
