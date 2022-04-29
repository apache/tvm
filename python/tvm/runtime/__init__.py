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

# class exposures
from .packed_func import PackedFunc
from .object import Object
from .object_generic import ObjectGeneric, ObjectTypes
from .ndarray import NDArray, DataType, DataTypeCode, Device
from .module import Module, num_threads
from .profiling import Report

# function exposures
from .object_generic import convert_to_object, convert, const
from .ndarray import device, cpu, cuda, gpu, opencl, cl, vulkan, metal, mtl
from .ndarray import vpi, rocm, ext_dev
from .module import load_module, enabled, system_lib
from .container import String, ShapeTuple
from .params import save_param_dict, load_param_dict

from . import executor
