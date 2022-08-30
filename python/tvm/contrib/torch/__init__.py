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
# pylint: disable=wrong-import-position,redefined-builtin,invalid-name
"""Module container of Pytorch custom class"""
import os
import platform
import warnings
import torch
from tvm._ffi import libinfo


def _load_platform_specific_library(lib_name):
    system = platform.system()
    if system == "Darwin":
        lib_file_name = lib_name + ".dylib"
    elif system == "Windows":
        lib_file_name = lib_name + ".dll"
    else:
        lib_file_name = lib_name + ".so"
    lib_path = libinfo.find_lib_path()[0]
    lib_dir = os.path.dirname(lib_path)
    lib_file_path = os.path.join(lib_dir, lib_file_name)
    try:
        torch.classes.load_library(lib_file_path)
    except OSError as err:
        errmsg = str(err)
        if errmsg.find("undefined symbol") != -1:
            reason = " ".join(
                (
                    "Got undefined symbol error,",
                    "which might be due to the CXXABI incompatibility.",
                )
            )
        else:
            reason = errmsg
        warnings.warn(
            f"The library {lib_name} is not built successfully. {reason}",
            RuntimeWarning,
        )


_load_platform_specific_library("libpt_tvmdsoop")
_load_platform_specific_library("libpt_tvmdsoop_new")

from . import module

GraphModule = module.GraphModule
VMModule = module.VMModule
TraceTvmModule = module.TraceTvmModule

from . import pytorch_tvm

PyTorchTVMModule = pytorch_tvm.PyTorchTVMModule
compile = pytorch_tvm.compile

from . import as_torch

TVMScriptIRModule = as_torch.OperatorModuleWrapper
as_torch = as_torch.as_torch

from . import optimize_torch

GraphExecutorFactoryWrapper = optimize_torch.GraphExecutorFactoryWrapper
optimize_torch = optimize_torch.optimize_torch
