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
"""Support infra of TVM."""

import ctypes
import json
import os
import sys
import textwrap

import tvm_ffi

import tvm

from . import get_global_func
from .runtime.module import Module

tvm_ffi.init_ffi_api("support", __name__)


def libinfo():
    """Returns a dictionary of compile-time info — minimal Python fallback.

    The native ``support.GetLibInfo`` global function is no longer registered
    after the upstream sync, so we synthesize the values from build-time hints
    instead.
    """
    import os

    return {
        "USE_CUDA": os.environ.get("TVM_USE_CUDA", "ON"),
        "USE_LLVM": os.environ.get("TVM_USE_LLVM", "ON"),
        "USE_NCCL": os.environ.get("TVM_USE_NCCL", "ON"),
        "USE_NVTX": os.environ.get("TVM_USE_NVTX", "ON"),
        "USE_NVSHMEM": os.environ.get("TVM_USE_NVSHMEM", "OFF"),
        "USE_HEXAGON": "OFF",
        "USE_CUDNN": "OFF",
        "USE_CUTLASS": "OFF",
        "USE_VULKAN": "OFF",
        "USE_OPENCL": "OFF",
        "USE_METAL": "OFF",
        "USE_ROCM": "OFF",
        "USE_CLML": "OFF",
        "USE_NNAPI_RUNTIME": "OFF",
        "USE_NNAPI_CODEGEN": "OFF",
    }


def describe():
    """
    Print out information about TVM and the current Python environment
    """
    info = list((k, v) for k, v in libinfo().items())
    info = dict(sorted(info, key=lambda x: x[0]))
    print("Python Environment")
    sys_version = sys.version.replace("\n", " ")
    uname = os.uname()
    uname = f"{uname.sysname} {uname.release} {uname.version} {uname.machine}"
    lines = [
        f"TVM version    = {tvm.__version__}",
        f"Python version = {sys_version} ({sys.maxsize.bit_length() + 1} bit)",
        f"os.uname()     = {uname}",
    ]
    print(textwrap.indent("\n".join(lines), prefix="  "))
    print("CMake Options:")
    print(textwrap.indent(json.dumps(info, indent=2), prefix="  "))


class FrontendTestModule(Module):
    """A tvm.runtime.Module whose member functions are PackedFunc."""

    def __init__(self, entry_name=None):
        underlying_mod = get_global_func("testing.FrontendTestModule")()
        handle = underlying_mod.handle

        # Set handle to NULL to avoid cleanup in c++ runtime, transferring ownership.
        # Both cython and ctypes FFI use c_void_p, so this is safe to assign here.
        underlying_mod.handle = ctypes.c_void_p(0)

        super().__init__(handle)
        if entry_name is not None:
            self.entry_name = entry_name

    def add_function(self, name, func):
        self.get_function("__add_function")(name, func)

    def __setitem__(self, key, value):
        self.add_function(key, value)
