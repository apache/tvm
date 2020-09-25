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
# pylint: disable=redefined-builtin, wildcard-import
"""TVM: Open Deep Learning Compiler Stack."""
import multiprocessing
import sys
import traceback

# top-level alias
# tvm._ffi
from ._ffi.base import TVMError, __version__
from ._ffi.runtime_ctypes import DataTypeCode, DataType
from ._ffi import register_object, register_func, register_extension, get_global_func

# top-level alias
# tvm.runtime
from .runtime.object import Object
from .runtime.ndarray import context, cpu, gpu, opencl, cl, vulkan, metal, mtl
from .runtime.ndarray import vpi, rocm, ext_dev, micro_dev, hexagon
from .runtime import ndarray as nd

# tvm.error
from . import error

# tvm.ir
from .ir import IRModule
from .ir import transform
from .ir import container
from . import ir

# tvm.tir
from . import tir

# tvm.target
from . import target

# tvm.te
from . import te

# tvm.driver
from .driver import build, lower

# tvm.parser
from . import parser

# others
from . import arith

# support infra
from . import support

# Contrib initializers
from .contrib import rocm as _rocm, nvcc as _nvcc, sdaccel as _sdaccel


def tvm_wrap_excepthook(exception_hook):
    """Wrap given excepthook with TVM additional work."""

    def wrapper(exctype, value, trbk):
        """Clean subprocesses when TVM is interrupted."""
        exception_hook(exctype, value, trbk)
        if hasattr(multiprocessing, "active_children"):
            # pylint: disable=not-callable
            for p in multiprocessing.active_children():
                p.terminate()

    return wrapper


sys.excepthook = tvm_wrap_excepthook(sys.excepthook)
