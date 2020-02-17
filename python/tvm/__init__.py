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
from ._ffi.runtime_ctypes import TypeCode, DataType
from ._ffi.registry import register_object, register_func, register_extension

# top-level alias
# tvm.runtime
from .runtime.object import Object
from .runtime.ndarray import context, cpu, gpu, opencl, cl, vulkan, metal, mtl
from .runtime.ndarray import vpi, rocm, opengl, ext_dev, micro_dev
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

# others
from . import tensor
from . import arith
from . import make
from . import schedule
from . import hybrid
from . import testing

from .api import *
from .tensor_intrin import decl_tensor_intrin
from .schedule import create_schedule
from .build_module import build, lower, build_config
from .tag import tag_scope

# backward compact for topi, to be removed later
from .tir import expr, stmt, ir_builder, ir_pass, generic
from .tir.op import *
from . import intrin

# Contrib initializers
from .contrib import rocm as _rocm, nvcc as _nvcc, sdaccel as _sdaccel

# Clean subprocesses when TVM is interrupted
def tvm_excepthook(exctype, value, trbk):
    print('\n'.join(traceback.format_exception(exctype, value, trbk)))
    if hasattr(multiprocessing, 'active_children'):
        # pylint: disable=not-callable
        for p in multiprocessing.active_children():
            p.terminate()

sys.excepthook = tvm_excepthook
