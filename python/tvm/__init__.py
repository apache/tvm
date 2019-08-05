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
"""TVM: Low level DSL/IR stack for tensor computation."""
from __future__ import absolute_import as _abs

import multiprocessing
import sys
import traceback

from . import _pyversion

from . import tensor
from . import arith
from . import expr
from . import stmt
from . import make
from . import ir_pass
from . import codegen
from . import container
from . import schedule
from . import module
from . import node
from . import attrs
from . import ir_builder
from . import target
from . import generic
from . import hybrid
from . import testing
from . import error
from . import datatype

from . import ndarray as nd
from .ndarray import context, cpu, gpu, opencl, cl, vulkan, metal, mtl
from .ndarray import vpi, rocm, opengl, ext_dev, micro_dev

from ._ffi.runtime_ctypes import TypeCode, TVMType
from ._ffi.ndarray import TVMContext
from ._ffi.function import Function
from ._ffi.base import TVMError, __version__
from .api import *
from .intrin import *
from .tensor_intrin import decl_tensor_intrin
from .node import register_node
from .ndarray import register_extension
from .schedule import create_schedule
from .build_module import build, lower, build_config
from .tag import tag_scope

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
