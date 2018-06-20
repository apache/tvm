# pylint: disable=redefined-builtin, wildcard-import
"""TVM: Low level DSL/IR stack for tensor computation."""
from __future__ import absolute_import as _abs

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
from . import ir_builder
from . import target
from . import generic

from . import ndarray as nd
from .ndarray import context, cpu, gpu, opencl, cl, vulkan, metal, mtl
from .ndarray import vpi, rocm, opengl, ext_dev

from ._ffi.runtime_ctypes import TypeCode
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
from .contrib import rocm as _rocm, nvcc as _nvcc
