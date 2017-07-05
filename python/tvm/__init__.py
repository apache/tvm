# pylint: disable=redefined-builtin, wildcard-import
"""TVM: a DSL for tensor kernel compilation"""
from __future__ import absolute_import as _abs

from . import tensor
from . import arith
from . import expr
from . import stmt
from . import make
from . import ir_pass
from . import codegen
from . import collections
from . import schedule
from . import module
from . import node
from . import ir_builder

from . import ndarray as nd
from .ndarray import context, cpu, gpu, opencl, cl, metal, mtl, vpi

from ._ffi.runtime_ctypes import TypeCode
from ._ffi.function import Function
from ._ffi.base import TVMError, __version__
from .api import *
from .intrin import *
from .node import register_node
from .ndarray import register_extension
from .schedule import create_schedule
from .build import build, lower, build_config
from .op_tag import op_tag
