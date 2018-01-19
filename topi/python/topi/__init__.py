# pylint: disable=redefined-builtin, wildcard-import
"""TVM Operator Inventory.

TOPI is the operator collection library for TVM, to provide sugars
for constructing compute declaration as well as optimized schedules.

Some of the schedule function may have been specially optimized for a
specific workload.
"""
from __future__ import absolute_import as _abs

import sys
import os
import ctypes
from tvm._ffi import libinfo

def get_lib_names():
    if sys.platform.startswith('win32'):
        return ['libtvm_topi.dll', 'tvm_topi.dll']
    if sys.platform.startswith('darwin'):
        return ['libtvm_topi.dylib', 'tvm_topi.dylib']
    return ['libtvm_topi.so', 'tvm_topi.so']

curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
lib_search = curr_path

def _load_lib():
    """Load libary by searching possible path."""
    lib_path = libinfo.find_lib_path(get_lib_names(), lib_search, optional=True)
    if lib_path is None:
        return None, None
    lib = ctypes.CDLL(lib_path[0], ctypes.RTLD_GLOBAL)
    return lib, os.path.basename(lib_path[0])

__version__ = libinfo.__version__
_LIB, _LIB_NAME = _load_lib()

from .math import *
from .reduction import *
from .transform import *
from .broadcast import *
from . import nn
from . import x86
from . import cuda
from . import rasp
from . import testing
from . import util
from . import rocm
from . import cpp
