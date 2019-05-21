# pylint: disable=redefined-builtin, wildcard-import
"""TVM Operator Inventory.

TOPI is the operator collection library for TVM, to provide sugars
for constructing compute declaration as well as optimized schedules.

Some of the schedule function may have been specially optimized for a
specific workload.
"""
from __future__ import absolute_import as _abs

from tvm._ffi.libinfo import __version__

# Ensure C++ schedules get registered first, so python schedules can
# override them.
from . import cpp

from .math import *
from .tensor import *
from .generic_op_impl import *
from .reduction import *
from .transform import *
from .broadcast import *
from .sort import *
from . import nn
from . import x86
from . import cuda
from . import arm_cpu
from . import mali
from . import intel_graphics
from . import opengl
from . import util
from . import rocm
from . import vision
from . import image
from . import sparse
from . import hls
# error reporting
from .util import InvalidShapeError
# not import testing by default
# because testing can have extra deps that are not necessary
# we can import them from test cases explicitly
# from . import testing
