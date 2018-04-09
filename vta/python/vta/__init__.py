"""TVM-based VTA Compiler Toolchain"""
from __future__ import absolute_import as _abs

from .hw_spec import *

try:
    from .runtime import SCOPE_INP, SCOPE_OUT, SCOPE_WGT, DMA_COPY, ALU
    from .intrin import GEVM, GEMM
    from .build import debug_mode
    from . import mock, ir_pass
    from . import arm_conv2d, vta_conv2d
except AttributeError:
    pass

from .rpc_client import reconfig_runtime, program_fpga

try:
    from . import graph
except ImportError:
    pass
