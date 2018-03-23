"""TVM VTA runtime"""
from __future__ import absolute_import as _abs

from .hw_spec import *

from .runtime import SCOPE_INP, SCOPE_OUT, SCOPE_WGT, DMA_COPY, ALU
from .intrin import GEVM, GEMM
from .build import debug_mode

from . import mock, ir_pass
from . import arm_conv2d, vta_conv2d
from . import graph
