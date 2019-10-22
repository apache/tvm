# pylint: disable=redefined-builtin, wildcard-import
"""ARM Mali GPU specific declaration and schedules."""
from __future__ import absolute_import as _abs

from .gemm import *
from .conv2d import *
from .dense import *
from .depthwise_conv2d import *
