# pylint: disable=redefined-builtin, wildcard-import
"""CUDA specific declaration and schedules."""
from __future__ import absolute_import as _abs

from .conv2d_nchw import schedule_conv2d_nchw
from .conv2d_hwcn import schedule_conv2d_hwcn
from .depthwise_conv2d_map import schedule_depthwise_conv2d_map
from .reduction import schedule_reduce
from .broadcast import schedule_broadcast_to
