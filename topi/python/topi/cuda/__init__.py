# pylint: disable=redefined-builtin, wildcard-import
"""CUDA specific declaration and schedules."""
from __future__ import absolute_import as _abs

from . import conv2d_nchw
from . import conv2d_hwcn
from .depthwise_conv2d_map import schedule_depthwise_conv2d_map
