# pylint: disable=redefined-builtin, wildcard-import
"""CUDA specific declaration and schedules."""
from __future__ import absolute_import as _abs

from .conv2d_hwcn_map import schedule_conv2d_hwcn_map
from .depthwise_conv2d_map import schedule_depthwise_conv2d_map
