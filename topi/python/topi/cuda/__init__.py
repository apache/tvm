# pylint: disable=redefined-builtin, wildcard-import
"""CUDA specific declaration and schedules."""
from __future__ import absolute_import as _abs

from .depthwise_conv2d_nhwc import schedule_depthwise_conv2d_nhwc 
from .depthwise_conv2d_nchw import schedule_depthwise_conv2d_nchw
from .conv2d_nchw import schedule_conv2d_nchw
from .conv2d_hwcn import schedule_conv2d_hwcn
