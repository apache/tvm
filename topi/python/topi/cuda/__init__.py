# pylint: disable=redefined-builtin, wildcard-import
"""CUDA specific declaration and schedules."""
from __future__ import absolute_import as _abs

from .conv2d_nchw import schedule_conv2d_nchw
from .conv2d_hwcn import schedule_conv2d_hwcn
from .depthwise_conv2d import schedule_depthwise_conv2d_nchw, schedule_depthwise_conv2d_nhwc
from .depthwise_conv2d import schedule_depthwise_conv2d_backward_input_nhwc
from .depthwise_conv2d import schedule_depthwise_conv2d_backward_weight_nhwc
from .reduction import schedule_reduce
from .broadcast import schedule_broadcast_to
from .softmax import schedule_softmax
from .elemwise import schedule_elemwise
