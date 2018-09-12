# pylint: disable=redefined-builtin, wildcard-import
"""CUDA specific declaration and schedules."""
from __future__ import absolute_import as _abs

from . import conv2d, depthwise_conv2d, conv2d_transpose_nchw
from .conv2d_hwcn import schedule_conv2d_hwcn
from .depthwise_conv2d import schedule_depthwise_conv2d_backward_input_nhwc
from .depthwise_conv2d import schedule_depthwise_conv2d_backward_weight_nhwc
from .reduction import schedule_reduce
from .softmax import schedule_softmax
from .injective import schedule_injective, schedule_elemwise, schedule_broadcast
from .dense import dense_cuda, schedule_dense
from .pooling import schedule_pool, schedule_global_pool
from .extern import schedule_extern
from .nn import schedule_lrn, schedule_l2_normalize
from .vision import *
from . import ssd
from .ssd import *
from .nms import *
