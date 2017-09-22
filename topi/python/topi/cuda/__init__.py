# pylint: disable=redefined-builtin, wildcard-import
"""CUDA specific declaration and schedules."""
from __future__ import absolute_import as _abs

from .conv2d_nchw import schedule_conv2d_nchw
from .conv2d_hwcn import schedule_conv2d_hwcn
from .depthwise_conv2d import schedule_depthwise_conv2d_nchw, schedule_depthwise_conv2d_nhwc
from .depthwise_conv2d import schedule_depthwise_conv2d_backward_input_nhwc
from .depthwise_conv2d import schedule_depthwise_conv2d_backward_weight_nhwc
from .reduction import schedule_reduce
from .softmax import schedule_softmax
<<<<<<< 31fb14e42dae03b7c8609992bee2bb956b593f79
from .injective import schedule_injective, schedule_elemwise, schedule_broadcast
=======
from .elemwise import schedule_elemwise
from .fully_connected import schedule_fully_connected, schedule_fully_connected_with_bias
from .pooling import schedule_global_avg_pool
>>>>>>> migrate global_avg_pool, fully_connected
