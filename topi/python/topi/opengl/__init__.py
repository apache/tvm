# pylint: disable=redefined-builtin, wildcard-import
"""CUDA specific declaration and schedules."""
from __future__ import absolute_import as _abs

from .conv2d_nchw import schedule_conv2d_nchw
from .injective import schedule_injective, schedule_elemwise, schedule_broadcast
from .softmax import schedule_softmax
from .dense import schedule_dense
from .pooling import schedule_pool, schedule_adaptive_pool
