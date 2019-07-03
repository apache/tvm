# pylint: disable=redefined-builtin, wildcard-import
"""x86 specific declaration and schedules."""
from __future__ import absolute_import as _abs

from .conv2d import schedule_conv2d, schedule_conv2d_nhwc
from .binarize_pack import schedule_binarize_pack
from .binary_dense import schedule_binary_dense
from .nn import *
from .injective import *
from .pooling import schedule_pool, schedule_adaptive_pool
from .bitserial_conv2d import schedule_bitserial_conv2d
from .bitserial_dense import schedule_bitserial_dense
from .depthwise_conv2d import schedule_depthwise_conv2d_NCHWc
from .dense import _schedule_dense, _schedule_dense_pack, _schedule_dense_nopack
from .batch_matmul import schedule_batch_matmul
from .roi_align import roi_align_nchw
from .conv2d_transpose import schedule_conv2d_transpose
