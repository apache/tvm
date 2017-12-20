# pylint: disable=redefined-builtin, wildcard-import
"""x86 specific declaration and schedules."""
from __future__ import absolute_import as _abs

from .conv2d import schedule_conv2d
from .binarize_pack import schedule_binarize_pack
from .binary_dense import schedule_binary_dense
from .conv2d import schedule_conv2d, schedule_conv2d_nhwc
