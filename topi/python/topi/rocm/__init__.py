# pylint: disable=redefined-builtin, wildcard-import
"""rocm specific declaration and schedules."""
from __future__ import absolute_import as _abs

from .conv2d_nchw import schedule_conv2d_nchw
