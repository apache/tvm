# pylint: disable=redefined-builtin, wildcard-import
"""Raspberry pi specific declaration and schedules."""
from __future__ import absolute_import as _abs

from .conv2d import schedule_conv2d_nchw
from .depthwise_conv2d import schedule_depthwise_conv2d_nchw
