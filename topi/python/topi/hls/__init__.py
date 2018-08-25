# pylint: disable=redefined-builtin, wildcard-import
"""HLS specific declaration and schedules."""
from __future__ import absolute_import as _abs

from .injective import schedule_injective, schedule_elemwise, schedule_broadcast
from .nn import *
