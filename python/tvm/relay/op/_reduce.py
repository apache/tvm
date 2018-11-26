"""Backend compiler related feature registration"""
from __future__ import absolute_import

import topi
from . import op as _reg


def _schedule_reduce(_, outs, target):
    """Generic schedule for reduce"""
    with target:
        return topi.generic.schedule_reduce(outs)


_reg.register_schedule("argmax", _schedule_reduce)
_reg.register_schedule("argmin", _schedule_reduce)
_reg.register_schedule("sum", _schedule_reduce)
_reg.register_schedule("max", _schedule_reduce)
_reg.register_schedule("min", _schedule_reduce)
_reg.register_schedule("prod", _schedule_reduce)
_reg.register_schedule("mean", _schedule_reduce)
