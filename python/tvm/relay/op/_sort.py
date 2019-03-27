"""Definition of argsort op"""
# pylint: disable=invalid-name,unused-argument
from __future__ import absolute_import

import topi
from topi.util import get_const_int
from .op import OpPattern, register_compute, register_schedule, register_pattern


@register_schedule("argsort")
def schedule_argsort(_, outs, target):
    """Schedule definition of argsort"""
    with target:
        return topi.generic.schedule_argsort(outs)


@register_compute("argsort")
def compute_argsort(attrs, inputs, _, target):
    """Compute definition of argsort"""
    axis = get_const_int(attrs.axis)
    is_ascend = bool(get_const_int(attrs.is_ascend))
    return [
        topi.argsort(inputs[0], None, axis, is_ascend, flag=False)
    ]


register_pattern("argsort", OpPattern.OPAQUE)
