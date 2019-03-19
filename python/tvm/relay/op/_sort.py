"""Definition of argsort op"""
from __future__ import absolute_import

import topi
from topi.util import get_const_int, get_const_float, get_float_tuple
from .. import op as reg
from ..op import OpPattern


@reg.register_schedule("argsort")
def schedule_argsort(_, outs, target):
    """Schedule definition of argsort"""
    with target:
        return topi.generic.schedule_argsort(outs)


@reg.register_compute("argsort")
def compute_argsort(attrs, inputs, _, target):
    """Compute definition of argsort"""
    axis = get_const_int(attrs.axis)
    is_ascend = bool(get_const_int(attrs.is_ascend))
    flag = bool(get_const_int(attrs.flag))
    return [
        topi.argsort(inputs[0], inputs[1], axis, is_ascend, flag)
    ]


reg.register_pattern("argsort", OpPattern.OPAQUE)
