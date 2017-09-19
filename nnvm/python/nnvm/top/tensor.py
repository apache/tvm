# pylint: disable=invalid-name
"""Tensor ops"""
from __future__ import absolute_import

import tvm
import topi
import topi.cuda
from ..compiler import registry as reg
from ..compiler import OpPattern

def _schedule_broadcast(_, outs, target):
    """Generic schedule for binary bcast"""
    if target == "cuda":
        return topi.cuda.schedule_elemwise(outs)
    assert target.startswith("llvm")
    s = tvm.create_schedule([x.op for x in outs])
    tvm.schedule.AutoInlineInjective(s)
    return s

def _compute_binary_scalar(f):
    """auxiliary function"""
    @tvm.tag_scope("ewise")
    def _compute(attrs, x):
        x = x[0]
        scalar = attrs.get_float("scalar")
        scalar = tvm.const(scalar, x.dtype)
        return tvm.compute(x.shape, lambda *i: f(x(*i), scalar))
    return _compute


_fschedule_broadcast = tvm.convert(_schedule_broadcast)

# exp
reg.register_compute("exp",
                     lambda _, x: topi.exp(x[0]))
reg.register_pattern("exp", OpPattern.ELEM_WISE)
reg.register_schedule("exp", _fschedule_broadcast)

# add scalar
reg.register_compute("__add_scalar__",
                     _compute_binary_scalar(lambda x, y: x + y))
reg.register_pattern("__add_scalar__", OpPattern.ELEM_WISE)
reg.register_schedule("__add_scalar__", _fschedule_broadcast)

# broadcast_add
reg.register_compute("broadcast_add",
                     lambda _, x: topi.broadcast_add(x[0], x[1]))
reg.register_pattern("broadcast_add", OpPattern.BROADCAST)
reg.register_schedule("broadcast_add", _fschedule_broadcast)

# broadcast_sub
reg.register_compute("broadcast_sub",
                     lambda _, x: topi.broadcast_sub(x[0], x[1]))
reg.register_pattern("broadcast_sub", OpPattern.BROADCAST)
reg.register_schedule("broadcast_sub", _fschedule_broadcast)

# broadcast_mul
reg.register_compute("broadcast_mul",
                     lambda _, x: topi.broadcast_mul(x[0], x[1]))
reg.register_pattern("broadcast_mul", OpPattern.BROADCAST)
reg.register_schedule("broadcast_mul", _fschedule_broadcast)

# broadcast_div
reg.register_compute("broadcast_div",
                     lambda _, x: topi.broadcast_div(x[0], x[1]))
reg.register_pattern("broadcast_div", OpPattern.BROADCAST)
reg.register_schedule("broadcast_div", _fschedule_broadcast)
