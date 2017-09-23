# pylint: disable=invalid-name, unused-argument
"""Reduction ops"""
from __future__ import absolute_import

import tvm
import topi
import topi.cuda
from ..compiler import registry as reg
from ..compiler import OpPattern

def _schedule_reduce(_, outs, target):
    """Generic schedule for reduce"""
    if target == "cuda":
        return topi.cuda.schedule_reduce(outs)
    assert target.startswith("llvm")
    s = tvm.create_schedule([x.op for x in outs])
    x = outs[0]
    tvm.schedule.AutoInlineInjective(s)
    s[x].fuse(s[x].op.axis)
    return s

_fschedule_reduce = tvm.convert(_schedule_reduce)

def _compute_reduce(f):
    """auxiliary function"""
    def _compute(attrs, inputs, out_info):
        axis = attrs.get_int_tuple("axis")
        keepdims = attrs.get_bool("keepdims")
        if axis:
            return f(inputs[0], axis=axis, keepdims=keepdims)
        return f(inputs[0], keepdims=keepdims)
    return _compute

# sum
reg.register_compute("sum", _compute_reduce(topi.sum))
reg.register_pattern("sum", OpPattern.COMM_REDUCE)
reg.register_schedule("sum", _fschedule_reduce)

# max
reg.register_compute("max", _compute_reduce(topi.max))
reg.register_pattern("max", OpPattern.COMM_REDUCE)
reg.register_schedule("max", _fschedule_reduce)

# min
reg.register_compute("min", _compute_reduce(topi.min))
reg.register_pattern("min", OpPattern.COMM_REDUCE)
reg.register_schedule("min", _fschedule_reduce)
