# pylint: disable=invalid-name, unused-argument
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
    def _compute(attrs, x, _):
        x = x[0]
        scalar = attrs.get_float("scalar")
        scalar = tvm.const(scalar, x.dtype)
        return tvm.compute(x.shape, lambda *i: f(x(*i), scalar))
    return _compute


def _compute_unary(f):
    """auxiliary function"""
    def _compute(attrs, x, _):
        return f(x[0])
    return _compute


def _compute_binary(f):
    """auxiliary function"""
    def _compute(attrs, x, _):
        return f(x[0], x[1])
    return _compute


_fschedule_broadcast = tvm.convert(_schedule_broadcast)

# copy
reg.register_compute("copy", _compute_unary(topi.identity))
reg.register_pattern("copy", OpPattern.ELEM_WISE)
reg.register_schedule("copy", _fschedule_broadcast)

# exp
reg.register_compute("exp", _compute_unary(topi.exp))
reg.register_pattern("exp", OpPattern.ELEM_WISE)
reg.register_schedule("exp", _fschedule_broadcast)

# sqrt
reg.register_compute("sqrt", _compute_unary(topi.sqrt))
reg.register_pattern("sqrt", OpPattern.ELEM_WISE)
reg.register_schedule("sqrt", _fschedule_broadcast)

# log
reg.register_compute("log", _compute_unary(topi.log))
reg.register_pattern("log", OpPattern.ELEM_WISE)
reg.register_schedule("log", _fschedule_broadcast)

# tanh
reg.register_compute("tanh", _compute_unary(topi.tanh))
reg.register_pattern("tanh", OpPattern.ELEM_WISE)
reg.register_schedule("tanh", _fschedule_broadcast)

# negative
reg.register_compute("negative", _compute_unary(topi.negative))
reg.register_pattern("negative", OpPattern.ELEM_WISE)
reg.register_schedule("negative", _fschedule_broadcast)

# sigmoid
reg.register_compute("sigmoid", _compute_unary(topi.sigmoid))
reg.register_pattern("sigmoid", OpPattern.ELEM_WISE)
reg.register_schedule("sigmoid", _fschedule_broadcast)

# add_scalar
reg.register_compute("__add_scalar__",
                     _compute_binary_scalar(lambda x, y: x + y))
reg.register_pattern("__add_scalar__", OpPattern.ELEM_WISE)
reg.register_schedule("__add_scalar__", _fschedule_broadcast)

# sub_calar
reg.register_compute("__sub_scalar__",
                     _compute_binary_scalar(lambda x, y: x - y))
reg.register_pattern("__sub_scalar__", OpPattern.ELEM_WISE)
reg.register_schedule("__sub_scalar__", _fschedule_broadcast)

# rsub_scalar
reg.register_compute("__rsub_scalar__",
                     _compute_binary_scalar(lambda x, y: y - x))
reg.register_pattern("__rsub_scalar__", OpPattern.ELEM_WISE)
reg.register_schedule("__rsub_scalar__", _fschedule_broadcast)

# mul_scalar
reg.register_compute("__mul_scalar__",
                     _compute_binary_scalar(lambda x, y: x * y))
reg.register_pattern("__mul_scalar__", OpPattern.ELEM_WISE)
reg.register_schedule("__mul_scalar__", _fschedule_broadcast)

# div_scalar
reg.register_compute("__div_scalar__",
                     _compute_binary_scalar(lambda x, y: x / y))
reg.register_pattern("__div_scalar__", OpPattern.ELEM_WISE)
reg.register_schedule("__div_scalar__", _fschedule_broadcast)

# rdiv_scalar
reg.register_compute("__rdiv_scalar__",
                     _compute_binary_scalar(lambda x, y: y / x))
reg.register_pattern("__rdiv_scalar__", OpPattern.ELEM_WISE)
reg.register_schedule("__rdiv_scalar__", _fschedule_broadcast)

# elemwise_add
reg.register_compute("elemwise_add", _compute_binary(topi.broadcast_add))
reg.register_pattern("elemwise_add", OpPattern.BROADCAST)
reg.register_schedule("elemwise_add", _fschedule_broadcast)

# elemwise_sub
reg.register_compute("elemwise_sub", _compute_binary(topi.broadcast_sub))
reg.register_pattern("elemwise_sub", OpPattern.BROADCAST)
reg.register_schedule("elemwise_sub", _fschedule_broadcast)

# elemwise_mul
reg.register_compute("elemwise_mul", _compute_binary(topi.broadcast_mul))
reg.register_pattern("elemwise_mul", OpPattern.BROADCAST)
reg.register_schedule("elemwise_mul", _fschedule_broadcast)

# elemwise_div
reg.register_compute("elemwise_div", _compute_binary(topi.broadcast_div))
reg.register_pattern("elemwise_div", OpPattern.BROADCAST)
reg.register_schedule("elemwise_div", _fschedule_broadcast)

# broadcast_add
reg.register_compute("broadcast_add", _compute_binary(topi.broadcast_add))
reg.register_pattern("broadcast_add", OpPattern.BROADCAST)
reg.register_schedule("broadcast_add", _fschedule_broadcast)

# broadcast_sub
reg.register_compute("broadcast_sub", _compute_binary(topi.broadcast_sub))
reg.register_pattern("broadcast_sub", OpPattern.BROADCAST)
reg.register_schedule("broadcast_sub", _fschedule_broadcast)

# broadcast_mul
reg.register_compute("broadcast_mul", _compute_binary(topi.broadcast_mul))
reg.register_pattern("broadcast_mul", OpPattern.BROADCAST)
reg.register_schedule("broadcast_mul", _fschedule_broadcast)

# broadcast_div
reg.register_compute("broadcast_div", _compute_binary(topi.broadcast_div))
reg.register_pattern("broadcast_div", OpPattern.BROADCAST)
reg.register_schedule("broadcast_div", _fschedule_broadcast)

# broadcast_to
@reg.register_compute("broadcast_to")
def compute_softmax(attrs, inputs, out_info):
    """Compute definition of softmax"""
    return topi.broadcast_to(inputs[0], shape=out_info[0].shape)
reg.register_pattern("broadcast_to", OpPattern.BROADCAST)
reg.register_schedule("broadcast_to", _fschedule_broadcast)
