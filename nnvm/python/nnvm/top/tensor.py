# pylint: disable=invalid-name, unused-argument
"""Tensor ops"""
from __future__ import absolute_import

import tvm
import topi
import topi.cuda
from . import registry as reg
from .registry import OpPattern

def _schedule_injective(_, outs, target):
    """Generic schedule for binary bcast"""
    with tvm.target.create(target):
        return topi.generic.schedule_injective(outs)

def _compute_binary_scalar(f):
    """auxiliary function"""
    @tvm.tag_scope(topi.tag.ELEMWISE)
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


_fschedule_injective = tvm.convert(_schedule_injective)
_fschedule_broadcast = _fschedule_injective
_fschedule_elemwise = _fschedule_injective

# Assign requires special treatment in the compiler
# The compute and schedule are designed as
# copy from rhs to output
reg.register_pattern("_assign", OpPattern.OPAQUE)
reg.register_schedule("_assign", _fschedule_broadcast)

# copy
reg.register_pattern("copy", OpPattern.ELEMWISE)
reg.register_schedule("copy", _fschedule_broadcast)

# cast
reg.register_pattern("cast", OpPattern.ELEMWISE)
reg.register_schedule("cast", _fschedule_broadcast)

# floor
reg.register_pattern("floor", OpPattern.ELEMWISE)
reg.register_schedule("floor", _fschedule_broadcast)

# ceil
reg.register_pattern("ceil", OpPattern.ELEMWISE)
reg.register_schedule("ceil", _fschedule_broadcast)

# round
reg.register_pattern("round", OpPattern.ELEMWISE)
reg.register_schedule("round", _fschedule_broadcast)

# abs
reg.register_pattern("abs", OpPattern.ELEMWISE)
reg.register_schedule("abs", _fschedule_broadcast)

# trunc
reg.register_pattern("trunc", OpPattern.ELEMWISE)
reg.register_schedule("trunc", _fschedule_broadcast)

# exp
reg.register_pattern("exp", OpPattern.ELEMWISE)
reg.register_schedule("exp", _fschedule_broadcast)

# sqrt
reg.register_pattern("sqrt", OpPattern.ELEMWISE)
reg.register_schedule("sqrt", _fschedule_broadcast)

# log
reg.register_pattern("log", OpPattern.ELEMWISE)
reg.register_schedule("log", _fschedule_broadcast)

# tanh
reg.register_pattern("tanh", OpPattern.ELEMWISE)
reg.register_schedule("tanh", _fschedule_broadcast)

# negative
reg.register_pattern("negative", OpPattern.ELEMWISE)
reg.register_schedule("negative", _fschedule_broadcast)

# sigmoid
reg.register_pattern("sigmoid", OpPattern.ELEMWISE)
reg.register_schedule("sigmoid", _fschedule_broadcast)

# add_scalar
reg.register_pattern("__add_scalar__", OpPattern.ELEMWISE)
reg.register_schedule("__add_scalar__", _fschedule_broadcast)

# sub_calar
reg.register_pattern("__sub_scalar__", OpPattern.ELEMWISE)
reg.register_schedule("__sub_scalar__", _fschedule_broadcast)

# rsub_scalar
reg.register_pattern("__rsub_scalar__", OpPattern.ELEMWISE)
reg.register_schedule("__rsub_scalar__", _fschedule_broadcast)

# mul_scalar
reg.register_pattern("__mul_scalar__", OpPattern.ELEMWISE)
reg.register_schedule("__mul_scalar__", _fschedule_broadcast)

# div_scalar
reg.register_pattern("__div_scalar__", OpPattern.ELEMWISE)
reg.register_schedule("__div_scalar__", _fschedule_broadcast)

# rdiv_scalar
reg.register_pattern("__rdiv_scalar__", OpPattern.ELEMWISE)
reg.register_schedule("__rdiv_scalar__", _fschedule_broadcast)

# pow_scalar
reg.register_pattern("__pow_scalar__", OpPattern.ELEMWISE)
reg.register_schedule("__pow_scalar__", _fschedule_broadcast)

# rpow_scalar
reg.register_pattern("__rpow_scalar__", OpPattern.ELEMWISE)
reg.register_schedule("__rpow_scalar__", _fschedule_broadcast)

# lshift_scalar
reg.register_pattern("__lshift_scalar__", OpPattern.ELEMWISE)
reg.register_schedule("__lshift_scalar__", _fschedule_broadcast)

# rshift_scalar
reg.register_pattern("__rshift_scalar__", OpPattern.ELEMWISE)
reg.register_schedule("__rshift_scalar__", _fschedule_broadcast)

# elemwise_add
reg.register_pattern("elemwise_add", OpPattern.BROADCAST)
reg.register_schedule("elemwise_add", _fschedule_broadcast)

# elemwise_sub
reg.register_pattern("elemwise_sub", OpPattern.BROADCAST)
reg.register_schedule("elemwise_sub", _fschedule_broadcast)

# elemwise_mul
reg.register_pattern("elemwise_mul", OpPattern.BROADCAST)
reg.register_schedule("elemwise_mul", _fschedule_broadcast)

# elemwise_div
reg.register_pattern("elemwise_div", OpPattern.BROADCAST)
reg.register_schedule("elemwise_div", _fschedule_broadcast)

# elemwise_mod
reg.register_pattern("elemwise_mod", OpPattern.BROADCAST)
reg.register_schedule("elemwise_mod", _fschedule_broadcast)

# elemwise_pow
reg.register_pattern("elemwise_pow", OpPattern.BROADCAST)
reg.register_schedule("elemwise_pow", _fschedule_broadcast)

# broadcast_add
reg.register_pattern("broadcast_add", OpPattern.BROADCAST)
reg.register_schedule("broadcast_add", _fschedule_broadcast)

# broadcast_sub
reg.register_pattern("broadcast_sub", OpPattern.BROADCAST)
reg.register_schedule("broadcast_sub", _fschedule_broadcast)

# broadcast_mul
reg.register_pattern("broadcast_mul", OpPattern.BROADCAST)
reg.register_schedule("broadcast_mul", _fschedule_broadcast)

# broadcast_div
reg.register_pattern("broadcast_div", OpPattern.BROADCAST)
reg.register_schedule("broadcast_div", _fschedule_broadcast)

# broadcast mod
reg.register_pattern("broadcast_mod", OpPattern.BROADCAST)
reg.register_schedule("broadcast_mod", _fschedule_broadcast)

# broadcast max
reg.register_pattern("broadcast_max", OpPattern.BROADCAST)
reg.register_schedule("broadcast_max", _fschedule_broadcast)

# broadcast min
reg.register_pattern("broadcast_min", OpPattern.BROADCAST)
reg.register_schedule("broadcast_min", _fschedule_broadcast)

# broadcast pow
reg.register_pattern("broadcast_pow", OpPattern.BROADCAST)
reg.register_schedule("broadcast_pow", _fschedule_broadcast)

# broadcast left_shift
reg.register_pattern("broadcast_left_shift", OpPattern.BROADCAST)
reg.register_schedule("broadcast_left_shift", _fschedule_broadcast)

# broadcast right_shift
reg.register_pattern("broadcast_right_shift", OpPattern.BROADCAST)
reg.register_schedule("broadcast_right_shift", _fschedule_broadcast)

# broadcast greater
reg.register_pattern("broadcast_greater", OpPattern.BROADCAST)
reg.register_schedule("broadcast_greater", _fschedule_broadcast)

# broadcast less
reg.register_pattern("broadcast_less", OpPattern.BROADCAST)
reg.register_schedule("broadcast_less", _fschedule_broadcast)

# broadcast equal
reg.register_pattern("broadcast_equal", OpPattern.BROADCAST)
reg.register_schedule("broadcast_equal", _fschedule_broadcast)

# broadcast not_equal
reg.register_pattern("broadcast_not_equal", OpPattern.BROADCAST)
reg.register_schedule("broadcast_not_equal", _fschedule_broadcast)

# broadcast greater_equal
reg.register_pattern("broadcast_greater_equal", OpPattern.BROADCAST)
reg.register_schedule("broadcast_greater_equal", _fschedule_broadcast)

# broadcast less_equal
reg.register_pattern("broadcast_less_equal", OpPattern.BROADCAST)
reg.register_schedule("broadcast_less_equal", _fschedule_broadcast)

# broadcast_to
reg.register_pattern("broadcast_to", OpPattern.BROADCAST)
reg.register_schedule("broadcast_to", _fschedule_broadcast)

# clip
reg.register_pattern("clip", OpPattern.ELEMWISE)
reg.register_schedule("clip", _fschedule_elemwise)

# elemwise sum
reg.register_pattern("elemwise_sum", OpPattern.ELEMWISE)
reg.register_schedule("elemwise_sum", _fschedule_elemwise)

# full
reg.register_pattern("full", OpPattern.OUT_ELEMWISE_FUSABLE)
reg.register_schedule("full", _fschedule_elemwise)

# full_like
reg.register_pattern("full_like", OpPattern.ELEMWISE)
reg.register_schedule("full_like", _fschedule_elemwise)

# zeros
reg.register_pattern("zeros", OpPattern.OUT_ELEMWISE_FUSABLE)
reg.register_schedule("zeros", _fschedule_elemwise)

# zeros_like
reg.register_pattern("zeros_like", OpPattern.ELEMWISE)
reg.register_schedule("zeros_like", _fschedule_elemwise)

# ones
reg.register_pattern("ones", OpPattern.OUT_ELEMWISE_FUSABLE)
reg.register_schedule("ones", _fschedule_elemwise)

# ones_like
reg.register_pattern("ones_like", OpPattern.ELEMWISE)
reg.register_schedule("ones_like", _fschedule_elemwise)

# greater
reg.register_pattern("greater", OpPattern.ELEMWISE)
reg.register_schedule("greater", _fschedule_elemwise)

# less
reg.register_pattern("less", OpPattern.ELEMWISE)
reg.register_schedule("less", _fschedule_elemwise)

# block_grad
reg.register_compute("block_grad", _compute_unary(topi.identity))
reg.register_pattern("block_grad", OpPattern.ELEMWISE)
reg.register_schedule("block_grad", _fschedule_elemwise)
