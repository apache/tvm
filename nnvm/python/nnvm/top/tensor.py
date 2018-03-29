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

# broadcast_to
reg.register_pattern("broadcast_to", OpPattern.BROADCAST)
reg.register_schedule("broadcast_to", _fschedule_broadcast)

# clip
reg.register_pattern("clip", OpPattern.ELEMWISE)
reg.register_schedule("clip", _fschedule_elemwise)

# elemwise sum
@reg.register_compute("elemwise_sum")
def compute_elemwise_sum(attrs, inputs, _):
    """Compute definition of elemwise sum"""
    num_args = attrs.get_int("num_args")
    assert num_args == len(inputs), "Number of tensors does not match num_args."
    return topi.tensor.elemwise_sum(inputs, num_args)
reg.register_pattern("elemwise_sum", OpPattern.ELEMWISE)
reg.register_schedule("elemwise_sum", _fschedule_elemwise)

# full
@reg.register_compute("full")
def compute_full(attrs, inputs, _):
    """Compute definition of full"""
    shape = attrs.get_int_tuple("shape")
    dtype = attrs.get_string("dtype")
    fill_value = attrs.get_float("fill_value")
    return topi.tensor.full(shape, dtype, fill_value)
reg.register_pattern("full", OpPattern.OUT_ELEMWISE_FUSABLE)
reg.register_schedule("full", _fschedule_elemwise)

# full_like
@reg.register_compute("full_like")
def compute_full_like(attrs, inputs, _):
    """Compute definition of full_like"""
    fill_value = attrs.get_float("fill_value")
    return topi.tensor.full_like(inputs[0], fill_value)
reg.register_pattern("full_like", OpPattern.ELEMWISE)
reg.register_schedule("full_like", _fschedule_elemwise)

# zeros
@reg.register_compute("zeros")
def compute_zeros(attrs, inputs, _):
    """Compute definition of zeros"""
    shape = attrs.get_int_tuple("shape")
    dtype = attrs.get_string("dtype")
    return topi.tensor.full(shape, dtype, 0)
reg.register_pattern("zeros", OpPattern.OUT_ELEMWISE_FUSABLE)
reg.register_schedule("zeros", _fschedule_elemwise)

# zeros_like
@reg.register_compute("zeros_like")
def compute_zeros_like(_, inputs, out_info):
    """Compute definition of zeros_like"""
    return topi.tensor.full_like(inputs[0], 0)
reg.register_pattern("zeros_like", OpPattern.ELEMWISE)
reg.register_schedule("zeros_like", _fschedule_elemwise)

# ones
@reg.register_compute("ones")
def compute_ones(attrs, inputs, _):
    """Compute definition of ones"""
    shape = attrs.get_int_tuple("shape")
    dtype = attrs.get_string("dtype")
    #tvm.tensor.Tensor()
    return topi.tensor.full(shape, dtype, 1)
reg.register_pattern("ones", OpPattern.OUT_ELEMWISE_FUSABLE)
reg.register_schedule("ones", _fschedule_elemwise)

# ones_like
@reg.register_compute("ones_like")
def compute_ones_like(_, inputs, out_info):
    """Compute definition of ones_like"""
    return topi.tensor.full_like(inputs[0], 1)
reg.register_pattern("ones_like", OpPattern.ELEMWISE)
reg.register_schedule("ones_like", _fschedule_elemwise)

# greater
@reg.register_compute("greater")
def compute_greater(_, inputs, out_info):
    """Compute definition of greater"""
    return topi.tensor.greater(inputs[0], inputs[1], 'float32')
reg.register_pattern("greater", OpPattern.ELEMWISE)
reg.register_schedule("greater", _fschedule_elemwise)

# less
@reg.register_compute("less")
def compute_less(_, inputs, out_info):
    """Compute definition of less"""
    return topi.tensor.less(inputs[0], inputs[1], 'float32')
reg.register_pattern("less", OpPattern.ELEMWISE)
reg.register_schedule("less", _fschedule_elemwise)

# block_grad
reg.register_compute("block_grad", _compute_unary(topi.identity))
reg.register_pattern("block_grad", OpPattern.ELEMWISE)
reg.register_schedule("block_grad", _fschedule_elemwise)
