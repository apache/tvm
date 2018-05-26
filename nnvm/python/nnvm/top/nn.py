# pylint: disable=invalid-name, unused-argument
"""Definition of nn ops"""
from __future__ import absolute_import

import tvm
import topi
from topi.util import get_const_int
from .tensor import _fschedule_broadcast
from ..compiler import registry as reg
from ..compiler import OpPattern

# relu
@reg.register_compute("relu")
def compute_relu(attrs, inputs, _):
    """Compute definition of relu"""
    return topi.nn.relu(inputs[0])

reg.register_schedule("relu", _fschedule_broadcast)
reg.register_pattern("relu", OpPattern.ELEMWISE)


# leaky_relu
@reg.register_compute("leaky_relu")
def compute_leaky_relu(attrs, inputs, _):
    """Compute definition of relu"""
    return topi.nn.leaky_relu(inputs[0], attrs.get_float("alpha"))

reg.register_schedule("leaky_relu", _fschedule_broadcast)
reg.register_pattern("leaky_relu", OpPattern.ELEMWISE)


# flatten
@reg.register_compute("flatten")
def compute_flatten(attrs, inputs, _):
    """Compute definition of flatten"""
    return topi.nn.flatten(inputs[0])

reg.register_schedule("flatten", _fschedule_broadcast)
reg.register_pattern("flatten", OpPattern.INJECTIVE)


# softmax
@reg.register_compute("softmax")
def compute_softmax(attrs, inputs, _):
    """Compute definition of softmax"""
    axis = attrs.get_int("axis")
    assert axis == -1, "only support axis == -1 for now"
    return topi.nn.softmax(inputs[0])

@reg.register_schedule("softmax")
def schedule_softmax(_, outs, target):
    """Schedule definition of softmax"""
    if target == "cuda":
        return topi.cuda.schedule_softmax(outs)
    # naive schedule
    return tvm.create_schedule([x.op for x in outs])

# Mark softmax as extern as we do not fuse it in call cases
reg.register_pattern("softmax", OpPattern.OPAQUE)


# dense
@reg.register_compute("dense")
def compute_dense(attrs, inputs, _):
    """Compute definition of dense"""
    if attrs.get_bool("use_bias"):
        return topi.nn.dense(inputs[0], inputs[1], bias=inputs[2])
    return topi.nn.dense(inputs[0], inputs[1])

@reg.register_schedule("dense")
def schedule_dense(_, outs, target):
    """Schedule definition of dense"""
    if target == "cuda":
        return topi.cuda.schedule_dense(outs)
    # naive schedule
    return tvm.create_schedule([x.op for x in outs])

reg.register_pattern("dense", OpPattern.OUT_ELEMWISE_FUSABLE)


# conv2d
@reg.register_compute("conv2d")
def compute_conv2d(attrs, inputs, _):
    """Compute definition of conv2d"""
    padding = attrs.get_int_tuple("padding")
    strides = attrs.get_int_tuple("strides")
    dilation = attrs.get_int_tuple("dilation")
    groups = attrs.get_int("groups")
    channels = attrs.get_int("channels")
    layout = attrs["layout"]
    assert layout == "NCHW", "only support nchw for now"
    assert dilation == (1, 1), "not support dilate now"
    if groups == 1:
        out = topi.nn.conv2d_nchw(inputs[0], inputs[1], strides, padding)
    elif groups == get_const_int(inputs[0].shape[1]) and groups == channels:
        out = topi.nn.depthwise_conv2d_nchw(inputs[0], inputs[1], strides, padding)
    else:
        raise ValueError("not support arbitrary group number for now")
    if attrs.get_bool("use_bias"):
        bias = inputs[2]
        bias = topi.expand_dims(bias, axis=1, num_newaxis=2)
        out = topi.broadcast_add(out, bias)
    return out

@reg.register_schedule("conv2d")
def schedule_conv2d(attrs, outs, target):
    """Schedule definition of conv2d"""
    groups = attrs.get_int("groups")
    if target == "cuda":
        if groups == 1:
            return topi.cuda.schedule_conv2d_nchw(outs)
        return topi.cuda.schedule_depthwise_conv2d_nchw(outs)
    # naive schedule
    return tvm.create_schedule([x.op for x in outs])

reg.register_pattern("conv2d", OpPattern.OUT_ELEMWISE_FUSABLE)


# max_pool2d
@reg.register_compute("max_pool2d")
def compute_max_pool2d(attrs, inputs, _):
    """Compute definition of max_pool2d"""
    pool_size = attrs.get_int_tuple("pool_size")
    strides = attrs.get_int_tuple("strides")
    padding = attrs.get_int_tuple("padding")
    layout = attrs["layout"]
    ceil_mode = attrs["ceil_mode"]
    assert layout == "NCHW", "only support nchw for now"
    assert ceil_mode == "False", "not support ceil_mode now"
    return topi.nn.pool(inputs[0], pool_size, strides, padding, pool_type='max')

@reg.register_schedule("max_pool2d")
def schedule_max_pool2d(_, outs, target):
    """Schedule definition of max_pool2d"""
    if target == "cuda":
        return topi.cuda.schedule_pool(outs)
    # naive schedule
    return tvm.create_schedule([x.op for x in outs])

reg.register_pattern("max_pool2d", OpPattern.OUT_ELEMWISE_FUSABLE)


# avg_pool2d
@reg.register_compute("avg_pool2d")
def compute_avg_pool2d(attrs, inputs, _):
    """Compute definition of avg_pool2d"""
    pool_size = attrs.get_int_tuple("pool_size")
    strides = attrs.get_int_tuple("strides")
    padding = attrs.get_int_tuple("padding")
    layout = attrs["layout"]
    ceil_mode = attrs["ceil_mode"]
    assert layout == "NCHW", "only support nchw for now"
    assert ceil_mode == "False", "not support ceil_mode now"
    return topi.nn.pool(inputs[0], pool_size, strides, padding, pool_type='avg')

@reg.register_schedule("avg_pool2d")
def schedule_avg_pool2d(_, outs, target):
    """Schedule definition of avg_pool2d"""
    if target == "cuda":
        return topi.cuda.schedule_pool(outs)
    # naive schedule
    return tvm.create_schedule([x.op for x in outs])

reg.register_pattern("avg_pool2d", OpPattern.OUT_ELEMWISE_FUSABLE)


# global_max_pool2d
@reg.register_compute("global_max_pool2d")
def compute_global_max_pool2d(attrs, inputs, _):
    """Compute definition of global_max_pool2d"""
    layout = attrs["layout"]
    assert layout == "NCHW", "only support nchw for now"
    return topi.nn.global_pool(inputs[0], pool_type='max')

@reg.register_schedule("global_max_pool2d")
def schedule_global_max_pool2d(_, outs, target):
    """Schedule definition of global_max_pool2d"""
    if target == "cuda":
        return topi.cuda.schedule_global_pool(outs)
    # naive schedule
    return tvm.create_schedule([x.op for x in outs])

reg.register_pattern("global_max_pool2d", OpPattern.OUT_ELEMWISE_FUSABLE)


# global_avg_pool2d
@reg.register_compute("global_avg_pool2d")
def compute_global_avg_pool2d(attrs, inputs, _):
    """Compute definition of global_avg_pool2d"""
    layout = attrs["layout"]
    assert layout == "NCHW", "only support nchw for now"
    return topi.nn.global_pool(inputs[0], pool_type='avg')

@reg.register_schedule("global_avg_pool2d")
def schedule_global_avg_pool2d(_, outs, target):
    """Schedule definition of global_avg_pool2d"""
    if target == "cuda":
        return topi.cuda.schedule_global_pool(outs)
    # naive schedule
    return tvm.create_schedule([x.op for x in outs])

reg.register_pattern("global_avg_pool2d", OpPattern.OUT_ELEMWISE_FUSABLE)
