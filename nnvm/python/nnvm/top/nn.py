# pylint: disable=invalid-name, unused-argument
"""Definition of nn ops"""
from __future__ import absolute_import

import tvm
import topi
from topi.util import get_const_int
from .tensor import _fschedule_broadcast
from . import registry as reg
from .registry import OpPattern

# relu
reg.register_schedule("relu", _fschedule_broadcast)
reg.register_pattern("relu", OpPattern.ELEMWISE)


# leaky_relu
reg.register_schedule("leaky_relu", _fschedule_broadcast)
reg.register_pattern("leaky_relu", OpPattern.ELEMWISE)


# flatten
reg.register_schedule("flatten", _fschedule_broadcast)
reg.register_pattern("flatten", OpPattern.INJECTIVE)


# pad
reg.register_schedule("pad", _fschedule_broadcast)
reg.register_pattern("pad", OpPattern.INJECTIVE)


# softmax
@reg.register_schedule("softmax")
def schedule_softmax(_, outs, target):
    """Schedule definition of softmax"""
    with tvm.target.create(target):
        return topi.generic.schedule_softmax(outs)

reg.register_pattern("softmax", OpPattern.OPAQUE)


# log softmax
@reg.register_schedule("log_softmax")
def schedule_log_softmax(_, outs, target):
    """Schedule definition of softmax"""
    with tvm.target.create(target):
        return topi.generic.schedule_softmax(outs)

# Mark softmax as extern as we do not fuse it in call cases
reg.register_pattern("log_softmax", OpPattern.OPAQUE)


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
    with tvm.target.create(target):
        return topi.generic.schedule_dense(outs)

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
        out = topi.nn.conv2d(inputs[0], inputs[1], strides, padding)
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
    with tvm.target.create(target):
        if groups == 1:
            return topi.generic.schedule_conv2d_nchw(outs)
        return topi.generic.schedule_depthwise_conv2d_nchw(outs)

reg.register_pattern("conv2d", OpPattern.OUT_ELEMWISE_FUSABLE)


# conv2d_transpose
@reg.register_compute("conv2d_transpose")
def compute_conv2d_transpose(attrs, inputs, _):
    """Compute definition of conv2d_transpose"""
    padding = attrs.get_int_tuple("padding")
    strides = attrs.get_int_tuple("strides")
    dilation = attrs.get_int_tuple("dilation")
    groups = attrs.get_int("groups")
    layout = attrs["layout"]
    assert layout == "NCHW", "only support nchw for now"
    assert dilation == (1, 1), "not support dilate now"
    assert groups == 1, "only support groups == 1 for now"
    out = topi.nn.conv2d_transpose_nchw(inputs[0], inputs[1], strides, padding)
    if attrs.get_bool("use_bias"):
        bias = inputs[2]
        bias = topi.expand_dims(bias, axis=1, num_newaxis=2)
        out = topi.broadcast_add(out, bias)
    output_padding = attrs.get_int_tuple("output_padding")
    out = topi.nn.pad(out, \
        [0, 0, 0, 0], [0, 0, output_padding[0], output_padding[1]])
    return out

@reg.register_schedule("conv2d_transpose")
def schedule_conv2d_transpose(attrs, outs, target):
    """Schedule definition of conv2d_transpose"""
    with tvm.target.create(target):
        return topi.generic.schedule_conv2d_transpose_nchw(outs)

reg.register_pattern("conv2d_transpose", OpPattern.OUT_ELEMWISE_FUSABLE)


# max_pool2d
@reg.register_schedule("max_pool2d")
def schedule_max_pool2d(_, outs, target):
    """Schedule definition of max_pool2d"""
    with tvm.target.create(target):
        return topi.generic.schedule_pool(outs)

reg.register_pattern("max_pool2d", OpPattern.OUT_ELEMWISE_FUSABLE)


# avg_pool2d
@reg.register_schedule("avg_pool2d")
def schedule_avg_pool2d(_, outs, target):
    """Schedule definition of avg_pool2d"""
    with tvm.target.create(target):
        return topi.generic.schedule_pool(outs)

reg.register_pattern("avg_pool2d", OpPattern.OUT_ELEMWISE_FUSABLE)


# global_max_pool2d
@reg.register_schedule("global_max_pool2d")
def schedule_global_max_pool2d(_, outs, target):
    """Schedule definition of global_max_pool2d"""
    with tvm.target.create(target):
        return topi.generic.schedule_global_pool(outs)

reg.register_pattern("global_max_pool2d", OpPattern.OUT_ELEMWISE_FUSABLE)


# global_avg_pool2d
@reg.register_schedule("global_avg_pool2d")
def schedule_global_avg_pool2d(_, outs, target):
    """Schedule definition of global_avg_pool2d"""
    with tvm.target.create(target):
        return topi.generic.schedule_global_pool(outs)

reg.register_pattern("global_avg_pool2d", OpPattern.OUT_ELEMWISE_FUSABLE)


@reg.register_compute("upsampling")
def compute_upsampling(attrs, inputs, _):
    """Compute definition of upsampling"""
    scale = attrs.get_int("scale")
    return topi.nn.upsampling(inputs[0], scale)

@reg.register_schedule("upsampling")
def schedule_upsampling(_, outs, target):
    """Compute definition of upsampling"""
    with tvm.target.create(target):
        return topi.generic.schedule_injective(outs)

reg.register_pattern("upsampling", OpPattern.INJECTIVE)
