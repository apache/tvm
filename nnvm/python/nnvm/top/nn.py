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
def compute_relu(attrs, inputs, _):
    """Compute definition of relu"""
    return topi.nn.leaky_relu(inputs[0])

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
        return topi.nn.fully_connected_with_bias(
            inputs[0], inputs[1], inputs[2])
    return topi.nn.fully_connected(inputs[0], inputs[1])

@reg.register_schedule("dense")
def schedule_dense(_, outs, target):
    """Schedule definition of dense"""
    if target == "cuda":
        raise ValueError("fully_connected not yet implemented")
    # naive schedule
    return tvm.create_schedule([x.op for x in outs])

# register extern for now, change me when fusion is enabled.
reg.register_pattern("dense", OpPattern.OPAQUE)


# conv
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
