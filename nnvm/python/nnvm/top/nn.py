"""Definition of nn ops"""
from __future__ import absolute_import

import tvm
import topi
from topi.util import get_const_int
from .tensor import schedule_elemwise
from ..compiler import registry as reg
from ..compiler import OpPattern

# relu
@reg.register_compute("relu")
def compute_relu(_, inputs):
    """Compute definition of relu"""
    return topi.nn.relu(inputs[0])

@reg.register_schedule("relu")
def schedule_relu(_, outs, target):
    """Schedule definition of relu"""
    return schedule_elemwise(_, outs, target)

reg.register_pattern("relu", OpPattern.ELEM_WISE)


# softmax
@reg.register_compute("softmax")
def compute_softmax(attrs, inputs):
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

reg.register_pattern("softmax", OpPattern.COMPLEX)


# conv
@reg.register_compute("conv2d")
def compute_conv2d(attrs, inputs):
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
        bias = topi.broadcast_to(bias, (1, bias.shape[0], 1, 1))
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

reg.register_pattern("conv2d", OpPattern.COMPLEX)
