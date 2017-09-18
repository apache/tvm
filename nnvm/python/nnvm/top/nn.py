"""Definition of nn ops"""
from __future__ import absolute_import

import tvm
import topi
from ..compiler import registry as reg
from ..compiler import OpPattern

# conv
@reg.register_compute("conv2d")
def compute_conv2d(attrs, inputs):
    """Compute definition of conv2d"""
    padding = attrs.get_int_tuple("padding")
    strides = attrs.get_int_tuple("strides")
    dilation = attrs.get_int_tuple("dilation")
    layout = attrs["layout"]
    assert layout == "NCHW", "only support nchw for now"
    assert dilation == (1, 1), "not support dilate now"
    out = topi.nn.conv2d_nchw(inputs[0], inputs[1], strides, padding)
    if attrs.get_bool("use_bias"):
        bias = inputs[2]
        bias = topi.broadcast_to(bias, (1, bias.shape[0], 1, 1))
        out = topi.broadcast_add(out, bias)
    return out


@reg.register_schedule("conv2d")
def schedule_conv2d(_, outs, target):
    """Schedule definition of conv2d"""
    if target == "cuda":
        return topi.cuda.schedule_conv2d_nchw(outs)
    # naive schedule
    return tvm.create_schedule([x.op for x in outs])

reg.register_pattern("conv2d", OpPattern.COMPLEX)
