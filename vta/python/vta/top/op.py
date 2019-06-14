"""Namespace for supporting packed_conv2d + ewise variant of nnvm."""
from __future__ import absolute_import as _abs

import logging

import tvm
import topi

from tvm.relay.op import op as reg
from tvm.relay.op.op import OpPattern
from tvm.relay.op.nn import _nn

from .vta_conv2d import is_packed_layout
from ..environment import get_env

# override to force partition at copy
reg.register_pattern("copy", OpPattern.INJECTIVE, level=15)

@reg.register_compute("clip", level=15)
def compute_clip(attrs, inputs, output_type, target):
    """ Clip operator. """
    x = inputs[0]
    a_min = attrs.a_min
    a_max = attrs.a_max
    const_min = tvm.const(a_min, x.dtype)
    const_max = tvm.const(a_max, x.dtype)
    with tvm.tag_scope(topi.tag.ELEMWISE):
        x = tvm.compute(
            x.shape, lambda *i: tvm.min(x(*i), const_max), name="clipA")
        x = tvm.compute(
            x.shape, lambda *i: tvm.max(x(*i), const_min), name="clipB")
    return [x]


@reg.register_compute("nn.conv2d", level=15)
def compute_conv2d(attrs, inputs, output_type, target):
    """ Compute definition of conv2d """
    padding = topi.util.get_const_tuple(attrs.padding)
    strides = topi.util.get_const_tuple(attrs.strides)
    dilation = tuple([int(d) for d in attrs.dilation])
    groups = attrs.groups
    layout = attrs.data_layout
    out_dtype = attrs.out_dtype

    assert dilation == (1, 1), "support for dilation limited to (1, 1)"
    if is_packed_layout(layout):
        if groups == 1:
            assert groups == 1
            env = get_env()
            assert env.LOG_INP_WIDTH == 3, "only support 8bit inp for now"
            assert env.LOG_WGT_WIDTH == 3, "only support 8bit wgt for now"
            inputs = list(inputs)
            assert inputs[1].dtype == "int8"
            return [topi.nn.conv2d(inputs[0], inputs[1], strides, padding, dilation, layout, out_dtype)]
        else:
            return [topi.nn.group_conv2d_nchw(inputs[0], inputs[1], strides, padding, dilation, groups, out_dtype)]

    with tvm.target.arm_cpu(tvm.target.current_target().model):
        return _nn.compute_conv2d(attrs, inputs, output_type, target)


@reg.register_schedule("nn.conv2d", level=15)
def schedule_conv2d(attrs, outs, target):
    """ Schedule definition of conv2d """
    groups = attrs.groups
    layout = attrs.data_layout

    if is_packed_layout(layout):
        target = tvm.target.create(target)
        if target.device_name == "vta":
            if groups == 1:
                return topi.generic.schedule_conv2d_nchw(outs)
            else:
                return topi.generic.schedule_group_conv2d_nchw(outs)
        elif str(target).startswith("llvm"):
            return tvm.create_schedule([x.op for x in outs])
        else:
            raise RuntimeError("Target %s is not supported" % target)

    with tvm.target.arm_cpu(tvm.target.current_target().model):
        return _nn.schedule_conv2d(attrs, outs, tvm.target.current_target())


@reg.register_compute("nn.dense", level=15)
def compute_dense(attrs, inputs, out_type, target):
    """Compute definition of dense"""
    out_dtype = attrs.out_dtype
    out_dtype = inputs[0].dtype if out_dtype == "" else out_dtype

    if inputs[0].shape == 4: # this implies the layout is packed
        return [topi.nn.dense(inputs[0], inputs[1], None, out_dtype)]

    with tvm.target.arm_cpu(tvm.target.current_target().model):
        return _nn.compute_dense(attrs, inputs, out_type, target)


@reg.register_schedule("nn.dense", level=15)
def schedule_dense(attrs, outs, target):
    """Schedule definition of dense"""

    if outs[0].shape == 4: # this implies the layout is packed
        target = tvm.target.create(target)
        if target.device_name == "vta":
            return topi.generic.schedule_dense(outs)
        elif str(target).startswith("llvm"):
            return tvm.create_schedule([x.op for x in outs])
        else:
            raise RuntimeError("Target %s is not supported" % target)

    with tvm.target.arm_cpu(tvm.target.current_target().model):
        return _nn.schedule_dense(attrs, outs, tvm.target.current_target())
