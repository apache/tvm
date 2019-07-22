# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=unused-argument, ungrouped-imports
"""Namespace for supporting packed_conv2d + ewise variant of nnvm."""
from __future__ import absolute_import as _abs

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

    if target.device_name == "vta":
        assert dilation == (1, 1), "support for dilation limited to (1, 1)"
        if is_packed_layout(layout):
            if groups == 1:
                assert groups == 1
                env = get_env()
                assert env.LOG_INP_WIDTH == 3, "only support 8bit inp for now"
                assert env.LOG_WGT_WIDTH == 3, "only support 8bit wgt for now"
                inputs = list(inputs)
                assert inputs[1].dtype == "int8"
                return [topi.nn.conv2d(inputs[0],
                                       inputs[1],
                                       strides,
                                       padding,
                                       dilation,
                                       layout,
                                       out_dtype)]
            return [topi.nn.group_conv2d_nchw(inputs[0],
                                              inputs[1],
                                              strides,
                                              padding,
                                              dilation,
                                              groups,
                                              out_dtype)]
        # If it's not packed, run on ARM CPU
        with tvm.target.arm_cpu(tvm.target.current_target().model):
            return _nn.compute_conv2d(attrs, inputs, output_type, target)

    # If VTA is not the target, default to _nn def
    return _nn.compute_conv2d(attrs, inputs, output_type, target)


@reg.register_schedule("nn.conv2d", level=15)
def schedule_conv2d(attrs, outs, target):
    """ Schedule definition of conv2d """
    groups = attrs.groups
    layout = attrs.data_layout

    if target.device_name == "vta":
        if is_packed_layout(layout):
            target = tvm.target.create(target)
            assert target.device_name == "vta"
            if groups == 1:
                return topi.generic.schedule_conv2d_nchw(outs)
            return topi.generic.schedule_group_conv2d_nchw(outs)
        # If it's not packed, run on ARM CPU
        with tvm.target.arm_cpu(tvm.target.current_target().model):
            return _nn.schedule_conv2d(attrs, outs, tvm.target.current_target())

    # If VTA is not the target, default to _nn def
    return _nn.schedule_conv2d(attrs, outs, target)


@reg.register_compute("nn.dense", level=15)
def compute_dense(attrs, inputs, out_type, target):
    """Compute definition of dense"""
    out_dtype = attrs.out_dtype
    out_dtype = inputs[0].dtype if out_dtype == "" else out_dtype

    if target.device_name == "vta":
        if inputs[0].shape == 4: # this implies the layout is packed
            target = tvm.target.create(target)
            return [topi.nn.dense(inputs[0], inputs[1], None, out_dtype)]
        # If it's not packed, run on ARM CPU
        with tvm.target.arm_cpu(tvm.target.current_target().model):
            return _nn.compute_dense(attrs, inputs, out_type, target)

    # If VTA is not the target, default to _nn def
    return _nn.compute_dense(attrs, inputs, out_type, target)


@reg.register_schedule("nn.dense", level=15)
def schedule_dense(attrs, outs, target):
    """Schedule definition of dense"""
    if target.device_name == "vta":
        if outs[0].shape == 4: # this implies the layout is packed
            target = tvm.target.create(target)
            assert target.device_name == "vta"
            return topi.generic.schedule_dense(outs)
        # If it's not packed, run on ARM CPU
        with tvm.target.arm_cpu(tvm.target.current_target().model):
            return _nn.schedule_dense(attrs, outs, tvm.target.current_target())

    # If VTA is not the target, default to _nn def
    return _nn.schedule_dense(attrs, outs, target)
