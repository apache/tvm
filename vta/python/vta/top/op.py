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
"""Namespace for supporting Relay operators on VTA."""
from __future__ import absolute_import as _abs

import tvm
from tvm import te
from tvm import topi

from tvm.relay.op import op as reg
from tvm.relay.op import strategy as _strategy
from tvm.relay.op.op import OpPattern, OpStrategy

from .util import is_packed_layout
from .vta_conv2d import conv2d_packed, schedule_conv2d_packed
from .vta_conv2d_transpose import conv2d_transpose_packed, schedule_conv2d_transpose_packed
from .vta_group_conv2d import group_conv2d_packed, schedule_group_conv2d_packed
from .vta_dense import dense_packed, schedule_dense_packed
from ..environment import get_env


# override to force partition at copy
reg.register_pattern("copy", OpPattern.INJECTIVE, level=15)

# add clip vta strategy
def compute_clip_vta(attrs, inputs, output_type):
    """ Clip operator. """
    x = inputs[0]
    a_min = attrs.a_min
    a_max = attrs.a_max
    const_min = tvm.tir.const(a_min, x.dtype)
    const_max = tvm.tir.const(a_max, x.dtype)
    with tvm.te.tag_scope(topi.tag.ELEMWISE):
        x = te.compute(x.shape, lambda *i: tvm.te.min(x(*i), const_max), name="clipA")
        x = te.compute(x.shape, lambda *i: tvm.te.max(x(*i), const_min), name="clipB")
    return [x]


def clip_strategy_vta(attrs, inputs, out_type, target):
    strategy = OpStrategy()
    strategy.add_implementation(
        compute_clip_vta,
        _strategy.wrap_topi_schedule(topi.generic.schedule_injective),
        name="clip.vta",
    )
    return strategy


reg.get("clip").get_attr("FTVMStrategy").register(clip_strategy_vta, "vta")


@_strategy.conv2d_strategy.register("vta")
def conv2d_strategy_vta(attrs, inputs, out_type, target):
    """conv2d vta strategy"""
    strategy = OpStrategy()
    kernel = inputs[1]
    dilation = topi.util.get_const_tuple(attrs.dilation)
    groups = attrs.groups
    layout = attrs.data_layout

    assert dilation == (1, 1), "support for dilation limited to (1, 1)"
    if is_packed_layout(layout):
        if groups == 1:
            env = get_env()
            assert env.LOG_INP_WIDTH == 3, "only support 8bit inp for now"
            assert env.LOG_WGT_WIDTH == 3, "only support 8bit wgt for now"
            assert kernel.dtype == "int8"

            strategy.add_implementation(
                _strategy.wrap_compute_conv2d(conv2d_packed, True),
                _strategy.wrap_topi_schedule(schedule_conv2d_packed),
                name="conv2d_packed.vta",
            )
        else:  # group_conv2d
            strategy.add_implementation(
                _strategy.wrap_compute_conv2d(group_conv2d_packed, has_groups=True),
                _strategy.wrap_topi_schedule(schedule_group_conv2d_packed),
                name="group_conv2d_packed.vta",
            )
        return strategy

    # If it's not packed, run on ARM CPU
    arm_tgt = tvm.target.arm_cpu(target.model)
    return _strategy.arm_cpu.conv2d_strategy_arm_cpu(attrs, inputs, out_type, arm_tgt)


@_strategy.conv2d_transpose_strategy.register("vta")
def conv2d_transpose_strategy_vta(attrs, inputs, out_type, target):
    """conv2d_transpose vta strategy"""
    dilation = topi.util.get_const_tuple(attrs.dilation)
    layout = attrs.data_layout
    assert dilation == (1, 1), "support for dilation limited to (1, 1)"

    if is_packed_layout(layout):
        strategy = OpStrategy()
        strategy.add_implementation(
            _strategy.wrap_compute_conv2d_transpose(conv2d_transpose_packed),
            _strategy.wrap_topi_schedule(schedule_conv2d_transpose_packed),
            name="conv2d_transpose_packed.vta",
        )
        return strategy

    # If it's not packed, run on ARM CPU
    arm_tgt = tvm.target.arm_cpu(target.model)
    return _strategy.arm_cpu.conv2d_transpose_strategy_arm_cpu(attrs, inputs, out_type, arm_tgt)


@_strategy.dense_strategy.register("vta")
def dense_strategy_vta(attrs, inputs, out_type, target):
    """dense vta strategy"""
    if inputs[0].shape == 4:  # this implies the layout is packed
        strategy = OpStrategy()
        strategy.add_implementation(
            _strategy.wrap_compute_dense(dense_packed),
            _strategy.wrap_topi_schedule(schedule_dense_packed),
            name="dense_packed.vta",
        )
        return strategy
    # If it's not packed, run on ARM CPU
    arm_tgt = tvm.target.arm_cpu(target.model)
    return _strategy.x86.dense_strategy_cpu(attrs, inputs, out_type, arm_tgt)
