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
from tvm import autotvm
from tvm import topi

from tvm.relay.op import op as reg
from tvm.relay.op import strategy as _strategy
from tvm.relay.op.op import OpPattern, OpStrategy

from .utils import is_packed_layout
from .vta_conv2d import conv2d_packed, schedule_conv2d_packed
from .vta_conv2d_transpose import conv2d_transpose_packed, schedule_conv2d_transpose_packed
from .vta_group_conv2d import group_conv2d_packed, schedule_group_conv2d_packed
from .vta_dense import dense_packed, schedule_dense_packed
from ..environment import get_env

ENV = get_env()

# override to force partition at copy
reg.register_pattern("copy", OpPattern.INJECTIVE, level=15)

# add clip vta strategy
def compute_clip_vta(attrs, inputs, output_type):
    """Clip operator."""
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


@autotvm.register_topi_compute("add.vta")
def add_packed(cfg, lhs, rhs):
    return topi.add(lhs, rhs)


@autotvm.register_topi_compute("multiply.vta")
def multiply_packed(cfg, lhs, rhs):
    return topi.multiply(lhs, rhs)


def schedule_alu_packed(cfg, outs):
    """alu packed schedule"""
    assert len(outs) == 1

    def is_cast_op(op):
        return op.name == "T_cast"

    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    output = outs[0]
    s = te.create_schedule([x.op for x in outs])
    te.schedule.AutoInlineInjective(s)

    # other target does not support alu-only ops
    if not (ENV.TARGET in ["sim", "tsim", "intelfocl"]):
        return s

    # only put the int-related ops to vta
    if "int" in output.dtype and len(output.shape) == 6:
        ewise_inputs = []
        ewise_ops = []
        const_ops = []

        def _traverse(op):
            if topi.tag.is_broadcast(op.tag):
                if not op.same_as(output.op):
                    if not op.axis:
                        const_ops.append(op)
                    elif not is_cast_op(op):
                        ewise_ops.append(op)

                for tensor in op.input_tensors:
                    if isinstance(tensor.op, tvm.te.PlaceholderOp):
                        ewise_inputs.append((op, tensor))
                    elif is_cast_op(tensor.op) and not op.same_as(output.op):
                        ewise_inputs.append((op, tensor))
                    else:
                        _traverse(tensor.op)
            else:
                for tensor in op.input_tensors:
                    if (not isinstance(tensor.op, tvm.te.PlaceholderOp)) and (
                        not is_cast_op(tensor.op)
                    ):
                        _traverse(tensor.op)

        op = output.op
        _traverse(op)
        for _, t in ewise_inputs:
            if t.dtype == "float32":
                return s

        x_bo, x_co, x_i, x_j, x_bi, x_ci = s[output].op.axis

        cfg.define_split("tile_co", x_co, num_outputs=2)
        cfg.define_split("tile_h", x_i, num_outputs=2)
        cfg.define_split("tile_w", x_j, num_outputs=2)

        x_co0, x_co1 = cfg["tile_co"].apply(s, output, x_co)
        x_i0, x_i1 = cfg["tile_h"].apply(s, output, x_i)
        x_j0, x_j1 = cfg["tile_w"].apply(s, output, x_j)
        s[output].reorder(x_bo, x_i0, x_co0, x_j0, x_co1, x_i1, x_j1, x_bi, x_ci)
        store_pt = x_j0

        for e_o in ewise_ops:
            s[e_o].set_scope(ENV.acc_scope)
            s[e_o].pragma(s[e_o].op.axis[0], ENV.alu)
            s[e_o].compute_at(s[output], store_pt)

        # cache read input
        cache_read_ewise = []
        for consumer, tensor in ewise_inputs:
            cache_read_ewise.append(s.cache_read(tensor, ENV.acc_scope, [consumer]))

        for tensor in cache_read_ewise:
            if s[tensor].op.axis:
                s[tensor].pragma(s[tensor].op.axis[0], ENV.dma_copy)
            s[tensor].compute_at(s[output], store_pt)

        for op in const_ops:
            s[op].compute_inline()

        s[output].pragma(x_co1, ENV.dma_copy)

    return s


@autotvm.register_topi_schedule("add.vta")
def schedule_add_packed(cfg, outs):
    return schedule_alu_packed(cfg, outs)


@autotvm.register_topi_schedule("multiply.vta")
def schedule_multiply_packed(cfg, outs):
    return schedule_alu_packed(cfg, outs)


def add_strategy_vta(attrs, inputs, out_type, target):
    strategy = OpStrategy()
    strategy.add_implementation(
        _strategy.wrap_topi_compute(add_packed),
        _strategy.wrap_topi_schedule(schedule_add_packed),
        name="add.vta",
    )
    return strategy


def multiply_strategy_vta(attrs, inputs, out_type, target):
    strategy = OpStrategy()
    strategy.add_implementation(
        _strategy.wrap_topi_compute(multiply_packed),
        _strategy.wrap_topi_schedule(schedule_multiply_packed),
        name="multiply.vta",
    )
    return strategy


# other target does not support alu-only ops
if ENV.TARGET in ["sim", "intelfocl"]:
    reg.get("add").get_attr("FTVMStrategy").register(add_strategy_vta, "vta")
    reg.get("multiply").get_attr("FTVMStrategy").register(multiply_strategy_vta, "vta")


@_strategy.conv2d_strategy.register("vta")
def conv2d_strategy_vta(attrs, inputs, out_type, target):
    """conv2d vta strategy"""
    strategy = OpStrategy()
    kernel = inputs[1]
    dilation = topi.utils.get_const_tuple(attrs.dilation)
    groups = attrs.groups
    layout = attrs.data_layout

    assert dilation == (1, 1), "support for dilation limited to (1, 1)"
    if is_packed_layout(layout):
        if groups == 1:
            assert ENV.LOG_INP_WIDTH == 3, "only support 8bit inp for now"
            assert ENV.LOG_WGT_WIDTH == 3, "only support 8bit wgt for now"
            assert kernel.dtype == "int8"

            strategy.add_implementation(
                _strategy.wrap_compute_conv2d(conv2d_packed, need_data_layout=True),
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
    dilation = topi.utils.get_const_tuple(attrs.dilation)
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
    if len(inputs[0].shape) == 4:  # this implies the layout is packed
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
