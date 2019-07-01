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

"""Namespace for supporting packed_conv2d + ewise variant of nnvm."""
from __future__ import absolute_import as _abs

import logging

import tvm
import topi

from nnvm.top import registry as reg, OpPattern
from nnvm.top import nn as _nn

from .vta_conv2d import is_packed_layout
from ..environment import get_env

@tvm.register_func("nnvm.compiler.build_target", override=True)
def _build(funcs, target, target_host):
    tvm_t = tvm.target.create(target)
    if tvm_t.device_name == "vta":
        return tvm.build(funcs, target="ext_dev", target_host=target_host)
    if tvm_t.device_name == "rasp" or tvm_t.device_name == "vtacpu":
        return tvm.build(funcs, target=target_host)
    return tvm.build(funcs, target=target)

@tvm.register_func("nnvm.compiler.lower", override=True)
def _lower(sch, inputs, func_name, graph):
    import traceback
    # pylint: disable=broad-except
    try:
        f = tvm.lower(sch, inputs, name=func_name)
        if "quantized_conv2d" in func_name:
            logging.info(graph.ir(join_entry_attrs=["shape"]))
    except Exception:
        msg = traceback.format_exc()
        msg += "Error during compile graph\n"
        msg += "--------------------------\n"
        msg += graph.ir(join_entry_attrs=["shape"])
        raise RuntimeError(msg)
    return f if isinstance(
        f, (tvm.container.Array, tuple, list)) else [f]

# override to force partition at copy
reg.register_pattern("copy", OpPattern.INJECTIVE, level=15)

@reg.register_compute("clip", level=15)
def compute_clip(attrs, inputs, _):
    """ Clip operator. """
    x = inputs[0]
    a_min = attrs.get_float("a_min")
    a_max = attrs.get_float("a_max")
    const_min = tvm.const(a_min, x.dtype)
    const_max = tvm.const(a_max, x.dtype)
    with tvm.tag_scope(topi.tag.ELEMWISE):
        x = tvm.compute(
            x.shape, lambda *i: tvm.min(x(*i), const_max), name="clipA")
        x = tvm.compute(
            x.shape, lambda *i: tvm.max(x(*i), const_min), name="clipB")
    return x

@reg.register_compute("conv2d", level=15)
def compute_conv2d(attrs, inputs, out):
    """ Compute definition of conv2d """
    padding = attrs.get_int_tuple("padding")
    strides = attrs.get_int_tuple("strides")
    dilation = attrs.get_int_tuple("dilation")
    groups = attrs.get_int("groups")
    layout = attrs["layout"]
    out_dtype = attrs['out_dtype']

    assert dilation == (1, 1), "not support dilate now"
    if is_packed_layout(layout):
        if groups == 1:
            assert groups == 1
            env = get_env()
            assert env.LOG_INP_WIDTH == 3, "only support 8bit inp for now"
            assert env.LOG_OUT_WIDTH == 3, "only support 8bit inp for now"
            inputs = list(inputs)
            assert inputs[1].dtype == "int8"
            return topi.nn.conv2d(inputs[0], inputs[1], strides,
                                  padding, dilation, layout, out_dtype)
        return topi.nn.group_conv2d_nchw(inputs[0], inputs[1], strides,
                                         padding, dilation, groups, out_dtype)

    with tvm.target.arm_cpu(tvm.target.current_target().model):
        return _nn.compute_conv2d(attrs, inputs, out)

@reg.register_schedule("conv2d", level=15)
def schedule_conv2d(attrs, outs, target):
    """ Schedule definition of conv2d """
    layout = attrs["layout"]
    groups = attrs.get_int('groups')

    if is_packed_layout(layout):
        target = tvm.target.create(target)
        if target.device_name == "vta":
            if groups == 1:
                return topi.generic.schedule_conv2d_nchw(outs)
            return topi.generic.schedule_group_conv2d_nchw(outs)
        elif str(target).startswith("llvm"):
            return tvm.create_schedule([x.op for x in outs])
        else:
            raise RuntimeError("not support target %s" % target)

    with tvm.target.arm_cpu(tvm.target.current_target().model):
        return _nn.schedule_conv2d(attrs, outs, tvm.target.current_target())

@reg.register_alter_op_layout("conv2d", level=15)
def alter_conv2d_layout(attrs, inputs, out):
    layout = attrs['layout']
    if is_packed_layout(layout):
        return None

    with tvm.target.arm_cpu(tvm.target.current_target().model):
        return _nn.alter_conv2d_layout(attrs, inputs, out)
