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
"""
Namespace for the supported Relay operators on Gemmini
=====================
**Author**: `Federico Peccia <https://fPecc.github.io/>`_
"""

from __future__ import absolute_import as _abs

import tvm
from tvm import te
from tvm import autotvm
from tvm import topi

from tvm.relay.op import op as reg
from tvm.relay.op import strategy as _strategy
from tvm.relay.op.op import OpPattern, OpStrategy

from .gemmini_dense import gemm, schedule_gemm
from .gemmini_dense_cisc import gemm_cisc, schedule_gemm_cisc
from .gemmini_conv2d_cisc import conv2d_cisc, schedule_conv2d_cisc
from .gemmini_depthwise_conv2d_cisc import depthwise_conv2d_cisc, schedule_depthwise_conv2d_cisc
from .gemmini_add import add, schedule_add
from .gemmini_max_pool2d import max_pool2d, schedule_max_pool2d
from tvm.contrib.gemmini.environment import Environment

from tvm.topi.utils import const_vector, get_const_int, get_const_float
import numpy as np

ENV = Environment.instance()


def wrap_max_pool2d_topi_compute(topi_compute):
    """Wrapper for the max pool2d compute

    Args:
        topi_compute (function): function to wrap
    """

    def wrapper(attrs, inputs, out_type):
        return [
            topi_compute(
                *inputs,
                attrs.pool_size,
                attrs.pool_strides,
                attrs.pool_dilation,
                attrs.pool_padding,
            )
        ]

    return wrapper


@tvm.ir.register_op_attr("contrib.gemmini.max_pool2d", "FTVMStrategy")
def max_pool2d_strategy_gemmini(attrs, inputs, out_type, target):
    """Strategy implementations for Gemmini's max_pool2d operator

    Args:
        attrs (tvm.runtime.object.Object): attributes for the strategy
        inputs (tvm.ir.container.Array): inputs
        out_type (tvm.ir.tensor_type.TensorType): output type
        target (tvm.target.target.Target): target for the strategy

    Returns:
        OpStrategy: strategies implementation
    """
    if len(inputs) == 2:
        strategy = OpStrategy()
        strategy.add_implementation(
            wrap_max_pool2d_topi_compute(max_pool2d),
            _strategy.wrap_topi_schedule(schedule_max_pool2d),
            name="contrib.gemmini.max_pool2d",
            plevel=10,
        )
        return strategy
    return None


def wrap_add_topi_compute(topi_compute):
    """Wrapper for the add compute

    Args:
        topi_compute (function): function to wrap
    """

    def wrapper(attrs, inputs, out_type):
        ifm1_scale = float(attrs.ifm1_scale.data.numpy() / attrs.ofm_scale.data.numpy())
        ifm2_scale = float(attrs.ifm2_scale.data.numpy() / attrs.ofm_scale.data.numpy())
        return [topi_compute(*inputs, ifm1_scale, ifm2_scale)]

    return wrapper


@tvm.ir.register_op_attr("contrib.gemmini.add", "FTVMStrategy")
def add_strategy_gemmini(attrs, inputs, out_type, target):
    """Strategy implementations for Gemmini's add operator

    Args:
        attrs (tvm.runtime.object.Object): attributes for the strategy
        inputs (tvm.ir.container.Array): inputs
        out_type (tvm.ir.tensor_type.TensorType): output type
        target (tvm.target.target.Target): target for the strategy

    Returns:
        OpStrategy: strategies implementation
    """
    if len(inputs) == 3:
        strategy = OpStrategy()
        strategy.add_implementation(
            wrap_add_topi_compute(add),
            _strategy.wrap_topi_schedule(schedule_add),
            name="contrib.gemmini.add",
            plevel=10,
        )
        return strategy
    return None


def wrap_gemm_topi_compute(topi_compute):
    """Wrapper for the GEMM compute

    Args:
        topi_compute (function): function to wrap
    """

    def wrapper(attrs, inputs, out_type):
        return [
            topi_compute(
                *inputs, float(attrs.bias_scale.data.numpy() / attrs.ofm_scale.data.numpy())
            )
        ]

    return wrapper


@tvm.ir.register_op_attr("contrib.gemmini.gemm", "FTVMStrategy")
def gemm_strategy_gemmini(attrs, inputs, out_type, target):
    """Strategy implementations for Gemmini's GEMM operator

    Args:
        attrs (tvm.runtime.object.Object): attributes for the strategy
        inputs (tvm.ir.container.Array): inputs
        out_type (tvm.ir.tensor_type.TensorType): output type
        target (tvm.target.target.Target): target for the strategy

    Returns:
        OpStrategy: strategies implementation
    """
    if len(inputs) == 3:
        strategy = OpStrategy()
        strategy.add_implementation(
            wrap_gemm_topi_compute(gemm),
            _strategy.wrap_topi_schedule(schedule_gemm),
            name="contrib.gemmini.gemm",
            plevel=9,
        )
        strategy.add_implementation(
            wrap_gemm_topi_compute(gemm_cisc),
            _strategy.wrap_topi_schedule(schedule_gemm_cisc),
            name="contrib.gemmini.gemm_cisc",
            plevel=10,  # Higher -> used over the other one, unless AutoTVM says the other is better
        )
        return strategy
    return None


def wrap_conv2d_topi_compute(topi_compute):
    """Wrapper for the conv2d compute

    Args:
        topi_compute (function): function to wrap
    """

    def wrapper(attrs, inputs, out_type):
        if attrs.has_activation:
            gemmini_scale = float(
                attrs.activation_scale_in.data.numpy() / attrs.activation_scale_out.data.numpy()
            ) * float(attrs.bias_scale.data.numpy() / attrs.ofm_scale.data.numpy())
        else:
            gemmini_scale = float(attrs.bias_scale.data.numpy() / attrs.ofm_scale.data.numpy())
        return [
            topi_compute(
                *inputs,
                attrs.strides,
                attrs.padding,
                int(attrs.ifm_offset.data.numpy()),
                attrs.activation,
                gemmini_scale,
                attrs.pool_size,
                attrs.pool_strides,
                attrs.pool_dilation,
                attrs.pool_padding,
            )
        ]

    return wrapper


@tvm.ir.register_op_attr("contrib.gemmini.conv2d", "FTVMStrategy")
def conv2d_strategy_gemmini(attrs, inputs, out_type, target):
    """Strategy implementations for Gemmini's conv2d operator

    Args:
        attrs (tvm.runtime.object.Object): attributes for the strategy
        inputs (tvm.ir.container.Array): inputs
        out_type (tvm.ir.tensor_type.TensorType): output type
        target (tvm.target.target.Target): target for the strategy

    Returns:
        OpStrategy: strategies implementation
    """
    if len(inputs[0].shape) == 4:
        strategy = OpStrategy()
        if inputs[0].shape[1] == inputs[0].shape[2]:
            strategy.add_implementation(
                wrap_conv2d_topi_compute(conv2d_cisc),
                _strategy.wrap_topi_schedule(schedule_conv2d_cisc),
                name="contrib.gemmini.conv2d_cisc",
                plevel=10,
            )
        return strategy
    return None


def wrap_depthwise_conv2d_topi_compute(topi_compute):
    """Wrapper for the depthwise conv2d compute

    Args:
        topi_compute (function): function to wrap
    """

    def wrapper(attrs, inputs, out_type):
        return [
            topi_compute(
                *inputs,
                attrs.strides,
                attrs.padding,
                int(attrs.ifm_offset.data.numpy()),
                attrs.activation,
                float(attrs.bias_scale.data.numpy() / attrs.ofm_scale.data.numpy()),
            )
        ]

    return wrapper


@tvm.ir.register_op_attr("contrib.gemmini.depthwiseconv2d", "FTVMStrategy")
def depthwise_conv2d_strategy_gemmini(attrs, inputs, out_type, target):
    """Strategy implementations for Gemmini's depthwiseconv2d operator

    Args:
        attrs (tvm.runtime.object.Object): attributes for the strategy
        inputs (tvm.ir.container.Array): inputs
        out_type (tvm.ir.tensor_type.TensorType): output type
        target (tvm.target.target.Target): target for the strategy

    Returns:
        OpStrategy: strategies implementation
    """
    if len(inputs[0].shape) == 4:
        strategy = OpStrategy()
        if inputs[0].shape[1] == inputs[0].shape[2]:
            strategy.add_implementation(
                wrap_depthwise_conv2d_topi_compute(depthwise_conv2d_cisc),
                _strategy.wrap_topi_schedule(schedule_depthwise_conv2d_cisc),
                name="contrib.gemmini.depthwiseconv2d_cisc",
                plevel=10,
            )
        return strategy
    return None
