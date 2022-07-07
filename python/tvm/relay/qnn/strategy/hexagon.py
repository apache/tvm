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
"""Definition of Hexagon operator strategy."""
# pylint: disable=unused-argument,wildcard-import,unused-wildcard-import

from tvm import topi
from .generic import *
from ... import op as _op
from ...op.strategy.generic import is_depthwise_conv2d


# TODO: This is POC code. Change it on "hexagon" instead of "cpu"
@qnn_quantize_strategy.register("cpu")
def qnn_quantize_strategy_hexagon(attrs, inputs, out_type, target):
    """qnn.quantize strategy for Hexagon"""
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_quantize(topi.hexagon.qnn_quantize),
        wrap_topi_schedule(topi.hexagon.schedule_qnn_quantize),
        name="qnn_quantize.hexagon",
    )
    return strategy


# TODO: This is POC code. Change it on "hexagon" instead of "cpu"
@qnn_dequantize_strategy.register("cpu")
def qnn_dequantize_strategy_hexagon(attrs, inputs, out_type, target):
    """qnn.dequantize strategy for Hexagon"""
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_dequantize(topi.hexagon.qnn_dequantize),
        wrap_topi_schedule(topi.hexagon.schedule_qnn_dequantize),
        name="qnn_dequantize.hexagon",
    )
    return strategy


# TODO: This is POC code. Change it on "hexagon" instead of "cpu"
@qnn_requantize_strategy.register("cpu")
def qnn_requantize_strategy_hexagon(attrs, inputs, out_type, target):
    """qnn.requantize strategy for Hexagon"""
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_quantize(topi.hexagon.qnn_requantize),
        wrap_topi_schedule(topi.hexagon.schedule_qnn_requantize),
        name="qnn_requantize.hexagon",
    )
    return strategy


# TODO: This is POC code. Change it on "hexagon" instead of "cpu"
@qnn_add_strategy.register("cpu")
def qnn_add_strategy_hexagon(attrs, inputs, out_type, target):
    """qnn.add strategy for Hexagon"""
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_topi_compute(topi.hexagon.qnn_add),
        wrap_topi_schedule(topi.hexagon.schedule_qnn_add),
        name="qnn_add.hexagon",
    )
    return strategy


# TODO: This is POC code. Change it on "hexagon" instead of "cpu"
@qnn_conv2d_strategy.register("cpu")
def qnn_conv2d_strategy_hexagon(attrs, inputs, out_type, target):
    """qnn.conv2d strategy for Hexagon"""
    data = inputs[0]
    kernel = inputs[1]
    data_layout = attrs.data_layout
    kernel_layout = attrs.kernel_layout
    groups = attrs.groups
    strategy = _op.OpStrategy()
    if groups == 1:
        if data_layout == "NCHW" and kernel_layout == "OIHW":
            strategy.add_implementation(
                wrap_topi_qnn_conv2d(topi.hexagon.qnn_conv2d),
                wrap_topi_schedule(topi.hexagon.schedule_qnn_conv2d),
                name="qnn_conv2d.hexagon",
            )
    elif is_depthwise_conv2d(data.shape, data_layout, kernel.shape, kernel_layout, groups):
        if data_layout == "NCHW" and kernel_layout == "OIHW":
            strategy.add_implementation(
                wrap_topi_qnn_conv2d(topi.hexagon.qnn_depthwise_conv2d),
                wrap_topi_schedule(topi.hexagon.schedule_qnn_depthwise_conv2d),
                name="qnn_depthwise_conv2d.hexagon",
            )
    else:
        raise RuntimeError("Unsupported strategy for group qnn.conv2d")

    return strategy


# TODO: This is POC code. Change it on "hexagon" instead of "cpu"
@qnn_dense_strategy.register("cpu")
def qnn_dense_strategy_hexagon(attrs, inputs, out_type, target):
    """qnn.dense strategy for Hexagon"""
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_topi_qnn_dense(topi.hexagon.qnn_dense),
        wrap_topi_schedule(topi.hexagon.schedule_qnn_dense),
        name="qnn_dense.hexagon",
    )
    return strategy
