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

import re

from tvm import topi
from .generic import *
from ... import op as _op
from ...op.strategy.generic import is_depthwise_conv2d


NCHWC_MATCHER = re.compile("^NCHW[0-9]+c$")
OIHWIOI_MATCHER = re.compile("^OIHW[0-9]+i[0-9]+o[0-9]+i$")


@qnn_quantize_strategy.register("hexagon")
def qnn_quantize_strategy_hexagon(attrs, inputs, out_type, target):
    """qnn.quantize strategy for Hexagon"""
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_quantize(topi.hexagon.qnn_quantize),
        wrap_topi_schedule(topi.hexagon.schedule_qnn_quantize),
        name="qnn_quantize.hexagon",
    )
    return strategy


@qnn_dequantize_strategy.register("hexagon")
def qnn_dequantize_strategy_hexagon(attrs, inputs, out_type, target):
    """qnn.dequantize strategy for Hexagon"""
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_dequantize(topi.hexagon.qnn_dequantize),
        wrap_topi_schedule(topi.hexagon.schedule_qnn_dequantize),
        name="qnn_dequantize.hexagon",
    )
    return strategy


@qnn_requantize_strategy.register("hexagon")
def qnn_requantize_strategy_hexagon(attrs, inputs, out_type, target):
    """qnn.requantize strategy for Hexagon"""
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_quantize(topi.hexagon.qnn_requantize),
        wrap_topi_schedule(topi.hexagon.schedule_qnn_requantize),
        name="qnn_requantize.hexagon",
    )
    return strategy


@qnn_add_strategy.register("hexagon")
def qnn_add_strategy_hexagon(attrs, inputs, out_type, target):
    """qnn.add strategy for Hexagon"""
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_topi_compute(topi.hexagon.qnn_add),
        wrap_topi_schedule(topi.hexagon.schedule_qnn_add),
        name="qnn_add.hexagon",
    )
    return strategy


@qnn_subtract_strategy.register("hexagon")
def qnn_subtract_strategy_hexagon(attrs, inputs, out_type, target):
    """qnn.subtract strategy for Hexagon"""
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_topi_compute(topi.hexagon.qnn_subtract),
        wrap_topi_schedule(topi.hexagon.schedule_qnn_subtract),
        name="qnn_subtract.hexagon",
    )
    return strategy


@qnn_mul_strategy.register("hexagon")
def qnn_mul_strategy_hexagon(attrs, inputs, out_type, target):
    """qnn.mul strategy for Hexagon"""
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_topi_compute(topi.hexagon.qnn_mul),
        wrap_topi_schedule(topi.hexagon.schedule_qnn_mul),
        name="qnn_mul.hexagon",
    )
    return strategy


@qnn_tanh_strategy.register("hexagon")
def qnn_tanh_strategy_hexagon(attrs, inputs, out_type, target):
    """qnn.tanh strategy for Hexagon"""
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_topi_compute(topi.hexagon.qnn_tanh),
        wrap_topi_schedule(topi.hexagon.schedule_qnn_tanh),
        name="qnn_tanh.hexagon",
    )
    return strategy


@qnn_concatenate_strategy.register("hexagon")
def qnn_concatenate_strategy_hexagon(attrs, inputs, out_type, target):
    """qnn.concatenate strategy for Hexagon"""
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_topi_concatenate(topi.hexagon.qnn_concatenate),
        wrap_topi_schedule(topi.hexagon.schedule_qnn_concatenate),
        name="qnn_concatenate.hexagon",
    )
    return strategy


@qnn_conv2d_strategy.register("hexagon")
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
        elif NCHWC_MATCHER.match(data_layout) and OIHWIOI_MATCHER.match(kernel_layout):
            if data.dtype == "uint8" and kernel.dtype == "int8":
                strategy.add_implementation(
                    wrap_topi_qnn_conv2d(topi.hexagon.qnn_conv2d_NCHWc_int8),
                    wrap_topi_schedule(topi.hexagon.schedule_qnn_conv2d_NCHWc_int8),
                    name="qnn_conv2d_NCHWc_int8.hexagon",
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


@qnn_dense_strategy.register("hexagon")
def qnn_dense_strategy_hexagon(attrs, inputs, out_type, target):
    """qnn.dense strategy for Hexagon"""
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_topi_qnn_dense(topi.hexagon.qnn_dense),
        wrap_topi_schedule(topi.hexagon.schedule_qnn_dense),
        name="qnn_dense.hexagon",
    )
    return strategy


@qnn_dense_pack_strategy.register("hexagon")
def qnn_dense_pack_strategy_hexagon(attrs, inputs, out_type, target):
    """qnn.contrib_dense_pack strategy for Hexagon"""
    strategy = _op.OpStrategy()
    if (
        "uint8" in inputs[0].dtype
        and "int8" in inputs[1].dtype
        and attrs["weight_layout"] == "NC32n4c"
    ):
        # uint8 + uint8|int8 case
        strategy.add_implementation(
            wrap_topi_qnn_dense(topi.hexagon.qnn_dense_pack_vrmpy),
            wrap_topi_schedule(topi.hexagon.schedule_qnn_dense_pack_vrmpy),
            name="qnn_dense_pack_vrmpy.hexagon",
        )
    return strategy


@qnn_batch_matmul_strategy.register("hexagon")
def qnn_batch_matmul_strategy_hexagon(attrs, inputs, out_type, target):
    """qnn.batch_matmul strategy for Hexagon"""
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_topi_qnn_batch_matmul(topi.hexagon.qnn_batch_matmul),
        wrap_topi_schedule(topi.hexagon.schedule_qnn_batch_matmul),
        name="qnn_batch_matmul.hexagon",
    )
    return strategy


@qnn_avg_pool2d_strategy.register(["hexagon"])
def qnn_avg_pool2d_strategy_hexagon(attrs, inputs, out_type, target):
    """qnn.avg_pool2d strategy for Hexagon"""
    data_layout = attrs.layout
    if data_layout == "NHWC":
        strategy = _op.OpStrategy()
        strategy.add_implementation(
            wrap_compute_qnn_avg_pool2d(topi.hexagon.qnn.qnn_avg_pool2d_wrapper_compute_NHWC),
            wrap_topi_schedule(topi.hexagon.qnn.schedule_qnn_avg_pool2d),
            name="qnn_avg_pool2d.hexagon",
        )
        return strategy
    elif data_layout == "NCHW":
        strategy = _op.OpStrategy()
        strategy.add_implementation(
            wrap_compute_qnn_avg_pool2d(topi.hexagon.qnn.qnn_avg_pool2d_wrapper_compute_NCHW),
            wrap_topi_schedule(topi.hexagon.qnn.schedule_qnn_avg_pool2d),
            name="qnn_avg_pool2d.hexagon",
        )
        return strategy
    else:
        raise RuntimeError("Unsupported strategy for qnn.avg_pool2d")
