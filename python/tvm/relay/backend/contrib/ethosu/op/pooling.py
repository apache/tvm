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
# pylint: disable=unused-argument
"""Relay operators for pooling for Arm(R) Ethos(TM)-U NPU"""
from typing import Tuple

import tvm
from tvm.relay.op import _make
from tvm.topi.generic import schedule_injective
from tvm.relay.op.op import OpStrategy
from tvm.relay.op import strategy as _strategy

from ..te import pooling_compute


def _extract_ethosu_pooling_params(attrs, args):
    """Get the parameters necessary to construct a ethosu_pooling compute TE
    from a ethosu_pooling Relay call."""
    ifm = args[0]
    lut = args[1]
    pooling_type = attrs.pooling_type
    ifm_scale = attrs.ifm_scale
    ifm_zero_point = attrs.ifm_zero_point
    ofm_scale = attrs.ofm_scale
    ofm_zero_point = attrs.ofm_zero_point
    pool_shape = attrs.pool_shape
    ofm_channels = attrs.ofm_channels
    strides = attrs.strides
    padding = attrs.padding
    activation = attrs.activation
    clip_min = attrs.clip_min
    clip_max = attrs.clip_max
    upscale = attrs.upscale
    ifm_layout = attrs.ifm_layout
    ofm_layout = attrs.ofm_layout

    return (
        ifm,
        lut,
        pooling_type,
        ifm_scale,
        ifm_zero_point,
        ofm_scale,
        ofm_zero_point,
        pool_shape,
        ofm_channels,
        strides,
        padding,
        activation,
        clip_min,
        clip_max,
        upscale,
        ifm_layout,
        ofm_layout,
    )


@tvm.ir.register_op_attr("contrib.ethosu.pooling", "FTVMCompute")
def create_ethosu_pooling_compute(attrs, args, out_type):
    """Create an ethosu_pooling compute op."""
    params = _extract_ethosu_pooling_params(attrs, args)
    op = pooling_compute(*params)
    return [op]


@tvm.ir.register_op_attr("contrib.ethosu.pooling", "FTVMStrategy")
def pooling_strategy_ethosu(attrs, inputs, out_type, target):
    strategy = OpStrategy()
    strategy.add_implementation(
        create_ethosu_pooling_compute,
        _strategy.wrap_topi_schedule(schedule_injective),
        name="ethosu_pooling",
    )
    return strategy


def ethosu_pooling(
    ifm: tvm.relay.Expr,
    lut: tvm.relay.Expr,
    pooling_type: str,
    ifm_scale: float,
    ifm_zero_point: int,
    ofm_scale: float,
    ofm_zero_point: int,
    pool_shape: Tuple[int, int],
    ofm_channels: int,
    strides: Tuple[int, int] = (1, 1),
    padding: Tuple[int, int, int, int] = (0, 0, 0, 0),
    activation: str = "NONE",
    clip_min: int = 0,
    clip_max: int = 0,
    upscale: str = "NONE",
    ifm_layout: str = "NHWC",
    ofm_layout: str = "NHWC",
) -> tvm.relay.Call:
    """This is a quantized 2D pooling operation as supported by
    the NPU. It accepts either NHWC or NHCWB16 format
    for the input data.

    Parameters
    ----------
    ifm : tvm.relay.Expr
        The Input Feature Map tensor (IFM).
    lut : tvm.relay.Expr
         The look-up table of values to use if activation = "LUT".
    pooling_type: str
        The type of the pooling. "AVG" - average pool,   "MAX" - max pool.
    ifm_scale : float
        The quantization scale for the Input Feature Map tensor.
    ifm_zero_point : int
        The quantization zero point for the Input Feature Map tensor.
    ofm_scale : float
        The quantization scale for the Output Feature Map tensor.
    ofm_zero_point : int
       The quantization zero point for the Output Feature Map tensor.
    pool_shape : tuple of int
        The 2 dimensional pool shape as (pool_shape_height, pool_shape_width).
    ofm_channels : int
        The number of the Output Feature Map channels
    strides : tuple of int, optional
        The 2 dimensional strides as (stride_height, stride_width).
    padding : tuple of int, optional
        The 4 dimensional padding as (pad_top, pad_left, pad_bottom, pad_right).
    activation : str, optional
        The activation function to use.
            "NONE" - no activation function.
            "CLIP" - clip the output between clip_min and clip_max.
            "TANH" - tanh activation function.
            "SIGMOID" - sigmoid activation function.
            "LUT" - use a look-up table to perform the activation function.
    clip_min : int, optional
        The minimum clipping value if activation = "CLIP".
    clip_max : int, optional
        The maximum clipping value if activation = "CLIP".
    upscale: str, optional
        The 2x2 upscaling mode to apply to the Input Feature Map tensor.
            "NONE" - no upscaling.
            "NEAREST" - upscale using nearest neighbour.
            "ZEROS" - upscale using zeros.
    ifm_layout : str, optional
        The layout of the Input Feature Map tensor. Can be "NHWC" or "NHCWB16".
    ofm_layout : str, optional
        The layout of the Output Feature Map tensor. Can be "NHWC" or "NHCWB16".

    Returns
    -------
    out : tvm.relay.Call
        A call to the ethosu_pooling op.
    """
    return _make.ethosu_pooling(
        ifm,
        lut,
        pooling_type,
        ifm_scale,
        ifm_zero_point,
        ofm_scale,
        ofm_zero_point,
        pool_shape,
        ofm_channels,
        strides,
        padding,
        activation,
        clip_min,
        clip_max,
        upscale,
        ifm_layout,
        ofm_layout,
    )
