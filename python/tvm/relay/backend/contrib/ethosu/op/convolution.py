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
"""Relay operators for convolutions for Arm(R) Ethos(TM)-U NPU"""
from typing import Tuple

import tvm  # type: ignore
from tvm.relay.op import _make  # type: ignore
from tvm.topi.generic import schedule_injective  # type: ignore
from tvm.relay.op.op import OpStrategy  # type: ignore
from tvm.relay.op import strategy as _strategy

from ..te import conv2d_compute


def _extract_ethosu_conv2d_params(attrs, args):
    """Get the parameters necessary to construct a compute TE
    from a ethosu_conv2d Relay call."""
    ifm = args[0]
    weight = args[1]
    scale_bias = args[2]
    lut = args[3]
    ifm_scale = attrs.ifm_scale
    ifm_zero_point = attrs.ifm_zero_point
    weight_zero_point = attrs.weight_zero_point
    ofm_scale = attrs.ofm_scale
    ofm_zero_point = attrs.ofm_zero_point
    strides = attrs.strides
    padding = attrs.padding
    dilation = attrs.dilation
    activation = attrs.activation
    clip_min = attrs.clip_min
    clip_max = attrs.clip_max
    rounding_mode = attrs.rounding_mode
    upscale = attrs.upscale
    ifm_layout = attrs.ifm_layout
    ofm_layout = attrs.ofm_layout

    return (
        ifm,
        weight,
        scale_bias,
        lut,
        ifm_scale,
        ifm_zero_point,
        weight_zero_point,
        ofm_scale,
        ofm_zero_point,
        strides,
        padding,
        dilation,
        activation,
        clip_min,
        clip_max,
        rounding_mode,
        upscale,
        ifm_layout,
        ofm_layout,
    )


@tvm.ir.register_op_attr("contrib.ethosu.conv2d", "FTVMCompute")
def create_ethosu_conv2d_compute(attrs, args, out_type):
    """Create an ethosu_conv2d compute op."""
    params = _extract_ethosu_conv2d_params(attrs, args)
    op = conv2d_compute(*params)
    return [op]


@tvm.ir.register_op_attr("contrib.ethosu.conv2d", "FTVMStrategy")
def conv2d_strategy_ethosu(attrs, inputs, out_type, target):
    strategy = OpStrategy()
    strategy.add_implementation(
        create_ethosu_conv2d_compute,
        _strategy.wrap_topi_schedule(schedule_injective),
        name="ethosu_conv2d",
    )
    return strategy


def ethosu_conv2d(
    ifm: tvm.relay.Expr,
    weight: tvm.relay.Expr,
    scale_bias: tvm.relay.Expr,
    lut: tvm.relay.Expr,
    ifm_scale: float,
    ifm_zero_point: int,
    weight_zero_point: int,
    ofm_scale: float,
    ofm_zero_point: int,
    kernel_shape: Tuple[int, int],
    ofm_channels: int,
    strides: Tuple[int, int] = (1, 1),
    padding: Tuple[int, int, int, int] = (0, 0, 0, 0),
    dilation: Tuple[int, int] = (1, 1),
    activation: str = "NONE",
    clip_min: int = 0,
    clip_max: int = 0,
    rounding_mode: str = "TFL",
    upscale: str = "NONE",
    ifm_layout: str = "NHWC",
    ofm_layout: str = "NHWC",
) -> tvm.relay.Call:
    """This is a quantized 2D convolution operation as supported by
    the NPU. It accepts either NHWC or NHCWB16 format
    for the input data and OHWI format for the kernel weights.

    Reference: https://developer.arm.com/documentation/102420/0200/

    Note that the per-channel weight scale and bias tensor must be
    packed together into a combined tensor of uint80s. This is represented
    in TVM by a (channels, 10) tensor of type uint8. For more detail,
    refer to the Technical Reference Manual linked above.

    Parameters
    ----------
    ifm : tvm.relay.Expr
        The Input Feature Map tensor (IFM).
    weight : tvm.relay.Expr
        The weight tensor.
    scale_bias : tvm.relay.Expr
        The packed per-channel weight scale and bias tensor.
    lut : tvm.relay.Expr
        The look-up table of values to use if activation = "LUT".
    ifm_scale : float
        The quantization scale for the Input Feature Map tensor.
    ifm_zero_point : int
        The quantization zero point for the Input Feature Map tensor.
    weight_zero_point : int
        The quantization zero point for the weight tensor.
    ofm_scale : int
        The quantization scale for the Output Feature Map tensor.
    ofm_zero_point : int
        The quantization zero point for the Output Feature Map tensor.
    kernel_shape : tuple of int
        The 2 dimensional kernel shape as (kernel_height, kernel_width).
    ofm_channels : int
        The number of the Output Feature Map channels.
    strides : tuple of int, optional
        The 2 dimensional strides as (stride_height, stride_width).
    padding : tuple of int, optional
        The 4 dimensional padding as (pad_top, pad_left, pad_bottom, pad_right).
    dilation : tuple of int, optional
        The 2 dimensional dilation as (dilation_height, dilation_width).
    activation : str, optional
        The activation function to use.
            "NONE" - no activation function.
            "CLIP" - clip the output between clip_min and clip_max.
            "TANH" - tanh activation function.
            "SIGMOID" - sigmoid activation function.
            "LUT" - use a look-up table to perform the activation function.
    clip_min : int, optional
        The minimum clipping value if activation = "CLIP"
    clip_max : int, optional,
        The maximum clipping value if activation = "CLIP"
    rounding_mode : str, optional
        The rounding mode to apply to the Output Feature Map tensor.
            "TFL" - Tensorflow Lite rounding scheme.
            "TRUNCATE" - Truncate towards zero.
            "NATURAL" - Round to nearest value, with x.5 rounded up towards +infinity.
    upscale : str, optional
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
    tvm.relay.Call
        A call to the ethosu_conv2d op.

    """
    return _make.ethosu_conv2d(
        ifm,
        weight,
        scale_bias,
        lut,
        ifm_scale,
        ifm_zero_point,
        weight_zero_point,
        ofm_scale,
        ofm_zero_point,
        kernel_shape,
        ofm_channels,
        strides,
        padding,
        dilation,
        activation,
        clip_min,
        clip_max,
        rounding_mode,
        upscale,
        ifm_layout,
        ofm_layout,
    )
