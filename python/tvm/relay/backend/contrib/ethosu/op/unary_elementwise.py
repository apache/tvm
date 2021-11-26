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
"""Relay operator for unary elementwise operations for Arm(R) Ethos(TM)-U NPU"""
from typing import Optional
import tvm
from tvm.relay.op import _make
from tvm.topi.generic import schedule_injective
from tvm.relay.op.op import OpStrategy
from tvm.relay.op import strategy as _strategy

from ..te import unary_elementwise_compute


def _extract_ethosu_unary_elementwise_params(attrs, args):
    """Get the parameters necessary to construct a ethosu_unary_elementwise compute TE
    from a ethosu_unary_elementwise Relay call."""
    ifm = args[0]
    lut = args[1]
    operator_type = attrs.operator_type
    ifm_scale = attrs.ifm_scale
    ifm_zero_point = attrs.ifm_zero_point
    ofm_scale = attrs.ofm_scale
    ofm_zero_point = attrs.ofm_zero_point
    ofm_channels = attrs.ofm_channels
    activation = attrs.activation
    clip_min = attrs.clip_min
    clip_max = attrs.clip_max
    rounding_mode = attrs.rounding_mode
    ifm_layout = attrs.ifm_layout
    ofm_layout = attrs.ofm_layout

    return (
        ifm,
        lut,
        operator_type,
        ifm_scale,
        ifm_zero_point,
        ofm_scale,
        ofm_zero_point,
        ofm_channels,
        activation,
        clip_min,
        clip_max,
        rounding_mode,
        ifm_layout,
        ofm_layout,
    )


@tvm.ir.register_op_attr("contrib.ethosu.unary_elementwise", "FTVMCompute")
def create_ethosu_unary_elementwise_compute(attrs, args, out_type):
    """Create an ethosu_unary_elementwise compute op."""
    params = _extract_ethosu_unary_elementwise_params(attrs, args)
    op = unary_elementwise_compute(*params)
    return [op]


@tvm.ir.register_op_attr("contrib.ethosu.unary_elementwise", "FTVMStrategy")
def unary_elementwise_strategy_ethosu(attrs, inputs, out_type, target):
    strategy = OpStrategy()
    strategy.add_implementation(
        create_ethosu_unary_elementwise_compute,
        _strategy.wrap_topi_schedule(schedule_injective),
        name="ethosu_unary_elementwise",
    )
    return strategy


def ethosu_unary_elementwise(
    ifm: tvm.relay.Expr,
    lut: tvm.relay.Expr,
    operator_type: str,
    ifm_scale: float,
    ifm_zero_point: int,
    ofm_scale: float,
    ofm_zero_point: int,
    ofm_channels: int,
    activation: Optional[str] = "NONE",
    clip_min: Optional[int] = 0,
    clip_max: Optional[int] = 0,
    rounding_mode: Optional[str] = "TFL",
    ifm_layout: Optional[str] = "NHWC",
    ofm_layout: Optional[str] = "NHWC",
) -> tvm.relay.Call:
    """This is a quantized unary elementwise operation as supported by the
    NPU. It accepts either NHWC or NHCWB16 format for the input data.

    Parameters
    ----------
    ifm : tvm.relay.Expr
        The Input Feature Map tensor (IFM).
    lut : tvm.relay.Expr
        The look-up table values to use if activation = "LUT".
    operator_type: str
        The type of the unary elementwise operator.
            "ABS"
            "CLZ"
    ifm_scale : float
        The quantization scale for the Input Feature Map tensor.
    ifm_zero_point : int
        The quantization zero point for the Input Feature Map tensor.
    ofm_scale : float
        The quantization scale for the Output Feature Map tensor.
    ofm_zero_point : int
       The quantization zero point for the Output Feature Map tensor.
    ofm_channels : int
        The number of OFM channels.
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
    rounding_mode : str, optional
        The rounding mode to apply to the Output Feature Map tensor.
            "TFL" - Tensorflow Lite rounding scheme.
            "TRUNCATE" - Truncate towards zero.
            "NATURAL" - Round to nearest value, with x.5 rounded up towards +infinity.
    ifm_layout : str, optional
        The layout of the Input Feature Map tensor. Can be "NHWC" or "NHCWB16".
    ofm_layout : str, optional
        The layout of the Output Feature Map tensor. Can be "NHWC" or "NHCWB16".

    Returns
    -------
    out : tvm.relay.Call
        A call to the ethosu_unary_elementwise op.
    """
    return _make.ethosu_unary_elementwise(
        ifm,
        lut,
        operator_type,
        ifm_scale,
        ifm_zero_point,
        ofm_scale,
        ofm_zero_point,
        ofm_channels,
        activation,
        clip_min,
        clip_max,
        rounding_mode,
        ifm_layout,
        ofm_layout,
    )
