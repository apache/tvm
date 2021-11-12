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
"""Relay operators for binary elementwise operators for Arm(R) Ethos(TM)-U NPU"""
from typing import Optional
import tvm
from tvm.relay.op import _make
from tvm.topi.generic import schedule_injective
from tvm.relay.op.op import OpStrategy
from tvm.relay.op import strategy as _strategy

from ..te import binary_elementwise_compute


def _extract_ethosu_binary_elementwise_params(attrs, args):
    """Get the parameters necessary to construct a ethosu_binary_elementwise compute TE
    from a ethosu_binary_elementwise Relay call."""
    ifm = args[0]
    ifm2 = args[1]
    lut = args[2]
    operator_type = attrs.operator_type
    ifm_scale = attrs.ifm_scale
    ifm_zero_point = attrs.ifm_zero_point
    ifm2_scale = attrs.ifm2_scale
    ifm2_zero_point = attrs.ifm2_zero_point
    ofm_scale = attrs.ofm_scale
    ofm_zero_point = attrs.ofm_zero_point
    ifm_channels = attrs.ifm_channels
    ifm2_channels = attrs.ifm2_channels
    reversed_operands = attrs.reversed_operands
    activation = attrs.activation
    clip_min = attrs.clip_min
    clip_max = attrs.clip_max
    ifm_layout = attrs.ifm_layout
    ifm2_layout = attrs.ifm2_layout
    ofm_layout = attrs.ofm_layout
    ofm_dtype = attrs.ofm_dtype

    return (
        ifm,
        ifm2,
        lut,
        operator_type,
        ifm_scale,
        ifm_zero_point,
        ifm2_scale,
        ifm2_zero_point,
        ofm_scale,
        ofm_zero_point,
        ifm_channels,
        ifm2_channels,
        reversed_operands,
        activation,
        clip_min,
        clip_max,
        ifm_layout,
        ifm2_layout,
        ofm_layout,
        ofm_dtype,
    )


@tvm.ir.register_op_attr("contrib.ethosu.binary_elementwise", "FTVMCompute")
def create_ethosu_binary_elementwise_compute(attrs, args, out_type):
    """Create an ethosu_binary_elementwise compute op."""
    params = _extract_ethosu_binary_elementwise_params(attrs, args)
    op = binary_elementwise_compute(*params)
    return [op]


@tvm.ir.register_op_attr("contrib.ethosu.binary_elementwise", "FTVMStrategy")
def binary_elementwise_strategy_ethosu(attrs, inputs, out_type, target):
    strategy = OpStrategy()
    strategy.add_implementation(
        create_ethosu_binary_elementwise_compute,
        _strategy.wrap_topi_schedule(schedule_injective),
        name="ethosu_binary_elementwise",
    )
    return strategy


def ethosu_binary_elementwise(
    ifm: tvm.relay.Expr,
    ifm2: tvm.relay.Expr,
    lut: tvm.relay.Expr,
    operator_type: str,
    ifm_scale: float,
    ifm_zero_point: int,
    ifm2_scale: float,
    ifm2_zero_point: int,
    ofm_scale: float,
    ofm_zero_point: int,
    ifm_channels: int,
    ifm2_channels: int,
    reversed_operands: bool,
    ofm_dtype: str,
    activation: Optional[str] = "NONE",
    clip_min: Optional[int] = 0,
    clip_max: Optional[int] = 0,
    ifm_layout: Optional[str] = "NHWC",
    ifm2_layout: Optional[str] = "NHWC",
    ofm_layout: Optional[str] = "NHWC",
) -> tvm.relay.Call:
    """This is a quantized binary elementwise operation as supported by
    the NPU. It accepts either NHWC or NHCWB16 format
    for the input data.

    Parameters
    ----------
    ifm : tvm.relay.Expr
        The Input Feature Map tensor (IFM).
    ifm2 : tvm.relay.Expr
        The Input Feature Map tensor 2 (IFM2).
    lut : tvm.relay.Expr
        The look-up table of values to use if activation = "LUT".
    operator_type: str
        The type of the binary elementwise operator.
            "ADD"
            "SUB"
            "MUL"
            "MIN"
            "MAX"
            "SHR"
            "SHL"
    ifm_scale : float
        The quantization scale for the Input Feature Map tensor.
    ifm_zero_point : int
        The quantization zero point for the Input Feature Map tensor.
    ifm2_scale : float
        The quantization scale for the Input Feature Map tensor 2.
    ifm2_zero_point : int
        The quantization zero point for the Input Feature Map tensor 2.
    ofm_scale : float
        The quantization scale for the Output Feature Map tensor.
    ofm_zero_point : int
       The quantization zero point for the Output Feature Map tensor.
    ifm_channels : int
        The number of the Input Feature Map channels.
    ifm2_channels : int
        The number of the Input Feature Map 2 channels.
    reversed_operands : bool
        True if IFM2 is the first operand and IFM is the second operand.
    ofm_dtype: str
        The Output Feature Map tensor type.
        MUL, ADD, SUB {IFM}->{OFM}:
          {uint8, int8 int32} -> {uint8, int8, int32}, any pairing
        MAX, MIN:
          IFM and OFM must be of the same type, one of:
          {int8, uint8}
        SHR {IFM}->{OFM}:
          {int32}->{int8, uint8, int32}, any pairing"
        SHL:
          {int32}->{int32} only
    activation : str, optional
        The activation function to use.
            "NONE" - no activation function.
            "CLIP" - clip the output between clip_min and clip_max.
            "TANH" - tanh activation function.
            "SIGMOID" - sigmoid activation function.
            "LUT" - use a look-up table to perform the activation function.
        Available activations for activation type:
            {int8, uint8}: "NONE", "CLIP", "TANH", "SIGMOID", "LUT"
            {int32}: "NONE"
    clip_min : int, optional
        The minimum clipping value if activation = "CLIP".
    clip_max : int, optional
        The maximum clipping value if activation = "CLIP".
    ifm_layout : str, optional
        The layout of the Input Feature Map tensor. Can be "NHWC" or "NHCWB16".
    ifm2_layout : str, optional
        The layout of the Input Feature Map tensor 2. Can be "NHWC" or "NHCWB16".
    ofm_layout : str, optional
        The layout of the Output Feature Map tensor. Can be "NHWC" or "NHCWB16".

    Returns
    -------
    out : tvm.relay.Call
        A call to the ethosu_binary_elementwise op.
    """
    return _make.ethosu_binary_elementwise(
        ifm,
        ifm2,
        lut,
        operator_type,
        ifm_scale,
        ifm_zero_point,
        ifm2_scale,
        ifm2_zero_point,
        ofm_scale,
        ofm_zero_point,
        ifm_channels,
        ifm2_channels,
        reversed_operands,
        activation,
        clip_min,
        clip_max,
        ifm_layout,
        ifm2_layout,
        ofm_layout,
        ofm_dtype,
    )
