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
# pylint: disable=invalid-name,unused-argument
"""Tensor Expressions for binary_elementwise"""
import operator
from tvm import te
from .dma import dma_ofm_compute, dma_ifm_compute


def binary_elementwise_compute(
    ifm: te.Tensor,
    ifm2: te.Tensor,
    lut: te.Tensor,
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
    activation: str,
    clip_min: int,
    clip_max: int,
    ifm_layout: str,
    ifm2_layout: str,
    ofm_layout: str,
    ofm_dtype: str,
) -> te.Tensor:
    """A compute operator representing the capabilities of binary_elementwise for the NPU.

    Parameters
    ----------
    ifm : te.Tensor
        The Input Feature Map tensor (IFM).
    ifm2 : te.Tensor
        The Input Feature Map tensor 2 (IFM2).
    lut : te.Tensor
        The look-up table values to use if activation = "LUT".
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
        The quantization zero point for the Input Feature Map tensor 1.
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
    activation : str
        The activation function to use.
            "NONE" - no activation function.
            "CLIP" - clip the output between clip_min and clip_max.
            "TANH" - tanh activation function.
            "SIGMOID" - sigmoid activation function.
            "LUT" - use a look-up table to perform the activation function.
        Available activations for activation type:
            {int8, uint8}: "NONE", "CLIP", "TANH", "SIGMOID", "LUT"
            {int32}: "NONE"
    clip_min : int
        The minimum clipping value if activation = "CLIP".
    clip_max : int
        The maximum clipping value if activation = "CLIP".
    ifm_layout : str, optional
        The layout of the Input Feature Map tensor. Can be "NHWC" or "NHCWB16".
    ifm2_layout : str, optional
        The layout of the Input Feature Map tensor 2. Can be "NHWC" or "NHCWB16".
    ofm_layout : str, optional
        The layout of the Output Feature Map tensor. Can be "NHWC" or "NHCWB16".
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

    Returns
    -------
    te.Tensor
        The Output Feature Map tensor.
    """
    # Compute operation for the IFM DMA pipeline
    dmaed_ifm = dma_ifm_compute(
        ifm, ifm_layout, ifm_zero_point, ifm_scale, ifm_channels, (0, 0, 0, 0)
    )
    dmaed_ifm2 = dma_ifm_compute(
        ifm2, ifm2_layout, ifm2_zero_point, ifm2_scale, ifm2_channels, (0, 0, 0, 0)
    )

    # Binary elementwise compute operation
    ofm_height = dmaed_ifm.shape[1]
    ofm_width = dmaed_ifm.shape[2]

    binary_elementwise_attrs = {
        "op": "ethosu_binary_elementwise",
        "operator_type": operator_type,
        "reversed_operands": reversed_operands,
        "activation": activation,
        "clip_min": clip_min,
        "clip_max": clip_max,
    }

    operators = {
        "ADD": operator.add,
        "SUB": operator.sub,
        "MUL": operator.mul,
        "MIN": te.min,
        "MAX": te.max,
        "SHR": operator.add,
        "SHL": operator.add,
    }
    broadcast = [value == 1 for value in dmaed_ifm2.shape]

    if reversed_operands:
        binary_elementwise = te.compute(
            (1, ofm_height, ofm_width, ifm_channels),
            lambda nn, hh, ww, cc: operators[operator_type](
                dmaed_ifm2(
                    0 if broadcast[0] else nn,
                    0 if broadcast[1] else hh,
                    0 if broadcast[2] else ww,
                    0 if broadcast[3] else cc,
                ).astype(ifm.dtype),
                dmaed_ifm(nn, hh, ww, cc).astype(ifm.dtype),
            ).astype(ofm_dtype),
            name="ethosu_binary_elementwise",
            attrs=binary_elementwise_attrs,
        )
    else:
        binary_elementwise = te.compute(
            (1, ofm_height, ofm_width, ifm_channels),
            lambda nn, hh, ww, cc: operators[operator_type](
                dmaed_ifm(nn, hh, ww, cc).astype(ifm.dtype),
                dmaed_ifm2(
                    0 if broadcast[0] else nn,
                    0 if broadcast[1] else hh,
                    0 if broadcast[2] else ww,
                    0 if broadcast[3] else cc,
                ).astype(ifm.dtype),
            ).astype(ofm_dtype),
            name="ethosu_binary_elementwise",
            attrs=binary_elementwise_attrs,
        )

    # Compute operation for the OFM DMA pipeline
    return dma_ofm_compute(binary_elementwise, ofm_layout, ofm_zero_point, ofm_scale, ifm_channels)
