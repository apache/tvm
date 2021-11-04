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
"""Tensor Expressions for depthwise convolutions"""
from typing import Tuple, Union, List

from tvm import te
from .dma import dma_ofm_compute, dma_ifm_compute


def depthwise_conv2d_compute(
    ifm: te.Tensor,
    weight: te.Tensor,
    scale_bias: te.Tensor,
    lut: te.Tensor,
    ifm_scale: float,
    ifm_zero_point: int,
    weight_zero_point: int,
    ofm_scale: float,
    ofm_zero_point: int,
    strides: Tuple[int, int],
    padding: Tuple[int, int, int, int],
    dilation: Union[Tuple[int, int], List[int]],
    activation: str,
    clip_min: int,
    clip_max: int,
    upscale: str,
    ifm_layout: str,
    ofm_layout: str,
) -> te.Tensor:
    """A compute operator representing the capabilities of 2D convolution for the NPU.

    Parameters
    ----------
    ifm : te.Tensor
        The Input Feature Map tensor (IFM).
    weight : te.Tensor
        The weight tensor.
    scale_bias : te.Tensor
        The packed per-channel weight scale and bias tensor.
    lut : te.Tensor
        The look-up table of values to use if activation = "LUT".
    ifm_scale : float
        The quantization scale for the Input Feature Map tensor.
    ifm_zero_point : int
        The quantization zero point for the Input Feature Map tensor.
    weight_zero_point : int
        The quantization zero point for the weight tensor.
    ofm_scale : float
        The quantization scale for the Output Feature Map tensor.
    ofm_zero_point : int
        The quantization zero point for the Output Feature Map tensor.
    strides : tuple
        The 2 dimensional strides as (stride_height, stride_width).
    padding : tuple
        The 4 dimensional padding as (pad_top, pad_left, pad_bottom, pad_right).
    dilation : Union[int, tuple, list]
        The 2 dimensional dilation as (dilation_height, dilation_width).
    activation : str
        The activation function to use.
            "NONE" - no activation function.
            "CLIP" - clip the output between clip_min and clip_max.
            "TANH" - tanh activation function.
            "SIGMOID" - sigmoid activation function.
            "LUT" - use a look-up table to perform the activation function.
    clip_min : int
        The minimum clipping value if activation = "CLIP".
    clip_max : int
        The maximum clipping value if activation = "CLIP".
    upscale : str
        The 2x2 upscaling mode to apply to the Input Feature Map tensor.
            "NONE" - no upscaling.
            "NEAREST" - upscale using nearest neighbour.
            "ZEROS" - upscale using zeros.
    ifm_layout : str
        The layout of the Input Feature Map tensor. Can be "NHWC" or "NHCWB16".
    ofm_layout : str
        The layout of the Output Feature Map tensor. Can be "NHWC" or "NHCWB16".

    Returns
    -------
    te.Tensor
        The OFM tensor.

    """
    assert ifm.shape[0] == 1, f"Only batch size 1 is supported"
    assert ifm_layout in {"NHWC", "NHCWB16"}
    assert ofm_layout in {"NHWC", "NHCWB16"}

    stride_h, stride_w = strides
    dilation_h, dilation_w = dilation
    channels, kernel_h, kernel_w, _ = weight.shape

    # Compute operation for the IFM DMA pipeline
    dmaed_ifm = dma_ifm_compute(ifm, ifm_layout, ifm_zero_point, ifm_scale, channels, padding)

    # 2D Depthwise Convolution compute operation
    dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
    dilated_kernel_w = (kernel_w - 1) * dilation_w + 1
    ofm_height = (dmaed_ifm.shape[1] - dilated_kernel_h) // stride_h + 1
    ofm_width = (dmaed_ifm.shape[2] - dilated_kernel_w) // stride_w + 1
    rh = te.reduce_axis((0, kernel_h), name="ry")
    rw = te.reduce_axis((0, kernel_w), name="rx")

    depthwise_conv2d_attrs = {
        "op": "ethosu_depthwise_conv2d",
        "weight_zero_point": weight_zero_point,
        "activation": activation,
        "upscale": upscale,
        "clip_min": clip_min,
        "clip_max": clip_max,
        "stride_h": stride_h,
        "stride_w": stride_w,
        "dilation_h": dilation_h,
        "dilation_w": dilation_w,
    }

    depthwise = te.compute(
        (1, ofm_height, ofm_width, channels),
        lambda nn, hh, ww, cc: te.sum(
            dmaed_ifm(
                nn, hh * stride_h + rh * dilation_h, ww * stride_w + rw * dilation_w, cc
            ).astype(ifm.dtype)
            * weight[cc, rh, rw, 0].astype(ifm.dtype)
            # This is a trick to load 10 elements of the scale_bias at once, not accurate maths
            + (scale_bias[cc, 0] * scale_bias[cc, 9]).astype(ifm.dtype),
            axis=[rh, rw],
        ),
        name="ethosu_depthwise_conv2d",
        attrs=depthwise_conv2d_attrs,
    )

    # Compute operation for the OFM DMA pipeline
    return dma_ofm_compute(depthwise, ofm_layout, ofm_zero_point, ofm_scale, channels)
