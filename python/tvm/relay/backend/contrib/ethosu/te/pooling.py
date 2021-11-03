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
"""Tensor Expressions for poolings"""
from typing import Tuple

from tvm import te
from .dma import dma_ofm_compute, dma_ifm_compute


def pooling_compute(
    ifm: te.Tensor,
    lut: te.Tensor,
    pooling_type: str,
    ifm_scale: float,
    ifm_zero_point: int,
    ofm_scale: float,
    ofm_zero_point: int,
    pool_shape: Tuple[int, int],
    ofm_channels: int,
    strides: Tuple[int, int],
    padding: Tuple[int, int, int, int],
    activation: str,
    clip_min: int,
    clip_max: int,
    upscale: str,
    ifm_layout: str,
    ofm_layout: str,
) -> te.Tensor:
    """A compute operator representing the capabilities of pooling for the NPU.

    Parameters
    ----------
    ifm : te.Tensor
        The Input Feature Map tensor (IFM).
    lut : te.Tensor
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
    pool_shape : Tuple[int, int]
        The 2 dimensional pool shape as (pool_shape_height, pool_shape_width).
    ofm_channels : int
        The number of the Output Feature Map channels
    strides : Tuple[int, int]
        The 2 dimensional strides as (stride_height, stride_width).
    padding : Tuple[int, int, int, int]
        The 4 dimensional padding as (pad_top, pad_left, pad_bottom, pad_right).
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
    stride_h, stride_w = strides
    pool_shape_h, pool_shape_w = pool_shape

    # Compute operation for the IFM DMA pipeline
    dmaed_ifm = dma_ifm_compute(ifm, ifm_layout, ifm_zero_point, ifm_scale, ofm_channels, padding)

    # Pooling compute operation
    ofm_height = (dmaed_ifm.shape[1] - pool_shape_h) // stride_h + 1
    ofm_width = (dmaed_ifm.shape[2] - pool_shape_w) // stride_w + 1
    rh = te.reduce_axis((0, pool_shape_h), name="ry")
    rw = te.reduce_axis((0, pool_shape_w), name="rx")

    pooling_attrs = {
        "op": "ethosu_pooling",
        "pooling_type": pooling_type,
        "stride_h": stride_h,
        "stride_w": stride_w,
        "activation": activation,
        "clip_min": clip_min,
        "clip_max": clip_max,
        "upscale": upscale,
    }

    pooling = te.compute(
        (1, ofm_height, ofm_width, ofm_channels),
        lambda nn, hh, ww, cc: te.max(
            dmaed_ifm(nn, hh * stride_h + rh, ww * stride_w + rw, cc).astype(ifm.dtype),
            axis=[rh, rw],
        ),
        name="ethosu_pooling",
        attrs=pooling_attrs,
    )

    # Compute operation for the OFM DMA pipeline
    return dma_ofm_compute(pooling, ofm_layout, ofm_zero_point, ofm_scale, ofm_channels)
