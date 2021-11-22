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
"""Tensor Expressions for unary_elementwise for the NPU"""

from tvm import te
from .dma import dma_ofm_compute, dma_ifm_compute


def unary_elementwise_compute(
    ifm: te.Tensor,
    lut: te.Tensor,
    operator_type: str,
    ifm_scale: float,
    ifm_zero_point: int,
    ofm_scale: float,
    ofm_zero_point: int,
    ofm_channels: int,
    activation: str,
    clip_min: int,
    clip_max: int,
    rounding_mode: str,
    ifm_layout: str,
    ofm_layout: str,
) -> te.Tensor:
    """A compute operator representing the capabilities of unary_elementwise for the NPU.

    Parameters
    ----------
    ifm : te.Tensor
        The Input Feature Map tensor (IFM).
    lut : te.Tensor
        The look-up table values to use if activation = "LUT".
    operator_type: str
        The type of the unary elementwise operator.
            "ABS"
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
    rounding_mode : str
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
    te.Tensor
        The OFM tensor.

    """
    assert ifm.shape[0] == 1
    assert ifm_layout in {"NHWC", "NHCWB16"}
    assert ofm_layout in {"NHWC", "NHCWB16"}

    # Changing the ifm and ofm scale to conform with that expected by Vela API
    ofm_scale = ifm_scale / ofm_scale
    ifm_scale = 1.0

    # Compute operation for the IFM DMA pipeline
    dmaed_ifm = dma_ifm_compute(
        ifm, ifm_layout, ifm_zero_point, ifm_scale, ofm_channels, (0, 0, 0, 0)
    )

    # Unary elementwise compute operation
    ofm_height = dmaed_ifm.shape[1]
    ofm_width = dmaed_ifm.shape[2]

    unary_elementwise_attrs = {
        "op": "ethosu_unary_elementwise",
        "operator_type": operator_type,
        "activation": activation,
        "clip_min": clip_min,
        "clip_max": clip_max,
        "rounding_mode": rounding_mode,
    }

    operators = {"ABS": te.abs}

    unary_elementwise = te.compute(
        (1, ofm_height, ofm_width, ofm_channels),
        lambda nn, hh, ww, cc: operators[operator_type](
            dmaed_ifm(nn, hh, ww, cc).astype(ifm.dtype)
        ),
        name="ethosu_unary_elementwise",
        attrs=unary_elementwise_attrs,
    )

    # Compute operation for the OFM DMA pipeline
    return dma_ofm_compute(unary_elementwise, ofm_layout, ofm_zero_point, ofm_scale, ofm_channels)
