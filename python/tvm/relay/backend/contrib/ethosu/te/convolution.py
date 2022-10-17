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
"""Tensor Expressions for convolutions for the NPU"""
from typing import Tuple, Union, List
import numpy as np  # type: ignore

from tvm import te  # type: ignore
from tvm.contrib.ethosu.cascader import TESubgraph, EthosuPart, Propagator, register_matcher

from .dma import dma_ofm_compute, dma_ifm_compute
from .common import get_layout_transform_matrices


def conv2d_compute(
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
    rounding_mode: str,
    upscale: str,
    ifm_layout: str,
    ofm_layout: str,
) -> te.Tensor:
    """A compute operator representing the capabilities of a 2D convolution for the NPU.

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
    dilation : Union[Tuple[int, int], List[int]]
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
    rounding_mode : str
        The rounding mode to apply to the Output Feature Map tensor.
            "TFL" - Tensorflow Lite rounding scheme.
            "TRUNCATE" - Truncate towards zero.
            "NATURAL" - Round to nearest value, with x.5 rounded up towards +infinity.
    upscale : str
        The 2x2 upscaling mode to apply to the Input Feature Map tensor.
            "NONE" - no upscaling.
            "NEAREST" - upscale using nearest neighbour.
            "ZEROS" - upscale using zeros.
            "NATURAL" - Round to nearest value, with x.5 rounded up towards +infinity.
    ifm_layout : str
        The layout of the Input Feature Map tensor. Can be "NHWC" or "NHCWB16".
    ofm_layout : str
        The layout of the Output Feature Map tensor. Can be "NHWC" or "NHCWB16".

    Returns
    -------
    te.Tensor
        The OFM tensor.

    """
    assert ifm.shape[0] == 1
    assert ifm_layout in {"NHWC", "NHCWB16"}
    assert ofm_layout in {"NHWC", "NHCWB16"}

    padding = [int(v) for v in padding]
    stride_h, stride_w = [int(v) for v in strides]
    dilation_h, dilation_w = [int(v) for v in dilation]
    ofm_channels, kernel_h, kernel_w, ifm_channels = [int(v) for v in weight.shape]
    upscale_factor = 2 if upscale != "NONE" else 1

    # Compute operation for the IFM DMA pipeline
    dmaed_ifm = dma_ifm_compute(
        ifm,
        ifm_layout,
        ifm_zero_point,
        ifm_scale,
        weight.shape[3],
        padding,
        upscale_factor,
    )

    # 2D Convolution compute operation
    dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
    dilated_kernel_w = (kernel_w - 1) * dilation_w + 1
    ofm_height = (dmaed_ifm.shape[1] - dilated_kernel_h) // stride_h + 1
    ofm_width = (dmaed_ifm.shape[2] - dilated_kernel_w) // stride_w + 1
    rc = te.reduce_axis((0, ifm_channels), name="rc")
    rh = te.reduce_axis((0, kernel_h), name="ry")
    rw = te.reduce_axis((0, kernel_w), name="rx")

    conv2d_attrs = {
        "op": "ethosu_conv2d",
        "weight_zero_point": weight_zero_point,
        "activation": activation,
        "upscale": upscale,
        "clip_min": clip_min,
        "clip_max": clip_max,
        "rounding_mode": rounding_mode,
        "stride_h": stride_h,
        "stride_w": stride_w,
        "dilation_h": dilation_h,
        "dilation_w": dilation_w,
    }

    has_lut = activation in ("TANH", "LUT", "SIGMOID")

    # This is a trick to insert the LUT tensor into the TE graph if LUT is present
    lut_expr = (lut[0] + lut[255]).astype(ifm.dtype) if has_lut else 0

    # Add the LUT tensor to the attributes to be able to later tell which tensor is the LUT
    if has_lut:
        conv2d_attrs["lut"] = lut

    conv = te.compute(
        (1, ofm_height, ofm_width, ofm_channels),
        lambda nn, hh, ww, cc: te.sum(
            dmaed_ifm(
                nn, hh * stride_h + rh * dilation_h, ww * stride_w + rw * dilation_w, rc
            ).astype(ifm.dtype)
            * weight[cc, rh, rw, rc].astype(ifm.dtype)
            # This is a trick to load 10 elements of the scale_bias at once, not accurate maths
            + (scale_bias[cc, 0] * scale_bias[cc, 9] + lut_expr).astype(ifm.dtype),
            axis=[rh, rw, rc],
        ),
        name="ethosu_conv2d",
        attrs=conv2d_attrs,
    )

    nhwc_to_nhcwb16, nhcwb16_to_nhwc = get_layout_transform_matrices(ofm_channels)

    ifm_matrix = [
        [1, 0, 0, 0, 0],
        [0, stride_h, 0, 0, (dilated_kernel_h - stride_h)],
        [0, 0, stride_w, 0, (dilated_kernel_w - stride_w)],
        [0, 0, 0, 0, ifm_channels],
        [0, 0, 0, 0, 1],
    ]
    weights_matrix = [
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, kernel_h],
        [0, 0, 0, 0, kernel_w],
        [0, 0, 0, 0, ifm_channels],
        [0, 0, 0, 0, 1],
    ]
    bias_matrix = [
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 10],
        [0, 0, 0, 0, 1],
    ]
    if ofm_layout == "NHCWB16":
        ifm_matrix = np.matmul(ifm_matrix, nhcwb16_to_nhwc).tolist()
        weights_matrix = np.matmul(weights_matrix, nhcwb16_to_nhwc).tolist()
        bias_matrix = np.matmul(bias_matrix, nhcwb16_to_nhwc).tolist()
    if ifm_layout == "NHCWB16":
        ifm_matrix = np.matmul(nhwc_to_nhcwb16, ifm_matrix).tolist()
    ifm_propagator = Propagator(
        ifm_matrix,
        [0, -padding[0], -padding[1], 0]
        if ifm_layout == "NHWC"
        else [0, -padding[0], 0, -padding[1], 0],
    )
    weights_propagator = Propagator(
        weights_matrix,
        [0, 0, 0, 0],
    )
    bias_propagator = Propagator(
        bias_matrix,
        [0, 0],
    )
    propagator_attrs = {
        "ifm_propagator": ifm_propagator,
        "weights_propagator": weights_propagator,
        "bias_propagator": bias_propagator,
    }

    # Compute operation for the OFM DMA pipeline
    dma_ofm = dma_ofm_compute(
        conv, ofm_layout, ofm_zero_point, ofm_scale, ofm_channels, attrs=propagator_attrs
    )
    return dma_ofm


@register_matcher
def match_ethosu_conv2d(output_tensor, device_config):
    """Match a Tensor Expression corresponding to an NPU Conv2D.

    If the Tensor Expression matches, an EthosuPart will be created that models the
    matched Tensor Expression. Otherwise, None will be returned.

    Parameters
    ----------
    output_tensor : tvm.te.Tensor
        The tensor to attempt to match with.
    device_config : EthosuDeviceConfig
        Target device configuration

    Returns
    -------
    Union[None, EthosuPart]
        The created EthosuPart if there was a match, otherwise None.

    """
    write = output_tensor
    if write.op.name != "ethosu_write":
        return None
    convert_to_nhcwb16 = write.op.input_tensors[0]
    if convert_to_nhcwb16.op.name != "ethosu_convert_to_nhcwb16":
        return None
    conv2d = convert_to_nhcwb16.op.input_tensors[0]
    if conv2d.op.name != "ethosu_conv2d":
        return None
    pad = conv2d.op.input_tensors[0]
    if pad.op.name != "ethosu_pad":
        return None
    upscale = pad.op.input_tensors[0]
    if upscale.op.name != "ethosu_upscale":
        return None
    convert_to_nhwc = upscale.op.input_tensors[0]
    if convert_to_nhwc.op.name != "ethosu_convert_to_nhwc":
        return None
    read = convert_to_nhwc.op.input_tensors[0]
    if read.op.name != "ethosu_read":
        return None

    input_tensors = [
        read.op.input_tensors[0],
        conv2d.op.input_tensors[1],
        conv2d.op.input_tensors[2],
    ]

    subgraph = TESubgraph(input_tensors, output_tensor)
    propagators = [
        write.op.attrs["ifm_propagator"],
        write.op.attrs["weights_propagator"],
        write.op.attrs["bias_propagator"],
    ]
    ifm_dtype = input_tensors[0].dtype
    ofm_dtype = output_tensor.dtype

    # Use channels from the weights tensor since that its shape doesn't change during layout
    # conversion
    ifm_channels = int(input_tensors[1].shape[3])
    ofm_channels, kernel_height, kernel_width = (int(axis) for axis in input_tensors[1].shape[0:3])
    kernel_elements = kernel_height * kernel_width

    is_part_kernel = device_config.is_partkernel(
        conv2d.op.name, ifm_channels, ifm_dtype, kernel_elements
    )
    subkernels = len(
        device_config.get_kernel_steps(
            conv2d.op.name, kernel_height, kernel_width, ifm_dtype, is_part_kernel
        )
    )

    output_layout = convert_to_nhcwb16.op.attrs["layout"]
    input_layout = convert_to_nhwc.op.attrs["layout"]
    output_quantum = device_config.get_output_quantum(output_layout)

    valid_block_configs = device_config.get_valid_block_configs(
        propagators[0],
        conv2d.op.attrs,
        output_tensor.shape,
        ofm_channels,
        ifm_channels,
        output_layout,
        input_layout,
        ifm_dtype,
        ofm_dtype,
        kernel_height,
        kernel_width,
    )

    return EthosuPart(
        subgraph,
        propagators,
        output_quantum,
        subkernels,
        valid_block_configs,
        1,
    )
