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
import numpy as np
from tvm import te
from tvm.contrib.ethosu.cascader import TESubgraph, EthosuPart, Propagator, register_matcher

from .dma import dma_ofm_compute, dma_ifm_compute
from .common import get_layout_transform_matrices


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
    rounding_mode: str,
    ifm_layout: str,
    ifm2_layout: str,
    ofm_layout: str,
    ofm_dtype: str,
    use_rescale: bool,
    rescale_scale: int,
    rescale_shift: int,
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
    rounding_mode : str
        The rounding mode to apply to the Output Feature Map tensor.
            "TFL" - Tensorflow Lite rounding scheme.
            "TRUNCATE" - Truncate towards zero.
            "NATURAL" - Round to nearest value, with x.5 rounded up towards +infinity.
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
    use_rescale : bool
        Use explicit scaling if True.
    rescale_scale : int
        Scale value for rescale. For 32-bit operations scale is not applied but shift is.
    rescale_shift : int
        Shift value for rescale.

    Returns
    -------
    te.Tensor
        The Output Feature Map tensor.
    """
    assert ifm.shape[0] == 1
    assert ifm2.shape[0] == 1
    assert ifm_layout in {"NHWC", "NHCWB16"}
    assert ifm2_layout in {"NHWC", "NHCWB16"}
    assert ofm_layout in {"NHWC", "NHCWB16"}

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
        "rounding_mode": rounding_mode,
        "use_rescale": use_rescale,
        "rescale_scale": rescale_scale,
        "rescale_shift": rescale_shift,
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

    has_lut = activation in ("TANH", "LUT", "SIGMOID")
    # This is a trick to insert the LUT tensor into the TE graph if LUT is present
    lut_expr = (lut[0] + lut[255]).astype(ifm.dtype) if has_lut else 0

    # Add the LUT tensor to the attributes to be able to later tell which tensor is the LUT
    if has_lut:
        binary_elementwise_attrs["lut"] = lut

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
                dmaed_ifm(nn, hh, ww, cc).astype(ifm.dtype) + lut_expr,
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
                ).astype(ifm.dtype)
                + lut_expr,
            ).astype(ofm_dtype),
            name="ethosu_binary_elementwise",
            attrs=binary_elementwise_attrs,
        )

    nhwc_to_nhcwb16, nhcwb16_to_nhwc = get_layout_transform_matrices(int(ifm_channels))

    ifm_matrix = [
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],
    ]
    ifm2_matrix = [
        [1, 0, 0, 0, 0],
        [0, (1 - int(broadcast[1])), 0, 0, int(broadcast[1])],
        [0, 0, (1 - int(broadcast[2])), 0, int(broadcast[2])],
        [0, 0, 0, (1 - int(broadcast[3])), int(broadcast[3])],
        [0, 0, 0, 0, 1],
    ]
    if ofm_layout == "NHCWB16":
        ifm_matrix = np.matmul(ifm_matrix, nhcwb16_to_nhwc).tolist()
        ifm2_matrix = np.matmul(ifm2_matrix, nhcwb16_to_nhwc).tolist()
    if ifm_layout == "NHCWB16":
        ifm_matrix = np.matmul(nhwc_to_nhcwb16, ifm_matrix).tolist()
    if ifm2_layout == "NHCWB16":
        ifm2_matrix = np.matmul(nhwc_to_nhcwb16, ifm2_matrix).tolist()
    ifm_propagator = Propagator(
        ifm_matrix,
        [0, 0, 0, 0] if ifm_layout == "NHWC" else [0, 0, 0, 0, 0],
    )
    ifm2_propagator = Propagator(
        ifm2_matrix,
        [0, 0, 0, 0] if ifm2_layout == "NHWC" else [0, 0, 0, 0, 0],
    )
    propagator_attrs = {
        "ifm_propagator": ifm_propagator,
        "ifm2_propagator": ifm2_propagator,
    }

    # Compute operation for the OFM DMA pipeline
    return dma_ofm_compute(
        binary_elementwise,
        ofm_layout,
        ofm_zero_point,
        ofm_scale,
        ifm_channels,
        attrs=propagator_attrs,
    )


@register_matcher
def match_ethosu_binary_elementwise(output_tensor, device_config):
    """Match a Tensor Expression corresponding to an NPU Binary Elementwise.

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
    binary_elementwise = convert_to_nhcwb16.op.input_tensors[0]
    if binary_elementwise.op.name != "ethosu_binary_elementwise":
        return None
    pad = binary_elementwise.op.input_tensors[0]
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
    pad2 = binary_elementwise.op.input_tensors[1]
    if pad2.op.name != "ethosu_pad":
        return None
    upscale2 = pad2.op.input_tensors[0]
    if upscale2.op.name != "ethosu_upscale":
        return None
    convert_to_nhwc2 = upscale2.op.input_tensors[0]
    if convert_to_nhwc2.op.name != "ethosu_convert_to_nhwc":
        return None
    read2 = convert_to_nhwc2.op.input_tensors[0]
    if read2.op.name != "ethosu_read":
        return None

    input_tensors = [
        read.op.input_tensors[0],
        read2.op.input_tensors[0],
    ]
    subgraph = TESubgraph(input_tensors, output_tensor)
    propagators = [
        write.op.attrs["ifm_propagator"],
        write.op.attrs["ifm2_propagator"],
    ]
    ifm_dtype = input_tensors[0].dtype
    ofm_dtype = output_tensor.dtype

    output_layout = convert_to_nhcwb16.op.attrs["layout"]
    input_layout = convert_to_nhwc.op.attrs["layout"]
    input2_layout = convert_to_nhwc2.op.attrs["layout"]
    output_quantum = device_config.get_output_quantum(output_layout)

    block_config = device_config.get_elementwise_block_config(
        propagators[0],
        propagators[1],
        binary_elementwise.op.attrs,
        output_tensor.shape,
        output_layout,
        input_layout,
        input2_layout,
        ifm_dtype,
        ofm_dtype,
    )

    return EthosuPart(
        subgraph,
        propagators,
        output_quantum,
        1,
        block_config,
    )
