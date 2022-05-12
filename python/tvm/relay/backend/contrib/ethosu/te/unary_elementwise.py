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

import numpy as np
from tvm import te
from tvm.contrib.ethosu.cascader import TESubgraph, EthosuPart, Propagator, register_matcher
from .dma import dma_ofm_compute, dma_ifm_compute
from .common import get_layout_transform_matrices


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

    def clz_imp(inp):
        # Assuming that it's a 32 bit int
        return 32 - te.log2(inp)

    operators = {"ABS": te.abs, "CLZ": clz_imp}

    unary_elementwise = te.compute(
        (1, ofm_height, ofm_width, ofm_channels),
        lambda nn, hh, ww, cc: operators[operator_type](
            dmaed_ifm(nn, hh, ww, cc).astype(ifm.dtype)
        ),
        name="ethosu_unary_elementwise",
        attrs=unary_elementwise_attrs,
    )

    nhwc_to_nhcwb16, nhcwb16_to_nhwc = get_layout_transform_matrices(int(ofm_channels))

    ifm_matrix = [
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],
    ]
    if ofm_layout == "NHCWB16":
        ifm_matrix = np.matmul(ifm_matrix, nhcwb16_to_nhwc).tolist()
    if ifm_layout == "NHCWB16":
        ifm_matrix = np.matmul(nhwc_to_nhcwb16, ifm_matrix).tolist()

    ifm_propagator = Propagator(
        ifm_matrix,
        [0, 0, 0, 0] if ifm_layout == "NHWC" else [0, 0, 0, 0, 0],
    )
    propagator_attrs = {"ifm_propagator": ifm_propagator}

    # Compute operation for the OFM DMA pipeline
    return dma_ofm_compute(
        unary_elementwise,
        ofm_layout,
        ofm_zero_point,
        ofm_scale,
        ofm_channels,
        attrs=propagator_attrs,
    )


@register_matcher
def match_ethosu_unary_elementwise(output_tensor, device_config):
    """Match a Tensor Expression corresponding to an NPU Unary Elementwise.

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
    unary_elementwise = convert_to_nhcwb16.op.input_tensors[0]
    if unary_elementwise.op.name != "ethosu_unary_elementwise":
        return None
    pad = unary_elementwise.op.input_tensors[0]
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
    ]
    subgraph = TESubgraph(input_tensors, output_tensor)
    propagators = [
        write.op.attrs["ifm_propagator"],
    ]
    ifm_dtype = input_tensors[0].dtype
    ofm_dtype = output_tensor.dtype

    output_layout = convert_to_nhcwb16.op.attrs["layout"]
    input_layout = convert_to_nhwc.op.attrs["layout"]
    output_quantum = device_config.get_output_quantum(output_layout)

    block_config = device_config.get_elementwise_block_config(
        propagators[0],
        None,
        unary_elementwise.op.attrs,
        output_tensor.shape,
        output_layout,
        input_layout,
        None,
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
