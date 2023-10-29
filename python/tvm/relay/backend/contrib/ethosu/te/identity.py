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
"""Tensor Expression for identity"""
import numpy as np
from tvm import te
from tvm.contrib.ethosu.cascader import TESubgraph, EthosuPart, Propagator, register_matcher

from .dma import read_compute, write_compute


def identity_compute(
    ifm: te.Tensor,
    lut: te.Tensor,
    ifm_scale: float,
    ifm_zero_point: int,
    ofm_scale: float,
    ofm_zero_point: int,
    activation: str,
    rounding_mode: str,
) -> te.Tensor:
    """A compute operator for the NPU identity operator.

    Parameters
    ----------
    ifm : te.Tensor
        The Input Feature Map tensor (IFM).
    lut : te.Tensor
        The look-up table values to use if activation is "LUT", "TANH" or "SIGMOID".
    ifm_scale : float
        The quantization scale for the Input Feature Map tensor.
    ifm_zero_point : int
        The quantization zero point for the Input Feature Map tensor.
    ofm_scale : float
        The quantization scale for the Output Feature Map tensor.
    ofm_zero_point : int
        The quantization zero point for the Output Feature Map tensor.
    activation : str
        The activation function to use.
            "NONE" - no activation function.
            "TANH" - tanh activation function.
            "SIGMOID" - sigmoid activation function.
            "LUT" - use a look-up table to perform the activation function.
    rounding_mode : str
        The rounding mode to apply to the Output Feature Map tensor.
            "TFL" - Tensorflow Lite rounding scheme.
            "TRUNCATE" - Truncate towards zero.
            "NATURAL" - Round to nearest value, with x.5 rounded up towards +infinity.

    Returns
    -------
    te.Tensor
        The Output Feature Map tensor.
    """
    dmaed_ifm = read_compute(ifm, ifm_zero_point, ifm_scale)
    id_attrs = {"op": "ethosu_identity", "activation": activation, "rounding_mode": rounding_mode}

    has_lut = activation in ("TANH", "LUT", "SIGMOID")

    # This is a trick to insert the LUT tensor into the TE graph if LUT is present
    lut_expr = (lut[0] + lut[255]).astype(ifm.dtype) if has_lut else 0

    # Add the LUT tensor to the attributes to be able to later tell which tensor is the LUT
    if has_lut:
        id_attrs["lut"] = lut

    identity = te.compute(
        ifm.shape,
        lambda *i: (dmaed_ifm(*i) + lut_expr).astype(ifm.dtype),
        name="ethosu_identity",
        attrs=id_attrs,
    )
    length = len(ifm.shape)
    ifm_matrix = np.identity(length + 1)
    offset = np.zeros(length, dtype="int64")
    ifm_propagator = Propagator(
        ifm_matrix,
        offset.tolist(),
    )
    propagator_attrs = {
        "ifm_propagator": ifm_propagator,
    }
    return write_compute(identity, ofm_zero_point, ofm_scale, attrs=propagator_attrs)


@register_matcher
def match_ethosu_identity(output_tensor, device_config):
    """Match a Tensor Expression corresponding to an NPU identity.

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
    identity = write.op.input_tensors[0]
    if identity.op.name != "ethosu_identity":
        return None
    read = identity.op.input_tensors[0]
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

    input_tensors_shape = input_tensors[0].shape
    length = len(input_tensors_shape)
    assert length <= 4, "Input tensor shape must be <= 4 for the identity operator"
    channels = int(input_tensors_shape[length - 1]) if length >= 3 else 1

    subkernels = len(device_config.get_kernel_steps(identity.op.name, 1, 1, ifm_dtype))

    input_layout = output_layout = "NHWC"
    output_quantum = device_config.get_output_quantum(output_layout)

    valid_block_configs = device_config.get_valid_block_configs(
        propagators[0],
        identity.op.attrs,
        output_tensor.shape,
        channels,
        channels,
        output_layout,
        input_layout,
        ifm_dtype,
        ofm_dtype,
        1,
        1,
    )

    return EthosuPart(
        subgraph,
        propagators,
        output_quantum,
        subkernels,
        valid_block_configs,
    )
