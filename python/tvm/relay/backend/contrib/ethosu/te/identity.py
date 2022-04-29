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
from tvm import te
from .dma import read_compute, write_compute


def identity_compute(
    ifm: te.Tensor,
    lut: te.Tensor,
    ifm_scale: float,
    ifm_zero_point: int,
    ofm_scale: float,
    ofm_zero_point: int,
    activation: str,
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

    Returns
    -------
    te.Tensor
        The Output Feature Map tensor.

    """
    dmaed_ifm = read_compute(ifm, ifm_zero_point, ifm_scale)
    id_attrs = {"op": "ethosu_identity", "activation": activation}

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

    dmaed_ofm = write_compute(identity, ofm_zero_point, ofm_scale)

    return dmaed_ofm
