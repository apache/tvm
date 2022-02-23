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
"""Tensor Expressions for operations that will be inlined"""
import numpy as np  # type: ignore

from tvm.contrib.ethosu.cascader import TESubgraph, InlinePart, Propagator, register_matcher


INLINE_OPS = {"T_reshape", "T_strided_slice"}


@register_matcher
def match_ethosu_inline(output_tensor, device_config):
    """Match a Tensor Expression corresponding to an operator that will be inlined.

    If the Tensor Expression matches, an InlinePart will be created that models the
    matched Tensor Expression. Otherwise, None will be returned. This matcher is
    naive and assumes nothing about the compute of the Tensor Expression. Therefore,
    the resulting InlinePart will have full-tensor dependencies (i.e. each output
    element depends on every input element).

    Parameters
    ----------
    output_tensor : tvm.te.Tensor
        The tensor to attempt to match with.
    device_config : EthosuDeviceConfig
        Target device configuration

    Returns
    -------
    Union[None, InlinePart]
        The created InlinePart if there was a match, otherwise None.

    """
    if output_tensor.op.name not in INLINE_OPS:
        return None

    input_tensors = output_tensor.op.input_tensors
    propagators = []
    output_dims = len(output_tensor.shape)
    for input_tensor in input_tensors:
        input_dims = len(input_tensor.shape)
        transform_matrix = np.zeros((input_dims + 1, output_dims + 1))
        for i, axis in enumerate(input_tensor.shape):
            transform_matrix[i, output_dims] = int(axis)
        transform_matrix[input_dims, output_dims] = 1
        offset_vector = np.zeros(input_dims, dtype="int64")
        propagators.append(
            Propagator(
                transform_matrix.tolist(),
                offset_vector.tolist(),
            )
        )

    subgraph = TESubgraph(input_tensors, output_tensor)
    return InlinePart(
        subgraph,
        propagators,
    )
