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
# pylint: disable=invalid-name, line-too-long, unused-variable, too-many-locals
"""Space to batch ND in python"""
import numpy as np


def space_to_batch_nd_python(data, block_shape, pad_before, pad_after, pad_value=0):
    """Space to Batch operator in python for NHWC layout.

    Parameters
    ----------
    data : np.ndarray
        N-D with shape [batch, spatial_shape, remaining_shapes],
        where spatial_shape has M dimensions.

    block_shape : list of ints
        1-D array of size [M] where M is number of spatial dims, specifies block
        size for each spatial dimension.

    pad_before : list of ints
        list of shape [M] where M is number of spatial dims, specifies
        zero-padding size before each spatial dimension.

    pad_after : list of ints
        list of shape [M] where M is number of spatial dims, specifies
        zero-padding size after each spatial dimension.

    pad_value : float, optional
        the value used for padding. Defaults to 0.

    Returns
    -------
    s2b_out : np.ndarray
        N-D with shape [batch * prod(block_shape),
                        padded_data[1] / block_shape[0], ..., padded_data[M] / block_shape[M-1],
                        remaining_shape]
    """
    M = len(block_shape)
    in_batch = data.shape[0]
    block_shape_prod = np.prod(block_shape)

    # Apply padding to input data
    input_shape = data.shape
    # Add the paddings for batch and remaining dims
    paddings = map(list, zip(pad_before, pad_after))
    paddings = [[0, 0]] + list(paddings) + [[0, 0]] * (data.ndim - 1 - M)
    padded_data = np.pad(data, paddings, mode="constant", constant_values=pad_value)
    padded_shape = padded_data.shape

    # Get the reshape shape and transpose axes
    r_shape = []
    trans_axis = []
    r_shape.append(in_batch)
    for i in range(1, M + 1):
        r_shape.append((int(padded_shape[i] // block_shape[i - 1])))
        r_shape.append(block_shape[i - 1])
        trans_axis.append(len(r_shape) - 1)

    axis_len = len(trans_axis)
    trans_axis.append(0)
    for i in range(axis_len):
        trans_axis.append(trans_axis[i] - 1)

    out_shape = []
    out_shape.append(int((in_batch * block_shape_prod)))
    for i in range(1, M + 1):
        out_shape.append(int(padded_shape[i] // block_shape[i - 1]))

    for i in range(M + 1, len(input_shape)):
        r_shape.append(input_shape[i])
        trans_axis.append(len(r_shape) - 1)
        out_shape.append(input_shape[i])

    s2b_out = np.reshape(padded_data, newshape=r_shape)
    s2b_out = np.transpose(s2b_out, axes=trans_axis)
    s2b_out = np.reshape(s2b_out, newshape=out_shape)

    return s2b_out
