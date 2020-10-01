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
"""Batch to space ND in python"""
import numpy as np
from . import strided_slice_python


def batch_to_space_nd_python(data, block_shape, crop_begin_list, crop_end_list):
    """Batch to Space operator in python for NHWC layout.

    Parameters
    ----------
    data : np.ndarray
        N-D with shape [batch, spatial_shape, remaining_shapes],
        where spatial_shape has M dimensions.

    block_shape : list of ints
        1-D array of size [M] where M is number of spatial dims, specifies block
        size for each spatial dimension.

    crop_begin_list : list of ints
        list of shape [M] where M is number of spatial dims, specifies
        begin crop size for each spatial dimension.

    crop_end_list : list of ints
        list of shape [M] where M is number of spatial dims, specifies
        end crop size for each spatial dimension.

    Returns
    -------
    b2s_out : np.ndarray
        N-D with shape
        [batch / prod(block_shape),
        in_shape[1] * block_shape[0] - crop_begin_list[0] - crop_end_list[0], ...,
        in_shape[M] * block_shape[M-1] - crop_begin_list[M-1] - crop_end_list[M-1],
        remaining_shape]
    """
    in_shape = data.shape
    N = len(in_shape)
    M = len(block_shape)
    block_shape_prod = np.prod(block_shape)
    in_batch = data.shape[0]
    axis = []
    r_p_shape = []

    r_shape = [block_shape[i] for i in range(0, M)]
    axis.append(len(r_shape))
    r_shape.append(in_batch // block_shape_prod)

    for i in range(1, N):
        axis.append(len(r_shape))
        if len(axis) < (M + N):
            axis.append(len(r_shape) - (M + 1))
        r_shape.append(in_shape[i])

    r_p_shape.append(int((in_batch / block_shape_prod)))
    for i in range(1, M + 1):
        r_p_shape.append(in_shape[i] * block_shape[i - 1])
    for i in range(M + 1, N):
        r_p_shape.append(in_shape[i])

    b2s_out = np.reshape(data, newshape=r_shape)
    b2s_out = np.transpose(b2s_out, axes=axis)
    b2s_out = np.reshape(b2s_out, newshape=r_p_shape)

    # Crop the start and end of dimensions of b2s_out
    begin_idx = []
    end_idx = []
    strides = []

    for i, _ in enumerate(r_p_shape):
        strides.append(1)
        if 0 < i <= M:
            # begin and end index for spatial dimensions
            begin_idx.append(crop_begin_list[i - 1])
            end_idx.append(r_p_shape[i] - crop_end_list[i - 1])
        else:
            begin_idx.append(0)
            end_idx.append(r_p_shape[i])

    b2s_out = strided_slice_python(b2s_out, begin_idx, end_idx, strides)
    return b2s_out
