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
# pylint: disable=invalid-name, unused-argument, unused-variable
"""Ground truth max and average pooling operators in python."""
import itertools
import math
from typing import List, Tuple

import numpy as np
import tvm


def get_slice(
    spatial_dimensions: int,
    pad_np: np.array,
    dim_coord: Tuple[int],
    kernel: Tuple[int],
    strides: Tuple[int],
    dilation: Tuple[int],
) -> List[slice]:
    """
    Programmatically create a slice object of the right dimensions for pad_np.

    We assume pad_np's first two dimensions are not spatial and are not touched by the pad.

    pad_np[slice] should give the elements of the data that a pool operation will use for the
    step given in dim_coord.
    """
    slices = [slice(None)] * spatial_dimensions

    for nd in range(spatial_dimensions):
        slices[nd] = slice(
            dim_coord[nd] * strides[nd],
            dim_coord[nd] * strides[nd] + (kernel[nd] - 1) * dilation[nd] + 1,
            dilation[nd],
        )

    # Add back batch and channel dimensions
    slices = [slice(None), slice(None)] + slices

    return slices


def pad_tensor(
    np_arr: np.array,
    pad_value: float,
    padding_before: List[int],
    padding_after: List[int],
    dtype: str,
) -> np.array:
    """Pad the spatial dimensions of the given array."""
    orig_shape = list(np_arr.shape)
    padded_shape = list(np_arr.shape)
    n = len(orig_shape)
    for dim in range(2, n):
        i = dim - 2
        padded_shape[dim] += padding_after[i] + padding_before[i]

    pad_np = (np.zeros(shape=padded_shape) + pad_value).astype(dtype)
    ranges_it = [range(padded_shape[0]), range(padded_shape[1])]
    for dim in range(2, n):
        i = dim - 2
        ranges_it.append(range(padding_before[i], padding_before[i] + orig_shape[dim]))
    pad_np[np.ix_(*ranges_it)] = np_arr
    return pad_np


def poolnd_python(
    np_data: np.array,
    kernel: Tuple[int],
    strides: Tuple[int],
    dilation: Tuple[int],
    padding_before: Tuple[int],
    padding_after: Tuple[int],
    pool_type: str,
    count_include_pad: bool = True,
    ceil_mode: bool = False,
    dtype: str = "float32",
) -> np.array:
    """Ground truth pooling operator impelmented in numpy."""
    out_shape = [np_data.shape[0], np_data.shape[1]]
    for dim in range(2, len(np_data.shape)):
        i = dim - 2
        val = (
            float(
                np_data.shape[dim]
                - (kernel[i] - 1) * dilation[i]
                - 1
                + padding_before[i]
                + padding_after[i]
            )
            / strides[i]
        )

        if ceil_mode:
            out_shape.append(int(math.ceil(val) + 1))
        else:
            out_shape.append(int(math.floor(val) + 1))
    out_shape = tuple(out_shape)

    # Create a padded array, and a boolean mask showing which values are padded values
    pad_value = 0
    if pool_type == "max" and not count_include_pad:
        pad_value = tvm.te.min_value(dtype).value
    pad_data = pad_tensor(np_data, pad_value, padding_before, padding_after, dtype)
    pad_map = pad_tensor(np.ones_like(np_data), 0, padding_before, padding_after, "bool")

    # Create iterator which gives all indices for output array
    dim_iterators = []
    for spatial_dimension in range(2, len(np_data.shape)):
        dim_iterators.append(range(out_shape[spatial_dimension]))
    coord_iterator = itertools.product(*dim_iterators)

    ret_np = np.zeros(shape=out_shape).astype(dtype)
    for coordinate in coord_iterator:
        # Get index into the values that any pool operation will use for given coordinate
        np_index = get_slice(
            spatial_dimensions=len(out_shape) - 2,
            pad_np=pad_data,
            dim_coord=coordinate,
            kernel=kernel,
            strides=strides,
            dilation=dilation,
        )

        output_slice = [slice(None), slice(None)] + list(coordinate)
        reduction_axis = tuple(range(2, len(np_data.shape)))
        if pool_type == "avg":
            count_non_padded = (
                pad_data[np_index].size if count_include_pad else np.sum(pad_map[np_index])
            )
            # We summed over the non spatial dimensions too so divide by them
            count_non_padded /= out_shape[0] * out_shape[1]
            if count_non_padded == 0:
                ret_np[output_slice] = 0
            else:
                ret_np[output_slice] = (
                    np.sum(pad_data[np_index], axis=reduction_axis) / count_non_padded
                )
        elif pool_type == "max":
            count_non_padded = np.sum(pad_map[np_index])
            # All padded values, default to 0
            ret_np[output_slice] = np.max(pad_data[np_index], axis=reduction_axis)
        else:
            raise ValueError("Pool type {} is not supported".format(pool_type))

    return ret_np
