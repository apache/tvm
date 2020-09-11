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
# pylint: disable=invalid-name
"""OneHot in python"""
import numpy as np


def one_hot(indices, on_value, off_value, depth, axis, dtype):
    """one_hot operator implemented in numpy.

    Returns a one-hot tensor where the locations repsented by indices take value on_value,
    other locations take value off_value.
    Final dimension is <indices outer dimensions> x depth x <indices inner dimensions>.

    Parameters
    ----------
    indices : numpy.ndarray
        Locations to set to on_value.

    on_value : int/float
        Value to fill at indices.

    off_value : int/float
        Value to fill at all other positions besides indices.

    depth : int
        Depth of the one-hot dimension.

    axis : int
        Axis to fill.

    dtype : str
        Data type of the output tensor.

    Returns
    -------
    ret : relay.Expr
        The one-hot tensor.
    """
    oshape = []
    true_axis = len(indices.shape) if axis == -1 else axis
    ndim = len(indices.shape) + 1
    indices_index = 0
    for i in range(0, ndim):
        if i == true_axis:
            oshape.append(depth)
        else:
            oshape.append(indices.shape[indices_index])
            indices_index += 1

    out = np.empty(oshape)
    output_indices = list(np.ndindex(out.shape))
    for output_index in output_indices:
        indices_indices = []
        for i, out_idx in enumerate(output_index):
            if i == true_axis:
                continue
            indices_indices.append(out_idx)

        index = output_index[true_axis]
        if indices[tuple(indices_indices)] == index:
            out[output_index] = on_value
        else:
            out[output_index] = off_value

    return out.astype(dtype)
