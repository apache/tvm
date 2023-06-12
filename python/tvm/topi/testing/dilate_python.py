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
"""Dilate operation in python"""
import numpy as np


def dilate_python(input_np, strides, dilation_value=0.0, out_dtype=None):
    """Dilate operation.

    Parameters
    ----------
    input_np : numpy.ndarray
        n-D, can be any layout.

    strides : list / tuple of n ints
        Dilation stride on each dimension, 1 means no dilation.

    dilation_value : int/float, optional
        Value used to dilate the input.

    out_dtype : Option[str]
        The datatype of the dilated array.  If unspecified, will use
        the same dtype as the input array.

    Returns
    -------
    output_np : numpy.ndarray
        n-D, the same layout as Input.

    """
    assert len(input_np.shape) == len(
        strides
    ), f"Input dimension and strides size dismatch : {len(input_np.shape)} vs {len(strides)}"

    if out_dtype is None:
        out_dtype = input_np.dtype

    output_size = [
        (input_dim - 1) * stride + 1 for input_dim, stride in zip(input_np.shape, strides)
    ]
    non_zero_elements = np.ix_(
        *[range(0, output_dim, stride) for output_dim, stride in zip(output_size, strides)]
    )

    output_np = np.full(shape=output_size, fill_value=dilation_value, dtype=out_dtype)
    output_np[non_zero_elements] = input_np

    return output_np
