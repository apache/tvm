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
"""Correlation operators"""
from tvm import te

from .pad import pad
from ..utils import get_const_tuple


def correlation_nchw(
    data1, data2, kernel_size, max_displacement, stride1, stride2, padding, is_multiply
):
    """Correlation operator in NCHW layout.

    Parameters
    ----------
    data1 : tvm.te.Tensor
        4-D with shape [batch, channel, height, width]

    data2 : tvm.te.Tensor
        4-D with shape [batch, channel, height, width]

    kernel_size: int
        Kernel size for correlation, must be an odd number

    max_displacement: int
        Max displacement of Correlation

    stride1: int
        Stride for data1

    stride2: int
        Stride for data2 within the neightborhood centered around data1

    padding : int or a list/tuple of 2 or 4 ints
        Padding size, or
        [pad_height, pad_width] for 2 ints, or
        [pad_top, pad_left, pad_bottom, pad_right] for 4 ints

    is_multiply: bool
        operation type is either multiplication or substraction

    Returns
    -------
    Output : tvm.te.Tensor
        4-D with shape [batch, out_channel, out_height, out_width]
    """
    # pylint: disable=unnecessary-lambda, invalid-name
    data_shape = get_const_tuple(data1.shape)
    assert get_const_tuple(data2.shape) == data_shape, "data1 and data2 should have the same shape"
    assert kernel_size > 0 and kernel_size % 2, "kernel_size should be non-negative odd number"
    if isinstance(padding, (tuple, list)):
        if len(padding) == 2:
            pad_before_h = pad_after_h = padding[0]
            pad_before_w = pad_after_w = padding[1]
        elif len(padding) == 4:
            pad_before_h, pad_before_w, pad_after_h, pad_after_w = padding
        else:
            raise ValueError("invalid padding")
    elif isinstance(padding, int):
        pad_before_h = pad_after_h = pad_before_w = pad_after_w = padding
    else:
        raise ValueError("invalid padding")
    pad_before = [0, 0, pad_before_h, pad_before_w]
    pad_after = [0, 0, pad_after_h, pad_after_w]
    padded_data1 = pad(data1, pad_before, pad_after)
    padded_data2 = pad(data2, pad_before, pad_after)

    batch, channel, height, width = data_shape

    kernel_radius = (kernel_size - 1) // 2
    border_size = max_displacement + kernel_radius
    displacement_radius = max_displacement // stride2
    displacement_size = 2 * displacement_radius + 1

    padded_width = width + pad_before_w + pad_after_w
    padded_height = height + pad_before_h + pad_after_h
    out_channel = displacement_size * displacement_size
    out_height = (padded_height - 2 * border_size + stride1 - 1) // stride1
    out_width = (padded_width - 2 * border_size + stride1 - 1) // stride1

    rc = te.reduce_axis((0, channel), name="rc")
    ry = te.reduce_axis((0, kernel_size), name="ry")
    rx = te.reduce_axis((0, kernel_size), name="rx")

    if is_multiply:
        corr_func = lambda x, y: x * y
    else:
        corr_func = lambda x, y: te.abs(x - y)

    def _compute_correlation(n, q, i, j):
        # location in data1
        y1 = i * stride1 + max_displacement
        x1 = j * stride1 + max_displacement
        # location in data2
        y2 = y1 + (te.indexdiv(q, displacement_size) - displacement_radius) * stride2
        x2 = x1 + (te.indexmod(q, displacement_size) - displacement_radius) * stride2
        return te.sum(
            corr_func(padded_data1[n, rc, y1 + ry, x1 + rx], padded_data2[n, rc, y2 + ry, x2 + rx]),
            axis=[rc, ry, rx],
        )

    correlation = te.compute(
        (batch, out_channel, out_height, out_width),
        lambda n, q, i, j: _compute_correlation(n, q, i, j),
        tag="correlation_nchw",
    )
    return correlation / (kernel_size * kernel_size * channel)
