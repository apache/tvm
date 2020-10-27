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
# pylint: disable=invalid-name, unused-variable, too-many-locals
# pylint: disable=unused-argument, redefined-builtin
"""Dilation2D operators"""
from __future__ import absolute_import as _abs
from tvm import te
from tvm.topi.utils import simplify
from ..nn.pad import pad
from ..nn.utils import get_pad_tuple


def dilation2d_nchw(input, filter, stride, padding, dilations, out_dtype=None):
    """Morphological dilation operator in NCHW layout.

    Parameters
    ----------
    input : tvm.te.Tensor
        4-D with shape [batch, in_channel, in_height, in_width]

    filter : tvm.te.Tensor
        3-D with shape [ in_channel, filter_height, filter_width]

    stride : int or a list/tuple of two ints
        Stride size, or [stride_height, stride_width]

    padding : int or str
        Padding size

    dilations: int or a list/tuple of two ints
        dilation size, or [dilation_height, dilation_width]

    out_dtype : Optional[str]
        Specifies the output data type.

    Returns
    -------
    Output : tvm.te.Tensor
        4-D with shape [batch, in_channel, out_height, out_width]
    """
    if out_dtype is None:
        out_dtype = input.dtype
    assert isinstance(stride, int) or len(stride) == 2
    assert isinstance(dilations, int) or len(dilations) == 2
    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride

    if isinstance(dilations, int):
        dilation_h = dilation_w = dilations
    else:
        dilation_h, dilation_w = dilations

    batch, in_channel, in_height, in_width = input.shape
    channel, kernel_h, kernel_w = filter.shape
    assert (
        in_channel.value == channel.value
    ), "For Dilation2D input and filter channels should be same."

    # compute the output shape
    dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
    dilated_kernel_w = (kernel_w - 1) * dilation_w + 1
    pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
        padding, (dilated_kernel_h, dilated_kernel_w)
    )

    out_height = simplify((in_height - dilated_kernel_h + pad_top + pad_down) // stride_h + 1)
    out_width = simplify((in_width - dilated_kernel_w + pad_left + pad_right) // stride_w + 1)
    # compute graph
    pad_before = [0, 0, pad_top, pad_left]
    pad_after = [0, 0, pad_down, pad_right]
    temp = pad(input, pad_before, pad_after, name="pad_temp")
    ry = te.reduce_axis((0, kernel_h), name="ry")
    rx = te.reduce_axis((0, kernel_w), name="rx")

    return te.compute(
        (batch, in_channel, out_height, out_width),
        lambda nn, ff, yy, xx: te.max(
            temp[nn, ff, yy * stride_h + ry * dilation_h, xx * stride_w + rx * dilation_w].astype(
                out_dtype
            )
            + filter[ff, ry, rx].astype(out_dtype),
            axis=[ry, rx],
        ),
        tag="dilation2d_nchw",
    )


def dilation2d_nhwc(input, filter, stride, padding, dilations, out_dtype=None):
    """Morphological 2d dilation NHWC layout.

    Parameters
    ----------
    input : tvm.te.Tensor
        4-D with shape [batch, in_height, in_width, in_channel]

    filter : tvm.te.Tensor
        3-D with shape [filter_height, filter_width, in_channel]

    stride : int or a list/tuple of two ints
        Stride size, or [stride_height, stride_width]

    padding : int
        Padding size

    dilations: int or a list/tuple of two ints
        dilation size, or [dilation_height, dilation_width]

    out_dtype : Optional[str]
        Specifies the output data type.

    Returns
    -------
    Output : tvm.te.Tensor
        4-D with shape [batch, out_height, out_width, in_channel]
    """
    if out_dtype is None:
        out_dtype = input.dtype
    assert isinstance(stride, int) or len(stride) == 2
    assert isinstance(dilations, int) or len(dilations) == 2
    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride

    if isinstance(dilations, int):
        dilation_h = dilation_w = dilations
    else:
        dilation_h, dilation_w = dilations

    batch, in_height, in_width, in_channel = input.shape
    kernel_h, kernel_w, channel = filter.shape
    assert (
        in_channel.value == channel.value
    ), "For Dilation2D input and filter channels should be same."

    # compute the output shape
    dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
    dilated_kernel_w = (kernel_w - 1) * dilation_w + 1
    pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
        padding, (dilated_kernel_h, dilated_kernel_w)
    )

    out_height = simplify((in_height - dilated_kernel_h + pad_top + pad_down) // stride_h + 1)
    out_width = simplify((in_width - dilated_kernel_w + pad_left + pad_right) // stride_w + 1)
    pad_before = [0, pad_top, pad_left, 0]
    pad_after = [0, pad_down, pad_right, 0]
    padded_input = pad(input, pad_before, pad_after, name="padded_input")
    ry = te.reduce_axis((0, kernel_h), name="ry")
    rx = te.reduce_axis((0, kernel_w), name="rx")

    return te.compute(
        (batch, out_height, out_width, in_channel),
        lambda nn, yy, xx, ff: te.max(
            padded_input[
                nn, yy * stride_h + ry * dilation_h, xx * stride_w + rx * dilation_w, ff
            ].astype(out_dtype)
            + filter[ry, rx, ff].astype(out_dtype),
            axis=[ry, rx],
        ),
        tag="dilation2d_nhcw",
    )
