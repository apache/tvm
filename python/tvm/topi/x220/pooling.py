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
# pylint: disable=invalid-name, unused-variable
"""Schedule for pooling operators"""
import tvm
from tvm import te

from .dilate import dilate
from .pad import pad
from .utils import get_pad_tuple
from ..utils import simplify, get_const_tuple

from math import prod


def max_pool2d_nchw(Input, Filter, stride, dilation, padding, pool_type, out_dtype=None):
    """Max pool2d nchw forward operator.

    Parameters
    ----------
    Input : tvm.te.Tensor
        4-D with shape [batch, in_channel, in_height, in_width]

    Filter : int or a list/tuple of two ints
        filter_size, or [filter_height, filter_width]

    stride : int or a list/tuple of two ints
        The spatial stride, or (stride_height, stride_width).

    dilation: int or a list/tuple of two ints
        dilation size, or [dilation_height, dilation_width]

    padding : int or str
        Padding size, or ['VALID', 'SAME']

    Returns
    -------
    Output : tvm.te.Tensor
        4-D with shape [batch, out_channel, out_height, out_width]
    """
    out_dtype = Input.dtype if out_dtype is None else out_dtype

    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride

    if isinstance(dilation, int):
        dilation_h = dilation_w = dilation
    else:
        dilation_h, dilation_w = dilation

    if isinstance(Filter, int):
        filter_h = filter_w = Filter
    else:
        filter_h, filter_w = Filter

    batch, in_channel, in_height, in_width = Input.shape

    dilated_kernel_h = (filter_h - 1) * dilation_h + 1
    dilated_kernel_w = (filter_w - 1) * dilation_w + 1
    pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
        padding, (dilated_kernel_h, dilated_kernel_w)
    )
    out_channel = simplify(in_channel)
    out_height = simplify((in_height - dilated_kernel_h + pad_top + pad_down) // stride_h + 1)
    out_width = simplify((in_width - dilated_kernel_w + pad_left + pad_right) // stride_w + 1)

    # maxpool stage
    di = te.reduce_axis((0, filter_h), name="di")
    dj = te.reduce_axis((0, filter_w), name="dj")

    Input = te.compute(
        Input.shape,
        lambda b, ic, ih, iw:
        Input[b, ic, ih, iw], name="Input", tag="max_pool2d_nchw",
    )
    Acc = te.compute(
        (batch, out_channel, out_height, out_width),
        lambda b, c, i, j:
        te.max(
            (Input[b, c, i * stride_h + di * dilation_h, j * stride_w + dj * dilation_w,].astype(out_dtype)
            ), axis=[di, dj],
        ), name="Acc", tag="max_pool2d_nchw",
    )
    Output = te.compute(
        (batch, out_channel, out_height, out_width),
        lambda b, c, i, j:
            Acc[b, c, i, j]
        , name="Output", tag="max_pool2d_nchw",
    )
    return Output


def avg_pool2d_nchw(Input, Filter, stride, dilation, padding, pool_type, out_dtype=None):
    """Avg pool2d nchw forward operator.

    Parameters
    ----------
    Input : tvm.te.Tensor
        4-D with shape [batch, in_channel, in_height, in_width]

    Filter : int or a list/tuple of two ints
        filter_size, or [filter_height, filter_width]

    stride : int or a list/tuple of two ints
        The spatial stride, or (stride_height, stride_width).

    dilation: int or a list/tuple of two ints
        dilation size, or [dilation_height, dilation_width]

    padding : int or str
        Padding size, or ['VALID', 'SAME']

    Returns
    -------
    Output : tvm.te.Tensor
        4-D with shape [batch, out_channel, out_height, out_width]
    """
    out_dtype = Input.dtype if out_dtype is None else out_dtype

    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride

    if isinstance(dilation, int):
        dilation_h = dilation_w = dilation
    else:
        dilation_h, dilation_w = dilation

    if isinstance(Filter, int):
        filter_h = filter_w = Filter
    else:
        filter_h, filter_w = Filter

    batch, in_channel, in_height, in_width = Input.shape

    dilated_kernel_h = (filter_h - 1) * dilation_h + 1
    dilated_kernel_w = (filter_w - 1) * dilation_w + 1
    pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
        padding, (dilated_kernel_h, dilated_kernel_w)
    )
    out_channel = simplify(in_channel)
    out_height = simplify((in_height - dilated_kernel_h + pad_top + pad_down) // stride_h + 1)
    out_width = simplify((in_width - dilated_kernel_w + pad_left + pad_right) // stride_w + 1)

    # avgpool stage
    di = te.reduce_axis((0, filter_h), name="di")
    dj = te.reduce_axis((0, filter_w), name="dj")

    multiplier = tvm.te.const(1, "float")
    Multiplier = tvm.te.compute((batch, ), lambda b: multiplier, name="Multiplier")
    Input = te.compute(
        Input.shape,
        lambda b, ic, ih, iw:
        Input[b, ic, ih, iw], name="Input", tag="avg_pool2d_nchw",
    )
    Acc = te.compute(
        (batch, out_channel, out_height, out_width),
        lambda b, c, i, j:
        te.sum(
            (Input[b, c, i * stride_h + di * dilation_h, j * stride_w + dj * dilation_w,].astype(out_dtype)
                * Multiplier[b].astype(out_dtype)
            ), axis=[di, dj],
        ), name="Acc", tag="avg_pool2d_nchw",
    )
    Output = te.compute(
        (batch, out_channel, out_height, out_width),
        lambda b, c, i, j:
        (Acc[b, c, i, j].astype("int")>>(dilated_kernel_h*dilated_kernel_w)).astype(out_dtype)
        , name="Output", tag="avg_pool2d_nchw",
    )
    return Output


