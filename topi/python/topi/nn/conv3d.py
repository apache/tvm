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
# pylint: disable=unused-argument, redefined-builtin, no-else-return
"""Conv3D operators"""
from __future__ import absolute_import as _abs
import tvm

from .pad import pad
from .util import get_pad_tuple3d
from ..util import simplify


@tvm.target.generic_func
def conv3d(input, filter, strides, padding, dilation, layout='NCDHW', out_dtype=None):
    """Conv3D operator.

    Parameters
    ----------
    input : tvm.Tensor
        5-D with shape [batch, in_depth, in_channel, in_height, in_width]

    filter : tvm.Tensor
        5-D with shape [num_filter, in_channel, filter_depth, filter_height, filter_width]

    strides : int or a list/tuple of three ints
        stride size, or [stride_depth, stride_height, stride_width]

    padding : int or a list/tuple of three ints
        padding size, or [pad_depth, pad_height, pad_width]

    dilation: int or a list/tuple of three ints
        dilation size, or [dilation_depth, dilation_height, dilation_width]

    layout : str
        layout of data

    Returns
    -------
    output : tvm.Tensor
        5-D with shape [batch, out_depth, out_channel, out_height, out_width]
    """
    # search platform specific declaration first
    # default declaration
    if layout == 'NCDHW':
        return conv3d_ncdhw(input, filter, strides, padding, dilation, out_dtype)
    elif layout == 'NDHWC':
        return conv3d_ndhwc(input, filter, strides, padding, dilation, out_dtype)
    raise ValueError("not support this layout {} yet".format(layout))


def conv3d_ncdhw(Input, Filter, stride, padding, dilation, out_dtype=None):
    """Convolution operator in NCDHW layout.

    Parameters
    ----------
    Input : tvm.Tensor
        5-D with shape [batch, in_channel, in_depth, in_height, in_width]

    Filter : tvm.Tensor
        5-D with shape [num_filter, in_channel, filter_depth, filter_height, filter_width]

    stride : int or a list/tuple of three ints
        Stride size, or [strid_depth, stride_height, stride_width]

    padding : int or str
        Padding size, or ['VALID', 'SAME']

    dilation: int or a list/tuple of three ints
        dilation size, or [dilation_depth, dilation_height, dilation_width]

    Returns
    -------
    Output : tvm.Tensor
        5-D with shape [batch, out_channel, out_depth, out_height, out_width]
    """
    if out_dtype is None:
        out_dtype = Input.dtype
    assert isinstance(stride, int) or len(stride) == 3
    assert isinstance(dilation, int) or len(dilation) == 3
    if isinstance(stride, int):
        stride_d = stride_h = stride_w = stride
    else:
        stride_d, stride_h, stride_w = stride

    if isinstance(dilation, int):
        dilation_d = dilation_h = dilation_w = dilation
    else:
        dilation_d, dilation_h, dilation_w = dilation

    batch, in_channel, in_depth, in_height, in_width = Input.shape
    num_filter, channel, kernel_d, kernel_h, kernel_w = Filter.shape
    # compute the output shape
    dilated_kernel_d = (kernel_d - 1) * dilation_d + 1
    dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
    dilated_kernel_w = (kernel_w - 1) * dilation_w + 1
    pad_front, pad_top, pad_left, pad_back, pad_down, pad_right = get_pad_tuple3d(
        padding, (dilated_kernel_d, dilated_kernel_h, dilated_kernel_w))
    out_channel = num_filter
    out_depth = simplify((in_depth - dilated_kernel_d + pad_front + pad_back) // stride_d + 1)
    out_height = simplify((in_height - dilated_kernel_h + pad_top + pad_down) // stride_h + 1)
    out_width = simplify((in_width - dilated_kernel_w + pad_left + pad_right) // stride_w + 1)
    # compute graph
    pad_before = [0, 0, pad_front, pad_top, pad_left]
    pad_after = [0, 0, pad_back, pad_down, pad_right]
    temp = pad(Input, pad_before, pad_after, name="pad_temp")
    rc = tvm.reduce_axis((0, in_channel), name='rc')
    rz = tvm.reduce_axis((0, kernel_d), name='rz')
    ry = tvm.reduce_axis((0, kernel_h), name='ry')
    rx = tvm.reduce_axis((0, kernel_w), name='rx')

    return tvm.compute(
        (batch, out_channel, out_depth, out_height, out_width),
        lambda nn, ff, zz, yy, xx: tvm.sum(
            temp[nn, rc, zz * stride_d + rz * dilation_d, yy * stride_h + ry * dilation_h,
                 xx * stride_w + rx * dilation_w].astype(out_dtype) *
            Filter[ff, rc, rz, ry, rx].astype(out_dtype),
            axis=[rc, rz, ry, rx]), tag="conv3d_ncdhw")


def conv3d_ndhwc(Input, Filter, stride, padding, dilation, out_dtype='float32'):
    """Convolution operator in NDHWC layout.

    Parameters
    ----------
    Input : tvm.Tensor
        5-D with shape [batch, in_channel, in_depth, in_height, in_width]

    Filter : tvm.Tensor
        5-D with shape [num_filter, in_channel, filter_depth, filter_height, filter_width]

    stride : int or a list/tuple of three ints
        Stride size, or [strid_depth, stride_height, stride_width]

    padding : int or str
        Padding size, or ['VALID', 'SAME']

    dilation: int or a list/tuple of three ints
        dilation size, or [dilation_depth, dilation_height, dilation_width]

    Returns
    -------
    Output : tvm.Tensor
        5-D with shape [batch, out_channel, out_depth, out_height, out_width]
    """
    assert isinstance(stride, int) or len(stride) == 3
    assert isinstance(dilation, int) or len(dilation) == 3

    if isinstance(stride, int):
        stride_d = stride_h = stride_w = stride
    else:
        stride_d, stride_h, stride_w = stride

    if isinstance(dilation, int):
        dilation_d = dilation_h = dilation_w = dilation
    else:
        dilation_d, dilation_h, dilation_w = dilation

    batch, in_depth, in_height, in_width, in_channel = Input.shape
    kernel_d, kernel_h, kernel_w, channel, num_filter = Filter.shape
    # compute the output shape
    dilated_kernel_d = (kernel_d - 1) * dilation_d + 1
    dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
    dilated_kernel_w = (kernel_w - 1) * dilation_w + 1

    pad_front, pad_top, pad_left, pad_back, pad_down, pad_right = get_pad_tuple3d(
        padding, (dilated_kernel_d, dilated_kernel_h, dilated_kernel_w))
    out_channel = num_filter
    out_depth = simplify((in_depth - dilated_kernel_d + pad_front + pad_back) // stride_d + 1)
    out_height = simplify((in_height - dilated_kernel_h + pad_top + pad_down) // stride_h + 1)
    out_width = simplify((in_width - dilated_kernel_w + pad_left + pad_right) // stride_w + 1)
    pad_before = [0, pad_front, pad_top, pad_left, 0]
    pad_after = [0, pad_back, pad_down, pad_right, 0]
    PaddedInput = pad(Input, pad_before, pad_after, name="PaddedInput")
    rc = tvm.reduce_axis((0, in_channel), name='rc')
    rz = tvm.reduce_axis((0, kernel_d), name='rz')
    ry = tvm.reduce_axis((0, kernel_h), name='ry')
    rx = tvm.reduce_axis((0, kernel_w), name='rx')
    Output = tvm.compute(
        (batch, out_depth, out_height, out_width, out_channel),
        lambda nn, zz, yy, xx, ff: tvm.sum(
            PaddedInput[nn, zz * stride_d + rz * dilation_d, yy * stride_h + ry * dilation_h,
                        xx * stride_w + rx * dilation_w, rc].astype(out_dtype) *
            Filter[rz, ry, rx, rc, ff].astype(out_dtype), axis=[rz, ry, rx, rc]),
        name="Conv3dOutput", tag="conv3d_ndhwc")
    return Output
