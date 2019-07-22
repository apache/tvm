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
"""Transposed 2D convolution operators (sometimes called Deconvolution)."""
from __future__ import absolute_import as _abs
import tvm

from .dilate import dilate
from .pad import pad
from .util import get_pad_tuple
from ..util import simplify


@tvm.target.generic_func
def conv2d_transpose_nchw(Input, Filter, strides, padding, out_dtype):
    """Transposed 2D convolution nchw forward operator.

    Parameters
    ----------
    Input : tvm.Tensor
        4-D with shape [batch, in_channel, in_height, in_width]

    Filter : tvm.Tensor
        4-D with shape [in_channel, num_filter, filter_height, filter_width]

    strides : tuple of two ints
        The spatial stride along height and width

    padding : int or str
        Padding size, or ['VALID', 'SAME']

    out_dtype : str
        The output data type. This is used for mixed precision.

    Returns
    -------
    Output : tvm.Tensor
        4-D with shape [batch, out_channel, out_height, out_width]
    """
    return declaration_conv2d_transpose_impl(Input, Filter, strides, padding, out_dtype)

def declaration_conv2d_transpose_impl(data, kernel, strides, padding, out_dtype):
    """Implementation of conv2d transpose"""
    batch, in_c, in_h, in_w = data.shape
    _, out_c, filter_h, filter_w = kernel.shape
    stride_h, stride_w = strides
    # dilate stage
    DilatedInput = dilate(data, [1, 1, stride_h, stride_w], name='DilatedInput')
    # padding stage
    fpad_top, fpad_left, fpad_bottom, fpad_right = get_pad_tuple(padding, (filter_h, filter_w))
    bpad_top = filter_h - 1 - fpad_top
    bpad_bottom = filter_h - 1 - fpad_bottom
    bpad_left = filter_w - 1 - fpad_left
    bpad_right = filter_w - 1 - fpad_right
    PaddedInput = pad(DilatedInput, \
                        [0, 0, bpad_top, bpad_left], \
                        [0, 0, bpad_bottom, bpad_right], \
                        name='PaddedInput')
    # convolution stage
    out_c = simplify(out_c)
    out_h = simplify((in_h - 1) * stride_h - fpad_top - fpad_bottom + filter_h)
    out_w = simplify((in_w - 1) * stride_w - fpad_left - fpad_right + filter_w)
    dc = tvm.reduce_axis((0, in_c), name='dc')
    dh = tvm.reduce_axis((0, filter_h), name='dh')
    dw = tvm.reduce_axis((0, filter_w), name='dw')

    Output = tvm.compute(
        (batch, out_c, out_h, out_w),
        lambda b, c, h, w: tvm.sum(
            PaddedInput[b, dc, h+dh, w+dw].astype(out_dtype) *
            kernel[dc, c, filter_h-1-dh, filter_w-1-dw].astype(out_dtype),
            axis=[dc, dh, dw]), tag="conv2d_transpose_nchw")

    return Output
