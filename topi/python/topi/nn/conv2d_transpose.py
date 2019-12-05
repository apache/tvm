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
# pylint: disable=invalid-name, unused-variable, unused-argument
"""Transposed 2D convolution operators (sometimes called Deconvolution)."""
from __future__ import absolute_import as _abs
import tvm
from tvm import relay
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


def conv2d_transpose_nchw_preprocess(data, kernel, strides, padding, out_dtype):
    """Preprocess data and kernel to make the compute pattern
       of conv2d_transpose the same as conv2d"""
    batch, in_c, in_h, in_w = data.shape
    _, out_c, filter_h, filter_w = kernel.shape
    stride_h, stride_w = strides
    # dilate data
    data_dilate = dilate(data, [1, 1, stride_h, stride_w], name='data_dilate')
    # pad data
    fpad_top, fpad_left, fpad_bottom, fpad_right = get_pad_tuple(padding, (filter_h, filter_w))
    bpad_top = filter_h - 1 - fpad_top
    bpad_bottom = filter_h - 1 - fpad_bottom
    bpad_left = filter_w - 1 - fpad_left
    bpad_right = filter_w - 1 - fpad_right
    data_pad = pad(data_dilate, \
                   [0, 0, bpad_top, bpad_left], \
                   [0, 0, bpad_bottom, bpad_right], \
                   name='data_pad')
    # transform kernel layout from IOHW to OIHW, and rotate kernel by 180 degrees
    kernel_transform = tvm.compute((out_c, in_c, filter_h, filter_w), \
                                    lambda o, i, h, w: kernel[i][o][filter_h-1-h][filter_w-1-w], \
                                    name='kernel_transform')
    return data_pad, kernel_transform


def declaration_conv2d_transpose_impl(data, kernel, strides, padding, out_dtype):
    """Implementation of conv2d transpose"""
    data_pad, kernel_transform = \
        conv2d_transpose_nchw_preprocess(data, kernel, strides, padding, out_dtype)
    batch, in_c, in_h, in_w = data_pad.shape
    out_c, _, filter_h, filter_w = kernel_transform.shape
    stride_h, stride_w = strides

    # convolution stage
    out_c = simplify(out_c)
    out_h = simplify(in_h - filter_h + 1)
    out_w = simplify(in_w - filter_w + 1)
    dc = tvm.reduce_axis((0, in_c), name='dc')
    dh = tvm.reduce_axis((0, filter_h), name='dh')
    dw = tvm.reduce_axis((0, filter_w), name='dw')

    Output = tvm.compute(
        (batch, out_c, out_h, out_w),
        lambda b, c, h, w: tvm.sum(
            data_pad[b, dc, h+dh, w+dw].astype(out_dtype) *
            kernel_transform[c, dc, dh, dw].astype(out_dtype),
            axis=[dc, dh, dw]), tag="conv2d_transpose_nchw")

    return Output


@tvm.target.generic_func
def conv2d_transpose_legalize(attrs, inputs, types):
    """Legalizes Transposed 2D convolution op.

    Parameters
    ----------
    attrs : tvm.attrs.Attrs
        Attributes of current Transposed 2D convolution
    inputs : list of tvm.relay.Expr
        The args of the Relay expr to be legalized
    types : list of types
        List of input and output types

    Returns
    -------
    result : tvm.relay.Expr
        The legalized expr
    """
    if attrs['data_layout'] == 'NHWC':
        data, kernel = inputs
        kernel_layout = attrs['kernel_layout']
        # Convert Kernel layout to IOHW
        # kernel_layout is different from input kernel layout - IO is swapped
        if kernel_layout == 'HWIO':
            # input kernel layout is swapped to HWOI
            # output kernel layout will be IOHW
            kernel = relay.transpose(kernel, axes=(3, 2, 0, 1))
        elif kernel_layout == 'HWOI':
            # input kernel layout is swapped to HWIO
            # output kernel layout will be IOHW
            kernel = relay.transpose(kernel, axes=(2, 3, 0, 1))
        elif kernel_layout == 'IOHW':
            # input kernel layout is swapped to OIHW
            # output kernel layout will be IOHW
            kernel = relay.transpose(kernel, axes=(1, 0, 2, 3))
        elif kernel_layout == 'OIHW':
            # input kernel layout is swapped to IOHW
            # output kernel layout will be IOHW
            pass
        else:
            # Skip legalize. Let relay.nn.conv2d_transpose to handle the case
            return None

        # Set new attrs for conv2d_transpose.
        new_attrs = {k: attrs[k] for k in attrs.keys()}
        new_attrs['data_layout'] = 'NCHW'
        # layout of kernel should be IOHW, but kernel_layout should be swapped - OIHW
        new_attrs['kernel_layout'] = 'OIHW'

        # Convert data to NCHW.
        data = relay.transpose(data, axes=(0, 3, 1, 2))
        deconv = relay.nn.conv2d_transpose(data, kernel, **new_attrs)
        # Convert back to original NHWC layout.
        out = relay.transpose(deconv, axes=(0, 2, 3, 1))
        return out

    return None
