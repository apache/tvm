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
"""1D convolution operators."""
from __future__ import absolute_import as _abs
import tvm
from .pad import pad
from ..util import simplify
from .util import get_pad_tuple1d


def conv1d(data, kernel, layout='NCW', stride=1, padding='VALID', pad_method='SYMMETRIC', dilation=1, out_dtype=None):
    """ 1D convolution forward operator.

    Parameters
    ----------
    data : tvm.Tensor
        3-D input shape [batch, in_channel, in_width] for layout == 'NCW'
        and [batch, in_width, in_channel] for layout == 'NWC'

    kernel : tvm.Tensor
        3-D kernel with shape [num_filter, in_channel, filter_size] for layout == 'NCW'
        and [filter_size, in_channel, num_filter] for layout == 'NWC'

    layout : str
        How input data is laid out, must be one of ['NCW', 'NWC']

    stride : int
        The spatial stride along width

    padding : int or str
        Padding size, or ['VALID', 'SAME']

    pad_method : str
        How to add padding, must be one of ['SYMMETRIC', 'BEFORE', 'AFTER']
        Useful when dealing with Tensorflow and Keras funkiness.

    dilation : int
        Dilation rate if convolution should be dilated. 

    out_dtype : str
        The output data type. If None then output is same type as input.
    """
    if layout == 'NCW':
        return conv1d_ncw(data, kernel, stride, padding, pad_method, dilation, out_dtype)
    elif layout == 'NWC':
        return conv1d_nwc(data, kernel, stride, padding, pad_method, dilation, out_dtype)
    raise ValueError("This layout is not yet supported: {}".format(layout))


def conv1d_ncw(data, kernel, stride=1, padding='VALID', pad_method='SYMMETRIC', dilation=1, out_dtype=None):
    """ 1D convolution forward operator for NCW layout.

    Parameters
    ----------
    data : tvm.Tensor
        3-D with shape [batch, in_channel, in_width]

    kernel : tvm.Tensor
        3-D with shape [num_filter, in_channel, filter_size]

    stride : int
        The spatial stride along width

    padding : int or str
        Padding size, or ['VALID', 'SAME']

    pad_method : str
        How to add padding, must be one of ['SYMMETRIC', 'BEFORE', 'AFTER']

    dilation : int
        Dilation rate if convolution should be dilated. 

    out_dtype : str
        The output data type. If None then output is same type as input.
    """
    if out_dtype == None:
        out_dtype = data.dtype
    # Dilate and pad
    if isinstance(stride, (tuple, list)):
        stride = stride[0]
    if isinstance(dilation, (tuple, list)):
        dilation = dilation[0]
    batch, in_channels, data_width = data.shape
    out_channels, _, kernel_size = kernel.shape

    # Compute the output shape
    dilated_kernel_size = (kernel_size - 1) * dilation + 1
    pad_left, pad_right = get_pad_tuple1d(padding, (dilated_kernel_size,))
    out_channels = simplify(out_channels)
    out_width = simplify(
        (data_width - dilated_kernel_size + pad_left + pad_right) // stride + 1)

    # Apply selected padding
    if pad_method == 'SYMMETRIC':
        pad_before = [0, 0, pad_left]
        pad_after = [0, 0, pad_right]
    elif pad_method == 'BEFORE':
        pad_before = [0, 0, pad_right + pad_left]
        pad_after = [0, 0, 0]
    else:
        pad_before = [0, 0, 0]
        pad_after = [0, 0, pad_right + pad_left]
    temp = pad(data, pad_before, pad_after, name='pad_temp')

    # Compute graph
    rc = tvm.reduce_axis((0, in_channels), name='rc')
    rw = tvm.reduce_axis((0, kernel_size), name='rw')

    return tvm.compute(
        (batch, out_channels, out_width),
        lambda b, c, w: tvm.sum(
            temp[b, rc, w * stride + rw * dilation].astype(out_dtype) *
            kernel[c, rc, rw].astype(out_dtype),
            axis=[rc, rw]), tag="conv1d_ncw")


def conv1d_nwc(data, kernel, stride=1, padding='VALID', pad_method='SYMMETRIC', dilation=1, out_dtype=None):
    """ 1D convolution forward operator for NWC layout.

    Parameters
    ----------
    data : tvm.Tensor
        3-D with shape [batch, in_width, in_channel]

    kernel : tvm.Tensor
        3-D with shape [filter_size, in_channel, num_filter]

    stride : int
        The spatial stride along width

    padding : int or str
        Padding size, or ['VALID', 'SAME']

    pad_method : str
        How to add padding, must be one of ['SYMMETRIC', 'BEFORE', 'AFTER']

    dilation : int
        Dilation rate if convolution should be dilated. 

    out_dtype : str
        The output data type. If None then output is same type as input.
    """
    if out_dtype == None:
        out_dtype = data.dtype
    # Dilate and pad
    if isinstance(stride, (tuple, list)):
        stride = stride[0]
    if isinstance(dilation, (tuple, list)):
        dilation = dilation[0]
    batch, data_width, in_channels = data.shape
    kernel_size, _, out_channels = kernel.shape

    # Compute the output shape
    dilated_kernel_size = (kernel_size - 1) * dilation + 1
    pad_left, pad_right = get_pad_tuple1d(padding, (dilated_kernel_size,))
    out_channels = simplify(out_channels)
    out_width = simplify(
        (data_width - dilated_kernel_size + pad_left + pad_right) // stride + 1)

    # Apply selected padding
    if pad_method == 'SYMMETRIC':
        pad_before = [0, pad_left, 0]
        pad_after = [0, pad_right, 0]
    elif pad_method == 'BEFORE':
        pad_before = [0, pad_right + pad_left, 0]
        pad_after = [0, 0, 0]
    else:
        pad_before = [0, 0, 0]
        pad_after = [0, pad_right + pad_left, 0]
    temp = pad(data, pad_before, pad_after, name='pad_temp')

    # Compute graph
    rc = tvm.reduce_axis((0, in_channels), name='rc')
    rw = tvm.reduce_axis((0, kernel_size), name='rw')

    return tvm.compute(
        (batch, out_width, out_channels),
        lambda b, w, c: tvm.sum(
            temp[b, w * stride + rw * dilation, rc].astype(out_dtype) *
            kernel[rw, rc, c].astype(out_dtype),
            axis=[rc, rw]), tag="conv1d_nwc")
