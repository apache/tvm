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
from .conv2d import conv


def conv1d(
    data,
    kernel,
    strides=1,
    padding="VALID",
    dilation=1,
    data_layout="NCW",
    kernel_layout="",
    out_dtype=None,
):
    """1D convolution forward operator.

    Parameters
    ----------
    data : tvm.te.Tensor
        3-D input shape [batch, in_channel, in_width] for data_layout == 'NCW'
        and [batch, in_width, in_channel] for data_layout == 'NWC'

    kernel : tvm.te.Tensor
        3-D kernel with shape [num_filter, in_channel, filter_size] for kernel_layout == 'OIW'
        and [filter_size, in_channel, num_filter] for kernel_layout == 'WIO'

    strides : int or tuple
        The spatial stride along width

    padding : int or str
        Padding size, or ['VALID', 'SAME']

    dilation : int or tuple
        Dilation rate if convolution should be dilated.

    data_layout : str
        How input data is laid out, must be one of ['NCW', 'NWC']

    kernel_layout: Optiona[str]
        The layout of the kernel. If unspecified, use default layout. "OIW" if data_layout == "NCW",
        "WIO" if data_layout == "NWC".

    out_dtype : str
        The output data type. If None then output is same type as input.
    """
    return conv(data, kernel, strides, padding, dilation, 1, data_layout, kernel_layout, out_dtype)


def conv1d_nwc(data, kernel, strides=1, padding="VALID", dilation=1, out_dtype=None):
    """1D convolution in NWC layout. See :py:func:`conv` for details on parameters"""
    return conv(data, kernel, strides, padding, dilation, 1, "NWC", "WIO", out_dtype=out_dtype)


def conv1d_ncw(data, kernel, strides=1, padding="VALID", dilation=1, out_dtype=None):
    """1D convolution in NCW layout. See :py:func:`conv` for details on parameters"""
    return conv(data, kernel, strides, padding, dilation, 1, "NCW", "OIW", out_dtype=out_dtype)


def group_conv1d_nwc(
    data, kernel, strides=1, padding="VALID", dilation=1, groups=1, out_dtype=None
):
    """1D convolution forward operator for NWC layout.

    Parameters
    ----------
    data : tvm.te.Tensor
        3-D with shape [batch, in_width, in_channel]

    kernel : tvm.te.Tensor
        3-D with shape [filter_size, in_channel, num_filter]

    strides : int or tuple
        The spatial stride along width

    padding : int, tuple, or str
        Padding size can be an integer for equal padding,
        a tuple of (left, right) or a string in ['VALID', 'SAME'].

    dilation : int or tuple
        Dilation rate if convolution should be dilated.

    groups : int
        Number of groups

    out_dtype : str
        The output data type. If None then output is same type as input.
    """
    return conv(data, kernel, strides, padding, dilation, groups, "NWC", "WIO", out_dtype=out_dtype)


def group_conv1d_ncw(
    data, kernel, strides=1, padding="VALID", dilation=1, groups=1, out_dtype=None
):
    """1D convolution forward operator for NCW layout.

    Parameters
    ----------
    data : tvm.te.Tensor
        3-D with shape [batch, in_channel, in_width]

    kernel : tvm.te.Tensor
        3-D with shape [num_filter, in_channel, filter_size]

    strides : int or tuple
        The spatial stride along width

    padding : int, tuple, or str
        Padding size can be an integer for equal padding,
        a tuple of (left, right) or a string in ['VALID', 'SAME'].

    dilation : int or tuple
        Dilation rate if convolution should be dilated.

    groups : int
        Number of groups

    out_dtype : str
        The output data type. If None then output is same type as input.
    """
    return conv(data, kernel, strides, padding, dilation, groups, "NCW", "OIW", out_dtype=out_dtype)
