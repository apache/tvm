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
import tvm
from tvm import te
from .pad import pad
from ..utils import simplify
from .utils import get_pad_tuple1d
from .conv2d import conv


def conv1d(data, kernel, strides=1, padding="VALID", dilation=1, layout="NCW", out_dtype=None):
    """1D convolution forward operator.

    Parameters
    ----------
    data : tvm.te.Tensor
        3-D input shape [batch, in_channel, in_width] for layout == 'NCW'
        and [batch, in_width, in_channel] for layout == 'NWC'

    kernel : tvm.te.Tensor
        3-D kernel with shape [num_filter, in_channel, filter_size] for layout == 'NCW'
        and [filter_size, in_channel, num_filter] for layout == 'NWC'

    strides : int or tuple
        The spatial stride along width

    padding : int or str
        Padding size, or ['VALID', 'SAME']

    dilation : int or tuple
        Dilation rate if convolution should be dilated.

    layout : str
        How input data is laid out, must be one of ['NCW', 'NWC']

    out_dtype : str
        The output data type. If None then output is same type as input.
    """
    return conv(data, kernel, strides, padding, dilation, 1, layout, out_dtype)


def conv1d_nwc(data, kernel, strides=1, padding="VALID", dilation=1, out_dtype=None):
    """1D convolution in NWC layout. See :py:func:`conv` for details on parameters"""
    return conv(data, kernel, strides, padding, dilation, 1, "NWC", out_dtype=out_dtype)


def conv1d_ncw(data, kernel, strides=1, padding="VALID", dilation=1, out_dtype=None):
    """1D convolution in NCW layout. See :py:func:`conv` for details on parameters"""
    return conv(data, kernel, strides, padding, dilation, 1, "NCW", out_dtype=out_dtype)


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
    return conv(data, kernel, strides, padding, dilation, groups, "NWC", out_dtype=out_dtype)


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
    return conv(data, kernel, strides, padding, dilation, groups, "NCW", out_dtype=out_dtype)


def conv1d_nwc_owi(data, kernel, strides=1, padding="VALID", dilation=1, out_dtype=None):
    """1D convolution forward operator for NWC layout.

    Parameters
    ----------
    data : tvm.te.Tensor
        3-D with shape [batch, in_width, in_channel]

    kernel : tvm.te.Tensor
        3-D with shape [num_filter, filter_size, in_channel]

    strides : int or tuple
        The spatial stride along width

    padding : int, tuple, or str
        Padding size can be an integer for equal padding,
        a tuple of (left, right) or a string in ['VALID', 'SAME'].

    dilation : int or tuple
        Dilation rate if convolution should be dilated.

    out_dtype : str
        The output data type. If None then output is same type as input.
    """
    if out_dtype is None:
        out_dtype = data.dtype
    if isinstance(strides, (tuple, list)):
        strides = strides[0]
    if isinstance(dilation, (tuple, list)):
        dilation = dilation[0]

    batch, data_width, in_channels = data.shape
    out_channels, kernel_size, _ = kernel.shape

    # Compute the output shape
    dilated_kernel_size = (kernel_size - 1) * dilation + 1
    pad_left, pad_right = get_pad_tuple1d(padding, (dilated_kernel_size,))
    out_channels = simplify(out_channels)
    out_width = simplify((data_width - dilated_kernel_size + pad_left + pad_right) // strides + 1)

    # Apply padding
    pad_before = [0, pad_left, 0]
    pad_after = [0, pad_right, 0]
    temp = pad(data, pad_before, pad_after, name="pad_temp")

    # Compute graph
    rc = te.reduce_axis((0, in_channels), name="rc")
    rw = te.reduce_axis((0, kernel_size), name="rw")

    return te.compute(
        (batch, out_width, out_channels),
        lambda b, w, c: te.sum(
            temp[b, w * strides + rw * dilation, rc].astype(out_dtype)
            * kernel[c, rw, rc].astype(out_dtype),
            axis=[rw, rc],
        ),
        tag="conv1d_nwc_owi",
    )


@tvm.target.generic_func
def conv1d_legalize(attrs, inputs, types):
    """Legalizes Conv1D op.

    Parameters
    ----------
    attrs : tvm.ir.Attrs
        Attributes of current convolution
    inputs : list of tvm.relay.Expr
        The args of the Relay expr to be legalized
    types : list of types
        List of input and output types

    Returns
    -------
    result : tvm.relay.Expr
        The legalized expr
    """
    # not to change by default
    return None
