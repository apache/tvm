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
"""Transposed 1D convolution operators (sometimes called Deconvolution)."""
from tvm import te
from .dilate import dilate
from .pad import pad
from ..utils import simplify
from .utils import get_pad_tuple1d


def _conv1d_transpose_ncw_preprocess(data, kernel, stride, padding, out_dtype, output_padding):
    """Preprocess data and kernel to make the compute pattern
    of conv1d_transpose the same as conv1d.

    Parameters
    ----------
    data : tvm.te.Tensor
        3-D with shape [batch, in_channel, in_width]

    kernel : tvm.te.Tensor
        3-D with shape [in_channel, num_filter, filter_width]

    stride : ints
        The spatial stride along width

    padding : int or str
        Padding size, or ['VALID', 'SAME']

    out_dtype : str
        The output data type. This is used for mixed precision.

    output_padding : ints
        Used to recover the actual output shape in case there are more
        than one possible shape.  Must be smaller than stride.

    Returns
    -------
    data_pad : tvm.te.Tensor
        Padded input data. 3-D with shape [batch, in_channel, in_width]

    kernel: tvm.te.Tensor
        Transformed kernel. 3-D with shape [num_filter, in_channel, filter_width]
    """
    # some pre-processing and prelimnary checks
    if out_dtype is None:
        out_dtype = data.dtype

    # dilate and pad
    if isinstance(stride, (tuple, list)):
        stride = stride[0]
    if isinstance(output_padding, (tuple, list)):
        output_padding = output_padding[0]

    _, channels_in, _ = data.shape
    _, channels_out, kernel_width = kernel.shape
    assert output_padding < stride
    channels_out = simplify(channels_out)
    data_dilate = dilate(data, [1, 1, stride], name="data_dilate")
    pad_left, pad_right = get_pad_tuple1d(padding, (kernel_width,))
    pad_left = kernel_width - 1 - pad_left
    pad_right = kernel_width - 1 - pad_right + output_padding
    data_pad = pad(data_dilate, [0, 0, pad_left], [0, 0, pad_right], name="data_pad")

    # transform kernel layout from IOW to OIW, and rotate kernel by 180 degrees
    kernel = te.compute(
        (channels_out, channels_in, kernel_width),
        lambda o, i, w: kernel[i][o][kernel_width - 1 - w],
        name="kernel",
    )
    return data_pad, kernel


def conv1d_transpose_ncw(data, kernel, stride, padding, out_dtype, output_padding):
    """Transposed 1D convolution ncw forward operator.

    Parameters
    ----------
    data : tvm.te.Tensor
        3-D with shape [batch, in_channel, in_width]

    kernel : tvm.te.Tensor
        3-D with shape [in_channel, num_filter, filter_width]

    stride : ints
        The spatial stride along width

    padding : int or str
        Padding size, or ['VALID', 'SAME']

    out_dtype : str
        The output data type. This is used for mixed precision.

    output_padding : ints
        Used to recover the actual output shape in case there are more
        than one possible shape.  Must be smaller than stride.

    Returns
    -------
    output : tvm.te.Tensor
        3-D with shape [batch, out_channel, out_width]

    """

    batch, channels_in, _ = data.shape
    _, channels_out, kernel_width = kernel.shape

    data_pad, transformed_kernel = _conv1d_transpose_ncw_preprocess(
        data, kernel, stride, padding, out_dtype, output_padding
    )

    # convolution
    _, _, data_width = data_pad.shape
    out_w = simplify(data_width - kernel_width + 1)
    dc = te.reduce_axis((0, channels_in), name="dc")
    dw = te.reduce_axis((0, kernel_width), name="dw")
    output = te.compute(
        (batch, channels_out, out_w),
        lambda b, c, w: te.sum(
            data_pad[b, dc, w + dw].astype(out_dtype)
            * transformed_kernel[c, dc, dw].astype(out_dtype),
            axis=[dc, dw],
        ),
        tag="conv1d_transpose_ncw",
    )

    return output


def group_conv1d_transpose_ncw(data, kernel, stride, padding, out_dtype, output_padding, groups):
    """Transposed 1D group convolution ncw forward operator.

    Parameters
    ----------
    data : tvm.te.Tensor
        3-D with shape [batch, in_channel, in_width]

    kernel : tvm.te.Tensor
        3-D with shape [in_channel, num_filter, filter_width]

    stride : ints
        The spatial stride along width

    padding : int or str
        Padding size, or ['VALID', 'SAME']

    out_dtype : str
        The output data type. This is used for mixed precision.

    output_padding : ints
        Used to recover the actual output shape in case there are more
        than one possible shape.  Must be smaller than stride.

     groups : int
        number of groups

    Returns
    -------
    output : tvm.te.Tensor
        3-D with shape [batch, out_channel, out_width]

    """
    if groups == 1:
        return conv1d_transpose_ncw(data, kernel, stride, padding, out_dtype, output_padding)

    _, in_channels, _ = data.shape

    assert (
        in_channels % groups == 0
    ), f"input channels {in_channels} must divide group size {groups}"

    data_pad, transformed_kernel = _conv1d_transpose_ncw_preprocess(
        data, kernel, stride, padding, out_dtype, output_padding
    )

    batch, in_channels, in_w = data_pad.shape
    out_c, _, filter_w = transformed_kernel.shape

    # convolution stage
    out_channels = simplify(out_c * groups)
    out_w = simplify(in_w - filter_w + 1)
    dc = te.reduce_axis((0, in_channels // groups), name="dc")
    dw = te.reduce_axis((0, filter_w), name="dw")

    # data: batch, in_channels, out_w
    # weight: out_channels // G, in_channels, out_w
    return te.compute(
        (batch, out_channels, out_w),
        lambda b, c, w: te.sum(
            data_pad[
                b, c // (out_channels // groups) * (in_channels // groups) + dc, w + dw
            ].astype(out_dtype)
            * transformed_kernel[
                c % (out_channels // groups),
                c // (out_channels // groups) * (in_channels // groups) + dc,
                dw,
            ].astype(out_dtype),
            axis=[dc, dw],
        ),
        tag="group_conv1d_transpose_ncw",
    )
