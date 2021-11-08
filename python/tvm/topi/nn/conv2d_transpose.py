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
import collections

import tvm
from tvm import te
from tvm import relay
from .dilate import dilate
from .pad import pad
from .utils import get_pad_tuple
from ..utils import simplify

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            assert len(x) == n, f"Input can only have {n} elements, but got {len(x)} instead: {x}."
            return x
        return tuple(repeat(x, n))
    return parse

_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)
_quadruple = _ntuple(4)

def conv2d_transpose_nchw(Input, Filter, strides, padding, out_dtype, output_padding):
    """Transposed 2D convolution nchw forward operator.

    Parameters
    ----------
    Input : tvm.te.Tensor
        4-D with shape [batch, in_channel, in_height, in_width]

    Filter : tvm.te.Tensor
        4-D with shape [in_channel, num_filter, filter_height, filter_width]

    strides : tuple of two ints
        The spatial stride along height and width

    padding : int or str
        Padding size, or ['VALID', 'SAME']

    out_dtype : str
        The output data type. This is used for mixed precision.

    output_padding : tuple of ints
        Used to get the right output shape for gradients

    Returns
    -------
    Output : tvm.te.Tensor
        4-D with shape [batch, out_channel, out_height, out_width]
    """
    return declaration_conv2d_transpose_impl(
        Input, Filter, strides, padding, out_dtype, output_padding=output_padding
    )


def conv2d_transpose_nchw_preprocess(data, kernel, strides, padding, out_dtype, output_padding):
    """Preprocess data and kernel to make the compute pattern
    of conv2d_transpose the same as conv2d"""
    batch, in_c, in_h, in_w = data.shape
    _, out_c, filter_h, filter_w = kernel.shape
    stride_h, stride_w = strides
    opad_h, opad_w = output_padding
    assert opad_h < stride_h and opad_w < stride_w
    # dilate data
    data_dilate = dilate(data, [1, 1, stride_h, stride_w], name="data_dilate")
    # pad data
    fpad_top, fpad_left, fpad_bottom, fpad_right = get_pad_tuple(padding, (filter_h, filter_w))
    bpad_top = filter_h - 1 - fpad_top
    bpad_bottom = filter_h - 1 - fpad_bottom + opad_h
    bpad_left = filter_w - 1 - fpad_left
    bpad_right = filter_w - 1 - fpad_right + opad_w
    data_pad = pad(
        data_dilate, [0, 0, bpad_top, bpad_left], [0, 0, bpad_bottom, bpad_right], name="data_pad"
    )
    # transform kernel layout from IOHW to OIHW, and rotate kernel by 180 degrees
    kernel_transform = te.compute(
        (out_c, in_c, filter_h, filter_w),
        lambda o, i, h, w: kernel[i][o][filter_h - 1 - h][filter_w - 1 - w],
        name="kernel_transform",
    )
    return data_pad, kernel_transform


def declaration_conv2d_transpose_impl(data, kernel, strides, padding, out_dtype, output_padding):
    """Implementation of conv2d transpose"""
    data_pad, kernel_transform = conv2d_transpose_nchw_preprocess(
        data, kernel, strides, padding, out_dtype, output_padding
    )
    batch, in_c, in_h, in_w = data_pad.shape
    out_c, _, filter_h, filter_w = kernel_transform.shape

    # convolution stage
    out_c = simplify(out_c)

    out_h = simplify(in_h - filter_h + 1)
    out_w = simplify(in_w - filter_w + 1)
    dc = te.reduce_axis((0, in_c), name="dc")
    dh = te.reduce_axis((0, filter_h), name="dh")
    dw = te.reduce_axis((0, filter_w), name="dw")

    Output = te.compute(
        (batch, out_c, out_h, out_w),
        lambda b, c, h, w: te.sum(
            data_pad[b, dc, h + dh, w + dw].astype(out_dtype)
            * kernel_transform[c, dc, dh, dw].astype(out_dtype),
            axis=[dc, dh, dw],
        ),
        tag="conv2d_transpose_nchw",

    )

    return Output


def group_conv2d_transpose_nchw(data, kernel, strides=1, padding=0, output_padding=0, groups=1, dilation=1, out_dtype=None):
    # some pre-processing and prelimnary checks 
    if out_dtype is None:
        out_dtype = data.dtype

    # strides = _pair(strides)
    # padding = _pair(padding)
    # output_padding = _pair(output_padding)
    # dilation = _pair(dilation)
    batch, in_channels, in_height, in_width = data.shape
    _, out_c, filter_h, filter_w = kernel.shape
    assert in_channels % groups == 0, "input channels must divide group size"
    # assert out_c % groups == 0, "output channels must divide group size"

    batch, in_c, in_h, in_w = data.shape
    _, out_c, filter_h, filter_w = kernel.shape
    stride_h, stride_w = strides
    opad_h, opad_w = output_padding
    assert opad_h < stride_h and opad_w < stride_w, f"[{output_padding}] opad_h:{opad_h} < stride_h:{stride_h} and opad_w:{opad_w} < stride_w:{stride_w} does not satisfy."
    # dilate data
    data_dilate = dilate(data, [1, 1, stride_h, stride_w], name="data_dilate")
    # pad data
    fpad_top, fpad_left, fpad_bottom, fpad_right = get_pad_tuple(padding, (filter_h, filter_w))
    bpad_top = filter_h - 1 - fpad_top
    bpad_bottom = filter_h - 1 - fpad_bottom + opad_h
    bpad_left = filter_w - 1 - fpad_left
    bpad_right = filter_w - 1 - fpad_right + opad_w
    data_pad = pad(
        data_dilate, [0, 0, bpad_top, bpad_left], [0, 0, bpad_bottom, bpad_right], name="data_pad"
    )
    # transform kernel layout from IOHW to OIHW, and rotate kernel by 180 degrees
    kernel_transform = te.compute(
        (out_c, in_c, filter_h, filter_w),
        lambda i, o, h, w: kernel[o][i][filter_h - 1 - h][filter_w - 1 - w],
        name="kernel_transform",
    )

    batch, in_c, in_h, in_w = data_pad.shape
    out_c, _, filter_h, filter_w = kernel_transform.shape

    # convolution stage
    out_c = simplify(out_c)
    out_channels = simplify(out_c * groups)

    out_h = simplify(in_h - filter_h + 1)
    out_w = simplify(in_w - filter_w + 1)
    dc = te.reduce_axis((0, in_c // groups), name="dc")
    dh = te.reduce_axis((0, filter_h), name="dh")
    dw = te.reduce_axis((0, filter_w), name="dw")

    # data: batch, in_channels, out_h, out_w
    # weight: out_channels // G, in_channels, out_h, out_w
    return te.compute(
        (batch, out_channels, out_h, out_w),
        lambda b, c, h, w: te.sum(
            data_pad[
                b, 
                c // (out_channels // groups) * (in_channels // groups) + dc, 
                h + dh, 
                w + dw
            ].astype(out_dtype)
            * kernel_transform[
                c % (out_channels // groups), 
                c // (out_channels // groups) * (in_channels // groups) + dc, 
                dh, 
                dw
            ].astype(out_dtype),
            axis=[dc, dh, dw],
        ),
        tag="conv2d_transpose_nchw",
    )


@tvm.target.generic_func
def conv2d_transpose_legalize(attrs, inputs, types):
    """Legalizes Transposed 2D convolution op.

    Parameters
    ----------
    attrs : tvm.ir.Attrs
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
    if attrs["data_layout"] == "NHWC":
        data, kernel = inputs
        kernel_layout = attrs["kernel_layout"]
        # Convert Kernel layout to IOHW
        # kernel_layout is different from input kernel layout - IO is swapped
        if kernel_layout == "HWIO":
            # input kernel layout is swapped to HWOI
            # output kernel layout will be IOHW
            kernel = relay.transpose(kernel, axes=(3, 2, 0, 1))
        elif kernel_layout == "HWOI":
            # input kernel layout is swapped to HWIO
            # output kernel layout will be IOHW
            kernel = relay.transpose(kernel, axes=(2, 3, 0, 1))
        elif kernel_layout == "IOHW":
            # input kernel layout is swapped to OIHW
            # output kernel layout will be IOHW
            kernel = relay.transpose(kernel, axes=(1, 0, 2, 3))
        elif kernel_layout == "OIHW":
            # input kernel layout is swapped to IOHW
            # output kernel layout will be IOHW
            pass
        else:
            # Skip legalize. Let relay.nn.conv2d_transpose to handle the case
            return None

        # Set new attrs for conv2d_transpose.
        new_attrs = {k: attrs[k] for k in attrs.keys()}
        new_attrs["data_layout"] = "NCHW"
        # layout of kernel should be IOHW, but kernel_layout should be swapped - OIHW
        new_attrs["kernel_layout"] = "OIHW"

        # Convert data to NCHW.
        data = relay.transpose(data, axes=(0, 3, 1, 2))
        deconv = relay.nn.conv2d_transpose(data, kernel, **new_attrs)
        # Convert back to original NHWC layout.
        out = relay.transpose(deconv, axes=(0, 2, 3, 1))
        return out

    return None
