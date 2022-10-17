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
from tvm import relay, te

from ..utils import simplify
from .dilate import dilate
from .pad import pad
from .utils import get_pad_tuple


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


def group_conv2d_transpose_nchw(data, kernel, stride, padding, out_dtype, output_padding, groups):
    """Group convolution operator in NCHW layout.

    Parameters
    ----------
    data : tvm.te.Tensor
        4-D with shape [batch, in_channel, in_height, in_width]

    kernel : tvm.te.Tensor
        4-D with shape [in_channel, out_channel // groups, filter_height, filter_width]

    stride : int or a list/tuple of two ints
        Stride size, or [stride_height, stride_width]

    padding : int or a list/tuple of 2 or 4 ints
        padding size, or
        [pad_height, pad_width] for 2 ints, or
        [pad_top, pad_left, pad_bottom, pad_right] for 4 ints

    out_dtype : str
        The output data type. This is used for mixed precision.

    output_padding : tuple of ints
        Used to get the right output shape for gradients

    groups : int
        number of groups

    out_dtype : str
        The output type. This is used for mixed precision.

    Returns
    -------
    Output : tvm.te.Tensor
        4-D with shape [batch, out_channel, out_height, out_width]
    """
    if groups == 1:
        return conv2d_transpose_nchw(data, kernel, stride, padding, out_dtype, output_padding)

    # some pre-processing and prelimnary checks
    if out_dtype is None:
        out_dtype = data.dtype

    batch, in_channels, in_h, in_w = data.shape
    _, out_c, filter_h, filter_w = kernel.shape
    assert (
        in_channels % groups == 0
    ), f"input channels {in_channels} must divide group size {groups}"
    # assert out_c % groups == 0, f"output channels {in_c} must divide group size {groups}"

    strides = _pair(stride)
    # padding = _pair(padding)
    # output_padding = _pair(output_padding)
    # dilation = _pair(dilation)

    stride_h, stride_w = strides
    opad_h, opad_w = output_padding
    assert (
        opad_h < stride_h and opad_w < stride_w
    ), f"[{output_padding}] opad_h:{opad_h} < stride_h:{stride_h} \
        and opad_w:{opad_w} < stride_w:{stride_w} does not satisfy."
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
        (out_c, in_channels, filter_h, filter_w),
        lambda i, o, h, w: kernel[o][i][filter_h - 1 - h][filter_w - 1 - w],
        name="kernel_transform",
    )

    batch, in_channels, in_h, in_w = data_pad.shape
    out_c, _, filter_h, filter_w = kernel_transform.shape

    # convolution stage
    out_channels = simplify(out_c * groups)

    out_h = simplify(in_h - filter_h + 1)
    out_w = simplify(in_w - filter_w + 1)
    dc = te.reduce_axis((0, in_channels // groups), name="dc")
    dh = te.reduce_axis((0, filter_h), name="dh")
    dw = te.reduce_axis((0, filter_w), name="dw")

    # data: batch, in_channels, out_h, out_w
    # weight: out_channels // G, in_channels, out_h, out_w
    return te.compute(
        (batch, out_channels, out_h, out_w),
        lambda b, c, h, w: te.sum(
            data_pad[
                b, c // (out_channels // groups) * (in_channels // groups) + dc, h + dh, w + dw
            ].astype(out_dtype)
            * kernel_transform[
                c % (out_channels // groups),
                c // (out_channels // groups) * (in_channels // groups) + dc,
                dh,
                dw,
            ].astype(out_dtype),
            axis=[dc, dh, dw],
        ),
        tag="group_conv2d_transpose_nchw",
    )


def layout_transform(tensor: "relay.Expr", current_layout: str, desired_layout: str):
    """Transform a tensor with the current layout to the desired layout.

    E.g. layout_transform(t, "NCHW", "CNHW") --> relay.transpose(t, [1, 0, 2, 3])

    Parameters
    ----------
    tensor: relay.Expr
        The Tensor to transpose

    current_layout: str
        The current layout e.g. NCHW or OIHW

    desired_layout: str
        The desired layout, must be compatible with current_layout

    Returns
    -------
    The layout_transformed tensor.
    """
    if sorted(current_layout) != sorted(desired_layout):
        raise ValueError(f"Incompatible layouts: {current_layout} vs {desired_layout}")

    if current_layout == desired_layout:
        return tensor

    current_layout_map = {c: i for i, c in enumerate(current_layout)}
    desired_layout_map = {c: i for i, c in enumerate(desired_layout)}

    axes = [None] * len(current_layout)
    for c, i in desired_layout_map.items():
        axes[i] = current_layout_map[c]
    return relay.transpose(tensor, axes=axes)


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
    data, kernel = inputs
    kernel_layout = attrs["kernel_layout"]

    target = tvm.target.Target.current(allow_none=True)
    if target and "cudnn" in target.libs:
        # cuDNN backend can directly operate on NHWC layout.
        return None

    if attrs["data_layout"] == "NHWC":
        kernel = layout_transform(kernel, kernel_layout, "IOHW")

        # Set new attrs for conv2d_transpose.
        new_attrs = {k: attrs[k] for k in attrs.keys()}
        new_attrs["data_layout"] = "NCHW"
        # layout of kernel should be IOHW, but kernel_layout will be swapped - OIHW
        new_attrs["kernel_layout"] = "IOHW"

        # Convert data to NCHW.
        data = relay.transpose(data, axes=(0, 3, 1, 2))
        deconv = relay.nn.conv2d_transpose(data, kernel, **new_attrs)
        # Convert back to original NHWC layout.
        out = relay.transpose(deconv, axes=(0, 2, 3, 1))
        return out

    if attrs["data_layout"] == "NCHW":
        kernel = layout_transform(kernel, kernel_layout, "IOHW")
        new_attrs = {k: attrs[k] for k in attrs.keys()}

        # layout of kernel should be IOHW, but kernel_layout will be swapped - OIHW
        new_attrs["kernel_layout"] = "IOHW"
        return relay.nn.conv2d_transpose(data, kernel, **new_attrs)

    return None
