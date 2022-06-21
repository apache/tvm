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
# pylint: disable=unused-argument, redefined-builtin
"""Conv2D operators"""
from __future__ import absolute_import as _abs

import re
from collections import namedtuple
from typing import Optional, Sequence, Union

import numpy as np
import tvm
from tvm import auto_scheduler, te

from ..utils import get_const_int, get_const_tuple, simplify, tag
from .pad import pad
from .utils import get_pad_tuple, get_pad_tuple_generic
from .winograd_util import winograd_transform_matrices

# workload description of conv2d
Workload = namedtuple(
    "Workload",
    [
        "in_dtype",
        "out_dtype",
        "height",
        "width",
        "in_filter",
        "groups",
        "out_filter",
        "kernel_h",
        "kernel_w",
        "padt",
        "padl",
        "padb",
        "padr",
        "dilation_h",
        "dilation_w",
        "stride_h",
        "stride_w",
    ],
)


def conv2d(input, filter, strides, padding, dilation, layout="NCHW", out_dtype=None):
    """Conv2D operator.

    Parameters
    ----------
    input : tvm.te.Tensor
        4-D with shape [batch, in_channel, in_height, in_width]

    filter : tvm.te.Tensor
        4-D with shape [num_filter, in_channel, filter_height, filter_width]

    strides : int or a list/tuple of two ints
        stride size, or [stride_height, stride_width]

    padding : int or a list/tuple of 2 or 4 ints
        padding size, or
        [pad_height, pad_width] for 2 ints, or
        [pad_top, pad_left, pad_bottom, pad_right] for 4 ints

    dilation: int or a list/tuple of two ints
        dilation size, or [dilation_height, dilation_width]

    layout : str
        layout of data

    Returns
    -------
    output : tvm.te.Tensor
        4-D with shape [batch, out_channel, out_height, out_width]
    """
    # search platform specific declaration first
    # default declaration
    return conv(input, filter, strides, padding, dilation, 1, layout, out_dtype)


@tvm.target.generic_func
def conv2d_legalize(attrs, inputs, types):
    """Legalizes Conv2D op.

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


@tvm.target.generic_func
def conv2d_alter_layout(attrs, inputs, tinfos, out_type):
    """Change Conv2D layout.

    Parameters
    ----------
    attrs : tvm.ir.Attrs
        Attributes of current convolution
    inputs : tvm.relay.Expr
        Grouped input symbols
    tinfos : list
        Input shape and dtype
    out_type: type
        The output type

    Note
    ----
    Unlike other TOPI functions, this function operates on both graph level and operator level.
    """
    # not to change by default
    return None


@tvm.target.generic_func
def conv2d_infer_layout(workload, cfg):
    """Infer input/output shapes and layouts from a workload and cfg.

    Parameters
    ----------
    workload : tuple
        conv2d workload

    cfg : tuple
        tvm.autotvm config

    Returns
    -------
    Output : [tuple of tuple and str, tuple of tuple and str]
        Input shapes and layouts, and output shapes and layouts
    """
    raise ValueError("missing register for topi.nn.conv2d_infer_layout")


def _get_workload(data, kernel, stride, padding, dilation, out_dtype, data_layout="NCHW"):
    """Get the workload structure."""
    if data_layout == "NCHW":
        _, CI, IH, IW = get_const_tuple(data.shape)
    elif data_layout == "NHWC":
        _, IH, IW, CI = get_const_tuple(data.shape)
    elif data_layout == "HWCN":
        IH, IW, CI, _ = get_const_tuple(data.shape)
    else:
        raise ValueError("not support this layout {} yet".format(data_layout))

    if data_layout == "NCHW":
        CO, CIG, KH, KW = get_const_tuple(kernel.shape)
    else:
        KH, KW, CIG, CO = get_const_tuple(kernel.shape)

    dilation_h, dilation_w = (
        dilation if isinstance(dilation, (tuple, list)) else (dilation, dilation)
    )
    pt, pl, pb, pr = get_pad_tuple(
        padding,
        (get_const_int((KH - 1) * dilation_h + 1), get_const_int((KW - 1) * dilation_w + 1)),
    )
    GRPS = CI // CIG
    if isinstance(stride, (tuple, list)):
        HSTR, WSTR = stride
    else:
        HSTR, WSTR = stride, stride
    assert (data.dtype == kernel.dtype) or (
        data.dtype == "uint8" and kernel.dtype == "int8"
    ), "Do not support inputs with different data types now. ' \
        '{} vs. {}".format(
        data.dtype, kernel.dtype
    )
    return Workload(
        data.dtype,
        out_dtype,
        IH,
        IW,
        CI,
        GRPS,
        CO,
        KH,
        KW,
        pt,
        pl,
        pb,
        pr,
        dilation_h,
        dilation_w,
        HSTR,
        WSTR,
    )


def conv2d_nchw(Input, Filter, stride, padding, dilation, out_dtype=None):
    """Convolution operator in NCHW layout.

    Parameters
    ----------
    Input : tvm.te.Tensor
        4-D with shape [batch, in_channel, in_height, in_width]

    Filter : tvm.te.Tensor
        4-D with shape [num_filter, in_channel, filter_height, filter_width]

    stride : int or a list/tuple of two ints
        Stride size, or [stride_height, stride_width]

    padding : int or a list/tuple of 2 or 4 ints
        padding size, or
        [pad_height, pad_width] for 2 ints, or
        [pad_top, pad_left, pad_bottom, pad_right] for 4 ints

    dilation: int or a list/tuple of two ints
        dilation size, or [dilation_height, dilation_width]

    Returns
    -------
    Output : tvm.te.Tensor
        4-D with shape [batch, out_channel, out_height, out_width]
    """
    return conv(Input, Filter, stride, padding, dilation, 1, "NCHW", out_dtype=out_dtype)


def conv2d_hwcn(Input, Filter, stride, padding, dilation, out_dtype=None):
    """Convolution operator in HWCN layout.

    Parameters
    ----------
    Input : tvm.te.Tensor
        4-D with shape [in_height, in_width, in_channel, batch]

    Filter : tvm.te.Tensor
        4-D with shape [filter_height, filter_width, in_channel, num_filter]

    stride : int or a list/tuple of two ints
        Stride size, or [stride_height, stride_width]

    padding : int or a list/tuple of 2 or 4 ints
        padding size, or
        [pad_height, pad_width] for 2 ints, or
        [pad_top, pad_left, pad_bottom, pad_right] for 4 ints

    dilation: int or a list/tuple of two ints
        dilation size, or [dilation_height, dilation_width]

    Returns
    -------
    output : tvm.te.Tensor
        4-D with shape [out_height, out_width, out_channel, batch]
    """
    return conv(Input, Filter, stride, padding, dilation, 1, "HWCN", out_dtype=out_dtype)


def conv2d_nhwc(
    Input,
    Filter,
    stride,
    padding,
    dilation,
    out_dtype="float32",
    auto_scheduler_rewritten_layout="",
):
    """Convolution operator in NHWC layout.

    Parameters
    ----------
    Input : tvm.te.Tensor
        4-D with shape [batch, in_height, in_width, in_channel]

    Filter : tvm.te.Tensor
        4-D with shape [filter_height, filter_width, in_channel, num_filter]

    stride : int or a list/tuple of two ints
        Stride size, or [stride_height, stride_width]

    padding : int or a list/tuple of 2 or 4 ints
        padding size, or
        [pad_height, pad_width] for 2 ints, or
        [pad_top, pad_left, pad_bottom, pad_right] for 4 ints

    dilation: int or a list/tuple of two ints
        dilation size, or [dilation_height, dilation_width]

    out_dtype: str = "float32",
        The type of output tensor

    auto_scheduler_rewritten_layout: str = ""
        The layout after auto-scheduler's layout rewrite pass.

    Returns
    -------
    output : tvm.te.Tensor
        4-D with shape [batch, out_height, out_width, out_channel]
    """
    return conv(
        Input,
        Filter,
        stride,
        padding,
        dilation,
        1,
        "NHWC",
        out_dtype,
        auto_scheduler_rewritten_layout,
        auto_scheduler_should_rewrite_layout=True,
    )


def conv2d_NCHWc(data, kernel, stride, padding, dilation, layout, out_layout, out_dtype="float32"):
    """Conv2D operator for nChw[x]c layout.

    Parameters
    ----------
    data : tvm.te.Tensor
        5-D with shape [batch, in_channel_chunk, in_height, in_width, in_channel_block]

    kernel : tvm.te.Tensor
        6-D with shape
        [num_filter_chunk, in_channel_chunk, filter_height, filter_width,
        in_channel_block, num_filter_block]

    stride : int or a list/tuple of two ints
        stride size, or [stride_height, stride_width]

    padding : int or a list/tuple of 2 or 4 ints
        padding size, or
        [pad_height, pad_width] for 2 ints, or
        [pad_top, pad_left, pad_bottom, pad_right] for 4 ints

    dilation: int or a list/tuple of two ints
        dilation size, or [dilation_height, dilation_width]

    layout : str
        Input data layout

    out_layout : str
        Output data layout

    out_dtype : str
        output data type

    Returns
    -------
    output : tvm.te.Tensor
        5-D with shape [batch, out_channel_chunk, out_height, out_width, out_channel_block]
    """

    # layout and out_layout are not used here,
    # we keep them for debug convenience when dumping autotvm workload
    HSTR, WSTR = stride if isinstance(stride, (tuple, list)) else (stride, stride)
    dilation_h, dilation_w = (
        dilation if isinstance(dilation, (tuple, list)) else (dilation, dilation)
    )

    n, ic_chunk, ih, iw, ic_bn = get_const_tuple(data.shape)
    in_channel = ic_chunk * ic_bn
    target = tvm.target.Target.current(allow_none=False)
    oc_chunk, ic_chunk_group, kernel_height, kernel_width, _, oc_bn = get_const_tuple(kernel.shape)
    num_filter = oc_chunk * oc_bn
    groups = ic_chunk // ic_chunk_group

    dilated_kernel_h = (kernel_height - 1) * dilation_h + 1
    dilated_kernel_w = (kernel_width - 1) * dilation_w + 1

    pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
        padding, (dilated_kernel_h, dilated_kernel_w)
    )
    HPAD = pad_top + pad_down
    WPAD = pad_left + pad_right

    # output shape
    out_height = (ih + HPAD - dilated_kernel_h) // HSTR + 1
    out_width = (iw + WPAD - dilated_kernel_w) // WSTR + 1
    oshape = (n, oc_chunk, out_height, out_width, oc_bn)
    pad_before = (0, 0, pad_top, pad_left, 0)
    pad_after = (0, 0, pad_down, pad_right, 0)

    # DOPAD
    DOPAD = HPAD != 0 or WPAD != 0
    if DOPAD:
        data_pad = pad(data, pad_before, pad_after, name="data_pad")
    else:
        data_pad = data

    ic = te.reduce_axis((0, in_channel), name="ic")
    kh = te.reduce_axis((0, kernel_height), name="kh")
    kw = te.reduce_axis((0, kernel_width), name="kw")

    idxdiv = tvm.tir.indexdiv
    idxmod = tvm.tir.indexmod

    return te.compute(
        oshape,
        lambda n, oc_chunk, oh, ow, oc_block: te.sum(
            data_pad[
                n,
                idxdiv(ic, ic_bn),
                oh * HSTR + kh * dilation_h,
                ow * WSTR + kw * dilation_w,
                idxmod(ic, ic_bn),
            ].astype(out_dtype)
            * kernel[oc_chunk, idxdiv(ic, ic_bn), kh, kw, idxmod(ic, ic_bn), oc_block].astype(
                out_dtype
            ),
            axis=[ic, kh, kw],
        ),
        name="conv2d_NCHWc",
        tag="conv2d_NCHWc",
    )


def conv2d_NCHWc_int8(
    data, kernel, stride, padding, dilation, layout, out_layout, out_dtype="int32", n_elems=4
):
    """Conv2D operator for nChw[x]c layout.

    Parameters
    ----------
    data : tvm.te.Tensor
        5-D with shape [batch, in_channel_chunk, in_height, in_width, in_channel_block]

    kernel : tvm.te.Tensor
        7-D with shape
        [num_filter_chunk, in_channel_chunk, filter_height, filter_width, in_channel_block/4,
        num_filter_block, 4]

    stride : int or a list/tuple of two ints
        stride size, or [stride_height, stride_width]

    padding : int or a list/tuple of 2 or 4 ints
        padding size, or
        [pad_height, pad_width] for 2 ints, or
        [pad_top, pad_left, pad_bottom, pad_right] for 4 ints

    dilation: int or a list/tuple of two ints
        dilation size, or [dilation_height, dilation_width]

    layout : str
        Input data layout

    out_layout : str
        Output data layout

    out_dtype : str
        output data type

    n_elems : int
        numer of int8 elements accumulated

    Returns
    -------
    output : tvm.te.Tensor
        5-D with shape [batch, out_channel_chunk, out_height, out_width, out_channel_block]
    """

    # layout and out_layout are not used here,
    # we keep them for debug convenience when dumping autotvm workload
    HSTR, WSTR = stride if isinstance(stride, (tuple, list)) else (stride, stride)
    dilation_h, dilation_w = (
        dilation if isinstance(dilation, (tuple, list)) else (dilation, dilation)
    )

    n, ic_chunk, ih, iw, ic_bn = get_const_tuple(data.shape)
    in_channel = ic_chunk * ic_bn
    oc_chunk, ic_chunk_group, kernel_height, kernel_width, _, oc_bn, _ = get_const_tuple(
        kernel.shape
    )
    groups = ic_chunk // ic_chunk_group

    dilated_kernel_h = (kernel_height - 1) * dilation_h + 1
    dilated_kernel_w = (kernel_width - 1) * dilation_w + 1

    pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
        padding, (dilated_kernel_h, dilated_kernel_w)
    )
    HPAD = pad_top + pad_down
    WPAD = pad_left + pad_right

    # output shape
    out_height = (ih + HPAD - dilated_kernel_h) // HSTR + 1
    out_width = (iw + WPAD - dilated_kernel_w) // WSTR + 1
    oshape = (n, oc_chunk, out_height, out_width, oc_bn)
    pad_before = (0, 0, pad_top, pad_left, 0)
    pad_after = (0, 0, pad_down, pad_right, 0)

    # DOPAD
    DOPAD = HPAD != 0 or WPAD != 0
    if DOPAD:
        data_pad = pad(data, pad_before, pad_after, name="data_pad")
    else:
        data_pad = data

    ic = te.reduce_axis((0, in_channel), name="ic")
    kh = te.reduce_axis((0, kernel_height), name="kh")
    kw = te.reduce_axis((0, kernel_width), name="kw")

    if groups == 1:
        ic_outer = te.reduce_axis((0, in_channel // ic_bn), name="ic_outer")
        ic_f_inner = te.reduce_axis((0, ic_bn // n_elems), name="ic_f_inner")
        ic_s_inner = te.reduce_axis((0, n_elems), name="ic_s_inner")
        return te.compute(
            oshape,
            lambda n, oc_chunk, oh, ow, oc_block: te.sum(
                data_pad[
                    n,
                    ic_outer,
                    oh * HSTR + kh * dilation_h,
                    ow * WSTR + kw * dilation_w,
                    ic_f_inner * n_elems + ic_s_inner,
                ].astype(out_dtype)
                * kernel[oc_chunk, ic_outer, kh, kw, ic_f_inner, oc_block, ic_s_inner].astype(
                    out_dtype
                ),
                axis=[kh, kw, ic_outer, ic_f_inner, ic_s_inner],
            ),
            name="conv2d_NCHWc_int8",
            tag="conv2d_NCHWc_int8",
            attrs={"schedule_rule": "meta_schedule.conv2d_NCHWc_int8"},
        )
    # for int8 group conv support
    ic_chunk = in_channel // ic_bn
    ic_outer = te.reduce_axis((0, ic_chunk // groups), name="ic_outer")
    ic_f_inner = te.reduce_axis((0, ic_bn // n_elems), name="ic_f_inner")
    ic_s_inner = te.reduce_axis((0, n_elems), name="ic_s_inner")
    oshape = (n, oc_chunk, out_height, out_width, oc_bn)
    return te.compute(
        oshape,
        lambda n, occ, oh, ow, oc_block: te.sum(
            data_pad[
                n,
                (occ * oc_bn // (oc_chunk * oc_bn // groups)) * (ic_chunk // groups) + ic_outer,
                oh * HSTR + kh,
                ow * WSTR + kw,
                ic_f_inner * n_elems + ic_s_inner,
            ].astype(out_dtype)
            * kernel[occ, ic_outer, kh, kw, ic_f_inner, oc_block, ic_s_inner].astype(out_dtype),
            axis=[kh, kw, ic_outer, ic_f_inner, ic_s_inner],
        ),
        name="conv2d_NCHWc_int8",
        tag="conv2d_NCHWc_int8",
        attrs={"schedule_rule": "meta_schedule.conv2d_NCHWc_int8"},
    )


def conv2d_gemm_weight_transform(kernel, tile_rows, tile_cols):
    """Weight transformation for winograd

    Parameters
    ----------
    kernel: Tensor
        The raw kernel tensor with layout "NHWC".
    tile_rows: int
        Tile rows of the weight transformation for ConvGemm.
    tile_cols: int
        Tile columns of the weight transformation for ConvGemm.

    Returns
    -------
    output : tvm.te.Tensor
        2-D with shape [CI*KH*KW,CO]
    """
    KH, KW, IC, OC = get_const_tuple(kernel.shape)
    K = KH * KW * IC
    N = OC

    kernel_flat = te.compute(
        (K, N), lambda x, y: kernel[(x // IC) // KW, (x // IC) % KW, x % IC, y], "weight_flatten"
    )

    pad_K = 0
    pad_N = 0

    if N % tile_rows != 0:
        pad_N = tile_rows - (N % tile_rows)

    if K % tile_cols != 0:
        pad_K = tile_cols - (K % tile_cols)

    N_padded = N + pad_N
    K_padded = K + pad_K

    if pad_K != 0 or pad_N != 0:
        kernel_flat = pad(
            kernel_flat, pad_before=(0, 0), pad_after=(pad_K, pad_N), name="weight_padding"
        )

    return te.compute(
        (N_padded // tile_rows, K_padded // tile_cols, tile_rows, tile_cols),
        lambda x, y, z, w: kernel_flat[w + tile_cols * y, z + tile_rows * x],
        name="weight_block_reshape",
    )


def conv2d_winograd_weight_transform(kernel, tile_size):
    """Weight transformation for winograd

    Parameters
    ----------
    kernel: Tensor
        The raw kernel tensor with layout "NCHW".
    tile_size: int
        Tile size of winograd transform. e.g. 2 for F(2x2, 3x3) and 4 for F(4x4, 3x3)

    Returns
    -------
    output : tvm.te.Tensor
        4-D with shape [alpha, alpha, CO, CI]
    """
    shape = get_const_tuple(kernel.shape)
    assert shape[2] == shape[3], "Only support NxN kernel"

    K = shape[3]
    r = tile_size + K - 1
    shape = (r, r) + shape[:2]

    _, _, G = winograd_transform_matrices(tile_size, K, kernel.dtype)

    r_kh = te.reduce_axis((0, K), name="r_kh")
    r_kw = te.reduce_axis((0, K), name="r_kw")
    return te.compute(
        shape,
        lambda eps, nu, co, ci: te.sum(
            kernel[co][ci][r_kh][r_kw] * G[eps][r_kh] * G[nu][r_kw], axis=[r_kh, r_kw]
        ),
        name="transform_weight",
    )


def conv2d_winograd_nnpack_weight_transform(kernel, convolution_algorithm, out_dtype):
    """Weight transformation for winograd

    Parameters
    ----------
    kernel: Tensor
        The raw kernel tensor with layout "NCHW". Only 3x3 kernel is supported for now.
    convolution_algorithm: int
        The convolution algorithm for Winograd NNPACK.

    Returns
    -------
    output : tvm.te.Tensor
        4-D with shape [alpha, alpha, CO, CI]
    """
    # pylint: disable=import-outside-toplevel
    from tvm.contrib import nnpack

    return nnpack.convolution_inference_weight_transform(
        kernel, algorithm=convolution_algorithm, dtype=out_dtype
    )


def group_conv2d_nchw(Input, Filter, stride, padding, dilation, groups, out_dtype=None):
    """Group convolution operator in NCHW layout.

    Parameters
    ----------
    Input : tvm.te.Tensor
        4-D with shape [batch, in_channel, in_height, in_width]

    Filter : tvm.te.Tensor
        4-D with shape [num_filter, in_channel // groups, filter_height, filter_width]

    stride : int or a list/tuple of two ints
        Stride size, or [stride_height, stride_width]

    padding : int or a list/tuple of 2 or 4 ints
        padding size, or
        [pad_height, pad_width] for 2 ints, or
        [pad_top, pad_left, pad_bottom, pad_right] for 4 ints

    dilation : int or a list/tuple of two ints
        dilation size, or [dilation_height, dilation_width]

    groups : int
        number of groups

    out_dtype : str
        The output type. This is used for mixed precision.

    Returns
    -------
    Output : tvm.te.Tensor
        4-D with shape [batch, out_channel, out_height, out_width]
    """
    return conv(Input, Filter, stride, padding, dilation, groups, "NCHW", out_dtype=out_dtype)


def conv(
    inp: te.Tensor,
    filt: te.Tensor,
    stride: Union[int, Sequence[int]],
    padding: Union[int, Sequence[int]],
    dilation: Union[int, Sequence[int]],
    groups: int,
    order: str,
    out_dtype: Union[str, None] = None,
    auto_scheduler_rewritten_layout: Optional[str] = None,
    auto_scheduler_should_rewrite_layout: bool = False,
):
    """Convolution operator in NCHW or NHWC layout.

    Supports 1D, 2D, 3D, ... and grouping.

    Parameters
    ----------
    inp : tvm.te.Tensor
        N-D with shape [batch, in_channel, in_height, in_width, ...] ordered by `order`

    filt : tvm.te.Tensor
        N-D with shape [num_filter, in_channel // groups, filter_height, filter_width, ...]
        for NCHW or [filter_height, filter_width, ..., in_channel // groups, num_filter] for NHWC

    stride : int or a list/tuple of dim ints
        (where dim=2 for NCHW, dim=1 for NCH, etc.)
        Stride size, or [stride_height, stride_width, ...]

    padding : int or a list/tuple of dim or 2*dim ints
        (where dim=2 for NCHW, dim=1 for NCH, etc.)
        padding size, or
        [pad_height, pad_width, ...] for dim ints, or
        [pad_top, pad_left, pad_bottom, pad_right] for 2*dim ints

    dilation : int or a list/tuple of two ints
        dilation size, or [dilation_height, dilation_width]

    groups : int
        number of groups

    order : str
        Ordering of dimensions. N indicates batch dimension, C indicates
        channels, any other character indicates HW (or H or HWD for 1D and 3D).

    out_dtype : str
        Elements are converted to this type before elementwise multiplication
        and summation.

    auto_scheduler_should_rewrite_layout : bool
        Should auto scheduler be allowed to rewrite the layout of the filter
        tensor. Defaults to false. This can cause errors if used with grouped
        convs.

    auto_scheduler_rewritten_layout: str
        Layout from autoscheduler's layout rewritting.

    Returns
    -------
    Output : tvm.te.Tensor
        N-D with shape [batch, out_channel, out_height, out_width, ...] ordered by `order`.
    """
    dim = len(inp.shape) - 2
    if out_dtype is None:
        out_dtype = inp.dtype
    assert isinstance(stride, int) or len(stride) == dim
    assert isinstance(dilation, int) or len(dilation) == dim
    if isinstance(stride, int):
        strides = [stride for _ in range(dim)]
    else:
        strides = stride

    if isinstance(dilation, int):
        dilations = [dilation for _ in range(dim)]
    else:
        dilations = list(dilation)

    # transform from order to NCHW
    permutation_to = [order.find("N"), order.find("C")] + [
        x.span()[0] for x in re.finditer("[^NC]", order)
    ]
    # transform from NCHW to order
    permutation_from = np.argsort(permutation_to)
    # transform from CHW to order
    permutation_from_reductions = permutation_from[1:].copy()
    permutation_from_reductions[permutation_from_reductions > permutation_from[0]] -= 1

    # kernel permutation, if C appears before HW then num_filter is first, otherwise it is last
    # tkonolige: I don't really understand kernel ordering for NHWC, it seems
    # like num_filters should match the N dimension
    if order.find("C") < re.search("[^NC]", order).span()[0]:
        permutation_to_kernel = [0, 1] + list(range(2, dim + 2))
    else:
        permutation_to_kernel = [dim + 1, dim] + list(range(dim))
    permutation_from_kernel = np.argsort(permutation_to_kernel)

    batch, in_channel, *dimensions = np.array(get_const_tuple(inp.shape))[permutation_to].tolist()
    num_filter, _, *kernel_dimensions = np.array(get_const_tuple(filt.shape))[
        permutation_to_kernel
    ].tolist()

    # Autoscheduler may have messed with the input layout, so we extract the
    # dimensions that it gives us
    if auto_scheduler_rewritten_layout:
        num_filter, _, *kernel_dimensions = auto_scheduler.get_shape_from_rewritten_layout(
            auto_scheduler_rewritten_layout,
            ["ff", "rc"] + [f"r{i}" for i in ["y", "x", "z"][: len(kernel_dimensions)]],
        )
        auto_scheduler.remove_index_check(filt)

    assert in_channel % groups == 0, "input channels must divide group size"
    assert num_filter % groups == 0, "output channels must divide group size"

    dilated_kernel_dimensions = [(k - 1) * dil + 1 for k, dil in zip(kernel_dimensions, dilations)]
    pad_begin, pad_end = get_pad_tuple_generic(padding, dilated_kernel_dimensions)
    # compute the output shape
    out_channel = num_filter
    out_dimensions = [
        simplify(d - (k - 1) * dil - 1 + pb + pe) // stride + 1
        for d, k, dil, pb, pe, stride in zip(
            dimensions, kernel_dimensions, dilations, pad_begin, pad_end, strides
        )
    ]
    # compute graph
    pad_before = list(np.array([0, 0] + pad_begin)[permutation_from])
    pad_after = list(np.array([0, 0] + pad_end)[permutation_from])
    temp = pad(inp, pad_before, pad_after, name="pad_temp")
    rc = te.reduce_axis((0, in_channel // groups), name="rc")
    rs = [te.reduce_axis((0, k), name=f"r{i}") for i, k in zip(["y", "x", "z"], kernel_dimensions)]

    def compute(*args):
        nn, ff, *dim_indices = list(np.array(args)[permutation_to])

        if groups == 1:
            simplified_channel_index = rc
        else:
            simplified_channel_index = ff // (num_filter // groups) * (in_channel // groups) + rc

        return te.sum(
            temp.__getitem__(
                tuple(
                    np.array(
                        [nn, simplified_channel_index]
                        + [
                            di * stride + r * dil
                            for di, stride, r, dil in zip(dim_indices, strides, rs, dilations)
                        ]
                    )[permutation_from]
                )
            ).astype(out_dtype)
            * filt.__getitem__(tuple(np.array([ff, rc] + rs)[permutation_from_kernel])).astype(
                out_dtype
            ),
            # Schedules depend on reduction axes being in the same order as the
            # layout, so we reorder here.
            axis=np.array([rc, *rs])[permutation_from_reductions].tolist(),
        )

    out = te.compute(
        list(np.array([batch, out_channel] + out_dimensions)[permutation_from]),
        compute,
        # tag is expected to be lowercase
        tag=f"{'group_' if groups > 1 else ''}conv{dim}d_{order.lower()}",
        name=f"{'group_' if groups > 1 else ''}conv{dim}d_{order.lower()}",
        attrs={"layout_free_placeholders": [filt]} if auto_scheduler_should_rewrite_layout else {},
        varargs_names=list(np.array(["nn", "ff", "yy", "xx", "zz"])[permutation_from]),
    )
    # if we used autoscheduler's changed layout we need to rewrite the ordering
    # of the output dimensions
    if auto_scheduler_rewritten_layout:
        out = auto_scheduler.rewrite_compute_body(out, auto_scheduler_rewritten_layout)
    return out


def group_conv2d_nhwc(Input, Filter, stride, padding, dilation, groups, out_dtype=None):
    """Group convolution operator in NHWC layout.

    Parameters
    ----------
    Input : tvm.te.Tensor
        4-D with shape [batch, in_height, in_width, in_channel, ...]

    Filter : tvm.te.Tensor
        4-D with shape [filter_height, filter_width, in_channel // groups, num_filter]

    stride : int or a list/tuple of two ints
        Stride size, or [stride_height, stride_width]

    padding : int or a list/tuple of 2 or 4 ints
        padding size, or
        [pad_height, pad_width] for 2 ints, or
        [pad_top, pad_left, pad_bottom, pad_right] for 4 ints

    dilation : int or a list/tuple of two ints
        dilation size, or [dilation_height, dilation_width]

    groups : int
        number of groups

    out_dtype : str
        The output type. This is used for mixed precision.

    Returns
    -------
    Output : tvm.te.Tensor
        4-D with shape [batch, out_height, out_width, out_channel]
    """
    return conv(Input, Filter, stride, padding, dilation, groups, "NHWC", out_dtype=out_dtype)


def unpack_NCHWc_to_nchw(packed_out, out_dtype):
    """Unpack conv2d_NCHWc output from layout NCHWc to NCHW

    Parameters
    ----------
    packed_out : tvm.te.Tensor
        The output tensor of conv2d_NCHWc.

    out_dtype : str
        The output dtype.

    Returns
    -------
    unpacked_out : tvm.te.Tensor
        The unpacked output tensor in NCHW layout.
    """
    n, oc_chunk, oh, ow, oc_bn = get_const_tuple(packed_out.shape)

    idxmod = tvm.tir.indexmod
    idxdiv = tvm.tir.indexdiv

    oshape = (n, oc_chunk * oc_bn, oh, ow)
    unpacked_out = te.compute(
        oshape,
        lambda n, c, h, w: packed_out[n, idxdiv(c, oc_bn), h, w, idxmod(c, oc_bn)].astype(
            out_dtype
        ),
        name="output_unpack",
        tag=tag.INJECTIVE + ",unpack_nchwc",
    )
    return unpacked_out


def _conv2d_winograd_nhwc_impl(
    data,
    weight,
    strides,
    padding,
    dilation,
    out_dtype,
    tile_size,
    pre_computed=False,
    auto_scheduler_rewritten_layout="",
):
    """Conv2D Winograd implementation in NHWC layout.
    This is a clean version to be used by the auto-scheduler for both CPU and GPU.

    Parameters
    ----------
    data : tvm.Tensor
        4-D with shape [batch, in_height, in_width, in_channel]
    weight : tvm.Tensor
        4-D with shape [filter_height, filter_width, in_channel, num_filter]
    strides : int or a list/tuple of two ints
        stride size, or [stride_height, stride_width]
    padding : int or a list/tuple of two ints
        padding size, or [pad_height, pad_width]
    dilation: int or a list/tuple of two ints
        dilation size, or [dilation_height, dilation_width]
    out_dtype : str, optional
        Specifies the output data type.
    tile_size : int
        The size of the tile to use for the Winograd filter
    pre_computed: bool = False
        Whether the kernel is precomputed
    auto_scheduler_rewritten_layout: str = ""
        The layout after auto-scheduler's layout rewrite pass.

    Returns
    -------
    output : tvm.Tensor
        4-D with shape [batch, out_height, out_width, out_channel]
    """
    N, H, W, CI = get_const_tuple(data.shape)
    if isinstance(dilation, int):
        dilation_h = dilation_w = dilation
    else:
        dilation_h, dilation_w = dilation

    assert (dilation_h, dilation_w) == (1, 1), "Does not support dilation"
    if not pre_computed:
        KH, KW, CI, CO = get_const_tuple(weight.shape)
    else:
        if auto_scheduler_rewritten_layout:
            H_CAT, W_CAT, CO, CI = get_const_tuple(
                auto_scheduler.get_shape_from_rewritten_layout(
                    auto_scheduler_rewritten_layout, ["eps", "nu", "co", "ci"]
                )
            )
            auto_scheduler.remove_index_check(weight)
        else:
            H_CAT, W_CAT, CO, CI = get_const_tuple(weight.shape)

        KH, KW = H_CAT - tile_size + 1, W_CAT - tile_size + 1

    pad_t, pad_l, pad_b, pad_r = get_pad_tuple(padding, (KH, KW))
    HSTR, WSTR = (strides, strides) if isinstance(strides, int) else strides
    assert HSTR == 1 and WSTR == 1 and KH == 3 and KW == 3

    r = KW
    m = tile_size
    alpha = m + r - 1
    A, B, G = winograd_transform_matrices(m, r, out_dtype)

    H = (H + pad_t + pad_b - KH) // HSTR + 1
    W = (W + pad_l + pad_r - KW) // WSTR + 1
    nH, nW = (H + m - 1) // m, (W + m - 1) // m
    P = N * nH * nW

    pad_extra = (nW - 1) * m + alpha - (H + pad_t + pad_b)
    data_pad = pad(
        data,
        (0, pad_t, pad_l, 0),
        (0, pad_b + pad_extra, pad_r + pad_extra, 0),
        name="data_pad",
        attrs={"schedule_rule": "None"},
    )

    if not pre_computed:
        r_kh = te.reduce_axis((0, KH), name="r_kh")
        r_kw = te.reduce_axis((0, KW), name="r_kw")
        kernel_pack = te.compute(
            (alpha, alpha, CO, CI),
            lambda eps, nu, co, ci: te.sum(
                weight[r_kh][r_kw][ci][co] * G[eps][r_kh] * G[nu][r_kw], axis=[r_kh, r_kw]
            ),
            name="kernel_pack",
        )
        attrs = {}
    else:
        kernel_pack = weight
        attrs = {"layout_free_placeholders": [kernel_pack]}

    # pack data tile
    input_tile = te.compute(
        (alpha, alpha, P, CI),
        lambda eps, nu, p, ci: data_pad[p // (nH * nW)][((p // nW) % nH) * m + eps][
            (p % nW) * m + nu
        ][ci],
        name="input_tile",
        attrs={"schedule_rule": "None"},
    )

    # transform data
    target = tvm.target.Target.current(allow_none=True)
    if target is not None:
        target_kind = "meta_schedule.winograd_data_pack." + target.kind.name
    else:
        target_kind = "None"

    r_a = te.reduce_axis((0, alpha), "r_a")
    r_b = te.reduce_axis((0, alpha), "r_b")
    data_pack = te.compute(
        (alpha, alpha, P, CI),
        lambda eps, nu, p, ci: te.sum(
            input_tile[r_a][r_b][p][ci] * B[r_a][eps] * B[r_b][nu], axis=[r_a, r_b]
        ),
        name="data_pack",
        attrs={
            "auto_scheduler_simplify_const_tensor_indices": ["eps", "nu", "r_a", "r_b"],
            "schedule_rule": target_kind,
        },
        # the attrs are necessary hints for the auto-scheduler
    )

    # do batch gemm
    ci = te.reduce_axis((0, CI), name="ci")
    bgemm = te.compute(
        (alpha, alpha, P, CO),
        lambda eps, nu, p, co: te.sum(
            data_pack[eps][nu][p][ci] * kernel_pack[eps][nu][co][ci], axis=[ci]
        ),
        name="bgemm",
        attrs=attrs,
    )

    if auto_scheduler_rewritten_layout:
        bgemm = auto_scheduler.rewrite_compute_body(bgemm, auto_scheduler_rewritten_layout)

    # inverse transform
    if target is not None:
        target_kind = "meta_schedule.winograd_inverse." + target.kind.name
    else:
        target_kind = "None"

    r_a = te.reduce_axis((0, alpha), "r_a")
    r_b = te.reduce_axis((0, alpha), "r_b")
    inverse = te.compute(
        (m, m, P, CO),
        lambda vh, vw, p, co: te.sum(
            bgemm[r_a][r_b][p][co] * A[r_a][vh] * A[r_b][vw], axis=[r_a, r_b]
        ),
        name="inverse",
        attrs={
            "auto_scheduler_simplify_const_tensor_indices": ["vh", "vw", "r_a", "r_b"],
            "schedule_rule": target_kind,
        },
        # the attrs are necessary hints for the auto-scheduler
    )

    # output
    output = te.compute(
        (N, H, W, CO),
        lambda n, h, w, co: inverse[h % m, w % m, n * nH * nW + (h // m) * nW + (w // m), co],
        name="conv2d_winograd",
    )

    return output


@tvm.target.generic_func
def conv2d_winograd_nhwc(
    data,
    weight,
    strides,
    padding,
    dilation,
    out_dtype,
    pre_computed=False,
    auto_scheduler_rewritten_layout="",
):
    """Conv2D Winograd in NHWC layout.
    This is a clean version to be used by the auto-scheduler for both CPU and GPU.

    Parameters
    ----------
    data : tvm.Tensor
        4-D with shape [batch, in_height, in_width, in_channel]
    weight : tvm.Tensor
        4-D with shape [filter_height, filter_width, in_channel, num_filter]
    strides : int or a list/tuple of two ints
        stride size, or [stride_height, stride_width]
    padding : int or a list/tuple of two ints
        padding size, or [pad_height, pad_width]
    dilation: int or a list/tuple of two ints
        dilation size, or [dilation_height, dilation_width]
    out_dtype : str, optional
        Specifies the output data type.
    pre_computed: bool
        Whether the kernel is precomputed
    auto_scheduler_rewritten_layout: str = ""
        The layout after auto-scheduler's layout rewrite pass.

    Returns
    -------
    output : tvm.Tensor
        4-D with shape [batch, out_height, out_width, out_channel]
    """
    tile_size = 4

    return _conv2d_winograd_nhwc_impl(
        data,
        weight,
        strides,
        padding,
        dilation,
        out_dtype,
        tile_size,
        pre_computed,
        auto_scheduler_rewritten_layout,
    )


def conv2d_winograd_nhwc_without_weight_transform(
    data,
    weight,
    strides,
    padding,
    dilation,
    out_dtype,
    auto_scheduler_rewritten_layout="",
):
    """Conv2D Winograd without layout transform in NHWC layout.
    This is a clean version to be used by the auto-scheduler for both CPU and GPU.

    Parameters
    ----------
    data : tvm.Tensor
        4-D with shape [batch, in_height, in_width, in_channel]
    weight : tvm.Tensor
        4-D with shape [filter_height, filter_width, in_channel, num_filter]
    strides : int or a list/tuple of two ints
        stride size, or [stride_height, stride_width]
    padding : int or a list/tuple of two ints
        padding size, or [pad_height, pad_width]
    dilation: int or a list/tuple of two ints
        dilation size, or [dilation_height, dilation_width]
    out_dtype : str, optional
        Specifies the output data type.
    auto_scheduler_rewritten_layout: str = ""
        The layout after auto-scheduler's layout rewrite pass.

    Returns
    -------
    output : tvm.Tensor
        4-D with shape [batch, out_height, out_width, out_channel]
    """

    return conv2d_winograd_nhwc(
        data,
        weight,
        strides,
        padding,
        dilation,
        out_dtype,
        pre_computed=True,
        auto_scheduler_rewritten_layout=auto_scheduler_rewritten_layout,
    )
