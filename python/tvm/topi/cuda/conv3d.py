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
# pylint: disable=invalid-name, unused-argument
"""Compute definition for conv3d with cuda backend"""
from tvm import te
from tvm import autotvm
from tvm.contrib import cudnn

from .. import nn, generic
from ..utils import get_const_tuple, traverse_inline
from .conv3d_direct import schedule_direct_conv3d_cuda


@autotvm.register_topi_compute("conv3d_ncdhw.cuda")
def conv3d_ncdhw(cfg, data, kernel, strides, padding, dilation, groups, out_dtype="float32"):
    """Conv3D operator in NCDHW layout for cuda backend.

    Parameters
    ----------
    cfg: ConfigEntity
        The config for this template

    data : tvm.te.Tensor
        5-D with shape [batch, in_channel, in_depth, in_height, in_width]

    kernel : tvm.te.Tensor
        5-D with shape [num_filter, in_channel, filter_depth, filter_height, filter_width]

    strides : int or a list/tuple of three ints
        stride size, or [stride_depth, stride_height, stride_width]

    padding : int or a list/tuple of three ints
        padding size, or [pad_depth, pad_height, pad_width]

    dilation: int or a list/tuple of three ints
        dilation size, or [dilation_depth, dilation_height, dilation_width]

    groups: int
        Number of groups

    out_dtype: str
        The output type. This is used for mixed precision.

    Returns
    -------
    output : tvm.te.Tensor
        5-D with shape [batch, out_channel, out_depth, out_height, out_width]
    """
    return nn.conv3d_ncdhw(data, kernel, strides, padding, dilation, groups, out_dtype)


@autotvm.register_topi_schedule("conv3d_ncdhw.cuda")
def schedule_conv3d_ncdhw(cfg, outs):
    """TOPI schedule callback of conv3d for cuda gpu

    Parameters
    ----------
    cfg: ConfigEntity
        The config for this template

    outs: Array of Tensor
        The computation graph description of conv2d
        in the format of an array of tensors.

    Returns
    -------
    s: Schedule
        The computation schedule for conv2d.
    """
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        if "conv3d_ncdhw" in op.tag:
            schedule_direct_conv3d_cuda(cfg, s, op.output(0), "NCDHW", "conv3d_ncdhw.cuda")

    traverse_inline(s, outs[0].op, _callback)
    return s


@autotvm.register_topi_compute("conv3d_ndhwc.cuda")
def conv3d_ndhwc(cfg, data, kernel, strides, padding, dilation, groups, out_dtype="float32"):
    """Conv3d operator in NDHWC layout for cuda backend.

    Parameters
    ----------
    Input : tvm.te.Tensor
        5-D with shape [batch, in_depth, in_height, in_width, in_channel]

    Filter : tvm.te.Tensor
        5-D with shape [filter_depth, filter_height, filter_width, in_channel, num_filter]

    stride : int or a list/tuple of three ints
        Stride size, or [stride_depth, stride_height, stride_width]

    padding : int or str
        Padding size, or ['VALID', 'SAME']

    dilation: int or a list/tuple of three ints
        dilation size, or [dilation_depth, dilation_height, dilation_width]

    groups: int
        Number of groups

    Returns
    -------
    Output : tvm.te.Tensor
        5-D with shape [batch, out_depth, out_height, out_width, out_channel]
    """
    return nn.conv3d_ndhwc(data, kernel, strides, padding, dilation, groups, out_dtype)


@autotvm.register_topi_schedule("conv3d_ndhwc.cuda")
def schedule_conv3d_ndhwc(cfg, outs):
    """TOPI schedule callback of conv3d for cuda gpu

    Parameters
    ----------
    cfg: ConfigEntity
        The config for this template

    outs: Array of Tensor
        The computation graph description of conv3d
        in the format of an array of tensors.

    Returns
    -------
    s: Schedule
        The computation schedule for conv2d.
    """
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        if "conv3d_ndhwc" in op.tag:
            schedule_direct_conv3d_cuda(cfg, s, op.output(0), "NDHWC", "conv3d_ndhwc.cuda")

    traverse_inline(s, outs[0].op, _callback)
    return s


@autotvm.register_topi_compute("conv3d_cudnn.cuda")
def conv3d_cudnn(
    cfg, data, kernel, strides, padding, dilation, groups, layout="NCDHW", out_dtype="float32"
):
    """Conv3D operator for cuda backend.

    Parameters
    ----------
    cfg: ConfigEntity
        The config for this template

    data : tvm.te.Tensor
        5-D with shape [batch, in_channel, in_depth, in_height, in_width]

    kernel : tvm.te.Tensor
        5-D with shape [num_filter, in_channel, filter_depth, filter_height, filter_width]

    strides : int or a list/tuple of three ints
        stride size, or [stride_depth, stride_height, stride_width]

    padding : int or a list/tuple of three ints
        padding size, or [pad_depth, pad_height, pad_width]

    dilation: int or a list/tuple of three ints
        dilation size, or [dilation_depth, dilation_height, dilation_width]

    layout : str
        layout of data

    out_dtype: str
        The output type. This is used for mixed precision.

    Returns
    -------
    output : tvm.te.Tensor
        5-D with shape [batch, out_channel, out_depth, out_height, out_width]
    """
    if layout == "NCDHW":
        tensor_format = 0  # CUDNN_TENSOR_NCHW
        N, _, D, H, W = get_const_tuple(data.shape)
    elif layout == "NDHWC":
        tensor_format = 1  # CUDNN_TENSOR_NHWC
        N, D, H, W, _ = get_const_tuple(data.shape)
    else:
        raise ValueError("Unsupported layout %s in cudnn" % layout)
    CO, CI, KD, KH, KW = get_const_tuple(kernel.shape)

    assert groups == 1, "conv3d_cudnn does not support groups"

    # handle dilation
    stride_d, stride_h, stride_w = (
        (strides, strides, strides) if isinstance(strides, int) else strides
    )
    pad_d, pad_h, pad_w = (padding, padding, padding) if isinstance(padding, int) else padding
    dilation_d, dilation_h, dilation_w = (
        (dilation, dilation, dilation) if isinstance(dilation, int) else dilation
    )

    OD = (D + 2 * pad_d - KD) // stride_d + 1
    OH = (H + 2 * pad_h - KH) // stride_h + 1
    OW = (W + 2 * pad_w - KW) // stride_w + 1

    if isinstance(N, int):
        cfg.add_flop(
            2
            * N
            * OD
            * OH
            * OW
            * CO
            * CI
            * ((KD - 1) * dilation_d + 1)
            * ((KH - 1) * dilation_h + 1)
            * ((KW - 1) * dilation_w + 1)
        )

    cfg.define_knob("algo", range(cudnn.algo_to_index("fwd", "CUDNN_CONVOLUTION_FWD_ALGO_COUNT")))
    if cfg.is_fallback:
        if cudnn.exists():
            # Let CUDNN choose the best algo, based on benchmarks run
            # on the local machine.  In the future, this should be
            # based on parameters stored in the Target.
            cfg["algo"] = OtherOptionEntity(-1)
        else:
            cfg["algo"] = OtherOptionEntity(0)

    return cudnn.conv_forward(
        data,
        kernel,
        [pad_d, pad_h, pad_w],
        [stride_d, stride_h, stride_w],
        [dilation_d, dilation_h, dilation_w],
        conv_mode=1,
        tensor_format=tensor_format,
        algo=cfg["algo"].val,
        conv_dtype=dtype,
    )


@autotvm.register_topi_schedule("conv3d_cudnn.cuda")
def schedule_conv3d_cudnn(_, outs):
    """TOPI schedule callback of conv3d for cuda gpu

    Parameters
    ----------
    cfg: ConfigEntity
        The config for this template

    outs: Array of Tensor
        The computation graph description of conv2d
        in the format of an array of tensors.

    Returns
    -------
    s: Schedule
        The computation schedule for conv2d.
    """
    return generic.schedule_extern(outs)
