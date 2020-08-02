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
"""Compute definition for conv2d with rocm backend"""
from tvm import autotvm
from tvm.contrib import miopen

from .. import generic
from ..util import get_const_tuple
from ..nn.util import get_pad_tuple

@autotvm.register_topi_compute("conv2d_nchw_miopen.rocm")
def conv2d_nchw_miopen(cfg, data, kernel, strides, padding, dilation,
                       layout='NCHW', out_dtype='float32'):
    """Conv2D operator for rocm backend.

    Parameters
    ----------
    cfg: ConfigEntity
        The config for this template

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

    layout : str
        layout of data

    Returns
    -------
    output : tvm.te.Tensor
        4-D with shape [batch, out_channel, out_height, out_width]
    """

    CO, CI, KH, KW = get_const_tuple(kernel.shape)
    N, _, H, W = get_const_tuple(data.shape)

    assert layout == 'NCHW'

    # handle dilation
    stride_h, stride_w = (strides, strides) if isinstance(strides, int) else strides
    pt, pl, pb, pr = get_pad_tuple(padding, (KH, KW))
    pad_h, pad_w = pt + pb, pl + pr
    dilation_h, dilation_w = (dilation, dilation) if isinstance(dilation, int) else dilation
    assert (pt == pb) and (pl == pr)
    OH = (H + 2 * pad_h - KH) // stride_h + 1
    OW = (W + 2 * pad_w - KW) // stride_w + 1
    cfg.add_flop(2 * N * OH * OW * CO * CI * ((KH - 1) * dilation_h + 1) *\
                 ((KW - 1) * dilation_w + 1))

    return miopen.conv2d_forward(data,
                                 kernel,
                                 stride_h,
                                 stride_w,
                                 pt,
                                 pl,
                                 dilation_h,
                                 dilation_w,
                                 conv_mode=0,
                                 data_type=1)


@autotvm.register_topi_schedule("conv2d_nchw_miopen.rocm")
def schedule_conv2d_nchw_miopen(cfg, outs):
    """TOPI schedule callback of conv2d for rocm

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
