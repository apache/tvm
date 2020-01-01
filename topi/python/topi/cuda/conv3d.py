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
# pylint: disable=invalid-name
"""Compute definition for conv3d with cuda backend"""
import tvm
from tvm import autotvm
from tvm.contrib import cudnn

from .. import nn, generic
from ..nn.util import get_pad_tuple3d
from ..util import get_const_tuple, traverse_inline

from .conv3d_direct import schedule_direct_3d_cuda


@autotvm.register_topi_compute(nn.conv3d, ['cuda', 'gpu'], ['direct'])
def conv3d_cuda(cfg, data, kernel, strides, padding, dilation, layout='NCDHW', out_dtype='float32'):
    """Conv3D operator for cuda backend.

    Parameters
    ----------
    cfg: ConfigEntity
        The config for this template

    data : tvm.Tensor
        5-D with shape [batch, in_channel, in_depth, in_height, in_width]

    kernel : tvm.Tensor
        5-D with shape [num_filter, in_channel, filter_depth, filter_height, filter_width]

    strides : int or a list/tuple of three ints
        stride size, or [stride_depth, stride_height, stride_width]

    padding : int or a list/tuple of 3 or 6 ints
        padding size, or
        [pad_depth, pad_height, pad_width] for 3 ints, or
        [pad_front, pad_top, pad_left, pad_back, pad_bottom, pad_right] for 6 ints

    dilation: int or a list/tuple of three ints
        dilation size, or [dilation_depth, dilation_height, dilation_width]

    layout : str
        layout of data

    out_dtype: str
        The output type. This is used for mixed precision.

    Returns
    -------
    output : tvm.Tensor
        5-D with shape [batch, out_channel, out_depth, out_height, out_width]
    """
    target = tvm.target.current_target()

    if "cudnn" in target.libs:
        if layout == 'NCDHW':
            tensor_format = 0 # CUDNN_TENSOR_NCHW
            N, _, D, H, W = get_const_tuple(data.shape)
        elif layout == 'NDHWC':
            tensor_format = 1 # CUDNN_TENSOR_NHWC
            N, D, H, W, _ = get_const_tuple(data.shape)
        else:
            raise ValueError("Unsupported layout %s in cudnn" % layout)
        CO, CI, KD, KH, KW = get_const_tuple(kernel.shape)

        # handle dilation
        stride_d, stride_h, stride_w = (strides, strides, strides) if isinstance(strides, int) \
            else strides
        if isinstance(padding, (list, tuple)) and len(padding) > 3:
            raise ValueError("Cudnn doesn't support asymmetric padding.")
        pf, pt, pl, pk, pb, pr = get_pad_tuple3d(padding, (KD, KH, KW))
        dilation_d, dilation_h, dilation_w = (dilation, dilation, dilation) if \
            isinstance(dilation, int) else dilation

        OD = (D + pf + pk - KD) // stride_d + 1
        OH = (H + pt + pb - KH) // stride_h + 1
        OW = (W + pl + pr - KW) // stride_w + 1
        cfg.add_flop(2 * N * OD * OH * OW * CO * CI * ((KD - 1) * dilation_d + 1) *\
                    ((KH - 1) * dilation_h + 1) * ((KW - 1) * dilation_w + 1))

        return cudnn.conv_forward(data,
                                  kernel,
                                  [pf, pt, pl],  # cudnn padding pt, pl on both sides of input
                                  [stride_d, stride_h, stride_w],
                                  [dilation_d, dilation_h, dilation_w],
                                  conv_mode=1,
                                  tensor_format=tensor_format,
                                  algo=-1,         # let CUDNN choose the best algo
                                  conv_dtype=data.dtype)

    if layout == 'NCDHW':
        return nn.conv3d_ncdhw(data, kernel, strides, padding, dilation, out_dtype)
    raise ValueError("not support this layout {} yet".format(layout))


@autotvm.register_topi_schedule(generic.schedule_conv3d_ncdhw, ["cuda", "gpu"],
                                ["direct"])
def schedule_conv3d_ncdhw_cuda(cfg, outs):
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
    target = tvm.target.current_target()
    if 'cudnn' in target.libs:
        return generic.schedule_extern(outs)

    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    s = tvm.create_schedule([x.op for x in outs])

    def _callback(op):
        if op.tag == 'conv3d_ncdhw':
            schedule_direct_3d_cuda(cfg, s, op.output(0))

    traverse_inline(s, outs[0].op, _callback)
    return s


@autotvm.register_topi_schedule(generic.schedule_conv3d_ndhwc, ["cuda", "gpu"],
                                ["direct"])
def schedule_conv3d_ndhwc_cuda(cfg, outs):
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
    target = tvm.target.current_target()
    if 'cudnn' in target.libs:
        return generic.schedule_extern(outs)

    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    s = tvm.create_schedule([x.op for x in outs])

    def _callback(op):
        if op.tag == 'conv3d_ndhwc':
            schedule_direct_3d_cuda(cfg, s, op.output(0))

    traverse_inline(s, outs[0].op, _callback)
    return s
