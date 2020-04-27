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
"""Compute definition for conv2d with cuda backend"""
from tvm import te
from tvm import autotvm
from tvm.contrib import cudnn

from .. import nn, generic
from ..nn.util import get_pad_tuple
from ..util import get_const_tuple, traverse_inline
from .conv2d_direct import schedule_direct_cuda
from .conv2d_nhwc import schedule_conv2d_nhwc_direct


@autotvm.register_topi_compute("conv2d_nchw.cuda")
def conv2d_nchw(cfg, data, kernel, strides, padding, dilation, out_dtype='float32'):
    """Compute conv2d with NCHW layout"""
    return nn.conv2d_nchw(data, kernel, strides, padding, dilation, out_dtype)


@autotvm.register_topi_schedule("conv2d_nchw.cuda")
def schedule_conv2d_nchw(cfg, outs):
    """Create the schedule for conv2d_nchw"""
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        if op.tag == 'conv2d_nchw':
            schedule_direct_cuda(cfg, s, op.output(0))

    traverse_inline(s, outs[0].op, _callback)
    return s


@autotvm.register_topi_compute("conv2d_nhwc.cuda")
def conv2d_nhwc(cfg, data, kernel, strides, padding, dilation, out_dtype='float32'):
    """Compute conv2d with NHWC layout"""
    return nn.conv2d_nhwc(data, kernel, strides, padding, dilation, out_dtype)


@autotvm.register_topi_schedule("conv2d_nhwc.cuda")
def schedule_conv2d_nhwc(cfg, outs):
    """Create the schedule for conv2d_nhwc"""
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])
    def _callback(op):
        if op.tag == 'conv2d_nhwc':
            schedule_conv2d_nhwc_direct(cfg, s, op.output(0))
    traverse_inline(s, outs[0].op, _callback)
    return s


@autotvm.register_topi_compute("conv2d_cudnn.cuda")
def conv2d_cudnn(cfg, data, kernel, strides, padding, dilation, groups=1,
                 layout='NCHW', out_dtype='float32'):
    """Compute conv2d using CuDNN library"""
    if layout == 'NCHW':
        tensor_format = 0 # CUDNN_TENSOR_NCHW
        N, _, H, W = get_const_tuple(data.shape)
    elif layout == 'NHWC':
        tensor_format = 1 # CUDNN_TENSOR_NHWC
        N, H, W, _ = get_const_tuple(data.shape)
    else:
        raise ValueError("Unsupported layout %s in cudnn" % layout)
    CO, CI, KH, KW = get_const_tuple(kernel.shape)

    # handle dilation
    stride_h, stride_w = (strides, strides) if isinstance(strides, int) else strides
    dilation_h, dilation_w = (dilation, dilation) if isinstance(dilation, int) else dilation

    if isinstance(padding, (list, tuple)) and len(padding) == 4 and \
            (padding[0] != padding[2] or padding[1] != padding[3]):
        raise ValueError("Cudnn doesn't support asymmetric padding.")
    pt, pl, pb, pr = get_pad_tuple(padding, (KH, KW))
    OH = (H + pt + pb - KH) // stride_h + 1
    OW = (W + pl + pr - KW) // stride_w + 1
    cfg.add_flop(groups * 2 * N * OH * OW * CO * CI * ((KH - 1) * dilation_h + 1) * \
                 ((KW - 1) * dilation_w + 1))

    if data.dtype == "int8" or kernel.dtype == "int8":
        if layout == 'NCHW':
            raise ValueError("NCHW layout do not support int8 in cudnn")
        dtype = "int32"
    else:
        dtype = data.dtype

    return cudnn.conv_forward(data,
                              kernel,
                              [pt, pl], # cudnn padding pt, pl on both sides of input
                              [stride_h, stride_w],
                              [dilation_h, dilation_w],
                              conv_mode=1,
                              tensor_format=tensor_format,
                              algo=-1,         # let CUDNN choose the best algo
                              conv_dtype=dtype,
                              groups=groups)


@autotvm.register_topi_schedule("conv2d_cudnn.cuda")
def schedule_conv2d_cudnn(cfg, outs):
    """Create the schedule for conv2d_cudnn"""
    return generic.schedule_extern(outs)
