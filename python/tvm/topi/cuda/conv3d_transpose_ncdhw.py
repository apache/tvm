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
"""Conv3d transpose template for cuda backend"""

import tvm
from tvm import te
from tvm import autotvm
from .. import nn
from ..utils import get_const_tuple, traverse_inline
from .conv3d_direct import schedule_direct_conv3d_cuda


@autotvm.register_topi_compute("conv3d_transpose_ncdhw.cuda")
def conv3d_transpose_ncdhw(cfg, data, kernel, stride, padding, out_dtype, output_padding):
    """Transposed 3D convolution ncdhw forward operator.

    Parameters
    ----------
    cfg: ConfigEntity
        The config for this template
    Input : tvm.te.Tensor
        5-D with shape [batch, in_channel, in_depth, in_height, in_width]
    Filter : tvm.te.Tensor
        5-D with shape [in_channel, num_filter, filter_depth, filter_height, filter_width]
    strides : int or a list/tuple of three ints
        The spatial stride along height and width
    padding : int or str
        Padding size, or ['VALID', 'SAME']
    out_dtype: str
        The output type. This is used in mixed precision
    output_padding : tuple of three ints
        Used to disambiguate output shape

    Returns
    -------
    Output : tvm.te.Tensor
        5-D with shape [batch, out_channel, out_depth, out_height, out_width]
    """
    batch, inp_channels, inp_depth, inp_height, inp_width = get_const_tuple(data.shape)
    _, out_channels, kernel_depth, kernel_height, kernel_width = get_const_tuple(kernel.shape)
    stride_depth, stride_height, stride_width = stride
    outpad_depth, outpad_height, outpad_width = output_padding
    assert (
        outpad_height < stride_height
        and outpad_width < stride_width
        and outpad_depth < stride_depth
    )
    cfg.stride = stride
    pad_front, pad_top, pad_left, pad_back, pad_bottom, pad_right = nn.get_pad_tuple3d(
        padding, (kernel_depth, kernel_height, kernel_width)
    )

    out_depth = (inp_depth - 1) * stride_depth + kernel_depth - pad_front - pad_back + outpad_depth
    pad_front = kernel_depth - 1 - pad_front
    pad_back = kernel_depth - 1 - pad_back
    dilated_depth = stride_depth * (inp_depth - 1) + 1

    out_width = (inp_width - 1) * stride_width + kernel_width - pad_left - pad_right + outpad_width
    pad_left = kernel_width - 1 - pad_left
    pad_right = kernel_width - 1 - pad_right
    dilated_width = stride_width * (inp_width - 1) + 1

    out_height = (
        (inp_height - 1) * stride_height + kernel_height - pad_top - pad_bottom + outpad_height
    )
    pad_top = kernel_height - 1 - pad_top
    pad_bottom = kernel_height - 1 - pad_bottom
    dilated_height = stride_height * (inp_height - 1) + 1

    # compute pad
    data = te.compute(
        (
            batch,
            inp_channels,
            pad_front + dilated_depth + pad_back,
            pad_top + dilated_height + pad_bottom,
            pad_left + dilated_width + pad_right,
        ),
        lambda n, c, d, y, x: tvm.tir.if_then_else(
            tvm.tir.all(
                x >= pad_left,
                x < pad_left + dilated_width,
                tvm.tir.indexmod(x - pad_left, stride_width).equal(0),
                y >= pad_top,
                y < pad_top + dilated_height,
                tvm.tir.indexmod(y - pad_top, stride_height).equal(0),
                d >= pad_front,
                d < pad_front + dilated_depth,
                tvm.tir.indexmod(d - pad_front, stride_depth).equal(0),
            ),
            data[
                n,
                c,
                tvm.tir.indexdiv(d - pad_front, stride_depth),
                tvm.tir.indexdiv(y - pad_top, stride_height),
                tvm.tir.indexdiv(x - pad_left, stride_width),
            ],
            tvm.tir.const(0.0, "float32"),
        ),
        name="data_pad",
    )

    # compute transposed conv
    dc = te.reduce_axis((0, inp_channels), name="dc")
    dd = te.reduce_axis((0, kernel_depth), name="dd")
    dh = te.reduce_axis((0, kernel_height), name="dh")
    dw = te.reduce_axis((0, kernel_width), name="dw")
    data_out = te.compute(
        (batch, out_channels, out_depth, out_height, out_width),
        lambda b, c, d, h, w: te.sum(
            data[b, dc, d + dd, h + dh, w + dw].astype(out_dtype)
            * kernel[
                dc, c, kernel_depth - 1 - dd, kernel_height - 1 - dh, kernel_width - 1 - dw
            ].astype(out_dtype),
            axis=[dc, dd, dh, dw],
        ),
        tag="conv3d_transpose_ncdhw",
    )

    return data_out


@autotvm.register_topi_schedule("conv3d_transpose_ncdhw.cuda")
def schedule_conv3d_transpose_ncdhw(cfg, outs):
    """TOPI Schedule callback for conv3d transpose operator.

    Parameters
    ----------
    cfg: ConfigEntity
        The parameters for this template

    outs: Array of Tensor
        The computation graph description of conv3d transpose
        in the format of an array of tensors.

    Returns
    -------
    s: Schedule
        The computation schedule for conv3d transpose.
    """
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        if op.tag == "conv3d_transpose_ncdhw":
            schedule_direct_conv3d_cuda(
                cfg, s, op.output(0), "NCDHW", "conv3d_transpose_ncdhw.cuda"
            )

    traverse_inline(s, outs[0].op, _callback)
    return s
