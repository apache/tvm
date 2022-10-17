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
"""Schedule for conv2d"""

import tvm
from tvm import te
from .. import nn
from ..utils import traverse_inline
from .tensor_intrin import dot_vrmpy
from ..generic import conv2d as conv2d_generic


def schedule_conv2d_nhwc(outs):
    """Schedule for conv2d NHWC operator.

    Parameters
    ----------
    outs: Array of Tensor
        The computation graph description of conv2d in the format
        of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    outs = [outs] if isinstance(outs, tvm.te.tensor.Tensor) else outs
    s = tvm.te.create_schedule([x.op for x in outs])
    tvm.te.schedule.AutoInlineInjective(s)
    return s


def schedule_conv2d_nchw(outs):
    return schedule_conv2d_nhwc(outs)


def schedule_conv2d(outs, layout="NHWC"):
    layout_uncase = layout.casefold()
    if layout_uncase == "NHWC".casefold():
        return schedule_conv2d_nhwc(outs)
    if layout_uncase == "NCHW".casefold():
        return schedule_conv2d_nchw(outs)

    raise ValueError(f"Unexpected layout={layout}")


def schedule_depthwise_conv2d_nchw(outs):
    return schedule_conv2d_nchw(outs)


def schedule_depthwise_conv2d_nhwc(out):
    return schedule_conv2d_nhwc(out)


def schedule_conv2d_transpose_nchw(outs):
    """Create schedule for tensors"""
    outs = [outs] if isinstance(outs, tvm.te.tensor.Tensor) else outs
    s = schedule_conv2d_nchw(outs)

    def _callback(op):
        if "unpack_nchwc" in op.tag:
            conv_out = op.input_tensors[0]
            # retrieve data
            data_vec = conv_out.op.input_tensors[0]
            if isinstance(data_vec, tvm.te.ComputeOp):
                data_pad = data_vec.op.input_tensors[0]
                data_dilate = data_pad.op.input_tensors[0]
                s[data_dilate].compute_inline()
                s[data_pad].compute_inline()
            # retrieve kernel
            kernel_vec = conv_out.op.input_tensors[1]
            if isinstance(kernel_vec, tvm.te.ComputeOp):
                kernel_transform = kernel_vec.op.input_tensors[0]
                s[kernel_transform].compute_inline()

    traverse_inline(s, outs[0].op, _callback)
    return s


def conv2d_NCHWc_int8(
    data, kernel, stride, padding, dilation, layout, out_layout, out_dtype="int32"
):
    """Compute definition for int8 conv2d in NCHWc layout"""
    n_elems = int(kernel.shape[-1])
    return nn.conv2d_NCHWc_int8(
        data, kernel, stride, padding, dilation, layout, out_layout, out_dtype, n_elems=n_elems
    )


def schedule_conv2d_NCHWc_int8(outs):
    """Schedule for int8 conv2d in NCHWc layout using vrmpy tensorization"""
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        if "conv2d_NCHWc_int8" in op.tag:
            conv_out = op.output(0)
            kernel_vec = conv_out.op.input_tensors[1]
            data_vec = conv_out.op.input_tensors[0]
            out_width = conv_out.shape[3]

            reg_n = 1
            for n in range(31, 0, -1):
                if out_width % n == 0:
                    reg_n = n
                    break

            cfg = {"tile_ow": reg_n, "unroll_kw": False}
            args = [s, cfg, data_vec, kernel_vec, conv_out, outs[0]]
            intrin = dot_vrmpy(data_vec.dtype, kernel_vec.dtype)

            conv2d_generic.schedule_conv_NCHWc_cpu_common_int8(
                *args,
                int32_lanes=32,
                int8_elems=4,
                intrin=intrin,
                inline_fused=True,
            )

    traverse_inline(s, outs[0].op, _callback)
    return s
