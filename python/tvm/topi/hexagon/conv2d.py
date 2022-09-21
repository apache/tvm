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

"""Schedule for conv2d"""

import tvm
from tvm import te
from tvm.topi.nn.pad import pad
from .. import nn
from ..utils import traverse_inline
from tvm.topi.utils import get_const_tuple
from tvm.topi.nn.utils import get_pad_tuple
from .tensor_intrin import dot_vrmpy


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
    n_elems = int(kernel.shape[-1])
    return nn.conv2d_NCHWc_int8(
        data, kernel, stride, padding, dilation, layout, out_layout, out_dtype, n_elems=n_elems
    )


def schedule_conv2d_NCHWc_int8(outs):
    """Create schedule for tensors"""
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        if "conv2d_NCHWc_int8" in op.tag:
            conv_out = op.output(0)
            kernel_vec = conv_out.op.input_tensors[1]
            data_vec = conv_out.op.input_tensors[0]
            data = (
                data_vec.op.input_tensors[0]
                if isinstance(data_vec.op, te.tensor.ComputeOp) and "pad" not in data_vec.op.tag
                else data_vec
            )
            if isinstance(data.op, te.tensor.ComputeOp) and "pad" in data.op.tag:
                data_pad = data
                data = data_pad.op.input_tensors[0]

            out_width = conv_out.shape[3]
            reg_n = 1
            for n in range(31, 0, -1):
                if out_width % n == 0:
                    reg_n = n
                    break

            args = [s, data_vec, conv_out, outs[0]]
            # int8 conv kernel is 7-dim
            _, _, kh, kw, _, _, n_elems = get_const_tuple(kernel_vec.shape)
            # assert n_elems == 4
            intrin = dot_vrmpy(data.dtype, kernel_vec.dtype)

            inline_fused = True

            schedule_conv_NCHWc_cpu_common_int8(
                *args, reg_n=reg_n, int32_lanes=32, int8_elems=4, intrin=intrin, inline_fused=inline_fused
            )

    traverse_inline(s, outs[0].op, _callback)
    return s


def schedule_conv_NCHWc_cpu_common_int8(
    s,
    data_vec,
    conv_out,
    last,
    reg_n,
    int32_lanes=32,
    int8_elems=4,
    intrin=None,
    inline_fused=True,
):
    unroll_kw = False
    _, _, _, _, ic_bn = get_const_tuple(data_vec.shape)
    _, _, _, _, oc_bn = get_const_tuple(conv_out.shape)

    # schedule pad
    if isinstance(s[data_vec].op, te.tensor.ComputeOp) and "pad" in data_vec.op.tag:
        batch, ic_chunk, ih, iw, ic_block = s[data_vec].op.axis
        parallel_axis = s[data_vec].fuse(batch, ic_chunk, ih)
        # s[data_vec].parallel(parallel_axis)
        data_vec = data_vec.op.input_tensors[0]

    # schedule 5-D NCHW[x]c conv
    C, O = conv_out, last
    CC = s.cache_write(C, "global")

    batch, oc_chunk, oh, ow, oc_block = s[C].op.axis
    ow_chunk, ow_block = s[C].split(ow, factor=reg_n)
    s[C].reorder(oc_chunk, oh, ow_chunk, ow_block, oc_block)
    parallel_axis = s[C].fuse(batch, oc_chunk, oh)
    s[C].vectorize(oc_block)

    if C == O:
        s[C].parallel(parallel_axis)

    s[CC].compute_at(s[C], parallel_axis)
    _, oc_chunk, oh, ow, oc_block = s[CC].op.axis
    kh, kw, ic_outer, ic_f_inner, ic_s_inner = s[CC].op.reduce_axis

    ow_chunk, ow_block = s[CC].split(ow, factor=reg_n)

    assert oc_bn % int32_lanes == 0, f"oc_bn={oc_bn} % int32_lanes={int32_lanes} != 0"
    assert (
        ic_bn % int8_elems == 0
    ), f"ic_bn={ic_bn} % int8_elems={int8_elems} != 0"  # (u)int8 elements in (u)int32

    oc_f_inner, oc_s_inner = s[CC].split(oc_block, factor=int32_lanes)

    if unroll_kw:
        s[CC].reorder(
            oc_chunk,
            oh,
            ow_chunk,
            ic_outer,
            kh,
            ic_f_inner,
            kw,
            ow_block,
            oc_f_inner,
            oc_s_inner,
            ic_s_inner,
        )
        s[CC].unroll(kw)
    else:
        s[CC].reorder(
            oc_chunk,
            oh,
            ow_chunk,
            ic_outer,
            kh,
            kw,
            ic_f_inner,
            ow_block,
            oc_f_inner,
            oc_s_inner,
            ic_s_inner,
        )

    s[CC].tensorize(oc_s_inner, intrin)

    s[CC].unroll(ow_block)
    s[CC].unroll(oc_f_inner)

    if C != O:
        batch, oc_chunk, oh, ow, oc_block = s[O].op.axis
        ow_chunk, ow_block = s[O].split(ow, factor=reg_n)
        s[O].reorder(oc_chunk, oh, ow_chunk, ow_block, oc_block)
        parallel_axis = s[O].fuse(batch, oc_chunk, oh)

        if inline_fused:
            s[C].compute_at(s[O], ow_block)
        else:
            s[C].compute_at(s[O], parallel_axis)
        s[O].vectorize(oc_block)
        s[O].parallel(parallel_axis)

    return s
