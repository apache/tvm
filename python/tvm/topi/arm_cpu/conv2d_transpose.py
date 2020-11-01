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
# pylint: disable=invalid-name, unused-variable
"""Transposed 2D convolution operators (sometimes called Deconvolution)."""
from __future__ import absolute_import as _abs

import tvm
from tvm import te
from tvm import autotvm

from ..nn import dilate, pad, get_pad_tuple
from ..utils import get_const_tuple, traverse_inline
from .conv2d_spatial_pack import schedule_conv2d_spatial_pack_nchw


@autotvm.register_topi_compute("conv2d_transpose_nchw.arm_cpu")
def conv2d_transpose_nchw(cfg, Input, Filter, strides, padding, out_dtype, output_padding):
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

    out_dtype: str
        The output data type. This is used for mixed precision.

    output_padding : tuple of int
        Used to get the right output shape in gradients

    Returns
    -------
    Output : tvm.te.Tensor
        4-D with shape [batch, out_channel, out_height, out_width]
    """
    return _decl_spatial_pack(
        cfg, Input, Filter, strides, padding, "NCHW", out_dtype, 2, output_padding
    )


def _decl_spatial_pack(
    cfg, data, kernel, strides, padding, layout, out_dtype, num_tile, output_padding
):
    assert layout == "NCHW", "Only support NCHW"
    out_dtype = out_dtype or data.dtype

    N, CI, IH, IW = get_const_tuple(data.shape)
    if isinstance(N, tvm.tir.Any):
        N = tvm.te.size_var("n")
    if not isinstance(IH, int) or not isinstance(IW, int):
        raise RuntimeError("ARM winograd conv2d doesn't support dynamic input height or width.")

    _, CO, KH, KW = get_const_tuple(kernel.shape)
    HSTR, WSTR = strides if isinstance(strides, (tuple, list)) else (strides, strides)
    opad_h, opad_w = output_padding
    assert opad_h < HSTR and opad_w < WSTR

    pad_top, pad_left, pad_bottom, pad_right = get_pad_tuple(padding, (KH, KW))
    bpad_top, bpad_bottom = KH - 1 - pad_top, KH - 1 - pad_bottom + opad_h
    bpad_left, bpad_right = KW - 1 - pad_left, KW - 1 - pad_right + opad_w

    OH = (IH - 1) * HSTR - pad_top - pad_bottom + KH + opad_h
    OW = (IW - 1) * WSTR - pad_left - pad_right + KW + opad_w

    dilated_input = dilate(data, [1, 1, HSTR, WSTR])
    data_pad = pad(dilated_input, [0, 0, bpad_top, bpad_left], [0, 0, bpad_bottom, bpad_right])

    # ==================== define configuration space ====================
    # TODO(@kevinthesun): Support tuning/optimization for dynamic shape.
    n_tuning_axis = N if isinstance(N, int) else 1
    n, co, oh, ow = cfg.axis(n_tuning_axis), cfg.axis(CO), cfg.axis(OH), cfg.axis(OW)
    ci, kh, kw = cfg.reduce_axis(CI), cfg.reduce_axis(KH), cfg.reduce_axis(KW)

    if num_tile == 2:  # for arm cpu
        co, vc = cfg.define_split("tile_co", co, num_outputs=2)
        oh, vh = cfg.define_split("tile_oh", oh, num_outputs=2)
        ow, vw = cfg.define_split("tile_ow", ow, num_outputs=2)
    elif num_tile == 3:  # for mali gpu
        co, _, vc = cfg.define_split("tile_co", co, num_outputs=3)
        oh, _, vh = cfg.define_split("tile_oh", oh, num_outputs=3)
        ow, _, vw = cfg.define_split("tile_ow", ow, num_outputs=3)
    else:
        raise RuntimeError("Invalid num_tile")

    cfg.define_reorder(
        "reorder_0",
        [n, co, oh, ow, ci, kh, kw, vh, vw, vc],
        policy="candidate",
        candidate=[
            [n, co, oh, ow, ci, kh, kw, vh, vw, vc],
            [n, co, oh, ow, ci, kh, kw, vc, vh, vw],
        ],
    )

    cfg.define_annotate("ann_reduce", [kh, kw], policy="try_unroll")
    cfg.define_annotate("ann_spatial", [vh, vw, vc], policy="try_unroll_vec")
    # ====================================================================

    VC = cfg["tile_co"].size[-1]
    VH = cfg["tile_oh"].size[-1]
    VW = cfg["tile_ow"].size[-1]

    dvshape = (N, OH // VH, OW // VW, CI, VH + KH - 1, VW + KW - 1)
    kvshape = (CO // VC, CI, KH, KW, VC)
    ovshape = (N, CO // VC, OH // VH, OW // VW, VH, VW, VC)
    oshape = (N, CO, OH, OW)

    data_vec = te.compute(
        dvshape,
        lambda n, h, w, ci, vh, vw: data_pad[n][ci][h * VH + vh][w * VW + vw],
        name="data_vec",
    )

    kernel_vec = te.compute(
        kvshape,
        lambda co, ci, kh, kw, vc: kernel[ci][co * VC + vc][kh][kw],
        name="kernel_vec_conv2d_transpose",
    )

    ci = te.reduce_axis((0, CI), name="ci")
    kh = te.reduce_axis((0, KH), name="kh")
    kw = te.reduce_axis((0, KW), name="kw")

    conv = te.compute(
        ovshape,
        lambda n, co, h, w, vh, vw, vc: te.sum(
            data_vec[n, h, w, ci, vh + kh, vw + kw].astype(out_dtype)
            * kernel_vec[co, ci, KH - 1 - kh, KW - 1 - kw, vc].astype(out_dtype),
            axis=[ci, kh, kw],
        ),
        name="conv",
    )

    idxdiv = tvm.tir.indexdiv
    idxmod = tvm.tir.indexmod

    output = te.compute(
        oshape,
        lambda n, co, h, w: conv[
            n,
            idxdiv(co, VC),
            idxdiv(h, VH),
            idxdiv(w, VW),
            idxmod(h, VH),
            idxmod(w, VW),
            idxmod(co, VC),
        ],
        name="output_unpack",
        tag="spatial_conv2d_transpose_output",
    )
    return output


# register customized schedule for arm cpu.
@autotvm.register_topi_schedule("conv2d_transpose_nchw.arm_cpu")
def schedule_conv2d_transpose_nchw(cfg, outs):
    """Schedule conv2d transpose for arm cpu"""
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        if "spatial_conv2d_transpose_output" in op.tag:
            output = op.output(0)
            conv = op.input_tensors[0]

            data_vec = conv.op.input_tensors[0]
            data_pad = data_vec.op.input_tensors[0]
            dilated_input = data_pad.op.input_tensors[0]
            s[data_pad].compute_inline()
            s[dilated_input].compute_inline()

            kernel_vec = conv.op.input_tensors[1]
            if kernel_vec.op.name == "kernel_vec":
                kernel = kernel_vec.op.input_tensors[0]
            else:
                kernel = kernel_vec
            if isinstance(kernel.op, tvm.te.ComputeOp) and "dilate" in kernel.op.tag:
                s[kernel].compute_inline()

            schedule_conv2d_spatial_pack_nchw(cfg, s, data_vec, kernel_vec, conv, output, outs[0])

    traverse_inline(s, outs[0].op, _callback)
    return s
