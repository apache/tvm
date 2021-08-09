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
# pylint: disable=invalid-name,unused-variable,no-else-return
"""Conv2D spatial pack implementation for ARM CPU"""
from __future__ import absolute_import as _abs
import tvm
from tvm import te
from tvm import autotvm
from .. import nn
from ..utils import get_const_tuple
from ..nn.utils import get_const_int, get_pad_tuple


def conv2d_spatial_pack_nchw(cfg, data, kernel, strides, padding, dilation, out_dtype, num_tile):
    """compute define for Conv2d Spatial Pack with NCHW layout"""
    out_dtype = out_dtype or data.dtype
    N, CI, IH, IW = get_const_tuple(data.shape)
    if isinstance(N, tvm.tir.Any):
        N = tvm.te.size_var("n")
    if not isinstance(IH, int) or not isinstance(IW, int):
        raise RuntimeError("ARM winograd conv2d doesn't support dynamic input height or width.")

    if isinstance(dilation, int):
        dilation_h = dilation_w = dilation
    else:
        dilation_h, dilation_w = dilation

    if len(kernel.shape) == 4:
        pre_packed = False
        CO, _, KH, KW = get_const_tuple(kernel.shape)
    else:  # kernel tensor is pre packed
        pre_packed = True
        CO, _, KH, KW, VC = get_const_tuple(kernel.shape)
        CO = CO * VC

    dilated_kernel_h = (KH - 1) * dilation_h + 1
    dilated_kernel_w = (KW - 1) * dilation_w + 1
    pad_top, pad_left, pad_bottom, pad_right = get_pad_tuple(
        padding, (dilated_kernel_h, dilated_kernel_w)
    )
    HSTR, WSTR = strides if isinstance(strides, (tuple, list)) else (strides, strides)
    OH = (IH + pad_top + pad_bottom - dilated_kernel_h) // HSTR + 1
    OW = (IW + pad_left + pad_right - dilated_kernel_w) // WSTR + 1
    data_pad = nn.pad(data, [0, 0, pad_top, pad_left], [0, 0, pad_bottom, pad_right])

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

    # fallback support
    if cfg.is_fallback:
        if num_tile == 2:  # arm cpu
            ref_log = autotvm.tophub.load_reference_log(
                "arm_cpu", "rk3399", "conv2d_nchw_spatial_pack.arm_cpu"
            )
            cfg.fallback_with_reference_log(ref_log)
        elif num_tile == 3:  # mali gpu
            ref_log = autotvm.tophub.load_reference_log(
                "mali", "rk3399", "conv2d_nchw_spatial_pack.mali"
            )
            cfg.fallback_with_reference_log(ref_log)
    # ====================================================================

    VC = cfg["tile_co"].size[-1]
    VH = cfg["tile_oh"].size[-1]
    VW = cfg["tile_ow"].size[-1]

    kvshape = (CO // VC, CI, KH, KW, VC)
    ovshape = (N, CO // VC, OH // VH, OW // VW, VH, VW, VC)
    oshape = (N, CO, OH, OW)

    if dilation_h != 1 or dilation_w != 1:
        # undilate input data
        dvshape = (N, OH // VH, OW // VW, CI, KH, KW, VH, VW)
        data_vec = te.compute(
            dvshape,
            lambda n, h, w, ci, kh, kw, vh, vw: data_pad[n][ci][
                (h * VH + vh) * HSTR + kh * dilation_h
            ][(w * VW + vw) * WSTR + kw * dilation_w],
            name="data_vec_undilated",
        )
    else:
        dvshape = (N, OH // VH, OW // VW, CI, VH * HSTR + KH - 1, VW * WSTR + KW - 1)
        data_vec = te.compute(
            dvshape,
            lambda n, h, w, ci, vh, vw: data_pad[n][ci][h * VH * HSTR + vh][w * VW * WSTR + vw],
            name="data_vec",
        )

    if autotvm.GLOBAL_SCOPE.in_tuning:
        # use "kernel_autotvm" instead of "kernel" to avoid naming conflict with OpenCL keyword
        kernel_vec = tvm.te.placeholder(kvshape, kernel.dtype, name="kernel_autotvm")
    else:
        if pre_packed:
            kernel_vec = kernel
        else:
            kernel_vec = te.compute(
                kvshape,
                lambda co, ci, kh, kw, vc: kernel[co * VC + vc][ci][kh][kw],
                name="kernel_vec",
            )

    ci = te.reduce_axis((0, CI), name="ci")
    kh = te.reduce_axis((0, KH), name="kh")
    kw = te.reduce_axis((0, KW), name="kw")

    if dilation_h != 1 or dilation_w != 1:
        conv = te.compute(
            ovshape,
            lambda n, co, h, w, vh, vw, vc: te.sum(
                data_vec[n, h, w, ci, kh, kw, vh, vw].astype(out_dtype)
                * kernel_vec[co, ci, kh, kw, vc].astype(out_dtype),
                axis=[ci, kh, kw],
            ),
            name="conv",
        )
    else:
        conv = te.compute(
            ovshape,
            lambda n, co, h, w, vh, vw, vc: te.sum(
                data_vec[n, h, w, ci, vh * HSTR + kh, vw * WSTR + kw].astype(out_dtype)
                * kernel_vec[co, ci, kh, kw, vc].astype(out_dtype),
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
        tag="spatial_conv2d_output",
    )
    return output


def schedule_conv2d_spatial_pack_nchw(cfg, s, data_vec, kernel_vec, conv, output, last):
    """schedule implementation"""
    n, co, oh, ow, vh, vw, vc = s[conv].op.axis
    ci, kh, kw = s[conv].op.reduce_axis

    # schedule conv
    cfg["reorder_0"].apply(s, conv, [n, co, oh, ow, ci, kh, kw, vh, vw, vc])
    cfg["ann_reduce"].apply(
        s,
        conv,
        [kh, kw],
        axis_lens=[get_const_int(kh.dom.extent), get_const_int(kw.dom.extent)],
        max_unroll=None,
        cfg=cfg,
    )
    cfg["ann_spatial"].apply(
        s,
        conv,
        [vh, vw, vc],
        axis_lens=[cfg["tile_oh"].size[-1], cfg["tile_ow"].size[-1], cfg["tile_co"].size[-1]],
        max_unroll=None,
        cfg=cfg,
    )

    # schedule fusion
    n, co, h, w = s[last].op.axis
    co, vc = cfg["tile_co"].apply(s, last, co)
    oh, vh = cfg["tile_oh"].apply(s, last, h)
    ow, vw = cfg["tile_ow"].apply(s, last, w)
    s[last].reorder(n, co, oh, ow, vh, vw, vc)
    if last != output:
        s[output].compute_inline()
        cfg["ann_spatial"].apply(
            s,
            last,
            [vh, vw, vc],
            axis_lens=[cfg["tile_oh"].size[-1], cfg["tile_ow"].size[-1], cfg["tile_co"].size[-1]],
            max_unroll=16,
            cfg=cfg,
        )
    s[conv].compute_at(s[last], ow)

    # mark parallel
    s[last].parallel(co)

    if data_vec.op.name == "data_vec_undilated":
        _, h, _, _, _, _, _, _ = s[data_vec].op.axis
    else:
        _, h, _, _, _, _ = s[data_vec].op.axis
    s[data_vec].parallel(h)

    if kernel_vec.op.name == "kernel_vec":
        if not autotvm.GLOBAL_SCOPE.in_tuning:
            co, _, _, _, _ = s[kernel_vec].op.axis
            s[kernel_vec].parallel(co)
    elif kernel_vec.op.name == "kernel_vec_conv2d_transpose":  # for conv2d transpose
        co, _, _, _, _ = s[kernel_vec].op.axis
        s[kernel_vec].parallel(co)

    return s


def conv2d_spatial_pack_nhwc(cfg, data, kernel, strides, padding, dilation, out_dtype, num_tile=2):
    """Spatial pack compute for Conv2d NHWC"""
    out_dtype = out_dtype or data.dtype

    N, IH, IW, IC = get_const_tuple(data.shape)
    assert len(kernel.shape) == 4, "AlterOpLayout not enabled for NHWC yet"
    KH, KW, _, OC = get_const_tuple(kernel.shape)

    if isinstance(dilation, int):
        dilation_h = dilation_w = dilation
    else:
        dilation_h, dilation_w = dilation

    dilated_kernel_h = (KH - 1) * dilation_h + 1
    dilated_kernel_w = (KW - 1) * dilation_w + 1

    pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
        padding, (dilated_kernel_h, dilated_kernel_w)
    )
    HSTR, WSTR = strides if isinstance(strides, (tuple, list)) else (strides, strides)

    OH = (IH + pad_top + pad_down - dilated_kernel_h) // HSTR + 1
    OW = (IW + pad_left + pad_right - dilated_kernel_w) // WSTR + 1
    data_pad = nn.pad(data, [0, pad_top, pad_left, 0], [0, pad_down, pad_right, 0])

    # ==================== define configuration space ====================
    # If it has dynamic shape in batch, we fix the split factor to 1
    n = cfg.axis(N) if isinstance(N, int) else cfg.axis(1)
    oc, oh, ow = cfg.axis(OC), cfg.axis(OH), cfg.axis(OW)
    ic, kh, kw = cfg.reduce_axis(IC), cfg.reduce_axis(KH), cfg.reduce_axis(KW)

    if num_tile == 2:  # for arm cpu
        oco, oci = cfg.define_split("tile_co", oc, num_outputs=2)
        oho, ohi = cfg.define_split("tile_oh", oh, num_outputs=2)
        owo, owi = cfg.define_split("tile_ow", ow, num_outputs=2)
    elif num_tile == 3:  # for mali gpu
        oco, _, oci = cfg.define_split("tile_co", oc, num_outputs=3)
        oho, _, ohi = cfg.define_split("tile_oh", oh, num_outputs=3)
        owo, _, owi = cfg.define_split("tile_ow", ow, num_outputs=3)
    else:
        raise RuntimeError("Invalid num_tile")

    cfg.define_reorder(
        "reorder_conv",
        [n, oho, owo, oco, kh, kw, ic, ohi, owi, oci],
        policy="candidate",
        candidate=[
            [n, oho, owo, oco, kh, kw, ic, ohi, owi, oci],
            [n, oho, owo, oco, ohi, kh, kw, ic, owi, oci],
            [n, oho, owo, oco, ohi, kh, kw, owi, ic, oci],
            [n, oho, owo, ohi, oco, kh, kw, owi, ic, oci],
        ],
    )

    cfg.define_annotate("ann_reduce", [kh, kw], policy="try_unroll")
    cfg.define_annotate("ann_spatial", [ohi, owi, oci], policy="try_unroll_vec")
    # ====================================================================

    OCI = cfg["tile_co"].size[-1]
    OHI = cfg["tile_oh"].size[-1]
    OWI = cfg["tile_ow"].size[-1]
    OCO = OC // OCI
    OHO = OH // OHI
    OWO = OW // OWI

    kvshape = (OCO, KH, KW, IC, OCI)
    ovshape = (N, OHO, OWO, OCO, OHI, OWI, OCI)
    oshape = (N, OH, OW, OC)

    if dilation_h != 1 or dilation_w != 1:
        # undilate input data
        dvshape = (N, OHO, OWO, KH, KW, IC, OHI, OWI)
        data_vec = te.compute(
            dvshape,
            lambda n, oho, owo, kh, kw, ic, ohi, owi: data_pad[n][
                (oho * OHI + ohi) * HSTR + kh * dilation_h
            ][(owo * OWI + owi) * WSTR + kw * dilation_w][ic],
            name="data_vec_undilated",
        )
    else:
        dvshape = (N, OHO, OWO, KH + (OHI - 1) * HSTR, KW + (OWI - 1) * WSTR, IC)
        data_vec = te.compute(
            dvshape,
            lambda n, oho, owo, ohi, owi, ic: data_pad[n][oho * OHI * HSTR + ohi][
                owo * OWI * WSTR + owi
            ][ic],
            name="data_vec",
        )

    if autotvm.GLOBAL_SCOPE.in_tuning:
        kernel_vec = tvm.te.placeholder(kvshape, kernel.dtype, name="kernel")
    else:
        kernel_vec = te.compute(
            kvshape,
            lambda oco, kh, kw, ic, oci: kernel[kh][kw][ic][oco * OCI + oci],
            name="kernel_vec",
        )

    ic = te.reduce_axis((0, IC), name="ic")
    kh = te.reduce_axis((0, KH), name="kh")
    kw = te.reduce_axis((0, KW), name="kw")

    if dilation_h != 1 or dilation_w != 1:
        conv = te.compute(
            ovshape,
            lambda n, oho, owo, oco, ohi, owi, oci: te.sum(
                data_vec[n, oho, owo, kh, kw, ic, ohi, owi].astype(out_dtype)
                * kernel_vec[oco, kh, kw, ic, oci].astype(out_dtype),
                axis=[ic, kh, kw],
            ),
            name="conv",
        )
    else:
        conv = te.compute(
            ovshape,
            lambda n, oho, owo, oco, ohi, owi, oci: te.sum(
                data_vec[n, oho, owo, ohi * HSTR + kh, owi * WSTR + kw, ic].astype(out_dtype)
                * kernel_vec[oco, kh, kw, ic, oci].astype(out_dtype),
                axis=[ic, kh, kw],
            ),
            name="conv",
        )

    idiv = tvm.tir.indexdiv
    imod = tvm.tir.indexmod
    output = te.compute(
        oshape,
        lambda n, oho, owo, oc: conv[n][idiv(oho, OHI)][idiv(owo, OWI)][idiv(oc, OCI)][
            imod(oho, OHI)
        ][imod(owo, OWI)][imod(oc, OCI)],
        name="output_unpack",
        tag="spatial_conv_output_NHWC",
    )
    return output


def schedule_conv2d_spatial_pack_nhwc(cfg, s, op, output):
    """Spatial Pack schedule for Conv2d NHWC"""
    unpack = op.output(0)
    conv = unpack.op.input_tensors[0]
    data_vec = conv.op.input_tensors[0]
    kernel_vec = conv.op.input_tensors[1]
    data_pad = data_vec.op.input_tensors[0]
    OHI = cfg["tile_oh"].size[-1]
    OWI = cfg["tile_ow"].size[-1]
    OCI = cfg["tile_co"].size[-1]

    # schedule unpack/output
    if output != unpack:
        s[unpack].compute_inline()
    n, oh, ow, oc = s[output].op.axis
    oco, oci = cfg["tile_co"].apply(s, output, oc)
    oho, ohi = cfg["tile_oh"].apply(s, output, oh)
    owo, owi = cfg["tile_ow"].apply(s, output, ow)
    s[output].reorder(n, oho, owo, oco, ohi, owi, oci)
    cfg["ann_spatial"].apply(
        s, output, [ohi, owi, oci], axis_lens=[OHI, OWI, OCI], max_unroll=16, cfg=cfg
    )
    cfg.define_knob("compat", [0, 1, 2])
    if cfg["compat"].val < 2:
        compat_axis = [owo, oco][cfg["compat"].val]  # pylint: disable=R1706
        s[conv].compute_at(s[output], compat_axis)
    paxis = s[output].fuse(n, oho)
    s[output].parallel(paxis)

    # schedule conv
    n, oho, owo, oco, ohi, owi, oci = s[conv].op.axis
    ic, kh, kw = s[conv].op.reduce_axis
    cfg["reorder_conv"].apply(s, conv, [n, oho, owo, oco, kh, kw, ohi, owi, ic, oci])
    cfg["ann_reduce"].apply(
        s,
        conv,
        [kh, kw],
        axis_lens=[get_const_int(kh.dom.extent), get_const_int(kw.dom.extent)],
        max_unroll=16,
        cfg=cfg,
    )
    cfg["ann_spatial"].apply(
        s, conv, [ohi, owi, oci], axis_lens=[OHI, OWI, OCI], max_unroll=16, cfg=cfg
    )
    if cfg["compat"].val < 2:
        compat_axis = [owo, oco][cfg["compat"].val]  # pylint: disable=R1706
        s[kernel_vec].compute_at(s[conv], compat_axis)
        s[data_vec].compute_at(s[conv], compat_axis)

    if not autotvm.GLOBAL_SCOPE.in_tuning:
        # schedule kernel pack
        oco, kh, kw, ic, oci = kernel_vec.op.axis
        s[kernel_vec].vectorize(oci)
        s[kernel_vec].unroll(ic)
        if cfg["compat"].val == 2:
            s[kernel_vec].parallel(oco)

    # schedule data pack
    if data_vec.op.name == "data_vec_undilated":
        n, oho, owo, kh, kw, ic, ohi, owi = s[data_vec].op.axis
        s[data_vec].vectorize(owi)
        s[data_vec].unroll(ohi)
    else:
        n, oho, owo, ohi, owi, ic = s[data_vec].op.axis
        s[data_vec].vectorize(ic)
        s[data_vec].unroll(owi)
    if cfg["compat"].val == 2:
        paxis = s[data_vec].fuse(n, oho)
        s[data_vec].parallel(paxis)

    return s
