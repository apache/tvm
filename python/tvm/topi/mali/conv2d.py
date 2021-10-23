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
# pylint: disable=invalid-name,unused-variable,unused-argument,no-else-return
"""conv2d schedule on ARM Mali GPU"""
import logging
import tvm
from tvm import te
from tvm import relay
from tvm import autotvm
from tvm.autotvm.task.space import get_factors

from ..utils import traverse_inline, get_const_int, get_const_tuple
from .. import nn
from ..nn.winograd_util import winograd_transform_matrices
from ..nn.conv2d import conv2d_winograd_nhwc, _conv2d_winograd_nhwc_impl

# reuse some compute declarations from ARM CPU
from ..arm_cpu.conv2d_spatial_pack import conv2d_spatial_pack_nchw
from ..arm_cpu.conv2d_spatial_pack import conv2d_spatial_pack_nhwc

logger = logging.getLogger("topi")


@autotvm.register_topi_compute("conv2d_nchw_spatial_pack.mali")
def conv2d_nchw_spatial_pack(cfg, data, kernel, strides, padding, dilation, out_dtype):
    """TOPI compute callback for conv2d

    Parameters
    ----------
    cfg: ConfigEntity
        The config for this template

    data : tvm.te.Tensor
        4-D with shape [batch, in_channel, in_height, in_width]

    kernel : tvm.te.Tensor
        4-D with shape [num_filter, in_channel, filter_height, filter_width] or
        pre-packed 5-D with shape [num_filter_chunk, in_channel, filter_height,
        filter_width, num_filter_block]

    strides : list of two ints
        [stride_height, stride_width]

    padding : list of two ints
        [pad_height, pad_width]

    dilation : list of two ints
        [dilation_height, dilation_width]

    out_dtype: str
        The output type. This is used for mixed precision.

    Returns
    -------
    output : tvm.te.Tensor
        4-D with shape [batch, out_channel, out_height, out_width]
    """
    return conv2d_spatial_pack_nchw(
        cfg, data, kernel, strides, padding, dilation, out_dtype, num_tile=3
    )


@autotvm.register_topi_schedule("conv2d_nchw_spatial_pack.mali")
def schedule_conv2d_nchw_spatial_pack(cfg, outs):
    """TOPI schedule callback for conv2d

    Parameters
    ----------
    cfg: ConfigEntity
        The configuration of this template
    outs: Array of Tensor
        The computation graph description of convolution2d
        in the format of an array of tensors.

    Returns
    -------
    s: Schedule
        The computation schedule for conv2d
    """
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        # schedule conv2d
        if "spatial_conv2d_output" in op.tag:
            _schedule_spatial_pack(cfg, s, op, layout="NCHW")

    traverse_inline(s, outs[0].op, _callback)
    return s


@autotvm.register_topi_compute("conv2d_nhwc_spatial_pack.mali")
def conv2d_nhwc_spatial_pack(cfg, data, kernel, strides, padding, dilation, out_dtype):
    """Compute conv2d with NHWC layout"""
    return conv2d_spatial_pack_nhwc(
        cfg, data, kernel, strides, padding, dilation, out_dtype, num_tile=3
    )


@autotvm.register_topi_schedule("conv2d_nhwc_spatial_pack.mali")
def schedule_conv2d_nhwc_spatial_pack(cfg, outs):
    """Create schedule for conv2d_nhwc"""
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        # schedule conv2d
        if "spatial_conv_output_NHWC" in op.tag:
            _schedule_spatial_pack(cfg, s, op, layout="NHWC")

    traverse_inline(s, outs[0].op, _callback)
    return s


def _schedule_spatial_pack(cfg, s, op, layout):
    """schedule the spatial packing for conv2d"""

    assert layout in ("NCHW", "NHWC")

    output = op.output(0)
    conv = op.input_tensors[0]
    data_vec = conv.op.input_tensors[0]
    data_pad = data_vec.op.input_tensors[0]
    s[data_pad].compute_inline()
    kernel_vec = conv.op.input_tensors[1]
    if kernel_vec.op.name == "kernel_vec":
        kernel = kernel_vec.op.input_tensors[0]
    else:
        kernel = kernel_vec
    if isinstance(kernel.op, tvm.te.ComputeOp) and "dilate" in kernel.op.tag:
        s[kernel].compute_inline()
    data = s[data_vec].op.input_tensors[0]

    max_unroll = 16
    vec_size = [1, 2, 4, 8, 16]
    # get tunable parameters (they are defined in compute)
    _, TC, VC = cfg["tile_co"].size
    _, TH, VH = cfg["tile_oh"].size
    _, TW, VW = cfg["tile_ow"].size

    # schedule padding
    if isinstance(data.op, tvm.te.ComputeOp) and "pad" in data.op.tag:
        data_pad = data
        s[data_pad].compute_inline()

    # schedule data packing
    if layout == "NCHW":
        if isinstance(data_vec.op, tvm.te.ComputeOp) and data_vec.op.name == "data_vec_undilated":
            _, h, w, ci, _, _, vh, vw = s[data_vec].op.axis
        else:
            _, h, w, ci, vh, vw = s[data_vec].op.axis
        z, y, x, unroll1, unroll2 = h, w, ci, vh, vw
    else:
        if isinstance(data_vec.op, tvm.te.ComputeOp) and data_vec.op.name == "data_vec_undilated":
            _, oho, owo, _, _, ic, ohi, owi = s[data_vec].op.axis
        else:
            _, oho, owo, ohi, owi, ic = s[data_vec].op.axis
        z, y, x, unroll1, unroll2 = oho, owo, ohi, ic, owi
    tile_and_bind3d(s, data_vec, z, y, x, 1)
    if unroll1.dom.extent.value < max_unroll:
        s[data_vec].unroll(unroll1)
    if unroll2.dom.extent.value < max_unroll:
        s[data_vec].unroll(unroll2)

    if isinstance(kernel_vec.op, tvm.te.ComputeOp) and kernel_vec.name == "kernel_vec":
        if not autotvm.GLOBAL_SCOPE.in_tuning:
            max_threads = tvm.target.Target.current(allow_none=False).max_num_threads
            ax1, ax2, ax3, ax4, ax5 = s[kernel_vec].op.axis
            fused = s[kernel_vec].fuse(ax1, ax2, ax3, ax4, ax5)
            fused, vec = s[kernel_vec].split(fused, VC)
            bb, tt = s[kernel_vec].split(fused, max_threads)
            s[kernel_vec].bind(bb, te.thread_axis("blockIdx.x"))
            s[kernel_vec].bind(tt, te.thread_axis("threadIdx.x"))
            if VC in vec_size:
                s[kernel_vec].vectorize(vec)

    # schedule convolution
    ic, kh, kw = s[conv].op.reduce_axis
    if layout == "NCHW":
        kh_dim, kw_dim = kernel_vec.shape[2], kernel_vec.shape[3]
    else:
        kh_dim, kw_dim = kernel_vec.shape[0], kernel_vec.shape[1]
    cfg["ann_reduce"].apply(
        s,
        conv,
        [kh, kw],
        axis_lens=[get_const_int(kh_dim), get_const_int(kw_dim)],
        max_unroll=max_unroll,
    )

    if layout == "NCHW":
        n, c, h, w, vh, vw, vc = s[conv].op.axis
        cfg["reorder_0"].apply(s, conv, [n, c, h, w, ic, kh, kw, vh, vw, vc])
        tile_and_bind3d(s, conv, c, h, w, TC, TH, TW)
        unroll_vec_axes = [vh, vw, vc]
        axis_lens = [VH, VW, VC]
    else:
        n, oho, owo, oco, ohi, owi, oci = s[conv].op.axis
        cfg["reorder_conv"].apply(s, conv, [n, oho, owo, oco, kh, kw, ic, ohi, owi, oci])
        tile_and_bind3d(s, conv, oho, owo, oco, TH, TW, TC)
        unroll_vec_axes = [ohi, owi, oci]
        axis_lens = [VH, VW, VC]

    cfg["ann_spatial"].apply(
        s,
        conv,
        unroll_vec_axes,
        axis_lens,
        max_unroll=max_unroll,
        vec_size=vec_size,
        cfg=cfg,
    )

    # schedule output
    if output.op not in s.outputs:  # has bias
        s[output].compute_inline()
        output = s.outputs[0]
    if layout == "NCHW":
        _, co, oh, ow = s[output].op.axis
        tile_and_bind3d(s, output, co, oh, ow, TC, TH, TW)
    else:
        _, oh, ow, co = s[output].op.axis
        tile_and_bind3d(s, output, oh, ow, co, TH, TW, TC)

    return s


##### WINOGRAD TEMPLATE #####
def _pick_tile_size(data, kernel, layout="NCHW"):
    if layout == "NCHW":
        N, CI, H, W = get_const_tuple(data.shape)
    else:
        assert layout == "NHWC"
        N, H, W, CI = get_const_tuple(data.shape)

    if H % 4 == 0:
        return 4
    else:
        return 2


@autotvm.register_topi_compute("conv2d_nchw_winograd.mali")
def conv2d_nchw_winograd(cfg, data, kernel, strides, padding, dilation, out_dtype):
    tile_size = _pick_tile_size(data, kernel)
    return _decl_winograd(cfg, data, kernel, strides, padding, dilation, out_dtype, tile_size)


@autotvm.register_topi_schedule("conv2d_nchw_winograd.mali")
def schedule_conv2d_nchw_winograd(cfg, outs):
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        if "winograd_conv2d_output" in op.tag:
            _schedule_winograd(cfg, s, op)

    traverse_inline(s, outs[0].op, _callback)
    return s


def _decl_winograd(cfg, data, kernel, strides, padding, dilation, out_dtype, tile_size):
    N, CI, IH, IW = get_const_tuple(data.shape)
    if isinstance(dilation, int):
        dilation_h = dilation_w = dilation
    else:
        dilation_h, dilation_w = dilation

    if len(kernel.shape) == 4:
        if dilation_h != 1 or dilation_w != 1:
            kernel = nn.dilate(kernel, (1, 1, dilation_h, dilation_w))
        pre_computed = False
        CO, _, KH, KW = get_const_tuple(kernel.shape)
    else:
        assert (dilation_h, dilation_w) == (1, 1), "Does not support dilation"
        pre_computed = True
        H_CAT, W_CAT, CO, CI, VC = get_const_tuple(kernel.shape)
        CO *= VC
        KH, KW = H_CAT - tile_size + 1, W_CAT - tile_size + 1
    HSTR, WSTR = strides if isinstance(strides, (tuple, list)) else (strides, strides)
    pt, pl, pb, pr = nn.get_pad_tuple(padding, (KH, KW))

    assert KH == 3 and KW == 3 and HSTR == 1 and WSTR == 1
    data_pad = nn.pad(data, (0, 0, pt, pl), (0, 0, pb, pr), name="data_pad")

    r = KW
    m = tile_size
    alpha = m + r - 1
    A, B, G = winograd_transform_matrices(m, r, out_dtype)

    H = (IH + pt + pb - 3) // HSTR + 1
    W = (IW + pl + pr - 3) // WSTR + 1
    nH, nW = (H + m - 1) // m, (W + m - 1) // m
    P = N * nH * nW

    ##### space definition begin #####
    tile_bna_candidates = [1, 2, 4, 8, 16]
    factors = get_factors(CO)
    cfg.define_knob("tile_bna", [x for x in tile_bna_candidates if x in factors])
    cfg.define_knob("tile_bnb", [1, 2, 4, 8, 16])
    cfg.define_split("tile_t1", CI, num_outputs=2, max_factor=128)
    cfg.define_split("tile_t2", CO, num_outputs=2, max_factor=128)
    cfg.define_split("c_unroll", CI, num_outputs=2, max_factor=8)
    cfg.define_knob("yt", [1, 2, 4, 8, 16, 32])
    ##### space definition end #####

    if cfg.is_fallback:
        cfg["tile_bnb"].val = 4
        cfg["tile_bna"].val = 4
        while CO % cfg["tile_bna"].val != 0:
            cfg["tile_bna"].val //= 2
        cfg["yt"].val = 8
        cfg.fallback_split("tile_t1", [-1, 128])
        cfg.fallback_split("tile_t2", [-1, 128])
        cfg.fallback_split("c_unroll", [-1, 8])

    bna = cfg["tile_bna"].val
    bnb = cfg["tile_bnb"].val

    P_round = (P + bnb - 1) // bnb * bnb
    assert CO % bna == 0 and P_round % bnb == 0

    # pack input tile
    input_tile = te.compute(
        (CI, P_round // bnb, alpha, alpha, bnb),
        lambda ci, b, eps, nu, bb: tvm.tir.if_then_else(
            b * bnb + bb < P,
            data_pad[(b * bnb + bb) // (nH * nW)][ci][(b * bnb + bb) // nW % nH * m + eps][
                (b * bnb + bb) % nW * m + nu
            ],
            tvm.tir.const(0, data_pad.dtype),
        ),
        name="d",
    )

    if autotvm.GLOBAL_SCOPE.in_tuning:
        kvshape = (alpha, alpha, CO // bna, CI, bna)
        U = tvm.te.placeholder(kvshape, kernel.dtype, name="U")
    else:
        # transform kernel
        if pre_computed:
            U = kernel
        else:
            r_kh = te.reduce_axis((0, KH), "r_kh")
            r_kw = te.reduce_axis((0, KW), "r_kw")
            U = te.compute(
                (alpha, alpha, CO // bna, CI, bna),
                lambda eps, nu, co, ci, vco: te.sum(
                    kernel[co * bna + vco][ci][r_kh][r_kw] * G[eps][r_kh] * G[nu][r_kw],
                    axis=[r_kh, r_kw],
                ),
                name="U",
            )

    # transform image
    r_a = te.reduce_axis((0, alpha), "r_a")
    r_b = te.reduce_axis((0, alpha), "r_b")
    V = te.compute(
        (alpha, alpha, P_round // bnb, CI, bnb),
        lambda eps, nu, p, ci, vp: te.sum(
            input_tile[ci][p][r_a][r_b][vp] * B[r_a][eps] * B[r_b][nu], axis=[r_a, r_b]
        ),
        name="V",
    )

    idxdiv = tvm.tir.indexdiv
    idxmod = tvm.tir.indexmod

    # batch gemm
    ci = te.reduce_axis((0, CI), name="c")
    M = te.compute(
        (alpha, alpha, CO, P_round),
        lambda eps, nu, co, p: te.sum(
            U[eps][nu][idxdiv(co, bna)][ci][idxmod(co, bna)]
            * V[eps][nu][idxdiv(p, bnb)][ci][idxmod(p, bnb)],
            axis=ci,
        ),
        name="M",
    )

    r_a = te.reduce_axis((0, alpha), "r_a")
    r_b = te.reduce_axis((0, alpha), "r_b")
    Y = te.compute(
        (CO, P, m, m),
        lambda co, p, vh, vw: te.sum(M[r_a][r_b][co][p] * A[r_a][vh] * A[r_b][vw], axis=[r_a, r_b]),
        name="Y",
    )

    # unpack output
    output = te.compute(
        (N, CO, H, W),
        lambda n, co, h, w: Y[
            co, n * nH * nW + idxdiv(h, m) * nW + idxdiv(w, m), idxmod(h, m), idxmod(w, m)
        ]
        # The following hack term is used to make the padding in batch gemm ("M")
        # effective, otherwise the padding will be eliminated by bound inference.
        # Use `tvm.tir.Mul` instead of `*` to avoid issues in const folding.
        + tvm.tir.Mul(tvm.tir.const(0, out_dtype), M[alpha - 1][alpha - 1][CO - 1][P_round - 1]),
        name="output",
        tag="winograd_conv2d_output",
    )

    # we have to manually assign effective GFLOP for winograd
    cfg.add_flop(2 * N * CO * H * W * KH * KW * CI)
    return output


def _schedule_winograd(cfg, s, op):
    """schedule winograd fast convolution F(2x2, 3x3) for conv2d"""
    # get ops and tensors
    output = op.output(0)

    Y = op.input_tensors[0]
    M, A = s[Y].op.input_tensors
    U, V = s[M].op.input_tensors
    d, B = s[V].op.input_tensors
    data_pad = s[d].op.input_tensors[0]

    # padding
    s[data_pad].compute_inline()

    # transform kernel
    if isinstance(U.op, tvm.te.ComputeOp):
        kernel, G = s[U].op.input_tensors
        s[G].compute_inline()
        (
            eps,
            nu,
            co,
            ci,
            vco,
        ) = s[U].op.axis
        if not autotvm.GLOBAL_SCOPE.in_tuning:
            r_kh, r_kw = s[U].op.reduce_axis
            s[U].reorder(co, ci, eps, nu, r_kh, r_kw, vco)
            _ = [s[U].unroll(x) for x in [eps, nu, r_kh, r_kw]]
            s[U].vectorize(vco)
            tile_and_bind(s, U, co, ci, 1, 256)

        # dilation
        if isinstance(kernel.op, tvm.te.ComputeOp) and "dilate" in kernel.op.tag:
            s[kernel].compute_inline()

    # transform image
    s[B].compute_inline()
    VL = s.cache_write(V, "local")

    eps, nu, p, ci, vp = s[V].op.axis
    s[V].reorder(p, ci, eps, nu, vp)
    for axis in [eps, nu]:
        s[V].unroll(axis)
    s[V].vectorize(vp)
    fused = s[V].fuse(p, ci)

    bb, tt = cfg["tile_t1"].apply(s, V, fused)
    s[V].bind(bb, te.thread_axis("blockIdx.x"))
    s[V].bind(tt, te.thread_axis("threadIdx.x"))

    eps, nu, p, ci, vp = s[VL].op.axis
    r_a, r_b = s[VL].op.reduce_axis
    for axis in [eps, nu, r_a, r_b]:
        s[VL].unroll(axis)
    s[VL].vectorize(vp)
    s[d].compute_at(s[V], tt)
    s[VL].compute_at(s[V], tt)

    # batch gemm
    bna = cfg["tile_bna"].val
    bnb = cfg["tile_bnb"].val

    eps, nu, k, b = s[M].op.axis
    alpha = eps.dom.extent
    c = s[M].op.reduce_axis[0]
    yo, xo, yi, xi = s[M].tile(k, b, bna, bnb)
    c, c_unroll = cfg["c_unroll"].apply(s, M, c)
    s[M].reorder(yo, xo, c, c_unroll, yi, xi)
    s[M].unroll(c_unroll)
    s[M].unroll(yi)
    s[M].vectorize(xi)
    z = s[M].fuse(eps, nu)
    tile_and_bind3d(s, M, z, yo, xo, 1, cfg["yt"].val, 1)

    # inverse transform
    s[A].compute_inline()
    k, b, vh, vw = s[Y].op.axis
    r_a, r_b = s[Y].op.reduce_axis
    for axis in [vh, vw, r_a, r_b]:
        s[Y].unroll(axis)

    # schedule output and fusion
    if output.op not in s.outputs:
        s[output].compute_inline()
        output = s.outputs[0]

    n, co, h, w = s[output].op.axis
    m = alpha - 3 + 1
    h, w, hi, wi = s[output].tile(h, w, m, m)
    s[output].unroll(hi)
    s[output].unroll(wi)
    fused = s[output].fuse(n, co, h, w)
    bb, tt = cfg["tile_t2"].apply(s, output, fused)
    s[output].bind(bb, te.thread_axis("blockIdx.x"))
    s[output].bind(tt, te.thread_axis("threadIdx.x"))

    s[Y].compute_at(s[output], tt)


##### REGISTER ALTER OP LAYOUT #####
@nn.conv2d_alter_layout.register(["mali"])
def _alter_conv2d_layout(attrs, inputs, tinfos, out_type):
    target = tvm.target.Target.current(allow_none=False)
    dispatch_ctx = autotvm.task.DispatchContext.current

    new_attrs = {k: attrs[k] for k in attrs.keys()}

    strides = attrs.get_int_tuple("strides")
    padding = attrs.get_int_tuple("padding")
    dilation = attrs.get_int_tuple("dilation")
    data_layout = attrs["data_layout"]
    kernel_layout = attrs["kernel_layout"]
    data, kernel = tinfos
    out_dtype = out_type.dtype

    impl, outs = relay.backend.te_compiler.select_implementation(
        relay.op.get("nn.conv2d"), attrs, tinfos, out_type, target
    )
    workload = autotvm.task.get_workload(outs)
    if workload is None:
        # The best implementation is not an AutoTVM template.
        # It may be from the auto-scheduler
        if impl.name.find("winograd") != -1:
            if dilation != (1, 1):
                logger.warning("Does not support weight pre-transform for dilated convolution.")
                return None

            assert data_layout == "NHWC" and kernel_layout == "HWIO"
            N, H, W, CI = get_const_tuple(data.shape)
            KH, KW, _, CO = get_const_tuple(kernel.shape)

            # Pre-compute weight transformation in winograd
            tile_size = _pick_tile_size(tinfos[0], tinfos[1], layout="NHWC")

            # HWIO -> OIHW
            kernel_transform = relay.transpose(inputs[1], axes=[3, 2, 0, 1])
            # alpha, alpha, CO, CI
            weight = relay.nn.contrib_conv2d_winograd_weight_transform(
                kernel_transform, tile_size=tile_size
            )
            new_attrs["tile_size"] = tile_size
            new_attrs["channels"] = CO
            return relay.nn.contrib_conv2d_winograd_without_weight_transform(
                inputs[0], weight, **new_attrs
            )

        return None
    cfg = dispatch_ctx.query(target, workload)
    if cfg.is_fallback:  # if is fallback, clear query cache and return None
        autotvm.task.clear_fallback_cache(target, workload)
        return None

    topi_tmpl = workload[0]
    idxd = tvm.tir.indexdiv

    if topi_tmpl == "conv2d_nchw_spatial_pack.mali":
        assert data_layout == "NCHW" and kernel_layout == "OIHW"
        N, CI, H, W = get_const_tuple(data.shape)
        CO, _, KH, KW = get_const_tuple(kernel.shape)
        VC = cfg["tile_co"].size[-1]

        new_attrs["kernel_layout"] = "OIHW%do" % VC

        new_data = data
        new_kernel = te.placeholder((idxd(CO, VC), CI, KH, KW, VC), dtype=kernel.dtype)
        new_workload = autotvm.task.args_to_workload(
            [new_data, new_kernel, strides, padding, dilation, out_dtype],
            "conv2d_nchw_spatial_pack.mali",
        )
        dispatch_ctx.update(target, new_workload, cfg)

        return relay.nn.conv2d(*inputs, **new_attrs)
    elif topi_tmpl == "conv2d_nchw_winograd.mali":
        assert data_layout == "NCHW" and kernel_layout == "OIHW"
        N, CI, H, W = get_const_tuple(data.shape)
        CO, _, KH, KW = get_const_tuple(kernel.shape)
        tile_size = _pick_tile_size(data, kernel)
        VC = cfg["tile_bna"].val

        weight_expr = inputs[1]
        weight_expr = relay.nn.contrib_conv2d_winograd_weight_transform(
            weight_expr, tile_size=tile_size
        )
        weight_expr = relay.reshape(
            weight_expr, newshape=(KH + tile_size - 1, KW + tile_size - 1, idxd(CO, VC), VC, CI)
        )
        weight_expr = relay.transpose(weight_expr, axes=[0, 1, 2, 4, 3])

        new_attrs["tile_size"] = tile_size

        new_data = data
        new_kernel = te.placeholder(
            (KH + tile_size - 1, KW + tile_size - 1, idxd(CO, VC), CI, VC), kernel.dtype
        )
        new_workload = autotvm.task.args_to_workload(
            [new_data, new_kernel, strides, padding, dilation, out_dtype],
            "conv2d_nchw_winograd.mali",
        )
        dispatch_ctx.update(target, new_workload, cfg)

        return relay.nn.contrib_conv2d_winograd_without_weight_transform(
            inputs[0], weight_expr, **new_attrs
        )
    else:
        return None


@conv2d_winograd_nhwc.register(["mali"])
def conv2d_winograd_nhwc_mali(
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
    This is a clean version to be used by the auto-scheduler for mali.
    """
    tile_size = _pick_tile_size(data, weight, layout="NHWC")
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


##### SCHECULE UTILITIES #####
def tile_and_bind(s, tensor, y, x, y_factor, x_factor=None):
    """tile and bind to GPU threads"""
    x_factor = x_factor or y_factor
    yo, xo, yi, xi = s[tensor].tile(y, x, y_factor, x_factor)
    s[tensor].bind(xo, te.thread_axis("blockIdx.x"))
    s[tensor].bind(xi, te.thread_axis("threadIdx.x"))
    s[tensor].bind(yo, te.thread_axis("blockIdx.y"))
    s[tensor].bind(yi, te.thread_axis("threadIdx.y"))
    return yo, xo, yi, xi


def tile_and_bind3d(s, tensor, z, y, x, z_factor=2, y_factor=None, x_factor=None):
    """tile and bind 3d"""
    y_factor = y_factor or z_factor
    x_factor = x_factor or y_factor
    zo, zi = s[tensor].split(z, z_factor)
    yo, yi = s[tensor].split(y, y_factor)
    xo, xi = s[tensor].split(x, x_factor)
    s[tensor].bind(zo, te.thread_axis("blockIdx.z"))
    s[tensor].bind(zi, te.thread_axis("threadIdx.z"))
    s[tensor].bind(yo, te.thread_axis("blockIdx.y"))
    s[tensor].bind(yi, te.thread_axis("threadIdx.y"))
    s[tensor].bind(xo, te.thread_axis("blockIdx.x"))
    s[tensor].bind(xi, te.thread_axis("threadIdx.x"))
    s[tensor].reorder(zo, yo, xo, zi, yi, xi)
    return zo, yo, xo, zi, yi, xi
