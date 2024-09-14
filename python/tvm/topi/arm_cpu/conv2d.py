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
# pylint: disable=invalid-name, unused-variable, no-else-return, unused-argument, import-outside-toplevel
"""Conv2D schedule for ARM CPU"""
from __future__ import absolute_import as _abs

import tvm
from tvm import te
from tvm import autotvm
from tvm.script import tir as T
import tvm.contrib.nnpack
from tvm.tir.schedule.analysis import has_block

from ..utils import traverse_inline, get_const_tuple
from .. import nn
from ..nn.utils import get_const_int, get_pad_tuple
from ..nn.winograd_util import winograd_transform_matrices
from .arm_utils import get_tiling_A, get_tiling_B_transformed
from .conv2d_spatial_pack import (
    conv2d_spatial_pack_nchw,
    conv2d_spatial_pack_nhwc,
    schedule_conv2d_spatial_pack_nchw,
    schedule_conv2d_spatial_pack_nhwc,
)
from .conv2d_gemm import (
    compute_conv2d_gemm_without_weight_transform,
    schedule_conv2d_gemm_interleaved,
    schedule_conv2d_gemm_native,
)
from .mprofile.dsp.conv2d import conv2d_nhwc_dsp_compute, conv2d_nhwc_dsp_schedule


@autotvm.register_topi_compute("conv2d_nchw_spatial_pack.arm_cpu")
def conv2d_nchw_spatial_pack(cfg, data, kernel, strides, padding, dilation, out_dtype):
    """Compute conv2d with NCHW layout"""
    return conv2d_spatial_pack_nchw(
        cfg, data, kernel, strides, padding, dilation, out_dtype, num_tile=2
    )


@autotvm.register_topi_schedule("conv2d_nchw_spatial_pack.arm_cpu")
def schedule_conv2d_nchw_spatial_pack(cfg, outs):
    """Create schedule for conv2d_nchw"""
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        # schedule conv2d
        if "spatial_conv2d_output" in op.tag:
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

            schedule_conv2d_spatial_pack_nchw(cfg, s, data_vec, kernel_vec, conv, output, outs[0])

    traverse_inline(s, outs[0].op, _callback)
    return s


@autotvm.register_topi_compute("conv2d_nhwc_spatial_pack.arm_cpu")
def conv2d_nhwc_spatial_pack(cfg, data, kernel, strides, padding, dilation, out_dtype):
    """Compute conv2d with NHWC layout"""
    return conv2d_spatial_pack_nhwc(cfg, data, kernel, strides, padding, dilation, out_dtype)


@autotvm.register_topi_schedule("conv2d_nhwc_spatial_pack.arm_cpu")
def schedule_conv2d_nhwc_spatial_pack(cfg, outs):
    """Create schedule for conv2d_nhwc"""
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        if "spatial_conv_output_NHWC" in op.tag:
            schedule_conv2d_spatial_pack_nhwc(cfg, s, op, outs[0])

    traverse_inline(s, outs[0].op, _callback)
    return s


@autotvm.register_topi_compute("conv2d_nchw_winograd.arm_cpu")
def conv2d_nchw_winograd(cfg, data, kernel, strides, padding, dilation, out_dtype):
    """Compute conv2d_nchw layout using Winograd with weight transform"""
    tile_size = 4
    return _decl_winograd(cfg, data, kernel, strides, padding, dilation, out_dtype, tile_size)


@autotvm.register_topi_schedule("conv2d_nchw_winograd.arm_cpu")
def schedule_conv2d_nchw_winograd(cfg, outs):
    """Create schedule for conv2d_nchw_winograd"""
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        if "winograd_conv2d_output" in op.tag:
            output = op.output(0)
            _schedule_winograd(cfg, s, output, outs[0])

    traverse_inline(s, outs[0].op, _callback)
    return s


def _decl_winograd(cfg, data, kernel, strides, padding, dilation, out_dtype, tile_size):
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
    pt, pl, pb, pr = get_pad_tuple(padding, (KH, KW))

    assert KH == 3 and KW == 3 and HSTR == 1 and WSTR == 1
    data_pad = nn.pad(data, (0, 0, pt, pl), (0, 0, pb, pr), name="data_pad")

    idxd = tvm.tir.indexdiv
    idxm = tvm.tir.indexmod

    r = KW
    m = tile_size
    alpha = m + r - 1
    A, B, G = winograd_transform_matrices(m, r, out_dtype)

    K = CO
    C = CI

    H = (IH + pt + pb - 3) // HSTR + 1
    W = (IW + pl + pr - 3) // WSTR + 1
    nH, nW = (H + m - 1) // m, (W + m - 1) // m
    P = N * nH * nW

    # TODO(@kevinthesun): Support tuning/optimization for dynamic shape.
    tile_p = P if isinstance(N, int) else nH * nW
    cfg.define_split("tile_p", cfg.axis(tile_p), num_outputs=2, filter=lambda x: x.size[-1] <= 16)
    cfg.define_split("tile_k", cfg.axis(K), num_outputs=2, filter=lambda x: x.size[-1] <= 16)
    VP = cfg["tile_p"].size[-1]
    VK = cfg["tile_k"].size[-1]

    # pack input tile
    input_tile = te.compute(
        (C, idxd(P, VP), alpha, alpha, VP),
        lambda c, b, eps, nu, bb: data_pad[
            idxd(b * VP + bb, nH * nW),
            c,
            idxm(idxd(b * VP + bb, nW), nH) * m + eps,
            idxm(b * VP + bb, nW) * m + nu,
        ],
        name="d",
    )

    if autotvm.GLOBAL_SCOPE.in_tuning:
        VC = cfg["tile_k"].size[-1]
        kvshape = (KH + tile_size - 1, KW + tile_size - 1, idxd(CO, VC), CI, VC)
        U = tvm.te.placeholder(kvshape, kernel.dtype, name="U")
    else:
        # transform kernel
        if pre_computed:
            U = kernel
        else:
            r_kh = te.reduce_axis((0, KH), "r_kh")
            r_kw = te.reduce_axis((0, KW), "r_kw")
            U = te.compute(
                (alpha, alpha, idxd(K, VK), C, VK),
                lambda eps, nu, k, c, kk: te.sum(
                    kernel[k * VK + kk][c][r_kh][r_kw].astype(out_dtype)
                    * G[eps][r_kh]
                    * G[nu][r_kw],
                    axis=[r_kh, r_kw],
                ),
                name="U",
            )

    # transform image
    r_eps = te.reduce_axis((0, alpha), "r_eps")
    r_nu = te.reduce_axis((0, alpha), "r_nu")
    V = te.compute(
        (alpha, alpha, idxd(P, VP), C, VP),
        lambda eps, nu, b, c, bb: te.sum(
            input_tile[c][b][r_eps][r_nu][bb].astype(out_dtype) * B[r_eps][eps] * B[r_nu][nu],
            axis=[r_eps, r_nu],
        ),
        name="V",
    )

    # batch gemm
    c = te.reduce_axis((0, C), name="c")
    M = te.compute(
        (alpha, alpha, K, P),
        lambda eps, nu, k, b: te.sum(
            U[eps][nu][idxd(k, VK)][c][idxm(k, VK)] * V[eps][nu][idxd(b, VP)][c][idxm(b, VP)],
            axis=c,
        ),
        name="M",
    )

    # inverse transform
    r_eps = te.reduce_axis((0, alpha), "r_eps")
    r_nu = te.reduce_axis((0, alpha), "r_nu")
    Y = te.compute(
        (K, P, m, m),
        lambda k, b, vh, vw: te.sum(
            M[r_eps][r_nu][k][b] * A[r_eps][vh] * A[r_nu][vw], axis=[r_eps, r_nu]
        ),
        name="Y",
    )

    # unpack output
    output = te.compute(
        (N, K, H, W),
        lambda n, k, h, w: Y[k][n * nH * nW + idxd(h, m) * nW + idxd(w, m), idxm(h, m), idxm(w, m)],
        name="output",
        tag="winograd_conv2d_output",
    )

    # we have to manually assign effective GFLOP for winograd
    if isinstance(N, int):
        cfg.add_flop(2 * N * K * H * W * KH * KW * C)
    return output


def _schedule_winograd(cfg, s, output, last):
    Y = output.op.input_tensors[0]
    M, A = Y.op.input_tensors
    U, V = M.op.input_tensors
    d, B = V.op.input_tensors
    data_pad = d.op.input_tensors[0]

    # padding
    s[data_pad].compute_inline()

    # pack input tiles
    s[d].compute_inline()

    # transform kernel
    if isinstance(U.op, tvm.te.ComputeOp):
        kernel, G = U.op.input_tensors
        s[G].compute_inline()
        (eps, nu, k, c, kk) = s[U].op.axis
        if autotvm.GLOBAL_SCOPE.in_tuning:
            # kernel transformation will be pre-computed during compilation, so we skip
            # this part to make tuning records correct
            s[U].pragma(eps, "debug_skip_region")
        else:
            r_kh, r_kw = s[U].op.reduce_axis
            s[U].reorder(k, c, eps, nu, r_kh, r_kw, kk)
            for axis in [eps, nu, r_kh, r_kw]:
                s[U].unroll(axis)
            s[U].vectorize(kk)
            s[U].parallel(k)

        if isinstance(kernel.op, tvm.te.ComputeOp) and "dilate" in kernel.op.tag:
            s[kernel].compute_inline()

    # transform image
    DD = s.cache_read(d, "global", [V])
    s[B].compute_inline()
    eps, nu, b, c, bb = s[V].op.axis
    r_eps, r_nu = s[V].op.reduce_axis
    s[V].reorder(b, c, eps, nu, r_eps, r_nu, bb)
    for axis in [eps, nu, r_eps, r_nu]:
        s[V].unroll(axis)
    s[DD].compute_at(s[V], c)
    s[V].vectorize(bb)
    s[V].parallel(b)

    # batch gemm
    eps, nu, k, b = s[M].op.axis
    c = s[M].op.reduce_axis[0]
    cfg.define_split("tile_c", c, num_outputs=2, filter=lambda x: x.size[-1] <= 16)
    co, ci = cfg["tile_c"].apply(s, M, c)
    xo, xi = cfg["tile_p"].apply(s, M, b)
    s[M].reorder(eps, nu, xo, co, k, ci, xi)
    cfg.define_annotate("ann_reduce", [ci], policy="try_unroll")
    cfg.define_annotate("ann_spatial", [k, xi], policy="try_unroll_vec")
    cfg["ann_reduce"].apply(s, M, [ci], axis_lens=[cfg["tile_c"].size[-1]], max_unroll=16, cfg=cfg)
    cfg["ann_spatial"].apply(s, M, [k, xi])

    # inverse transform
    s[A].compute_inline()
    k, b, vh, vw = s[Y].op.axis
    r_eps, r_nu = s[Y].op.reduce_axis
    for axis in [vh, vw, r_eps, r_nu]:
        s[Y].unroll(axis)

    # output
    n, co, h, w = s[last].op.axis
    co, coi = cfg["tile_k"].apply(s, last, co)
    p = s[last].fuse(n, co)
    s[M].compute_at(s[last], p)
    s[last].parallel(p)

    MM = s.cache_read(M, "global", [Y])
    m = get_const_int(V.shape[0]) + 1 - 3
    ho, wo, hi, wi = s[last].tile(h, w, m, m)
    s[Y].compute_at(s[last], wo)
    s[MM].compute_at(s[last], wo)

    if output != last:
        s[output].compute_inline()


@autotvm.register_topi_compute("conv2d_nchw_winograd_nnpack.arm_cpu")
def conv2d_nchw_winograd_nnpack(cfg, data, kernel, strides, padding, dilation, out_dtype):
    """Compute conv2d_nchw using nnpack Winograd implementation"""
    dtype = data.dtype
    if dtype == "float32":
        return _conv2d_arm_cpu_winograd_nnpack(
            cfg,
            data,
            kernel,
            strides,
            padding,
            dilation,
            out_dtype,
            tvm.contrib.nnpack.ConvolutionAlgorithm.WT_8x8,
        )
    elif dtype == "float16":
        return _conv2d_arm_cpu_winograd_nnpack(
            cfg,
            data,
            kernel,
            strides,
            padding,
            dilation,
            out_dtype,
            tvm.contrib.nnpack.ConvolutionAlgorithm.WT_8x8_FP16,
        )
    else:
        raise ValueError(f"Unsupported data type {dtype} for conv2d winograd nnpack")


@autotvm.register_topi_schedule("conv2d_nchw_winograd_nnpack.arm_cpu")
def schedule_conv2d_nchw_winograd_nnpack(cfg, outs):
    """Create schedule for conv2d_nchw_winograd_nnpack"""
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        if "winograd_nnpack_conv2d_output" in op.tag:
            output = op.output(0)
            _schedule_winograd_nnpack(cfg, s, output, outs[0])

    traverse_inline(s, outs[0].op, _callback)
    return s


def _conv2d_arm_cpu_winograd_nnpack(
    cfg, data, kernel, strides, padding, dilation, out_dtype, convolution_algorithm
):
    """TOPI compute callback. Use winograd NNPACK template"""
    N, CI, IH, IW = get_const_tuple(data.shape)

    if isinstance(dilation, int):
        dilation_h = dilation_w = dilation
    else:
        dilation_h, dilation_w = dilation
    assert (dilation_h, dilation_w) == (1, 1)
    assert len(kernel.shape) == 4
    CO, _, KH, KW = get_const_tuple(kernel.shape)
    HSTR, WSTR = strides if isinstance(strides, (tuple, list)) else (strides, strides)
    pt, pl, pb, pr = get_pad_tuple(padding, (KH, KW))

    assert (
        KH == 3
        and KW == 3
        and pt == 1
        and pb == 1
        and pl == 1
        and pr == 1
        and HSTR == 1
        and WSTR == 1
    )
    H = (IH + pt + pb - 3) // HSTR + 1
    W = (IW + pl + pr - 3) // WSTR + 1

    cfg.define_knob("winograd_nnpack_algorithm", [convolution_algorithm])

    assert N == 1
    with tvm.te.tag_scope("winograd_nnpack_conv2d_weight_transform"):
        transformed_kernel = tvm.contrib.nnpack.convolution_inference_weight_transform(
            kernel, algorithm=cfg["winograd_nnpack_algorithm"].val
        )
        if autotvm.GLOBAL_SCOPE.in_tuning:
            transformed_kernel = te.compute(transformed_kernel.shape, lambda *args: 0.0)

    with tvm.te.tag_scope("winograd_nnpack_conv2d_output"):
        output = tvm.contrib.nnpack.convolution_inference_without_weight_transform(
            data,
            transformed_kernel,
            bias=None,
            padding=[pt, pb, pl, pr],
            stride=[HSTR, WSTR],
            algorithm=cfg["winograd_nnpack_algorithm"].val,
        )

    # we have to manually assign effective GFLOP for winograd
    cfg.add_flop(2 * N * CI * H * W * KH * KW * CO)
    return output


def _schedule_winograd_nnpack(cfg, s, output, last):
    # Could have bias.

    (X, TK) = output.op.input_tensors[:2]

    # transform kernel
    assert isinstance(TK.op, (te.tensor.ComputeOp, te.tensor.ExternOp, te.tensor.PlaceholderOp))
    if autotvm.GLOBAL_SCOPE.in_tuning and isinstance(TK.op, te.tensor.ComputeOp):
        # kernel transformation will be pre-computed during compilation, so we skip
        # this part to make tuning records correct
        s[TK].pragma(s[TK].op.axis[0], "debug_skip_region")


@autotvm.register_topi_compute("conv2d_nchw_winograd_nnpack_without_weight_transform.arm_cpu")
def conv2d_nchw_winograd_nnpack_without_weight_transform(
    cfg, data, transformed_kernel, bias, strides, padding, dilation, out_dtype
):
    """Compute conv2d_nchw using NNPack winograd without weight transform"""
    N, CI, IH, IW = get_const_tuple(data.shape)
    if isinstance(dilation, int):
        dilation_h = dilation_w = dilation
    else:
        dilation_h, dilation_w = dilation
    assert (dilation_h, dilation_w) == (1, 1)
    assert len(transformed_kernel.shape) == 4
    CO, _, _, _ = get_const_tuple(transformed_kernel.shape)
    HSTR, WSTR = strides if isinstance(strides, (tuple, list)) else (strides, strides)
    KH, KW = 3, 3
    pt, pl, pb, pr = get_pad_tuple(padding, (KH, KW))

    assert (
        KH == 3
        and KW == 3
        and pt == 1
        and pb == 1
        and pl == 1
        and pr == 1
        and HSTR == 1
        and WSTR == 1
    )
    H = (IH + pt + pb - 3) // HSTR + 1
    W = (IW + pl + pr - 3) // WSTR + 1

    assert N == 1
    with tvm.te.tag_scope("winograd_nnpack_conv2d_output"):
        output = tvm.contrib.nnpack.convolution_inference_without_weight_transform(
            data=data,
            transformed_kernel=transformed_kernel,
            bias=bias,
            padding=[pt, pb, pl, pr],
            stride=[HSTR, WSTR],
            algorithm=cfg["winograd_nnpack_algorithm"].val,
        )

    # we have to manually assign effective GFLOP for winograd
    cfg.add_flop(2 * N * CI * H * W * KH * KW * CO)
    return output


@autotvm.register_topi_schedule("conv2d_nchw_winograd_nnpack_without_weight_transform.arm_cpu")
def schedule_conv2d_nchw_winograd_nnpack_without_weight_transform(cfg, outs):
    """TOPI schedule callback"""
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        if "winograd_nnpack_conv2d_output" in op.tag:
            output = op.output(0)
            _schedule_winograd_nnpack(cfg, s, output, outs[0])

    traverse_inline(s, outs[0].op, _callback)
    return s


@autotvm.register_topi_compute("conv2d_nhwc_dsp.arm_cpu")
def conv2d_nhwc_dsp(cfg, data, kernel, strides, padding, dilation, out_dtype):
    """Compute conv2d_nhwc with v7e-m DSP instructions."""
    return conv2d_nhwc_dsp_compute(cfg, data, kernel, strides, padding, dilation, out_dtype)


@autotvm.register_topi_schedule("conv2d_nhwc_dsp.arm_cpu")
def schedule_conv2d_nhwc_dsp(cfg, outs):
    """Create schedule for conv2d_nhwc_dsp"""
    return conv2d_nhwc_dsp_schedule(cfg, outs)


def compute_conv2d_NHWC(
    cfg,
    data,
    kernel,
    strides,
    padding,
    dilation,
    out_dtype,
    interleave_A,
    use_scalable_vectors=False,
    use_sme=False,
):
    """Compute definition for conv2d NHWC"""
    N, IH, IW, IC = get_const_tuple(data.shape)
    KH, KW, _, OC = get_const_tuple(kernel.shape)
    tile_N, tile_K = get_tiling_B_transformed(
        interleave_A, data.dtype, use_scalable_vectors, use_sme
    )

    kernel = nn.conv2d_gemm_weight_transform(kernel, tile_N, tile_K, use_scalable_vectors, use_sme)
    return compute_conv2d_gemm_without_weight_transform(
        cfg,
        data,
        kernel,
        strides,
        padding,
        dilation,
        out_dtype,
        (KH, KW),
        OC,
        interleave_A,
        use_scalable_vectors,
        use_sme,
    )


def compute_conv2d_NHWC_without_transform(
    cfg,
    data,
    B,
    strides,
    padding,
    dilation,
    out_dtype,
    kernel_size=None,
    output_channels=None,
    interleave_A=False,
):
    """Compute conv2d NHWC without weight transform"""
    return compute_conv2d_gemm_without_weight_transform(
        cfg,
        data,
        B,
        strides,
        padding,
        dilation,
        out_dtype,
        kernel_size,
        output_channels,
        interleave_A,
    )


def schedule_conv2d_NHWC(cfg, outs, interleave_A):
    """Create schedule for tensors"""
    s = te.create_schedule([x.op for x in outs])
    # Vectorize the output and then inline all the rest
    out = outs[0]
    n, h, w, c = out.op.axis
    n_h_fused = s[out].fuse(n, h)
    _, inner = s[out].split(c, 4)
    s[out].vectorize(inner)
    s[out].parallel(n_h_fused)

    def _callback(op):
        """Traverse operators from computation graph"""
        if op.name == "conv2d_gemm_output":
            conv_out = op.output(0)
            if interleave_A:
                schedule_conv2d_gemm_interleaved(cfg, s, conv_out, out)
            else:
                schedule_conv2d_gemm_native(cfg, s, conv_out, out)
            if out != conv_out:
                s[conv_out].compute_at(s[out], inner)
            else:
                C = conv_out.op.input_tensors[0]
                if interleave_A:
                    s[C].compute_at(s[out], inner)

    traverse_inline(s, outs[0].op, _callback)
    return s


@autotvm.register_topi_compute("conv2d_NHWC_hybrid.arm_cpu")
def compute_conv2d_NHWC_hybrid(cfg, data, kernel, strides, padding, dilation, out_dtype):
    """Interface for hybrid compute_conv2d_NHWC_hybrid"""
    return compute_conv2d_NHWC(cfg, data, kernel, strides, padding, dilation, out_dtype, False)


@autotvm.register_topi_compute("conv2d_NHWC_hybrid_without_transform.arm_cpu")
def compute_conv2d_NHWC_hybrid_without_transform(
    cfg, data, kernel, strides, padding, dilation, out_dtype, kernel_size, output_channels
):
    """Interface for hybrid compute_conv2d_NHWC_hybrid_without_transform"""
    return compute_conv2d_NHWC_without_transform(
        cfg,
        data,
        kernel,
        strides,
        padding,
        dilation,
        out_dtype,
        kernel_size,
        output_channels,
        False,
    )


@autotvm.register_topi_schedule("conv2d_NHWC_hybrid.arm_cpu")
def schedule_conv2d_NHWC_hybrid(cfg, outs):
    """Interface for hybrid schedule_conv2d_NHWC_hybrid"""
    return schedule_conv2d_NHWC(cfg, outs, False)


@autotvm.register_topi_schedule("conv2d_NHWC_hybrid_without_transform.arm_cpu")
def schedule_conv2d_NHWC_hybrid_without_transform(cfg, outs):
    """Interface for hybrid schedule_conv2d_NHWC_hybrid"""
    return schedule_conv2d_NHWC(cfg, outs, False)


@autotvm.register_topi_compute("conv2d_NHWC_hybrid_SVE.arm_cpu")
def compute_conv2d_NHWC_hybrid_SVE(cfg, data, kernel, strides, padding, dilation, out_dtype):
    """Interface for hybrid compute_conv2d_NHWC_hybrid_SVE"""
    return compute_conv2d_NHWC(
        cfg, data, kernel, strides, padding, dilation, out_dtype, False, True
    )


@autotvm.register_topi_schedule("conv2d_NHWC_hybrid_SVE.arm_cpu")
def schedule_conv2d_NHWC_hybrid_SVE(cfg, outs):
    """Interface for hybrid schedule_conv2d_NHWC_hybrid_SVE"""
    return schedule_conv2d_NHWC(cfg, outs, False)


@autotvm.register_topi_compute("conv2d_NHWC_hybrid_SME.arm_cpu")
def compute_conv2d_NHWC_hybrid_SME(cfg, data, kernel, strides, padding, dilation, out_dtype):
    """Interface for hybrid compute_conv2d_NHWC_hybrid_SME"""
    return compute_conv2d_NHWC(
        cfg,
        data,
        kernel,
        strides,
        padding,
        dilation,
        out_dtype,
        False,
        True,
        True,
    )


@autotvm.register_topi_compute("conv2d_NHWC_hybrid_SME_transposed_B.arm_cpu")
def compute_conv2d_NHWC_SME_transposed_B(
    cfg,
    data,
    kernel,
    strides,
    padding,
    dilation,
    out_dtype,
    kernel_size,
    output_channels,
):
    """Compute conv2d NHWC hybrid SME transposed B"""
    N, K = get_const_tuple(kernel.shape)
    tile_N, tile_K = get_tiling_B_transformed(False, data.dtype, True, True)
    pad_N, pad_K = tvm.topi.arm_cpu.arm_utils.get_conv2d_weights_padding(N, K, tile_N, tile_K)

    kernel = tvm.topi.nn.pad(
        kernel, pad_before=(0, 0), pad_after=(pad_N, pad_K), name="weight_padding"
    )

    return compute_conv2d_gemm_without_weight_transform(
        cfg,
        data,
        kernel,
        strides,
        padding,
        dilation,
        out_dtype,
        kernel_size,
        output_channels,
        interleave_A=False,
        use_scalable_vectors=True,
        use_sme=True,
    )


def schedule_conv2d_NHWC_hybrid_TIR(sch: tvm.tir.Schedule):
    """
    Perform TIR scheduling for conv2d NHWC.
    """
    # Get ordered buffer list
    primfunc = sch.mod["main"]
    buffer_names = primfunc.params
    buffer_list = [primfunc.buffer_map[buf] for buf in buffer_names]
    in_dtype = buffer_list[0].dtype
    out_dtype = "float32"

    # Determine PrimFunc blocks
    block_list = [
        "data_pad",
        "data_im2col",
        "T_reshape",
        "A_padded_K",
        "A_padded_M",
        "weight_flatten",
        "weight_padding",
        "weight_transpose",
        "C",
        "conv2d_gemm_output",
    ]
    func_blocks = {}
    for block in block_list:
        func_blocks[block] = sch.get_block(block) if has_block(sch, block) else None

    gemm_block = func_blocks["C"]
    b, m, n, k = sch.get_loops(gemm_block)

    # Get tiling information
    use_scalable_vectors = sch.get(func_blocks["conv2d_gemm_output"]).annotations[
        "use_scalable_vectors"
    ]
    use_sme = sch.get(func_blocks["conv2d_gemm_output"]).annotations["use_sme"]
    M_padded = sch.get(m).extent
    N_padded = sch.get(n).extent
    K_padded = sch.get(k).extent
    tile_M, tile_K = get_tiling_A(False, in_dtype, use_sme)
    tile_N, _ = get_tiling_B_transformed(False, in_dtype, use_scalable_vectors, use_sme)
    tile_M = T.cast(tile_M, M_padded.dtype)
    tile_N = T.cast(tile_N, N_padded.dtype)
    tile_K = T.cast(tile_K, K_padded.dtype)

    # GeMM
    # Compute each tile_M x tile_N tile
    # By summing up K outer products
    if use_sme:
        # pylint: disable=import-outside-toplevel
        from tvm.topi.arm_cpu.pstate_attributes import SMEAttributes
        from tvm.tir.tensor_intrin.arm_cpu import (
            ARM_SME_2SVLx2SVL_GEMM_INTERLEAVED_MOPA,
            ARM_SME_INIT,
            get_sme_gemm_interleaved_mopa_2svlx2svl_intrin,
            get_transpose_interleave_intrin_name,
        )

        # Interleave the padded im2col matrix utilizing the matrix tile
        interleave_t_A_block = sch.cache_read(gemm_block, 0, "global")
        sch.transform_layout(interleave_t_A_block, ("write", 0), lambda b, m, k: (b, k, m))
        b, m, k = sch.get_loops(interleave_t_A_block)
        mo, mi = sch.split(m, factors=(None, tile_M), disable_predication=True)
        ko, ki = sch.split(k, factors=(None, tile_K), disable_predication=True)
        sch.parallel(b)
        sch.reorder(b, ko, mo, ki, mi)
        sch.tensorize(
            ki, get_transpose_interleave_intrin_name(in_dtype, out_dtype, M_padded, K_padded)
        )

        # Interleave the padded weights matrix utilizing the matrix tile
        if in_dtype == "float16":
            interleave_b_block = sch.cache_read(gemm_block, 1, "global")
            sch.transform_layout(interleave_b_block, ("write", 0), lambda n, k: (k, n))
            n, k = sch.get_loops(interleave_b_block)
            ko, ki = sch.split(k, factors=(None, tile_K), disable_predication=True)
            no, ni = sch.split(n, factors=(None, tile_N), disable_predication=True)
            sch.reorder(ko, no, ki, ni)
            sch.tensorize(
                ki, get_transpose_interleave_intrin_name(in_dtype, out_dtype, M_padded, K_padded)
            )

        # Split and reorder the loops of the GeMM for tensorization
        b, m, n, k = sch.get_loops(gemm_block)
        tile_M, _ = get_tiling_A(False, out_dtype, True)
        tile_N, _ = get_tiling_B_transformed(False, out_dtype, True, True)
        tile_M = T.cast(tile_M, M_padded.dtype)
        tile_N = T.cast(tile_N, N_padded.dtype)
        mo, mi = sch.split(m, factors=(None, tile_M), disable_predication=True)
        no, ni = sch.split(n, factors=(None, tile_N), disable_predication=True)
        sch.parallel(b)
        sch.reorder(b, mo, no, mi, ni, k)

        # Tensorize the GeMM initialization
        init_block = sch.decompose_reduction(gemm_block, mi)
        sch.tensorize(sch.get_loops(init_block)[-2], ARM_SME_INIT)

        # Tensorize the GeMM update
        sme_gemm_interleaved_intrin_name = (
            ARM_SME_2SVLx2SVL_GEMM_INTERLEAVED_MOPA + f"_{M_padded}_{K_padded}_{in_dtype}"
        )
        tvm.tir.TensorIntrin.register(
            sme_gemm_interleaved_intrin_name,
            *get_sme_gemm_interleaved_mopa_2svlx2svl_intrin(M_padded, K_padded, in_dtype),
            override=True,
        )
        sch.tensorize(mi, sme_gemm_interleaved_intrin_name)

        # Add pstate annotations
        root_block = sch.get_block("root")
        sch.annotate(
            root_block, SMEAttributes.STREAMING_MODE, SMEAttributes.StreamingModeValues.ENABLED
        )
        sch.annotate(root_block, SMEAttributes.ZA_STORAGE, SMEAttributes.ZAStorageValues.NEW)
    elif use_scalable_vectors:
        mo, mi = sch.split(m, [None, tile_M])
        no, ni = sch.split(n, [None, tile_N], disable_predication=True)
        ko, ki = sch.split(k, [None, tile_K])
        b_mo_fused = sch.fuse(b, mo)
        sch.parallel(b_mo_fused)
        sch.reorder(
            b_mo_fused,
            no,
            ko,
            ki,
            mi,
            ni,
        )
        sch.vectorize(ni)
        sch.unroll(mi)

        # GeMM - Init
        # Initialise an entire GeMM tile at once
        sch.decompose_reduction(gemm_block, ko)
    else:
        mo, mi = sch.split(m, [None, tile_M])
        no, ni = sch.split(n, [None, tile_N])
        ko, ki = sch.split(k, [None, tile_K])
        ni_outer, ni_inner = sch.split(ni, [4, None])
        b_mo_fused = sch.fuse(b, mo)
        sch.parallel(b_mo_fused)
        sch.reorder(
            b_mo_fused,
            no,
            ko,
            ki,
            ni_outer,
            mi,
            ni_inner,
        )
        sch.vectorize(ni_inner)
        sch.unroll(mi)
        sch.unroll(ni_outer)

        # GeMM - Init
        # Initialise an entire GeMM tile at once
        sch.decompose_reduction(gemm_block, ko)

    # Input padding
    if func_blocks["data_pad"]:
        input_padding_block = func_blocks["data_pad"]
        b, h, w, ic = sch.get_loops(input_padding_block)
        b_h_fused = sch.fuse(b, h)
        sch.parallel(b_h_fused)

    # Im2col + padding to tile size
    # Computed outside GeMM
    if func_blocks["data_im2col"]:
        im2col_block = func_blocks["data_im2col"]
        b1, m1, k1 = sch.get_loops(im2col_block)
        b_m_fused_1 = sch.fuse(b1, m1)
        if func_blocks["A_padded_K"]:
            im2col_pad_K_block = func_blocks["A_padded_K"]
            b2, m2, k2 = sch.get_loops(im2col_pad_K_block)
            b_m_fused_2 = sch.fuse(b2, m2)
            sch.parallel(b_m_fused_2)
            sch.compute_at(im2col_block, b_m_fused_2)
            _, k1 = sch.get_loops(sch.get_block("data_im2col"))
        elif func_blocks["A_padded_M"]:
            im2col_pad_M_block = func_blocks["A_padded_M"]
            b2, m2, k2 = sch.get_loops(im2col_pad_M_block)
            b_m_fused_2 = sch.fuse(b2, m2)
            sch.parallel(b_m_fused_1)
            sch.parallel(b_m_fused_2)
        else:
            sch.parallel(b_m_fused_1)

        K = sch.get(k1).extent.value
        if K % 16 == 0:
            split_factor = 16
        elif K % 8 == 0:
            split_factor = 8
        else:
            IC = buffer_list[0].shape[3]
            split_factor = IC
        k_outer, k_inner = sch.split(k1, [None, split_factor])
        sch.vectorize(k_inner)
        sch.unroll(k_outer)

    # Reshape + padding to tile size
    # Computed inside GeMM
    elif func_blocks["T_reshape"]:
        reshape_block = func_blocks["T_reshape"]
        A_pad_block = func_blocks["A_padded_K"] if func_blocks["A_padded_K"] else None
        A_pad_block = func_blocks["A_padded_M"] if func_blocks["A_padded_M"] else A_pad_block
        use_explicit_predication = use_sme and in_dtype == "float32"
        if not use_explicit_predication:
            if use_sme:
                sch.compute_inline(reshape_block)
            elif A_pad_block:
                sch.compute_inline(reshape_block)
                b, m, k = sch.get_loops(A_pad_block)
                _, k_inner = sch.split(k, [None, tile_N])
                sch.vectorize(k_inner)
                sch.compute_at(A_pad_block, mi)
            else:
                sch.compute_at(reshape_block, mi)

    # Weight flattening
    if func_blocks["weight_flatten"]:
        weight_flatten_block = func_blocks["weight_flatten"]
        sch.compute_inline(weight_flatten_block)

    # Weight transpose
    if func_blocks["weight_transpose"] and func_blocks["weight_padding"]:
        weight_padding_block = func_blocks["weight_padding"]
        sch.compute_inline(weight_padding_block)

    # Conv2d output block
    output_block = func_blocks["conv2d_gemm_output"]
    n, h, w, c = sch.get_loops(output_block)
    n_h_fused = sch.fuse(n, h)
    _, inner = sch.split(c, [None, 4])
    sch.vectorize(inner)
    sch.parallel(n_h_fused)

    return sch
