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
# pylint: disable=invalid-name,unused-variable,unused-argument
"""Common Winograd implementation for Adreno backend"""

import tvm
from tvm import te
from tvm import autotvm

from tvm.topi import nn
from tvm.topi.utils import get_const_int, get_const_tuple, traverse_inline
from ..nn.winograd_util import winograd_transform_matrices
from .utils import (
    split_to_chunks,
    pack_input,
    pack_filter,
    bind_data_copy,
    get_texture_storage,
    infer_tile_size,
)


def conv2d_winograd_comp(
    cfg, data, kernel, strides, padding, dilation, out_dtype, pre_computed, layout
):
    """Compute declaration for winograd

    Parameters
    ----------
    cfg: ConfigEntity
        The config for this template

    data: tvm.te.Tensor
        4-D or 5-D Data tensor with shape NCHW or NCHW4c

    kernel: tvm.te.Tensor
        4-D or 5-D tensor with shape OIHW or OIHW4o

    strides: int or a list/tuple of two ints
        stride size, or [stride_height, stride_width]

    padding: int or a list/tuple of 2 or 4 ints
        padding size, or
        [pad_height, pad_width] for 2 ints, or
        [pad_top, pad_left, pad_bottom, pad_right] for 4 ints

    dilation: int or a list/tuple of two ints
        dilation size, or [dilation_height, dilation_width]

    out_dtype: str
        The output type. This is used for mixed precision.

    pre_computed: bool
        Flag if weights were pre computed if true or the weights should be
        computed in runtime

    layout: str
        NHWC or NCHW values are accepted

    Returns
    -------
    output: tvm.te.Tensor
        4-D or 5-D with shape NCHW or NCHW4c
    """
    assert layout in ("NCHW", "NHWC")
    tile_size = infer_tile_size(data, layout)

    if isinstance(dilation, int):
        dilation_h = dilation_w = dilation
    else:
        dilation_h, dilation_w = dilation
    HSTR, WSTR = (strides, strides) if isinstance(strides, int) else strides

    convert_from4d = False
    if len(data.shape) == 4:
        convert_from4d = True
        if layout == "NCHW":
            N, DCI, H, W = get_const_tuple(data.shape)
        else:
            N, H, W, DCI = get_const_tuple(data.shape)
        if not pre_computed:
            if layout == "NCHW":
                out_channels, CI, KH, KW = get_const_tuple(kernel.shape)
            else:
                KH, KW, CI, out_channels = get_const_tuple(kernel.shape)
        else:
            alpha, _, CI, out_channels = get_const_tuple(kernel.shape)
            KH = KW = alpha + 1 - tile_size

        in_channel_chunks, in_channel_block, in_channel_tail = split_to_chunks(CI, 4)
        out_channel_chunks, out_channel_block, out_channel_tail = split_to_chunks(out_channels, 4)
        if autotvm.GLOBAL_SCOPE.in_tuning is True:
            if layout == "NCHW":
                dshape = (N, in_channel_chunks, H, W, in_channel_block)
            else:
                dshape = (N, H, W, in_channel_chunks, in_channel_block)
            if not pre_computed:  # kernel tensor is raw tensor, do strict check
                if layout == "NCHW":
                    kshape = (out_channel_chunks, CI, KH, KW, out_channel_block)
                else:
                    kshape = (KH, KW, CI, out_channel_chunks, out_channel_block)
            else:
                kshape = (alpha, alpha, CI, out_channel_chunks, out_channel_block)
            data = tvm.te.placeholder(dshape, data.dtype, name="data_placeholder")
            kernel = tvm.te.placeholder(kshape, kernel.dtype, name="kernel_placeholder")
        else:
            data = pack_input(
                data, layout, N, in_channel_chunks, in_channel_block, in_channel_tail, H, W
            )
            kernel_layout = "OIHW" if layout == "NCHW" else "HWIO"
            if not pre_computed:  # kernel tensor is raw tensor, do strict check
                kernel = pack_filter(
                    kernel,
                    kernel_layout,
                    out_channel_chunks,
                    out_channel_block,
                    out_channel_tail,
                    CI,
                    in_channel_chunks,
                    in_channel_block,
                    in_channel_tail,
                    KH,
                    KW,
                )
            else:
                kernel = pack_filter(
                    kernel,
                    "HWIO",
                    out_channel_chunks,
                    out_channel_block,
                    out_channel_tail,
                    CI,
                    in_channel_chunks,
                    in_channel_block,
                    in_channel_tail,
                    alpha,
                    alpha,
                )
    if layout == "NCHW":
        N, DCI, H, W, CB = get_const_tuple(data.shape)
    else:
        N, H, W, DCI, CB = get_const_tuple(data.shape)
    if not pre_computed:  # kernel tensor is raw tensor, do strict check
        if layout == "NCHW":
            CO, CI, KH, KW, COB = get_const_tuple(kernel.shape)
        else:
            KH, KW, CI, CO, COB = get_const_tuple(kernel.shape)
        alpha = KW + tile_size - 1
        assert HSTR == 1 and WSTR == 1 and KH == KW
    else:
        alpha, _, CI, CO, COB = get_const_tuple(kernel.shape)
        KH = KW = alpha + 1 - tile_size
        assert HSTR == 1 and WSTR == 1 and dilation_h == 1 and dilation_w == 1

    if isinstance(N, tvm.tir.Any):
        N = tvm.te.size_var("n")

    if not isinstance(H, int) or not isinstance(W, int):
        raise RuntimeError(
            "adreno winograd conv2d doesn't support dynamic input\
                           height or width."
        )

    pt, pl, pb, pr = nn.get_pad_tuple(padding, (KH, KW))
    if layout == "NCHW":
        data_pad = nn.pad(data, (0, 0, pt, pl, 0), (0, 0, pb, pr, 0), name="data_pad")
    else:
        data_pad = nn.pad(data, (0, pt, pl, 0, 0), (0, pb, pr, 0, 0), name="data_pad")

    r = KW
    m = tile_size
    A, B, G = winograd_transform_matrices(m, r, data.dtype)

    H = (H + pt + pb - KH) // HSTR + 1
    W = (W + pl + pr - KW) // WSTR + 1
    nH, nW = (H + m - 1) // m, (W + m - 1) // m

    P = N * nH * nW if isinstance(N, int) else nH * nW

    # transform kernel
    if not pre_computed:
        r_kh = te.reduce_axis((0, KH), name="r_kh")
        r_kw = te.reduce_axis((0, KW), name="r_kw")
        if layout == "NCHW":
            kernel_pack = te.compute(
                (alpha, alpha, CI, CO, COB),
                lambda eps, nu, ci, co, cob: te.sum(
                    kernel[co][ci][r_kh][r_kw][cob] * G[eps][r_kh] * G[nu][r_kw], axis=[r_kh, r_kw]
                ),
                name="kernel_pack",
            )
        else:
            kernel_pack = te.compute(
                (alpha, alpha, CI, CO, COB),
                lambda eps, nu, ci, co, cob: te.sum(
                    kernel[r_kh][r_kw][ci][co][cob] * G[eps][r_kh] * G[nu][r_kw], axis=[r_kh, r_kw]
                ),
                name="kernel_pack",
            )
    else:
        kernel_pack = kernel

    idxdiv = tvm.tir.indexdiv
    idxmod = tvm.tir.indexmod
    if layout == "NCHW":
        N, CI, _, _, CB = get_const_tuple(data.shape)
    else:
        N, _, _, CI, CB = get_const_tuple(data.shape)

    # pack input tile
    if layout == "NCHW":
        input_tile = te.compute(
            (alpha, alpha, CI, P, CB),
            lambda eps, nu, c, p, cb: data_pad[idxdiv(p, (nH * nW))][c][
                idxmod(idxdiv(p, nW), nH) * m + eps
            ][idxmod(p, nW) * m + nu][cb],
            name="d",
        )
    else:
        input_tile = te.compute(
            (alpha, alpha, CI, P, CB),
            lambda eps, nu, c, p, cb: data_pad[idxdiv(p, (nH * nW))][
                idxmod(idxdiv(p, nW), nH) * m + eps
            ][idxmod(p, nW) * m + nu][c][cb],
            name="d",
        )

    # transform data
    r_a = te.reduce_axis((0, alpha), "r_a")
    r_b = te.reduce_axis((0, alpha), "r_a")
    data_pack = te.compute(
        (P, CI, alpha, alpha, CB),
        lambda p, ci, eps, nu, cb: te.sum(
            input_tile[r_a][r_b][ci][p][cb] * B[r_a][eps] * B[r_b][nu], axis=[r_a, r_b]
        ),
        name="data_pack",
    )

    # repack transformed data
    data_pack_trans = te.compute(
        (alpha, alpha, CI, P, CB),
        lambda eps, nu, c, p, cb: data_pack[p][c][eps][nu][cb],
        name="data_pack_trans",
    )

    # do batch gemm
    ci = te.reduce_axis((0, CI), name="ci")
    cb = te.reduce_axis((0, CB), name="cb")
    bgemm = te.compute(
        (alpha, alpha, CO, P, COB),
        lambda eps, nu, co, p, cob: te.sum(
            (
                kernel_pack[eps][nu][ci * CB + cb][co][cob] * data_pack_trans[eps][nu][ci][p][cb]
            ).astype(out_dtype),
            axis=[ci, cb],
        ),
        name="bgemm",
    )

    # inverse transform
    r_a = te.reduce_axis((0, alpha), "r_a")
    r_b = te.reduce_axis((0, alpha), "r_a")
    inverse = te.compute(
        (CO, P, m, m, COB),
        lambda co, p, vh, vw, cob: te.sum(
            bgemm[r_a][r_b][co][p][cob] * (A[r_a][vh] * A[r_b][vw]).astype(out_dtype),
            axis=[r_a, r_b],
        ),
        name="inverse",
    )

    # output
    if layout == "NCHW":
        if convert_from4d and autotvm.GLOBAL_SCOPE.in_tuning is False:
            output = te.compute(
                (N, out_channels, H, W),
                lambda n, c, h, w: inverse[c // CB][n * nH * nW + idxdiv(h, m) * nW + idxdiv(w, m)][
                    idxmod(h, m)
                ][idxmod(w, m)][c % CB].astype(out_dtype),
                name="output",
                tag="dummy_compute_at",
            )
        else:
            output = te.compute(
                (N, CO, H, W, COB),
                lambda n, co, h, w, cob: inverse[co][
                    n * nH * nW + idxdiv(h, m) * nW + idxdiv(w, m)
                ][idxmod(h, m)][idxmod(w, m)][cob].astype(out_dtype),
                name="output",
                tag="dummy_compute_at",
            )
    else:
        if convert_from4d and autotvm.GLOBAL_SCOPE.in_tuning is False:
            output = te.compute(
                (N, H, W, out_channels),
                lambda n, h, w, c: inverse[c // CB][n * nH * nW + idxdiv(h, m) * nW + idxdiv(w, m)][
                    idxmod(h, m)
                ][idxmod(w, m)][c % CB].astype(out_dtype),
                name="output",
                tag="dummy_compute_at",
            )
        else:
            output = te.compute(
                (N, H, W, CO, COB),
                lambda n, h, w, co, cob: inverse[co][
                    n * nH * nW + idxdiv(h, m) * nW + idxdiv(w, m)
                ][idxmod(h, m)][idxmod(w, m)][cob].astype(out_dtype),
                name="output",
                tag="dummy_compute_at",
            )

    if isinstance(N, int):
        cfg.add_flop(2 * N * CO * COB * H * W * CI * CB * KH * KW)

    return output


def schedule_conv2d_winograd_impl(cfg, outs, tag, pre_computed=False):
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        if op.tag == tag:
            schedule_conv2d_winograd(cfg, s, op.output(0), pre_computed=pre_computed)

    traverse_inline(s, outs[0].op, _callback)
    return s


def schedule_conv2d_winograd(cfg, s, output, pre_computed):
    """Schedule winograd template"""
    inverse = s[output].op.input_tensors[0]
    bgemm, A = s[inverse].op.input_tensors
    kernel_pack, data_pack_trans = s[bgemm].op.input_tensors
    data_pack = s[data_pack_trans].op.input_tensors[0]
    input_tile, B = s[data_pack].op.input_tensors
    pad_data = s[input_tile].op.input_tensors[0]

    # data transform
    s[B].compute_inline()
    s[A].compute_inline()

    # probably will improve real topology execution
    if autotvm.GLOBAL_SCOPE.in_tuning:
        # Padding to texture
        AA = s.cache_read(pad_data, get_texture_storage(pad_data.shape), [input_tile])
        bind_data_copy(s[AA])

    s[input_tile].compute_inline()

    OL = s.cache_write(data_pack, "local")
    c, p, eps, nu, cb = s[data_pack].op.axis
    fused = s[data_pack].fuse(c, p, eps, nu)
    bx, tx = s[data_pack].split(fused, 128)
    s[data_pack].vectorize(cb)
    s[data_pack].bind(bx, te.thread_axis("blockIdx.x"))
    s[data_pack].bind(tx, te.thread_axis("threadIdx.x"))

    _, _, eps, nu, cb = s[OL].op.axis
    r_a, r_b = s[OL].op.reduce_axis
    s[OL].unroll(eps)
    s[OL].unroll(nu)
    s[OL].unroll(r_a)
    s[OL].unroll(r_b)
    s[OL].vectorize(cb)
    s[OL].compute_at(s[data_pack], tx)
    s[data_pack].set_scope(get_texture_storage(data_pack.shape))

    s[data_pack_trans].compute_inline()

    # transform kernel
    if not pre_computed:
        kernel, G = s[kernel_pack].op.input_tensors
        eps, nu, ci, co, cob = s[kernel_pack].op.axis
        if autotvm.GLOBAL_SCOPE.in_tuning:
            # skip this part during tuning to make recrods accurate
            # this part will be pre-computed during pre-compute optimization pass
            s[G].pragma(s[G].op.axis[0], "debug_skip_region")
            s[kernel_pack].pragma(eps, "debug_skip_region")
        else:
            s[G].compute_inline()
            r_a, r_b = s[kernel_pack].op.reduce_axis
            for axis in [eps, nu, r_a, r_b]:
                s[kernel_pack].unroll(axis)

            fused = s[kernel_pack].fuse(ci, co)
            bb, tt = s[kernel_pack].split(fused, 128)
            s[kernel_pack].reorder(bb, tt, eps, nu, r_a, r_b, cob)
            s[kernel_pack].vectorize(cob)
            s[kernel_pack].bind(bb, te.thread_axis("blockIdx.x"))
            s[kernel_pack].bind(tt, te.thread_axis("threadIdx.x"))
    else:
        kernel = kernel_pack

    if isinstance(kernel.op, tvm.te.ComputeOp) and "filter_pack" in kernel.op.tag:
        # manage scheduling of datacopy
        pack_data = pad_data.op.input_tensors[0]
        bind_data_copy(s[pack_data])
        bind_data_copy(s[kernel])
    elif isinstance(kernel.op, tvm.te.ComputeOp) and "dilate" in kernel.op.tag:
        s[kernel].compute_inline()
    s[pad_data].compute_inline()

    ##### space definition begin #####
    cfg.define_knob("auto_unroll_max_step", [0, 4, 16])
    b1, b2, y, x, cb = s[bgemm].op.axis
    rcc = s[bgemm].op.reduce_axis[0]
    alpha = get_const_int(b1.dom.extent)

    cfg.define_split(
        "tile_y", y, num_outputs=3, filter=lambda entry: entry.size[2] <= 64 and entry.size[1] <= 16
    )

    min_x_div = 1
    for bn in range(4, 0, -1):
        if bgemm.shape[3] % bn == 0:
            min_x_div = bn
            break

    cfg.define_split(
        "tile_x",
        x,
        num_outputs=3,
        filter=lambda entry: entry.size[2] <= 64
        and entry.size[1] >= min_x_div
        and entry.size[1] <= 16,
    )
    cfg.define_split("tile_rc", rcc, num_outputs=2)
    # TODO: Uncomment the following lines when multi_filter will be introduced
    # cfg.multi_filter(
    # filter=lambda entity: entity["tile_y"].size[2] * entity["tile_x"].size[2] in range(32,1024)
    # )
    ##### space definition end #####

    # batch gemm
    OL = s.cache_write(bgemm, "local")
    if (
        autotvm.GLOBAL_SCOPE.in_tuning
        or isinstance(kernel.op, tvm.te.ComputeOp)
        and "filter_pack" in kernel.op.tag
    ):
        BB = s.cache_read(kernel_pack, get_texture_storage(kernel_pack.shape), [OL])
        bind_data_copy(s[BB])

    by = s[bgemm].fuse(b1, b2, y)

    # tile and bind spatial axes
    bgemm_scope, by = s[bgemm].split(by, nparts=1)
    by, vy, ty = cfg["tile_y"].apply(s, bgemm, by)
    bx, vx, tx = cfg["tile_x"].apply(s, bgemm, x)
    s[bgemm].bind(by, te.thread_axis("blockIdx.y"))
    s[bgemm].bind(bx, te.thread_axis("blockIdx.x"))
    s[bgemm].bind(vy, te.thread_axis("vthread"))
    s[bgemm].bind(vx, te.thread_axis("vthread"))
    s[bgemm].bind(ty, te.thread_axis("threadIdx.y"))
    s[bgemm].bind(tx, te.thread_axis("threadIdx.x"))
    s[bgemm].reorder(bgemm_scope, by, bx, vy, vx, ty, tx, cb)
    s[bgemm].vectorize(cb)
    s[bgemm].set_scope(get_texture_storage(bgemm.shape))

    # tile reduction axes
    s[OL].compute_at(s[bgemm], tx)
    b1, b2, y, x, cb = s[OL].op.axis
    (rcc, rcb) = s[OL].op.reduce_axis
    b = s[OL].fuse(b1, b2)
    s[OL].reorder(b, y, x, rcc, rcb, cb)
    # s[OL].unroll(rcb)
    s[OL].pragma(rcb, "auto_unroll_max_step", cfg["auto_unroll_max_step"].val)
    s[OL].pragma(rcb, "unroll_explicit", True)
    s[OL].vectorize(cb)

    # schedule inverse, output and fusion
    if output.op in s.outputs:
        OL = None
    else:
        OL = output
        s[OL].set_scope("local")
        output = s.outputs[0]

    if len(s[output].op.axis) == 4:
        n, co, h, w = s[output].op.axis
        cb = None
    else:
        n, co, h, w, cb = s[output].op.axis
    inverse_scope, n = s[output].split(n, nparts=1)

    fused = s[output].fuse(n, co, h, w)
    bb, tt = s[output].split(fused, 128)
    if cb is not None:
        s[output].reorder(bb, tt, cb)
        s[output].vectorize(cb)

    s[output].bind(bb, te.thread_axis("blockIdx.x"))
    s[output].bind(tt, te.thread_axis("threadIdx.x"))

    if OL is not None:
        s[OL].compute_at(s[output], tt)

    co, p, vh, vw, cb = s[inverse].op.axis
    r_a, r_b = s[inverse].op.reduce_axis
    for axis in [vh, vw, r_a, r_b]:
        s[inverse].unroll(axis)
    s[inverse].vectorize(cb)
    s[inverse].compute_at(s[output], tt)

    return s
