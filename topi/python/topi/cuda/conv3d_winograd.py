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
"""Winograd template for cuda backend"""

import logging
import tvm
from tvm import te
from tvm import autotvm

from .. import nn
from ..util import get_const_int, get_const_tuple, traverse_inline
from ..nn.winograd_util import winograd_transform_matrices

logger = logging.getLogger('conv3d_winograd')


def _infer_tile_size(data, kernel):
    N, CI, D, H, W = get_const_tuple(data.shape)

    if D % 8 == 0:
        return 4
    return 2


def winograd_cuda(cfg, data, kernel, strides, padding, dilation, out_dtype, pre_computed):
    """Compute declaration for winograd"""
    tile_size = _infer_tile_size(data, kernel)

    N, CI, D, H, W = get_const_tuple(data.shape)

    if isinstance(dilation, int):
        dilation_d = dilation_h = dilation_w = dilation
    else:
        dilation_d, dilation_h, dilation_w = dilation
    DSTR, HSTR, WSTR = (strides, strides, strides) if isinstance(strides, int) else strides

    if not pre_computed:  # kernel tensor is raw tensor, do strict check
        if dilation_d != 1 or dilation_h != 1 or dilation_w != 1:
            kernel = nn.dilate(kernel, (1, 1, dilation_d, dilation_h, dilation_w))
        CO, CI, KD, KH, KW = get_const_tuple(kernel.shape)
        alpha = KW + tile_size - 1
        assert DSTR == 1 and HSTR == 1 and WSTR == 1 and KD == KH and KH == KW
    else:
        # kernel tensor is pre-transfomred. this op is created by alter op layout.
        # dilation is not supported
        alpha, _, _, CI, CO = get_const_tuple(kernel.shape)
        KD = KH = KW = alpha + 1 - tile_size
        assert DSTR == 1 and HSTR == 1 and WSTR == 1 and dilation_d == 1 and dilation_h == 1 and dilation_w == 1

    pf, pt, pl, pb, pd, pr = nn.get_pad_tuple3d(padding, (KD, KH, KW))
    data_pad = nn.pad(data, (0, 0, pf, pt, pl), (0, 0, pb, pd, pr), name="data_pad")

    r = KW
    m = tile_size
    A, B, G = winograd_transform_matrices(m, r, out_dtype)

    D = (D + pf + pb - KD) // DSTR + 1
    H = (H + pt + pd - KH) // HSTR + 1
    W = (W + pl + pr - KW) // WSTR + 1
    nD, nH, nW = (D + m - 1) // m, (H + m - 1) // m, (W + m - 1) // m
    P = N * nD * nH * nW

    # transform kernel
    if not pre_computed:
        r_kd = te.reduce_axis((0, KD), name='r_kd')
        r_kh = te.reduce_axis((0, KH), name='r_kh')
        r_kw = te.reduce_axis((0, KW), name='r_kw')
        kernel_pack = te.compute(
            (alpha, alpha, alpha, CI, CO),
            lambda omg, eps, nu, ci, co: te.sum(
                kernel[co][ci][r_kd][r_kh][r_kw] * G[omg][r_kd] * G[eps][r_kh] * G[nu][r_kw],
                axis=[r_kd, r_kh, r_kw]),
            name='kernel_pack')
    else:
        kernel_pack = kernel

    idxdiv = tvm.tir.indexdiv
    idxmod = tvm.tir.indexmod
    # pack input tile
    input_tile = te.compute((CI, P, alpha, alpha, alpha),
                            lambda c, p, omg, eps, nu: data_pad[idxdiv(p, (nD * nH * nW))]
                            [c]
                            [idxmod(idxdiv(p, nH * nW), nD) * m + omg]
                            [idxmod(idxdiv(p, nW), nH) * m + eps]
                            [idxmod(p, nW) * m + nu],
                            name='d')

    # transform data
    r_a = te.reduce_axis((0, alpha), 'r_a')
    r_b = te.reduce_axis((0, alpha), 'r_b')
    r_c = te.reduce_axis((0, alpha), 'r_c')
    data_pack = te.compute(
        (alpha, alpha, alpha, CI, P),
        lambda omg, eps, nu, ci, p: te.sum(
            input_tile[ci][p][r_a][r_b][r_c] * B[r_a][omg] * B[r_b][eps] * B[r_c][nu],
            axis=[r_a, r_b, r_c]),
        name='data_pack')

    # do batch gemm
    ci = te.reduce_axis((0, CI), name='ci')
    bgemm = te.compute(
        (alpha, alpha, alpha, CO, P),
        lambda omg, eps, nu, co, p: te.sum(
            kernel_pack[omg][eps][nu][ci][co] * data_pack[omg][eps][nu][ci][p], axis=[ci]),
        name='bgemm')

    # inverse transform
    r_a = te.reduce_axis((0, alpha), 'r_a')
    r_b = te.reduce_axis((0, alpha), 'r_b')
    r_c = te.reduce_axis((0, alpha), 'r_c')
    inverse = te.compute((CO, P, m, m, m),
                         lambda co, p, vd, vh, vw: te.sum(
                             bgemm[r_a][r_b][r_c][co][p] * A[r_a][vd] * A[r_b][vh] * A[r_c][vw],
                             axis=[r_a, r_b, r_c]),
                         name='inverse')

    # output
    output = te.compute((N, CO, D, H, W),
                        lambda n, co, d, h, w: inverse[co, n * nD * nH * nW + idxdiv(d, m) * nH * nW
                                                       + idxdiv(h, m) * nW + idxdiv(w, m),
                                                       idxmod(d, m),
                                                       idxmod(h, m),
                                                       idxmod(w, m)],
                        name='output',
                        tag='conv3d_ncdhw_winograd')
    cfg.add_flop(2 * N * CO * D * H * W * CI * KD * KH * KW)

    return output


def schedule_winograd_cuda(cfg, s, output, pre_computed):
    """Schedule winograd template"""
    # get stages
    inverse = s[output].op.input_tensors[0]
    bgemm, A = s[inverse].op.input_tensors
    kernel_pack, data_pack = s[bgemm].op.input_tensors
    input_tile, B = s[data_pack].op.input_tensors
    pad_data = s[input_tile].op.input_tensors[0]

    # data transform
    s[B].compute_inline()

    data_l = s.cache_write(data_pack, 'local')
    omg, eps, nu, c, p = s[data_l].op.axis
    r_a, r_b, r_c = s[data_l].op.reduce_axis
    # TODO unrolling by omg, eps, nu may improve performance but
    # in some cases causes extremely long build times due to imperfect tiling.
    for axis in [r_a, r_b, r_c]:
        s[data_l].unroll(axis)

    omg, eps, nu, c, p = s[data_pack].op.axis
    p, pi = s[data_pack].split(p, 1)
    fused = s[data_pack].fuse(c, p)
    bb, tt = s[data_pack].split(fused, 128)
    s[data_pack].reorder(bb, tt, pi, omg, eps, nu)
    s[data_pack].bind(bb, te.thread_axis("blockIdx.x"))
    s[data_pack].bind(tt, te.thread_axis("threadIdx.x"))

    s[data_l].compute_at(s[data_pack], pi)
    s[input_tile].compute_at(s[data_pack], pi)
    s[pad_data].compute_inline()

    # transform kernel
    if not pre_computed:
        kernel, G = s[kernel_pack].op.input_tensors
        omg, eps, nu, ci, co = s[kernel_pack].op.axis
        if autotvm.GLOBAL_SCOPE.in_tuning:
            # skip this part during tuning to make recrods accurate
            # this part will be pre-computed during pre-compute optimization pass
            s[G].pragma(s[G].op.axis[0], 'debug_skip_region')
            s[kernel_pack].pragma(eps, 'debug_skip_region')
        else:
            s[G].compute_inline()
            r_a, r_b, r_c = s[kernel_pack].op.reduce_axis
            for axis in [omg, eps, nu, r_a, r_b, r_c]:
                s[kernel_pack].unroll(axis)

            fused = s[kernel_pack].fuse(ci, co)
            bb, tt = s[kernel_pack].split(fused, 128)
            s[kernel_pack].reorder(bb, tt, omg, eps, nu, r_a, r_b, r_c)
            s[kernel_pack].bind(bb, te.thread_axis("blockIdx.x"))
            s[kernel_pack].bind(tt, te.thread_axis("threadIdx.x"))
    else:
        kernel = kernel_pack

    if isinstance(kernel.op, tvm.te.ComputeOp) and "dilate" in kernel.op.tag:
        s[kernel].compute_inline()

    ##### space definition begin #####
    b1, b2, b3, y, x = s[bgemm].op.axis
    rc = s[bgemm].op.reduce_axis[0]
    alpha = get_const_int(b1.dom.extent)

    cfg.define_split(
        "tile_b",
        cfg.axis(alpha * alpha * alpha),
        num_outputs=4,
        filter=lambda x: x.size[-3:] == [1, 1, 1])
    cfg.define_split("tile_y", y, num_outputs=4)
    cfg.define_split("tile_x", x, num_outputs=4)
    cfg.define_split("tile_rc", rc, num_outputs=2)
    cfg.define_knob("auto_unroll_max_step", [0, 128, 1500])
    target = tvm.target.Target.current()
    if target.target_name in ['nvptx', 'rocm']:
        cfg.define_knob("unroll_explicit", [1])
    else:
        cfg.define_knob("unroll_explicit", [0, 1])
    ##### space definition end #####

    # batch gemm
    C = bgemm
    A0, B0 = kernel_pack, data_pack

    OL = s.cache_write(C, 'local')
    AA = s.cache_read(A0, 'shared', [OL])
    BB = s.cache_read(B0, 'shared', [OL])

    b = s[bgemm].fuse(b1, b2, b3)

    # tile and bind spatial axes
    bgemm_scope, b = s[bgemm].split(b, nparts=1)
    bz, vz, tz, zi = cfg["tile_b"].apply(s, C, b)
    by, vy, ty, yi = cfg["tile_y"].apply(s, C, y)
    bx, vx, tx, xi = cfg["tile_x"].apply(s, C, x)
    s[C].bind(bz, te.thread_axis("blockIdx.z"))
    s[C].bind(by, te.thread_axis("blockIdx.y"))
    s[C].bind(bx, te.thread_axis("blockIdx.x"))
    s[C].bind(vz, te.thread_axis("vthread"))
    s[C].bind(vy, te.thread_axis("vthread"))
    s[C].bind(vx, te.thread_axis("vthread"))
    s[C].bind(tz, te.thread_axis("threadIdx.z"))
    s[C].bind(ty, te.thread_axis("threadIdx.y"))
    s[C].bind(tx, te.thread_axis("threadIdx.x"))
    s[C].reorder(bgemm_scope, bz, by, bx, vz, vy, vx, tz, ty, tx, zi, yi, xi)

    # tile reduction axes
    s[OL].compute_at(s[C], tx)
    b1, b2, b3, y, x = s[OL].op.axis
    b = s[OL].fuse(b1, b2, b3)
    rc, = s[OL].op.reduce_axis
    rco, rci = cfg['tile_rc'].apply(s, OL, rc)
    s[OL].reorder(rco, rci, b, y, x)

    s[AA].compute_at(s[OL], rco)
    s[BB].compute_at(s[OL], rco)

    # cooperative fetching
    for load in [AA, BB]:
        fused = s[load].fuse(*list(s[load].op.axis))
        fused, tx = s[load].split(fused, cfg["tile_x"].size[2])
        fused, ty = s[load].split(fused, cfg["tile_y"].size[2])
        fused, tz = s[load].split(fused, cfg["tile_b"].size[2])
        s[load].bind(tz, te.thread_axis("threadIdx.z"))
        s[load].bind(ty, te.thread_axis("threadIdx.y"))
        s[load].bind(tx, te.thread_axis("threadIdx.x"))

    s[C].pragma(bgemm_scope, 'auto_unroll_max_step', cfg['auto_unroll_max_step'].val)
    s[C].pragma(bgemm_scope, 'unroll_explicit', cfg['unroll_explicit'].val)

    # schedule inverse, output and fusion
    if output.op in s.outputs:
        OL = None
    else:
        OL = output
        s[OL].set_scope('local')
        output = s.outputs[0]

    m = alpha - 3 + 1
    n, co, d, h, w = s[output].op.axis
    do, di = s[output].split(d, m)
    ho, hi = s[output].split(w, m)
    wo, wi = s[output].split(h, m)
    s[output].reorder(n, co, do, ho, wo, di, hi, wi)
    inverse_scope, n = s[output].split(n, nparts=1)

    fused = s[output].fuse(n, co, do, ho, wo)
    bb, tt = s[output].split(fused, 128)

    s[output].bind(bb, te.thread_axis("blockIdx.x"))
    s[output].bind(tt, te.thread_axis("threadIdx.x"))

    if OL is not None:
        s[OL].compute_at(s[output], tt)

    s[A].compute_inline()
    co, p, vd, vh, vw = s[inverse].op.axis
    r_a, r_b, r_c = s[inverse].op.reduce_axis
    for axis in [vd, vh, vw, r_a, r_b, r_c]:
        s[inverse].unroll(axis)
    s[inverse].compute_at(s[output], tt)

    return s


@autotvm.register_topi_compute("conv3d_ncdhw_winograd.cuda")
def conv3d_ncdhw_winograd(cfg, data, kernel, strides, padding, dilation, out_dtype):
    return winograd_cuda(
        cfg, data, kernel, strides, padding, dilation, out_dtype, pre_computed=False)


@autotvm.register_topi_schedule("conv3d_ncdhw_winograd.cuda")
def schedule_conv3d_ncdhw_winograd(cfg, outs):
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        if 'conv3d_ncdhw_winograd' in op.tag:
            schedule_winograd_cuda(cfg, s, op.output(0), pre_computed=False)

    traverse_inline(s, outs[0].op, _callback)
    return s


@autotvm.register_topi_compute("conv3d_ncdhw_winograd_without_weight_transform.cuda")
def conv3d_ncdhw_winograd_without_weight_transform(cfg, data, kernel, strides, padding, dilation,
                                                   out_dtype):
    return winograd_cuda(
        cfg, data, kernel, strides, padding, dilation, out_dtype, pre_computed=True)


@autotvm.register_topi_schedule("conv3d_ncdhw_winograd_without_weight_transform.cuda")
def schedule_conv3d_ncdhw_winograd_without_weight_transform(cfg, outs):
    """TOPI schedule callback"""
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        if 'conv3d_ncdhw_winograd' in op.tag:
            schedule_winograd_cuda(cfg, s, op.output(0), pre_computed=True)

    traverse_inline(s, outs[0].op, _callback)
    return s
