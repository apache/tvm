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
import tvm
from tvm import autotvm
from tvm.autotvm.task.space import get_factors

from ..generic import schedule_conv2d_nchw, schedule_conv2d_winograd_without_weight_transform
from ..util import traverse_inline, get_const_int, get_const_tuple
from ..nn import conv2d, conv2d_winograd_without_weight_transform, \
    get_pad_tuple, pad, conv2d_alter_layout
from ..nn.winograd_util import winograd_transform_matrices

# reuse some compute declarations from ARM CPU
from ..arm_cpu.conv2d import _alter_conv2d_layout_arm
from ..arm_cpu.conv2d_spatial_pack import conv2d_spatial_pack_nchw


@autotvm.register_topi_compute(conv2d, 'mali', ['direct'])
def conv2d_mali(cfg, data, kernel, strides, padding, dilation, layout, out_dtype):
    """TOPI compute callback for conv2d

    Parameters
    ----------
    cfg: ConfigEntity
        The config for this template

    data : tvm.Tensor
        4-D with shape [batch, in_channel, in_height, in_width]

    kernel : tvm.Tensor
        4-D with shape [num_filter, in_channel, filter_height, filter_width] or
        pre-packed 5-D with shape [num_filter_chunk, in_channel, filter_height,
        filter_width, num_filter_block]

    strides : list of two ints
        [stride_height, stride_width]

    padding : list of two ints
        [pad_height, pad_width]

    dilation : list of two ints
        [dilation_height, dilation_width]

    layout : str
        layout of data

    out_dtype: str
        The output type. This is used for mixed precision.

    Returns
    -------
    output : tvm.Tensor
        4-D with shape [batch, out_channel, out_height, out_width]
    """
    if layout == 'NCHW':
        return conv2d_spatial_pack_nchw(cfg, data, kernel, strides, padding,
                                        dilation, out_dtype, num_tile=3)
    else:
        raise ValueError("Unsupported layout {}".format(layout))

@autotvm.register_topi_schedule(schedule_conv2d_nchw, 'mali', ['direct', 'winograd'])
def schedule_conv2d_nchw_mali(cfg, outs):
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
    s = tvm.create_schedule([x.op for x in outs])

    def _callback(op):
        # schedule conv2d
        if 'spatial_conv2d_output' in op.tag:
            output = op.output(0)
            conv = op.input_tensors[0]

            data_vec = conv.op.input_tensors[0]
            data_pad = data_vec.op.input_tensors[0]
            s[data_pad].compute_inline()

            kernel_vec = conv.op.input_tensors[1]
            if kernel_vec.op.name == 'kernel_vec':
                kernel = kernel_vec.op.input_tensors[0]
            else:
                kernel = kernel_vec
            if isinstance(kernel.op, tvm.tensor.ComputeOp) and "dilate" in kernel.op.tag:
                s[kernel].compute_inline()

            _schedule_spatial_pack(cfg, s, output, conv, data_vec, kernel_vec)

        if 'winograd_conv2d_output' in op.tag:
            _schedule_winograd(cfg, s, op)

    traverse_inline(s, outs[0].op, _callback)
    return s


def _schedule_spatial_pack(cfg, s, output, conv, data_vec, kernel_vec):
    """schedule the spatial packing for conv2d"""
    data = s[data_vec].op.input_tensors[0]

    max_unroll = 16
    vec_size = [1, 2, 4, 8, 16]
    # get tunable parameters (they are defined in compute)
    BC, TC, VC = cfg["tile_co"].size
    BH, TH, VH = cfg["tile_oh"].size
    BW, TW, VW = cfg["tile_ow"].size

    # schedule padding
    if isinstance(data.op, tvm.tensor.ComputeOp) and "pad" in data.op.tag:
        data_pad = data
        s[data_pad].compute_inline()

    # schedule data packing
    if isinstance(data_vec.op, tvm.tensor.ComputeOp) and data_vec.op.name == 'data_vec_undilated':
        _, h, w, ci, _, _, vh, vw = s[data_vec].op.axis
    else:
        _, h, w, ci, vh, vw = s[data_vec].op.axis
    tile_and_bind3d(s, data_vec, h, w, ci, 1)
    if vh.dom.extent.value < max_unroll:
        s[data_vec].unroll(vh)
    if vw.dom.extent.value < max_unroll:
        s[data_vec].unroll(vw)

    if isinstance(kernel_vec.op, tvm.tensor.ComputeOp) and kernel_vec.name == 'kernel_vec':
        if autotvm.GLOBAL_SCOPE.in_tuning:
            # kernel packing will be pre-computed during compilation, so we skip
            # this part to make tuning records correct
            s[kernel_vec].pragma(s[kernel_vec].op.axis[0], 'debug_skip_region')
        else:
            max_threads = tvm.target.current_target(allow_none=False).max_num_threads
            co, ci, kh, kw, vc = s[kernel_vec].op.axis
            fused = s[kernel_vec].fuse(co, ci, kh, kw, vc)
            fused, vec = s[kernel_vec].split(fused, VC)
            bb, tt = s[kernel_vec].split(fused, max_threads)
            s[kernel_vec].bind(bb, tvm.thread_axis("blockIdx.x"))
            s[kernel_vec].bind(tt, tvm.thread_axis("threadIdx.x"))
            if VC in vec_size:
                s[kernel_vec].vectorize(vec)

    # schedule convolution
    n, c, h, w, vh, vw, vc = s[conv].op.axis
    kc, kh, kw = s[conv].op.reduce_axis

    cfg["reorder_0"].apply(s, conv, [n, c, h, w, kc, kh, kw, vh, vw, vc])
    tile_and_bind3d(s, conv, c, h, w, TC, TH, TW)

    cfg["ann_reduce"].apply(s, conv, [kh, kw],
                            axis_lens=[get_const_int(kernel_vec.shape[2]),
                                       get_const_int(kernel_vec.shape[3])],
                            max_unroll=max_unroll)

    cfg["ann_spatial"].apply(s, conv, [vh, vw, vc],
                             axis_lens=[VH, VW, VC],
                             max_unroll=max_unroll,
                             vec_size=vec_size,
                             cfg=cfg)

    # schedule output
    if output.op not in s.outputs:  # has bias
        s[output].compute_inline()
        output = s.outputs[0]

    _, co, oh, ow = s[output].op.axis
    tile_and_bind3d(s, output, co, oh, ow, TC, TH, TW)

    return s

##### WINOGRAD TEMPLATE #####
def _pick_tile_size(data, kernel):
    N, CI, H, W = get_const_tuple(data.shape)

    if H % 4 == 0:
        return 4
    else:
        return 2

@autotvm.register_topi_compute(conv2d, 'mali', ['winograd'])
def conv2d_mali_winograd(cfg, data, kernel, strides, padding, dilation, layout, out_dtype):
    tile_size = _pick_tile_size(data, kernel)
    return _decl_winograd(cfg, data, kernel, strides, padding, dilation, layout, out_dtype,
                          tile_size)

def _decl_winograd(cfg, data, kernel, strides, padding, dilation, layout, out_dtype, tile_size):
    N, CI, IH, IW = get_const_tuple(data.shape)
    if isinstance(dilation, int):
        dilation_h = dilation_w = dilation
    else:
        dilation_h, dilation_w = dilation

    if len(kernel.shape) == 4:

        if dilation_h != 1 or dilation_w != 1:
            kernel = dilate(kernel, (1, 1, dilation_h, dilation_w))
        pre_computed = False
        CO, _, KH, KW = get_const_tuple(kernel.shape)
    else:
        assert (dilation_h, dilation_w) == (1, 1), "Does not support dilation"
        pre_computed = True
        H_CAT, W_CAT, CO, CI, VC = get_const_tuple(kernel.shape)
        CO *= VC
        KH, KW = H_CAT - tile_size + 1, W_CAT - tile_size + 1
    HSTR, WSTR = strides if isinstance(strides, (tuple, list)) else (strides, strides)
    HPAD, WPAD, _, _ = get_pad_tuple(padding, kernel)

    assert layout == 'NCHW'
    assert KH == 3 and KW == 3 and HSTR == 1 and WSTR == 1
    data_pad = pad(data, (0, 0, HPAD, WPAD), name="data_pad")

    r = KW
    m = tile_size
    alpha = m + r - 1
    A, B, G = winograd_transform_matrices(m, r, out_dtype)

    H = (IH + 2 * HPAD - 3) // HSTR + 1
    W = (IW + 2 * WPAD - 3) // WSTR + 1
    nH, nW = (H + m-1) // m, (W + m-1) // m
    P = N * nH * nW

    ##### space definition begin #####
    tile_bna_candidates = [1, 2, 4, 8, 16]
    factors = get_factors(CO)
    cfg.define_knob('tile_bna', [x for x in tile_bna_candidates if x in factors])
    cfg.define_knob('tile_bnb', [1, 2, 4, 8, 16])
    cfg.define_split('tile_t1', CI, num_outputs=2, max_factor=128)
    cfg.define_split('tile_t2', CO, num_outputs=2, max_factor=128)
    cfg.define_split('c_unroll', CI, num_outputs=2, max_factor=8)
    cfg.define_knob('yt', [1, 2, 4, 8, 16, 32])
    ##### space definition end #####

    if cfg.is_fallback:
        cfg['tile_bnb'].val = 4
        cfg['tile_bna'].val = 4
        while CO % cfg['tile_bna'].val != 0:
            cfg['tile_bna'].val //= 2
        cfg['yt'].val = 8
        cfg.fallback_split('tile_t1', [-1, 128])
        cfg.fallback_split('tile_t2', [-1, 128])
        cfg.fallback_split('c_unroll', [-1, 8])

    bna = cfg['tile_bna'].val
    bnb = cfg['tile_bnb'].val

    P_round = (P + bnb - 1) // bnb * bnb
    assert CO % bna == 0 and P_round % bnb == 0

    # pack input tile
    input_tile = tvm.compute((CI, P_round // bnb, alpha, alpha, bnb), lambda ci, b, eps, nu, bb: \
         tvm.if_then_else(
             b * bnb + bb < P,
             data_pad[(b*bnb+bb) // (nH*nW)][ci][(b*bnb+bb) // nW % nH * m + eps]
             [(b*bnb+bb) % nW * m + nu], tvm.const(0, data_pad.dtype)), name='d')

    # transform kernel
    if pre_computed:
        U = kernel
    else:
        r_kh = tvm.reduce_axis((0, KH), 'r_kh')
        r_kw = tvm.reduce_axis((0, KW), 'r_kw')
        U = tvm.compute((alpha, alpha, CO // bna, CI, bna), lambda eps, nu, co, ci, vco:
                        tvm.sum(kernel[co * bna + vco][ci][r_kh][r_kw] * G[eps][r_kh] * G[nu][r_kw],
                                axis=[r_kh, r_kw]), name='U')

    # transform image
    r_a = tvm.reduce_axis((0, alpha), 'r_a')
    r_b = tvm.reduce_axis((0, alpha), 'r_b')
    V = tvm.compute((alpha, alpha, P_round // bnb, CI, bnb), lambda eps, nu, p, ci, vp:
                    tvm.sum(input_tile[ci][p][r_a][r_b][vp] * B[r_a][eps] * B[r_b][nu],
                            axis=[r_a, r_b]), name='V')

    idxdiv = tvm.indexdiv
    idxmod = tvm.indexmod

    # batch gemm
    ci = tvm.reduce_axis((0, CI), name='c')
    M = tvm.compute((alpha, alpha, CO, P_round), lambda eps, nu, co, p:
                    tvm.sum(U[eps][nu][idxdiv(co, bna)][ci][idxmod(co, bna)] *
                            V[eps][nu][idxdiv(p, bnb)][ci][idxmod(p, bnb)], axis=ci), name='M')

    r_a = tvm.reduce_axis((0, alpha), 'r_a')
    r_b = tvm.reduce_axis((0, alpha), 'r_b')
    Y = tvm.compute((CO, P, m, m), lambda co, p, vh, vw:
                    tvm.sum(M[r_a][r_b][co][p] * A[r_a][vh] * A[r_b][vw],
                            axis=[r_a, r_b]), name='Y')

    # unpack output
    output = tvm.compute((N, CO, H, W), lambda n, co, h, w:
                         Y[co, n * nH * nW + idxdiv(h, m) * nW + idxdiv(w, m),
                           idxmod(h, m), idxmod(w, m)]
                         # The following hack term is used to make the padding in batch gemm ("M")
                         # effective, otherwise the padding will be eliminated by bound inference.
                         # Use `tvm.expr.Mul` instead of `*` to avoid issues in const folding.
                         + tvm.expr.Mul(tvm.const(0, out_dtype),
                                        M[alpha-1][alpha-1][CO-1][P_round-1]),
                         name='output', tag='winograd_conv2d_output')

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
    if isinstance(U.op, tvm.tensor.ComputeOp):
        kernel, G = s[U].op.input_tensors
        s[G].compute_inline()
        eps, nu, co, ci, vco, = s[U].op.axis
        if autotvm.GLOBAL_SCOPE.in_tuning:
            # kernel transformation will be pre-computed during compilation, so we skip
            # this part to make tuning records correct
            s[U].pragma(eps, 'debug_skip_region')
        else:
            r_kh, r_kw = s[U].op.reduce_axis
            s[U].reorder(co, ci, eps, nu, r_kh, r_kw, vco)
            _ = [s[U].unroll(x) for x in [eps, nu, r_kh, r_kw]]
            s[U].vectorize(vco)
            tile_and_bind(s, U, co, ci, 1, 256)

        # dilation
        if isinstance(kernel.op, tvm.tensor.ComputeOp) and "dilate" in kernel.op.tag:
            s[kernel].compute_inline()

    # transform image
    s[B].compute_inline()
    VL = s.cache_write(V, 'local')

    eps, nu, p, ci, vp = s[V].op.axis
    s[V].reorder(p, ci, eps, nu, vp)
    for axis in [eps, nu]:
        s[V].unroll(axis)
    s[V].vectorize(vp)
    fused = s[V].fuse(p, ci)

    bb, tt = cfg['tile_t1'].apply(s, V, fused)
    s[V].bind(bb, tvm.thread_axis('blockIdx.x'))
    s[V].bind(tt, tvm.thread_axis('threadIdx.x'))

    eps, nu, p, ci, vp = s[VL].op.axis
    r_a, r_b = s[VL].op.reduce_axis
    for axis in [eps, nu, r_a, r_b]:
        s[VL].unroll(axis)
    s[VL].vectorize(vp)
    s[d].compute_at(s[V], tt)
    s[VL].compute_at(s[V], tt)

    # batch gemm
    bna = cfg['tile_bna'].val
    bnb = cfg['tile_bnb'].val

    eps, nu, k, b = s[M].op.axis
    alpha = eps.dom.extent
    c = s[M].op.reduce_axis[0]
    yo, xo, yi, xi = s[M].tile(k, b, bna, bnb)
    c, c_unroll = cfg['c_unroll'].apply(s, M, c)
    s[M].reorder(yo, xo, c, c_unroll, yi, xi)
    s[M].unroll(c_unroll)
    s[M].unroll(yi)
    s[M].vectorize(xi)
    z = s[M].fuse(eps, nu)
    tile_and_bind3d(s, M, z, yo, xo, 1, cfg['yt'].val, 1)

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
    bb, tt = cfg['tile_t2'].apply(s, output, fused)
    s[output].bind(bb, tvm.thread_axis('blockIdx.x'))
    s[output].bind(tt, tvm.thread_axis('threadIdx.x'))

    s[Y].compute_at(s[output], tt)

##### REGISTER TOPI COMPUTE / SCHEDULE FOR WINOGRAD WITH WEIGHT TRANSFORM #####
@autotvm.register_topi_compute(conv2d_winograd_without_weight_transform, 'mali', ['winograd'])
def conv2d_winograd_ww(cfg, data, kernel, strides, padding, dilation, layout, out_dtype, tile_size):
    """TOPI compute callback"""
    return _decl_winograd(cfg, data, kernel, strides, padding, dilation, layout, out_dtype,
                          tile_size)


@autotvm.register_topi_schedule(schedule_conv2d_winograd_without_weight_transform,
                                'mali', ['winograd'])
def schedule_conv2d_winograd_without_weight_transform_(cfg, outs):
    """TOPI schedule callback"""
    s = tvm.create_schedule([x.op for x in outs])

    def _callback(op):
        if 'winograd_conv2d_output' in op.tag:
            _schedule_winograd(cfg, s, op)

    traverse_inline(s, outs[0].op, _callback)
    return s


##### REGISTER ALTER OP LAYOUT #####
@conv2d_alter_layout.register(["mali"])
def _alter_conv2d_layout(attrs, inputs, tinfos, F):
    try:
        return _alter_conv2d_layout_arm(attrs, inputs, tinfos, F)
    except KeyError:  # to filter out fallback opencl templates
        return None


##### SCHECULE UTILITIES #####
def tile_and_bind(s, tensor, y, x, y_factor, x_factor=None):
    """ tile and bind to GPU threads """
    x_factor = x_factor or y_factor
    yo, xo, yi, xi = s[tensor].tile(y, x, y_factor, x_factor)
    s[tensor].bind(xo, tvm.thread_axis("blockIdx.x"))
    s[tensor].bind(xi, tvm.thread_axis("threadIdx.x"))
    s[tensor].bind(yo, tvm.thread_axis("blockIdx.y"))
    s[tensor].bind(yi, tvm.thread_axis("threadIdx.y"))
    return yo, xo, yi, xi


def tile_and_bind3d(s, tensor, z, y, x, z_factor=2, y_factor=None, x_factor=None):
    """ tile and bind 3d """
    y_factor = y_factor or z_factor
    x_factor = x_factor or y_factor
    zo, zi = s[tensor].split(z, z_factor)
    yo, yi = s[tensor].split(y, y_factor)
    xo, xi = s[tensor].split(x, x_factor)
    s[tensor].bind(zo, tvm.thread_axis("blockIdx.z"))
    s[tensor].bind(zi, tvm.thread_axis("threadIdx.z"))
    s[tensor].bind(yo, tvm.thread_axis("blockIdx.y"))
    s[tensor].bind(yi, tvm.thread_axis("threadIdx.y"))
    s[tensor].bind(xo, tvm.thread_axis("blockIdx.x"))
    s[tensor].bind(xi, tvm.thread_axis("threadIdx.x"))
    s[tensor].reorder(zo, yo, xo, zi, yi, xi)
    return zo, yo, xo, zi, yi, xi
