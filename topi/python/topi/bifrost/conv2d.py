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
"""conv2d schedule on ARM Mali (Bifrost) GPU"""

import tvm
from tvm import autotvm

from .gemm import decl_winograd_gemm, schedule_gemm
from .transforms import tile_and_bind, tile_and_bind3d
from ..generic import schedule_conv2d_nchw, schedule_conv2d_winograd_without_weight_transform
from ..util import traverse_inline, get_const_int, get_const_tuple
from ..nn import conv2d, conv2d_winograd_without_weight_transform, \
    get_pad_tuple, pad, conv2d_alter_layout, dilate
from ..nn.winograd_util import winograd_transform_matrices

# reuse some compute declarations from ARM CPU
from ..arm_cpu.conv2d_spatial_pack import conv2d_spatial_pack_nchw
from ..arm_cpu.conv2d import _alter_conv2d_layout_arm


@autotvm.register_topi_compute(conv2d, 'bifrost', ['direct'])
def conv2d_bifrost(cfg, data, kernel, strides, padding, dilation, layout, out_dtype):
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


@autotvm.register_topi_schedule(schedule_conv2d_nchw, 'bifrost', ['direct', 'winograd'])
def schedule_conv2d_nchw_bifrost(cfg, outs):
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


@autotvm.register_topi_compute(conv2d, 'bifrost', ['winograd'])
def conv2d_bifrost_winograd(cfg, data, kernel, strides, padding, dilation, layout, out_dtype):
    """Use Winograd as the convolution method"""
    return _decl_winograd(cfg, data, kernel, strides, padding, dilation, layout, out_dtype)


def _decl_winograd_kernel_transform(kernel, tile_size, G):
    """Declare a Winograd kernel transform
    This exists separately to allow for precomputation
    The precomputation will most often happen on CPU

    Parameters
    ----------
    kernel : tvm.Tensor
        The kernel to transform

    tile_size : int
        The size of the tile to use for the Winograd filter

    Returns
    -------
    U : tvm.Tensor
        Transformed kernel

    """
    CO, CI, KH, KW = [get_const_int(x) for x in kernel.shape]
    # Only support 32 bit floats
    out_dtype = 'float32'

    alpha = G.shape[0]
    K = CO
    C = CI

    def upround(x, align):
        return (x + align - 1) // align * align

    ALIGN = 16
    K_round = upround(K, ALIGN)

    # Padded Kernel [K_round, C, KH, KW]
    # Pad the number of kernels to multiple of ALIGN
    padded_kernel = tvm.compute((K_round, C, KH, KW),
                                lambda k, c, h, w:
                                tvm.if_then_else(k < K,
                                                 kernel[k][c][h][w],
                                                 tvm.const(0, out_dtype)),
                                name='padded_kernel')

    # U [alpha, alpha, K_round, C]
    # Perform the kernel transform
    r_kh = tvm.reduce_axis((0, KH), 'r_kh')
    r_kw = tvm.reduce_axis((0, KW), 'r_kw')
    U = tvm.compute((alpha, alpha, K_round, C),
                    lambda eps, nu, k, c:
                    tvm.sum(padded_kernel[k][c][r_kh][r_kw] * G[eps][r_kh] * G[nu][r_kw],
                            axis=[r_kh, r_kw]),
                    name='U')

    return U


def _decl_winograd(cfg, data, kernel, strides, padding, dilation, layout, out_dtype, tile_size=2):
    """Declare a winograd convolution - only tile_size=2 is currently supported"""
    N, CI, IH, IW = get_const_tuple(data.shape)
    if isinstance(dilation, int):
        dilation_h = dilation_w = dilation
    else:
        dilation_h, dilation_w = dilation

    if int(kernel.shape[2]) == 3:
        if dilation_h != 1 or dilation_w != 1:
            kernel = dilate(kernel, (1, 1, dilation_h, dilation_w))
        pre_computed = False
        CO, _, KH, KW = get_const_tuple(kernel.shape)
    else:
        assert (dilation_h, dilation_w) == (1, 1), "Does not support dilation"
        pre_computed = True
        H_CAT, W_CAT, CO, CI = get_const_tuple(kernel.shape)
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

    K = CO
    C = CI
    H = (IH + 2 * HPAD - 3) // HSTR + 1
    W = (IW + 2 * WPAD - 3) // WSTR + 1
    nH, nW = (H + m-1) // m, (W + m-1) // m
    P = N * nH * nW

    def upround(x, align):
        return (x + align - 1) // align * align

    ALIGN = 16
    P_round = upround(P, ALIGN)
    K_round = upround(K, ALIGN)

    # CONFIG

    cfg.define_knob("data_transform_wgx", [1, 2, 4, 8, 16, 32, 64])
    cfg.define_knob("data_transform_wgy", [1, 2, 4, 8, 16, 32, 64])

    # Pack input tile
    input_tile = tvm.compute((N, C, H + 2, W + 2),
                             lambda n, c, h, w:
                             data_pad[n][c][h][w],
                             name='d')

    if pre_computed:
        U = kernel
    else:
        U = _decl_winograd_kernel_transform(kernel, tile_size, G)

    # V [alpha * alpha, C, P_round)
    # Perform the image transform
    r_eps = tvm.reduce_axis((0, alpha), 'r_eps')
    r_nu = tvm.reduce_axis((0, alpha), 'r_nu')
    V = tvm.compute((alpha * alpha, C, P_round),
                    lambda epsnu, c, b:
                    tvm.sum(input_tile[b // (nH*nW)][c][b // nW % nH * m + r_eps][b % nW * m +r_nu]\
                            * B[r_eps][epsnu // alpha] * B[r_nu][epsnu % alpha],
                            axis=[r_eps, r_nu]),
                    name='V')

    # Winograd GEMM is a wrapper around batched GEMM to convert U to a 3D Tensor
    _, M = decl_winograd_gemm(cfg, U, V)

    # Y [K, P, m, m]
    # Winograd output transform
    r_eps = tvm.reduce_axis((0, alpha), 'r_eps')
    r_nu = tvm.reduce_axis((0, alpha), 'r_nu')
    Y = tvm.compute((K, P, m, m), lambda k, b, vh, vw:
                    tvm.sum(M[r_eps * alpha + r_nu][k][b] * A[r_eps][vh] * A[r_nu][vw],
                            axis=[r_eps, r_nu]), name='Y')

    # Output [N, K, H, W]
    # Unpack back to NCHW format
    # The last term ensures alignment is not lost to bound inference
    output = tvm.compute((N, K, H, W), lambda n, k, h, w:
                         Y[k][n * nH * nW + (h//m) * nW + w//m][h % m][w % m]
                         + tvm.const(0, out_dtype) * M[(alpha*alpha)-1][K_round-1][P_round-1],
                         name='output', tag='winograd_conv2d_output')

    return output


def _schedule_winograd(cfg, s, op):
    """Schedule Winograd convolution for Bifrost"""

    # Get ops and tensors
    output = op.output(0)

    Y = op.input_tensors[0]
    M, A = s[Y].op.input_tensors
    U_3D, V = s[M].op.input_tensors
    U = s[U_3D].op.input_tensors[0]
    d, B = s[V].op.input_tensors
    data_pad = s[d].op.input_tensors[0]

    if isinstance(U.op, tvm.tensor.ComputeOp):
        padded_kernel, G = s[U].op.input_tensors
        kernel = s[padded_kernel].op.input_tensors[0]
        s[G].compute_inline()
        eps, _, _, _ = s[U].op.axis
        y, _, _, _ = s[padded_kernel].op.axis
        if autotvm.GLOBAL_SCOPE.in_tuning:
            # Kernel transformation will be pre-computed during compilation, so we skip
            # this part to make tuning records correct
            s[U].pragma(eps, 'debug_skip_region')
            s[padded_kernel].pragma(y, 'debug_skip_region')
        else:
            # Pad kernel
            y, x, ky, kx = s[padded_kernel].op.axis
            s[padded_kernel].unroll(ky)
            s[padded_kernel].unroll(kx)
            tile_and_bind(s, padded_kernel, y, x, 1, 8)

            # Transform kernel
            eps, nu, k, c = s[U].op.axis
            s[U].reorder(k, c, eps, nu)
            r_kh, r_kw = s[U].op.reduce_axis
            _ = [s[U].unroll(x) for x in [eps, nu, r_kh, r_kw]]

            yo, xo, yi, xi = tile_and_bind(s, U, k, c, 1, 4)

        # Dilation
        if isinstance(kernel.op, tvm.tensor.ComputeOp) and "dilate" in kernel.op.tag:
            s[kernel].compute_inline()

    # Pad data
    s[data_pad].compute_inline()

    # Pack data
    n, c, h, w = s[d].op.axis
    w, wi = s[d].split(w, 4)
    s[d].unroll(wi)
    b = s[d].fuse(n, c)
    tile_and_bind3d(s, d, b, h, w, 1, 4, 2)

    # Transform data
    bIL_d = s.cache_read(d, 'local', [V])

    s[B].compute_inline()
    epsnu, c, b = s[V].op.axis
    r_eps, r_nu = s[V].op.reduce_axis
    s[V].reorder(b, c, epsnu, r_nu, r_eps)
    _ = [s[V].unroll(x) for x in [epsnu, r_eps, r_nu]]
    yo, xo, yi, xi = tile_and_bind(
        s, V, b, c, cfg["data_transform_wgy"].val, cfg["data_transform_wgx"].val
    )

    s[bIL_d].compute_at(s[V], xi)
    n, c, h, w = s[bIL_d].op.axis
    s[bIL_d].unroll(h)
    s[bIL_d].vectorize(w)

    # Batched GEMM
    # Inline the 4D -> 3D tensor transform on the kernel
    s[U_3D].compute_inline()
    U_transform, V_transform = schedule_gemm(
        cfg, s, U_3D, V, M, batched=True, schedule_transforms=True
    )

    # Inverse transform
    CR_M = s.cache_read(M, 'local', [Y])
    CW_Y = s.cache_write(Y, 'local')

    s[A].compute_inline()
    k, b, vh, vw = s[Y].op.axis
    fused = s[Y].fuse(vh, vw)
    s[Y].vectorize(fused)
    yo, xo, yi, xi = tile_and_bind(s, Y, k, b, 1, 4)

    s[CR_M].compute_at(s[Y], xi)
    k, b, epsnu = s[CR_M].op.axis
    s[CR_M].unroll(k)

    s[CW_Y].compute_at(s[Y], xi)
    k, b, vh, vw = s[CW_Y].op.axis
    r_eps, r_nu = s[CW_Y].op.reduce_axis
    _ = [s[CW_Y].unroll(x) for x in [vh, vw, r_eps, r_nu]]

    # Schedule output and fusion
    if output.op not in s.outputs:
        s[output].compute_inline()
        output = s.outputs[0]

    _, k, h, w = s[output].op.axis
    tile_and_bind3d(s, output, k, h, w, 1, 2, 2)


##### REGISTER TOPI COMPUTE / SCHEDULE FOR WINOGRAD WITH WEIGHT TRANSFORM #####
@autotvm.register_topi_compute(conv2d_winograd_without_weight_transform, 'bifrost', ['winograd'])
def conv2d_winograd_ww(cfg, data, kernel, strides, padding, dilation, layout, out_dtype, tile_size):
    """TOPI compute callback"""
    return _decl_winograd(cfg, data, kernel, strides, padding, dilation, layout, out_dtype)


@autotvm.register_topi_schedule(schedule_conv2d_winograd_without_weight_transform,
                                'bifrost', ['winograd'])
def schedule_conv2d_winograd_without_weight_transform_(cfg, outs):
    """TOPI schedule callback"""
    s = tvm.create_schedule([x.op for x in outs])

    def _callback(op):
        if 'winograd_conv2d_output' in op.tag:
            _schedule_winograd(cfg, s, op)

    traverse_inline(s, outs[0].op, _callback)
    return s


##### REGISTER ALTER OP LAYOUT #####
@conv2d_alter_layout.register(["bifrost"])
def _alter_conv2d_layout(attrs, inputs, tinfos, F):
    try:
        return _alter_conv2d_layout_arm(attrs, inputs, tinfos, F)
    except KeyError:  # to filter out fallback opencl templates
        return None
