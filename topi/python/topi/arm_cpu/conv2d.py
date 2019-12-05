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
# pylint: disable=invalid-name, unused-variable, no-else-return, unused-argument
"""Conv2D schedule for ARM CPU"""
from __future__ import absolute_import as _abs

import logging

import tvm
from tvm import autotvm
import tvm.contrib.nnpack

from ..generic import schedule_conv2d_nchw, schedule_conv2d_winograd_without_weight_transform, \
                      schedule_conv2d_winograd_nnpack_without_weight_transform
from ..util import traverse_inline, get_const_tuple
from ..nn import dilate, pad, conv2d, conv2d_alter_layout, \
                 conv2d_winograd_without_weight_transform, \
                 conv2d_winograd_nnpack_without_weight_transform, \
                 depthwise_conv2d_nchw
from ..nn.util import get_const_int, get_pad_tuple
from ..nn.winograd_util import winograd_transform_matrices
from .conv2d_spatial_pack import conv2d_spatial_pack_nchw, \
                                 schedule_conv2d_spatial_pack_nchw

logger = logging.getLogger('topi')

@autotvm.register_topi_compute(conv2d, 'arm_cpu', ['direct'])
def conv2d_arm_cpu(cfg, data, kernel, strides, padding, dilation, layout, out_dtype):
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
                                        dilation, out_dtype, num_tile=2)
    else:
        raise ValueError("Unsupported layout {}".format(layout))


@autotvm.register_topi_schedule(
    schedule_conv2d_nchw, 'arm_cpu',
    ['direct', 'winograd', 'winograd_nnpack_fp16', 'winograd_nnpack_fp32'])
def schedule_conv2d_nchw_arm_cpu(cfg, outs):
    """TOPI schedule callback for conv2d

    Parameters
    ----------
    cfg: ConfigEntity
        The config for this template

    outs: Array of Tensor
        The computation graph description of conv2d
        in the format of an array of tensors.

    Returns
    -------
    s: Schedule
        The computation schedule for conv2d.
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

            schedule_conv2d_spatial_pack_nchw(cfg, s, data_vec, kernel_vec,
                                              conv, output, outs[0])

        if 'winograd_conv2d_output' in op.tag:
            output = op.output(0)
            _schedule_winograd(cfg, s, output, outs[0])

        if 'winograd_nnpack_conv2d_output' in op.tag:
            output = op.output(0)
            _schedule_winograd_nnpack(cfg, s, output, outs[0])

    traverse_inline(s, outs[0].op, _callback)
    return s

@autotvm.register_topi_compute(conv2d, 'arm_cpu', ['winograd'])
def conv2d_arm_cpu_winograd(cfg, data, kernel, strides, padding, dilation, layout, out_dtype):
    """ TOPI compute callback. Use winograd template """
    tile_size = 4
    return _decl_winograd(cfg, data, kernel, strides, padding, dilation, layout,
                          out_dtype, tile_size)

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

    idxd = tvm.indexdiv
    idxm = tvm.indexmod

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

    cfg.define_split('tile_p', cfg.axis(P), num_outputs=2, filter=lambda x: x.size[-1] <= 16)
    cfg.define_split('tile_k', cfg.axis(K), num_outputs=2, filter=lambda x: x.size[-1] <= 16)
    VP = cfg['tile_p'].size[-1]
    VK = cfg['tile_k'].size[-1]

    # pack input tile
    input_tile = tvm.compute((C, idxd(P, VP), alpha, alpha, VP),
                             lambda c, b, eps, nu, bb:
                             data_pad[idxd(b*VP + bb, nH*nW), c,
                                      idxm(idxd(b*VP + bb, nW), nH) * m + eps,
                                      idxm(b*VP + bb, nW) * m + nu],
                             name='d')

    # transform kernel
    if pre_computed:
        U = kernel
    else:
        r_kh = tvm.reduce_axis((0, KH), 'r_kh')
        r_kw = tvm.reduce_axis((0, KW), 'r_kw')
        U = tvm.compute((alpha, alpha, idxd(K, VK), C, VK), lambda eps, nu, k, c, kk:
                        tvm.sum(kernel[k * VK + kk][c][r_kh][r_kw].astype(out_dtype) *
                                G[eps][r_kh] * G[nu][r_kw], axis=[r_kh, r_kw]), name='U')

    # transform image
    r_eps = tvm.reduce_axis((0, alpha), 'r_eps')
    r_nu = tvm.reduce_axis((0, alpha), 'r_nu')
    V = tvm.compute((alpha, alpha, idxd(P, VP), C, VP), lambda eps, nu, b, c, bb:
                    tvm.sum(input_tile[c][b][r_eps][r_nu][bb].astype(out_dtype) *
                            B[r_eps][eps] * B[r_nu][nu], axis=[r_eps, r_nu]), name='V')

    # batch gemm
    c = tvm.reduce_axis((0, C), name='c')
    M = tvm.compute((alpha, alpha, K, P), lambda eps, nu, k, b:
                    tvm.sum(U[eps][nu][idxd(k, VK)][c][idxm(k, VK)] *
                            V[eps][nu][idxd(b, VP)][c][idxm(b, VP)], axis=c), name='M')

    # inverse transform
    r_eps = tvm.reduce_axis((0, alpha), 'r_eps')
    r_nu = tvm.reduce_axis((0, alpha), 'r_nu')
    Y = tvm.compute((K, P, m, m), lambda k, b, vh, vw:
                    tvm.sum(M[r_eps][r_nu][k][b] * A[r_eps][vh] * A[r_nu][vw],
                            axis=[r_eps, r_nu]), name='Y')

    # unpack output
    output = tvm.compute((N, K, H, W), lambda n, k, h, w:
                         Y[k][n * nH * nW + idxd(h, m) * nW + idxd(w, m),
                              idxm(h, m), idxm(w, m)],
                         name='output', tag='winograd_conv2d_output')

    # we have to manually assign effective GFLOP for winograd
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
    if isinstance(U.op, tvm.tensor.ComputeOp):
        kernel, G = U.op.input_tensors
        s[G].compute_inline()
        eps, nu, k, c, kk, = s[U].op.axis
        if autotvm.GLOBAL_SCOPE.in_tuning:
            # kernel transformation will be pre-computed during compilation, so we skip
            # this part to make tuning records correct
            s[U].pragma(eps, 'debug_skip_region')
        else:
            r_kh, r_kw = s[U].op.reduce_axis
            s[U].reorder(k, c, eps, nu, r_kh, r_kw, kk)
            for axis in [eps, nu, r_kh, r_kw]:
                s[U].unroll(axis)
            s[U].vectorize(kk)
            s[U].parallel(k)

        if isinstance(kernel.op, tvm.tensor.ComputeOp) and "dilate" in kernel.op.tag:
            s[kernel].compute_inline()

    # transform image
    DD = s.cache_read(d, 'global', [V])
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
    cfg.define_split('tile_c', c, num_outputs=2, filter=lambda x: x.size[-1] <= 16)
    co, ci = cfg['tile_c'].apply(s, M, c)
    xo, xi = cfg['tile_p'].apply(s, M, b)
    s[M].reorder(eps, nu, xo, co, k, ci, xi)
    cfg.define_annotate('ann_reduce', [ci], policy='try_unroll')
    cfg.define_annotate('ann_spatial', [k, xi], policy='try_unroll_vec')
    cfg['ann_reduce'].apply(s, M, [ci],
                            axis_lens=[cfg['tile_c'].size[-1]],
                            max_unroll=16,
                            cfg=cfg)
    cfg['ann_spatial'].apply(s, M, [k, xi])

    # inverse transform
    s[A].compute_inline()
    k, b, vh, vw = s[Y].op.axis
    r_eps, r_nu = s[Y].op.reduce_axis
    for axis in [vh, vw, r_eps, r_nu]:
        s[Y].unroll(axis)

    # output
    n, co, h, w = s[last].op.axis
    co, coi = cfg['tile_k'].apply(s, last, co)
    p = s[last].fuse(n, co)
    s[M].compute_at(s[last], p)
    s[last].parallel(p)

    MM = s.cache_read(M, 'global', [Y])
    m = get_const_int(V.shape[0]) + 1 - 3
    ho, wo, hi, wi = s[last].tile(h, w, m, m)
    s[Y].compute_at(s[last], wo)
    s[MM].compute_at(s[last], wo)

    if output != last:
        s[output].compute_inline()


@autotvm.register_topi_compute(conv2d, 'arm_cpu', ['winograd_nnpack_fp16'])
def conv2d_arm_cpu_winograd_nnpack_fp16(
        cfg, data, kernel, strides, padding, dilation, layout, out_dtype):
    """ TOPI compute callback. Use winograd_nnpack_fp16 template """
    return conv2d_arm_cpu_winograd_nnpack(
        cfg, data, kernel, strides, padding, dilation, layout, out_dtype,
        tvm.contrib.nnpack.ConvolutionAlgorithm.WT_8x8_FP16)


@autotvm.register_topi_compute(conv2d, 'arm_cpu', ['winograd_nnpack_fp32'])
def conv2d_arm_cpu_winograd_nnpack_fp32(
        cfg, data, kernel, strides, padding, dilation, layout, out_dtype):
    """ TOPI compute callback. Use winograd_nnpack_fp32 template """
    return conv2d_arm_cpu_winograd_nnpack(
        cfg, data, kernel, strides, padding, dilation, layout, out_dtype,
        tvm.contrib.nnpack.ConvolutionAlgorithm.WT_8x8)


def conv2d_arm_cpu_winograd_nnpack(
        cfg, data, kernel, strides, padding, dilation, layout, out_dtype, convolution_algorithm):
    """ TOPI compute callback. Use winograd NNPACK template """
    N, CI, IH, IW = get_const_tuple(data.shape)

    if isinstance(dilation, int):
        dilation_h = dilation_w = dilation
    else:
        dilation_h, dilation_w = dilation
    assert (dilation_h, dilation_w) == (1, 1)
    assert len(kernel.shape) == 4
    CO, _, KH, KW = get_const_tuple(kernel.shape)
    HSTR, WSTR = strides if isinstance(strides, (tuple, list)) else (strides, strides)
    HPAD, WPAD, _, _ = get_pad_tuple(padding, kernel)

    assert layout == 'NCHW'
    assert KH == 3 and KW == 3 and HPAD == 1 and WPAD == 1 and HSTR == 1 and WSTR == 1
    H = (IH + 2 * HPAD - 3) // HSTR + 1
    W = (IW + 2 * WPAD - 3) // WSTR + 1

    cfg.define_knob('winograd_nnpack_algorithm', [convolution_algorithm])

    assert N == 1
    with tvm.tag_scope("winograd_nnpack_conv2d_weight_transform"):
        transformed_kernel = tvm.contrib.nnpack.convolution_inference_weight_transform(
            kernel, algorithm=cfg['winograd_nnpack_algorithm'].val)
        if autotvm.GLOBAL_SCOPE.in_tuning:
            transformed_kernel = tvm.compute(transformed_kernel.shape, lambda *args: 0.0)

    with tvm.tag_scope("winograd_nnpack_conv2d_output"):
        output = tvm.contrib.nnpack.convolution_inference_without_weight_transform(
            data, transformed_kernel,
            bias=None,
            padding=[HPAD, HPAD, WPAD, WPAD],
            stride=[HSTR, WSTR],
            algorithm=cfg['winograd_nnpack_algorithm'].val)

    # we have to manually assign effective GFLOP for winograd
    cfg.add_flop(2 * N * CI * H * W * KH * KW * CO)
    return output

def _schedule_winograd_nnpack(cfg, s, output, last):
    # Could have bias.

    (X, TK) = output.op.input_tensors[:2]

    # transform kernel
    assert isinstance(TK.op, (tvm.tensor.ComputeOp, tvm.tensor.ExternOp, tvm.tensor.PlaceholderOp))
    if autotvm.GLOBAL_SCOPE.in_tuning and isinstance(TK.op, tvm.tensor.ComputeOp):
        # kernel transformation will be pre-computed during compilation, so we skip
        # this part to make tuning records correct
        s[TK].pragma(s[TK].op.axis[0], 'debug_skip_region')


##### REGISTER TOPI COMPUTE / SCHEDULE FOR WINOGRAD WITH WEIGHT TRANSFORM #####
@autotvm.register_topi_compute(conv2d_winograd_without_weight_transform, 'arm_cpu', ['winograd'])
def conv2d_winograd_ww(cfg, data, kernel, strides, padding, dilation, layout, out_dtype, tile_size):
    """TOPI compute callback"""
    return _decl_winograd(cfg, data, kernel, strides, padding, dilation, layout, out_dtype,\
                          tile_size)


@autotvm.register_topi_schedule(schedule_conv2d_winograd_without_weight_transform,
                                'arm_cpu', ['winograd'])
def schedule_conv2d_winograd_without_weight_transform_(cfg, outs):
    """TOPI schedule callback"""
    s = tvm.create_schedule([x.op for x in outs])

    def _callback(op):
        if 'winograd_conv2d_output' in op.tag:
            output = op.output(0)
            _schedule_winograd(cfg, s, output, outs[0])

    traverse_inline(s, outs[0].op, _callback)
    return s


##### REGISTER TOPI COMPUTE / SCHEDULE FOR WINOGRAD NNPACK WITHOUT WEIGHT TRANSFORM #####
@autotvm.register_topi_compute(conv2d_winograd_nnpack_without_weight_transform,
                               'arm_cpu',
                               ['winograd_nnpack_fp16', 'winograd_nnpack_fp32'])
def conv2d_winograd_nnpack_ww(cfg, data, transformed_kernel, bias, strides,
                              padding, dilation, layout, out_dtype):
    """ TOPI compute callback. Use winograd NNPACK template """
    N, CI, IH, IW = get_const_tuple(data.shape)
    if isinstance(dilation, int):
        dilation_h = dilation_w = dilation
    else:
        dilation_h, dilation_w = dilation
    assert (dilation_h, dilation_w) == (1, 1)
    assert len(transformed_kernel.shape) == 4
    CO, _, _, _ = get_const_tuple(transformed_kernel.shape)
    HSTR, WSTR = strides if isinstance(strides, (tuple, list)) else (strides, strides)
    HPAD, WPAD, _, _ = get_pad_tuple(padding, (3, 3))
    KH, KW = 3, 3

    assert layout == 'NCHW'
    assert KH == 3 and KW == 3 and HPAD == 1 and WPAD == 1 and HSTR == 1 and WSTR == 1
    H = (IH + 2 * HPAD - 3) // HSTR + 1
    W = (IW + 2 * WPAD - 3) // WSTR + 1

    assert N == 1
    with tvm.tag_scope("winograd_nnpack_conv2d_output"):
        output = tvm.contrib.nnpack.convolution_inference_without_weight_transform(
            data=data,
            transformed_kernel=transformed_kernel,
            bias=bias,
            padding=[HPAD, HPAD, WPAD, WPAD],
            stride=[HSTR, WSTR],
            algorithm=cfg['winograd_nnpack_algorithm'].val)

    # we have to manually assign effective GFLOP for winograd
    cfg.add_flop(2 * N * CI * H * W * KH * KW * CO)
    return output


@autotvm.register_topi_schedule(schedule_conv2d_winograd_nnpack_without_weight_transform,
                                'arm_cpu', ['winograd_nnpack_fp16', 'winograd_nnpack_fp32'])
def schedule_conv2d_winograd_nnpack_without_weight_transform_(cfg, outs):
    """TOPI schedule callback"""
    s = tvm.create_schedule([x.op for x in outs])

    def _callback(op):
        if 'winograd_nnpack_conv2d_output' in op.tag:
            output = op.output(0)
            _schedule_winograd_nnpack(cfg, s, output, outs[0])

    traverse_inline(s, outs[0].op, _callback)
    return s


##### REGISTER ALTER OP LAYOUT #####
@conv2d_alter_layout.register(["arm_cpu"])
def _alter_conv2d_layout_arm(attrs, inputs, tinfos, F):
    """Alter op layout for pre-computing kernel transformation

    Parameters
    ----------
    attrs : nnvm.top.AttrDict or tvm.attrs.Attrs
        Attributes of current convolution
    inputs : nnvm.symbol or tvm.relay.Expr
        Grouped input symbols
    tinfos : list
        Input shape and dtype
    F: symbol
        The context, can be either nnvm.sym or relay.op

    Note
    ----
    Unlike other TOPI functions, this function operates on both graph level and operator level,
    so we have to pass 'F' to make it support our two versions of graph IR, NNVM and Relay.
    """
    copy_inputs = [s for s in inputs]

    new_attrs = {k: attrs[k] for k in attrs.keys()}

    if F.__name__ == 'tvm.relay.op':
        # Derive channels for frontends (e.g ONNX) that miss "channel" field.
        new_attrs["channels"] = inputs[1].checked_type.shape[attrs['kernel_layout'].index('O')]

    dilation = attrs.get_int_tuple("dilation")
    strides = attrs.get_int_tuple("strides")
    padding = attrs.get_int_tuple("padding")
    groups = attrs.get_int('groups')
    data_layout_key = "data_layout" if "data_layout" in new_attrs else "layout"
    layout = attrs[data_layout_key]
    kernel_layout = attrs['kernel_layout']
    out_dtype = attrs["out_dtype"]
    if out_dtype in ("same", ""):
        out_dtype = tinfos[0].dtype

    if dilation != (1, 1):
        logger.warning("Does not support weight pre-transform for dilated convolution.")
        return None

    # query config of this workload
    data, kernel = tinfos[0:2]
    if groups == 1:
        workload = autotvm.task.args_to_workload(
            [data, kernel, strides, padding, dilation, layout, out_dtype], conv2d)
    else:
        workload = autotvm.task.args_to_workload(
            [data, kernel, strides, padding, dilation, out_dtype], depthwise_conv2d_nchw)

    if layout == 'NCHW' and kernel_layout == 'OIHW':
        N, CI, H, W = get_const_tuple(data.shape)
        CO, _, KH, KW = get_const_tuple(kernel.shape)
    elif layout == 'NHWC' and kernel_layout == 'HWIO':
        N, H, W, CI = get_const_tuple(data.shape)
        KH, KW, _, CO = get_const_tuple(kernel.shape)
        # Also modify the workload to pick up because later we convert to NCHW
        # layout.
        new_data = tvm.placeholder((N, CI, H, W), dtype=data.dtype)
        new_kernel = tvm.placeholder((CO, CI, KH, KW), dtype=kernel.dtype)
        new_layout = 'NCHW'
        workload = autotvm.task.args_to_workload(
            [new_data, new_kernel, strides, padding, dilation, new_layout, out_dtype], conv2d)
    elif layout == 'NHWC' and kernel_layout == 'HWOI':
        # This is the case for depthwise convolution.
        N, H, W, CI = get_const_tuple(data.shape)
        KH, KW, CO, M = get_const_tuple(kernel.shape)
        # Also modify the workload to pick up because later we convert to NCHW
        # layout.
        new_data = tvm.placeholder((N, CI, H, W), dtype=data.dtype)
        new_kernel = tvm.placeholder((CO, M, KH, KW), dtype=kernel.dtype)
        workload = autotvm.task.args_to_workload(
            [new_data, new_kernel, strides, padding, dilation, out_dtype], depthwise_conv2d_nchw)
    else:
        return None

    idxd = tvm.indexdiv

    if groups == 1:
        target = tvm.target.current_target()
        dispatch_ctx = autotvm.DispatchContext.current
        cfg = dispatch_ctx.query(target, workload)

        if cfg.is_fallback:  # if is fallback, clear query cache and return None
            autotvm.task.clear_fallback_cache(target, workload)
            if layout == 'NHWC' and kernel_layout == 'HWIO':
                new_attrs['data_layout'] = 'NCHW'
                new_attrs['kernel_layout'] = 'OIHW'
                return F.nn.conv2d(*copy_inputs, **new_attrs)
            return None

        if cfg.template_key == 'direct':  # pack weight tensor
            VC = cfg['tile_co'].size[-1]
            new_attrs['kernel_layout'] = 'OIHW%do' % VC

            # Store the same config for the altered operator (workload)
            new_data = tvm.placeholder((N, CI, H, W), dtype=data.dtype)
            new_attrs[data_layout_key] = 'NCHW'
            new_kernel = tvm.placeholder((idxd(CO, VC), CI, KH, KW, VC), dtype=kernel.dtype)
            new_workload = autotvm.task.args_to_workload(
                [new_data, new_kernel, strides, padding, dilation, 'NCHW', out_dtype], conv2d)
            dispatch_ctx.update(target, new_workload, cfg)

            return F.nn.conv2d(*copy_inputs, **new_attrs)
        elif cfg.template_key == "winograd":  # pre-compute weight transformation in winograd
            if "-device=arm_cpu" in target.options:
                tile_size = 4
                VC = cfg['tile_k'].size[-1]
            elif "-device=bifrost" in target.options:
                tile_size = 2
                VC = 0
            else:
                from ..mali.conv2d import _pick_tile_size
                tile_size = _pick_tile_size(tinfos[0], tinfos[1])
                VC = cfg['tile_bna'].val

            weight = copy_inputs[1]
            if kernel_layout != 'OIHW':
                weight = F.transpose(weight, axes=(2, 3, 0, 1))
            weight = F.nn.contrib_conv2d_winograd_weight_transform(weight,
                                                                   tile_size=tile_size)
            if VC > 0:
                weight = F.reshape(weight,
                                   newshape=(KH + tile_size - 1,
                                             KW + tile_size - 1,
                                             idxd(CO, VC), VC, CI))
                weight = F.transpose(weight, axes=[0, 1, 2, 4, 3])
                new_weight = tvm.placeholder((KH + tile_size - 1,
                                              KW + tile_size -1,
                                              idxd(CO, VC), CI, VC),
                                             kernel.dtype)
            else:
                weight = F.reshape(weight,
                                   newshape=(KH + tile_size - 1, KW + tile_size - 1, CO, CI))
                new_weight = tvm.placeholder(
                    (KH + tile_size - 1, KW + tile_size -1, CO, CI), kernel.dtype
                )

            copy_inputs[1] = weight
            new_attrs['tile_size'] = tile_size
            new_attrs[data_layout_key] = 'NCHW'

            # Store the same config for the altered operator (workload)
            new_data = tvm.placeholder((N, CI, H, W), dtype=data.dtype)
            new_workload = autotvm.task.args_to_workload(
                [new_data, new_weight, strides, padding, dilation,
                 new_attrs[data_layout_key], out_dtype, tile_size],
                conv2d_winograd_without_weight_transform)
            dispatch_ctx.update(target, new_workload, cfg)

            return F.nn.contrib_conv2d_winograd_without_weight_transform(*copy_inputs, **new_attrs)
        elif cfg.template_key in ["winograd_nnpack_fp16", "winograd_nnpack_fp32"]:
            # pre-compute winograd_nnpack transform
            # for winograd_nnpack_fp16, the the precomputeprune pass must run on device,
            # where float16 is supported
            weight_dtype = 'float32'
            weight = copy_inputs[1]
            if kernel_layout != 'OIHW':
                weight = F.transpose(weight, axes=(2, 3, 0, 1))
            weight = F.nn.contrib_conv2d_winograd_weight_transform(weight,
                                                                   tile_size=tile_size)
            transformed_kernel = F.nn.contrib_conv2d_winograd_nnpack_weight_transform(
                weight,
                convolution_algorithm=cfg['winograd_nnpack_algorithm'].val,
                out_dtype=weight_dtype)
            copy_inputs[1] = transformed_kernel

            new_data = tvm.placeholder((N, CI, H, W), dtype=data.dtype)
            new_kernel = tvm.placeholder((CO, CI, 8, 8), "float32")
            bias = tvm.placeholder((CO, ), "float32")
            new_attrs[data_layout_key] = 'NCHW'
            new_workload = autotvm.task.args_to_workload(
                [new_data, new_kernel, bias, strides,
                 padding, dilation, new_attrs[data_layout_key], out_dtype]
                if len(copy_inputs) == 3 else
                [new_data, new_kernel, strides,
                 padding, dilation, new_attrs[data_layout_key], out_dtype],
                conv2d_winograd_nnpack_without_weight_transform)
            dispatch_ctx.update(target, new_workload, cfg)
            return F.nn.contrib_conv2d_winograd_nnpack_without_weight_transform(
                *copy_inputs, **new_attrs)
        else:
            raise RuntimeError("Unsupported template_key '%s'" % cfg.template_key)
    else:
        target = tvm.target.current_target()
        dispatch_ctx = autotvm.DispatchContext.current
        cfg = dispatch_ctx.query(target, workload)

        if cfg.is_fallback:  # if is fallback, clear query cache and return None
            autotvm.task.clear_fallback_cache(tvm.target.current_target(), workload)
            if layout == 'NHWC' and kernel_layout == 'HWOI':
                new_attrs['data_layout'] = 'NCHW'
                new_attrs['kernel_layout'] = 'OIHW'
                return F.nn.conv2d(*copy_inputs, **new_attrs)
            return None
        if cfg.template_key == 'contrib_spatial_pack':
            VC = cfg['tile_co'].size[-1]
            new_attrs['kernel_layout'] = 'OIHW%do' % (cfg['tile_co'].size[-1])

            # Store the same config for the altered operator (workload)
            new_data = tvm.placeholder((N, CI, H, W), dtype=data.dtype)
            new_attrs[data_layout_key] = 'NCHW'
            if attrs['kernel_layout'] == 'OIHW':
                CO, M, KH, KW = get_const_tuple(kernel.shape)
            elif attrs['kernel_layout'] == 'HWOI':
                KH, KW, CO, M = get_const_tuple(kernel.shape)
            else:
                raise RuntimeError("Depthwise conv should either have OIHW/HWIO kernel layout")
            new_kernel = tvm.placeholder((idxd(CO, VC), M, KH, KW, VC), dtype=kernel.dtype)
            new_workload = autotvm.task.args_to_workload(
                [new_data, new_kernel, strides, padding, dilation, out_dtype],
                depthwise_conv2d_nchw)
            dispatch_ctx.update(target, new_workload, cfg)

            return F.nn.conv2d(*copy_inputs, **new_attrs)
        else:
            # currently we only have contrib_spatial_pack and direct template
            # add more schedule templates.
            return None
