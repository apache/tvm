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
# pylint: disable=invalid-name,unused-variable,unused-argument,no-member
"""Conv2D schedule on x86"""

import logging
import re

import tvm
from tvm import autotvm
from tvm.autotvm.task.topi_integration import deserialize_args
from tvm.autotvm.task import get_config
from .. import generic, tag
from .. import nn
from ..nn.conv2d import conv2d, conv2d_NCHWc, \
    conv2d_infer_layout, _get_workload as _get_conv2d_workload
from ..nn.depthwise_conv2d import _get_workload as _get_depthwise_conv2d_workload
from ..nn.pad import pad
from ..util import get_const_tuple

from . import conv2d_avx_1x1, conv2d_avx_common

logger = logging.getLogger('topi')

def _get_default_config(cfg, data, kernel, strides, padding, out_dtype, is_depthwise=False,
                        layout='NCHW'):
    """
    Get default schedule config for the workload
    """
    static_data_shape = []
    for dim in get_const_tuple(data.shape):
        if isinstance(dim, tvm.expr.Var):
            static_data_shape.append(1)
        else:
            static_data_shape.append(dim)
    data = tvm.placeholder(static_data_shape, dtype=data.dtype)
    if is_depthwise:
        wkl = _get_depthwise_conv2d_workload(data, kernel, strides, padding, out_dtype)
        from .depthwise_conv2d import _fallback_schedule
        _fallback_schedule(cfg, wkl)
    else:
        wkl = _get_conv2d_workload(data, kernel, strides, padding, out_dtype, layout)
        is_kernel_1x1 = wkl.hkernel == 1 and wkl.wkernel == 1
        if is_kernel_1x1:
            conv2d_avx_1x1._fallback_schedule(cfg, wkl)
        else:
            conv2d_avx_common._fallback_schedule(cfg, wkl)

def _create_tuning_space(cfg, data, kernel, strides, padding, dilation, layout):
    """Create schedule configuration from input arguments"""
    dshape = get_const_tuple(data.shape)
    kshape = get_const_tuple(kernel.shape)
    pat = re.compile(r'NCHW.+(\d+)c')
    if layout == 'NCHW':
        n, ic, h, w = dshape
        oc, _, kh, kw = kshape
    elif layout == 'NHWC':
        n, h, w, ic = dshape
        kh, kw, oc, _ = kshape
    elif pat.match(layout) is not None:
        n, ic_chunk, h, w, ic_bn = dshape
        target = tvm.target.current_target(allow_none=False)
        oc_chunk, k_ic_chunk, kh, kw, k_ic_bn, oc_bn = kshape
        assert ic_chunk == k_ic_chunk
        assert ic_bn == k_ic_bn
        ic = ic_chunk*ic_bn
        oc = oc_chunk*oc_bn
    else:
        raise ValueError("Not support this layout {} with "
                         "schedule template.".format(layout))

    is_kernel_1x1 = kh == 1 and kw == 1
    ph, pw = padding if isinstance(padding, (tuple, list)) else (padding, padding)
    sh, sw = strides if isinstance(strides, (tuple, list)) else (strides, strides)
    oh = (h - kh + 2 * ph) // sh + 1
    ow = (w - kw + 2 * pw) // sw + 1

    # Create schedule config
    cfg.define_split("tile_ic", ic, num_outputs=2)
    cfg.define_split("tile_oc", oc, num_outputs=2)
    cfg.define_split("tile_ow", ow, num_outputs=2, filter=lambda y: y.size[-1] <= 64)
    if is_kernel_1x1:
        cfg.define_knob("tile_oh", [1, 2] if oh > 1 else [1])
    else:
        cfg.define_knob("unroll_kw", [True, False])


@autotvm.register_topi_compute(conv2d, 'cpu', ['direct'])
def _declaration_conv(cfg, data, kernel, strides, padding, dilation, layout, out_dtype):
    out_dtype = data.dtype if out_dtype is None else out_dtype
    padding = padding if isinstance(padding, (tuple, list)) else (padding, padding)
    strides = strides if isinstance(strides, (tuple, list)) else (strides, strides)
    dilation = dilation if isinstance(dilation, (tuple, list)) else (dilation, dilation)

    if layout == 'NCHW':
        _create_tuning_space(cfg, data, kernel, strides, padding, dilation, layout)
        if cfg.is_fallback:
            _get_default_config(cfg, data, kernel, strides, padding, out_dtype)
        return _declaration_conv_impl(cfg, data, kernel, strides,
                                      padding, dilation, layout, out_dtype)

    # HWOI kernel layout is for NHWC and HWCN
    kh, kw, _, _ = get_const_tuple(kernel.shape)
    if layout == 'HWCN':
        return nn.conv2d_hwcn(data, kernel, strides, padding, dilation, out_dtype)
    # FIXME - https://github.com/apache/incubator-tvm/issues/4122
    # _declaration_conv_nhwc_pack expects kernel layout to be HWOI. However, the tests use HWIO
    # layout. Commenting until we have clarity about the nhwc_pack implementation from the author.
    # elif layout == 'NHWC' and kh == 1 and kw == 1 and kernel.dtype == "int8":
    #     if cfg.is_fallback:
    #         _get_default_config(cfg, data, kernel, strides, padding, out_dtype, False, layout)
    #     # specialize for INT8 1X1 conv on X86
    #     return conv2d_avx_1x1._declaration_conv_nhwc_pack(cfg, data, kernel, strides,
    #                                                       padding, dilation, out_dtype)
    elif layout == 'NHWC':
        return nn.conv2d_nhwc(data, kernel, strides, padding, dilation, out_dtype)
    raise ValueError("not support this layout {} yet".format(layout))


def _declaration_conv_impl(cfg, data, kernel, strides, padding, dilation, layout, out_dtype):
    out_dtype = data.dtype if out_dtype is None else out_dtype
    assert layout == 'NCHW', "only support NCHW convolution for AVX"

    assert isinstance(dilation, int) or len(dilation) == 2
    if isinstance(dilation, int):
        dilation_h, dilation_w = dilation
    else:
        dilation_h, dilation_w = dilation

    HPAD, WPAD = padding
    HSTR, WSTR = strides

    batch_size, in_channel, in_height, in_width = get_const_tuple(data.shape)
    num_filter, _, kernel_height, kernel_width = get_const_tuple(kernel.shape)

    pad_height = in_height + 2 * HPAD
    pad_width = in_width + 2 * WPAD

    dilated_kernel_h = (kernel_height - 1) * dilation_h + 1
    dilated_kernel_w = (kernel_width - 1) * dilation_w + 1
    out_height = (in_height + 2 * HPAD - dilated_kernel_h) // HSTR + 1
    out_width = (in_width + 2 * WPAD - dilated_kernel_w) // WSTR + 1

    # pack data
    DOPAD = (HPAD != 0 or WPAD != 0)
    if DOPAD:
        data_pad = pad(data, (0, 0, HPAD, WPAD), name="data_pad")
    else:
        data_pad = data

    # fetch schedule
    ic_bn, oc_bn = cfg["tile_ic"].size[-1], cfg["tile_oc"].size[-1]

    shape = (batch_size, in_channel // ic_bn, pad_height, ic_bn, pad_width)
    data_vec = tvm.compute(shape,
                           lambda n, C, h, c, w: data_pad[n, C * ic_bn + c, h, w],
                           name='data_vec')

    # pack kernel
    shape = (num_filter//oc_bn, in_channel//ic_bn,
             kernel_height, kernel_width, ic_bn, oc_bn)
    kernel_vec = tvm.compute(shape,
                             lambda CO, CI, h, w, ci, co:
                             kernel[CO * oc_bn + co, CI * ic_bn + ci, h, w],
                             name='kernel_vec')

    # convolution
    oshape = (batch_size, num_filter//oc_bn, out_height, out_width, oc_bn)
    unpack_shape = (batch_size, num_filter, out_height, out_width)

    ic = tvm.reduce_axis((0, in_channel), name='ic')
    kh = tvm.reduce_axis((0, kernel_height), name='kh')
    kw = tvm.reduce_axis((0, kernel_width), name='kw')
    idxmod = tvm.indexmod
    idxdiv = tvm.indexdiv

    conv = tvm.compute(oshape, lambda n, oc_chunk, oh, ow, oc_block:
                       tvm.sum(data_vec[n, idxdiv(ic, ic_bn), oh*HSTR+kh*dilation_h,
                                        idxmod(ic, ic_bn),
                                        ow*WSTR+kw*dilation_w].astype(out_dtype) *
                               kernel_vec[oc_chunk, idxdiv(ic, ic_bn), kh, kw,
                                          idxmod(ic, ic_bn),
                                          oc_block].astype(out_dtype),
                               axis=[ic, kh, kw]), name='conv')

    unpack = tvm.compute(unpack_shape,
                         lambda n, c, h, w: conv[n, idxdiv(c, oc_bn), h, w, idxmod(c, oc_bn)]
                         .astype(out_dtype),
                         name='output_unpack',
                         tag='conv2d_nchw')
    return unpack


@autotvm.register_topi_schedule(generic.schedule_conv2d_nchw, 'cpu', ['direct'])
def schedule_conv2d(cfg, outs):
    """Create schedule for tensors"""
    s = tvm.create_schedule([x.op for x in outs])
    scheduled_ops = []

    def traverse(op):
        """Traverse operators from computation graph"""
        # inline all one-to-one-mapping operators except the last stage (output)
        if tag.is_broadcast(op.tag):
            if op not in s.outputs:
                s[op].compute_inline()
            for tensor in op.input_tensors:
                if isinstance(tensor.op, tvm.tensor.ComputeOp) and tensor.op not in scheduled_ops:
                    traverse(tensor.op)

        if 'conv2d_nchw' in op.tag:
            output = op.output(0)
            conv_out = op.input_tensors[0]
            kernel_vec = conv_out.op.input_tensors[1]
            kernel = kernel_vec.op.input_tensors[0]
            if isinstance(kernel.op, tvm.tensor.ComputeOp) and "dilate" in kernel.op.tag:
                s[kernel].compute_inline()
            data_vec = conv_out.op.input_tensors[0]
            data = data_vec.op.input_tensors[0]
            data_pad = None
            if isinstance(data.op, tvm.tensor.ComputeOp) and "pad" in data.op.tag:
                data_pad = data
                data = data_pad.op.input_tensors[0]

            _, _, kh, kw = get_const_tuple(kernel.shape)
            is_kernel_1x1 = kh == 1 and kw == 1
            args = [s, cfg, data, data_pad, data_vec, kernel_vec, conv_out, output, outs[0]]
            if is_kernel_1x1:
                conv2d_avx_1x1._schedule_conv(*args)
            else:
                conv2d_avx_common._schedule_conv(*args)

        scheduled_ops.append(op)

    traverse(outs[0].op)
    return s

@generic.schedule_conv2d_nhwc.register("cpu")
def schedule_conv2d_nhwc(outs):
    """Create schedule for tensors"""
    s = tvm.create_schedule([x.op for x in outs])
    output_op = outs[0].op
    scheduled_ops = []

    def traverse(op):
        """Traverse operators from computation graph"""
        # inline all one-to-one-mapping operators except the last stage (output)
        if tag.is_broadcast(op.tag):
            if op not in s.outputs:
                s[op].compute_inline()
            else: # inject custom schedule
                if len(op.axis) == 4: # schedule bias + bn + relu
                    n, h, w, c = op.axis
                    fused = s[op].fuse(n, h, w)
                    s[op].parallel(fused)
                    s[op].vectorize(c)
            for tensor in op.input_tensors:
                if isinstance(tensor.op, tvm.tensor.ComputeOp) and tensor.op not in scheduled_ops:
                    traverse(tensor.op)

        if 'conv2d_nhwc' in op.tag:
            conv = op.output(0)
            kernel = op.input_tensors[1]
            if isinstance(kernel.op, tvm.tensor.ComputeOp) and "dilate" in kernel.op.tag:
                s[kernel].compute_inline()

            data = op.input_tensors[0]
            data_pad = None
            if isinstance(data.op, tvm.tensor.ComputeOp) and "pad" in data.op.tag:
                data_pad = data
                data = data_pad.op.input_tensors[0]

            n_pad, h_pad, w_pad, c_pad = data_pad.op.axis
            pad_fused = s[data_pad].fuse(n_pad, h_pad)
            s[data_pad].parallel(pad_fused)
            C = conv
            n, h, w, c = C.op.axis
            ry, rx, rc = C.op.reduce_axis
            n_out, h_out, w_out, c_out = output_op.axis
            s[C].vectorize(c)
            if op != output_op: # fuse bias + bn + relu into conv
                s[C].compute_at(s[output_op], c_out)
            else:
                fused = s[C].fuse(n, h, w)
                s[C].parallel(fused)

        scheduled_ops.append(op)

    traverse(output_op)
    return s


# Define template function for autotvm task
# We define schedule template in this function instead of
# declaration function since actual input arguments need
# to be altered by the schedule selected.
@autotvm.task.register("topi_x86_conv2d_NCHWc")
def _topi_nn_conv2d_NCHWc(*args, **kwargs):
    assert not kwargs, "Do not support kwargs in template function call"
    args = deserialize_args(args)

    if len(args) == 7:
        data, kernel, strides, padding, dilation, origin_layout, dtype = args
    else:
        assert len(args) == 8
        data, kernel, strides, padding, dilation, origin_layout, out_layout, dtype = args

    raw_data_shape = get_const_tuple(data.shape)
    raw_kernel_shape = get_const_tuple(kernel.shape)

    # get config here
    cfg = get_config()
    _create_tuning_space(cfg, data, kernel, strides, padding, dilation, origin_layout)

    idxdiv = tvm.indexdiv
    idxmod = tvm.indexmod
    # change shape with the value in config
    ic_bn, oc_bn, ow_bn = (cfg["tile_ic"].size[-1], cfg["tile_oc"].size[-1],
                           cfg["tile_ow"].size[-1])
    new_data_shape = (raw_data_shape[0], idxdiv(raw_data_shape[1], ic_bn),
                      raw_data_shape[2], raw_data_shape[3], ic_bn)
    data_layout = "NCHW%dc" % ic_bn
    out_layout = "NCHW%dc" % oc_bn
    new_kernel_shape = (idxdiv(raw_kernel_shape[0], oc_bn),
                        idxdiv(raw_kernel_shape[1], ic_bn),
                        raw_kernel_shape[2], raw_kernel_shape[3], ic_bn, oc_bn)
    new_data = tvm.placeholder(new_data_shape, data.dtype)
    new_kernel = tvm.placeholder(new_kernel_shape, kernel.dtype)

    C = _declaration_conv_NCHWc(cfg, new_data, new_kernel, strides, padding, dilation,
                                data_layout, out_layout, dtype)
    s = _schedule_conv2d_NCHWc(cfg, [C])
    return s, [new_data, new_kernel, C]


@conv2d_infer_layout.register("cpu")
def _conv2d_infer_layout(workload, cfg):
    _, data, kernel, strides, padding, dilation, layout, dtype = workload
    batch_size, in_channel, in_height, in_width = data[:-1]
    out_channel, _, k_height, k_width = kernel[:-1]
    idxdiv = tvm.indexdiv

    out_height = idxdiv(in_height + 2 * padding[0] - k_height, strides[0]) + 1
    out_width = idxdiv(in_width + 2 * padding[1] - k_width, strides[1]) + 1
    tile_ic, tile_oc = cfg["tile_ic"].size[-1], cfg["tile_oc"].size[-1]
    in_shape = (batch_size, idxdiv(in_channel, tile_ic), in_height, in_width, tile_ic)
    in_layout = "NCHW%dc" % tile_ic
    out_shape = (batch_size, idxdiv(out_channel, tile_oc), out_height, out_width, tile_oc)
    out_layout = "NCHW%dc" % tile_oc
    return ((in_shape, in_layout),), ((out_shape, out_layout),)


@autotvm.register_topi_compute(conv2d_NCHWc, 'cpu', 'direct')
def _declaration_conv_NCHWc(cfg, data, kernel, strides,
                            padding, dilation, layout, out_layout, out_dtype):
    # layout and out_layout are not used here,
    # we keep them for debug convenience when dumping autotvm workload
    n, ic_chunk, ih, iw, ic_bn = get_const_tuple(data.shape)
    in_channel = ic_chunk * ic_bn
    oc_chunk, ic_chunk_group, kernel_height, kernel_width, _, oc_bn = \
            get_const_tuple(kernel.shape)
    num_filter = oc_chunk * oc_bn

    # If no config was set, we can fallback to NCHW config.
    if cfg.is_fallback:
        _get_default_config(cfg, tvm.placeholder((n, in_channel, ih, iw), dtype=data.dtype),
                            tvm.placeholder((num_filter, in_channel, kernel_height, kernel_width),
                                            dtype=kernel.dtype),
                            strides, padding, out_dtype)

    return nn.conv2d_NCHWc_compute(data,
                                   kernel,
                                   strides,
                                   padding,
                                   dilation,
                                   layout,
                                   out_layout,
                                   out_dtype)


@autotvm.register_topi_schedule(generic.schedule_conv2d_NCHWc, 'cpu', ['direct'])
def _schedule_conv2d_NCHWc(cfg, outs):
    """Create schedule for tensors"""
    s = tvm.create_schedule([x.op for x in outs])
    scheduled_ops = []

    def traverse(op):
        """Traverse operators from computation graph"""
        # inline all one-to-one-mapping operators except the last stage (output)
        if tag.is_broadcast(op.tag):
            if op not in s.outputs:
                s[op].compute_inline()
            for tensor in op.input_tensors:
                if isinstance(tensor.op, tvm.tensor.ComputeOp) and tensor.op not in scheduled_ops:
                    traverse(tensor.op)

        if 'conv2d_NCHWc' in op.tag:
            conv_out = op.output(0)
            kernel = conv_out.op.input_tensors[1]
            data_vec = conv_out.op.input_tensors[0]
            data = data_vec.op.input_tensors[0] \
                if isinstance(data_vec.op, tvm.tensor.ComputeOp) and "pad" not in data_vec.op.tag \
                else data_vec
            if isinstance(data.op, tvm.tensor.ComputeOp) and "pad" in data.op.tag:
                data_pad = data
                data = data_pad.op.input_tensors[0]

            args = [s, cfg, data_vec, conv_out, outs[0]]
            target = tvm.target.current_target(allow_none=False)
            _, _, kh, kw, _, _, = get_const_tuple(kernel.shape)
            if kh == 1 and kw == 1:
                conv2d_avx_1x1._schedule_conv_NCHWc(*args)
            else:
                conv2d_avx_common._schedule_conv_NCHWc(*args)

        scheduled_ops.append(op)

    traverse(outs[0].op)
    return s
